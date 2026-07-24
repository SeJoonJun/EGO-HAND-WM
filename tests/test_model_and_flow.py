from dataclasses import replace

import pytest
import torch

from ego_hand_wm.contracts.batch import canonical_collate
from ego_hand_wm.contracts.schema import SCHEMA
from ego_hand_wm.data.synthetic import SyntheticCanonicalDataset
from ego_hand_wm.flow.rectified_flow import (
    make_flow_training_sample,
    make_visual_flow_training_sample,
    normalize_visual_flow_target,
    sample_flow_time,
    sample_ode,
)
from ego_hand_wm.losses import WorldActionLoss, wrist_trajectory_losses
from ego_hand_wm.models.world_action_model import WorldActionModel


def tiny_model_config() -> dict:
    return {
        "hidden_dim": 32,
        "heads": 4,
        "context_depth": 1,
        "depth": 1,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "physical_max_period": 10.0,
        "future_visual_latent_dim": 0,
        "vision": {"kind": "tiny"},
        "text": {"kind": "hash", "max_tokens": 8},
    }


def test_model_loss_backward_and_sampler() -> None:
    dataset = SyntheticCanonicalDataset(length=2, history_steps=2, future_steps=3, image_size=16)
    batch = canonical_collate([dataset[0], dataset[1]])
    model = WorldActionModel(tiny_model_config())
    flow = make_flow_training_sample(batch)
    prediction, _, _, _, _ = model(batch, flow.noisy_state, flow.flow_time)
    criterion = WorldActionLoss({"rotation_weight": 0.01, "ego_weight": 0.01})
    metrics = criterion(
        batch, prediction, flow.target_velocity, flow.noisy_state, flow.flow_time
    )
    assert torch.isfinite(metrics["loss"])
    metrics["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    generated = sample_ode(model.eval(), batch, steps=2)
    assert generated.shape == batch.future_state.shape
    assert torch.isfinite(generated).all()


def test_future_supervision_mask_does_not_change_flow_input() -> None:
    dataset = SyntheticCanonicalDataset(length=1, history_steps=2, future_steps=3, image_size=16)
    batch = canonical_collate([dataset[0]])
    changed = replace(batch, future_stream_mask=torch.zeros_like(batch.future_stream_mask))
    torch.manual_seed(123)
    original_flow = make_flow_training_sample(batch)
    torch.manual_seed(123)
    changed_flow = make_flow_training_sample(changed)
    torch.testing.assert_close(original_flow.noisy_state, changed_flow.noisy_state)
    torch.testing.assert_close(original_flow.target_velocity, changed_flow.target_velocity)
    torch.testing.assert_close(original_flow.flow_time, changed_flow.flow_time)


def test_future_visual_target_per_token_layer_norm_and_mask() -> None:
    torch.manual_seed(9)
    latent = torch.randn(2, 3, 17, 8) * 4.0 + 7.0
    original = latent.clone()
    valid = torch.tensor([[True, True, False], [True, False, False]])
    flow_time = torch.tensor([0.25, 0.75])

    normalized = normalize_visual_flow_target(
        latent, mode="per_token_layer_norm", eps=1e-6
    )
    torch.testing.assert_close(latent, original)
    torch.testing.assert_close(
        normalized.mean(dim=-1), torch.zeros_like(normalized[..., 0]), atol=2e-6, rtol=0
    )
    torch.testing.assert_close(
        normalized.var(dim=-1, unbiased=False),
        torch.ones_like(normalized[..., 0]),
        atol=2e-5,
        rtol=0,
    )

    sample = make_visual_flow_training_sample(
        latent,
        flow_time,
        valid,
        normalization="per_token_layer_norm",
        normalization_eps=1e-6,
    )
    assert sample.noisy_latent.shape == latent.shape
    assert sample.target_velocity.shape == latent.shape
    assert not sample.noisy_latent[~valid].any()
    assert not sample.clean_latent[~valid].any()
    assert not sample.target_velocity[~valid].any()
    torch.testing.assert_close(sample.clean_latent[valid], normalized[valid])


def test_unknown_future_visual_target_normalization_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unknown visual target normalization"):
        normalize_visual_flow_target(torch.randn(1, 2, 3, 4), mode="not-a-mode")


def test_visual_flow_time_can_be_shifted_independently() -> None:
    torch.manual_seed(31)
    uniform = sample_flow_time(
        16, device=torch.device("cpu"), dtype=torch.float32
    )
    torch.manual_seed(31)
    shifted = sample_flow_time(
        16,
        device=torch.device("cpu"),
        dtype=torch.float32,
        config={"distribution": "uniform", "shift": 5.0},
    )
    torch.testing.assert_close(shifted, 5.0 * uniform / (1.0 + 4.0 * uniform))
    assert shifted.mean() > uniform.mean()


def test_hand_kinematics_head_starts_from_persistence_and_backpropagates() -> None:
    sample = SyntheticCanonicalDataset(
        length=2, history_steps=2, future_steps=3, image_size=16
    )[1]
    generator = torch.Generator().manual_seed(5)
    history = torch.randn(2, 2, 21, 3, generator=generator) * 0.03
    future = history[-1:].expand(3, -1, -1, -1).clone()
    future = future + torch.linspace(0.0, 0.02, 3)[:, None, None, None]
    sample["history_hand_joints_local"] = history
    sample["future_hand_joints_local"] = future
    batch = canonical_collate([sample])
    config = tiny_model_config()
    config["hand_kinematics"] = {"enabled": True, "depth": 1, "heads": 4}
    model = WorldActionModel(config)
    flow = make_flow_training_sample(batch)
    output = model(batch, flow.noisy_state, flow.flow_time)
    assert output.hand_joints_local is not None
    expected_persistence = history[-1:].expand(3, -1, -1, -1)
    torch.testing.assert_close(output.hand_joints_local[0], expected_persistence)

    criterion = WorldActionLoss(
        {
            "rotation_weight": 0.01,
            "ego_weight": 0.01,
            "hand_joint_weight": 10.0,
            "fingertip_weight": 5.0,
            "hand_motion_weight": 0.1,
        }
    )
    metrics = criterion(
        batch,
        output.velocity,
        flow.target_velocity,
        flow.noisy_state,
        flow.flow_time,
        predicted_hand_joints_local=output.hand_joints_local,
    )
    assert {"hand_joint", "fingertip", "hand_motion"}.issubset(metrics)
    metrics["loss"].backward()
    assert model.hand_kinematics_head.output.weight.grad is not None


def test_metric_aligned_trajectory_losses_use_clean_estimate_and_masks() -> None:
    sample = SyntheticCanonicalDataset(
        length=1, history_steps=2, future_steps=3, image_size=16
    )[0]
    batch = canonical_collate([sample])
    exact = wrist_trajectory_losses(
        batch,
        batch.future_state.clone(),
        horizon_power=1.0,
        velocity_beta=0.05,
    )
    for value in exact.values():
        torch.testing.assert_close(value, torch.zeros_like(value))

    shifted = batch.future_state.clone()
    shifted[:, -1, SCHEMA.right_wrist.start] += 0.03
    losses = wrist_trajectory_losses(
        batch,
        shifted,
        horizon_power=1.0,
        velocity_beta=0.05,
    )
    assert losses["trajectory/position"] > 0
    assert losses["trajectory/endpoint"] > losses["trajectory/position"]
    assert losses["trajectory/velocity"] > 0


def test_trajectory_losses_backpropagate_through_flow_clean_estimate() -> None:
    dataset = SyntheticCanonicalDataset(
        length=1, history_steps=2, future_steps=3, image_size=16
    )
    batch = canonical_collate([dataset[0]])
    flow = make_flow_training_sample(batch)
    prediction = torch.nn.Parameter(flow.target_velocity.detach() + 0.1)
    criterion = WorldActionLoss(
        {
            "rotation_weight": 0.0,
            "ego_weight": 0.0,
            "trajectory_position_weight": 2.0,
            "trajectory_endpoint_weight": 3.0,
            "trajectory_velocity_weight": 0.1,
            "trajectory_horizon_power": 1.0,
        }
    )
    metrics = criterion(
        batch,
        prediction,
        flow.target_velocity,
        flow.noisy_state,
        flow.flow_time,
    )
    assert {
        "trajectory/position",
        "trajectory/endpoint",
        "trajectory/velocity",
    }.issubset(metrics)
    metrics["loss"].backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
