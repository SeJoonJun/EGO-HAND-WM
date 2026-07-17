from dataclasses import replace

import torch

from ego_hand_wm.contracts.batch import canonical_collate
from ego_hand_wm.data.synthetic import SyntheticCanonicalDataset
from ego_hand_wm.flow.rectified_flow import make_flow_training_sample, sample_ode
from ego_hand_wm.losses import WorldActionLoss
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
