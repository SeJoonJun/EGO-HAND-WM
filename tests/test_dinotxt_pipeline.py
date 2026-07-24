import json
from pathlib import Path

import numpy as np
import torch

from ego_hand_wm.contracts.batch import canonical_collate
from ego_hand_wm.data.adapters.vitra import enumerate_vitra_prompts
from ego_hand_wm.data.dinotxt_text import (
    DinoTxtTextFeatureStore,
    TEXT_CACHE_CONTRACT,
    write_text_feature_cache,
)
from ego_hand_wm.data.synthetic import SyntheticCanonicalDataset
from ego_hand_wm.data.unique_features import UNIQUE_VISUAL_CONTRACT, UniqueVisualFeatureStore
from ego_hand_wm.models.encoders import ContextEncoder
from ego_hand_wm.models.world_action_model import WorldActionModel


def test_enumerate_vitra_prompts_matches_overlapping_half_open_windows() -> None:
    episode = {
        "extrinsics": np.zeros((10, 4, 4)),
        "anno_type": "right",
        "text": {
            "right": [("pick cup", [1, 7])],
            "left": [("hold plate", [4, 9])],
        },
    }
    assert enumerate_vitra_prompts(episode) == {
        "",
        "Right hand: pick cup",
        "Right hand: pick cup Left hand: hold plate",
        "Left hand: hold plate",
    }


def test_text_feature_cache_round_trip(tmp_path: Path) -> None:
    features = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    write_text_feature_cache(
        output_root=tmp_path,
        texts=["", "pick cup"],
        features=features,
        metadata={"backend": "fake"},
    )
    store = DinoTxtTextFeatureStore(tmp_path)
    assert store.success["contract"] == TEXT_CACHE_CONTRACT
    torch.testing.assert_close(store.lookup("pick cup"), torch.tensor([3.0, 4.0]))


def test_unique_visual_feature_lookup_uses_physical_frame_ids(tmp_path: Path) -> None:
    feature_root = tmp_path / "features"
    rgb_root = tmp_path / "rgb"
    (feature_root / "epic").mkdir(parents=True)
    (rgb_root / "epic").mkdir(parents=True)
    extractor = "a" * 64
    (feature_root / "_SUCCESS").write_text(
        json.dumps(
            {
                "complete": True,
                "contract": UNIQUE_VISUAL_CONTRACT,
                "extractor_id": extractor,
                "total_tokens": 17,
                "feature_dim": 4,
            }
        )
    )
    frame_ids = np.asarray([2, 7, 11], dtype=np.int64)
    values = np.arange(3 * 17 * 4, dtype=np.float16).reshape(3, 17, 4)
    np.save(rgb_root / "epic/sample.frames.npy", frame_ids, allow_pickle=False)
    np.save(feature_root / "epic/sample.features.npy", values, allow_pickle=False)
    (feature_root / "epic/sample.json").write_text(
        json.dumps({"complete": True, "extractor_id": extractor})
    )
    store = UniqueVisualFeatureStore(feature_root=feature_root, staged_rgb_root=rgb_root)
    selected = store.lookup("epic", "sample", np.asarray([11, 2]))
    assert selected.shape == (2, 17, 4)
    torch.testing.assert_close(selected[0], torch.from_numpy(values[2].astype(np.float32)))

    compact = UniqueVisualFeatureStore(
        feature_root=feature_root,
        staged_rgb_root=rgb_root,
        output_dtype=np.float16,
    ).lookup("epic", "sample", np.asarray([11, 2]))
    assert compact.dtype == torch.float16
    torch.testing.assert_close(compact[0], torch.from_numpy(values[2]))


def _dinotxt_batch() -> object:
    sample = SyntheticCanonicalDataset(
        length=1, history_steps=3, future_steps=4, image_size=16
    )[0]
    sample.pop("context_images")
    sample["context_visual_features"] = torch.randn(3, 17, 8)
    sample["context_text_features"] = torch.randn(12)
    sample["future_visual_latents"] = torch.randn(4, 17, 8)
    return canonical_collate([sample])


def test_dinotxt_context_reconstructs_global_plus_16_spatial_tokens() -> None:
    batch = _dinotxt_batch()
    encoder = ContextEncoder(
        {
            "hidden_dim": 32,
            "heads": 4,
            "context_depth": 1,
            "mlp_ratio": 2.0,
            "vision": {
                "kind": "precomputed_dinotxt",
                "feature_dim": 8,
                "spatial_tokens": 16,
            },
            "text": {"kind": "precomputed_dinotxt", "feature_dim": 12},
        }
    )
    tokens, valid = encoder(batch)
    assert valid[0, : 3 * 17].all()
    assert tokens.shape[-1] == 32


def test_parallel_visual_flow_expert_supports_17_tokens_and_shared_context() -> None:
    batch = _dinotxt_batch()
    model = WorldActionModel(
        {
            "hidden_dim": 32,
            "heads": 4,
            "context_depth": 1,
            "depth": 2,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "vision": {
                "kind": "precomputed_dinotxt",
                "feature_dim": 8,
                "spatial_tokens": 16,
            },
            "text": {"kind": "precomputed_dinotxt", "feature_dim": 12},
            "future_visual_latent_dim": 8,
            "future_visual_tokens": 17,
            "future_visual_depth": 1,
        }
    )
    noisy_state = torch.randn_like(batch.future_state)
    noisy_visual = torch.randn_like(batch.future_visual_latents)
    velocity, _, visual_velocity, _, _ = model(
        batch,
        noisy_state,
        torch.tensor([0.4]),
        noisy_visual_latent=noisy_visual,
    )
    assert velocity.shape == batch.future_state.shape
    assert visual_velocity is not None
    assert visual_velocity.shape == noisy_visual.shape


def test_cache_reuse_matches_fresh_prefill_and_future_visual_cannot_leak() -> None:
    torch.manual_seed(7)
    batch = _dinotxt_batch()
    model = WorldActionModel(
        {
            "hidden_dim": 32,
            "heads": 4,
            "context_depth": 1,
            "depth": 2,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "vision": {
                "kind": "precomputed_dinotxt",
                "feature_dim": 8,
                "spatial_tokens": 16,
            },
            "text": {"kind": "precomputed_dinotxt", "feature_dim": 12},
            "future_visual_latent_dim": 8,
            "future_visual_tokens": 17,
            "future_visual_depth": 2,
        }
    ).eval()
    torch.nn.init.normal_(model.future_visual_expert.output_projection.weight, std=0.02)
    torch.nn.init.constant_(
        model.future_visual_expert.blocks[0].modulation[-1].bias, 0.1
    )
    noisy_state = torch.randn_like(batch.future_state)
    first_visual = torch.randn_like(batch.future_visual_latents)
    second_visual = torch.randn_like(batch.future_visual_latents)
    flow_time = torch.tensor([0.4])

    fresh_geometry, _, first_prediction, _, _ = model(
        batch,
        noisy_state,
        flow_time,
        noisy_visual_latent=first_visual,
    )
    cache = model.prefill_context(batch)
    assert cache.depth == 2
    assert cache.keys[0].shape[:3] == (1, 4, cache.valid.shape[1])
    cached_geometry, _, second_prediction, _, _ = model(
        batch,
        noisy_state,
        flow_time,
        context_cache=cache,
        noisy_visual_latent=second_visual,
    )
    torch.testing.assert_close(cached_geometry, fresh_geometry)
    assert first_prediction is not None and second_prediction is not None
    assert not torch.equal(first_prediction, second_prediction)

    # Geometry is computed before and independently of the future-visual expert.  Changing a
    # future-DINO input therefore cannot alter the trajectory prediction.
    repeated_geometry, _, _, _, _ = model(
        batch,
        noisy_state,
        flow_time,
        context_cache=cache,
        noisy_visual_latent=first_visual,
    )
    torch.testing.assert_close(repeated_geometry, cached_geometry)


def test_visual_loss_reaches_shared_context_but_not_through_geometry_hidden() -> None:
    torch.manual_seed(11)
    batch = _dinotxt_batch()
    model = WorldActionModel(
        {
            "hidden_dim": 32,
            "heads": 4,
            "context_depth": 1,
            "depth": 1,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "vision": {
                "kind": "precomputed_dinotxt",
                "feature_dim": 8,
                "spatial_tokens": 16,
            },
            "text": {"kind": "precomputed_dinotxt", "feature_dim": 12},
            "future_visual_latent_dim": 8,
            "future_visual_tokens": 17,
            "future_visual_depth": 1,
        }
    )
    # Move the zero-initialized visual output projection off zero so this unit test observes
    # the intended gradient path on its first backward pass.
    torch.nn.init.normal_(model.future_visual_expert.output_projection.weight, std=0.02)
    torch.nn.init.constant_(
        model.future_visual_expert.blocks[0].modulation[-1].bias, 0.1
    )
    _, _, visual_velocity, _, _ = model(
        batch,
        torch.randn_like(batch.future_state),
        torch.tensor([0.3]),
        noisy_visual_latent=torch.randn_like(batch.future_visual_latents),
    )
    assert visual_velocity is not None
    visual_velocity.square().mean().backward()
    assert model.context_kv_projector.projections[0].weight.grad is not None
    assert model.context_encoder.fusion.layers[0].linear1.weight.grad is not None
    assert model.denoiser.output_projections[0].weight.grad is None


def test_training_only_visual_expert_can_be_discarded_for_inference() -> None:
    batch = _dinotxt_batch()
    model = WorldActionModel(
        {
            "hidden_dim": 32,
            "heads": 4,
            "context_depth": 1,
            "depth": 1,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "vision": {
                "kind": "precomputed_dinotxt",
                "feature_dim": 8,
                "spatial_tokens": 16,
            },
            "text": {"kind": "precomputed_dinotxt", "feature_dim": 12},
            "future_visual_latent_dim": 8,
            "future_visual_tokens": 17,
            "future_visual_depth": 1,
        }
    ).eval()
    parameters_before = sum(parameter.numel() for parameter in model.parameters())
    model.discard_future_visual_expert()
    parameters_after = sum(parameter.numel() for parameter in model.parameters())
    assert model.future_visual_expert is None
    assert parameters_after < parameters_before
    prediction, _, visual, _, _ = model(
        batch,
        torch.randn_like(batch.future_state),
        torch.tensor([0.5]),
    )
    assert prediction.shape == batch.future_state.shape
    assert visual is None
