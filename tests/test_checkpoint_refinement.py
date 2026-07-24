from pathlib import Path

import torch

from ego_hand_wm.models.world_action_model import WorldActionModel
from ego_hand_wm.training.checkpoint import (
    initialize_model_from_checkpoint,
    save_model_checkpoint,
)


def _config(*, hand: bool) -> dict:
    config = {
        "hidden_dim": 32,
        "heads": 4,
        "context_depth": 1,
        "depth": 1,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "future_visual_latent_dim": 0,
        "vision": {"kind": "tiny"},
        "text": {"kind": "hash", "max_tokens": 4},
    }
    if hand:
        config["hand_kinematics"] = {"enabled": True, "heads": 4, "depth": 1}
    return config


def test_model_only_initialization_allows_new_auxiliary_head(tmp_path: Path) -> None:
    pretrained = WorldActionModel(_config(hand=False))
    path = tmp_path / "pretrained.pt"
    save_model_checkpoint(path, model=pretrained, step=7, config={})

    refined = WorldActionModel(_config(hand=True))
    missing, unexpected = initialize_model_from_checkpoint(
        path, model=refined, strict=False
    )
    assert missing
    assert all(name.startswith("hand_kinematics_head.") for name in missing)
    assert unexpected == []
    torch.testing.assert_close(
        refined.denoiser.input_projections[0].weight,
        pretrained.denoiser.input_projections[0].weight,
    )
