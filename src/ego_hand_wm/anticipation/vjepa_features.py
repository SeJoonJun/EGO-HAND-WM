"""Frozen V-JEPA 2.1 ViT-G/16 loading and compact token extraction."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


VJEPA_FRAMES = 32
VJEPA_RESOLUTION = 384
VJEPA_PATCH_SIZE = 16
VJEPA_TUBELET_SIZE = 2
VJEPA_EMBED_DIM = 1664
VJEPA_TEMPORAL_TOKENS = VJEPA_FRAMES // VJEPA_TUBELET_SIZE
VJEPA_SPATIAL_GRID = VJEPA_RESOLUTION // VJEPA_PATCH_SIZE


def load_frozen_vjepa2_1_vitg(
    *, checkpoint: str | Path, repository: str | Path
) -> nn.Module:
    """Load the official 2B ViT-G/384 target encoder without its predictor."""

    repository = str(Path(repository).resolve())
    if repository not in sys.path:
        sys.path.insert(0, repository)
    from app.vjepa_2_1.models import vision_transformer as vit

    encoder = vit.vit_gigantic_xformers(
        img_size=(VJEPA_RESOLUTION, VJEPA_RESOLUTION),
        num_frames=VJEPA_FRAMES,
        patch_size=VJEPA_PATCH_SIZE,
        tubelet_size=VJEPA_TUBELET_SIZE,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=True,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    pretrained = payload["target_encoder"]
    pretrained = {
        key.removeprefix("module.").removeprefix("backbone."): value
        for key, value in pretrained.items()
    }
    compatible = {
        key: value
        for key, value in pretrained.items()
        if key in encoder.state_dict() and encoder.state_dict()[key].shape == value.shape
    }
    missing, unexpected = encoder.load_state_dict(compatible, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected V-JEPA target-encoder keys: {unexpected[:10]}")
    # RoPE models legitimately omit learned positional tensors, but every learned block and
    # projection must have loaded.
    learned_missing = [
        key
        for key in missing
        if not key.startswith(("pos_embed", "mask_token"))
    ]
    if learned_missing:
        raise RuntimeError(f"Missing V-JEPA target-encoder keys: {learned_missing[:10]}")
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def compact_vjepa_tokens(tokens: torch.Tensor, *, spatial_size: int = 4) -> torch.Tensor:
    """Pool each 24x24 spatial grid to 4x4 while preserving all 16 tubelets."""

    expected = VJEPA_TEMPORAL_TOKENS * VJEPA_SPATIAL_GRID**2
    if tokens.ndim != 3 or tokens.shape[1:] != (expected, VJEPA_EMBED_DIM):
        raise ValueError(
            f"Expected V-JEPA tokens [B,{expected},{VJEPA_EMBED_DIM}], got {tuple(tokens.shape)}"
        )
    batch = tokens.shape[0]
    grids = tokens.view(
        batch,
        VJEPA_TEMPORAL_TOKENS,
        VJEPA_SPATIAL_GRID,
        VJEPA_SPATIAL_GRID,
        VJEPA_EMBED_DIM,
    )
    grids = grids.permute(0, 1, 4, 2, 3).reshape(
        batch * VJEPA_TEMPORAL_TOKENS,
        VJEPA_EMBED_DIM,
        VJEPA_SPATIAL_GRID,
        VJEPA_SPATIAL_GRID,
    )
    pooled = F.adaptive_avg_pool2d(grids.float(), (spatial_size, spatial_size))
    pooled = pooled.view(
        batch, VJEPA_TEMPORAL_TOKENS, VJEPA_EMBED_DIM, spatial_size, spatial_size
    )
    return pooled.permute(0, 1, 3, 4, 2).reshape(
        batch, VJEPA_TEMPORAL_TOKENS * spatial_size**2, VJEPA_EMBED_DIM
    )


@torch.inference_mode()
def extract_compact_vjepa_tokens(encoder: nn.Module, video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 5 or video.shape[1:] != (
        3,
        VJEPA_FRAMES,
        VJEPA_RESOLUTION,
        VJEPA_RESOLUTION,
    ):
        raise ValueError("video must have shape [B,3,32,384,384]")
    tokens = encoder(video)
    if isinstance(tokens, (tuple, list)):
        tokens = tokens[-1]
    return compact_vjepa_tokens(tokens)
