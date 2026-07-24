"""Past/future geometry tokens for Assembly101 oracle anticipation diagnostics.

The encoder follows the useful parts of the supplied STA hand-pose reference: input
normalization, a per-timestep projection, learned past/future type embeddings, masked temporal
self-attention, and a zero-initialized cross-attention residual.  Assembly101 semantic
anticipation has no object-proposal branch, so verb/object/action queries replace ROI features.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from ego_hand_wm.models.time_embedding import TimeMLP


GeometryMode = Literal[
    "camera",
    "wrist",
    "handpose",
    "whole_hand",
    "camera_wrist",
    "camera_handpose",
    "camera_whole_hand",
]
GEOMETRY_INPUT_DIMS: dict[str, int] = {
    "camera": 9,
    "wrist": 2 * 9 + 2,
    "handpose": 2 * 21 * 3 + 2,
    "whole_hand": 2 * 9 + 2 * 21 * 3 + 2 + 2,
    # Camera, two pose-9 roots, and two released validity indicators.
    "camera_wrist": 9 + 2 * 9 + 2,
    # Camera, two wrist-local 21x3 poses, and two validity indicators.
    "camera_handpose": 9 + 2 * 21 * 3 + 2,
    # Camera, wrist roots, local poses, wrist validity, and pose validity.
    "camera_whole_hand": 9 + 2 * 9 + 2 * 21 * 3 + 2 + 2,
}


def assemble_geometry_sequence(
    mode: GeometryMode,
    *,
    camera_pose: torch.Tensor,
    wrist_pose: torch.Tensor,
    hand_pose: torch.Tensor,
    wrist_valid: torch.Tensor,
    hand_pose_valid: torch.Tensor,
    future_mask: torch.Tensor,
    include_future: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack one requested geometry ablation into one token vector per timestamp.

    Args:
        camera_pose: Anchored camera pose-9, ``[B,T,9]``.
        wrist_pose: Two anchored wrist pose-9 streams, ``[B,T,2,9]``.
        hand_pose: Two wrist-local 21-joint poses, ``[B,T,2,21,3]``.
        wrist_valid: Released wrist validity, ``[B,T,2]``.
        hand_pose_valid: Landmark validity, ``[B,T,2]``.
        future_mask: True for oracle-future timestamps, ``[B,T]``.
        include_future: False constructs the matched past-only control.

    Returns:
        Packed values ``[B,T,D_mode]`` and a valid-timestep mask ``[B,T]``.
    """

    if mode not in GEOMETRY_INPUT_DIMS:
        raise ValueError(f"Unsupported geometry mode: {mode}")
    if camera_pose.ndim != 3 or camera_pose.shape[-1] != 9:
        raise ValueError("camera_pose must have shape [B,T,9]")
    batch, steps, _ = camera_pose.shape
    if wrist_pose.shape != (batch, steps, 2, 9):
        raise ValueError("wrist_pose must have shape [B,T,2,9]")
    if hand_pose.shape != (batch, steps, 2, 21, 3):
        raise ValueError("hand_pose must have shape [B,T,2,21,3]")
    if wrist_valid.shape != (batch, steps, 2):
        raise ValueError("wrist_valid must have shape [B,T,2]")
    if hand_pose_valid.shape != (batch, steps, 2):
        raise ValueError("hand_pose_valid must have shape [B,T,2]")
    if future_mask.shape != (batch, steps):
        raise ValueError("future_mask must have shape [B,T]")

    wrist_valid = wrist_valid.to(device=camera_pose.device, dtype=torch.bool)
    hand_pose_valid = hand_pose_valid.to(device=camera_pose.device, dtype=torch.bool)
    future_mask = future_mask.to(device=camera_pose.device, dtype=torch.bool)
    wrist = torch.where(wrist_valid[..., None], wrist_pose, torch.zeros_like(wrist_pose))
    pose = torch.where(
        hand_pose_valid[..., None, None], hand_pose, torch.zeros_like(hand_pose)
    )

    include_camera = mode.startswith("camera")
    include_wrist = mode in {"wrist", "whole_hand", "camera_wrist", "camera_whole_hand"}
    include_handpose = mode in {
        "handpose",
        "whole_hand",
        "camera_handpose",
        "camera_whole_hand",
    }
    parts: list[torch.Tensor] = []
    if include_camera:
        parts.append(camera_pose)
    if include_wrist:
        parts.extend((wrist.flatten(start_dim=2), wrist_valid.to(wrist.dtype)))
    if include_handpose:
        parts.extend((pose.flatten(start_dim=2), hand_pose_valid.to(pose.dtype)))
    values = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    if include_camera:
        # Camera-inclusive rows remain informative when one or both hands are untracked.
        valid = torch.isfinite(camera_pose).all(dim=-1)
    else:
        valid = torch.zeros((batch, steps), dtype=torch.bool, device=camera_pose.device)
        if include_wrist:
            valid = valid | wrist_valid.any(dim=-1)
        if include_handpose:
            valid = valid | hand_pose_valid.any(dim=-1)

    if not include_future:
        valid = valid & ~future_mask
    values = torch.nan_to_num(values.float())
    values = values.masked_fill(~valid.unsqueeze(-1), 0.0)
    expected = GEOMETRY_INPUT_DIMS[mode]
    if values.shape[-1] != expected:
        raise RuntimeError(f"Packed {mode} dim {values.shape[-1]} does not match {expected}")
    return values, valid


class PastFutureGeometryEncoder(nn.Module):
    """Encode an ordered history-plus-oracle sequence without pooling away time."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = 512,
        depth: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_frames: int = 48,
        use_input_norm: bool = True,
        geometry_drop_prob: float = 0.0,
        physical_max_period: float = 10.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        self.input_dim = int(input_dim)
        self.max_frames = int(max_frames)
        self.geometry_drop_prob = max(0.0, min(1.0, float(geometry_drop_prob)))
        self.input_norm = nn.LayerNorm(self.input_dim) if use_input_norm else nn.Identity()
        self.input_projection = nn.Linear(self.input_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.max_frames, hidden_dim))
        # Observed history, oracle pre-action gap, and oracle target execution.
        self.phase_embedding = nn.Parameter(torch.zeros(1, 3, hidden_dim))
        self.physical_time = TimeMLP(hidden_dim, max_period=physical_max_period)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.phase_embedding, std=0.02)

    def forward_tokens(
        self,
        geometry: torch.Tensor,
        geometry_mask: torch.Tensor,
        time_seconds: torch.Tensor,
        future_mask: torch.Tensor,
        execution_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if geometry.ndim != 3 or geometry.shape[-1] != self.input_dim:
            raise ValueError(f"geometry must have shape [B,T,{self.input_dim}]")
        batch, frames, _ = geometry.shape
        if frames > self.max_frames:
            raise ValueError(f"geometry has {frames} frames, max_frames={self.max_frames}")
        expected = (batch, frames)
        if geometry_mask.shape != expected or time_seconds.shape != expected:
            raise ValueError("geometry_mask and time_seconds must have shape [B,T]")
        if future_mask.shape != expected:
            raise ValueError("future_mask must have shape [B,T]")
        if execution_mask is not None and execution_mask.shape != expected:
            raise ValueError("execution_mask must have shape [B,T]")

        mask = geometry_mask.to(device=geometry.device, dtype=torch.bool)
        future_mask = future_mask.to(device=geometry.device, dtype=torch.bool)
        if execution_mask is None:
            execution_mask = torch.zeros_like(future_mask)
        else:
            execution_mask = execution_mask.to(device=geometry.device, dtype=torch.bool)
        if bool((execution_mask & ~future_mask).any().item()):
            raise ValueError("execution timestamps must also be marked as future")
        time_seconds = time_seconds.to(device=geometry.device, dtype=torch.float32)
        values = torch.nan_to_num(geometry.float())
        if self.training and self.geometry_drop_prob > 0.0:
            drop = torch.rand(batch, device=geometry.device) < self.geometry_drop_prob
            values = values.masked_fill(drop.view(batch, 1, 1), 0.0)
            mask = mask & ~drop.view(batch, 1)

        # TransformerEncoder cannot receive an all-masked row. The temporary safe token is
        # removed from the returned mask and zeroed from the returned representation.
        empty = ~mask.any(dim=1)
        safe_mask = mask.clone()
        if empty.any():
            safe_mask[empty, 0] = True

        tokens = self.input_projection(self.input_norm(values))
        tokens = tokens + self.position_embedding[:, :frames].to(tokens.dtype)
        phase_ids = future_mask.long()
        phase_ids = torch.where(execution_mask, torch.full_like(phase_ids, 2), phase_ids)
        tokens = tokens + self.phase_embedding[:, phase_ids].squeeze(0).to(tokens.dtype)
        tokens = tokens + self.physical_time(time_seconds).to(tokens.dtype)
        tokens = self.encoder(tokens, src_key_padding_mask=~safe_mask)
        tokens = self.output(tokens)
        tokens = tokens.masked_fill(~mask.unsqueeze(-1), 0.0)
        return tokens, mask

    def forward(
        self,
        geometry: torch.Tensor,
        geometry_mask: torch.Tensor,
        time_seconds: torch.Tensor,
        future_mask: torch.Tensor,
        execution_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens, mask = self.forward_tokens(
            geometry, geometry_mask, time_seconds, future_mask, execution_mask
        )
        weights = mask.to(tokens.dtype).unsqueeze(-1)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class SemanticGeometryCrossAttention(nn.Module):
    """Zero-initialized geometry residual for verb/object/action query tokens."""

    def __init__(self, dim: int, *, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.geometry_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        semantic_queries: torch.Tensor,
        geometry_tokens: torch.Tensor,
        geometry_mask: torch.Tensor,
    ) -> torch.Tensor:
        if semantic_queries.ndim != 3 or geometry_tokens.ndim != 3:
            raise ValueError("semantic_queries and geometry_tokens must be rank-3")
        if semantic_queries.shape[0] != geometry_tokens.shape[0]:
            raise ValueError("semantic and geometry batch sizes must match")
        if semantic_queries.shape[-1] != geometry_tokens.shape[-1]:
            raise ValueError("semantic and geometry widths must match")
        expected = geometry_tokens.shape[:2]
        if geometry_mask.shape != expected:
            raise ValueError("geometry_mask must have shape [B,T]")

        mask = geometry_mask.to(device=geometry_tokens.device, dtype=torch.bool)
        empty = ~mask.any(dim=1)
        safe_mask = mask.clone()
        if empty.any():
            safe_mask[empty, 0] = True
        residual, _ = self.cross_attention(
            self.query_norm(semantic_queries),
            self.geometry_norm(geometry_tokens),
            self.geometry_norm(geometry_tokens),
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        residual = self.output(residual)
        residual = residual.masked_fill(empty.view(-1, 1, 1), 0.0)
        return semantic_queries + residual
