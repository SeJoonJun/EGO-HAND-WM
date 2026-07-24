"""Lightweight 3D hand-kinematics auxiliary head.

The canonical flow state predicts MANO rotations, but a geodesic angle alone does not expose
whether those rotations preserve fingertips or the articulated hand shape.  This head reads the
same future wrist/finger-chain tokens and predicts all 21 joints in each instantaneous wrist
frame.  Its zero-initialized residual makes last-observation persistence the starting solution.
"""

from __future__ import annotations

import torch
from torch import nn


HAND_ENTITY_INDICES = (
    (1, 3, 4, 5, 6, 7),
    (2, 8, 9, 10, 11, 12),
)


class FutureHandKinematicsHead(nn.Module):
    """Decode wrist-local 21-joint positions from anatomical geometry tokens."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        heads: int,
        depth: int = 1,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth < 0:
            raise ValueError("Hand-kinematics depth must be non-negative")
        self.joint_queries = nn.Parameter(torch.randn(1, 1, 21, hidden_dim) * 0.02)
        self.side_embedding = nn.Parameter(torch.randn(1, 2, 1, hidden_dim) * 0.02)
        self.source_norm = nn.LayerNorm(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        feedforward = max(hidden_dim, int(hidden_dim * mlp_ratio))
        self.joint_blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        geometry_tokens: torch.Tensor,
        history_hand_joints_local: torch.Tensor,
        future_hand_mask: torch.Tensor,
        history_hand_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if geometry_tokens.ndim != 4 or geometry_tokens.shape[2] != 13:
            raise ValueError("geometry_tokens must be [B,F,13,D]")
        if history_hand_joints_local.ndim != 5 or history_hand_joints_local.shape[2:] != (
            2,
            21,
            3,
        ):
            raise ValueError("history_hand_joints_local must be [B,H,2,21,3]")
        batch, future_steps, _, hidden_dim = geometry_tokens.shape
        if history_hand_joints_local.shape[0] != batch:
            raise ValueError("History joints and geometry tokens must have the same batch size")
        if future_hand_mask.shape != (batch, future_steps, 2):
            raise ValueError("future_hand_mask must be [B,F,2]")

        sources = torch.stack(
            [geometry_tokens[:, :, indices, :] for indices in HAND_ENTITY_INDICES],
            dim=2,
        )
        sources = sources.reshape(batch * future_steps * 2, 6, hidden_dim)
        queries = self.joint_queries + self.side_embedding
        queries = queries[:, None].expand(batch, future_steps, 2, 21, hidden_dim)
        queries = queries.reshape(batch * future_steps * 2, 21, hidden_dim)
        attended, _ = self.cross_attention(
            self.query_norm(queries),
            self.source_norm(sources),
            self.source_norm(sources),
            need_weights=False,
        )
        joints = queries + attended
        for block in self.joint_blocks:
            joints = block(joints)
        residual = self.output(self.final_norm(joints)).reshape(
            batch, future_steps, 2, 21, 3
        )

        baseline = history_hand_joints_local[:, -1]
        if history_hand_mask is not None:
            if history_hand_mask.shape != (batch, 2):
                raise ValueError("history_hand_mask must be [B,2]")
            baseline = baseline * history_hand_mask[:, :, None, None].to(baseline.dtype)
        prediction = baseline[:, None] + residual
        return prediction * future_hand_mask[:, :, :, None, None].to(prediction.dtype)
