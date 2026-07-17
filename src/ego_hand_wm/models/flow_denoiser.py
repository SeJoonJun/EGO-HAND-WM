"""Structured DiT-like flow denoiser over camera, wrist, and hand entity tokens."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ego_hand_wm.contracts.schema import SCHEMA, STREAM_DIMS, STREAM_NAMES
from ego_hand_wm.models.time_embedding import TimeMLP


def _modulate(value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return value * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNCrossBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.norm_mlp = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        inner = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, inner), nn.GELU(), nn.Dropout(dropout), nn.Linear(inner, hidden_dim)
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        tokens: torch.Tensor,
        conditioning: torch.Tensor,
        token_valid: torch.Tensor,
        context: torch.Tensor,
        context_valid: torch.Tensor,
    ) -> torch.Tensor:
        shift_self, scale_self, gate_self, shift_mlp, scale_mlp, gate_mlp = self.modulation(
            conditioning
        ).chunk(6, dim=-1)
        normalized = _modulate(self.norm_self(tokens), shift_self, scale_self)
        attended = self.self_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~token_valid,
            need_weights=False,
        )[0]
        tokens = tokens + gate_self.unsqueeze(1) * attended
        cross = self.cross_attention(
            self.norm_cross(tokens),
            context,
            context,
            key_padding_mask=~context_valid,
            need_weights=False,
        )[0]
        tokens = tokens + cross
        feedforward = self.mlp(_modulate(self.norm_mlp(tokens), shift_mlp, scale_mlp))
        tokens = tokens + gate_mlp.unsqueeze(1) * feedforward
        return tokens.masked_fill(~token_valid.unsqueeze(-1), 0.0)


class StructuredFlowDenoiser(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        hidden_dim = int(config["hidden_dim"])
        self.hidden_dim = hidden_dim
        self.input_projections = nn.ModuleList(
            [nn.Linear(width, hidden_dim) for width in STREAM_DIMS]
        )
        self.output_projections = nn.ModuleList(
            [nn.Linear(hidden_dim, width) for width in STREAM_DIMS]
        )
        self.entity_embedding = nn.Parameter(torch.randn(len(STREAM_NAMES), hidden_dim) * 0.02)
        self.physical_time = TimeMLP(
            hidden_dim, max_period=float(config.get("physical_max_period", 10.0))
        )
        self.flow_time = TimeMLP(hidden_dim, max_period=10_000.0)
        self.blocks = nn.ModuleList(
            [
                AdaLNCrossBlock(
                    hidden_dim,
                    int(config["heads"]),
                    float(config.get("mlp_ratio", 4.0)),
                    float(config.get("dropout", 0.0)),
                )
                for _ in range(int(config["depth"]))
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        for projection in self.output_projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(
        self,
        noisy_state: torch.Tensor,
        physical_time: torch.Tensor,
        flow_time: torch.Tensor,
        stream_mask: torch.Tensor,
        context: torch.Tensor,
        context_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        split = SCHEMA.split(noisy_state)
        entities = [
            projection(split[name])
            for name, projection in zip(STREAM_NAMES, self.input_projections, strict=True)
        ]
        tokens = torch.stack(entities, dim=2)
        tokens = tokens + self.entity_embedding.view(1, 1, len(STREAM_NAMES), -1)
        tokens = tokens + self.physical_time(physical_time).unsqueeze(2)
        batch, future_steps, entity_count, hidden_dim = tokens.shape
        tokens = tokens.reshape(batch, future_steps * entity_count, hidden_dim)
        token_valid = stream_mask.reshape(batch, future_steps * entity_count)
        conditioning = self.flow_time(flow_time)
        for block in self.blocks:
            tokens = block(tokens, conditioning, token_valid, context, context_valid)
        tokens = self.final_norm(tokens).reshape(batch, future_steps, entity_count, hidden_dim)
        velocity_parts = [
            projection(tokens[:, :, index])
            for index, projection in enumerate(self.output_projections)
        ]
        velocity = SCHEMA.pack(*velocity_parts)
        velocity = velocity * SCHEMA.expand_stream_mask(stream_mask).to(velocity.dtype)
        return velocity, tokens

