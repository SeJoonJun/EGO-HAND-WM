"""Flow denoiser over anatomical camera, wrist, and MANO-chain entity tokens."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ego_hand_wm.contracts.schema import ENTITY_DIMS, ENTITY_NAMES, SCHEMA
from ego_hand_wm.models.attention import CachedContextAttentionBlock, ContextKVCache
from ego_hand_wm.models.time_embedding import TimeMLP


class StructuredFlowDenoiser(nn.Module):
    """Predict canonical geometry flow while attending leakage-safe cached context."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        hidden_dim = int(config["hidden_dim"])
        self.hidden_dim = hidden_dim
        self.input_projections = nn.ModuleList(
            nn.Linear(width, hidden_dim) for width in ENTITY_DIMS
        )
        self.output_projections = nn.ModuleList(
            nn.Linear(hidden_dim, width) for width in ENTITY_DIMS
        )
        self.entity_embedding = nn.Parameter(
            torch.randn(len(ENTITY_NAMES), hidden_dim) * 0.02
        )
        self.physical_time = TimeMLP(
            hidden_dim, max_period=float(config.get("physical_max_period", 10.0))
        )
        self.flow_time = TimeMLP(hidden_dim, max_period=10_000.0)
        self.blocks = nn.ModuleList(
            CachedContextAttentionBlock(
                hidden_dim,
                int(config["heads"]),
                float(config.get("mlp_ratio", 4.0)),
                float(config.get("dropout", 0.0)),
                qk_norm=bool(config.get("qk_norm", True)),
                qk_norm_eps=float(config.get("qk_norm_eps", 1e-6)),
            )
            for _ in range(int(config["depth"]))
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
        context_cache: ContextKVCache,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context_cache.depth < len(self.blocks):
            raise ValueError(
                f"Context cache depth {context_cache.depth} is smaller than geometry depth "
                f"{len(self.blocks)}"
            )
        entities = [
            projection(entity)
            for entity, projection in zip(
                SCHEMA.split_entities(noisy_state), self.input_projections, strict=True
            )
        ]
        tokens = torch.stack(entities, dim=2)
        tokens = tokens + self.entity_embedding.view(1, 1, len(ENTITY_NAMES), -1)
        tokens = tokens + self.physical_time(physical_time).unsqueeze(2)
        batch, future_steps, entity_count, hidden_dim = tokens.shape
        tokens = tokens.reshape(batch, future_steps * entity_count, hidden_dim)
        token_valid = SCHEMA.expand_entity_mask(stream_mask).reshape(
            batch, future_steps * entity_count
        )
        conditioning = self.flow_time(flow_time)
        for index, block in enumerate(self.blocks):
            context_key, context_value = context_cache.layer(index)
            tokens = block(
                tokens,
                conditioning,
                token_valid,
                context_key,
                context_value,
                context_cache.valid,
            )
        tokens = self.final_norm(tokens).masked_fill(~token_valid.unsqueeze(-1), 0.0)
        tokens = tokens.reshape(batch, future_steps, entity_count, hidden_dim)
        velocity_parts = [
            projection(tokens[:, :, index])
            for index, projection in enumerate(self.output_projections)
        ]
        velocity = SCHEMA.pack_entities(velocity_parts)
        velocity = velocity * SCHEMA.expand_stream_mask(stream_mask).to(velocity.dtype)
        return velocity, tokens
