from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ego_hand_wm.contracts.batch import CanonicalBatch
from ego_hand_wm.models.encoders import ContextEncoder
from ego_hand_wm.models.flow_denoiser import StructuredFlowDenoiser


class FutureVisualFlowHead(nn.Module):
    """One-way auxiliary: geometry hidden states condition visual flow, never the reverse."""

    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, latent_dim),
        )

    def forward(self, noisy_latent: torch.Tensor, geometry_hidden: torch.Tensor) -> torch.Tensor:
        pooled_geometry = geometry_hidden.mean(dim=2)
        return self.network(torch.cat((noisy_latent, pooled_geometry), dim=-1))


class WorldActionModel(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.context_encoder = ContextEncoder(config)
        self.denoiser = StructuredFlowDenoiser(config)
        latent_dim = int(config.get("future_visual_latent_dim", 0))
        self.future_visual_head = (
            FutureVisualFlowHead(int(config["hidden_dim"]), latent_dim) if latent_dim > 0 else None
        )

    def encode_context(self, batch: CanonicalBatch) -> tuple[torch.Tensor, torch.Tensor]:
        return self.context_encoder(batch)

    def forward(
        self,
        batch: CanonicalBatch,
        noisy_state: torch.Tensor,
        flow_time: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        context_valid: torch.Tensor | None = None,
        noisy_visual_latent: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
    ]:
        if context is None or context_valid is None:
            context, context_valid = self.encode_context(batch)
        velocity, hidden = self.denoiser(
            noisy_state,
            batch.future_time,
            flow_time,
            batch.future_query_stream_mask,
            context,
            context_valid,
        )
        visual_velocity = None
        if noisy_visual_latent is not None:
            if self.future_visual_head is None:
                raise ValueError("Received visual latent input while the visual head is disabled")
            visual_velocity = self.future_visual_head(noisy_visual_latent, hidden)
        return velocity, hidden, visual_velocity, context, context_valid
