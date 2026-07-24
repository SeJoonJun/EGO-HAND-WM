from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import nn

from ego_hand_wm.contracts.batch import CanonicalBatch
from ego_hand_wm.models.attention import (
    CachedContextAttentionBlock,
    ContextKVCache,
    SharedContextKVProjector,
)
from ego_hand_wm.models.encoders import ContextEncoder
from ego_hand_wm.models.flow_denoiser import StructuredFlowDenoiser
from ego_hand_wm.models.hand_kinematics import FutureHandKinematicsHead
from ego_hand_wm.models.time_embedding import TimeMLP


@dataclass
class WorldActionOutput:
    """Named model output with legacy five-value unpacking compatibility."""

    velocity: torch.Tensor
    geometry_hidden: torch.Tensor
    visual_velocity: torch.Tensor | None
    context_tokens: torch.Tensor
    context_valid: torch.Tensor
    hand_joints_local: torch.Tensor | None = None

    def __iter__(self) -> Iterator[torch.Tensor | None]:
        # Existing samplers and downstream code unpack five values.  Keep that API stable while
        # exposing optional auxiliary predictions by name.
        yield self.velocity
        yield self.geometry_hidden
        yield self.visual_velocity
        yield self.context_tokens
        yield self.context_valid


class FutureVisualFlowExpert(nn.Module):
    """Training-only DINO flow expert grounded in the same cache as geometry.

    Future visual tokens and future geometry tokens never attend one another.  Their only
    connection is the observed world-context representation, so visual targets cannot leak
    into geometry while the visual loss still trains the context encoder and shared K/V.
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        *,
        tokens_per_frame: int,
        heads: int,
        depth: int,
        mlp_ratio: float,
        dropout: float,
        physical_max_period: float,
        qk_norm: bool,
        qk_norm_eps: float,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.tokens_per_frame = tokens_per_frame
        self.input_projection = nn.Linear(latent_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, latent_dim)
        self.flow_time = TimeMLP(hidden_dim, max_period=10_000.0)
        self.physical_time = TimeMLP(hidden_dim, max_period=physical_max_period)
        self.token_embedding = nn.Parameter(
            torch.zeros(1, 1, tokens_per_frame, hidden_dim)
        )
        nn.init.normal_(self.token_embedding, std=0.02)
        self.blocks = nn.ModuleList(
            CachedContextAttentionBlock(
                hidden_dim,
                heads,
                mlp_ratio,
                dropout,
                qk_norm=qk_norm,
                qk_norm_eps=qk_norm_eps,
            )
            for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        future_time: torch.Tensor,
        flow_time: torch.Tensor,
        future_valid: torch.Tensor,
        context_cache: ContextKVCache,
    ) -> torch.Tensor:
        if noisy_latent.ndim != 4 or noisy_latent.shape[2] != self.tokens_per_frame:
            raise ValueError(
                "Visual flow input must be [B,K,P,D] with "
                f"P={self.tokens_per_frame}; got {tuple(noisy_latent.shape)}"
            )
        if context_cache.depth < len(self.blocks):
            raise ValueError(
                f"Context cache depth {context_cache.depth} is smaller than visual depth "
                f"{len(self.blocks)}"
            )
        tokens = (
            self.input_projection(noisy_latent)
            + self.physical_time(future_time).unsqueeze(2)
            + self.token_embedding.to(noisy_latent.dtype)
        )
        batch, steps, visual_tokens, hidden = tokens.shape
        tokens = tokens.reshape(batch, steps * visual_tokens, hidden)
        valid = future_valid.unsqueeze(-1).expand(-1, -1, visual_tokens).reshape(batch, -1)
        conditioning = self.flow_time(flow_time)
        for index, block in enumerate(self.blocks):
            context_key, context_value = context_cache.layer(index)
            tokens = block(
                tokens,
                conditioning,
                valid,
                context_key,
                context_value,
                context_cache.valid,
            )
        tokens = self.final_norm(tokens).masked_fill(~valid.unsqueeze(-1), 0.0)
        return self.output_projection(
            tokens.reshape(batch, steps, visual_tokens, hidden)
        )


class WorldActionModel(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.context_encoder = ContextEncoder(config)
        self.denoiser = StructuredFlowDenoiser(config)
        hidden_dim = int(config["hidden_dim"])
        heads = int(config["heads"])
        geometry_depth = int(config["depth"])
        latent_dim = int(config.get("future_visual_latent_dim", 0))
        visual_depth = int(config.get("future_visual_depth", geometry_depth))
        self.future_visual_expert = (
            FutureVisualFlowExpert(
                hidden_dim,
                latent_dim,
                tokens_per_frame=int(config.get("future_visual_tokens", 17)),
                heads=heads,
                depth=visual_depth,
                mlp_ratio=float(config.get("mlp_ratio", 4.0)),
                dropout=float(config.get("dropout", 0.0)),
                physical_max_period=float(config.get("physical_max_period", 10.0)),
                qk_norm=bool(config.get("qk_norm", True)),
                qk_norm_eps=float(config.get("qk_norm_eps", 1e-6)),
            )
            if latent_dim > 0
            else None
        )
        cache_depth = max(geometry_depth, visual_depth if latent_dim > 0 else 0)
        self.context_kv_projector = SharedContextKVProjector(
            hidden_dim,
            heads,
            cache_depth,
            qk_norm=bool(config.get("qk_norm", True)),
            qk_norm_eps=float(config.get("qk_norm_eps", 1e-6)),
        )
        hand_config = dict(config.get("hand_kinematics", {}))
        self.hand_kinematics_head = (
            FutureHandKinematicsHead(
                hidden_dim,
                heads=int(hand_config.get("heads", heads)),
                depth=int(hand_config.get("depth", 1)),
                mlp_ratio=float(hand_config.get("mlp_ratio", 2.0)),
                dropout=float(hand_config.get("dropout", config.get("dropout", 0.0))),
            )
            if bool(hand_config.get("enabled", False))
            else None
        )

    def encode_context(self, batch: CanonicalBatch) -> tuple[torch.Tensor, torch.Tensor]:
        return self.context_encoder(batch)

    def discard_future_visual_expert(self) -> None:
        """Remove training-only visual parameters after loading a pretraining checkpoint."""
        self.future_visual_expert = None

    def prefill_context(
        self,
        batch: CanonicalBatch,
        *,
        context: torch.Tensor | None = None,
        context_valid: torch.Tensor | None = None,
    ) -> ContextKVCache:
        """Encode one observation and project reusable K/V for every flow layer."""
        if (context is None) != (context_valid is None):
            raise ValueError("context and context_valid must be provided together")
        if context is None or context_valid is None:
            context, context_valid = self.encode_context(batch)
        return self.context_kv_projector(context, context_valid)

    def forward(
        self,
        batch: CanonicalBatch,
        noisy_state: torch.Tensor,
        flow_time: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        context_valid: torch.Tensor | None = None,
        context_cache: ContextKVCache | None = None,
        noisy_visual_latent: torch.Tensor | None = None,
        visual_flow_time: torch.Tensor | None = None,
        compute_hand_joints: bool | None = None,
    ) -> WorldActionOutput:
        if context_cache is not None and (context is not None or context_valid is not None):
            raise ValueError("Pass either context_cache or raw context tensors, not both")
        if context_cache is None:
            context_cache = self.prefill_context(
                batch, context=context, context_valid=context_valid
            )
        velocity, hidden = self.denoiser(
            noisy_state,
            batch.future_time,
            flow_time,
            batch.future_query_stream_mask,
            context_cache,
        )
        visual_velocity = None
        if noisy_visual_latent is not None:
            if self.future_visual_expert is None:
                raise ValueError("Received visual latent input while the visual expert is disabled")
            visual_velocity = self.future_visual_expert(
                noisy_visual_latent,
                batch.future_time,
                flow_time if visual_flow_time is None else visual_flow_time,
                batch.future_query_stream_mask.any(dim=-1),
                context_cache,
            )
        hand_joints_local = None
        if compute_hand_joints is None:
            compute_hand_joints = self.training
        if compute_hand_joints and self.hand_kinematics_head is not None:
            if batch.history_hand_joints_local is None:
                raise ValueError(
                    "The hand-kinematics head requires history_hand_joints_local"
                )
            hand_joints_local = self.hand_kinematics_head(
                hidden,
                batch.history_hand_joints_local,
                batch.future_query_stream_mask[..., [3, 4]],
                batch.history_stream_mask[:, -1, [3, 4]],
            )
        return WorldActionOutput(
            velocity=velocity,
            geometry_hidden=hidden,
            visual_velocity=visual_velocity,
            context_tokens=context_cache.tokens,
            context_valid=context_cache.valid,
            hand_joints_local=hand_joints_local,
        )
