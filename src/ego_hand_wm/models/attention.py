"""Leakage-safe mixed attention and reusable observed-context K/V caches."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn


class HeadRMSNorm(nn.Module):
    """RMS-normalize each attention head in fp32, then restore the activation dtype."""

    def __init__(self, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if eps <= 0:
            raise ValueError("RMSNorm epsilon must be positive")
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        dtype = value.dtype
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(dtype) * self.weight.to(dtype)


@dataclass(frozen=True)
class ContextKVCache:
    """Observed context plus its per-layer projected keys and values."""

    tokens: torch.Tensor
    valid: torch.Tensor
    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]

    @property
    def depth(self) -> int:
        return len(self.keys)

    def layer(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.depth:
            raise IndexError(f"Context cache has {self.depth} layers, requested {index}")
        return self.keys[index], self.values[index]


class SharedContextKVProjector(nn.Module):
    """Project one fixed observed representation into cache entries for every expert layer."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        depth: int,
        *,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        if depth <= 0:
            raise ValueError("Context K/V cache depth must be positive")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.projections = nn.ModuleList(
            nn.Linear(hidden_dim, 2 * hidden_dim) for _ in range(depth)
        )
        self.key_norms = nn.ModuleList(
            HeadRMSNorm(self.head_dim, qk_norm_eps) if qk_norm else nn.Identity()
            for _ in range(depth)
        )

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = value.shape
        return value.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, context: torch.Tensor, valid: torch.Tensor) -> ContextKVCache:
        if context.ndim != 3 or valid.shape != context.shape[:2]:
            raise ValueError("Context and validity must be [B,S,D] and [B,S]")
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for projection, key_norm in zip(self.projections, self.key_norms, strict=True):
            key, value = projection(context).chunk(2, dim=-1)
            keys.append(key_norm(self._split_heads(key)))
            values.append(self._split_heads(value))
        return ContextKVCache(context, valid, tuple(keys), tuple(values))


def _modulate(value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return value * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class CachedContextAttentionBlock(nn.Module):
    """Expert block whose queries jointly attend cached context and peer expert tokens.

    A single softmax is evaluated over ``[context K/V, expert K/V]``.  Context is never
    updated by the noisy expert tokens, so future targets cannot leak into the observed
    representation and the context projections can be reused across every flow step.
    """

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
        *,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.dropout = dropout
        self.norm_attention = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.query_norm = HeadRMSNorm(self.head_dim, qk_norm_eps) if qk_norm else nn.Identity()
        self.key_norm = HeadRMSNorm(self.head_dim, qk_norm_eps) if qk_norm else nn.Identity()
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm_mlp = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        inner = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_dim),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = value.shape
        return value.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        tokens: torch.Tensor,
        conditioning: torch.Tensor,
        token_valid: torch.Tensor,
        context_key: torch.Tensor,
        context_value: torch.Tensor,
        context_valid: torch.Tensor,
    ) -> torch.Tensor:
        if token_valid.shape != tokens.shape[:2]:
            raise ValueError("Expert validity must match token shape [B,S]")
        shift_attention, scale_attention, gate_attention, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation(conditioning).chunk(6, dim=-1)
        )
        normalized = _modulate(
            self.norm_attention(tokens), shift_attention, scale_attention
        )
        query, key, value = self.qkv(normalized).chunk(3, dim=-1)
        query = self.query_norm(self._split_heads(query))
        key = self.key_norm(self._split_heads(key))
        value = self._split_heads(value)
        if context_key.shape[:2] != (tokens.shape[0], self.heads):
            raise ValueError("Cached context K/V batch or head count does not match expert")
        key = torch.cat((context_key, key), dim=2)
        value = torch.cat((context_value, value), dim=2)
        key_valid = torch.cat((context_valid, token_valid), dim=1)
        # PyTorch SDPA boolean masks use True for locations that participate in attention.
        attention_mask = key_valid[:, None, None, :]
        attended = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(tokens.shape)
        tokens = tokens + gate_attention.unsqueeze(1) * self.output(attended)
        feedforward = self.mlp(_modulate(self.norm_mlp(tokens), shift_mlp, scale_mlp))
        tokens = tokens + gate_mlp.unsqueeze(1) * feedforward
        return tokens.masked_fill(~token_valid.unsqueeze(-1), 0.0)
