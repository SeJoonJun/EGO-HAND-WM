from __future__ import annotations

import math

import torch


def sinusoidal_time_embedding(time: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Embed arbitrary real-valued seconds or flow coordinates without conflating the two."""
    if dim < 2:
        raise ValueError("Time embedding dimension must be at least 2")
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=time.device)
        / max(half - 1, 1)
    )
    phase = time.float().unsqueeze(-1) * frequencies
    embedding = torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)
    if dim % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


class TimeMLP(torch.nn.Module):
    def __init__(self, hidden_dim: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_period = max_period
        self.network = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        return self.network(sinusoidal_time_embedding(time, self.hidden_dim, self.max_period))

