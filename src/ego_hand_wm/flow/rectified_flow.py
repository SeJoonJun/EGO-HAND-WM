from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from ego_hand_wm.contracts.batch import CanonicalBatch
from ego_hand_wm.contracts.schema import SCHEMA
from ego_hand_wm.models.world_action_model import WorldActionModel
from ego_hand_wm.models.attention import ContextKVCache


@dataclass
class FlowTrainingSample:
    noisy_state: torch.Tensor
    noise: torch.Tensor
    target_velocity: torch.Tensor
    flow_time: torch.Tensor


@dataclass
class VisualFlowTrainingSample:
    noisy_latent: torch.Tensor
    noise: torch.Tensor
    clean_latent: torch.Tensor
    target_velocity: torch.Tensor


def sample_flow_time(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    config: dict | None = None,
) -> torch.Tensor:
    """Sample configurable continuous flow time in the model's noise->clean convention."""
    options = dict(config or {})
    distribution = str(options.get("distribution", "uniform"))
    if distribution == "uniform":
        time = torch.rand(batch_size, device=device, dtype=torch.float32)
    elif distribution == "logit_normal":
        mean = float(options.get("mean", 0.0))
        std = float(options.get("std", 1.0))
        if std <= 0:
            raise ValueError("logit-normal flow-time std must be positive")
        time = torch.sigmoid(
            torch.randn(batch_size, device=device, dtype=torch.float32) * std + mean
        )
    else:
        raise ValueError(f"Unknown flow-time distribution: {distribution!r}")
    shift = float(options.get("shift", 1.0))
    if shift <= 0:
        raise ValueError("flow-time shift must be positive")
    time = shift * time / (1.0 + (shift - 1.0) * time)
    epsilon = float(options.get("epsilon", 0.0))
    if epsilon < 0 or epsilon >= 0.5:
        raise ValueError("flow-time epsilon must lie in [0, 0.5)")
    if epsilon:
        time = time.clamp(epsilon, 1.0 - epsilon)
    return time.to(dtype=dtype)


def normalize_visual_flow_target(
    latent: torch.Tensor,
    *,
    mode: str = "none",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize a future visual target without altering observed context features.

    Per-token LayerNorm is applied independently to every ``[D]`` class or spatial token.
    It has no learned affine parameters and is performed in fp32 for stable statistics before
    restoring the input dtype.  This places semantic latents on a unit-variance scale suitable
    for interpolation with unit Gaussian flow noise.
    """
    if mode == "none":
        return latent
    if mode != "per_token_layer_norm":
        raise ValueError(f"Unknown visual target normalization: {mode!r}")
    if latent.ndim != 4 or latent.shape[-1] <= 1:
        raise ValueError("Visual targets must be [B,K,P,D] with D > 1")
    if eps <= 0:
        raise ValueError("Visual target normalization epsilon must be positive")
    normalized = functional.layer_norm(
        latent.float(),
        (latent.shape[-1],),
        weight=None,
        bias=None,
        eps=eps,
    )
    return normalized.to(latent.dtype)


def make_visual_flow_training_sample(
    clean_latent: torch.Tensor,
    flow_time: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    normalization: str = "none",
    normalization_eps: float = 1e-6,
) -> VisualFlowTrainingSample:
    if clean_latent.ndim != 4:
        raise ValueError("Visual targets must be [B,K,P,D]")
    if flow_time.shape != clean_latent.shape[:1]:
        raise ValueError("flow_time must have shape [B]")
    if valid_mask.shape != clean_latent.shape[:2]:
        raise ValueError("Visual valid_mask must have shape [B,K]")
    mask = valid_mask[..., None, None].to(clean_latent.dtype)
    clean = normalize_visual_flow_target(
        clean_latent,
        mode=normalization,
        eps=normalization_eps,
    ) * mask
    noise = torch.randn_like(clean) * mask
    interpolation = flow_time[:, None, None, None]
    noisy = ((1.0 - interpolation) * noise + interpolation * clean) * mask
    target_velocity = (clean - noise) * mask
    return VisualFlowTrainingSample(noisy, noise, clean, target_velocity)


def make_flow_training_sample(
    batch: CanonicalBatch, *, time_config: dict | None = None
) -> FlowTrainingSample:
    query_mask = SCHEMA.expand_stream_mask(batch.future_query_stream_mask).to(
        batch.future_state.dtype
    )
    noise = torch.randn_like(batch.future_state) * query_mask
    clean = batch.future_state * query_mask
    flow_time = sample_flow_time(
        batch.batch_size,
        device=clean.device,
        dtype=clean.dtype,
        config=time_config,
    )
    interpolation = flow_time[:, None, None]
    # The flow input depends only on requested query capabilities, never on GT validity. Dataset
    # adapters must safe-fill unavailable state values; supervision validity is used only by loss.
    noisy_state = (1.0 - interpolation) * noise + interpolation * clean
    target_velocity = clean - noise
    return FlowTrainingSample(noisy_state, noise, target_velocity, flow_time)


@torch.no_grad()
def sample_ode(
    model: WorldActionModel,
    batch: CanonicalBatch,
    steps: int = 16,
    method: str = "heun",
    initial_state: torch.Tensor | None = None,
    context_cache: ContextKVCache | None = None,
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive")
    mask = SCHEMA.expand_stream_mask(batch.future_query_stream_mask).to(batch.future_state.dtype)
    if initial_state is None:
        state = torch.randn_like(batch.future_state)
    else:
        if initial_state.shape != batch.future_state.shape:
            raise ValueError("initial_state must match batch.future_state")
        state = initial_state.to(
            device=batch.future_state.device,
            dtype=batch.future_state.dtype,
        ).clone()
    state = state * mask
    if context_cache is None:
        context_cache = model.prefill_context(batch)
    step_size = 1.0 / steps
    for index in range(steps):
        time = torch.full(
            (batch.batch_size,), index / steps, device=state.device, dtype=state.dtype
        )
        velocity, _, _, _, _ = model(
            batch, state, time, context_cache=context_cache
        )
        if method == "euler" or index == steps - 1:
            state = state + step_size * velocity
        elif method == "heun":
            proposal = state + step_size * velocity
            next_time = torch.full_like(time, (index + 1) / steps)
            next_velocity, _, _, _, _ = model(
                batch, proposal, next_time, context_cache=context_cache
            )
            state = state + 0.5 * step_size * (velocity + next_velocity)
        else:
            raise ValueError(f"Unknown ODE method: {method}")
        state = state * mask
    return state
