from __future__ import annotations

from dataclasses import dataclass

import torch

from ego_hand_wm.contracts.batch import CanonicalBatch
from ego_hand_wm.contracts.schema import SCHEMA
from ego_hand_wm.models.world_action_model import WorldActionModel


@dataclass
class FlowTrainingSample:
    noisy_state: torch.Tensor
    noise: torch.Tensor
    target_velocity: torch.Tensor
    flow_time: torch.Tensor


def make_flow_training_sample(batch: CanonicalBatch) -> FlowTrainingSample:
    query_mask = SCHEMA.expand_stream_mask(batch.future_query_stream_mask).to(
        batch.future_state.dtype
    )
    noise = torch.randn_like(batch.future_state) * query_mask
    clean = batch.future_state * query_mask
    flow_time = torch.rand(batch.batch_size, device=clean.device, dtype=clean.dtype)
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
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive")
    mask = SCHEMA.expand_stream_mask(batch.future_query_stream_mask).to(batch.future_state.dtype)
    state = torch.randn_like(batch.future_state) * mask
    context, context_valid = model.encode_context(batch)
    step_size = 1.0 / steps
    for index in range(steps):
        time = torch.full(
            (batch.batch_size,), index / steps, device=state.device, dtype=state.dtype
        )
        velocity, _, _, _, _ = model(
            batch, state, time, context=context, context_valid=context_valid
        )
        if method == "euler" or index == steps - 1:
            state = state + step_size * velocity
        elif method == "heun":
            proposal = state + step_size * velocity
            next_time = torch.full_like(time, (index + 1) / steps)
            next_velocity, _, _, _, _ = model(
                batch, proposal, next_time, context=context, context_valid=context_valid
            )
            state = state + 0.5 * step_size * (velocity + next_velocity)
        else:
            raise ValueError(f"Unknown ODE method: {method}")
        state = state * mask
    return state
