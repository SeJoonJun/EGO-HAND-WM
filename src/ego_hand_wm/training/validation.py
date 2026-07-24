"""Deterministic, source-stratified validation for VITRA world-action pretraining."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
import torch
import torch.distributed as distributed

from ego_hand_wm.contracts.batch import CanonicalBatch
from ego_hand_wm.contracts.schema import SCHEMA
from ego_hand_wm.flow.rectified_flow import normalize_visual_flow_target, sample_ode
from ego_hand_wm.geometry.rotations import rotation_6d_to_matrix, so3_geodesic_angle


REPORTING_SOURCES = ("ego4d", "egoexo4d", "epic", "ssv2")
METRICS = (
    "camera_translation_cm",
    "camera_rotation_deg",
    "wrist_translation_cm",
    "wrist_rotation_deg",
    "mano_rotation_deg",
    "hand_joint_mpjpe_cm",
    "fingertip_mpjpe_cm",
    "geometry_flow_mse",
    "future_visual_flow_mse",
    "future_visual_cosine",
    "horizon_seconds",
)


def reporting_source(logical_source: str) -> str:
    if logical_source in {"ego4d_cooking_and_cleaning", "ego4d_other"}:
        return "ego4d"
    if logical_source not in REPORTING_SOURCES:
        raise ValueError(f"Unknown VITRA reporting source: {logical_source!r}")
    return logical_source


def _member_seed(member_name: str, namespace: str) -> int:
    digest = hashlib.sha256(f"42\0{namespace}\0{member_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _deterministic_noise(
    batch: CanonicalBatch, shape: torch.Size, *, namespace: str
) -> torch.Tensor:
    if batch.metadata is None or len(batch.metadata) != batch.batch_size:
        raise ValueError("Validation requires one metadata record per sample")
    samples: list[torch.Tensor] = []
    for metadata in batch.metadata:
        member = str(metadata.get("archive_member") or metadata.get("sample_id") or "")
        if not member:
            raise ValueError("Validation metadata is missing archive_member/sample_id")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_member_seed(member, namespace))
        samples.append(torch.randn(tuple(shape[1:]), generator=generator, dtype=torch.float32))
    return torch.stack(samples).to(device=batch.future_state.device, dtype=batch.future_state.dtype)


def _add(
    statistics: torch.Tensor,
    source_index: int,
    metric: str,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    metric_index = METRICS.index(metric)
    valid = mask.bool()
    statistics[source_index, metric_index, 0] += values[valid].double().sum()
    statistics[source_index, metric_index, 1] += valid.sum().double()


def _accumulate_outcome_metrics(
    statistics: torch.Tensor,
    prediction: torch.Tensor,
    batch: CanonicalBatch,
    predicted_hand_joints_local: torch.Tensor | None = None,
) -> None:
    predicted = SCHEMA.split(prediction.float())
    target = SCHEMA.split(batch.future_state.float())
    radians_to_degrees = 180.0 / math.pi
    for sample_index, metadata in enumerate(batch.metadata or ()):
        source_index = REPORTING_SOURCES.index(
            reporting_source(str(metadata["source_dataset"]))
        )
        stream_mask = batch.future_stream_mask[sample_index]

        camera_translation = torch.linalg.vector_norm(
            predicted["camera"][sample_index, :, :3] - target["camera"][sample_index, :, :3],
            dim=-1,
        ) * 100.0
        camera_rotation = so3_geodesic_angle(
            rotation_6d_to_matrix(predicted["camera"][sample_index, :, 3:9]),
            rotation_6d_to_matrix(target["camera"][sample_index, :, 3:9]),
        ) * radians_to_degrees
        _add(statistics, source_index, "camera_translation_cm", camera_translation, stream_mask[:, 0])
        _add(statistics, source_index, "camera_rotation_deg", camera_rotation, stream_mask[:, 0])

        wrist_translation: list[torch.Tensor] = []
        wrist_rotation: list[torch.Tensor] = []
        wrist_mask: list[torch.Tensor] = []
        mano_rotation: list[torch.Tensor] = []
        mano_mask: list[torch.Tensor] = []
        for stream_index, side in ((1, "left"), (2, "right")):
            wrist_name = f"{side}_wrist"
            wrist_translation.append(
                torch.linalg.vector_norm(
                    predicted[wrist_name][sample_index, :, :3]
                    - target[wrist_name][sample_index, :, :3],
                    dim=-1,
                )
                * 100.0
            )
            wrist_rotation.append(
                so3_geodesic_angle(
                    rotation_6d_to_matrix(predicted[wrist_name][sample_index, :, 3:9]),
                    rotation_6d_to_matrix(target[wrist_name][sample_index, :, 3:9]),
                )
                * radians_to_degrees
            )
            wrist_mask.append(stream_mask[:, stream_index])

            mano_name = f"{side}_mano"
            predicted_mano = predicted[mano_name][sample_index].reshape(-1, 15, 6)
            target_mano = target[mano_name][sample_index].reshape(-1, 15, 6)
            mano_rotation.append(
                so3_geodesic_angle(
                    rotation_6d_to_matrix(predicted_mano),
                    rotation_6d_to_matrix(target_mano),
                ).mean(dim=-1)
                * radians_to_degrees
            )
            mano_mask.append(stream_mask[:, stream_index + 2])

        combined_wrist_mask = torch.stack(wrist_mask, dim=-1)
        _add(
            statistics,
            source_index,
            "wrist_translation_cm",
            torch.stack(wrist_translation, dim=-1),
            combined_wrist_mask,
        )
        _add(
            statistics,
            source_index,
            "wrist_rotation_deg",
            torch.stack(wrist_rotation, dim=-1),
            combined_wrist_mask,
        )
        _add(
            statistics,
            source_index,
            "mano_rotation_deg",
            torch.stack(mano_rotation, dim=-1),
            torch.stack(mano_mask, dim=-1),
        )
        if (
            predicted_hand_joints_local is not None
            and batch.future_hand_joints_local is not None
        ):
            hand_error = torch.linalg.vector_norm(
                predicted_hand_joints_local[sample_index].float()
                - batch.future_hand_joints_local[sample_index].float(),
                dim=-1,
            ) * 100.0
            hand_valid = stream_mask[:, [3, 4]]
            _add(
                statistics,
                source_index,
                "hand_joint_mpjpe_cm",
                hand_error,
                hand_valid[..., None].expand_as(hand_error),
            )
            fingertip_error = hand_error[..., (4, 8, 12, 16, 20)]
            _add(
                statistics,
                source_index,
                "fingertip_mpjpe_cm",
                fingertip_error,
                hand_valid[..., None].expand_as(fingertip_error),
            )
        horizon = torch.tensor(
            [float(metadata["horizon_seconds"])],
            device=statistics.device,
            dtype=torch.float64,
        )
        _add(statistics, source_index, "horizon_seconds", horizon, torch.ones_like(horizon, dtype=torch.bool))


def _accumulate_flow_metrics(
    statistics: torch.Tensor,
    batch: CanonicalBatch,
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    predicted_visual_velocity: torch.Tensor | None,
    target_visual_velocity: torch.Tensor | None,
    estimated_visual: torch.Tensor | None,
    clean_visual: torch.Tensor | None,
) -> None:
    geometry_error = (predicted_velocity.float() - target_velocity.float()).square()
    geometry_mask = SCHEMA.expand_stream_mask(batch.future_stream_mask)
    for sample_index, metadata in enumerate(batch.metadata or ()):
        source_index = REPORTING_SOURCES.index(
            reporting_source(str(metadata["source_dataset"]))
        )
        _add(
            statistics,
            source_index,
            "geometry_flow_mse",
            geometry_error[sample_index],
            geometry_mask[sample_index],
        )
        if predicted_visual_velocity is None or target_visual_velocity is None:
            continue
        visual_error = (
            predicted_visual_velocity[sample_index].float()
            - target_visual_velocity[sample_index].float()
        ).square()
        visual_mask = batch.future_valid_mask[sample_index, :, None, None].expand_as(visual_error)
        _add(
            statistics,
            source_index,
            "future_visual_flow_mse",
            visual_error,
            visual_mask,
        )
        if estimated_visual is None or clean_visual is None:
            continue
        cosine = torch.nn.functional.cosine_similarity(
            estimated_visual[sample_index].float(), clean_visual[sample_index].float(), dim=-1
        )
        cosine_mask = batch.future_valid_mask[sample_index, :, None].expand_as(cosine)
        _add(statistics, source_index, "future_visual_cosine", cosine, cosine_mask)


@torch.no_grad()
def evaluate_vitra(
    model: torch.nn.Module,
    loader: Iterable[CanonicalBatch],
    *,
    device: torch.device,
    use_bf16: bool,
    ode_steps: int,
    ode_method: str,
    visual_normalization: str,
    visual_normalization_eps: float,
) -> dict[str, float]:
    """Evaluate fixed held-out episodes and reduce exact sums/counts across ranks."""

    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.eval()
    statistics = torch.zeros(
        len(REPORTING_SOURCES), len(METRICS), 2, dtype=torch.float64, device=device
    )
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        state_noise = _deterministic_noise(batch, batch.future_state.shape, namespace="geometry")
        geometry_mask = SCHEMA.expand_stream_mask(batch.future_query_stream_mask).to(
            batch.future_state.dtype
        )
        state_noise = state_noise * geometry_mask
        clean_state = batch.future_state * geometry_mask
        flow_time = torch.full(
            (batch.batch_size,), 0.5, device=device, dtype=batch.future_state.dtype
        )
        noisy_state = 0.5 * (state_noise + clean_state)
        target_velocity = clean_state - state_noise

        noisy_visual = None
        target_visual = None
        clean_visual = None
        visual_noise = None
        if batch.future_visual_latents is not None:
            clean_visual = normalize_visual_flow_target(
                batch.future_visual_latents,
                mode=visual_normalization,
                eps=visual_normalization_eps,
            )
            visual_mask = batch.future_valid_mask[..., None, None].to(clean_visual.dtype)
            clean_visual = clean_visual * visual_mask
            visual_noise = _deterministic_noise(
                batch, clean_visual.shape, namespace="future-visual"
            ) * visual_mask
            noisy_visual = 0.5 * (visual_noise + clean_visual)
            target_visual = clean_visual - visual_noise

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            context_cache = unwrapped.prefill_context(batch)
            model_output = unwrapped(
                batch,
                noisy_state,
                flow_time,
                context_cache=context_cache,
                noisy_visual_latent=noisy_visual,
                compute_hand_joints=False,
            )
            predicted_velocity, _, predicted_visual, _, _ = model_output
            generated = sample_ode(
                unwrapped,
                batch,
                steps=ode_steps,
                method=ode_method,
                initial_state=state_noise,
                context_cache=context_cache,
            )
            predicted_hand_joints = None
            if unwrapped.hand_kinematics_head is not None:
                final_output = unwrapped(
                    batch,
                    generated,
                    torch.ones_like(flow_time),
                    context_cache=context_cache,
                    compute_hand_joints=True,
                )
                predicted_hand_joints = final_output.hand_joints_local
        estimated_visual = None
        if predicted_visual is not None and noisy_visual is not None:
            estimated_visual = noisy_visual + 0.5 * predicted_visual
        _accumulate_flow_metrics(
            statistics,
            batch,
            predicted_velocity,
            target_velocity,
            predicted_visual,
            target_visual,
            estimated_visual,
            clean_visual,
        )
        _accumulate_outcome_metrics(
            statistics,
            generated,
            batch,
            predicted_hand_joints_local=predicted_hand_joints,
        )

    if distributed.is_available() and distributed.is_initialized():
        distributed.all_reduce(statistics, op=distributed.ReduceOp.SUM)

    result: dict[str, float] = {}
    for metric_index, metric in enumerate(METRICS):
        source_values: list[float] = []
        total_sum = 0.0
        total_count = 0.0
        for source_index, source in enumerate(REPORTING_SOURCES):
            numerator, count = statistics[source_index, metric_index].tolist()
            if count <= 0:
                continue
            value = numerator / count
            result[f"validation/{source}/{metric}"] = value
            source_values.append(value)
            total_sum += numerator
            total_count += count
        if source_values:
            result[f"validation/macro/{metric}"] = sum(source_values) / len(source_values)
            result[f"validation/micro/{metric}"] = total_sum / total_count
    return result


@torch.no_grad()
def evaluate_trajectory(
    model: torch.nn.Module,
    loader: Iterable[CanonicalBatch],
    *,
    device: torch.device,
    use_bf16: bool,
    ode_steps: int,
    ode_method: str,
    initial_noise_scale: float = 1.0,
) -> dict[str, float]:
    """N=1 ADE/FDE for H2O/EgoPAT3D/HOT3D-Clips wrist trajectories.

    The target and prediction are already in the final observed camera frame.  Only the tracked
    wrist XYZ is scored; unavailable wrist orientation and MANO values never enter the metric.
    """

    if initial_noise_scale < 0:
        raise ValueError("initial_noise_scale must be non-negative")
    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.eval()
    # ADE sum/count, FDE sum/count, camera translation sum/count, examples.
    statistics = torch.zeros(7, dtype=torch.float64, device=device)
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        state_noise = _deterministic_noise(
            batch, batch.future_state.shape, namespace="trajectory-geometry"
        )
        query_mask = SCHEMA.expand_stream_mask(batch.future_query_stream_mask).to(
            batch.future_state.dtype
        )
        state_noise = state_noise * query_mask * initial_noise_scale
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            context_cache = unwrapped.prefill_context(batch)
            generated = sample_ode(
                unwrapped,
                batch,
                steps=ode_steps,
                method=ode_method,
                initial_state=state_noise,
                context_cache=context_cache,
            )
        predicted_streams = SCHEMA.split(generated.float())
        target_streams = SCHEMA.split(batch.future_state.float())
        for sample_index, metadata in enumerate(batch.metadata or ()):
            hand = str(metadata["tracked_hand"])
            name = f"{hand}_wrist"
            error = torch.linalg.vector_norm(
                predicted_streams[name][sample_index, :, :3]
                - target_streams[name][sample_index, :, :3],
                dim=-1,
            )
            valid = batch.effective_future_state_mask[
                sample_index,
                :,
                (SCHEMA.left_wrist if hand == "left" else SCHEMA.right_wrist).start,
            ]
            valid_error = error[valid]
            if not len(valid_error):
                continue
            statistics[0] += valid_error.double().sum()
            statistics[1] += len(valid_error)
            statistics[2] += valid_error[-1].double()
            statistics[3] += 1
            camera_error = torch.linalg.vector_norm(
                predicted_streams["camera"][sample_index, :, :3]
                - target_streams["camera"][sample_index, :, :3],
                dim=-1,
            )
            camera_valid = batch.effective_future_state_mask[
                sample_index, :, SCHEMA.camera.start
            ]
            statistics[4] += camera_error[camera_valid].double().sum()
            statistics[5] += camera_valid.sum().double()
            statistics[6] += 1
    if distributed.is_available() and distributed.is_initialized():
        distributed.all_reduce(statistics, op=distributed.ReduceOp.SUM)
    ade = float(statistics[0] / statistics[1].clamp_min(1))
    fde = float(statistics[2] / statistics[3].clamp_min(1))
    return {
        "validation/trajectory/ade_m": ade,
        "validation/trajectory/fde_m": fde,
        # A single predeclared selection statistic prevents choosing a checkpoint that improves
        # average displacement while silently degrading the manipulation-critical endpoint.
        "validation/trajectory/mean_ade_fde_m": 0.5 * (ade + fde),
        "validation/trajectory/camera_translation_m": float(
            statistics[4] / statistics[5].clamp_min(1)
        ),
        "validation/trajectory/examples": float(statistics[6]),
    }
