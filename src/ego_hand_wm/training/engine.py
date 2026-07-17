from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset

from ego_hand_wm.contracts.batch import canonical_collate
from ego_hand_wm.data.build import build_dataset
from ego_hand_wm.flow.rectified_flow import make_flow_training_sample
from ego_hand_wm.losses import WorldActionLoss
from ego_hand_wm.models.world_action_model import WorldActionModel
from ego_hand_wm.training.checkpoint import load_checkpoint, save_checkpoint
from ego_hand_wm.training.distributed import cleanup, initialize, seed_everything


def _validate_dataset_capabilities(dataset: object, data_config: dict[str, Any]) -> None:
    requirements = {
        "require_context_visual": "provides_context_visual",
        "require_future_visual": "provides_future_visual",
    }
    for requirement, capability in requirements.items():
        if bool(data_config.get(requirement, False)) and not bool(
            getattr(dataset, capability, False)
        ):
            raise RuntimeError(
                f"data.{requirement}=true, but {type(dataset).__name__} does not provide "
                f"that modality. Build/attach the visual feature shards before launching training."
            )


def _validate_batch_modalities(
    batch: object, data_config: dict[str, Any], model_config: dict[str, Any]
) -> None:
    if bool(data_config.get("require_context_visual", False)):
        if batch.context_images is None and batch.context_visual_features is None:
            raise RuntimeError("This run requires context RGB or context visual features")
    expected_visual_dim = int(model_config.get("future_visual_latent_dim", 0))
    if bool(data_config.get("require_future_visual", False)) and batch.future_visual_latents is None:
        raise RuntimeError("This run requires future visual latent targets")
    if batch.future_visual_latents is not None:
        if expected_visual_dim <= 0:
            raise RuntimeError(
                "Batch contains future visual latents but model.future_visual_latent_dim is disabled"
            )
        if batch.future_visual_latents.shape[-1] != expected_visual_dim:
            raise RuntimeError(
                "Future visual target width does not match model.future_visual_latent_dim: "
                f"{batch.future_visual_latents.shape[-1]} != {expected_visual_dim}"
            )


def _build_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, max_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def run_training(config: dict[str, Any]) -> dict[str, float | int | str]:
    runtime = config["runtime"]
    training = config["training"]
    context = initialize(str(runtime.get("device", "auto")))
    seed_everything(int(config.get("seed", 42)), context.rank)
    output_dir = Path(training["output_dir"])
    resume = training.get("resume")
    overwrite = bool(training.get("overwrite", False))
    if output_dir.exists() and any(output_dir.iterdir()) and not resume and not overwrite:
        cleanup()
        raise FileExistsError(
            f"Refusing to overwrite nonempty fresh-run directory: {output_dir}. "
            "Set training.resume or explicitly set training.overwrite=true."
        )
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)

    dataset = build_dataset(config["data"])
    _validate_dataset_capabilities(dataset, config["data"])
    sampler = None
    if context.world_size > 1 and not isinstance(dataset, IterableDataset):
        sampler = DistributedSampler(
            dataset, num_replicas=context.world_size, rank=context.rank, shuffle=True
        )
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=sampler is None and not isinstance(dataset, IterableDataset),
        sampler=sampler,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=context.device.type == "cuda",
        collate_fn=canonical_collate,
        drop_last=bool(training.get("drop_last", True)),
    )

    model = WorldActionModel(config["model"]).to(context.device)
    if context.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            find_unused_parameters=bool(training.get("find_unused_parameters", True)),
        )
    criterion = WorldActionLoss(config["loss"])
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
        betas=(0.9, 0.95),
    )
    max_steps = int(training["max_steps"])
    scheduler = _build_scheduler(optimizer, int(training.get("warmup_steps", 0)), max_steps)
    requested_bf16 = bool(training.get("bf16", True))
    if (
        context.device.type == "cuda"
        and requested_bf16
        and not torch.cuda.is_bf16_supported()
    ):
        cleanup()
        raise RuntimeError("This run requests BF16, but the selected CUDA device lacks BF16 support")
    use_amp = context.device.type == "cuda" and requested_bf16
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    start_step = 0
    if resume:
        start_step = load_checkpoint(
            resume, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    if accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    last_metrics: dict[str, float] = {}
    epoch = 0
    microstep = 0
    while step < max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_batches = 0
        for batch in loader:
            epoch_batches += 1
            microstep += 1
            should_step = microstep % accumulation == 0
            batch = batch.to(context.device, non_blocking=True)
            _validate_batch_modalities(batch, config["data"], config["model"])
            flow = make_flow_training_sample(batch)
            noisy_visual = None
            target_visual = None
            if batch.future_visual_latents is not None:
                visual_mask = batch.future_valid_mask[..., None].to(
                    batch.future_visual_latents.dtype
                )
                visual_noise = torch.randn_like(batch.future_visual_latents) * visual_mask
                interpolation = flow.flow_time[:, None, None]
                noisy_visual = (
                    (1.0 - interpolation) * visual_noise
                    + interpolation * batch.future_visual_latents
                ) * visual_mask
                target_visual = (batch.future_visual_latents - visual_noise) * visual_mask
            synchronization = (
                model.no_sync()
                if isinstance(model, DistributedDataParallel) and not should_step
                else nullcontext()
            )
            with synchronization:
                with torch.autocast(
                    device_type=context.device.type,
                    dtype=torch.bfloat16,
                    enabled=use_amp,
                ):
                    predicted, _, predicted_visual, _, _ = model(
                        batch,
                        flow.noisy_state,
                        flow.flow_time,
                        noisy_visual_latent=noisy_visual,
                    )
                    metrics = criterion(
                        batch,
                        predicted,
                        flow.target_velocity,
                        flow.noisy_state,
                        flow.flow_time,
                        predicted_visual_velocity=predicted_visual,
                        target_visual_velocity=target_visual,
                    )
                    loss = metrics["loss"] / accumulation
                if not torch.isfinite(metrics["loss"]):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
            if not should_step:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training.get("grad_clip", 1.0))
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Non-finite gradient norm")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            last_metrics = {key: float(value.detach().float()) for key, value in metrics.items()}
            if context.is_main and step % int(training.get("log_every", 1)) == 0:
                print(json.dumps({"step": step, "lr": scheduler.get_last_lr()[0], **last_metrics}))
            save_every = int(training.get("save_every", 0))
            if context.is_main and save_every > 0 and step % save_every == 0:
                save_checkpoint(
                    output_dir / f"step-{step:08d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    step=step,
                    config=config,
                )
            if step >= max_steps:
                break
        epoch += 1
        if epoch_batches == 0:
            raise RuntimeError("Dataset produced no valid batches")

    if context.is_main:
        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            step=step,
            config=config,
        )
    result: dict[str, float | int | str] = {
        "step": step,
        "output_dir": str(output_dir),
        **last_metrics,
    }
    cleanup()
    return result
