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
from ego_hand_wm.flow.rectified_flow import (
    make_flow_training_sample,
    make_visual_flow_training_sample,
    sample_flow_time,
)
from ego_hand_wm.losses import WorldActionLoss
from ego_hand_wm.models.world_action_model import WorldActionModel
from ego_hand_wm.training.checkpoint import (
    initialize_model_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
    save_model_checkpoint,
)
from ego_hand_wm.training.distributed import cleanup, initialize, seed_everything
from ego_hand_wm.training.validation import evaluate_trajectory, evaluate_vitra


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
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("training.min_lr_ratio must lie in [0,1]")

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def run_training(config: dict[str, Any]) -> dict[str, float | int | str]:
    runtime = config["runtime"]
    training = config["training"]
    context = initialize(str(runtime.get("device", "auto")))
    seed_everything(int(config.get("seed", 42)), context.rank)
    output_dir = Path(training["output_dir"])
    resume = training.get("resume")
    init_checkpoint = training.get("init_checkpoint")
    if resume and init_checkpoint:
        raise ValueError("Configure either training.resume or training.init_checkpoint, not both")
    overwrite = bool(training.get("overwrite", False))
    output_error: str | None = None
    if (
        context.is_main
        and output_dir.exists()
        and any(output_dir.iterdir())
        and not resume
        and not overwrite
    ):
        output_error = (
            f"Refusing to overwrite nonempty fresh-run directory: {output_dir}. "
            "Set training.resume or explicitly set training.overwrite=true."
        )
    if context.world_size > 1:
        synchronized_error = [output_error]
        torch.distributed.broadcast_object_list(synchronized_error, src=0)
        output_error = synchronized_error[0]
    if output_error is not None:
        cleanup()
        raise FileExistsError(output_error)
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
    if context.world_size > 1:
        torch.distributed.barrier()

    dataset = build_dataset(config["data"])
    _validate_dataset_capabilities(dataset, config["data"])
    sampler = None
    if context.world_size > 1 and not isinstance(dataset, IterableDataset):
        sampler = DistributedSampler(
            dataset, num_replicas=context.world_size, rank=context.rank, shuffle=True
        )
    train_workers = int(training.get("num_workers", 0))
    train_loader_options: dict[str, Any] = {}
    if train_workers > 0:
        train_loader_options["persistent_workers"] = bool(
            training.get("persistent_workers", True)
        )
        train_loader_options["prefetch_factor"] = int(training.get("prefetch_factor", 2))
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=sampler is None and not isinstance(dataset, IterableDataset),
        sampler=sampler,
        num_workers=train_workers,
        pin_memory=context.device.type == "cuda",
        collate_fn=canonical_collate,
        drop_last=bool(training.get("drop_last", True)),
        **train_loader_options,
    )

    validation = dict(config.get("validation", {}))
    validation_loader = None
    if bool(validation.get("enabled", False)):
        validation_data = dict(config["data"])
        if validation_data.get("kind") in {"trajectory_h6k16", "hot3d_clips_h6k16"}:
            if not validation_data.get("manifests"):
                raise ValueError("Trajectory validation requires data.manifests")
        elif not validation_data.get("split_manifest"):
            raise ValueError("VITRA validation requires data.split_manifest")
        validation_data["split"] = "validation"
        validation_data["shuffle_buffer"] = 0
        validation_dataset = build_dataset(validation_data)
        _validate_dataset_capabilities(validation_dataset, validation_data)
        validation_workers = int(
            validation.get("num_workers", training.get("num_workers", 0))
        )
        validation_loader_options: dict[str, Any] = {}
        if validation_workers > 0:
            validation_loader_options["persistent_workers"] = bool(
                validation.get("persistent_workers", True)
            )
            validation_loader_options["prefetch_factor"] = int(
                validation.get("prefetch_factor", training.get("prefetch_factor", 2))
            )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=int(validation.get("batch_size", training["batch_size"])),
            shuffle=False,
            num_workers=validation_workers,
            pin_memory=context.device.type == "cuda",
            collate_fn=canonical_collate,
            drop_last=False,
            **validation_loader_options,
        )

    model = WorldActionModel(config["model"]).to(context.device)
    if init_checkpoint:
        missing, unexpected = initialize_model_from_checkpoint(
            init_checkpoint,
            model=model,
            strict=bool(training.get("init_strict", False)),
        )
        if context.is_main:
            initialization_report = {
                "initialized_from": str(init_checkpoint),
                "missing_model_keys": missing,
                "unexpected_model_keys": unexpected,
            }
            with (output_dir / "initialization_report.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(initialization_report, handle, indent=2, sort_keys=True)
            print(
                json.dumps(
                    {
                        "initialized_from": str(init_checkpoint),
                        "missing_model_key_count": len(missing),
                        "unexpected_model_key_count": len(unexpected),
                        "missing_model_key_examples": missing[:10],
                        "unexpected_model_key_examples": unexpected[:10],
                    }
                )
            )
    # Wrap only after model-only initialization so DDP broadcasts the newly initialized
    # auxiliary parameters from rank zero; missing non-strict keys must not differ by rank.
    if context.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            find_unused_parameters=bool(training.get("find_unused_parameters", True)),
            gradient_as_bucket_view=bool(
                training.get("gradient_as_bucket_view", False)
            ),
            static_graph=bool(training.get("static_graph", False)),
            broadcast_buffers=bool(training.get("broadcast_buffers", True)),
        )
    criterion = WorldActionLoss(config["loss"])
    optimizer_options: dict[str, Any] = {}
    if context.device.type == "cuda" and bool(training.get("fused_optimizer", False)):
        optimizer_options["fused"] = True
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
        betas=(0.9, 0.95),
        **optimizer_options,
    )
    max_steps = int(training["max_steps"])
    scheduler = _build_scheduler(
        optimizer,
        int(training.get("warmup_steps", 0)),
        max_steps,
        float(training.get("min_lr_ratio", 0.0)),
    )
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
    best_metric: float | None = None
    stale_validations = 0
    early_stop = False
    selection_metric = validation.get("selection_metric")
    selection_mode = str(validation.get("selection_mode", "min"))
    if selection_mode not in {"min", "max"}:
        raise ValueError("validation.selection_mode must be 'min' or 'max'")
    selection_min_delta = float(validation.get("selection_min_delta", 0.0))
    selection_patience = int(validation.get("early_stopping_patience", 0))
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
            flow = make_flow_training_sample(
                batch,
                time_config=training.get("geometry_flow_time"),
            )
            noisy_visual = None
            target_visual = None
            visual_flow_time = None
            if batch.future_visual_latents is not None:
                visual_time_config = training.get("visual_flow_time")
                visual_flow_time = (
                    flow.flow_time
                    if visual_time_config is None
                    else sample_flow_time(
                        batch.batch_size,
                        device=batch.future_state.device,
                        dtype=batch.future_state.dtype,
                        config=visual_time_config,
                    )
                )
                visual_flow = make_visual_flow_training_sample(
                    batch.future_visual_latents,
                    visual_flow_time,
                    batch.future_valid_mask,
                    normalization=str(
                        config["loss"].get("visual_target_normalization", "none")
                    ),
                    normalization_eps=float(
                        config["loss"].get("visual_target_normalization_eps", 1e-6)
                    ),
                )
                noisy_visual = visual_flow.noisy_latent
                target_visual = visual_flow.target_velocity
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
                    model_output = model(
                        batch,
                        flow.noisy_state,
                        flow.flow_time,
                        noisy_visual_latent=noisy_visual,
                        visual_flow_time=visual_flow_time,
                    )
                    predicted, _, predicted_visual, _, _ = model_output
                    metrics = criterion(
                        batch,
                        predicted,
                        flow.target_velocity,
                        flow.noisy_state,
                        flow.flow_time,
                        predicted_visual_velocity=predicted_visual,
                        target_visual_velocity=target_visual,
                        predicted_hand_joints_local=model_output.hand_joints_local,
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
            validation_every = int(validation.get("every", 0))
            if (
                validation_loader is not None
                and validation_every > 0
                and step % validation_every == 0
            ):
                validation_kind = str(
                    validation.get(
                        "kind",
                        "trajectory"
                        if config["data"].get("kind")
                        in {"trajectory_h6k16", "hot3d_clips_h6k16"}
                        else "vitra",
                    )
                )
                if validation_kind == "trajectory":
                    validation_metrics = evaluate_trajectory(
                        model,
                        validation_loader,
                        device=context.device,
                        use_bf16=use_amp,
                        ode_steps=int(validation.get("ode_steps", 4)),
                        ode_method=str(validation.get("ode_method", "heun")),
                        initial_noise_scale=float(
                            validation.get("initial_noise_scale", 1.0)
                        ),
                    )
                elif validation_kind == "vitra":
                    validation_metrics = evaluate_vitra(
                        model,
                        validation_loader,
                        device=context.device,
                        use_bf16=use_amp,
                        ode_steps=int(validation.get("ode_steps", 4)),
                        ode_method=str(validation.get("ode_method", "heun")),
                        visual_normalization=str(
                            config["loss"].get("visual_target_normalization", "none")
                        ),
                        visual_normalization_eps=float(
                            config["loss"].get("visual_target_normalization_eps", 1e-6)
                        ),
                    )
                else:
                    raise ValueError(f"Unknown validation kind: {validation_kind!r}")
                model.train()
                if context.is_main:
                    record = {"step": step, **validation_metrics}
                    print(json.dumps(record))
                    with (output_dir / "validation.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                if selection_metric is not None:
                    if selection_metric not in validation_metrics:
                        raise KeyError(
                            f"Selection metric {selection_metric!r} was not produced by validation"
                        )
                    candidate = float(validation_metrics[selection_metric])
                    improved = best_metric is None or (
                        candidate < best_metric - selection_min_delta
                        if selection_mode == "min"
                        else candidate > best_metric + selection_min_delta
                    )
                    if improved:
                        best_metric = candidate
                        stale_validations = 0
                        if context.is_main:
                            save_model_checkpoint(
                                output_dir / "best.pt",
                                model=model,
                                step=step,
                                config=config,
                                metrics=validation_metrics,
                            )
                    else:
                        stale_validations += 1
                    if selection_patience > 0 and stale_validations >= selection_patience:
                        early_stop = True
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
            model_save_every = int(training.get("model_only_save_every", 0))
            if (
                context.is_main
                and model_save_every > 0
                and step % model_save_every == 0
            ):
                save_model_checkpoint(
                    output_dir / f"model-step-{step:08d}.pt",
                    model=model,
                    step=step,
                    config=config,
                )
            if step >= max_steps or early_stop:
                break
        epoch += 1
        if epoch_batches == 0:
            raise RuntimeError("Dataset produced no valid batches")
        if early_stop:
            break

    if context.is_main and bool(training.get("save_last", True)):
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
