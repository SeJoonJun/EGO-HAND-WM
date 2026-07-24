"""Trainer for the controlled Assembly101 e4 oracle-geometry ablation."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ego_hand_wm.anticipation.dataset import Assembly101E4OracleDataset
from ego_hand_wm.anticipation.metrics import semantic_anticipation_metrics
from ego_hand_wm.anticipation.model import (
    OracleGeometryAnticipationModel,
    focal_semantic_loss,
)
from ego_hand_wm.anticipation.protocol import load_tail_segment_ids, load_unseen_recordings


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _dataset(config: dict[str, Any], split: str) -> Assembly101E4OracleDataset:
    data = config["data"]
    annotations = Path(data["annotations_root"]) / f"{split}.csv"
    return Assembly101E4OracleDataset(
        split=split,
        annotations_csv=annotations,
        feature_root=data["feature_root"],
        geometry_root=data["geometry_root"],
        confidence_threshold=float(data.get("confidence_threshold", 0.25)),
        wrist_reference=str(data.get("wrist_reference", "camera_anchor")),
        require_all_caches=bool(data.get("require_all_caches", True)),
        cache_size=int(data.get("cache_size", 8)),
    )


def _loader(
    dataset: Assembly101E4OracleDataset,
    config: dict[str, Any],
    *,
    train: bool,
) -> DataLoader:
    training = config["training"]
    # Keep the train permutation independent of model construction. Geometry
    # modes instantiate different parameter counts and therefore advance the
    # global torch RNG by different amounts.
    generator = torch.Generator()
    generator.manual_seed(int(config.get("seed", 42)) + (0 if train else 1))
    return DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=train,
        num_workers=int(training.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(training.get("num_workers", 4)) > 0,
        drop_last=train and bool(training.get("drop_last", True)),
        generator=generator,
    )


def _model_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: batch[name].to(device, non_blocking=True)
        for name in (
            "visual_tokens",
            "camera_pose",
            "wrist_pose",
            "hand_pose",
            "wrist_valid",
            "hand_pose_valid",
            "geometry_time_mask",
            "time_seconds",
            "future_mask",
            "execution_mask",
        )
    }


@torch.inference_mode()
def evaluate(
    model: OracleGeometryAnticipationModel,
    loader: DataLoader,
    device: torch.device,
    *,
    tail_segment_ids: set[int] | None = None,
    unseen_recordings: set[str] | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    score_parts: dict[str, list[torch.Tensor]] = {name: [] for name in ("verb", "object", "action")}
    labels: list[torch.Tensor] = []
    segment_ids: list[torch.Tensor] = []
    recordings: list[str] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        output = model(**_model_inputs(batch, device))
        for name, score in output.scores().items():
            score_parts[name].append(score.float().cpu())
        labels.append(batch["labels"].cpu())
        segment_ids.append(torch.as_tensor(batch["segment_id"]).cpu())
        recordings.extend(list(batch["recording"]))
    if not labels:
        raise ValueError("Evaluation loader produced no batches")
    return semantic_anticipation_metrics(
        {name: torch.cat(parts) for name, parts in score_parts.items()},
        torch.cat(labels),
        segment_ids=torch.cat(segment_ids),
        recordings=recordings,
        tail_segment_ids=tail_segment_ids,
        unseen_recordings=unseen_recordings,
    )


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def run_anticipation_training(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    _seed_everything(seed)
    device = _device(str(config.get("runtime", {}).get("device", "auto")))
    train_dataset = _dataset(config, "train")
    validation_dataset = _dataset(config, "validation")
    train_loader = _loader(train_dataset, config, train=True)
    validation_loader = _loader(validation_dataset, config, train=False)

    model_config = dict(config["model"])
    model = OracleGeometryAnticipationModel(**model_config).to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.05)),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(training.get("schedule_epoch", 10)),
        gamma=float(training.get("schedule_gamma", 0.1)),
    )
    output_dir = Path(training["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    evaluation_root = Path(config["data"]["evaluation_root"])
    tail_path = evaluation_root / "tail_validation_segments.txt"
    unseen_path = evaluation_root / "validation_split_seq.txt"
    tail_ids = load_tail_segment_ids(tail_path) if tail_path.is_file() else None
    unseen_recordings = load_unseen_recordings(unseen_path) if unseen_path.is_file() else None

    epochs = int(training.get("epochs", 15))
    validation_interval = int(training.get("validation_interval", 2))
    if validation_interval <= 0:
        raise ValueError("training.validation_interval must be positive")
    max_train_batches = training.get("max_train_batches")
    max_eval_batches = training.get("max_eval_batches")
    best_recall = float("-inf")
    history: list[dict[str, Any]] = []
    use_bf16 = bool(training.get("bf16", True)) and device.type == "cuda"
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"total": 0.0, "action": 0.0, "verb": 0.0, "object": 0.0}
        batches = 0
        for batch_index, batch in enumerate(train_loader):
            if max_train_batches is not None and batch_index >= int(max_train_batches):
                break
            optimizer.zero_grad(set_to_none=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                output = model(**_model_inputs(batch, device))
                loss, parts = focal_semantic_loss(
                    output, labels, gamma=float(training.get("focal_gamma", 2.0))
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training.get("grad_clip", 5.0))
            )
            optimizer.step()
            totals["total"] += float(loss.detach())
            for name, value in parts.items():
                totals[name] += float(value.detach())
            batches += 1
        if batches == 0:
            raise ValueError("Training loader produced no batches")
        scheduler.step()
        report: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train/{name}_loss": value / batches for name, value in totals.items()},
        }
        run_validation = epoch % validation_interval == 0 or epoch == epochs
        metrics: dict[str, float] = {}
        if run_validation:
            metrics = evaluate(
                model,
                validation_loader,
                device,
                tail_segment_ids=tail_ids,
                unseen_recordings=unseen_recordings,
                max_batches=None if max_eval_batches is None else int(max_eval_batches),
            )
            report.update(metrics)
        history.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "metrics": metrics if run_validation else None,
        }
        _atomic_torch_save(checkpoint, output_dir / "last.pt")
        if run_validation:
            recall = metrics["overall/action_mean_top5_recall"]
            if recall > best_recall:
                best_recall = recall
                _atomic_torch_save(checkpoint, output_dir / "best.pt")
        (output_dir / "metrics.json").write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n"
        )
    return {"best_action_mean_top5_recall": best_recall, "epochs": epochs, "output_dir": str(output_dir)}
