#!/usr/bin/env python3
"""Audit legacy trajectory checkpoints under alternate validation-time samplers.

This script intentionally never opens a test manifest.  It isolates the effect of
ODE step count and initial-noise scale without using held-out test results for
model or sampler selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ego_hand_wm.config import load_config
from ego_hand_wm.contracts.batch import canonical_collate
from ego_hand_wm.data.build import build_dataset
from ego_hand_wm.models.world_action_model import WorldActionModel
from ego_hand_wm.training.checkpoint import initialize_model_from_checkpoint
from ego_hand_wm.training.validation import evaluate_trajectory


EXPERIMENTS = {
    "h2o": {
        "config": "/scratch/jun.se/EGO-HAND-WM/runs/h2o-vitra-h6k16-v1/resolved_config.json",
        "checkpoint": "/scratch/jun.se/EGO-HAND-WM/runs/h2o-vitra-h6k16-v1/best.pt",
    },
    "egopat3d": {
        "config": "/scratch/jun.se/EGO-HAND-WM/runs/egopat3d-vitra-h6k16-v1/resolved_config.json",
        "checkpoint": "/scratch/jun.se/EGO-HAND-WM/runs/egopat3d-vitra-h6k16-v1/best.pt",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ode-steps", type=int, nargs="+", default=(8, 16))
    parser.add_argument(
        "--noise-scales", type=float, nargs="+", default=(0.0, 0.5, 1.0)
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_validation_loader(
    config: dict[str, Any], batch_size: int, num_workers: int
) -> tuple[DataLoader, int]:
    data = dict(config["data"])
    data["split"] = "validation"
    dataset = build_dataset(data)
    options: dict[str, Any] = {}
    if num_workers > 0:
        options = {"persistent_workers": True, "prefetch_factor": 2}
    return (
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=canonical_collate,
            drop_last=False,
            **options,
        ),
        len(dataset),
    )


def main() -> None:
    args = parse_args()
    if any(step <= 0 for step in args.ode_steps):
        raise ValueError("--ode-steps must be positive")
    if any(scale < 0 for scale in args.noise_scales):
        raise ValueError("--noise-scales must be non-negative")

    experiment = EXPERIMENTS[args.dataset]
    config = load_config(experiment["config"])
    checkpoint = Path(experiment["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, examples = make_validation_loader(
        config, args.batch_size, args.num_workers
    )
    model = WorldActionModel(config["model"]).to(device)
    missing, unexpected = initialize_model_from_checkpoint(
        checkpoint, model=model, strict=True
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.eval()

    candidates = []
    for ode_steps in args.ode_steps:
        for noise_scale in args.noise_scales:
            metrics = evaluate_trajectory(
                model,
                loader,
                device=device,
                use_bf16=device.type == "cuda" and torch.cuda.is_bf16_supported(),
                ode_steps=ode_steps,
                ode_method="heun",
                initial_noise_scale=noise_scale,
            )
            candidates.append(
                {
                    "ode_steps": ode_steps,
                    "initial_noise_scale": noise_scale,
                    "validation_ade_m": metrics["validation/trajectory/ade_m"],
                    "validation_fde_m": metrics["validation/trajectory/fde_m"],
                    "validation_mean_ade_fde_m": metrics[
                        "validation/trajectory/mean_ade_fde_m"
                    ],
                }
            )

    result = {
        "scope": "validation-only inference audit; test manifests were not opened",
        "dataset": args.dataset,
        "config": experiment["config"],
        "checkpoint": str(checkpoint),
        "validation_examples": examples,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
