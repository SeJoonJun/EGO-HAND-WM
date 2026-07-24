#!/usr/bin/env python3
"""Select a trajectory recipe on validation, then evaluate held-out test exactly once."""

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
    "h2o": (
        (
            "fresh",
            "configs/benchmarks/h2o_vitra_finetune_h6k16_trajectory_v2.yaml",
            "/scratch/jun.se/EGO-HAND-WM/runs/h2o-vitra-h6k16-trajectory-v2/best.pt",
        ),
        (
            "refine",
            "configs/benchmarks/h2o_vitra_finetune_h6k16_trajectory_v2_refine.yaml",
            "/scratch/jun.se/EGO-HAND-WM/runs/h2o-vitra-h6k16-trajectory-v2-refine/best.pt",
        ),
    ),
    "egopat3d": (
        (
            "fresh",
            "configs/benchmarks/egopat3d_vitra_finetune_h6k16_trajectory_v2.yaml",
            "/scratch/jun.se/EGO-HAND-WM/runs/egopat3d-vitra-h6k16-trajectory-v2-iofix/best.pt",
        ),
        (
            "refine",
            "configs/benchmarks/egopat3d_vitra_finetune_h6k16_trajectory_v2_refine.yaml",
            "/scratch/jun.se/EGO-HAND-WM/runs/egopat3d-vitra-h6k16-trajectory-v2-refine-iofix/best.pt",
        ),
    ),
}

TEST_SPLITS = {
    "h2o": ("test",),
    "egopat3d": ("test_seen", "test_novel"),
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


def make_loader(
    config: dict[str, Any], split: str, batch_size: int, num_workers: int
) -> tuple[DataLoader, int]:
    data = dict(config["data"])
    data["split"] = split
    dataset = build_dataset(data)
    worker_options: dict[str, Any] = {}
    if num_workers > 0:
        worker_options = {"persistent_workers": True, "prefetch_factor": 2}
    return (
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=canonical_collate,
            drop_last=False,
            **worker_options,
        ),
        len(dataset),
    )


def load_model(
    config: dict[str, Any], checkpoint: str, device: torch.device
) -> WorldActionModel:
    model = WorldActionModel(config["model"]).to(device)
    missing, unexpected = initialize_model_from_checkpoint(
        checkpoint, model=model, strict=True
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {checkpoint}: missing={missing}, unexpected={unexpected}"
        )
    return model.eval()


def evaluate(
    model: WorldActionModel,
    loader: DataLoader,
    *,
    device: torch.device,
    ode_steps: int,
    noise_scale: float,
) -> dict[str, float]:
    return evaluate_trajectory(
        model,
        loader,
        device=device,
        use_bf16=device.type == "cuda" and torch.cuda.is_bf16_supported(),
        ode_steps=ode_steps,
        ode_method="heun",
        initial_noise_scale=noise_scale,
    )


def main() -> None:
    args = parse_args()
    if len(set(args.ode_steps)) != len(args.ode_steps) or any(
        step <= 0 for step in args.ode_steps
    ):
        raise ValueError("--ode-steps must contain unique positive integers")
    if len(set(args.noise_scales)) != len(args.noise_scales) or any(
        scale < 0 for scale in args.noise_scales
    ):
        raise ValueError("--noise-scales must contain unique non-negative values")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates: list[dict[str, Any]] = []

    # Recipe and integration-step selection use the complete validation split only.
    for name, config_path, checkpoint in EXPERIMENTS[args.dataset]:
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Training did not produce {checkpoint}")
        config = load_config(config_path)
        loader, examples = make_loader(
            config, "validation", args.batch_size, args.num_workers
        )
        model = load_model(config, checkpoint, device)
        for ode_steps in args.ode_steps:
            for noise_scale in args.noise_scales:
                metrics = evaluate(
                    model,
                    loader,
                    device=device,
                    ode_steps=ode_steps,
                    noise_scale=noise_scale,
                )
                candidates.append(
                    {
                        "recipe": name,
                        "config": config_path,
                        "checkpoint": checkpoint,
                        "ode_steps": ode_steps,
                        "initial_noise_scale": noise_scale,
                        "validation_examples": examples,
                        "validation_ade_m": metrics["validation/trajectory/ade_m"],
                        "validation_fde_m": metrics["validation/trajectory/fde_m"],
                        "validation_mean_ade_fde_m": metrics[
                            "validation/trajectory/mean_ade_fde_m"
                        ],
                    }
                )
        del model, loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selected = min(candidates, key=lambda item: item["validation_mean_ade_fde_m"])
    selected_config = load_config(selected["config"])
    selected_model = load_model(selected_config, selected["checkpoint"], device)

    # The test split is opened only after validation has fixed both checkpoint and ODE steps.
    test_results: dict[str, Any] = {}
    for split in TEST_SPLITS[args.dataset]:
        loader, examples = make_loader(
            selected_config, split, args.batch_size, args.num_workers
        )
        metrics = evaluate(
            selected_model,
            loader,
            device=device,
            ode_steps=int(selected["ode_steps"]),
            noise_scale=float(selected["initial_noise_scale"]),
        )
        test_results[split] = {
            "examples": examples,
            "ade_m": metrics["validation/trajectory/ade_m"],
            "fde_m": metrics["validation/trajectory/fde_m"],
            "mean_ade_fde_m": metrics[
                "validation/trajectory/mean_ade_fde_m"
            ],
        }

    result = {
        "protocol": {
            "history_steps": 6,
            "future_steps": 16,
            "fps": 30,
            "prediction_horizon_seconds": 16 / 30,
            "samples_per_context": 1,
            "selection_data": "complete validation split only",
            "test_evaluations_per_selected_model": 1,
            "persistence_residual": False,
        },
        "dataset": args.dataset,
        "validation_candidates": candidates,
        "selected": selected,
        "test": test_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
