#!/usr/bin/env python3
"""Evaluate one VITRA-fine-tuned checkpoint on an official trajectory split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ego_hand_wm.config import load_config
from ego_hand_wm.contracts.batch import canonical_collate
from ego_hand_wm.data.build import build_dataset
from ego_hand_wm.models.world_action_model import WorldActionModel
from ego_hand_wm.training.checkpoint import initialize_model_from_checkpoint
from ego_hand_wm.training.validation import evaluate_trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ode-steps", type=int, default=8)
    parser.add_argument("--ode-method", choices=("euler", "heun"), default="heun")
    parser.add_argument("--initial-noise-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, [])
    data_config = dict(config["data"])
    data_config["split"] = args.split
    dataset = build_dataset(data_config)
    loader_options = {}
    if args.num_workers > 0:
        loader_options = {"persistent_workers": True, "prefetch_factor": 2}
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=canonical_collate,
        drop_last=False,
        **loader_options,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WorldActionModel(config["model"]).to(device)
    missing, unexpected = initialize_model_from_checkpoint(
        args.checkpoint, model=model, strict=True
    )
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    metrics = evaluate_trajectory(
        model,
        loader,
        device=device,
        use_bf16=device.type == "cuda" and torch.cuda.is_bf16_supported(),
        ode_steps=args.ode_steps,
        ode_method=args.ode_method,
        initial_noise_scale=args.initial_noise_scale,
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "samples": len(dataset),
        **{
            key.replace("validation/trajectory/", f"test/{args.split}/"): value
            for key, value in metrics.items()
        },
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
