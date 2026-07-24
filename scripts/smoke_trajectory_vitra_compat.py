#!/usr/bin/env python3
"""Real-checkpoint compatibility gate for H2O/EgoPAT3D canonical fine-tuning."""

from __future__ import annotations

import argparse
import json

import torch

from ego_hand_wm.config import load_config
from ego_hand_wm.contracts.batch import canonical_collate
from ego_hand_wm.data.build import build_dataset
from ego_hand_wm.flow.rectified_flow import make_flow_training_sample
from ego_hand_wm.losses import WorldActionLoss
from ego_hand_wm.models.world_action_model import WorldActionModel
from ego_hand_wm.training.checkpoint import initialize_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, [])
    dataset = build_dataset(config["data"])
    # Exercise two physically distinct records and, for H2O, commonly different hand sides.
    batch = canonical_collate([dataset[0], dataset[-1]])
    batch.validate()
    if batch.context_visual_features is None:
        raise RuntimeError("Real compatibility gate requires cached DINO.txt visual features")
    if tuple(batch.context_visual_features.shape) != (2, 6, 17, 1024):
        raise ValueError(
            f"Unexpected DINO.txt context shape: {tuple(batch.context_visual_features.shape)}"
        )
    if batch.context_text_mask is None or batch.context_text_mask.any():
        raise ValueError("Trajectory text must be explicitly present-and-masked")

    model = WorldActionModel(config["model"])
    missing, unexpected = initialize_model_from_checkpoint(
        args.checkpoint, model=model, strict=False
    )
    if missing:
        raise RuntimeError(f"VITRA checkpoint is missing downstream model keys: {missing}")
    invalid_unexpected = [
        key for key in unexpected if not key.startswith("future_visual_expert.")
    ]
    if invalid_unexpected:
        raise RuntimeError(
            "Unexpected non-training-only VITRA checkpoint keys: "
            f"{invalid_unexpected[:20]}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).train()
    batch = batch.to(device)
    criterion = WorldActionLoss(config["loss"])
    flow = make_flow_training_sample(batch)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        output = model(batch, flow.noisy_state, flow.flow_time)
        metrics = criterion(
            batch,
            output.velocity,
            flow.target_velocity,
            flow.noisy_state,
            flow.flow_time,
        )
    if not torch.isfinite(metrics["loss"]):
        raise FloatingPointError("Compatibility loss is not finite")
    metrics["loss"].backward()
    finite_gradients = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("Compatibility backward produced a non-finite gradient")
        finite_gradients += 1
    if finite_gradients == 0:
        raise RuntimeError("Compatibility backward produced no gradients")

    print(
        json.dumps(
            {
                "status": "compatible",
                "config": args.config,
                "dataset": batch.metadata[0]["source_dataset"],
                "samples": batch.batch_size,
                "history_steps": batch.history_state.shape[1],
                "future_steps": batch.future_state.shape[1],
                "horizon_seconds": batch.metadata[0]["horizon_seconds"],
                "dinotxt_shape": list(batch.context_visual_features.shape),
                "text_condition": "masked_unavailable",
                "checkpoint_missing_keys": len(missing),
                "checkpoint_training_only_unexpected_keys": len(unexpected),
                "finite_gradient_tensors": finite_gradients,
                "loss": float(metrics["loss"].detach()),
                "flow": float(metrics["flow"].detach()),
                "camera_flow": float(metrics["flow/camera"].detach()),
                "left_wrist_flow": float(metrics["flow/left_wrist"].detach()),
                "right_wrist_flow": float(metrics["flow/right_wrist"].detach()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

