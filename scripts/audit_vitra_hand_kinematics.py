#!/usr/bin/env python
"""Measure wrist-local hand motion and persistence on a configured VITRA split."""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from ego_hand_wm.config import load_config
from ego_hand_wm.data.build import build_dataset


FINGERTIPS = (4, 8, 12, 16, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vitra_geometry_kinematics_refine.yaml")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data = dict(config["data"])
    data["split"] = args.split
    data["shuffle_buffer"] = 0
    # This diagnostic uses annotations only and should not fault visual/text mmap pages into RAM.
    for key in ("unique_feature_root", "staged_rgb_root", "feature_root", "text_feature_root"):
        data.pop(key, None)
    data["attach_future_visual"] = False
    data["require_context_visual"] = False
    data["require_future_visual"] = False
    dataset = build_dataset(data)
    samples_iterable = DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.num_workers,
        persistent_workers=False,
    )

    errors: list[torch.Tensor] = []
    fingertip_errors: list[torch.Tensor] = []
    movement: list[torch.Tensor] = []
    samples = 0
    valid_hands = 0
    for sample in samples_iterable:
        history = sample["history_hand_joints_local"]
        future = sample["future_hand_joints_local"]
        mask = sample["future_stream_mask"][:, [3, 4]]
        persistence = history[-1][None].expand_as(future)
        distance = torch.linalg.vector_norm(persistence - future, dim=-1)
        selected = distance[mask[..., None].expand_as(distance)]
        if selected.numel():
            errors.append(selected)
            fingertip_errors.append(
                distance[..., FINGERTIPS][mask[..., None].expand_as(distance[..., FINGERTIPS])]
            )
            movement.append(
                torch.linalg.vector_norm(future - history[-1][None], dim=-1)[
                    mask[..., None].expand_as(distance)
                ]
            )
            valid_hands += int(mask.sum())
        samples += 1
        if samples >= args.max_samples:
            break
    if not errors:
        raise RuntimeError("No valid hand targets were found")
    all_error = torch.cat(errors)
    all_fingertips = torch.cat(fingertip_errors)
    all_movement = torch.cat(movement)
    report = {
        "samples": samples,
        "valid_hand_steps": valid_hands,
        "persistence_joint_mpjpe_cm": float(all_error.mean() * 100.0),
        "persistence_fingertip_mpjpe_cm": float(all_fingertips.mean() * 100.0),
        "mean_joint_displacement_cm": float(all_movement.mean() * 100.0),
        "p90_joint_displacement_cm": float(torch.quantile(all_movement, 0.9) * 100.0),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
