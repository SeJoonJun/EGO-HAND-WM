"""Aligned frozen DINO.txt feature caches for H2O/EgoPAT3D windows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


TRAJECTORY_FEATURE_CONTRACT = "ego_hand_wm.trajectory_dinotxt_visual_features"


class TrajectoryVisualFeatureStore:
    """Memory-map context features keyed by the manifest's stable sample identifier."""

    def __init__(
        self,
        root: str | Path,
        *,
        dataset: str,
        split: str,
        sequence: str = "history",
        output_dtype: np.dtype = np.float32,
    ) -> None:
        self.root = Path(root)
        self.dataset = str(dataset)
        self.split = str(split)
        self.sequence = str(sequence)
        if self.sequence not in {"history", "future"}:
            raise ValueError("Trajectory feature sequence must be 'history' or 'future'")
        self.output_dtype = np.dtype(output_dtype)
        if self.output_dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
            raise ValueError("Trajectory visual output_dtype must be float16 or float32")
        directory = self.root / self.dataset
        prefix = self.split if self.sequence == "history" else f"{self.split}.future"
        success_path = directory / f"{prefix}.SUCCESS.json"
        try:
            self.success: dict[str, Any] = json.loads(success_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Missing or invalid trajectory feature gate: {success_path}") from error
        if (
            self.success.get("complete") is not True
            or self.success.get("contract") != TRAJECTORY_FEATURE_CONTRACT
        ):
            raise ValueError(f"Incomplete trajectory feature cache: {success_path}")
        self.sample_ids = np.load(
            directory / f"{prefix}.sample_ids.npy", allow_pickle=False, mmap_mode="r"
        )
        self.features = np.load(
            directory / f"{prefix}.features.npy", allow_pickle=False, mmap_mode="r"
        )
        steps_key = "history_steps" if self.sequence == "history" else "future_steps"
        expected = (
            len(self.sample_ids),
            int(self.success[steps_key]),
            int(self.success["total_tokens"]),
            int(self.success["feature_dim"]),
        )
        if self.sample_ids.ndim != 1 or self.features.shape != expected:
            raise ValueError(
                f"Invalid trajectory feature arrays: {self.sample_ids.shape}, "
                f"{self.features.shape}; expected {expected}"
            )
        if self.features.dtype != np.float16:
            raise ValueError("Trajectory DINO.txt features must be stored as float16")
        self._indices = {str(sample_id): index for index, sample_id in enumerate(self.sample_ids)}
        if len(self._indices) != len(self.sample_ids):
            raise ValueError("Trajectory feature cache contains duplicate sample IDs")

    def lookup(self, sample_id: str) -> torch.Tensor:
        try:
            index = self._indices[str(sample_id)]
        except KeyError as error:
            raise KeyError(f"Trajectory visual features missing for {sample_id!r}") from error
        # A memmap slice is read-only.  Collation needs an owned tensor and PyTorch otherwise
        # warns that accidental writes would be undefined behavior.
        return torch.from_numpy(
            np.array(self.features[index], dtype=self.output_dtype, copy=True)
        )
