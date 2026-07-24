"""Read globally deduplicated staged visual features without episode-level duplication."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


UNIQUE_VISUAL_CONTRACT = "ego_hand_wm.vitra_unique_dinotxt_visual_features"


class UniqueVisualFeatureStore:
    def __init__(
        self,
        *,
        feature_root: str | Path,
        staged_rgb_root: str | Path,
        max_open_videos: int = 16,
        output_dtype: np.dtype = np.float32,
    ) -> None:
        self.feature_root = Path(feature_root)
        self.staged_rgb_root = Path(staged_rgb_root)
        try:
            success = json.loads((self.feature_root / "_SUCCESS").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Missing or invalid unique visual completion gate: {self.feature_root}"
            ) from error
        if success.get("complete") is not True or success.get("contract") != UNIQUE_VISUAL_CONTRACT:
            raise ValueError(f"Incomplete or incompatible unique visual root: {self.feature_root}")
        self.success = success
        self.total_tokens = int(success["total_tokens"])
        self.feature_dim = int(success["feature_dim"])
        self.max_open_videos = int(max_open_videos)
        self.output_dtype = np.dtype(output_dtype)
        if self.output_dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
            raise ValueError("Unique visual output_dtype must be float16 or float32")
        if self.max_open_videos <= 0:
            raise ValueError("max_open_videos must be positive")
        self._cache: OrderedDict[
            tuple[str, str], tuple[np.ndarray, np.ndarray]
        ] = OrderedDict()

    @staticmethod
    def _close(array: np.ndarray) -> None:
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()

    def _load(self, dataset: str, video: str) -> tuple[np.ndarray, np.ndarray]:
        key = (str(dataset), str(video))
        cached = self._cache.pop(key, None)
        if cached is not None:
            self._cache[key] = cached
            return cached
        manifest_path = self.feature_root / dataset / f"{video}.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Missing unique feature manifest: {manifest_path}") from error
        if (
            manifest.get("complete") is not True
            or manifest.get("extractor_id") != self.success["extractor_id"]
        ):
            raise ValueError(f"Invalid unique feature manifest: {manifest_path}")
        frame_ids = np.load(
            self.staged_rgb_root / dataset / f"{video}.frames.npy",
            allow_pickle=False,
            mmap_mode="r",
        )
        features = np.load(
            self.feature_root / dataset / f"{video}.features.npy",
            allow_pickle=False,
            mmap_mode="r",
        )
        expected = (len(frame_ids), self.total_tokens, self.feature_dim)
        if frame_ids.ndim != 1 or features.shape != expected or features.dtype != np.float16:
            raise ValueError(
                f"Invalid unique arrays for {dataset}/{video}: {frame_ids.shape}, {features.shape}"
            )
        cached = (frame_ids, features)
        self._cache[key] = cached
        while len(self._cache) > self.max_open_videos:
            _, (old_ids, old_features) = self._cache.popitem(last=False)
            self._close(old_ids)
            self._close(old_features)
        return cached

    def lookup(self, dataset: str, video: str, physical_frame_ids: np.ndarray) -> torch.Tensor:
        requested = np.asarray(physical_frame_ids, dtype=np.int64)
        if requested.ndim != 1:
            raise ValueError("physical_frame_ids must be one-dimensional")
        available, features = self._load(dataset, video)
        positions = np.searchsorted(available, requested)
        valid = positions < len(available)
        in_bounds = np.flatnonzero(valid)
        valid[in_bounds] = available[positions[in_bounds]] == requested[in_bounds]
        if not valid.all():
            missing = requested[~valid][:10].tolist()
            raise KeyError(f"Unique visual features missing for {dataset}/{video}: {missing}")
        values = np.asarray(features[positions], dtype=self.output_dtype)
        return torch.from_numpy(values)
