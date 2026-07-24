"""Deterministic local cache for frozen DINO.txt prompt embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


TEXT_CACHE_CONTRACT = "ego_hand_wm.dinotxt_text_features"


class DinoTxtTextFeatureStore:
    """Memory-map DINO.txt embeddings and provide exact prompt lookup."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        try:
            success = json.loads((self.root / "_SUCCESS").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Missing or invalid DINO.txt text cache: {self.root}") from error
        if success.get("complete") is not True or success.get("contract") != TEXT_CACHE_CONTRACT:
            raise ValueError(f"Incomplete or incompatible DINO.txt text cache: {self.root}")
        self.success: dict[str, Any] = success
        self.texts = np.load(self.root / "texts.npy", allow_pickle=False, mmap_mode="r")
        self.features = np.load(
            self.root / "features.npy", allow_pickle=False, mmap_mode="r"
        )
        expected = (len(self.texts), int(success["feature_dim"]))
        if self.texts.ndim != 1 or self.features.shape != expected:
            raise ValueError(
                f"Invalid DINO.txt text cache arrays: {self.texts.shape}, {self.features.shape}"
            )
        if self.features.dtype != np.float16:
            raise ValueError("DINO.txt cached text features must be float16")
        self._indices = {str(text): index for index, text in enumerate(self.texts)}
        if len(self._indices) != len(self.texts):
            raise ValueError("DINO.txt text cache contains duplicate prompts")

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    def lookup(self, text: str) -> torch.Tensor:
        try:
            index = self._indices[str(text)]
        except KeyError as error:
            raise KeyError(f"Prompt is absent from DINO.txt cache: {text!r}") from error
        return torch.from_numpy(np.asarray(self.features[index], dtype=np.float32))


def write_text_feature_cache(
    *,
    output_root: str | Path,
    texts: Sequence[str],
    features: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Atomically publish sorted unique prompts and their aligned features."""
    from ego_hand_wm.data.feature_shards import atomic_write_json, extractor_id

    root = Path(output_root)
    ordered = [str(text) for text in texts]
    if ordered != sorted(set(ordered)):
        raise ValueError("DINO.txt cache prompts must be sorted and unique")
    values = np.asarray(features)
    if values.ndim != 2 or values.shape[0] != len(ordered) or values.shape[1] <= 0:
        raise ValueError("DINO.txt text features must be [N,D] and align with prompts")
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise ValueError("DINO.txt text features must be finite floating-point values")
    root.mkdir(parents=True, exist_ok=True)
    (root / "_SUCCESS").unlink(missing_ok=True)
    max_length = max((len(text) for text in ordered), default=1)
    text_tmp = root / "texts.npy.tmp"
    feature_tmp = root / "features.npy.tmp"
    text_array = np.lib.format.open_memmap(
        text_tmp, mode="w+", dtype=f"<U{max_length}", shape=(len(ordered),)
    )
    text_array[:] = ordered
    text_array.flush()
    del text_array
    feature_array = np.lib.format.open_memmap(
        feature_tmp, mode="w+", dtype=np.float16, shape=values.shape
    )
    feature_array[:] = values.astype(np.float16)
    feature_array.flush()
    del feature_array
    text_tmp.replace(root / "texts.npy")
    feature_tmp.replace(root / "features.npy")
    success = {
        "complete": True,
        "contract": TEXT_CACHE_CONTRACT,
        "prompts": len(ordered),
        "feature_dim": int(values.shape[1]),
        "dtype": "float16",
        "normalized": True,
        "extractor_id": extractor_id(metadata),
        "extractor": metadata,
    }
    atomic_write_json(root / "_SUCCESS", success)
    return success
