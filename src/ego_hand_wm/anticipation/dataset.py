"""Datasets for the Assembly101 e4 semantic-anticipation experiments."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ego_hand_wm.anticipation.protocol import AnticipationRecord, context_frame_indices, read_e4_anticipation_csv
from ego_hand_wm.data.adapters.assembly101 import (
    ANNOTATION_FPS,
    WRIST_REFERENCES,
    WristReference,
    canonicalize_assembly101_geometry,
    canonicalize_assembly101_oracle_geometry,
)


HISTORY_STEPS = 32
GEOMETRY_FPS = 8
GAP_STEPS = 8
EXECUTION_STEPS = 8
ORACLE_STEPS = HISTORY_STEPS + GAP_STEPS + EXECUTION_STEPS


def feature_cache_path(root: str | Path, record: AnticipationRecord) -> Path:
    return Path(root) / f"{record.recording}__{record.video_stem}.npy"


def geometry_cache_path(root: str | Path, record: AnticipationRecord) -> Path:
    return Path(root) / f"{record.recording}.npz"


def oracle_feature_cache_path(
    root: str | Path, split: str, record: AnticipationRecord
) -> Path:
    return Path(root) / split / f"{record.segment_id:07d}.npy"


class _LRUFiles:
    """Small per-worker cache for memory maps and decompressed recording geometry."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        self.capacity = capacity
        self.values: OrderedDict[Path, Any] = OrderedDict()

    def get(self, path: Path, loader: Any) -> Any:
        value = self.values.pop(path, None)
        if value is None:
            value = loader(path)
        self.values[path] = value
        while len(self.values) > self.capacity:
            stale = self.values.popitem(last=False)[1]
            close = getattr(stale, "close", None)
            if callable(close):
                close()
        return value


def _load_geometry(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = ("camera_world_from_camera", "wrist_world_from_hand", "wrist_confidence")
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f"Geometry cache {path} lacks {missing}")
        return {name: np.asarray(archive[name]) for name in required}


def _load_oracle_geometry(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = (
            "camera_world_from_camera",
            "wrist_world_from_hand",
            "landmarks_world",
            "wrist_confidence",
        )
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f"Geometry cache {path} lacks {missing}")
        result = {name: np.asarray(archive[name]) for name in required}
        result["frame_valid"] = (
            np.asarray(archive["frame_valid"], dtype=bool)
            if "frame_valid" in archive
            else np.ones((result["camera_world_from_camera"].shape[0],), dtype=bool)
        )
        return result


def oracle_relative_times() -> np.ndarray:
    """Fixed signed timestamps relative to the official observation cutoff."""

    history = np.arange(-HISTORY_STEPS + 1, 1, dtype=np.float64) / GEOMETRY_FPS
    gap = np.arange(1, GAP_STEPS + 1, dtype=np.float64) / GEOMETRY_FPS
    execution = 1.0 + np.arange(1, EXECUTION_STEPS + 1, dtype=np.float64) / GEOMETRY_FPS
    return np.concatenate((history, gap, execution))


class Assembly101E4AnticipationDataset(Dataset[dict[str, Any]]):
    """Official one-second e4 segments with frozen DINOv3 and final-camera geometry."""

    def __init__(
        self,
        *,
        annotations_csv: str | Path,
        feature_root: str | Path,
        geometry_root: str | Path,
        confidence_threshold: float = 0.25,
        require_labels: bool = True,
        require_all_caches: bool = True,
        cache_size: int = 8,
    ) -> None:
        self.feature_root = Path(feature_root)
        self.geometry_root = Path(geometry_root)
        self.confidence_threshold = float(confidence_threshold)
        records = read_e4_anticipation_csv(annotations_csv, require_labels=require_labels)
        missing: list[str] = []
        retained: list[AnticipationRecord] = []
        for record in records:
            feature_path = feature_cache_path(self.feature_root, record)
            geometry_path = geometry_cache_path(self.geometry_root, record)
            if feature_path.is_file() and geometry_path.is_file():
                retained.append(record)
            else:
                missing.append(
                    f"{record.recording}: feature={feature_path.is_file()} geometry={geometry_path.is_file()}"
                )
        if missing and require_all_caches:
            preview = "\n".join(missing[:10])
            raise FileNotFoundError(
                f"Missing Assembly101 anticipation caches for {len(missing)} segments; first entries:\n{preview}"
            )
        self.records = retained
        if not self.records:
            raise FileNotFoundError("No Assembly101 records have both DINOv3 and geometry caches")
        self._features = _LRUFiles(cache_size)
        self._geometry = _LRUFiles(cache_size)

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _open_features(path: Path) -> np.ndarray:
        features = np.load(path, allow_pickle=False, mmap_mode="r")
        if features.ndim not in (2, 3) or features.shape[0] == 0:
            raise ValueError(f"DINOv3 cache must be [T,D] or [T,P,D]: {path} {features.shape}")
        return features

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        features = self._features.get(
            feature_cache_path(self.feature_root, record), self._open_features
        )
        geometry = self._geometry.get(
            geometry_cache_path(self.geometry_root, record), _load_geometry
        )
        lengths = [features.shape[0], *[array.shape[0] for array in geometry.values()]]
        available = min(lengths)
        indices = context_frame_indices(record.anchor_frame)
        if indices[-1] >= available:
            raise IndexError(
                f"Segment {record.segment_id} anchor {indices[-1]} exceeds cache length {available}"
            )

        visual = np.asarray(features[indices], dtype=np.float32)
        if visual.ndim == 3:
            visual = visual.mean(axis=1)
        canonical = canonicalize_assembly101_geometry(
            geometry["camera_world_from_camera"][indices],
            geometry["wrist_world_from_hand"][indices],
            geometry["wrist_confidence"][indices],
            confidence_threshold=self.confidence_threshold,
        )
        labels = torch.tensor(
            [
                -1 if record.verb is None else record.verb,
                -1 if record.object is None else record.object,
                -1 if record.action is None else record.action,
            ],
            dtype=torch.long,
        )
        return {
            "rgb_features": torch.from_numpy(visual),
            **canonical,
            "labels": labels,
            "segment_id": record.segment_id,
            "recording": record.recording,
            "video_stem": record.video_stem,
            "anchor_frame": record.anchor_frame,
        }


class Assembly101E4OracleDataset(Dataset[dict[str, Any]]):
    """V-JEPA visual tokens plus history and target-execution oracle geometry."""

    def __init__(
        self,
        *,
        split: str,
        annotations_csv: str | Path,
        feature_root: str | Path,
        geometry_root: str | Path,
        confidence_threshold: float = 0.25,
        wrist_reference: WristReference = "camera_anchor",
        require_all_caches: bool = True,
        cache_size: int = 8,
    ) -> None:
        self.split = str(split)
        self.feature_root = Path(feature_root)
        self.geometry_root = Path(geometry_root)
        self.confidence_threshold = float(confidence_threshold)
        if wrist_reference not in WRIST_REFERENCES:
            raise ValueError(
                f"Unsupported wrist_reference={wrist_reference!r}; expected one of "
                f"{sorted(WRIST_REFERENCES)}"
            )
        self.wrist_reference: WristReference = wrist_reference
        records = read_e4_anticipation_csv(annotations_csv, require_labels=True)
        missing_visual: list[str] = []
        missing_geometry: set[str] = set()
        retained: list[AnticipationRecord] = []
        for record in records:
            visual = oracle_feature_cache_path(self.feature_root, self.split, record)
            geometry = geometry_cache_path(self.geometry_root, record)
            if visual.is_file():
                retained.append(record)
                if not geometry.is_file():
                    missing_geometry.add(record.recording)
            else:
                missing_visual.append(
                    f"id={record.segment_id} recording={record.recording} visual={visual}"
                )
        if missing_visual and require_all_caches:
            raise FileNotFoundError(
                f"Missing V-JEPA caches for {len(missing_visual)} e4 segments; first entries:\n"
                + "\n".join(missing_visual[:10])
            )
        self.records = retained
        if not self.records:
            raise FileNotFoundError("No e4 oracle records have V-JEPA visual caches")
        # Assembly101 does not release poses for every official anticipation
        # recording. Keep the official sample set identical across all eight
        # ablations and represent those recordings with an all-invalid geometry
        # stream.  The model consequently reduces to RGB for those samples.
        self.missing_geometry_recordings = frozenset(missing_geometry)
        self._features = _LRUFiles(cache_size)
        self._geometry = _LRUFiles(cache_size)
        self._relative_times = oracle_relative_times()

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _open_features(path: Path) -> np.ndarray:
        features = np.load(path, allow_pickle=False, mmap_mode="r")
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError(f"V-JEPA cache must be [tokens,dim]: {path} {features.shape}")
        return features

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        visual = self._features.get(
            oracle_feature_cache_path(self.feature_root, self.split, record),
            self._open_features,
        )
        geometry_path = geometry_cache_path(self.geometry_root, record)
        if geometry_path.is_file():
            geometry = self._geometry.get(geometry_path, _load_oracle_geometry)
            available = min(array.shape[0] for array in geometry.values())
            if available <= 0:
                raise ValueError(f"Geometry cache is empty: {geometry_path}")

            frame_offsets = np.rint(self._relative_times * ANNOTATION_FPS).astype(np.int64)
            indices = record.anchor_frame + frame_offsets
            timestamp_valid = np.ones((ORACLE_STEPS,), dtype=bool)
            # Early clips match the official visual loader by repeating frame zero.
            indices[:HISTORY_STEPS] = np.maximum(indices[:HISTORY_STEPS], 0)
            execution = np.arange(HISTORY_STEPS + GAP_STEPS, ORACLE_STEPS)
            timestamp_valid[execution] = indices[execution] <= record.end_frame
            indices[execution] = np.minimum(indices[execution], record.end_frame)
            timestamp_valid &= indices < available
            # Canonical coordinates require the e4 camera at the legal
            # observation cutoff. If that anchor is absent, mask the complete
            # geometry sequence instead of using a shifted reference frame.
            if record.anchor_frame >= available:
                timestamp_valid[:] = False
            indices = np.minimum(indices, available - 1)
            timestamp_valid &= geometry["frame_valid"][indices]

            canonical = canonicalize_assembly101_oracle_geometry(
                geometry["camera_world_from_camera"][indices],
                geometry["wrist_world_from_hand"][indices],
                geometry["landmarks_world"][indices],
                geometry["wrist_confidence"][indices],
                anchor_index=HISTORY_STEPS - 1,
                confidence_threshold=self.confidence_threshold,
                wrist_reference=self.wrist_reference,
            )
            geometry_available = bool(timestamp_valid.any())
        else:
            timestamp_valid = np.zeros((ORACLE_STEPS,), dtype=bool)
            canonical = {
                "camera_pose": torch.zeros((ORACLE_STEPS, 9), dtype=torch.float32),
                "wrist_pose": torch.zeros((ORACLE_STEPS, 2, 9), dtype=torch.float32),
                "hand_pose": torch.zeros(
                    (ORACLE_STEPS, 2, 21, 3), dtype=torch.float32
                ),
                "wrist_confidence": torch.zeros(
                    (ORACLE_STEPS, 2), dtype=torch.float32
                ),
                "wrist_valid": torch.zeros((ORACLE_STEPS, 2), dtype=torch.bool),
                "hand_pose_valid": torch.zeros(
                    (ORACLE_STEPS, 2), dtype=torch.bool
                ),
            }
            geometry_available = False
        future_mask = torch.zeros(ORACLE_STEPS, dtype=torch.bool)
        future_mask[HISTORY_STEPS:] = True
        execution_mask = torch.zeros(ORACLE_STEPS, dtype=torch.bool)
        execution_mask[HISTORY_STEPS + GAP_STEPS :] = True
        labels = torch.tensor(
            [record.verb, record.object, record.action], dtype=torch.long
        )
        return {
            "visual_tokens": torch.from_numpy(np.asarray(visual, dtype=np.float32)),
            **canonical,
            "geometry_time_mask": torch.from_numpy(timestamp_valid),
            "time_seconds": torch.from_numpy(self._relative_times.astype(np.float32)),
            "future_mask": future_mask,
            "execution_mask": execution_mask,
            "geometry_available": torch.tensor(geometry_available),
            "labels": labels,
            "segment_id": record.segment_id,
            "recording": record.recording,
            "video_stem": record.video_stem,
            "anchor_frame": record.anchor_frame,
        }
