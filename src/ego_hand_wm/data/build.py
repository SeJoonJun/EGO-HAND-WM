from __future__ import annotations

import glob
import io
import random
import tarfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as distributed
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from ego_hand_wm.data.adapters.vitra import canonicalize_vitra_episode, load_vitra_episode
from ego_hand_wm.data.synthetic import SyntheticCanonicalDataset


def _window_indices(length: int, history_steps: int, future_steps: int, index: int) -> tuple[list[int], list[int]]:
    needed = history_steps + future_steps
    if length < needed:
        raise ValueError(f"Episode has {length} frames but window requires {needed}")
    max_start = length - needed
    start = (index * 104729) % (max_start + 1)
    anchor = start + history_steps - 1
    return list(range(start, anchor + 1)), list(range(anchor + 1, anchor + 1 + future_steps))


def _validated_time_offsets(
    history_offsets_seconds: list[float] | tuple[float, ...],
    future_offsets_seconds: list[float] | tuple[float, ...],
    max_time_error_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    history = np.asarray(history_offsets_seconds, dtype=np.float64)
    future = np.asarray(future_offsets_seconds, dtype=np.float64)
    if history.ndim != 1 or not len(history):
        raise ValueError("history_offsets_seconds must be a non-empty 1D sequence")
    if future.ndim != 1 or not len(future):
        raise ValueError("future_offsets_seconds must be a non-empty 1D sequence")
    if not np.isfinite(history).all() or not np.isfinite(future).all():
        raise ValueError("Physical-time offsets must be finite")
    if not np.all(np.diff(history) > 0) or not np.all(np.diff(future) > 0):
        raise ValueError("Physical-time offsets must be strictly increasing")
    if not np.isclose(history[-1], 0.0, rtol=0.0, atol=1e-9):
        raise ValueError("The final history offset must be the anchor at 0 seconds")
    if future[0] <= 0:
        raise ValueError("Future offsets must be strictly after the anchor")
    if not np.isfinite(max_time_error_seconds) or max_time_error_seconds < 0:
        raise ValueError("max_time_error_seconds must be finite and non-negative")
    return history, future


def _physical_time_window_indices(
    frame_times: np.ndarray,
    anchor_index: int,
    history_offsets_seconds: list[float] | tuple[float, ...],
    future_offsets_seconds: list[float] | tuple[float, ...],
    max_time_error_seconds: float,
) -> tuple[list[int], list[int]]:
    history_offsets, future_offsets = _validated_time_offsets(
        history_offsets_seconds, future_offsets_seconds, max_time_error_seconds
    )
    times = np.asarray(frame_times, dtype=np.float64)
    if times.ndim != 1 or not len(times):
        raise ValueError("frame_times must be a non-empty 1D array")
    if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("Episode-aligned frame_times must be finite and strictly increasing")
    if anchor_index < 0 or anchor_index >= len(times):
        raise IndexError(f"Anchor index {anchor_index} is outside an episode of length {len(times)}")

    offsets = np.concatenate((history_offsets, future_offsets))
    targets = times[anchor_index] + offsets
    right = np.searchsorted(times, targets, side="left")
    left = np.clip(right - 1, 0, len(times) - 1)
    right = np.clip(right, 0, len(times) - 1)
    use_right = np.abs(times[right] - targets) < np.abs(times[left] - targets)
    selected = np.where(use_right, right, left)

    if selected[len(history_offsets) - 1] != anchor_index:
        raise ValueError("The zero-second history target did not select its anchor")
    if not np.all(np.diff(selected) > 0):
        raise ValueError("Nearest annotation frames are not strictly increasing (duplicate selection)")
    errors = np.abs(times[selected] - targets)
    if np.any(errors > max_time_error_seconds):
        raise ValueError(
            f"Nearest annotation frame exceeds {max_time_error_seconds:g}s tolerance "
            f"(maximum error {errors.max():g}s)"
        )

    split = len(history_offsets)
    return selected[:split].tolist(), selected[split:].tolist()


def _sample_physical_time_window(
    frame_times: np.ndarray,
    history_offsets_seconds: list[float] | tuple[float, ...],
    future_offsets_seconds: list[float] | tuple[float, ...],
    max_time_error_seconds: float,
    rng: random.Random,
) -> tuple[int, list[int], list[int]]:
    eligible: list[tuple[int, list[int], list[int]]] = []
    for anchor_index in range(len(frame_times)):
        try:
            history, future = _physical_time_window_indices(
                frame_times,
                anchor_index,
                history_offsets_seconds,
                future_offsets_seconds,
                max_time_error_seconds,
            )
        except (IndexError, ValueError):
            continue
        eligible.append((anchor_index, history, future))
    if not eligible:
        raise ValueError("Episode has no anchor satisfying the physical-time sampling request")
    return rng.choice(eligible)


def _fallback_frame_times(episode: dict[str, Any], fps: float) -> np.ndarray:
    if fps <= 0:
        raise ValueError("fallback_fps must be positive")
    return np.asarray(episode["video_decode_frame"], dtype=np.float64) / fps


def _episode_frame_times(
    episode: dict[str, Any],
    *,
    dataset_name: str,
    pts_root: str | None,
    fallback_fps: float | None,
) -> tuple[np.ndarray, str]:
    if pts_root:
        pts_path = Path(pts_root) / dataset_name / f"{episode['video_name']}.npy"
        if pts_path.is_file():
            video_pts = np.load(pts_path, allow_pickle=False)
            frame_indices = np.asarray(episode["video_decode_frame"], dtype=np.int64)
            if frame_indices.max(initial=-1) >= len(video_pts):
                raise IndexError(f"VITRA frame index exceeds cached PTS array: {pts_path}")
            return np.asarray(video_pts[frame_indices], dtype=np.float64), f"pts:{pts_path}"
        if fallback_fps is None:
            raise FileNotFoundError(f"Required video PTS cache is missing: {pts_path}")
    if fallback_fps is None:
        raise ValueError("No PTS cache and fallback_fps is disabled")
    return _fallback_frame_times(episode, fallback_fps), f"fallback_fps:{fallback_fps}"


class VitraDirectoryDataset(Dataset):
    """Geometry gate for a small extracted annotation set; not the 1.2M production path."""

    provides_context_visual = False
    provides_future_visual = False

    def __init__(self, config: dict[str, Any]) -> None:
        pattern = str(config["annotation_glob"])
        self.paths = [Path(path) for path in sorted(glob.glob(pattern, recursive=True))]
        if not self.paths:
            raise FileNotFoundError(f"No VITRA annotations match {pattern}")
        self.history_steps = int(config["history_steps"])
        self.future_steps = int(config["future_steps"])
        fallback = config.get("fallback_fps")
        self.fallback_fps = float(fallback) if fallback is not None else None
        self.pts_root = config.get("pts_root")
        self.dataset_name = str(config["dataset_name"])
        self.left_mano_policy = str(config.get("left_mano_policy", "mask"))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode = load_vitra_episode(self.paths[index])
        history, future = _window_indices(
            len(episode["extrinsics"]), self.history_steps, self.future_steps, index
        )
        frame_times, time_source = _episode_frame_times(
            episode,
            dataset_name=self.dataset_name,
            pts_root=self.pts_root,
            fallback_fps=self.fallback_fps,
        )
        sample = canonicalize_vitra_episode(
            episode,
            history,
            future,
            frame_times,
            left_mano_policy=self.left_mano_policy,
            source_dataset=self.dataset_name,
            episode_id=self.paths[index].stem,
        )
        sample["metadata"]["annotation_path"] = str(self.paths[index])
        sample["metadata"]["time_source"] = time_source
        return sample


class VitraShardDataset(IterableDataset):
    """Streams physical-time windows from worker-partitioned uncompressed shards."""

    provides_context_visual = False
    provides_future_visual = False

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.shards = [Path(path) for path in sorted(glob.glob(str(config["shard_glob"])))]
        if not self.shards:
            raise FileNotFoundError(f"No VITRA shards match {config['shard_glob']}")
        self.history_offsets_seconds = tuple(float(value) for value in config["history_offsets_seconds"])
        self.future_offsets_seconds = tuple(float(value) for value in config["future_offsets_seconds"])
        self.max_time_error_seconds = float(config["max_time_error_seconds"])
        _validated_time_offsets(
            self.history_offsets_seconds,
            self.future_offsets_seconds,
            self.max_time_error_seconds,
        )
        fallback = config.get("fallback_fps")
        self.fallback_fps = float(fallback) if fallback is not None else None
        self.pts_root = config.get("pts_root")
        self.left_mano_policy = str(config.get("left_mano_policy", "mask"))
        self.shuffle_buffer = int(config.get("shuffle_buffer", 0))
        self._iterator_count = 0

    def _iterator_rng(self) -> random.Random:
        worker = get_worker_info()
        base_seed = worker.seed if worker else torch.initial_seed()
        rank = distributed.get_rank() if distributed.is_available() and distributed.is_initialized() else 0
        iteration = self._iterator_count
        self._iterator_count += 1
        seed = (int(base_seed) + rank * 1_000_003 + iteration * 10_000_019) % (2**64)
        return random.Random(seed)

    def _assigned_shards(self) -> list[Path]:
        rank = distributed.get_rank() if distributed.is_available() and distributed.is_initialized() else 0
        world = distributed.get_world_size() if distributed.is_available() and distributed.is_initialized() else 1
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        workers = worker.num_workers if worker else 1
        global_worker = rank * workers + worker_id
        global_workers = world * workers
        return self.shards[global_worker::global_workers]

    def _episodes(self) -> Iterator[tuple[str, dict[str, Any]]]:
        for shard in self._assigned_shards():
            with tarfile.open(shard, "r:") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith(".npy"):
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        continue
                    episode = np.load(io.BytesIO(extracted.read()), allow_pickle=True).item()
                    yield member.name, episode

    def __iter__(self) -> Iterator[dict[str, Any]]:
        buffer: list[dict[str, Any]] = []
        rng = self._iterator_rng()
        for member_name, episode in self._episodes():
            dataset_name = member_name.split("/", 1)[0]
            frame_times, time_source = _episode_frame_times(
                episode,
                dataset_name=dataset_name,
                pts_root=self.pts_root,
                fallback_fps=self.fallback_fps,
            )
            try:
                _, history, future = _sample_physical_time_window(
                    frame_times,
                    self.history_offsets_seconds,
                    self.future_offsets_seconds,
                    self.max_time_error_seconds,
                    rng,
                )
            except ValueError:
                continue
            sample = canonicalize_vitra_episode(
                episode,
                history,
                future,
                frame_times,
                left_mano_policy=self.left_mano_policy,
                source_dataset=dataset_name,
                episode_id=member_name,
            )
            sample["metadata"]["archive_member"] = member_name
            sample["metadata"]["time_source"] = time_source
            if self.shuffle_buffer <= 1:
                yield sample
                continue
            buffer.append(sample)
            if len(buffer) >= self.shuffle_buffer:
                position = rng.randrange(len(buffer))
                yield buffer.pop(position)
        rng.shuffle(buffer)
        yield from buffer


def build_dataset(config: dict[str, Any]) -> Dataset | IterableDataset:
    kind = config["kind"]
    if kind == "synthetic":
        return SyntheticCanonicalDataset(
            length=int(config["length"]),
            history_steps=int(config["history_steps"]),
            future_steps=int(config["future_steps"]),
            horizon_seconds=float(config["horizon_seconds"]),
            image_size=int(config["image_size"]),
            seed=int(config.get("seed", 17)),
        )
    if kind == "vitra_directory":
        return VitraDirectoryDataset(config)
    if kind == "vitra_shards":
        return VitraShardDataset(config)
    raise ValueError(f"Unknown dataset kind: {kind}")
