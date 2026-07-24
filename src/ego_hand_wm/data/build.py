from __future__ import annotations

import glob
import io
import json
import random
import sqlite3
import tarfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as distributed
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from ego_hand_wm.data.adapters.vitra import canonicalize_vitra_episode, load_vitra_episode
from ego_hand_wm.benchmarks.trajectory_dataset import CanonicalTrajectoryDataset
from ego_hand_wm.benchmarks.hot3d_clips_dataset import Hot3DClipsForecastDataset
from ego_hand_wm.data.feature_shards import (
    EpisodeFeatureRecord,
    feature_shard_paths,
    iter_aligned_annotation_features,
    validate_feature_root_success,
)
from ego_hand_wm.data.synthetic import SyntheticCanonicalDataset
from ego_hand_wm.data.dinotxt_text import DinoTxtTextFeatureStore
from ego_hand_wm.data.unique_features import UniqueVisualFeatureStore
from ego_hand_wm.data.vitra_split import (
    VitraVideoSplit,
    episode_member_identity,
    validate_aliases,
)


def _load_excluded_members(database_path: str | Path | None) -> frozenset[str]:
    """Load the exact preprocessing exclusions from the finalized frame-request index."""
    if database_path is None:
        return frozenset()
    path = Path(database_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing VITRA exclusion database: {path}")
    try:
        success_path = path.with_suffix(path.suffix + ".SUCCESS.json")
        success = json.loads(success_path.read_text())
        if success.get("complete") is not True:
            raise ValueError(f"Incomplete VITRA exclusion database: {path}")
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            return frozenset(
                str(member)
                for (member,) in connection.execute("SELECT member FROM excluded")
            )
    except (OSError, json.JSONDecodeError, sqlite3.DatabaseError) as error:
        raise ValueError(f"Invalid VITRA exclusion database: {path}") from error


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


def _primary_action_intervals(episode: dict[str, Any]) -> tuple[str, list[tuple[int, int]]]:
    primary = str(episode.get("anno_type", "right")).lower()
    if primary not in {"left", "right"}:
        raise ValueError(f"Unsupported VITRA primary hand: {primary!r}")
    length = len(episode["extrinsics"])
    intervals: list[tuple[int, int]] = []
    for _, frame_range in episode.get("text", {}).get(primary, []):
        start = max(int(frame_range[0]), 0)
        end = min(int(frame_range[1]), length)
        if start < end:
            intervals.append((start, end))
    if not intervals:
        raise ValueError(f"VITRA episode has no language interval for primary {primary} hand")
    return primary, intervals


def _choose_target_count(
    supported: list[int],
    *,
    core_target_counts: tuple[int, ...],
    rare_target_count: int | None,
    rare_target_probability: float,
    rng: random.Random,
) -> int:
    core = [count for count in core_target_counts if count in supported]
    rare_supported = rare_target_count is not None and rare_target_count in supported
    if rare_supported and rng.random() < rare_target_probability:
        return int(rare_target_count)
    if core:
        return rng.choice(core)
    if rare_supported:
        return int(rare_target_count)
    raise ValueError("Episode cannot support any configured native target count")


def _spread_indices(indices: list[int], count: int) -> list[int]:
    if count <= 0 or len(indices) < count:
        raise ValueError("Cannot spread more targets than the available genuine frames")
    if count == 1:
        return [indices[-1]]
    positions = [(index * (len(indices) - 1)) // (count - 1) for index in range(count)]
    selected = [indices[position] for position in positions]
    if len(set(selected)) != count:
        raise ValueError("Spread sampling produced duplicate targets")
    return selected


def _sample_native_action_window(
    episode: dict[str, Any],
    *,
    history_steps: int,
    core_target_counts: tuple[int, ...],
    rare_target_count: int | None,
    rare_target_probability: float,
    consecutive_probability: float,
    rng: random.Random,
) -> tuple[int, list[int], list[int], dict[str, Any]]:
    """Sample real VITRA frames with variable K inside the primary action interval.

    Dense samples take the next K valid native annotations.  Spread samples select K ordered,
    non-duplicated annotations over the remaining action, so query density can change without
    interpolating geometry or inventing targets.
    """
    if history_steps <= 0:
        raise ValueError("history_steps must be positive")
    if not core_target_counts or any(count <= 0 for count in core_target_counts):
        raise ValueError("core_target_counts must contain positive values")
    if tuple(sorted(set(core_target_counts))) != core_target_counts:
        raise ValueError("core_target_counts must be unique and increasing")
    if not 0.0 <= rare_target_probability <= 1.0:
        raise ValueError("rare_target_probability must lie in [0,1]")
    if not 0.0 <= consecutive_probability <= 1.0:
        raise ValueError("consecutive_probability must lie in [0,1]")

    primary, intervals = _primary_action_intervals(episode)
    kept = np.asarray(episode[primary]["kept_frames"], dtype=bool)
    length = len(episode["extrinsics"])
    if kept.shape != (length,):
        raise ValueError(f"Primary kept_frames must have shape ({length},)")

    requested_counts = list(core_target_counts)
    if rare_target_count is not None:
        if rare_target_count <= 0:
            raise ValueError("rare_target_count must be positive")
        requested_counts.append(rare_target_count)
    anchors_by_count: dict[int, list[tuple[int, tuple[int, int], list[int]]]] = {
        count: [] for count in requested_counts
    }
    for interval in intervals:
        start, end = interval
        first_anchor = max(start, history_steps - 1)
        for anchor in range(first_anchor, end - 1):
            if not kept[anchor]:
                continue
            valid_future = [index for index in range(anchor + 1, end) if kept[index]]
            for count in requested_counts:
                if len(valid_future) >= count:
                    anchors_by_count[count].append((anchor, interval, valid_future))
    supported = [count for count, anchors in anchors_by_count.items() if anchors]
    target_count = _choose_target_count(
        supported,
        core_target_counts=core_target_counts,
        rare_target_count=rare_target_count,
        rare_target_probability=rare_target_probability,
        rng=rng,
    )
    anchor, interval, valid_future = rng.choice(anchors_by_count[target_count])
    sampling_mode = "consecutive"
    if rng.random() < consecutive_probability:
        future = valid_future[:target_count]
    else:
        sampling_mode = "spread"
        future = _spread_indices(valid_future, target_count)
    history = list(range(anchor - history_steps + 1, anchor + 1))
    metadata = {
        "sampling_kind": "native_variable",
        "sampling_mode": sampling_mode,
        "target_count": target_count,
        "primary_hand": primary,
        "primary_action_interval": interval,
    }
    return anchor, history, future, metadata


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
        self.left_mano_policy = str(config.get("left_mano_policy", "as_stored"))

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
    """Streams variable-time windows and optional aligned DINO features from VITRA shards."""

    provides_context_visual = False
    provides_future_visual = False

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.shards = [Path(path) for path in sorted(glob.glob(str(config["shard_glob"])))]
        if not self.shards:
            raise FileNotFoundError(f"No VITRA shards match {config['shard_glob']}")
        sampling = dict(config.get("sampling", {}))
        self.sampling_kind = str(sampling.get("kind", "physical_offsets"))
        if self.sampling_kind == "physical_offsets":
            self.history_offsets_seconds = tuple(
                float(value) for value in config["history_offsets_seconds"]
            )
            self.future_offsets_seconds = tuple(
                float(value) for value in config["future_offsets_seconds"]
            )
            self.max_time_error_seconds = float(config["max_time_error_seconds"])
            _validated_time_offsets(
                self.history_offsets_seconds,
                self.future_offsets_seconds,
                self.max_time_error_seconds,
            )
        elif self.sampling_kind == "native_variable":
            self.history_steps = int(sampling.get("history_steps", 6))
            self.core_target_counts = tuple(
                int(value) for value in sampling.get("target_counts", (4, 8, 12, 16))
            )
            rare_count = sampling.get("rare_target_count", 24)
            self.rare_target_count = int(rare_count) if rare_count is not None else None
            self.rare_target_probability = float(sampling.get("rare_target_probability", 0.1))
            self.consecutive_probability = float(sampling.get("consecutive_probability", 0.7))
        else:
            raise ValueError(f"Unknown VITRA sampling kind: {self.sampling_kind}")
        fallback = config.get("fallback_fps")
        self.fallback_fps = float(fallback) if fallback is not None else None
        self.pts_root = config.get("pts_root")
        aliases = config.get("dataset_aliases", {})
        if not isinstance(aliases, dict) or any(
            not isinstance(source, str)
            or not source
            or not isinstance(destination, str)
            or not destination
            for source, destination in aliases.items()
        ):
            raise ValueError("data.dataset_aliases must map nonempty dataset names")
        self.dataset_aliases = dict(aliases)
        split_manifest = config.get("split_manifest")
        self.video_split = VitraVideoSplit.load(split_manifest) if split_manifest else None
        self.split = str(config.get("split", "train"))
        if self.video_split is not None:
            if self.split not in {"train", "validation"}:
                raise ValueError("data.split must be 'train' or 'validation'")
            validate_aliases(self.dataset_aliases, self.video_split)
        elif "split" in config:
            raise ValueError("data.split requires data.split_manifest")
        self.excluded_members = _load_excluded_members(config.get("exclusion_database"))
        self.left_mano_policy = str(config.get("left_mano_policy", "as_stored"))
        self.shuffle_buffer = int(config.get("shuffle_buffer", 0))
        feature_root = config.get("feature_root")
        self.feature_root = Path(feature_root) if feature_root else None
        unique_feature_root = config.get("unique_feature_root")
        staged_rgb_root = config.get("staged_rgb_root")
        self.unique_feature_root = Path(unique_feature_root) if unique_feature_root else None
        self.staged_rgb_root = Path(staged_rgb_root) if staged_rgb_root else None
        if self.feature_root is not None and self.unique_feature_root is not None:
            raise ValueError("Configure either feature_root or unique_feature_root, not both")
        if (self.unique_feature_root is None) != (self.staged_rgb_root is None):
            raise ValueError("unique_feature_root and staged_rgb_root must be configured together")
        self.attach_future_visual = bool(config.get("attach_future_visual", False))
        self.expected_extractor_id: str | None = None
        if self.feature_root is not None:
            success = validate_feature_root_success(self.feature_root)
            self.expected_extractor_id = str(success["extractor_id"])
            self.provides_context_visual = True
            self.provides_future_visual = self.attach_future_visual
        elif self.unique_feature_root is not None:
            self.provides_context_visual = True
            self.provides_future_visual = self.attach_future_visual
        elif self.attach_future_visual:
            raise ValueError("attach_future_visual requires a visual feature root")
        text_feature_root = config.get("text_feature_root")
        self.text_feature_root = Path(text_feature_root) if text_feature_root else None
        self._unique_store: UniqueVisualFeatureStore | None = None
        visual_feature_dtype = str(config.get("visual_feature_dtype", "float32"))
        if visual_feature_dtype not in {"float16", "float32"}:
            raise ValueError("data.visual_feature_dtype must be 'float16' or 'float32'")
        self.visual_feature_dtype = (
            np.float16 if visual_feature_dtype == "float16" else np.float32
        )
        self._text_store: DinoTxtTextFeatureStore | None = None
        self._iterator_count = 0

    def _get_unique_store(self) -> UniqueVisualFeatureStore:
        if self.unique_feature_root is None or self.staged_rgb_root is None:
            raise RuntimeError("Unique feature store is not configured")
        if self._unique_store is None:
            self._unique_store = UniqueVisualFeatureStore(
                feature_root=self.unique_feature_root,
                staged_rgb_root=self.staged_rgb_root,
                output_dtype=self.visual_feature_dtype,
            )
        return self._unique_store

    def _get_text_store(self) -> DinoTxtTextFeatureStore:
        if self.text_feature_root is None:
            raise RuntimeError("Text feature store is not configured")
        if self._text_store is None:
            self._text_store = DinoTxtTextFeatureStore(self.text_feature_root)
        return self._text_store

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

    def _episodes(
        self, shards: list[Path]
    ) -> Iterator[tuple[str, dict[str, Any], EpisodeFeatureRecord | None]]:
        for shard in shards:
            if self.feature_root is not None:
                feature_shard, _ = feature_shard_paths(self.feature_root, shard)
                yield from (
                    (member_name, episode, features)
                    for member_name, episode, features in iter_aligned_annotation_features(
                        shard,
                        feature_shard,
                        expected_extractor_id=self.expected_extractor_id,
                    )
                    if member_name not in self.excluded_members
                )
                continue
            with tarfile.open(shard, "r:") as archive:
                for member in archive:
                    if (
                        not member.isfile()
                        or not member.name.endswith(".npy")
                        or member.name in self.excluded_members
                    ):
                        continue
                    # Split identity is encoded in the member path.  Reject held-out training
                    # videos and non-selected validation episodes before reading/decompressing
                    # their NumPy payloads; frequent validation otherwise scans 1.2M episodes.
                    if self.video_split is not None:
                        logical_source, video_name = episode_member_identity(member.name)
                        if not self.video_split.includes(
                            self.split,
                            logical_source=logical_source,
                            video_name=video_name,
                            member_name=member.name,
                        ):
                            continue
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        continue
                    episode = np.load(io.BytesIO(extracted.read()), allow_pickle=True).item()
                    yield member.name, episode, None

    def __iter__(self) -> Iterator[dict[str, Any]]:
        buffer: list[dict[str, Any]] = []
        rng = self._iterator_rng()
        shards = self._assigned_shards()
        if self.split != "validation":
            rng.shuffle(shards)
        for member_name, episode, features in self._episodes(shards):
            dataset_name = member_name.split("/", 1)[0]
            storage_dataset_name = self.dataset_aliases.get(dataset_name, dataset_name)
            if self.video_split is not None and not self.video_split.includes(
                self.split,
                logical_source=dataset_name,
                video_name=str(episode["video_name"]),
                member_name=member_name,
            ):
                continue
            episode_rng = rng
            if self.split == "validation" and self.video_split is not None:
                episode_rng = random.Random(self.video_split.episode_seed(member_name))
            frame_times, time_source = _episode_frame_times(
                episode,
                dataset_name=storage_dataset_name,
                pts_root=self.pts_root,
                fallback_fps=self.fallback_fps,
            )
            try:
                if self.sampling_kind == "physical_offsets":
                    _, history, future = _sample_physical_time_window(
                        frame_times,
                        self.history_offsets_seconds,
                        self.future_offsets_seconds,
                        self.max_time_error_seconds,
                        episode_rng,
                    )
                    sampling_metadata = {"sampling_kind": "physical_offsets"}
                else:
                    _, history, future, sampling_metadata = _sample_native_action_window(
                        episode,
                        history_steps=self.history_steps,
                        core_target_counts=self.core_target_counts,
                        rare_target_count=self.rare_target_count,
                        rare_target_probability=self.rare_target_probability,
                        consecutive_probability=self.consecutive_probability,
                        rng=episode_rng,
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
            sample["metadata"].update(sampling_metadata)
            sample["metadata"]["horizon_seconds"] = float(
                frame_times[future[-1]] - frame_times[history[-1]]
            )
            if features is not None:
                pooled = torch.from_numpy(features.pooled_features.astype(np.float32))
                sample["context_visual_features"] = pooled[history]
                sample["metadata"]["visual_extractor_id"] = features.extractor_id
                if self.attach_future_visual:
                    sample["future_visual_latents"] = pooled[future]
            elif self.unique_feature_root is not None:
                physical = np.asarray(episode["video_decode_frame"], dtype=np.int64)
                selected = np.concatenate((np.asarray(history), np.asarray(future)))
                visual = self._get_unique_store().lookup(
                    storage_dataset_name,
                    str(episode["video_name"]),
                    physical[selected],
                )
                split = len(history)
                sample["context_visual_features"] = visual[:split]
                sample["metadata"]["visual_extractor_id"] = self._get_unique_store().success[
                    "extractor_id"
                ]
                if self.attach_future_visual:
                    sample["future_visual_latents"] = visual[split:]
            if self.text_feature_root is not None:
                sample["context_text_features"] = self._get_text_store().lookup(sample["text"])
                sample["metadata"]["text_extractor_id"] = self._get_text_store().success[
                    "extractor_id"
                ]
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
    if kind == "trajectory_h6k16":
        return CanonicalTrajectoryDataset(config)
    if kind == "hot3d_clips_h6k16":
        return Hot3DClipsForecastDataset(config)
    raise ValueError(f"Unknown dataset kind: {kind}")
