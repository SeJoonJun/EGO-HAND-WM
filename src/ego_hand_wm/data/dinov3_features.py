"""Offline DINOv3 extraction for VITRA annotation shards.

Only local repositories and local checkpoint files are accepted.  The extraction unit is an
annotation tar shard: episodes are grouped by source video, requested physical frame IDs are
decoded once per video within that shard, and pooled features are written to one aligned tar.
"""

from __future__ import annotations

import io
import os
import tarfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as functional

from ego_hand_wm.data.feature_shards import (
    FEATURE_CONTRACT,
    FEATURE_SCHEMA_VERSION,
    ROLE_CAPABILITIES,
    EpisodeFeatureRecord,
    FeatureShardError,
    annotation_member_to_feature_member,
    atomic_write_json,
    encode_feature_record,
    extractor_id,
    feature_shard_paths,
    load_feature_manifest,
    sha256_bytes,
    sha256_file,
)


VIDEO_SUFFIXES = frozenset({".mp4", ".webm"})
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PooledFeatureEncoder(Protocol):
    def encode(self, rgb_frames: np.ndarray) -> np.ndarray:
        """Return finite ``[B,D]`` pooled features for uint8 ``[B,H,W,3]`` RGB frames."""


FrameReader = Callable[[Path, Sequence[int]], Iterator[tuple[int, np.ndarray]]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_local_file(path: str | Path, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Local {description} is missing: {resolved}")
    return resolved


def _require_local_directory(path: str | Path, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Local {description} is missing: {resolved}")
    return resolved


def build_extractor_metadata(
    *,
    repo_path: str | Path,
    weights_path: str | Path,
    model_name: str,
    input_size: int,
) -> dict[str, Any]:
    repo = _require_local_directory(repo_path, "DINOv3 repository")
    weights = _require_local_file(weights_path, "DINOv3 weights")
    if input_size <= 0:
        raise ValueError("input_size must be positive")
    return {
        "backend": "torch_hub_local",
        "model_name": str(model_name),
        "repo_path": str(repo),
        "repo_hubconf_sha256": sha256_file(repo / "hubconf.py"),
        "weights_path": str(weights),
        "weights_size_bytes": weights.stat().st_size,
        "weights_sha256": sha256_file(weights),
        "input_size": int(input_size),
        "resize": "bilinear_square_antialias",
        "input_color": "rgb",
        "input_range": "uint8_0_255",
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "pooling": "mean_x_norm_patchtokens",
        "storage_dtype": "float16",
    }


class LocalDinoV3PooledEncoder:
    """Frozen local-only DINOv3 patch-token mean encoder."""

    def __init__(
        self,
        *,
        repo_path: str | Path,
        weights_path: str | Path,
        model_name: str = "dinov3_vitl16",
        input_size: int = 256,
        device: str = "auto",
    ) -> None:
        self.repo_path = _require_local_directory(repo_path, "DINOv3 repository")
        self.weights_path = _require_local_file(weights_path, "DINOv3 weights")
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        self.input_size = int(input_size)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for DINO extraction but is unavailable")

        # Passing a verified local checkpoint and source='local' prevents torch.hub downloads.
        self.model = torch.hub.load(
            str(self.repo_path),
            str(model_name),
            source="local",
            weights=str(self.weights_path),
        ).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def encode(self, rgb_frames: np.ndarray) -> np.ndarray:
        frames = np.asarray(rgb_frames)
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise ValueError(f"Expected uint8 RGB [B,H,W,3], got {frames.shape} {frames.dtype}")
        images = torch.from_numpy(frames.copy()).to(self.device)
        images = images.permute(0, 3, 1, 2).float().div_(255.0)
        images = functional.interpolate(
            images,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        images = (images - self.mean) / self.std
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            output = self.model.forward_features(images)
            if not isinstance(output, dict) or "x_norm_patchtokens" not in output:
                raise FeatureShardError(
                    "DINOv3 forward_features did not return x_norm_patchtokens"
                )
            pooled = output["x_norm_patchtokens"].mean(dim=1)
        result = pooled.float().cpu().numpy()
        if result.ndim != 2 or not np.isfinite(result).all():
            raise FeatureShardError("DINOv3 returned invalid pooled features")
        return result


class VideoResolver:
    """Resolve VITRA ``video_name`` by exact stem within a dataset-specific root."""

    def __init__(self, video_roots: dict[str, str | Path]) -> None:
        if not video_roots:
            raise ValueError("At least one dataset video root is required")
        self.roots = {
            str(dataset): _require_local_directory(root, f"video root for {dataset}")
            for dataset, root in video_roots.items()
        }
        self._indices: dict[str, dict[str, Path]] = {}

    def _build_index(self, dataset_name: str) -> dict[str, Path]:
        if dataset_name not in self.roots:
            raise FeatureShardError(f"No video root configured for dataset {dataset_name!r}")
        owners: dict[str, Path] = {}
        for path in self.roots[dataset_name].rglob("*"):
            if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            previous = owners.setdefault(path.stem, path)
            if previous != path:
                raise FeatureShardError(
                    f"Duplicate video stem {path.stem!r}: {previous} and {path}"
                )
        if not owners:
            raise FeatureShardError(f"No MP4/WebM files found under {self.roots[dataset_name]}")
        return owners

    def resolve(self, dataset_name: str, video_name: str) -> Path:
        index = self._indices.setdefault(dataset_name, self._build_index(dataset_name))
        try:
            return index[video_name]
        except KeyError as error:
            raise FileNotFoundError(
                f"No unique video stem {video_name!r} under {self.roots[dataset_name]}"
            ) from error


def decode_video_frames(path: Path, frame_ids: Sequence[int]) -> Iterator[tuple[int, np.ndarray]]:
    """Decode requested presentation-order frame IDs exactly once in ascending order."""
    import av

    requested = np.asarray(frame_ids, dtype=np.int64)
    if requested.ndim != 1 or len(requested) == 0:
        raise ValueError("frame_ids must be a non-empty one-dimensional sequence")
    if np.any(requested < 0) or np.any(np.diff(requested) <= 0):
        raise ValueError("frame_ids must be non-negative and strictly increasing")
    cursor = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for decoded_index, frame in enumerate(container.decode(stream)):
            target = int(requested[cursor])
            if decoded_index < target:
                continue
            if decoded_index != target:
                raise FeatureShardError(
                    f"Decoder skipped requested frame {target} in {path}; reached {decoded_index}"
                )
            yield target, frame.to_ndarray(format="rgb24")
            cursor += 1
            if cursor == len(requested):
                break
    if cursor != len(requested):
        raise FeatureShardError(
            f"Video {path} ended before requested frame {int(requested[cursor])}"
        )


@dataclass(frozen=True)
class AnnotationWork:
    member_name: str
    payload: bytes
    digest: str
    dataset_name: str
    video_name: str
    frame_ids: np.ndarray
    frame_times_seconds: np.ndarray


def _load_annotation_work(annotation_shard: Path, pts_root: Path) -> list[AnnotationWork]:
    pts_cache: dict[tuple[str, str], np.ndarray] = {}
    work: list[AnnotationWork] = []
    with tarfile.open(annotation_shard, "r:*") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".npy"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise FeatureShardError(f"Could not read annotation member {member.name}")
            payload = extracted.read()
            try:
                episode = np.load(io.BytesIO(payload), allow_pickle=True).item()
            except (OSError, TypeError, ValueError) as error:
                raise FeatureShardError(f"Invalid annotation member {member.name}") from error
            if not isinstance(episode, dict):
                raise FeatureShardError(f"Annotation is not a dictionary: {member.name}")
            parts = Path(member.name).parts
            if not parts:
                raise FeatureShardError(f"Cannot derive dataset namespace from {member.name}")
            dataset_name = parts[0]
            video_name = str(episode.get("video_name", ""))
            frame_ids = np.asarray(episode.get("video_decode_frame"), dtype=np.int64)
            if (
                not video_name
                or frame_ids.ndim != 1
                or len(frame_ids) == 0
                or np.any(frame_ids < 0)
                or np.any(np.diff(frame_ids) <= 0)
            ):
                raise FeatureShardError(f"Invalid video/frame identity in {member.name}")
            cache_key = (dataset_name, video_name)
            if cache_key not in pts_cache:
                pts_path = pts_root / dataset_name / f"{video_name}.npy"
                if not pts_path.is_file():
                    raise FileNotFoundError(f"Required physical PTS cache is missing: {pts_path}")
                video_times = np.load(pts_path, allow_pickle=False, mmap_mode="r")
                if (
                    video_times.ndim != 1
                    or len(video_times) == 0
                    or not np.isfinite(video_times).all()
                    or np.any(np.diff(video_times.astype(np.float64)) <= 0)
                ):
                    raise FeatureShardError(f"Invalid physical PTS cache: {pts_path}")
                pts_cache[cache_key] = video_times
            video_times = pts_cache[cache_key]
            if frame_ids[-1] >= len(video_times):
                raise FeatureShardError(
                    f"Frame {frame_ids[-1]} exceeds PTS cache for {member.name}"
                )
            work.append(
                AnnotationWork(
                    member_name=member.name,
                    payload=payload,
                    digest=sha256_bytes(payload),
                    dataset_name=dataset_name,
                    video_name=video_name,
                    frame_ids=frame_ids,
                    frame_times_seconds=np.asarray(video_times[frame_ids], dtype=np.float64),
                )
            )
    if not work:
        raise FeatureShardError(f"Annotation shard has no .npy episodes: {annotation_shard}")
    return work


def _encode_requested_frames(
    video_path: Path,
    frame_ids: np.ndarray,
    *,
    encoder: PooledFeatureEncoder,
    frame_reader: FrameReader,
    batch_size: int,
) -> dict[int, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    result: dict[int, np.ndarray] = {}
    batch_ids: list[int] = []
    batch_frames: list[np.ndarray] = []
    feature_dim: int | None = None

    def flush() -> None:
        nonlocal feature_dim
        if not batch_ids:
            return
        encoded = np.asarray(encoder.encode(np.stack(batch_frames, axis=0)))
        if encoded.ndim != 2 or encoded.shape[0] != len(batch_ids) or encoded.shape[1] <= 0:
            raise FeatureShardError(
                f"Encoder must return [B,D]; got {encoded.shape} for B={len(batch_ids)}"
            )
        if not np.issubdtype(encoded.dtype, np.floating) or not np.isfinite(encoded).all():
            raise FeatureShardError("Encoder returned non-finite or non-floating features")
        if feature_dim is None:
            feature_dim = int(encoded.shape[1])
        elif feature_dim != encoded.shape[1]:
            raise FeatureShardError("Encoder feature dimension changed between batches")
        for frame_id, feature in zip(batch_ids, encoded, strict=True):
            result[frame_id] = np.asarray(feature, dtype=np.float16)
        batch_ids.clear()
        batch_frames.clear()

    for frame_id, frame in frame_reader(video_path, frame_ids.tolist()):
        expected = int(frame_ids[len(result) + len(batch_ids)])
        if frame_id != expected:
            raise FeatureShardError(
                f"Frame reader order mismatch for {video_path}: expected {expected}, got {frame_id}"
            )
        rgb = np.asarray(frame)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise FeatureShardError(
                f"Frame reader must emit uint8 RGB [H,W,3], got {rgb.shape} {rgb.dtype}"
            )
        batch_ids.append(frame_id)
        batch_frames.append(rgb)
        if len(batch_ids) >= batch_size:
            flush()
    flush()
    missing = [int(frame_id) for frame_id in frame_ids if int(frame_id) not in result]
    if missing:
        raise FeatureShardError(f"Frame reader did not emit requested IDs: {missing[:10]}")
    return result


def _source_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def _existing_shard_is_valid(
    annotation_shard: Path,
    feature_path: Path,
    manifest_path: Path,
    expected_extractor_id: str,
) -> dict[str, Any] | None:
    if not feature_path.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = load_feature_manifest(manifest_path)
        source = manifest["annotation_shard"]
        stat = annotation_shard.stat()
        if (
            source["name"] != annotation_shard.name
            or int(source["size_bytes"]) != stat.st_size
            or int(source["mtime_ns"]) != stat.st_mtime_ns
            or manifest["extractor_id"] != expected_extractor_id
            or int(manifest["feature_size_bytes"]) != feature_path.stat().st_size
        ):
            return None
    except (FeatureShardError, KeyError, OSError, TypeError, ValueError):
        return None
    return manifest


def extract_feature_shard(
    *,
    annotation_shard: str | Path,
    output_root: str | Path,
    pts_root: str | Path,
    video_resolver: VideoResolver,
    encoder: PooledFeatureEncoder,
    extractor_metadata: dict[str, Any],
    frame_reader: FrameReader = decode_video_frames,
    batch_size: int = 32,
    force: bool = False,
) -> dict[str, Any]:
    """Extract one atomic feature tar and publish one complete/incomplete manifest."""
    annotation_path = _require_local_file(annotation_shard, "annotation shard")
    pts_path = _require_local_directory(pts_root, "PTS root")
    root = Path(output_root)
    feature_path, manifest_path = feature_shard_paths(root, annotation_path)
    current_extractor_id = extractor_id(extractor_metadata)
    if not force:
        existing = _existing_shard_is_valid(
            annotation_path, feature_path, manifest_path, current_extractor_id
        )
        if existing is not None:
            return {**existing, "status": "validated_skip"}

    root.mkdir(parents=True, exist_ok=True)
    # Any extraction attempt invalidates a previous global completion claim.
    (root / "_SUCCESS").unlink(missing_ok=True)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = feature_path.with_name(
        f".{feature_path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial"
    )
    started = utc_now()
    base_manifest: dict[str, Any] = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "contract": FEATURE_CONTRACT,
        "complete": False,
        "status": "failed",
        "started_at_utc": started,
        "annotation_shard": _source_fingerprint(annotation_path),
        "feature_shard": str(feature_path),
        "extractor_id": current_extractor_id,
        "extractor": extractor_metadata,
        "role_capabilities": list(ROLE_CAPABILITIES),
    }
    try:
        work = _load_annotation_work(annotation_path, pts_path)
        groups: dict[tuple[str, str], set[int]] = {}
        for episode in work:
            groups.setdefault((episode.dataset_name, episode.video_name), set()).update(
                int(frame_id) for frame_id in episode.frame_ids
            )
        feature_maps: dict[tuple[str, str], dict[int, np.ndarray]] = {}
        for (dataset_name, video_name), ids in groups.items():
            requested = np.asarray(sorted(ids), dtype=np.int64)
            video_path = video_resolver.resolve(dataset_name, video_name)
            feature_maps[(dataset_name, video_name)] = _encode_requested_frames(
                video_path,
                requested,
                encoder=encoder,
                frame_reader=frame_reader,
                batch_size=batch_size,
            )

        logical_digest = torch.sha256 if False else None  # keep torch out of tar hashing
        import hashlib

        digest = hashlib.sha256()
        feature_dim: int | None = None
        with tarfile.open(temporary, "w") as writer:
            for episode in work:
                mapping = feature_maps[(episode.dataset_name, episode.video_name)]
                pooled = np.stack([mapping[int(frame_id)] for frame_id in episode.frame_ids])
                if feature_dim is None:
                    feature_dim = int(pooled.shape[1])
                elif feature_dim != pooled.shape[1]:
                    raise FeatureShardError("Feature dimension changed across source videos")
                record = EpisodeFeatureRecord(
                    annotation_member=episode.member_name,
                    annotation_sha256=episode.digest,
                    dataset_name=episode.dataset_name,
                    video_name=episode.video_name,
                    frame_ids=episode.frame_ids,
                    frame_times_seconds=episode.frame_times_seconds,
                    pooled_features=pooled,
                    valid_mask=np.ones(len(episode.frame_ids), dtype=np.bool_),
                    extractor_id=current_extractor_id,
                )
                payload = encode_feature_record(record)
                member_name = annotation_member_to_feature_member(episode.member_name)
                tar_info = tarfile.TarInfo(member_name)
                tar_info.size = len(payload)
                tar_info.mode = 0o644
                tar_info.mtime = 0
                writer.addfile(tar_info, io.BytesIO(payload))
                digest.update(member_name.encode("utf-8"))
                digest.update(sha256_bytes(payload).encode("ascii"))
        if feature_dim is None:
            raise FeatureShardError("No feature records were written")
        os.replace(temporary, feature_path)
        manifest = {
            **base_manifest,
            "complete": True,
            "status": "rebuilt",
            "finished_at_utc": utc_now(),
            "episode_count": len(work),
            "source_video_count": len(groups),
            "unique_decoded_frames": sum(len(ids) for ids in groups.values()),
            "feature_dim": feature_dim,
            "storage_dtype": "float16",
            "feature_size_bytes": feature_path.stat().st_size,
            "logical_content_sha256": digest.hexdigest(),
        }
        atomic_write_json(manifest_path, manifest)
        return manifest
    except Exception as error:
        failure = {
            **base_manifest,
            "finished_at_utc": utc_now(),
            "error_type": type(error).__name__,
            "message": str(error),
        }
        atomic_write_json(manifest_path, failure)
        raise
    finally:
        temporary.unlink(missing_ok=True)
