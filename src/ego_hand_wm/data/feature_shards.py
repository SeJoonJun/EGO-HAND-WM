"""Compact, streaming feature-shard contract for VITRA visual supervision.

One feature tar is paired with one byte-preserving annotation tar.  Every annotation ``.npy``
member has one same-path ``.npz`` feature member, so training can stream both archives in lockstep
without creating 1.2 million filesystem entries or loading a whole shard into memory.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import uuid
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator

import numpy as np


FEATURE_SCHEMA_VERSION = 2
FEATURE_CONTRACT = "ego_hand_wm.vitra_spatial_visual_features"
ROLE_CAPABILITIES = ("context", "future_target")


class FeatureShardError(ValueError):
    """An annotation/feature pair cannot be joined without guessing."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def extractor_id(metadata: dict[str, Any]) -> str:
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def annotation_member_to_feature_member(annotation_member: str) -> str:
    path = PurePosixPath(annotation_member)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".npy":
        raise FeatureShardError(
            f"Annotation member must be a safe relative .npy path, got {annotation_member!r}"
        )
    return str(path.with_suffix(".npz"))


@dataclass(frozen=True)
class EpisodeFeatureRecord:
    annotation_member: str
    annotation_sha256: str
    dataset_name: str
    video_name: str
    frame_ids: np.ndarray
    frame_times_seconds: np.ndarray
    pooled_features: np.ndarray
    valid_mask: np.ndarray
    extractor_id: str

    @property
    def feature_dim(self) -> int:
        return int(self.pooled_features.shape[2])

    @property
    def spatial_tokens(self) -> int:
        return int(self.pooled_features.shape[1])

    def validate(self) -> None:
        annotation_member_to_feature_member(self.annotation_member)
        if len(self.annotation_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.annotation_sha256
        ):
            raise FeatureShardError("annotation_sha256 must be a lowercase SHA-256 digest")
        if not self.dataset_name or not self.video_name:
            raise FeatureShardError("dataset_name and video_name must be non-empty")
        if len(self.extractor_id) != 64 or any(
            character not in "0123456789abcdef" for character in self.extractor_id
        ):
            raise FeatureShardError("extractor_id must be a lowercase SHA-256 digest")

        frame_ids = np.asarray(self.frame_ids)
        times = np.asarray(self.frame_times_seconds)
        features = np.asarray(self.pooled_features)
        valid = np.asarray(self.valid_mask)
        if frame_ids.ndim != 1 or not np.issubdtype(frame_ids.dtype, np.integer):
            raise FeatureShardError("frame_ids must be a one-dimensional integer array")
        if len(frame_ids) == 0:
            raise FeatureShardError("An episode feature record cannot be empty")
        if np.any(frame_ids < 0) or np.any(np.diff(frame_ids.astype(np.int64)) <= 0):
            raise FeatureShardError("frame_ids must be non-negative and strictly increasing")
        if times.shape != frame_ids.shape or not np.issubdtype(times.dtype, np.floating):
            raise FeatureShardError("frame_times_seconds must be floating [T]")
        if not np.isfinite(times).all() or np.any(np.diff(times.astype(np.float64)) <= 0):
            raise FeatureShardError("frame_times_seconds must be finite and strictly increasing")
        if (
            features.ndim != 3
            or features.shape[0] != len(frame_ids)
            or features.shape[1] <= 0
            or features.shape[2] <= 0
        ):
            raise FeatureShardError("pooled_features must be [T,P,D] with P,D > 0")
        if not np.issubdtype(features.dtype, np.floating) or not np.isfinite(features).all():
            raise FeatureShardError("pooled_features must contain finite floating-point values")
        if valid.shape != frame_ids.shape or valid.dtype != np.bool_:
            raise FeatureShardError("valid_mask must be boolean [T]")
        if not valid.all():
            raise FeatureShardError(
                "Partial feature records are forbidden; an incomplete decode must fail the shard"
            )


def encode_feature_record(record: EpisodeFeatureRecord) -> bytes:
    record.validate()
    buffer = io.BytesIO()
    np.savez(
        buffer,
        schema_version=np.asarray(FEATURE_SCHEMA_VERSION, dtype=np.int64),
        contract=np.asarray(FEATURE_CONTRACT),
        annotation_member=np.asarray(record.annotation_member),
        annotation_sha256=np.asarray(record.annotation_sha256),
        dataset_name=np.asarray(record.dataset_name),
        video_name=np.asarray(record.video_name),
        frame_ids=np.asarray(record.frame_ids, dtype=np.int64),
        frame_times_seconds=np.asarray(record.frame_times_seconds, dtype=np.float64),
        pooled_features=np.asarray(record.pooled_features, dtype=np.float16),
        valid_mask=np.asarray(record.valid_mask, dtype=np.bool_),
        extractor_id=np.asarray(record.extractor_id),
    )
    return buffer.getvalue()


def _scalar_text(archive: Any, key: str) -> str:
    value = np.asarray(archive[key])
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise FeatureShardError(f"{key} must be a scalar string")
    item = value.item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def decode_feature_record(payload: bytes) -> EpisodeFeatureRecord:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            version = int(np.asarray(archive["schema_version"]).item())
            contract = _scalar_text(archive, "contract")
            if version != FEATURE_SCHEMA_VERSION or contract != FEATURE_CONTRACT:
                raise FeatureShardError(
                    f"Unsupported feature record: schema={version}, contract={contract!r}"
                )
            record = EpisodeFeatureRecord(
                annotation_member=_scalar_text(archive, "annotation_member"),
                annotation_sha256=_scalar_text(archive, "annotation_sha256"),
                dataset_name=_scalar_text(archive, "dataset_name"),
                video_name=_scalar_text(archive, "video_name"),
                frame_ids=np.asarray(archive["frame_ids"], dtype=np.int64).copy(),
                frame_times_seconds=np.asarray(
                    archive["frame_times_seconds"], dtype=np.float64
                ).copy(),
                pooled_features=np.asarray(archive["pooled_features"], dtype=np.float16).copy(),
                valid_mask=np.asarray(archive["valid_mask"], dtype=np.bool_).copy(),
                extractor_id=_scalar_text(archive, "extractor_id"),
            )
    except (KeyError, OSError, TypeError, ValueError) as error:
        if isinstance(error, FeatureShardError):
            raise
        raise FeatureShardError("Invalid or incomplete feature NPZ member") from error
    record.validate()
    return record


def feature_shard_paths(
    output_root: str | Path, annotation_shard: str | Path
) -> tuple[Path, Path]:
    root = Path(output_root)
    name = Path(annotation_shard).name
    return root / "shards" / name, root / "_manifests" / f"{name}.json"


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def load_feature_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureShardError(f"Cannot read feature manifest: {manifest_path}") from error
    if manifest.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise FeatureShardError(f"Unsupported feature manifest: {manifest_path}")
    if manifest.get("contract") != FEATURE_CONTRACT or not manifest.get("complete", False):
        raise FeatureShardError(f"Feature manifest is not complete: {manifest_path}")
    return manifest


def validate_feature_root_success(output_root: str | Path) -> dict[str, Any]:
    success_path = Path(output_root) / "_SUCCESS"
    try:
        success = json.loads(success_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureShardError(f"Missing or invalid feature completion gate: {success_path}") from error
    if (
        success.get("schema_version") != FEATURE_SCHEMA_VERSION
        or success.get("contract") != FEATURE_CONTRACT
        or not success.get("complete", False)
    ):
        raise FeatureShardError(f"Feature completion gate is not valid: {success_path}")
    return success


def _next_payload(archive: tarfile.TarFile, suffix: str) -> Iterator[tuple[str, bytes]]:
    for member in archive:
        if not member.isfile() or not member.name.endswith(suffix):
            continue
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FeatureShardError(f"Could not read tar member: {member.name}")
        yield member.name, extracted.read()


def iter_aligned_annotation_features(
    annotation_shard: str | Path,
    feature_shard: str | Path,
    *,
    expected_extractor_id: str | None = None,
) -> Iterator[tuple[str, dict[str, Any], EpisodeFeatureRecord]]:
    """Stream and verify a one-to-one annotation/feature tar pair in member order."""
    with tarfile.open(annotation_shard, "r:*") as annotations, tarfile.open(
        feature_shard, "r:"
    ) as features:
        pairs = zip_longest(
            _next_payload(annotations, ".npy"),
            _next_payload(features, ".npz"),
        )
        for annotation_item, feature_item in pairs:
            if annotation_item is None or feature_item is None:
                raise FeatureShardError("Annotation and feature shard member counts differ")
            annotation_name, annotation_payload = annotation_item
            feature_name, feature_payload = feature_item
            expected_name = annotation_member_to_feature_member(annotation_name)
            if feature_name != expected_name:
                raise FeatureShardError(
                    f"Feature member order/name mismatch: expected {expected_name}, got {feature_name}"
                )
            record = decode_feature_record(feature_payload)
            if record.annotation_member != annotation_name:
                raise FeatureShardError("Feature record annotation_member does not match tar path")
            if record.annotation_sha256 != sha256_bytes(annotation_payload):
                raise FeatureShardError(f"Annotation payload digest mismatch: {annotation_name}")
            if expected_extractor_id is not None and record.extractor_id != expected_extractor_id:
                raise FeatureShardError(
                    f"Extractor mismatch for {annotation_name}: {record.extractor_id}"
                )
            try:
                episode = np.load(io.BytesIO(annotation_payload), allow_pickle=True).item()
            except (OSError, TypeError, ValueError) as error:
                raise FeatureShardError(f"Invalid VITRA annotation: {annotation_name}") from error
            if not isinstance(episode, dict):
                raise FeatureShardError(f"VITRA annotation is not a dictionary: {annotation_name}")
            annotation_frame_ids = np.asarray(episode.get("video_decode_frame"), dtype=np.int64)
            if not np.array_equal(annotation_frame_ids, record.frame_ids):
                raise FeatureShardError(f"Physical frame IDs do not align: {annotation_name}")
            if str(episode.get("video_name", "")) != record.video_name:
                raise FeatureShardError(f"Video identity does not align: {annotation_name}")
            yield annotation_name, episode, record


def finalize_feature_root(
    annotation_shards: Iterable[str | Path], output_root: str | Path
) -> dict[str, Any]:
    """Validate all expected per-shard manifests and atomically publish the global gate."""
    shards = [Path(path) for path in annotation_shards]
    if not shards:
        raise FeatureShardError("No annotation shards were supplied to finalization")
    names = [path.name for path in shards]
    if len(names) != len(set(names)):
        raise FeatureShardError("Annotation shard basenames must be unique")

    root = Path(output_root)
    records: list[dict[str, Any]] = []
    extractor_ids: set[str] = set()
    feature_dims: set[int] = set()
    spatial_token_counts: set[int] = set()
    total_episodes = 0
    for annotation_shard in sorted(shards):
        feature_path, manifest_path = feature_shard_paths(root, annotation_shard)
        manifest = load_feature_manifest(manifest_path)
        stat = annotation_shard.stat()
        annotation_fingerprint = manifest.get("annotation_shard", {})
        if (
            annotation_fingerprint.get("name") != annotation_shard.name
            or int(annotation_fingerprint.get("size_bytes", -1)) != stat.st_size
            or int(annotation_fingerprint.get("mtime_ns", -1)) != stat.st_mtime_ns
        ):
            raise FeatureShardError(f"Annotation shard changed after extraction: {annotation_shard}")
        if not feature_path.is_file() or int(manifest.get("feature_size_bytes", -1)) != feature_path.stat().st_size:
            raise FeatureShardError(f"Feature shard missing or changed: {feature_path}")
        extractor_ids.add(str(manifest["extractor_id"]))
        feature_dims.add(int(manifest["feature_dim"]))
        spatial_token_counts.add(int(manifest["spatial_tokens"]))
        episode_count = int(manifest["episode_count"])
        total_episodes += episode_count
        records.append(
            {
                "annotation_shard": annotation_shard.name,
                "feature_shard": str(feature_path.relative_to(root)),
                "manifest": str(manifest_path.relative_to(root)),
                "episodes": episode_count,
            }
        )
    if len(extractor_ids) != 1 or len(feature_dims) != 1 or len(spatial_token_counts) != 1:
        raise FeatureShardError(
            "All feature shards must use exactly one extractor, feature dimension, and grid"
        )

    success = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "contract": FEATURE_CONTRACT,
        "complete": True,
        "role_capabilities": list(ROLE_CAPABILITIES),
        "extractor_id": next(iter(extractor_ids)),
        "feature_dim": next(iter(feature_dims)),
        "spatial_tokens": next(iter(spatial_token_counts)),
        "annotation_shards": len(shards),
        "episodes": total_episodes,
        "records": records,
    }
    atomic_write_json(root / "_SUCCESS", success)
    return success
