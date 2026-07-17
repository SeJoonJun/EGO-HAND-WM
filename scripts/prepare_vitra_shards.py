#!/usr/bin/env python3
"""Repack the five sealed VITRA archives into validated, resumable tar shards.

The annotation payloads are copied byte-for-byte and are never extracted onto the filesystem.
Published shards, progress state, the final manifest, and ``_SUCCESS`` are all written
transactionally.  ``_SUCCESS`` is created only after the expected per-source episode counts and
all published shard metadata have been validated.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import tarfile
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO


SCHEMA_VERSION = 1
EXPECTED_SOURCE_COUNTS: dict[str, int] = {
    "ego4d_cooking_and_cleaning": 454_244,
    "ego4d_other": 494_439,
    "epic": 154_464,
    "egoexo4d": 67_053,
    "ssv2": 52_718,
}
EXPECTED_TOTAL_EPISODES = sum(EXPECTED_SOURCE_COUNTS.values())
STATE_FILENAME = ".shard-state.json"
MANIFEST_FILENAME = "manifest.json"
JSONL_FILENAME = "shards.jsonl"
SUCCESS_FILENAME = "_SUCCESS"
LOCK_FILENAME = ".shard.lock"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "archives",
        nargs="+",
        type=Path,
        help=(
            "The five VITRA .tar.gz archives. Each basename (without .tar.gz) must equal an "
            "expected source name; command-line order is ignored."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-shard", type=int, default=2048)
    parser.add_argument("--prefix", default="vitra")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _archive_source_name(path: Path) -> str:
    name = path.name
    if name.endswith(".tar.gz"):
        return name[: -len(".tar.gz")]
    if name.endswith(".tgz"):
        return name[: -len(".tgz")]
    raise ValueError(f"Expected a .tar.gz or .tgz archive, got {path}")


def order_archives(
    archives: Sequence[Path], expected_counts: Mapping[str, int]
) -> list[tuple[str, Path]]:
    by_source: dict[str, Path] = {}
    for archive in archives:
        source = _archive_source_name(archive)
        if source not in expected_counts:
            raise ValueError(f"Unexpected VITRA source archive {archive.name!r}")
        if source in by_source:
            raise ValueError(f"Duplicate archive for VITRA source {source!r}")
        if not archive.is_file():
            raise FileNotFoundError(archive)
        by_source[source] = archive
    missing = set(expected_counts).difference(by_source)
    if missing:
        raise ValueError(f"Missing VITRA source archives: {sorted(missing)}")
    return [(source, by_source[source]) for source in expected_counts]


def archive_identity(source: str, path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "source": source,
        "path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def make_run_spec(
    ordered_archives: Sequence[tuple[str, Path]],
    expected_counts: Mapping[str, int],
    *,
    prefix: str,
    episodes_per_shard: int,
) -> dict[str, Any]:
    if episodes_per_shard <= 0:
        raise ValueError("episodes_per_shard must be positive")
    if not prefix or "/" in prefix:
        raise ValueError("prefix must be a non-empty filename prefix")
    counts = {source: int(count) for source, count in expected_counts.items()}
    if any(count < 0 for count in counts.values()):
        raise ValueError("Expected source counts must be non-negative")
    return {
        "schema_version": SCHEMA_VERSION,
        "prefix": prefix,
        "episodes_per_shard": episodes_per_shard,
        "expected_source_counts": counts,
        "expected_total_episodes": sum(counts.values()),
        "source_archives": [archive_identity(source, path) for source, path in ordered_archives],
    }


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


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


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(path, _json_bytes(payload))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


class _DigestingWriter:
    """A write-only file wrapper that hashes the tar stream without a second disk read."""

    def __init__(self, handle: BinaryIO) -> None:
        self.handle = handle
        self.digest = hashlib.sha256()
        self.bytes_written = 0

    def write(self, data: bytes) -> int:
        written = self.handle.write(data)
        if written != len(data):
            raise OSError(f"Short shard write: expected {len(data)} bytes, wrote {written}")
        self.digest.update(data)
        self.bytes_written += written
        return written


class _ShardWriter:
    def __init__(self, output_dir: Path, prefix: str, index: int) -> None:
        self.index = index
        self.final_path = output_dir / f"{prefix}-{index:06d}.tar"
        self.partial_path = output_dir / f".{prefix}-{index:06d}.tar.partial"
        self.handle = self.partial_path.open("wb")
        self.digesting = _DigestingWriter(self.handle)
        self.tar = tarfile.open(fileobj=self.digesting, mode="w|")
        self.episodes = 0
        self.source_counts: dict[str, int] = {}

    def add(self, source: str, member: tarfile.TarInfo, payload: BinaryIO) -> None:
        target = tarfile.TarInfo(name=member.name.lstrip("./"))
        target.size = member.size
        target.mode = 0o644
        self.tar.addfile(target, payload)
        self.episodes += 1
        self.source_counts[source] = self.source_counts.get(source, 0) + 1

    def publish(self, output_dir: Path) -> dict[str, Any]:
        if self.episodes <= 0:
            raise RuntimeError("Refusing to publish an empty shard")
        self.tar.close()
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        if self.partial_path.stat().st_size != self.digesting.bytes_written:
            raise OSError(f"Shard size changed before publication: {self.partial_path}")
        os.replace(self.partial_path, self.final_path)
        _fsync_directory(output_dir)
        return {
            "index": self.index,
            "shard": self.final_path.name,
            "episodes": self.episodes,
            "source_counts": dict(sorted(self.source_counts.items())),
            "bytes": self.digesting.bytes_written,
            "sha256": self.digesting.digest.hexdigest(),
        }

    def abort(self) -> None:
        try:
            self.tar.close()
        finally:
            if not self.handle.closed:
                self.handle.close()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read valid JSON from {path}") from error


def _sum_source_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for record in records:
        for source, count in record["source_counts"].items():
            totals[source] = totals.get(source, 0) + int(count)
    return totals


def validate_records(
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    run_spec: Mapping[str, Any],
    *,
    verify_hashes: bool,
    require_complete: bool,
) -> None:
    episodes_per_shard = int(run_spec["episodes_per_shard"])
    prefix = str(run_spec["prefix"])
    expected_sources = set(run_spec["expected_source_counts"])
    for index, record in enumerate(records):
        expected_name = f"{prefix}-{index:06d}.tar"
        if int(record.get("index", -1)) != index or record.get("shard") != expected_name:
            raise RuntimeError(f"Non-contiguous or misnamed shard record at index {index}")
        episodes = int(record.get("episodes", 0))
        if episodes <= 0 or episodes > episodes_per_shard:
            raise RuntimeError(f"Invalid episode count for {expected_name}: {episodes}")
        if index < len(records) - 1 and episodes != episodes_per_shard:
            raise RuntimeError(f"Only the final shard may be short: {expected_name}")
        source_counts = record.get("source_counts", {})
        if not isinstance(source_counts, dict) or set(source_counts).difference(expected_sources):
            raise RuntimeError(f"Invalid source counts for {expected_name}")
        if sum(int(value) for value in source_counts.values()) != episodes:
            raise RuntimeError(f"Source counts do not sum to episodes for {expected_name}")
        shard_path = output_dir / expected_name
        if not shard_path.is_file() or shard_path.stat().st_size != int(record.get("bytes", -1)):
            raise RuntimeError(f"Missing or size-mismatched shard: {shard_path}")
        expected_hash = str(record.get("sha256", ""))
        if len(expected_hash) != 64:
            raise RuntimeError(f"Missing SHA-256 for {expected_name}")
        if verify_hashes and sha256_file(shard_path) != expected_hash:
            raise RuntimeError(f"SHA-256 mismatch for {shard_path}")

    if require_complete:
        observed = _sum_source_counts(records)
        expected = {key: int(value) for key, value in run_spec["expected_source_counts"].items()}
        if observed != expected:
            raise RuntimeError(
                f"Final source counts differ: observed={observed}, expected={expected}"
            )
        if sum(int(record["episodes"]) for record in records) != int(
            run_spec["expected_total_episodes"]
        ):
            raise RuntimeError("Final total episode count differs from expected total")
        expected_files = {str(record["shard"]) for record in records}
        actual_files = {path.name for path in output_dir.glob(f"{prefix}-*.tar")}
        if actual_files != expected_files:
            raise RuntimeError(
                f"Unexpected published shards: {sorted(actual_files.difference(expected_files))}"
            )


def _write_state(path: Path, run_spec: Mapping[str, Any], records: Sequence[Any]) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "updated_at_utc": utc_now(),
            "run": run_spec,
            "shards": list(records),
        },
    )


def _validate_completed_run(output_dir: Path, run_spec: Mapping[str, Any]) -> dict[str, Any]:
    success_path = output_dir / SUCCESS_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    success = _load_json(success_path)
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != success.get("manifest_sha256"):
        raise RuntimeError("_SUCCESS does not match manifest.json")
    manifest = json.loads(manifest_bytes)
    if manifest.get("run") != run_spec or not manifest.get("complete"):
        raise RuntimeError("Completed manifest does not match this invocation")
    validate_records(
        output_dir,
        manifest["shards"],
        run_spec,
        verify_hashes=False,
        require_complete=True,
    )
    return manifest


def prepare_vitra_shards(
    archives: Sequence[Path],
    *,
    output_dir: Path,
    episodes_per_shard: int = 2048,
    prefix: str = "vitra",
    expected_counts: Mapping[str, int] = EXPECTED_SOURCE_COUNTS,
) -> dict[str, Any]:
    ordered = order_archives(archives, expected_counts)
    run_spec = make_run_spec(
        ordered,
        expected_counts,
        prefix=prefix,
        episodes_per_shard=episodes_per_shard,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (output_dir / LOCK_FILENAME).open("a+b")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_handle.close()
        raise RuntimeError(f"Another sharding process holds {output_dir / LOCK_FILENAME}") from error
    state_path = output_dir / STATE_FILENAME
    success_path = output_dir / SUCCESS_FILENAME

    if success_path.is_file():
        manifest = _validate_completed_run(output_dir, run_spec)
        print(json.dumps({"status": "already_complete", **manifest["summary"]}), flush=True)
        return manifest

    records: list[dict[str, Any]] = []
    if state_path.exists():
        state = _load_json(state_path)
        if state.get("run") != run_spec:
            raise RuntimeError(f"Existing {STATE_FILENAME} belongs to a different run")
        records = list(state.get("shards", []))
        validate_records(
            output_dir,
            records,
            run_spec,
            verify_hashes=True,
            require_complete=False,
        )
    else:
        existing = sorted(output_dir.glob(f"{prefix}-*.tar"))
        first_orphan = output_dir / f"{prefix}-000000.tar"
        if existing and existing != [first_orphan]:
            raise RuntimeError(
                "Published shards exist without resumable state; use a clean output directory"
            )
        _write_state(state_path, run_spec, records)

    next_index = len(records)
    expected_next_orphan = output_dir / f"{prefix}-{next_index:06d}.tar"
    recorded_names = {str(record["shard"]) for record in records}
    extras = [
        path
        for path in output_dir.glob(f"{prefix}-*.tar")
        if path.name not in recorded_names and path != expected_next_orphan
    ]
    if extras:
        raise RuntimeError(f"Unexpected unrecorded shards: {[path.name for path in extras]}")

    completed_episodes = sum(int(record["episodes"]) for record in records)
    recorded_source_counts = _sum_source_counts(records)
    if records and int(records[-1]["episodes"]) < episodes_per_shard:
        final_short_shard = True
    else:
        final_short_shard = False

    seen_episodes = 0
    observed_counts = {source: 0 for source in expected_counts}
    writer: _ShardWriter | None = None
    prefix_checked = completed_episodes == 0
    try:
        for source, archive in ordered:
            with tarfile.open(archive, "r|gz") as source_tar:
                for member in source_tar:
                    if not member.isfile() or not member.name.endswith(".npy"):
                        continue
                    seen_episodes += 1
                    observed_counts[source] += 1
                    if observed_counts[source] > int(expected_counts[source]):
                        raise RuntimeError(
                            f"{source} exceeds expected count {expected_counts[source]}"
                        )
                    if seen_episodes <= completed_episodes:
                        if seen_episodes == completed_episodes:
                            prefix_observed = {
                                key: value for key, value in observed_counts.items() if value
                            }
                            if prefix_observed != recorded_source_counts:
                                raise RuntimeError(
                                    "Source archive contents no longer match the recorded prefix"
                                )
                            prefix_checked = True
                        continue
                    if final_short_shard:
                        raise RuntimeError("A recorded short shard is not the final dataset shard")
                    extracted = source_tar.extractfile(member)
                    if extracted is None:
                        raise OSError(f"Could not read {member.name} from {archive}")
                    if writer is None:
                        writer = _ShardWriter(output_dir, prefix, len(records))
                    writer.add(source, member, extracted)
                    if writer.episodes == episodes_per_shard:
                        record = writer.publish(output_dir)
                        writer = None
                        records.append(record)
                        _write_state(state_path, run_spec, records)
                        print(json.dumps(record, sort_keys=True), flush=True)
            if observed_counts[source] != int(expected_counts[source]):
                raise RuntimeError(
                    f"{source} has {observed_counts[source]} episodes; "
                    f"expected {expected_counts[source]}"
                )
        if not prefix_checked:
            raise RuntimeError(
                f"Archives contain fewer than the {completed_episodes} recorded episodes"
            )
        if writer is not None:
            record = writer.publish(output_dir)
            writer = None
            records.append(record)
            _write_state(state_path, run_spec, records)
            print(json.dumps(record, sort_keys=True), flush=True)
    except Exception:
        if writer is not None:
            writer.abort()
        raise

    if seen_episodes != int(run_spec["expected_total_episodes"]):
        raise RuntimeError(
            f"Observed {seen_episodes} total episodes; expected "
            f"{run_spec['expected_total_episodes']}"
        )
    final_archive_identities = [
        archive_identity(source, archive) for source, archive in ordered
    ]
    if final_archive_identities != run_spec["source_archives"]:
        raise RuntimeError("A source archive changed while shards were being prepared")
    validate_records(
        output_dir,
        records,
        run_spec,
        verify_hashes=False,
        require_complete=True,
    )

    summary = {
        "total_episodes": seen_episodes,
        "total_shards": len(records),
        "total_bytes": sum(int(record["bytes"]) for record in records),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "created_at_utc": utc_now(),
        "run": run_spec,
        "observed_source_counts": observed_counts,
        "summary": summary,
        "shards": records,
    }
    jsonl = b"".join(
        (json.dumps(record, sort_keys=True) + "\n").encode("utf-8") for record in records
    )
    atomic_write_bytes(output_dir / JSONL_FILENAME, jsonl)
    manifest_bytes = _json_bytes(manifest)
    atomic_write_bytes(output_dir / MANIFEST_FILENAME, manifest_bytes)

    # Re-read the final manifest and validate all sizes/counts before publishing the success gate.
    persisted_manifest = _load_json(output_dir / MANIFEST_FILENAME)
    if persisted_manifest != manifest:
        raise RuntimeError("Persisted final manifest does not round-trip")
    validate_records(
        output_dir,
        records,
        run_spec,
        verify_hashes=False,
        require_complete=True,
    )
    atomic_write_json(
        success_path,
        {
            "schema_version": SCHEMA_VERSION,
            "manifest": MANIFEST_FILENAME,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            **summary,
        },
    )
    print(json.dumps({"status": "complete", **summary}), flush=True)
    return manifest


def main() -> None:
    args = parse_args()
    prepare_vitra_shards(
        args.archives,
        output_dir=args.output_dir,
        episodes_per_shard=args.episodes_per_shard,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
