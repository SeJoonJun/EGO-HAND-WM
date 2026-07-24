from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_video_pts_index.py"
SPEC = importlib.util.spec_from_file_location("build_video_pts_index", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
pts_index = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pts_index)


def _write_video_placeholder(path: Path, payload: bytes = b"video") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _reader(path: Path) -> tuple[np.ndarray, str]:
    offset = float(len(path.read_bytes()))
    return np.asarray([offset, offset + 0.1, offset + 0.2], dtype=np.float64), "test_reader"


def test_discovery_is_case_insensitive_and_partition_is_exact(tmp_path: Path) -> None:
    root = tmp_path / "videos"
    expected = [root / "a.mp4", root / "nested/b.MP4", root / "nested/c.webm"]
    for path in expected:
        _write_video_placeholder(path)
    _write_video_placeholder(root / "ignored.mov")

    videos = pts_index.discover_videos(root)
    assert videos == expected
    partitions = [pts_index.assigned_videos(videos, shard, 2) for shard in range(2)]
    flattened = partitions[0] + partitions[1]
    assert sorted(flattened) == expected
    assert set(partitions[0]).isdisjoint(partitions[1])


def test_duplicate_pts_repair_preserves_frames_and_surrounding_times() -> None:
    tick = 1.0 / 90_000.0
    source = np.asarray([0.0, 1.0 / 30.0, 1.0 / 30.0, 2.0 / 30.0])

    repaired = pts_index._repair_duplicate_timestamps(source, minimum_step=tick)

    assert len(repaired) == len(source)
    assert np.all(np.diff(repaired) > 0)
    assert repaired[0] == source[0]
    assert repaired[1] == source[1]
    assert repaired[2] == source[2] + tick
    assert repaired[3] == source[3]


def test_cache_skip_requires_matching_metadata_and_source_fingerprint(tmp_path: Path) -> None:
    root = tmp_path / "videos"
    video = root / "sample.MP4"
    _write_video_placeholder(video, b"first")
    output = tmp_path / "pts" / "epic"

    first = pts_index.build_or_validate_cache(
        video,
        video_root=root,
        output_dir=output,
        dataset_name="epic",
        timestamp_reader=_reader,
    )
    second = pts_index.build_or_validate_cache(
        video,
        video_root=root,
        output_dir=output,
        dataset_name="epic",
        timestamp_reader=_reader,
    )
    assert first["status"] == "rebuilt"
    assert second["status"] == "validated_skip"

    _write_video_placeholder(video, b"changed-and-longer")
    third = pts_index.build_or_validate_cache(
        video,
        video_root=root,
        output_dir=output,
        dataset_name="epic",
        timestamp_reader=_reader,
    )
    assert third["status"] == "rebuilt"

    np.save(output / "sample.npy", np.asarray([0.0, 0.0]), allow_pickle=False)
    fourth = pts_index.build_or_validate_cache(
        video,
        video_root=root,
        output_dir=output,
        dataset_name="epic",
        timestamp_reader=_reader,
    )
    assert fourth["status"] == "rebuilt"
    assert not list(output.glob(".*.tmp"))


def test_run_index_writes_atomic_success_and_failure_manifests(tmp_path: Path) -> None:
    root = tmp_path / "videos"
    _write_video_placeholder(root / "good.mp4", b"good")
    _write_video_placeholder(root / "bad.webm", b"bad")
    output_root = tmp_path / "pts"

    def sometimes_fails(path: Path) -> tuple[np.ndarray, str]:
        if path.stem == "bad":
            raise RuntimeError("intentional failure")
        return _reader(path)

    failed = pts_index.run_index(
        video_root=root,
        output_root=output_root,
        dataset_name="ssv2",
        patterns=None,
        shard_id=0,
        num_shards=1,
        timestamp_reader=sometimes_fails,
    )
    manifest_path = pts_index.shard_manifest_path(output_root / "ssv2", 0, 1)
    assert manifest_path.is_file()
    assert failed["complete"] is False
    assert failed["failure_count"] == 1
    assert json.loads(manifest_path.read_text())["complete"] is False
    assert not list(manifest_path.parent.glob(".*.tmp"))

    succeeded = pts_index.run_index(
        video_root=root,
        output_root=output_root,
        dataset_name="ssv2",
        patterns=None,
        shard_id=0,
        num_shards=1,
        timestamp_reader=_reader,
    )
    assert succeeded["complete"] is True
    assert succeeded["successful_videos"] == 2
    assert json.loads(manifest_path.read_text())["complete"] is True
