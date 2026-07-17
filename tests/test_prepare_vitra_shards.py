from __future__ import annotations

import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path


SCRIPT = Path("/n/home08/sjmathy/EGO-HAND-WM/scripts/prepare_vitra_shards.py")
SPEC = importlib.util.spec_from_file_location("prepare_vitra_shards", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sharder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sharder)


def _make_archive(root: Path, source: str, episodes: int) -> Path:
    archive = root / f"{source}.tar.gz"
    payload_dir = root / f"{source}-payloads"
    payload_dir.mkdir()
    with tarfile.open(archive, "w:gz") as target:
        for index in range(episodes):
            payload = payload_dir / f"{source}-{index}.npy"
            payload.write_bytes(f"payload-{source}-{index}".encode())
            target.add(payload, arcname=f"nested/{payload.name}")
        ignored = payload_dir / "README.txt"
        ignored.write_text("not an episode")
        target.add(ignored, arcname=ignored.name)
    return archive


def _fixture(tmp_path: Path) -> tuple[list[Path], dict[str, int]]:
    expected = {"source_a": 3, "source_b": 2}
    archives = [_make_archive(tmp_path, source, count) for source, count in expected.items()]
    return archives, expected


def test_complete_sharding_is_hashed_atomic_and_idempotent(tmp_path: Path) -> None:
    archives, expected = _fixture(tmp_path)
    output = tmp_path / "output"
    manifest = sharder.prepare_vitra_shards(
        archives,
        output_dir=output,
        episodes_per_shard=2,
        expected_counts=expected,
    )

    assert manifest["complete"] is True
    assert manifest["observed_source_counts"] == expected
    assert [record["episodes"] for record in manifest["shards"]] == [2, 2, 1]
    assert (output / "_SUCCESS").is_file()
    assert len((output / "shards.jsonl").read_text().splitlines()) == 3
    for record in manifest["shards"]:
        shard = output / record["shard"]
        assert shard.stat().st_size == record["bytes"]
        assert hashlib.sha256(shard.read_bytes()).hexdigest() == record["sha256"]
    assert not list(output.glob("*.partial"))
    assert not list(output.glob(".*.partial"))

    mtimes = {
        record["shard"]: (output / record["shard"]).stat().st_mtime_ns
        for record in manifest["shards"]
    }
    rerun = sharder.prepare_vitra_shards(
        list(reversed(archives)),
        output_dir=output,
        episodes_per_shard=2,
        expected_counts=expected,
    )
    assert rerun["summary"] == manifest["summary"]
    assert mtimes == {
        record["shard"]: (output / record["shard"]).stat().st_mtime_ns
        for record in manifest["shards"]
    }


def test_resume_verifies_prefix_and_rebuilds_remaining_shards(tmp_path: Path) -> None:
    archives, expected = _fixture(tmp_path)
    output = tmp_path / "output"
    manifest = sharder.prepare_vitra_shards(
        archives,
        output_dir=output,
        episodes_per_shard=2,
        expected_counts=expected,
    )
    first = manifest["shards"][0]
    state_path = output / sharder.STATE_FILENAME
    state = json.loads(state_path.read_text())
    state["shards"] = [first]
    sharder.atomic_write_json(state_path, state)
    (output / sharder.SUCCESS_FILENAME).unlink()
    (output / sharder.MANIFEST_FILENAME).unlink()
    (output / sharder.JSONL_FILENAME).unlink()
    for record in manifest["shards"][1:]:
        (output / record["shard"]).unlink()

    resumed = sharder.prepare_vitra_shards(
        archives,
        output_dir=output,
        episodes_per_shard=2,
        expected_counts=expected,
    )
    assert resumed["summary"]["total_episodes"] == 5
    assert resumed["shards"][0] == first
    assert (output / sharder.SUCCESS_FILENAME).is_file()


def test_count_mismatch_never_publishes_success(tmp_path: Path) -> None:
    archives, _ = _fixture(tmp_path)
    output = tmp_path / "output"
    try:
        sharder.prepare_vitra_shards(
            archives,
            output_dir=output,
            episodes_per_shard=2,
            expected_counts={"source_a": 4, "source_b": 2},
        )
    except RuntimeError as error:
        assert "source_a has 3 episodes" in str(error)
    else:
        raise AssertionError("Expected the source-count gate to fail")
    assert not (output / sharder.SUCCESS_FILENAME).exists()
    assert not (output / sharder.MANIFEST_FILENAME).exists()
