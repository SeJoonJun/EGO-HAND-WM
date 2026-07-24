from __future__ import annotations

import io
import sqlite3
import tarfile
from pathlib import Path

import numpy as np

from ego_hand_wm.data.build import _load_excluded_members

from scripts.build_vitra_frame_requests import (
    RGB_BYTES_PER_FRAME,
    TRUNCATED_EGO4D_VIDEO,
    build_partition,
    merge_partitions,
)


def _episode(video: str, frame_ids: list[int]) -> bytes:
    buffer = io.BytesIO()
    np.save(
        buffer,
        {"video_name": video, "video_decode_frame": np.asarray(frame_ids, dtype=np.int64)},
        allow_pickle=True,
    )
    return buffer.getvalue()


def _write_shard(path: Path, records: list[tuple[str, str, list[int]]]) -> None:
    with tarfile.open(path, "w") as archive:
        for member, video, frame_ids in records:
            payload = _episode(video, frame_ids)
            info = tarfile.TarInfo(member)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_global_index_deduplicates_aliases_and_applies_exclusions(tmp_path: Path) -> None:
    first = tmp_path / "vitra-000000.tar"
    second = tmp_path / "vitra-000001.tar"
    _write_shard(
        first,
        [
            ("ego4d_cooking_and_cleaning/episodes/a.npy", "shared", [1, 2, 3]),
            ("epic/episodes/missing.npy", "P01_19", [5, 6]),
        ],
    )
    _write_shard(
        second,
        [
            ("ego4d_other/episodes/b.npy", "shared", [3, 4]),
            (
                "ego4d_other/episodes/truncated.npy",
                TRUNCATED_EGO4D_VIDEO,
                [15_787, 15_788],
            ),
        ],
    )
    parts = [tmp_path / f"part-{index}.sqlite" for index in range(2)]
    for index, output in enumerate(parts):
        build_partition(
            [first, second], output, partition_id=index, num_partitions=2
        )

    pts = tmp_path / "pts/ego4d_cooking_and_cleaning"
    pts.mkdir(parents=True)
    np.save(pts / "shared.npy", np.arange(10, dtype=np.float64) / 30.0)
    output = tmp_path / "requests.sqlite"
    summary = merge_partitions(
        parts,
        output,
        pts_root=tmp_path / "pts",
        pts_aliases={"ego4d_other": "ego4d_cooking_and_cleaning"},
    )

    assert summary["stats"]["episodes_total"] == 4
    assert summary["stats"]["episodes_kept"] == 2
    assert summary["stats"]["episodes_excluded"] == 2
    assert summary["logical_requested_frames"] == 5
    assert summary["unique_requested_frames"] == 4
    assert summary["staged_rgb"]["total_bytes"] == 4 * RGB_BYTES_PER_FRAME
    assert summary["exclusion_reasons"] == {
        "missing_source_video:P01_19": 1,
        "truncated_source_frame>=15788": 1,
    }
    with sqlite3.connect(output) as connection:
        frames = connection.execute(
            "SELECT physical_dataset,video,frame_id FROM physical_frames ORDER BY frame_id"
        ).fetchall()
        excluded = connection.execute("SELECT COUNT(*) FROM excluded").fetchone()[0]
    assert frames == [
        ("ego4d_cooking_and_cleaning", "shared", 1),
        ("ego4d_cooking_and_cleaning", "shared", 2),
        ("ego4d_cooking_and_cleaning", "shared", 3),
        ("ego4d_cooking_and_cleaning", "shared", 4),
    ]
    assert excluded == 2
    assert output.with_suffix(".sqlite.SUCCESS.json").is_file()
    assert _load_excluded_members(output) == {
        "ego4d_other/episodes/truncated.npy",
        "epic/episodes/missing.npy",
    }
