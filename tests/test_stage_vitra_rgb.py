from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from scripts.stage_vitra_rgb import assigned_videos, stage_video


def _request_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE physical_frames(
                physical_dataset TEXT, video TEXT, frame_id INTEGER,
                PRIMARY KEY(physical_dataset,video,frame_id)
            );
            CREATE TABLE videos(
                physical_dataset TEXT, video TEXT, requested_frames INTEGER,
                minimum_frame INTEGER, maximum_frame INTEGER, pts_frames INTEGER,
                PRIMARY KEY(physical_dataset,video)
            );
            """
        )
        connection.executemany(
            "INSERT INTO physical_frames VALUES ('epic','sample',?)", [(2,), (5,), (8,)]
        )
        connection.execute("INSERT INTO videos VALUES ('epic','sample',3,2,8,10)")


def test_stage_video_is_atomic_resumable_and_exact(tmp_path: Path) -> None:
    database = tmp_path / "requests.sqlite"
    _request_database(database)
    video_root = tmp_path / "videos"
    video_root.mkdir()
    video = video_root / "sample.mp4"
    video.write_bytes(b"synthetic-video")
    pts_root = tmp_path / "pts/epic"
    pts_root.mkdir(parents=True)
    np.save(pts_root / "sample.npy", np.arange(10, dtype=np.float64) / 30.0)
    stat = video.stat()
    (pts_root / "sample.meta.json").write_text(
        json.dumps(
            {
                "source_fingerprint": {
                    "relative_path": "sample.mp4",
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sampled_sha256": "unused-in-test",
                }
            }
        )
    )

    def reader(path, frame_ids, frame_times, *, output_size):
        assert path == video
        assert frame_ids == [2, 5, 8]
        assert output_size == (256, 256)
        for frame_id in frame_ids:
            yield frame_id, np.full((256, 256, 3), frame_id, dtype=np.uint8)

    output = tmp_path / "rgb"
    first = stage_video(
        request_database=database,
        output_root=output,
        pts_root=tmp_path / "pts",
        video_roots={"epic": video_root},
        dataset="epic",
        video="sample",
        frame_reader=reader,
    )
    second = stage_video(
        request_database=database,
        output_root=output,
        pts_root=tmp_path / "pts",
        video_roots={"epic": video_root},
        dataset="epic",
        video="sample",
        frame_reader=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()),
    )

    frames = np.load(output / "epic/sample.frames.npy", allow_pickle=False)
    rgb = np.load(output / "epic/sample.rgb.npy", allow_pickle=False, mmap_mode="r")
    assert first["status"] == "rebuilt"
    assert second["status"] == "validated_skip"
    assert frames.tolist() == [2, 5, 8]
    assert rgb.shape == (3, 256, 256, 3)
    assert int(rgb[1, 0, 0, 0]) == 5
    assert not list((output / "epic").glob(".*.tmp"))


def test_video_assignment_is_deterministic_and_balanced(tmp_path: Path) -> None:
    database = tmp_path / "requests.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE videos(
                physical_dataset TEXT, video TEXT, requested_frames INTEGER,
                minimum_frame INTEGER, maximum_frame INTEGER, pts_frames INTEGER
            )
            """
        )
        connection.executemany(
            "INSERT INTO videos VALUES ('ssv2',?, ?,0,0,1)",
            [("large", 10), ("medium", 6), ("small_a", 2), ("small_b", 2)],
        )
    first = assigned_videos(database, worker_id=0, num_workers=2)
    second = assigned_videos(database, worker_id=1, num_workers=2)
    assert {row[1] for row in first + second} == {"large", "medium", "small_a", "small_b"}
    assert abs(sum(row[2] for row in first) - sum(row[2] for row in second)) <= 2
