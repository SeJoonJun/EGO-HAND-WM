import json
import sqlite3
from pathlib import Path

import numpy as np

from scripts.extract_staged_vitra_dinov3 import (
    _sha256_array,
    assigned_tail_assist_videos,
    encode_staged_video,
)


class FakeSpatialEncoder:
    spatial_grid_size = 2

    def encode(self, rgb_frames: np.ndarray) -> np.ndarray:
        values = rgb_frames[:, 0, 0, 0].astype(np.float32)
        return np.broadcast_to(values[:, None, None], (len(values), 4, 3)).copy()


def test_encode_staged_video_is_atomic_and_resumable(tmp_path: Path) -> None:
    rgb_root = tmp_path / "rgb"
    directory = rgb_root / "epic"
    directory.mkdir(parents=True)
    frame_ids = np.asarray([2, 5], dtype=np.int64)
    rgb = np.zeros((2, 256, 256, 3), dtype=np.uint8)
    rgb[1] = 7
    np.save(directory / "sample.frames.npy", frame_ids, allow_pickle=False)
    np.save(directory / "sample.rgb.npy", rgb, allow_pickle=False)
    (directory / "sample.json").write_text(
        json.dumps(
            {
                "complete": True,
                "frames": 2,
                "frame_ids_sha256": _sha256_array(frame_ids),
            }
        )
    )

    arguments = {
        "rgb_root": rgb_root,
        "output_root": tmp_path / "features",
        "dataset": "epic",
        "video": "sample",
        "encoder": FakeSpatialEncoder(),
        "extractor_metadata": {"backend": "fake", "feature_dim": 3},
        "batch_size": 1,
    }
    first = encode_staged_video(**arguments)
    second = encode_staged_video(**arguments)
    features = np.load(tmp_path / "features/epic/sample.features.npy", allow_pickle=False)

    assert first["status"] == "rebuilt"
    assert second["status"] == "validated_skip"
    assert features.shape == (2, 4, 3)
    assert features.dtype == np.float16
    assert np.all(features[0] == 0)
    assert np.all(features[1] == 7)
    assert not list((tmp_path / "features/epic").glob(".*.tmp"))


def test_tail_assist_assignment_is_disjoint_and_uses_bin_tails(tmp_path: Path) -> None:
    database = tmp_path / "requests.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE videos(physical_dataset TEXT, video TEXT, requested_frames INTEGER)"
        )
        connection.executemany(
            "INSERT INTO videos VALUES ('epic', ?, ?)",
            [(f"video-{index}", frames) for index, frames in enumerate(range(24, 0, -1))],
        )

    first = assigned_tail_assist_videos(
        database,
        worker_id=0,
        num_workers=2,
        base_num_workers=3,
        tail_fraction=0.2,
    )
    second = assigned_tail_assist_videos(
        database,
        worker_id=1,
        num_workers=2,
        base_num_workers=3,
        tail_fraction=0.2,
    )
    first_ids = {(dataset, video) for dataset, video, _ in first}
    second_ids = {(dataset, video) for dataset, video, _ in second}

    assert first_ids
    assert second_ids
    assert first_ids.isdisjoint(second_ids)
    for base_worker_id in range(3):
        from scripts.stage_vitra_rgb import assigned_videos

        base_bin = assigned_videos(database, worker_id=base_worker_id, num_workers=3)
        selected = [item for item in base_bin if (item[0], item[1]) in first_ids | second_ids]
        assert selected == base_bin[-len(selected) :]
