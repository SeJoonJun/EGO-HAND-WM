import os
from pathlib import Path

import numpy as np
import pytest
import torch

from ego_hand_wm.data.adapters.vitra import canonicalize_vitra_episode, load_vitra_episode
from ego_hand_wm.geometry.se3 import pose9_to_matrix


EXAMPLE = Path(
    os.environ.get(
        "VITRA_EXAMPLE_ANNOTATION",
        "/scratch/jun.se/VITRA-1M/ego4d_cooking_and_cleaning/episodic_annotations/"
        "Ego4D_03cc49c3-a7d1-445b-9a2a-545c4fae6843_ep_example.npy",
    )
)


def _minimal_episode(length: int = 5) -> dict:
    rotations = np.broadcast_to(np.eye(3), (length, 3, 3)).copy()
    hand_pose = np.broadcast_to(np.eye(3), (length, 15, 3, 3)).copy()
    side = {
        "global_orient_worldspace": rotations,
        "transl_worldspace": np.zeros((length, 3)),
        "hand_pose": hand_pose,
        "kept_frames": np.ones(length, dtype=bool),
        "joints_worldspace": np.zeros((length, 21, 3)),
        "beta": np.zeros(10),
    }
    return {
        "extrinsics": np.broadcast_to(np.eye(4), (length, 4, 4)).copy(),
        "intrinsics": np.asarray([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]),
        "video_decode_frame": np.arange(length),
        "video_name": "synthetic",
        "anno_type": "right",
        "text": {
            "left": [("hold", [0, length])],
            "right": [("move", [0, length])],
        },
        "left": {key: np.array(value, copy=True) for key, value in side.items()},
        "right": {key: np.array(value, copy=True) for key, value in side.items()},
    }


def test_as_stored_policy_keeps_bilateral_vitra_supervision() -> None:
    episode = _minimal_episode()
    sample = canonicalize_vitra_episode(
        episode,
        [0, 1],
        [2, 3, 4],
        np.arange(5, dtype=np.float64) / 30.0,
        left_mano_policy="as_stored",
    )
    assert sample["history_stream_mask"][:, [1, 3]].all()
    assert sample["future_stream_mask"][:, [1, 3]].all()
    assert sample["future_query_stream_mask"].all()
    assert sample["metadata"]["left_mano_policy"] == "as_stored"


def test_wrist_local_joints_remove_global_root_translation() -> None:
    episode = _minimal_episode()
    offset = np.linspace(0.0, 0.2, 21, dtype=np.float64)[:, None] * np.asarray(
        [[1.0, -0.5, 0.25]]
    )
    for side in ("left", "right"):
        root = np.stack(
            [np.asarray([float(index), 2.0, -1.0]) for index in range(5)]
        )
        episode[side]["transl_worldspace"] = root
        episode[side]["joints_worldspace"] = root[:, None, :] + offset[None, :, :]
    sample = canonicalize_vitra_episode(
        episode,
        [0, 1],
        [2, 3, 4],
        np.arange(5, dtype=np.float64) / 30.0,
    )
    expected = torch.as_tensor(offset, dtype=torch.float32)
    torch.testing.assert_close(sample["history_hand_joints_local"][0, 0], expected)
    torch.testing.assert_close(sample["future_hand_joints_local"][-1, 1], expected)


def test_vitra_anchor_and_intrinsics() -> None:
    if not EXAMPLE.is_file():
        pytest.skip(f"Set VITRA_EXAMPLE_ANNOTATION to an extracted VITRA episode: {EXAMPLE}")
    episode = load_vitra_episode(EXAMPLE)
    times = np.asarray(episode["video_decode_frame"], dtype=np.float64) / 30.0
    sample = canonicalize_vitra_episode(
        episode,
        [0, 1, 2],
        [3, 4, 5, 6],
        times,
        calibration_size=(1440, 1920),
        left_mano_policy="mask",
        source_dataset="ego4d_cooking_and_cleaning",
        episode_id=EXAMPLE.stem,
    )
    anchor_camera = pose9_to_matrix(sample["history_state"][-1, :9])
    torch.testing.assert_close(anchor_camera, torch.eye(4), atol=1e-5, rtol=1e-5)
    expected = torch.tensor(
        [
            episode["intrinsics"][0, 0] / 1920,
            episode["intrinsics"][1, 1] / 1440,
            episode["intrinsics"][0, 2] / 1920,
            episode["intrinsics"][1, 2] / 1440,
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(sample["intrinsics"], expected)
    assert sample["history_fingertips"].shape == (3, 2, 5, 3)
    assert sample["future_fingertips"].shape == (4, 2, 5, 3)
    assert sample["history_hand_joints_local"].shape == (3, 2, 21, 3)
    assert sample["future_hand_joints_local"].shape == (4, 2, 21, 3)
    assert not sample["future_stream_mask"][:, 3].any()
    assert not sample["future_stream_mask"][:, 1].any()
    assert not sample["future_query_stream_mask"][:, 3].any()
    assert not sample["future_query_stream_mask"][:, 1].any()
    assert sample["future_query_stream_mask"][:, [0, 2, 4]].all()
    assert sample["metadata"]["source_dataset"] == "ego4d_cooking_and_cleaning"
    assert "Right hand:" in sample["text"]
