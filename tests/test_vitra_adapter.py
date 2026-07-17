from pathlib import Path

import numpy as np
import torch

from ego_hand_wm.data.adapters.vitra import canonicalize_vitra_episode, load_vitra_episode
from ego_hand_wm.geometry.se3 import pose9_to_matrix


EXAMPLE = Path(
    "/n/home08/sjmathy/EGO-HAND-WM/VITRA/data/examples/annotations/"
    "Ego4D_03cc49c3-a7d1-445b-9a2a-545c4fae6843_ep_example.npy"
)


def test_vitra_anchor_and_intrinsics() -> None:
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
    assert not sample["future_stream_mask"][:, 3].any()
    assert not sample["future_stream_mask"][:, 1].any()
    assert not sample["future_query_stream_mask"][:, 3].any()
    assert not sample["future_query_stream_mask"][:, 1].any()
    assert sample["future_query_stream_mask"][:, [0, 2, 4]].all()
    assert sample["metadata"]["source_dataset"] == "ego4d_cooking_and_cleaning"
    assert "Right hand:" in sample["text"]
