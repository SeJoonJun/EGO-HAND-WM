import numpy as np

from scripts.audit_vitra_horizons import episode_capacity


def test_episode_capacity_uses_primary_language_kept_frames_and_physical_time() -> None:
    episode = {
        "anno_type": "right",
        "extrinsics": np.repeat(np.eye(4)[None], 10, axis=0),
        "text": {"right": [("act", (0, 10))], "left": []},
        "right": {"kept_frames": np.ones(10, dtype=bool)},
        "left": {"kept_frames": np.ones(10, dtype=bool)},
    }
    times = np.arange(10, dtype=np.float64) * 0.1
    targets, horizon = episode_capacity(episode, times, history_steps=3)
    assert targets == 7
    assert np.isclose(horizon, 0.7)

    episode["right"]["kept_frames"][[4, 6, 8]] = False
    targets, horizon = episode_capacity(episode, times, history_steps=3)
    assert targets == 4
    assert np.isclose(horizon, 0.7)


def test_episode_capacity_does_not_cross_language_interval() -> None:
    episode = {
        "anno_type": "left",
        "extrinsics": np.repeat(np.eye(4)[None], 12, axis=0),
        "text": {"left": [("first", (0, 6)), ("second", (6, 12))], "right": []},
        "left": {"kept_frames": np.ones(12, dtype=bool)},
        "right": {"kept_frames": np.ones(12, dtype=bool)},
    }
    targets, horizon = episode_capacity(
        episode, np.arange(12, dtype=np.float64) / 10.0, history_steps=3
    )
    assert targets == 5
    assert np.isclose(horizon, 0.5)
