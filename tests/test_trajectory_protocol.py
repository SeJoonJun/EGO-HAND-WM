import json
import pickle

import numpy as np
import pytest
import torch

from ego_hand_wm.benchmarks import (
    CanonicalTrajectoryDataset,
    FixedTrajectoryProtocol,
    TrajectoryWindowDataset,
)
from ego_hand_wm.contracts import canonical_collate
from ego_hand_wm.contracts.schema import SCHEMA
from ego_hand_wm.geometry.se3 import pose9_to_matrix
from ego_hand_wm.losses import masked_stream_flow_loss


def test_h6_k16_temporal_contract():
    protocol = FixedTrajectoryProtocol()

    assert protocol.total_steps == 22
    assert protocol.observation_ratio == pytest.approx(6 / 22)
    assert protocol.history_span_seconds == pytest.approx(5 / 30)
    assert protocol.final_horizon_seconds == pytest.approx(16 / 30)
    assert protocol.history_relative_times[-1] == 0
    assert protocol.future_relative_times[0] == pytest.approx(1 / 30)
    assert protocol.future_relative_times[-1] == pytest.approx(16 / 30)


def test_exact_windows_drop_short_clips_and_include_tail():
    protocol = FixedTrajectoryProtocol()

    assert protocol.window_starts(21) == ()
    assert protocol.window_starts(22) == (0,)
    assert protocol.window_starts(23) == (0, 1)
    assert protocol.window_starts(54) == (0, 16, 32)


def test_indices_have_six_history_and_sixteen_strict_future_frames():
    protocol = FixedTrajectoryProtocol()
    history, future = protocol.frame_indices(10)

    assert history == tuple(range(10, 16))
    assert future == tuple(range(16, 32))
    assert set(history).isdisjoint(future)


def _record(protocol, **updates):
    record = {
        "schema_version": 1,
        "protocol": "h6_k16_30hz",
        "split": "train",
        "sample_id": "sample",
        "source_group": "group",
        "video_path": "/unused.mp4",
        "trajectory_window_start": 0,
        **protocol.manifest_fields(0),
    }
    record.update(updates)
    return record


def test_h2o_and_egopat_are_both_anchored_to_last_observed_camera(tmp_path):
    protocol = FixedTrajectoryProtocol()
    points = np.tile(np.array([[1.0, 0.0, 1.0]]), (22, 1))
    cam2world = np.repeat(np.eye(4)[None], 22, axis=0)
    cam2world[:, 0, 3] = np.arange(22) * 0.1
    h2o_path = tmp_path / "h2o.pkl"
    with h2o_path.open("wb") as handle:
        pickle.dump(
            {
                "left_hand": [
                    {"start": 0, "end": 21, "traj3d": points, "cam2world": cam2world}
                ],
                "right_hand": [],
            },
            handle,
        )
    h2o_record = _record(
        protocol,
        dataset="h2o",
        trajectory_path=str(h2o_path),
        hand="left",
        trajectory_segment_index=0,
    )

    egopat_path = tmp_path / "egopat.pkl"
    with egopat_path.open("wb") as handle:
        pickle.dump({"traj3d": points, "num_preserve": 22}, handle)
    odometry = np.repeat(np.eye(4)[None], 22, axis=0)
    odometry[1:, 0, 3] = 0.1
    odometry_path = tmp_path / "odometry.npy"
    np.save(odometry_path, odometry)
    egopat_record = _record(
        protocol,
        dataset="egopat3d",
        trajectory_path=str(egopat_path),
        odometry_path=str(odometry_path),
    )

    manifest = tmp_path / "windows.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(record) for record in (h2o_record, egopat_record)) + "\n"
    )
    dataset = TrajectoryWindowDataset(manifest)

    assert len(dataset) == 2
    assert dataset[0]["history_xyz_anchor"].shape == (6, 3)
    assert dataset[0]["future_xyz_anchor"].shape == (16, 3)
    assert dataset[0]["history_xyz_anchor"][-1, 0] == pytest.approx(0.5)
    assert dataset[1]["history_xyz_anchor"][-1, 0] == pytest.approx(1.0)
    assert dataset[1]["future_xyz_anchor"][-1, 0] == pytest.approx(2.6)


def test_h2o_canonical_adapter_matches_vitra_anchor_and_masks_sparse_wrist(tmp_path):
    protocol = FixedTrajectoryProtocol()
    points = np.tile(np.array([[1.0, 0.0, 1.0]]), (22, 1))
    cam2world = np.repeat(np.eye(4)[None], 22, axis=0)
    cam2world[:, 0, 3] = np.arange(22) * 0.1
    trajectory_path = tmp_path / "h2o.pkl"
    with trajectory_path.open("wb") as handle:
        pickle.dump(
            {
                "left_hand": [
                    {"start": 0, "end": 21, "traj3d": points, "cam2world": cam2world}
                ],
                "right_hand": [],
                "intrinsics": {
                    "fx": 640.0,
                    "fy": 360.0,
                    "cx": 640.0,
                    "cy": 360.0,
                    "width": 1280.0,
                    "height": 720.0,
                },
            },
            handle,
        )
    record = _record(
        protocol,
        dataset="h2o",
        trajectory_path=str(trajectory_path),
        hand="left",
        trajectory_segment_index=0,
    )
    manifest = tmp_path / "h2o_train.jsonl"
    manifest.write_text(json.dumps(record) + "\n")
    dataset = CanonicalTrajectoryDataset(
        {
            "split": "train",
            "manifest": str(manifest),
            "missing_text_feature_dim": 12,
        }
    )
    sample = dataset[0]

    assert sample["history_state"].shape == (6, 207)
    assert sample["future_state"].shape == (16, 207)
    assert sample["history_stream_mask"][-1].tolist() == [True, True, False, False, False]
    assert sample["future_state_mask"][:, SCHEMA.left_wrist].sum().item() == 16 * 3
    assert sample["future_state_mask"][:, SCHEMA.camera].all()
    assert not sample["future_state_mask"][:, SCHEMA.left_wrist.start + 3 : SCHEMA.left_wrist.stop].any()
    camera_anchor = pose9_to_matrix(sample["history_state"][-1, SCHEMA.camera])
    torch.testing.assert_close(camera_anchor, torch.eye(4))
    assert sample["history_state"][-1, SCHEMA.left_wrist.start].item() == pytest.approx(0.5)
    assert sample["context_text_mask"].item() is False
    batch = canonical_collate([sample])
    batch.validate()


def test_component_mask_does_not_supervise_unobserved_wrist_rotation():
    prediction = torch.zeros(1, 2, 207)
    target = torch.zeros_like(prediction)
    prediction[..., SCHEMA.left_wrist.start + 3 : SCHEMA.left_wrist.stop] = 10.0
    stream_mask = torch.zeros(1, 2, 5, dtype=torch.bool)
    stream_mask[..., 1] = True
    state_mask = torch.zeros_like(prediction, dtype=torch.bool)
    state_mask[..., SCHEMA.left_wrist.start : SCHEMA.left_wrist.start + 3] = True

    loss, components = masked_stream_flow_loss(
        prediction, target, stream_mask, state_mask
    )

    assert loss.item() == pytest.approx(0.0)
    assert components["flow/left_wrist"].item() == pytest.approx(0.0)
