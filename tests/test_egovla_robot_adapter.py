from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ego_hand_wm.data.adapters.egovla_robot import (
    CALIBRATED_CAMERA_QUATERNION_WXYZ,
    MAIN_CAMERA_INTRINSICS,
    MissingActualEndEffectorPoseError,
    MissingManoAssetsError,
    camera_points_to_environment,
    camera_pose_matrices_to_environment,
    environment_points_to_camera,
    environment_poses_to_camera,
    fallback_timestamps,
    inspect_egovla_robot_handle,
    normalized_main_camera_intrinsics,
    pose_wxyz_to_matrix,
    quaternion_wxyz_to_matrix,
    require_mano_assets,
    static_camera_pose9,
)


class ShapeOnlyDataset:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class FakeHDF5(dict[str, Any]):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(payload)
        self.attrs = {"sim": True}


def fake_episode(*, actual_ee: bool = True, num_frames: int = 3) -> FakeHDF5:
    observations: dict[str, Any] = {
        "images": {"main": ShapeOnlyDataset((num_frames, 720, 1280, 3))},
        "qpos": ShapeOnlyDataset((num_frames, 50)),
        "qvel": ShapeOnlyDataset((num_frames, 50)),
        "left_finger_tip_pos": ShapeOnlyDataset((num_frames, 5, 3)),
        "right_finger_tip_pos": ShapeOnlyDataset((num_frames, 5, 3)),
    }
    if actual_ee:
        observations["left_ee_pose"] = ShapeOnlyDataset((num_frames, 7))
        observations["right_ee_pose"] = ShapeOnlyDataset((num_frames, 7))
    else:
        observations["left_target_ee_pose"] = ShapeOnlyDataset((num_frames, 7))
        observations["right_target_ee_pose"] = ShapeOnlyDataset((num_frames, 7))
    return FakeHDF5(
        {
            "action": ShapeOnlyDataset((num_frames, 50)),
            "observations": observations,
        }
    )


def test_wxyz_pose_and_frame_round_trip() -> None:
    root_half = np.sqrt(0.5)
    rotation = quaternion_wxyz_to_matrix([root_half, 0.0, 0.0, root_half])
    np.testing.assert_allclose(
        rotation,
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        atol=1e-7,
    )

    poses = np.array(
        [
            [0.2, -0.1, 1.0, 1.0, 0.0, 0.0, 0.0],
            [-0.4, 0.3, 0.8, root_half, 0.0, 0.0, root_half],
        ]
    )
    recovered = camera_pose_matrices_to_environment(environment_poses_to_camera(poses))
    np.testing.assert_allclose(recovered, pose_wxyz_to_matrix(poses), atol=1e-9)

    points = np.array([[0.1, 0.2, 1.1], [-0.4, 0.5, 0.3]])
    np.testing.assert_allclose(
        camera_points_to_environment(environment_points_to_camera(points)), points, atol=1e-9
    )


def test_calibrated_camera_rotation_and_intrinsics() -> None:
    calibrated_rotation = quaternion_wxyz_to_matrix(CALIBRATED_CAMERA_QUATERNION_WXYZ)
    assert np.linalg.det(calibrated_rotation) == pytest.approx(1.0)
    normalized = normalized_main_camera_intrinsics()
    np.testing.assert_allclose(
        normalized,
        [
            MAIN_CAMERA_INTRINSICS[0, 0] / 1280.0,
            MAIN_CAMERA_INTRINSICS[1, 1] / 720.0,
            0.5,
            0.5,
        ],
    )
    np.testing.assert_allclose(
        normalized, [0.38177046875, 0.6787030555555556, 0.5, 0.5]
    )


def test_static_camera_pose9_and_30hz_fallback() -> None:
    camera = static_camera_pose9(3)
    np.testing.assert_array_equal(camera[:, :3], 0.0)
    np.testing.assert_array_equal(camera[:, 3:], [[1, 0, 0, 0, 1, 0]] * 3)
    np.testing.assert_allclose(fallback_timestamps(3), [0.0, 1.0 / 30.0, 2.0 / 30.0])


def test_strict_actual_ee_and_labeled_target_override() -> None:
    actual = inspect_egovla_robot_handle(fake_episode())
    assert actual.end_effector_source == "actual"
    assert actual.timestamp_source == "derived:30hz"
    np.testing.assert_allclose(actual.timestamps_s, [0.0, 1.0 / 30.0, 2.0 / 30.0])

    target_only = fake_episode(actual_ee=False)
    with pytest.raises(MissingActualEndEffectorPoseError, match="silently falls back"):
        inspect_egovla_robot_handle(target_only)
    permitted = inspect_egovla_robot_handle(target_only, allow_target_ee=True)
    assert permitted.end_effector_source == "commanded_target"
    assert permitted.end_effector_keys == (
        "left_target_ee_pose",
        "right_target_ee_pose",
    )


def test_explicit_timestamps_are_preferred_and_validated() -> None:
    episode = fake_episode()
    episode["timestamps_s"] = np.array([0.1, 0.14, 0.2])
    inspection = inspect_egovla_robot_handle(episode)
    assert inspection.timestamp_source == "hdf5:timestamps_s"
    np.testing.assert_array_equal(inspection.timestamps_s, episode["timestamps_s"])

    episode["timestamps_s"] = np.array([0.0, 0.1, 0.1])
    with pytest.raises(ValueError, match="strictly increasing"):
        inspect_egovla_robot_handle(episode)


def test_mano_asset_check_fails_closed(tmp_path: Path) -> None:
    model_dir = tmp_path / "mano_v1_2" / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "MANO_LEFT.pkl").write_bytes(b"left-test-placeholder")
    with pytest.raises(MissingManoAssetsError, match="Full MANO export is disabled"):
        require_mano_assets(tmp_path)

    (model_dir / "MANO_RIGHT.pkl").write_bytes(b"right-test-placeholder")
    assets = require_mano_assets(tmp_path)
    assert assets.model_directory == model_dir
    assert assets.left_model.name == "MANO_LEFT.pkl"
    assert assets.right_model.name == "MANO_RIGHT.pkl"
