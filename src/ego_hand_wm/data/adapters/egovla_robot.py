"""Read-only contract checks and geometry transforms for EgoVLA robot demonstrations.

This module deliberately stops before MANO fitting.  It validates the raw simulator episode,
reproduces EgoVLA's manually calibrated environment-to-CV-camera conversion, and proves that the
licensed MANO assets needed by a later exporter are present.  It never substitutes zeros for a
missing MANO fit and never imports Isaac Lab or the upstream EgoVLA package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_HZ = 30.0
ROBOT_STATE_DIM = 50
FINGERTIPS_PER_HAND = 5

MAIN_CAMERA_HEIGHT = 720
MAIN_CAMERA_WIDTH = 1280
MAIN_CAMERA_TRANSLATION = np.array([0.09, 0.0, 1.7], dtype=np.float64)
# Isaac Lab quaternion convention is WXYZ.
MAIN_CAMERA_QUATERNION_WXYZ = np.array(
    [0.66446, 0.24184, -0.24184, -0.664464], dtype=np.float64
)
# EgoVLA's manually calibrated target camera orientation, also WXYZ.
CALIBRATED_CAMERA_QUATERNION_WXYZ = np.array(
    [0.9063077870366499, 0.0, 0.42261826174069944, 0.0], dtype=np.float64
)
MAIN_CAMERA_INTRINSICS = np.array(
    [
        [488.6662, 0.0, 640.0],
        [0.0, 488.6662, 360.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
CAM_AXIS_TRANSFORM = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

ACTUAL_EE_KEYS = ("left_ee_pose", "right_ee_pose")
TARGET_EE_KEYS = ("left_target_ee_pose", "right_target_ee_pose")
TIMESTAMP_DATASET_PATHS = ("timestamps_s", "observations/timestamps_s")


class RobotHDF5ContractError(ValueError):
    """A raw robot episode cannot be interpreted without guessing."""


class MissingActualEndEffectorPoseError(RobotHDF5ContractError):
    """An episode contains commanded targets but not realized end-effector poses."""


class MissingManoAssetsError(FileNotFoundError):
    """Both licensed, side-specific MANO v1.2 model files are required."""


def quaternion_wxyz_to_matrix(quaternion: Any) -> np.ndarray:
    """Convert normalized or unnormalized WXYZ quaternions to rotation matrices."""
    value = np.asarray(quaternion, dtype=np.float64)
    if value.ndim == 0 or value.shape[-1] != 4:
        raise ValueError(f"Expected WXYZ quaternion shape [...,4], got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("Quaternion contains NaN or Inf")
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(norm <= np.finfo(np.float64).eps):
        raise ValueError("A zero-norm quaternion has no rotation")
    w, x, y, z = np.moveaxis(value / norm, -1, 0)
    result = np.empty(value.shape[:-1] + (3, 3), dtype=np.float64)
    result[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    result[..., 0, 1] = 2.0 * (x * y - z * w)
    result[..., 0, 2] = 2.0 * (x * z + y * w)
    result[..., 1, 0] = 2.0 * (x * y + z * w)
    result[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    result[..., 1, 2] = 2.0 * (y * z - x * w)
    result[..., 2, 0] = 2.0 * (x * z - y * w)
    result[..., 2, 1] = 2.0 * (y * z + x * w)
    result[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return result


def pose_wxyz_to_matrix(pose: Any) -> np.ndarray:
    """Convert ``[..., xyz, qw, qx, qy, qz]`` Isaac Lab poses to SE(3) matrices."""
    value = np.asarray(pose, dtype=np.float64)
    if value.ndim == 0 or value.shape[-1] != 7:
        raise ValueError(f"Expected pose shape [...,7], got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("Pose contains NaN or Inf")
    result = np.zeros(value.shape[:-1] + (4, 4), dtype=np.float64)
    result[..., :3, :3] = quaternion_wxyz_to_matrix(value[..., 3:])
    result[..., :3, 3] = value[..., :3]
    result[..., 3, 3] = 1.0
    return result


def _build_main_camera_transform() -> tuple[np.ndarray, np.ndarray]:
    raw_rotation = quaternion_wxyz_to_matrix(MAIN_CAMERA_QUATERNION_WXYZ)
    calibrated_rotation = quaternion_wxyz_to_matrix(CALIBRATED_CAMERA_QUATERNION_WXYZ)
    # This is the exact construction in EgoVLA's transformation.py and otv_isaaclab/utils.py.
    frame_change = calibrated_rotation @ np.linalg.inv(raw_rotation)
    camera_to_environment = np.eye(4, dtype=np.float64)
    camera_to_environment[:3, :3] = frame_change @ raw_rotation
    camera_to_environment[:3, 3] = MAIN_CAMERA_TRANSLATION
    return frame_change, camera_to_environment


ISAAC_LAB_CAMERA_FRAME_CHANGE, MAIN_CAMERA_TO_ENVIRONMENT = _build_main_camera_transform()
ENVIRONMENT_TO_CV_CAMERA = CAM_AXIS_TRANSFORM @ np.linalg.inv(MAIN_CAMERA_TO_ENVIRONMENT)
CV_CAMERA_TO_ENVIRONMENT = np.linalg.inv(ENVIRONMENT_TO_CV_CAMERA)


def environment_points_to_camera(points: Any) -> np.ndarray:
    """Map environment-local points to EgoVLA's OpenCV camera coordinates."""
    value = np.asarray(points, dtype=np.float64)
    if value.ndim == 0 or value.shape[-1] != 3:
        raise ValueError(f"Expected point shape [...,3], got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("Points contain NaN or Inf")
    return (
        np.einsum("ij,...j->...i", ENVIRONMENT_TO_CV_CAMERA[:3, :3], value)
        + ENVIRONMENT_TO_CV_CAMERA[:3, 3]
    )


def camera_points_to_environment(points: Any) -> np.ndarray:
    """Invert :func:`environment_points_to_camera`."""
    value = np.asarray(points, dtype=np.float64)
    if value.ndim == 0 or value.shape[-1] != 3:
        raise ValueError(f"Expected point shape [...,3], got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("Points contain NaN or Inf")
    return (
        np.einsum("ij,...j->...i", CV_CAMERA_TO_ENVIRONMENT[:3, :3], value)
        + CV_CAMERA_TO_ENVIRONMENT[:3, 3]
    )


def environment_poses_to_camera(poses_wxyz: Any) -> np.ndarray:
    """Map Isaac Lab WXYZ poses from the environment-local frame to the CV camera frame."""
    return np.matmul(ENVIRONMENT_TO_CV_CAMERA, pose_wxyz_to_matrix(poses_wxyz))


def camera_pose_matrices_to_environment(camera_poses: Any) -> np.ndarray:
    """Map camera-frame SE(3) matrices back to the environment-local frame."""
    value = np.asarray(camera_poses, dtype=np.float64)
    if value.ndim < 2 or value.shape[-2:] != (4, 4):
        raise ValueError(f"Expected pose matrices [...,4,4], got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("Pose matrices contain NaN or Inf")
    return np.matmul(CV_CAMERA_TO_ENVIRONMENT, value)


def normalized_main_camera_intrinsics() -> np.ndarray:
    """Return ``[fx/W, fy/H, cx/W, cy/H]`` for the fixed EgoVLA camera."""
    return np.array(
        [
            MAIN_CAMERA_INTRINSICS[0, 0] / MAIN_CAMERA_WIDTH,
            MAIN_CAMERA_INTRINSICS[1, 1] / MAIN_CAMERA_HEIGHT,
            MAIN_CAMERA_INTRINSICS[0, 2] / MAIN_CAMERA_WIDTH,
            MAIN_CAMERA_INTRINSICS[1, 2] / MAIN_CAMERA_HEIGHT,
        ],
        dtype=np.float64,
    )


def static_camera_pose9(num_frames: int, *, dtype: np.dtype[Any] = np.float32) -> np.ndarray:
    """Return identity camera motion in the canonical translation-plus-rot6D layout."""
    if num_frames < 0:
        raise ValueError("num_frames must be non-negative")
    pose = np.zeros((num_frames, 9), dtype=dtype)
    # The project uses the first-two-rows rot6D convention.
    pose[:, 3:] = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=dtype)
    return pose


def fallback_timestamps(num_frames: int, source_hz: float = SOURCE_HZ) -> np.ndarray:
    """Generate state timestamps only when the source has no explicit seconds array."""
    if num_frames < 0:
        raise ValueError("num_frames must be non-negative")
    if not np.isfinite(source_hz) or source_hz <= 0:
        raise ValueError("source_hz must be finite and positive")
    return np.arange(num_frames, dtype=np.float64) / float(source_hz)


@dataclass(frozen=True)
class EndEffectorPoseSelection:
    left_key: str
    right_key: str
    source: str


def _has_key(group: Any, key: str) -> bool:
    try:
        return key in group
    except TypeError as error:
        raise RobotHDF5ContractError(f"Expected an HDF5-like group, got {type(group)!r}") from error


def _get_key(group: Any, key: str, path: str) -> Any:
    if not _has_key(group, key):
        raise RobotHDF5ContractError(f"Missing required HDF5 key: {path}")
    return group[key]


def resolve_end_effector_pose_keys(
    observations: Any, *, allow_target_ee: bool = False
) -> EndEffectorPoseSelection:
    """Select paired actual EE arrays, refusing silent commanded-target substitution."""
    actual_present = tuple(_has_key(observations, key) for key in ACTUAL_EE_KEYS)
    if all(actual_present):
        return EndEffectorPoseSelection(*ACTUAL_EE_KEYS, source="actual")
    if any(actual_present):
        missing = ACTUAL_EE_KEYS[actual_present.index(False)]
        raise MissingActualEndEffectorPoseError(
            f"Actual EE arrays must be paired; observations/{missing} is missing"
        )
    if not allow_target_ee:
        raise MissingActualEndEffectorPoseError(
            "The episode has no actual left/right EE poses. EgoVLA's upstream loader silently "
            "falls back to commanded target EE poses; this adapter refuses that substitution. "
            "Pass allow_target_ee=True only for an explicitly labeled diagnostic export."
        )
    target_present = tuple(_has_key(observations, key) for key in TARGET_EE_KEYS)
    if not all(target_present):
        missing = TARGET_EE_KEYS[target_present.index(False)]
        raise MissingActualEndEffectorPoseError(
            f"Target-EE fallback was requested, but observations/{missing} is missing"
        )
    return EndEffectorPoseSelection(*TARGET_EE_KEYS, source="commanded_target")


def _shape(node: Any, path: str) -> tuple[int, ...]:
    if not hasattr(node, "shape"):
        raise RobotHDF5ContractError(f"HDF5 key is not an array dataset: {path}")
    return tuple(int(size) for size in node.shape)


def _validate_shape(
    node: Any,
    path: str,
    *,
    num_frames: int | None,
    trailing_shape: tuple[int, ...],
) -> tuple[int, ...]:
    shape = _shape(node, path)
    expected_rank = len(trailing_shape) + 1
    if len(shape) != expected_rank or shape[1:] != trailing_shape:
        expected = ("T", *trailing_shape)
        raise RobotHDF5ContractError(f"{path} must have shape {expected}, got {shape}")
    if num_frames is not None and shape[0] != num_frames:
        raise RobotHDF5ContractError(
            f"{path} has {shape[0]} frames but action establishes T={num_frames}"
        )
    return shape


def _lookup_path(root: Any, path: str) -> Any | None:
    node = root
    for component in path.split("/"):
        if not _has_key(node, component):
            return None
        node = node[component]
    return node


def _read_dataset(node: Any) -> np.ndarray:
    if isinstance(node, np.ndarray):
        return node
    try:
        return np.asarray(node[...])
    except (IndexError, TypeError, ValueError) as error:
        raise RobotHDF5ContractError("Could not read timestamp dataset") from error


def resolve_timestamps(
    handle: Any, num_frames: int, *, source_hz: float = SOURCE_HZ
) -> tuple[np.ndarray, str]:
    """Use an explicit ``*_s`` dataset or derive the documented 30 Hz control clock."""
    for path in TIMESTAMP_DATASET_PATHS:
        node = _lookup_path(handle, path)
        if node is None:
            continue
        values = np.asarray(_read_dataset(node), dtype=np.float64)
        if values.shape != (num_frames,):
            raise RobotHDF5ContractError(
                f"{path} must have shape ({num_frames},), got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise RobotHDF5ContractError(f"{path} contains NaN or Inf")
        if num_frames > 1 and np.any(np.diff(values) <= 0):
            raise RobotHDF5ContractError(f"{path} must be strictly increasing")
        return values, f"hdf5:{path}"
    return fallback_timestamps(num_frames, source_hz), f"derived:{source_hz:g}hz"


@dataclass(frozen=True)
class RobotHDF5Inspection:
    num_frames: int
    dataset_shapes: dict[str, tuple[int, ...]]
    end_effector_source: str
    end_effector_keys: tuple[str, str]
    timestamps_s: np.ndarray
    timestamp_source: str
    effective_hz: float

    def as_dict(self) -> dict[str, Any]:
        duration = (
            float(self.timestamps_s[-1] - self.timestamps_s[0])
            if self.num_frames > 1
            else 0.0
        )
        return {
            "valid": True,
            "num_frames": self.num_frames,
            "dataset_shapes": {
                key: list(shape) for key, shape in sorted(self.dataset_shapes.items())
            },
            "end_effector_source": self.end_effector_source,
            "end_effector_keys": list(self.end_effector_keys),
            "timestamp_source": self.timestamp_source,
            "effective_hz": self.effective_hz,
            "start_time_s": float(self.timestamps_s[0]) if self.num_frames else None,
            "end_time_s": float(self.timestamps_s[-1]) if self.num_frames else None,
            "duration_s": duration,
            "intrinsics_normalized": normalized_main_camera_intrinsics().tolist(),
            "camera_motion": "static_identity",
        }


def inspect_egovla_robot_handle(
    handle: Any, *, allow_target_ee: bool = False, source_hz: float = SOURCE_HZ
) -> RobotHDF5Inspection:
    """Validate an already-open, read-only EgoVLA simulation HDF5 handle."""
    attributes = getattr(handle, "attrs", None)
    if attributes is None or "sim" not in attributes:
        raise RobotHDF5ContractError("Missing required root HDF5 attribute: sim")

    shapes: dict[str, tuple[int, ...]] = {}
    action = _get_key(handle, "action", "action")
    action_shape = _validate_shape(
        action, "action", num_frames=None, trailing_shape=(ROBOT_STATE_DIM,)
    )
    num_frames = action_shape[0]
    if num_frames == 0:
        raise RobotHDF5ContractError("Robot episode contains zero frames")
    shapes["action"] = action_shape

    observations = _get_key(handle, "observations", "observations")
    images = _get_key(observations, "images", "observations/images")
    checks = {
        "observations/images/main": (
            _get_key(images, "main", "observations/images/main"),
            (MAIN_CAMERA_HEIGHT, MAIN_CAMERA_WIDTH, 3),
        ),
        "observations/qpos": (
            _get_key(observations, "qpos", "observations/qpos"),
            (ROBOT_STATE_DIM,),
        ),
        "observations/qvel": (
            _get_key(observations, "qvel", "observations/qvel"),
            (ROBOT_STATE_DIM,),
        ),
        "observations/left_finger_tip_pos": (
            _get_key(observations, "left_finger_tip_pos", "observations/left_finger_tip_pos"),
            (FINGERTIPS_PER_HAND, 3),
        ),
        "observations/right_finger_tip_pos": (
            _get_key(
                observations, "right_finger_tip_pos", "observations/right_finger_tip_pos"
            ),
            (FINGERTIPS_PER_HAND, 3),
        ),
    }
    selection = resolve_end_effector_pose_keys(observations, allow_target_ee=allow_target_ee)
    checks[f"observations/{selection.left_key}"] = (
        observations[selection.left_key],
        (7,),
    )
    checks[f"observations/{selection.right_key}"] = (
        observations[selection.right_key],
        (7,),
    )
    for path, (node, trailing_shape) in checks.items():
        shapes[path] = _validate_shape(
            node, path, num_frames=num_frames, trailing_shape=trailing_shape
        )

    timestamps, timestamp_source = resolve_timestamps(handle, num_frames, source_hz=source_hz)
    effective_hz = (
        float(1.0 / np.median(np.diff(timestamps))) if num_frames > 1 else float(source_hz)
    )
    return RobotHDF5Inspection(
        num_frames=num_frames,
        dataset_shapes=shapes,
        end_effector_source=selection.source,
        end_effector_keys=(selection.left_key, selection.right_key),
        timestamps_s=timestamps,
        timestamp_source=timestamp_source,
        effective_hz=effective_hz,
    )


def inspect_egovla_robot_hdf5(
    path: str | Path, *, allow_target_ee: bool = False, source_hz: float = SOURCE_HZ
) -> RobotHDF5Inspection:
    """Open one HDF5 file read-only and validate it; importing h5py is intentionally lazy."""
    try:
        import h5py
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "HDF5 inspection requires h5py; install the `robot-export` project extra"
        ) from error
    episode_path = Path(path)
    if not episode_path.is_file():
        raise FileNotFoundError(episode_path)
    with h5py.File(episode_path, "r") as handle:
        return inspect_egovla_robot_handle(
            handle, allow_target_ee=allow_target_ee, source_hz=source_hz
        )


@dataclass(frozen=True)
class ManoModelAssets:
    model_directory: Path
    left_model: Path
    right_model: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "ready": True,
            "model_directory": str(self.model_directory),
            "left_model": str(self.left_model),
            "right_model": str(self.right_model),
        }


def require_mano_assets(mano_root: str | Path) -> ManoModelAssets:
    """Fail unless non-empty side-specific MANO v1.2 pickle files are available.

    Files are only checked for presence and size; this function never unpickles untrusted data.
    """
    root = Path(mano_root).expanduser()
    candidates = (root, root / "models", root / "mano_v1_2" / "models")
    checked: list[str] = []
    for directory in dict.fromkeys(candidates):
        left = directory / "MANO_LEFT.pkl"
        right = directory / "MANO_RIGHT.pkl"
        checked.extend((str(left), str(right)))
        if (
            left.is_file()
            and right.is_file()
            and left.stat().st_size > 0
            and right.stat().st_size > 0
        ):
            return ManoModelAssets(directory, left, right)
    raise MissingManoAssetsError(
        "Full MANO export is disabled until both licensed, non-empty MANO v1.2 files exist. "
        "Checked: " + ", ".join(checked)
    )
