import torch

from ego_hand_wm.geometry.rotations import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    so3_geodesic_angle,
)
from ego_hand_wm.geometry.se3 import invert, transform_points


def test_rot6d_round_trip() -> None:
    generator = torch.Generator().manual_seed(3)
    matrix = axis_angle_to_matrix(torch.randn(32, 3, generator=generator) * 0.4)
    recovered = rotation_6d_to_matrix(matrix_to_rotation_6d(matrix))
    torch.testing.assert_close(recovered, matrix, atol=1e-5, rtol=1e-5)


def test_se3_inverse_and_points() -> None:
    transform = torch.eye(4)
    transform[:3, :3] = axis_angle_to_matrix(torch.tensor([0.1, -0.2, 0.05]))
    transform[:3, 3] = torch.tensor([0.4, -0.1, 0.2])
    point = torch.tensor([[0.2, 0.3, 0.4]])
    recovered = transform_points(invert(transform), transform_points(transform, point))
    torch.testing.assert_close(recovered, point, atol=1e-6, rtol=1e-6)


def test_so3_geodesic_identity_near_zero_and_pi() -> None:
    identity = torch.eye(3)
    assert so3_geodesic_angle(identity, identity).item() == 0.0

    small = torch.tensor([1e-4, 0.0, 0.0])
    small_rotation = axis_angle_to_matrix(small)
    torch.testing.assert_close(
        so3_geodesic_angle(small_rotation, identity),
        torch.tensor(1e-4),
        atol=1e-7,
        rtol=1e-4,
    )

    pi_rotation = axis_angle_to_matrix(torch.tensor([torch.pi, 0.0, 0.0]))
    torch.testing.assert_close(
        so3_geodesic_angle(pi_rotation, identity), torch.tensor(torch.pi), atol=1e-6, rtol=0
    )
