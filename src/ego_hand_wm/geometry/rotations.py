"""Torch SO(3) utilities using the first-two-rows rot6D convention."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("matrix_to_rotation_6d expects [...,3,3]")
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(rotation_6d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if rotation_6d.shape[-1] != 6:
        raise ValueError("rotation_6d_to_matrix expects [...,6]")
    first = rotation_6d[..., :3]
    second = rotation_6d[..., 3:]
    basis_1 = functional.normalize(first, dim=-1, eps=eps)
    second_orthogonal = second - (basis_1 * second).sum(dim=-1, keepdim=True) * basis_1
    basis_2 = functional.normalize(second_orthogonal, dim=-1, eps=eps)
    basis_3 = torch.cross(basis_1, basis_2, dim=-1)
    return torch.stack((basis_1, basis_2, basis_3), dim=-2)


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        *vector.shape[:-1], 3, 3
    )


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    if axis_angle.shape[-1] != 3:
        raise ValueError("axis_angle_to_matrix expects [...,3]")
    theta_sq = (axis_angle * axis_angle).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta_sq.clamp_min(1e-16))
    small = theta_sq < 1e-8
    coefficient_a = torch.where(
        small,
        1.0 - theta_sq / 6.0 + theta_sq.square() / 120.0,
        torch.sin(theta) / theta.clamp_min(1e-8),
    )
    coefficient_b = torch.where(
        small,
        0.5 - theta_sq / 24.0 + theta_sq.square() / 720.0,
        (1.0 - torch.cos(theta)) / theta_sq.clamp_min(1e-16),
    )
    skew = _skew(axis_angle)
    identity = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    identity = identity.expand(*axis_angle.shape[:-1], 3, 3)
    return identity + coefficient_a[..., None] * skew + coefficient_b[..., None] * (skew @ skew)


def so3_geodesic_angle(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if prediction.shape[-2:] != (3, 3) or target.shape[-2:] != (3, 3):
        raise ValueError("so3_geodesic_angle expects [...,3,3] matrices")
    relative = prediction @ target.transpose(-1, -2)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0).clamp(
        -1.0, 1.0
    )
    skew_vector = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(skew_vector, dim=-1)
    return torch.atan2(sine, cosine)
