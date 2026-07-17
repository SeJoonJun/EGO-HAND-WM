"""Homogeneous transforms. `T_X_from_Y` maps coordinates in Y into X."""

from __future__ import annotations

import torch

from ego_hand_wm.geometry.rotations import matrix_to_rotation_6d, rotation_6d_to_matrix


def invert(transform: torch.Tensor) -> torch.Tensor:
    if transform.shape[-2:] != (4, 4):
        raise ValueError("invert expects [...,4,4]")
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    rotation_inverse = rotation.transpose(-1, -2)
    result = torch.zeros_like(transform)
    result[..., :3, :3] = rotation_inverse
    result[..., :3, 3] = -(rotation_inverse @ translation[..., None]).squeeze(-1)
    result[..., 3, 3] = 1.0
    return result


def compose(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.shape[-2:] != (4, 4) or second.shape[-2:] != (4, 4):
        raise ValueError("compose expects [...,4,4]")
    return first @ second


def transform_points(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    if transform.shape[-2:] != (4, 4) or points.shape[-1] != 3:
        raise ValueError("transform_points expects transform [...,4,4] and points [...,N,3]")
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    return points @ rotation.transpose(-1, -2) + translation.unsqueeze(-2)


def encode_pose9(transform: torch.Tensor) -> torch.Tensor:
    if transform.shape[-2:] != (4, 4):
        raise ValueError("encode_pose9 expects [...,4,4]")
    return torch.cat(
        (transform[..., :3, 3], matrix_to_rotation_6d(transform[..., :3, :3])), dim=-1
    )


def pose9_to_matrix(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape[-1] != 9:
        raise ValueError("pose9_to_matrix expects [...,9]")
    transform = torch.zeros(*pose.shape[:-1], 4, 4, dtype=pose.dtype, device=pose.device)
    transform[..., :3, :3] = rotation_6d_to_matrix(pose[..., 3:])
    transform[..., :3, 3] = pose[..., :3]
    transform[..., 3, 3] = 1.0
    return transform

