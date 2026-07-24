import numpy as np
import torch

from ego_hand_wm.anticipation.geometry_encoder import (
    GEOMETRY_INPUT_DIMS,
    PastFutureGeometryEncoder,
    SemanticGeometryCrossAttention,
    assemble_geometry_sequence,
)
from ego_hand_wm.data.adapters.assembly101 import (
    canonicalize_assembly101_oracle_geometry,
)
from ego_hand_wm.geometry.se3 import pose9_to_matrix


def _transform(x_mm: float = 0.0, y_mm: float = 0.0) -> np.ndarray:
    value = np.eye(4, dtype=np.float32)
    value[:3, 3] = [x_mm, y_mm, 0.0]
    return value


def _geometry_batch(batch: int = 2, steps: int = 6) -> dict[str, torch.Tensor]:
    return {
        "camera_pose": torch.randn(batch, steps, 9),
        "wrist_pose": torch.randn(batch, steps, 2, 9),
        "hand_pose": torch.randn(batch, steps, 2, 21, 3),
        "wrist_valid": torch.ones(batch, steps, 2, dtype=torch.bool),
        "hand_pose_valid": torch.ones(batch, steps, 2, dtype=torch.bool),
        "future_mask": torch.arange(steps).view(1, -1).expand(batch, -1) >= 4,
    }


def test_geometry_modes_and_past_only_masks() -> None:
    inputs = _geometry_batch()
    for mode, width in GEOMETRY_INPUT_DIMS.items():
        full, full_mask = assemble_geometry_sequence(
            mode, **inputs, include_future=True
        )
        past, past_mask = assemble_geometry_sequence(
            mode, **inputs, include_future=False
        )
        assert full.shape == (2, 6, width)
        assert full_mask.all()
        assert past_mask[:, :4].all()
        assert not past_mask[:, 4:].any()
        assert not past[:, 4:].any()


def test_every_hand_condition_includes_camera_motion() -> None:
    inputs = _geometry_batch(batch=1, steps=3)
    inputs["wrist_valid"].zero_()
    inputs["hand_pose_valid"].zero_()
    for mode in ("camera_wrist", "camera_handpose", "camera_whole_hand"):
        values, mask = assemble_geometry_sequence(
            mode, **inputs, include_future=True
        )
        torch.testing.assert_close(values[..., :9], inputs["camera_pose"])
        # Camera remains a usable conditioning signal when neither hand is tracked.
        assert mask.all()


def test_temporal_geometry_encoder_and_zero_initialized_semantic_residual() -> None:
    inputs = _geometry_batch()
    values, mask = assemble_geometry_sequence(
        "camera_whole_hand", **inputs, include_future=True
    )
    encoder = PastFutureGeometryEncoder(
        GEOMETRY_INPUT_DIMS["camera_whole_hand"],
        32,
        hidden_dim=32,
        depth=2,
        num_heads=4,
        dropout=0.0,
        max_frames=6,
    ).eval()
    times = torch.linspace(-0.5, 0.25, 6).view(1, -1).expand(2, -1)
    tokens, returned_mask = encoder.forward_tokens(
        values, mask, times, inputs["future_mask"]
    )
    assert tokens.shape == (2, 6, 32)
    assert torch.equal(mask, returned_mask)
    assert torch.isfinite(tokens).all()

    fusion = SemanticGeometryCrossAttention(32, num_heads=4, dropout=0.0).eval()
    queries = torch.randn(2, 3, 32)
    # As in the supplied ROI-hand reference, the final residual projection starts at zero.
    torch.testing.assert_close(fusion(queries, tokens, mask), queries)

    empty_mask = torch.zeros_like(mask)
    torch.testing.assert_close(fusion(queries, tokens, empty_mask), queries)


def test_oracle_geometry_anchors_at_last_past_frame_and_makes_pose_wrist_local() -> None:
    steps = 4
    anchor_index = 1
    camera = np.stack([_transform(index * 100.0) for index in range(steps)])
    wrist = np.stack(
        [
            np.stack(
                [
                    _transform(1000.0 + index * 100.0),
                    _transform(2000.0 + index * 100.0, 300.0),
                ]
            )
            for index in range(steps)
        ]
    )
    local = np.zeros((2, 21, 3), dtype=np.float32)
    local[0, :, 0] = np.arange(21, dtype=np.float32)
    local[1, :, 1] = np.arange(21, dtype=np.float32) * 2.0
    landmarks = np.empty((steps, 2, 21, 3), dtype=np.float32)
    for time in range(steps):
        for hand in range(2):
            landmarks[time, hand] = local[hand] + wrist[time, hand, :3, 3]

    result = canonicalize_assembly101_oracle_geometry(
        camera,
        wrist,
        landmarks,
        np.ones((steps, 2), dtype=np.float32),
        anchor_index=anchor_index,
    )
    torch.testing.assert_close(
        result["camera_pose"][anchor_index, :3], torch.zeros(3), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        result["camera_pose"][-1, :3],
        torch.tensor([0.2, 0.0, 0.0]),
        atol=1e-6,
        rtol=0,
    )
    expected_local = torch.from_numpy(local / 1000.0)
    for time in range(steps):
        torch.testing.assert_close(
            result["hand_pose"][time], expected_local, atol=1e-6, rtol=0
        )


def test_oracle_geometry_can_express_each_wrist_relative_to_its_t0_pose() -> None:
    steps = 4
    anchor_index = 1
    camera = np.stack([_transform(index * 250.0) for index in range(steps)])
    wrist = np.stack(
        [
            np.stack(
                [
                    _transform(1000.0 + index * 100.0),
                    _transform(2000.0 + index * 100.0, 300.0),
                ]
            )
            for index in range(steps)
        ]
    )
    landmarks = np.repeat(wrist[:, :, None, :3, 3], 21, axis=2)
    result = canonicalize_assembly101_oracle_geometry(
        camera,
        wrist,
        landmarks,
        np.ones((steps, 2), dtype=np.float32),
        anchor_index=anchor_index,
        wrist_reference="last_observed_wrist",
    )

    wrist_matrices = pose9_to_matrix(result["wrist_pose"])
    torch.testing.assert_close(
        wrist_matrices[anchor_index],
        torch.eye(4).expand(2, -1, -1),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        result["wrist_pose"][:, :, 0],
        torch.tensor([[-0.1, -0.1], [0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]),
        atol=1e-6,
        rtol=0,
    )
    # Camera motion retains its independent last-observed-camera anchor.
    torch.testing.assert_close(
        result["camera_pose"][:, 0],
        torch.tensor([-0.25, 0.0, 0.25, 0.5]),
        atol=1e-6,
        rtol=0,
    )


def test_wrist_relative_sequence_masks_a_hand_with_an_invalid_t0_anchor() -> None:
    steps = 4
    anchor_index = 1
    camera = np.stack([_transform() for _ in range(steps)])
    wrist = np.stack(
        [np.stack([_transform(index * 100.0), _transform(index * 200.0)]) for index in range(steps)]
    )
    landmarks = np.repeat(wrist[:, :, None, :3, 3], 21, axis=2)
    confidence = np.ones((steps, 2), dtype=np.float32)
    confidence[anchor_index, 0] = 0.0
    result = canonicalize_assembly101_oracle_geometry(
        camera,
        wrist,
        landmarks,
        confidence,
        anchor_index=anchor_index,
        wrist_reference="last_observed_wrist",
    )

    assert not result["wrist_valid"][:, 0].any()
    assert not result["wrist_pose"][:, 0].any()
    assert result["wrist_valid"][:, 1].all()
    # Wrist-local articulation remains usable at timestamps where that hand is tracked.
    assert result["hand_pose_valid"][:, 0].tolist() == [True, False, True, True]
