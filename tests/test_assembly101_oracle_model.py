from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from ego_hand_wm.anticipation.dataset import (
    Assembly101E4OracleDataset,
    oracle_relative_times,
)
from ego_hand_wm.anticipation.model import (
    ORACLE_ABLATION_MODES,
    OracleGeometryAnticipationModel,
    focal_semantic_loss,
)
from ego_hand_wm.anticipation.training import _loader, run_anticipation_training


def _batch(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "visual_tokens": torch.randn(batch, 5, 16),
        "camera_pose": torch.randn(batch, 48, 9),
        "wrist_pose": torch.randn(batch, 48, 2, 9),
        "hand_pose": torch.randn(batch, 48, 2, 21, 3),
        "wrist_valid": torch.ones(batch, 48, 2, dtype=torch.bool),
        "hand_pose_valid": torch.ones(batch, 48, 2, dtype=torch.bool),
        "geometry_time_mask": torch.ones(batch, 48, dtype=torch.bool),
        "time_seconds": torch.from_numpy(oracle_relative_times().astype(np.float32))
        .unsqueeze(0)
        .expand(batch, -1),
        "future_mask": torch.cat(
            (torch.zeros(batch, 32), torch.ones(batch, 16)), dim=1
        ).bool(),
        "execution_mask": torch.cat(
            (torch.zeros(batch, 40), torch.ones(batch, 8)), dim=1
        ).bool(),
    }


def test_oracle_ablation_set_is_camera_inclusive() -> None:
    assert ORACLE_ABLATION_MODES == {
        "rgb",
        "rgb_gt_camera",
        "rgb_gt_wrist",
        "rgb_gt_handpose",
        "rgb_gt_whole_hand",
        "rgb_gt_camera_wrist",
        "rgb_gt_camera_handpose",
        "rgb_gt_camera_whole_hand",
    }


def test_all_ablation_modes_share_identical_common_initialization() -> None:
    common_prefixes = (
        "visual_projection.",
        "semantic_queries",
        "visual_probe.",
        "output_norm.",
        "verb_head.",
        "object_head.",
        "action_head.",
    )
    shared_states: list[dict[str, torch.Tensor]] = []
    for mode in sorted(ORACLE_ABLATION_MODES):
        torch.manual_seed(17)
        model = OracleGeometryAnticipationModel(
            visual_dim=16,
            hidden_dim=32,
            heads=4,
            visual_depth=1,
            geometry_depth=1,
            mode=mode,
            verb_classes=3,
            object_classes=4,
            action_classes=5,
            dropout=0.0,
        )
        shared_states.append(
            {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
                if name.startswith(common_prefixes)
            }
        )
    reference = shared_states[0]
    assert reference
    for state in shared_states[1:]:
        assert state.keys() == reference.keys()
        for name in reference:
            torch.testing.assert_close(state[name], reference[name])


def test_training_order_is_independent_of_model_rng_consumption() -> None:
    dataset = TensorDataset(torch.arange(23))
    config = {
        "seed": 19,
        "training": {
            "batch_size": 4,
            "num_workers": 0,
            "drop_last": False,
        },
    }
    first = torch.cat([batch[0] for batch in _loader(dataset, config, train=True)])
    torch.rand(10_000)
    second = torch.cat([batch[0] for batch in _loader(dataset, config, train=True)])
    torch.testing.assert_close(first, second)


@pytest.mark.parametrize("mode", sorted(ORACLE_ABLATION_MODES))
def test_oracle_model_five_modes(mode: str) -> None:
    model = OracleGeometryAnticipationModel(
        visual_dim=16,
        hidden_dim=32,
        heads=4,
        visual_depth=1,
        geometry_depth=1,
        mode=mode,
        verb_classes=3,
        object_classes=4,
        action_classes=5,
        dropout=0.0,
    )
    output = model(**_batch())
    assert output.verb_logits.shape == (2, 3)
    assert output.object_logits.shape == (2, 4)
    assert output.action_logits.shape == (2, 5)
    labels = torch.tensor([[0, 1, 2], [2, 3, 4]])
    loss, parts = focal_semantic_loss(output, labels)
    assert set(parts) == {"verb", "object", "action"}
    loss.backward()


def test_oracle_dataset_builds_history_gap_and_execution(tmp_path: Path) -> None:
    annotations = tmp_path / "train.csv"
    annotations.write_text(
        "id,video,start_frame,end_frame,action_id,verb_id,noun_id,toyid,is_shared\n"
        "7,recording/HMC_84358933_mono10bit.mp4,150,165,4,2,3,a01,0\n"
    )
    feature_root = tmp_path / "features"
    (feature_root / "train").mkdir(parents=True)
    np.save(feature_root / "train" / "0000007.npy", np.ones((6, 16), np.float16))
    geometry_root = tmp_path / "geometry"
    geometry_root.mkdir()
    steps = 200
    cameras = np.broadcast_to(np.eye(4, dtype=np.float32), (steps, 4, 4)).copy()
    wrists = np.broadcast_to(np.eye(4, dtype=np.float32), (steps, 2, 4, 4)).copy()
    landmarks = np.zeros((steps, 2, 21, 3), dtype=np.float32)
    confidence = np.ones((steps, 2), dtype=np.float32)
    np.savez(
        geometry_root / "recording.npz",
        camera_world_from_camera=cameras,
        wrist_world_from_hand=wrists,
        landmarks_world=landmarks,
        wrist_confidence=confidence,
    )
    dataset = Assembly101E4OracleDataset(
        split="train",
        annotations_csv=annotations,
        feature_root=feature_root,
        geometry_root=geometry_root,
    )
    item = dataset[0]
    assert item["camera_pose"].shape == (48, 9)
    assert item["hand_pose"].shape == (48, 2, 21, 3)
    assert item["future_mask"].sum().item() == 16
    assert item["execution_mask"].sum().item() == 8
    # A 0.5-second target action exposes four 8-Hz execution samples, then masks padding.
    assert item["geometry_time_mask"][40:].tolist() == [True] * 4 + [False] * 4


def test_oracle_dataset_keeps_official_sample_without_released_pose(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "validation.csv"
    annotations.write_text(
        "id,video,start_frame,end_frame,action_id,verb_id,noun_id,toyid,is_shared\n"
        "9,missing/HMC_21179183_mono10bit.mp4,150,180,4,2,3,a01,0\n"
    )
    feature_root = tmp_path / "features"
    (feature_root / "validation").mkdir(parents=True)
    np.save(
        feature_root / "validation" / "0000009.npy",
        np.ones((6, 16), np.float16),
    )
    geometry_root = tmp_path / "geometry"
    geometry_root.mkdir()

    dataset = Assembly101E4OracleDataset(
        split="validation",
        annotations_csv=annotations,
        feature_root=feature_root,
        geometry_root=geometry_root,
        require_all_caches=True,
    )
    item = dataset[0]
    assert len(dataset) == 1
    assert dataset.missing_geometry_recordings == {"missing"}
    assert not item["geometry_available"]
    assert not item["geometry_time_mask"].any()
    assert not item["wrist_valid"].any()
    assert not item["hand_pose_valid"].any()


@pytest.mark.parametrize(
    ("steps", "expected_available", "expected_valid_steps"),
    (
        # The canonical cutoff camera exists, but the released pose stream ends
        # during the oracle future window.  Only those missing timestamps mask.
        (135, True, 35),
        # The released pose stream ends before the observation cutoff, so a
        # canonical anchor cannot be constructed and the entire stream masks.
        (110, False, 0),
    ),
)
def test_oracle_dataset_masks_truncated_pose_stream_without_shifting_sample(
    tmp_path: Path,
    steps: int,
    expected_available: bool,
    expected_valid_steps: int,
) -> None:
    annotations = tmp_path / "validation.csv"
    annotations.write_text(
        "id,video,start_frame,end_frame,action_id,verb_id,noun_id,toyid,is_shared\n"
        "9,recording/HMC_21179183_mono10bit.mp4,150,180,4,2,3,a01,0\n"
    )
    feature_root = tmp_path / "features"
    (feature_root / "validation").mkdir(parents=True)
    np.save(
        feature_root / "validation" / "0000009.npy",
        np.ones((6, 16), np.float16),
    )
    geometry_root = tmp_path / "geometry"
    geometry_root.mkdir()
    np.savez(
        geometry_root / "recording.npz",
        camera_world_from_camera=np.broadcast_to(
            np.eye(4, dtype=np.float32), (steps, 4, 4)
        ).copy(),
        wrist_world_from_hand=np.broadcast_to(
            np.eye(4, dtype=np.float32), (steps, 2, 4, 4)
        ).copy(),
        landmarks_world=np.zeros((steps, 2, 21, 3), dtype=np.float32),
        wrist_confidence=np.ones((steps, 2), dtype=np.float32),
    )

    item = Assembly101E4OracleDataset(
        split="validation",
        annotations_csv=annotations,
        feature_root=feature_root,
        geometry_root=geometry_root,
        require_all_caches=True,
    )[0]
    assert bool(item["geometry_available"]) is expected_available
    assert item["geometry_time_mask"].sum().item() == expected_valid_steps
    assert item["anchor_frame"] == 120


def test_training_runs_full_validation_every_two_epochs(tmp_path: Path) -> None:
    annotation_root = tmp_path / "annotations"
    annotation_root.mkdir()
    feature_root = tmp_path / "features"
    geometry_root = tmp_path / "geometry"
    geometry_root.mkdir()
    for split, segment_id in (("train", 1), ("validation", 2)):
        (annotation_root / f"{split}.csv").write_text(
            "id,video,start_frame,end_frame,action_id,verb_id,noun_id,toyid,is_shared\n"
            f"{segment_id},recording/HMC_84358933_mono10bit.mp4,150,180,4,2,3,a01,0\n"
        )
        (feature_root / split).mkdir(parents=True)
        np.save(
            feature_root / split / f"{segment_id:07d}.npy",
            np.ones((6, 16), np.float16),
        )
    steps = 200
    np.savez(
        geometry_root / "recording.npz",
        camera_world_from_camera=np.broadcast_to(
            np.eye(4, dtype=np.float32), (steps, 4, 4)
        ).copy(),
        wrist_world_from_hand=np.broadcast_to(
            np.eye(4, dtype=np.float32), (steps, 2, 4, 4)
        ).copy(),
        landmarks_world=np.zeros((steps, 2, 21, 3), dtype=np.float32),
        wrist_confidence=np.ones((steps, 2), dtype=np.float32),
    )
    output_dir = tmp_path / "run"
    config = {
        "seed": 1,
        "runtime": {"device": "cpu"},
        "data": {
            "annotations_root": str(annotation_root),
            "evaluation_root": str(annotation_root),
            "feature_root": str(feature_root),
            "geometry_root": str(geometry_root),
            "require_all_caches": True,
        },
        "model": {
            "visual_dim": 16,
            "hidden_dim": 32,
            "heads": 4,
            "visual_depth": 1,
            "geometry_depth": 1,
            "mode": "rgb_gt_camera_whole_hand",
            "verb_classes": 3,
            "object_classes": 4,
            "action_classes": 5,
            "dropout": 0.0,
        },
        "training": {
            "output_dir": str(output_dir),
            "batch_size": 1,
            "num_workers": 0,
            "drop_last": False,
            "epochs": 3,
            "validation_interval": 2,
            "learning_rate": 1e-3,
            "schedule_epoch": 2,
            "bf16": False,
        },
    }
    result = run_anticipation_training(config)
    history = json.loads((output_dir / "metrics.json").read_text())
    assert result["epochs"] == 3
    assert "overall/action_mean_top5_recall" not in history[0]
    assert "overall/action_mean_top5_recall" in history[1]
    # The final epoch is always validated even when it is not divisible by two.
    assert "overall/action_mean_top5_recall" in history[2]
    assert (output_dir / "best.pt").is_file()
    assert (output_dir / "last.pt").is_file()
