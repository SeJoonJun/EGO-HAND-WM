from pathlib import Path

import numpy as np
import torch

from ego_hand_wm.anticipation.dataset import Assembly101E4AnticipationDataset
from ego_hand_wm.anticipation.metrics import class_mean_topk_recall
from ego_hand_wm.anticipation.model import TempAggGeometryModel, anticipation_loss
from ego_hand_wm.anticipation.protocol import CONTEXT_FRAMES, read_e4_anticipation_csv
from ego_hand_wm.data.adapters.assembly101 import (
    annotation_to_raw_frame,
    canonicalize_assembly101_geometry,
    e4_pose_camera_key,
)


def _transform(x_mm: float = 0.0, y_mm: float = 0.0) -> np.ndarray:
    value = np.eye(4, dtype=np.float32)
    value[:3, 3] = [x_mm, y_mm, 0.0]
    return value


def test_assembly_geometry_uses_last_camera_anchor_and_masks_wrists() -> None:
    camera = np.stack([_transform(0), _transform(100), _transform(200)])
    wrist = np.stack(
        [
            np.stack([_transform(1000 + step * 100), _transform(0, 500)])
            for step in range(3)
        ]
    )
    confidence = np.asarray([[1.0, 0.0], [1.0, 0.1], [1.0, 0.2]], dtype=np.float32)
    result = canonicalize_assembly101_geometry(camera, wrist, confidence)
    torch.testing.assert_close(result["camera_pose"][-1, :3], torch.zeros(3), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        result["camera_pose"][0, :3], torch.tensor([-0.2, 0.0, 0.0]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        result["wrist_pose"][:, 0, :3],
        torch.tensor([[0.8, 0.0, 0.0], [0.9, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        atol=1e-6,
        rtol=0,
    )
    assert not result["wrist_valid"][:, 1].any()
    assert not result["wrist_pose"][:, 1].any()
    assert annotation_to_raw_frame(37) == 74
    assert e4_pose_camera_key("HMC_84358933_mono10bit.mp4") == "84358933:mono10bit"


def test_official_csv_normalization_filters_e4_and_retains_zero_duration(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "id,video,start_frame,end_frame,action_id,verb_id,noun_id,toyid,is_shared\n"
        "3,rec/HMC_84358933_mono10bit.mp4,100,100,7,2,4,a01,0\n"
        "3,rec/C10095_rgb.mp4,100,100,7,2,4,a01,0\n"
    )
    records = read_e4_anticipation_csv(csv_path)
    assert len(records) == 1
    assert records[0].anchor_frame == 70
    assert records[0].action == 7


def test_real_dataset_contract_from_synthetic_caches(tmp_path: Path) -> None:
    annotations = tmp_path / "train.csv"
    recording = "rec"
    stem = "HMC_84358933_mono10bit"
    annotations.write_text(
        "id,video,start_frame,end_frame,action_id,verb_id,noun_id,toyid,is_shared\n"
        f"3,{recording}/{stem}.mp4,100,120,7,2,4,a01,0\n"
    )
    feature_root = tmp_path / "features"
    geometry_root = tmp_path / "geometry"
    feature_root.mkdir()
    geometry_root.mkdir()
    np.save(feature_root / f"{recording}__{stem}.npy", np.zeros((100, 16), dtype=np.float16))
    camera = np.stack([_transform(index) for index in range(100)])
    wrist = np.stack(
        [np.stack([_transform(100 + index), _transform(200 + index)]) for index in range(100)]
    )
    np.savez_compressed(
        geometry_root / f"{recording}.npz",
        camera_world_from_camera=camera,
        wrist_world_from_hand=wrist,
        wrist_confidence=np.ones((100, 2), dtype=np.float32),
    )
    dataset = Assembly101E4AnticipationDataset(
        annotations_csv=annotations,
        feature_root=feature_root,
        geometry_root=geometry_root,
    )
    sample = dataset[0]
    assert sample["rgb_features"].shape == (CONTEXT_FRAMES, 16)
    assert sample["camera_pose"].shape == (CONTEXT_FRAMES, 9)
    assert sample["wrist_pose"].shape == (CONTEXT_FRAMES, 2, 9)
    assert sample["labels"].tolist() == [2, 4, 7]


def test_all_ablation_modes_and_rgb_geometry_isolation() -> None:
    batch = 2
    inputs = {
        "rgb_features": torch.randn(batch, CONTEXT_FRAMES, 16),
        "camera_pose": torch.randn(batch, CONTEXT_FRAMES, 9),
        "wrist_pose": torch.randn(batch, CONTEXT_FRAMES, 2, 9),
        "wrist_confidence": torch.rand(batch, CONTEXT_FRAMES, 2),
        "wrist_valid": torch.ones(batch, CONTEXT_FRAMES, 2, dtype=torch.bool),
    }
    for mode in ("rgb", "rgb_camera", "rgb_wrist", "rgb_camera_wrist"):
        model = TempAggGeometryModel(
            rgb_dim=16,
            hidden_dim=32,
            heads=4,
            mode=mode,
            action_classes=7,
            verb_classes=3,
            object_classes=5,
            dropout=0.0,
        ).eval()
        output = model(**inputs)
        assert len(output.action_logits) == 4
        assert output.ensemble()["action"].shape == (batch, 7)
        loss, _ = anticipation_loss(output, torch.tensor([[0, 1, 2], [2, 4, 6]]))
        assert torch.isfinite(loss)
        if mode == "rgb":
            changed = dict(inputs)
            changed["camera_pose"] = inputs["camera_pose"] + 1000
            changed["wrist_pose"] = inputs["wrist_pose"] - 1000
            torch.testing.assert_close(
                output.ensemble()["action"], model(**changed).ensemble()["action"]
            )


def test_class_mean_top5_recall_is_not_frequency_weighted() -> None:
    logits = torch.tensor(
        [
            [9.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
        ]
    )
    targets = torch.tensor([0, 0, 0, 1])
    # Class 0 recall=1 and class 1 recall=0, so the class mean is 0.5 (sample accuracy is 0.75).
    assert class_mean_topk_recall(logits, targets, k=1) == 0.5

