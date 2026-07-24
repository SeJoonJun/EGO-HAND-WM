import io
import json
import runpy
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ego_hand_wm.benchmarks import Hot3DClipsForecastDataset
from ego_hand_wm.contracts import canonical_collate
from ego_hand_wm.contracts.schema import SCHEMA
from ego_hand_wm.data.adapters.hot3d_clips import canonicalize_hot3d_clip_window
from ego_hand_wm.flow.rectified_flow import make_flow_training_sample
from ego_hand_wm.geometry.se3 import pose9_to_matrix
from ego_hand_wm.losses import WorldActionLoss
from ego_hand_wm.models.world_action_model import WorldActionModel
from ego_hand_wm.data.trajectory_features import TRAJECTORY_FEATURE_CONTRACT


def _pose(x):
    return {
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "translation_xyz": [float(x), 0.0, 0.0],
    }


def _add_bytes(archive, name, payload):
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _add_json(archive, name, value):
    _add_bytes(archive, name, json.dumps(value).encode("utf-8"))


def _make_clip(path, frames=22):
    image = Image.new("RGB", (8, 8), color=(40, 80, 120))
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    with tarfile.open(path, "w") as archive:
        for frame in range(frames):
            key = f"{frame:06d}"
            calibration = {
                "image_width": 8,
                "image_height": 8,
                "projection_params": [4.0, 3.5, 3.5],
            }
            _add_json(
                archive,
                f"{key}.cameras.json",
                {"214-1": {"T_world_from_camera": _pose(0.1 * frame), "calibration": calibration}},
            )
            _add_json(
                archive,
                f"{key}.hands.json",
                {
                    "left": {
                        "umetrack_pose": {
                            "T_world_from_wrist": _pose(1.0 + 0.2 * frame),
                            "joint_angles": [0.0] * 22,
                        },
                        "mano_pose": {"thetas": [0.0] * 15, "wrist_xform": [0.0] * 6},
                    },
                    "right": {
                        "umetrack_pose": {
                            "T_world_from_wrist": _pose(-1.0),
                            "joint_angles": [0.0] * 22,
                        },
                        "mano_pose": {"thetas": [0.0] * 15, "wrist_xform": [0.0] * 6},
                    },
                },
            )
            timestamp = 1_000_000_000 + round(frame * 1e9 / 30.0)
            _add_json(
                archive,
                f"{key}.info.json",
                {
                    "participant_id": "P0001",
                    "sequence_id": "P0001_synthetic",
                    "ref_timestamp_ns": timestamp,
                    "image_timestamps_ns": {"214-1": timestamp},
                },
            )
            _add_bytes(archive, f"{key}.image_214-1.jpg", image_bytes.getvalue())


def _temporal():
    history = list(range(6))
    future = list(range(6, 22))
    history_time = [(index - 5) / 30 for index in history]
    future_time = [(index - 5) / 30 for index in future]
    return history, future, history_time, future_time


def test_hot3d_clip_uses_last_observed_camera_and_metric_world_wrist(tmp_path):
    tar_path = tmp_path / "clip-001849.tar"
    _make_clip(tar_path)
    history, future, history_time, future_time = _temporal()
    sample = canonicalize_hot3d_clip_window(
        tar_path,
        history,
        future,
        history_time,
        future_time,
        tracked_hand="left",
    )

    anchor_camera = pose9_to_matrix(sample["history_state"][-1, SCHEMA.camera])
    torch.testing.assert_close(anchor_camera, torch.eye(4), atol=1e-6, rtol=1e-6)
    assert sample["history_state"][-1, SCHEMA.left_wrist.start].item() == pytest.approx(1.5)
    assert sample["future_state"][-1, SCHEMA.left_wrist.start].item() == pytest.approx(4.7)
    assert sample["history_stream_mask"][-1].tolist() == [True, True, True, False, False]
    assert sample["future_query_stream_mask"][0].tolist() == [True, True, False, False, False]
    assert sample["future_state_mask"][:, SCHEMA.left_wrist.start : SCHEMA.left_wrist.start + 3].all()
    assert not sample["future_state_mask"][:, SCHEMA.left_wrist.start + 3 : SCHEMA.left_wrist.stop].any()
    assert not sample["future_state_mask"][:, SCHEMA.left_mano].any()
    assert sample["metadata"]["mano_streams"] == "masked_requires_official_hand_model_decode"
    assert sample["metadata"]["supervised_geometry"] == "camera_se3_and_tracked_wrist_xyz"


def test_hot3d_full_wrist_se3_is_an_explicit_ablation(tmp_path):
    tar_path = tmp_path / "clip-001849.tar"
    _make_clip(tar_path)
    history, future, history_time, future_time = _temporal()
    sample = canonicalize_hot3d_clip_window(
        tar_path,
        history,
        future,
        history_time,
        future_time,
        tracked_hand="left",
        wrist_target="se3",
    )

    assert sample["future_state_mask"][:, SCHEMA.left_wrist].all()
    assert sample["metadata"]["supervised_geometry"] == "camera_se3_and_tracked_wrist_se3"


def test_hot3d_first_observed_spatial_anchor_is_explicit_option(tmp_path):
    tar_path = tmp_path / "clip-001849.tar"
    _make_clip(tar_path)
    history, future, history_time, future_time = _temporal()
    sample = canonicalize_hot3d_clip_window(
        tar_path,
        history,
        future,
        history_time,
        future_time,
        tracked_hand="left",
        anchor="first_observed",
    )
    first_camera = pose9_to_matrix(sample["history_state"][0, SCHEMA.camera])
    last_history_camera = pose9_to_matrix(sample["history_state"][-1, SCHEMA.camera])
    torch.testing.assert_close(first_camera, torch.eye(4), atol=1e-6, rtol=1e-6)
    assert last_history_camera[0, 3].item() == pytest.approx(0.5)
    assert sample["history_time"][-1].item() == pytest.approx(0.0)


def test_hot3d_manifest_dataset_decodes_rgb_and_collates(tmp_path):
    tar_path = tmp_path / "clip-001849.tar"
    _make_clip(tar_path)
    history, future, history_time, future_time = _temporal()
    record = {
        "schema_version": 1,
        "protocol": "h6_k16_30hz",
        "dataset": "hot3d_clips_aria",
        "split": "train",
        "sample_id": "hot3d:001849:000:left",
        "participant_id": "P0001",
        "sequence_id": "P0001_synthetic",
        "clip_id": 1849,
        "tar_path": str(tar_path),
        "tracked_hand": "left",
        "history_indices": history,
        "future_indices": future,
        "history_time_seconds": history_time,
        "future_time_seconds": future_time,
    }
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    dataset = Hot3DClipsForecastDataset(
        {
            "split": "train",
            "manifest": str(manifest),
            "decode_rgb": True,
            "image_size": 16,
            "missing_text_feature_dim": 8,
        }
    )
    sample = dataset[0]
    assert sample["context_images"].shape == (6, 3, 16, 16)
    assert sample["history_state"].shape == (6, 207)
    assert sample["future_state"].shape == (16, 207)
    assert sample["metadata"]["source_dataset"] == "hot3d_clips_aria"
    batch = canonical_collate([sample])
    batch.validate()
    assert np.isfinite(batch.intrinsics.numpy()).all()


def test_hot3d_batch_runs_world_model_loss_and_backward(tmp_path):
    tar_path = tmp_path / "clip-001849.tar"
    _make_clip(tar_path)
    history, future, history_time, future_time = _temporal()
    record = {
        "schema_version": 1,
        "protocol": "h6_k16_30hz",
        "dataset": "hot3d_clips_aria",
        "split": "train",
        "sample_id": "hot3d:001849:000:left",
        "participant_id": "P0001",
        "sequence_id": "P0001_synthetic",
        "clip_id": 1849,
        "tar_path": str(tar_path),
        "tracked_hand": "left",
        "history_indices": history,
        "future_indices": future,
        "history_time_seconds": history_time,
        "future_time_seconds": future_time,
    }
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    dataset = Hot3DClipsForecastDataset(
        {
            "split": "train",
            "manifest": str(manifest),
            "decode_rgb": True,
            "image_size": 16,
        }
    )
    batch = canonical_collate([dataset[0]])
    model = WorldActionModel(
        {
            "hidden_dim": 32,
            "heads": 4,
            "context_depth": 1,
            "depth": 1,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "physical_max_period": 10.0,
            "future_visual_latent_dim": 0,
            "vision": {"kind": "tiny"},
            "text": {"kind": "hash", "max_tokens": 8},
        }
    )
    flow = make_flow_training_sample(batch)
    prediction, _, _, _, _ = model(batch, flow.noisy_state, flow.flow_time)
    metrics = WorldActionLoss({"rotation_weight": 0.1, "ego_weight": 0.2})(
        batch,
        prediction,
        flow.target_velocity,
        flow.noisy_state,
        flow.flow_time,
    )
    assert torch.isfinite(metrics["loss"])
    metrics["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_hot3d_dinotxt_extractor_reads_tar_frames(tmp_path):
    tar_path = tmp_path / "clip-001849.tar"
    _make_clip(tar_path)
    module = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/extract_trajectory_dinotxt.py")
    )

    class FakeEncoder:
        def encode(self, frames):
            values = frames.astype(np.float32).mean(axis=(1, 2, 3))
            return np.repeat(values[:, None, None], 2, axis=2)

    output = np.zeros((1, 2, 1, 2), dtype=np.float16)
    encoded = module["_encode_hot3d_tar"](
        str(tar_path),
        {0: [(0, 0)], 5: [(0, 1)]},
        output=output,
        encoder=FakeEncoder(),
        batch_size=1,
        camera_stream="214-1",
    )

    assert encoded == 2
    assert np.isfinite(output).all()
    assert output[0, 0].any() and output[0, 1].any()


def test_hot3d_dataset_attaches_history_and_future_dinotxt_features(tmp_path):
    tar_path = tmp_path / "clip-001849.tar"
    _make_clip(tar_path)
    history, future, history_time, future_time = _temporal()
    sample_id = "hot3d:001849:000:left"
    record = {
        "schema_version": 1,
        "protocol": "h6_k16_30hz",
        "dataset": "hot3d_clips_aria",
        "split": "train",
        "sample_id": sample_id,
        "participant_id": "P0001",
        "sequence_id": "P0001_synthetic",
        "clip_id": 1849,
        "tar_path": str(tar_path),
        "tracked_hand": "left",
        "history_indices": history,
        "future_indices": future,
        "history_time_seconds": history_time,
        "future_time_seconds": future_time,
    }
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    feature_dir = tmp_path / "features/hot3d_clips_aria"
    feature_dir.mkdir(parents=True)
    for prefix, steps in (("train", 6), ("train.future", 16)):
        np.save(
            feature_dir / f"{prefix}.sample_ids.npy",
            np.asarray([sample_id]),
            allow_pickle=False,
        )
        np.save(
            feature_dir / f"{prefix}.features.npy",
            np.ones((1, steps, 17, 8), dtype=np.float16),
            allow_pickle=False,
        )
        (feature_dir / f"{prefix}.SUCCESS.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "contract": TRAJECTORY_FEATURE_CONTRACT,
                    "history_steps" if prefix == "train" else "future_steps": steps,
                    "total_tokens": 17,
                    "feature_dim": 8,
                }
            ),
            encoding="utf-8",
        )
    dataset = Hot3DClipsForecastDataset(
        {
            "split": "train",
            "manifest": str(manifest),
            "visual_feature_root": str(tmp_path / "features"),
            "future_visual_feature_root": str(tmp_path / "features"),
            "future_visual_splits": ["train"],
            "visual_feature_dtype": "float16",
        }
    )

    sample = dataset[0]
    assert sample["context_visual_features"].shape == (6, 17, 8)
    assert sample["future_visual_latents"].shape == (16, 17, 8)
    assert sample["context_visual_features"].dtype == torch.float16
