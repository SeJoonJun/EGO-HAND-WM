from pathlib import Path

import numpy as np
import torch

from ego_hand_wm.contracts.batch import canonical_collate
from ego_hand_wm.data.dinov3_features import _encode_requested_frames
from ego_hand_wm.data.feature_shards import (
    EpisodeFeatureRecord,
    decode_feature_record,
    encode_feature_record,
)
from ego_hand_wm.data.synthetic import SyntheticCanonicalDataset
from ego_hand_wm.models.encoders import ContextEncoder


def test_spatial_feature_record_round_trip() -> None:
    features = np.arange(3 * 16 * 8, dtype=np.float32).reshape(3, 16, 8)
    record = EpisodeFeatureRecord(
        annotation_member="epic/example.npy",
        annotation_sha256="a" * 64,
        dataset_name="epic",
        video_name="P01_01",
        frame_ids=np.asarray([4, 7, 12], dtype=np.int64),
        frame_times_seconds=np.asarray([0.2, 0.4, 0.7], dtype=np.float64),
        pooled_features=features,
        valid_mask=np.ones(3, dtype=np.bool_),
        extractor_id="b" * 64,
    )

    restored = decode_feature_record(encode_feature_record(record))

    assert restored.pooled_features.shape == (3, 16, 8)
    assert restored.spatial_tokens == 16
    assert restored.feature_dim == 8
    np.testing.assert_allclose(restored.pooled_features, features.astype(np.float16))


def test_requested_frame_encoding_preserves_ids_times_and_spatial_grid() -> None:
    frame_ids = np.asarray([3, 9, 20], dtype=np.int64)
    frame_times = np.asarray([0.1, 0.3, 0.7], dtype=np.float64)

    def reader(path: Path, ids: list[int], times: list[float]):
        assert path == Path("video.mp4")
        assert ids == frame_ids.tolist()
        assert times == frame_times.tolist()
        for frame_id in ids:
            yield frame_id, np.full((4, 5, 3), frame_id, dtype=np.uint8)

    class Encoder:
        def encode(self, frames: np.ndarray) -> np.ndarray:
            values = frames[:, 0, 0, 0].astype(np.float32)
            return np.broadcast_to(values[:, None, None], (len(frames), 16, 8)).copy()

    result = _encode_requested_frames(
        Path("video.mp4"),
        frame_ids,
        frame_times_seconds=frame_times,
        encoder=Encoder(),
        frame_reader=reader,
        batch_size=2,
    )

    assert list(result) == frame_ids.tolist()
    assert all(feature.shape == (16, 8) for feature in result.values())
    assert float(result[20][0, 0]) == 20.0


def test_context_encoder_consumes_all_spatial_tokens() -> None:
    sample = SyntheticCanonicalDataset(
        length=1, history_steps=3, future_steps=4, image_size=16
    )[0]
    sample.pop("context_images")
    sample["context_visual_features"] = torch.randn(3, 16, 8)
    batch = canonical_collate([sample])
    encoder = ContextEncoder(
        {
            "hidden_dim": 32,
            "heads": 4,
            "context_depth": 1,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "vision": {"kind": "precomputed", "precomputed_dim": 8, "spatial_tokens": 16},
            "text": {"kind": "hash", "max_tokens": 8},
        }
    )

    tokens, valid = encoder(batch)

    # The first H*P fused tokens are the 3 frames x 16 spatial DINO cells.
    assert tokens.shape[1] >= 3 * 16
    assert valid[0, : 3 * 16].all()
