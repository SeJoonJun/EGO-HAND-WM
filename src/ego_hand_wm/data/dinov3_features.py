"""Offline DINOv3 extraction for VITRA annotation shards.

Only local repositories and local checkpoint files are accepted.  The extraction unit is an
annotation tar shard: episodes are grouped by source video, requested physical frame IDs are
decoded once per video within that shard, and spatial features are written to one aligned tar.
"""

from __future__ import annotations

import io
import os
import sys
import tarfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as functional

from ego_hand_wm.data.feature_shards import (
    FEATURE_CONTRACT,
    FEATURE_SCHEMA_VERSION,
    ROLE_CAPABILITIES,
    EpisodeFeatureRecord,
    FeatureShardError,
    annotation_member_to_feature_member,
    atomic_write_json,
    encode_feature_record,
    extractor_id,
    feature_shard_paths,
    load_feature_manifest,
    sha256_bytes,
    sha256_file,
)


VIDEO_SUFFIXES = frozenset({".mp4", ".webm"})
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SpatialFeatureEncoder(Protocol):
    def encode(self, rgb_frames: np.ndarray) -> np.ndarray:
        """Return finite ``[B,P,D]`` spatial features for uint8 RGB frames."""


FrameReader = Callable[
    [Path, Sequence[int], Sequence[float]], Iterator[tuple[int, np.ndarray]]
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_local_file(path: str | Path, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Local {description} is missing: {resolved}")
    return resolved


def _require_local_directory(path: str | Path, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Local {description} is missing: {resolved}")
    return resolved


def build_extractor_metadata(
    *,
    repo_path: str | Path,
    weights_path: str | Path,
    model_name: str,
    input_size: int,
    spatial_grid_size: int,
) -> dict[str, Any]:
    repo = _require_local_directory(repo_path, "DINOv3 repository")
    weights = _require_local_file(weights_path, "DINOv3 weights")
    if input_size <= 0 or spatial_grid_size <= 0:
        raise ValueError("input_size and spatial_grid_size must be positive")
    return {
        "backend": "dinov3_local_backbone",
        "model_name": str(model_name),
        "repo_path": str(repo),
        "repo_hubconf_sha256": sha256_file(repo / "hubconf.py"),
        "weights_path": str(weights),
        "weights_size_bytes": weights.stat().st_size,
        "weights_sha256": sha256_file(weights),
        "input_size": int(input_size),
        "resize": "bilinear_square_antialias",
        "input_color": "rgb",
        "input_range": "uint8_0_255",
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "pooling": "adaptive_average_spatial_grid",
        "spatial_grid_size": int(spatial_grid_size),
        "spatial_tokens": int(spatial_grid_size**2),
        "storage_dtype": "float16",
    }


def build_dinotxt_extractor_metadata(
    *,
    repo_path: str | Path,
    weights_path: str | Path,
    dinotxt_weights_path: str | Path,
    bpe_path: str | Path,
    model_name: str,
    input_size: int,
    spatial_grid_size: int,
) -> dict[str, Any]:
    """Describe the exact local DINO.txt visual/text representation contract."""
    metadata = build_extractor_metadata(
        repo_path=repo_path,
        weights_path=weights_path,
        model_name=model_name,
        input_size=input_size,
        spatial_grid_size=spatial_grid_size,
    )
    adapter = _require_local_file(dinotxt_weights_path, "DINO.txt adapter weights")
    bpe = _require_local_file(bpe_path, "DINO.txt BPE vocabulary")
    spatial_tokens = int(spatial_grid_size**2)
    metadata.update(
        {
            "backend": "dinov3_local_dinotxt",
            "dinotxt_weights_path": str(adapter),
            "dinotxt_weights_size_bytes": adapter.stat().st_size,
            "dinotxt_weights_sha256": sha256_file(adapter),
            "bpe_path": str(bpe),
            "bpe_sha256": sha256_file(bpe),
            "vision_head": "official_dinotxt_two_transformer_blocks",
            "pooling": "post_dinotxt_head_adaptive_average_spatial_grid",
            "token_layout": "post_head_cls_then_row_major_spatial",
            "class_tokens": 1,
            "spatial_tokens": spatial_tokens,
            "total_tokens": 1 + spatial_tokens,
            "feature_dim": 1024,
            "global_descriptor": "l2_normalize(concat(cls,mean(spatial)))",
            "global_dim": 2048,
            "text_descriptor": "official_dinotxt_l2_normalized_eot",
            "text_dim": 2048,
        }
    )
    return metadata


def _load_local_dinov3_backbone(
    *, repo_path: Path, weights_path: Path, model_name: str
) -> torch.nn.Module:
    sys.path.insert(0, str(repo_path))
    try:
        from dinov3.hub import backbones

        constructor = getattr(backbones, str(model_name))
        backbone = constructor(pretrained=False)
    finally:
        if sys.path[0] == str(repo_path):
            sys.path.pop(0)
    state = torch.load(weights_path, map_location="cpu", weights_only=True, mmap=True)
    backbone.load_state_dict(state, strict=True)
    return backbone


class LocalDinoTxtVisualEncoder:
    """Official frozen DINO.txt ViT-L vision head with compact spatial retention.

    Each frame is stored as 17 tokens by default: the post-head class token followed by a
    row-major 4x4 average pool of the post-head patch tokens.  The exact aligned global image
    descriptor can therefore be reconstructed losslessly as ``concat(cls, patches.mean(0))``
    relative to this pooling (equal-area 16x16 -> 4x4 pooling preserves the patch mean).
    """

    def __init__(
        self,
        *,
        repo_path: str | Path,
        weights_path: str | Path,
        dinotxt_weights_path: str | Path,
        model_name: str = "dinov3_vitl16",
        input_size: int = 256,
        spatial_grid_size: int = 4,
        device: str = "auto",
    ) -> None:
        self.repo_path = _require_local_directory(repo_path, "DINOv3 repository")
        self.weights_path = _require_local_file(weights_path, "DINOv3 weights")
        self.dinotxt_weights_path = _require_local_file(
            dinotxt_weights_path, "DINO.txt adapter weights"
        )
        if input_size <= 0 or spatial_grid_size <= 0:
            raise ValueError("input_size and spatial_grid_size must be positive")
        self.input_size = int(input_size)
        self.spatial_grid_size = int(spatial_grid_size)
        self.output_tokens = 1 + self.spatial_grid_size**2
        self.feature_dim = 1024
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for DINO.txt extraction but is unavailable")

        backbone = _load_local_dinov3_backbone(
            repo_path=self.repo_path,
            weights_path=self.weights_path,
            model_name=model_name,
        )
        sys.path.insert(0, str(self.repo_path))
        try:
            from dinov3.eval.text.vision_tower import VisionTower

            self.model = VisionTower(
                backbone=backbone,
                freeze_backbone=True,
                embed_dim=2048,
                num_head_blocks=2,
                head_blocks_block_drop_path=0.3,
                use_class_token=True,
                use_patch_tokens=True,
                patch_token_layer=1,
                patch_tokens_pooler_type="mean",
                use_linear_projection=False,
            )
        finally:
            if sys.path[0] == str(self.repo_path):
                sys.path.pop(0)
        adapter = torch.load(
            self.dinotxt_weights_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        head_state = {
            key.removeprefix("visual_model.head."): value
            for key, value in adapter.items()
            if key.startswith("visual_model.head.")
        }
        self.model.head.load_state_dict(head_state, strict=True)
        del adapter, head_state
        self.model = self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def encode(self, rgb_frames: np.ndarray) -> np.ndarray:
        frames = np.asarray(rgb_frames)
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise ValueError(f"Expected uint8 RGB [B,H,W,3], got {frames.shape} {frames.dtype}")
        host_frames = (
            frames
            if frames.flags.c_contiguous and frames.flags.writeable
            else frames.copy()
        )
        images = torch.from_numpy(host_frames).to(self.device)
        images = images.permute(0, 3, 1, 2).float().div_(255.0)
        images = functional.interpolate(
            images,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        images = (images - self.mean) / self.std
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            class_token, patches, _ = self.model.get_class_and_patch_tokens(images)
            patch_side = int(round(patches.shape[1] ** 0.5))
            if patch_side * patch_side != patches.shape[1]:
                raise FeatureShardError(
                    f"DINO.txt patch count is not square: {patches.shape[1]}"
                )
            spatial = patches.reshape(
                patches.shape[0], patch_side, patch_side, patches.shape[2]
            ).permute(0, 3, 1, 2)
            spatial = functional.adaptive_avg_pool2d(
                spatial.float(), (self.spatial_grid_size, self.spatial_grid_size)
            ).flatten(2).transpose(1, 2)
            tokens = torch.cat((class_token.float().unsqueeze(1), spatial), dim=1)
        result = tokens.cpu().numpy()
        expected = (len(frames), self.output_tokens, self.feature_dim)
        if result.shape != expected or not np.isfinite(result).all():
            raise FeatureShardError(
                f"DINO.txt returned invalid visual features: {result.shape}, expected {expected}"
            )
        return result


class LocalDinoTxtTextEncoder:
    """Load only the official DINO.txt text tower for one-time prompt caching."""

    output_dim = 2048

    def __init__(
        self,
        *,
        repo_path: str | Path,
        dinotxt_weights_path: str | Path,
        bpe_path: str | Path,
        device: str = "auto",
    ) -> None:
        self.repo_path = _require_local_directory(repo_path, "DINOv3 repository")
        self.dinotxt_weights_path = _require_local_file(
            dinotxt_weights_path, "DINO.txt adapter weights"
        )
        self.bpe_path = _require_local_file(bpe_path, "DINO.txt BPE vocabulary")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for DINO.txt text encoding but is unavailable")

        sys.path.insert(0, str(self.repo_path))
        try:
            from dinov3.eval.text.text_tower import TextTower
            from dinov3.eval.text.text_transformer import TextTransformer
            from dinov3.eval.text.tokenizer import get_tokenizer

            backbone = TextTransformer(
                context_length=77,
                vocab_size=49408,
                dim=1280,
                num_heads=20,
                num_layers=24,
                ffn_ratio=4,
                is_causal=True,
                ls_init_value=None,
                dropout_prob=0.0,
            )
            self.model = TextTower(
                backbone=backbone,
                freeze_backbone=True,
                embed_dim=2048,
                num_head_blocks=0,
                head_blocks_is_causal=False,
                head_blocks_block_drop_prob=0.0,
                tokens_pooler_type="argmax",
                use_linear_projection=True,
            )
            self.tokenizer = get_tokenizer(str(self.bpe_path))
        finally:
            if sys.path[0] == str(self.repo_path):
                sys.path.pop(0)
        adapter = torch.load(
            self.dinotxt_weights_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        text_state = {
            key.removeprefix("text_model."): value
            for key, value in adapter.items()
            if key.startswith("text_model.")
        }
        self.model.load_state_dict(text_state, strict=True)
        del adapter, text_state
        self.model = self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.output_dim), dtype=np.float32)
        token_indices = self.tokenizer.tokenize([str(text) for text in texts]).to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            features = functional.normalize(self.model(token_indices).float(), dim=-1)
        result = features.cpu().numpy()
        if result.shape != (len(texts), self.output_dim) or not np.isfinite(result).all():
            raise FeatureShardError(f"DINO.txt returned invalid text features: {result.shape}")
        return result


class LocalDinoV3SpatialEncoder:
    """Frozen local-only DINOv3 encoder retaining a compact spatial token grid."""

    def __init__(
        self,
        *,
        repo_path: str | Path,
        weights_path: str | Path,
        model_name: str = "dinov3_vitl16",
        input_size: int = 256,
        spatial_grid_size: int = 4,
        device: str = "auto",
    ) -> None:
        self.repo_path = _require_local_directory(repo_path, "DINOv3 repository")
        self.weights_path = _require_local_file(weights_path, "DINOv3 weights")
        if input_size <= 0 or spatial_grid_size <= 0:
            raise ValueError("input_size and spatial_grid_size must be positive")
        self.input_size = int(input_size)
        self.spatial_grid_size = int(spatial_grid_size)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for DINO extraction but is unavailable")

        # Import only the backbone module.  DINOv3's broad hubconf imports optional evaluation
        # dependencies and its file:// weights path copies multi-GB checkpoints into $HOME.
        sys.path.insert(0, str(self.repo_path))
        try:
            from dinov3.hub import backbones

            constructor = getattr(backbones, str(model_name))
            self.model = constructor(pretrained=False)
        finally:
            if sys.path[0] == str(self.repo_path):
                sys.path.pop(0)
        state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def encode(self, rgb_frames: np.ndarray) -> np.ndarray:
        frames = np.asarray(rgb_frames)
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise ValueError(f"Expected uint8 RGB [B,H,W,3], got {frames.shape} {frames.dtype}")
        images = torch.from_numpy(frames.copy()).to(self.device)
        images = images.permute(0, 3, 1, 2).float().div_(255.0)
        images = functional.interpolate(
            images,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        images = (images - self.mean) / self.std
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            output = self.model.forward_features(images)
            if not isinstance(output, dict) or "x_norm_patchtokens" not in output:
                raise FeatureShardError(
                    "DINOv3 forward_features did not return x_norm_patchtokens"
                )
            patches = output["x_norm_patchtokens"]
            patch_side = int(round(patches.shape[1] ** 0.5))
            if patch_side * patch_side != patches.shape[1]:
                raise FeatureShardError(
                    f"DINOv3 patch count is not square: {patches.shape[1]}"
                )
            spatial = patches.reshape(
                patches.shape[0], patch_side, patch_side, patches.shape[2]
            ).permute(0, 3, 1, 2)
            spatial = functional.adaptive_avg_pool2d(
                spatial.float(), (self.spatial_grid_size, self.spatial_grid_size)
            )
            spatial = spatial.flatten(2).transpose(1, 2)
        result = spatial.cpu().numpy()
        if result.ndim != 3 or not np.isfinite(result).all():
            raise FeatureShardError("DINOv3 returned invalid spatial features")
        return result


class VideoResolver:
    """Resolve VITRA ``video_name`` by exact stem within a dataset-specific root."""

    def __init__(self, video_roots: dict[str, str | Path]) -> None:
        if not video_roots:
            raise ValueError("At least one dataset video root is required")
        self.roots = {
            str(dataset): _require_local_directory(root, f"video root for {dataset}")
            for dataset, root in video_roots.items()
        }
        self._indices: dict[str, dict[str, Path]] = {}

    def _build_index(self, dataset_name: str) -> dict[str, Path]:
        if dataset_name not in self.roots:
            raise FeatureShardError(f"No video root configured for dataset {dataset_name!r}")
        owners: dict[str, Path] = {}
        for path in self.roots[dataset_name].rglob("*"):
            if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            previous = owners.setdefault(path.stem, path)
            if previous != path:
                raise FeatureShardError(
                    f"Duplicate video stem {path.stem!r}: {previous} and {path}"
                )
        if not owners:
            raise FeatureShardError(f"No MP4/WebM files found under {self.roots[dataset_name]}")
        return owners

    def resolve(self, dataset_name: str, video_name: str) -> Path:
        index = self._indices.setdefault(dataset_name, self._build_index(dataset_name))
        try:
            return index[video_name]
        except KeyError as error:
            raise FileNotFoundError(
                f"No unique video stem {video_name!r} under {self.roots[dataset_name]}"
            ) from error


def decode_video_frames(
    path: Path,
    frame_ids: Sequence[int],
    frame_times_seconds: Sequence[float],
    *,
    seek_margin_seconds: float = 2.0,
    reseek_gap_seconds: float = 2.0,
    output_size: tuple[int, int] | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Seek by validated PTS and decode only short spans around requested frames.

    Inter-frame codecs require decoding from a preceding keyframe.  Cached presentation times
    let us seek near each sparse run while still matching every returned RGB frame exactly.
    """
    import av

    requested = np.asarray(frame_ids, dtype=np.int64)
    requested_times = np.asarray(frame_times_seconds, dtype=np.float64)
    if requested.ndim != 1 or len(requested) == 0:
        raise ValueError("frame_ids must be a non-empty one-dimensional sequence")
    if np.any(requested < 0) or np.any(np.diff(requested) <= 0):
        raise ValueError("frame_ids must be non-negative and strictly increasing")
    if (
        requested_times.shape != requested.shape
        or not np.isfinite(requested_times).all()
        or np.any(np.diff(requested_times) <= 0)
    ):
        raise ValueError("frame_times_seconds must be finite, aligned, and increasing")
    if seek_margin_seconds <= 0 or reseek_gap_seconds <= 0:
        raise ValueError("seek margins must be positive")
    if output_size is not None and (output_size[0] <= 0 or output_size[1] <= 0):
        raise ValueError("output_size must contain positive (height, width)")

    cursor = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        requested_threads = int(os.environ.get("EGO_HAND_WM_DECODE_THREADS", "0"))
        if requested_threads > 0:
            stream.thread_type = "AUTO"
            stream.codec_context.thread_count = requested_threads
        if stream.time_base is None:
            raise FeatureShardError(f"Video stream has no time base: {path}")
        stream_tick = float(stream.time_base)
        while cursor < len(requested):
            seek_seconds = max(float(requested_times[cursor]) - seek_margin_seconds, 0.0)
            seek_pts = max(int(seek_seconds / stream_tick), 0)
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
            made_progress = False
            # Some upstream videos contain isolated corrupt packets even though their MP4
            # container and timestamp table are intact (EPIC P30_08 is one example).  PyAV's
            # high-level decoder raises for those packets.  Continue to the next decodable
            # packet, while retaining the strict PTS comparison below so a corrupt *requested*
            # frame still fails loudly instead of silently shifting the frame index.
            stop_span = False
            for packet in container.demux(stream):
                try:
                    decoded_frames = packet.decode()
                except (av.error.InvalidDataError, av.error.EOFError):
                    # Packet-level decoding is important here: unlike the high-level
                    # container.decode() iterator, skipping this packet advances demux state and
                    # cannot spin forever on the same corrupt packet.
                    continue
                for frame in decoded_frames:
                    if frame.pts is None:
                        raise FeatureShardError(f"Decoded frame lacks PTS in {path}")
                    time_base = frame.time_base or stream.time_base
                    current_time = float(frame.pts * time_base)
                    target_time = float(requested_times[cursor])
                    local_step = (
                        float(requested_times[cursor + 1] - target_time)
                        if cursor + 1 < len(requested)
                        else 1.0 / float(stream.average_rate or 30.0)
                    )
                    tolerance = max(2.0 * stream_tick, min(local_step * 0.2, 1e-3))
                    if current_time < target_time - tolerance:
                        continue
                    if current_time > target_time + tolerance:
                        raise FeatureShardError(
                            f"PTS seek skipped frame {int(requested[cursor])} in {path}: "
                            f"target={target_time:.9f}, decoded={current_time:.9f}"
                        )
                    if output_size is not None:
                        height, width = output_size
                        frame = frame.reformat(width=width, height=height, format="rgb24")
                    yield int(requested[cursor]), frame.to_ndarray(format="rgb24")
                    made_progress = True
                    cursor += 1
                    if cursor == len(requested):
                        stop_span = True
                        break
                    if float(requested_times[cursor]) - current_time > reseek_gap_seconds:
                        stop_span = True
                        break
                if stop_span:
                    break
            if not made_progress:
                raise FeatureShardError(
                    f"Video {path} ended before requested frame {int(requested[cursor])}"
                )
    if cursor != len(requested):
        raise FeatureShardError(
            f"Video {path} ended before requested frame {int(requested[cursor])}"
        )


@dataclass(frozen=True)
class AnnotationWork:
    member_name: str
    payload: bytes
    digest: str
    dataset_name: str
    video_name: str
    frame_ids: np.ndarray
    frame_times_seconds: np.ndarray


def _load_annotation_work(annotation_shard: Path, pts_root: Path) -> list[AnnotationWork]:
    pts_cache: dict[tuple[str, str], np.ndarray] = {}
    work: list[AnnotationWork] = []
    with tarfile.open(annotation_shard, "r:*") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".npy"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise FeatureShardError(f"Could not read annotation member {member.name}")
            payload = extracted.read()
            try:
                episode = np.load(io.BytesIO(payload), allow_pickle=True).item()
            except (OSError, TypeError, ValueError) as error:
                raise FeatureShardError(f"Invalid annotation member {member.name}") from error
            if not isinstance(episode, dict):
                raise FeatureShardError(f"Annotation is not a dictionary: {member.name}")
            parts = Path(member.name).parts
            if not parts:
                raise FeatureShardError(f"Cannot derive dataset namespace from {member.name}")
            dataset_name = parts[0]
            video_name = str(episode.get("video_name", ""))
            frame_ids = np.asarray(episode.get("video_decode_frame"), dtype=np.int64)
            if (
                not video_name
                or frame_ids.ndim != 1
                or len(frame_ids) == 0
                or np.any(frame_ids < 0)
                or np.any(np.diff(frame_ids) <= 0)
            ):
                raise FeatureShardError(f"Invalid video/frame identity in {member.name}")
            cache_key = (dataset_name, video_name)
            if cache_key not in pts_cache:
                pts_path = pts_root / dataset_name / f"{video_name}.npy"
                if not pts_path.is_file():
                    raise FileNotFoundError(f"Required physical PTS cache is missing: {pts_path}")
                video_times = np.load(pts_path, allow_pickle=False, mmap_mode="r")
                if (
                    video_times.ndim != 1
                    or len(video_times) == 0
                    or not np.isfinite(video_times).all()
                    or np.any(np.diff(video_times.astype(np.float64)) <= 0)
                ):
                    raise FeatureShardError(f"Invalid physical PTS cache: {pts_path}")
                pts_cache[cache_key] = video_times
            video_times = pts_cache[cache_key]
            if frame_ids[-1] >= len(video_times):
                raise FeatureShardError(
                    f"Frame {frame_ids[-1]} exceeds PTS cache for {member.name}"
                )
            work.append(
                AnnotationWork(
                    member_name=member.name,
                    payload=payload,
                    digest=sha256_bytes(payload),
                    dataset_name=dataset_name,
                    video_name=video_name,
                    frame_ids=frame_ids,
                    frame_times_seconds=np.asarray(video_times[frame_ids], dtype=np.float64),
                )
            )
    if not work:
        raise FeatureShardError(f"Annotation shard has no .npy episodes: {annotation_shard}")
    return work


def _encode_requested_frames(
    video_path: Path,
    frame_ids: np.ndarray,
    *,
    frame_times_seconds: np.ndarray,
    encoder: SpatialFeatureEncoder,
    frame_reader: FrameReader,
    batch_size: int,
) -> dict[int, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    result: dict[int, np.ndarray] = {}
    batch_ids: list[int] = []
    batch_frames: list[np.ndarray] = []
    feature_shape: tuple[int, int] | None = None

    def flush() -> None:
        nonlocal feature_shape
        if not batch_ids:
            return
        encoded = np.asarray(encoder.encode(np.stack(batch_frames, axis=0)))
        if (
            encoded.ndim != 3
            or encoded.shape[0] != len(batch_ids)
            or encoded.shape[1] <= 0
            or encoded.shape[2] <= 0
        ):
            raise FeatureShardError(
                f"Encoder must return [B,P,D]; got {encoded.shape} for B={len(batch_ids)}"
            )
        if not np.issubdtype(encoded.dtype, np.floating) or not np.isfinite(encoded).all():
            raise FeatureShardError("Encoder returned non-finite or non-floating features")
        current_shape = (int(encoded.shape[1]), int(encoded.shape[2]))
        if feature_shape is None:
            feature_shape = current_shape
        elif feature_shape != current_shape:
            raise FeatureShardError("Encoder feature grid changed between batches")
        for frame_id, feature in zip(batch_ids, encoded, strict=True):
            result[frame_id] = np.asarray(feature, dtype=np.float16)
        batch_ids.clear()
        batch_frames.clear()

    for frame_id, frame in frame_reader(
        video_path, frame_ids.tolist(), frame_times_seconds.tolist()
    ):
        expected = int(frame_ids[len(result) + len(batch_ids)])
        if frame_id != expected:
            raise FeatureShardError(
                f"Frame reader order mismatch for {video_path}: expected {expected}, got {frame_id}"
            )
        rgb = np.asarray(frame)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise FeatureShardError(
                f"Frame reader must emit uint8 RGB [H,W,3], got {rgb.shape} {rgb.dtype}"
            )
        batch_ids.append(frame_id)
        batch_frames.append(rgb)
        if len(batch_ids) >= batch_size:
            flush()
    flush()
    missing = [int(frame_id) for frame_id in frame_ids if int(frame_id) not in result]
    if missing:
        raise FeatureShardError(f"Frame reader did not emit requested IDs: {missing[:10]}")
    return result


def _source_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def _existing_shard_is_valid(
    annotation_shard: Path,
    feature_path: Path,
    manifest_path: Path,
    expected_extractor_id: str,
) -> dict[str, Any] | None:
    if not feature_path.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = load_feature_manifest(manifest_path)
        source = manifest["annotation_shard"]
        stat = annotation_shard.stat()
        if (
            source["name"] != annotation_shard.name
            or int(source["size_bytes"]) != stat.st_size
            or int(source["mtime_ns"]) != stat.st_mtime_ns
            or manifest["extractor_id"] != expected_extractor_id
            or int(manifest["feature_size_bytes"]) != feature_path.stat().st_size
        ):
            return None
    except (FeatureShardError, KeyError, OSError, TypeError, ValueError):
        return None
    return manifest


def extract_feature_shard(
    *,
    annotation_shard: str | Path,
    output_root: str | Path,
    pts_root: str | Path,
    video_resolver: VideoResolver,
    encoder: SpatialFeatureEncoder,
    extractor_metadata: dict[str, Any],
    frame_reader: FrameReader = decode_video_frames,
    batch_size: int = 32,
    force: bool = False,
) -> dict[str, Any]:
    """Extract one atomic feature tar and publish one complete/incomplete manifest."""
    annotation_path = _require_local_file(annotation_shard, "annotation shard")
    pts_path = _require_local_directory(pts_root, "PTS root")
    root = Path(output_root)
    feature_path, manifest_path = feature_shard_paths(root, annotation_path)
    current_extractor_id = extractor_id(extractor_metadata)
    if not force:
        existing = _existing_shard_is_valid(
            annotation_path, feature_path, manifest_path, current_extractor_id
        )
        if existing is not None:
            return {**existing, "status": "validated_skip"}

    root.mkdir(parents=True, exist_ok=True)
    # Any extraction attempt invalidates a previous global completion claim.
    (root / "_SUCCESS").unlink(missing_ok=True)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = feature_path.with_name(
        f".{feature_path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial"
    )
    started = utc_now()
    base_manifest: dict[str, Any] = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "contract": FEATURE_CONTRACT,
        "complete": False,
        "status": "failed",
        "started_at_utc": started,
        "annotation_shard": _source_fingerprint(annotation_path),
        "feature_shard": str(feature_path),
        "extractor_id": current_extractor_id,
        "extractor": extractor_metadata,
        "role_capabilities": list(ROLE_CAPABILITIES),
    }
    try:
        work = _load_annotation_work(annotation_path, pts_path)
        groups: dict[tuple[str, str], dict[int, float]] = {}
        for episode in work:
            group = groups.setdefault((episode.dataset_name, episode.video_name), {})
            for frame_id, frame_time in zip(
                episode.frame_ids, episode.frame_times_seconds, strict=True
            ):
                frame_id = int(frame_id)
                previous = group.setdefault(frame_id, float(frame_time))
                if not np.isclose(previous, frame_time, rtol=0.0, atol=1e-9):
                    raise FeatureShardError(
                        f"Conflicting PTS for {episode.dataset_name}/{episode.video_name} "
                        f"frame {frame_id}"
                    )
        feature_maps: dict[tuple[str, str], dict[int, np.ndarray]] = {}
        for (dataset_name, video_name), id_to_time in groups.items():
            requested = np.asarray(sorted(id_to_time), dtype=np.int64)
            requested_times = np.asarray(
                [id_to_time[int(frame_id)] for frame_id in requested], dtype=np.float64
            )
            video_path = video_resolver.resolve(dataset_name, video_name)
            feature_maps[(dataset_name, video_name)] = _encode_requested_frames(
                video_path,
                requested,
                frame_times_seconds=requested_times,
                encoder=encoder,
                frame_reader=frame_reader,
                batch_size=batch_size,
            )

        import hashlib

        digest = hashlib.sha256()
        feature_dim: int | None = None
        spatial_tokens: int | None = None
        with tarfile.open(temporary, "w") as writer:
            for episode in work:
                mapping = feature_maps[(episode.dataset_name, episode.video_name)]
                pooled = np.stack([mapping[int(frame_id)] for frame_id in episode.frame_ids])
                if spatial_tokens is None:
                    spatial_tokens = int(pooled.shape[1])
                    feature_dim = int(pooled.shape[2])
                elif spatial_tokens != pooled.shape[1] or feature_dim != pooled.shape[2]:
                    raise FeatureShardError("Feature dimension changed across source videos")
                record = EpisodeFeatureRecord(
                    annotation_member=episode.member_name,
                    annotation_sha256=episode.digest,
                    dataset_name=episode.dataset_name,
                    video_name=episode.video_name,
                    frame_ids=episode.frame_ids,
                    frame_times_seconds=episode.frame_times_seconds,
                    pooled_features=pooled,
                    valid_mask=np.ones(len(episode.frame_ids), dtype=np.bool_),
                    extractor_id=current_extractor_id,
                )
                payload = encode_feature_record(record)
                member_name = annotation_member_to_feature_member(episode.member_name)
                tar_info = tarfile.TarInfo(member_name)
                tar_info.size = len(payload)
                tar_info.mode = 0o644
                tar_info.mtime = 0
                writer.addfile(tar_info, io.BytesIO(payload))
                digest.update(member_name.encode("utf-8"))
                digest.update(sha256_bytes(payload).encode("ascii"))
        if feature_dim is None or spatial_tokens is None:
            raise FeatureShardError("No feature records were written")
        os.replace(temporary, feature_path)
        manifest = {
            **base_manifest,
            "complete": True,
            "status": "rebuilt",
            "finished_at_utc": utc_now(),
            "episode_count": len(work),
            "source_video_count": len(groups),
            "unique_decoded_frames": sum(len(ids) for ids in groups.values()),
            "feature_dim": feature_dim,
            "spatial_tokens": spatial_tokens,
            "storage_dtype": "float16",
            "feature_size_bytes": feature_path.stat().st_size,
            "logical_content_sha256": digest.hexdigest(),
        }
        atomic_write_json(manifest_path, manifest)
        return manifest
    except Exception as error:
        failure = {
            **base_manifest,
            "finished_at_utc": utc_now(),
            "error_type": type(error).__name__,
            "message": str(error),
        }
        atomic_write_json(manifest_path, failure)
        raise
    finally:
        temporary.unlink(missing_ok=True)
