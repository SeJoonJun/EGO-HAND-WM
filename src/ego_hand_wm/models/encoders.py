"""No-download smoke encoders and local-only production encoder wrappers."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ego_hand_wm.contracts.batch import CanonicalBatch
from ego_hand_wm.contracts.schema import GEOMETRY_DIM, STREAM_NAMES
from ego_hand_wm.models.time_embedding import TimeMLP


class TinyVisionEncoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, hidden_dim // 4, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = images.shape
        encoded = self.network(images.reshape(batch * time, channels, height, width))
        return encoded.flatten(1).reshape(batch, time, -1)


class DinoV3VisionEncoder(nn.Module):
    """Frozen DINOv3 wrapper that refuses implicit network downloads."""

    def __init__(self, config: dict[str, Any], hidden_dim: int) -> None:
        super().__init__()
        backend = config.get("backend", "huggingface")
        self.backend = backend
        if backend == "huggingface":
            from transformers import AutoModel

            model_name = str(config["model_name"])
            self.backbone = AutoModel.from_pretrained(model_name, local_files_only=True)
        elif backend == "hub_local":
            repo_path = Path(config["repo_path"])
            weights_path = Path(config["weights_path"])
            if not repo_path.is_dir() or not weights_path.is_file():
                raise FileNotFoundError(
                    f"Local DINOv3 repo/weights missing: {repo_path}, {weights_path}"
                )
            self.backbone = torch.hub.load(
                str(repo_path),
                str(config.get("model_name", "dinov3_vitl16")),
                source="local",
                weights=str(weights_path),
            )
        else:
            raise ValueError(f"Unknown DINOv3 backend: {backend}")

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        backbone_dim = int(
            config.get("output_dim")
            or getattr(getattr(self.backbone, "config", None), "hidden_size", 0)
            or getattr(self.backbone, "embed_dim", 0)
        )
        if backbone_dim <= 0:
            raise ValueError("Set vision.output_dim; DINOv3 hidden width could not be inferred")
        self.projection = nn.Linear(backbone_dim, hidden_dim)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )

    def train(self, mode: bool = True) -> "DinoV3VisionEncoder":
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def _extract(self, images: torch.Tensor) -> torch.Tensor:
        normalized = (images - self.mean) / self.std
        if self.backend == "hub_local":
            output = self.backbone.forward_features(normalized)
            if isinstance(output, dict):
                patches = output.get("x_norm_patchtokens")
                if patches is None:
                    raise KeyError("DINOv3 forward_features lacks x_norm_patchtokens")
            else:
                patches = output
        else:
            output = self.backbone(normalized)
            patches = output.last_hidden_state
            register_count = int(getattr(self.backbone.config, "num_register_tokens", 4))
            patches = patches[:, 1 + register_count :]
        return patches.mean(dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = images.shape
        pooled = self._extract(images.reshape(batch * time, channels, height, width))
        return self.projection(pooled).reshape(batch, time, -1)


class HashTextEncoder(nn.Module):
    """Stable lightweight encoder for tests; not the production language model."""

    def __init__(self, hidden_dim: int, vocabulary_size: int = 8192, max_tokens: int = 24) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.max_tokens = max_tokens
        self.embedding = nn.Embedding(vocabulary_size, hidden_dim, padding_idx=0)

    def _token_id(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") % (self.vocabulary_size - 1) + 1

    def forward(self, texts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.zeros(len(texts), self.max_tokens, dtype=torch.long, device=device)
        valid = torch.zeros(len(texts), self.max_tokens, dtype=torch.bool, device=device)
        for row, text in enumerate(texts):
            words = re.findall(r"[A-Za-z0-9']+", text.lower())[: self.max_tokens]
            if not words:
                words = ["<empty>"]
            ids = [self._token_id(word) for word in words]
            tokens[row, : len(ids)] = torch.tensor(ids, device=device)
            valid[row, : len(ids)] = True
        return self.embedding(tokens), valid


class HFTextEncoder(nn.Module):
    def __init__(self, config: dict[str, Any], hidden_dim: int) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        model_name = str(config["model_name"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        loaded = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.backbone = loaded.get_encoder() if loaded.config.is_encoder_decoder else loaded
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        source_dim = int(
            getattr(self.backbone.config, "d_model", 0)
            or self.backbone.config.hidden_size
        )
        self.projection = nn.Linear(source_dim, hidden_dim)
        self.max_tokens = int(config.get("max_tokens", 32))

    def train(self, mode: bool = True) -> "HFTextEncoder":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, texts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            hidden = self.backbone(**tokenized).last_hidden_state
        return self.projection(hidden), tokenized.attention_mask.bool()


class ContextEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        hidden_dim = int(config["hidden_dim"])
        vision_config = config.get("vision", {"kind": "tiny"})
        vision_kind = vision_config.get("kind", "tiny")
        if vision_kind == "tiny":
            self.vision = TinyVisionEncoder(hidden_dim)
            self.precomputed_visual_projection = None
        elif vision_kind == "dinov3":
            self.vision = DinoV3VisionEncoder(vision_config, hidden_dim)
            self.precomputed_visual_projection = None
        elif vision_kind == "precomputed":
            feature_dim = int(vision_config["precomputed_dim"])
            if feature_dim <= 0:
                raise ValueError("vision.precomputed_dim must be positive")
            self.vision = None
            self.precomputed_visual_projection = nn.Linear(feature_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown vision encoder: {vision_kind}")

        text_config = config.get("text", {"kind": "hash"})
        text_kind = text_config.get("kind", "hash")
        if text_kind == "hash":
            self.text = HashTextEncoder(hidden_dim, max_tokens=int(text_config.get("max_tokens", 24)))
        elif text_kind == "hf":
            self.text = HFTextEncoder(text_config, hidden_dim)
        else:
            raise ValueError(f"Unknown text encoder: {text_kind}")

        self.state_projection = nn.Linear(GEOMETRY_DIM + len(STREAM_NAMES), hidden_dim)
        self.fingertip_projection = (
            nn.Linear(2 * 5 * 3 + 2, hidden_dim)
            if bool(config.get("use_fingertips", False))
            else None
        )
        self.intrinsics_projection = nn.Linear(4, hidden_dim)
        self.physical_time = TimeMLP(hidden_dim, max_period=float(config.get("physical_max_period", 10.0)))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=int(config["heads"]),
            dim_feedforward=int(hidden_dim * float(config.get("mlp_ratio", 4.0))),
            dropout=float(config.get("dropout", 0.0)),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=int(config.get("context_depth", 2)))

    def forward(self, batch: CanonicalBatch) -> tuple[torch.Tensor, torch.Tensor]:
        device = batch.history_state.device
        history_time_embedding = self.physical_time(batch.history_time)
        if batch.context_visual_features is not None:
            if self.precomputed_visual_projection is None:
                raise ValueError(
                    "Received precomputed visual features, but vision.kind is not precomputed"
                )
            visual = self.precomputed_visual_projection(batch.context_visual_features)
        elif batch.context_images is not None:
            if self.vision is None:
                raise ValueError("Received RGB images, but vision.kind=precomputed")
            visual = self.vision(batch.context_images)
        else:
            visual = torch.zeros(
                batch.batch_size,
                batch.history_state.shape[1],
                self.state_projection.out_features,
                device=device,
            )
        visual = visual + history_time_embedding
        visual_valid = batch.history_query_mask & (
            torch.ones_like(batch.history_query_mask)
            if batch.context_visual_features is not None or batch.context_images is not None
            else torch.zeros_like(batch.history_query_mask)
        )

        text, text_valid = self.text(batch.text, device)
        from ego_hand_wm.contracts.schema import SCHEMA

        masked_history = batch.history_state * SCHEMA.expand_stream_mask(
            batch.history_stream_mask
        ).to(batch.history_state.dtype)
        state_input = torch.cat((masked_history, batch.history_stream_mask.float()), dim=-1)
        state = self.state_projection(state_input) + history_time_embedding
        state_valid = batch.history_stream_mask.any(dim=-1) & batch.history_query_mask
        fingertip = None
        fingertip_valid = None
        if batch.history_fingertips is not None and self.fingertip_projection is not None:
            hand_valid = torch.stack(
                (batch.history_stream_mask[..., 1], batch.history_stream_mask[..., 2]), dim=-1
            )
            masked_fingertips = batch.history_fingertips * hand_valid[
                ..., None, None
            ].to(batch.history_fingertips.dtype)
            fingertip_input = torch.cat(
                (
                    masked_fingertips.reshape(*batch.history_time.shape, 30),
                    hand_valid.float(),
                ),
                dim=-1,
            )
            fingertip = self.fingertip_projection(fingertip_input) + self.physical_time(
                batch.history_time
            )
            fingertip_valid = hand_valid.any(dim=-1) & batch.history_query_mask
        intrinsics = self.intrinsics_projection(batch.intrinsics).unsqueeze(1)
        intrinsics_valid = torch.ones(batch.batch_size, 1, dtype=torch.bool, device=device)

        token_parts = [visual, text, state]
        valid_parts = [visual_valid, text_valid, state_valid]
        if fingertip is not None and fingertip_valid is not None:
            token_parts.append(fingertip)
            valid_parts.append(fingertip_valid)
        token_parts.append(intrinsics)
        valid_parts.append(intrinsics_valid)
        tokens = torch.cat(token_parts, dim=1)
        valid = torch.cat(valid_parts, dim=1)
        tokens = self.fusion(tokens, src_key_padding_mask=~valid)
        return tokens, valid
