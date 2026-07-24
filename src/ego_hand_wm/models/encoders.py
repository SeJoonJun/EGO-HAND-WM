"""No-download smoke encoders and local-only production encoder wrappers."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn

from ego_hand_wm.contracts.batch import CanonicalBatch
from ego_hand_wm.contracts.schema import ENTITY_DIMS, ENTITY_NAMES, SCHEMA
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
        return encoded.flatten(1).reshape(batch, time, 1, -1)


class DinoV3VisionEncoder(nn.Module):
    """Frozen DINOv3 wrapper that refuses implicit network downloads."""

    def __init__(self, config: dict[str, Any], hidden_dim: int) -> None:
        super().__init__()
        backend = config.get("backend", "huggingface")
        self.backend = backend
        self.spatial_grid_size = int(config.get("spatial_grid_size", 4))
        if self.spatial_grid_size <= 0:
            raise ValueError("vision.spatial_grid_size must be positive")
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
            sys.path.insert(0, str(repo_path.resolve()))
            try:
                from dinov3.hub import backbones

                constructor = getattr(
                    backbones, str(config.get("model_name", "dinov3_vitl16"))
                )
                self.backbone = constructor(pretrained=False)
            finally:
                if sys.path[0] == str(repo_path.resolve()):
                    sys.path.pop(0)
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(state, strict=True)
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
        patch_side = int(round(patches.shape[1] ** 0.5))
        if patch_side * patch_side != patches.shape[1]:
            raise ValueError(f"DINOv3 patch count is not square: {patches.shape[1]}")
        spatial = patches.reshape(
            patches.shape[0], patch_side, patch_side, patches.shape[2]
        ).permute(0, 3, 1, 2)
        spatial = torch.nn.functional.adaptive_avg_pool2d(
            spatial.float(), (self.spatial_grid_size, self.spatial_grid_size)
        )
        return spatial.flatten(2).transpose(1, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = images.shape
        spatial = self._extract(images.reshape(batch * time, channels, height, width))
        projected = self.projection(spatial)
        return projected.reshape(batch, time, spatial.shape[1], -1)


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
        self.hidden_dim = hidden_dim
        vision_config = config.get("vision", {"kind": "tiny"})
        vision_kind = vision_config.get("kind", "tiny")
        if vision_kind == "tiny":
            self.vision = TinyVisionEncoder(hidden_dim)
            self.precomputed_visual_projection = None
            self.dinotxt_global_projection = None
            self.dinotxt_spatial_projection = None
            self.visual_spatial_tokens = 1
        elif vision_kind == "dinov3":
            self.vision = DinoV3VisionEncoder(vision_config, hidden_dim)
            self.precomputed_visual_projection = None
            self.dinotxt_global_projection = None
            self.dinotxt_spatial_projection = None
            grid_size = int(vision_config.get("spatial_grid_size", 4))
            self.visual_spatial_tokens = grid_size**2
        elif vision_kind == "precomputed":
            feature_dim = int(vision_config["precomputed_dim"])
            if feature_dim <= 0:
                raise ValueError("vision.precomputed_dim must be positive")
            self.vision = None
            self.precomputed_visual_projection = nn.Linear(feature_dim, hidden_dim)
            self.dinotxt_global_projection = None
            self.dinotxt_spatial_projection = None
            self.visual_spatial_tokens = int(vision_config.get("spatial_tokens", 1))
            if self.visual_spatial_tokens <= 0:
                raise ValueError("vision.spatial_tokens must be positive")
        elif vision_kind == "precomputed_dinotxt":
            feature_dim = int(vision_config.get("feature_dim", 1024))
            spatial_tokens = int(vision_config.get("spatial_tokens", 16))
            if feature_dim <= 0 or spatial_tokens <= 0:
                raise ValueError("DINO.txt feature_dim and spatial_tokens must be positive")
            self.vision = None
            self.precomputed_visual_projection = None
            self.dinotxt_global_projection = nn.Linear(feature_dim * 2, hidden_dim)
            self.dinotxt_spatial_projection = nn.Linear(feature_dim, hidden_dim)
            self.visual_spatial_tokens = 1 + spatial_tokens
        else:
            raise ValueError(f"Unknown vision encoder: {vision_kind}")

        text_config = config.get("text", {"kind": "hash"})
        text_kind = text_config.get("kind", "hash")
        if text_kind == "hash":
            self.text = HashTextEncoder(hidden_dim, max_tokens=int(text_config.get("max_tokens", 24)))
            self.precomputed_text_projection = None
        elif text_kind == "hf":
            self.text = HFTextEncoder(text_config, hidden_dim)
            self.precomputed_text_projection = None
        elif text_kind == "precomputed_dinotxt":
            feature_dim = int(text_config.get("feature_dim", 2048))
            if feature_dim <= 0:
                raise ValueError("text.feature_dim must be positive")
            self.text = None
            self.precomputed_text_projection = nn.Linear(feature_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown text encoder: {text_kind}")

        self.state_projections = nn.ModuleList(
            nn.Linear(width + 1, hidden_dim) for width in ENTITY_DIMS
        )
        self.state_entity_embedding = nn.Parameter(
            torch.zeros(1, 1, len(ENTITY_NAMES), hidden_dim)
        )
        nn.init.normal_(self.state_entity_embedding, std=0.02)
        self.fingertip_projection = (
            nn.Linear(2 * 5 * 3 + 2, hidden_dim)
            if bool(config.get("use_fingertips", False))
            else None
        )
        self.intrinsics_projection = nn.Linear(4, hidden_dim)
        self.visual_spatial_embedding = nn.Parameter(
            torch.zeros(1, 1, self.visual_spatial_tokens, hidden_dim)
        )
        nn.init.normal_(self.visual_spatial_embedding, std=0.02)
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
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, batch: CanonicalBatch) -> tuple[torch.Tensor, torch.Tensor]:
        device = batch.history_state.device
        history_time_embedding = self.physical_time(batch.history_time)
        if batch.context_visual_features is not None:
            if self.dinotxt_global_projection is not None:
                if batch.context_visual_features.ndim != 4:
                    raise ValueError("DINO.txt visual features must be [B,H,17,D]")
                class_token = batch.context_visual_features[:, :, 0]
                spatial_tokens = batch.context_visual_features[:, :, 1:]
                global_descriptor = functional.normalize(
                    torch.cat((class_token, spatial_tokens.mean(dim=2)), dim=-1), dim=-1
                )
                visual = torch.cat(
                    (
                        self.dinotxt_global_projection(global_descriptor).unsqueeze(2),
                        self.dinotxt_spatial_projection(spatial_tokens),
                    ),
                    dim=2,
                )
            elif self.precomputed_visual_projection is not None:
                visual = self.precomputed_visual_projection(batch.context_visual_features)
            else:
                raise ValueError(
                    "Received precomputed visual features, but vision.kind is not precomputed"
                )
        elif batch.context_images is not None:
            if self.vision is None:
                raise ValueError("Received RGB images, but vision.kind=precomputed")
            visual = self.vision(batch.context_images)
        else:
            visual = torch.zeros(
                batch.batch_size,
                batch.history_state.shape[1],
                self.visual_spatial_tokens,
                self.hidden_dim,
                device=device,
            )
        if visual.ndim != 4 or visual.shape[2] != self.visual_spatial_tokens:
            raise ValueError(
                "Visual features must be [B,H,P,D] with "
                f"P={self.visual_spatial_tokens}; got {tuple(visual.shape)}"
            )
        visual = (
            visual
            + history_time_embedding.unsqueeze(2)
            + self.visual_spatial_embedding.to(visual.dtype)
        )
        visual = visual.flatten(1, 2)
        visual_valid = batch.history_query_mask.unsqueeze(-1).expand(
            -1, -1, self.visual_spatial_tokens
        ) & (
            torch.ones(
                *batch.history_query_mask.shape,
                self.visual_spatial_tokens,
                dtype=torch.bool,
                device=device,
            )
            if batch.context_visual_features is not None or batch.context_images is not None
            else torch.zeros(
                *batch.history_query_mask.shape,
                self.visual_spatial_tokens,
                dtype=torch.bool,
                device=device,
            )
        )
        visual_valid = visual_valid.flatten(1, 2)

        if batch.context_text_features is not None:
            if self.precomputed_text_projection is None:
                raise ValueError(
                    "Received precomputed text features, but text.kind is not precomputed_dinotxt"
                )
            text = self.precomputed_text_projection(batch.context_text_features).unsqueeze(1)
            text_valid = (
                torch.ones(batch.batch_size, 1, dtype=torch.bool, device=device)
                if batch.context_text_mask is None
                else batch.context_text_mask[:, None]
            )
        else:
            if self.text is None:
                raise ValueError("text.kind=precomputed_dinotxt requires context_text_features")
            text, text_valid = self.text(batch.text, device)
            if batch.context_text_mask is not None:
                text_valid = text_valid & batch.context_text_mask[:, None]
        history_entities = SCHEMA.split_entities(batch.history_state)
        history_component_masks = SCHEMA.split_entities(
            batch.effective_history_state_mask.to(batch.history_state.dtype)
        )
        entity_valid = SCHEMA.expand_entity_mask(batch.history_stream_mask)
        state_parts = []
        for index, (entity, component_mask, projection) in enumerate(
            zip(
                history_entities,
                history_component_masks,
                self.state_projections,
                strict=True,
            )
        ):
            valid = entity_valid[..., index : index + 1]
            state_parts.append(
                projection(
                    torch.cat(
                        (
                            entity * component_mask * valid.to(entity.dtype),
                            valid.to(entity.dtype),
                        ),
                        dim=-1,
                    )
                )
            )
        state = torch.stack(state_parts, dim=2)
        state = (
            state
            + history_time_embedding.unsqueeze(2)
            + self.state_entity_embedding.to(state.dtype)
        )
        state = state.flatten(1, 2)
        state_valid = (
            entity_valid & batch.history_query_mask.unsqueeze(-1)
        ).flatten(1, 2)
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
        tokens = self.final_norm(tokens).masked_fill(~valid.unsqueeze(-1), 0.0)
        return tokens, valid
