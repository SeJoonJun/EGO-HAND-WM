"""TempAgg-style DINOv3/geometry model for Assembly101 one-second anticipation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as functional
from torch import nn

from ego_hand_wm.anticipation.geometry_encoder import (
    GEOMETRY_INPUT_DIMS,
    PastFutureGeometryEncoder,
    SemanticGeometryCrossAttention,
    assemble_geometry_sequence,
)
from ego_hand_wm.anticipation.protocol import CONTEXT_FRAMES, temporal_bin_ranges


AblationMode = Literal["rgb", "rgb_camera", "rgb_wrist", "rgb_camera_wrist"]
ABLATION_MODES = frozenset({"rgb", "rgb_camera", "rgb_wrist", "rgb_camera_wrist"})


@dataclass
class AnticipationOutput:
    action_logits: tuple[torch.Tensor, ...]
    verb_logits: tuple[torch.Tensor, ...]
    object_logits: tuple[torch.Tensor, ...]

    def ensemble(self) -> dict[str, torch.Tensor]:
        return {
            "action": torch.stack(self.action_logits).mean(dim=0),
            "verb": torch.stack(self.verb_logits).mean(dim=0),
            "object": torch.stack(self.object_logits).mean(dim=0),
        }


class FrameModalityFusion(nn.Module):
    """Fuse RGB, camera, and two confidence-masked wrist tokens at each frame."""

    def __init__(
        self,
        *,
        rgb_dim: int,
        hidden_dim: int,
        heads: int,
        mode: AblationMode,
        dropout: float,
    ) -> None:
        super().__init__()
        if mode not in ABLATION_MODES:
            raise ValueError(f"Unknown anticipation ablation mode: {mode}")
        self.mode = mode
        self.use_camera = mode in {"rgb_camera", "rgb_camera_wrist"}
        self.use_wrist = mode in {"rgb_wrist", "rgb_camera_wrist"}
        self.rgb_projection = nn.Sequential(nn.LayerNorm(rgb_dim), nn.Linear(rgb_dim, hidden_dim))
        self.camera_projection = nn.Sequential(
            nn.LayerNorm(9), nn.Linear(9, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        # pose9 + released confidence + binary validity
        self.wrist_projection = nn.Sequential(
            nn.LayerNorm(11), nn.Linear(11, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.modality_embedding = nn.Parameter(torch.zeros(4, hidden_dim))
        nn.init.normal_(self.modality_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        rgb_features: torch.Tensor,
        camera_pose: torch.Tensor,
        wrist_pose: torch.Tensor,
        wrist_confidence: torch.Tensor,
        wrist_valid: torch.Tensor,
    ) -> torch.Tensor:
        if rgb_features.ndim != 3:
            raise ValueError("rgb_features must have shape [B,T,D]")
        batch, time, _ = rgb_features.shape
        if camera_pose.shape != (batch, time, 9):
            raise ValueError("camera_pose must have shape [B,T,9]")
        if wrist_pose.shape != (batch, time, 2, 9):
            raise ValueError("wrist_pose must have shape [B,T,2,9]")
        if wrist_confidence.shape != (batch, time, 2) or wrist_valid.shape != (batch, time, 2):
            raise ValueError("Wrist confidence/validity must have shape [B,T,2]")

        token_groups = [self.rgb_projection(rgb_features).unsqueeze(2)]
        masks = [torch.zeros(batch, time, 1, dtype=torch.bool, device=rgb_features.device)]
        modality_ids = [0]
        if self.use_camera:
            token_groups.append(self.camera_projection(camera_pose).unsqueeze(2))
            masks.append(torch.zeros(batch, time, 1, dtype=torch.bool, device=rgb_features.device))
            modality_ids.append(1)
        if self.use_wrist:
            wrist_input = torch.cat(
                (
                    wrist_pose,
                    wrist_confidence.unsqueeze(-1),
                    wrist_valid.to(wrist_pose.dtype).unsqueeze(-1),
                ),
                dim=-1,
            )
            token_groups.append(self.wrist_projection(wrist_input))
            masks.append(~wrist_valid.bool())
            modality_ids.extend((2, 3))

        tokens = torch.cat(token_groups, dim=2)
        invalid = torch.cat(masks, dim=2)
        embeddings = self.modality_embedding[
            torch.tensor(modality_ids, dtype=torch.long, device=tokens.device)
        ]
        tokens = tokens + embeddings.view(1, 1, len(modality_ids), -1)
        fused = self.fusion(
            tokens.reshape(batch * time, len(modality_ids), -1),
            src_key_padding_mask=invalid.reshape(batch * time, len(modality_ids)),
        ).reshape(batch, time, len(modality_ids), -1)
        valid_weights = (~invalid).to(fused.dtype).unsqueeze(-1)
        pooled = (fused * valid_weights).sum(dim=2) / valid_weights.sum(dim=2).clamp_min(1.0)
        return self.output_norm(pooled)


def pool_temporal_bins(
    features: torch.Tensor, ranges: tuple[tuple[int, int], ...]
) -> torch.Tensor:
    """Max-pool inclusive temporal intervals, as in the released TempAgg feature loader."""

    if features.ndim != 3:
        raise ValueError("features must have shape [B,T,D]")
    pooled: list[torch.Tensor] = []
    for start, end in ranges:
        if start < 0 or end < start or end >= features.shape[1]:
            raise IndexError(f"Invalid temporal bin [{start},{end}] for T={features.shape[1]}")
        pooled.append(features[:, start : end + 1].amax(dim=1))
    return torch.stack(pooled, dim=1)


class CouplingBlock(nn.Module):
    """Modern cross-attention equivalent of TempAgg's recent-to-spanning non-local block."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.recent_norm = nn.LayerNorm(hidden_dim)
        self.spanning_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.recent_output = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context_output = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self, spanning: torch.Tensor, recent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, _ = self.cross_attention(
            self.recent_norm(recent),
            self.spanning_norm(spanning),
            self.spanning_norm(spanning),
            need_weights=False,
        )
        recent_summary = torch.cat((recent.mean(dim=1), attended.mean(dim=1)), dim=-1)
        context_summary = torch.cat((spanning.mean(dim=1), attended.mean(dim=1)), dim=-1)
        return self.recent_output(recent_summary), self.context_output(context_summary)


class TemporalAggregateBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.couplings = nn.ModuleList(
            CouplingBlock(hidden_dim, heads, dropout) for _ in range(3)
        )
        self.recent_fusion = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(
        self, spanning: tuple[torch.Tensor, ...], recent: torch.Tensor
    ) -> torch.Tensor:
        if len(spanning) != len(self.couplings):
            raise ValueError("TempAgg expects exactly three spanning granularities")
        outputs = [
            coupling(scale, recent) for coupling, scale in zip(self.couplings, spanning, strict=True)
        ]
        recent_fused = self.recent_fusion(torch.cat([pair[0] for pair in outputs], dim=-1))
        context_fused = torch.stack([pair[1] for pair in outputs], dim=0).amax(dim=0)
        return torch.cat((recent_fused, context_fused), dim=-1)


class TempAggGeometryModel(nn.Module):
    """Four-branch TempAgg head with a frozen-DINO feature interface and geometry tokens."""

    def __init__(
        self,
        *,
        rgb_dim: int = 1024,
        hidden_dim: int = 512,
        heads: int = 8,
        mode: AblationMode = "rgb_camera_wrist",
        action_classes: int = 1064,
        verb_classes: int = 17,
        object_classes: int = 90,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.mode = mode
        self.frame_fusion = FrameModalityFusion(
            rgb_dim=rgb_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            mode=mode,
            dropout=dropout,
        )
        self.temporal_position = nn.Parameter(torch.zeros(1, CONTEXT_FRAMES, hidden_dim))
        nn.init.normal_(self.temporal_position, std=0.02)
        self.aggregate_blocks = nn.ModuleList(
            TemporalAggregateBlock(hidden_dim, heads, dropout) for _ in range(4)
        )
        self.action_heads = nn.ModuleList(
            nn.Linear(hidden_dim * 2, action_classes) for _ in range(4)
        )
        self.verb_heads = nn.ModuleList(nn.Linear(hidden_dim * 2, verb_classes) for _ in range(4))
        self.object_heads = nn.ModuleList(
            nn.Linear(hidden_dim * 2, object_classes) for _ in range(4)
        )
        self.spanning_ranges, self.recent_ranges = temporal_bin_ranges()

    def forward(
        self,
        *,
        rgb_features: torch.Tensor,
        camera_pose: torch.Tensor,
        wrist_pose: torch.Tensor,
        wrist_confidence: torch.Tensor,
        wrist_valid: torch.Tensor,
    ) -> AnticipationOutput:
        if rgb_features.shape[1] != CONTEXT_FRAMES:
            raise ValueError(
                f"Expected {CONTEXT_FRAMES} context frames, got {rgb_features.shape[1]}"
            )
        frames = self.frame_fusion(
            rgb_features, camera_pose, wrist_pose, wrist_confidence, wrist_valid
        )
        frames = frames + self.temporal_position
        spanning = tuple(pool_temporal_bins(frames, ranges) for ranges in self.spanning_ranges)
        recent = tuple(pool_temporal_bins(frames, ranges) for ranges in self.recent_ranges)
        representations = tuple(
            block(spanning, recent_scale)
            for block, recent_scale in zip(self.aggregate_blocks, recent, strict=True)
        )
        return AnticipationOutput(
            action_logits=tuple(
                head(representation)
                for head, representation in zip(self.action_heads, representations, strict=True)
            ),
            verb_logits=tuple(
                head(representation)
                for head, representation in zip(self.verb_heads, representations, strict=True)
            ),
            object_logits=tuple(
                head(representation)
                for head, representation in zip(self.object_heads, representations, strict=True)
            ),
        )


def anticipation_loss(
    output: AnticipationOutput,
    labels: torch.Tensor,
    *,
    verb_weight: float = 1.0,
    object_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Official four-branch action/verb/object supervision with stable mean scaling."""

    if labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError("labels must be [B,3] ordered as verb, object, action")
    verb_loss = torch.stack(
        [functional.cross_entropy(logits, labels[:, 0]) for logits in output.verb_logits]
    ).mean()
    object_loss = torch.stack(
        [functional.cross_entropy(logits, labels[:, 1]) for logits in output.object_logits]
    ).mean()
    action_loss = torch.stack(
        [functional.cross_entropy(logits, labels[:, 2]) for logits in output.action_logits]
    ).mean()
    total = action_loss + float(verb_weight) * verb_loss + float(object_weight) * object_loss
    return total, {"action": action_loss, "verb": verb_loss, "object": object_loss}


OracleAblationMode = Literal[
    "rgb",
    "rgb_gt_camera",
    "rgb_gt_wrist",
    "rgb_gt_handpose",
    "rgb_gt_whole_hand",
    "rgb_gt_camera_wrist",
    "rgb_gt_camera_handpose",
    "rgb_gt_camera_whole_hand",
]
ORACLE_ABLATION_MODES = frozenset(
    {
        "rgb",
        "rgb_gt_camera",
        "rgb_gt_wrist",
        "rgb_gt_handpose",
        "rgb_gt_whole_hand",
        "rgb_gt_camera_wrist",
        "rgb_gt_camera_handpose",
        "rgb_gt_camera_whole_hand",
    }
)


@dataclass
class SemanticAnticipationOutput:
    verb_logits: torch.Tensor
    object_logits: torch.Tensor
    action_logits: torch.Tensor

    def scores(self) -> dict[str, torch.Tensor]:
        return {
            "verb": self.verb_logits,
            "object": self.object_logits,
            "action": self.action_logits,
        }


class QueryCrossAttentionBlock(nn.Module):
    """Attentive-probe block that decodes semantic queries from frozen visual tokens."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        update, _ = self.cross(
            self.query_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        queries = queries + update
        normalized = self.self_norm(queries)
        update, _ = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )
        queries = queries + update
        return queries + self.mlp(queries)


class OracleGeometryAnticipationModel(nn.Module):
    """Frozen-V-JEPA-token semantic decoder with optional GT oracle geometry."""

    def __init__(
        self,
        *,
        visual_dim: int = 1664,
        hidden_dim: int = 512,
        heads: int = 8,
        visual_depth: int = 2,
        geometry_depth: int = 2,
        mode: OracleAblationMode = "rgb",
        action_classes: int = 1064,
        verb_classes: int = 17,
        object_classes: int = 90,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if mode not in ORACLE_ABLATION_MODES:
            raise ValueError(f"Unknown oracle ablation mode: {mode}")
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.mode = mode
        self.geometry_mode = None if mode == "rgb" else mode.removeprefix("rgb_gt_")
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden_dim)
        )
        self.semantic_queries = nn.Parameter(torch.empty(1, 3, hidden_dim))
        nn.init.trunc_normal_(self.semantic_queries, std=0.02)
        self.visual_probe = nn.ModuleList(
            QueryCrossAttentionBlock(hidden_dim, heads, dropout)
            for _ in range(visual_depth)
        )
        # Construct every parameter shared by all ablations before any
        # mode-specific geometry modules. With the same seed this guarantees
        # identical RGB/query/classifier initialization across the eight runs;
        # otherwise the variable-width geometry encoder advances the RNG before
        # the classifier heads are initialized.
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.verb_head = nn.Linear(hidden_dim, verb_classes)
        self.object_head = nn.Linear(hidden_dim, object_classes)
        self.action_head = nn.Linear(hidden_dim, action_classes)
        if self.geometry_mode is None:
            self.geometry_encoder = None
            self.geometry_fusion = None
        else:
            self.geometry_encoder = PastFutureGeometryEncoder(
                GEOMETRY_INPUT_DIMS[self.geometry_mode],
                hidden_dim,
                hidden_dim=hidden_dim,
                depth=geometry_depth,
                num_heads=heads,
                dropout=dropout,
                max_frames=48,
            )
            self.geometry_fusion = SemanticGeometryCrossAttention(
                hidden_dim, num_heads=heads, dropout=dropout
            )

    def forward(
        self,
        *,
        visual_tokens: torch.Tensor,
        camera_pose: torch.Tensor,
        wrist_pose: torch.Tensor,
        hand_pose: torch.Tensor,
        wrist_valid: torch.Tensor,
        hand_pose_valid: torch.Tensor,
        geometry_time_mask: torch.Tensor,
        time_seconds: torch.Tensor,
        future_mask: torch.Tensor,
        execution_mask: torch.Tensor,
    ) -> SemanticAnticipationOutput:
        if visual_tokens.ndim != 3:
            raise ValueError("visual_tokens must have shape [B,N,D]")
        memory = self.visual_projection(visual_tokens.float())
        queries = self.semantic_queries.expand(visual_tokens.shape[0], -1, -1)
        for block in self.visual_probe:
            queries = block(queries, memory)

        if self.geometry_mode is not None:
            assert self.geometry_encoder is not None
            assert self.geometry_fusion is not None
            geometry, geometry_mask = assemble_geometry_sequence(
                self.geometry_mode,
                camera_pose=camera_pose,
                wrist_pose=wrist_pose,
                hand_pose=hand_pose,
                wrist_valid=wrist_valid,
                hand_pose_valid=hand_pose_valid,
                future_mask=future_mask,
                include_future=True,
            )
            geometry_mask = geometry_mask & geometry_time_mask.bool()
            geometry_tokens, geometry_mask = self.geometry_encoder.forward_tokens(
                geometry,
                geometry_mask,
                time_seconds,
                future_mask,
                execution_mask,
            )
            queries = self.geometry_fusion(queries, geometry_tokens, geometry_mask)

        queries = self.output_norm(queries)
        return SemanticAnticipationOutput(
            verb_logits=self.verb_head(queries[:, 0]),
            object_logits=self.object_head(queries[:, 1]),
            action_logits=self.action_head(queries[:, 2]),
        )


def focal_semantic_loss(
    output: SemanticAnticipationOutput,
    labels: torch.Tensor,
    *,
    gamma: float = 2.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Equal-weight focal losses for official verb, object, and action labels."""

    if labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError("labels must be [B,3] ordered as verb, object, action")

    def focal(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = functional.cross_entropy(logits, targets, reduction="none")
        return ((1.0 - torch.exp(-ce)) ** float(gamma) * ce).mean()

    parts = {
        "verb": focal(output.verb_logits, labels[:, 0]),
        "object": focal(output.object_logits, labels[:, 1]),
        "action": focal(output.action_logits, labels[:, 2]),
    }
    return sum(parts.values()), parts
