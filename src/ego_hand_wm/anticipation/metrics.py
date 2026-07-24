"""Metrics for the Assembly101 semantic anticipation protocol."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def class_mean_topk_recall(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    k: int = 5,
    sample_mask: torch.Tensor | None = None,
) -> float:
    """Mean per-class Top-k recall over target classes represented in the subset."""

    if logits.ndim != 2 or targets.shape != (logits.shape[0],):
        raise ValueError("Expected logits [N,C] and targets [N]")
    if sample_mask is not None:
        if sample_mask.shape != targets.shape:
            raise ValueError("sample_mask must align with targets")
        logits = logits[sample_mask.bool()]
        targets = targets[sample_mask.bool()]
    if targets.numel() == 0:
        return float("nan")
    if targets.min() < 0 or targets.max() >= logits.shape[1]:
        raise ValueError("Target lies outside the classifier range")
    topk = logits.topk(min(k, logits.shape[1]), dim=1).indices
    correct = (topk == targets[:, None]).any(dim=1).float()
    recalls = [correct[targets == class_id].mean() for class_id in targets.unique(sorted=True)]
    return float(torch.stack(recalls).mean().item())


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, *, k: int = 1) -> float:
    if logits.ndim != 2 or targets.shape != (logits.shape[0],):
        raise ValueError("Expected logits [N,C] and targets [N]")
    if targets.numel() == 0:
        return float("nan")
    topk = logits.topk(min(k, logits.shape[1]), dim=1).indices
    return float((topk == targets[:, None]).any(dim=1).float().mean().item())


def semantic_anticipation_metrics(
    scores: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    segment_ids: torch.Tensor | None = None,
    recordings: Iterable[str] | None = None,
    tail_segment_ids: set[int] | None = None,
    unseen_recordings: set[str] | None = None,
) -> dict[str, float]:
    """Compute official overall metrics and optional validation tail/unseen subsets."""

    if labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError("labels must be [N,3] ordered as verb, object, action")
    targets = {"verb": labels[:, 0], "object": labels[:, 1], "action": labels[:, 2]}
    result: dict[str, float] = {}
    for name in ("verb", "object", "action"):
        result[f"overall/{name}_top1_accuracy"] = topk_accuracy(
            scores[name], targets[name], k=1
        )
        result[f"overall/{name}_mean_top5_recall"] = class_mean_topk_recall(
            scores[name], targets[name], k=5
        )

    subsets: dict[str, torch.Tensor] = {}
    if tail_segment_ids is not None:
        if segment_ids is None:
            raise ValueError("segment_ids are required for tail evaluation")
        subsets["tail"] = torch.tensor(
            [int(value) in tail_segment_ids for value in segment_ids.tolist()], dtype=torch.bool
        )
    if unseen_recordings is not None:
        if recordings is None:
            raise ValueError("recordings are required for unseen evaluation")
        subsets["unseen"] = torch.tensor(
            [name in unseen_recordings for name in recordings], dtype=torch.bool
        )
    for subset_name, mask in subsets.items():
        for name in ("verb", "object", "action"):
            result[f"{subset_name}/{name}_mean_top5_recall"] = class_mean_topk_recall(
                scores[name], targets[name], k=5, sample_mask=mask
            )
    return result

