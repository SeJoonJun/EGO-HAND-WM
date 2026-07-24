#!/usr/bin/env python3
"""Extract compact frozen V-JEPA 2.1 ViT-G/384 tokens for e4 anticipation clips."""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

from ego_hand_wm.anticipation.dataset import HISTORY_STEPS, oracle_feature_cache_path
from ego_hand_wm.anticipation.protocol import AnticipationRecord, read_e4_anticipation_csv
from ego_hand_wm.anticipation.vjepa_features import (
    VJEPA_RESOLUTION,
    extract_compact_vjepa_tokens,
    load_frozen_vjepa2_1_vitg,
)
from ego_hand_wm.data.adapters.assembly101 import ANNOTATION_FPS, RAW_FPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-root", required=True)
    parser.add_argument("--recordings-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vjepa-repository", required=True)
    parser.add_argument("--index", type=int, default=None, help="One combined split/recording job")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args()


def extraction_jobs(annotations_root: Path) -> list[tuple[str, str, list[AnticipationRecord]]]:
    grouped: dict[tuple[str, str], list[AnticipationRecord]] = defaultdict(list)
    for split in ("train", "validation"):
        records = read_e4_anticipation_csv(annotations_root / f"{split}.csv")
        for record in records:
            grouped[(split, record.recording)].append(record)
    return [
        (split, recording, sorted(records, key=lambda item: item.anchor_frame))
        for (split, recording), records in sorted(grouped.items())
    ]


def clip_raw_indices(record: AnticipationRecord) -> np.ndarray:
    relative = np.arange(-HISTORY_STEPS + 1, 1, dtype=np.float64) / 8.0
    anchor_raw = record.anchor_frame * (RAW_FPS // ANNOTATION_FPS)
    return np.maximum(anchor_raw + np.rint(relative * RAW_FPS).astype(np.int64), 0)


def _center_crop_384(array: np.ndarray) -> np.ndarray:
    image = Image.fromarray(array, mode="RGB")
    width, height = image.size
    short = int(VJEPA_RESOLUTION * 256 / 224)
    scale = short / min(width, height)
    resized = image.resize(
        (int(round(width * scale)), int(round(height * scale))), Image.Resampling.BILINEAR
    )
    width, height = resized.size
    left = (width - VJEPA_RESOLUTION) // 2
    top = (height - VJEPA_RESOLUTION) // 2
    return np.asarray(
        resized.crop((left, top, left + VJEPA_RESOLUTION, top + VJEPA_RESOLUTION)),
        dtype=np.uint8,
    )


def decode_selected_frames(video_path: Path, indices: set[int]) -> dict[int, np.ndarray]:
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not indices:
        return {}
    selected: dict[int, np.ndarray] = {}
    final = max(indices)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index in indices:
                selected[frame_index] = _center_crop_384(frame.to_ndarray(format="rgb24"))
            if frame_index >= final:
                break
    missing = indices.difference(selected)
    if missing:
        raise IndexError(
            f"Video {video_path} ended before requested frames; first missing={min(missing)}"
        )
    return selected


def video_tensor(clips: list[np.ndarray], device: torch.device) -> torch.Tensor:
    array = np.stack(clips, axis=0)
    tensor = torch.from_numpy(array).to(device=device, dtype=torch.float32) / 255.0
    tensor = tensor.permute(0, 4, 1, 2, 3).contiguous()
    mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
    std = tensor.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
    return (tensor - mean) / std


def atomic_save(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, value, allow_pickle=False)
    os.replace(temporary, path)


def select_jobs(
    jobs: list[tuple[str, str, list[AnticipationRecord]]],
    *,
    index: int | None,
    shard_index: int | None,
    num_shards: int | None,
) -> list[tuple[str, str, list[AnticipationRecord]]]:
    if index is not None and (shard_index is not None or num_shards is not None):
        raise ValueError("--index cannot be combined with sharding")
    if index is not None:
        if not 0 <= index < len(jobs):
            raise IndexError(f"--index must be in [0,{len(jobs) - 1}]")
        return [jobs[index]]
    if shard_index is None or num_shards is None:
        raise ValueError("Provide --index or both --shard-index and --num-shards")
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("Shards require num_shards>0 and 0<=shard_index<num_shards")
    return jobs[shard_index::num_shards]


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    jobs = extraction_jobs(Path(args.annotations_root))
    if args.list_only:
        for index, (split, recording, records) in enumerate(jobs):
            print(f"{index}\t{split}\t{recording}\t{len(records)}")
        return
    selected_jobs = select_jobs(
        jobs,
        index=args.index,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    output_root = Path(args.output_root)
    pending_jobs = [
        (
            split,
            recording,
            [
                record
                for record in records
                if args.overwrite
                or not oracle_feature_cache_path(output_root, split, record).is_file()
            ],
        )
        for split, recording, records in selected_jobs
    ]
    pending_jobs = [job for job in pending_jobs if job[2]]
    if not pending_jobs:
        print("complete\tselected_shard")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("V-JEPA ViT-G extraction requires CUDA")
    device = torch.device("cuda")
    load_started = time.perf_counter()
    encoder = load_frozen_vjepa2_1_vitg(
        checkpoint=args.checkpoint, repository=args.vjepa_repository
    ).to(device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started
    torch.cuda.reset_peak_memory_stats(device)
    total_decode_seconds = 0.0
    total_forward_seconds = 0.0
    total_save_seconds = 0.0
    total_segments = 0
    for split, recording, pending in pending_jobs:
        video_stems = {record.video_stem for record in pending}
        if len(video_stems) != 1:
            raise ValueError(f"Recording {recording} has multiple e4 streams: {video_stems}")
        video_path = (
            Path(args.recordings_root)
            / recording
            / f"{next(iter(video_stems))}.mp4"
        )
        indices_by_id = {
            record.segment_id: clip_raw_indices(record) for record in pending
        }
        wanted = set(np.concatenate(list(indices_by_id.values())).tolist())
        decode_started = time.perf_counter()
        decoded = decode_selected_frames(video_path, wanted)
        decode_seconds = time.perf_counter() - decode_started
        forward_seconds = 0.0
        save_seconds = 0.0
        for start in range(0, len(pending), args.batch_size):
            batch_records = pending[start : start + args.batch_size]
            clips = [
                np.stack(
                    [decoded[int(index)] for index in indices_by_id[record.segment_id]]
                )
                for record in batch_records
            ]
            torch.cuda.synchronize(device)
            forward_started = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                tokens = extract_compact_vjepa_tokens(
                    encoder, video_tensor(clips, device)
                )
            tokens = tokens.to(dtype=torch.float16).cpu().numpy()
            torch.cuda.synchronize(device)
            forward_seconds += time.perf_counter() - forward_started
            save_started = time.perf_counter()
            for record, value in zip(batch_records, tokens, strict=True):
                atomic_save(oracle_feature_cache_path(output_root, split, record), value)
            save_seconds += time.perf_counter() - save_started
        total_decode_seconds += decode_seconds
        total_forward_seconds += forward_seconds
        total_save_seconds += save_seconds
        total_segments += len(pending)
        print(
            f"wrote\t{split}\t{recording}\tsegments={len(pending)}"
            f"\tdecode_s={decode_seconds:.3f}\tforward_s={forward_seconds:.3f}"
            f"\tsave_s={save_seconds:.3f}",
            flush=True,
        )
    peak_allocated_gib = torch.cuda.max_memory_allocated(device) / 2**30
    peak_reserved_gib = torch.cuda.max_memory_reserved(device) / 2**30
    print(
        f"summary\tsegments={total_segments}\tbatch_size={args.batch_size}"
        f"\tmodel_load_s={load_seconds:.3f}\tdecode_s={total_decode_seconds:.3f}"
        f"\tforward_s={total_forward_seconds:.3f}\tsave_s={total_save_seconds:.3f}"
        f"\tpeak_allocated_gib={peak_allocated_gib:.3f}"
        f"\tpeak_reserved_gib={peak_reserved_gib:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
