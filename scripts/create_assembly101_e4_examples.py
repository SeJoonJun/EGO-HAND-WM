#!/usr/bin/env python3
"""Create annotated Assembly101-e4 anticipation clips from official segment IDs."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import av
from PIL import ImageDraw, ImageFont

from ego_hand_wm.data.adapters.assembly101 import ANNOTATION_FPS, RAW_FPS, is_e4_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--recordings-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--segment-id", type=int, action="append", required=True)
    parser.add_argument("--observed-seconds", type=float, default=1.0)
    parser.add_argument("--execution-seconds", type=float, default=1.0)
    return parser.parse_args()


def integer(row: dict[str, str], first: str, second: str) -> int:
    value = row.get(first, row.get(second))
    if value is None:
        raise KeyError(f"Annotation lacks both {first!r} and {second!r}")
    return int(value)


def read_rows(path: Path) -> tuple[dict[int, dict[str, object]], dict[str, list[dict[str, object]]]]:
    by_id: dict[int, dict[str, object]] = {}
    by_recording: dict[str, list[dict[str, object]]] = defaultdict(list)
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            video = Path(raw["video"])
            if not is_e4_video(video):
                continue
            row: dict[str, object] = {
                "id": int(raw["id"]),
                "recording": video.parent.name,
                "video_stem": video.stem,
                "start": integer(raw, "start", "start_frame"),
                "end": integer(raw, "end", "end_frame"),
                "action": raw.get("action_cls", f"action {raw.get('action', raw.get('action_id'))}"),
            }
            by_id[int(row["id"])] = row
            by_recording[str(row["recording"])].append(row)
    for rows in by_recording.values():
        rows.sort(key=lambda item: (int(item["start"]), int(item["end"])))
    return by_id, by_recording


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def active_labels(rows: list[dict[str, object]], annotation_frame: float) -> list[str]:
    return [
        str(row["action"])
        for row in rows
        if int(row["start"]) <= annotation_frame < int(row["end"])
    ]


def fitting_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    maximum_width: int,
    bold: bool,
    start_size: int,
    minimum_size: int = 12,
) -> ImageFont.FreeTypeFont:
    path = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    for size in range(start_size, minimum_size - 1, -1):
        font = ImageFont.truetype(path, size)
        if draw.textbbox((0, 0), text, font=font)[2] <= maximum_width:
            return font
    return ImageFont.truetype(path, minimum_size)


def create_clip(
    target: dict[str, object],
    rows: list[dict[str, object]],
    recordings_root: Path,
    output_dir: Path,
    observed_seconds: float,
    execution_seconds: float,
) -> Path:
    start = int(target["start"])
    end = int(target["end"])
    anchor = start - ANNOTATION_FPS
    clip_start = max(0, anchor - int(round(observed_seconds * ANNOTATION_FPS)))
    clip_end = min(end, start + int(round(execution_seconds * ANNOTATION_FPS)))
    raw_ratio = RAW_FPS // ANNOTATION_FPS
    raw_start = clip_start * raw_ratio
    raw_anchor = anchor * raw_ratio
    raw_action = start * raw_ratio
    raw_end = clip_end * raw_ratio
    source = (
        recordings_root
        / str(target["recording"])
        / f"{target['video_stem']}.mp4"
    )
    if not source.is_file():
        raise FileNotFoundError(source)

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        f"segment_{int(target['id']):05d}_{safe_name(str(target['action']))}_e4.mp4"
    )
    source_container = av.open(str(source))
    source_stream = source_container.streams.video[0]
    output_container = av.open(str(output), mode="w")
    encoder = output_container.add_stream("libx264", rate=RAW_FPS)
    encoder.width = source_stream.width
    encoder.height = source_stream.height
    encoder.pix_fmt = "yuv420p"
    encoder.options = {"crf": "20", "preset": "medium"}
    regular = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17
    )
    written = 0
    for raw_index, frame in enumerate(source_container.decode(source_stream)):
        if raw_index < raw_start:
            continue
        if raw_index > raw_end:
            break
        image = frame.to_image().convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        if raw_index <= raw_anchor:
            phase, color = "OBSERVED HISTORY", (45, 145, 245, 235)
        elif raw_index < raw_action:
            phase, color = "UNSEEN 1-SECOND GAP", (245, 165, 30, 235)
        else:
            phase = f"TARGET: {str(target['action']).upper()}"
            color = (35, 185, 85, 235)
        relative_seconds = (raw_index - raw_anchor) / RAW_FPS
        annotation_frame = raw_index / raw_ratio
        labels = active_labels(rows, annotation_frame)
        active = " + ".join(labels[:2]) if labels else "unlabelled transition"
        active_text = f"Official active label: {active}"
        phase_font = fitting_font(
            draw,
            phase,
            maximum_width=image.width - 26,
            bold=True,
            start_size=21,
        )
        active_font = fitting_font(
            draw,
            active_text,
            maximum_width=image.width - 26,
            bold=False,
            start_size=17,
        )
        draw.rectangle((0, 0, image.width, 61), fill=(8, 8, 8, 190))
        draw.text((13, 7), phase, font=phase_font, fill=color)
        draw.text(
            (13, 35),
            f"t = {relative_seconds:+.2f} s from last observation",
            font=regular,
            fill=(255, 255, 255, 240),
        )
        draw.rectangle(
            (0, image.height - 35, image.width, image.height), fill=(8, 8, 8, 185)
        )
        draw.text(
            (13, image.height - 29),
            active_text,
            font=active_font,
            fill=(255, 255, 255, 240),
        )
        output_frame = av.VideoFrame.from_image(image)
        output_frame.pts = written
        output_frame.time_base = Fraction(1, RAW_FPS)
        for packet in encoder.encode(output_frame):
            output_container.mux(packet)
        written += 1
    for packet in encoder.encode():
        output_container.mux(packet)
    output_container.close()
    source_container.close()
    if written == 0:
        raise RuntimeError(f"No frames written for segment {target['id']}")
    print(
        f"wrote\t{output}\tframes={written}\tduration={written / RAW_FPS:.3f}s"
        f"\ttarget={target['action']}"
    )
    return output


def main() -> None:
    args = parse_args()
    if args.observed_seconds <= 0 or args.execution_seconds <= 0:
        raise ValueError("Observed and execution durations must be positive")
    by_id, by_recording = read_rows(Path(args.annotations))
    for segment_id in args.segment_id:
        if segment_id not in by_id:
            raise KeyError(f"No e4 annotation for segment {segment_id}")
        target = by_id[segment_id]
        create_clip(
            target,
            by_recording[str(target["recording"])],
            Path(args.recordings_root),
            Path(args.output_dir),
            args.observed_seconds,
            args.execution_seconds,
        )


if __name__ == "__main__":
    main()
