#!/usr/bin/env python3
"""Render VITRA training/validation curves without a matplotlib dependency."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "loss": "#2563eb",
    "flow": "#dc2626",
    "visual": "#16a34a",
    "rotation": "#9333ea",
    "ego": "#ea580c",
    "cosine_error": "#0891b2",
    "camera_translation_cm": "#2563eb",
    "wrist_translation_cm": "#16a34a",
    "camera_rotation_deg": "#dc2626",
    "wrist_rotation_deg": "#9333ea",
    "mano_rotation_deg": "#ea580c",
}


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size=size) if path.exists() else ImageFont.load_default()


def load_training(path: Path) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith('{"step":'):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "lr" in record and "loss" in record:
                records.append(record)
    return records


def load_jsonl(path: Path) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values.copy()
    total = np.cumsum(np.insert(values, 0, 0.0))
    core = (total[width:] - total[:-width]) / width
    return np.concatenate((np.full(width - 1, np.nan), core))


def format_tick(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value / 1000:.0f}k"
    if abs(value) >= 10:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.1f}"
    return f"{value:.3f}" if value else "0"


def panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    title: str,
    series: list[tuple[str, np.ndarray, np.ndarray]],
    y_log: bool = False,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    stop_step: int | None = None,
) -> None:
    left, top, right, bottom = box
    pad_l, pad_r, pad_t, pad_b = 84, 24, 52, 58
    x0, y0, x1, y1 = left + pad_l, top + pad_t, right - pad_r, bottom - pad_b
    draw.rectangle(box, fill="#ffffff", outline="#d1d5db", width=2)
    draw.text((left + 18, top + 13), title, fill="#111827", font=font(22, bold=True))

    all_x = np.concatenate([x for _, x, _ in series])
    selected: list[tuple[str, np.ndarray, np.ndarray]] = []
    xlo = float(np.min(all_x)) if x_min is None else x_min
    xhi = float(np.max(all_x)) if x_max is None else x_max
    for name, xs, ys in series:
        mask = np.isfinite(ys) & (xs >= xlo) & (xs <= xhi)
        if y_log:
            mask &= ys > 0
        selected.append((name, xs[mask], ys[mask]))
    all_y = np.concatenate([ys for _, _, ys in selected if len(ys)])
    transformed = np.log10(all_y) if y_log else all_y
    ylo_t = float(np.min(transformed)) if y_min is None else (math.log10(y_min) if y_log else y_min)
    yhi_t = float(np.max(transformed)) if y_max is None else (math.log10(y_max) if y_log else y_max)
    if yhi_t <= ylo_t:
        yhi_t = ylo_t + 1.0
    margin = 0.04 * (yhi_t - ylo_t)
    ylo_t -= margin
    yhi_t += margin

    def xp(value: float) -> float:
        return x0 + (value - xlo) / max(xhi - xlo, 1e-12) * (x1 - x0)

    def yp(value: float) -> float:
        transformed_value = math.log10(value) if y_log else value
        return y1 - (transformed_value - ylo_t) / (yhi_t - ylo_t) * (y1 - y0)

    for i in range(6):
        fraction = i / 5
        x = x0 + fraction * (x1 - x0)
        value = xlo + fraction * (xhi - xlo)
        draw.line((x, y0, x, y1), fill="#e5e7eb", width=1)
        label = format_tick(value)
        draw.text((x - 16, y1 + 12), label, fill="#4b5563", font=font(14))
    for i in range(6):
        fraction = i / 5
        y = y1 - fraction * (y1 - y0)
        transformed_value = ylo_t + fraction * (yhi_t - ylo_t)
        value = 10**transformed_value if y_log else transformed_value
        draw.line((x0, y, x1, y), fill="#e5e7eb", width=1)
        draw.text((left + 8, y - 8), format_tick(value), fill="#4b5563", font=font(14))
    draw.line((x0, y1, x1, y1), fill="#6b7280", width=2)
    draw.line((x0, y0, x0, y1), fill="#6b7280", width=2)
    draw.text(((x0 + x1) / 2 - 34, bottom - 34), "training step", fill="#374151", font=font(15))
    if y_log:
        draw.text((left + 8, top + 36), "log scale", fill="#6b7280", font=font(12))

    legend_x = x0 + 12
    for name, xs, ys in selected:
        color = COLORS.get(name, "#111827")
        if len(xs) > 1:
            stride = max(1, len(xs) // 1800)
            points = [(xp(float(x)), yp(float(y))) for x, y in zip(xs[::stride], ys[::stride])]
            draw.line(points, fill=color, width=4, joint="curve")
        draw.line((legend_x, top + 28, legend_x + 28, top + 28), fill=color, width=4)
        draw.text((legend_x + 34, top + 17), name.replace("_", " "), fill="#111827", font=font(14))
        legend_x += 42 + int(draw.textlength(name.replace("_", " "), font=font(14)))
    if stop_step is not None and xlo <= stop_step <= xhi:
        x = xp(stop_step)
        draw.line((x, y0, x, y1), fill="#111827", width=2)
        draw.text((x - 104, y0 + 7), f"abrupt stop: {stop_step:,}", fill="#111827", font=font(13, bold=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--validation-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = load_training(args.train_log)
    validation = load_jsonl(args.validation_log)
    if not train or not validation:
        raise SystemExit("Training or validation records are empty")
    steps = np.asarray([row["step"] for row in train], dtype=np.float64)
    smooth = 100  # 100 logged records = 1,000 optimizer steps.
    smoothed = {
        key: moving_average(np.asarray([row[key] for row in train], dtype=np.float64), smooth)
        for key in ("loss", "flow", "visual", "rotation", "ego")
    }
    val_steps = np.asarray([row["step"] for row in validation], dtype=np.float64)
    macro = lambda key: np.asarray(
        [row[f"validation/macro/{key}"] for row in validation], dtype=np.float64
    )
    stop_step = int(steps[-1])

    image = Image.new("RGB", (1900, 1320), "#f3f4f6")
    draw = ImageDraw.Draw(image)
    draw.text((42, 18), "EGO-HAND-WM · VITRA pretraining diagnostics", fill="#111827", font=font(30, bold=True))
    draw.text(
        (42, 58),
        f"Job 8482436 · {len(train):,} train records · smoothing = 1,000 steps · stopped after step {stop_step:,}",
        fill="#4b5563",
        font=font(17),
    )

    panel(
        draw,
        (34, 92, 940, 680),
        title="Training loss — full run",
        series=[(key, steps, smoothed[key]) for key in ("loss", "flow", "visual")],
        y_log=True,
        stop_step=stop_step,
    )
    panel(
        draw,
        (960, 92, 1866, 680),
        title="Training components — late-stage detail",
        series=[(key, steps, smoothed[key]) for key in ("flow", "rotation", "ego", "visual")],
        x_min=20_000,
        y_min=0.0,
        y_max=1.6,
        stop_step=stop_step,
    )
    panel(
        draw,
        (34, 704, 940, 1288),
        title="Validation macro losses",
        series=[
            ("flow", val_steps, macro("geometry_flow_mse")),
            ("visual", val_steps, macro("future_visual_flow_mse")),
            ("cosine_error", val_steps, 1.0 - macro("future_visual_cosine")),
        ],
        y_log=True,
        stop_step=stop_step,
    )
    panel(
        draw,
        (960, 704, 1866, 1288),
        title="Validation macro physical errors",
        series=[
            ("camera_translation_cm", val_steps, macro("camera_translation_cm")),
            ("wrist_translation_cm", val_steps, macro("wrist_translation_cm")),
            ("camera_rotation_deg", val_steps, macro("camera_rotation_deg")),
            ("wrist_rotation_deg", val_steps, macro("wrist_rotation_deg")),
            ("mano_rotation_deg", val_steps, macro("mano_rotation_deg")),
        ],
        stop_step=stop_step,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, optimize=True)

    last = train[-1]
    latest_val = validation[-1]
    summary = {
        "train_records": len(train),
        "first_step": int(steps[0]),
        "last_step": stop_step,
        "first_loss": train[0]["loss"],
        "last_loss": last["loss"],
        "last_smoothed_loss": float(smoothed["loss"][-1]),
        "last_smoothed_flow": float(smoothed["flow"][-1]),
        "last_smoothed_visual": float(smoothed["visual"][-1]),
        "last_validation_step": int(latest_val["step"]),
        "last_validation_macro": {
            key: latest_val[f"validation/macro/{key}"]
            for key in (
                "geometry_flow_mse",
                "future_visual_flow_mse",
                "future_visual_cosine",
                "camera_translation_cm",
                "camera_rotation_deg",
                "wrist_translation_cm",
                "wrist_rotation_deg",
                "mano_rotation_deg",
            )
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
