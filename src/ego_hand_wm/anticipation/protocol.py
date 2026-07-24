"""Official Assembly101 one-second anticipation sampling utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from ego_hand_wm.data.adapters.assembly101 import ANNOTATION_FPS, is_e4_video


ACTION_CLASSES = 1064
VERB_CLASSES = 17
OBJECT_CLASSES = 90
ANTICIPATION_SECONDS = 1.0
SPANNING_SECONDS = 6.0
RECENT_SECONDS = (1.6, 1.2, 0.8, 0.4)
SPANNING_BINS = (5, 3, 2)
RECENT_BINS = 2
CONTEXT_FRAMES = int(SPANNING_SECONDS * ANNOTATION_FPS) + 1


@dataclass(frozen=True)
class AnticipationRecord:
    segment_id: int
    video: str
    recording: str
    video_stem: str
    start_frame: int
    end_frame: int
    action: int | None
    verb: int | None
    object: int | None
    toy_id: str
    shared: bool

    @property
    def anchor_frame(self) -> int:
        """Final observation frame, exactly one second before the target action."""

        return self.start_frame - ANNOTATION_FPS


def _first(row: Mapping[str, str], *names: str, default: str | None = None) -> str | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return default


def _optional_int(row: Mapping[str, str], *names: str) -> int | None:
    value = _first(row, *names)
    return None if value is None else int(value)


def parse_anticipation_row(row: Mapping[str, str]) -> AnticipationRecord:
    """Normalize the released train/validation/test CSV column-name variants."""

    video = str(_first(row, "video"))
    if not video:
        raise ValueError("Assembly101 anticipation row has no video")
    path = Path(video)
    start = _optional_int(row, "start_frame", "start")
    end = _optional_int(row, "end_frame", "end")
    # The released anticipation CSV contains a small number of zero-duration target segments.
    # TempAgg still uses their start frame as a valid anticipation boundary, so retain them.
    if start is None or end is None or start < 0 or end < start:
        raise ValueError(f"Invalid Assembly101 action interval: {start}, {end}")
    return AnticipationRecord(
        segment_id=int(str(_first(row, "id"))),
        video=video,
        recording=path.parent.name,
        video_stem=path.stem,
        start_frame=start,
        end_frame=end,
        action=_optional_int(row, "action_id", "action"),
        verb=_optional_int(row, "verb_id", "verb"),
        object=_optional_int(row, "noun_id", "noun", "object_id", "object"),
        toy_id=str(_first(row, "toyid", "toy_id", default="")),
        shared=bool(int(str(_first(row, "is_shared", "shared", default="0")))),
    )


def read_e4_anticipation_csv(
    path: str | Path,
    *,
    require_labels: bool = True,
    require_official_history: bool = True,
) -> list[AnticipationRecord]:
    """Read one official split, retaining exactly one e4 stream per action segment."""

    csv_path = Path(path)
    with csv_path.open(newline="") as handle:
        records = [
            parse_anticipation_row(row)
            for row in csv.DictReader(handle)
            if is_e4_video(str(row["video"]))
        ]
    if require_official_history:
        # This is the same two-second guard used by the released TempAgg SequenceDataset.  The
        # six-second spanning branch clips to frame zero for early sequences.
        records = [record for record in records if record.anchor_frame >= 2 * ANNOTATION_FPS]
    if require_labels and any(
        record.action is None or record.verb is None or record.object is None for record in records
    ):
        raise ValueError(f"Split has no semantic labels: {csv_path}")
    if not records:
        raise ValueError(f"No usable Assembly101 e4 segments in {csv_path}")
    return records


def context_frame_indices(anchor_frame: int) -> np.ndarray:
    """Return a fixed six-second logical 30 fps context ending at ``anchor_frame``.

    TempAgg clips early spanning history to frame zero.  Repeating index zero here produces a
    batchable equivalent and preserves the endpoint of every official temporal bin.
    """

    if anchor_frame < 0:
        raise ValueError("anchor_frame must be non-negative")
    offsets = np.arange(-CONTEXT_FRAMES + 1, 1, dtype=np.int64)
    return np.maximum(anchor_frame + offsets, 0)


def temporal_bin_ranges(
    *,
    fps: int = ANNOTATION_FPS,
    spanning_seconds: float = SPANNING_SECONDS,
    spanning_bins: Iterable[int] = SPANNING_BINS,
    recent_seconds: Iterable[float] = RECENT_SECONDS,
    recent_bins: int = RECENT_BINS,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[tuple[int, int], ...], ...]]:
    """Return inclusive TempAgg bin ranges over the fixed context tensor.

    The released implementation uses ``np.linspace(..., dtype=int)`` and includes both ends of
    each interval, so adjacent bins share their boundary frame.  We retain that detail.
    """

    context_length = int(round(spanning_seconds * fps)) + 1

    def ranges(start: int, end: int, bins: int) -> tuple[tuple[int, int], ...]:
        if bins <= 0:
            raise ValueError("Temporal bin counts must be positive")
        boundaries = np.linspace(start, end, bins + 1, dtype=np.int64)
        return tuple((int(left), int(right)) for left, right in zip(boundaries[:-1], boundaries[1:]))

    spanning = tuple(ranges(0, context_length - 1, bins) for bins in spanning_bins)
    recent = tuple(
        ranges(context_length - 1 - int(round(seconds * fps)), context_length - 1, recent_bins)
        for seconds in recent_seconds
    )
    return spanning, recent


def load_tail_segment_ids(path: str | Path) -> set[int]:
    return {int(line) for line in Path(path).read_text().splitlines() if line.strip()}


def load_unseen_recordings(path: str | Path) -> set[str]:
    recordings: set[str] = set()
    for line in Path(path).read_text().splitlines():
        fields = line.split("\t")
        if len(fields) >= 2 and fields[1].strip().lower() == "notshared":
            recordings.add(fields[0].strip())
    return recordings
