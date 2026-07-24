"""A model-independent temporal contract for trajectory anticipation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FixedTrajectoryProtocol:
    """Fixed-rate history/future sampling shared by all adapted baselines.

    Time zero is the last observed frame.  Therefore a six-frame history at
    30 Hz spans 5/30 seconds and the sixteenth future target lies 16/30
    seconds after the anchor.
    """

    history_steps: int = 6
    future_steps: int = 16
    fps: float = 30.0

    def __post_init__(self) -> None:
        if self.history_steps < 1:
            raise ValueError("history_steps must be positive")
        if self.future_steps < 1:
            raise ValueError("future_steps must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")

    @property
    def total_steps(self) -> int:
        return self.history_steps + self.future_steps

    @property
    def observation_ratio(self) -> float:
        return self.history_steps / self.total_steps

    @property
    def history_span_seconds(self) -> float:
        return (self.history_steps - 1) / self.fps

    @property
    def final_horizon_seconds(self) -> float:
        return self.future_steps / self.fps

    @property
    def history_relative_times(self) -> tuple[float, ...]:
        return tuple(
            (index - self.history_steps + 1) / self.fps
            for index in range(self.history_steps)
        )

    @property
    def future_relative_times(self) -> tuple[float, ...]:
        return tuple(index / self.fps for index in range(1, self.future_steps + 1))

    def window_starts(
        self,
        length: int,
        *,
        stride: int | None = None,
        include_tail: bool = True,
    ) -> tuple[int, ...]:
        """Return starts for exact-length windows without padding.

        The default stride equals the prediction length.  This limits nearly
        duplicate targets while still covering long clips.  ``include_tail``
        adds a final end-aligned window when the stride misses the clip tail.
        """

        if length < 0:
            raise ValueError("length cannot be negative")
        stride = self.future_steps if stride is None else stride
        if stride < 1:
            raise ValueError("stride must be positive")
        if length < self.total_steps:
            return ()

        last_start = length - self.total_steps
        starts = list(range(0, last_start + 1, stride))
        if include_tail and starts[-1] != last_start:
            starts.append(last_start)
        return tuple(starts)

    def frame_indices(self, start: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if start < 0:
            raise ValueError("start cannot be negative")
        anchor = start + self.history_steps - 1
        history = tuple(range(start, anchor + 1))
        future = tuple(range(anchor + 1, anchor + self.future_steps + 1))
        return history, future

    def manifest_fields(self, start: int) -> dict[str, object]:
        history, future = self.frame_indices(start)
        return {
            "fps": self.fps,
            "history_steps": self.history_steps,
            "future_steps": self.future_steps,
            "anchor_index": history[-1],
            "history_indices": list(history),
            "future_indices": list(future),
            "history_time_seconds": list(self.history_relative_times),
            "future_time_seconds": list(self.future_relative_times),
        }
