"""Deterministic time source selection for live streams and local videos."""

from __future__ import annotations

import math
from typing import Optional


class PipelineClock:
    """Use monotonic wall time for live sources and robust PTS for files."""

    def __init__(self, fallback_fps: float = 30.0):
        self.fallback_fps = float(fallback_fps)
        if not math.isfinite(self.fallback_fps) or self.fallback_fps <= 0.0:
            raise ValueError("fallback_fps must be finite and greater than zero")
        self.reset()

    def reset(self) -> None:
        self.frame_index = 0
        self.last_source_frame_index: Optional[int] = None
        self.last_valid_pts: Optional[float] = None
        self.last_timestamp: Optional[float] = None

    def next_timestamp(
        self,
        source_kind: Optional[str],
        monotonic_time: float,
        pts_ms: Optional[float] = None,
        fps: Optional[float] = None,
        frame_index: Optional[int] = None,
    ) -> float:
        if source_kind != "file":
            return float(monotonic_time)

        if frame_index is None:
            current_frame_index = self.frame_index
            self.frame_index += 1
        else:
            current_frame_index = max(0, int(frame_index))
            self.frame_index = max(self.frame_index, current_frame_index + 1)
        source_fps = float(fps) if fps is not None else float("nan")
        if not math.isfinite(source_fps) or source_fps <= 0.0:
            source_fps = self.fallback_fps
        frame_duration = 1.0 / source_fps
        frame_time = current_frame_index * frame_duration

        pts_seconds = float(pts_ms) / 1000.0 if pts_ms is not None else float("nan")
        pts_advances = (
            math.isfinite(pts_seconds)
            and pts_seconds >= 0.0
            and (
                self.last_valid_pts is None
                or pts_seconds > self.last_valid_pts + 1e-9
            )
        )
        if pts_advances:
            self.last_valid_pts = pts_seconds
            candidate = pts_seconds
        else:
            candidate = frame_time

        if self.last_timestamp is not None and candidate <= self.last_timestamp:
            elapsed_frames = max(
                1,
                current_frame_index - (
                    self.last_source_frame_index
                    if self.last_source_frame_index is not None
                    else current_frame_index - 1
                ),
            )
            candidate = self.last_timestamp + elapsed_frames * frame_duration
        self.last_source_frame_index = current_frame_index
        self.last_timestamp = candidate
        return candidate
