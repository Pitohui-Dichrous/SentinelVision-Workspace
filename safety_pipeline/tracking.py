"""Small deterministic one-to-one tracker for logical head observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from .config import TrackingConfig
from .matching import maximum_total_weight_matching
from .ppe_conflict import box_iou
from .types import Box, HeadObservation, TrackedHead, TrackerFrame


@dataclass
class _Track:
    track_id: int
    box: Box
    observation: HeadObservation
    first_seen: float
    last_seen: float
    age_frames: int = 1
    hits: int = 1
    lost_frames: int = 0


def _smooth(old_box: Box, new_box: Box, amount: float) -> Box:
    return tuple(
        amount * new_value + (1.0 - amount) * old_value
        for old_value, new_value in zip(old_box, new_box)
    )  # type: ignore[return-value]


class HeadTracker:
    """Track heads independently of their current helmet classification."""

    def __init__(self, config: TrackingConfig):
        self.config = config
        self._tracks: Dict[int, _Track] = {}
        self._next_track_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1

    @property
    def active_count(self) -> int:
        return len(self._tracks)

    def update(self, observations: Iterable[HeadObservation], timestamp: float) -> TrackerFrame:
        items = tuple(observations)
        for track in self._tracks.values():
            track.age_frames += 1
            track.lost_frames += 1

        candidate_weights: Dict[Tuple[int, int], float] = {}
        for track_id, track in self._tracks.items():
            for observation_index, observation in enumerate(items):
                overlap = box_iou(track.box, observation.box)
                if overlap + 1e-12 >= self.config.iou_threshold:
                    candidate_weights[(track_id, observation_index)] = overlap

        track_to_observation = maximum_total_weight_matching(
            sorted(self._tracks),
            range(len(items)),
            candidate_weights,
        )
        assignments = {
            observation_index: track_id
            for track_id, observation_index in track_to_observation.items()
        }

        visible: List[TrackedHead] = []
        for observation_index, observation in enumerate(items):
            track_id = assignments.get(observation_index)
            if track_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1
                track = _Track(
                    track_id=track_id,
                    box=observation.box,
                    observation=observation,
                    first_seen=timestamp,
                    last_seen=timestamp,
                )
                self._tracks[track_id] = track
            else:
                track = self._tracks[track_id]
                track.box = _smooth(track.box, observation.box, self.config.smoothing_alpha)
                track.observation = observation
                track.last_seen = timestamp
                track.hits += 1
                track.lost_frames = 0
            visible.append(self._public(track))

        retired = tuple(
            sorted(
                track_id
                for track_id, track in self._tracks.items()
                if track.lost_frames > self.config.max_lost_frames
            )
        )
        for track_id in retired:
            self._tracks.pop(track_id, None)
        missing = tuple(
            sorted(
                track_id
                for track_id, track in self._tracks.items()
                if track.lost_frames > 0
            )
        )
        return TrackerFrame(
            visible=tuple(sorted(visible, key=lambda item: item.track_id)),
            missing_track_ids=missing,
            retired_track_ids=retired,
        )

    def _public(self, track: _Track) -> TrackedHead:
        return TrackedHead(
            track_id=track.track_id,
            observation=track.observation,
            box=track.box,
            first_seen=track.first_seen,
            last_seen=track.last_seen,
            age_frames=track.age_frames,
            hits=track.hits,
            lost_frames=track.lost_frames,
            confirmed=track.hits >= self.config.min_hits,
        )
