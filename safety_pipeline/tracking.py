"""Small deterministic one-to-one tracker for logical head observations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .config import TrackingConfig
from .matching import maximum_total_weight_matching
from .ppe_conflict import box_iou
from .types import Box, HeadObservation, TrackedHead, TrackerFrame


_EPSILON = 1e-12
_ZERO_VELOCITY = (0.0, 0.0, 0.0, 0.0)


@dataclass
class _Track:
    candidate_id: int
    track_id: Optional[int]
    box: Box
    measurement_box: Box
    observation: HeadObservation
    first_seen: float
    last_seen: float
    velocity: Tuple[float, float, float, float] = _ZERO_VELOCITY
    age_frames: int = 1
    hits: int = 1
    lost_frames: int = 0


def _smooth(old_box: Box, new_box: Box, amount: float) -> Box:
    return tuple(
        amount * new_value + (1.0 - amount) * old_value
        for old_value, new_value in zip(old_box, new_box)
    )  # type: ignore[return-value]


def _box_state(box: Sequence[float]) -> Tuple[float, float, float, float]:
    width = max(0.0, float(box[2]) - float(box[0]))
    height = max(0.0, float(box[3]) - float(box[1]))
    return (
        (float(box[0]) + float(box[2])) / 2.0,
        (float(box[1]) + float(box[3])) / 2.0,
        width,
        height,
    )


def _state_box(state: Sequence[float]) -> Box:
    center_x, center_y = float(state[0]), float(state[1])
    width, height = max(0.0, float(state[2])), max(0.0, float(state[3]))
    return (
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    )


def _predict_box(track: _Track, timestamp: float) -> Box:
    elapsed = max(0.0, timestamp - track.last_seen)
    state = _box_state(track.measurement_box)
    predicted = tuple(
        value + velocity * elapsed
        for value, velocity in zip(state, track.velocity)
    )
    return _state_box(predicted)


def _area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _geometry_compatible(
    predicted_box: Sequence[float],
    observation_box: Sequence[float],
    config: TrackingConfig,
) -> bool:
    predicted_area = _area(predicted_box)
    observation_area = _area(observation_box)
    if predicted_area <= 0.0 or observation_area <= 0.0:
        return False
    area_ratio = min(predicted_area, observation_area) / max(
        predicted_area,
        observation_area,
    )
    if area_ratio + _EPSILON < config.reacquire_min_area_ratio:
        return False

    predicted_center = (
        (predicted_box[0] + predicted_box[2]) / 2.0,
        (predicted_box[1] + predicted_box[3]) / 2.0,
    )
    observation_center = (
        (observation_box[0] + observation_box[2]) / 2.0,
        (observation_box[1] + observation_box[3]) / 2.0,
    )
    center_distance = math.hypot(
        predicted_center[0] - observation_center[0],
        predicted_center[1] - observation_center[1],
    )
    reference_length = max(math.sqrt(predicted_area), math.sqrt(observation_area), _EPSILON)
    return (
        center_distance / reference_length
        <= config.reacquire_max_center_distance_ratio + _EPSILON
    )


def _unambiguous_edges(
    weights: Mapping[Tuple[int, int], float],
    margin: float,
    restricted_tracks: Optional[Set[int]] = None,
) -> Dict[Tuple[int, int], float]:
    """Keep selected-track edges that beat alternatives on both sides.

    ``restricted_tracks`` lets the normal strict matcher retain continuity for
    tracks seen in the immediately preceding update while applying the safer
    ambiguity rule to tracks that are being recovered after a real gap.
    """

    by_track: Dict[int, List[Tuple[int, float]]] = {}
    by_observation: Dict[int, List[Tuple[int, float]]] = {}
    for (candidate_id, observation_index), score in weights.items():
        by_track.setdefault(candidate_id, []).append((observation_index, score))
        by_observation.setdefault(observation_index, []).append((candidate_id, score))

    result: Dict[Tuple[int, int], float] = {}
    for pair, score in weights.items():
        candidate_id, observation_index = pair
        if restricted_tracks is not None and candidate_id not in restricted_tracks:
            result[pair] = score
            continue
        track_alternatives = [
            other_score
            for other_index, other_score in by_track[candidate_id]
            if other_index != observation_index
        ]
        observation_alternatives = [
            other_score
            for other_id, other_score in by_observation[observation_index]
            if other_id != candidate_id
        ]
        if track_alternatives and score - max(track_alternatives) + _EPSILON < margin:
            continue
        if observation_alternatives and score - max(observation_alternatives) + _EPSILON < margin:
            continue
        result[pair] = score
    return result


class HeadTracker:
    """Track heads independently of their current helmet classification.

    Candidate IDs are private lifecycle keys allocated for every observation.
    Public track IDs are a separate namespace, so schema-v2 tentative noise
    cannot create gaps in the operator-visible numbering.
    """

    def __init__(self, config: TrackingConfig):
        self.config = config
        self._tracks: Dict[int, _Track] = {}
        self._next_candidate_id = 1
        self._next_track_id = 1
        self._last_timestamp: Optional[float] = None

    def reset(self) -> None:
        self._tracks.clear()
        self._next_candidate_id = 1
        self._next_track_id = 1
        self._last_timestamp = None

    @property
    def active_count(self) -> int:
        """Number of operator-visible tracks.

        In the v1 ``immediate`` compatibility policy every candidate is public;
        with ``confirmed_only`` this naturally counts confirmed tracks only.
        """

        return sum(track.track_id is not None for track in self._tracks.values())

    @property
    def candidate_count(self) -> int:
        return len(self._tracks)

    @property
    def public_count(self) -> int:
        return self.active_count

    @property
    def confirmed_count(self) -> int:
        return sum(self._is_confirmed(track) for track in self._tracks.values())

    def update(self, observations: Iterable[HeadObservation], timestamp: float) -> TrackerFrame:
        now = self._monotonic_timestamp(timestamp)
        items = tuple(observations)
        previously_missing = {
            candidate_id
            for candidate_id, track in self._tracks.items()
            if track.lost_frames > 0
        }
        for track in self._tracks.values():
            track.age_frames += 1
            track.lost_frames += 1

        retired: Set[int] = set()
        if self.config.max_lost_frames is None:
            # Time-based tracks must not be resurrected by a detection that
            # arrives after their configured lifetime has already elapsed.
            retired.update(
                candidate_id
                for candidate_id, track in self._tracks.items()
                if self._expired_by_time(track, now)
            )
            for candidate_id in retired:
                self._tracks.pop(candidate_id, None)

        unmatched_tracks: Set[int] = set(self._tracks)
        unmatched_observations: Set[int] = set(range(len(items)))
        assignments: Dict[int, int] = {}

        strict_weights: Dict[Tuple[int, int], float] = {}
        for candidate_id in sorted(unmatched_tracks):
            track = self._tracks[candidate_id]
            for observation_index in sorted(unmatched_observations):
                overlap = box_iou(track.box, items[observation_index].box)
                if overlap + _EPSILON >= self.config.iou_threshold and overlap > 0.0:
                    strict_weights[(candidate_id, observation_index)] = overlap
        if self.config.reacquire_enabled and previously_missing:
            # A detection gap makes even a high-IoU association a
            # reacquisition.  Do not inherit Risk/Event/Cooldown when two old
            # identities explain the same returning box almost equally well.
            strict_weights = _unambiguous_edges(
                strict_weights,
                self.config.reacquire_ambiguity_margin,
                restricted_tracks=previously_missing,
            )
        self._apply_matches(
            maximum_total_weight_matching(
                sorted(unmatched_tracks),
                sorted(unmatched_observations),
                strict_weights,
            ),
            assignments,
            unmatched_tracks,
            unmatched_observations,
        )

        if self.config.reacquire_enabled and unmatched_tracks and unmatched_observations:
            relaxed_weights: Dict[Tuple[int, int], float] = {}
            for candidate_id in sorted(unmatched_tracks):
                track = self._tracks[candidate_id]
                elapsed = max(0.0, now - track.last_seen)
                if not self._is_confirmed(track):
                    continue
                if elapsed > self.config.motion_max_seconds + _EPSILON:
                    continue
                predicted_box = _predict_box(track, now)
                for observation_index in sorted(unmatched_observations):
                    observation = items[observation_index]
                    overlap = box_iou(predicted_box, observation.box)
                    if overlap + _EPSILON < self.config.reacquire_min_iou:
                        continue
                    if not _geometry_compatible(predicted_box, observation.box, self.config):
                        continue
                    # Optional matching ignores zero-weight edges.  A tiny
                    # positive floor lets an explicit min_iou=0 configuration
                    # rely on the still-active distance and area gates.
                    relaxed_weights[(candidate_id, observation_index)] = max(overlap, 1e-9)

            unambiguous_weights = _unambiguous_edges(
                relaxed_weights,
                self.config.reacquire_ambiguity_margin,
            )
            self._apply_matches(
                maximum_total_weight_matching(
                    sorted(unmatched_tracks),
                    sorted(unmatched_observations),
                    unambiguous_weights,
                ),
                assignments,
                unmatched_tracks,
                unmatched_observations,
            )

        visible_tracks: List[_Track] = []
        for observation_index, observation in enumerate(items):
            candidate_id = assignments.get(observation_index)
            if candidate_id is None:
                track = self._new_track(observation, now)
            else:
                track = self._tracks[candidate_id]
                self._update_track(track, observation, now)
            visible_tracks.append(track)

        # Allocate IDs only after every hit count has been updated.  Spatial
        # ordering makes simultaneous promotions independent of detector input
        # order, while candidate_id remains a final deterministic tie-break.
        self._assign_public_ids()
        visible = [self._public(track) for track in visible_tracks]

        if self.config.max_lost_frames is not None:
            # Preserve schema-v1 behavior: a track may still match on the frame
            # where its loss counter would otherwise exceed the old threshold.
            retired.update(
                candidate_id
                for candidate_id, track in self._tracks.items()
                if track.lost_frames > self.config.max_lost_frames
            )
            for candidate_id in retired:
                self._tracks.pop(candidate_id, None)

        missing = tuple(
            sorted(
                candidate_id
                for candidate_id, track in self._tracks.items()
                if track.lost_frames > 0
            )
        )
        return TrackerFrame(
            visible=tuple(sorted(visible, key=lambda item: item.candidate_id)),
            missing_candidate_ids=missing,
            retired_candidate_ids=tuple(sorted(retired)),
        )

    @staticmethod
    def _apply_matches(
        track_to_observation: Mapping[int, int],
        assignments: Dict[int, int],
        unmatched_tracks: Set[int],
        unmatched_observations: Set[int],
    ) -> None:
        for candidate_id, observation_index in track_to_observation.items():
            assignments[observation_index] = candidate_id
            unmatched_tracks.discard(candidate_id)
            unmatched_observations.discard(observation_index)

    def _monotonic_timestamp(self, timestamp: float) -> float:
        try:
            result = float(timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError("tracker timestamp must be a finite number") from exc
        if not math.isfinite(result):
            raise ValueError("tracker timestamp must be a finite number")
        if self._last_timestamp is not None:
            result = max(result, self._last_timestamp)
        self._last_timestamp = result
        return result

    def _new_track(self, observation: HeadObservation, timestamp: float) -> _Track:
        candidate_id = self._next_candidate_id
        self._next_candidate_id += 1
        track = _Track(
            candidate_id=candidate_id,
            track_id=None,
            box=observation.box,
            measurement_box=observation.box,
            observation=observation,
            first_seen=timestamp,
            last_seen=timestamp,
        )
        self._tracks[candidate_id] = track
        return track

    def _update_track(
        self,
        track: _Track,
        observation: HeadObservation,
        timestamp: float,
    ) -> None:
        elapsed = timestamp - track.last_seen
        if elapsed > _EPSILON:
            previous_state = _box_state(track.measurement_box)
            current_state = _box_state(observation.box)
            instantaneous_velocity = tuple(
                (current - previous) / elapsed
                for previous, current in zip(previous_state, current_state)
            )
            alpha = self.config.velocity_alpha
            track.velocity = tuple(
                alpha * current + (1.0 - alpha) * previous
                for previous, current in zip(track.velocity, instantaneous_velocity)
            )  # type: ignore[assignment]
        track.box = _smooth(track.box, observation.box, self.config.smoothing_alpha)
        track.measurement_box = observation.box
        track.observation = observation
        track.last_seen = timestamp
        track.hits += 1
        track.lost_frames = 0

    def _assign_public_ids(self) -> None:
        eligible = [
            track
            for track in self._tracks.values()
            if track.track_id is None
            and (
                self.config.public_id_policy == "immediate"
                or self._is_confirmed(track)
            )
        ]

        def spatial_key(track: _Track) -> Tuple[float, float, int]:
            center_x, center_y, _, _ = _box_state(track.measurement_box)
            return center_x, center_y, track.candidate_id

        for track in sorted(eligible, key=spatial_key):
            track.track_id = self._next_track_id
            self._next_track_id += 1

    def _is_confirmed(self, track: _Track) -> bool:
        return track.hits >= self.config.min_hits

    def _expired_by_time(self, track: _Track, timestamp: float) -> bool:
        if self._is_confirmed(track):
            tolerance = self.config.max_lost_seconds
        else:
            tolerance = self.config.tentative_max_lost_seconds
        if tolerance is None:
            return False
        return timestamp - track.last_seen > tolerance + _EPSILON

    def _public(self, track: _Track) -> TrackedHead:
        return TrackedHead(
            candidate_id=track.candidate_id,
            track_id=track.track_id,
            observation=track.observation,
            box=track.box,
            first_seen=track.first_seen,
            last_seen=track.last_seen,
            age_frames=track.age_frames,
            hits=track.hits,
            lost_frames=track.lost_frames,
            confirmed=self._is_confirmed(track),
        )
