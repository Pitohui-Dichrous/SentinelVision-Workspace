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
    motion_box: Optional[Box] = None
    confidence_ema: float = 0.0
    publication_hits: int = 1
    publication_started_at: Optional[float] = None
    publication_confidence_ema: Optional[float] = None


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


def _predict_alpha_beta_state(
    track: _Track,
    timestamp: float,
    config: TrackingConfig,
) -> Tuple[Box, Tuple[float, float]]:
    """Predict a center-only motion state with bounded velocity decay.

    Size is deliberately not extrapolated. Detector width/height estimates are
    much noisier than center motion and previously made short gaps expand or
    collapse the association box catastrophically.
    """

    elapsed = max(0.0, timestamp - track.last_seen)
    prediction_horizon = max(0.0, float(config.motion_max_seconds))
    prediction_elapsed = min(elapsed, prediction_horizon)
    base_box = track.motion_box or track.measurement_box
    center_x, center_y, width, height = _box_state(base_box)
    velocity_x, velocity_y = track.velocity[0], track.velocity[1]
    decay_rate = max(
        0.0,
        float(getattr(config, "velocity_decay_per_second", 0.0)),
    )
    if decay_rate > _EPSILON:
        decay = math.exp(-decay_rate * elapsed)
        displacement_scale = (
            1.0 - math.exp(-decay_rate * prediction_elapsed)
        ) / decay_rate
    else:
        decay = 1.0
        displacement_scale = prediction_elapsed
    predicted = _state_box((
        center_x + velocity_x * displacement_scale,
        center_y + velocity_y * displacement_scale,
        width,
        height,
    ))
    return predicted, (velocity_x * decay, velocity_y * decay)


def _valid_observation(observation: HeadObservation) -> bool:
    try:
        box = tuple(float(value) for value in observation.box)
        confidence = float(observation.confidence)
    except (TypeError, ValueError):
        return False
    return (
        len(box) == 4
        and all(math.isfinite(value) for value in box)
        and box[2] > box[0]
        and box[3] > box[1]
        and math.isfinite(confidence)
        and 0.0 <= confidence <= 1.0
    )


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
        self._stats: Dict[str, int] = {}
        self._reset_stats()

    def reset(self) -> None:
        self._tracks.clear()
        self._next_candidate_id = 1
        self._next_track_id = 1
        self._last_timestamp = None
        self._reset_stats()

    def _reset_stats(self) -> None:
        self._stats = {
            "created": 0,
            "promoted": 0,
            "retired": 0,
            "tentative_retired": 0,
            "confirmed_retired": 0,
            "dropped": 0,
            "low_confidence_dropped": 0,
            "capacity_dropped": 0,
            "invalid": 0,
            "highwater": 0,
            "strict": 0,
            "reacquire": 0,
            "reacquire_attempts": 0,
            "reacquire_successes": 0,
            "ambiguity": 0,
        }

    @property
    def stats(self) -> Mapping[str, int]:
        return dict(self._stats)

    @property
    def statistics(self) -> Mapping[str, int]:
        """Descriptive alias for integrations that avoid the short name."""

        return self.stats

    def _update_highwater(self) -> None:
        self._stats["highwater"] = max(
            self._stats["highwater"],
            len(self._tracks),
        )

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

    def update(
        self,
        observations: Iterable[HeadObservation],
        timestamp: float,
        *,
        high_confidence_threshold: Optional[float] = None,
    ) -> TrackerFrame:
        if getattr(self.config, "motion_model", "legacy") == "alpha_beta":
            return self._update_alpha_beta(
                observations,
                timestamp,
                high_confidence_threshold=high_confidence_threshold,
            )
        return self._update_legacy(observations, timestamp)

    def _update_legacy(
        self,
        observations: Iterable[HeadObservation],
        timestamp: float,
    ) -> TrackerFrame:
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

    def _update_alpha_beta(
        self,
        observations: Iterable[HeadObservation],
        timestamp: float,
        *,
        high_confidence_threshold: Optional[float] = None,
    ) -> TrackerFrame:
        """Hardened association path used by schema-3 opt-in profiles.

        Every active track is predicted before either association stage. Low
        confidence observations may preserve an already-published identity,
        but can neither create a candidate nor become safety evidence.
        """

        now = self._monotonic_timestamp(timestamp)
        association_floor = float(
            getattr(self.config, "association_min_confidence", 0.0)
        )
        creation_floor = float(
            getattr(self.config, "new_candidate_min_confidence", 0.0)
        )
        if high_confidence_threshold is not None:
            requested_floor = float(high_confidence_threshold)
            if not math.isfinite(requested_floor) or not 0.0 <= requested_floor <= 1.0:
                raise ValueError("high_confidence_threshold must be finite and between 0 and 1")
            creation_floor = max(creation_floor, requested_floor)
        items: List[HeadObservation] = []
        high_confidence: List[bool] = []
        for observation in observations:
            if not _valid_observation(observation):
                self._stats["invalid"] += 1
                continue
            confidence = float(observation.confidence)
            if confidence + _EPSILON < association_floor:
                self._stats["dropped"] += 1
                self._stats["low_confidence_dropped"] += 1
                continue
            items.append(observation)
            high_confidence.append(confidence + _EPSILON >= creation_floor)

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
            retired.update(
                candidate_id
                for candidate_id, track in self._tracks.items()
                if self._expired_by_time(track, now)
            )
            self._stats["tentative_retired"] += sum(
                self._tracks[candidate_id].track_id is None
                for candidate_id in retired
            )
            self._stats["confirmed_retired"] += sum(
                self._tracks[candidate_id].track_id is not None
                for candidate_id in retired
            )
            for candidate_id in retired:
                self._tracks.pop(candidate_id, None)

        unmatched_tracks: Set[int] = set(self._tracks)
        unmatched_observations: Set[int] = set(range(len(items)))
        assignments: Dict[int, int] = {}
        predicted_boxes = {
            candidate_id: _predict_alpha_beta_state(track, now, self.config)[0]
            for candidate_id, track in self._tracks.items()
        }
        prediction_horizon = max(0.0, float(self.config.motion_max_seconds))
        recovery_eligible_tracks = {
            candidate_id
            for candidate_id, track in self._tracks.items()
            if track.track_id is not None
            and now - track.last_seen <= prediction_horizon + _EPSILON
        }
        reacquire_attempt_pairs: Set[Tuple[int, int]] = set()
        for candidate_id in previously_missing & recovery_eligible_tracks:
            for observation_index in unmatched_observations:
                reacquire_attempt_pairs.add((candidate_id, observation_index))

        high_observations = {
            index for index, is_high in enumerate(high_confidence) if is_high
        }
        low_observations = set(range(len(items))) - high_observations

        def apply_strict_stage(
            observation_pool: Set[int],
            *,
            public_only: bool,
        ) -> None:
            stage_observations = unmatched_observations & observation_pool
            if not unmatched_tracks or not stage_observations:
                return
            strict_weights: Dict[Tuple[int, int], float] = {}
            for candidate_id in sorted(unmatched_tracks):
                track = self._tracks[candidate_id]
                if public_only and track.track_id is None:
                    continue
                predicted_box = predicted_boxes[candidate_id]
                for observation_index in sorted(stage_observations):
                    overlap = box_iou(predicted_box, items[observation_index].box)
                    if overlap + _EPSILON >= self.config.iou_threshold and overlap > 0.0:
                        strict_weights[(candidate_id, observation_index)] = overlap
            if previously_missing:
                before = len(strict_weights)
                strict_weights = _unambiguous_edges(
                    strict_weights,
                    self.config.reacquire_ambiguity_margin,
                    restricted_tracks=previously_missing,
                )
                self._stats["ambiguity"] += before - len(strict_weights)
            strict_matches = maximum_total_weight_matching(
                sorted(unmatched_tracks),
                sorted(stage_observations),
                strict_weights,
            )
            strict_reacquire = sum(
                candidate_id in previously_missing
                and candidate_id in recovery_eligible_tracks
                for candidate_id in strict_matches
            )
            self._stats["reacquire"] += strict_reacquire
            self._stats["reacquire_successes"] += strict_reacquire
            self._stats["strict"] += sum(
                candidate_id not in previously_missing
                or candidate_id not in recovery_eligible_tracks
                for candidate_id in strict_matches
            )
            self._apply_matches(
                strict_matches,
                assignments,
                unmatched_tracks,
                unmatched_observations,
            )

        def apply_relaxed_stage(
            observation_pool: Set[int],
            *,
            public_only: bool,
        ) -> None:
            stage_observations = unmatched_observations & observation_pool
            if (
                not self.config.reacquire_enabled
                or not unmatched_tracks
                or not stage_observations
            ):
                return
            relaxed_weights: Dict[Tuple[int, int], float] = {}
            for candidate_id in sorted(unmatched_tracks):
                track = self._tracks[candidate_id]
                if public_only and track.track_id is None:
                    continue
                if not public_only and track.track_id is None:
                    # A private hit-confirmed candidate may use a strong
                    # relaxed match only to bridge a real detection gap.  Do
                    # not run an O(private*observation) relaxed search for
                    # continuously visible, publication-waiting noise.
                    if (
                        not self._is_confirmed(track)
                        or candidate_id not in previously_missing
                    ):
                        continue
                if now - track.last_seen > prediction_horizon + _EPSILON:
                    continue
                predicted_box = predicted_boxes[candidate_id]
                for observation_index in sorted(stage_observations):
                    reacquire_attempt_pairs.add((candidate_id, observation_index))
                    observation = items[observation_index]
                    overlap = box_iou(predicted_box, observation.box)
                    if overlap + _EPSILON < self.config.reacquire_min_iou:
                        continue
                    if not _geometry_compatible(predicted_box, observation.box, self.config):
                        continue
                    relaxed_weights[(candidate_id, observation_index)] = max(overlap, 1e-9)
            before = len(relaxed_weights)
            relaxed_weights = _unambiguous_edges(
                relaxed_weights,
                self.config.reacquire_ambiguity_margin,
            )
            self._stats["ambiguity"] += before - len(relaxed_weights)
            relaxed_matches = maximum_total_weight_matching(
                sorted(unmatched_tracks),
                sorted(stage_observations),
                relaxed_weights,
            )
            self._stats["reacquire"] += len(relaxed_matches)
            self._stats["reacquire_successes"] += len(relaxed_matches)
            self._apply_matches(
                relaxed_matches,
                assignments,
                unmatched_tracks,
                unmatched_observations,
            )

        # ByteTrack-style confidence ordering: strong observations get the
        # first opportunity to claim any trajectory.  Weak observations can
        # only preserve a still-unmatched public identity and can never
        # steal it from a stronger observation or create a new candidate.
        apply_strict_stage(high_observations, public_only=False)
        apply_relaxed_stage(high_observations, public_only=False)
        apply_strict_stage(low_observations, public_only=True)
        apply_relaxed_stage(low_observations, public_only=True)

        self._stats["reacquire_attempts"] += len(reacquire_attempt_pairs)

        observation_tracks: Dict[int, _Track] = {}
        evidence_eligible: Dict[int, bool] = {}
        for observation_index, candidate_id in sorted(assignments.items()):
            track = self._tracks[candidate_id]
            publication_gap = candidate_id in previously_missing
            self._update_alpha_beta_track(track, items[observation_index], now)
            self._update_publication_streak(
                track,
                now,
                eligible=high_confidence[observation_index],
                reset=publication_gap,
                confidence=float(items[observation_index].confidence),
            )
            observation_tracks[observation_index] = track
            evidence_eligible[observation_index] = high_confidence[observation_index]

        unmatched_low = [
            index
            for index in unmatched_observations
            if not high_confidence[index]
        ]
        self._stats["dropped"] += len(unmatched_low)
        self._stats["low_confidence_dropped"] += len(unmatched_low)
        new_candidates = [
            index
            for index in unmatched_observations
            if high_confidence[index]
        ]

        def admission_key(index: int) -> Tuple[float, float, float, int]:
            center_x, center_y, _, _ = _box_state(items[index].box)
            return -float(items[index].confidence), center_x, center_y, index

        capacity = int(getattr(self.config, "max_tentative_candidates", 0))
        private_candidate_count = sum(
            track.track_id is None
            for track in self._tracks.values()
        )
        slots = (
            len(new_candidates)
            if capacity <= 0
            else max(0, capacity - private_candidate_count)
        )
        admitted = set(sorted(new_candidates, key=admission_key)[:slots])
        self._stats["dropped"] += len(new_candidates) - len(admitted)
        self._stats["capacity_dropped"] += len(new_candidates) - len(admitted)
        for observation_index in sorted(admitted):
            track = self._new_alpha_beta_track(items[observation_index], now)
            observation_tracks[observation_index] = track
            evidence_eligible[observation_index] = True

        matched_candidates = {
            track.candidate_id for track in observation_tracks.values()
        }
        for candidate_id, track in self._tracks.items():
            if candidate_id not in matched_candidates and track.track_id is None:
                self._reset_publication_streak(track)

        self._assign_public_ids(now)
        visible = [
            self._public(
                track,
                evidence_eligible=evidence_eligible[observation_index],
            )
            for observation_index, track in observation_tracks.items()
        ]

        if self.config.max_lost_frames is not None:
            retired.update(
                candidate_id
                for candidate_id, track in self._tracks.items()
                if track.lost_frames > self.config.max_lost_frames
            )
            newly_retired = tuple(
                candidate_id for candidate_id in retired if candidate_id in self._tracks
            )
            self._stats["tentative_retired"] += sum(
                self._tracks[candidate_id].track_id is None
                for candidate_id in newly_retired
            )
            self._stats["confirmed_retired"] += sum(
                self._tracks[candidate_id].track_id is not None
                for candidate_id in newly_retired
            )
            for candidate_id in retired:
                self._tracks.pop(candidate_id, None)

        self._stats["retired"] += len(retired)
        self._update_highwater()
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
            motion_box=observation.box,
            confidence_ema=float(observation.confidence),
            publication_hits=1,
            publication_started_at=timestamp,
            publication_confidence_ema=float(observation.confidence),
        )
        self._tracks[candidate_id] = track
        return track

    def _new_alpha_beta_track(
        self,
        observation: HeadObservation,
        timestamp: float,
    ) -> _Track:
        track = self._new_track(observation, timestamp)
        self._stats["created"] += 1
        self._update_highwater()
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

    def _update_alpha_beta_track(
        self,
        track: _Track,
        observation: HeadObservation,
        timestamp: float,
    ) -> None:
        previous_box = track.motion_box or track.measurement_box
        previous_center_x, previous_center_y, previous_width, previous_height = _box_state(
            previous_box
        )
        predicted_box, predicted_velocity = _predict_alpha_beta_state(
            track,
            timestamp,
            self.config,
        )
        predicted_center_x, predicted_center_y, _, _ = _box_state(predicted_box)
        observed_center_x, observed_center_y, observed_width, observed_height = _box_state(
            observation.box
        )
        residual_x = observed_center_x - predicted_center_x
        residual_y = observed_center_y - predicted_center_y
        position_gain = float(getattr(self.config, "position_gain", 0.75))
        filtered_center_x = predicted_center_x + position_gain * residual_x
        filtered_center_y = predicted_center_y + position_gain * residual_y

        elapsed = max(0.0, timestamp - track.last_seen)
        effective_dt = max(
            elapsed,
            float(getattr(self.config, "min_velocity_dt_seconds", 0.02)),
            _EPSILON,
        )
        if track.hits <= 1:
            velocity_x = (observed_center_x - previous_center_x) / effective_dt
            velocity_y = (observed_center_y - previous_center_y) / effective_dt
        else:
            velocity_gain = float(getattr(self.config, "velocity_gain", 0.20))
            velocity_x = predicted_velocity[0] + velocity_gain * residual_x / effective_dt
            velocity_y = predicted_velocity[1] + velocity_gain * residual_y / effective_dt

        reference_length = max(
            math.sqrt(max(previous_width * previous_height, _EPSILON)),
            math.sqrt(max(observed_width * observed_height, _EPSILON)),
        )
        maximum_speed = (
            float(getattr(self.config, "max_center_speed_ratio", 8.0))
            * reference_length
        )
        speed = math.hypot(velocity_x, velocity_y)
        if maximum_speed > 0.0 and speed > maximum_speed:
            scale = maximum_speed / speed
            velocity_x *= scale
            velocity_y *= scale

        size_alpha = self.config.smoothing_alpha
        filtered_width = (
            size_alpha * observed_width + (1.0 - size_alpha) * previous_width
        )
        filtered_height = (
            size_alpha * observed_height + (1.0 - size_alpha) * previous_height
        )
        track.motion_box = _state_box((
            filtered_center_x,
            filtered_center_y,
            filtered_width,
            filtered_height,
        ))
        track.velocity = (velocity_x, velocity_y, 0.0, 0.0)
        track.box = track.motion_box
        track.measurement_box = observation.box
        track.observation = observation
        track.last_seen = timestamp
        track.hits += 1
        track.lost_frames = 0
        confidence_alpha = float(getattr(self.config, "confidence_alpha", 0.35))
        track.confidence_ema = (
            confidence_alpha * float(observation.confidence)
            + (1.0 - confidence_alpha) * track.confidence_ema
        )

    @staticmethod
    def _reset_publication_streak(track: _Track) -> None:
        if track.track_id is not None:
            return
        track.publication_hits = 0
        track.publication_started_at = None
        track.publication_confidence_ema = None

    def _update_publication_streak(
        self,
        track: _Track,
        timestamp: float,
        *,
        eligible: bool,
        reset: bool,
        confidence: float,
    ) -> None:
        """Advance only uninterrupted high-confidence publication evidence.

        Association continuity and operator-visible publication are separate:
        a private candidate may survive a short gap for motion reacquisition,
        but that gap must never help it earn a public ID or visible box.
        """

        if track.track_id is not None:
            return
        if reset or not eligible:
            self._reset_publication_streak(track)
        if not eligible:
            return
        if track.publication_started_at is None:
            track.publication_started_at = timestamp
            track.publication_hits = 1
            track.publication_confidence_ema = confidence
        else:
            track.publication_hits += 1
            confidence_alpha = float(
                getattr(self.config, "confidence_alpha", 0.35)
            )
            previous = (
                confidence
                if track.publication_confidence_ema is None
                else track.publication_confidence_ema
            )
            track.publication_confidence_ema = (
                confidence_alpha * confidence
                + (1.0 - confidence_alpha) * previous
            )

    def _assign_public_ids(self, timestamp: Optional[float] = None) -> None:
        hardened = getattr(self.config, "motion_model", "legacy") == "alpha_beta"
        publication_min_hits = max(
            int(self.config.min_hits),
            int(getattr(self.config, "publication_min_hits", 0)),
        )
        publication_min_seconds = float(
            getattr(self.config, "publication_min_seconds", 0.0)
        )
        publication_min_confidence = float(
            getattr(self.config, "publication_min_confidence", 0.0)
        )
        eligible = [
            track
            for track in self._tracks.values()
            if track.track_id is None
            and (
                self.config.public_id_policy == "immediate"
                or self._is_confirmed(track)
            )
            and (
                not hardened
                or track.publication_hits >= publication_min_hits
            )
            and (
                not hardened
                or (
                    track.publication_started_at is not None
                    and track.last_seen - track.publication_started_at + _EPSILON
                    >= publication_min_seconds
                )
            )
            and (
                not hardened
                or (
                    track.publication_confidence_ema is not None
                    and track.publication_confidence_ema + _EPSILON
                    >= publication_min_confidence
                )
            )
        ]

        def spatial_key(track: _Track) -> Tuple[float, float, int]:
            center_x, center_y, _, _ = _box_state(track.measurement_box)
            return center_x, center_y, track.candidate_id

        for track in sorted(eligible, key=spatial_key):
            track.track_id = self._next_track_id
            self._next_track_id += 1
            if hardened:
                self._stats["promoted"] += 1

    def _is_confirmed(self, track: _Track) -> bool:
        return track.hits >= self.config.min_hits

    def _expired_by_time(self, track: _Track, timestamp: float) -> bool:
        hardened = getattr(self.config, "motion_model", "legacy") == "alpha_beta"
        retained_as_confirmed = track.track_id is not None if hardened else self._is_confirmed(track)
        if retained_as_confirmed:
            tolerance = self.config.max_lost_seconds
        else:
            tolerance = self.config.tentative_max_lost_seconds
        if tolerance is None:
            return False
        return timestamp - track.last_seen > tolerance + _EPSILON

    def _public(
        self,
        track: _Track,
        evidence_eligible: bool = True,
    ) -> TrackedHead:
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
            evidence_eligible=evidence_eligible,
        )
