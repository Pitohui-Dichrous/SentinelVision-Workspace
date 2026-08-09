"""Immutable data contracts shared by the safety reasoning modules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Tuple


Box = Tuple[float, float, float, float]


class PPEState(str, Enum):
    HELMET = "HELMET"
    NO_HELMET = "NO_HELMET"
    UNCERTAIN = "UNCERTAIN"
    UNKNOWN = "UNKNOWN"


class RiskState(str, Enum):
    NORMAL = "NORMAL"
    SUSPECT = "SUSPECT"
    CONFIRMED = "CONFIRMED"
    ALARMED = "ALARMED"
    COOLDOWN = "COOLDOWN"
    RECOVERING = "RECOVERING"


@dataclass(frozen=True)
class HeadObservation:
    box: Box
    state: PPEState
    helmet_score: Optional[float]
    nohelmet_score: Optional[float]
    confidence: float
    raw_detections: Tuple[Any, ...]
    source_model_ids: Tuple[str, ...]
    resolution: str


@dataclass(frozen=True)
class ResolvedDetection:
    box: Box
    confidence: float
    class_id: str
    source_model_ids: Tuple[str, ...]
    head_observation: HeadObservation


@dataclass(frozen=True)
class ResolutionFrame:
    detections: Tuple[Any, ...]
    input_count: int
    output_count: int
    ppe_raw_count: int
    head_count: int
    conflicts_resolved: int
    uncertain_count: int


@dataclass(frozen=True)
class TrackedHead:
    candidate_id: int
    track_id: Optional[int]
    observation: HeadObservation
    box: Box
    first_seen: float
    last_seen: float
    age_frames: int
    hits: int
    lost_frames: int
    confirmed: bool
    evidence_eligible: bool = True


@dataclass(frozen=True)
class TrackerFrame:
    visible: Tuple[TrackedHead, ...]
    missing_candidate_ids: Tuple[int, ...]
    retired_candidate_ids: Tuple[int, ...]

    @property
    def missing_track_ids(self) -> Tuple[int, ...]:
        """Compatibility alias; values belong to the private candidate namespace."""

        return self.missing_candidate_ids

    @property
    def retired_track_ids(self) -> Tuple[int, ...]:
        """Compatibility alias; values belong to the private candidate namespace."""

        return self.retired_candidate_ids


@dataclass(frozen=True)
class TemporalEstimate:
    track_id: int
    raw_state: PPEState
    stable_state: PPEState
    raw_confidence: float
    stable_confidence: float
    helmet_ema: float
    nohelmet_ema: float
    stability: float
    evidence_count: int
    missing_frames: int
    changed: bool


@dataclass(frozen=True)
class StateTransition:
    track_id: int
    risk_type: str
    from_state: RiskState
    to_state: RiskState
    at: float
    reason: str
    event_id: Optional[str] = None


@dataclass(frozen=True)
class PipelineAlert:
    event_id: str
    track_id: int
    risk_type: str
    class_id: str
    confidence: float
    first_seen: float
    confirmed_at: float
    alarmed_at: float
    source_model_ids: Tuple[str, ...]
    stable_state: PPEState
    risk_state: RiskState
    evidence_hits: int = 0
    evidence_duration: float = 0.0
    evidence_stability: float = 0.0
    decision_reason: str = ""


@dataclass(frozen=True)
class PipelineDetection:
    box: Box
    confidence: float
    class_id: str
    source_model_ids: Tuple[str, ...]
    track_id: int
    raw_state: PPEState
    stable_state: PPEState
    risk_state: RiskState
    stability: float
    head_observation: HeadObservation


@dataclass(frozen=True)
class PipelineMetrics:
    session_id: str
    mode: str
    frames_processed: int
    raw_ppe_detections: int
    head_observations: int
    conflicts_resolved: int
    uncertain_observations: int
    stable_state_changes: int
    suppressed_transients: int
    confirmed_events: int
    alerts_emitted: int
    active_tracks: int
    active_candidates: int = 0
    candidates_created: int = 0
    candidates_promoted: int = 0
    tentative_retired: int = 0
    confirmed_retired: int = 0
    admission_dropped: int = 0
    low_confidence_filtered: int = 0
    invalid_observations: int = 0
    high_water_candidates: int = 0
    strict_matches: int = 0
    reacquire_attempts: int = 0
    reacquire_successes: int = 0
    ambiguity_rejections: int = 0
    pipeline_ms_last: float = 0.0
    pipeline_ms_p95: float = 0.0
    pipeline_ms_max: float = 0.0

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "frames_processed": self.frames_processed,
            "raw_ppe_detections": self.raw_ppe_detections,
            "head_observations": self.head_observations,
            "conflicts_resolved": self.conflicts_resolved,
            "uncertain_observations": self.uncertain_observations,
            "stable_state_changes": self.stable_state_changes,
            "suppressed_transients": self.suppressed_transients,
            "confirmed_events": self.confirmed_events,
            "alerts_emitted": self.alerts_emitted,
            "active_tracks": self.active_tracks,
            "active_candidates": self.active_candidates,
            "candidates_created": self.candidates_created,
            "candidates_promoted": self.candidates_promoted,
            "tentative_retired": self.tentative_retired,
            "confirmed_retired": self.confirmed_retired,
            "admission_dropped": self.admission_dropped,
            "low_confidence_filtered": self.low_confidence_filtered,
            "invalid_observations": self.invalid_observations,
            "high_water_candidates": self.high_water_candidates,
            "strict_matches": self.strict_matches,
            "reacquire_attempts": self.reacquire_attempts,
            "reacquire_successes": self.reacquire_successes,
            "ambiguity_rejections": self.ambiguity_rejections,
            "pipeline_ms_last": self.pipeline_ms_last,
            "pipeline_ms_p95": self.pipeline_ms_p95,
            "pipeline_ms_max": self.pipeline_ms_max,
        }


@dataclass(frozen=True)
class PipelineFrame:
    detections: Tuple[PipelineDetection, ...]
    alerts: Tuple[PipelineAlert, ...]
    transitions: Tuple[StateTransition, ...]
    metrics: PipelineMetrics
