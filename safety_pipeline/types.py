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
        }


@dataclass(frozen=True)
class PipelineFrame:
    detections: Tuple[PipelineDetection, ...]
    alerts: Tuple[PipelineAlert, ...]
    transitions: Tuple[StateTransition, ...]
    metrics: PipelineMetrics
