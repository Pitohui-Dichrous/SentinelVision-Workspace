"""Per-track PPE risk state machine and one-event-per-episode alerting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config import RiskConfig
from .types import PPEState, PipelineAlert, RiskState, StateTransition


LOGGER = logging.getLogger(__name__)


@dataclass
class _RiskTrack:
    public_track_id: Optional[int] = None
    state: RiskState = RiskState.NORMAL
    violation_frames: int = 0
    suspect_frames: int = 0
    recovery_frames: int = 0
    episode_sequence: int = 0
    event_id: Optional[str] = None
    raw_first_seen: Optional[float] = None
    first_seen: Optional[float] = None
    confirmed_at: Optional[float] = None
    alarmed_at: Optional[float] = None
    cooldown_until: float = 0.0
    last_alert_at: float = float("-inf")
    event_alerted: bool = False


@dataclass(frozen=True)
class RiskUpdate:
    state: RiskState
    alerts: Tuple[PipelineAlert, ...]
    transitions: Tuple[StateTransition, ...]


class PPERiskStateMachine:
    RISK_TYPE = "no_helmet"

    def __init__(self, config: RiskConfig, session_id: str):
        self.config = config
        self.session_id = session_id
        self._tracks: Dict[int, _RiskTrack] = {}
        self.suppressed_transients = 0
        self.confirmed_events = 0
        self.alerts_emitted = 0

    def reset(self, session_id: str) -> None:
        self.session_id = session_id
        self._tracks.clear()
        self.suppressed_transients = 0
        self.confirmed_events = 0
        self.alerts_emitted = 0

    def state_for(self, candidate_id: int) -> RiskState:
        """Return state from the private tracker-candidate namespace."""

        return self._tracks.get(candidate_id, _RiskTrack()).state

    def remove(self, candidate_id: int, timestamp: float) -> Tuple[StateTransition, ...]:
        """Retire candidate-owned state without leaking a tentative ID."""

        track = self._tracks.pop(candidate_id, None)
        if track is None:
            return ()
        if track.state == RiskState.SUSPECT:
            self.suppressed_transients += 1
        if track.state == RiskState.NORMAL or track.public_track_id is None:
            return ()
        return (
            StateTransition(
                track_id=track.public_track_id,
                risk_type=self.RISK_TYPE,
                from_state=track.state,
                to_state=RiskState.NORMAL,
                at=timestamp,
                reason="track_retired",
                event_id=track.event_id,
            ),
        )

    def observe_raw(self, candidate_id: int, raw_state: PPEState, timestamp: float) -> None:
        """Record pre-confirmation evidence without advancing the risk state."""

        track = self._tracks.setdefault(candidate_id, _RiskTrack())
        if raw_state == PPEState.NO_HELMET and track.raw_first_seen is None:
            track.raw_first_seen = timestamp
        elif raw_state == PPEState.HELMET and track.state == RiskState.NORMAL:
            track.raw_first_seen = None

    def update(
        self,
        candidate_id: int,
        stable_state: PPEState,
        confidence: float,
        timestamp: float,
        class_id: str,
        source_model_ids: Tuple[str, ...],
        raw_state: PPEState = PPEState.UNKNOWN,
        alerts_enabled: bool = True,
        public_track_id: Optional[int] = None,
    ) -> RiskUpdate:
        track = self._tracks.setdefault(candidate_id, _RiskTrack())
        track_id = self._bind_public_track_id(candidate_id, track, public_track_id)
        alerts: List[PipelineAlert] = []
        transitions: List[StateTransition] = []
        violating = stable_state == PPEState.NO_HELMET
        recovered = stable_state == PPEState.HELMET
        self.observe_raw(candidate_id, raw_state, timestamp)

        if track.state == RiskState.NORMAL:
            if violating:
                track.violation_frames += 1
                if track.first_seen is None:
                    track.first_seen = track.raw_first_seen if track.raw_first_seen is not None else timestamp
                if track.violation_frames >= self.config.suspect_frames:
                    self._transition(track_id, track, RiskState.SUSPECT, timestamp, "stable_no_helmet", transitions)
                    track.suspect_frames = 0
            else:
                track.violation_frames = 0

        elif track.state == RiskState.SUSPECT:
            if violating:
                track.recovery_frames = 0
                track.suspect_frames += 1
                if track.suspect_frames >= self.config.confirm_frames:
                    track.confirmed_at = timestamp
                    track.episode_sequence += 1
                    track.event_id = "%s-ppe-%04d-%03d" % (
                        self.session_id,
                        track_id,
                        track.episode_sequence,
                    )
                    track.event_alerted = False
                    self.confirmed_events += 1
                    self._transition(track_id, track, RiskState.CONFIRMED, timestamp, "confirmation_window_met", transitions)
                    if alerts_enabled and timestamp - track.last_alert_at >= self.config.cooldown_seconds:
                        self._emit_alert(
                            track_id, track, confidence, timestamp, class_id, source_model_ids,
                            stable_state, alerts, transitions,
                        )
                    elif alerts_enabled:
                        track.cooldown_until = track.last_alert_at + self.config.cooldown_seconds
                        self._transition(
                            track_id, track, RiskState.COOLDOWN, timestamp,
                            "per_track_cooldown_suppressed", transitions,
                        )
            elif recovered:
                track.suspect_frames = 0
                track.recovery_frames += 1
                if track.recovery_frames >= self.config.recovery_frames:
                    self.suppressed_transients += 1
                    self._transition(track_id, track, RiskState.NORMAL, timestamp, "suspect_recovered", transitions)
                    self._reset_episode(track)
            else:
                track.suspect_frames = 0
                track.recovery_frames = 0

        elif track.state == RiskState.CONFIRMED:
            if recovered:
                track.recovery_frames = 1
                self._transition(track_id, track, RiskState.RECOVERING, timestamp, "protected_evidence", transitions)
            elif violating and alerts_enabled:
                if timestamp - track.last_alert_at >= self.config.cooldown_seconds:
                    self._emit_alert(
                        track_id, track, confidence, timestamp, class_id, source_model_ids,
                        stable_state, alerts, transitions,
                    )
                else:
                    track.cooldown_until = track.last_alert_at + self.config.cooldown_seconds
                    self._transition(
                        track_id, track, RiskState.COOLDOWN, timestamp,
                        "per_track_cooldown_suppressed", transitions,
                    )

        elif track.state == RiskState.ALARMED:
            if recovered:
                track.recovery_frames = 1
                self._transition(track_id, track, RiskState.RECOVERING, timestamp, "protected_evidence", transitions)
            elif violating and timestamp < track.cooldown_until:
                self._transition(track_id, track, RiskState.COOLDOWN, timestamp, "per_track_cooldown", transitions)

        elif track.state == RiskState.COOLDOWN:
            if recovered:
                track.recovery_frames = 1
                self._transition(track_id, track, RiskState.RECOVERING, timestamp, "protected_evidence", transitions)
            elif violating and timestamp >= track.cooldown_until:
                if not track.event_alerted and alerts_enabled:
                    self._emit_alert(
                        track_id, track, confidence, timestamp, class_id, source_model_ids,
                        stable_state, alerts, transitions,
                    )
                elif track.event_alerted:
                    self._transition(track_id, track, RiskState.ALARMED, timestamp, "cooldown_elapsed", transitions)

        elif track.state == RiskState.RECOVERING:
            if recovered:
                track.recovery_frames += 1
                if track.recovery_frames >= self.config.recovery_frames:
                    self._transition(track_id, track, RiskState.NORMAL, timestamp, "recovery_hysteresis_met", transitions)
                    self._reset_episode(track)
            elif violating:
                destination = RiskState.COOLDOWN if track.event_id else RiskState.SUSPECT
                self._transition(track_id, track, destination, timestamp, "violation_returned", transitions)
                track.recovery_frames = 0
            else:
                track.recovery_frames = 0

        return RiskUpdate(track.state, tuple(alerts), tuple(transitions))

    @staticmethod
    def _bind_public_track_id(
        candidate_id: int,
        track: _RiskTrack,
        public_track_id: Optional[int],
    ) -> int:
        """Bind an immutable public ID while keeping v1 direct calls compatible.

        Legacy callers supplied only one ID because schema v1 exposed candidate
        IDs immediately.  Omitting ``public_track_id`` therefore preserves that
        behavior; schema v2 callers pass the tracker-assigned public ID.
        """

        resolved_id = (
            track.public_track_id
            if public_track_id is None and track.public_track_id is not None
            else candidate_id if public_track_id is None else public_track_id
        )
        if isinstance(resolved_id, bool) or not isinstance(resolved_id, int) or resolved_id < 1:
            raise ValueError("public_track_id must be a positive integer")
        if track.public_track_id is None:
            track.public_track_id = resolved_id
        elif track.public_track_id != resolved_id:
            raise ValueError(
                "candidate %s is already bound to public track %s"
                % (candidate_id, track.public_track_id)
            )
        return resolved_id

    def _emit_alert(
        self,
        track_id: int,
        track: _RiskTrack,
        confidence: float,
        timestamp: float,
        class_id: str,
        source_model_ids: Tuple[str, ...],
        stable_state: PPEState,
        alerts: List[PipelineAlert],
        transitions: List[StateTransition],
    ) -> None:
        if track.event_id is None:
            return
        self._transition(track_id, track, RiskState.ALARMED, timestamp, "event_alert_emitted", transitions)
        track.alarmed_at = timestamp
        track.last_alert_at = timestamp
        track.cooldown_until = timestamp + self.config.cooldown_seconds
        track.event_alerted = True
        alerts.append(PipelineAlert(
            event_id=track.event_id,
            track_id=track_id,
            risk_type=self.RISK_TYPE,
            class_id=class_id,
            confidence=float(confidence),
            first_seen=float(track.first_seen if track.first_seen is not None else timestamp),
            confirmed_at=float(track.confirmed_at if track.confirmed_at is not None else timestamp),
            alarmed_at=timestamp,
            source_model_ids=source_model_ids,
            stable_state=stable_state,
            risk_state=RiskState.ALARMED,
        ))
        self.alerts_emitted += 1

    def _transition(
        self,
        track_id: int,
        track: _RiskTrack,
        destination: RiskState,
        timestamp: float,
        reason: str,
        output: List[StateTransition],
    ) -> None:
        if track.state == destination:
            return
        transition = StateTransition(
            track_id=track_id,
            risk_type=self.RISK_TYPE,
            from_state=track.state,
            to_state=destination,
            at=timestamp,
            reason=reason,
            event_id=track.event_id,
        )
        LOGGER.info(
            "PPE risk transition track=%s %s->%s reason=%s event=%s",
            track_id,
            track.state.value,
            destination.value,
            reason,
            track.event_id or "-",
        )
        track.state = destination
        output.append(transition)

    @staticmethod
    def _reset_episode(track: _RiskTrack) -> None:
        track.violation_frames = 0
        track.suspect_frames = 0
        track.recovery_frames = 0
        track.event_id = None
        track.raw_first_seen = None
        track.first_seen = None
        track.confirmed_at = None
        track.alarmed_at = None
        track.event_alerted = False
