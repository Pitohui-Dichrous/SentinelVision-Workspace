"""Per-track multi-frame evidence fusion with hysteresis."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

from .config import TemporalConfig
from .types import HeadObservation, PPEState, TemporalEstimate


@dataclass(frozen=True)
class _Evidence:
    state: PPEState
    helmet_score: float
    nohelmet_score: float
    missing: bool = False


@dataclass
class _TemporalTrack:
    history: Deque[_Evidence]
    stable_state: PPEState = PPEState.UNKNOWN
    pending_state: Optional[PPEState] = None
    pending_frames: int = 0
    helmet_ema: float = 0.0
    nohelmet_ema: float = 0.0
    ema_initialized: bool = False
    missing_frames: int = 0


class TemporalEvidenceEngine:
    def __init__(self, config: TemporalConfig):
        self.config = config
        self._tracks: Dict[int, _TemporalTrack] = {}

    def reset(self) -> None:
        self._tracks.clear()

    def remove(self, track_id: int) -> None:
        self._tracks.pop(track_id, None)

    def _state(self, track_id: int) -> _TemporalTrack:
        state = self._tracks.get(track_id)
        if state is None:
            state = _TemporalTrack(history=deque(maxlen=self.config.window_size))
            self._tracks[track_id] = state
        return state

    def update(self, track_id: int, observation: HeadObservation) -> TemporalEstimate:
        state = self._state(track_id)
        helmet_score = float(observation.helmet_score or 0.0)
        nohelmet_score = float(observation.nohelmet_score or 0.0)
        state.history.append(_Evidence(observation.state, helmet_score, nohelmet_score))
        state.missing_frames = 0
        if not state.ema_initialized:
            state.helmet_ema = helmet_score
            state.nohelmet_ema = nohelmet_score
            state.ema_initialized = True
        else:
            alpha = self.config.ema_alpha
            state.helmet_ema = alpha * helmet_score + (1.0 - alpha) * state.helmet_ema
            state.nohelmet_ema = alpha * nohelmet_score + (1.0 - alpha) * state.nohelmet_ema

        candidate = self._candidate(state)
        changed = self._apply_candidate(state, candidate)
        return self._estimate(track_id, observation.state, observation.confidence, state, changed)

    def update_missing(self, track_id: int) -> TemporalEstimate:
        state = self._state(track_id)
        state.history.append(_Evidence(PPEState.UNKNOWN, 0.0, 0.0, missing=True))
        state.missing_frames += 1
        changed = False
        if state.missing_frames > self.config.missing_tolerance:
            changed = state.stable_state != PPEState.UNKNOWN
            state.stable_state = PPEState.UNKNOWN
            state.pending_state = None
            state.pending_frames = 0
        return self._estimate(track_id, PPEState.UNKNOWN, 0.0, state, changed)

    def snapshot(
        self,
        track_id: int,
        raw_state: PPEState = PPEState.UNKNOWN,
        raw_confidence: float = 0.0,
    ) -> TemporalEstimate:
        """Read the current estimate without adding alert evidence.

        This is used when a low-confidence observation is accepted only for
        geometric association.  It may move a box, but it cannot manufacture
        temporal votes, recovery evidence, or an alarm.
        """

        state = self._state(track_id)
        return self._estimate(track_id, raw_state, raw_confidence, state, False)

    def _candidate(self, state: _TemporalTrack) -> PPEState:
        evidence = tuple(item for item in state.history if not item.missing)
        if not evidence:
            return PPEState.UNKNOWN
        helmet_votes = sum(item.state == PPEState.HELMET for item in evidence)
        nohelmet_votes = sum(item.state == PPEState.NO_HELMET for item in evidence)
        helmet_ready = (
            helmet_votes >= self.config.min_votes
            and state.helmet_ema - state.nohelmet_ema >= self.config.score_margin - 1e-12
        )
        nohelmet_ready = (
            nohelmet_votes >= self.config.min_votes
            and state.nohelmet_ema - state.helmet_ema >= self.config.score_margin - 1e-12
        )
        if helmet_ready and not nohelmet_ready:
            return PPEState.HELMET
        if nohelmet_ready and not helmet_ready:
            return PPEState.NO_HELMET
        if helmet_ready and nohelmet_ready:
            return PPEState.HELMET if state.helmet_ema >= state.nohelmet_ema else PPEState.NO_HELMET
        return PPEState.UNCERTAIN

    def _apply_candidate(self, state: _TemporalTrack, candidate: PPEState) -> bool:
        if candidate == state.stable_state:
            state.pending_state = None
            state.pending_frames = 0
            return False
        if candidate != state.pending_state:
            state.pending_state = candidate
            state.pending_frames = 1
        else:
            state.pending_frames += 1
        required = (
            self.config.recovery_frames
            if state.stable_state in (PPEState.HELMET, PPEState.NO_HELMET)
            else self.config.min_stable_frames
        )
        if state.pending_frames < required:
            return False
        state.stable_state = candidate
        state.pending_state = None
        state.pending_frames = 0
        return True

    def _estimate(
        self,
        track_id: int,
        raw_state: PPEState,
        raw_confidence: float,
        state: _TemporalTrack,
        changed: bool,
    ) -> TemporalEstimate:
        evidence = tuple(item for item in state.history if not item.missing)
        if state.stable_state == PPEState.HELMET:
            stable_confidence = state.helmet_ema
        elif state.stable_state == PPEState.NO_HELMET:
            stable_confidence = state.nohelmet_ema
        else:
            stable_confidence = max(state.helmet_ema, state.nohelmet_ema)
        matching = sum(item.state == state.stable_state for item in evidence)
        stability = matching / len(evidence) if evidence else 0.0
        return TemporalEstimate(
            track_id=track_id,
            raw_state=raw_state,
            stable_state=state.stable_state,
            raw_confidence=float(raw_confidence),
            stable_confidence=max(0.0, min(1.0, stable_confidence)),
            helmet_ema=max(0.0, min(1.0, state.helmet_ema)),
            nohelmet_ema=max(0.0, min(1.0, state.nohelmet_ema)),
            stability=max(0.0, min(1.0, stability)),
            evidence_count=len(evidence),
            missing_frames=state.missing_frames,
            changed=changed,
        )
