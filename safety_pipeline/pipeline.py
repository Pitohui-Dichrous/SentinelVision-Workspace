"""Orchestrates PPE tracking, temporal evidence, risk, and metrics."""

from __future__ import annotations

import uuid
from typing import Iterable, List, Optional

from .config import SafetyPipelineConfig
from .ppe_conflict import PPEConflictResolver
from .risk import PPERiskStateMachine
from .temporal import TemporalEvidenceEngine
from .tracking import HeadTracker
from .types import (
    PPEState,
    PipelineDetection,
    PipelineFrame,
    PipelineMetrics,
    ResolutionFrame,
    ResolvedDetection,
    RiskState,
    StateTransition,
)


class PPETemporalPipeline:
    """A deterministic PPE-only safety pipeline.

    Fire and other classes are intentionally outside this first research slice
    and remain on SentinelVision's legacy-compatible path.
    """

    def __init__(self, config: SafetyPipelineConfig, session_id: Optional[str] = None):
        self.config = config
        self.resolver = PPEConflictResolver(config.conflict)
        self.tracker = HeadTracker(config.tracking)
        self.temporal = TemporalEvidenceEngine(config.temporal)
        self.session_id = session_id or self._new_session_id()
        self.risk = PPERiskStateMachine(config.risk, self.session_id)
        self._frames_processed = 0
        self._raw_ppe_detections = 0
        self._head_observations = 0
        self._conflicts_resolved = 0
        self._uncertain_observations = 0
        self._stable_state_changes = 0

    @staticmethod
    def _new_session_id() -> str:
        return uuid.uuid4().hex[:12]

    def reset(self, session_id: Optional[str] = None) -> None:
        self.session_id = session_id or self._new_session_id()
        self.tracker.reset()
        self.temporal.reset()
        self.risk.reset(self.session_id)
        self._frames_processed = 0
        self._raw_ppe_detections = 0
        self._head_observations = 0
        self._conflicts_resolved = 0
        self._uncertain_observations = 0
        self._stable_state_changes = 0

    def resolve(self, detections: Iterable[object]) -> ResolutionFrame:
        return self.resolver.resolve(detections, enabled=True)

    @staticmethod
    def summarize_filtered(detections: Iterable[ResolvedDetection]) -> ResolutionFrame:
        """Build metrics for PPE evidence inside the active monitoring domain."""
        items = tuple(detections)
        raw_count = sum(len(item.head_observation.raw_detections) for item in items)
        conflicts = sum(len(item.head_observation.raw_detections) > 1 for item in items)
        uncertain = sum(item.head_observation.state == PPEState.UNCERTAIN for item in items)
        return ResolutionFrame(
            detections=items,
            input_count=raw_count,
            output_count=len(items),
            ppe_raw_count=raw_count,
            head_count=len(items),
            conflicts_resolved=conflicts,
            uncertain_count=uncertain,
        )

    def process(
        self,
        ppe_detections: Iterable[ResolvedDetection],
        resolution: ResolutionFrame,
        timestamp: float,
        alerts_enabled: bool = True,
    ) -> PipelineFrame:
        items = tuple(ppe_detections)
        observations = tuple(item.head_observation for item in items)
        tracker_frame = self.tracker.update(observations, timestamp)
        self._frames_processed += 1
        self._raw_ppe_detections += resolution.ppe_raw_count
        self._head_observations += resolution.head_count
        self._conflicts_resolved += resolution.conflicts_resolved
        self._uncertain_observations += resolution.uncertain_count

        transitions: List[StateTransition] = []
        alerts = []
        detections: List[PipelineDetection] = []
        for candidate_id in tracker_frame.missing_candidate_ids:
            self.temporal.update_missing(candidate_id)
        for candidate_id in tracker_frame.retired_candidate_ids:
            self.temporal.remove(candidate_id)
            transitions.extend(self.risk.remove(candidate_id, timestamp))

        protected_id = self.config.conflict.protected_class_id
        unprotected_id = self.config.conflict.unprotected_class_id
        uncertain_id = self.config.conflict.uncertain_class_id
        for tracked in tracker_frame.visible:
            candidate_id = tracked.candidate_id
            estimate = self.temporal.update(candidate_id, tracked.observation)
            self.risk.observe_raw(candidate_id, estimate.raw_state, timestamp)
            if estimate.changed:
                self._stable_state_changes += 1
            if tracked.confirmed:
                if tracked.track_id is None:
                    raise RuntimeError("confirmed tracker candidate has no public track ID")
                risk_update = self.risk.update(
                    candidate_id=candidate_id,
                    stable_state=estimate.stable_state,
                    confidence=estimate.stable_confidence,
                    timestamp=timestamp,
                    class_id=unprotected_id,
                    source_model_ids=tracked.observation.source_model_ids,
                    raw_state=estimate.raw_state,
                    alerts_enabled=alerts_enabled,
                    public_track_id=tracked.track_id,
                )
                risk_state = risk_update.state
                alerts.extend(risk_update.alerts)
                transitions.extend(risk_update.transitions)
            else:
                risk_state = RiskState.NORMAL

            # Schema v2 keeps tentative candidates private.  Schema v1's
            # immediate policy still supplies a public ID before confirmation,
            # preserving its historical rendering contract without advancing
            # the risk state early.
            if tracked.track_id is None:
                continue

            if estimate.stable_state == PPEState.HELMET:
                class_id = protected_id
            elif estimate.stable_state == PPEState.NO_HELMET:
                class_id = unprotected_id
            else:
                class_id = uncertain_id
            confidence = (
                estimate.stable_confidence
                if estimate.stable_state in (PPEState.HELMET, PPEState.NO_HELMET)
                else tracked.observation.confidence
            )
            detections.append(PipelineDetection(
                box=tracked.box,
                confidence=confidence,
                class_id=class_id,
                source_model_ids=tracked.observation.source_model_ids,
                track_id=tracked.track_id,
                raw_state=estimate.raw_state,
                stable_state=estimate.stable_state,
                risk_state=risk_state,
                stability=estimate.stability,
                head_observation=tracked.observation,
            ))

        metrics = PipelineMetrics(
            session_id=self.session_id,
            mode="ppe_temporal",
            frames_processed=self._frames_processed,
            raw_ppe_detections=self._raw_ppe_detections,
            head_observations=self._head_observations,
            conflicts_resolved=self._conflicts_resolved,
            uncertain_observations=self._uncertain_observations,
            stable_state_changes=self._stable_state_changes,
            suppressed_transients=self.risk.suppressed_transients,
            confirmed_events=self.risk.confirmed_events,
            alerts_emitted=self.risk.alerts_emitted,
            active_tracks=self.tracker.active_count,
            active_candidates=self.tracker.candidate_count,
        )
        return PipelineFrame(
            detections=tuple(detections),
            alerts=tuple(alerts),
            transitions=tuple(transitions),
            metrics=metrics,
        )
