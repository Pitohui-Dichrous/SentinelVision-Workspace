from __future__ import annotations

import unittest

from safety_pipeline.config import (
    RiskConfig,
    SafetyPipelineConfig,
    TemporalConfig,
    TrackingConfig,
)
from safety_pipeline.pipeline import PPETemporalPipeline
from safety_pipeline.ppe_conflict import box_iou
from safety_pipeline.risk import PPERiskStateMachine
from safety_pipeline.types import (
    HeadObservation,
    PPEState,
    ResolvedDetection,
    RiskState,
    TrackedHead,
    TrackerFrame,
)


def head(
    state: PPEState = PPEState.NO_HELMET,
    confidence: float = 0.94,
    box=(10.0, 10.0, 50.0, 50.0),
) -> HeadObservation:
    return HeadObservation(
        box=box,
        state=state,
        helmet_score=confidence if state == PPEState.HELMET else None,
        nohelmet_score=confidence if state == PPEState.NO_HELMET else None,
        confidence=confidence,
        raw_detections=(),
        source_model_ids=("combined",),
        resolution="test",
    )


def resolved(observation: HeadObservation) -> ResolvedDetection:
    if observation.state == PPEState.HELMET:
        class_id = "hat"
    elif observation.state == PPEState.NO_HELMET:
        class_id = "person"
    else:
        class_id = "ppe_uncertain"
    return ResolvedDetection(
        box=observation.box,
        confidence=observation.confidence,
        class_id=class_id,
        source_model_ids=observation.source_model_ids,
        head_observation=observation,
    )


def tracked(
    candidate_id: int,
    track_id: int | None,
    *,
    confirmed: bool,
    hits: int,
    observation: HeadObservation | None = None,
) -> TrackedHead:
    item = observation or head()
    return TrackedHead(
        candidate_id=candidate_id,
        track_id=track_id,
        observation=item,
        box=item.box,
        first_seen=0.0,
        last_seen=float(max(0, hits - 1)),
        age_frames=hits,
        hits=hits,
        lost_frames=0,
        confirmed=confirmed,
    )


class ScriptedTracker:
    def __init__(self, frames):
        self._frames = list(frames)
        self.active_count = 0
        self.candidate_count = 0

    def reset(self):
        self._frames.clear()
        self.active_count = 0
        self.candidate_count = 0

    def update(self, observations, timestamp):
        del observations, timestamp
        frame = self._frames.pop(0)
        self.candidate_count = len(frame.visible) + len(frame.missing_candidate_ids)
        self.active_count = sum(item.track_id is not None for item in frame.visible)
        return frame


def frame(*items: TrackedHead) -> TrackerFrame:
    return TrackerFrame(
        visible=tuple(items),
        missing_candidate_ids=(),
        retired_candidate_ids=(),
    )


def fast_pipeline() -> PPETemporalPipeline:
    config = SafetyPipelineConfig(
        default_mode="ppe_temporal",
        temporal=TemporalConfig(
            window_size=3,
            min_votes=1,
            ema_alpha=0.5,
            score_margin=0.1,
            min_stable_frames=1,
            recovery_frames=1,
            missing_tolerance=1,
        ),
        risk=RiskConfig(
            suspect_frames=1,
            confirm_frames=1,
            recovery_frames=2,
            cooldown_seconds=10.0,
        ),
    )
    return PPETemporalPipeline(config, session_id="dual")


class RiskPublicIdTests(unittest.TestCase):
    def setUp(self):
        self.risk = PPERiskStateMachine(
            RiskConfig(
                suspect_frames=1,
                confirm_frames=1,
                recovery_frames=2,
                cooldown_seconds=10.0,
            ),
            "risk",
        )

    def update(self, state: PPEState, timestamp: float):
        return self.risk.update(
            candidate_id=41,
            public_track_id=7,
            stable_state=state,
            confidence=0.9,
            timestamp=timestamp,
            class_id="person",
            source_model_ids=("combined",),
            raw_state=state,
        )

    def test_public_events_keep_candidate_owned_preconfirmation_time(self):
        self.risk.observe_raw(41, PPEState.NO_HELMET, 0.0)
        first = self.update(PPEState.NO_HELMET, 5.0)
        alarmed = self.update(PPEState.NO_HELMET, 6.0)

        self.assertEqual(first.state, RiskState.SUSPECT)
        self.assertEqual(self.risk.state_for(41), RiskState.ALARMED)
        self.assertEqual(self.risk.state_for(7), RiskState.NORMAL)
        self.assertEqual({item.track_id for item in first.transitions + alarmed.transitions}, {7})
        self.assertEqual(len(alarmed.alerts), 1)
        self.assertEqual(alarmed.alerts[0].track_id, 7)
        self.assertIn("-ppe-0007-", alarmed.alerts[0].event_id)
        self.assertEqual(alarmed.alerts[0].first_seen, 0.0)
        self.assertEqual(alarmed.alerts[0].alarmed_at - alarmed.alerts[0].first_seen, 6.0)

    def test_cooldown_survives_public_id_binding_and_new_episode(self):
        self.update(PPEState.NO_HELMET, 0.0)
        first = self.update(PPEState.NO_HELMET, 1.0).alerts[0]
        self.update(PPEState.NO_HELMET, 2.0)
        self.update(PPEState.HELMET, 3.0)
        self.update(PPEState.HELMET, 4.0)
        self.update(PPEState.NO_HELMET, 5.0)
        suppressed = self.update(PPEState.NO_HELMET, 6.0)

        self.assertEqual(suppressed.state, RiskState.COOLDOWN)
        self.assertEqual(suppressed.alerts, ())
        second = self.update(PPEState.NO_HELMET, 11.0).alerts[0]
        self.assertEqual(second.track_id, 7)
        self.assertNotEqual(first.event_id, second.event_id)

    def test_retiring_tentative_candidate_never_leaks_internal_id(self):
        self.risk.observe_raw(41, PPEState.NO_HELMET, 0.0)
        self.assertEqual(self.risk.remove(41, 1.0), ())

    def test_public_id_cannot_be_rebound(self):
        self.update(PPEState.NO_HELMET, 0.0)
        with self.assertRaises(ValueError):
            self.risk.update(
                candidate_id=41,
                public_track_id=8,
                stable_state=PPEState.NO_HELMET,
                confidence=0.9,
                timestamp=1.0,
                class_id="person",
                source_model_ids=("combined",),
                raw_state=PPEState.NO_HELMET,
            )


class PipelinePublicIdTests(unittest.TestCase):
    def test_real_tracker_reacquires_public_track_without_duplicate_event(self):
        tracking = TrackingConfig(
            iou_threshold=0.5,
            min_hits=2,
            smoothing_alpha=1.0,
            max_lost_frames=None,
            max_lost_seconds=1.0,
            tentative_max_lost_seconds=0.15,
            reacquire_enabled=True,
            reacquire_min_iou=0.1,
            reacquire_max_center_distance_ratio=0.8,
            reacquire_min_area_ratio=0.5,
            reacquire_ambiguity_margin=0.1,
            motion_max_seconds=0.5,
            velocity_alpha=1.0,
            public_id_policy="confirmed_only",
        )
        pipeline = PPETemporalPipeline(
            SafetyPipelineConfig(
                default_mode="ppe_temporal",
                tracking=tracking,
                temporal=TemporalConfig(
                    window_size=3,
                    min_votes=1,
                    ema_alpha=0.5,
                    score_margin=0.1,
                    min_stable_frames=1,
                    recovery_frames=1,
                    missing_tolerance=1,
                ),
                risk=RiskConfig(
                    suspect_frames=1,
                    confirm_frames=1,
                    recovery_frames=2,
                    cooldown_seconds=10.0,
                ),
            ),
            session_id="reacquire",
        )

        def process(observation, timestamp):
            items = () if observation is None else (resolved(observation),)
            return pipeline.process(
                items,
                pipeline.summarize_filtered(items),
                timestamp,
            )

        first_box = (0.0, 0.0, 40.0, 40.0)
        second_box = (10.0, 0.0, 50.0, 40.0)
        third_box = (20.0, 0.0, 60.0, 40.0)
        reacquired_box = (40.0, 0.0, 80.0, 40.0)
        self.assertLess(box_iou(third_box, reacquired_box), tracking.iou_threshold)

        tentative = process(head(box=first_box), 0.0)
        promoted = process(head(box=second_box), 0.1)
        alarmed = process(head(box=third_box), 0.2)
        missing = process(None, 0.3)
        reacquired = process(head(box=reacquired_box), 0.4)

        self.assertEqual(tentative.detections, ())
        self.assertEqual([item.track_id for item in promoted.detections], [1])
        self.assertEqual([item.track_id for item in alarmed.detections], [1])
        self.assertEqual([item.track_id for item in reacquired.detections], [1])
        self.assertEqual(missing.metrics.active_candidates, 1)
        self.assertEqual(reacquired.metrics.active_candidates, 1)
        self.assertEqual(reacquired.metrics.active_tracks, 1)

        frames = (tentative, promoted, alarmed, missing, reacquired)
        alerts = [alert for result in frames for alert in result.alerts]
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].track_id, 1)
        self.assertEqual(alerts[0].first_seen, 0.0)
        self.assertAlmostEqual(alerts[0].alarmed_at - alerts[0].first_seen, 0.2)
        event_ids = {
            transition.event_id
            for result in frames
            for transition in result.transitions
            if transition.event_id is not None
        }
        self.assertEqual(event_ids, {alerts[0].event_id})
        self.assertEqual(reacquired.metrics.confirmed_events, 1)
        self.assertEqual(reacquired.metrics.alerts_emitted, 1)
        self.assertEqual(reacquired.detections[0].risk_state, RiskState.COOLDOWN)

    def test_v2_tentative_candidate_is_private_but_contributes_latency(self):
        pipeline = fast_pipeline()
        pipeline.tracker = ScriptedTracker((
            frame(tracked(41, None, confirmed=False, hits=1)),
            frame(tracked(41, 7, confirmed=True, hits=2)),
            frame(tracked(41, 7, confirmed=True, hits=3)),
        ))
        item = resolved(head())
        resolution = pipeline.summarize_filtered((item,))

        tentative = pipeline.process((item,), resolution, 0.0)
        suspect = pipeline.process((item,), resolution, 1.0)
        alarmed = pipeline.process((item,), resolution, 2.0)

        self.assertEqual(tentative.detections, ())
        self.assertEqual(tentative.metrics.active_candidates, 1)
        self.assertEqual(tentative.metrics.active_tracks, 0)
        self.assertEqual([item.track_id for item in suspect.detections], [7])
        self.assertEqual([item.track_id for item in alarmed.detections], [7])
        self.assertEqual({item.track_id for item in suspect.transitions + alarmed.transitions}, {7})
        self.assertEqual(len(alarmed.alerts), 1)
        self.assertEqual(alarmed.alerts[0].track_id, 7)
        self.assertEqual(alarmed.alerts[0].first_seen, 0.0)
        self.assertEqual(alarmed.alerts[0].alarmed_at - alarmed.alerts[0].first_seen, 2.0)
        self.assertEqual(alarmed.metrics.active_candidates, 1)
        self.assertEqual(alarmed.metrics.active_tracks, 1)

    def test_v1_immediate_id_remains_visible_before_confirmation(self):
        pipeline = fast_pipeline()
        pipeline.tracker = ScriptedTracker((
            frame(tracked(41, 1, confirmed=False, hits=1)),
        ))
        item = resolved(head())
        resolution = pipeline.summarize_filtered((item,))

        result = pipeline.process((item,), resolution, 0.0)

        self.assertEqual([item.track_id for item in result.detections], [1])
        self.assertEqual(result.alerts, ())
        self.assertEqual(result.transitions, ())
        self.assertEqual(result.metrics.active_candidates, 1)
        self.assertEqual(result.metrics.active_tracks, 1)


if __name__ == "__main__":
    unittest.main()
