from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from safety_pipeline.config import (
    PPEConflictConfig,
    RiskConfig,
    SafetyPipelineConfig,
    TemporalConfig,
    TrackingConfig,
)
from safety_pipeline.pipeline import PPETemporalPipeline
from safety_pipeline.journal import JsonlEventJournal
from safety_pipeline.ppe_conflict import PPEConflictResolver
from safety_pipeline.risk import PPERiskStateMachine
from safety_pipeline.temporal import TemporalEvidenceEngine
from safety_pipeline.tracking import HeadTracker
from safety_pipeline.types import HeadObservation, PPEState, RiskState


class RawDetection:
    def __init__(self, class_id, confidence, box=(10.0, 10.0, 50.0, 50.0)):
        self.class_id = class_id
        self.confidence = confidence
        self.box = box
        self.source_model_ids = ("combined",)


def head(state, confidence=0.9, box=(10.0, 10.0, 50.0, 50.0)):
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


class HeadTrackerTests(unittest.TestCase):
    def test_class_flip_keeps_track_id(self):
        tracker = HeadTracker(TrackingConfig(iou_threshold=0.3, min_hits=1))
        first = tracker.update((head(PPEState.HELMET),), 0.0).visible[0]
        second = tracker.update((head(PPEState.NO_HELMET, box=(11.0, 10.0, 51.0, 50.0)),), 1.0).visible[0]
        self.assertEqual(first.track_id, second.track_id)
        self.assertEqual(second.observation.state, PPEState.NO_HELMET)

    def test_one_observation_cannot_update_two_tracks(self):
        tracker = HeadTracker(TrackingConfig(iou_threshold=0.1, min_hits=1))
        first_frame = tracker.update((
            head(PPEState.HELMET, box=(0.0, 0.0, 40.0, 40.0)),
            head(PPEState.HELMET, box=(30.0, 0.0, 70.0, 40.0)),
        ), 0.0)
        self.assertEqual(len(first_frame.visible), 2)
        second_frame = tracker.update((head(PPEState.HELMET, box=(20.0, 0.0, 60.0, 40.0)),), 1.0)
        self.assertEqual(len(second_frame.visible), 1)
        self.assertEqual(len(second_frame.missing_track_ids), 1)

    def test_tracker_does_not_swap_identity_to_maximize_cardinality(self):
        tracker = HeadTracker(TrackingConfig(
            iou_threshold=0.3,
            min_hits=1,
            smoothing_alpha=1.0,
        ))
        first_frame = tracker.update((
            head(PPEState.HELMET, box=(0.0, 0.0, 100.0, 100.0)),
            head(PPEState.NO_HELMET, box=(-53.0, 0.0, 47.0, 100.0)),
        ), 0.0)
        helmet_track = next(
            item for item in first_frame.visible
            if item.observation.state == PPEState.HELMET
        )
        nohelmet_track = next(
            item for item in first_frame.visible
            if item.observation.state == PPEState.NO_HELMET
        )

        second_frame = tracker.update((
            head(PPEState.HELMET, box=(0.0, 0.0, 100.0, 100.0)),
            head(PPEState.NO_HELMET, box=(52.0, 0.0, 152.0, 100.0)),
        ), 1.0)

        visible = {item.candidate_id: item for item in second_frame.visible}
        self.assertIn(helmet_track.candidate_id, visible)
        preserved = visible[helmet_track.candidate_id]
        self.assertEqual(preserved.track_id, helmet_track.track_id)
        self.assertEqual(preserved.box, (0.0, 0.0, 100.0, 100.0))
        self.assertEqual(preserved.observation.state, PPEState.HELMET)
        self.assertEqual(
            second_frame.missing_candidate_ids,
            (nohelmet_track.candidate_id,),
        )

    def test_short_loss_is_retained_then_retired(self):
        tracker = HeadTracker(TrackingConfig(max_lost_frames=2, min_hits=1))
        track_id = tracker.update((head(PPEState.HELMET),), 0.0).visible[0].track_id
        self.assertIn(track_id, tracker.update((), 1.0).missing_track_ids)
        self.assertIn(track_id, tracker.update((), 2.0).missing_track_ids)
        self.assertIn(track_id, tracker.update((), 3.0).retired_track_ids)


class TemporalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.engine = TemporalEvidenceEngine(TemporalConfig(
            window_size=5,
            min_votes=3,
            ema_alpha=0.5,
            score_margin=0.1,
            min_stable_frames=1,
            recovery_frames=2,
            missing_tolerance=2,
        ))

    def test_single_frame_does_not_confirm_nohelmet(self):
        result = self.engine.update(17, head(PPEState.NO_HELMET, 0.95))
        self.assertNotEqual(result.stable_state, PPEState.NO_HELMET)

    def test_n_of_m_confirms_and_one_jitter_frame_does_not_flip(self):
        for _ in range(3):
            result = self.engine.update(17, head(PPEState.NO_HELMET, 0.95))
        self.assertEqual(result.stable_state, PPEState.NO_HELMET)
        jitter = self.engine.update(17, head(PPEState.HELMET, 0.90))
        self.assertEqual(jitter.stable_state, PPEState.NO_HELMET)

    def test_missing_tolerance_preserves_then_clears_state(self):
        for _ in range(3):
            result = self.engine.update(17, head(PPEState.HELMET, 0.95))
        self.assertEqual(result.stable_state, PPEState.HELMET)
        self.assertEqual(self.engine.update_missing(17).stable_state, PPEState.HELMET)
        self.assertEqual(self.engine.update_missing(17).stable_state, PPEState.HELMET)
        self.assertEqual(self.engine.update_missing(17).stable_state, PPEState.UNKNOWN)


class RiskStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.risk = PPERiskStateMachine(
            RiskConfig(suspect_frames=1, confirm_frames=1, recovery_frames=2, cooldown_seconds=10.0),
            "session",
        )

    def update(self, track_id, state, timestamp):
        return self.risk.update(
            track_id, state, 0.9, timestamp, "person", ("combined",), raw_state=state
        )

    def test_each_track_can_alarm_independently(self):
        self.assertEqual(self.update(17, PPEState.NO_HELMET, 0.0).state, RiskState.SUSPECT)
        first = self.update(17, PPEState.NO_HELMET, 1.0)
        self.assertEqual(len(first.alerts), 1)
        self.assertEqual(self.update(26, PPEState.NO_HELMET, 2.0).state, RiskState.SUSPECT)
        second = self.update(26, PPEState.NO_HELMET, 3.0)
        self.assertEqual(len(second.alerts), 1)
        self.assertNotEqual(first.alerts[0].event_id, second.alerts[0].event_id)
        self.assertEqual(len(self.update(17, PPEState.NO_HELMET, 4.0).alerts), 0)

    def test_recovery_hysteresis_allows_new_episode(self):
        self.update(17, PPEState.NO_HELMET, 0.0)
        first = self.update(17, PPEState.NO_HELMET, 1.0).alerts[0]
        self.update(17, PPEState.NO_HELMET, 2.0)
        self.assertEqual(self.update(17, PPEState.HELMET, 3.0).state, RiskState.RECOVERING)
        self.assertEqual(self.update(17, PPEState.HELMET, 4.0).state, RiskState.NORMAL)
        self.update(17, PPEState.NO_HELMET, 5.0)
        suppressed = self.update(17, PPEState.NO_HELMET, 6.0)
        self.assertEqual(suppressed.state, RiskState.COOLDOWN)
        self.assertEqual(suppressed.alerts, ())
        second = self.update(17, PPEState.NO_HELMET, 11.0).alerts[0]
        self.assertNotEqual(first.event_id, second.event_id)

    def test_cooldown_expiry_does_not_duplicate_same_episode(self):
        self.update(17, PPEState.NO_HELMET, 0.0)
        self.assertEqual(len(self.update(17, PPEState.NO_HELMET, 1.0).alerts), 1)
        self.assertEqual(self.update(17, PPEState.NO_HELMET, 2.0).state, RiskState.COOLDOWN)
        after_cooldown = self.update(17, PPEState.NO_HELMET, 12.0)
        self.assertEqual(after_cooldown.state, RiskState.ALARMED)
        self.assertEqual(after_cooldown.alerts, ())

    def test_confirmation_requires_consecutive_violation_frames(self):
        risk = PPERiskStateMachine(
            RiskConfig(suspect_frames=1, confirm_frames=2, recovery_frames=2, cooldown_seconds=10.0),
            "continuous",
        )
        def update(state, timestamp):
            return risk.update(
                17, state, 0.9, timestamp, "person", ("combined",), raw_state=state
            )
        update(PPEState.NO_HELMET, 0.0)
        update(PPEState.NO_HELMET, 1.0)
        update(PPEState.HELMET, 2.0)
        update(PPEState.NO_HELMET, 3.0)
        interrupted = update(PPEState.UNCERTAIN, 4.0)
        self.assertEqual(interrupted.alerts, ())
        self.assertEqual(update(PPEState.NO_HELMET, 5.0).alerts, ())
        confirmed = update(PPEState.NO_HELMET, 6.0)
        self.assertEqual(len(confirmed.alerts), 1)

    def test_disabled_alert_policy_stops_at_confirmed(self):
        self.risk.update(
            17, PPEState.NO_HELMET, 0.9, 0.0, "person", ("combined",),
            raw_state=PPEState.NO_HELMET, alerts_enabled=False,
        )
        result = self.risk.update(
            17, PPEState.NO_HELMET, 0.9, 1.0, "person", ("combined",),
            raw_state=PPEState.NO_HELMET, alerts_enabled=False,
        )
        self.assertEqual(result.state, RiskState.CONFIRMED)
        self.assertEqual(result.alerts, ())
        self.assertEqual(self.risk.alerts_emitted, 0)

    def test_alert_delay_starts_at_first_raw_violation_evidence(self):
        self.risk.update(
            17, PPEState.UNCERTAIN, 0.8, 0.0, "person", ("combined",),
            raw_state=PPEState.NO_HELMET,
        )
        self.update(17, PPEState.NO_HELMET, 5.0)
        alert = self.update(17, PPEState.NO_HELMET, 6.0).alerts[0]
        self.assertEqual(alert.first_seen, 0.0)
        self.assertEqual(alert.alarmed_at - alert.first_seen, 6.0)

    def test_opposite_raw_evidence_resets_pre_suspect_latency_origin(self):
        self.risk.update(
            17, PPEState.UNCERTAIN, 0.8, 0.0, "person", ("combined",),
            raw_state=PPEState.NO_HELMET,
        )
        self.risk.update(
            17, PPEState.UNCERTAIN, 0.8, 2.0, "person", ("combined",),
            raw_state=PPEState.HELMET,
        )
        self.update(17, PPEState.NO_HELMET, 5.0)
        alert = self.update(17, PPEState.NO_HELMET, 6.0).alerts[0]
        self.assertEqual(alert.first_seen, 5.0)

    def test_recovery_requires_consecutive_protected_frames(self):
        self.update(17, PPEState.NO_HELMET, 0.0)
        self.update(17, PPEState.NO_HELMET, 1.0)
        self.update(17, PPEState.NO_HELMET, 2.0)
        self.assertEqual(self.update(17, PPEState.HELMET, 3.0).state, RiskState.RECOVERING)
        self.assertEqual(self.update(17, PPEState.UNCERTAIN, 4.0).state, RiskState.RECOVERING)
        self.assertEqual(self.update(17, PPEState.HELMET, 5.0).state, RiskState.RECOVERING)
        self.assertEqual(self.update(17, PPEState.HELMET, 6.0).state, RiskState.NORMAL)


class PipelineIntegrationTests(unittest.TestCase):
    def test_end_to_end_emits_one_alert_per_track_episode(self):
        config = SafetyPipelineConfig(
            default_mode="ppe_temporal",
            conflict=PPEConflictConfig(),
            tracking=TrackingConfig(iou_threshold=0.3, max_lost_frames=2, min_hits=1),
            temporal=TemporalConfig(
                window_size=3,
                min_votes=2,
                ema_alpha=0.5,
                score_margin=0.1,
                min_stable_frames=1,
                recovery_frames=1,
                missing_tolerance=1,
            ),
            risk=RiskConfig(suspect_frames=1, confirm_frames=1, recovery_frames=2, cooldown_seconds=10.0),
        )
        pipeline = PPETemporalPipeline(config, session_id="fixed")
        emitted = []
        track_ids = []
        for frame_index in range(5):
            resolution = pipeline.resolve((RawDetection("person", 0.94),))
            ppe = tuple(item for item in resolution.detections if hasattr(item, "head_observation"))
            result = pipeline.process(ppe, resolution, float(frame_index))
            emitted.extend(result.alerts)
            track_ids.extend(item.track_id for item in result.detections)
        self.assertEqual(len(set(track_ids)), 1)
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].track_id, track_ids[0])
        self.assertEqual(result.metrics.alerts_emitted, 1)
        self.assertEqual(result.metrics.confirmed_events, 1)
        self.assertGreaterEqual(result.metrics.frames_processed, 5)

    def test_confirmation_delay_includes_pre_tracker_confirmation_evidence(self):
        config = SafetyPipelineConfig(
            default_mode="ppe_temporal",
            tracking=TrackingConfig(iou_threshold=0.3, max_lost_frames=2, min_hits=3),
            temporal=TemporalConfig(
                window_size=3,
                min_votes=2,
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
        pipeline = PPETemporalPipeline(config, session_id="latency")
        emitted = []
        for frame_index in range(4):
            resolution = pipeline.resolve((RawDetection("person", 0.94),))
            result = pipeline.process(resolution.detections, resolution, float(frame_index))
            emitted.extend(result.alerts)
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].first_seen, 0.0)
        self.assertEqual(emitted[0].alarmed_at - emitted[0].first_seen, 3.0)

    def test_metrics_only_count_detections_inside_active_monitoring_domain(self):
        pipeline = PPETemporalPipeline(SafetyPipelineConfig(default_mode="ppe_temporal"), session_id="domain")
        resolution = pipeline.resolve((
            RawDetection("hat", 0.96),
            RawDetection("person", 0.75),
        ))
        filtered = pipeline.summarize_filtered(())
        result = pipeline.process((), filtered, 0.0)
        self.assertEqual(result.metrics.raw_ppe_detections, 0)
        self.assertEqual(result.metrics.head_observations, 0)
        self.assertEqual(result.metrics.conflicts_resolved, 0)


class EventJournalTests(unittest.TestCase):
    def test_jsonl_records_are_structured_and_append_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events" / "ppe.jsonl"
            journal = JsonlEventJournal(path)
            self.assertTrue(journal.append("risk_transition", {"track_id": 17, "to_state": "ALARMED"}))
            self.assertTrue(journal.append("human_review", {"event_id": "evt-1", "conclusion": "valid"}))
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([record["record_type"] for record in records], ["risk_transition", "human_review"])
        self.assertEqual(records[0]["track_id"], 17)
        self.assertEqual(records[1]["conclusion"], "valid")


if __name__ == "__main__":
    unittest.main()
