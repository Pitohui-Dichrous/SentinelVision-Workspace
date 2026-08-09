from __future__ import annotations

import unittest
from pathlib import Path

from safety_pipeline.config import (
    RiskConfig,
    load_safety_pipeline_config_checked,
)
from safety_pipeline.pipeline import PPETemporalPipeline
from safety_pipeline.risk import PPERiskStateMachine
from safety_pipeline.types import PPEState, RiskState


class RawDetection:
    def __init__(
        self,
        confidence: float,
        box=(10.0, 10.0, 50.0, 50.0),
        class_id="person",
    ):
        self.class_id = class_id
        self.confidence = confidence
        self.box = box
        self.source_model_ids = ("helmet_only",)


class ProductionPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "config" / "safety_pipeline.yaml"
        cls.config, cls.valid = load_safety_pipeline_config_checked(path)

    def process(self, pipeline, confidence, timestamp, class_id="person"):
        resolution = pipeline.resolve((RawDetection(confidence, class_id=class_id),))
        return pipeline.process(resolution.detections, resolution, timestamp)

    def test_tracked_schema_three_configuration_is_valid(self):
        self.assertTrue(self.valid)
        self.assertEqual(self.config.schema_version, 3)
        self.assertEqual(self.config.tracking.motion_model, "alpha_beta")
        self.assertTrue(self.config.alert_validation.enabled)

    def test_short_high_confidence_burst_does_not_alarm(self):
        pipeline = PPETemporalPipeline(self.config, session_id="short-noise")
        alerts = []
        for frame_index in range(8):
            result = self.process(pipeline, 0.95, frame_index / 25.0)
            alerts.extend(result.alerts)
        self.assertEqual(alerts, [])
        self.assertEqual(result.metrics.alerts_emitted, 0)

    def test_sustained_violation_alarms_only_after_elapsed_time_gate(self):
        pipeline = PPETemporalPipeline(self.config, session_id="sustained")
        emitted = []
        for frame_index in range(40):
            timestamp = frame_index / 25.0
            result = self.process(pipeline, 0.95, timestamp)
            emitted.extend(result.alerts)
            if emitted:
                break
        self.assertEqual(len(emitted), 1)
        self.assertGreaterEqual(emitted[0].evidence_duration, 0.8 - 1e-9)
        self.assertGreaterEqual(emitted[0].evidence_hits, 2)
        self.assertEqual(emitted[0].decision_reason, "verified_temporal_evidence")

    def test_low_confidence_noise_cannot_create_track_or_risk_event(self):
        pipeline = PPETemporalPipeline(self.config, session_id="low-noise")
        for frame_index in range(40):
            result = self.process(pipeline, 0.30, frame_index / 25.0)
        self.assertEqual(result.detections, ())
        self.assertEqual(result.alerts, ())
        self.assertEqual(result.metrics.active_candidates, 0)
        self.assertGreater(result.metrics.low_confidence_filtered, 0)

    def test_strong_helmet_evidence_clears_stale_raw_violation_origin(self):
        pipeline = PPETemporalPipeline(self.config, session_id="fresh-origin")
        self.process(pipeline, 0.95, 0.0, "person")
        for frame_index in range(1, 21):
            self.process(pipeline, 0.95, frame_index / 25.0, "hat")

        emitted = []
        violation_started_at = 21 / 25.0
        for frame_index in range(21, 80):
            result = self.process(pipeline, 0.95, frame_index / 25.0, "person")
            emitted.extend(result.alerts)
            if emitted:
                break

        self.assertEqual(len(emitted), 1)
        self.assertGreaterEqual(emitted[0].first_seen, violation_started_at - 1e-9)


class ProductionRiskGateTests(unittest.TestCase):
    def setUp(self):
        self.config = RiskConfig(
            suspect_frames=1,
            confirm_frames=1,
            recovery_frames=2,
            cooldown_seconds=10.0,
            min_violation_seconds=0.5,
            min_stable_confidence=0.6,
            min_stability=0.75,
            max_evidence_gap_seconds=0.35,
        )
        self.risk = PPERiskStateMachine(self.config, "risk-gate")

    def update(self, timestamp, *, confidence=0.9, stability=1.0, eligible=True):
        return self.risk.update(
            candidate_id=1,
            public_track_id=1,
            stable_state=PPEState.NO_HELMET,
            raw_state=PPEState.NO_HELMET,
            confidence=confidence,
            stability=stability,
            evidence_eligible=eligible,
            raw_evidence_eligible=eligible and confidence >= 0.6,
            timestamp=timestamp,
            class_id="person",
            source_model_ids=("helmet_only",),
        )

    def test_duration_boundary_and_gap_reset_are_seconds_based(self):
        self.assertEqual(self.update(0.0).state, RiskState.SUSPECT)
        self.assertEqual(self.update(0.25).alerts, ())
        self.assertEqual(self.update(0.49).alerts, ())
        at_boundary = self.update(0.50)
        self.assertEqual(len(at_boundary.alerts), 1)

        other = PPERiskStateMachine(self.config, "gap")
        other.update(
            1, PPEState.NO_HELMET, 0.9, 0.0, "person", ("helmet_only",),
            raw_state=PPEState.NO_HELMET, public_track_id=1,
        )
        other.observe_gap(1, 0.351)
        result = other.update(
            1, PPEState.NO_HELMET, 0.9, 0.36, "person", ("helmet_only",),
            raw_state=PPEState.NO_HELMET, public_track_id=1,
        )
        self.assertEqual(result.alerts, ())
        self.assertEqual(other.evidence_gap_resets, 1)

    def test_low_confidence_or_unstable_observation_cannot_advance(self):
        first = self.update(0.0)
        self.assertEqual(first.state, RiskState.SUSPECT)
        self.assertEqual(self.update(1.0, confidence=0.4, eligible=False).alerts, ())
        self.assertEqual(self.update(1.1, stability=0.5).alerts, ())
        self.assertEqual(self.risk.alerts_emitted, 0)


if __name__ == "__main__":
    unittest.main()
