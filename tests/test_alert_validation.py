import math
import unittest
from dataclasses import dataclass, replace

from safety_pipeline.alert_validation import AlertEvidenceGate, AlertEvidenceState


@dataclass(frozen=True)
class Config:
    min_high_confidence_hits: int = 4
    min_duration_seconds: float = 0.6
    window_seconds: float = 1.2
    max_gap_seconds: float = 0.25
    min_presence_ratio: float = 0.6
    min_ema_confidence: float = 0.45
    ema_alpha: float = 0.35
    cooldown_seconds: float = 10.0
    max_active_events: int = 128


KEY = ("camera-01", "person", 1)


class AlertEvidenceGateTests(unittest.TestCase):
    def test_single_frame_and_predicted_or_missing_updates_never_emit(self):
        gate = AlertEvidenceGate(Config())

        first = gate.observe(KEY, 0.0, 0.95)
        predicted = gate.missing(KEY, 0.1, predicted=True)
        missing = gate.missing(KEY, 0.2)

        self.assertFalse(first.emitted)
        self.assertEqual(first.actual_hits, 1)
        self.assertEqual(predicted.actual_hits, 1)
        self.assertEqual(missing.actual_hits, 1)
        self.assertEqual(predicted.reason, "predicted_ignored")
        self.assertEqual(missing.reason, "missing_ignored")
        self.assertEqual(gate.metrics.emitted, 0)

    def test_different_track_keys_never_aggregate(self):
        gate = AlertEvidenceGate(replace(Config(), min_duration_seconds=0.0))
        keys = [("camera-01", "person", number) for number in range(1, 5)]

        decisions = [gate.observe(key, 0.0, 0.95) for key in keys]

        self.assertTrue(all(decision.actual_hits == 1 for decision in decisions))
        self.assertFalse(any(decision.emitted for decision in decisions))
        self.assertEqual(gate.active_count, 4)

    def test_sustained_high_confidence_evidence_emits_with_audit_statistics(self):
        gate = AlertEvidenceGate(Config())
        decisions = [gate.observe(KEY, timestamp, 0.8) for timestamp in (0.0, 0.2, 0.4, 0.6)]

        result = decisions[-1]
        self.assertTrue(result.emitted)
        self.assertTrue(result.confirmed)
        self.assertEqual(result.state, AlertEvidenceState.COOLDOWN)
        self.assertEqual(result.actual_hits, 4)
        self.assertAlmostEqual(result.duration, 0.6)
        self.assertAlmostEqual(result.presence_ratio, 1.0)
        self.assertAlmostEqual(result.ema, 0.8)
        self.assertEqual(result.reason, "emitted")
        self.assertEqual(gate.metrics.emitted, 1)

    def test_hit_count_without_minimum_elapsed_duration_does_not_emit(self):
        gate = AlertEvidenceGate(Config())
        result = None
        for timestamp in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
            result = gate.observe(KEY, timestamp, 0.95)

        self.assertIsNotNone(result)
        self.assertFalse(result.emitted)
        self.assertEqual(result.reason, "insufficient_duration")
        self.assertGreaterEqual(result.actual_hits, 4)

    def test_low_ema_blocks_an_otherwise_complete_candidate(self):
        gate = AlertEvidenceGate(Config())
        result = None
        for timestamp in (0.0, 0.2, 0.4, 0.6):
            result = gate.observe(KEY, timestamp, 0.40)

        self.assertIsNotNone(result)
        self.assertFalse(result.emitted)
        self.assertAlmostEqual(result.ema, 0.40)
        self.assertEqual(result.reason, "insufficient_ema")

    def test_gap_boundary_is_inclusive_but_exceeding_it_restarts_candidate(self):
        config = replace(
            Config(),
            min_high_confidence_hits=2,
            min_duration_seconds=0.0,
            min_presence_ratio=0.0,
            min_ema_confidence=0.0,
        )
        at_boundary = AlertEvidenceGate(config)
        at_boundary.observe(KEY, 0.0, 0.9)
        confirmed = at_boundary.observe(KEY, 0.25, 0.9)
        self.assertTrue(confirmed.emitted)
        self.assertFalse(confirmed.gap_reset)

        over_boundary = AlertEvidenceGate(config)
        over_boundary.observe(KEY, 0.0, 0.9)
        restarted = over_boundary.observe(KEY, 0.250001, 0.9)
        self.assertFalse(restarted.emitted)
        self.assertTrue(restarted.gap_reset)
        self.assertEqual(restarted.reason, "gap_reset")
        self.assertEqual(restarted.actual_hits, 1)
        self.assertEqual(over_boundary.metrics.suppressed, 1)

    def test_same_key_is_suppressed_during_cooldown_then_can_emit_again(self):
        config = replace(
            Config(),
            min_high_confidence_hits=2,
            min_duration_seconds=0.1,
            window_seconds=20.0,
            max_gap_seconds=20.0,
            cooldown_seconds=1.0,
        )
        gate = AlertEvidenceGate(config)
        gate.observe(KEY, 0.0, 0.9)
        first = gate.observe(KEY, 0.1, 0.9)
        during = gate.observe(KEY, 0.5, 0.9)
        after = gate.observe(KEY, 1.1, 0.9)

        self.assertTrue(first.emitted)
        self.assertFalse(during.emitted)
        self.assertEqual(during.reason, "cooldown_active")
        self.assertAlmostEqual(during.cooldown_remaining, 0.6)
        self.assertTrue(after.emitted)
        self.assertEqual(gate.metrics.emitted, 2)
        self.assertGreaterEqual(gate.metrics.suppressed, 1)

    def test_capacity_evicts_oldest_candidate_but_never_confirmed_event(self):
        config = replace(
            Config(),
            min_high_confidence_hits=2,
            min_duration_seconds=0.0,
            min_presence_ratio=0.0,
            min_ema_confidence=0.0,
            max_gap_seconds=100.0,
            cooldown_seconds=100.0,
            max_active_events=2,
        )
        gate = AlertEvidenceGate(config)
        confirmed_key = ("camera-01", "fire", "confirmed")
        candidate_key = ("camera-01", "fire", "candidate")
        replacement_key = ("camera-01", "fire", "replacement")
        rejected_key = ("camera-01", "fire", "rejected")

        gate.observe(confirmed_key, 0.0, 0.9)
        self.assertTrue(gate.observe(confirmed_key, 0.1, 0.9).emitted)
        gate.observe(candidate_key, 0.2, 0.9)
        admitted = gate.observe(replacement_key, 0.3, 0.9)

        self.assertEqual(admitted.actual_hits, 1)
        self.assertEqual(gate.active_count, 2)
        protected = gate.observe(confirmed_key, 0.4, 0.9)
        self.assertTrue(protected.confirmed)
        self.assertEqual(protected.reason, "cooldown_active")

        # Confirm the replacement too; with only protected events left, a new
        # key is rejected rather than evicting either confirmed event.
        gate.observe(replacement_key, 0.5, 0.9)
        rejected = gate.observe(rejected_key, 0.6, 0.9)
        self.assertEqual(rejected.reason, "capacity_rejected")
        self.assertEqual(rejected.actual_hits, 0)
        self.assertEqual(gate.active_count, 2)
        self.assertGreaterEqual(gate.metrics.dropped, 2)

    def test_cleanup_expires_inactive_candidate_and_counts_suppression(self):
        gate = AlertEvidenceGate(Config())
        gate.observe(KEY, 0.0, 0.9)

        expired = gate.cleanup(0.250001)

        self.assertEqual(expired, (KEY,))
        self.assertEqual(gate.metrics.expired, 1)
        self.assertEqual(gate.metrics.suppressed, 1)
        self.assertEqual(gate.metrics.active_events, 0)

    def test_invalid_timestamp_and_confidence_fail_safe_without_mutation(self):
        gate = AlertEvidenceGate(Config())

        invalid_time = gate.observe(KEY, math.nan, 0.9)
        invalid_confidence = gate.observe(KEY, 0.0, math.inf)
        valid = gate.observe(KEY, 1.0, 0.9)
        regression = gate.observe(KEY, 0.5, 0.9)

        self.assertFalse(invalid_time.emitted)
        self.assertEqual(invalid_time.reason, "invalid_timestamp")
        self.assertFalse(invalid_confidence.emitted)
        self.assertEqual(invalid_confidence.reason, "invalid_confidence")
        self.assertEqual(valid.actual_hits, 1)
        self.assertFalse(regression.emitted)
        self.assertEqual(regression.reason, "timestamp_regression")
        self.assertEqual(gate.observe(KEY, 1.1, 0.9).actual_hits, 2)
        self.assertEqual(gate.metrics.dropped, 3)


if __name__ == "__main__":
    unittest.main()
