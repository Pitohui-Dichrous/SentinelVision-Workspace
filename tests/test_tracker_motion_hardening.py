from __future__ import annotations

import math
import unittest

from safety_pipeline.config import TrackingConfig
from safety_pipeline.tracking import HeadTracker
from safety_pipeline.types import HeadObservation, PPEState


def head(box, label="A", confidence=0.90):
    return HeadObservation(
        box=tuple(float(value) for value in box),
        state=PPEState.HELMET,
        helmet_score=float(confidence),
        nohelmet_score=None,
        confidence=float(confidence),
        raw_detections=(),
        source_model_ids=("combined",),
        resolution=label,
    )


def hardened_config(**overrides):
    values = dict(
        iou_threshold=0.30,
        min_hits=2,
        smoothing_alpha=0.70,
        max_lost_frames=None,
        max_lost_seconds=1.0,
        tentative_max_lost_seconds=0.40,
        reacquire_enabled=True,
        reacquire_min_iou=0.10,
        reacquire_max_center_distance_ratio=0.65,
        reacquire_min_area_ratio=0.65,
        reacquire_ambiguity_margin=0.10,
        motion_max_seconds=0.85,
        velocity_alpha=0.65,
        public_id_policy="confirmed_only",
        motion_model="alpha_beta",
        position_gain=0.75,
        velocity_gain=0.20,
        velocity_decay_per_second=0.35,
        min_velocity_dt_seconds=0.02,
        max_center_speed_ratio=8.0,
        max_tentative_candidates=8,
        association_min_confidence=0.20,
        new_candidate_min_confidence=0.45,
        publication_min_seconds=0.0,
        publication_min_confidence=0.0,
        confidence_alpha=0.35,
    )
    values.update(overrides)
    return TrackingConfig(**values)


class AlphaBetaAssociationTests(unittest.TestCase):
    def test_crossing_after_gap_uses_predicted_identity_not_stale_display_box(self):
        tracker = HeadTracker(hardened_config())
        tracker.update((
            head((0, 0, 40, 40), "A"),
            head((60, 0, 100, 40), "B"),
        ), 0.0)
        confirmed = tracker.update((
            head((20, 0, 60, 40), "A"),
            head((40, 0, 80, 40), "B"),
        ), 0.1)
        public_ids = {
            item.observation.resolution: item.track_id
            for item in confirmed.visible
        }
        tracker.update((), 0.2)

        returned = tracker.update((
            head((60, 0, 100, 40), "A"),
            head((0, 0, 40, 40), "B"),
        ), 0.3)

        self.assertEqual(
            {
                item.observation.resolution: item.track_id
                for item in returned.visible
            },
            public_ids,
        )
        self.assertEqual(tracker.stats["reacquire"], 2)
        self.assertEqual(tracker.stats["reacquire_successes"], 2)
        self.assertGreaterEqual(tracker.stats["reacquire_attempts"], 2)
        self.assertEqual(tracker.candidate_count, 2)

    def test_constant_motion_reacquires_after_six_tenths_of_a_second(self):
        tracker = HeadTracker(hardened_config())
        tracker.update((head((0, 0, 40, 40)),), 0.0)
        confirmed = tracker.update((head((10, 0, 50, 40)),), 0.1).visible[0]
        tracker.update((), 0.2)

        returned = tracker.update((head((70, 0, 110, 40)),), 0.7).visible[0]

        self.assertEqual(returned.candidate_id, confirmed.candidate_id)
        self.assertEqual(returned.track_id, confirmed.track_id)
        self.assertEqual(tracker.candidate_count, 1)

    def test_tiny_timestamp_delta_is_speed_limited(self):
        tracker = HeadTracker(hardened_config())
        tracker.update((head((0, 0, 40, 40)),), 0.0)
        confirmed = tracker.update((head((10, 0, 50, 40)),), 0.001).visible[0]
        velocity = tracker._tracks[confirmed.candidate_id].velocity
        self.assertLessEqual(math.hypot(velocity[0], velocity[1]), 320.0 + 1e-9)
        tracker.update((), 0.002)

        returned = tracker.update((head((40, 0, 80, 40)),), 0.102).visible[0]

        self.assertEqual(returned.candidate_id, confirmed.candidate_id)
        self.assertEqual(returned.track_id, confirmed.track_id)

    def test_size_jitter_is_smoothed_but_never_extrapolated(self):
        tracker = HeadTracker(hardened_config())
        tracker.update((head((0, 0, 40, 40)),), 0.0)
        confirmed = tracker.update((head((10, -10, 70, 50)),), 0.1).visible[0]
        tracker.update((), 0.2)

        returned = tracker.update((head((80, 0, 120, 40)),), 0.5).visible[0]
        width = returned.box[2] - returned.box[0]
        height = returned.box[3] - returned.box[1]

        self.assertEqual(returned.candidate_id, confirmed.candidate_id)
        self.assertAlmostEqual(width, 44.2, places=6)
        self.assertAlmostEqual(height, 44.2, places=6)

    def test_relaxed_reacquisition_stops_beyond_motion_horizon(self):
        tracker = HeadTracker(hardened_config(
            motion_max_seconds=0.15,
            reacquire_max_center_distance_ratio=0.80,
        ))
        tracker.update((head((0, 0, 40, 40)),), 0.0)
        confirmed = tracker.update((head((10, 0, 50, 40)),), 0.1).visible[0]
        tracker.update((), 0.2)

        returned = tracker.update((head((50, 0, 90, 40)),), 0.4).visible[0]

        self.assertNotEqual(returned.candidate_id, confirmed.candidate_id)
        self.assertIsNone(returned.track_id)
        self.assertEqual(tracker.stats["reacquire_successes"], 0)

    def test_high_confidence_relaxed_match_preserves_publication_waiting_candidate(self):
        tracker = HeadTracker(hardened_config(
            publication_min_seconds=0.20,
            reacquire_max_center_distance_ratio=0.80,
        ))
        tracker.update((head((0, 0, 40, 40)),), 0.0)
        waiting = tracker.update((head((10, 0, 50, 40)),), 0.01).visible[0]
        self.assertTrue(waiting.confirmed)
        self.assertIsNone(waiting.track_id)
        tracker.update((), 0.05)

        returned = tracker.update((head((60, 0, 100, 40)),), 0.10).visible[0]

        self.assertEqual(returned.candidate_id, waiting.candidate_id)
        self.assertIsNone(returned.track_id)
        self.assertEqual(tracker.candidate_count, 1)
        self.assertEqual(tracker.stats["reacquire_successes"], 1)


class ConfidenceAdmissionTests(unittest.TestCase):
    def test_low_confidence_cannot_continue_publication_waiting_candidate(self):
        tracker = HeadTracker(hardened_config(
            association_min_confidence=0.20,
            new_candidate_min_confidence=0.60,
            publication_min_seconds=0.20,
        ))
        tracker.update((head((0, 0, 40, 40), confidence=0.90),), 0.0)
        waiting = tracker.update(
            (head((2, 0, 42, 40), confidence=0.90),),
            0.01,
        ).visible[0]

        continued = tracker.update(
            (head((4, 0, 44, 40), confidence=0.30),),
            0.02,
        )

        self.assertTrue(waiting.confirmed)
        self.assertIsNone(waiting.track_id)
        self.assertEqual(continued.visible, ())
        self.assertEqual(tracker.candidate_count, 1)
        self.assertEqual(tracker.stats["low_confidence_dropped"], 1)

    def test_stale_low_confidence_box_cannot_steal_track_from_true_high_box(self):
        tracker = HeadTracker(hardened_config(
            association_min_confidence=0.20,
            new_candidate_min_confidence=0.45,
        ))
        tracker.update((head((0, 0, 40, 40), "true", 0.90),), 0.0)
        tracker.update((head((2, 0, 42, 40), "true", 0.90),), 0.1)
        confirmed = tracker.update(
            (head((4, 0, 44, 40), "true", 0.90),),
            0.2,
        ).visible[0]

        result = tracker.update((
            head((10, 0, 50, 40), "high-true", 0.90),
            head((6, 0, 46, 40), "low-stale", 0.30),
        ), 0.3)

        self.assertEqual(len(result.visible), 1)
        self.assertEqual(result.visible[0].candidate_id, confirmed.candidate_id)
        self.assertEqual(result.visible[0].track_id, confirmed.track_id)
        self.assertEqual(result.visible[0].observation.resolution, "high-true")
        self.assertTrue(result.visible[0].evidence_eligible)
        self.assertEqual(tracker.candidate_count, 1)
        self.assertEqual(tracker.stats["low_confidence_dropped"], 1)

    def test_low_confidence_can_continue_confirmed_track_without_evidence(self):
        tracker = HeadTracker(hardened_config(
            association_min_confidence=0.20,
            new_candidate_min_confidence=0.60,
        ))
        empty = tracker.update((head((0, 0, 40, 40), confidence=0.30),), 0.0)
        self.assertEqual(empty.visible, ())
        self.assertEqual(tracker.candidate_count, 0)

        tracker.update((head((0, 0, 40, 40), confidence=0.90),), 0.1)
        confirmed = tracker.update(
            (head((10, 0, 50, 40), confidence=0.90),),
            0.2,
        ).visible[0]
        tracker.update((), 0.3)

        continued = tracker.update(
            (head((30, 0, 70, 40), confidence=0.30),),
            0.4,
        ).visible[0]

        self.assertEqual(continued.track_id, confirmed.track_id)
        self.assertFalse(continued.evidence_eligible)
        self.assertEqual(tracker.candidate_count, 1)

    def test_low_confidence_far_from_confirmed_track_does_not_start_candidate(self):
        tracker = HeadTracker(hardened_config(
            association_min_confidence=0.20,
            new_candidate_min_confidence=0.60,
        ))
        tracker.update((head((0, 0, 40, 40), confidence=0.90),), 0.0)
        tracker.update((head((5, 0, 45, 40), confidence=0.90),), 0.1)

        result = tracker.update(
            (head((500, 0, 540, 40), confidence=0.30),),
            0.2,
        )

        self.assertEqual(result.visible, ())
        self.assertEqual(tracker.candidate_count, 1)
        self.assertEqual(tracker.stats["dropped"], 1)

    def test_publication_requires_hits_duration_and_confidence(self):
        tracker = HeadTracker(hardened_config(
            min_hits=2,
            publication_min_seconds=0.20,
            publication_min_confidence=0.80,
        ))
        first = tracker.update((head((0, 0, 40, 40), confidence=0.90),), 0.0).visible[0]
        second = tracker.update((head((2, 0, 42, 40), confidence=0.90),), 0.1).visible[0]
        third = tracker.update((head((4, 0, 44, 40), confidence=0.90),), 0.2).visible[0]

        self.assertIsNone(first.track_id)
        self.assertIsNone(second.track_id)
        self.assertTrue(second.confirmed)
        self.assertEqual(third.track_id, 1)
        self.assertEqual(tracker.stats["promoted"], 1)

    def test_runtime_high_threshold_controls_candidate_creation(self):
        tracker = HeadTracker(hardened_config(
            new_candidate_min_confidence=0.45,
        ))

        rejected = tracker.update(
            (head((0, 0, 40, 40), confidence=0.55),),
            0.0,
            high_confidence_threshold=0.60,
        )
        accepted = tracker.update(
            (head((0, 0, 40, 40), confidence=0.70),),
            0.1,
            high_confidence_threshold=0.60,
        )

        self.assertEqual(rejected.visible, ())
        self.assertEqual(len(accepted.visible), 1)
        self.assertEqual(tracker.candidate_count, 1)


class CapacityAndStatisticsTests(unittest.TestCase):
    def test_publication_waiting_candidates_cannot_escape_private_capacity(self):
        tracker = HeadTracker(hardened_config(
            min_hits=2,
            max_tentative_candidates=128,
            publication_min_seconds=0.20,
        ))
        first_burst = tuple(
            head((index * 100, 0, index * 100 + 40, 40), str(index), 0.90)
            for index in range(128)
        )
        second_burst = tuple(
            head((20000 + index * 100, 0, 20040 + index * 100, 40), str(index), 0.90)
            for index in range(128)
        )

        tracker.update(first_burst, 0.0)
        waiting = tracker.update(first_burst, 0.01)
        rejected = tracker.update(second_burst, 0.02)

        self.assertEqual(len(waiting.visible), 128)
        self.assertTrue(all(item.confirmed for item in waiting.visible))
        self.assertTrue(all(item.track_id is None for item in waiting.visible))
        self.assertEqual(rejected.visible, ())
        self.assertEqual(tracker.candidate_count, 128)
        self.assertEqual(tracker.active_count, 0)
        self.assertEqual(tracker.stats["created"], 128)
        self.assertEqual(tracker.stats["capacity_dropped"], 128)
        self.assertEqual(tracker.stats["reacquire_attempts"], 0)

    def test_tentative_capacity_never_evicts_confirmed_tracks(self):
        tracker = HeadTracker(hardened_config(
            min_hits=3,
            max_tentative_candidates=2,
        ))
        first = tracker.update(tuple(
            head((index * 100, 0, index * 100 + 40, 40), str(index), 0.90 - index * 0.01)
            for index in range(5)
        ), 0.0)
        self.assertEqual(len(first.visible), 2)
        self.assertEqual(tracker.candidate_count, 2)
        self.assertEqual(tracker.stats["dropped"], 3)

        boxes = tuple(item.observation for item in first.visible)
        tracker.update(boxes, 0.1)
        confirmed = tracker.update(boxes, 0.2)
        confirmed_ids = {item.track_id for item in confirmed.visible}
        self.assertEqual(len(confirmed_ids), 2)

        result = tracker.update(
            boxes + (
                head((500, 0, 540, 40), "new-1", 0.90),
                head((600, 0, 640, 40), "new-2", 0.89),
                head((700, 0, 740, 40), "new-3", 0.88),
            ),
            0.3,
        )

        self.assertTrue(confirmed_ids.issubset({item.track_id for item in result.visible}))
        self.assertEqual(tracker.active_count, 2)
        self.assertEqual(tracker.candidate_count, 4)
        self.assertEqual(tracker.stats["created"], 4)
        self.assertEqual(tracker.stats["highwater"], 4)
        self.assertEqual(tracker.stats["dropped"], 4)

    def test_invalid_observation_is_counted_without_allocating_state(self):
        tracker = HeadTracker(hardened_config())

        result = tracker.update((head((0, 0, float("nan"), 40)),), 0.0)

        self.assertEqual(result.visible, ())
        self.assertEqual(tracker.candidate_count, 0)
        self.assertEqual(tracker.stats["invalid"], 1)

    def test_unpublished_candidate_uses_tentative_retirement_and_metrics(self):
        tracker = HeadTracker(hardened_config(
            tentative_max_lost_seconds=0.05,
            publication_min_seconds=0.20,
        ))
        tracker.update((head((0, 0, 40, 40)),), 0.0)
        waiting = tracker.update((head((2, 0, 42, 40)),), 0.01).visible[0]
        self.assertTrue(waiting.confirmed)
        self.assertIsNone(waiting.track_id)

        retired = tracker.update((), 0.061)

        self.assertEqual(retired.retired_candidate_ids, (waiting.candidate_id,))
        self.assertEqual(tracker.stats["tentative_retired"], 1)
        self.assertEqual(tracker.stats["confirmed_retired"], 0)


if __name__ == "__main__":
    unittest.main()
