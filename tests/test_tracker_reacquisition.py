from __future__ import annotations

import unittest

from safety_pipeline.config import TrackingConfig
from safety_pipeline.tracking import HeadTracker
from safety_pipeline.types import HeadObservation, PPEState


def head(box, state=PPEState.HELMET, confidence=0.9):
    return HeadObservation(
        box=tuple(float(value) for value in box),
        state=state,
        helmet_score=confidence if state == PPEState.HELMET else None,
        nohelmet_score=confidence if state == PPEState.NO_HELMET else None,
        confidence=confidence,
        raw_detections=(),
        source_model_ids=("combined",),
        resolution="test",
    )


def motion_config(**overrides):
    values = dict(
        iou_threshold=0.5,
        min_hits=2,
        smoothing_alpha=1.0,
        max_lost_frames=None,
        max_lost_seconds=1.0,
        tentative_max_lost_seconds=0.2,
        reacquire_enabled=True,
        reacquire_min_iou=0.1,
        reacquire_max_center_distance_ratio=0.8,
        reacquire_min_area_ratio=0.5,
        reacquire_ambiguity_margin=0.1,
        motion_max_seconds=0.5,
        velocity_alpha=1.0,
        public_id_policy="confirmed_only",
    )
    values.update(overrides)
    return TrackingConfig(**values)


class TrackerIdentityTests(unittest.TestCase):
    def test_default_tentative_survives_sparse_detection_until_confirmation(self):
        tracker = HeadTracker(TrackingConfig())

        first = tracker.update((head((0, 0, 40, 40)),), 0.0).visible[0]
        self.assertIsNone(first.track_id)
        self.assertEqual(tracker.update((), 0.2).missing_candidate_ids, (1,))
        confirmed = tracker.update((head((1, 0, 41, 40)),), 0.4).visible[0]

        self.assertEqual(confirmed.candidate_id, first.candidate_id)
        self.assertEqual(confirmed.track_id, 1)
        self.assertTrue(confirmed.confirmed)

    def test_tentative_candidates_do_not_consume_public_ids(self):
        tracker = HeadTracker(motion_config(
            reacquire_enabled=False,
            tentative_max_lost_seconds=0.01,
        ))

        first = tracker.update((head((0, 0, 40, 40)),), 0.0).visible[0]
        self.assertEqual(first.candidate_id, 1)
        self.assertIsNone(first.track_id)
        self.assertEqual(tracker.active_count, 0)
        self.assertEqual(tracker.candidate_count, 1)
        self.assertEqual(tracker.update((), 0.02).retired_candidate_ids, (1,))

        second = tracker.update((head((100, 0, 140, 40)),), 0.03).visible[0]
        self.assertEqual(second.candidate_id, 2)
        self.assertIsNone(second.track_id)
        self.assertEqual(tracker.update((), 0.05).retired_candidate_ids, (2,))

        tentative = tracker.update((head((200, 0, 240, 40)),), 0.06).visible[0]
        confirmed = tracker.update((head((201, 0, 241, 40)),), 0.065).visible[0]
        self.assertEqual(tentative.candidate_id, 3)
        self.assertIsNone(tentative.track_id)
        self.assertEqual(confirmed.candidate_id, 3)
        self.assertEqual(confirmed.track_id, 1)
        self.assertTrue(confirmed.confirmed)
        self.assertEqual(tracker.active_count, 1)
        self.assertEqual(tracker.candidate_count, 1)
        self.assertEqual(tracker.confirmed_count, 1)

    def test_simultaneous_promotions_use_spatial_not_detection_order(self):
        def public_ids(initial, confirmation):
            tracker = HeadTracker(motion_config(reacquire_enabled=False))
            tracker.update(tuple(head(box) for box in initial), 0.0)
            visible = tracker.update(tuple(head(box) for box in confirmation), 0.1).visible
            return {
                round(item.box[0]): item.track_id
                for item in visible
            }

        left = (0, 0, 40, 40)
        right = (100, 0, 140, 40)
        self.assertEqual(public_ids((left, right), (right, left)), {0: 1, 100: 2})
        self.assertEqual(public_ids((right, left), (left, right)), {0: 1, 100: 2})

    def test_schema_one_assigns_public_id_immediately_and_uses_frame_lifetime(self):
        config = TrackingConfig.from_mapping(
            {"min_hits": 3, "max_lost_frames": 2},
            schema_version=1,
        )
        tracker = HeadTracker(config)

        first = tracker.update((head((0, 0, 40, 40)),), 0.0).visible[0]
        self.assertEqual(first.candidate_id, 1)
        self.assertEqual(first.track_id, 1)
        self.assertFalse(first.confirmed)
        self.assertEqual(tracker.active_count, 1)

        self.assertEqual(tracker.update((), 100.0).missing_candidate_ids, (1,))
        self.assertEqual(tracker.update((), 200.0).missing_candidate_ids, (1,))
        retired = tracker.update((), 300.0)
        self.assertEqual(retired.retired_candidate_ids, (1,))
        self.assertEqual(tracker.active_count, 0)
        self.assertEqual(tracker.candidate_count, 0)


class TrackerReacquisitionTests(unittest.TestCase):
    def test_ambiguous_strict_iou_after_gap_does_not_inherit_old_public_id(self):
        tracker = HeadTracker(motion_config(iou_threshold=0.3))
        left = head((0, 0, 40, 40))
        right = head((20, 0, 60, 40))
        tracker.update((left, right), 0.0)
        confirmed = tracker.update((left, right), 0.1)
        self.assertEqual({item.track_id for item in confirmed.visible}, {1, 2})
        tracker.update((), 0.2)

        ambiguous = tracker.update((head((10, 0, 50, 40)),), 0.3)

        self.assertEqual(ambiguous.missing_candidate_ids, (1, 2))
        self.assertEqual(len(ambiguous.visible), 1)
        self.assertEqual(ambiguous.visible[0].candidate_id, 3)
        self.assertIsNone(ambiguous.visible[0].track_id)

    def test_predicted_box_recaptures_after_one_low_confidence_miss(self):
        tracker = HeadTracker(motion_config())
        initial = tracker.update((head((0, 0, 40, 40)),), 0.0).visible[0]
        confirmed = tracker.update((head((10, 0, 50, 40)),), 0.1).visible[0]
        self.assertEqual(initial.candidate_id, confirmed.candidate_id)
        self.assertEqual(confirmed.track_id, 1)

        missing = tracker.update((), 0.2)
        self.assertEqual(missing.missing_candidate_ids, (1,))
        reacquired = tracker.update((head((30, 0, 70, 40)),), 0.3).visible[0]

        self.assertEqual(reacquired.candidate_id, 1)
        self.assertEqual(reacquired.track_id, 1)
        self.assertEqual(reacquired.hits, 3)
        self.assertEqual(tracker.candidate_count, 1)

    def test_strict_match_is_locked_before_relaxed_reacquisition(self):
        tracker = HeadTracker(motion_config())
        first = tracker.update((
            head((-10, 0, 30, 40)),
            head((20, 0, 60, 40)),
        ), 0.0)
        second = tracker.update((
            head((0, 0, 40, 40)),
            head((20, 0, 60, 40)),
        ), 0.1)
        public_by_candidate = {
            item.candidate_id: item.track_id
            for item in second.visible
        }
        self.assertEqual(public_by_candidate, {1: 1, 2: 2})

        tracker.update((), 0.2)
        result = tracker.update((head((20, 0, 60, 40)),), 0.3)

        self.assertEqual(len(result.visible), 1)
        self.assertEqual(result.visible[0].candidate_id, 2)
        self.assertEqual(result.visible[0].track_id, 2)
        self.assertEqual(result.missing_candidate_ids, (1,))

    def test_ambiguous_motion_match_creates_a_new_tentative_candidate(self):
        tracker = HeadTracker(motion_config(
            iou_threshold=0.3,
            reacquire_ambiguity_margin=0.05,
        ))
        tracker.update((
            head((-30, 0, 10, 40)),
            head((70, 0, 110, 40)),
        ), 0.0)
        tracker.update((
            head((-10, 0, 30, 40)),
            head((50, 0, 90, 40)),
        ), 0.1)
        tracker.update((), 0.2)

        result = tracker.update((head((20, 0, 60, 40)),), 0.3)

        self.assertEqual(len(result.visible), 1)
        self.assertEqual(result.visible[0].candidate_id, 3)
        self.assertIsNone(result.visible[0].track_id)
        self.assertEqual(result.missing_candidate_ids, (1, 2))
        self.assertEqual(tracker.active_count, 2)
        self.assertEqual(tracker.candidate_count, 3)

    def test_motion_reacquisition_stops_at_configured_horizon(self):
        tracker = HeadTracker(motion_config(motion_max_seconds=0.15))
        tracker.update((head((0, 0, 40, 40)),), 0.0)
        tracker.update((head((10, 0, 50, 40)),), 0.1)
        tracker.update((), 0.2)

        result = tracker.update((head((40, 0, 80, 40)),), 0.4)

        self.assertEqual(result.visible[0].candidate_id, 2)
        self.assertIsNone(result.visible[0].track_id)
        self.assertEqual(result.missing_candidate_ids, (1,))

    def test_relaxed_match_obeys_center_distance_and_area_gates(self):
        cases = (
            (
                "center distance",
                {"reacquire_max_center_distance_ratio": 0.5},
                (25, 0, 65, 40),
            ),
            (
                "area ratio",
                {"reacquire_min_area_ratio": 0.65},
                (10, 10, 30, 30),
            ),
        )
        for name, overrides, candidate_box in cases:
            with self.subTest(gate=name):
                tracker = HeadTracker(motion_config(**overrides))
                tracker.update((head((0, 0, 40, 40)),), 0.0)
                tracker.update((head((0, 0, 40, 40)),), 0.1)
                tracker.update((), 0.2)

                result = tracker.update((head(candidate_box),), 0.3)

                self.assertEqual(result.visible[0].candidate_id, 2)
                self.assertIsNone(result.visible[0].track_id)
                self.assertEqual(result.missing_candidate_ids, (1,))

    def test_zero_reacquire_iou_can_rely_on_other_geometry_gates(self):
        tracker = HeadTracker(motion_config(
            reacquire_min_iou=0.0,
            reacquire_max_center_distance_ratio=1.1,
        ))
        tracker.update((head((0, 0, 40, 40)),), 0.0)
        confirmed = tracker.update((head((0, 0, 40, 40)),), 0.1).visible[0]
        tracker.update((), 0.2)

        result = tracker.update((head((41, 0, 81, 40)),), 0.3)

        self.assertEqual(result.visible[0].candidate_id, confirmed.candidate_id)
        self.assertEqual(result.visible[0].track_id, confirmed.track_id)


class TrackerLifetimeTests(unittest.TestCase):
    def test_confirmed_lifetime_depends_on_elapsed_seconds_not_frame_count(self):
        tracker = HeadTracker(motion_config(
            min_hits=1,
            max_lost_seconds=0.5,
            tentative_max_lost_seconds=0.1,
        ))
        tracked = tracker.update((head((0, 0, 40, 40)),), 0.0).visible[0]
        for timestamp in (0.1, 0.2, 0.3, 0.4, 0.5):
            frame = tracker.update((), timestamp)
            self.assertIn(tracked.candidate_id, frame.missing_candidate_ids)

        retired = tracker.update((), 0.501)
        self.assertEqual(retired.retired_candidate_ids, (tracked.candidate_id,))

    def test_tentative_track_uses_shorter_seconds_lifetime(self):
        tracker = HeadTracker(motion_config(
            min_hits=2,
            max_lost_seconds=1.0,
            tentative_max_lost_seconds=0.1,
        ))
        tentative = tracker.update((head((0, 0, 40, 40)),), 0.0).visible[0]
        self.assertEqual(tracker.update((), 0.1).missing_candidate_ids, (tentative.candidate_id,))
        self.assertEqual(
            tracker.update((), 0.101).retired_candidate_ids,
            (tentative.candidate_id,),
        )

    def test_non_monotonic_timestamp_does_not_rewind_lifecycle(self):
        tracker = HeadTracker(motion_config(
            min_hits=2,
            max_lost_seconds=0.1,
            tentative_max_lost_seconds=0.1,
        ))
        tracker.update((head((0, 0, 40, 40)),), 10.0)
        confirmed = tracker.update((head((1, 0, 41, 40)),), 9.0).visible[0]
        self.assertEqual(confirmed.last_seen, 10.0)
        self.assertEqual(confirmed.track_id, 1)
        self.assertEqual(tracker.update((), 9.0).missing_candidate_ids, (1,))
        self.assertEqual(tracker.update((), 10.101).retired_candidate_ids, (1,))

    def test_non_finite_timestamp_is_rejected(self):
        tracker = HeadTracker(motion_config())
        with self.assertRaisesRegex(ValueError, "finite number"):
            tracker.update((), float("nan"))


if __name__ == "__main__":
    unittest.main()
