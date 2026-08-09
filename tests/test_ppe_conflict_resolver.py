from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Tuple

from safety_pipeline.config import PPEConflictConfig
from safety_pipeline.ppe_conflict import PPEConflictResolver
from safety_pipeline.types import PPEState, ResolvedDetection


@dataclass(frozen=True)
class RawDetection:
    box: Tuple[float, float, float, float]
    confidence: float
    class_id: str
    source_model_ids: Tuple[str, ...] = ("combined",)


class PPEConflictResolverTests(unittest.TestCase):
    def setUp(self):
        self.resolver = PPEConflictResolver(PPEConflictConfig())

    @staticmethod
    def detection(class_id, confidence, box=(10.0, 10.0, 50.0, 50.0), model="combined"):
        return RawDetection(box, confidence, class_id, (model,))

    def test_helmet_095_nohelmet_075_becomes_one_uncertain_observation_at_strict_boundary(self):
        helmet = self.detection("hat", 0.95)
        nohelmet = self.detection("person", 0.75)
        frame = self.resolver.resolve((helmet, nohelmet))
        self.assertEqual(len(frame.detections), 1)
        result = frame.detections[0]
        self.assertIsInstance(result, ResolvedDetection)
        self.assertEqual(result.head_observation.state, PPEState.UNCERTAIN)
        self.assertAlmostEqual(result.head_observation.helmet_score, 0.95)
        self.assertAlmostEqual(result.head_observation.nohelmet_score, 0.75)
        self.assertEqual(result.head_observation.raw_detections, (helmet, nohelmet))
        self.assertEqual(frame.conflicts_resolved, 1)

    def test_score_difference_above_margin_selects_helmet(self):
        frame = self.resolver.resolve((
            self.detection("hat", 0.96),
            self.detection("person", 0.75),
        ))
        self.assertEqual(frame.detections[0].head_observation.state, PPEState.HELMET)

    def test_close_scores_are_uncertain(self):
        frame = self.resolver.resolve((
            self.detection("hat", 0.91),
            self.detection("person", 0.90),
        ))
        result = frame.detections[0]
        self.assertEqual(result.class_id, "ppe_uncertain")
        self.assertEqual(result.head_observation.state, PPEState.UNCERTAIN)

    def test_nohelmet_score_can_win(self):
        frame = self.resolver.resolve((
            self.detection("hat", 0.50),
            self.detection("person", 0.94),
        ))
        result = frame.detections[0]
        self.assertEqual(result.class_id, "person")
        self.assertEqual(result.head_observation.state, PPEState.NO_HELMET)

    def test_adjacent_people_do_not_merge(self):
        frame = self.resolver.resolve((
            self.detection("hat", 0.95, (0.0, 0.0, 30.0, 30.0)),
            self.detection("person", 0.94, (32.0, 0.0, 62.0, 30.0)),
        ))
        self.assertEqual(len(frame.detections), 2)
        self.assertEqual(frame.conflicts_resolved, 0)
        self.assertTrue(all(isinstance(item, ResolvedDetection) for item in frame.detections))

    def test_disabled_mode_is_exact_passthrough(self):
        items = (
            self.detection("hat", 0.95),
            self.detection("person", 0.75),
            self.detection("fire", 0.88),
        )
        frame = self.resolver.resolve(items, enabled=False)
        self.assertEqual(frame.detections, items)
        self.assertTrue(all(output is original for output, original in zip(frame.detections, items)))

    def test_non_ppe_detection_is_not_modified(self):
        fire = self.detection("fire", 0.88)
        frame = self.resolver.resolve((fire,))
        self.assertIs(frame.detections[0], fire)

    def test_two_pairs_are_matched_one_to_one(self):
        detections = (
            self.detection("hat", 0.95, (0.0, 0.0, 40.0, 40.0), "helmet_model"),
            self.detection("hat", 0.90, (100.0, 0.0, 140.0, 40.0), "helmet_model"),
            self.detection("person", 0.72, (1.0, 1.0, 41.0, 41.0), "ppe_model"),
            self.detection("person", 0.88, (101.0, 1.0, 141.0, 41.0), "ppe_model"),
        )
        frame = self.resolver.resolve(detections)
        self.assertEqual(frame.conflicts_resolved, 2)
        self.assertEqual(len(frame.detections), 2)
        for result in frame.detections:
            self.assertEqual(result.source_model_ids, ("helmet_model", "ppe_model"))

    def test_input_order_does_not_change_classification(self):
        helmet = self.detection("hat", 0.96)
        nohelmet = self.detection("person", 0.75)
        forward = self.resolver.resolve((helmet, nohelmet)).detections[0]
        reverse = self.resolver.resolve((nohelmet, helmet)).detections[0]
        self.assertEqual(forward.class_id, reverse.class_id)
        self.assertEqual(forward.head_observation.state, reverse.head_observation.state)
        self.assertEqual(forward.box, reverse.box)


if __name__ == "__main__":
    unittest.main()
