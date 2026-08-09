from __future__ import annotations

import unittest

from safety_pipeline.config import PPEConflictConfig
from safety_pipeline.policy import ppe_evidence_enabled, stable_class_visible
from safety_pipeline.ppe_conflict import PPEConflictResolver
from safety_pipeline.spatial import filter_monitoring_domain
from safety_pipeline.types import PPEState


class RawDetection:
    def __init__(self, class_id, confidence, box):
        self.class_id = class_id
        self.confidence = confidence
        self.box = box
        self.source_model_ids = ("combined",)


class MonitoringDomainTests(unittest.TestCase):
    def test_raw_roi_filter_prevents_outside_box_from_moving_merged_center(self):
        resolver = PPEConflictResolver(PPEConflictConfig())
        inside_nohelmet = RawDetection("person", 0.95, (29.9, 0.0, 69.9, 40.0))
        outside_helmet = RawDetection("hat", 0.74, (35.0, 0.0, 75.0, 40.0))
        roi = ((0.0, -10.0), (50.0, -10.0), (50.0, 60.0), (0.0, 60.0))

        spatial = filter_monitoring_domain((inside_nohelmet, outside_helmet), rois=(roi,))
        self.assertEqual(spatial, (inside_nohelmet,))
        resolution = resolver.resolve(spatial)
        self.assertEqual(len(resolution.detections), 1)
        self.assertEqual(resolution.detections[0].head_observation.state, PPEState.NO_HELMET)
        self.assertLess(sum(resolution.detections[0].box[::2]) / 2.0, 50.0)

    def test_mask_excludes_raw_detection(self):
        item = RawDetection("person", 0.9, (10.0, 10.0, 30.0, 30.0))
        mask = ((0.0, 0.0), (40.0, 0.0), (40.0, 40.0), (0.0, 40.0))
        self.assertEqual(filter_monitoring_domain((item,), masks=(mask,)), ())


class EvidenceVisibilityPolicyTests(unittest.TestCase):
    def test_either_ppe_switch_keeps_full_evidence_path_enabled(self):
        self.assertTrue(ppe_evidence_enabled({"hat": False, "person": True}, "hat", "person"))
        self.assertTrue(ppe_evidence_enabled({"hat": True, "person": False}, "hat", "person"))
        self.assertFalse(ppe_evidence_enabled({"hat": False, "person": False}, "hat", "person"))

    def test_stable_output_obeys_its_own_switch(self):
        enabled = {"hat": False, "person": True}
        self.assertFalse(stable_class_visible("hat", enabled, "hat", "person", "ppe_uncertain"))
        self.assertTrue(stable_class_visible("person", enabled, "hat", "person", "ppe_uncertain"))
        self.assertTrue(stable_class_visible("ppe_uncertain", enabled, "hat", "person", "ppe_uncertain"))


if __name__ == "__main__":
    unittest.main()
