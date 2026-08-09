from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from safety_pipeline.config import (
    ConfigError,
    SafetyPipelineConfig,
    config_fingerprint,
    load_safety_pipeline_config,
    load_safety_pipeline_config_checked,
)


class SafetyPipelineConfigTests(unittest.TestCase):
    def test_old_or_missing_runtime_override_uses_baseline_default(self):
        config = SafetyPipelineConfig.from_mapping({"schema_version": 1})
        self.assertEqual(config.default_mode, "baseline")
        self.assertEqual(config.conflict.protected_class_id, "hat")
        self.assertEqual(config.conflict.unprotected_class_id, "person")

    def test_invalid_temporal_window_is_rejected(self):
        with self.assertRaises(ConfigError):
            SafetyPipelineConfig.from_mapping({
                "schema_version": 1,
                "ppe": {"temporal": {"window_size": 4, "min_votes": 5}},
            })

    def test_invalid_file_falls_back_without_interrupting_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("schema_version: 99\n", encoding="utf-8")
            with self.assertLogs("safety_pipeline.config", level="WARNING"):
                config = load_safety_pipeline_config(path)
        self.assertEqual(config.default_mode, "baseline")
        self.assertEqual(config.schema_version, 2)

    def test_checked_load_marks_invalid_file_for_fail_closed_ui_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("default_mode: ppe_temporal\nppe: [invalid]\n", encoding="utf-8")
            with self.assertLogs("safety_pipeline.config", level="WARNING"):
                config, valid = load_safety_pipeline_config_checked(path)
        self.assertFalse(valid)
        self.assertEqual(config.default_mode, "baseline")

    def test_checked_load_accepts_valid_tracked_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "valid.yaml"
            path.write_text("schema_version: 1\ndefault_mode: ppe_temporal\n", encoding="utf-8")
            config, valid = load_safety_pipeline_config_checked(path)
        self.assertTrue(valid)
        self.assertEqual(config.default_mode, "ppe_temporal")

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ConfigError):
            SafetyPipelineConfig.from_mapping({"schema_version": 1, "default_mode": "magic"})

    def test_non_finite_numbers_are_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                SafetyPipelineConfig.from_mapping({
                    "schema_version": 1,
                    "ppe": {"conflict": {"min_iou": value}},
                })

    def test_schema_one_preserves_legacy_tracking_contract(self):
        config = SafetyPipelineConfig.from_mapping({
            "schema_version": 1,
            "ppe": {"tracking": {"max_lost_frames": 9, "min_hits": 3}},
        })
        self.assertEqual(config.tracking.max_lost_frames, 9)
        self.assertIsNone(config.tracking.max_lost_seconds)
        self.assertFalse(config.tracking.reacquire_enabled)
        self.assertEqual(config.tracking.public_id_policy, "immediate")

    def test_schema_two_loads_time_based_reacquisition(self):
        config = SafetyPipelineConfig.from_mapping({
            "schema_version": 2,
            "ppe": {
                "tracking": {
                    "loss_tolerance": {"unit": "seconds", "value": 0.8},
                    "tentative_loss_tolerance": {"unit": "seconds", "value": 0.1},
                    "reacquisition": {
                        "enabled": True,
                        "min_iou": 0.12,
                        "max_center_distance_ratio": 0.7,
                        "min_area_ratio": 0.6,
                        "ambiguity_margin": 0.08,
                        "motion_max_seconds": 0.4,
                        "velocity_alpha": 0.5,
                    },
                    "public_id_policy": "confirmed_only",
                },
            },
        })
        tracking = config.tracking
        self.assertIsNone(tracking.max_lost_frames)
        self.assertEqual(tracking.max_lost_seconds, 0.8)
        self.assertEqual(tracking.tentative_max_lost_seconds, 0.1)
        self.assertTrue(tracking.reacquire_enabled)
        self.assertEqual(tracking.reacquire_min_iou, 0.12)
        self.assertEqual(tracking.public_id_policy, "confirmed_only")

    def test_schema_two_rejects_invalid_tracking_fields(self):
        invalid_tracking = (
            {"loss_tolerance": {"unit": "frames", "value": 1.0}},
            {
                "loss_tolerance": {"unit": "seconds", "value": 0.1},
                "tentative_loss_tolerance": {"unit": "seconds", "value": 0.2},
            },
            {"reacquisition": {"enabled": "yes"}},
            {"reacquisition": {"min_iou": float("nan")}},
            {"public_id_policy": "candidate"},
        )
        for tracking in invalid_tracking:
            with self.subTest(tracking=tracking), self.assertRaises(ConfigError):
                SafetyPipelineConfig.from_mapping({
                    "schema_version": 2,
                    "ppe": {"tracking": tracking},
                })

    def test_reacquisition_fields_change_config_fingerprint(self):
        original = SafetyPipelineConfig.from_mapping({"schema_version": 2})
        changed = SafetyPipelineConfig.from_mapping({
            "schema_version": 2,
            "ppe": {"tracking": {"reacquisition": {"min_iou": 0.11}}},
        })
        self.assertNotEqual(config_fingerprint(original), config_fingerprint(changed))


if __name__ == "__main__":
    unittest.main()
