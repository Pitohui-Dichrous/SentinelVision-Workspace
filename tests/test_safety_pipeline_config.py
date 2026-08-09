from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from safety_pipeline.config import (
    ConfigError,
    SafetyPipelineConfig,
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
        self.assertEqual(config.schema_version, 1)

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


if __name__ == "__main__":
    unittest.main()
