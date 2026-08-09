from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SENTINEL_OFFLINE", "1")

from PySide6 import QtCore, QtWidgets

import SentinelVision4


class FakeVideoWorker(QtCore.QObject):
    frameReady = QtCore.Signal(object, int, int)
    statusMsg = QtCore.Signal(str)
    fpsReady = QtCore.Signal(float)
    newAlert = QtCore.Signal(dict)
    modelStatus = QtCore.Signal(str, str, str)
    modelsApplied = QtCore.Signal(object, object)
    pipelineStatsReady = QtCore.Signal(dict)
    pipelineTransition = QtCore.Signal(dict)

    def __init__(self, snapshot, safety_config):
        super().__init__()
        self.snapshot = snapshot
        self.safety_config = safety_config
        self.pipeline_settings = None

    def configure_models(self, model_ids):
        self.model_ids = tuple(model_ids)

    def configure_classes(self, values):
        self.class_values = dict(values)

    def configure_safety_pipeline(self, mode, debug_overlay=False):
        self.pipeline_settings = (mode, bool(debug_overlay))

    def configure_thresholds(self, *_):
        pass

    def configure_polys(self, *_):
        pass

    def start(self):
        pass

    def isRunning(self):
        return False

    def request_stop(self):
        pass

    def wait(self, *_):
        return True


class DetectorUIContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_analysis_mode_is_explicit_and_switchable_without_inference(self):
        settings = {
            "schema_version": 2,
            "selected_model_ids": [],
            "class_enabled": {},
        }
        with (
            mock.patch.object(SentinelVision4, "VideoWorker", FakeVideoWorker),
            mock.patch.object(SentinelVision4.MainWindow, "_load_settings", return_value=settings),
            mock.patch.object(SentinelVision4.MainWindow, "_save_settings", autospec=True),
        ):
            window = SentinelVision4.MainWindow()
            try:
                window.resize(1200, 800)
                window.show()
                self.app.processEvents()
                self.assertEqual(window.safety_mode_combo.currentData(), "baseline")
                self.assertFalse(window.safety_debug_check.isEnabled())
                enhanced_index = window.safety_mode_combo.findData("ppe_temporal")
                window.safety_mode_combo.setCurrentIndex(enhanced_index)
                self.app.processEvents()
                self.assertTrue(window.safety_debug_check.isEnabled())
                self.assertEqual(window.worker.pipeline_settings, ("ppe_temporal", False))
                image = window.grab()
                self.assertFalse(image.isNull())
                self.assertGreater(image.width(), 0)
                self.assertGreater(image.height(), 0)
            finally:
                window.close()
                self.app.processEvents()

    def test_schema_three_save_preserves_unknown_settings(self):
        settings = {
            "schema_version": 2,
            "selected_model_ids": [],
            "class_enabled": {"future_class": False},
            "future_module": {"keep": "value"},
        }
        with (
            mock.patch.object(SentinelVision4, "VideoWorker", FakeVideoWorker),
            mock.patch.object(SentinelVision4.MainWindow, "_load_settings", return_value=settings),
        ):
            window = SentinelVision4.MainWindow()
            try:
                with tempfile.TemporaryDirectory() as directory:
                    window.settings_path = Path(directory) / "sentinel_settings.json"
                    window.safety_mode = "ppe_temporal"
                    window.safety_debug_overlay = True
                    window._save_settings()
                    saved = json.loads(window.settings_path.read_text(encoding="utf-8"))
                self.assertEqual(saved["schema_version"], 3)
                self.assertEqual(saved["future_module"], {"keep": "value"})
                self.assertFalse(saved["class_enabled"]["future_class"])
                self.assertEqual(saved["safety_pipeline"]["mode"], "ppe_temporal")
                self.assertTrue(saved["safety_pipeline"]["debug_overlay"])
            finally:
                window.close()
                self.app.processEvents()

    def test_invalid_tracked_algorithm_config_forces_fail_closed_baseline(self):
        settings = {
            "schema_version": 3,
            "selected_model_ids": [],
            "class_enabled": {},
            "safety_pipeline": {"mode": "ppe_temporal", "debug_overlay": True},
        }
        with (
            mock.patch.object(SentinelVision4, "VideoWorker", FakeVideoWorker),
            mock.patch.object(SentinelVision4, "SAFETY_PIPELINE_CONFIG_VALID", False),
            mock.patch.object(SentinelVision4.MainWindow, "_load_settings", return_value=settings),
            mock.patch.object(SentinelVision4.MainWindow, "_save_settings", autospec=True),
        ):
            window = SentinelVision4.MainWindow()
            try:
                self.assertEqual(window.safety_mode, "baseline")
                self.assertEqual(window.safety_mode_combo.currentData(), "baseline")
                self.assertFalse(window.safety_mode_combo.isEnabled())
                self.assertFalse(window.safety_debug_check.isEnabled())
                self.assertEqual(window.worker.pipeline_settings, ("baseline", False))
            finally:
                window.close()
                self.app.processEvents()

    def test_ppe_only_reset_preserves_fire_tracker_and_cooldown(self):
        worker = SentinelVision4.VideoWorker(
            SentinelVision4.CATALOG_SNAPSHOT,
            SentinelVision4.SAFETY_PIPELINE_DEFAULTS,
        )
        worker.tracks = [
            {"cls": "fire", "id": 1},
            {"cls": "hat", "id": 2},
            {"cls": "person", "id": 3},
        ]
        worker.frame_hits[("camera-01", "fire")].append(1)
        worker.frame_hits[("camera-01", "person")].append(1)
        worker.last_alert_at[("camera-01", "fire")] = 12.0
        worker.last_alert_at[("camera-01", "person")] = 13.0

        worker._reset_ppe_state()

        self.assertEqual([track["cls"] for track in worker.tracks], ["fire"])
        self.assertIn(("camera-01", "fire"), worker.frame_hits)
        self.assertNotIn(("camera-01", "person"), worker.frame_hits)
        self.assertEqual(worker.last_alert_at[("camera-01", "fire")], 12.0)
        self.assertNotIn(("camera-01", "person"), worker.last_alert_at)

    def test_public_track_session_is_preserved_in_alert_and_review_journal(self):
        worker = SentinelVision4.VideoWorker(
            SentinelVision4.CATALOG_SNAPSHOT,
            SentinelVision4.SAFETY_PIPELINE_DEFAULTS,
        )
        worker.event_journal = mock.Mock()
        alert = {
            "id": "session-a-ppe-0001-001",
            "event_id": "session-a-ppe-0001-001",
            "session_id": "session-a",
            "track_id": 1,
            "track_id_namespace": "session_public",
            "source_id": "camera-01",
            "cls": "person",
        }

        worker.record_alert(alert)
        worker.record_review(
            alert["id"],
            "valid",
            alert["track_id"],
            alert["event_id"],
            alert["session_id"],
            alert["track_id_namespace"],
        )

        alert_record = worker.event_journal.append.call_args_list[0]
        review_record = worker.event_journal.append.call_args_list[1]
        self.assertEqual(alert_record.args[0], "alert")
        self.assertEqual(alert_record.args[1]["session_id"], "session-a")
        self.assertEqual(alert_record.args[1]["track_id"], 1)
        self.assertEqual(alert_record.args[1]["track_id_namespace"], "session_public")
        self.assertEqual(review_record.args[0], "human_review")
        self.assertEqual(review_record.args[1]["session_id"], "session-a")
        self.assertEqual(review_record.args[1]["track_id"], 1)


if __name__ == "__main__":
    unittest.main()
