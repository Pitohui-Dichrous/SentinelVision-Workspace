"""Behavioral checks for the redesigned UI; never train, deploy or open media."""

from contextlib import ExitStack
import os
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtTest import QTest

import SentinelVision4 as detector
import workspace_manager as workspace
from deployment_manager import CandidateInfo
from tests.test_detector_ui_contract import FakeVideoWorker
from ui_font import install_ui_font
from ui_theme import PressableButton, tokens


class RedesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        install_ui_font(cls.app)

    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(mock.patch.dict(os.environ, {"SENTINEL_REDUCE_MOTION": "1"}))
        self.patches.enter_context(mock.patch.object(detector, "VideoWorker", FakeVideoWorker))
        self.patches.enter_context(mock.patch.object(detector.MainWindow, "_save_settings"))
        self.patches.enter_context(mock.patch.object(detector.MainWindow, "_load_settings", return_value={
            "schema_version": 3, "selected_model_ids": [], "class_enabled": {},
            "safety_pipeline": {"mode": "baseline"},
        }))
        self.patches.enter_context(mock.patch.object(workspace.WorkspaceManager, "_refresh_models"))
        self.windows = []
        self.addCleanup(self.close_windows)

    def close_windows(self):
        for window in self.windows:
            window.close()
            window.deleteLater()
        self.app.processEvents()

    def create(self, cls):
        window = cls()
        self.windows.append(window)
        window.show()
        self.app.processEvents()
        return window

    def test_training_choices_survive_disclosure_and_navigation(self):
        window = self.create(workspace.WorkspaceManager)
        window._show_page(1)
        self.assertFalse(window.advanced_training.body.isVisible())
        window.advanced_training.toggle.click()
        self.assertTrue(window.advanced_training.body.isVisible())
        window.batch_size.setValue(8)
        window.workers.setValue(2)
        window.experiment_name.setText("new_ppe")
        window._show_page(0)
        window._show_page(1)
        self.assertEqual(window.batch_size.value(), 8)
        self.assertEqual(window.workers.value(), 2)
        self.assertEqual(window.experiment_name.text(), "new_ppe")
        self.assertIsNone(window.training_process)

    def test_class_summary_edits_preserve_mapping_and_unknown_alert_opt_in(self):
        window = self.create(workspace.WorkspaceManager)
        window._populate_class_table(("person", "hat", "new_target"))
        window._select_class_editor(2)
        editors = window.class_editors["new_target"]
        self.assertFalse(editors[4].isChecked())
        editors[0].setText("vehicle")
        editors[1].setText("车辆")
        editors[4].setChecked(True)
        window._select_class_editor(0)
        self.assertEqual(window._class_settings()["new_target"]["canonical_id"], "vehicle")
        self.assertEqual(window.class_table.item(2, 1).text(), "车辆")
        self.assertEqual(window.class_table.item(2, 3).text(), "启用")
        window._candidate_selection_changed()
        self.assertEqual(window.class_editor_pages.count(), 0)
        self.assertEqual(window.class_editors, {})
        self.assertFalse(window.btn_deploy.isEnabled())

    def test_candidate_change_clears_approval_and_final_confirmation_remains_required(self):
        window = self.create(workspace.WorkspaceManager)
        path = (workspace.PROJECT_ROOT / "TRAINING_OUTPUTS" / "ui-only" / "best.pt").resolve()
        window.candidate_combo.setCurrentText(str(path))
        window.candidate_info = CandidateInfo(path, ("hat",), 100, "1" * 64)
        window._populate_class_table(("hat",))
        window.deploy_model_id.setText("new_helmet")
        for checkbox in window.review_checks.values():
            checkbox.setChecked(True)
        self.assertTrue(window.btn_deploy.isEnabled())
        with mock.patch.object(QtWidgets.QMessageBox, "warning", return_value=QtWidgets.QMessageBox.StandardButton.No) as confirm, \
             mock.patch.object(workspace.DeploymentManager, "deploy") as deploy:
            window._deploy_candidate()
            confirm.assert_called_once()
            deploy.assert_not_called()
        window.candidate_combo.setCurrentText(str(path.with_name("another.pt")))
        self.assertFalse(window.btn_deploy.isEnabled())
        self.assertFalse(any(check.isChecked() for check in window.review_checks.values()))

    def test_reduced_motion_interrupts_navigation_at_a_readable_final_state(self):
        window = self.create(workspace.WorkspaceManager)
        window.motion_check.setChecked(False)
        window._show_page(1)
        window._show_page(2)
        window.motion_check.setChecked(True)
        self.assertEqual(window._page_effect.opacity(), 1.0)
        self.assertEqual(window._page_animation.state(), QtCore.QAbstractAnimation.State.Stopped)
        window._show_page(3)
        self.assertEqual(window._page_effect.opacity(), 1.0)
        self.assertEqual(window.pages.currentIndex(), 3)

    def test_disabling_a_pressed_button_never_leaves_a_shrunken_control(self):
        os.environ["SENTINEL_REDUCE_MOTION"] = "0"
        button = self.create(PressableButton)
        button.setText("Run")
        QTest.mousePress(button, QtCore.Qt.MouseButton.LeftButton)
        QTest.qWait(60)
        self.assertLess(button.visualScale, 1.0)
        button.setEnabled(False)
        self.assertEqual(button.visualScale, 1.0)
        self.assertEqual(button._press_animation.state(), QtCore.QAbstractAnimation.State.Stopped)

    def test_detection_controls_fit_small_window_and_regions_remain_reachable(self):
        window = self.create(detector.MainWindow)
        window.resize(1100, 720)
        self.app.processEvents()
        self.assertEqual(window.size(), QtCore.QSize(1100, 720))
        for expanded in (False, True):
            window.region_toggle.setChecked(expanded)
            self.app.processEvents()
            self.assertEqual(window.region_bar.isVisible(), expanded)
            self.assertLess(window.canvas.geometry().bottom(), window.transport_bar.geometry().top())
            self.assertLessEqual(window.transport_bar.geometry().bottom(), window.viewport_card.height())
        self.assertEqual(window.selected_model_ids, ())
        self.assertTrue(all(not checkbox.isChecked() for checkbox in window.model_checks.values()))
        window.role.setCurrentText("Viewer")
        self.assertFalse(window.btn_file.isEnabled())
        self.assertFalse(window.btn_webcam.isEnabled())
        self.assertFalse(window.safety_mode_combo.isEnabled())
        window.role.setCurrentText("Admin")
        self.assertTrue(window.btn_file.isEnabled())

    def test_real_frame_replaces_empty_state_without_changing_image_geometry(self):
        window = self.create(detector.MainWindow)
        frame = QtGui.QImage(640, 360, QtGui.QImage.Format.Format_RGB32)
        frame.fill(QtGui.QColor("#394A5D"))
        window._on_frame(frame, 640, 360)
        self.app.processEvents()
        self.assertEqual(window.canvas._src_w, 640)
        self.assertEqual(window.canvas._src_h, 360)
        rendered = window.canvas.grab().toImage()
        self.assertEqual(rendered.pixelColor(rendered.width() // 2, rendered.height() // 2).name(), "#394a5d")

    def test_expanded_forms_fit_workbench_minimum_width(self):
        window = self.create(workspace.WorkspaceManager)
        window.resize(960, 640)
        window._show_page(1)
        window.advanced_training.set_expanded(True)
        window.weight_disclosure.set_expanded(True)
        self.app.processEvents()
        self.assertEqual(window.pages.currentWidget().horizontalScrollBar().maximum(), 0)
        window._show_page(2)
        window._populate_class_table(("person", "hat", "fire"))
        window._select_class_editor(0)
        self.app.processEvents()
        self.assertEqual(window.pages.currentWidget().horizontalScrollBar().maximum(), 0)
        editor_page = window.class_editor_pages.currentWidget()
        for editor in window.class_editors["person"]:
            self.assertGreater(editor.width(), 100)
            self.assertLessEqual(editor.geometry().right(), editor_page.width())

    def test_drawer_escape_returns_keyboard_focus_even_after_reopening(self):
        self.patches.enter_context(mock.patch.object(detector, "REDUCE_MOTION", True))
        window = self.create(detector.MainWindow)
        window.btn_file.setFocus()
        window._open_drawer()
        window._open_drawer()
        self.assertTrue(window.btn_close_drawer.hasFocus())
        QTest.keyClick(window.btn_close_drawer, QtCore.Qt.Key.Key_Escape)
        self.assertFalse(window.drawer_open)
        self.assertEqual(window.drawer.x(), window.width())
        self.assertTrue(window.btn_file.hasFocus())

    def test_palette_text_and_action_contrast_in_both_themes(self):
        def luminance(value):
            rgb = [int(value[i:i+2], 16) / 255 for i in (1, 3, 5)]
            linear = [n / 12.92 if n <= .04045 else ((n + .055) / 1.055) ** 2.4 for n in rgb]
            return sum(a*b for a, b in zip(linear, (.2126, .7152, .0722)))
        for dark in (False, True):
            palette = tokens(dark)
            for foreground, background in (("text", "bg"), ("muted", "surface"), ("subtle", "surface_alt"),
                                           ("cyan", "primary_soft"), ("on_primary", "primary"),
                                           ("success", "success_soft"), ("warning", "warning_soft")):
                a, b = sorted((luminance(palette[foreground]), luminance(palette[background])))
                self.assertGreaterEqual((b + .05) / (a + .05), 4.5, (dark, foreground, background))


if __name__ == "__main__":
    unittest.main()
