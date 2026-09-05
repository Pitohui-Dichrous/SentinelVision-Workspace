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

    def test_rapid_navigation_continues_from_visible_frame_and_releases_snapshots(self):
        window = self.create(workspace.WorkspaceManager)
        window._show_page(1)
        QTest.qWait(40)
        transition = window._page_transition
        bounds = QtCore.QRect(window.page_stage.mapTo(window, QtCore.QPoint()), window.page_stage.size())
        visible = window.grab(bounds).toImage()
        window._show_page(2)
        self.assertEqual(transition._before.toImage(), visible)
        window._show_page(3)
        self.assertEqual(window.pages.currentIndex(), 3)
        self.assertTrue(transition.isVisible())
        QTest.qWait(300)
        self.assertFalse(transition.isVisible())
        self.assertTrue(transition._before.isNull())
        self.assertTrue(transition._after.isNull())
        self.assertEqual(window.page_title.text(), window.PAGE_META[3][0])

    def test_transition_preserves_the_page_background_until_the_last_frame(self):
        window = self.create(workspace.WorkspaceManager)
        window._show_page(1)
        transition = window._page_transition
        transition.animation.pause()
        transition.animation.setCurrentTime(transition.animation.duration() - 1)
        before = window.grab().toImage()
        transition.cancel()
        after = window.grab().toImage()
        point = window.page_stage.mapTo(window, QtCore.QPoint(10, 20))
        point = QtCore.QPoint(round(point.x() * after.devicePixelRatio()), round(point.y() * after.devicePixelRatio()))
        self.assertEqual(before.pixelColor(point), after.pixelColor(point))
        self.assertEqual(after.pixelColor(point).name(), tokens(False)["bg"].lower())

    def test_resize_cancels_transition_and_keyboard_navigation_is_immediate(self):
        window = self.create(workspace.WorkspaceManager)
        window._show_page(1)
        window.resize(960, 640)
        self.app.processEvents()
        self.assertFalse(window._page_transition.isVisible())
        self.assertEqual(window.pages.currentWidget().horizontalScrollBar().maximum(), 0)
        window.nav_buttons[2].setFocus()
        QTest.keyClick(window.nav_buttons[2], QtCore.Qt.Key.Key_Space)
        self.assertEqual(window.pages.currentIndex(), 2)
        self.assertFalse(window._page_transition.isVisible())
        self.assertTrue(window.nav_buttons[2].hasFocus())

    def test_fast_pointer_tap_responds_immediately_with_visible_feedback(self):
        button = self.create(PressableButton)
        button.setText("Refresh")
        clicked = mock.Mock()
        button.clicked.connect(clicked)
        original = button.geometry()
        QTest.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)
        clicked.assert_called_once()
        QTest.qWait(40)
        self.assertLess(button.visualScale, .999)
        QTest.mousePress(button, QtCore.Qt.MouseButton.LeftButton)
        previous_scale = button.visualScale
        self.assertEqual(button._press_animation.startValue(), previous_scale)
        QTest.mouseRelease(button, QtCore.Qt.MouseButton.LeftButton)
        self.assertEqual(clicked.call_count, 2)
        QTest.qWait(300)
        self.assertEqual(button.visualScale, 1.0)
        self.assertEqual(button.geometry(), original)
        QTest.keyClick(button, QtCore.Qt.Key.Key_Space)
        self.assertEqual(clicked.call_count, 3)
        self.assertEqual(button._press_animation.state(), QtCore.QAbstractAnimation.State.Stopped)

    def test_dragging_out_of_a_button_cancels_activation_and_restores_paint(self):
        button = self.create(PressableButton)
        clicked = mock.Mock()
        button.clicked.connect(clicked)
        QTest.mousePress(button, QtCore.Qt.MouseButton.LeftButton)
        QTest.qWait(40)
        outside = QtCore.QPoint(button.width() + 20, button.height() + 20)
        QTest.mouseMove(button, outside)
        QTest.mouseRelease(button, QtCore.Qt.MouseButton.LeftButton, pos=outside)
        QTest.qWait(300)
        clicked.assert_not_called()
        self.assertEqual(button.visualScale, 1.0)

    def test_disabling_a_pressed_button_never_leaves_a_shrunken_control(self):
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
        window = self.create(detector.MainWindow)
        window.btn_file.setFocus()
        window._open_drawer()
        window._open_drawer()
        self.assertTrue(window.btn_close_drawer.hasFocus())
        QTest.keyClick(window.btn_close_drawer, QtCore.Qt.Key.Key_Escape)
        self.assertFalse(window.drawer_open)
        self.assertEqual(window.drawer.x(), window.width())
        self.assertTrue(window.btn_file.hasFocus())

    def test_detector_tabs_animate_pointer_changes_and_interrupt_for_keyboard(self):
        window = self.create(detector.MainWindow)
        tabs = window.inspector_tabs
        bar = tabs.tabBar()
        QTest.mouseClick(bar, QtCore.Qt.MouseButton.LeftButton, pos=bar.tabRect(2).center())
        self.assertEqual(tabs.currentIndex(), 2)
        self.assertTrue(tabs.page_transition().isVisible())
        bar.setFocus()
        QTest.keyClick(bar, QtCore.Qt.Key.Key_Left)
        self.assertEqual(tabs.currentIndex(), 1)
        self.assertFalse(tabs.page_transition().isVisible())

    def test_drawer_reverses_from_current_position_and_settles_after_resize(self):
        window = self.create(detector.MainWindow)
        window._open_drawer()
        QTest.qWait(60)
        current = window.drawer.pos()
        self.assertLess(current.x(), window.width())
        window._close_drawer()
        self.assertEqual(window.drawer_anim.startValue(), current)
        QTest.qWait(40)
        current = window.drawer.pos()
        window._open_drawer()
        self.assertEqual(window.drawer_anim.startValue(), current)
        window.resize(1100, 720)
        self.app.processEvents()
        self.assertEqual(window.drawer.x(), window.width() - window.drawer.width())
        self.assertEqual(window.drawer_anim.state(), QtCore.QAbstractAnimation.State.Stopped)

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

    def test_dropdowns_cover_system_dark_background_and_keep_edge_rows_visible(self):
        original_palette = self.app.palette()
        self.addCleanup(self.app.setPalette, original_palette)
        system_palette = QtGui.QPalette(original_palette)
        for role in (QtGui.QPalette.ColorRole.Window, QtGui.QPalette.ColorRole.Base,
                     QtGui.QPalette.ColorRole.Button):
            system_palette.setColor(role, QtGui.QColor("#2b2b2b"))
        self.app.setPalette(system_palette)
        workbench = self.create(workspace.WorkspaceManager)
        workbench._show_page(1)
        workbench._page_transition.cancel()
        console = self.create(detector.MainWindow)
        # Reparent nothing: language stays inside the real preferences menu.
        preferences = console.preferences_button.menu()
        combos = (workbench.weights_preset, console.safety_mode_combo, console.lang)
        for combo in combos:
            with self.subTest(combo=combo):
                if combo is console.lang:
                    preferences.popup(console.preferences_button.mapToGlobal(QtCore.QPoint()))
                combo.showPopup()
                self.app.processEvents()
                view = combo.view()
                popup = view.window()
                rendered = popup.grab().toImage()
                ratio = rendered.devicePixelRatio()
                # Sample the whole outer band, including formerly black top
                # and bottom menu gutters. Allow the thin gray rounded border
                # (also sampled at fractional display scaling), not dark fill.
                for x in range(2, popup.width() - 2, 7):
                    for y in (2, popup.height() - 3):
                        color = rendered.pixelColor(round(x * ratio), round(y * ratio))
                        self.assertGreater(min(color.red(), color.green(), color.blue()), 100,
                                           (x, y, color.name()))
                last = combo.model().index(combo.count() - 1, combo.modelColumn())
                view.scrollTo(last)
                self.app.processEvents()
                self.assertTrue(view.viewport().rect().contains(view.visualRect(last)))
                QTest.keyClick(view, QtCore.Qt.Key.Key_Escape)
                self.assertFalse(popup.isVisible())
                preferences.hide()

    def test_dropdown_keyboard_selection_and_theme_switch_remain_native(self):
        console = self.create(detector.MainWindow)
        original_dark = detector.DARK
        self.addCleanup(setattr, detector, "DARK", original_dark)
        combo = console.safety_mode_combo
        original = combo.currentData()
        combo.showPopup()
        self.app.processEvents()
        QTest.keyClick(combo.view(), QtCore.Qt.Key.Key_End)
        QTest.keyClick(combo.view(), QtCore.Qt.Key.Key_Return)
        self.assertEqual(combo.currentIndex(), combo.count() - 1)
        self.assertFalse(combo.view().window().isVisible())
        combo.setCurrentIndex(combo.findData(original))
        console.toggle_theme()
        combo.showPopup()
        self.app.processEvents()
        image = combo.view().window().grab().toImage()
        ratio = image.devicePixelRatio()
        self.assertEqual(image.pixelColor(round(10 * ratio), round(2 * ratio)).name(),
                         tokens(detector.DARK)["surface"].lower())
        QTest.keyClick(combo.view(), QtCore.Qt.Key.Key_Escape)


if __name__ == "__main__":
    unittest.main()
