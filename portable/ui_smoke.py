"""Read-only offscreen smoke check for the WorkspaceManager interface.

This script deliberately avoids pytest and never clicks actions that start
SENTINEL, training, candidate inspection, deployment, or model inference.
Optional screenshots must stay inside the workspace.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SmokeFailure(RuntimeError):
    """Raised when the UI does not satisfy the smoke-test contract."""


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SentinelVision WorkspaceManager UI smoke check offscreen."
    )
    parser.add_argument(
        "--screenshots",
        metavar="DIR",
        help="save one PNG per page to DIR (DIR must be inside the workspace)",
    )
    parser.add_argument("--width", type=int, default=1440, help="window width (default: 1440)")
    parser.add_argument("--height", type=int, default=960, help="window height (default: 960)")
    return parser.parse_args()


def _screenshot_directory(value: str | None) -> Path | None:
    if not value:
        return None

    requested = Path(value)
    target = requested if requested.is_absolute() else PROJECT_ROOT / requested
    target = target.resolve()
    try:
        target.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise SmokeFailure("截图目录必须位于工作库内：%s" % target) from exc

    settings_directory = (PROJECT_ROOT / ".runtime" / "config").resolve()
    try:
        target.relative_to(settings_directory)
    except ValueError:
        pass
    else:
        raise SmokeFailure("截图目录不得位于 .runtime/config 用户设置目录内。")

    target.mkdir(parents=True, exist_ok=True)
    return target


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def _require_widget(owner, name: str, expected_type):
    widget = getattr(owner, name, None)
    _require(
        isinstance(widget, expected_type),
        "缺少关键控件 %s（应为 %s）。" % (name, expected_type.__name__),
    )
    return widget


def _install_safety_guards(workspace_module) -> None:
    """Disable every external action while leaving page construction intact."""

    def blocked_action(*_args, **_kwargs):
        raise SmokeFailure("UI 烟雾检查期间意外触发了外部操作。")

    def lightweight_model_refresh(window):
        # The production implementation may hash and inspect .pt files.  A UI
        # smoke check only needs the table and summary widgets to be usable.
        if hasattr(window, "models_table"):
            window.models_table.setRowCount(0)
        if hasattr(window, "models_hint"):
            window.models_hint.setText("UI 烟雾检查：已跳过生产模型文件扫描。")

    manager_type = workspace_module.WorkspaceManager
    for method_name in (
        "_start_sentinel",
        "_start_training",
        "_resume_training",
        "_inspect_candidate",
        "_test_candidate_media",
        "_deploy_candidate",
        "_run_utility",
        "_launch_command_file",
        "_open_path",
    ):
        setattr(manager_type, method_name, blocked_action)
    manager_type._refresh_models = lightweight_model_refresh


def _check_key_widgets(window, QtWidgets) -> None:
    pages = _require_widget(window, "pages", QtWidgets.QStackedWidget)
    title = _require_widget(window, "page_title", QtWidgets.QLabel)
    subtitle = _require_widget(window, "page_subtitle", QtWidgets.QLabel)

    _require(pages.count() == 4, "WorkspaceManager 必须包含四个主页面。")
    _require(len(getattr(window, "nav_buttons", ())) == 4, "必须包含四个主导航按钮。")
    _require(all(isinstance(item, QtWidgets.QPushButton) for item in window.nav_buttons),
             "主导航项目必须全部是按钮。")
    _require(bool(title.text().strip()), "当前页面标题为空。")
    _require(bool(subtitle.text().strip()), "当前页面说明为空。")

    # Overview: a launch/action surface and its read-only feedback area.
    _require_widget(window, "home_log", QtWidgets.QPlainTextEdit)
    _require(bool(pages.widget(0).findChildren(QtWidgets.QPushButton)),
             "概览页缺少主要操作按钮。")

    # Training: preserve the dataset/weights inputs and explicit start action.
    _require_widget(window, "dataset_combo", QtWidgets.QComboBox)
    _require_widget(window, "weights_preset", QtWidgets.QComboBox)
    _require_widget(window, "btn_start_training", QtWidgets.QPushButton)
    _require_widget(window, "training_log", QtWidgets.QPlainTextEdit)

    # Review/deployment: the manual approval gate must remain present.
    _require_widget(window, "candidate_combo", QtWidgets.QComboBox)
    _require_widget(window, "class_table", QtWidgets.QTableWidget)
    _require_widget(window, "btn_deploy", QtWidgets.QPushButton)
    expected_review_checks = {
        "metrics_reviewed",
        "class_order_reviewed",
        "failure_samples_reviewed",
        "real_media_reviewed",
    }
    review_checks = getattr(window, "review_checks", {})
    _require(set(review_checks) == expected_review_checks, "人工审核清单必须保留四项门禁。")
    _require(all(isinstance(item, QtWidgets.QCheckBox) for item in review_checks.values()),
             "人工审核门禁必须使用复选框。")

    # Models: the deployed-model inventory remains inspectable.
    _require_widget(window, "models_table", QtWidgets.QTableWidget)
    _require_widget(window, "models_hint", QtWidgets.QLabel)


def _check_safety_and_responsiveness(window, QtWidgets) -> None:
    """Exercise UI-only state rules without invoking any external action."""

    original_candidate = window.candidate_combo.currentText()
    _require(not window.btn_deploy.isEnabled(), "未读取候选模型时部署按钮必须禁用。")
    for checkbox in window.review_checks.values():
        checkbox.setChecked(True)
    _require(not window.btn_deploy.isEnabled(), "仅勾选清单不得绕过候选模型技术检查。")
    for checkbox in window.review_checks.values():
        checkbox.setChecked(False)

    fake_candidate = (PROJECT_ROOT / "TRAINING_OUTPUTS" / "ui_smoke" / "weights" / "best.pt").resolve()
    window.candidate_combo.setCurrentText(str(fake_candidate))
    window.candidate_info = SimpleNamespace(path=fake_candidate)
    for checkbox in window.review_checks.values():
        checkbox.setChecked(True)
    window.deploy_model_id.setText("!!!")
    _require(not window.btn_deploy.isEnabled(), "无效模型 ID 不得解锁部署门禁。")
    window.deploy_model_id.setText("ui_smoke")
    _require(window.btn_deploy.isEnabled(), "候选已检查且 4 项齐全时门禁应进入可确认状态。")
    window._candidate_selection_changed()
    _require(not window.btn_deploy.isEnabled(), "切换候选后必须重新锁定部署门禁。")

    disclosure = getattr(window, "home_log_disclosure", None)
    _require(disclosure is not None, "概览页必须保留可展开的任务输出。")
    disclosure.set_expanded(True)
    _require(disclosure.body.isVisible(), "任务输出展开状态无效。")
    disclosure.set_expanded(False)
    _require(not disclosure.body.isVisible(), "任务输出收起状态无效。")

    window.resize(1150, 700)
    window._apply_responsive_layout()
    _require(not window.models_table.isColumnHidden(4), "非紧凑布局应显示权重路径列。")
    window.resize(1050, 700)
    window._apply_responsive_layout()
    _require(window.models_table.isColumnHidden(4), "跨紧凑断点时应隐藏权重路径列。")
    _require(
        window.models_summary_layout.direction() == QtWidgets.QBoxLayout.Direction.TopToBottom,
        "跨紧凑断点时模型摘要应切换为单列。",
    )

    window.resize(960, 640)
    window._apply_responsive_layout()
    _require(all(button.isVisible() for button in window.nav_buttons), "窄屏顶部导航必须保持可见。")
    _require(
        window.review_columns_layout.direction() == QtWidgets.QBoxLayout.Direction.TopToBottom,
        "窄屏审核页应切换为单列。",
    )
    _require(window.models_table.isColumnHidden(4), "窄屏应隐藏低优先级权重路径列。")

    window.resize(1440, 960)
    window._apply_responsive_layout()
    _require(window.navigation.isVisible(), "宽屏顶部导航必须保持可见。")
    _require(
        window.review_columns_layout.direction() == QtWidgets.QBoxLayout.Direction.LeftToRight,
        "宽屏审核页应恢复证据/门禁双栏。",
    )
    _require(not window.models_table.isColumnHidden(4), "宽屏应显示完整模型信息。")
    window.candidate_combo.setCurrentText(original_candidate)
    window._candidate_selection_changed()


def _pump_events(application, QtCore) -> None:
    from PySide6.QtTest import QTest

    QTest.qWait(220)
    for _ in range(3):
        application.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 100)


def _save_page_screenshot(window, destination: Path, page_number: int, slug: str) -> Path:
    path = destination / ("%02d-%s.png" % (page_number + 1, slug))
    pixmap = window.grab()
    _require(not pixmap.isNull(), "无法抓取第 %d 页画面。" % (page_number + 1))
    _require(pixmap.save(str(path), "PNG"), "无法保存截图：%s" % path)
    return path


def run_smoke(screenshot_value: str | None, width: int = 1440, height: int = 960) -> int:
    # These variables must be set before importing PySide6/workspace_manager.
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_OPENGL"] = "software"
    os.environ["YOLOV5_AUTOINSTALL"] = "false"
    os.environ["SENTINEL_OFFLINE"] = "1"

    root_text = str(PROJECT_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    os.chdir(PROJECT_ROOT)

    from PySide6 import QtCore, QtWidgets

    import workspace_manager
    from ui_font import install_ui_font

    screenshot_dir = _screenshot_directory(screenshot_value)
    _require(800 <= width <= 3840, "窗口宽度必须介于 800 和 3840。")
    _require(600 <= height <= 2160, "窗口高度必须介于 600 和 2160。")
    _install_safety_guards(workspace_module=workspace_manager)

    application = QtWidgets.QApplication([sys.argv[0]])
    application.setApplicationName("SentinelVision Workspace UI Smoke")
    install_ui_font(application)
    window = None
    saved_paths = []
    page_slugs = ("overview", "training", "review", "models")

    try:
        window = workspace_manager.WorkspaceManager()
        window.resize(width, height)
        window.show()
        _pump_events(application, QtCore)
        _check_key_widgets(window, QtWidgets)
        _check_safety_and_responsiveness(window, QtWidgets)
        window.resize(width, height)
        window._apply_responsive_layout()
        _pump_events(application, QtCore)

        pages = window.pages
        for index, slug in enumerate(page_slugs):
            window.nav_buttons[index].click()
            _pump_events(application, QtCore)
            _require(pages.currentIndex() == index, "无法切换到第 %d 页。" % (index + 1))
            _require(window.nav_buttons[index].isChecked(), "第 %d 个导航按钮未选中。" % (index + 1))
            _require(pages.currentWidget() is pages.widget(index), "页面堆栈状态不一致。")
            _require(bool(window.page_title.text().strip()), "第 %d 页标题为空。" % (index + 1))
            page_scroll = pages.currentWidget()
            _require(
                page_scroll.horizontalScrollBar().maximum() == 0,
                "第 %d 页出现外层横向滚动，响应式布局已溢出。" % (index + 1),
            )
            if index == 0:
                occupied_right = max(card.geometry().right() for card in window.home_metric_cards)
                grid_right = window.home_metrics_layout.geometry().right()
                _require(
                    grid_right - occupied_right <= 24,
                    "概览状态卡未占满可用宽度，可能残留空白网格列。",
                )
            if screenshot_dir is not None:
                saved_paths.append(_save_page_screenshot(window, screenshot_dir, index, slug))

        _require(window.training_process is None, "烟雾检查不得启动训练进程。")
        _require(window.utility_process is None, "烟雾检查不得启动工具进程。")
        print("[通过] WorkspaceManager 四页已在 offscreen 模式依次切换。")
        print("[通过] 训练、人工审核/部署与模型清单关键控件完整。")
        print("[通过] 未启动 SENTINEL、训练、工具进程或模型扫描/推理。")
        for path in saved_paths:
            print("[截图] %s" % path.relative_to(PROJECT_ROOT).as_posix())
        return 0
    finally:
        if window is not None:
            window.close()
            window.deleteLater()
            _pump_events(application, QtCore)
        application.quit()


def main() -> int:
    arguments = _parse_arguments()
    try:
        return run_smoke(arguments.screenshots, arguments.width, arguments.height)
    except Exception as exc:
        print("[失败] WorkspaceManager UI 烟雾检查未通过：%s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
