"""Beginner-facing SentinelVision training, review, and deployment workspace."""

from __future__ import annotations

import codecs
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

os.environ["YOLOV5_AUTOINSTALL"] = "false"
os.environ["SENTINEL_OFFLINE"] = "1"

import torch
from PySide6 import QtCore, QtGui, QtWidgets

from deployment_manager import DeploymentManager, inspect_candidate, safe_model_id
from model_catalog import ModelCatalog
from pretrained_weights import PRESETS, WeightPreset, validate_training_weight
from project_paths import (
    PRETRAINED_WEIGHTS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    RUNTIME_PYTHON,
    TRAINING_OUTPUTS_DIR,
)
from ui_font import install_ui_font
from ui_theme import (
    PressableButton,
    build_stylesheet,
    make_badge,
    make_card,
    set_property,
)


PYTHON_EXECUTABLE = str(RUNTIME_PYTHON)


class AuroraBackdrop(QtWidgets.QWidget):
    """Static, low-cost light field behind the translucent workspace surfaces."""

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor("#050A12"))

        width = max(1, self.width())
        height = max(1, self.height())
        lights = (
            (QtCore.QPointF(width * 0.18, height * 0.06), max(width, height) * 0.58, QtGui.QColor(68, 112, 255, 78)),
            (QtCore.QPointF(width * 0.92, height * 0.24), max(width, height) * 0.48, QtGui.QColor(39, 197, 214, 52)),
            (QtCore.QPointF(width * 0.70, height * 1.02), max(width, height) * 0.52, QtGui.QColor(111, 73, 255, 44)),
        )
        for center, radius, color in lights:
            gradient = QtGui.QRadialGradient(center, radius)
            gradient.setColorAt(0.0, color)
            color.setAlpha(0)
            gradient.setColorAt(1.0, color)
            painter.fillRect(self.rect(), gradient)
        painter.end()
        super().paintEvent(event)


class DisclosureCard(QtWidgets.QFrame):
    """Compact glass section with an immediate, keyboard-friendly reveal."""

    def __init__(self, title, detail, expanded=False, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setProperty("card", True)
        self.setProperty("glass", "panel")
        self._title = title
        self._expanded = bool(expanded)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)
        header = QtWidgets.QHBoxLayout()
        header.setSpacing(12)
        copy = QtWidgets.QVBoxLayout()
        copy.setSpacing(2)
        heading = QtWidgets.QLabel(title)
        heading.setProperty("textRole", "sectionTitle")
        supporting = QtWidgets.QLabel(detail)
        supporting.setProperty("textRole", "muted")
        supporting.setWordWrap(True)
        copy.addWidget(heading)
        copy.addWidget(supporting)
        self.toggle = PressableButton()
        self.toggle.setProperty("variant", "ghost")
        self.toggle.setCheckable(True)
        self.toggle.setChecked(self._expanded)
        self.toggle.setAccessibleName("展开或收起%s" % title)
        self.toggle.clicked.connect(self.set_expanded)
        header.addLayout(copy, 1)
        header.addWidget(self.toggle, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        outer.addLayout(header)

        self.body = QtWidgets.QWidget()
        self.body.setObjectName("DisclosureBody")
        self.body_layout = QtWidgets.QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 2, 0, 0)
        self.body_layout.setSpacing(8)
        outer.addWidget(self.body)
        self.set_expanded(self._expanded)

    def set_expanded(self, expanded):
        expanded = bool(expanded)
        self._expanded = expanded
        self.toggle.blockSignals(True)
        self.toggle.setChecked(expanded)
        self.toggle.setText("收起" if expanded else "展开")
        self.toggle.blockSignals(False)
        self.body.setVisible(expanded)


class WorkspaceManager(QtWidgets.QMainWindow):
    PAGE_META = (
        ("概览", "掌握当前状态，只处理下一步"),
        ("训练", "使用安全默认值创建候选模型"),
        ("审核与部署", "人工确认后，候选模型才能进入生产区"),
        ("生产模型", "查看 RESULTS 中已批准的正式模型"),
    )

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SentinelVision · Liquid Glass AI 工作台")
        self.resize(1400, 900)
        self.setMinimumSize(960, 640)
        self.training_process = None
        self.utility_process = None
        self.candidate_info = None
        self.class_editors = {}
        self._last_recommended_image_size = 640

        self._build_shell()
        self.statusBar().showMessage("工作库：%s" % PROJECT_ROOT)
        self._apply_style()
        self._refresh_datasets()
        self._refresh_candidates()
        self._refresh_models()

    def _build_shell(self):
        shell = AuroraBackdrop()
        shell.setObjectName("AppShell")
        shell_layout = QtWidgets.QHBoxLayout(shell)
        shell_layout.setContentsMargins(14, 14, 14, 12)
        shell_layout.setSpacing(14)

        sidebar = QtWidgets.QFrame()
        sidebar.setObjectName("SideNav")
        sidebar.setProperty("glass", "strong")
        sidebar.setFixedWidth(228)
        self.sidebar = sidebar
        nav_layout = QtWidgets.QVBoxLayout(sidebar)
        nav_layout.setContentsMargins(17, 19, 17, 16)
        nav_layout.setSpacing(7)

        brand_row = QtWidgets.QHBoxLayout()
        logo = QtWidgets.QLabel("SV")
        logo.setObjectName("BrandMark")
        logo.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        logo.setFixedSize(42, 42)
        brand_text = QtWidgets.QVBoxLayout()
        brand_text.setSpacing(1)
        brand = QtWidgets.QLabel("SentinelVision")
        brand.setProperty("textRole", "brand")
        edition = QtWidgets.QLabel("AI SAFETY WORKSPACE")
        edition.setProperty("textRole", "eyebrow")
        self.brand_label = brand
        self.edition_label = edition
        brand_text.addWidget(brand)
        brand_text.addWidget(edition)
        brand_row.addWidget(logo)
        brand_row.addLayout(brand_text, 1)
        nav_layout.addLayout(brand_row)
        nav_layout.addSpacing(23)

        nav_caption = QtWidgets.QLabel("工作区")
        nav_caption.setProperty("textRole", "navCaption")
        self.nav_caption = nav_caption
        nav_layout.addWidget(nav_caption)
        nav_layout.addSpacing(3)

        self.nav_group = QtWidgets.QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons = []
        self.nav_item_labels = []
        nav_items = (
            ("01", "概览", "运行状态与下一步"),
            ("02", "训练", "创建候选模型"),
            ("03", "审核与部署", "人工生产门禁"),
            ("04", "生产模型", "正式模型目录"),
        )
        for index, (number, text, hint) in enumerate(nav_items):
            button = QtWidgets.QPushButton("%s    %s" % (number, text))
            button.setCheckable(True)
            button.setProperty("variant", "nav")
            button.setToolTip(hint)
            button.setAccessibleName(text)
            self.nav_group.addButton(button, index)
            self.nav_buttons.append(button)
            self.nav_item_labels.append((number, text))
            nav_layout.addWidget(button)
        nav_layout.addStretch(1)

        local_caption = QtWidgets.QLabel("运行环境")
        local_caption.setProperty("textRole", "navCaption")
        self.local_caption = local_caption
        nav_layout.addWidget(local_caption)
        offline = make_badge("●  本地安全运行", "success")
        offline.setToolTip("运行环境、缓存和训练输出全部保存在本地移动工作库中")
        self.sidebar_offline_badge = offline
        nav_layout.addWidget(offline)
        location = QtWidgets.QLabel(PROJECT_ROOT.name)
        location.setProperty("textRole", "muted")
        location.setWordWrap(True)
        location.setToolTip(str(PROJECT_ROOT))
        self.sidebar_location = location
        nav_layout.addWidget(location)

        content = QtWidgets.QFrame()
        content.setObjectName("ContentShell")
        content.setProperty("glass", "strong")
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        top_bar = QtWidgets.QFrame()
        top_bar.setObjectName("TopBar")
        top_layout = QtWidgets.QHBoxLayout(top_bar)
        top_layout.setContentsMargins(25, 16, 25, 15)
        page_text = QtWidgets.QVBoxLayout()
        page_text.setSpacing(3)
        self.page_eyebrow = QtWidgets.QLabel()
        self.page_eyebrow.setProperty("textRole", "eyebrow")
        self.page_title = QtWidgets.QLabel()
        self.page_title.setProperty("textRole", "pageTitle")
        self.page_subtitle = QtWidgets.QLabel()
        self.page_subtitle.setProperty("textRole", "muted")
        page_text.addWidget(self.page_eyebrow)
        page_text.addWidget(self.page_title)
        page_text.addWidget(self.page_subtitle)
        top_layout.addLayout(page_text)
        top_layout.addStretch(1)
        gpu_ready = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_ready else "CUDA 不可用"
        gpu_text = (
            gpu_name.replace("NVIDIA GeForce ", "")
            .replace("NVIDIA ", "")
            .replace(" Laptop GPU", "")
        )
        offline_badge = make_badge("OFFLINE", "neutral")
        offline_badge.setToolTip("工作库以本地离线模式运行")
        self.header_device_badge = make_badge("GPU · %s" % gpu_text, "success" if gpu_ready else "warning")
        self.header_device_badge.setToolTip(gpu_name)
        top_layout.addWidget(offline_badge)
        top_layout.addWidget(self.header_device_badge)

        self.pages = QtWidgets.QStackedWidget()
        self.pages.addWidget(self._build_home_tab())
        self.pages.addWidget(self._build_training_tab())
        self.pages.addWidget(self._build_deployment_tab())
        self.pages.addWidget(self._build_models_tab())
        content_layout.addWidget(top_bar)
        content_layout.addWidget(self.pages, 1)

        shell_layout.addWidget(sidebar)
        shell_layout.addWidget(content, 1)
        self.setCentralWidget(shell)
        self.nav_group.idClicked.connect(self._show_page)
        self.nav_buttons[0].setChecked(True)
        self._show_page(0)
        QtCore.QTimer.singleShot(0, self._apply_responsive_layout)

    def _show_page(self, index):
        if not 0 <= index < len(self.PAGE_META):
            return
        self.pages.setCurrentIndex(index)
        title, subtitle = self.PAGE_META[index]
        self.page_eyebrow.setText("WORKSPACE  /  %02d" % (index + 1))
        self.page_title.setText(title)
        self.page_subtitle.setText(subtitle)

    def _apply_responsive_layout(self):
        if not hasattr(self, "sidebar"):
            return
        compact = self.width() < 1100
        stacked = self.width() < 1180
        if getattr(self, "_compact", None) != compact:
            self._compact = compact
            self.sidebar.setFixedWidth(78 if compact else 228)
            self.brand_label.setVisible(not compact)
            self.edition_label.setVisible(not compact)
            self.nav_caption.setVisible(not compact)
            self.local_caption.setVisible(not compact)
            self.sidebar_location.setVisible(not compact)
            self.sidebar_offline_badge.setText("●" if compact else "●  本地安全运行")
            self.sidebar_offline_badge.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignCenter
                if compact else QtCore.Qt.AlignmentFlag.AlignLeft
            )
            for button, (number, label) in zip(self.nav_buttons, self.nav_item_labels):
                button.setText(label[:2] if compact else "%s    %s" % (number, label))
                set_property(button, "compact", compact)
            self.models_summary_layout.setDirection(
                QtWidgets.QBoxLayout.Direction.TopToBottom
                if compact else QtWidgets.QBoxLayout.Direction.LeftToRight
            )
            if hasattr(self, "models_table"):
                self.models_table.setColumnHidden(4, compact)

        if getattr(self, "_stacked_pages", None) != stacked:
            self._stacked_pages = stacked
            direction = (
                QtWidgets.QBoxLayout.Direction.TopToBottom
                if stacked else QtWidgets.QBoxLayout.Direction.LeftToRight
            )
            self.home_lower_layout.setDirection(direction)
            self.training_columns_layout.setDirection(direction)
            self.review_columns_layout.setDirection(direction)
            for column in range(4):
                self.home_metrics_layout.setColumnStretch(
                    column, 1 if not stacked or column < 2 else 0
                )
            for index, card in enumerate(self.home_metric_cards):
                row, column = (divmod(index, 2) if stacked else (0, index))
                self.home_metrics_layout.addWidget(card, row, column)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_layout()

    def _apply_style(self):
        QtWidgets.QApplication.setStyle("Fusion")
        self.setStyleSheet(build_stylesheet(dark=True))

    @staticmethod
    def _page():
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        outer = QtWidgets.QWidget()
        outer.setObjectName("PageContent")
        layout = QtWidgets.QVBoxLayout(outer)
        layout.setContentsMargins(24, 20, 24, 24)
        layout.setSpacing(14)
        scroll.setWidget(outer)
        return scroll, layout

    @staticmethod
    def _button(text, callback, primary=False, variant=None):
        button = PressableButton(text)
        resolved_variant = "primary" if primary else variant
        if resolved_variant:
            button.setProperty("variant", resolved_variant)
        button.clicked.connect(callback)
        return button

    @staticmethod
    def _section_header(title, detail=None, badge=None):
        container = QtWidgets.QWidget()
        container.setObjectName("SectionHeader")
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        copy = QtWidgets.QVBoxLayout()
        copy.setSpacing(3)
        heading = QtWidgets.QLabel(title)
        heading.setProperty("textRole", "sectionTitle")
        copy.addWidget(heading)
        if detail:
            supporting = QtWidgets.QLabel(detail)
            supporting.setProperty("textRole", "muted")
            supporting.setWordWrap(True)
            copy.addWidget(supporting)
        layout.addLayout(copy, 1)
        if badge is not None:
            layout.addWidget(badge, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        return container

    @classmethod
    def _glass_card(cls, title=None, detail=None, elevated=False, badge=None):
        card = make_card(elevated=elevated)
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(18, 17, 18, 18)
        layout.setSpacing(13)
        if title:
            layout.addWidget(cls._section_header(title, detail, badge))
        return card, layout

    @staticmethod
    def _metric_card(label, value, detail, tone="neutral"):
        card = make_card(elevated=tone != "neutral")
        card.setProperty("cardStyle", "metric")
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(16, 15, 16, 15)
        layout.setSpacing(5)
        caption = QtWidgets.QLabel(label)
        caption.setProperty("textRole", "eyebrow")
        metric = QtWidgets.QLabel(value)
        metric.setProperty("textRole", "metric")
        description = QtWidgets.QLabel(detail)
        description.setProperty("textRole", "muted")
        description.setWordWrap(True)
        layout.addWidget(caption)
        layout.addWidget(metric)
        layout.addWidget(description)
        return card, metric

    @staticmethod
    def _open_path(path):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(Path(path).resolve())))

    def _launch_command_file(self, filename):
        path = (PROJECT_ROOT / filename).resolve()
        if not path.is_file():
            QtWidgets.QMessageBox.warning(self, "文件缺失", "找不到：%s" % path)
            return
        try:
            os.startfile(str(path))
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "启动失败", str(exc))

    @staticmethod
    def _prepare_process(process):
        """Force deterministic UTF-8 logs and project-local caches."""
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        computer = os.environ.get("COMPUTERNAME", "THIS_PC")
        runtime_cache = PROJECT_ROOT / ".runtime" / "cache" / computer
        runtime_temp = PROJECT_ROOT / ".runtime" / "temp" / computer
        python_root = RUNTIME_PYTHON.parent
        runtime_cache.mkdir(parents=True, exist_ok=True)
        runtime_temp.mkdir(parents=True, exist_ok=True)
        for variable in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV"):
            environment.remove(variable)
        environment.insert("PYTHONUTF8", "1")
        environment.insert("PYTHONIOENCODING", "utf-8")
        environment.insert("PYTHONNOUSERSITE", "1")
        environment.insert("PYTHONDONTWRITEBYTECODE", "1")
        environment.insert("YOLOV5_AUTOINSTALL", "false")
        environment.insert("SENTINEL_OFFLINE", "1")
        environment.insert("GIT_PYTHON_REFRESH", "quiet")
        environment.insert("YOLOV5_CONFIG_DIR", str(runtime_cache / "yolov5"))
        environment.insert("TORCH_HOME", str(runtime_cache / "torch"))
        environment.insert("TORCH_EXTENSIONS_DIR", str(runtime_cache / "torch_extensions"))
        environment.insert("MPLCONFIGDIR", str(runtime_cache / "matplotlib"))
        environment.insert("CUDA_CACHE_PATH", str(runtime_cache / "cuda"))
        environment.insert("TEMP", str(runtime_temp))
        environment.insert("TMP", str(runtime_temp))
        qt_plugins = python_root / "Lib" / "site-packages" / "PySide6" / "plugins"
        environment.insert("QT_PLUGIN_PATH", str(qt_plugins))
        environment.insert("QT_QPA_PLATFORM_PLUGIN_PATH", str(qt_plugins / "platforms"))
        path_prefixes = (
            python_root,
            python_root / "DLLs",
            python_root / "Lib" / "site-packages" / "torch" / "lib",
        )
        environment.insert(
            "PATH",
            os.pathsep.join(str(path) for path in path_prefixes) + os.pathsep + environment.value("PATH"),
        )
        process.setProcessEnvironment(environment)

    def _build_home_tab(self):
        page, layout = self._page()
        hero = make_card(elevated=True)
        hero.setObjectName("HeroCard")
        hero.setProperty("cardStyle", "hero")
        hero_layout = QtWidgets.QHBoxLayout(hero)
        hero_layout.setContentsMargins(26, 24, 24, 24)
        hero_layout.setSpacing(24)
        hero_copy = QtWidgets.QVBoxLayout()
        hero_copy.setSpacing(7)
        eyebrow = QtWidgets.QLabel("LOCAL AI SAFETY · READY")
        eyebrow.setProperty("textRole", "eyebrow")
        headline = QtWidgets.QLabel("今天需要处理什么？")
        headline.setProperty("textRole", "heroTitle")
        description = QtWidgets.QLabel("启动实时检测，或从训练开始构建下一版生产模型。所有数据留在本机。")
        description.setProperty("textRole", "muted")
        description.setWordWrap(True)
        hero_copy.addWidget(eyebrow)
        hero_copy.addWidget(headline)
        hero_copy.addWidget(description)
        hero_layout.addLayout(hero_copy, 1)
        hero_actions = QtWidgets.QVBoxLayout()
        hero_actions.setSpacing(8)
        launch = self._button("启动实时检测", self._start_sentinel, primary=True)
        launch.setAccessibleName("启动实时检测")
        launch.setMinimumWidth(178)
        launch.setMinimumHeight(42)
        environment = self._button("检查运行环境", lambda: self._run_utility("portable_check.py"))
        environment.setAccessibleName("检查运行环境")
        hero_actions.addWidget(launch)
        hero_actions.addWidget(environment)
        hero_layout.addLayout(hero_actions)
        layout.addWidget(hero)

        deployed_count = len(list(RESULTS_DIR.glob("**/weights/best.pt")))
        candidate_count = len(list(TRAINING_OUTPUTS_DIR.glob("**/weights/best.pt")))
        gpu_value = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "未检测到 CUDA"
        metrics = QtWidgets.QGridLayout()
        metrics.setHorizontalSpacing(10)
        metrics.setVerticalSpacing(10)
        gpu_short = (
            gpu_value.replace("NVIDIA GeForce ", "")
            .replace("NVIDIA ", "")
            .replace(" Laptop GPU", "")
        )
        gpu_card, self.home_gpu = self._metric_card("计算设备", gpu_short, "CUDA 加速" if torch.cuda.is_available() else "需要检查环境", "success" if torch.cuda.is_available() else "warning")
        model_card, self.home_model_metric = self._metric_card("生产模型", str(deployed_count), "已人工批准", "primary")
        candidate_card, self.home_candidate_metric = self._metric_card("待审候选", str(candidate_count), "不会自动部署", "warning" if candidate_count else "neutral")
        runtime_card, self.home_runtime_metric = self._metric_card("运行方式", "本地离线", "便携运行时", "success")
        for index, card in enumerate((gpu_card, model_card, candidate_card, runtime_card)):
            card.setMinimumWidth(130)
            metrics.addWidget(card, 0, index)
            metrics.setColumnStretch(index, 1)
        self.home_metrics_layout = metrics
        self.home_metric_cards = (gpu_card, model_card, candidate_card, runtime_card)
        layout.addLayout(metrics)

        lower = QtWidgets.QHBoxLayout()
        lower.setSpacing(12)
        workflow, workflow_layout = self._glass_card(
            "模型发布路径",
            "每一步都有明确边界，候选权重不会自行进入生产区。",
            elevated=True,
            badge=make_badge("人工门禁", "primary"),
        )
        workflow_steps = (
            ("01", "训练", "生成候选权重", "进入训练", lambda: self._show_page(1)),
            ("02", "审核", "核对指标与真实媒体", "进入审核", lambda: self._show_page(2)),
            ("03", "部署", "批准后复制到 RESULTS", "查看模型", lambda: self._show_page(3)),
        )
        for index, (number, title, detail, action_text, callback) in enumerate(workflow_steps):
            row = QtWidgets.QFrame()
            row.setProperty("surface", "soft")
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(13, 10, 10, 10)
            row_layout.setSpacing(11)
            step = QtWidgets.QLabel(number)
            step.setProperty("textRole", "stepNumber")
            step.setFixedWidth(28)
            copy = QtWidgets.QVBoxLayout()
            copy.setSpacing(1)
            title_label = QtWidgets.QLabel(title)
            title_label.setProperty("textRole", "bodyStrong")
            detail_label = QtWidgets.QLabel(detail)
            detail_label.setProperty("textRole", "muted")
            copy.addWidget(title_label)
            copy.addWidget(detail_label)
            action = self._button(action_text, callback, variant="ghost")
            action.setMinimumWidth(86)
            row_layout.addWidget(step)
            row_layout.addLayout(copy, 1)
            row_layout.addWidget(action)
            workflow_layout.addWidget(row)
            if index < len(workflow_steps) - 1:
                connector = QtWidgets.QFrame()
                connector.setObjectName("WorkflowConnector")
                connector.setFixedHeight(1)
                workflow_layout.addWidget(connector)
        workflow_layout.addStretch(1)

        tools = DisclosureCard("维护与帮助", "检查、文档与版本工具", expanded=False)
        tools_layout = tools.body_layout
        tool_actions = (
            ("检查文件完整", lambda: self._run_utility("portable/verify_workspace.py")),
            ("打开工作库", lambda: self._open_path(PROJECT_ROOT)),
            ("中文使用说明", lambda: self._open_path(PROJECT_ROOT / "START_HERE_CN.md")),
            ("Git 使用说明", lambda: self._open_path(PROJECT_ROOT / "GIT_GUIDE_CN.md")),
            ("保存代码版本", lambda: self._launch_command_file("SAVE_VERSION.cmd")),
            ("同步 GitHub", lambda: self._launch_command_file("SYNC_GITHUB.cmd")),
            ("查看 Git 历史", lambda: self._launch_command_file("GIT_STATUS.cmd")),
        )
        for text, callback in tool_actions:
            tools_layout.addWidget(self._button(text, callback, variant="quiet"))
        lower.addWidget(workflow, 3)
        lower.addWidget(tools, 2, QtCore.Qt.AlignmentFlag.AlignTop)
        self.home_lower_layout = lower
        layout.addLayout(lower)

        self.home_log_disclosure = DisclosureCard(
            "任务输出", "环境与文件检查的结果会显示在这里。", expanded=False
        )
        log_layout = self.home_log_disclosure.body_layout
        self.home_log = QtWidgets.QPlainTextEdit()
        self.home_log.setReadOnly(True)
        self.home_log.setPlaceholderText("环境检查结果会显示在这里。")
        self.home_log.setMinimumHeight(150)
        self.home_log.setMaximumHeight(230)
        log_layout.addWidget(self.home_log)
        layout.addWidget(self.home_log_disclosure)
        layout.addStretch(1)
        return page

    def _build_training_tab(self):
        page, layout = self._page()
        safety = QtWidgets.QFrame()
        safety.setObjectName("SafetyStrip")
        safety.setProperty("tone", "info")
        safety_layout = QtWidgets.QHBoxLayout(safety)
        safety_layout.setContentsMargins(14, 11, 14, 11)
        safety_layout.setSpacing(10)
        safety_layout.addWidget(make_badge("候选区", "primary"))
        intro = QtWidgets.QLabel(
            "训练结果只写入 TRAINING_OUTPUTS。数据集会先被审计，完成后仍需人工审核与批准。"
        )
        intro.setProperty("textRole", "body")
        intro.setWordWrap(True)
        safety_layout.addWidget(intro, 1)
        layout.addWidget(safety)

        columns = QtWidgets.QHBoxLayout()
        columns.setSpacing(12)
        data_card, data_layout = self._glass_card(
            "数据与基础模型",
            "选择训练输入；工作台会验证路径与权重兼容性。",
            elevated=True,
            badge=make_badge("步骤 1", "neutral"),
        )
        self.dataset_combo = QtWidgets.QComboBox()
        self.dataset_combo.setEditable(True)
        self.dataset_combo.setMinimumWidth(0)
        self.dataset_combo.setMinimumContentsLength(20)
        self.dataset_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.dataset_combo.setAccessibleName("数据集 YAML")
        self.dataset_browse = self._button("浏览", self._browse_dataset, variant="quiet")
        dataset_row = QtWidgets.QHBoxLayout()
        dataset_row.setSpacing(8)
        dataset_row.addWidget(self.dataset_combo, 1)
        dataset_row.addWidget(self.dataset_browse)
        dataset_label = QtWidgets.QLabel("数据集 YAML")
        dataset_label.setProperty("textRole", "fieldLabel")
        data_layout.addWidget(dataset_label)
        data_layout.addLayout(dataset_row)

        self.weights_preset = QtWidgets.QComboBox()
        self.weights_preset.setMinimumWidth(0)
        self.weights_preset.setMinimumContentsLength(20)
        self.weights_preset.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.weights_preset.setToolTip("仅列出与当前目标检测训练流程兼容的官方 YOLOv5 v7.0 权重")
        self.weights_preset.setAccessibleName("预训练权重型号")
        preset_label = QtWidgets.QLabel("预训练型号")
        preset_label.setProperty("textRole", "fieldLabel")
        data_layout.addWidget(preset_label)
        data_layout.addWidget(self.weights_preset)

        self.weights_edit = QtWidgets.QLineEdit()
        self.weights_edit.setReadOnly(True)
        self.weights_edit.setPlaceholderText("自动查找失败，请手动浏览可信的自定义 .pt")
        self.weights_edit.setAccessibleName("实际训练权重文件")
        self.weights_browse = self._button("更换", self._browse_training_weights, variant="quiet")
        weight_row = QtWidgets.QHBoxLayout()
        weight_row.setSpacing(8)
        weight_row.addWidget(self.weights_edit, 1)
        weight_row.addWidget(self.weights_browse)
        weight_label = QtWidgets.QLabel("实际权重文件")
        weight_label.setProperty("textRole", "fieldLabel")
        data_layout.addWidget(weight_label)
        data_layout.addLayout(weight_row)

        weight_surface = QtWidgets.QFrame()
        weight_surface.setProperty("surface", "soft")
        weight_layout = QtWidgets.QHBoxLayout(weight_surface)
        weight_layout.setContentsMargins(12, 10, 12, 10)
        weight_layout.setSpacing(10)
        self.weight_family_badge = make_badge("P5 · 640px", "primary")
        self.weight_advice = QtWidgets.QLabel()
        self.weight_advice.setProperty("textRole", "muted")
        self.weight_advice.setWordWrap(True)
        weight_layout.addWidget(self.weight_family_badge, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        weight_layout.addWidget(self.weight_advice, 1)
        data_layout.addWidget(weight_surface)

        run_card, run_layout = self._glass_card(
            "训练参数",
            "默认值适合当前 Windows 便携环境。",
            elevated=False,
            badge=make_badge("步骤 2", "neutral"),
        )
        self.experiment_name = QtWidgets.QLineEdit("new_target")
        self.experiment_name.setAccessibleName("训练任务名称")
        self.epochs = QtWidgets.QSpinBox()
        self.epochs.setRange(1, 10000)
        self.epochs.setValue(100)
        self.epochs.setSuffix(" 轮")
        self.batch_size = QtWidgets.QSpinBox()
        self.batch_size.setRange(-1, 1024)
        self.batch_size.setValue(-1)
        self.batch_size.setSpecialValueText("自动")
        self.image_size = QtWidgets.QSpinBox()
        self.image_size.setRange(160, 2048)
        self.image_size.setSingleStep(32)
        self.image_size.setValue(640)
        self.image_size.setSuffix(" px")
        self.workers = QtWidgets.QSpinBox()
        self.workers.setRange(0, 32)
        self.workers.setValue(0)
        self.workers.setSuffix(" 线程")
        name_label = QtWidgets.QLabel("任务名称")
        name_label.setProperty("textRole", "fieldLabel")
        run_layout.addWidget(name_label)
        run_layout.addWidget(self.experiment_name)
        parameter_grid = QtWidgets.QGridLayout()
        parameter_grid.setHorizontalSpacing(10)
        parameter_grid.setVerticalSpacing(8)
        parameters = (
            ("训练轮数", self.epochs),
            ("批量大小", self.batch_size),
            ("图像尺寸", self.image_size),
            ("数据线程", self.workers),
        )
        for index, (label, editor) in enumerate(parameters):
            field = QtWidgets.QVBoxLayout()
            field.setSpacing(5)
            label_widget = QtWidgets.QLabel(label)
            label_widget.setProperty("textRole", "fieldLabel")
            field.addWidget(label_widget)
            field.addWidget(editor)
            parameter_grid.addLayout(field, index // 2, index % 2)
            parameter_grid.setColumnStretch(index % 2, 1)
        run_layout.addLayout(parameter_grid)
        recommendation = QtWidgets.QLabel("批量大小设为“自动”、数据线程保持 0，可降低便携环境中的启动失败风险。")
        recommendation.setProperty("textRole", "muted")
        recommendation.setWordWrap(True)
        run_layout.addWidget(recommendation)
        run_layout.addStretch(1)
        columns.addWidget(data_card, 6)
        columns.addWidget(run_card, 4)
        self.training_columns_layout = columns
        layout.addLayout(columns)

        self._populate_weight_presets()
        self.weights_preset.currentIndexChanged.connect(self._on_weight_preset_changed)
        default_index = next((i for i, preset in enumerate(PRESETS) if preset.key == "s"), 0)
        self.weights_preset.setCurrentIndex(default_index)
        self._on_weight_preset_changed(default_index)

        action_card = QtWidgets.QFrame()
        action_card.setObjectName("ActionBar")
        action_card.setProperty("glass", "subtle")
        action_layout = QtWidgets.QHBoxLayout(action_card)
        action_layout.setContentsMargins(14, 11, 14, 11)
        action_layout.setSpacing(8)
        self.training_status_badge = make_badge("待机", "neutral")
        self.btn_check_dataset = self._button("检查数据集", self._check_selected_dataset, variant="quiet")
        self.btn_start_training = self._button("开始训练", self._start_training, primary=True)
        self.btn_resume_training = self._button("从检查点继续", self._resume_training, variant="quiet")
        self.btn_stop_training = self._button("停止", self._stop_training, variant="danger")
        self.btn_stop_training.setEnabled(False)
        action_layout.addWidget(self.training_status_badge)
        action_layout.addStretch(1)
        action_layout.addWidget(self.btn_check_dataset)
        action_layout.addWidget(self.btn_resume_training)
        action_layout.addWidget(self.btn_stop_training)
        action_layout.addWidget(self.btn_start_training)
        layout.addWidget(action_card)

        self.training_log_disclosure = DisclosureCard(
            "训练输出", "进度、审计结果和错误会实时显示。", expanded=False
        )
        log_layout = self.training_log_disclosure.body_layout
        self.training_log = QtWidgets.QPlainTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setPlaceholderText("训练日志会实时显示在这里。关闭工作台前请先确认训练已结束。")
        self.training_log.setMinimumHeight(220)
        log_layout.addWidget(self.training_log)
        layout.addWidget(self.training_log_disclosure)
        return page

    def _build_deployment_tab(self):
        page, layout = self._page()
        safety = QtWidgets.QFrame()
        safety.setObjectName("SafetyStrip")
        safety.setProperty("tone", "warning")
        safety_layout = QtWidgets.QHBoxLayout(safety)
        safety_layout.setContentsMargins(14, 11, 14, 11)
        safety_layout.setSpacing(10)
        safety_layout.addWidget(make_badge("人工门禁", "warning"))
        warning = QtWidgets.QLabel("程序不会自动批准。完成技术检查与 4 项人工复核后，才可复制到 RESULTS。")
        warning.setProperty("textRole", "body")
        warning.setWordWrap(True)
        safety_layout.addWidget(warning, 1)
        layout.addWidget(safety)

        columns = QtWidgets.QHBoxLayout()
        columns.setSpacing(12)
        evidence_card, evidence_layout = self._glass_card(
            "候选与证据",
            "先读取技术指纹，再用训练记录和真实媒体判断是否适合生产。",
            elevated=False,
            badge=make_badge("步骤 1", "neutral"),
        )
        candidate_label = QtWidgets.QLabel("候选 best.pt")
        candidate_label.setProperty("textRole", "fieldLabel")
        evidence_layout.addWidget(candidate_label)
        candidate_row = QtWidgets.QHBoxLayout()
        candidate_row.setSpacing(8)
        self.candidate_combo = QtWidgets.QComboBox()
        self.candidate_combo.setEditable(True)
        self.candidate_combo.setMinimumWidth(0)
        self.candidate_combo.setMinimumContentsLength(16)
        self.candidate_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.candidate_combo.setAccessibleName("候选模型 best.pt")
        candidate_row.addWidget(self.candidate_combo, 1)
        candidate_row.addWidget(self._button("浏览", self._browse_candidate, variant="quiet"))
        candidate_row.addWidget(self._button("读取指纹与类别", self._inspect_candidate, primary=True))
        evidence_layout.addLayout(candidate_row)

        self.candidate_summary = QtWidgets.QLabel("尚未读取候选指纹与类别")
        self.candidate_summary.setObjectName("CandidateSummary")
        self.candidate_summary.setProperty("surface", "soft")
        self.candidate_summary.setWordWrap(True)
        self.candidate_summary.setContentsMargins(12, 10, 12, 10)
        evidence_layout.addWidget(self.candidate_summary)

        evidence_actions = QtWidgets.QHBoxLayout()
        evidence_actions.setSpacing(8)
        evidence_actions.addWidget(self._button("打开训练目录", self._open_candidate_run, variant="quiet"))
        evidence_actions.addWidget(self._button("使用真实媒体测试", self._test_candidate_media, variant="quiet"))
        evidence_actions.addWidget(self._button("查看测试结果", lambda: self._open_path(TRAINING_OUTPUTS_DIR / "_REVIEWS"), variant="quiet"))
        evidence_layout.addLayout(evidence_actions)

        identity_header = QtWidgets.QLabel("生产身份与类别映射")
        identity_header.setProperty("textRole", "sectionTitle")
        evidence_layout.addWidget(identity_header)
        identity_grid = QtWidgets.QGridLayout()
        identity_grid.setHorizontalSpacing(10)
        identity_grid.setVerticalSpacing(6)
        self.deploy_model_id = QtWidgets.QLineEdit()
        self.deploy_display_name = QtWidgets.QLineEdit()
        self.deploy_model_id.setPlaceholderText("例如 fire_v2")
        self.deploy_display_name.setPlaceholderText("界面中的模型名称")
        for column, (label, editor) in enumerate((
            ("模型 ID", self.deploy_model_id),
            ("显示名称", self.deploy_display_name),
        )):
            field_label = QtWidgets.QLabel(label)
            field_label.setProperty("textRole", "fieldLabel")
            identity_grid.addWidget(field_label, 0, column)
            identity_grid.addWidget(editor, 1, column)
            identity_grid.setColumnStretch(column, 1)
        evidence_layout.addLayout(identity_grid)

        self.class_table = QtWidgets.QTableWidget(0, 7)
        self.class_table.setHorizontalHeaderLabels([
            "权重类别", "统一 ID", "中文名", "英文名", "等级", "告警", "颜色",
        ])
        header = self.class_table.horizontalHeader()
        fixed_widths = {0: 80, 1: 108, 4: 112, 5: 58, 6: 102}
        for column, width in fixed_widths.items():
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.Fixed)
            header.resizeSection(column, width)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.class_table.setAlternatingRowColors(True)
        self.class_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.class_table.verticalHeader().setVisible(False)
        self.class_table.setMinimumHeight(220)
        evidence_layout.addWidget(self.class_table, 1)

        self.review_progress_badge = make_badge("0 / 4", "warning")
        gate_card, gate_layout = self._glass_card(
            "生产安全门禁",
            "全部条件都必须由审核人逐项确认。",
            elevated=True,
            badge=self.review_progress_badge,
        )
        self.review_checks = {
            "metrics_reviewed": QtWidgets.QCheckBox("指标与训练曲线已核对"),
            "class_order_reviewed": QtWidgets.QCheckBox("类别名称与编号顺序已核对"),
            "failure_samples_reviewed": QtWidgets.QCheckBox("漏报、误报与失败样本已检查"),
            "real_media_reviewed": QtWidgets.QCheckBox("真实图片或视频已完成测试"),
        }
        for index, checkbox in enumerate(self.review_checks.values(), start=1):
            checkbox.setAccessibleName("人工审核条件 %d" % index)
            check_surface = QtWidgets.QFrame()
            check_surface.setProperty("surface", "soft")
            check_layout = QtWidgets.QHBoxLayout(check_surface)
            check_layout.setContentsMargins(11, 8, 11, 8)
            check_layout.addWidget(checkbox)
            gate_layout.addWidget(check_surface)

        reviewer_label = QtWidgets.QLabel("审核人")
        reviewer_label.setProperty("textRole", "fieldLabel")
        self.reviewer = QtWidgets.QLineEdit("user")
        notes_label = QtWidgets.QLabel("审核备注")
        notes_label.setProperty("textRole", "fieldLabel")
        self.review_notes = QtWidgets.QLineEdit()
        self.review_notes.setPlaceholderText("说明批准依据或已知限制")
        gate_layout.addWidget(reviewer_label)
        gate_layout.addWidget(self.reviewer)
        gate_layout.addWidget(notes_label)
        gate_layout.addWidget(self.review_notes)

        replace_surface = QtWidgets.QFrame()
        replace_surface.setObjectName("ReplaceWarning")
        replace_surface.setProperty("tone", "danger")
        replace_layout = QtWidgets.QVBoxLayout(replace_surface)
        replace_layout.setContentsMargins(11, 9, 11, 9)
        self.replace_existing = QtWidgets.QCheckBox("替换同 ID 的生产模型")
        replace_detail = QtWidgets.QLabel("仅在明确需要覆盖时勾选；旧版本会自动归档。")
        replace_detail.setProperty("textRole", "muted")
        replace_detail.setWordWrap(True)
        replace_layout.addWidget(self.replace_existing)
        replace_layout.addWidget(replace_detail)
        gate_layout.addWidget(replace_surface)
        gate_layout.addStretch(1)
        self.review_gate_hint = QtWidgets.QLabel("先读取候选模型，再完成 4 项人工确认。")
        self.review_gate_hint.setProperty("textRole", "muted")
        self.review_gate_hint.setWordWrap(True)
        gate_layout.addWidget(self.review_gate_hint)
        self.btn_deploy = self._button("人工批准并部署", self._deploy_candidate, primary=True)
        self.btn_deploy.setMinimumHeight(42)
        self.btn_deploy.setEnabled(False)
        gate_layout.addWidget(self.btn_deploy)

        columns.addWidget(evidence_card, 7)
        columns.addWidget(gate_card, 4)
        self.review_columns_layout = columns
        layout.addLayout(columns)
        layout.addStretch(1)
        self.candidate_combo.currentTextChanged.connect(self._candidate_selection_changed)
        self.deploy_model_id.textChanged.connect(self._update_review_gate)
        for checkbox in self.review_checks.values():
            checkbox.toggled.connect(self._update_review_gate)
        return page

    def _build_models_tab(self):
        page, layout = self._page()
        summary = QtWidgets.QHBoxLayout()
        summary.setSpacing(10)
        total_card, self.models_total_metric = self._metric_card("生产模型", "—", "RESULTS 中已批准", "primary")
        ready_card, self.models_ready_metric = self._metric_card("当前可用", "—", "通过完整性检查", "success")
        policy_card, self.models_policy_metric = self._metric_card("加载策略", "手动选择", "新模型默认不启用", "neutral")
        for card in (total_card, ready_card, policy_card):
            summary.addWidget(card, 1)
        self.models_summary_layout = summary
        self.model_summary_cards = (total_card, ready_card, policy_card)
        layout.addLayout(summary)

        catalog_card, catalog_layout = self._glass_card(
            "生产模型目录",
            "SENTINEL 动态扫描 RESULTS；新增模型仍需在检测窗口中手动勾选。",
            elevated=True,
            badge=make_badge("只读视图", "neutral"),
        )
        actions = QtWidgets.QHBoxLayout()
        actions.setSpacing(8)
        actions.addWidget(self._button("重新扫描 RESULTS", self._refresh_models, primary=True))
        actions.addWidget(self._button("打开生产目录", lambda: self._open_path(RESULTS_DIR), variant="quiet"))
        actions.addWidget(self._button("打开归档", lambda: self._open_path(PROJECT_ROOT / "MODEL_ARCHIVE"), variant="quiet"))
        actions.addStretch(1)
        catalog_layout.addLayout(actions)
        self.models_table = QtWidgets.QTableWidget(0, 5)
        self.models_table.setHorizontalHeaderLabels(["模型 ID", "显示名", "类别", "状态", "权重路径"])
        header = self.models_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.models_table.setAlternatingRowColors(True)
        self.models_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.models_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.models_table.setSortingEnabled(True)
        self.models_table.setShowGrid(False)
        self.models_table.verticalHeader().setVisible(False)
        self.models_table.setMinimumHeight(420)
        catalog_layout.addWidget(self.models_table, 1)
        self.models_hint = QtWidgets.QLabel()
        self.models_hint.setProperty("textRole", "muted")
        self.models_hint.setWordWrap(True)
        catalog_layout.addWidget(self.models_hint)
        layout.addWidget(catalog_card, 1)
        return page

    def _populate_weight_presets(self):
        self.weights_preset.clear()
        for preset in PRESETS:
            prefix = "标准 640" if not preset.is_p6 else "高级 1280"
            label = "%s  |  %s  ·  精度%s / 显存%s" % (
                prefix, preset.display_name, preset.accuracy, preset.memory,
            )
            self.weights_preset.addItem(label, preset)
            index = self.weights_preset.count() - 1
            if not preset.path.is_file():
                item = self.weights_preset.model().item(index)
                if item is not None:
                    item.setEnabled(False)
                self.weights_preset.setItemData(
                    index, "文件缺失：%s" % preset.path, QtCore.Qt.ItemDataRole.ToolTipRole
                )
            else:
                self.weights_preset.setItemData(
                    index, preset.recommendation, QtCore.Qt.ItemDataRole.ToolTipRole
                )

    def _on_weight_preset_changed(self, index):
        preset = self.weights_preset.itemData(index)
        if not isinstance(preset, WeightPreset):
            return
        self.weights_edit.setText(str(preset.path))
        recommended_size = preset.image_size
        if self.image_size.value() in (self._last_recommended_image_size, 640, 1280):
            self.image_size.setValue(recommended_size)
        self._last_recommended_image_size = recommended_size
        self.weight_family_badge.setText("%s · %dpx" % (preset.family, recommended_size))
        set_property(self.weight_family_badge, "badge", "warning" if preset.is_p6 else "primary")
        self.weight_advice.setText(
            "%s：速度%s，精度%s，显存占用%s。%s。官方 COCO 预训练权重仅用于迁移学习初始化。"
            % (preset.display_name, preset.speed, preset.accuracy, preset.memory, preset.recommendation)
        )

    def _show_custom_weight_advice(self, path):
        self.weights_preset.blockSignals(True)
        self.weights_preset.setCurrentIndex(-1)
        self.weights_preset.blockSignals(False)
        self.weight_family_badge.setText("CUSTOM")
        set_property(self.weight_family_badge, "badge", "warning")
        self.weight_advice.setText(
            "自定义权重：%s。只可选择来源可信、与 YOLOv5 目标检测 train.py 兼容的 .pt；"
            "分割或分类权重不能用于此页面。" % Path(path).name
        )

    def _refresh_datasets(self):
        paths = set(PROJECT_ROOT.glob("DATASETS/**/dataset.yaml"))
        paths.update(PROJECT_ROOT.glob("data/*.yaml"))
        current = self.dataset_combo.currentText() if hasattr(self, "dataset_combo") else ""
        self.dataset_combo.clear()
        for path in sorted(paths, key=lambda item: str(item).lower()):
            self.dataset_combo.addItem(str(path))
        if current:
            self.dataset_combo.setCurrentText(current)
        else:
            recommended = PROJECT_ROOT / "DATASETS" / "combined_legacy_v1" / "dataset.yaml"
            if recommended.is_file():
                self.dataset_combo.setCurrentText(str(recommended))

    def _refresh_candidates(self):
        if not hasattr(self, "candidate_combo"):
            return
        candidates = sorted(
            TRAINING_OUTPUTS_DIR.glob("**/weights/best.pt"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        current = self.candidate_combo.currentText()
        blocker = QtCore.QSignalBlocker(self.candidate_combo)
        self.candidate_combo.clear()
        for path in candidates:
            self.candidate_combo.addItem(str(path))
        if current:
            self.candidate_combo.setCurrentText(current)
        new_current = self.candidate_combo.currentText()
        if self.candidate_combo.lineEdit() is not None:
            self.candidate_combo.lineEdit().setCursorPosition(0)
            self.candidate_combo.lineEdit().deselect()
        del blocker
        if current and new_current != current:
            self._candidate_selection_changed(new_current)
        if hasattr(self, "home_candidate_metric"):
            self.home_candidate_metric.setText(str(len(candidates)))

    def _refresh_models(self):
        if not hasattr(self, "models_table"):
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            snapshot = ModelCatalog().scan(verify_hashes=True, inspect_new_models=True)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        sorting_enabled = self.models_table.isSortingEnabled()
        self.models_table.setSortingEnabled(False)
        self.models_table.setRowCount(len(snapshot.models))
        for row, model in enumerate(snapshot.models):
            values = [
                model.model_id,
                model.display_name,
                ", ".join(model.canonical_classes),
                "可用" if model.selectable else "已隔离：%s" % (model.issue or "不可用"),
                model.relative_weight_path,
            ]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if column == 3:
                    item.setForeground(QtGui.QColor("#78F0C5" if model.selectable else "#FF9AAE"))
                self.models_table.setItem(row, column, item)
        self.models_table.setSortingEnabled(sorting_enabled)
        total = len(snapshot.models)
        ready = len(snapshot.selectable_models)
        self.models_hint.setText(
            "共发现 %d 个生产模型，其中 %d 个可用。新发现模型在实时检测窗口中默认不选中。"
            % (total, ready)
        )
        if hasattr(self, "models_total_metric"):
            self.models_total_metric.setText(str(total))
            self.models_ready_metric.setText(str(ready))
        if hasattr(self, "home_model_metric"):
            self.home_model_metric.setText(str(total))

    def _browse_dataset(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择数据集 YAML", str(PROJECT_ROOT), "YAML (*.yaml *.yml)")
        if path:
            self.dataset_combo.setCurrentText(path)

    def _browse_training_weights(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择初始权重", str(PRETRAINED_WEIGHTS_DIR), "PyTorch 权重 (*.pt)"
        )
        if path:
            self.weights_edit.setText(path)
            self._show_custom_weight_advice(path)

    def _browse_candidate(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择候选模型", str(TRAINING_OUTPUTS_DIR), "PyTorch 权重 (*.pt)")
        if path:
            self.candidate_combo.setCurrentText(path)

    def _run_utility(self, script, arguments=None, target_log=None):
        if self.utility_process is not None and self.utility_process.state() != QtCore.QProcess.ProcessState.NotRunning:
            QtWidgets.QMessageBox.information(self, "任务进行中", "请等待当前检查完成。")
            return
        log = target_log or self.home_log
        log.clear()
        if log is self.home_log and hasattr(self, "home_log_disclosure"):
            self.home_log_disclosure.set_expanded(True)
        process = QtCore.QProcess(self)
        self._prepare_process(process)
        process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        process.setWorkingDirectory(str(PROJECT_ROOT))
        process.readyReadStandardOutput.connect(lambda: self._append_process_output(process, log))
        process.finished.connect(lambda code, status: log.appendPlainText("\n[结束] 退出码 %d" % code))
        process.finished.connect(lambda *_: self._refresh_candidates())
        self.utility_process = process
        process.start(PYTHON_EXECUTABLE, [str(PROJECT_ROOT / script)] + list(arguments or ()))
        return process

    @staticmethod
    def _render_terminal_text(log, text, overwrite_current=False):
        """Render terminal CR/LF semantics without duplicating tqdm progress rows."""
        cursor = log.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)

        for token in re.split(r"(\r\n|\r|\n)", text):
            if not token:
                continue
            if token == "\r":
                overwrite_current = True
                continue
            if token in ("\n", "\r\n"):
                cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
                cursor.insertBlock()
                overwrite_current = False
                continue

            cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
            if overwrite_current:
                cursor.movePosition(
                    QtGui.QTextCursor.MoveOperation.StartOfBlock,
                    QtGui.QTextCursor.MoveMode.KeepAnchor,
                )
                cursor.removeSelectedText()
                overwrite_current = False
            cursor.insertText(token)

        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        log.setTextCursor(cursor)
        log.ensureCursorVisible()
        return overwrite_current

    @classmethod
    def _append_process_output(cls, process, log):
        raw = bytes(process.readAllStandardOutput())
        if not raw:
            return

        decoder = getattr(process, "_sentinel_utf8_decoder", None)
        if decoder is None:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            process._sentinel_utf8_decoder = decoder
        text = decoder.decode(raw, final=False)
        if not text:
            return

        overwrite_current = getattr(process, "_sentinel_overwrite_current", False)
        process._sentinel_overwrite_current = cls._render_terminal_text(
            log, text, overwrite_current
        )

    def _check_selected_dataset(self):
        dataset = self.dataset_combo.currentText().strip()
        if not dataset:
            QtWidgets.QMessageBox.warning(self, "缺少数据集", "请先选择数据集 YAML。")
            return
        process = self._run_utility("dataset_audit.py", [dataset], self.training_log)
        if process is not None:
            self.training_log_disclosure.set_expanded(True)
            self.training_status_badge.setText("正在审计")
            set_property(self.training_status_badge, "badge", "primary")
            process.finished.connect(self._dataset_check_finished)

    def _dataset_check_finished(self, exit_code, _status):
        self.training_status_badge.setText("数据可用" if exit_code == 0 else "审计失败")
        set_property(self.training_status_badge, "badge", "success" if exit_code == 0 else "danger")

    def _start_training(self):
        if self.training_process is not None and self.training_process.state() != QtCore.QProcess.ProcessState.NotRunning:
            QtWidgets.QMessageBox.information(self, "训练进行中", "当前已有训练任务。")
            return
        dataset = self.dataset_combo.currentText().strip()
        weights = self.weights_edit.text().strip()
        if not Path(dataset).is_file() or not Path(weights).is_file():
            QtWidgets.QMessageBox.warning(self, "路径错误", "数据集 YAML 或初始权重不存在。")
            return
        valid_weight, weight_message = validate_training_weight(weights)
        if not valid_weight:
            QtWidgets.QMessageBox.critical(self, "权重校验失败", weight_message)
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "开始训练",
            "训练会长期占用 GPU。建议先关闭 SENTINEL 检测窗口。\n\n确认开始？",
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        name = re.sub(r"[^a-zA-Z0-9_-]+", "_", self.experiment_name.text()).strip("_") or "candidate"
        arguments = [
            str(PROJECT_ROOT / "safe_train.py"),
            "--data", dataset,
            "--weights", weights,
            "--epochs", str(self.epochs.value()),
            "--batch-size", str(self.batch_size.value()),
            "--imgsz", str(self.image_size.value()),
            "--workers", str(self.workers.value()),
            "--device", "0",
            "--name", name,
        ]
        self.training_log.clear()
        self.training_log_disclosure.set_expanded(True)
        self.training_log.appendPlainText(
            "[OK] %s\n\n即将执行：\n%s %s\n"
            % (weight_message, PYTHON_EXECUTABLE, " ".join(arguments))
        )
        process = QtCore.QProcess(self)
        self._prepare_process(process)
        process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        process.setWorkingDirectory(str(PROJECT_ROOT))
        process.readyReadStandardOutput.connect(lambda: self._append_process_output(process, self.training_log))
        process.finished.connect(self._training_finished)
        self.training_process = process
        self.btn_start_training.setEnabled(False)
        self.btn_resume_training.setEnabled(False)
        self.btn_stop_training.setEnabled(True)
        self.training_status_badge.setText("训练中")
        set_property(self.training_status_badge, "badge", "primary")
        process.start(PYTHON_EXECUTABLE, arguments)

    def _resume_training(self):
        if self.training_process is not None and self.training_process.state() != QtCore.QProcess.ProcessState.NotRunning:
            QtWidgets.QMessageBox.information(self, "训练进行中", "当前已有训练任务。")
            return
        checkpoint, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择候选续训 checkpoint", str(TRAINING_OUTPUTS_DIR), "YOLO last checkpoint (last.pt)"
        )
        if not checkpoint:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "安全续训", "程序会重新审计原数据集，再从所选 last.pt 续训。\n建议先关闭 SENTINEL。确认继续？"
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        arguments = [str(PROJECT_ROOT / "safe_train.py"), "--safe-resume", checkpoint]
        self.training_log.clear()
        self.training_log_disclosure.set_expanded(True)
        process = QtCore.QProcess(self)
        self._prepare_process(process)
        process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        process.setWorkingDirectory(str(PROJECT_ROOT))
        process.readyReadStandardOutput.connect(lambda: self._append_process_output(process, self.training_log))
        process.finished.connect(self._training_finished)
        self.training_process = process
        self.btn_start_training.setEnabled(False)
        self.btn_resume_training.setEnabled(False)
        self.btn_stop_training.setEnabled(True)
        self.training_status_badge.setText("续训中")
        set_property(self.training_status_badge, "badge", "primary")
        process.start(PYTHON_EXECUTABLE, arguments)

    def _stop_training(self):
        if self.training_process is None or self.training_process.state() == QtCore.QProcess.ProcessState.NotRunning:
            return
        answer = QtWidgets.QMessageBox.question(self, "停止训练", "确认停止当前训练？已写入的 checkpoint 会保留。")
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self.training_status_badge.setText("正在停止")
            set_property(self.training_status_badge, "badge", "warning")
            self._terminate_training_tree()

    def _terminate_training_tree(self):
        process = self.training_process
        if process is None or process.state() == QtCore.QProcess.ProcessState.NotRunning:
            return
        pid = int(process.processId())
        self.training_log.appendPlainText("\n[正在停止] 终止本次训练及其数据加载子进程…")
        if os.name == "nt" and pid > 0:
            try:
                subprocess.run(
                    ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    timeout=15,
                )
            except (OSError, subprocess.SubprocessError):
                process.kill()
        else:
            process.terminate()
            if not process.waitForFinished(5000):
                process.kill()

    def _training_finished(self, exit_code, _status):
        self.training_log.appendPlainText("\n[训练进程结束] 退出码 %d" % exit_code)
        self.btn_start_training.setEnabled(True)
        self.btn_resume_training.setEnabled(True)
        self.btn_stop_training.setEnabled(False)
        self.training_status_badge.setText("训练完成" if exit_code == 0 else "训练已结束")
        set_property(self.training_status_badge, "badge", "success" if exit_code == 0 else "danger")
        self._refresh_candidates()
        if exit_code == 0:
            QtWidgets.QMessageBox.information(self, "训练完成", "候选模型已保存在 TRAINING_OUTPUTS。\n请进入“审核与部署”页并在人工检查后部署。")

    def _start_sentinel(self):
        ok, _pid = QtCore.QProcess.startDetached(PYTHON_EXECUTABLE, [str(PROJECT_ROOT / "launcher_GPT.py")], str(PROJECT_ROOT))
        if not ok:
            QtWidgets.QMessageBox.warning(self, "启动失败", "无法启动 SENTINEL。请先运行环境检查。")

    def _candidate_selection_changed(self, _text=None):
        self.candidate_info = None
        self.candidate_summary.setText("尚未读取候选指纹与类别")
        self.deploy_model_id.clear()
        self.deploy_display_name.clear()
        self.class_table.setRowCount(0)
        self.class_editors = {}
        for checkbox in self.review_checks.values():
            checkbox.setChecked(False)
        self.review_notes.clear()
        self.replace_existing.setChecked(False)
        self._update_review_gate()

    def _update_review_gate(self, *_args):
        if not hasattr(self, "review_checks"):
            return
        checked = sum(checkbox.isChecked() for checkbox in self.review_checks.values())
        current_path = Path(self.candidate_combo.currentText().strip()).resolve()
        inspected = self.candidate_info is not None and self.candidate_info.path == current_path
        try:
            safe_model_id(self.deploy_model_id.text())
            valid_model_id = True
        except ValueError:
            valid_model_id = False
        ready = inspected and checked == len(self.review_checks) and valid_model_id
        self.review_progress_badge.setText("%d / %d" % (checked, len(self.review_checks)))
        set_property(
            self.review_progress_badge,
            "badge",
            "success" if ready else ("primary" if checked else "warning"),
        )
        self.btn_deploy.setEnabled(ready)
        if not inspected:
            message = "先读取候选模型的指纹与类别。"
        elif not valid_model_id:
            message = "填写有效的生产模型 ID。"
        elif checked < len(self.review_checks):
            message = "还需确认 %d 项人工检查。" % (len(self.review_checks) - checked)
        else:
            message = "门禁已完成。点击后仍会显示最终部署确认。"
        self.review_gate_hint.setText(message)

    def _inspect_candidate(self):
        path = self.candidate_combo.currentText().strip()
        self._candidate_selection_changed(path)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            info = inspect_candidate(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "候选模型不可用", str(exc))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.candidate_info = info
        run_name = info.path.parent.parent.name if info.path.parent.name.lower() == "weights" else info.path.stem
        suggested_id = safe_model_id(run_name)
        self.deploy_model_id.setText(suggested_id)
        self.deploy_display_name.setText(run_name.replace("_", " "))
        self.candidate_summary.setText(
            "技术检查通过 · %.1f MB · SHA-256 %s… · 类别顺序：%s"
            % (info.bytes / (1024 ** 2), info.sha256[:16], ", ".join(info.classes))
        )
        self._populate_class_table(info.classes)
        self._update_review_gate()

    def _populate_class_table(self, classes):
        defaults = {
            "fire": ("fire", "火焰", "Fire", "critical", True, "#F43F5E"),
            "person": ("person", "未戴安全帽", "NO HELMET", "warning", True, "#38BDF8"),
            "hat": ("hat", "已戴安全帽", "Helmet", "none", False, "#EAB308"),
        }
        self.class_table.setRowCount(len(classes))
        self.class_editors = {}
        for row, raw_name in enumerate(classes):
            self.class_table.setRowHeight(row, 46)
            fallback_id = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", raw_name).strip("_.:-").lower()
            if not fallback_id:
                fallback_id = "target_%d" % (row + 1)
            canonical, zh, en, severity_value, alert_value, color_value = defaults.get(
                raw_name, (fallback_id, raw_name, raw_name, "none", False, "#60A5FA")
            )
            raw_item = QtWidgets.QTableWidgetItem(raw_name); raw_item.setFlags(raw_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            canonical_edit = QtWidgets.QLineEdit(canonical)
            zh_edit = QtWidgets.QLineEdit(zh)
            en_edit = QtWidgets.QLineEdit(en)
            severity = QtWidgets.QComboBox(); severity.addItems(["none", "info", "warning", "critical"]); severity.setCurrentText(severity_value)
            alert = QtWidgets.QCheckBox(); alert.setChecked(alert_value)
            color = QtWidgets.QLineEdit(color_value)
            self.class_table.setItem(row, 0, raw_item)
            for column, widget in ((1, canonical_edit), (2, zh_edit), (3, en_edit), (4, severity), (5, alert), (6, color)):
                self.class_table.setCellWidget(row, column, widget)
            self.class_editors[raw_name] = (canonical_edit, zh_edit, en_edit, severity, alert, color)

    def _class_settings(self):
        settings = {}
        for raw_name, editors in self.class_editors.items():
            canonical, zh, en, severity, alert, color = editors
            settings[raw_name] = {
                "canonical_id": canonical.text().strip(),
                "display_zh": zh.text().strip(),
                "display_en": en.text().strip(),
                "severity": None if severity.currentText() == "none" else severity.currentText(),
                "alert_enabled": alert.isChecked(),
                "color": color.text().strip(),
            }
        return settings

    def _open_candidate_run(self):
        path = Path(self.candidate_combo.currentText().strip())
        if path.is_file():
            self._open_path(path.parent.parent if path.parent.name.lower() == "weights" else path.parent)

    def _test_candidate_media(self):
        current_path = Path(self.candidate_combo.currentText().strip()).resolve()
        if self.candidate_info is None or self.candidate_info.path != current_path:
            QtWidgets.QMessageBox.warning(self, "尚未技术检查", "请先点击“读取指纹与类别”。")
            return
        media_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择人工测试图片或视频",
            str(PROJECT_ROOT),
            "媒体 (*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mkv *.mov)",
        )
        if not media_path:
            return
        review_root = TRAINING_OUTPUTS_DIR / "_REVIEWS"
        try:
            review_model_id = safe_model_id(self.deploy_model_id.text() or current_path.stem)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "模型 ID 错误", str(exc))
            return
        review_name = "%s_%s" % (datetime.now().strftime("%Y%m%d_%H%M%S"), review_model_id)
        output_directory = review_root / review_name
        process = self._run_utility(
            "detect.py",
            [
                "--weights", str(current_path),
                "--source", media_path,
                "--project", str(review_root),
                "--name", review_name,
                "--exist-ok",
                "--save-txt",
                "--save-conf",
            ],
            self.home_log,
        )
        if process is None:
            return

        def finished(exit_code, _status):
            if exit_code == 0 and output_directory.is_dir():
                self._open_path(output_directory)
                QtWidgets.QMessageBox.information(
                    self,
                    "人工测试完成",
                    "已打开带检测框的结果。请同时检查漏报、误报和类别名称；程序不会自动勾选审核项。",
                )
            else:
                QtWidgets.QMessageBox.warning(self, "人工测试失败", "请在“开始”页查看运行日志。")

        process.finished.connect(finished)

    def _deploy_candidate(self):
        if self.candidate_info is None or self.candidate_info.path != Path(self.candidate_combo.currentText().strip()).resolve():
            QtWidgets.QMessageBox.warning(self, "尚未技术检查", "请先点击“读取指纹与类别”。")
            return
        try:
            model_id = safe_model_id(self.deploy_model_id.text())
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "模型 ID 错误", str(exc)); return
        review = {key: checkbox.isChecked() for key, checkbox in self.review_checks.items()}
        review["notes"] = self.review_notes.text().strip()
        missing = [checkbox.text() for key, checkbox in self.review_checks.items() if not review[key]]
        if missing:
            QtWidgets.QMessageBox.warning(self, "审核未完成", "以下项目尚未确认：\n- " + "\n- ".join(missing))
            return
        message = (
            "这是最终部署动作。\n\n模型：%s\n类别：%s\nSHA-256：%s\n\n"
            "模型会复制到 RESULTS，SENTINEL 下次启动或重新扫描后可勾选。确认批准？"
            % (model_id, ", ".join(self.candidate_info.classes), self.candidate_info.sha256)
        )
        class_settings = self._class_settings()
        existing_profiles = ModelCatalog().scan(
            verify_hashes=False, inspect_new_models=False
        ).class_profiles
        reused_profiles = sorted({
            str(item.get("canonical_id", "")).strip().lower()
            for item in class_settings.values()
            if str(item.get("canonical_id", "")).strip().lower() in existing_profiles
        })
        if reused_profiles:
            message += (
                "\n\n这些统一类别沿用现有全局名称与告警规则，不会被本次部署静默修改：\n%s"
                % ", ".join(reused_profiles)
            )
        answer = QtWidgets.QMessageBox.warning(
            self, "最终批准", message,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            result = DeploymentManager().deploy(
                candidate=self.candidate_info,
                model_id=model_id,
                display_name=self.deploy_display_name.text(),
                class_settings=class_settings,
                review=review,
                approved=True,
                replace_existing=self.replace_existing.isChecked(),
                reviewer=self.reviewer.text().strip() or "user",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "部署失败", str(exc))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._refresh_models()
        QtWidgets.QMessageBox.information(
            self, "部署完成",
            "模型已部署：\n%s\n\nSENTINEL 会在下次启动或点击“重新扫描 RESULTS”后发现它。" % result.weight_path,
        )

    def closeEvent(self, event):
        if self.training_process is not None and self.training_process.state() != QtCore.QProcess.ProcessState.NotRunning:
            answer = QtWidgets.QMessageBox.question(self, "训练仍在进行", "关闭工作台会停止当前训练。确认关闭？")
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore(); return
            self._terminate_training_tree()
            self.training_process.waitForFinished(5000)
        event.accept()


def main():
    os.chdir(str(PROJECT_ROOT))
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("SentinelVision Workspace")
    install_ui_font(app)
    window = WorkspaceManager()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
