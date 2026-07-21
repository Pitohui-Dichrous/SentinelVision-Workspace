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
from ui_theme import build_stylesheet, make_badge, make_card, set_property


PYTHON_EXECUTABLE = str(RUNTIME_PYTHON)


class WorkspaceManager(QtWidgets.QMainWindow):
    PAGE_META = (
        ("工作库概览", "完事开头难，不如先从检查环境开始吧！"),
        ("训练新模型", "选择训练用数据集和YOLO官方预训练权重"),
        ("审核与部署", "按步骤检查候选模型"),
        ("已部署模型", "在这里查看 SENTINEL 启动时能够扫描到的正式模型"),
    )

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SentinelVision · AI 安全视觉工作台")
        self.resize(1320, 860)
        self.setMinimumSize(900, 600)
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
        shell = QtWidgets.QWidget()
        shell.setObjectName("AppShell")
        shell_layout = QtWidgets.QHBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        sidebar = QtWidgets.QFrame()
        sidebar.setObjectName("SideNav")
        sidebar.setFixedWidth(226)
        nav_layout = QtWidgets.QVBoxLayout(sidebar)
        nav_layout.setContentsMargins(18, 22, 18, 18)
        nav_layout.setSpacing(8)

        brand_row = QtWidgets.QHBoxLayout()
        logo = make_badge("🧠🤖", "primary")
        logo.setFixedSize(48, 34)
        brand_text = QtWidgets.QVBoxLayout()
        brand_text.setSpacing(0)
        brand = QtWidgets.QLabel("SentinelVision")
        brand.setProperty("textRole", "brand")
        edition = QtWidgets.QLabel("AI SAFETY PLATFORM")
        edition.setProperty("textRole", "eyebrow")
        brand_text.addWidget(brand)
        brand_text.addWidget(edition)
        brand_row.addWidget(logo)
        brand_row.addLayout(brand_text, 1)
        nav_layout.addLayout(brand_row)
        nav_layout.addSpacing(24)

        self.nav_group = QtWidgets.QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons = []
        nav_items = (
            "概览    OVERVIEW",
            "训练    TRAIN",
            "审核    REVIEW",
            "模型    MODELS",
        )
        for index, text in enumerate(nav_items):
            button = QtWidgets.QPushButton(text)
            button.setCheckable(True)
            button.setProperty("variant", "nav")
            self.nav_group.addButton(button, index)
            self.nav_buttons.append(button)
            nav_layout.addWidget(button)
        nav_layout.addStretch(1)

        offline = make_badge("●  离线版本运行中...", "success")
        offline.setToolTip("运行环境、缓存和训练输出全部保存在本地移动工作库中")
        nav_layout.addWidget(offline)
        location = QtWidgets.QLabel(str(PROJECT_ROOT))
        location.setProperty("textRole", "muted")
        location.setWordWrap(True)
        location.setToolTip(str(PROJECT_ROOT))
        nav_layout.addWidget(location)

        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        top_bar = QtWidgets.QFrame()
        top_bar.setObjectName("TopBar")
        top_layout = QtWidgets.QHBoxLayout(top_bar)
        top_layout.setContentsMargins(26, 16, 26, 16)
        page_text = QtWidgets.QVBoxLayout()
        page_text.setSpacing(2)
        self.page_title = QtWidgets.QLabel()
        self.page_title.setProperty("textRole", "pageTitle")
        self.page_subtitle = QtWidgets.QLabel()
        self.page_subtitle.setProperty("textRole", "muted")
        page_text.addWidget(self.page_title)
        page_text.addWidget(self.page_subtitle)
        top_layout.addLayout(page_text)
        top_layout.addStretch(1)
        gpu_ready = torch.cuda.is_available()
        gpu_text = torch.cuda.get_device_name(0) if gpu_ready else "CUDA 不可用"
        self.header_device_badge = make_badge(gpu_text, "success" if gpu_ready else "warning")
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

    def _show_page(self, index):
        if not 0 <= index < len(self.PAGE_META):
            return
        self.pages.setCurrentIndex(index)
        title, subtitle = self.PAGE_META[index]
        self.page_title.setText(title)
        self.page_subtitle.setText(subtitle)

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
        layout.setContentsMargins(26, 22, 26, 26)
        layout.setSpacing(16)
        scroll.setWidget(outer)
        return scroll, layout

    @staticmethod
    def _button(text, callback, primary=False):
        button = QtWidgets.QPushButton(text)
        if primary:
            button.setProperty("variant", "primary")
        button.clicked.connect(callback)
        return button

    @staticmethod
    def _metric_card(label, value, detail, tone="neutral"):
        card = make_card(elevated=tone != "neutral")
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(4)
        caption = QtWidgets.QLabel(label.upper())
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
        hero_layout = QtWidgets.QHBoxLayout(hero)
        hero_layout.setContentsMargins(22, 20, 22, 20)
        hero_copy = QtWidgets.QVBoxLayout()
        hero_copy.setSpacing(6)
        eyebrow = QtWidgets.QLabel("SENTINEL OPERATIONS")
        eyebrow.setProperty("textRole", "eyebrow")
        headline = QtWidgets.QLabel("欢迎使用 SentinelVision [Beta] 控制台")
        headline.setProperty("textRole", "pageTitle")
        description = QtWidgets.QLabel("仅作内部学习用途，请勿商用。支持PITO谢谢喵")
        description.setProperty("textRole", "muted")
        description.setWordWrap(True)
        hero_copy.addWidget(eyebrow)
        hero_copy.addWidget(headline)
        hero_copy.addWidget(description)
        hero_layout.addLayout(hero_copy, 1)
        launch = self._button("启动 SENTINEL  →", self._start_sentinel, primary=True)
        launch.setMinimumWidth(178)
        launch.setMinimumHeight(44)
        hero_layout.addWidget(launch)
        layout.addWidget(hero)

        deployed_count = len(list(RESULTS_DIR.glob("**/weights/best.pt")))
        candidate_count = len(list(TRAINING_OUTPUTS_DIR.glob("**/weights/best.pt")))
        gpu_value = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "未检测到 CUDA"
        metrics = QtWidgets.QGridLayout()
        metrics.setHorizontalSpacing(12)
        metrics.setVerticalSpacing(12)
        gpu_card, self.home_gpu = self._metric_card("计算设备", gpu_value, "咦？怎么不是 H100？", "success")
        model_card, _ = self._metric_card("正式模型", str(deployed_count), "也就是 RESULTS 文件夹中的模型数量", "primary")
        candidate_card, _ = self._metric_card("待审候选模型", str(candidate_count), "尚未接入 SENTINEL 的数量", "warning" if candidate_count else "neutral")
        runtime_card, _ = self._metric_card("运行模式", "离线版本", "移动工作站内嵌 Python，不依赖本机环境", "success")
        for index, card in enumerate((gpu_card, model_card, candidate_card, runtime_card)):
            metrics.addWidget(card, index // 2, index % 2)
        metrics.setColumnStretch(0, 1)
        metrics.setColumnStretch(1, 1)
        layout.addLayout(metrics)

        actions = make_card()
        action_layout = QtWidgets.QGridLayout(actions)
        action_layout.setContentsMargins(18, 16, 18, 16)
        action_layout.setSpacing(10)
        action_title = QtWidgets.QLabel("快速操作")
        action_title.setProperty("textRole", "sectionTitle")
        action_layout.addWidget(action_title, 0, 0, 1, 2)
        action_layout.addWidget(self._button("运行完整环境自检", lambda: self._run_utility("portable_check.py")), 1, 0)
        action_layout.addWidget(self._button("打开工作库文件夹", lambda: self._open_path(PROJECT_ROOT)), 1, 1)
        action_layout.addWidget(self._button("打开中文使用说明", lambda: self._open_path(PROJECT_ROOT / "START_HERE_CN.md")), 2, 0)
        action_layout.addWidget(self._button("快速检查关键文件", lambda: self._run_utility("portable/verify_workspace.py")), 2, 1)
        action_layout.addWidget(self._button("保存当前代码版本", lambda: self._launch_command_file("SAVE_VERSION.cmd"), primary=True), 3, 0)
        action_layout.addWidget(self._button("与 GitHub 最新版本同步", lambda: self._launch_command_file("SYNC_GITHUB.cmd")), 3, 1)
        action_layout.addWidget(self._button("查看 Git 历史", lambda: self._launch_command_file("GIT_STATUS.cmd")), 4, 0)
        action_layout.addWidget(self._button("Git 使用说明", lambda: self._open_path(PROJECT_ROOT / "GIT_GUIDE_CN.md")), 4, 1)
        layout.addWidget(actions)

        log_title = QtWidgets.QLabel("运行日志")
        log_title.setProperty("textRole", "sectionTitle")
        layout.addWidget(log_title)
        self.home_log = QtWidgets.QPlainTextEdit()
        self.home_log.setReadOnly(True)
        self.home_log.setPlaceholderText("环境检查结果会显示在这里。")
        self.home_log.setMinimumHeight(220)
        layout.addWidget(self.home_log)
        return page

    def _build_training_tab(self):
        page, layout = self._page()
        intro = QtWidgets.QLabel(
            "训练结果只存入 TRAINING_OUTPUTS 候选区。训练开始前会强制检查数据集，完成后仍需在“审核”页人工批准。"
        )
        intro.setWordWrap(True)
        intro.setObjectName("Hint")
        layout.addWidget(intro)

        form_group = QtWidgets.QGroupBox("训练设置")
        form = QtWidgets.QGridLayout(form_group)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        form.setColumnStretch(1, 1)
        self.dataset_combo = QtWidgets.QComboBox()
        self.dataset_combo.setEditable(True)
        self.dataset_browse = self._button("浏览…", self._browse_dataset)
        self.weights_preset = QtWidgets.QComboBox()
        self.weights_preset.setToolTip("仅列出与当前目标检测训练流程兼容的官方 YOLOv5 v7.0 权重")
        self.weights_edit = QtWidgets.QLineEdit()
        self.weights_edit.setReadOnly(True)
        self.weights_edit.setPlaceholderText("自动查找失败，请手动浏览可信的自定义 .pt")
        self.weights_browse = self._button("浏览…", self._browse_training_weights)
        self.experiment_name = QtWidgets.QLineEdit("new_target")
        self.epochs = QtWidgets.QSpinBox(); self.epochs.setRange(1, 10000); self.epochs.setValue(100)
        self.batch_size = QtWidgets.QSpinBox(); self.batch_size.setRange(-1, 1024); self.batch_size.setValue(-1)
        self.image_size = QtWidgets.QSpinBox(); self.image_size.setRange(160, 2048); self.image_size.setSingleStep(32); self.image_size.setValue(640)
        self.workers = QtWidgets.QSpinBox(); self.workers.setRange(0, 32); self.workers.setValue(0)
        rows = [
            ("数据集 YAML", self.dataset_combo, self.dataset_browse),
            ("预训练型号", self.weights_preset, None),
            ("实际权重文件", self.weights_edit, self.weights_browse),
            ("训练名称", self.experiment_name, None),
            ("训练轮数 epochs", self.epochs, None),
            ("批量 batch（默认 -1）", self.batch_size, None),
            ("图像尺寸", self.image_size, None),
            ("数据线程（Windows 建议 0）", self.workers, None),
        ]
        for row, (label, editor, extra) in enumerate(rows):
            form.addWidget(QtWidgets.QLabel(label), row, 0)
            form.addWidget(editor, row, 1)
            if extra is not None:
                form.addWidget(extra, row, 2)
        layout.addWidget(form_group)

        weight_card = make_card(elevated=True)
        weight_layout = QtWidgets.QHBoxLayout(weight_card)
        weight_layout.setContentsMargins(16, 13, 16, 13)
        self.weight_family_badge = make_badge("P5 · 640px", "primary")
        self.weight_advice = QtWidgets.QLabel()
        self.weight_advice.setProperty("textRole", "muted")
        self.weight_advice.setWordWrap(True)
        weight_layout.addWidget(self.weight_family_badge)
        weight_layout.addWidget(self.weight_advice, 1)
        layout.addWidget(weight_card)

        self._populate_weight_presets()
        self.weights_preset.currentIndexChanged.connect(self._on_weight_preset_changed)
        default_index = next((i for i, preset in enumerate(PRESETS) if preset.key == "s"), 0)
        self.weights_preset.setCurrentIndex(default_index)
        self._on_weight_preset_changed(default_index)

        buttons = QtWidgets.QHBoxLayout()
        self.btn_check_dataset = self._button("检查数据集", self._check_selected_dataset)
        self.btn_start_training = self._button("开始训练候选模型", self._start_training, primary=True)
        self.btn_resume_training = self._button("从 last.pt 继续", self._resume_training)
        self.btn_stop_training = self._button("停止训练", self._stop_training)
        self.btn_stop_training.setProperty("variant", "danger")
        self.btn_stop_training.setEnabled(False)
        buttons.addWidget(self.btn_check_dataset)
        buttons.addWidget(self.btn_start_training)
        buttons.addWidget(self.btn_resume_training)
        buttons.addWidget(self.btn_stop_training)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.training_log = QtWidgets.QPlainTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setPlaceholderText("训练日志会实时显示在这里。关闭工作台前请先确认训练已结束。")
        self.training_log.setMinimumHeight(260)
        layout.addWidget(self.training_log)
        return page

    def _build_deployment_tab(self):
        page, layout = self._page()
        warning = QtWidgets.QLabel(
            "注意：人工完成指标、类别、失败样本和真实视频检查并点击最终批准后模型才会被 SENTINEL 使用。"
        )
        warning.setObjectName("Warning")
        warning.setWordWrap(True)
        layout.addWidget(warning)

        candidate_group = QtWidgets.QGroupBox("1. 选择需要检查的候选权重")
        candidate_layout = QtWidgets.QGridLayout(candidate_group)
        self.candidate_combo = QtWidgets.QComboBox(); self.candidate_combo.setEditable(True)
        candidate_layout.addWidget(QtWidgets.QLabel("候选 best.pt"), 0, 0)
        candidate_layout.addWidget(self.candidate_combo, 0, 1)
        candidate_layout.addWidget(self._button("浏览…", self._browse_candidate), 0, 2)
        candidate_layout.addWidget(self._button("读取模型详情", self._inspect_candidate, primary=True), 0, 3)
        self.candidate_summary = QtWidgets.QLabel("尚未检查候选模型")
        self.candidate_summary.setObjectName("Hint")
        self.candidate_summary.setWordWrap(True)
        candidate_layout.addWidget(self.candidate_summary, 1, 0, 1, 4)
        candidate_layout.addWidget(self._button("查看候选模型目录", self._open_candidate_run), 2, 0, 1, 2)
        candidate_layout.addWidget(self._button("选择图片或视频进行测试", self._test_candidate_media), 2, 2)
        candidate_layout.addWidget(self._button("查看测试结果", lambda: self._open_path(TRAINING_OUTPUTS_DIR / "_REVIEWS")), 2, 3)
        layout.addWidget(candidate_group)

        identity_group = QtWidgets.QGroupBox("2. 模型与类别定义")
        identity_layout = QtWidgets.QGridLayout(identity_group)
        self.deploy_model_id = QtWidgets.QLineEdit()
        self.deploy_display_name = QtWidgets.QLineEdit()
        identity_layout.addWidget(QtWidgets.QLabel("模型 ID（英文/数字）"), 0, 0)
        identity_layout.addWidget(self.deploy_model_id, 0, 1)
        identity_layout.addWidget(QtWidgets.QLabel("界面显示名"), 0, 2)
        identity_layout.addWidget(self.deploy_display_name, 0, 3)
        self.class_table = QtWidgets.QTableWidget(0, 7)
        self.class_table.setHorizontalHeaderLabels([
            "权重类别", "统一类别 ID", "中文名", "英文名", "告警等级", "触发告警", "颜色",
        ])
        self.class_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.class_table.setMinimumHeight(180)
        identity_layout.addWidget(self.class_table, 1, 0, 1, 4)
        layout.addWidget(identity_group)

        review_group = QtWidgets.QGroupBox("3. 人工审核清单")
        review_layout = QtWidgets.QGridLayout(review_group)
        self.review_checks = {
            "metrics_reviewed": QtWidgets.QCheckBox("已检查 Precision / Recall / mAP / 曲线"),
            "class_order_reviewed": QtWidgets.QCheckBox("已核对类别名称和编号顺序"),
            "failure_samples_reviewed": QtWidgets.QCheckBox("已检查漏报、误报和失败样本"),
            "real_media_reviewed": QtWidgets.QCheckBox("已用真实图片或视频人工测试"),
        }
        for index, checkbox in enumerate(self.review_checks.values()):
            review_layout.addWidget(checkbox, index // 2, index % 2)
        self.reviewer = QtWidgets.QLineEdit("user")
        self.review_notes = QtWidgets.QLineEdit()
        self.replace_existing = QtWidgets.QCheckBox("替换同 ID 的现有模型（旧版本自动归档）")
        review_layout.addWidget(QtWidgets.QLabel("审核人"), 2, 0)
        review_layout.addWidget(self.reviewer, 2, 1)
        review_layout.addWidget(QtWidgets.QLabel("审核备注"), 3, 0)
        review_layout.addWidget(self.review_notes, 3, 1)
        review_layout.addWidget(self.replace_existing, 4, 0, 1, 2)
        layout.addWidget(review_group)
        self.btn_deploy = self._button("最终批准并部署到 SENTINEL", self._deploy_candidate, primary=True)
        layout.addWidget(self.btn_deploy)
        self.candidate_combo.currentTextChanged.connect(self._candidate_selection_changed)
        return page

    def _build_models_tab(self):
        page, layout = self._page()
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._button("重新扫描可用模型", self._refresh_models, primary=True))
        row.addWidget(self._button("打开正式模型文件夹", lambda: self._open_path(RESULTS_DIR)))
        row.addWidget(self._button("打开归档文件夹", lambda: self._open_path(PROJECT_ROOT / "MODEL_ARCHIVE")))
        row.addStretch(1)
        layout.addLayout(row)
        self.models_table = QtWidgets.QTableWidget(0, 5)
        self.models_table.setHorizontalHeaderLabels(["模型 ID", "显示名", "类别", "状态", "权重路径"])
        self.models_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.models_table.setAlternatingRowColors(True)
        self.models_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.models_table.verticalHeader().setVisible(False)
        self.models_table.setMinimumHeight(420)
        layout.addWidget(self.models_table)
        self.models_hint = QtWidgets.QLabel()
        self.models_hint.setObjectName("Hint")
        layout.addWidget(self.models_hint)
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
        self.candidate_combo.clear()
        for path in candidates:
            self.candidate_combo.addItem(str(path))
        if current:
            self.candidate_combo.setCurrentText(current)

    def _refresh_models(self):
        if not hasattr(self, "models_table"):
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            snapshot = ModelCatalog().scan(verify_hashes=True, inspect_new_models=True)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
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
                self.models_table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
        self.models_hint.setText("共发现 %d 个模型，其中 %d 个可用。" % (
            len(snapshot.models), len(snapshot.selectable_models)
        ))

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
        self._run_utility("dataset_audit.py", [dataset], self.training_log)

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
        process.start(PYTHON_EXECUTABLE, arguments)

    def _stop_training(self):
        if self.training_process is None or self.training_process.state() == QtCore.QProcess.ProcessState.NotRunning:
            return
        answer = QtWidgets.QMessageBox.question(self, "停止训练", "确认停止当前训练？已写入的 checkpoint 会保留。")
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
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
        self._refresh_candidates()
        if exit_code == 0:
            QtWidgets.QMessageBox.information(self, "训练完成", "候选模型已保存在 TRAINING_OUTPUTS。\n请进入“审核与部署”页，人工检查后再部署。")

    def _start_sentinel(self):
        ok, _pid = QtCore.QProcess.startDetached(PYTHON_EXECUTABLE, [str(PROJECT_ROOT / "launcher_GPT.py")], str(PROJECT_ROOT))
        if not ok:
            QtWidgets.QMessageBox.warning(self, "启动失败", "无法启动 SENTINEL。请先运行环境自检。")

    def _candidate_selection_changed(self, _text=None):
        self.candidate_info = None
        self.candidate_summary.setText("尚未检查候选模型")
        self.deploy_model_id.clear()
        self.deploy_display_name.clear()
        self.class_table.setRowCount(0)
        self.class_editors = {}
        for checkbox in self.review_checks.values():
            checkbox.setChecked(False)
        self.review_notes.clear()
        self.replace_existing.setChecked(False)

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

    def _populate_class_table(self, classes):
        defaults = {
            "fire": ("fire", "火焰", "Fire", "critical", True, "#F43F5E"),
            "person": ("person", "未戴安全帽", "NO HELMET", "warning", True, "#38BDF8"),
            "hat": ("hat", "已戴安全帽", "Helmet", "none", False, "#EAB308"),
        }
        self.class_table.setRowCount(len(classes))
        self.class_editors = {}
        for row, raw_name in enumerate(classes):
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
            QtWidgets.QMessageBox.warning(self, "尚未技术检查", "请先点击“读取类别与指纹”。")
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
            QtWidgets.QMessageBox.warning(self, "尚未技术检查", "请先点击“读取类别与指纹”。")
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
