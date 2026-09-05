# -*- coding: utf-8 -*-
"""Sentinel Vision dynamic YOLOv5 desktop detector.

Deployment models are discovered from RESULTS on every start.  Users select
any non-conflicting combination with checkboxes; future registered classes are
rendered and filtered without changing this source file.
"""

import sys, os, time, json, threading, csv, uuid, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Runtime dependencies are installed explicitly by SETUP_ENVIRONMENT.cmd.
# Never let YOLOv5 modify a public computer's environment during application
# startup, especially when the machine is offline.
os.environ["YOLOv5_AUTOINSTALL"] = "false"
os.environ["SENTINEL_OFFLINE"] = "1"
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import cv2
import torch
from PySide6 import QtCore, QtGui, QtWidgets

import time
import threading

from collections import defaultdict, deque
import threading, time  # time 以后用 time.time()

from model_catalog import CatalogSnapshot, ClassProfile, ModelCatalog
from model_runtime import ModelRuntime
from project_paths import EVENTS_DIR, PROJECT_ROOT, SAFETY_PIPELINE_CONFIG
from safety_pipeline import (
    AlertEvidenceGate,
    JsonlEventJournal,
    PipelineClock,
    PPETemporalPipeline,
    config_fingerprint,
    filter_monitoring_domain,
    load_safety_pipeline_config_checked,
    ppe_evidence_enabled,
    stable_class_visible,
)
from safety_pipeline.types import ResolvedDetection
from ui_font import install_ui_font
from ui_theme import (
    PressableButton,
    build_stylesheet as build_app_stylesheet,
    reduce_motion_enabled,
    set_property,
    tokens as app_tokens,
)

# ---- 统一的告警ID生成器（毫秒*1000，极小概率如果同时触发两个警报，则赋予并发序号000/001/002...）----
_ALERT_LAST_MS = 0
_ALERT_SEQ = 0
_ALERT_LOCK = threading.Lock()
def _next_alert_id():
    global _ALERT_LAST_MS, _ALERT_SEQ
    now_ms = int(time.time() * 1000)
    with _ALERT_LOCK:
        if now_ms == _ALERT_LAST_MS:
            _ALERT_SEQ = (_ALERT_SEQ + 1) % 1000
        else:
            _ALERT_LAST_MS = now_ms
            _ALERT_SEQ = 0
        return now_ms * 1000 + _ALERT_SEQ

# ===================== 路径与模型 =====================
# 前端名称以及简介：
APP_NAME = "SentinelVision"
SUBTITLE = "REAL-TIME AI SAFETY OPERATIONS"


INFER_IMG_SIZE  = 640
DETECT_EVERY_N  = 1   # 当前设置为 1，减少“跳帧检测”造成的闪烁；如算力要求过高可以修改为 2 或更高，降低检测频率

# ===================== 主题（暗/亮） =====================
TOKENS_DARK = dict(
    bg="#07101D", bg2="#0C1727", card="#101E31", line="#223651",
    text="#F4F7FB", text2="#9CB0C8",
    primary="#4F8CFF", accent="#22D3EE", accentSoft="#4F8CFF",
    success="#2DD4A3", warn="#F6B94A", danger="#F05D6C", info="#4F8CFF",
)
TOKENS_LIGHT = dict(
    bg="#F3F6FA", bg2="#FFFFFF", card="#F7F9FC", line="#D7E0EB",
    text="#142033", text2="#586B82",
    primary="#2F6FED", accent="#0891B2", accentSoft="#2F6FED",
    success="#159A75", warn="#C78312", danger="#D74252", info="#2F6FED",
)
DARK = False
REDUCE_MOTION = reduce_motion_enabled()

# ===================== 使用者 / 权限 =====================
RBAC_ENABLED = True
DEFAULT_ROLE = "Admin"  # Admin / Operator / Viewer

# ===================== 动态模型/类别 =====================
MODEL_CATALOG = ModelCatalog()
CATALOG_SNAPSHOT = MODEL_CATALOG.scan(verify_hashes=True, inspect_new_models=True)
SAFETY_PIPELINE_DEFAULTS, SAFETY_PIPELINE_CONFIG_VALID = load_safety_pipeline_config_checked(
    SAFETY_PIPELINE_CONFIG
)


def class_profile(class_id: str) -> ClassProfile:
    profile = CATALOG_SNAPSHOT.class_profiles.get(class_id)
    if profile is not None:
        return profile
    return ClassProfile(class_id, class_id, class_id)


def class_display_name(class_id: str, language: str = "en") -> str:
    return class_profile(class_id).display_name(language)


def all_class_ids(snapshot: Optional[CatalogSnapshot] = None) -> Tuple[str, ...]:
    active_snapshot = snapshot or CATALOG_SNAPSHOT
    result = []
    for model in active_snapshot.models:
        for class_id in model.canonical_classes:
            if class_id not in result:
                result.append(class_id)
    return tuple(result)

def P(): return TOKENS_DARK if DARK else TOKENS_LIGHT
def qcolor(hexstr: str, alpha: int = None):
    c = QtGui.QColor(hexstr)
    if alpha is not None: c.setAlpha(alpha)
    return c

# ===================== i18n =====================
I18N = {
    "zh": {
        "model": "模型",
        "add_roi": "新增 ROI",
        "add_mask": "新增屏蔽区",
        "edit": "编辑",
        "reselect": "重选区域",
        "pick_file": "选择本地视频",
        "url_ph": "RTSP / HTTP(s) URL",
        "load": "载入",
        "webcam": "Webcam",
        "demo1": "明火CCTV DEMO 1",
        "demo2": "明火CCTV DEMO 2",
        "conf": "置信度阈值 0.05 ↔ 0.95",
        "iou": "NMS IoU 0.10 ↔ 0.90",
        "timeline": "告警时间轴（最新→最旧）",
        "clear_alerts": "清空告警",
        "export_csv": "导出 CSV",
        "export_json": "导出 JSON",
        "alert_detail": "告警详情",
        "status_switching": "正在切换视频源…",
        "status_no_perm_src": "无权限：Viewer 无法切换视频源",
        "status_no_perm_edit": "无权限：仅 Operator/Admin 可编辑区域",
        "status_applied": "已应用阈值：conf={conf:.2f}, iou={iou:.2f}",
        "status_stopped": "已停止并释放视频源",
        "status_opened": "视频源已打开",
        "status_read_fail": "读取帧失败，尝试重连…",
        "status_infer_err": "推理异常：{err}",
        "no_alert": "未选中任何告警",
    },
    "en": {
        "model": "Model",
        "add_roi": "Add ROI",
        "add_mask": "Add Mask",
        "edit": "Edit",
        "reselect": "Reset Regions",
        "pick_file": "Pick Local Video",
        "url_ph": "RTSP / HTTP(s) URL",
        "load": "Load",
        "webcam": "Webcam",
        "demo1": "Fire CCTV Demo 1",
        "demo2": "Fire CCTV Demo 2",
        "conf": "Conf Threshold 0.05 ↔ 0.95",
        "iou": "NMS IoU 0.10 ↔ 0.90",
        "timeline": "Alert Timeline (Newest → Oldest)",
        "clear_alerts": "Clear",
        "export_csv": "Export CSV",
        "export_json": "Export JSON",
        "alert_detail": "Alert Detail",
        "status_switching": "Switching source…",
        "status_no_perm_src": "No permission: Viewer cannot switch source",
        "status_no_perm_edit": "No permission: Only Operator/Admin can edit regions",
        "status_applied": "Applied: conf={conf:.2f}, iou={iou:.2f}",
        "status_stopped": "Stopped and released source",
        "status_opened": "Video source opened",
        "status_read_fail": "Read failed, reconnecting…",
        "status_infer_err": "Inference error: {err}",
        "no_alert": "No alert selected",
    }
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===================== 工具 =====================
# ===================== 统一按压反馈 =====================
class AnimatedButton(PressableButton):
    """Compatibility name backed by the shared pointer feedback primitive."""

    pass

QBtn = AnimatedButton

# ===================== 主题化下拉 =====================
class ThemedComboBox(QtWidgets.QComboBox):
    """Native combobox using the shared light/dark controls and chevron."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setEditable(False)
        self.setMinimumHeight(40)
        self.setMinimumContentsLength(8)
        self.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)

    def apply_theme(self):
        self.update()

# ===================== 动态多模型推理线程 =====================
class VideoWorker(QtCore.QThread):
    frameReady = QtCore.Signal(object, int, int)
    statusMsg = QtCore.Signal(str)
    fpsReady = QtCore.Signal(float)
    newAlert = QtCore.Signal(dict)
    modelStatus = QtCore.Signal(str, str, str)
    modelsApplied = QtCore.Signal(object, object)
    pipelineStatsReady = QtCore.Signal(dict)
    pipelineTransition = QtCore.Signal(dict)

    def __init__(self, snapshot: CatalogSnapshot, safety_config=SAFETY_PIPELINE_DEFAULTS):
        super().__init__()
        self._lock = threading.Lock()
        self._source = None
        self._want_open = False
        self._stop = False
        self._paused = False
        self.cap = None
        self.frame_count = 0
        self.conf = 0.25
        self.iou = 0.45
        self.rois = []
        self.masks = []
        self.current_source_id = None
        self.current_source_kind = None
        self.source_frame_index = 0
        self.pipeline_clock = PipelineClock()
        self._fps_t0 = time.time()
        self._fps_counter = 0

        self.catalog_snapshot = snapshot
        self._catalog_generation = 0
        self._applied_catalog_generation = -1
        self.selected_model_ids: Tuple[str, ...] = ()
        self._selection_generation = 0
        self._applied_selection_generation = -1
        self.active_model_ids: Tuple[str, ...] = ()
        self.class_enabled = {
            class_id: profile.enabled_by_default
            for class_id, profile in snapshot.class_profiles.items()
        }

        self.runtime: Optional[ModelRuntime] = None
        self.tracks = []
        self.track_next_id = 1
        self.min_hits = 2
        self.max_miss = 6
        self.iou_match_thr = self.iou
        self.smooth_alpha = 0.7
        self.alert_win_n = 30
        self.alert_min_hits = 15
        self.frame_hits = defaultdict(lambda: deque(maxlen=self.alert_win_n))
        self.last_alert_at = {}
        self.legacy_admission_dropped = 0
        self.legacy_invalid_detections = 0
        self.safety_config = safety_config
        self.safety_mode = safety_config.default_mode
        self.safety_config_fingerprint = config_fingerprint(safety_config.with_mode(self.safety_mode))
        self.safety_debug_overlay = False
        self._pipeline_reset_requested = False
        self._full_pipeline_reset_requested = False
        self.ppe_pipeline = PPETemporalPipeline(safety_config)
        self.alert_evidence_gate = AlertEvidenceGate(safety_config.alert_validation)
        self.generic_alert_session_id = uuid.uuid4().hex[:12]
        self._last_pipeline_stats_at = 0.0
        self.event_journal = JsonlEventJournal(EVENTS_DIR / "ppe_events.jsonl")

    # ---------- thread-safe configuration ----------
    def configure_thresholds(self, conf, iou):
        conf = float(max(0.0, min(1.0, conf)))
        iou = float(max(0.0, min(1.0, iou)))
        with self._lock:
            changed = abs(conf - self.conf) > 1e-12 or abs(iou - self.iou) > 1e-12
            self.conf = conf
            self.iou = iou
            self.iou_match_thr = iou
            if changed and self.safety_mode == "ppe_temporal":
                # Confidence changes alter the high/low evidence boundary and
                # IoU changes alter canonical detections.  Never mix evidence
                # collected under two threshold policies in one episode.
                self._full_pipeline_reset_requested = True

    def configure_classes(self, values):
        with self._lock:
            updated = dict(values)
            if updated != self.class_enabled:
                old_values = self.class_enabled
                self.class_enabled = updated
                ppe_ids = {
                    self.safety_config.conflict.protected_class_id,
                    self.safety_config.conflict.unprotected_class_id,
                }
                if self.safety_mode == "ppe_temporal":
                    changed_ids = {
                        class_id
                        for class_id in set(old_values) | set(updated)
                        if bool(old_values.get(class_id, True))
                        != bool(updated.get(class_id, True))
                    }
                    if changed_ids - ppe_ids:
                        # A generic alert policy changed.  Reset its evidence
                        # rather than allowing a disable/enable cycle to reuse
                        # a pre-change candidate or cooldown namespace.
                        self._full_pipeline_reset_requested = True
                    elif changed_ids & ppe_ids:
                        self._pipeline_reset_requested = True

    def configure_safety_pipeline(self, mode, debug_overlay=False):
        if mode not in ("baseline", "ppe_temporal"):
            mode = "baseline"
        with self._lock:
            debug_overlay = bool(debug_overlay)
            if mode != self.safety_mode:
                self.safety_mode = mode
                self.safety_config_fingerprint = config_fingerprint(self.safety_config.with_mode(mode))
                self._full_pipeline_reset_requested = True
            self.safety_debug_overlay = debug_overlay

    def configure_polys(self, rois, masks):
        with self._lock:
            changed = rois != self.rois or masks != self.masks
            self.rois = rois
            self.masks = masks
            if changed and self.safety_mode == "ppe_temporal":
                # ROI/Mask edits redefine the monitored domain.  Evidence
                # collected under the previous geometry must not survive.
                self._full_pipeline_reset_requested = True

    def configure_catalog(self, snapshot: CatalogSnapshot):
        with self._lock:
            self.catalog_snapshot = snapshot
            self._catalog_generation += 1

    def configure_models(self, model_ids):
        with self._lock:
            requested = tuple(dict.fromkeys(str(model_id) for model_id in model_ids))
            if requested != self.selected_model_ids:
                self.selected_model_ids = requested
                self._selection_generation += 1

    def record_alert(self, alert):
        fields = (
            "id", "event_id", "session_id", "track_id", "track_id_namespace",
            "ts", "source_id", "cls", "conf", "severity",
            "model_ids", "risk_type", "stable_state", "risk_state", "confirmation_delay",
            "evidence_hits", "evidence_duration", "evidence_presence_ratio",
            "evidence_ema", "evidence_stability", "decision_reason",
            "pipeline_mode", "config_fingerprint",
        )
        payload = {key: alert.get(key) for key in fields if key in alert}
        self.event_journal.append("alert", payload)

    def record_review(
        self,
        alert_id,
        conclusion,
        track_id=None,
        event_id=None,
        session_id=None,
        track_id_namespace=None,
    ):
        self.event_journal.append("human_review", {
            "alert_id": str(alert_id),
            "event_id": event_id,
            "session_id": session_id,
            "track_id": track_id,
            "track_id_namespace": track_id_namespace,
            "conclusion": str(conclusion),
        })

    def set_source(self, source_obj, source_id, source_kind=None):
        with self._lock:
            self._source = source_obj
            self.current_source_id = source_id
            self.current_source_kind = source_kind
            self._want_open = True

    def stop_source(self):
        with self._lock:
            self._source = None
            self.current_source_id = None
            self.current_source_kind = None
            self._want_open = True

    def set_paused(self, flag: bool):
        with self._lock:
            self._paused = flag

    def _open_cap(self, source):
        if isinstance(source, int):
            return cv2.VideoCapture(source, cv2.CAP_DSHOW)
        return cv2.VideoCapture(source)

    def _close_cap(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _reset_temporal_state(self):
        self.tracks = []
        self.track_next_id = 1
        self.frame_hits.clear()
        self.last_alert_at.clear()
        self.legacy_admission_dropped = 0
        self.legacy_invalid_detections = 0
        self.ppe_pipeline.reset()
        self.alert_evidence_gate.reset()
        self.generic_alert_session_id = uuid.uuid4().hex[:12]
        self._last_pipeline_stats_at = 0.0

    def _reset_ppe_state(self):
        """Reset PPE reasoning without erasing the legacy Fire cooldown."""

        ppe_class_ids = {
            self.safety_config.conflict.protected_class_id,
            self.safety_config.conflict.unprotected_class_id,
            self.safety_config.conflict.uncertain_class_id,
        }
        self.tracks = [track for track in self.tracks if track["cls"] not in ppe_class_ids]
        for state_map in (self.frame_hits, self.last_alert_at):
            for key in tuple(state_map):
                if len(key) >= 2 and key[1] in ppe_class_ids:
                    state_map.pop(key, None)
        self.ppe_pipeline.reset()
        self._last_pipeline_stats_at = 0.0

    @staticmethod
    def _box_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return intersection / max(1e-6, area_a + area_b - intersection)

    @staticmethod
    def _smooth(old_box, new_box, amount=0.7):
        return tuple(amount * new + (1.0 - amount) * old for new, old in zip(new_box, old_box))

    def _update_tracks(
        self,
        detections,
        hardened=False,
        high_confidence_threshold=None,
    ):
        for track in self.tracks:
            track["updated"] = False
            track["evidence_eligible"] = False
            track["miss"] += 1

        valid_detections = detections
        if hardened:
            valid_detections = []
            for detection in detections:
                try:
                    confidence = float(detection["conf"])
                    box = tuple(float(value) for value in detection["box"])
                except (KeyError, TypeError, ValueError):
                    self.legacy_invalid_detections += 1
                    continue
                if (
                    len(box) != 4
                    or not math.isfinite(confidence)
                    or confidence < 0.0
                    or confidence > 1.0
                    or not all(math.isfinite(value) for value in box)
                    or box[2] <= box[0]
                    or box[3] <= box[1]
                ):
                    self.legacy_invalid_detections += 1
                    continue
                valid_detections.append(detection)
        ordered_detections = (
            sorted(
                valid_detections,
                key=lambda item: (
                    -float(item.get("conf", 0.0)),
                    tuple(float(value) for value in item.get("box", (0, 0, 0, 0))),
                    str(item.get("cls", "")),
                ),
            )
            if hardened else detections
        )
        matched_track_indexes = set()
        association_floor = self.safety_config.tracking.association_min_confidence
        creation_floor = self.safety_config.tracking.new_candidate_min_confidence
        if hardened and high_confidence_threshold is not None:
            creation_floor = max(creation_floor, float(high_confidence_threshold))
        for detection in ordered_detections:
            confidence = float(detection["conf"])
            high_confidence = not hardened or confidence + 1e-12 >= creation_floor
            if hardened and confidence + 1e-12 < association_floor:
                continue
            best_iou = 0.0
            best_index = -1
            for index, track in enumerate(self.tracks):
                if hardened and index in matched_track_indexes:
                    continue
                if track["cls"] != detection["cls"]:
                    continue
                if hardened and not high_confidence and track["hits"] < self.min_hits:
                    continue
                overlap = self._box_iou(track["box"], detection["box"])
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = index
            if best_index >= 0 and best_iou >= self.iou_match_thr:
                track = self.tracks[best_index]
                if hardened:
                    matched_track_indexes.add(best_index)
                track["box"] = self._smooth(track["box"], detection["box"], self.smooth_alpha)
                if hardened:
                    alpha = self.safety_config.alert_validation.ema_alpha
                    track["conf"] = alpha * confidence + (1.0 - alpha) * track["conf"]
                    if high_confidence:
                        track["hits"] += 1
                else:
                    track["conf"] = max(track["conf"], detection["conf"])
                    track["hits"] += 1
                track["miss"] = 0
                track["updated"] = True
                track["actual_conf"] = confidence
                track["evidence_eligible"] = bool(high_confidence)
                track["model_ids"] = tuple(dict.fromkeys(track["model_ids"] + detection["model_ids"]))
            else:
                if hardened and not high_confidence:
                    continue
                if hardened:
                    tentative_count = sum(
                        track["hits"] < self.min_hits for track in self.tracks
                    )
                    capacity = self.safety_config.tracking.max_tentative_candidates
                    if capacity > 0 and tentative_count >= capacity:
                        self.legacy_admission_dropped += 1
                        continue
                self.tracks.append({
                    "id": self.track_next_id,
                    "cls": detection["cls"],
                    "box": detection["box"],
                    "conf": detection["conf"],
                    "model_ids": detection["model_ids"],
                    "hits": 1,
                    "miss": 0,
                    "updated": True,
                    "actual_conf": confidence,
                    "evidence_eligible": bool(high_confidence),
                })
                self.track_next_id += 1
        self.tracks = [track for track in self.tracks if track["miss"] <= self.max_miss]

    def _inference_confidence_floor(self, requested_confidence, safety_mode):
        requested = float(max(0.0, min(1.0, requested_confidence)))
        if self._production_temporal_enabled(safety_mode):
            # Keep low-confidence observations that are useful only for
            # association.  The UI threshold remains the high-confidence
            # creation/evidence boundary and is enforced downstream.
            return min(
                requested,
                float(self.safety_config.tracking.association_min_confidence),
            )
        return requested

    def _production_temporal_enabled(self, safety_mode):
        return bool(
            safety_mode == "ppe_temporal"
            and self.safety_config.schema_version >= 3
        )

    def _apply_pending_models(self):
        with self._lock:
            snapshot = self.catalog_snapshot
            catalog_generation = self._catalog_generation
            selected_ids = self.selected_model_ids
            selection_generation = self._selection_generation

        if self.runtime is None:
            self.runtime = ModelRuntime(snapshot, device=device)
        if catalog_generation != self._applied_catalog_generation:
            self.runtime.update_catalog(snapshot)
            self._applied_catalog_generation = catalog_generation
            self._applied_selection_generation = -1
        if selection_generation == self._applied_selection_generation:
            return

        result = self.runtime.apply_selection(
            selected_ids,
            status_callback=lambda model_id, state, message: self.modelStatus.emit(model_id, state, message),
        )
        self.active_model_ids = result.active_ids
        self._applied_selection_generation = selection_generation
        self._reset_temporal_state()
        self.modelsApplied.emit(list(result.active_ids), dict(result.failures))

    def _watermark(self, overlay):
        if self.active_model_ids:
            label = "MODELS: " + ", ".join(self.active_model_ids)
        else:
            label = "NO DETECTION MODEL SELECTED"
        if len(label) > 72:
            label = "MODELS ACTIVE: %d" % len(self.active_model_ids)
        cv2.putText(
            overlay,
            label,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def run(self):
        while not self._stop:
            try:
                self._apply_pending_models()
            except Exception as exc:
                self.statusMsg.emit("模型切换失败：%s" % exc)

            with self._lock:
                want_open = self._want_open
                source = self._source
                paused = self._paused
                conf = self.conf
                iou = self.iou
                rois = self.rois
                masks = self.masks
                class_enabled = dict(self.class_enabled)
                snapshot = self.catalog_snapshot
                source_id = self.current_source_id or "unknown"
                source_kind = self.current_source_kind
                safety_mode = self.safety_mode
                safety_debug_overlay = self.safety_debug_overlay
                safety_config_fingerprint = self.safety_config_fingerprint
                pipeline_reset_requested = self._pipeline_reset_requested
                self._pipeline_reset_requested = False
                full_pipeline_reset_requested = self._full_pipeline_reset_requested
                self._full_pipeline_reset_requested = False
            if full_pipeline_reset_requested:
                self._reset_temporal_state()
            if pipeline_reset_requested:
                self._reset_ppe_state()
                self.pipelineStatsReady.emit({
                    "mode": safety_mode,
                    "frames_processed": 0,
                    "conflicts_resolved": 0,
                    "active_tracks": 0,
                    "confirmed_events": 0,
                    "alerts_emitted": 0,
                })
            if want_open:
                self._close_cap()
                with self._lock:
                    self._want_open = False
                self._reset_temporal_state()
                self.source_frame_index = 0
                self.pipeline_clock.reset()
                if source is None:
                    self.statusMsg.emit("已停止并释放视频源")
                else:
                    self.cap = self._open_cap(source)
                    if not self.cap or not self.cap.isOpened():
                        self.statusMsg.emit("无法打开视频源")
                        self._close_cap()
                    else:
                        self.statusMsg.emit("视频源已打开")

            if self.cap is None:
                time.sleep(0.05)
                continue
            if paused:
                time.sleep(0.03)
                continue

            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.statusMsg.emit("读取帧失败，正在尝试重新连接")
                self._close_cap()
                time.sleep(0.3)
                with self._lock:
                    self._want_open = True
                continue

            captured_at_monotonic = time.monotonic()
            pipeline_timestamp = captured_at_monotonic
            if source_kind == "file":
                position_ms = float(self.cap.get(cv2.CAP_PROP_POS_MSEC))
                source_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
                pipeline_timestamp = self.pipeline_clock.next_timestamp(
                    source_kind,
                    captured_at_monotonic,
                    pts_ms=position_ms,
                    fps=source_fps,
                    frame_index=max(0, self.source_frame_index),
                )

            self._fps_counter += 1
            self.source_frame_index += 1
            now_time = time.time()
            if now_time - self._fps_t0 >= 1.0:
                self.fpsReady.emit(self._fps_counter / (now_time - self._fps_t0))
                self._fps_t0 = now_time
                self._fps_counter = 0

            overlay = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pending_alerts = []
            self.frame_count += 1
            self._watermark(overlay)
            do_infer = self.frame_count % max(1, DETECT_EVERY_N) == 0

            if do_infer and self.runtime is not None and self.active_model_ids:
                try:
                    inference_confidence = self._inference_confidence_floor(
                        conf,
                        safety_mode,
                    )
                    raw_detections = self.runtime.infer(
                        overlay,
                        inference_confidence,
                        iou,
                        INFER_IMG_SIZE,
                    )
                    inference_failures = dict(self.runtime.inference_failures)
                    if inference_failures:
                        for model_id, message in inference_failures.items():
                            self.modelStatus.emit(model_id, "failed", "推理失败：%s" % message)
                        self.active_model_ids = tuple(
                            model_id for model_id in self.active_model_ids
                            if model_id in self.runtime.loaded
                        )
                        self._reset_temporal_state()
                        self.modelsApplied.emit(list(self.active_model_ids), inference_failures)
                    spatial_detections = filter_monitoring_domain(raw_detections, rois, masks)
                    if safety_mode == "ppe_temporal":
                        resolution = self.ppe_pipeline.resolve(spatial_detections)
                        pipeline_inputs = resolution.detections
                    else:
                        resolution = None
                        pipeline_inputs = spatial_detections

                    legacy_detections = []
                    ppe_detections = []
                    uncertain_id = self.safety_config.conflict.uncertain_class_id
                    protected_id = self.safety_config.conflict.protected_class_id
                    unprotected_id = self.safety_config.conflict.unprotected_class_id
                    collect_ppe_evidence = ppe_evidence_enabled(
                        class_enabled,
                        protected_id,
                        unprotected_id,
                    )
                    for detection in pipeline_inputs:
                        class_id = detection.class_id
                        if safety_mode == "ppe_temporal" and isinstance(detection, ResolvedDetection):
                            if collect_ppe_evidence:
                                ppe_detections.append(detection)
                        else:
                            if not class_enabled.get(class_id, True):
                                continue
                            legacy_detections.append({
                                "cls": class_id,
                                "conf": detection.confidence,
                                "box": detection.box,
                                "model_ids": detection.source_model_ids,
                            })
                    production_temporal = self._production_temporal_enabled(
                        safety_mode
                    )
                    generic_gate_enabled = bool(
                        production_temporal
                        and self.safety_config.alert_validation.enabled
                    )
                    self._update_tracks(
                        legacy_detections,
                        hardened=production_temporal,
                        high_confidence_threshold=conf,
                    )

                    pipeline_frame = None
                    if safety_mode == "ppe_temporal" and resolution is not None:
                        filtered_resolution = self.ppe_pipeline.summarize_filtered(ppe_detections)
                        ppe_alert_profile = snapshot.class_profiles.get(
                            unprotected_id,
                            class_profile(unprotected_id),
                        )
                        ppe_alerts_enabled = bool(
                            ppe_alert_profile.alert_enabled
                            and ppe_alert_profile.severity
                            and class_enabled.get(unprotected_id, True)
                        )
                        pipeline_frame = self.ppe_pipeline.process(
                            ppe_detections,
                            filtered_resolution,
                            pipeline_timestamp,
                            alerts_enabled=ppe_alerts_enabled,
                            high_confidence_threshold=conf,
                        )
                        for transition in pipeline_frame.transitions:
                            transition_record = {
                                "session_id": pipeline_frame.metrics.session_id,
                                "source_id": source_id,
                                "pipeline_mode": safety_mode,
                                "config_fingerprint": safety_config_fingerprint,
                                "track_id": transition.track_id,
                                "track_id_namespace": "session_public",
                                "risk_type": transition.risk_type,
                                "from_state": transition.from_state.value,
                                "to_state": transition.to_state.value,
                                "at": transition.at,
                                "reason": transition.reason,
                                "event_id": transition.event_id,
                            }
                            self.pipelineTransition.emit(transition_record)
                            self.event_journal.append("risk_transition", transition_record)
                        if now_time - self._last_pipeline_stats_at >= 1.0:
                            stats_payload = dict(pipeline_frame.metrics.as_dict())
                            gate_metrics = self.alert_evidence_gate.metrics.as_dict()
                            stats_payload.update({
                                "alert_candidates": gate_metrics["active_events"],
                                "alert_transients_filtered": gate_metrics["suppressed"],
                                "alert_gate_dropped": gate_metrics["dropped"],
                                "legacy_admission_dropped": self.legacy_admission_dropped,
                                "legacy_invalid_detections": self.legacy_invalid_detections,
                            })
                            self.pipelineStatsReady.emit(stats_payload)
                            self._last_pipeline_stats_at = now_time

                    if generic_gate_enabled:
                        for track in self.tracks:
                            profile = snapshot.class_profiles.get(
                                track["cls"], class_profile(track["cls"])
                            )
                            if (
                                not profile.alert_enabled
                                or not profile.severity
                                or not class_enabled.get(track["cls"], True)
                            ):
                                continue
                            event_key = (source_id, track["cls"], int(track["id"]))
                            if track["updated"]:
                                decision = self.alert_evidence_gate.observe(
                                    event_key,
                                    pipeline_timestamp,
                                    float(track.get("actual_conf", track["conf"])),
                                    confirmed=track["hits"] >= self.min_hits,
                                    high_confidence=bool(track.get("evidence_eligible", False)),
                                )
                            else:
                                decision = self.alert_evidence_gate.missing(
                                    event_key,
                                    pipeline_timestamp,
                                )
                            if not decision.emitted:
                                continue
                            alert_id = str(_next_alert_id())
                            event_id = "%s-class-%s-%04d-%s" % (
                                self.generic_alert_session_id,
                                track["cls"],
                                int(track["id"]),
                                alert_id,
                            )
                            pending_alerts.append({
                                "id": alert_id,
                                "event_id": event_id,
                                "session_id": self.generic_alert_session_id,
                                "track_id": int(track["id"]),
                                "track_id_namespace": "session_class_track",
                                "ts": time.time(),
                                "source_id": source_id,
                                "cls": track["cls"],
                                "conf": float(decision.ema),
                                "severity": profile.severity,
                                "snapshot": None,
                                "model_ids": list(track["model_ids"]),
                                "risk_type": track["cls"],
                                "stable_state": "CONFIRMED",
                                "risk_state": "ALARMED",
                                "confirmation_delay": float(decision.duration),
                                "evidence_hits": int(decision.actual_hits),
                                "evidence_duration": float(decision.duration),
                                "evidence_presence_ratio": float(decision.presence_ratio),
                                "evidence_ema": float(decision.ema),
                                "decision_reason": decision.reason,
                                "pipeline_mode": "production_temporal_class_gate",
                                "config_fingerprint": safety_config_fingerprint,
                            })
                        self.alert_evidence_gate.cleanup(pipeline_timestamp)

                    present_now = set()
                    present_conf = {}
                    present_models: Dict[Tuple[str, str], Tuple[str, ...]] = {}
                    render_tracks = list(self.tracks)
                    if pipeline_frame is not None:
                        for detection in pipeline_frame.detections:
                            if not stable_class_visible(
                                detection.class_id,
                                class_enabled,
                                protected_id,
                                unprotected_id,
                                uncertain_id,
                            ):
                                continue
                            render_tracks.append({
                                "id": detection.track_id,
                                "cls": detection.class_id,
                                "conf": detection.confidence,
                                "box": detection.box,
                                "model_ids": detection.source_model_ids,
                                "hits": self.safety_config.tracking.min_hits,
                                "miss": 0,
                                "updated": True,
                                "ppe_result": detection,
                            })
                        for alert in pipeline_frame.alerts:
                            profile = snapshot.class_profiles.get(
                                alert.class_id,
                                class_profile(alert.class_id),
                            )
                            if not profile.alert_enabled or not profile.severity:
                                continue
                            pending_alerts.append({
                                "id": alert.event_id,
                                "ts": time.time(),
                                "source_id": source_id,
                                "cls": alert.class_id,
                                "conf": alert.confidence,
                                "severity": profile.severity,
                                "snapshot": None,
                                "model_ids": list(alert.source_model_ids),
                                "session_id": pipeline_frame.metrics.session_id,
                                "track_id": alert.track_id,
                                "track_id_namespace": "session_public",
                                "event_id": alert.event_id,
                                "risk_type": alert.risk_type,
                                "stable_state": alert.stable_state.value,
                                "risk_state": alert.risk_state.value,
                                "confirmation_delay": max(0.0, alert.alarmed_at - alert.first_seen),
                                "evidence_hits": alert.evidence_hits,
                                "evidence_duration": alert.evidence_duration,
                                "evidence_stability": alert.evidence_stability,
                                "evidence_ema": alert.confidence,
                                "decision_reason": alert.decision_reason,
                                "pipeline_mode": safety_mode,
                                "config_fingerprint": safety_config_fingerprint,
                            })

                    for track in render_tracks:
                        ppe_result = track.get("ppe_result")
                        if (
                            production_temporal
                            and ppe_result is None
                            and (not track["updated"] or track["hits"] < self.min_hits)
                        ):
                            continue
                        if not production_temporal and track["hits"] < self.min_hits and track["miss"] == 0:
                            continue
                        x1, y1, x2, y2 = map(int, track["box"])
                        profile = snapshot.class_profiles.get(track["cls"], class_profile(track["cls"]))
                        color = profile.color_rgb
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                        if ppe_result is None:
                            label = "%s %.2f" % (profile.display_en, track["conf"])
                        else:
                            label = "#%s %s %.2f" % (track["id"], profile.display_en, track["conf"])
                            if safety_debug_overlay:
                                label += " | raw:%s stable:%s risk:%s" % (
                                    ppe_result.raw_state.value,
                                    ppe_result.stable_state.value,
                                    ppe_result.risk_state.value,
                                )
                        (text_width, text_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                        )
                        top = max(0, y1 - text_height - 6)
                        cv2.rectangle(overlay, (x1, top), (x1 + text_width + 6, y1), color, -1)
                        cv2.putText(
                            overlay,
                            label,
                            (x1 + 3, max(text_height, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        if (
                            profile.alert_enabled
                            and profile.severity
                            and not (generic_gate_enabled and ppe_result is None)
                            and not (
                                production_temporal
                                and ppe_result is None
                                and not track.get("evidence_eligible", False)
                            )
                        ):
                            key = (source_id, track["cls"])
                            present_now.add(key)
                            if track["conf"] > present_conf.get(key, 0.0):
                                present_conf[key] = float(track["conf"])
                                present_models[key] = tuple(track["model_ids"])

                    now_seconds = time.time()
                    for class_id, profile in snapshot.class_profiles.items():
                        if not profile.alert_enabled or not profile.severity:
                            continue
                        if safety_mode == "ppe_temporal" and class_id == unprotected_id:
                            continue
                        if generic_gate_enabled:
                            continue
                        if not class_enabled.get(class_id, True):
                            continue
                        key = (source_id, class_id)
                        window = self.frame_hits[key]
                        window.append(1 if key in present_now else 0)
                        if (
                            sum(window) >= self.alert_min_hits
                            and now_seconds - self.last_alert_at.get(key, 0.0) >= 10.0
                        ):
                            self.last_alert_at[key] = now_seconds
                            pending_alerts.append({
                                "id": str(_next_alert_id()),
                                "ts": now_seconds,
                                "source_id": source_id,
                                "cls": class_id,
                                "conf": float(present_conf.get(key, 0.0)),
                                "severity": profile.severity,
                                "snapshot": None,
                                "model_ids": list(present_models.get(key, ())),
                                "pipeline_mode": "baseline" if safety_mode == "baseline" else "legacy_class_path",
                            })
                except Exception as exc:
                    self.statusMsg.emit("推理异常：%s" % exc)

            qimage = QtGui.QImage(
                overlay.data,
                overlay.shape[1],
                overlay.shape[0],
                overlay.shape[1] * 3,
                QtGui.QImage.Format.Format_RGB888,
            ).copy()
            self.frameReady.emit(qimage, overlay.shape[1], overlay.shape[0])
            for alert in pending_alerts:
                self.newAlert.emit(alert)
            time.sleep(0.005)

        self._close_cap()
        if self.runtime is not None:
            self.runtime.unload_all()

    def request_stop(self):
        self._stop = True


# ===================== 叠加层（ROI/Mask，仅此处绘制） =====================
class VideoCanvas(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("VideoCanvas")
        self.setMinimumSize(320, 200)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self._qimg=None; self._src_w=0; self._src_h=0
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        self.mode = "select"
        self.rois: List[List[QtCore.QPointF]] = []
        self.masks: List[List[QtCore.QPointF]] = []
        self.active_list=None; self.active_index=-1
        self.dragging=(-1,-1)

    def update_frame(self, qimg, w, h):
        self._qimg=qimg; self._src_w=w; self._src_h=h; self.update()
    def clear_all_polys(self):
        self.rois.clear(); self.masks.clear()
        self.active_list=None; self.active_index=-1; self.update()
    def set_mode(self, m: str): self.mode=m
    def export_polys_int(self):
        to_int = lambda poly: [(int(p.x()), int(p.y())) for p in poly]
        return [to_int(p) for p in self.rois], [to_int(p) for p in self.masks]

    def _fit_rect(self):
        if self._src_w<=0 or self._src_h<=0:
            return QtCore.QRect(0,0,self.width(), self.height()), 1.0
        scale = min(self.width()/self._src_w, self.height()/self._src_h)
        nw, nh = int(self._src_w*scale), int(self._src_h*scale)
        ox, oy = (self.width()-nw)//2, (self.height()-nh)//2
        return QtCore.QRect(ox, oy, nw, nh), scale

    def _map_to_frame(self, x_disp, y_disp):
        rect, scale = self._fit_rect()
        x = int((x_disp - rect.left()) / scale); y = int((y_disp - rect.top()) / scale)
        x = max(0, min(self._src_w-1, x)); y = max(0, min(self._src_h-1, y))
        return x,y

    def _hit_test(self, x_disp, y_disp):
        rect, scale = self._fit_rect()
        for typ, lst in (("roi", self.rois), ("mask", self.masks)):
            for pi, poly in enumerate(lst):
                for idx, pt in enumerate(poly):
                    px = pt.x()*scale + rect.left(); py = pt.y()*scale + rect.top()
                    if (px - x_disp)**2 + (py - y_disp)**2 <= 64:
                        return (typ, pi, idx)
        return None

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if self._src_w > 0:
            x_frame, y_frame = self._map_to_frame(e.position().x(), e.position().y())
        else:
            x_frame, y_frame = int(e.position().x()), int(e.position().y())

        if self.mode in ("add-roi","add-mask"):
            poly_list = self.rois if self.mode=="add-roi" else self.masks
            if self.active_list is not poly_list or self.active_index<0:
                poly_list.append([QtCore.QPointF(x_frame, y_frame)])
                self.active_list = poly_list; self.active_index = len(poly_list)-1
            else:
                poly_list[self.active_index].append(QtCore.QPointF(x_frame, y_frame))
            self.update()
        elif self.mode=="edit":
            hit = self._hit_test(e.position().x(), e.position().y())
            if hit:
                typ, pi, idx = hit
                self.active_list = self.rois if typ=="roi" else self.masks
                self.active_index = pi
                self.dragging = (pi, idx)

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self.mode=="edit" and self.dragging!=( -1,-1 ):
            pi, idx = self.dragging
            x_frame, y_frame = self._map_to_frame(e.position().x(), e.position().y())
            if self.active_list is not None and 0<=pi<len(self.active_list) and 0<=idx<len(self.active_list[pi]):
                self.active_list[pi][idx] = QtCore.QPointF(x_frame, y_frame)
                self.update()

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        self.dragging = (-1,-1)

    def mouseDoubleClickEvent(self, e: QtGui.QMouseEvent):
        if self.mode in ("add-roi","add-mask") and self.active_list is not None and self.active_index>=0:
            if len(self.active_list[self.active_index])>=3:
                self.mode = "select"; self.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.Antialiasing)
        # The image stage stays optically neutral in both app themes so video
        # luminance and detection overlays are judged against a stable field.
        if self._qimg:
            p.fillRect(self.rect(), QtGui.QColor("#101113"))
        else:
            t = app_tokens(DARK)
            p.fillRect(self.rect(), QtGui.QColor(t["surface"]))
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor(t["surface_alt"]))
            p.drawRoundedRect(self.rect(), 16, 16)
            center = self.rect().center()
            cx, cy = center.x(), center.y() - 44
            pen = QtGui.QPen(QtGui.QColor(t["border_strong"]), 2)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            for x, y, sx, sy in ((cx-34, cy-25, 1, 1), (cx+34, cy-25, -1, 1), (cx-34, cy+25, 1, -1), (cx+34, cy+25, -1, -1)):
                path = QtGui.QPainterPath(QtCore.QPointF(x+sx*12, y))
                path.lineTo(x, y)
                path.lineTo(x, y+sy*12)
                p.drawPath(path)
            font = self.font()
            font.setPixelSize(24)
            font.setWeight(QtGui.QFont.Weight.DemiBold)
            p.setFont(font)
            p.setPen(QtGui.QColor(t["text"]))
            p.drawText(QtCore.QRect(12, cy+48, self.width()-24, 36), QtCore.Qt.AlignmentFlag.AlignCenter,
                       getattr(self, "empty_title", "准备好，就开始。"))
            font.setPixelSize(14)
            font.setWeight(QtGui.QFont.Weight.Normal)
            p.setFont(font)
            p.setPen(QtGui.QColor(t["muted"]))
            p.drawText(QtCore.QRect(24, cy+92, self.width()-48, 56), QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.TextFlag.TextWordWrap,
                       getattr(self, "empty_detail", "选择本地视频或摄像头，画面将在这里显示。"))
        rect, scale = self._fit_rect()
        if self._qimg: p.drawImage(rect, self._qimg)

        def map_pt(pt: QtCore.QPointF) -> QtCore.QPointF:
            return QtCore.QPointF(pt.x()*scale + rect.left(), pt.y()*scale + rect.top())

        def draw_poly(poly: List[QtCore.QPointF], edge_hex: str, fill_hex: str, fill_alpha: int):
            if not poly: return
            mapped = [map_pt(pt) for pt in poly]
            path = QtGui.QPainterPath(); path.moveTo(mapped[0])
            for pt in mapped[1:]: path.lineTo(pt)
            if self.mode not in ("add-roi","add-mask") and len(mapped)>=3:
                path.closeSubpath()
            p.save()
            p.setPen(QtGui.QPen(qcolor(edge_hex), 2.5))
            p.setBrush(QtGui.QBrush(qcolor(fill_hex, fill_alpha)))
            p.drawPath(path)
            p.setBrush(QtGui.QBrush(qcolor(TOKENS_DARK["accent"] if DARK else TOKENS_LIGHT["accentSoft"])))
            p.setPen(QtGui.QPen(qcolor("#FFFFFF"), 1))
            for pt in mapped: p.drawEllipse(pt, 5, 5)
            p.restore()

        for poly in self.rois:  draw_poly(poly, P()["primary"], P()["primary"], 28)
        for poly in self.masks: draw_poly(poly, P()["danger"],  P()["danger"],  22)

        p.end()

# ===================== 告警列表模型 =====================
class Alert:
    def __init__(self, **kw): self.__dict__.update(kw)

class AlertsModel(QtCore.QAbstractListModel):
    def __init__(self, language_getter=None):
        super().__init__(); self.items=[]
        self.language_getter = language_getter or (lambda: "zh")
    def rowCount(self, parent=QtCore.QModelIndex()): return len(self.items)
    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid(): return None
        a = self.items[index.row()]
        if role==QtCore.Qt.ItemDataRole.DisplayRole:
            cls_disp = class_display_name(a.cls, self.language_getter())
            model_tag = ", ".join(getattr(a, "model_ids", ()) or ()) or "—"
            track_tag = " · Track #%s" % a.track_id if getattr(a, "track_id", None) is not None else ""
            timestamp = time.strftime('%Y-%m-%d  %H:%M:%S', time.localtime(a.ts))
            return f"{cls_disp}  ·  {a.conf:.0%}{track_tag}\n{timestamp}  ·  {a.source_id}  ·  {model_tag}"
        if role==QtCore.Qt.ItemDataRole.SizeHintRole:
            return QtCore.QSize(280, 54)
        if role==QtCore.Qt.ItemDataRole.ToolTipRole:
            return "双击或单击查看告警详情"
        if role==QtCore.Qt.ItemDataRole.UserRole: return a
        return None
    def add_alert(self, a: Alert):
        self.beginInsertRows(QtCore.QModelIndex(), 0, 0); self.items.insert(0, a); self.endInsertRows()
        if len(self.items) > 500:
            last = len(self.items) - 1
            self.beginRemoveRows(QtCore.QModelIndex(), last, last)
            self.items.pop()
            self.endRemoveRows()
    def clear(self):
        self.beginResetModel(); self.items.clear(); self.endResetModel()

# ===================== 👇主窗口👇 =====================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — Detection Console")
        self.resize(1520, 940)
        self.setMinimumSize(1100, 720)

        # 状态
        self.lang_code = "zh"   # zh / en
        self.is_viewer = False
        self.active_source_id = None
        self.catalog_snapshot = CATALOG_SNAPSHOT
        self.settings_path = PROJECT_ROOT / ".runtime" / "config" / "sentinel_settings.json"
        self.saved_settings = self._load_settings()
        self.safety_config = SAFETY_PIPELINE_DEFAULTS
        saved_pipeline = self.saved_settings.get("safety_pipeline", {})
        if not isinstance(saved_pipeline, dict):
            saved_pipeline = {}
        saved_mode = str(saved_pipeline.get("mode", self.safety_config.default_mode))
        if not SAFETY_PIPELINE_CONFIG_VALID:
            saved_mode = "baseline"
        self.safety_mode = saved_mode if saved_mode in ("baseline", "ppe_temporal") else "baseline"
        self.safety_debug_overlay = bool(saved_pipeline.get("debug_overlay", False))
        self.selected_model_ids = self._initial_model_selection()
        self._closing_after_worker = False

        # 顶栏：结构性重材质，状态与身份集中在右侧。
        self.toolbar = QtWidgets.QToolBar()
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        self.toolbar.setObjectName("AppBar")
        self.toolbar.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.PreventContextMenu)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self.toolbar)
        self.brand_mark = QtWidgets.QLabel("SV")
        self.brand_mark.setObjectName("BrandMark")
        self.brand_mark.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.brand_mark.setFixedSize(42, 34)
        self.title = QtWidgets.QLabel(APP_NAME)
        self.title.setProperty("textRole", "brand")
        self.sub   = QtWidgets.QLabel(SUBTITLE)
        self.sub.setProperty("textRole", "caption")
        brand_copy = QtWidgets.QVBoxLayout()
        brand_copy.setContentsMargins(0, 0, 0, 0)
        brand_copy.setSpacing(0)
        brand_copy.addWidget(self.title)
        brand_copy.addWidget(self.sub)
        brand_block = QtWidgets.QWidget()
        brand_block.setLayout(brand_copy)

        self.system_badge = QtWidgets.QLabel("LOCAL · OFFLINE")
        self.system_badge.setProperty("badge", "primary")
        self.system_badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_fps = QtWidgets.QLabel("FPS —")
        self.lbl_fps.setProperty("badge", "neutral")
        self.lbl_fps.setMinimumWidth(76)
        self.lbl_fps.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.btn_theme = QBtn("主题")
        self.btn_theme.clicked.connect(self.toggle_theme)
        self.btn_rm = QBtn("动效")
        self.btn_rm.clicked.connect(self.toggle_reduce_motion)
        self.btn_theme.setProperty("variant", "ghost")
        self.btn_rm.setProperty("variant", "ghost")
        self.btn_theme.setToolTip("切换明暗主题")
        self.btn_rm.setToolTip("减少非必要位移动效")
        self.lang = ThemedComboBox()
        self.lang.addItems(["zh-CN", "en-NZ"])
        self.lang.setFixedWidth(94)
        self.role = ThemedComboBox()
        self.role.addItems(["Viewer", "Operator", "Admin"])
        self.role.setCurrentText(DEFAULT_ROLE)
        self.role.setFixedWidth(108)
        self.role.currentTextChanged.connect(self.apply_role)
        self.lang.currentTextChanged.connect(self.on_language_change)

        self.brand_mark.hide()
        self.toolbar.addWidget(brand_block)
        self.toolbar.addWidget(self._spacer())
        self.toolbar.addWidget(self.system_badge)
        self.toolbar.addWidget(self.lbl_fps)
        self.preferences_button = QBtn("显示与权限")
        self.preferences_button.setProperty("variant", "ghost")
        preferences_menu = QtWidgets.QMenu(self.preferences_button)
        preferences_widget = QtWidgets.QWidget()
        preferences_form = QtWidgets.QFormLayout(preferences_widget)
        preferences_form.setContentsMargins(18, 16, 18, 16)
        preferences_form.setSpacing(12)
        self.preference_language_label = QtWidgets.QLabel("界面语言")
        self.preference_role_label = QtWidgets.QLabel("操作身份")
        self.preference_language_label.setBuddy(self.lang)
        self.preference_role_label.setBuddy(self.role)
        preferences_form.addRow(self.preference_language_label, self.lang)
        preferences_form.addRow(self.preference_role_label, self.role)
        preferences_form.addRow(self.btn_theme, self.btn_rm)
        preferences_action = QtWidgets.QWidgetAction(preferences_menu)
        preferences_action.setDefaultWidget(preferences_widget)
        preferences_menu.addAction(preferences_action)
        self.preferences_button.setMenu(preferences_menu)
        self.toolbar.addWidget(self.preferences_button)

        # 中心布局：检测画布与 Inspector 两个同级区域，无整列嵌套滚动。
        central = QtWidgets.QWidget()
        central.setObjectName("AppShell")
        central_layout = QtWidgets.QVBoxLayout(central)
        central_layout.setContentsMargins(24, 24, 24, 24)
        central_layout.setSpacing(0)
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(20)

        # 主域：视频画布。
        self.viewport_card = self._card("ViewportFrame", "strong")
        L = QtWidgets.QVBoxLayout(self.viewport_card)
        L.setContentsMargins(24, 20, 24, 20)
        L.setSpacing(12)
        monitor_header = QtWidgets.QHBoxLayout()
        monitor_header.setSpacing(10)
        monitor_copy = QtWidgets.QVBoxLayout()
        monitor_copy.setContentsMargins(0, 0, 0, 0)
        monitor_copy.setSpacing(2)
        self.monitor_eyebrow = QtWidgets.QLabel("LIVE DETECTION")
        self.monitor_eyebrow.setProperty("textRole", "eyebrow")
        self.monitor_title = QtWidgets.QLabel("实时检测")
        self.monitor_title.setProperty("textRole", "sectionTitle")
        monitor_copy.addWidget(self.monitor_eyebrow)
        monitor_copy.addWidget(self.monitor_title)
        self.live_badge = QtWidgets.QLabel("●  STANDBY")
        self.live_badge.setProperty("badge", "neutral")
        self.source_state = QtWidgets.QLabel("等待选择视频源")
        self.source_state.setObjectName("SourceState")
        self.source_state.setProperty("textRole", "body")
        self.source_state.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.source_state.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        monitor_header.addLayout(monitor_copy)
        monitor_header.addStretch(1)
        monitor_header.addWidget(self.live_badge)
        monitor_header.addWidget(self.source_state)
        L.addLayout(monitor_header)

        self.canvas = VideoCanvas()
        self.canvas.setMinimumHeight(300)
        self.canvas.setToolTip("单击放置区域顶点，双击闭合；编辑模式下可拖动顶点")
        L.addWidget(self.canvas, 1)

        # 来源与 transport 合并为靠近画布的操作条。
        self.transport_bar = self._card("TransportBar", "subtle")
        transport = QtWidgets.QHBoxLayout(self.transport_bar)
        transport.setContentsMargins(12, 9, 12, 9)
        transport.setSpacing(8)
        self.transport_label = QtWidgets.QLabel("来源与播放")
        self.transport_label.setProperty("textRole", "fieldLabel")
        self.src_quick = ThemedComboBox()
        self.src_quick.setMinimumWidth(120)
        self.src_quick.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.btn_play = QBtn("播放")
        self.btn_pause = QBtn("暂停")
        self.btn_stop = QBtn("停止")
        self.btn_play.clicked.connect(lambda: self.worker.set_paused(False))
        self.btn_pause.clicked.connect(lambda: self.worker.set_paused(True))
        self.btn_stop.clicked.connect(self.stop_source)
        self.btn_play.setProperty("variant", "secondary")
        self.btn_pause.setProperty("variant", "secondary")
        self.btn_stop.setProperty("variant", "danger")
        self.btn_play.setMinimumWidth(72)
        self.btn_pause.setMinimumWidth(72)
        self.btn_stop.setMinimumWidth(72)
        self.transport_label.hide()
        transport.addWidget(self.src_quick, 1)
        transport.addWidget(self.btn_play)
        transport.addWidget(self.btn_pause)
        transport.addWidget(self.btn_stop)
        L.addWidget(self.transport_bar)

        # 区域工具独立成低权重操作条，不与 transport 争夺主操作。
        self.region_bar = self._card("RegionBar", "subtle")
        region = QtWidgets.QHBoxLayout(self.region_bar)
        region.setContentsMargins(12, 8, 12, 8)
        region.setSpacing(8)
        self.regions_label = QtWidgets.QLabel("分析区域")
        self.regions_label.setProperty("textRole", "fieldLabel")
        self.regions_hint = QtWidgets.QLabel("ROI 参与检测；屏蔽区会被忽略")
        self.regions_hint.setProperty("textRole", "caption")
        self.btn_add_roi = QBtn()
        self.btn_add_mask = QBtn()
        self.btn_edit_poly = QBtn()
        self.btn_clear_poly = QBtn()
        for button in (self.btn_add_roi, self.btn_add_mask, self.btn_edit_poly, self.btn_clear_poly):
            button.setProperty("variant", "quiet")
        self.btn_add_roi.clicked.connect(lambda: self._set_edit_mode("add-roi"))
        self.btn_add_mask.clicked.connect(lambda: self._set_edit_mode("add-mask"))
        self.btn_edit_poly.clicked.connect(lambda: self._set_edit_mode("edit"))
        self.btn_clear_poly.clicked.connect(self._clear_polys)
        region.addWidget(self.regions_label)
        region.addWidget(self.regions_hint)
        region.addStretch(1)
        region.addWidget(self.btn_add_roi)
        region.addWidget(self.btn_add_mask)
        region.addWidget(self.btn_edit_poly)
        region.addWidget(self.btn_clear_poly)
        self.region_toggle = QtWidgets.QToolButton()
        self.region_toggle.setCheckable(True)
        self.region_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.region_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.region_toggle.setProperty("variant", "ghost")
        self.region_toggle.toggled.connect(self._set_region_tools_visible)
        L.addWidget(self.region_toggle, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        L.addWidget(self.region_bar)
        self.region_bar.hide()

        # Inspector：来源、检测配置、事件复核按任务分层。
        self.inspector_panel = self._card("InspectorFrame", "strong")
        self.inspector_panel.setMinimumWidth(380)
        self.inspector_panel.setMaximumWidth(500)
        inspector = QtWidgets.QVBoxLayout(self.inspector_panel)
        inspector.setContentsMargins(14, 15, 14, 14)
        inspector.setSpacing(10)
        self.inspector_eyebrow = QtWidgets.QLabel("INSPECTOR")
        self.inspector_eyebrow.setProperty("textRole", "eyebrow")
        self.inspector_title = QtWidgets.QLabel("检测控制")
        self.inspector_title.setProperty("textRole", "sectionTitle")
        self.inspector_eyebrow.hide()
        self.inspector_title.hide()
        self.inspector_tabs = QtWidgets.QTabWidget()
        self.inspector_tabs.setObjectName("InspectorTabs")
        self.inspector_tabs.setDocumentMode(True)
        self.inspector_tabs.tabBar().setDrawBase(False)
        self.inspector_tabs.setUsesScrollButtons(False)
        inspector.addWidget(self.inspector_tabs, 1)

        # 来源页。
        self.source_tab = QtWidgets.QWidget()
        S = QtWidgets.QVBoxLayout(self.source_tab)
        S.setContentsMargins(14, 16, 14, 14)
        S.setSpacing(10)
        self.source_title = QtWidgets.QLabel("视频源")
        self.source_title.setProperty("textRole", "sectionTitle")
        self.source_helper = QtWidgets.QLabel("选择本地视频、设备或网络流；输入不会上传。")
        self.source_helper.setWordWrap(True)
        self.source_helper.setProperty("textRole", "body")
        self.btn_file = QBtn()
        self.btn_file.setProperty("variant", "primary")
        self.btn_file.clicked.connect(self.pick_file)
        self.source_url_label = QtWidgets.QLabel("网络地址")
        self.source_url_label.setProperty("textRole", "fieldLabel")
        self.txt_url = QtWidgets.QLineEdit()
        self.btn_load = QBtn()
        self.btn_load.setProperty("variant", "secondary")
        self.btn_load.clicked.connect(self.load_url)
        url_row = QtWidgets.QHBoxLayout()
        url_row.setSpacing(8)
        url_row.addWidget(self.txt_url, 1)
        url_row.addWidget(self.btn_load)
        self.source_quick_label = QtWidgets.QLabel("快速来源")
        self.source_quick_label.setProperty("textRole", "fieldLabel")
        self.btn_webcam = QBtn()
        self.btn_webcam.clicked.connect(lambda: self.set_source(0, "webcam"))
        self.btn_demo1 = QBtn()
        self.btn_demo1.clicked.connect(lambda: self.set_source("https://media.istockphoto.com/id/1272087364/video/4k-firefighters-extinguish-a-fire-in-oil-refinery-plant.mp4?s=mp4-640x640-is&k=20&c=INonYLBCKuPgbTXfs-eSe4JlhuGEAVykvm10GaVnO6E=", "http"))
        self.btn_demo2 = QBtn()
        self.btn_demo2.clicked.connect(lambda: self.set_source("https://media.istockphoto.com/id/615753036/video/factory-worker-in-blue-uniform-is-putting-his-hard-hat-and-goggles-on-while-walking.mp4?s=mp4-640x640-is&k=20&c=ghY3U2O5e4eKoMoY-Oh8pscFBGUACy2HZAKTW99XNKM=", "http"))
        self.demo_toggle = QtWidgets.QToolButton()
        self.demo_toggle.setCheckable(True)
        self.demo_toggle.setChecked(False)
        self.demo_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.demo_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.demo_toggle.setProperty("variant", "ghost")
        self.demo_toggle.toggled.connect(self._set_demo_sources_visible)
        self.demo_sources_panel = QtWidgets.QFrame()
        self.demo_sources_panel.setProperty("surface", "soft")
        demo_layout = QtWidgets.QGridLayout(self.demo_sources_panel)
        demo_layout.setContentsMargins(9, 9, 9, 9)
        demo_layout.setSpacing(8)
        demo_layout.addWidget(self.btn_demo1, 0, 0)
        demo_layout.addWidget(self.btn_demo2, 0, 1)
        self.demo_sources_panel.setVisible(False)
        S.addWidget(self.source_title)
        S.addWidget(self.source_helper)
        S.addSpacing(4)
        S.addWidget(self.btn_file)
        S.addSpacing(4)
        S.addWidget(self.source_url_label)
        S.addLayout(url_row)
        S.addSpacing(4)
        S.addWidget(self.source_quick_label)
        S.addWidget(self.btn_webcam)
        S.addWidget(self.demo_toggle)
        S.addWidget(self.demo_sources_panel)
        S.addStretch(1)

        # 检测页：模型、阈值与目标类别是一个连续配置任务。
        self.detection_tab = QtWidgets.QScrollArea()
        self.detection_tab.setWidgetResizable(True)
        self.detection_tab.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.detection_tab.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.detection_content = QtWidgets.QWidget()
        C = QtWidgets.QGridLayout(self.detection_content)
        C.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        self.detection_tab.setWidget(self.detection_content)
        C.setContentsMargins(14, 16, 14, 14)
        C.setHorizontalSpacing(8)
        C.setVerticalSpacing(9)
        self.lbl_model = QtWidgets.QLabel()
        self.lbl_model.setProperty("textRole", "sectionTitle")
        self.lbl_model_summary = QtWidgets.QLabel()
        self.lbl_model_summary.setObjectName("MutedLabel")
        self.lbl_model_summary.setWordWrap(True)
        self.btn_rescan_models = QBtn("重新扫描 RESULTS")
        self.btn_rescan_models.setProperty("variant", "quiet")
        self.btn_rescan_models.clicked.connect(self._rescan_models)
        self.s_conf = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.s_conf.setRange(5, 95)
        saved_thresholds = self.saved_settings.get("detection_thresholds", {})
        if not isinstance(saved_thresholds, dict):
            saved_thresholds = {}
        try:
            saved_conf = float(saved_thresholds.get("confidence", 0.25))
        except (TypeError, ValueError):
            saved_conf = 0.25
        try:
            saved_iou = float(saved_thresholds.get("iou", 0.45))
        except (TypeError, ValueError):
            saved_iou = 0.45
        default_conf = int(round(max(0.05, min(0.95, saved_conf)) * 100))
        default_iou = int(round(max(0.10, min(0.90, saved_iou)) * 100))
        self.s_conf.setValue(default_conf)
        self.v_conf = QtWidgets.QDoubleSpinBox()
        self.v_conf.setRange(0.05, 0.95)
        self.v_conf.setDecimals(2)
        self.v_conf.setSingleStep(0.01)
        self.v_conf.setValue(self.s_conf.value() / 100)
        self.s_iou = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.s_iou.setRange(10, 90)
        self.s_iou.setValue(default_iou)
        self.v_iou = QtWidgets.QDoubleSpinBox()
        self.v_iou.setRange(0.10, 0.90)
        self.v_iou.setDecimals(2)
        self.v_iou.setSingleStep(0.01)
        self.v_iou.setValue(self.s_iou.value() / 100)
        self.s_conf.valueChanged.connect(lambda value: self.v_conf.setValue(value / 100.0))
        self.v_conf.valueChanged.connect(lambda value: self.s_conf.setValue(int(round(value * 100))))
        self.s_iou.valueChanged.connect(lambda value: self.v_iou.setValue(value / 100.0))
        self.v_iou.valueChanged.connect(lambda value: self.s_iou.setValue(int(round(value * 100))))
        self.v_conf.valueChanged.connect(self._apply_thresholds)
        self.v_iou.valueChanged.connect(self._apply_thresholds)

        row = 0
        C.addWidget(self.lbl_model, row, 0, 1, 2)
        C.addWidget(self.btn_rescan_models, row, 2)
        row += 1
        C.addWidget(self.lbl_model_summary, row, 0, 1, 3)
        row += 1
        self.model_list_widget = QtWidgets.QWidget()
        self.model_list_layout = QtWidgets.QVBoxLayout(self.model_list_widget)
        self.model_list_layout.setContentsMargins(0, 0, 0, 0)
        self.model_list_layout.setSpacing(5)
        self.model_scroll = QtWidgets.QScrollArea()
        self.model_scroll.setWidgetResizable(True)
        self.model_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.model_scroll.setMinimumHeight(138)
        self.model_scroll.setMaximumHeight(220)
        self.model_scroll.setWidget(self.model_list_widget)
        C.addWidget(self.model_scroll, row, 0, 1, 3)
        row += 1
        C.addWidget(self._hline(), row, 0, 1, 3)
        row += 1
        self.threshold_title = QtWidgets.QLabel("检测灵敏度")
        self.threshold_title.setProperty("textRole", "sectionTitle")
        C.addWidget(self.threshold_title, row, 0, 1, 3)
        row += 1
        self.lbl_conf = QtWidgets.QLabel()
        self.lbl_conf.setProperty("textRole", "fieldLabel")
        C.addWidget(self.lbl_conf, row, 0)
        C.addWidget(self.s_conf, row, 1)
        C.addWidget(self.v_conf, row, 2)
        row += 1
        self.lbl_iou = QtWidgets.QLabel()
        self.lbl_iou.setProperty("textRole", "fieldLabel")
        C.addWidget(self.lbl_iou, row, 0)
        C.addWidget(self.s_iou, row, 1)
        C.addWidget(self.v_iou, row, 2)
        row += 1
        C.addWidget(self._hline(), row, 0, 1, 3)
        row += 1
        self.analysis_title = QtWidgets.QLabel("分析模式")
        self.analysis_title.setProperty("textRole", "sectionTitle")
        C.addWidget(self.analysis_title, row, 0, 1, 3)
        row += 1
        self.safety_mode_combo = ThemedComboBox()
        self.safety_mode_combo.addItem("实验兼容基线", "baseline")
        self.safety_mode_combo.addItem("生产时序防护（推荐）", "ppe_temporal")
        mode_index = self.safety_mode_combo.findData(self.safety_mode)
        self.safety_mode_combo.setCurrentIndex(max(0, mode_index))
        self.safety_mode_combo.setEnabled(SAFETY_PIPELINE_CONFIG_VALID)
        if not SAFETY_PIPELINE_CONFIG_VALID:
            self.safety_mode_combo.setToolTip(
                "安全管线配置无效，已锁定兼容基线"
                if self.lang_code == "zh" else
                "Safety pipeline config is invalid; baseline is locked"
            )
        self.safety_debug_check = QtWidgets.QCheckBox("显示 raw / stable 调试信息")
        self.safety_debug_check.setChecked(self.safety_debug_overlay)
        self.safety_debug_check.setEnabled(self.safety_mode == "ppe_temporal")
        C.addWidget(self.safety_mode_combo, row, 0, 1, 3)
        row += 1
        C.addWidget(self.safety_debug_check, row, 0, 1, 3)
        row += 1
        self.pipeline_stats_label = QtWidgets.QLabel()
        self.pipeline_stats_label.setObjectName("MutedLabel")
        self.pipeline_stats_label.setWordWrap(True)
        C.addWidget(self.pipeline_stats_label, row, 0, 1, 3)
        row += 1
        C.addWidget(self._hline(), row, 0, 1, 3)
        row += 1
        self.lbl_classes = QtWidgets.QLabel()
        self.lbl_classes.setProperty("textRole", "sectionTitle")
        C.addWidget(self.lbl_classes, row, 0, 1, 3)
        row += 1
        self.class_list_widget = QtWidgets.QWidget()
        self.class_list_layout = QtWidgets.QGridLayout(self.class_list_widget)
        self.class_list_layout.setContentsMargins(0, 0, 0, 0)
        self.class_list_layout.setHorizontalSpacing(10)
        self.class_list_layout.setVerticalSpacing(6)
        self.class_scroll = QtWidgets.QScrollArea()
        self.class_scroll.setWidgetResizable(True)
        self.class_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.class_scroll.setMinimumHeight(88)
        self.class_scroll.setMaximumHeight(145)
        self.class_scroll.setWidget(self.class_list_widget)
        C.addWidget(self.class_scroll, row, 0, 1, 3)
        C.setRowStretch(row, 1)
        self.model_checks = {}
        self.model_status_labels = {}
        self.class_checks = {}
        self._rebuild_model_controls()
        self._rebuild_class_controls()
        self.safety_mode_combo.currentIndexChanged.connect(self._apply_safety_pipeline_settings)
        self.safety_debug_check.toggled.connect(self._apply_safety_pipeline_settings)
        self._update_pipeline_stats({"mode": self.safety_mode})

        # 事件页：时间轴与人工复核入口。
        self.events_tab = QtWidgets.QWidget()
        T = QtWidgets.QGridLayout(self.events_tab)
        T.setContentsMargins(14, 16, 14, 14)
        T.setHorizontalSpacing(8)
        T.setVerticalSpacing(9)
        self.timeline_title = QtWidgets.QLabel()
        self.timeline_title.setProperty("textRole", "sectionTitle")
        self.timeline_hint = QtWidgets.QLabel("选择事件可打开详情并记录人工结论。")
        self.timeline_hint.setWordWrap(True)
        self.timeline_hint.setProperty("textRole", "body")
        self.alert_count_badge = QtWidgets.QLabel("0")
        self.alert_count_badge.setProperty("badge", "neutral")
        self.alert_count_badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.alerts_model = AlertsModel(lambda: self.lang_code)
        self.list_alerts = QtWidgets.QListView()
        self.list_alerts.setModel(self.alerts_model)
        self.list_alerts.setWordWrap(True)
        self.list_alerts.setSpacing(3)
        self.list_alerts.setMinimumHeight(320)
        self.list_alerts.clicked.connect(self._open_alert_detail)
        self.btn_clear_alerts = QBtn()
        self.btn_clear_alerts.setProperty("variant", "danger")
        self.btn_clear_alerts.clicked.connect(self._clear_alerts)
        self.btn_export_csv = QBtn()
        self.btn_export_json = QBtn()
        self.btn_export_csv.setProperty("variant", "secondary")
        self.btn_export_json.setProperty("variant", "secondary")
        self.btn_export_csv.clicked.connect(self._export_csv)
        self.btn_export_json.clicked.connect(self._export_json)
        T.addWidget(self.timeline_title, 0, 0, 1, 2)
        T.addWidget(self.alert_count_badge, 0, 2)
        T.addWidget(self.timeline_hint, 1, 0, 1, 3)
        T.addWidget(self.list_alerts, 2, 0, 1, 3)
        T.addWidget(self.btn_clear_alerts, 3, 0)
        T.addWidget(self.btn_export_csv, 3, 1)
        T.addWidget(self.btn_export_json, 3, 2)
        T.setRowStretch(2, 1)

        # Daily input controls sit beside the image, with network feeds in their tab.
        self.source_actions = QtWidgets.QHBoxLayout()
        self.source_actions.setSpacing(8)
        self.source_actions.addWidget(self.btn_file)
        self.btn_webcam.setProperty("variant", "quiet")
        self.source_actions.addWidget(self.btn_webcam)
        self.source_actions.addStretch()
        L.insertLayout(1, self.source_actions)
        self.source_quick_label.hide()
        self.inspector_tabs.addTab(self.source_tab, "来源")
        self.inspector_tabs.addTab(self.detection_tab, "检测")
        self.inspector_tabs.addTab(self.events_tab, "事件")
        self.inspector_tabs.setCurrentWidget(self.detection_tab)

        self.main_splitter.addWidget(self.viewport_card)
        self.main_splitter.addWidget(self.inspector_panel)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 0)
        self.main_splitter.setSizes([1060, 440])
        central_layout.addWidget(self.main_splitter)
        self.setCentralWidget(central)

        # 告警详情抽屉：固定尺寸，仅平移以避免逐帧重排内容。
        self.drawer = QtWidgets.QFrame(self)
        self.drawer.setObjectName("AlertDrawer")
        self.drawer.setProperty("glass", "strong")
        self.drawer.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        initial_drawer_width = min(520, max(380, self.width() - 48))
        self.drawer.resize(initial_drawer_width, self.height())
        self.drawer.move(self.width(), 0)
        self.drawer.raise_()
        self.drawer_open = False
        self.drawer_anim = QtCore.QPropertyAnimation(self.drawer, b"pos")
        self.drawer_anim.setDuration(220)
        self.drawer_anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self.drawer_layout = QtWidgets.QVBoxLayout(self.drawer)
        self.drawer_layout.setContentsMargins(22, 20, 22, 20)
        self.drawer_layout.setSpacing(12)
        self.drawer_title = QtWidgets.QLabel()
        self.drawer_title.setProperty("textRole", "sectionTitle")
        self.btn_close_drawer = QBtn("关闭详情")
        self.btn_close_drawer.setProperty("variant", "ghost")
        self.btn_close_drawer.clicked.connect(self._close_drawer)
        self.drawer_info = QtWidgets.QTextEdit()
        self.drawer_info.setReadOnly(True)
        self.btn_mark_ok = QBtn()
        self.btn_mark_ok.setProperty("variant", "primary")
        self.btn_mark_ng = QBtn()
        self.btn_mark_ng.setProperty("variant", "danger")
        self.btn_snapshot = QBtn("保存截图")
        self.btn_snapshot.setProperty("variant", "secondary")
        self.btn_mark_ok.clicked.connect(lambda: self._mark_current_alert("valid"))
        self.btn_mark_ng.clicked.connect(lambda: self._mark_current_alert("false_alarm"))
        self.btn_snapshot.clicked.connect(self._save_current_alert_snapshot)
        self.current_alert = None
        self.last_qimage = None
        drawer_header = QtWidgets.QHBoxLayout()
        drawer_header.addWidget(self.drawer_title)
        drawer_header.addStretch(1)
        drawer_header.addWidget(self.btn_close_drawer)
        self.drawer_layout.addLayout(drawer_header)
        self.drawer_layout.addWidget(self.drawer_info)
        row_b = QtWidgets.QHBoxLayout()
        row_b.setSpacing(8)
        row_b.addWidget(self.btn_mark_ok)
        row_b.addWidget(self.btn_mark_ng)
        row_b.addWidget(self.btn_snapshot)
        self.drawer_layout.addLayout(row_b)

        # 源：先填充但屏蔽信号；worker 创建后再绑定切换信号，避免提前触发
        self.sources = [
            {"id":"none", "name":"-- 请选择视频源 --", "kind":"none", "url":None},
            {"id":"webcam","name":"Webcam (device)","kind":"webcam","url":None},
        ]
        self.src_quick.blockSignals(True)
        for s in self.sources: self.src_quick.addItem(s["name"], userData=s["id"])
        self.src_quick.setCurrentIndex(0)
        self.src_quick.blockSignals(False)

        # 线程
        self.worker = VideoWorker(self.catalog_snapshot, self.safety_config)
        self.worker.frameReady.connect(self._on_frame)
        self.worker.statusMsg.connect(self.statusBar().showMessage)
        self.worker.fpsReady.connect(lambda f:self.lbl_fps.setText(f"FPS {f:.1f}"))
        self.worker.newAlert.connect(self._on_new_alert)
        self.worker.modelStatus.connect(self._on_model_status)
        self.worker.modelsApplied.connect(self._on_models_applied)
        self.worker.pipelineStatsReady.connect(self._update_pipeline_stats)
        self.worker.configure_models(self.selected_model_ids)
        self.worker.configure_classes(self._class_selection_map())
        self.worker.configure_thresholds(self.v_conf.value(), self.v_iou.value())
        self.worker.configure_safety_pipeline(
            self.safety_mode,
            self.safety_debug_overlay and self.safety_mode == "ppe_temporal",
        )
        self.worker.start()

        # 现在再连接源切换信号，并主动切一次当前源
        self.src_quick.currentIndexChanged.connect(self._on_quick_switch)

        # 初始化主题 & 文案 & 权限
        self.apply_theme()
        self.apply_i18n()
        self.apply_role(self.role.currentText())
        self._apply_responsive_layout()

    # ---------- 设置与动态模型 ----------
    def _load_settings(self):
        try:
            with self.settings_path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def _initial_model_selection(self):
        selectable = {model.model_id for model in self.catalog_snapshot.models if model.selectable}
        configured = self.saved_settings.get("selected_model_ids")
        if isinstance(configured, list):
            selected = tuple(str(model_id) for model_id in configured if str(model_id) in selectable)
            return self._nonconflicting_selection(self.catalog_snapshot, selected)

        # Migrate settings written by the former SINGLE/DUAL interface.
        old_mode = str(self.saved_settings.get("mode", "")).lower()
        if old_mode == "dual":
            migrated = tuple(model_id for model_id in ("fire", "helmet_only") if model_id in selectable)
            if migrated:
                return self._nonconflicting_selection(self.catalog_snapshot, migrated)
        if old_mode == "single" and "combined" in selectable:
            return ("combined",)

        defaults = tuple(
            model.model_id
            for model in self.catalog_snapshot.models
            if model.selectable and model.default_selected
        )
        if defaults:
            return self._nonconflicting_selection(self.catalog_snapshot, defaults)
        # Newly discovered/unregistered models are deliberately never selected
        # merely because they are the only model present.
        return ()

    @staticmethod
    def _nonconflicting_selection(snapshot, model_ids):
        """Keep catalog order while refusing persisted or newly mapped overlaps."""
        selectable = {model.model_id for model in snapshot.models if model.selectable}
        accepted = []
        for model_id in model_ids:
            if model_id not in selectable or model_id in accepted:
                continue
            candidate = tuple(accepted + [model_id])
            if not snapshot.selection_conflicts(candidate):
                accepted.append(model_id)
        return tuple(accepted)

    def _save_settings(self):
        class_values = self._class_selection_map() if hasattr(self, "class_checks") else {}
        preserved_class_values = self.saved_settings.get("class_enabled", {})
        if not isinstance(preserved_class_values, dict):
            preserved_class_values = {}
        preserved_class_values = dict(preserved_class_values)
        preserved_class_values.update(class_values)
        # Preserve fields written by newer optional modules instead of
        # rebuilding and silently deleting them on every model/class change.
        document = dict(self.saved_settings)
        pipeline_settings = document.get("safety_pipeline", {})
        if not isinstance(pipeline_settings, dict):
            pipeline_settings = {}
        pipeline_settings = dict(pipeline_settings)
        pipeline_settings.update({
            "mode": self.safety_mode,
            "debug_overlay": bool(self.safety_debug_overlay),
        })
        document.update({
            "schema_version": 3,
            "selected_model_ids": list(self.selected_model_ids),
            "class_enabled": preserved_class_values,
            "safety_pipeline": pipeline_settings,
            "detection_thresholds": {
                "confidence": float(self.v_conf.value()) if hasattr(self, "v_conf") else 0.25,
                "iou": float(self.v_iou.value()) if hasattr(self, "v_iou") else 0.45,
            },
        })
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.settings_path.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(document, handle, ensure_ascii=False, indent=2)
            temporary.replace(self.settings_path)
            self.saved_settings = document
        except OSError as exc:
            self.statusBar().showMessage("无法保存模型选择：%s" % exc)

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            child_layout = item.layout()
            if child_layout is not None:
                MainWindow._clear_layout(child_layout)

    def _rebuild_model_controls(self):
        self._clear_layout(self.model_list_layout)
        self.model_checks = {}
        self.model_status_labels = {}
        is_zh = self.lang_code == "zh"
        for spec in self.catalog_snapshot.models:
            row = QtWidgets.QFrame()
            row.setObjectName("ModelRow")
            layout = QtWidgets.QHBoxLayout(row)
            layout.setContentsMargins(8, 5, 8, 5)
            checkbox = QtWidgets.QCheckBox(spec.display_name)
            checkbox.setChecked(spec.model_id in self.selected_model_ids)
            checkbox.setEnabled(spec.selectable)
            classes = ", ".join(
                self.catalog_snapshot.class_profiles.get(class_id, class_profile(class_id)).display_en
                for class_id in spec.canonical_classes
            ) or "unknown classes"
            details = QtWidgets.QLabel(classes)
            details.setObjectName("MutedLabel")
            details.setToolTip(str(spec.weight_path))
            if spec.issue:
                checkbox.setToolTip(spec.issue + "\n" + str(spec.weight_path))
            status_names = (
                {"ready": "可用", "new": "新发现"}
                if is_zh else
                {"ready": "READY", "new": "NEW"}
            )
            status = QtWidgets.QLabel(status_names.get(spec.status, "不可用" if is_zh else "UNAVAILABLE"))
            status.setObjectName("ModelStatus")
            status.setProperty(
                "badge", "success" if spec.status == "ready" else "warning" if spec.status == "new" else "danger"
            )
            status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            if spec.issue:
                status.setToolTip(spec.issue)
            checkbox.stateChanged.connect(
                lambda state, model_id=spec.model_id: self._on_model_selection_changed(model_id, state)
            )
            layout.addWidget(checkbox, 2)
            layout.addWidget(details, 2)
            layout.addWidget(status, 1)
            self.model_checks[spec.model_id] = checkbox
            self.model_status_labels[spec.model_id] = status
            self.model_list_layout.addWidget(row)
        self.model_list_layout.addStretch(1)
        self._update_model_summary()

    def _rebuild_class_controls(self):
        previous = self._class_selection_map() if getattr(self, "class_checks", None) else {}
        configured = self.saved_settings.get("class_enabled", {})
        if not isinstance(configured, dict):
            configured = {}
        self._clear_layout(self.class_list_layout)
        self.class_checks = {}
        class_ids = self.catalog_snapshot.selected_class_ids(self.selected_model_ids)
        if not class_ids:
            empty = QtWidgets.QLabel("未选择检测模型" if self.lang_code == "zh" else "No detection model selected")
            empty.setObjectName("MutedLabel")
            self.class_list_layout.addWidget(empty, 0, 0, 1, 2)
            return
        for index, class_id in enumerate(class_ids):
            profile = self.catalog_snapshot.class_profiles.get(class_id, class_profile(class_id))
            checkbox = QtWidgets.QCheckBox(profile.display_name(self.lang_code))
            checked = previous.get(class_id, configured.get(class_id, profile.enabled_by_default))
            checkbox.setChecked(bool(checked))
            if class_id in (
                self.safety_config.conflict.protected_class_id,
                self.safety_config.conflict.unprotected_class_id,
            ):
                checkbox.setToolTip(
                    "%s · 增强模式仍保留互斥类别作为判断证据" % class_id
                    if self.lang_code == "zh" else
                    "%s · temporal mode retains mutually-exclusive evidence" % class_id
                )
            else:
                checkbox.setToolTip(class_id)
            checkbox.stateChanged.connect(self._apply_classes)
            self.class_checks[class_id] = checkbox
            self.class_list_layout.addWidget(checkbox, index // 2, index % 2)

    def _class_selection_map(self):
        return {class_id: checkbox.isChecked() for class_id, checkbox in self.class_checks.items()}

    def _on_model_selection_changed(self, changed_model_id, state):
        selected = tuple(
            model.model_id
            for model in self.catalog_snapshot.models
            if model.model_id in self.model_checks and self.model_checks[model.model_id].isChecked()
        )
        conflicts = self.catalog_snapshot.selection_conflicts(selected)
        if state and conflicts:
            checkbox = self.model_checks.get(changed_model_id)
            if checkbox is not None:
                checkbox.blockSignals(True)
                checkbox.setChecked(False)
                checkbox.blockSignals(False)
            lines = []
            for class_id, providers in conflicts.items():
                profile = self.catalog_snapshot.class_profiles.get(class_id, class_profile(class_id))
                lines.append("%s: %s" % (profile.display_name(self.lang_code), ", ".join(providers)))
            QtWidgets.QMessageBox.warning(
                self,
                "模型目标重复",
                "这些模型会重复检测相同目标，已取消本次选择：\n\n" + "\n".join(lines),
            )
            self._update_model_summary()
            return

        self.selected_model_ids = selected
        self._rebuild_class_controls()
        if hasattr(self, "worker"):
            self.worker.configure_models(selected)
            self.worker.configure_classes(self._class_selection_map())
        self._save_settings()
        self._update_model_summary()

    def _update_model_summary(self, active_ids=None):
        selected_count = len(self.selected_model_ids)
        class_count = len(self.catalog_snapshot.selected_class_ids(self.selected_model_ids))
        active_count = len(active_ids) if active_ids is not None else getattr(self, "active_model_count", 0)
        device_name = "CUDA" if torch.cuda.is_available() else "CPU"
        if self.lang_code == "zh":
            summary = "已选 %d 个 · 已加载 %d 个 · %d 个目标 · %s"
        else:
            summary = "%d selected · %d loaded · %d targets · %s"
        self.lbl_model_summary.setText(summary % (selected_count, active_count, class_count, device_name))

    def _on_model_status(self, model_id, state, message):
        label = self.model_status_labels.get(model_id)
        if label is None:
            return
        names = (
            {"loading": "加载中", "active": "已启用", "inactive": "可用", "failed": "加载失败"}
            if self.lang_code == "zh" else
            {"loading": "LOADING", "active": "ACTIVE", "inactive": "READY", "failed": "FAILED"}
        )
        label.setText(names.get(state, state))
        label.setToolTip(message)
        tone = {"failed": "danger", "active": "success", "inactive": "neutral", "loading": "warning"}.get(
            state, "neutral"
        )
        set_property(label, "badge", tone)

    def _on_models_applied(self, active_ids, failures):
        self.active_model_count = len(active_ids)
        self._update_model_summary(active_ids)
        if failures:
            self.statusBar().showMessage(
                "部分模型未能启用：" + "; ".join("%s: %s" % item for item in failures.items())
            )
        elif active_ids:
            self.statusBar().showMessage("检测模型已启用：" + ", ".join(active_ids))
        else:
            self.statusBar().showMessage("当前未选择检测模型；视频仍可预览")

    def _rescan_models(self):
        global CATALOG_SNAPSHOT
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            snapshot = MODEL_CATALOG.scan(verify_hashes=True, inspect_new_models=True)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        CATALOG_SNAPSHOT = snapshot
        self.catalog_snapshot = snapshot
        selectable = {model.model_id for model in snapshot.models if model.selectable}
        requested_selection = tuple(
            model_id for model_id in self.selected_model_ids if model_id in selectable
        )
        self.selected_model_ids = self._nonconflicting_selection(snapshot, requested_selection)
        selection_adjusted = self.selected_model_ids != requested_selection
        self._rebuild_model_controls()
        self._rebuild_class_controls()
        self.worker.configure_catalog(snapshot)
        self.worker.configure_models(self.selected_model_ids)
        self.worker.configure_classes(self._class_selection_map())
        self._save_settings()
        self.apply_role(self.role.currentText())
        scan_warnings = list(snapshot.warnings)
        if selection_adjusted:
            scan_warnings.append("模型类别映射发生冲突，已自动取消有冲突的重复模型。")
        if scan_warnings:
            QtWidgets.QMessageBox.warning(self, "RESULTS 扫描提示", "\n".join(scan_warnings))
        else:
            self.statusBar().showMessage("RESULTS 扫描完成：发现 %d 个模型" % len(snapshot.models))

    # ---------- 样式 ----------
    def _build_stylesheet(self):
        t = app_tokens(DARK)
        return build_app_stylesheet(dark=DARK) + f"""
        QFrame#ViewportFrame, QFrame#InspectorFrame {{
            background:{t['surface']}; border:0; border-radius:20px;
        }}
        QFrame#TransportBar, QFrame#RegionBar {{
            background:transparent; border:0; border-radius:0;
        }}
        QFrame#AlertDrawer {{ background:{t['surface']}; border:0; border-left:1px solid {t['border']}; }}
        QFrame#ModelRow {{ background:{t['surface_alt']}; border:0; border-radius:12px; }}
        QFrame#Separator {{ background:{t['border']}; min-height:1px; max-height:1px; border:0; }}
        QTabWidget#InspectorTabs::pane {{ background:transparent; border:0; top:-1px; }}
        QTabWidget#InspectorTabs > QWidget {{ background:transparent; }}
        QTabBar::tab {{ min-height:38px; padding:0 18px; background:transparent; border:0; color:{t['muted']}; }}
        QTabBar::tab:selected {{ background:{t['primary_soft']}; color:{t['cyan']}; border-radius:10px; }}
        QTabBar::tab:hover {{ background:{t['surface_alt']}; }}
        QLabel#SourceState {{ color:{t['muted']}; }}
        QLabel#MutedLabel {{ color:{t['muted']}; font-size:13px; }}
        QListView::item {{ border-bottom:1px solid {t['border']}; padding:12px; }}
        QListView::item:selected {{ border-left:3px solid {t['primary']}; }}
        QToolButton {{ text-align:left; }}
        QToolBar#AppBar {{ padding:10px 24px; spacing:12px; }}
        """

    def _card(self, object_name="CardFrame", material="panel"):
        card = QtWidgets.QFrame()
        card.setObjectName(object_name)
        card.setProperty("glass", material)
        card.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        return card

    def _hline(self):
        line = QtWidgets.QFrame()
        line.setObjectName("Separator")
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        return line
    def _spacer(self):
        s = QtWidgets.QWidget(); s.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding); return s

    def _set_region_tools_visible(self, visible):
        self.region_bar.setVisible(bool(visible))
        self.region_toggle.setArrowType(QtCore.Qt.ArrowType.DownArrow if visible else QtCore.Qt.ArrowType.RightArrow)

    def _set_demo_sources_visible(self, visible):
        self.demo_sources_panel.setVisible(bool(visible))
        self.demo_toggle.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if visible else QtCore.Qt.ArrowType.RightArrow
        )

    def _apply_responsive_layout(self):
        compact = self.width() < 1320
        short = self.height() < 880
        self.canvas.setMinimumHeight(180 if short else 300)
        self.sub.setVisible(not compact)
        self.regions_hint.hide()
        self.regions_label.hide()
        self.system_badge.setVisible(not compact)
        self.source_state.setMaximumWidth(150 if compact else 240)
        self.inspector_panel.setMinimumWidth(390 if compact else 420)
        self.inspector_panel.setMaximumWidth(430 if compact else 520)
        self.model_scroll.setMinimumHeight(115 if short else 138)
        self.model_scroll.setMaximumHeight(115 if short else 200)
        self.class_scroll.setMinimumHeight(54 if short else 88)
        self.class_scroll.setMaximumHeight(72 if short else 145)

    # ---------- 文字和图标 ----------
    def L(self, key, **fmt):
        s = I18N["zh" if self.lang_code=="zh" else "en"].get(key, key)
        return s.format(**fmt) if fmt else s

    def on_language_change(self, text):
        self.lang_code = "zh" if text.startswith("zh") else "en"
        self.apply_i18n()

    def apply_i18n(self):
        self.title.setText(APP_NAME)
        self.sub.setText(SUBTITLE)
        is_zh = self.lang_code == "zh"
        self.system_badge.setText("本地计算" if is_zh else "LOCAL")
        self.preferences_button.setText("显示与权限" if is_zh else "Preferences")
        self.preference_language_label.setText("界面语言" if is_zh else "Language")
        self.preference_role_label.setText("操作身份" if is_zh else "Role")
        self.region_toggle.setText("分析区域" if is_zh else "Analysis regions")
        self.canvas.empty_title = "准备好，就开始。" if is_zh else "Ready when you are."
        self.canvas.empty_detail = "选择本地视频或摄像头，画面将在这里显示。" if is_zh else "Choose a video or webcam. Your feed will appear here."
        self.canvas.update()
        self.btn_theme.setText(("浅色" if DARK else "深色") if is_zh else ("Light" if DARK else "Dark"))
        self.btn_rm.setText(
            ("动效：减弱" if REDUCE_MOTION else "动效：标准")
            if is_zh else
            ("Motion: Reduced" if REDUCE_MOTION else "Motion: Full")
        )
        self.btn_play.setText("播放" if is_zh else "Play")
        self.btn_pause.setText("暂停" if is_zh else "Pause")
        self.btn_stop.setText("停止" if is_zh else "Stop")
        self.btn_add_roi.setText("检测区域" if is_zh else "Add ROI")
        self.btn_add_mask.setText("屏蔽区域" if is_zh else "Add Mask")
        self.btn_edit_poly.setText("编辑" if is_zh else "Edit Points")
        self.btn_clear_poly.setText("清除" if is_zh else "Clear Regions")
        self.btn_file.setText("选择本地视频" if is_zh else "Choose Local Video")
        self.txt_url.setPlaceholderText(self.L('url_ph'))
        self.btn_load.setText("连接" if is_zh else "Connect")
        self.btn_webcam.setText("打开摄像头" if is_zh else "Open Webcam")
        self.btn_demo1.setText("火情示例" if is_zh else "Fire Sample")
        self.btn_demo2.setText("防护示例" if is_zh else "PPE Sample")
        self.demo_toggle.setText("示例源 · 需要网络" if is_zh else "Sample feeds · network required")
        self.monitor_eyebrow.setText("LIVE DETECTION")
        self.monitor_title.setText("实时检测" if is_zh else "Live detection")
        self.transport_label.setText("当前来源" if is_zh else "Current source")
        self.regions_label.setText("分析区域" if is_zh else "Analysis regions")
        self.regions_hint.setText(
            "ROI 参与检测；屏蔽区会被忽略" if is_zh else "ROIs are analysed; masks are ignored"
        )
        self.inspector_eyebrow.setText("INSPECTOR")
        self.inspector_title.setText("操作与审查" if is_zh else "Controls & Review")
        self.source_title.setText("网络视频源" if is_zh else "Network sources")
        self.source_helper.setText(
            "连接 RTSP 或 HTTP 视频流。也可以使用示例源检查界面。"
            if is_zh else
            "Connect an RTSP or HTTP stream, or try a sample feed."
        )
        self.source_url_label.setText("网络地址" if is_zh else "Network address")
        self.source_quick_label.setText("设备" if is_zh else "Device")
        for index in range(self.src_quick.count()):
            if self.src_quick.itemData(index) == "none":
                self.src_quick.setItemText(
                    index,
                    "-- 请选择视频源 --" if is_zh else "-- Choose a video source --",
                )
                break
        if not self.active_source_id:
            self.source_state.setText("等待选择视频源" if is_zh else "Waiting for a video source")
        self.lbl_model.setText("检测模型" if is_zh else "Detection models")
        self.threshold_title.setText("检测灵敏度" if is_zh else "Detection sensitivity")
        self.analysis_title.setText("分析模式" if is_zh else "Analysis mode")
        self.safety_mode_combo.blockSignals(True)
        self.safety_mode_combo.setItemText(0, "实验兼容基线" if is_zh else "Experimental baseline")
        self.safety_mode_combo.setItemText(1, "生产时序防护（推荐）" if is_zh else "Production temporal protection (recommended)")
        self.safety_mode_combo.blockSignals(False)
        self.safety_debug_check.setText(
            "显示 raw / stable 调试信息" if is_zh else "Show raw / stable debug overlay"
        )
        if not SAFETY_PIPELINE_CONFIG_VALID:
            self.safety_mode_combo.setToolTip(
                "安全管线配置无效，已锁定兼容基线"
                if is_zh else
                "Safety pipeline config is invalid; baseline is locked"
            )
        self.lbl_classes.setText("识别目标" if is_zh else "Detection targets")
        self.btn_rescan_models.setText("刷新模型" if is_zh else "Refresh")
        self.lbl_conf.setText("置信阈值" if is_zh else "Confidence")
        self.lbl_iou.setText("重叠阈值" if is_zh else "NMS IoU")
        self.timeline_title.setText("事件审查" if is_zh else "Event review")
        self.timeline_hint.setText(
            "选择事件可打开详情并记录人工结论。"
            if is_zh else
            "Select an event to inspect it and record a human conclusion."
        )
        self.btn_clear_alerts.setText("清空" if is_zh else "Clear")
        self.btn_export_csv.setText("导出 CSV" if is_zh else "Export CSV")
        self.btn_export_json.setText("导出 JSON" if is_zh else "Export JSON")
        self.inspector_tabs.setTabText(0, "来源" if is_zh else "Source")
        self.inspector_tabs.setTabText(1, "检测" if is_zh else "Detection")
        self.inspector_tabs.setTabText(2, "事件" if is_zh else "Events")
        status_names = (
            {"ready": "可用", "new": "新发现"}
            if is_zh else
            {"ready": "READY", "new": "NEW"}
        )
        for spec in self.catalog_snapshot.models:
            label = self.model_status_labels.get(spec.model_id)
            if label is not None:
                label.setText(status_names.get(spec.status, "不可用" if is_zh else "UNAVAILABLE"))
        self.drawer_title.setText(self.L('alert_detail'))
        self.btn_mark_ok.setText("标注为有效" if is_zh else "Mark Valid")
        self.btn_mark_ng.setText("标注为误报" if is_zh else "Mark False")
        self.btn_snapshot.setText("保存截图" if is_zh else "Save Snapshot")
        self.btn_close_drawer.setText("关闭详情" if is_zh else "Close")
        self._rebuild_class_controls()
        self._update_model_summary()
        self._update_pipeline_stats({"mode": self.safety_mode})

    # ---------- 权限 ----------
    def apply_role(self, role_text=None):
        role = role_text or self.role.currentText()
        self.is_viewer = (role == "Viewer")
        allow = not self.is_viewer
        widgets = [
            self.btn_theme, self.btn_rm, self.lang,
            self.btn_play, self.btn_pause, self.btn_stop, self.src_quick,
            self.btn_add_roi, self.btn_add_mask, self.btn_edit_poly, self.btn_clear_poly,
            self.btn_file, self.txt_url, self.btn_load, self.btn_webcam, self.btn_demo1, self.btn_demo2,
            self.s_conf, self.v_conf, self.s_iou, self.v_iou, self.btn_rescan_models,
            self.safety_mode_combo, self.safety_debug_check,
            *self.model_checks.values(), *self.class_checks.values(),
            self.list_alerts, self.btn_clear_alerts, self.btn_export_csv, self.btn_export_json,
            self.btn_mark_ok, self.btn_mark_ng, self.btn_snapshot
        ]
        for w in widgets: w.setEnabled(allow)
        self.safety_mode_combo.setEnabled(allow and SAFETY_PIPELINE_CONFIG_VALID)
        self.safety_debug_check.setEnabled(allow and self.safety_mode == "ppe_temporal")
        if allow:
            for spec in self.catalog_snapshot.models:
                checkbox = self.model_checks.get(spec.model_id)
                if checkbox is not None and not spec.selectable:
                    checkbox.setEnabled(False)
        self.role.setEnabled(True)

        if not allow: self.statusBar().showMessage("Viewer mode: read-only")
        else: self.statusBar().clearMessage()

    # ---------- 编辑模式 ----------
    def _set_edit_mode(self, m: str):
        if RBAC_ENABLED and self.is_viewer:
            self.statusBar().showMessage(self.L("status_no_perm_edit")); return
        try:
            self.canvas.set_mode(m)
        except Exception as e:
            print("set_mode error:", e); return
        tips_zh = {
            "add-roi": "编辑模式：新增 ROI（单击落点，双击闭合）",
            "add-mask": "编辑模式：新增屏蔽区（单击落点，双击闭合）",
            "edit": "编辑模式：拖拽顶点微调",
            "select": "编辑模式：选择",
        }
        tips_en = {
            "add-roi": "Mode: Add ROI (click to add points, double-click to close)",
            "add-mask": "Mode: Add Mask (click to add points, double-click to close)",
            "edit": "Mode: Edit (drag vertices)",
            "select": "Mode: Select",
        }
        if self.lang_code=="zh": self.statusBar().showMessage(tips_zh.get(m, f"模式：{m}"))
        else: self.statusBar().showMessage(tips_en.get(m, f"Mode: {m}"))

    def _clear_polys(self):
        try:
            self.canvas.clear_all_polys()
            self.worker.configure_polys([], [])
            self.statusBar().showMessage("已清空区域" if self.lang_code=="zh" else "Regions cleared")
        except Exception as e:
            print("clear_polys error:", e)

    # ---------- 事件 ----------
    def closeEvent(self, e):
        try:
            if self.worker.isRunning():
                self.worker.request_stop()
                if not self.worker.wait(300):
                    e.ignore()
                    if not self._closing_after_worker:
                        self._closing_after_worker = True
                        self.statusBar().showMessage("正在安全释放模型，请稍候…")
                        self.worker.finished.connect(self.close)
                    return
        except Exception:
            pass
        return super().closeEvent(e)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Escape and self.drawer_open:
            self._close_drawer()
            event.accept()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "inspector_panel"):
            self._apply_responsive_layout()
        if not hasattr(self, "drawer"):
            return
        self.drawer_anim.stop()
        drawer_width = self._drawer_width()
        self.drawer.resize(drawer_width, self.height())
        if self.drawer_open:
            self.drawer.move(self.width() - drawer_width, 0)
        else:
            self.drawer.move(self.width(), 0)

    # ---------- 源 ----------
    def _on_quick_switch(self, idx):
        if not hasattr(self, "worker") or self.worker is None: return
        if idx<0 or idx>=len(self.sources): return
        s = self.sources[idx]
        if s["kind"] == "none":
            return
        self.set_source(0 if s["kind"]=="webcam" else s["url"], s["kind"], s["id"])
    def set_source(self, src, kind="file", sid: Optional[str]=None):
        if RBAC_ENABLED and self.is_viewer:
            self.statusBar().showMessage(self.L("status_no_perm_src")); return
        sid = sid or (kind if kind=="webcam" else str(src))
        self.active_source_id = sid
        self.source_state.setText(str(sid))
        self.live_badge.setText("●  CONNECTING")
        set_property(self.live_badge, "badge", "warning")
        self.statusBar().showMessage(self.L("status_switching"))
        self.canvas.clear_all_polys()
        self.worker.set_source(src, sid, kind)
    def stop_source(self):
        self.worker.stop_source(); self.active_source_id = None
        self.source_state.setText("等待选择视频源" if self.lang_code == "zh" else "Waiting for a video source")
        self.live_badge.setText("●  STANDBY")
        set_property(self.live_badge, "badge", "neutral")
    def pick_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, self.L("pick_file"), "", "Video (*.mp4 *.avi *.mkv *.mov)")
        if path:
            sid = f"file_{int(time.time())}"
            self.sources.insert(0, {"id":sid,"name":os.path.basename(path),"kind":"file","url":path})
            self.src_quick.insertItem(0, os.path.basename(path), userData=sid)
            self.src_quick.setCurrentIndex(0)
            self.set_source(path, "file", sid)
    def load_url(self):
        url = self.txt_url.text().strip()
        if not url: return
        kind = "rtsp" if url.lower().startswith("rtsp") else "http"
        sid = f"url_{int(time.time())}"
        self.sources.insert(0, {"id":sid,"name":url,"kind":kind,"url":url})
        self.src_quick.insertItem(0, url, userData=sid)
        self.src_quick.setCurrentIndex(0)
        self.set_source(url, kind, sid)

    # ---------- 阈值/类别 ----------
    def _apply_safety_pipeline_settings(self, *_):
        previous_mode = self.safety_mode
        mode = self.safety_mode_combo.currentData()
        if not SAFETY_PIPELINE_CONFIG_VALID:
            mode = "baseline"
            self.safety_mode_combo.blockSignals(True)
            self.safety_mode_combo.setCurrentIndex(
                max(0, self.safety_mode_combo.findData("baseline"))
            )
            self.safety_mode_combo.blockSignals(False)
        self.safety_mode = mode if mode in ("baseline", "ppe_temporal") else "baseline"
        self.safety_debug_overlay = bool(self.safety_debug_check.isChecked())
        enhanced = self.safety_mode == "ppe_temporal"
        self.safety_debug_check.setEnabled(enhanced and not self.is_viewer)
        if hasattr(self, "worker"):
            self.worker.configure_safety_pipeline(
                self.safety_mode,
                self.safety_debug_overlay and enhanced,
            )
        self._save_settings()
        self._update_pipeline_stats({"mode": self.safety_mode})
        if previous_mode != self.safety_mode:
            message = (
                ("已切换到生产时序防护；旧轨迹和证据已安全清空" if enhanced else "已切换到实验兼容基线；旧轨迹和证据已安全清空")
                if self.lang_code == "zh" else
                ("Production temporal protection enabled; prior evidence was reset" if enhanced else "Experimental baseline enabled; prior evidence was reset")
            )
        else:
            message = (
                ("已显示 raw / stable 调试信息" if self.safety_debug_overlay else "已隐藏 raw / stable 调试信息")
                if self.lang_code == "zh" else
                ("Raw / stable debug overlay shown" if self.safety_debug_overlay else "Raw / stable debug overlay hidden")
            )
        self.statusBar().showMessage(message)

    def _update_pipeline_stats(self, stats):
        stats = stats if isinstance(stats, dict) else {}
        mode = str(stats.get("mode", self.safety_mode))
        previous = getattr(self, "_pipeline_stats_snapshot", {})
        if isinstance(previous, dict) and previous.get("mode") == mode:
            merged = dict(previous)
            merged.update(stats)
            stats = merged
        self._pipeline_stats_snapshot = dict(stats)
        is_zh = self.lang_code == "zh"
        if mode != "ppe_temporal":
            text = (
                "兼容基线 · 原有按来源/类别的多帧门控"
                if is_zh else
                "Compatibility baseline · legacy source/class temporal gate"
            )
        else:
            tracks = int(stats.get("active_tracks", 0) or 0)
            candidates = int(stats.get("active_candidates", 0) or 0)
            conflicts = int(stats.get("conflicts_resolved", 0) or 0)
            events = int(stats.get("confirmed_events", stats.get("alerts_emitted", 0)) or 0)
            frames = int(stats.get("frames_processed", 0) or 0)
            reacquired = int(stats.get("reacquire_successes", 0) or 0)
            filtered = int(stats.get("alert_transients_filtered", 0) or 0)
            filtered += int(stats.get("low_confidence_filtered", 0) or 0)
            dropped = int(stats.get("admission_dropped", 0) or 0) + int(
                stats.get("alert_gate_dropped", 0) or 0
            )
            dropped += int(stats.get("legacy_admission_dropped", 0) or 0)
            dropped += int(stats.get("legacy_invalid_detections", 0) or 0)
            p95_ms = float(stats.get("pipeline_ms_p95", 0.0) or 0.0)
            text = (
                "生产时序防护 · %d 帧 · %d 轨迹 / %d 候选 · 重捕获 %d · 过滤 %d / 拒绝 %d · %.1f ms p95 · %d 事件"
                if is_zh else
                "Production temporal · %d frames · %d tracks / %d candidates · %d reacquired · %d filtered / %d dropped · %.1f ms p95 · %d events"
            ) % (frames, tracks, candidates, reacquired, filtered + conflicts, dropped, p95_ms, events)
        self.pipeline_stats_label.setText(text)

    def _apply_thresholds(self):
        conf = float(self.v_conf.value()); iou = float(self.v_iou.value())
        self.worker.configure_thresholds(conf, iou)
        self._save_settings()
        self.statusBar().showMessage(self.L("status_applied", conf=conf, iou=iou))
    def _apply_classes(self):
        self.worker.configure_classes(self._class_selection_map())
        self._save_settings()

    # ---------- 帧，告警 ----------
    def _on_frame(self, qimg: QtGui.QImage, w, h):
        rois, masks = self.canvas.export_polys_int()
        self.worker.configure_polys(rois, masks)
        self.last_qimage = qimg.copy()
        self.canvas.update_frame(qimg, w, h)
        if self.live_badge.text() != "●  LIVE":
            self.live_badge.setText("●  LIVE")
            set_property(self.live_badge, "badge", "success")
    def _on_new_alert(self, a_dict):
        a_dict["review"] = "unreviewed"
        snapshot_bytes = None
        if self.last_qimage is not None:
            data = QtCore.QByteArray()
            buffer = QtCore.QBuffer(data)
            if buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly):
                self.last_qimage.save(buffer, "JPG", 85)
                buffer.close()
                snapshot_bytes = bytes(data)
        a_dict["_snapshot_bytes"] = snapshot_bytes
        self.alerts_model.add_alert(Alert(**a_dict))
        if hasattr(self.worker, "record_alert"):
            self.worker.record_alert(a_dict)
        self.alert_count_badge.setText(str(len(self.alerts_model.items)))
        set_property(self.alert_count_badge, "badge", "warning")
    def _open_alert_detail(self, index: QtCore.QModelIndex):
        a: Alert = self.alerts_model.data(index, QtCore.Qt.ItemDataRole.UserRole)
        if not a:
            self.statusBar().showMessage(self.L('no_alert')); return
        self.current_alert = a
        info = {
            ("时间" if self.lang_code=="zh" else "Time"): time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(a.ts)),
            ("来源" if self.lang_code=="zh" else "Source"): a.source_id,
            ("类别" if self.lang_code=="zh" else "Class"): class_display_name(a.cls, self.lang_code),
            ("置信度" if self.lang_code=="zh" else "Confidence"): round(a.conf, 2),
            ("模型" if self.lang_code=="zh" else "Models"): ", ".join(getattr(a, "model_ids", ()) or ()) or "—",
            ("告警等级" if self.lang_code=="zh" else "Severity"): getattr(a, "severity", "info"),
            ("人工结论" if self.lang_code=="zh" else "Review"): getattr(a, "review", "unreviewed"),
            ("检测参数" if self.lang_code=="zh" else "Detection settings"): f"YOLOv5 | conf≥{self.v_conf.value():.2f} | IoU≤{self.v_iou.value():.2f}"
        }
        if getattr(a, "track_id", None) is not None:
            info[("轨迹 ID" if self.lang_code=="zh" else "Track ID")] = a.track_id
            info[("轨迹会话" if self.lang_code=="zh" else "Track session")] = getattr(
                a, "session_id", "—"
            )
            info[("编号作用域" if self.lang_code=="zh" else "ID namespace")] = getattr(
                a, "track_id_namespace", "legacy"
            )
            info[("事件 ID" if self.lang_code=="zh" else "Event ID")] = getattr(a, "event_id", a.id)
            info[("风险类型" if self.lang_code=="zh" else "Risk type")] = getattr(a, "risk_type", "—")
            info[("稳定状态" if self.lang_code=="zh" else "Stable state")] = getattr(a, "stable_state", "—")
            info[("风险状态" if self.lang_code=="zh" else "Risk state")] = getattr(a, "risk_state", "—")
            info[("确认延迟（秒）" if self.lang_code=="zh" else "Confirmation delay (s)")] = round(
                float(getattr(a, "confirmation_delay", 0.0)), 3
            )
        if getattr(a, "evidence_hits", None) is not None:
            info[("有效证据帧" if self.lang_code=="zh" else "Verified evidence hits")] = int(
                getattr(a, "evidence_hits", 0)
            )
            info[("证据持续（秒）" if self.lang_code=="zh" else "Evidence duration (s)")] = round(
                float(getattr(a, "evidence_duration", 0.0)), 3
            )
            info[("证据 EMA" if self.lang_code=="zh" else "Evidence EMA")] = round(
                float(getattr(a, "evidence_ema", getattr(a, "conf", 0.0))), 4
            )
            info[("判定依据" if self.lang_code=="zh" else "Decision reason")] = getattr(
                a, "decision_reason", "—"
            )
        info[("分析模式" if self.lang_code=="zh" else "Pipeline mode")] = getattr(
            a, "pipeline_mode", "baseline"
        )
        info[("配置指纹" if self.lang_code=="zh" else "Config fingerprint")] = getattr(
            a, "config_fingerprint", "—"
        )
        self.drawer_info.setPlainText(json.dumps(info, indent=2, ensure_ascii=False))
        self._open_drawer()

    def _mark_current_alert(self, conclusion):
        if self.current_alert is None or self.current_alert not in self.alerts_model.items:
            self.current_alert = None
            self.statusBar().showMessage(self.L("no_alert"))
            return
        self.current_alert.review = conclusion
        if hasattr(self.worker, "record_review"):
            self.worker.record_review(
                self.current_alert.id,
                conclusion,
                getattr(self.current_alert, "track_id", None),
                getattr(self.current_alert, "event_id", None),
                getattr(self.current_alert, "session_id", None),
                getattr(self.current_alert, "track_id_namespace", None),
            )
        self.statusBar().showMessage("告警人工结论已记录：%s" % conclusion)
        index = self.alerts_model.index(self.alerts_model.items.index(self.current_alert), 0)
        self._open_alert_detail(index)

    def _clear_alerts(self):
        self.alerts_model.clear()
        self.alert_count_badge.setText("0")
        set_property(self.alert_count_badge, "badge", "neutral")
        self.current_alert = None
        self._close_drawer()
        self.statusBar().showMessage("告警记录已清空" if self.lang_code == "zh" else "Alerts cleared")

    def _save_current_alert_snapshot(self):
        if self.current_alert is None or not getattr(self.current_alert, "_snapshot_bytes", None):
            self.statusBar().showMessage("该告警没有可保存的画面")
            return
        default_name = "alert_%s.png" % self.current_alert.id
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存告警截图", default_name, "PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if not path:
            return
        image = QtGui.QImage.fromData(self.current_alert._snapshot_bytes, "JPG")
        if image.isNull() or not image.save(path):
            QtWidgets.QMessageBox.warning(self, "保存失败", "无法写入截图：%s" % path)
            return
        self.current_alert.snapshot = path
        self.statusBar().showMessage("告警截图已保存：%s" % path)

    # ---------- 抽屉 ----------
    def _drawer_width(self):
        return min(520, max(380, self.width() - 48))

    def _open_drawer(self):
        drawer_width = self._drawer_width()
        self.drawer.resize(drawer_width, self.height())
        start = self.drawer.pos()
        end = QtCore.QPoint(self.width() - drawer_width, 0)
        if not self.drawer_open:
            self._drawer_return_focus = QtWidgets.QApplication.focusWidget()
        self.drawer_open = True
        self.drawer.raise_()
        self.btn_close_drawer.setFocus()
        self.drawer_anim.stop()
        if REDUCE_MOTION:
            self.drawer.move(end)
        else:
            self.drawer_anim.setDuration(220)
            self.drawer_anim.setStartValue(start)
            self.drawer_anim.setEndValue(end)
            self.drawer_anim.start()

    def _close_drawer(self):
        start = self.drawer.pos()
        end = QtCore.QPoint(self.width(), 0)
        self.drawer_open = False
        if getattr(self, "_drawer_return_focus", None) is not None:
            self._drawer_return_focus.setFocus()
        self.drawer_anim.stop()
        if REDUCE_MOTION:
            self.drawer.move(end)
        else:
            self.drawer_anim.setDuration(160)
            self.drawer_anim.setStartValue(start)
            self.drawer_anim.setEndValue(end)
            self.drawer_anim.start()

    # ---------- 导出 ----------
    def _export_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, self.L('export_csv'), "alerts.csv", "CSV (*.csv)")
        if not path: return
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "event_id", "session_id", "track_id", "track_id_namespace",
                "ts", "iso", "source", "class_id", "class_name",
                "confidence", "severity", "models", "pipeline_mode", "config_fingerprint", "risk_type",
                "stable_state", "risk_state", "confirmation_delay", "evidence_hits",
                "evidence_duration", "evidence_presence_ratio", "evidence_ema",
                "evidence_stability", "decision_reason", "review", "snapshot",
            ])
            for a in self.alerts_model.items:
                writer.writerow([
                    a.id, getattr(a, "event_id", ""), getattr(a, "session_id", ""),
                    getattr(a, "track_id", ""), getattr(a, "track_id_namespace", ""),
                    a.ts, time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(a.ts)),
                    a.source_id, a.cls, class_display_name(a.cls, self.lang_code), "%.4f" % a.conf,
                    getattr(a, "severity", "info"), ";".join(getattr(a, "model_ids", ()) or ()),
                    getattr(a, "pipeline_mode", "baseline"), getattr(a, "config_fingerprint", ""),
                    getattr(a, "risk_type", ""), getattr(a, "stable_state", ""),
                    getattr(a, "risk_state", ""), getattr(a, "confirmation_delay", ""),
                    getattr(a, "evidence_hits", ""), getattr(a, "evidence_duration", ""),
                    getattr(a, "evidence_presence_ratio", ""), getattr(a, "evidence_ema", ""),
                    getattr(a, "evidence_stability", ""), getattr(a, "decision_reason", ""),
                    getattr(a, "review", "unreviewed"), getattr(a, "snapshot", None) or "",
                ])
        self.statusBar().showMessage(f"CSV -> {path}")
    def _export_json(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, self.L('export_json'), "alerts.json", "JSON (*.json)")
        if not path: return
        with open(path, "w", encoding="utf-8") as f:
            records = []
            for alert in self.alerts_model.items:
                record = {key: value for key, value in alert.__dict__.items() if key != "_snapshot_bytes"}
                record["class_name"] = class_display_name(alert.cls, self.lang_code)
                records.append(record)
            json.dump(records, f, ensure_ascii=False, indent=2)
        self.statusBar().showMessage(f"JSON -> {path}")

    # ---------- 主题/动效 ----------
    def toggle_theme(self):
        global DARK; DARK = not DARK
        self.apply_theme()
        is_zh = self.lang_code == "zh"
        self.btn_theme.setText(("浅色" if DARK else "深色") if is_zh else ("Light" if DARK else "Dark"))
    def toggle_reduce_motion(self):
        global REDUCE_MOTION; REDUCE_MOTION = not REDUCE_MOTION
        os.environ["SENTINEL_REDUCE_MOTION"] = "1" if REDUCE_MOTION else "0"
        if REDUCE_MOTION and self.drawer_anim.state() == QtCore.QAbstractAnimation.State.Running:
            self.drawer_anim.stop()
            target_x = self.width() - self.drawer.width() if self.drawer_open else self.width()
            self.drawer.move(target_x, 0)
        is_zh = self.lang_code == "zh"
        self.btn_rm.setText(
            ("动效：减弱" if REDUCE_MOTION else "动效：标准")
            if is_zh else
            ("Motion: Reduced" if REDUCE_MOTION else "Motion: Full")
        )
        self.statusBar().showMessage(
            ("已启用减弱动态" if REDUCE_MOTION else "已恢复标准动效")
            if is_zh else
            ("Reduced motion enabled" if REDUCE_MOTION else "Standard motion restored")
        )
    def apply_theme(self):
        self.setStyleSheet(self._build_stylesheet())
        self.drawer.setStyleSheet("")
        for combo in (self.lang, self.role, self.src_quick, self.safety_mode_combo):
            combo.apply_theme()
        self.canvas.update()

# ===================== 入口 =====================
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    install_ui_font(app)
    w = MainWindow(); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
