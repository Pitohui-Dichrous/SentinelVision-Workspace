# -*- coding: utf-8 -*-
"""Sentinel Vision dynamic YOLOv5 desktop detector.

Deployment models are discovered from RESULTS on every start.  Users select
any non-conflicting combination with checkboxes; future registered classes are
rendered and filtered without changing this source file.
"""

import sys, os, time, json, threading, csv
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
from project_paths import PROJECT_ROOT
from ui_font import install_ui_font
from ui_theme import build_stylesheet as build_app_stylesheet, set_property

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
DARK = True
REDUCE_MOTION = False

# ===================== 使用者 / 权限 =====================
RBAC_ENABLED = True
DEFAULT_ROLE = "Admin"  # Admin / Operator / Viewer

# ===================== 动态模型/类别 =====================
MODEL_CATALOG = ModelCatalog()
CATALOG_SNAPSHOT = MODEL_CATALOG.scan(verify_hashes=True, inspect_new_models=True)


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
def point_in_polygon(pt, poly):
    x, y = pt; inside = False; n = len(poly)
    for i in range(n):
        x1,y1 = poly[i]; x2,y2 = poly[(i+1)%n]
        if ((y1>y)!=(y2>y)) and (x < (x2-x1)*(y-y1)/(y2-y1+1e-9) + x1):
            inside = not inside
    return inside

# ===================== 点击动画按钮 =====================
class AnimatedButton(QtWidgets.QPushButton):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._scale = 1.0
        self._anim = QtCore.QPropertyAnimation(self, b"scale", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        eff = QtWidgets.QGraphicsDropShadowEffect(self)
        eff.setBlurRadius(0); eff.setOffset(0, 2)
        eff.setColor(qcolor(TOKENS_DARK["primary"], 160))
        self.setGraphicsEffect(eff); self._glow = eff
    def getScale(self): return self._scale
    def setScale(self, v: float): self._scale = v; self.update()
    scale = QtCore.Property(float, getScale, setScale)
    def mousePressEvent(self, e):
        if not REDUCE_MOTION:
            self._anim.stop(); self._anim.setStartValue(self._scale); self._anim.setEndValue(0.96); self._anim.start()
            self._glow.setBlurRadius(18)
        super().mousePressEvent(e)
    def mouseReleaseEvent(self, e):
        if not REDUCE_MOTION:
            self._anim.stop(); self._anim.setStartValue(self._scale); self._anim.setEndValue(1.0); self._anim.start()
            QtCore.QTimer.singleShot(120, lambda: self._glow.setBlurRadius(0))
        super().mouseReleaseEvent(e)
    def paintEvent(self, ev):
        p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.translate(self.rect().center()); p.scale(self._scale, self._scale); p.translate(-self.rect().center())
        opt = QtWidgets.QStyleOptionButton(); opt.initFrom(self); opt.text = self.text(); opt.icon = self.icon()
        self.style().drawControl(QtWidgets.QStyle.ControlElement.CE_PushButton, opt, p, self)

QBtn = AnimatedButton

# ===================== 主题化下拉 =====================
class ThemedComboBox(QtWidgets.QComboBox):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.setEditable(False); self.setMinimumHeight(36)
        self._apply_qss()
    def _apply_qss(self):
        t = P()
        self.setStyleSheet(f"""
            QComboBox {{
                background:{t['bg2']}; color:{t['text']};
                border:1px solid {t['line']}; border-radius:9px;
                padding:5px 30px 5px 10px;
            }}
            QComboBox::drop-down {{
                width:24px; subcontrol-origin: padding; subcontrol-position: top right;
                border-left: 1px solid {t['line']}; background:{t['bg2']};
                border-top-right-radius:9px; border-bottom-right-radius:9px;
            }}
            QComboBox::down-arrow {{ image: none; width:0; height:0; }}
            QComboBox QAbstractItemView {{
                background:{t['card']}; color:{t['text']};
                border:1px solid {t['line']}; outline:0;
                selection-background-color: {t['accentSoft']};
                selection-color: {t['text']};
            }}
        """)
    def paintEvent(self, e):
        super().paintEvent(e)
        t = P(); p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.Antialiasing)
        r = self.rect(); cx = r.right()-12; cy = r.center().y()
        tri = QtGui.QPolygonF([QtCore.QPointF(cx-5, cy-2), QtCore.QPointF(cx+5, cy-2), QtCore.QPointF(cx, cy+4)])
        p.setBrush(qcolor(t["accent"])); p.setPen(QtCore.Qt.PenStyle.NoPen); p.drawPolygon(tri); p.end()
    def apply_theme(self): self._apply_qss(); self.update()

# ===================== 动态多模型推理线程 =====================
class VideoWorker(QtCore.QThread):
    frameReady = QtCore.Signal(object, int, int)
    statusMsg = QtCore.Signal(str)
    fpsReady = QtCore.Signal(float)
    newAlert = QtCore.Signal(dict)
    modelStatus = QtCore.Signal(str, str, str)
    modelsApplied = QtCore.Signal(object, object)

    def __init__(self, snapshot: CatalogSnapshot):
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

    # ---------- thread-safe configuration ----------
    def configure_thresholds(self, conf, iou):
        iou = float(max(0.0, min(1.0, iou)))
        with self._lock:
            self.conf = float(conf)
            self.iou = iou
            self.iou_match_thr = iou

    def configure_classes(self, values):
        with self._lock:
            self.class_enabled = dict(values)

    def configure_polys(self, rois, masks):
        with self._lock:
            self.rois = rois
            self.masks = masks

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

    def set_source(self, source_obj, source_id):
        with self._lock:
            self._source = source_obj
            self.current_source_id = source_id
            self._want_open = True

    def stop_source(self):
        with self._lock:
            self._source = None
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

    def _update_tracks(self, detections):
        for track in self.tracks:
            track["updated"] = False
            track["miss"] += 1

        for detection in detections:
            best_iou = 0.0
            best_index = -1
            for index, track in enumerate(self.tracks):
                if track["cls"] != detection["cls"]:
                    continue
                overlap = self._box_iou(track["box"], detection["box"])
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = index
            if best_index >= 0 and best_iou >= self.iou_match_thr:
                track = self.tracks[best_index]
                track["box"] = self._smooth(track["box"], detection["box"], self.smooth_alpha)
                track["conf"] = max(track["conf"], detection["conf"])
                track["hits"] += 1
                track["miss"] = 0
                track["updated"] = True
                track["model_ids"] = tuple(dict.fromkeys(track["model_ids"] + detection["model_ids"]))
            else:
                self.tracks.append({
                    "id": self.track_next_id,
                    "cls": detection["cls"],
                    "box": detection["box"],
                    "conf": detection["conf"],
                    "model_ids": detection["model_ids"],
                    "hits": 1,
                    "miss": 0,
                    "updated": True,
                })
                self.track_next_id += 1
        self.tracks = [track for track in self.tracks if track["miss"] <= self.max_miss]

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
            if want_open:
                self._close_cap()
                with self._lock:
                    self._want_open = False
                self._reset_temporal_state()
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

            self._fps_counter += 1
            now_time = time.time()
            if now_time - self._fps_t0 >= 1.0:
                self.fpsReady.emit(self._fps_counter / (now_time - self._fps_t0))
                self._fps_t0 = now_time
                self._fps_counter = 0

            overlay = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_count += 1
            self._watermark(overlay)
            do_infer = self.frame_count % max(1, DETECT_EVERY_N) == 0

            if do_infer and self.runtime is not None and self.active_model_ids:
                try:
                    raw_detections = self.runtime.infer(overlay, conf, iou, INFER_IMG_SIZE)
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
                    detections = []
                    for detection in raw_detections:
                        if not class_enabled.get(detection.class_id, True):
                            continue
                        x1, y1, x2, y2 = detection.box
                        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                        if any(point_in_polygon(center, mask) for mask in masks):
                            continue
                        if rois and not any(point_in_polygon(center, roi) for roi in rois):
                            continue
                        detections.append({
                            "cls": detection.class_id,
                            "conf": detection.confidence,
                            "box": detection.box,
                            "model_ids": detection.source_model_ids,
                        })
                    self._update_tracks(detections)

                    present_now = set()
                    present_conf = {}
                    present_models: Dict[Tuple[str, str], Tuple[str, ...]] = {}
                    source_id = self.current_source_id or "unknown"
                    for track in self.tracks:
                        if track["hits"] < self.min_hits and track["miss"] == 0:
                            continue
                        x1, y1, x2, y2 = map(int, track["box"])
                        profile = snapshot.class_profiles.get(track["cls"], class_profile(track["cls"]))
                        color = profile.color_rgb
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                        label = "%s %.2f" % (profile.display_en, track["conf"])
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
                        if profile.alert_enabled and profile.severity:
                            key = (source_id, track["cls"])
                            present_now.add(key)
                            if track["conf"] > present_conf.get(key, 0.0):
                                present_conf[key] = float(track["conf"])
                                present_models[key] = tuple(track["model_ids"])

                    now_seconds = time.time()
                    for class_id, profile in snapshot.class_profiles.items():
                        if not profile.alert_enabled or not profile.severity:
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
                            self.newAlert.emit({
                                "id": str(_next_alert_id()),
                                "ts": now_seconds,
                                "source_id": source_id,
                                "cls": class_id,
                                "conf": float(present_conf.get(key, 0.0)),
                                "severity": profile.severity,
                                "snapshot": None,
                                "model_ids": list(present_models.get(key, ())),
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
        self.setMinimumSize(520, 292)
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
        self.reduce_motion = REDUCE_MOTION
        self._ripples=[]; self._timer=QtCore.QTimer(self)
        self._timer.timeout.connect(self.update); self._timer.start(16)

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

        if not self.reduce_motion:
            self._ripples.append((QtCore.QPointF(e.position()), time.time()))

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
        t = P(); p.fillRect(self.rect(), qcolor(t["bg2"]))
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

        if not self.reduce_motion and self._ripples:
            now = time.time(); keep=[]
            for pos, t0 in self._ripples:
                dt = now - t0
                if dt>0.6: continue
                r = 60*dt/0.6; alpha = max(0, int(140*(1 - dt/0.6)))
                pen = QtGui.QPen(qcolor(P()["accentSoft"], alpha)); pen.setWidth(2)
                p.setPen(pen); p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                p.drawEllipse(pos, r, r)
                keep.append((pos,t0))
            self._ripples = keep
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
            timestamp = time.strftime('%Y-%m-%d  %H:%M:%S', time.localtime(a.ts))
            return f"{cls_disp}  ·  {a.conf:.0%}\n{timestamp}  ·  {a.source_id}  ·  {model_tag}"
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
        self.setWindowTitle(APP_NAME)
        self.resize(1400, 900)

        # 状态
        self.lang_code = "zh"   # zh / en
        self.is_viewer = False
        self.active_source_id = None
        self.catalog_snapshot = CATALOG_SNAPSHOT
        self.settings_path = PROJECT_ROOT / ".runtime" / "config" / "sentinel_settings.json"
        self.saved_settings = self._load_settings()
        self.selected_model_ids = self._initial_model_selection()
        self._closing_after_worker = False

        # 顶栏
        self.toolbar = QtWidgets.QToolBar(); self.toolbar.setMovable(False)
        self.toolbar.setObjectName("AppBar")
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self.toolbar)
        self.brand_mark = QtWidgets.QLabel("SV")
        self.brand_mark.setProperty("badge", "primary")
        self.brand_mark.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.brand_mark.setFixedSize(40, 30)
        self.title = QtWidgets.QLabel(APP_NAME)
        self.title.setProperty("textRole", "brand")
        self.sub   = QtWidgets.QLabel(SUBTITLE)
        self.sub.setProperty("textRole", "muted")
        self.lbl_fps = QtWidgets.QLabel("FPS —")
        self.lbl_fps.setProperty("badge", "neutral")
        self.lbl_fps.setMinimumWidth(76)
        self.lbl_fps.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.btn_theme = QBtn("主题"); self.btn_theme.clicked.connect(self.toggle_theme)
        self.btn_rm    = QBtn("动效"); self.btn_rm.clicked.connect(self.toggle_reduce_motion)
        self.btn_theme.setProperty("variant", "ghost")
        self.btn_rm.setProperty("variant", "ghost")
        self.lang = ThemedComboBox(); self.lang.addItems(["zh-CN","en-NZ"])
        self.role = ThemedComboBox(); self.role.addItems(["Viewer","Operator","Admin"]); self.role.setCurrentText(DEFAULT_ROLE)
        self.role.currentTextChanged.connect(self.apply_role)
        self.lang.currentTextChanged.connect(self.on_language_change)

        self.toolbar.addWidget(self.brand_mark); self.toolbar.addWidget(self.title); self.toolbar.addWidget(self.sub)
        self.toolbar.addWidget(self._spacer())
        self.toolbar.addWidget(self.lbl_fps)
        for w in (self.btn_theme, self.btn_rm, self.lang, self.role): self.toolbar.addWidget(w)

        # 中心布局
        central = QtWidgets.QWidget()
        central.setObjectName("AppShell")
        central_layout = QtWidgets.QVBoxLayout(central)
        central_layout.setContentsMargins(16,16,16,16)
        central_layout.setSpacing(0)
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)

        # 左：监控区
        left_card = self._card(); L = QtWidgets.QGridLayout(left_card); L.setContentsMargins(12,12,12,12)
        L.setHorizontalSpacing(8); L.setVerticalSpacing(9)
        monitor_header = QtWidgets.QHBoxLayout()
        self.monitor_title = QtWidgets.QLabel("实时监控")
        self.monitor_title.setProperty("textRole", "sectionTitle")
        self.live_badge = QtWidgets.QLabel("●  STANDBY")
        self.live_badge.setProperty("badge", "neutral")
        self.source_state = QtWidgets.QLabel("等待选择视频源")
        self.source_state.setProperty("textRole", "muted")
        monitor_header.addWidget(self.monitor_title)
        monitor_header.addWidget(self.live_badge)
        monitor_header.addStretch(1)
        monitor_header.addWidget(self.source_state)
        L.addLayout(monitor_header, 0, 0, 1, 5)
        self.canvas = VideoCanvas(); L.addWidget(self.canvas, 1, 0, 1, 5)
        self.btn_play  = QBtn("播放"); self.btn_pause = QBtn("暂停"); self.btn_stop  = QBtn("停止")
        self.btn_play.clicked.connect(lambda: self.worker.set_paused(False))
        self.btn_pause.clicked.connect(lambda: self.worker.set_paused(True))
        self.btn_stop.clicked.connect(self.stop_source)
        self.src_quick = ThemedComboBox(); self.src_quick.setMinimumWidth(260)
        self.btn_play.setProperty("variant", "primary")
        self.btn_stop.setProperty("variant", "danger")
        L.addWidget(self.btn_play,  2, 1); L.addWidget(self.btn_pause, 2, 2); L.addWidget(self.btn_stop,  2, 3); L.addWidget(self.src_quick, 2, 4)

        self.btn_add_roi   = QBtn();  self.btn_add_mask = QBtn(); self.btn_edit_poly = QBtn(); self.btn_clear_poly= QBtn()
        self.btn_add_roi.clicked.connect(lambda:self._set_edit_mode("add-roi"))
        self.btn_add_mask.clicked.connect(lambda:self._set_edit_mode("add-mask"))
        self.btn_edit_poly.clicked.connect(lambda:self._set_edit_mode("edit"))
        self.btn_clear_poly.clicked.connect(self._clear_polys)
        L.addWidget(self.btn_add_roi,  3, 1); L.addWidget(self.btn_add_mask, 3, 2); L.addWidget(self.btn_edit_poly, 3, 3); L.addWidget(self.btn_clear_poly, 3, 4)

        # 右：源/阈值/模式/时间轴
        right_col = QtWidgets.QVBoxLayout(); right_col.setSpacing(10)

        # 源管理
        src_card = self._card(); S = QtWidgets.QGridLayout(src_card); S.setContentsMargins(12,12,12,12)
        S.setHorizontalSpacing(8); S.setVerticalSpacing(8)
        self.source_title = QtWidgets.QLabel("视频源")
        self.source_title.setProperty("textRole", "sectionTitle")
        self.btn_file = QBtn(); self.btn_file.clicked.connect(self.pick_file)
        self.txt_url  = QtWidgets.QLineEdit()
        self.btn_load = QBtn(); self.btn_load.clicked.connect(self.load_url)
        self.btn_webcam = QBtn(); self.btn_webcam.clicked.connect(lambda:self.set_source(0, "webcam"))
        self.btn_demo1  = QBtn(); self.btn_demo1.clicked.connect(lambda:self.set_source("https://media.istockphoto.com/id/1272087364/video/4k-firefighters-extinguish-a-fire-in-oil-refinery-plant.mp4?s=mp4-640x640-is&k=20&c=INonYLBCKuPgbTXfs-eSe4JlhuGEAVykvm10GaVnO6E=","http"))
        self.btn_demo2  = QBtn(); self.btn_demo2.clicked.connect(lambda:self.set_source("https://media.istockphoto.com/id/615753036/video/factory-worker-in-blue-uniform-is-putting-his-hard-hat-and-goggles-on-while-walking.mp4?s=mp4-640x640-is&k=20&c=ghY3U2O5e4eKoMoY-Oh8pscFBGUACy2HZAKTW99XNKM=","http"))
        S.addWidget(self.source_title, 0, 0, 1, 2)
        S.addWidget(self.btn_file, 1, 0, 1, 2)
        S.addWidget(self.txt_url,  2, 0, 1, 1); S.addWidget(self.btn_load, 2, 1, 1, 1)
        S.addWidget(self.btn_webcam, 3, 0, 1, 1); S.addWidget(self.btn_demo1, 3, 1, 1, 1)
        S.addWidget(self.btn_demo2, 4, 0, 1, 1)

        # 阈值、动态模型、动态类别
        cfg_card = self._card(); C = QtWidgets.QGridLayout(cfg_card); C.setContentsMargins(12,12,12,12)
        self.lbl_model = QtWidgets.QLabel()
        self.lbl_model_summary = QtWidgets.QLabel()
        self.lbl_model_summary.setObjectName("MutedLabel")
        self.btn_rescan_models = QBtn("重新扫描 RESULTS")
        self.btn_rescan_models.clicked.connect(self._rescan_models)
        self.s_conf = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.s_conf.setRange(5,95)
        default_conf = 25
        default_iou  = 45
        self.s_conf.setValue(default_conf)
        self.v_conf = QtWidgets.QDoubleSpinBox(); self.v_conf.setRange(0.05,0.95); self.v_conf.setDecimals(2); self.v_conf.setSingleStep(0.01); self.v_conf.setValue(self.s_conf.value()/100)
        self.s_iou  = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.s_iou.setRange(10,90)
        self.s_iou.setValue(default_iou)
        self.v_iou  = QtWidgets.QDoubleSpinBox(); self.v_iou.setRange(0.10,0.90); self.v_iou.setDecimals(2); self.v_iou.setSingleStep(0.01); self.v_iou.setValue(self.s_iou.value()/100)
        self.s_conf.valueChanged.connect(lambda v:self.v_conf.setValue(v/100.0))
        self.v_conf.valueChanged.connect(lambda x:self.s_conf.setValue(int(round(x*100))))
        self.s_iou.valueChanged.connect(lambda v:self.v_iou.setValue(v/100.0))
        self.v_iou.valueChanged.connect(lambda x:self.s_iou.setValue(int(round(x*100))))
        self.v_conf.valueChanged.connect(self._apply_thresholds); self.v_iou.valueChanged.connect(self._apply_thresholds)

        row=0
        C.addWidget(self.lbl_model, row,0,1,2)
        C.addWidget(self.btn_rescan_models, row,2,1,1); row+=1
        C.addWidget(self.lbl_model_summary, row,0,1,3); row+=1
        self.model_list_widget = QtWidgets.QWidget()
        self.model_list_layout = QtWidgets.QVBoxLayout(self.model_list_widget)
        self.model_list_layout.setContentsMargins(0,0,0,0)
        self.model_list_layout.setSpacing(5)
        self.model_scroll = QtWidgets.QScrollArea()
        self.model_scroll.setWidgetResizable(True)
        self.model_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.model_scroll.setMinimumHeight(125)
        self.model_scroll.setMaximumHeight(180)
        self.model_scroll.setWidget(self.model_list_widget)
        C.addWidget(self.model_scroll, row,0,1,3); row+=1
        self.lbl_conf = QtWidgets.QLabel(); C.addWidget(self.lbl_conf, row,0); C.addWidget(self.s_conf, row,1); C.addWidget(self.v_conf, row,2); row+=1
        self.lbl_iou  = QtWidgets.QLabel(); C.addWidget(self.lbl_iou,  row,0); C.addWidget(self.s_iou,  row,1); C.addWidget(self.v_iou,  row,2); row+=1
        C.addWidget(self._hline(), row,0,1,3); row+=1
        self.lbl_classes = QtWidgets.QLabel()
        C.addWidget(self.lbl_classes, row,0,1,3); row+=1
        self.class_list_widget = QtWidgets.QWidget()
        self.class_list_layout = QtWidgets.QGridLayout(self.class_list_widget)
        self.class_list_layout.setContentsMargins(0,0,0,0)
        self.class_scroll = QtWidgets.QScrollArea()
        self.class_scroll.setWidgetResizable(True)
        self.class_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.class_scroll.setMaximumHeight(100)
        self.class_scroll.setWidget(self.class_list_widget)
        C.addWidget(self.class_scroll, row,0,1,3); row+=1
        self.model_checks = {}
        self.model_status_labels = {}
        self.class_checks = {}
        self._rebuild_model_controls()
        self._rebuild_class_controls()

        # 告警时间轴
        timeline_card = self._card(); T = QtWidgets.QGridLayout(timeline_card); T.setContentsMargins(12,12,12,12)
        self.timeline_title = QtWidgets.QLabel()
        self.alerts_model = AlertsModel(lambda: self.lang_code)
        self.list_alerts = QtWidgets.QListView(); self.list_alerts.setModel(self.alerts_model)
        self.list_alerts.setWordWrap(True)
        self.list_alerts.setSpacing(3)
        self.list_alerts.setMinimumHeight(190)
        self.list_alerts.clicked.connect(self._open_alert_detail)
        self.btn_clear_alerts = QBtn(); self.btn_clear_alerts.clicked.connect(self._clear_alerts)
        self.btn_export_csv   = QBtn(); self.btn_export_json  = QBtn()
        self.btn_export_csv.clicked.connect(self._export_csv); self.btn_export_json.clicked.connect(self._export_json)
        T.addWidget(self.timeline_title, 0,0,1,3)
        T.addWidget(self.list_alerts, 1,0,1,3)
        T.addWidget(self.btn_clear_alerts, 2,0); T.addWidget(self.btn_export_csv, 2,1); T.addWidget(self.btn_export_json, 2,2)

        # 右列组合：整体可滚动，兼容 768p 和 Windows 高 DPI。
        right_col.addWidget(src_card); right_col.addWidget(cfg_card); right_col.addWidget(timeline_card)
        right_col.addStretch(1)
        containerR = QtWidgets.QWidget(); containerR.setLayout(right_col)
        containerR.setMinimumWidth(310)
        right_scroll = QtWidgets.QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setWidget(containerR)
        right_scroll.setMinimumWidth(330)
        right_scroll.setMaximumWidth(460)
        self.main_splitter.addWidget(left_card)
        self.main_splitter.addWidget(right_scroll)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 0)
        self.main_splitter.setSizes([980, 400])
        central_layout.addWidget(self.main_splitter)
        self.setCentralWidget(central)

        # 抽屉
        self.drawer = QtWidgets.QFrame(self)
        self.drawer.setGeometry(self.width(), 0, 0, self.height()); self.drawer.raise_()
        self.drawer_open = False
        self.drawer_anim = QtCore.QPropertyAnimation(self.drawer, b"geometry"); self.drawer_anim.setDuration(240)
        self.drawer_anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self.drawer_layout = QtWidgets.QVBoxLayout(self.drawer); self.drawer_layout.setContentsMargins(12,12,12,12)
        self.drawer_title = QtWidgets.QLabel()
        self.btn_close_drawer = QBtn("关闭详情"); self.btn_close_drawer.clicked.connect(self._close_drawer)
        self.drawer_info  = QtWidgets.QTextEdit(); self.drawer_info.setReadOnly(True)
        self.btn_mark_ok  = QBtn(); self.btn_mark_ng  = QBtn(); self.btn_snapshot = QBtn("保存截图")
        self.btn_mark_ok.clicked.connect(lambda: self._mark_current_alert("valid"))
        self.btn_mark_ng.clicked.connect(lambda: self._mark_current_alert("false_alarm"))
        self.btn_snapshot.clicked.connect(self._save_current_alert_snapshot)
        self.current_alert = None
        self.last_qimage = None
        drawer_header = QtWidgets.QHBoxLayout()
        drawer_header.addWidget(self.drawer_title); drawer_header.addStretch(1); drawer_header.addWidget(self.btn_close_drawer)
        self.drawer_layout.addLayout(drawer_header); self.drawer_layout.addWidget(self.drawer_info)
        row_b = QtWidgets.QHBoxLayout(); row_b.addWidget(self.btn_mark_ok); row_b.addWidget(self.btn_mark_ng); row_b.addWidget(self.btn_snapshot)
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
        self.worker = VideoWorker(self.catalog_snapshot)
        self.worker.frameReady.connect(self._on_frame)
        self.worker.statusMsg.connect(self.statusBar().showMessage)
        self.worker.fpsReady.connect(lambda f:self.lbl_fps.setText(f"FPS {f:.1f}"))
        self.worker.newAlert.connect(self._on_new_alert)
        self.worker.modelStatus.connect(self._on_model_status)
        self.worker.modelsApplied.connect(self._on_models_applied)
        self.worker.configure_models(self.selected_model_ids)
        self.worker.configure_classes(self._class_selection_map())
        self.worker.start()

        # 现在再连接源切换信号，并主动切一次当前源
        self.src_quick.currentIndexChanged.connect(self._on_quick_switch)

        # 初始化主题 & 文案 & 权限
        self.apply_theme()
        self.apply_i18n()
        self.apply_role(self.role.currentText())

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
        document = {
            "schema_version": 2,
            "selected_model_ids": list(self.selected_model_ids),
            "class_enabled": preserved_class_values,
        }
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
            status = QtWidgets.QLabel(
                "可用" if spec.status == "ready" else "新发现" if spec.status == "new" else "不可用"
            )
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
        self.lbl_model_summary.setText(
            "已选 %d 个 · 已加载 %d 个 · %d 个目标 · %s"
            % (selected_count, active_count, class_count, device_name)
        )

    def _on_model_status(self, model_id, state, message):
        label = self.model_status_labels.get(model_id)
        if label is None:
            return
        names = {"loading": "加载中", "active": "已启用", "inactive": "可用", "failed": "加载失败"}
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
        t = P()
        return build_app_stylesheet(dark=DARK) + f"""
        QFrame#CardFrame {{ background:{t['card']}; border:1px solid {t['line']}; border-radius:14px; }}
        QFrame#ModelRow {{ background:{t['bg2']}; border:1px solid {t['line']}; border-radius:9px; }}
        QFrame#Separator {{ background:{t['line']}; min-height:1px; max-height:1px; border:0; }}
        QWidget#VideoCanvas {{ background:#02060D; border:1px solid {t['line']}; border-radius:10px; }}
        QLabel#MutedLabel {{ color:{t['text2']}; font-size:12px; }}
        QListView::item {{ border-bottom:1px solid {t['line']}; padding:7px 9px; }}
        QListView::item:selected {{ border-left:3px solid {t['primary']}; }}
        """
    def _card(self):
        w = QtWidgets.QFrame(); w.setObjectName("CardFrame"); return w
    def _hline(self):
        line = QtWidgets.QFrame()
        line.setObjectName("Separator")
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        return line
    def _spacer(self):
        s = QtWidgets.QWidget(); s.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding); return s

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
        self.btn_theme.setText("主题" if is_zh else "Theme")
        self.btn_rm.setText("动效" if is_zh else "Motion")
        self.btn_play.setText("播放" if is_zh else "Play")
        self.btn_pause.setText("暂停" if is_zh else "Pause")
        self.btn_stop.setText("停止" if is_zh else "Stop")
        self.btn_add_roi.setText(self.L('add_roi'))
        self.btn_add_mask.setText(self.L('add_mask'))
        self.btn_edit_poly.setText(self.L('edit'))
        self.btn_clear_poly.setText(self.L('reselect'))
        self.btn_file.setText(self.L('pick_file'))
        self.txt_url.setPlaceholderText(self.L('url_ph'))
        self.btn_load.setText(self.L('load'))
        self.btn_webcam.setText(self.L('webcam'))
        self.btn_demo1.setText(self.L('demo1'))
        self.btn_demo2.setText(self.L('demo2'))
        self.monitor_title.setText("实时监控" if is_zh else "Live monitoring")
        self.source_title.setText("视频源" if is_zh else "Video source")
        if not self.active_source_id:
            self.source_state.setText("等待选择视频源" if is_zh else "Waiting for a video source")
        self.lbl_model.setText("检测模型" if is_zh else "Detection models")
        self.lbl_classes.setText("识别目标" if self.lang_code == "zh" else "Detection targets")
        self.btn_rescan_models.setText("重新扫描 RESULTS" if self.lang_code == "zh" else "Rescan RESULTS")
        self.lbl_conf.setText(self.L('conf'))
        self.lbl_iou.setText(self.L('iou'))
        self.timeline_title.setText(self.L('timeline'))
        self.btn_clear_alerts.setText(self.L('clear_alerts'))
        self.btn_export_csv.setText(self.L('export_csv'))
        self.btn_export_json.setText(self.L('export_json'))
        self.drawer_title.setText(self.L('alert_detail'))
        self.btn_mark_ok.setText("标注为有效" if is_zh else "Mark Valid")
        self.btn_mark_ng.setText("标注为误报" if is_zh else "Mark False")
        self.btn_snapshot.setText("保存截图" if is_zh else "Save Snapshot")
        self.btn_close_drawer.setText("关闭详情" if is_zh else "Close")
        self._rebuild_class_controls()
        self._update_model_summary()

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
            *self.model_checks.values(), *self.class_checks.values(),
            self.list_alerts, self.btn_clear_alerts, self.btn_export_csv, self.btn_export_json,
            self.btn_mark_ok, self.btn_mark_ng, self.btn_snapshot
        ]
        for w in widgets: w.setEnabled(allow)
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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not hasattr(self, "drawer"):
            return
        self.drawer_anim.stop()
        if self.drawer_open:
            drawer_width = min(560, max(360, self.width() - 40))
            self.drawer.setGeometry(self.width() - drawer_width, 0, drawer_width, self.height())
        else:
            self.drawer.setGeometry(self.width(), 0, 0, self.height())

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
        self.worker.set_source(src, sid)
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
    def _apply_thresholds(self):
        conf = float(self.v_conf.value()); iou = float(self.v_iou.value())
        self.worker.configure_thresholds(conf, iou)
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
        self.drawer_info.setPlainText(json.dumps(info, indent=2, ensure_ascii=False))
        self._open_drawer()

    def _mark_current_alert(self, conclusion):
        if self.current_alert is None or self.current_alert not in self.alerts_model.items:
            self.current_alert = None
            self.statusBar().showMessage(self.L("no_alert"))
            return
        self.current_alert.review = conclusion
        self.statusBar().showMessage("告警人工结论已记录：%s" % conclusion)
        index = self.alerts_model.index(self.alerts_model.items.index(self.current_alert), 0)
        self._open_alert_detail(index)

    def _clear_alerts(self):
        self.alerts_model.clear()
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
    def _open_drawer(self):
        drawer_width = min(560, max(360, self.width() - 40))
        self.drawer.setStyleSheet(f"background:{P()['bg2']}; border-left:1px solid {P()['line']}; color:{P()['text']};")
        start = self.drawer.geometry()
        end = QtCore.QRect(self.width() - drawer_width, 0, drawer_width, self.height())
        self.drawer_open = True
        self.drawer.raise_()
        self.drawer_anim.stop()
        if REDUCE_MOTION:
            self.drawer.setGeometry(end)
        else:
            self.drawer_anim.setStartValue(start)
            self.drawer_anim.setEndValue(end)
            self.drawer_anim.start()

    def _close_drawer(self):
        start = self.drawer.geometry()
        end = QtCore.QRect(self.width(), 0, 0, self.height())
        self.drawer_open = False
        self.drawer_anim.stop()
        if REDUCE_MOTION:
            self.drawer.setGeometry(end)
        else:
            self.drawer_anim.setStartValue(start)
            self.drawer_anim.setEndValue(end)
            self.drawer_anim.start()

    # ---------- 导出 ----------
    def _export_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, self.L('export_csv'), "alerts.csv", "CSV (*.csv)")
        if not path: return
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "ts", "iso", "source", "class_id", "class_name", "confidence", "severity", "models", "review", "snapshot"])
            for a in self.alerts_model.items:
                writer.writerow([
                    a.id, a.ts, time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(a.ts)),
                    a.source_id, a.cls, class_display_name(a.cls, self.lang_code), "%.4f" % a.conf,
                    getattr(a, "severity", "info"), ";".join(getattr(a, "model_ids", ()) or ()),
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
    def toggle_reduce_motion(self):
        global REDUCE_MOTION; REDUCE_MOTION = not REDUCE_MOTION
        self.canvas.reduce_motion = REDUCE_MOTION
        self.statusBar().showMessage(f"Motion {'Off' if REDUCE_MOTION else 'On'}")
    def apply_theme(self):
        self.setStyleSheet(self._build_stylesheet())
        self.drawer.setStyleSheet(f"background:{P()['bg2']}; border-left:1px solid {P()['line']}; color:{P()['text']};")
        for combo in (self.lang, self.role, self.src_quick): combo.apply_theme()
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
