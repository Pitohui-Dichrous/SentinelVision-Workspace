# -*- coding: utf-8 -*-
"""
YOLOv5 Desktop UI (PySide6) – Fire CCTV
- 本地窗口，无需浏览器
- 切源BUG修复：切换源时释放旧cap并重连；提供“停止/释放源”
- 阈值滑块：实时作用于 model.conf / model.iou，并显示数值
- 画面自适应任意分辨率；点击坐标正确映射到原始帧
"""

import sys, os, time, math, threading
import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import torch

from project_paths import LEGACY_MODEL_PATH, YOLO_REPO_DIR

# ===================== 你的模型与目录 =====================
MODEL_PATH = LEGACY_MODEL_PATH
INFER_IMG_SIZE = 640                     # 推理尺寸
DETECT_EVERY_N = 2                       # 每N帧推一次

# --------------------- 加载模型 --------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    model = torch.hub.load(
        repo_or_dir=str(YOLO_REPO_DIR),
        model='custom',
        path=str(MODEL_PATH),
        source='local'
    ).to(device).eval()
    model.conf = 0.25
    model.iou  = 0.45
    MODEL_OK = True
except Exception as e:
    print("模型加载失败：", e)
    model = None
    MODEL_OK = False

# ===================== 工具函数 =====================
def qimage_from_rgb(rgb: np.ndarray) -> QtGui.QImage:
    h, w, ch = rgb.shape
    return QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format.Format_RGB888).copy()

def point_in_polygon(pt, poly):
    x, y = pt; inside = False; n = len(poly)
    for i in range(n):
        x1,y1 = poly[i]; x2,y2 = poly[(i+1)%n]
        if ((y1>y)!=(y2>y)) and (x < (x2-x1)*(y-y1)/(y2-y1+1e-9) + x1):
            inside = not inside
    return inside

def draw_glow_polygon(img: np.ndarray, pts, phase=0.0):
    if len(pts) < 3: return img
    overlay = img.copy()
    poly = np.array(pts, dtype=np.int32)
    cv2.fillPoly(overlay, [poly], color=(56,189,248))  # 浅蓝填充
    img = cv2.addWeighted(overlay, 0.12, img, 0.88, 0)
    cv2.polylines(img, [poly], True, (59,130,246), 3, cv2.LINE_AA)
    return img

# ===================== 推理线程 =====================
class VideoDetector(QtCore.QThread):
    frameReady = QtCore.Signal(object, int, int)  # QImage, w, h（原始帧尺寸）
    statReady  = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._source = None           # 当前目标源（路径/rtsp/int）
        self._current_src = None      # 实际已打开的源
        self._source_dirty = False    # 需要重开
        self._paused = False
        self._stop_flag = False
        self.cap = None
        self.frame_count = 0
        self.roi = []                 # 原始帧坐标系中的点 [(x,y)*4]
        self.phase = 0.0

    # --- 外部接口 ---
    def setSource(self, src):
        with self._lock:
            self._source = src
            self._source_dirty = True   # 标记重开
        self.statReady.emit(f"切换视频源：{src}")

    def stopSource(self):
        with self._lock:
            self._source = None
            self._source_dirty = True   # 强制关闭
        self.statReady.emit("已停止并释放视频源")

    def setPaused(self, flag: bool):
        with self._lock:
            self._paused = flag
        self.statReady.emit("暂停" if flag else "继续")

    def setROI(self, pts):
        with self._lock:
            self.roi = [(int(x),int(y)) for (x,y) in pts][:4]

    def stop(self):
        self._stop_flag = True

    # --- 内部 ---
    def _open_cap(self, src):
        if isinstance(src, int):
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(src)
        return cap if cap.isOpened() else None

    def _close_cap(self):
        if self.cap is not None:
            try: self.cap.release()
            except: pass
            self.cap = None

    def run(self):
        global model, MODEL_OK
        while not self._stop_flag:
            with self._lock:
                need_reopen = self._source_dirty
                paused       = self._paused
                target_src   = self._source

            # 源变更或停止 → 释放并按需重开
            if need_reopen:
                self._close_cap()
                self._current_src = None
                with self._lock: self._source_dirty = False
                if target_src is not None:
                    self.cap = self._open_cap(target_src)
                    if self.cap is not None:
                        self._current_src = target_src
                        self.statReady.emit("视频源已打开")
                    else:
                        self.statReady.emit("无法打开视频源")
                        time.sleep(0.5)
                        continue
                else:
                    time.sleep(0.1)
                    continue

            if self.cap is None:
                time.sleep(0.05)
                continue

            if paused:
                time.sleep(0.03)
                continue

            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.statReady.emit("读取帧失败，尝试重连…")
                self._close_cap()
                self._current_src = None
                time.sleep(0.3)
                # 交给上面的 need_reopen 流程在下次迭代处理
                with self._lock: self._source_dirty = True
                continue

            h0, w0 = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            overlay = rgb

            # 每 N 帧推一次
            self.frame_count += 1
            boxes, names = [], []
            if MODEL_OK and model is not None and self.frame_count % max(1, DETECT_EVERY_N) == 0:
                try:
                    with torch.no_grad():
                        results = model(overlay, size=INFER_IMG_SIZE)
                    try:
                        det = results.xyxy[0].cpu().numpy()
                        cls_names = results.names if hasattr(results, "names") else model.names
                    except Exception:
                        det = results.pandas().xyxy[0].to_numpy()
                        cls_names = results.names if hasattr(results, "names") else model.names
                    with self._lock:
                        roi = self.roi.copy()
                    for row in det:
                        x1,y1,x2,y2,conf,cls = row[:6]
                        cx, cy = (x1+x2)/2, (y1+y2)/2
                        show = True
                        if len(roi) == 4:
                            show = point_in_polygon((cx,cy), roi)
                        if show:
                            boxes.append((int(x1),int(y1),int(x2),int(y2),float(conf),int(cls)))
                            try: names.append(cls_names[int(cls)])
                            except: names.append(str(int(cls)))
                except Exception as e:
                    self.statReady.emit(f"推理异常：{e}")

            # 画框
            for i,(x1,y1,x2,y2,conf,cls) in enumerate(boxes):
                color = (255,80,80)  # 火焰红
                cv2.rectangle(overlay,(x1,y1),(x2,y2),color,2)
                label = f"{names[i] if i < len(names) else cls} {conf:.2f}"
                (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(overlay,(x1,y1-th-6),(x1+tw+6,y1),color,-1)
                cv2.putText(overlay,label,(x1+3,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1,cv2.LINE_AA)

            # 画ROI
            with self._lock: roi_pts = self.roi.copy()
            for (px,py) in roi_pts:
                cv2.circle(overlay,(px,py),6,(14,165,233),-1)
                cv2.circle(overlay,(px,py),10,(56,189,248),2)
            if len(roi_pts)==4:
                overlay = draw_glow_polygon(overlay, roi_pts)

            qimg = qimage_from_rgb(overlay)
            self.frameReady.emit(qimg, w0, h0)

            time.sleep(0.005)

        self._close_cap()

# ===================== 可缩放视频Label =====================
class VideoLabel(QtWidgets.QLabel):
    clicked = QtCore.Signal(int, int)  # 原始帧坐标

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 360)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self._qimg = None
        self._src_w = 0
        self._src_h = 0
        self.ripples = []  # (x_disp, y_disp, t0)

        # 重绘定时器用于涟漪
        self._timer = QtCore.QTimer(self); self._timer.timeout.connect(self.update); self._timer.start(16)

    def update_frame(self, qimg: QtGui.QImage, w, h):
        self._qimg = qimg
        self._src_w, self._src_h = w, h
        self.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        # 背景
        p.fillRect(self.rect(), QtGui.QColor(241,245,249))
        if self._qimg:
            # 等比缩放绘制
            label_w, label_h = self.width(), self.height()
            scale = min(label_w / self._src_w, label_h / self._src_h)
            nw, nh = int(self._src_w * scale), int(self._src_h * scale)
            ox, oy = (label_w - nw)//2, (label_h - nh)//2
            p.drawImage(QtCore.QRect(ox, oy, nw, nh), self._qimg)

        # 画点击涟漪
        now = time.time()
        keep = []
        for (x, y, t0) in self.ripples:
            dt = now - t0
            if dt <= 0.6:
                r = 60 * dt / 0.6
                alpha = max(0, int(140*(1 - dt/0.6)))
                pen = QtGui.QPen(QtGui.QColor(59,130,246,alpha)); pen.setWidth(2)
                p.setPen(pen); p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                p.drawEllipse(QtCore.QPointF(x,y), r, r)
                keep.append((x,y,t0))
        self.ripples = keep
        p.end()

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if not self._qimg: return
        label_w, label_h = self.width(), self.height()
        scale = min(label_w / self._src_w, label_h / self._src_h)
        nw, nh = int(self._src_w * scale), int(self._src_h * scale)
        ox, oy = (label_w - nw)//2, (label_h - nh)//2
        x_disp, y_disp = e.position().x(), e.position().y()
        self.ripples.append((x_disp, y_disp, time.time()))
        # 映射回原始坐标
        x = int((x_disp - ox) / scale)
        y = int((y_disp - oy) / scale)
        x = max(0, min(self._src_w-1, x))
        y = max(0, min(self._src_h-1, y))
        self.clicked.emit(x, y)

# ===================== 主窗口 =====================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv5 Project")
        self.resize(1200, 800)

        # 顶部UI
        header = QtWidgets.QWidget(); hlyt = QtWidgets.QVBoxLayout(header)
        title = QtWidgets.QLabel("YOLOv5 Project"); title.setAlignment(QtCore.Qt.AlignHCenter)
        title.setStyleSheet("font:900 28px 'Segoe UI'; color:#2563eb;")
        subtitle = QtWidgets.QLabel("See what you see   See what you see   See what you see")
        subtitle.setAlignment(QtCore.Qt.AlignHCenter); subtitle.setStyleSheet("color:#0284c7; font:600 13px 'Segoe UI';")
        names = QtWidgets.QLabel("制作：Bryan Park · Shibo Xu")
        names.setAlignment(QtCore.Qt.AlignHCenter)
        names.setStyleSheet("border:1px solid rgba(125,211,252,.6); border-radius:999px; padding:4px 10px; background:rgba(255,255,255,.7);")
        hlyt.addWidget(title); hlyt.addWidget(subtitle); hlyt.addWidget(names)

        # 中心画面 + 控件
        self.video = VideoLabel()

        play_btn  = QtWidgets.QPushButton("开始/继续")
        pause_btn = QtWidgets.QPushButton("暂停")
        stop_btn  = QtWidgets.QPushButton("停止/释放源")
        play_btn.clicked.connect(lambda: self.worker.setPaused(False))
        pause_btn.clicked.connect(lambda: self.worker.setPaused(True))
        stop_btn.clicked.connect(self.stop_source)

        # 源选择
        src_box = QtWidgets.QGroupBox("视频源"); src_grid = QtWidgets.QGridLayout(src_box)
        pick_btn = QtWidgets.QPushButton("选择本地视频"); pick_btn.clicked.connect(self.pick_file)
        self.url_edit = QtWidgets.QLineEdit(); self.url_edit.setPlaceholderText("RTSP / HTTP(s) 流地址")
        load_btn = QtWidgets.QPushButton("载入"); load_btn.clicked.connect(self.load_url)
        webcam_btn = QtWidgets.QPushButton("● Webcam"); webcam_btn.clicked.connect(lambda: self.set_source(0))
        demo1_btn  = QtWidgets.QPushButton("▶ 明火CCTV DEMO 1"); demo1_btn.clicked.connect(lambda: self.set_source("REPLACE_WITH_YOUR_FIRE_CCTV_DEMO_1.mp4"))
        demo2_btn  = QtWidgets.QPushButton("▶ 明火CCTV DEMO 2"); demo2_btn.clicked.connect(lambda: self.set_source("REPLACE_WITH_YOUR_FIRE_CCTV_DEMO_2.mp4"))
        src_grid.addWidget(pick_btn, 0, 0, 1, 2)
        src_grid.addWidget(QtWidgets.QLabel("流地址："), 1, 0)
        src_grid.addWidget(self.url_edit, 1, 1)
        src_grid.addWidget(load_btn, 1, 2)
        src_grid.addWidget(webcam_btn, 2, 0)
        src_grid.addWidget(demo1_btn, 2, 1)
        src_grid.addWidget(demo2_btn, 2, 2)

        # 阈值与模型信息
        cfg_box = QtWidgets.QGroupBox("模型与阈值"); cfg = QtWidgets.QGridLayout(cfg_box)
        self.lbl_model = QtWidgets.QLabel(f"MODEL_PATH:\n{MODEL_PATH}")
        self.lbl_repo  = QtWidgets.QLabel(f"YOLO_REPO_DIR:\n{YOLO_REPO_DIR}")
        self.s_conf = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.s_conf.setRange(5, 95); self.s_conf.setValue(int(getattr(model,'conf',0.25)*100) if MODEL_OK else 25)
        self.s_iou  = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.s_iou.setRange(10, 90); self.s_iou.setValue(int(getattr(model,'iou',0.45)*100) if MODEL_OK else 45)
        self.v_conf = QtWidgets.QDoubleSpinBox(); self.v_conf.setRange(0.05, 0.95); self.v_conf.setSingleStep(0.01); self.v_conf.setDecimals(2); self.v_conf.setValue(self.s_conf.value()/100.0)
        self.v_iou  = QtWidgets.QDoubleSpinBox(); self.v_iou.setRange(0.10, 0.90); self.v_iou.setSingleStep(0.01); self.v_iou.setDecimals(2); self.v_iou.setValue(self.s_iou.value()/100.0)
        # 双向绑定
        self.s_conf.valueChanged.connect(lambda v: self.v_conf.setValue(v/100.0))
        self.v_conf.valueChanged.connect(lambda x: self.s_conf.setValue(int(round(x*100))))
        self.s_iou.valueChanged.connect(lambda v: self.v_iou.setValue(v/100.0))
        self.v_iou.valueChanged.connect(lambda x: self.s_iou.setValue(int(round(x*100))))
        # 实时应用
        self.s_conf.valueChanged.connect(self.apply_thresholds)
        self.s_iou.valueChanged.connect(self.apply_thresholds)
        self.v_conf.valueChanged.connect(self.apply_thresholds)
        self.v_iou.valueChanged.connect(self.apply_thresholds)
        cfg.addWidget(self.lbl_model, 0, 0, 1, 2)
        cfg.addWidget(self.lbl_repo,  1, 0, 1, 2)
        cfg.addWidget(QtWidgets.QLabel("置信度阈值 (0.05 ↔ 0.95)"), 2, 0); cfg.addWidget(self.s_conf, 2, 1); cfg.addWidget(self.v_conf, 2, 2)
        cfg.addWidget(QtWidgets.QLabel("NMS IoU (0.10 ↔ 0.90)"),   3, 0); cfg.addWidget(self.s_iou,  3, 1); cfg.addWidget(self.v_iou,  3, 2)

        # 顶部按钮行
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(play_btn); btn_row.addWidget(pause_btn); btn_row.addWidget(stop_btn)

        # 主布局
        central = QtWidgets.QWidget(); layout = QtWidgets.QGridLayout(central)
        layout.addWidget(header, 0, 0, 1, 3)
        layout.addLayout(btn_row, 1, 0, 1, 3)
        layout.addWidget(self.video, 2, 0, 2, 2)
        layout.addWidget(src_box,  2, 2, 1, 1)
        layout.addWidget(cfg_box,  3, 2, 1, 1)
        self.setCentralWidget(central)

        # 线程
        self.worker = VideoDetector()
        self.worker.frameReady.connect(self.on_frame_ready)
        self.worker.statReady.connect(self.statusBar().showMessage)
        self.worker.start()

        # ROI
        self.roi_points = []
        self.video.clicked.connect(self.on_video_clicked)

        if not MODEL_OK:
            self.statusBar().showMessage("⚠ 模型加载失败，请检查 MODEL_PATH 与 YOLO_REPO_DIR")

    # --- 源控制 ---
    def pick_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mkv *.mov)")
        if path: self.set_source(path)

    def load_url(self):
        url = self.url_edit.text().strip()
        if url: self.set_source(url)

    def set_source(self, src):
        self.worker.setSource(src)
        self.clear_roi()  # 不同分辨率下ROI无效，切源时清空

    def stop_source(self):
        self.worker.stopSource()

    # --- ROI 控制 ---
    def on_video_clicked(self, x, y):
        if len(self.roi_points) < 4:
            self.roi_points.append((x, y))
            self.worker.setROI(self.roi_points)
            self.statusBar().showMessage(f"已选择点 P{len(self.roi_points)}: ({x},{y})")

    def clear_roi(self):
        self.roi_points = []
        self.worker.setROI(self.roi_points)
        self.statusBar().showMessage("已清空ROI点位")

    # --- 帧回调 & 阈值 ---
    def on_frame_ready(self, qimg: QtGui.QImage, w, h):
        self.video.update_frame(qimg, w, h)

    def apply_thresholds(self, *args):
        if model is None: return
        conf = round(self.v_conf.value(), 2)
        iou  = round(self.v_iou.value(), 2)
        model.conf = conf
        model.iou  = iou
        self.statusBar().showMessage(f"已应用阈值：conf={conf:.2f}, iou={iou:.2f}")

    # --- 清理 ---
    def closeEvent(self, e):
        try:
            self.worker.stop()
            self.worker.wait(800)
        except Exception:
            pass
        return super().closeEvent(e)

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
