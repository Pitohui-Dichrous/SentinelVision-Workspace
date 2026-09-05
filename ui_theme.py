"""Neutral, typography-led design system for the SentinelVision desktop apps."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets


# Keep the public token names compatible with the detection console.
def _palette(dark):
    base = {
        "bg": "#111113" if dark else "#F5F5F7",
        "surface": "#1C1C1E" if dark else "#FFFFFF",
        "surface_alt": "#262629" if dark else "#F5F5F7",
        "surface_hover": "#343438" if dark else "#E8E8ED",
        "border": "#424247" if dark else "#D2D2D7",
        "border_strong": "#86868B" if dark else "#86868B",
        "text": "#F5F5F7" if dark else "#1D1D1F",
        "muted": "#B8B8BE" if dark else "#626269",
        "subtle": "#AAAAAF" if dark else "#6E6E73",
        "primary": "#0066CC",
        "primary_hover": "#0055AA",
        "success": "#6ED69A" if dark else "#237A46",
        "warning": "#F0C46B" if dark else "#916000",
        "danger": "#FF858A" if dark else "#BC2731",
        "focus": "#8EC5FF" if dark else "#0066CC",
        "on_primary": "#FFFFFF",
        "primary_deep": "#0055AA",
        "primary_soft": "#173553" if dark else "#EAF3FF",
        "success_soft": "#18372A" if dark else "#EAF5EE",
        "warning_soft": "#3A3020" if dark else "#FFF5DF",
        "danger_soft": "#3B2225" if dark else "#FFF0F0",
    }
    for key, source in {
        "bg_gradient": "bg", "sidebar_glass": "surface", "glass": "surface",
        "glass_alt": "surface_alt", "glass_strong": "surface", "glass_hover": "surface_hover",
        "glass_pressed": "surface_alt", "hairline": "border", "hairline_strong": "border_strong",
        "cyan": "focus" if dark else "primary",
    }.items():
        base[key] = base[source]
    return base


DARK = _palette(True)
LIGHT = _palette(False)


def tokens(dark: bool = True):
    """Return the live theme token dictionary (legacy behavior)."""

    return DARK if dark else LIGHT


_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "reduce", "reduced"})


def reduce_motion_enabled() -> bool:
    """Honor the portable app's explicit reduced-motion environment switch."""

    value = os.environ.get("SENTINEL_REDUCE_MOTION", "")
    return value.strip().lower() in _TRUE_VALUES


def set_property(widget, name: str, value):
    """Set a dynamic property and immediately refresh its QSS selectors."""

    widget.setProperty(name, value)
    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)
    widget.update()
    return widget


def apply_glass_shadow(widget, elevated: bool = True):
    """Apply a restrained depth shadow that works with translucent Qt widgets."""

    effect = QtWidgets.QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(34 if elevated else 22)
    effect.setOffset(0, 10 if elevated else 6)
    effect.setColor(QtGui.QColor(0, 7, 16, 105 if elevated else 72))
    widget.setGraphicsEffect(effect)
    return widget


def make_card(parent=None, elevated: bool = False):
    frame = QtWidgets.QFrame(parent)
    frame.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
    frame.setProperty("card", True)
    frame.setProperty("glass", "strong" if elevated else "panel")
    frame.setProperty("elevated", elevated)
    return frame


def make_badge(text: str, tone: str = "neutral", parent=None):
    label = QtWidgets.QLabel(text, parent)
    label.setProperty("badge", tone)
    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
    return label


class PressableButton(QtWidgets.QPushButton):
    """QPushButton with pointer-down feedback and zero keyboard-trigger motion.

    The animation is deliberately limited to mouse/touch-generated mouse
    events.  Space/Enter activation therefore stays instant for expert users.
    Rapid reversals begin at the current presentation value, so the interaction
    remains interruptible.
    """

    PRESS_SCALE = 0.975
    PRESS_DURATION_MS = 140
    RELEASE_DURATION_MS = 90

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._visual_scale = 1.0
        self._mouse_press_active = False
        self._press_inside = False
        self._press_animation = QtCore.QPropertyAnimation(self, b"visualScale", self)
        self._press_animation.setDuration(self.PRESS_DURATION_MS)
        self._press_animation.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)

    def _get_visual_scale(self) -> float:
        return self._visual_scale

    def _set_visual_scale(self, value: float) -> None:
        self._visual_scale = float(value)
        self.update()

    visualScale = QtCore.Property(float, _get_visual_scale, _set_visual_scale)

    def _animate_scale(self, target: float, duration_ms: int) -> None:
        self._press_animation.stop()
        if reduce_motion_enabled():
            self._set_visual_scale(1.0)
            return
        if abs(self._visual_scale - target) < 0.0001:
            return
        self._press_animation.setDuration(duration_ms)
        self._press_animation.setStartValue(self._visual_scale)
        self._press_animation.setEndValue(target)
        self._press_animation.start()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.isEnabled():
            self._mouse_press_active = True
            self._press_inside = True
            self._animate_scale(self.PRESS_SCALE, self.PRESS_DURATION_MS)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._mouse_press_active:
            inside = self.rect().contains(event.position().toPoint())
            if inside != self._press_inside:
                self._press_inside = inside
                self._animate_scale(
                    self.PRESS_SCALE if inside else 1.0,
                    self.PRESS_DURATION_MS if inside else self.RELEASE_DURATION_MS,
                )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._mouse_press_active:
            self._mouse_press_active = False
            self._press_inside = False
            self._animate_scale(1.0, self.RELEASE_DURATION_MS)
        super().mouseReleaseEvent(event)

    def _reset_press(self):
        self._press_animation.stop()
        self._mouse_press_active = False
        self._press_inside = False
        self._set_visual_scale(1.0)

    def changeEvent(self, event):
        if event.type() == QtCore.QEvent.Type.EnabledChange and not self.isEnabled():
            self._reset_press()
        super().changeEvent(event)

    def hideEvent(self, event):
        self._reset_press()
        super().hideEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        if abs(self._visual_scale - 1.0) < 0.0001:
            super().paintEvent(event)
            return
        painter = QtWidgets.QStylePainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        center = self.rect().center()
        painter.translate(center)
        painter.scale(self._visual_scale, self._visual_scale)
        painter.translate(-center)
        option = QtWidgets.QStyleOptionButton()
        self.initStyleOption(option)
        painter.drawControl(QtWidgets.QStyle.ControlElement.CE_PushButton, option)


def make_button(text: str = "", parent=None, variant: str | None = None):
    """Create a themed button with the shared pointer feedback behavior."""

    # Also accept make_button("Label", "primary") as a convenient shorthand.
    if isinstance(parent, str) and variant is None:
        variant, parent = parent, None
    button = PressableButton(text, parent)
    if variant:
        button.setProperty("variant", variant)
    return button


def make_glass_panel(
    parent=None,
    material: str = "panel",
    card_style: str | None = None,
    elevated: bool = False,
):
    """Create a free-form glass surface for newly composed pages."""

    frame = QtWidgets.QFrame(parent)
    frame.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
    frame.setProperty("glass", material)
    if card_style:
        frame.setProperty("cardStyle", card_style)
    return frame


def build_stylesheet(dark: bool = True) -> str:
    """Build QSS using only properties supported reliably by Qt 6."""

    t = tokens(dark)
    icons = (Path(__file__).resolve().parent / "assets" / "ui").as_posix()
    return f"""
    QMainWindow, QDialog {{
        background: {t['bg_gradient']};
        color: {t['text']};
    }}
    QWidget {{
        color: {t['text']};
        font-size: 14px;
        font-weight: 400;
        selection-background-color: {t['primary']};
        selection-color: {t['on_primary']};
    }}
    QWidget#AppShell {{ background: {t['bg_gradient']}; }}
    QWidget#PageContent, QStackedWidget, QStackedWidget > QWidget {{ background: transparent; }}

    QFrame#SideNav {{
        background: {t['sidebar_glass']};
        border: 0;
        border-right: 1px solid {t['hairline']};
    }}
    QFrame#TopBar {{
        background: {t['glass']};
        border: 0;
        border-bottom: 1px solid {t['hairline']};
    }}
    QFrame#ActionBar {{
        background: {t['glass_alt']};
        border: 1px solid {t['hairline']};
        border-radius: 16px;
    }}
    QFrame#SafetyStrip {{
        background: {t['primary_soft']};
        border: 1px solid {t['hairline']};
        border-left: 3px solid {t['primary']};
        border-radius: 12px;
    }}
    QFrame#SafetyStrip[tone="warning"] {{
        background: {t['warning_soft']};
        border-left-color: {t['warning']};
    }}
    QFrame#ReplaceWarning {{
        background: {t['danger_soft']};
        border: 1px solid rgba(255, 115, 132, 100);
        border-radius: 11px;
    }}
    QFrame[surface="soft"] {{
        background: rgba(255, 255, 255, 10);
        border: 1px solid {t['hairline']};
        border-radius: 11px;
    }}
    QFrame#WorkflowConnector {{
        background: {t['hairline']};
        border: 0;
    }}
    QFrame[card="true"] {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 {t['glass_strong']}, stop:1 {t['glass']});
        border: 1px solid {t['hairline']};
        border-radius: 18px;
    }}
    QFrame[card="true"][elevated="true"] {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 {t['glass_strong']}, stop:1 {t['glass_alt']});
        border-color: {t['hairline_strong']};
    }}
    QFrame[glassPanel="true"], QFrame[glass="true"], QFrame[glass="panel"] {{
        background: {t['glass']};
        border: 1px solid {t['hairline']};
        border-radius: 14px;
    }}
    QFrame[glass="subtle"] {{
        background: {t['glass_alt']};
        border: 1px solid {t['hairline']};
        border-radius: 14px;
    }}
    QFrame[glass="strong"] {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 {t['glass_strong']}, stop:1 {t['glass']});
        border: 1px solid {t['hairline_strong']};
        border-radius: 18px;
    }}
    QFrame[cardStyle="hero"], QFrame[card="true"][cardStyle="hero"] {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 {t['glass_strong']}, stop:0.72 {t['glass']}, stop:1 {t['primary_soft']});
        border: 1px solid {t['hairline_strong']};
        border-radius: 22px;
    }}
    QFrame[cardStyle="metric"], QFrame[card="true"][cardStyle="metric"] {{
        background: {t['glass_alt']};
        border: 1px solid {t['hairline']};
        border-radius: 16px;
    }}
    QFrame[cardStyle="workflow"], QFrame[card="true"][cardStyle="workflow"] {{
        background: {t['primary_soft']};
        border: 1px solid {t['primary']};
        border-radius: 16px;
    }}
    QFrame[cardStyle="warning"], QFrame[card="true"][cardStyle="warning"] {{
        background: {t['warning_soft']};
        border: 1px solid {t['warning']};
        border-radius: 16px;
    }}
    QFrame[cardStyle="danger"], QFrame[card="true"][cardStyle="danger"] {{
        background: {t['danger_soft']};
        border: 1px solid {t['danger']};
        border-radius: 16px;
    }}
    QFrame[cardStyle="flat"], QFrame[card="true"][cardStyle="flat"] {{
        background: transparent;
        border: 1px solid {t['hairline']};
        border-radius: 14px;
    }}

    QLabel {{ background: transparent; }}
    QLabel#BrandMark {{
        color: {t['cyan']};
        background: {t['primary_soft']};
        border: 1px solid {t['hairline_strong']};
        border-radius: 13px;
        font-size: 14px;
        font-weight: 800;
    }}
    QLabel[textRole="brand"] {{ color: {t['text']}; font-size: 19px; font-weight: 700; }}
    QLabel[textRole="eyebrow"] {{ color: {t['muted']}; font-size: 12px; font-weight: 600; }}
    QLabel[textRole="navCaption"] {{ color: {t['subtle']}; font-size: 12px; font-weight: 700; }}
    QLabel[textRole="pageTitle"] {{ color: {t['text']}; font-size: 36px; font-weight: 600; }}
    QLabel[textRole="sectionTitle"] {{ color: {t['text']}; font-size: 20px; font-weight: 600; }}
    QLabel[textRole="metric"] {{ color: {t['text']}; font-size: 24px; font-weight: 700; }}
    QLabel[textRole="heroTitle"] {{ color: {t['text']}; font-size: 54px; font-weight: 600; }}
    QLabel[textRole="heroAccent"] {{ color: {t['primary']}; font-size: 54px; font-weight: 600; }}
    QLabel[textRole="body"] {{ color: {t['muted']}; font-size: 14px; }}
    QLabel[textRole="bodyStrong"] {{ color: {t['text']}; font-size: 13px; font-weight: 700; }}
    QLabel[textRole="fieldLabel"] {{ color: {t['text']}; font-size: 13px; font-weight: 600; }}
    QLabel[textRole="stepNumber"] {{ color: {t['cyan']}; font-size: 12px; font-weight: 800; }}
    QLabel[textRole="caption"] {{ color: {t['subtle']}; font-size: 12px; font-weight: 600; }}
    QLabel[textRole="success"] {{ color: {t['success']}; font-weight: 700; }}
    QLabel[textRole="warning"] {{ color: {t['warning']}; font-weight: 700; }}
    QLabel[textRole="danger"] {{ color: {t['danger']}; font-weight: 700; }}
    QLabel[textRole="muted"], QLabel#Hint, QLabel#MutedLabel {{ color: {t['muted']}; }}
    QLabel#CandidateSummary {{
        color: {t['muted']};
        background: rgba(255, 255, 255, 10);
        border: 1px solid {t['hairline']};
        border-radius: 10px;
        padding: 9px 11px;
    }}
    QLabel#Good {{ color: {t['success']}; font-weight: 700; }}
    QLabel#Warning {{ color: {t['warning']}; font-weight: 600; }}

    QLabel[badge] {{
        border-radius: 10px;
        padding: 4px 10px;
        font-size: 12px;
        font-weight: 700;
    }}
    QLabel[badge="neutral"] {{ color: {t['muted']}; background: {t['glass_alt']}; border: 1px solid {t['hairline']}; }}
    QLabel[badge="primary"] {{ color: {t['cyan']}; background: {t['primary_soft']}; border: 1px solid {t['primary']}; }}
    QLabel[badge="success"] {{ color: {t['success']}; background: {t['success_soft']}; border: 1px solid {t['success']}; }}
    QLabel[badge="warning"] {{ color: {t['warning']}; background: {t['warning_soft']}; border: 1px solid {t['warning']}; }}
    QLabel[badge="danger"] {{ color: {t['danger']}; background: {t['danger_soft']}; border: 1px solid {t['danger']}; }}

    QPushButton, QToolButton {{
        min-height: 42px;
        padding: 0 14px;
        border-radius: 10px;
        background: {t['glass_alt']};
        color: {t['text']};
        border: 1px solid {t['hairline']};
        font-weight: 600;
    }}
    QPushButton:hover, QToolButton:hover {{
        background: {t['glass_hover']};
        border-color: {t['hairline_strong']};
    }}
    QPushButton:pressed, QToolButton:pressed {{
        background: {t['glass_pressed']};
        border-color: {t['border_strong']};
    }}
    QPushButton:disabled, QToolButton:disabled {{
        color: {t['subtle']};
        background: {t['glass']};
        border-color: {t['border']};
    }}
    QPushButton[variant="primary"], QPushButton#Primary {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {t['primary']}, stop:1 {t['primary_hover']});
        color: {t['on_primary']};
        border-color: {t['primary_hover']};
        font-weight: 700;
    }}
    QPushButton[variant="primary"]:hover, QPushButton#Primary:hover {{
        background: {t['primary_hover']};
        border-color: {t['focus']};
    }}
    QPushButton[variant="primary"]:pressed, QPushButton#Primary:pressed {{
        background: {t['primary_deep']};
        border-color: {t['primary']};
    }}
    QPushButton[variant="danger"], QPushButton#Danger {{
        color: {t['danger']};
        border-color: {t['danger']};
        background: {t['danger_soft']};
    }}
    QPushButton[variant="danger"]:hover, QPushButton#Danger:hover {{ background: {t['danger_soft']}; border-color: {t['danger']}; }}
    QPushButton[variant="ghost"] {{ background: transparent; border-color: transparent; color: {t['muted']}; }}
    QPushButton[variant="ghost"]:hover {{ color: {t['text']}; background: {t['glass_hover']}; border-color: {t['hairline']}; }}
    QPushButton[variant="quiet"] {{ background: transparent; color: {t['muted']}; border-color: {t['hairline']}; }}
    QPushButton[variant="quiet"]:hover {{ color: {t['text']}; background: {t['glass_hover']}; border-color: {t['hairline_strong']}; }}
    QPushButton[variant="secondary"] {{ background: {t['glass_alt']}; color: {t['text']}; border-color: {t['hairline_strong']}; }}
    QPushButton[variant="workflow"] {{ background: {t['primary_soft']}; color: {t['cyan']}; border-color: {t['primary']}; }}
    QPushButton[variant="nav"] {{
        min-height: 44px;
        text-align: left;
        padding: 0 14px;
        background: transparent;
        border: 1px solid transparent;
        color: {t['muted']};
    }}
    QPushButton[variant="nav"][compact="true"] {{
        padding: 0;
        text-align: center;
    }}
    QPushButton[variant="nav"]:hover {{ color: {t['text']}; background: {t['glass_hover']}; border-color: {t['hairline']}; }}
    QPushButton[variant="nav"]:checked {{
        color: {t['text']};
        background: {t['primary_soft']};
        border: 1px solid {t['hairline_strong']};
        border-left: 3px solid {t['primary']};
    }}
    QPushButton[variant="primary"]:disabled, QPushButton#Primary:disabled {{
        color: {t['subtle']};
        background: {t['glass']};
        border-color: {t['border']};
    }}
    QPushButton[variant="danger"]:disabled, QPushButton#Danger:disabled {{
        color: {t['subtle']};
        background: {t['glass']};
        border-color: {t['border']};
    }}
    QPushButton:focus, QToolButton:focus {{
        border: 2px solid {t['focus']};
    }}

    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
        background: {t['glass_alt']};
        color: {t['text']};
        border: 1px solid {t['hairline']};
        border-radius: 10px;
        padding: 6px 10px;
        placeholder-text-color: {t['subtle']};
    }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ min-height: 28px; }}
    QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover, QPlainTextEdit:hover, QTextEdit:hover {{ border-color: {t['hairline_strong']}; }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
        background: {t['glass_strong']};
        border: 2px solid {t['focus']};
    }}
    QLineEdit:read-only, QPlainTextEdit:read-only, QTextEdit:read-only {{ background: {t['glass']}; color: {t['muted']}; }}
    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{ color: {t['subtle']}; border-color: {t['border']}; }}
    QComboBox::drop-down {{ width: 30px; border: 0; border-left: 1px solid {t['hairline']}; }}
    QComboBox:on {{ background: {t['glass_strong']}; border-color: {t['focus']}; }}
    QComboBox QAbstractItemView {{
        background: {t['surface']};
        color: {t['text']};
        border: 1px solid {t['hairline_strong']};
        selection-background-color: {t['primary_soft']};
        selection-color: {t['text']};
        padding: 6px;
    }}

    QGroupBox {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 {t['glass_strong']}, stop:1 {t['glass']});
        border: 1px solid {t['hairline']};
        border-radius: 18px;
        margin-top: 18px;
        padding: 18px 16px 16px 16px;
        font-weight: 700;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 16px;
        padding: 0 7px;
        color: {t['text']};
        background: {t['surface']};
    }}

    QTableWidget, QTableView, QListView, QTreeView {{
        background: {t['glass']};
        alternate-background-color: {t['glass_alt']};
        color: {t['text']};
        border: 1px solid {t['hairline']};
        border-radius: 12px;
        gridline-color: {t['hairline']};
        selection-background-color: {t['primary_soft']};
        selection-color: {t['text']};
    }}
    QTableWidget::item, QTableView::item, QListView::item, QTreeView::item {{ padding: 8px; border: 0; }}
    QTableWidget::item:hover, QTableView::item:hover, QListView::item:hover, QTreeView::item:hover {{ background: {t['glass_hover']}; }}
    QTableWidget::item:selected, QTableView::item:selected, QListView::item:selected, QTreeView::item:selected {{ background: {t['primary_soft']}; color: {t['text']}; }}
    QHeaderView::section {{
        background: {t['glass_alt']};
        color: {t['muted']};
        padding: 9px;
        border: 0;
        border-bottom: 1px solid {t['hairline']};
        font-weight: 700;
    }}

    QCheckBox, QRadioButton {{ spacing: 9px; }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 18px;
        height: 18px;
        border: 1px solid {t['border_strong']};
        background: {t['glass_alt']};
    }}
    QCheckBox::indicator {{ border-radius: 6px; }}
    QRadioButton::indicator {{ border-radius: 9px; }}
    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {t['focus']}; background: {t['glass_hover']}; }}
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{ background: {t['primary']}; border: 3px solid {t['focus']}; }}
    QCheckBox:focus, QRadioButton:focus {{ color: {t['focus']}; }}

    QSlider::groove:horizontal {{ height: 6px; border-radius: 3px; background: {t['border']}; }}
    QSlider::sub-page:horizontal {{ background: {t['primary']}; border-radius: 3px; }}
    QSlider::handle:horizontal {{ width: 17px; margin: -6px 0; border-radius: 8px; background: {t['text']}; border: 2px solid {t['primary']}; }}
    QSlider::handle:horizontal:hover {{ background: {t['focus']}; border-color: {t['focus']}; }}
    QProgressBar {{ background: {t['glass_alt']}; border: 1px solid {t['hairline']}; border-radius: 6px; text-align: center; color: {t['text']}; }}
    QProgressBar::chunk {{ background: {t['primary']}; border-radius: 5px; }}

    QTabWidget::pane {{ background: {t['glass']}; border: 1px solid {t['hairline']}; border-radius: 12px; top: -1px; }}
    QTabBar::tab {{ background: transparent; color: {t['muted']}; padding: 9px 14px; border-bottom: 2px solid transparent; }}
    QTabBar::tab:hover {{ color: {t['text']}; background: {t['glass_hover']}; }}
    QTabBar::tab:selected {{ color: {t['text']}; border-bottom-color: {t['primary']}; }}
    QTabBar::tab:focus {{ color: {t['focus']}; }}

    QMenu {{ background: {t['surface']}; color: {t['text']}; border: 1px solid {t['hairline_strong']}; padding: 6px; }}
    QMenu::item {{ border-radius: 7px; padding: 7px 22px 7px 10px; }}
    QMenu::item:selected {{ background: {t['primary_soft']}; color: {t['text']}; }}
    QMenu::separator {{ height: 1px; background: {t['hairline']}; margin: 5px 8px; }}

    QScrollArea {{ background: transparent; border: 0; }}
    QScrollArea > QWidget > QWidget {{ background: transparent; }}
    QScrollBar:vertical {{ width: 11px; background: transparent; margin: 3px; }}
    QScrollBar:horizontal {{ height: 11px; background: transparent; margin: 3px; }}
    QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{ min-height: 34px; min-width: 34px; border-radius: 4px; background: {t['border_strong']}; }}
    QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{ background: {t['primary']}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    QSplitter::handle {{ background: transparent; width: 7px; height: 7px; }}
    QSplitter::handle:hover {{ background: {t['primary_soft']}; }}
    QToolBar#AppBar {{ background: {t['glass']}; border: 0; border-bottom: 1px solid {t['hairline']}; padding: 8px 12px; spacing: 8px; }}
    QStatusBar {{ background: {t['glass']}; color: {t['muted']}; border-top: 1px solid {t['hairline']}; }}
    QStatusBar::item {{ border: 0; }}
    QToolTip {{ color: {t['text']}; background: {t['surface_alt']}; border: 1px solid {t['hairline_strong']}; padding: 7px; }}
    QFrame#Navigation {{ background: {t['surface']}; border-bottom: 1px solid {t['border']}; }}
    QPushButton[variant="navigation"] {{ background: transparent; border: 2px solid transparent; border-radius: 8px; color: {t['muted']}; }}
    QPushButton[variant="navigation"]:checked {{ color: {t['text']}; background: {t['surface_alt']}; }}
    QPushButton[variant="navigation"]:hover {{ color: {t['text']}; background: {t['surface_hover']}; }}
    QPushButton[variant="navigation"]:focus {{ border-color: {t['focus']}; }}
    QFrame[cardStyle="stat"] {{ background: transparent; border: 0; border-radius: 0; }}
    QLabel[textRole="heroDescription"] {{ color: {t['muted']}; font-size: 17px; }}
    QPushButton[variant="primary"] {{ border-radius: 21px; background: {t['primary']}; }}
    QPushButton[variant="primary"]:hover {{ background: {t['primary_hover']}; }}
    QPushButton[variant="primary"]:disabled {{ background: {t['surface_alt']}; color: {t['subtle']}; }}
    QPushButton[variant="ghost"] {{ color: {t['primary']}; }}
    QMessageBox {{ background: {t['bg_gradient']}; }}

    /* Quiet surfaces. Hierarchy comes from type and space, not nested borders. */
    QFrame[card="true"], QFrame[card="true"][elevated="true"],
    QFrame[glass="panel"], QFrame[glass="strong"] {{
        background: {t['surface']}; border: 1px solid transparent; border-radius: 20px;
    }}
    QFrame[surface="soft"], QFrame[glass="subtle"] {{
        background: {t['surface_alt']}; border: 0; border-radius: 12px;
    }}
    QFrame[cardStyle="stat"], QFrame[cardStyle="step"] {{
        background: transparent; border: 0; border-radius: 0;
    }}
    QFrame[cardStyle="metric"] {{ background: {t['surface']}; border: 0; }}
    QFrame#Navigation {{ background: {t['surface']}; border: 0; border-bottom: 1px solid {t['border']}; }}
    QPushButton[variant="navigation"] {{ min-height: 40px; padding: 0 18px; font-weight: 500; }}
    QPushButton[variant="navigation"]:checked {{ color: {t['cyan']}; background: {t['primary_soft']}; }}
    QFrame#HomeHero {{ background: transparent; border: 0; }}
    QFrame#HomeStats {{ background: transparent; border: 0; border-top: 1px solid {t['border']}; border-bottom: 1px solid {t['border']}; }}
    QFrame#Workflow {{ background: {t['surface']}; border: 0; border-radius: 20px; }}
    QFrame#DisclosureCard {{ background: transparent; border: 0; border-top: 1px solid {t['border']}; border-radius: 0; }}
    QFrame#DisclosureCard QLabel[textRole="sectionTitle"] {{ font-size: 14px; }}
    QFrame#PageHeader {{ background: transparent; }}
    QFrame#SafetyStrip, QFrame#SafetyStrip[tone="warning"] {{ background: transparent; border: 0; border-radius: 0; }}
    QFrame#ActionBar {{ background: {t['surface']}; border: 0; border-radius: 16px; }}
    QFrame#ReplaceWarning {{ background: {t['surface_alt']}; border: 0; border-radius: 12px; }}
    QLabel[badge] {{ border: 0; border-radius: 10px; padding: 5px 10px; font-weight: 500; }}
    QLabel[badge="primary"] {{ color: {t['cyan']}; }}
    QLabel#CandidateSummary {{ background: {t['surface_alt']}; border: 0; padding: 14px; }}
    QPushButton, QToolButton {{ min-height: 42px; border: 2px solid transparent; padding: 0 16px; font-weight: 500; }}
    QPushButton:focus, QToolButton:focus {{ border: 2px solid {t['focus']}; }}
    QPushButton[variant="primary"] {{ border: 2px solid transparent; }}
    QPushButton[variant="primary"]:focus {{ border: 2px solid {t['text']}; }}
    QPushButton[variant="ghost"] {{ color: {t['cyan']}; }}
    QPushButton[variant="quiet"] {{ border: 1px solid {t['border']}; }}
    QPushButton[variant="quiet"]:focus {{ border: 2px solid {t['focus']}; }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        min-height: 28px; border: 1px solid {t['border_strong']}; background: {t['surface']};
        padding: 7px 12px; border-radius: 10px;
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border: 2px solid {t['focus']}; padding: 6px 11px; }}
    QComboBox::drop-down {{ border: 0; width: 28px; }}
    QComboBox::down-arrow {{ image: url("{icons}/chevron-down.svg"); width: 16px; height: 16px; }}
    QAbstractSpinBox {{ padding-right: 28px; }}
    QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right; width: 26px; border: 0; background: transparent; }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right; width: 26px; border: 0; background: transparent; }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url("{icons}/chevron-up.svg"); width: 14px; height: 14px; }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url("{icons}/chevron-down.svg"); width: 14px; height: 14px; }}
    QCheckBox {{ spacing: 10px; min-height: 30px; }}
    QCheckBox::indicator {{ width: 18px; height: 18px; border: 1px solid {t['border_strong']}; border-radius: 5px; background: {t['surface']}; }}
    QCheckBox::indicator:checked {{ image: url("{icons}/check.svg"); background: {t['primary']}; border: 1px solid {t['primary']}; }}
    QCheckBox::indicator:focus {{ border: 2px solid {t['focus']}; }}
    QTableWidget, QTableView, QListView {{ border: 0; background: {t['surface']}; alternate-background-color: {t['surface_alt']}; }}
    QHeaderView::section {{ background: {t['surface_alt']}; color: {t['muted']}; border: 0; padding: 14px 10px; font-weight: 500; }}
    QMenu {{ background: {t['surface']}; color: {t['text']}; border: 1px solid {t['border']}; padding: 6px; }}
    QMenu::item {{ padding: 10px 24px; border-radius: 6px; }}
    QMenu::item:selected {{ background: {t['primary_soft']}; color: {t['cyan']}; }}
    QMenu::separator {{ background: {t['border']}; height: 1px; margin: 6px; }}
    """


__all__ = [
    "DARK",
    "LIGHT",
    "PressableButton",
    "apply_glass_shadow",
    "build_stylesheet",
    "make_badge",
    "make_button",
    "make_card",
    "make_glass_panel",
    "reduce_motion_enabled",
    "set_property",
    "tokens",
]
