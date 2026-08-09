"""Liquid Glass design system shared by the SentinelVision desktop apps.

Qt Style Sheets cannot blur the content behind a widget.  The theme therefore
uses translucent, cyan-tinted layers, bright hairline borders and restrained
shadows to preserve the same depth hierarchy without relying on unsupported
CSS properties.
"""

from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets


# The original public keys remain available for older screens.  Additional
# tokens describe the translucent material layers used by the new theme.
DARK = {
    "bg": "#061119",
    "surface": "#0B202C",
    "surface_alt": "#102D3C",
    "surface_hover": "#164359",
    "border": "#254958",
    "border_strong": "#3F7B8D",
    "text": "#F2FBFF",
    "muted": "#A8C0C9",
    "subtle": "#718C97",
    "primary": "#27C5E7",
    "primary_hover": "#58DCF2",
    "cyan": "#7AE7F6",
    "success": "#54D6A1",
    "warning": "#F7C66C",
    "danger": "#FF7384",
    "bg_gradient": "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #061119, stop:0.52 #071A24, stop:1 #05131D)",
    "sidebar_glass": "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgba(11, 34, 46, 244), stop:1 rgba(7, 25, 36, 230))",
    "glass": "rgba(10, 30, 42, 222)",
    "glass_alt": "rgba(16, 45, 59, 210)",
    "glass_strong": "rgba(18, 55, 70, 232)",
    "glass_hover": "rgba(44, 104, 122, 150)",
    "glass_pressed": "rgba(7, 25, 35, 238)",
    "hairline": "rgba(183, 242, 255, 56)",
    "hairline_strong": "rgba(176, 240, 255, 112)",
    "focus": "#9BEFFF",
    "on_primary": "#03171E",
    "primary_deep": "#1597C5",
    "primary_soft": "rgba(39, 197, 231, 42)",
    "success_soft": "rgba(84, 214, 161, 36)",
    "warning_soft": "rgba(247, 198, 108, 34)",
    "danger_soft": "rgba(255, 115, 132, 36)",
}

LIGHT = {
    "bg": "#EAF4F7",
    "surface": "#F6FBFC",
    "surface_alt": "#E9F5F7",
    "surface_hover": "#D9EEF2",
    "border": "#BCD5DA",
    "border_strong": "#87B6C0",
    "text": "#102A34",
    "muted": "#4F6972",
    "subtle": "#718991",
    "primary": "#087FA3",
    "primary_hover": "#086B8C",
    "cyan": "#087F99",
    "success": "#147B5C",
    "warning": "#9A6410",
    "danger": "#C33D51",
    "bg_gradient": "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #F1FAFB, stop:0.55 #E8F4F7, stop:1 #DDEEF2)",
    "sidebar_glass": "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgba(250, 254, 255, 238), stop:1 rgba(229, 243, 246, 226))",
    "glass": "rgba(250, 254, 255, 220)",
    "glass_alt": "rgba(229, 245, 248, 210)",
    "glass_strong": "rgba(252, 255, 255, 238)",
    "glass_hover": "rgba(185, 225, 233, 166)",
    "glass_pressed": "rgba(211, 235, 240, 232)",
    "hairline": "rgba(255, 255, 255, 190)",
    "hairline_strong": "rgba(81, 144, 158, 118)",
    "focus": "#056F92",
    "on_primary": "#FFFFFF",
    "primary_deep": "#065F7D",
    "primary_soft": "rgba(8, 127, 163, 30)",
    "success_soft": "rgba(20, 123, 92, 28)",
    "warning_soft": "rgba(154, 100, 16, 26)",
    "danger_soft": "rgba(195, 61, 81, 26)",
}


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
    if elevated:
        apply_glass_shadow(frame, elevated=True)
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
    if elevated:
        apply_glass_shadow(frame, elevated=True)
    return frame


def build_stylesheet(dark: bool = True) -> str:
    """Build QSS using only properties supported reliably by Qt 6."""

    t = tokens(dark)
    return f"""
    QMainWindow, QDialog {{
        background: {t['bg_gradient']};
        color: {t['text']};
    }}
    QWidget {{
        color: {t['text']};
        font-size: 13px;
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
    QLabel[textRole="eyebrow"] {{ color: {t['cyan']}; font-size: 11px; font-weight: 700; }}
    QLabel[textRole="navCaption"] {{ color: {t['subtle']}; font-size: 11px; font-weight: 700; }}
    QLabel[textRole="pageTitle"] {{ color: {t['text']}; font-size: 25px; font-weight: 700; }}
    QLabel[textRole="sectionTitle"] {{ color: {t['text']}; font-size: 16px; font-weight: 700; }}
    QLabel[textRole="metric"] {{ color: {t['text']}; font-size: 22px; font-weight: 700; }}
    QLabel[textRole="heroTitle"] {{ color: {t['text']}; font-size: 30px; font-weight: 700; }}
    QLabel[textRole="body"] {{ color: {t['muted']}; font-size: 13px; }}
    QLabel[textRole="bodyStrong"] {{ color: {t['text']}; font-size: 13px; font-weight: 700; }}
    QLabel[textRole="fieldLabel"] {{ color: {t['muted']}; font-size: 12px; font-weight: 650; }}
    QLabel[textRole="stepNumber"] {{ color: {t['cyan']}; font-size: 11px; font-weight: 800; }}
    QLabel[textRole="caption"] {{ color: {t['subtle']}; font-size: 11px; font-weight: 600; }}
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
        font-size: 11px;
        font-weight: 700;
    }}
    QLabel[badge="neutral"] {{ color: {t['muted']}; background: {t['glass_alt']}; border: 1px solid {t['hairline']}; }}
    QLabel[badge="primary"] {{ color: {t['cyan']}; background: {t['primary_soft']}; border: 1px solid {t['primary']}; }}
    QLabel[badge="success"] {{ color: {t['success']}; background: {t['success_soft']}; border: 1px solid {t['success']}; }}
    QLabel[badge="warning"] {{ color: {t['warning']}; background: {t['warning_soft']}; border: 1px solid {t['warning']}; }}
    QLabel[badge="danger"] {{ color: {t['danger']}; background: {t['danger_soft']}; border: 1px solid {t['danger']}; }}

    QPushButton, QToolButton {{
        min-height: 36px;
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
    QMessageBox {{ background: {t['bg_gradient']}; }}
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
