"""Shared SentinelVision desktop design system."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


DARK = {
    "bg": "#07101D",
    "surface": "#0C1727",
    "surface_alt": "#101E31",
    "surface_hover": "#162844",
    "border": "#223651",
    "border_strong": "#315176",
    "text": "#F4F7FB",
    "muted": "#9CB0C8",
    "subtle": "#6F87A4",
    "primary": "#4F8CFF",
    "primary_hover": "#6DA1FF",
    "cyan": "#22D3EE",
    "success": "#2DD4A3",
    "warning": "#F6B94A",
    "danger": "#F05D6C",
}

LIGHT = {
    "bg": "#F3F6FA",
    "surface": "#FFFFFF",
    "surface_alt": "#F7F9FC",
    "surface_hover": "#EBF1FA",
    "border": "#D7E0EB",
    "border_strong": "#B8C8DA",
    "text": "#142033",
    "muted": "#586B82",
    "subtle": "#8191A5",
    "primary": "#2F6FED",
    "primary_hover": "#245FD1",
    "cyan": "#0891B2",
    "success": "#159A75",
    "warning": "#C78312",
    "danger": "#D74252",
}


def tokens(dark: bool = True):
    return DARK if dark else LIGHT


def set_property(widget, name: str, value):
    widget.setProperty(name, value)
    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)
    widget.update()
    return widget


def make_card(parent=None, elevated: bool = False):
    frame = QtWidgets.QFrame(parent)
    frame.setProperty("card", True)
    frame.setProperty("elevated", elevated)
    return frame


def make_badge(text: str, tone: str = "neutral", parent=None):
    label = QtWidgets.QLabel(text, parent)
    label.setProperty("badge", tone)
    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    return label


def build_stylesheet(dark: bool = True) -> str:
    t = tokens(dark)
    return f"""
    * {{ outline: none; }}
    QMainWindow, QDialog {{ background: {t['bg']}; color: {t['text']}; }}
    QWidget {{ color: {t['text']}; font-size: 13px; }}
    QWidget#AppShell, QWidget#PageContent, QStackedWidget {{ background: {t['bg']}; }}
    QFrame#SideNav {{ background: {t['surface']}; border-right: 1px solid {t['border']}; }}
    QFrame#TopBar {{ background: {t['bg']}; border-bottom: 1px solid {t['border']}; }}
    QFrame[card="true"] {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 14px; }}
    QFrame[card="true"][elevated="true"] {{ background: {t['surface_alt']}; border-color: {t['border_strong']}; }}
    QLabel[textRole="brand"] {{ color: {t['text']}; font-size: 18px; font-weight: 700; }}
    QLabel[textRole="eyebrow"] {{ color: {t['cyan']}; font-size: 11px; font-weight: 700; }}
    QLabel[textRole="pageTitle"] {{ color: {t['text']}; font-size: 24px; font-weight: 700; }}
    QLabel[textRole="sectionTitle"] {{ color: {t['text']}; font-size: 15px; font-weight: 700; }}
    QLabel[textRole="metric"] {{ color: {t['text']}; font-size: 20px; font-weight: 700; }}
    QLabel[textRole="muted"], QLabel#Hint, QLabel#MutedLabel {{ color: {t['muted']}; }}
    QLabel#Good {{ color: {t['success']}; font-weight: 650; }}
    QLabel#Warning {{ color: {t['warning']}; }}
    QLabel[badge] {{ border-radius: 10px; padding: 3px 9px; font-size: 11px; font-weight: 650; }}
    QLabel[badge="neutral"] {{ color: {t['muted']}; background: {t['surface_alt']}; border: 1px solid {t['border']}; }}
    QLabel[badge="primary"] {{ color: {t['primary']}; background: {t['surface_alt']}; border: 1px solid {t['primary']}; }}
    QLabel[badge="success"] {{ color: {t['success']}; background: {t['surface_alt']}; border: 1px solid {t['success']}; }}
    QLabel[badge="warning"] {{ color: {t['warning']}; background: {t['surface_alt']}; border: 1px solid {t['warning']}; }}
    QLabel[badge="danger"] {{ color: {t['danger']}; background: {t['surface_alt']}; border: 1px solid {t['danger']}; }}
    QPushButton {{ min-height: 36px; padding: 0 14px; border-radius: 9px; background: {t['surface_alt']}; color: {t['text']}; border: 1px solid {t['border']}; font-weight: 600; }}
    QPushButton:hover {{ background: {t['surface_hover']}; border-color: {t['border_strong']}; }}
    QPushButton:pressed {{ padding-top: 1px; background: {t['surface']}; }}
    QPushButton:focus {{ border-color: {t['primary']}; }}
    QPushButton:disabled {{ color: {t['subtle']}; background: {t['surface']}; border-color: {t['border']}; }}
    QPushButton[variant="primary"], QPushButton#Primary {{ background: {t['primary']}; color: #FFFFFF; border-color: {t['primary']}; }}
    QPushButton[variant="primary"]:hover, QPushButton#Primary:hover {{ background: {t['primary_hover']}; border-color: {t['primary_hover']}; }}
    QPushButton[variant="danger"], QPushButton#Danger {{ color: {t['danger']}; border-color: {t['danger']}; background: {t['surface_alt']}; }}
    QPushButton[variant="ghost"] {{ background: transparent; border-color: transparent; color: {t['muted']}; }}
    QPushButton[variant="nav"] {{ min-height: 42px; text-align: left; padding: 0 14px; background: transparent; border: 1px solid transparent; color: {t['muted']}; }}
    QPushButton[variant="nav"]:hover {{ color: {t['text']}; background: {t['surface_hover']}; }}
    QPushButton[variant="nav"]:checked {{ color: #FFFFFF; background: {t['primary']}; border-color: {t['primary']}; }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{ background: {t['surface_alt']}; color: {t['text']}; border: 1px solid {t['border']}; border-radius: 9px; padding: 6px 10px; selection-background-color: {t['primary']}; }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ min-height: 26px; }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{ border-color: {t['primary']}; }}
    QComboBox::drop-down {{ width: 28px; border: 0; }}
    QComboBox QAbstractItemView {{ background: {t['surface']}; color: {t['text']}; border: 1px solid {t['border_strong']}; selection-background-color: {t['primary']}; padding: 5px; }}
    QGroupBox {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 14px; margin-top: 16px; padding: 16px 14px 14px 14px; font-weight: 700; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 14px; padding: 0 6px; color: {t['text']}; }}
    QTableWidget, QTableView, QListView {{ background: {t['surface']}; alternate-background-color: {t['surface_alt']}; color: {t['text']}; border: 1px solid {t['border']}; border-radius: 10px; gridline-color: {t['border']}; selection-background-color: {t['surface_hover']}; selection-color: {t['text']}; }}
    QTableWidget::item, QTableView::item, QListView::item {{ padding: 8px; border: 0; }}
    QHeaderView::section {{ background: {t['surface_alt']}; color: {t['muted']}; padding: 8px; border: 0; border-bottom: 1px solid {t['border']}; font-weight: 650; }}
    QCheckBox {{ spacing: 8px; }}
    QCheckBox::indicator {{ width: 17px; height: 17px; border: 1px solid {t['border_strong']}; border-radius: 5px; background: {t['surface_alt']}; }}
    QCheckBox::indicator:checked {{ background: {t['primary']}; border-color: {t['primary']}; }}
    QSlider::groove:horizontal {{ height: 5px; border-radius: 2px; background: {t['border']}; }}
    QSlider::sub-page:horizontal {{ background: {t['primary']}; border-radius: 2px; }}
    QSlider::handle:horizontal {{ width: 15px; margin: -5px 0; border-radius: 7px; background: {t['text']}; border: 2px solid {t['primary']}; }}
    QScrollArea {{ background: transparent; border: 0; }}
    QScrollArea > QWidget > QWidget {{ background: transparent; }}
    QScrollBar:vertical {{ width: 10px; background: transparent; margin: 3px; }}
    QScrollBar::handle:vertical {{ min-height: 32px; border-radius: 4px; background: {t['border_strong']}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QSplitter::handle {{ background: transparent; width: 6px; }}
    QToolBar#AppBar {{ background: {t['surface']}; border: 0; border-bottom: 1px solid {t['border']}; padding: 7px 12px; spacing: 7px; }}
    QStatusBar {{ background: {t['surface']}; color: {t['muted']}; border-top: 1px solid {t['border']}; }}
    QToolTip {{ color: {t['text']}; background: {t['surface_alt']}; border: 1px solid {t['border_strong']}; padding: 6px; }}
    QMessageBox {{ background: {t['surface']}; }}
    """


__all__ = ["DARK", "LIGHT", "build_stylesheet", "make_badge", "make_card", "set_property", "tokens"]
