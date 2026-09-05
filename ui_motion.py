"""Shared native motion: short transitions, stable layouts, immediate actions."""

from contextlib import contextmanager

from PySide6 import QtCore, QtGui, QtWidgets


PAGE_DURATION_MS = 240
TAB_DURATION_MS = 180
FADE_DURATION_MS = 120
PRESS_DURATION_MS = 120
RELEASE_DURATION_MS = 180
DRAWER_ENTER_MS = 260
DRAWER_EXIT_MS = 200
_keyboard_depth = 0


def motion_curve(drawer=False):
    """Use the design system's strong ease-out and spatial drawer curves."""
    curve = QtCore.QEasingCurve(QtCore.QEasingCurve.Type.BezierSpline)
    first, second = ((.32, .72), (0, 1)) if drawer else ((.23, 1), (.32, 1))
    curve.addCubicBezierSegment(QtCore.QPointF(*first), QtCore.QPointF(*second), QtCore.QPointF(1, 1))
    return curve


@contextmanager
def keyboard_activation():
    """Propagate synchronous keyboard activation through ordinary Qt signals."""
    global _keyboard_depth
    _keyboard_depth += 1
    try:
        yield
    finally:
        _keyboard_depth -= 1


def is_keyboard_activation():
    return _keyboard_depth > 0


class PageTransition(QtWidgets.QWidget):
    """Briefly composite two page snapshots without moving live controls.

    State changes happen synchronously. This mouse-transparent overlay only
    paints; it never holds up clicks, layout, focus or application logic. A new
    transition captures the current composite, so rapid navigation has no jump
    back to a stale page. Pixmaps are released as soon as motion settles.
    """

    def __init__(self, target, duration=PAGE_DURATION_MS, distance=12):
        super().__init__(target)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._before = QtGui.QPixmap()
        self._after = QtGui.QPixmap()
        self._progress = 1.0
        self._distance = distance
        self._fade_curve = motion_curve()
        self.animation = QtCore.QPropertyAnimation(self, b"progress", self)
        self.animation.setDuration(duration)
        self.animation.setEasingCurve(motion_curve())
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.finished.connect(self.cancel)
        target.installEventFilter(self)
        self.hide()

    def _get_progress(self):
        return self._progress

    def _set_progress(self, value):
        self._progress = float(value)
        self.update()

    progress = QtCore.Property(float, _get_progress, _set_progress)

    def cancel(self):
        self.animation.stop()
        self.hide()
        self._progress = 1.0
        self._before = QtGui.QPixmap()
        self._after = QtGui.QPixmap()

    def run(self, change, animate=True):
        target = self.parentWidget()
        if not animate or is_keyboard_activation() or not target.isVisible():
            self.cancel()
            change()
            return
        # Capture through the window to include the real ancestor background;
        # grabbing a transparent child directly substitutes Qt's default gray.
        before = self._capture()
        self.cancel()
        change()
        if target.layout() is not None:
            target.layout().activate()
        after = self._capture()
        self._before, self._after = before, after
        self.setGeometry(target.rect())
        self._progress = 0.0
        self.show()
        self.raise_()
        self.animation.start()

    def _capture(self):
        target = self.parentWidget()
        window = target.window()
        bounds = QtCore.QRect(target.mapTo(window, QtCore.QPoint()), target.size())
        return window.grab(bounds)

    def eventFilter(self, watched, event):
        if event.type() in (QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.Hide,
                            QtCore.QEvent.Type.StyleChange, QtCore.QEvent.Type.PaletteChange):
            self.cancel()
        return False

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        p = self._progress
        painter.drawPixmap(QtCore.QPointF(0, -self._distance * .25 * p), self._before)
        # Let text resolve before the remaining movement settles.
        fade_time = min(1.0, self.animation.currentTime() / FADE_DURATION_MS)
        painter.setOpacity(self._fade_curve.valueForProgress(fade_time))
        painter.drawPixmap(QtCore.QPointF(0, self._distance * (1 - p)), self._after)


class _TransitionTabBar(QtWidgets.QTabBar):
    def mousePressEvent(self, event):
        owner = self.parentWidget()
        index = self.tabAt(event.position().toPoint())
        if (event.button() == QtCore.Qt.MouseButton.LeftButton and index >= 0
                and index != self.currentIndex() and self.isTabEnabled(index)):
            owner.page_transition().run(lambda: super(_TransitionTabBar, self).mousePressEvent(event))
        else:
            super().mousePressEvent(event)

    def keyPressEvent(self, event):
        self.parentWidget().page_transition().cancel()
        with keyboard_activation():
            super().keyPressEvent(event)


class AnimatedTabWidget(QtWidgets.QTabWidget):
    """Pointer-only inspector transitions; keyboard and data updates stay live."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setTabBar(_TransitionTabBar(self))
        self._transition = None
        self.currentChanged.connect(self._cancel_transition)

    def page_transition(self):
        if self._transition is None:
            self._transition = PageTransition(self.currentWidget().parentWidget(), TAB_DURATION_MS, 8)
        return self._transition

    def _cancel_transition(self, _index):
        if self._transition is not None:
            self._transition.cancel()
