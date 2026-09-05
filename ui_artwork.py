"""Small resolution-independent brand illustration; no animation or data claims."""

from PySide6 import QtCore, QtGui, QtWidgets


class VisionArtwork(QtWidgets.QWidget):
    """An optical field and focus brackets, drawn entirely with native vectors."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(240, 250)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.translate(self.width() / 2, self.height() / 2)
        scale = min(self.width() / 390, self.height() / 310)
        p.scale(scale, scale)
        halo = QtGui.QRadialGradient(0, 0, 160)
        halo.setColorAt(0, QtGui.QColor("#E0EDFF"))
        halo.setColorAt(.65, QtGui.QColor("#ECF3FC"))
        halo.setColorAt(1, QtGui.QColor(245, 245, 247, 0))
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(halo)
        p.drawEllipse(QtCore.QPointF(), 165, 150)
        for radius, color in ((116, "#D9E0EA"), (93, "#CFD9E7"), (67, "#A8C4E9")):
            p.setPen(QtGui.QPen(QtGui.QColor(color), 1))
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawEllipse(QtCore.QPointF(), radius, radius)
        # The lens is a brand motif, not a fabricated camera preview.
        lens = QtGui.QLinearGradient(-45, -50, 50, 60)
        lens.setColorAt(0, QtGui.QColor("#89BFFF"))
        lens.setColorAt(.5, QtGui.QColor("#0066CC"))
        lens.setColorAt(1, QtGui.QColor("#00366D"))
        p.setPen(QtGui.QPen(QtGui.QColor("#0059B2"), 1))
        p.setBrush(lens)
        p.drawEllipse(QtCore.QPointF(), 46, 46)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 100), 1))
        p.setBrush(QtGui.QColor(255, 255, 255, 30))
        p.drawEllipse(QtCore.QPointF(-8, -10), 22, 22)
        pen = QtGui.QPen(QtGui.QColor("#0066CC"), 2)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        for x, y, sx, sy in ((-136, -105, 1, 1), (136, -105, -1, 1), (-136, 105, 1, -1), (136, 105, -1, -1)):
            path = QtGui.QPainterPath(QtCore.QPointF(x + sx * 22, y))
            path.lineTo(x, y)
            path.lineTo(x, y + sy * 22)
            p.drawPath(path)
        p.setPen(QtGui.QPen(QtGui.QColor("#A0B0C2"), 1))
        for x in (-164, 164):
            p.drawLine(QtCore.QPointF(x - 5, 0), QtCore.QPointF(x + 5, 0))
            p.drawLine(QtCore.QPointF(x, -5), QtCore.QPointF(x, 5))
        p.end()
