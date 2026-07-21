"""Load a Chinese-capable UI font without depending on the Windows locale."""

from pathlib import Path

from PySide6 import QtGui

from project_paths import PROJECT_ROOT


def install_ui_font(application):
    candidates = [
        PROJECT_ROOT / "assets" / "NotoSansSC-VF.ttf",
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    for path in candidates:
        if not path.is_file():
            continue
        font_id = QtGui.QFontDatabase.addApplicationFont(str(path))
        if font_id < 0:
            continue
        families = QtGui.QFontDatabase.applicationFontFamilies(font_id)
        if families:
            application.setFont(QtGui.QFont(families[0], 10))
            return families[0]
    return application.font().family()


__all__ = ["install_ui_font"]
