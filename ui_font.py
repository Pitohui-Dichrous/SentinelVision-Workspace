"""Load a Chinese-capable UI font without depending on the Windows locale."""

import os
from pathlib import Path

from PySide6 import QtGui

from project_paths import PROJECT_ROOT


def install_ui_font(application):
    # Qt 6.6 can render the bundled variable CJK font with an excessively light
    # face. Prefer Windows' installed UI family, with Segoe for Latin text.
    families = QtGui.QFontDatabase.families()
    if "Microsoft YaHei UI" in families:
        font = QtGui.QFont()
        font.setFamilies(["Segoe UI", "Microsoft YaHei UI"])
        font.setPointSize(10)
        font.setWeight(QtGui.QFont.Weight.Normal)
        application.setFont(font)
        return "Segoe UI / Microsoft YaHei UI"
    windows_directory = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
    system_fonts = Path(windows_directory) / "Fonts" if windows_directory else None
    candidates = ([system_fonts / "msyh.ttc"] if system_fonts else [])
    candidates.append(PROJECT_ROOT / "assets" / "NotoSansSC-VF.ttf")
    if system_fonts:
        candidates.extend((system_fonts / "NotoSansSC-VF.ttf", system_fonts / "simhei.ttf"))
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
