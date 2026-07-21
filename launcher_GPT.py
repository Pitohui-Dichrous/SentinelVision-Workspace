"""Portable launcher for the PySide6 SentinelVision desktop application."""

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    # Some YOLOv5 defaults are relative to the working directory.  Always run
    # from the project root, even when this file is opened from Explorer.
    os.chdir(str(PROJECT_ROOT))
    try:
        import SentinelVision4 as sentinel
    except ModuleNotFoundError as exc:
        print("无法启动 SentinelVision：缺少 Python 依赖 %s。" % exc.name)
        print("请双击 SETUP_ENVIRONMENT.cmd，从移动硬盘离线修复内置运行时。")
        return 1

    usable_models = sentinel.CATALOG_SNAPSHOT.selectable_models
    if not usable_models:
        print("无法启动 SentinelVision：RESULTS 中没有通过扫描校验的部署模型。")
        for model in sentinel.CATALOG_SNAPSHOT.models:
            print("- %s: %s" % (model.display_name, model.issue or "不可用"))
        print("请通过工作台检查或部署至少一个模型。")
        return 2

    print("已发现 %d 个可用部署模型：%s" % (
        len(usable_models), ", ".join(model.display_name for model in usable_models)
    ))

    sentinel.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
