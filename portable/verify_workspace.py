"""Fast fixed-scope check for the portable SentinelVision workspace.

This intentionally does not walk the whole drive or calculate file hashes.
"""

from __future__ import annotations

import sys
from pathlib import Path

from project_paths import PROJECT_ROOT


REQUIRED_FILES = (
    "START_HERE.cmd",
    "START_SENTINEL.cmd",
    "TRAIN_SAFE.cmd",
    "VERIFY_WORKSPACE.cmd",
    "SentinelVision4.py",
    "workspace_manager.py",
    "ui_motion.py",
    "ui_theme.py",
    "ui_font.py",
    "ui_artwork.py",
    "portable_check.py",
    "safe_train.py",
    "model_catalog.py",
    "model_runtime.py",
    "deployment_manager.py",
    "pretrained_weights.py",
    "models/yolov5s.yaml",
    "RESULTS/model_registry.yaml",
    "DATASETS/combined_legacy_v1/dataset.yaml",
    "DATASETS/fire_only_legacy_v1/dataset.yaml",
    "RUNTIME/python/python.exe",
    "TOOLS/Git/cmd/git.exe",
    "TOOLS/GitHubCLI/bin/gh.exe",
)

REQUIRED_DIRECTORIES = (
    "DATASETS",
    "MODEL_ARCHIVE",
    "PRETRAINED_WEIGHTS",
    "RESULTS",
    "RUNTIME",
    "TOOLS",
    "TRAINING_OUTPUTS",
)

PRETRAINED_WEIGHTS = (
    "yolov5n.pt",
    "yolov5s.pt",
    "yolov5m.pt",
    "yolov5l.pt",
    "yolov5x.pt",
    "yolov5n6.pt",
    "yolov5s6.pt",
    "yolov5m6.pt",
    "yolov5l6.pt",
    "yolov5x6.pt",
)


def main() -> int:
    print("SentinelVision 工作库快速检查")
    print("工作库：%s" % PROJECT_ROOT)
    print("说明：仅检查固定关键文件，不遍历全盘、不计算哈希。")
    problems = []

    for relative in REQUIRED_DIRECTORIES:
        path = PROJECT_ROOT / relative
        if not path.is_dir():
            problems.append("缺少目录：%s" % relative)

    for relative in REQUIRED_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            problems.append("缺少文件：%s" % relative)
        elif path.stat().st_size <= 0:
            problems.append("空文件：%s" % relative)

    weight_directory = PROJECT_ROOT / "PRETRAINED_WEIGHTS"
    for filename in PRETRAINED_WEIGHTS:
        path = weight_directory / filename
        if not path.is_file():
            problems.append("缺少训练权重：PRETRAINED_WEIGHTS/%s" % filename)
        elif path.stat().st_size < 1024 * 1024:
            problems.append("训练权重大小明显异常：PRETRAINED_WEIGHTS/%s" % filename)

    deployed = sorted((PROJECT_ROOT / "RESULTS").glob("*/weights/best.pt"))
    if not deployed:
        problems.append("RESULTS 中没有可用的 best.pt")
    for path in deployed:
        if path.stat().st_size < 1024 * 1024:
            problems.append("部署模型大小明显异常：%s" % path.relative_to(PROJECT_ROOT).as_posix())

    executable = Path(sys.executable).resolve()
    try:
        executable.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        problems.append("当前未使用工作库自带 Python：%s" % executable)

    print("=" * 60)
    if problems:
        print("[失败] 发现 %d 个关键项异常：" % len(problems))
        for problem in problems:
            print("- %s" % problem)
        return 1

    print("[通过] 固定关键文件齐备：10 个训练权重、%d 个部署模型。" % len(deployed))
    print("[通过] 本检查未遍历数据集内容，也未计算任何 SHA-256。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
