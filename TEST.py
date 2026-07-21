"""Run one portable command-line detection as a smoke test."""

import subprocess
import sys

from project_paths import MODEL_PATH_COMBINED, PROJECT_ROOT


def main():
    images = PROJECT_ROOT / "data" / "helmetdata" / "images"
    source = next((path for path in images.iterdir() if path.is_file()), None)
    if source is None:
        print("测试失败：数据集中没有找到图片。")
        return 1

    command = [
        sys.executable,
        str(PROJECT_ROOT / "detect.py"),
        "--weights",
        str(MODEL_PATH_COMBINED),
        "--source",
        str(source),
        "--img",
        "640",
        "--conf",
        "0.25",
        "--project",
        str(PROJECT_ROOT / "runs" / "detect"),
        "--name",
        "portable_smoke_test",
        "--exist-ok",
    ]
    return subprocess.call(command, cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    sys.exit(main())
