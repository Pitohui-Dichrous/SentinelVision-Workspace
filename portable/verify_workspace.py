"""Verify every delivered workspace file against the portable SHA-256 manifest."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from project_paths import PROJECT_ROOT


MANIFEST_PATH = PROJECT_ROOT / "INTEGRITY" / "workspace_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    print("SentinelVision 工作库完整性校验")
    print("工作库：%s" % PROJECT_ROOT)
    if not MANIFEST_PATH.is_file():
        print("[错误] 缺少完整性清单：%s" % MANIFEST_PATH)
        return 2

    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8-sig"))
        entries = manifest["files"]
        if manifest.get("schema_version") != 1 or not isinstance(entries, list):
            raise ValueError("不支持的清单格式")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print("[错误] 无法读取完整性清单：%s" % exc)
        return 2

    failures = []
    total = len(entries)
    for index, entry in enumerate(entries, 1):
        try:
            relative = Path(entry["path"])
            candidate = (PROJECT_ROOT / relative).resolve()
            candidate.relative_to(PROJECT_ROOT.resolve())
            if not candidate.is_file():
                failures.append("缺失：%s" % relative.as_posix())
                continue
            if candidate.stat().st_size != int(entry["size"]):
                failures.append("大小异常：%s" % relative.as_posix())
                continue
            if sha256_file(candidate) != str(entry["sha256"]).lower():
                failures.append("内容异常：%s" % relative.as_posix())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append("无法检查：%s (%s)" % (entry.get("path", "?"), exc))
        if index % 2000 == 0 or index == total:
            print("进度：%d / %d，异常：%d" % (index, total, len(failures)))

    print("=" * 60)
    if failures:
        print("[失败] 发现 %d 个损坏或缺失文件。" % len(failures))
        for message in failures[:100]:
            print("- %s" % message)
        if len(failures) > 100:
            print("- 其余 %d 项未显示" % (len(failures) - 100))
        print("不要继续训练；请从 D 盘源工作库重新同步，或联系维护者。")
        return 1

    expected_bytes = sum(int(entry["size"]) for entry in entries)
    print("[通过] %d 个文件、%.2f GB 均与 SHA-256 清单一致。" % (
        total, expected_bytes / (1024 ** 3)
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
