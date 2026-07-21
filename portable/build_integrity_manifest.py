"""Maintainer utility: rebuild the portable workspace SHA-256 manifest."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from project_paths import PROJECT_ROOT


OUTPUT_DIR = PROJECT_ROOT / "INTEGRITY"
OUTPUT_PATH = OUTPUT_DIR / "workspace_manifest.json"
EXCLUDED_TOP_LEVEL = {".runtime", ".git", ".idea", ".agents", "INTEGRITY"}
EXCLUDED_DIRECTORY_NAMES = {"__pycache__"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".cache", ".tmp", ".partial"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def delivered_files():
    paths = []
    for directory, dirnames, filenames in os.walk(PROJECT_ROOT):
        current = Path(directory)
        relative_directory = current.relative_to(PROJECT_ROOT)
        dirnames[:] = [
            name for name in dirnames
            if name not in EXCLUDED_DIRECTORY_NAMES
            and not (relative_directory == Path(".") and name in EXCLUDED_TOP_LEVEL)
        ]
        for filename in filenames:
            path = current / filename
            relative = path.relative_to(PROJECT_ROOT)
            if relative.parts and relative.parts[0] in EXCLUDED_TOP_LEVEL:
                continue
            if path.suffix.lower() in EXCLUDED_SUFFIXES:
                continue
            paths.append((relative.as_posix(), path))
    return sorted(paths, key=lambda item: item[0].casefold())


def main() -> int:
    entries = []
    total_bytes = 0
    files = delivered_files()
    for index, (relative, path) in enumerate(files, 1):
        size = path.stat().st_size
        entries.append({"path": relative, "size": size, "sha256": sha256_file(path)})
        total_bytes += size
        if index % 2000 == 0 or index == len(files):
            print("Progress: %d / %d" % (index, len(files)), flush=True)

    document = {
        "schema_version": 1,
        "product": "SentinelVision Portable Workspace",
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "exclusions": [
            ".runtime", ".git", ".idea", ".agents", "INTEGRITY",
            "__pycache__", "*.pyc", "*.pyo", "*.cache", "*.tmp", "*.partial",
        ],
        "files": entries,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(OUTPUT_PATH)
    print("Complete: %d files, %.2f GiB" % (len(entries), total_bytes / (1024 ** 3)))
    print("Manifest: %s" % OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
