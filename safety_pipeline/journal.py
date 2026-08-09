"""Low-volume structured event journal for production and experiment audit."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Mapping


class JsonlEventJournal:
    """Append state transitions, alerts, and human reviews as JSON Lines.

    Frame-by-frame detections are deliberately not written here; those belong
    to an explicitly enabled experiment trace so 24/7 operation cannot grow a
    log at video-frame rate.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, record_type: str, payload: Mapping[str, Any]) -> bool:
        record = {
            "schema_version": 1,
            "record_type": str(record_type),
            "wall_time": time.time(),
            **dict(payload),
        }
        try:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line + "\n")
            return True
        except (OSError, TypeError, ValueError):
            return False
