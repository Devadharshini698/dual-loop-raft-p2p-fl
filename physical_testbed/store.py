"""Atomic stable state and append-only evidence storage."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StableStateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def read(self) -> dict[str, Any]:
        with self._lock:
            if not self.path.exists():
                return {}
            return json.loads(self.path.read_text(encoding="utf-8"))

    def start_incarnation(self) -> int:
        state = self.read()
        state["boot_id"] = int(state.get("boot_id", 0)) + 1
        self._write(state)
        return int(state["boot_id"])

    def save_raft(self, term: int, voted_for: str | None, log: list,
                  commit_index: int, boot_id: int) -> None:
        state = {
            "boot_id": int(boot_id), "current_term": int(term),
            "voted_for": voted_for, "commit_index": int(commit_index),
            "log": [[int(entry.term), entry.value] for entry in log],
        }
        self._write(state)

    def _write(self, state: dict[str, Any]) -> None:
        encoded = json.dumps(state, separators=(",", ":"), allow_nan=False)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with self._lock:
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)


class EvidenceLogger:
    def __init__(self, path: str | Path, run_id: str, node_id: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id, self.node_id = run_id, node_id
        self._lock = threading.Lock()

    def write(self, event: str, monotonic_ms: float, **fields: Any) -> None:
        record = {
            "schema": 1, "run_id": self.run_id, "node_id": self.node_id,
            "event": event, "utc": datetime.now(timezone.utc).isoformat(),
            "monotonic_ms": round(float(monotonic_ms), 3), **fields,
        }
        encoded = json.dumps(record, separators=(",", ":"), allow_nan=False)
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")


def log_checksum(entries: list) -> str:
    serial = [[int(entry.term), entry.value] for entry in entries]
    return hashlib.sha256(json.dumps(serial, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()
