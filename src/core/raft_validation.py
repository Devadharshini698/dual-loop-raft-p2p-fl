"""Fail-closed parsing helpers for untrusted Raft messages."""

from __future__ import annotations

import math
from typing import Any, Collection

from .types import LogEntry, Message, MessageType


def valid_message(message: object, receiver_id: str,
                  peers: Collection[str]) -> bool:
    """Return whether the message has a valid envelope and known sender."""
    return (
        isinstance(message, Message)
        and message.receiver_id == receiver_id
        and message.sender_id in peers
        and isinstance(message.term, int) and message.term >= 0
        and isinstance(message.boot_id, int) and message.boot_id >= 0
        and isinstance(message.log_index, int) and message.log_index >= -1
        and isinstance(message.log_term, int) and message.log_term >= 0
        and isinstance(message.msg_type, MessageType)
        and isinstance(message.payload, dict)
        and isinstance(message.peer_heard_map, dict)
    )


def parse_entries(payload: dict[str, Any]) -> list[LogEntry] | None:
    """Parse a complete log payload, rejecting partial or malformed input."""
    raw = payload.get("entries", [])
    if not isinstance(raw, list):
        return None
    try:
        entries = [LogEntry(int(item[0]), item[1]) for item in raw
                   if isinstance(item, (list, tuple)) and len(item) == 2]
    except (TypeError, ValueError):
        return None
    if len(entries) != len(raw) or any(entry.term < 0 for entry in entries):
        return None
    return entries


def parse_int(payload: dict[str, Any], key: str,
              default: int, minimum: int) -> int | None:
    """Parse a bounded integer field without accepting malformed values."""
    try:
        value = int(payload.get(key, default))
    except (TypeError, ValueError):
        return None
    return value if value >= minimum else None


def parse_timestamp(payload: dict[str, Any], key: str) -> float | None:
    """Parse a witness timestamp while rejecting NaN values."""
    try:
        value = float(payload.get(key, float("-inf")))
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value
