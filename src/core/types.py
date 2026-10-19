"""Compact shared types for simulation, Raft, and telemetry."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class MessageType(Enum):
    HEARTBEAT = auto()
    HEARTBEAT_ACK = auto()
    VOTE_REQ = auto()
    VOTE_RESP = auto()
    WITNESS_REQ = auto()
    WITNESS_RESP = auto()
    MODEL_SYNC = auto()


class FaultType(Enum):
    LINK_DROP = auto()
    LINK_DELAY = auto()
    ASYMMETRIC_LOSS = auto()
    NODE_CRASH = auto()
    DRIFT_SHIFT = auto()


@dataclass(slots=True)
class Message:
    sender_id: str
    receiver_id: str
    term: int
    msg_type: MessageType
    boot_id: int = 0
    log_index: int = -1
    log_term: int = 0
    peer_heard_map: dict[str, float] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FaultEvent:
    start_time: float
    end_time: float
    fault_type: FaultType
    target_nodes: tuple[str, ...]
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class LogEntry:
    term: int
    value: Any


@dataclass(slots=True)
class PeerState:
    peer_id: str
    last_heard_ms: float = float("-inf")
    boot_id: int = 0


@dataclass(slots=True)
class TelemetrySnapshot:
    timestamp_ms: float
    peer_id: str
    rtt_ms: float = 0.0
    jitter_ms: float = 0.0
    loss_ratio: float = 0.0
    missed_heartbeats: int = 0
    local_tick_lag_ms: float = 0.0


@dataclass(slots=True, frozen=True)
class DeliveryRecord:
    sender_id: str
    receiver_id: str
    sent_at: float
    delivery_at: float | None
    dropped: bool
    reason: str = ""
