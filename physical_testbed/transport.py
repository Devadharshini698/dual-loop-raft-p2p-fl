"""Authenticated-by-topology UDP transport for physical lab Raft messages."""

from __future__ import annotations

import json
import math
import socket
import threading
from collections.abc import Callable
from typing import Any

from src.core.types import DeliveryRecord, Message, MessageType

from .clock import MonotonicClock


def encode_message(message: Message) -> bytes:
    data = {
        "sender_id": message.sender_id,
        "receiver_id": message.receiver_id,
        "term": message.term,
        "msg_type": message.msg_type.name,
        "boot_id": message.boot_id,
        "log_index": message.log_index,
        "log_term": message.log_term,
        "peer_heard_map": message.peer_heard_map,
        "payload": message.payload,
    }
    encoded = json.dumps(data, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > 60_000:
        raise ValueError("Raft datagram exceeds the safe UDP payload budget")
    return encoded


def decode_message(payload: bytes) -> Message:
    if len(payload) > 60_000:
        raise ValueError("Raft datagram exceeds the safe UDP payload budget")
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid Raft JSON") from error
    if not isinstance(data, dict):
        raise ValueError("Raft message must be a JSON object")
    required = {"sender_id", "receiver_id", "term", "msg_type"}
    if not required <= data.keys():
        raise ValueError("incomplete Raft message")
    sender, receiver = data["sender_id"], data["receiver_id"]
    if not isinstance(sender, str) or not isinstance(receiver, str):
        raise ValueError("Raft node identifiers must be strings")
    message_type = data["msg_type"]
    if not isinstance(message_type, str):
        raise ValueError("invalid Raft message type")
    heard = _finite_float_map(data.get("peer_heard_map", {}))
    body = data.get("payload", {})
    if not isinstance(body, dict):
        raise ValueError("Raft payload must be an object")
    return Message(
        sender_id=sender, receiver_id=receiver,
        term=_bounded_int(data["term"], 0), msg_type=MessageType[message_type],
        boot_id=_bounded_int(data.get("boot_id", 0), 0),
        log_index=_bounded_int(data.get("log_index", -1), -1),
        log_term=_bounded_int(data.get("log_term", 0), 0),
        peer_heard_map=heard, payload=body,
    )


def _bounded_int(value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Raft integer field has the wrong type")
    if not minimum <= value <= 2**63 - 1:
        raise ValueError("Raft integer field is out of range")
    return value


def _finite_float_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("peer_heard_map must be an object")
    result = {str(key): float(item) for key, item in value.items()}
    if any(not math.isfinite(item) for item in result.values()):
        raise ValueError("peer_heard_map values must be finite")
    return result


class UdpRaftNetwork:
    """A SimNetwork-compatible endpoint backed by a real UDP socket."""

    def __init__(self, local_id: str, addresses: dict[str, tuple[str, int]],
                 clock: MonotonicClock, boot_id: int) -> None:
        if local_id not in addresses:
            raise KeyError(f"missing address for {local_id}")
        self.local_id, self.addresses, self.clock = local_id, dict(addresses), clock
        self.boot_ids = {local_id: int(boot_id)}
        self.crashed: set[str] = set()
        self.records: list[DeliveryRecord] = []
        self._receiver: Callable[[Message], None] | None = None
        self._closed = threading.Event()
        self._resolved = {node: (socket.gethostbyname(host), port)
                          for node, (host, port) in self.addresses.items()}
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(self.addresses[local_id])
        self._socket.settimeout(0.2)
        self._thread = threading.Thread(target=self._receive_loop,
                                        name=f"raft-udp-{local_id}", daemon=True)
        self._thread.start()

    def register(self, node_id: str, receiver: Callable[[Message], None],
                 on_reboot: Callable[[int], None] | None = None) -> None:
        if node_id != self.local_id:
            raise ValueError("a physical endpoint can register only its local node")
        self._receiver = receiver

    def send(self, message: Message) -> bool:
        if message.sender_id != self.local_id or message.receiver_id not in self.addresses:
            return False
        sent = self.clock.now_ms
        try:
            self._socket.sendto(encode_message(message), self._resolved[message.receiver_id])
        except (OSError, TypeError, ValueError):
            self.records.append(DeliveryRecord(self.local_id, message.receiver_id,
                                               sent, None, True, "socket_error"))
            return False
        self.records.append(DeliveryRecord(self.local_id, message.receiver_id,
                                           sent, sent, False, "udp"))
        return True

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._socket.close()
        self._thread.join(timeout=2)

    def _receive_loop(self) -> None:
        while not self._closed.is_set():
            try:
                payload, source = self._socket.recvfrom(65_535)
                message = decode_message(payload)
                self._accept(message, source)
            except socket.timeout:
                continue
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def _accept(self, message: Message, source: tuple[str, int]) -> None:
        expected = self._resolved.get(message.sender_id)
        if message.receiver_id != self.local_id or expected is None:
            return
        if source[0] != expected[0] or source[1] != expected[1]:
            return
        received = self.clock.now_ms
        self.records.append(DeliveryRecord(message.sender_id, self.local_id,
                                           received, received, False, "udp"))
        if self._receiver is not None:
            self.clock.call_later(0, lambda m=message: self._receiver(m))
