"""Virtual time, deterministic messaging, and scheduled fault injection."""

from __future__ import annotations

import heapq
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .types import DeliveryRecord, FaultEvent, FaultType, Message


Callback = Callable[[], None]
Receiver = Callable[[Message], None]


class VirtualClock:
    """Monotonic millisecond clock driven by a stable event queue."""

    def __init__(self) -> None:
        self.now_ms = 0.0
        self._queue: list[tuple[float, int, Callback]] = []
        self._sequence = 0

    def call_at(self, timestamp_ms: float, callback: Callback) -> None:
        if timestamp_ms < self.now_ms:
            raise ValueError("cannot schedule an event in the past")
        self._sequence += 1
        heapq.heappush(self._queue, (float(timestamp_ms), self._sequence, callback))

    def call_later(self, delay_ms: float, callback: Callback) -> None:
        if delay_ms < 0:
            raise ValueError("delay must be non-negative")
        self.call_at(self.now_ms + delay_ms, callback)

    def advance_to(self, timestamp_ms: float, max_events: int | None = None) -> int:
        if timestamp_ms < self.now_ms:
            raise ValueError("virtual time is monotonic")
        count = 0
        while self._queue and self._queue[0][0] <= timestamp_ms:
            if max_events is not None and count >= max_events:
                break
            when, _, callback = heapq.heappop(self._queue)
            self.now_ms = when
            callback()
            count += 1
        if max_events is None or count < max_events:
            self.now_ms = float(timestamp_ms)
        return count

    def run(self, max_events: int | None = None) -> int:
        count = 0
        while self._queue and (max_events is None or count < max_events):
            when = self._queue[0][0]
            count += self.advance_to(when, 1)
        return count


@dataclass(slots=True)
class LinkProfile:
    mean_ms: float = 10.0
    sigma_ms: float = 0.0
    loss_ratio: float = 0.0


class SimNetwork:
    """Seeded message bus with link faults and persistent incarnations."""

    def __init__(self, clock: VirtualClock, seed: int = 42) -> None:
        self.clock = clock
        self.rng = np.random.default_rng(seed)
        self.default_link = LinkProfile()
        self.links: dict[tuple[str, str], LinkProfile] = {}
        self.receivers: dict[str, Receiver] = {}
        self.reboot_hooks: dict[str, Callable[[int], None]] = {}
        self.boot_ids: dict[str, int] = {}
        self.crashed: set[str] = set()
        self.faults: list[FaultEvent] = []
        self.records: list[DeliveryRecord] = []

    def register(self, node_id: str, receiver: Receiver,
                 on_reboot: Callable[[int], None] | None = None) -> None:
        self.receivers[node_id] = receiver
        self.boot_ids.setdefault(node_id, 0)
        if on_reboot is not None:
            self.reboot_hooks[node_id] = on_reboot

    def set_link(self, source: str, target: str, mean_ms: float = 10.0,
                 sigma_ms: float = 0.0, loss_ratio: float = 0.0) -> None:
        if mean_ms < 0 or sigma_ms < 0 or not 0 <= loss_ratio <= 1:
            raise ValueError("invalid link profile")
        self.links[(source, target)] = LinkProfile(mean_ms, sigma_ms, loss_ratio)

    def add_fault(self, event: FaultEvent) -> None:
        if event.end_time < event.start_time:
            raise ValueError("fault end precedes start")
        self.faults.append(event)
        if event.fault_type is FaultType.DRIFT_SHIFT:
            self.clock.call_at(event.start_time, lambda e=event: self._apply_drift(e))
        elif event.fault_type is FaultType.NODE_CRASH:
            self.clock.call_at(event.start_time, lambda e=event: self._crash_targets(e))
            self.clock.call_at(event.end_time, lambda e=event: self._reboot_targets(e))

    def active_faults(self, timestamp_ms: float | None = None) -> tuple[FaultEvent, ...]:
        now = self.clock.now_ms if timestamp_ms is None else timestamp_ms
        return tuple(f for f in self.faults if f.start_time <= now < f.end_time)

    def send(self, message: Message) -> bool:
        sent = self.clock.now_ms
        reason, extra_delay, extra_loss = self._fault_effect(message)
        profile = self.links.get((message.sender_id, message.receiver_id), self.default_link)
        unavailable = message.sender_id in self.crashed or message.receiver_id in self.crashed
        drop = unavailable or bool(reason) or self.rng.random() < max(profile.loss_ratio, extra_loss)
        if drop:
            why = "node_crashed" if unavailable else reason or "packet_loss"
            self.records.append(DeliveryRecord(message.sender_id, message.receiver_id, sent, None, True, why))
            return False
        latency = max(0.0, float(self.rng.normal(profile.mean_ms, profile.sigma_ms))) + extra_delay
        delivery = sent + latency
        self.records.append(DeliveryRecord(message.sender_id, message.receiver_id, sent, delivery, False))
        self.clock.call_at(delivery, lambda m=message: self._deliver(m))
        return True

    def crash_node(self, node_id: str) -> None:
        self.crashed.add(node_id)

    def reboot_node(self, node_id: str) -> int:
        self.crashed.discard(node_id)
        self.boot_ids[node_id] = self.boot_ids.get(node_id, 0) + 1
        hook = self.reboot_hooks.get(node_id)
        if hook is not None:
            hook(self.boot_ids[node_id])
        return self.boot_ids[node_id]

    def _deliver(self, message: Message) -> None:
        if message.receiver_id not in self.crashed:
            receiver = self.receivers.get(message.receiver_id)
            if receiver is not None:
                receiver(message)

    def _fault_effect(self, message: Message) -> tuple[str, float, float]:
        reason, delay, loss = "", 0.0, 0.0
        edge = (message.sender_id, message.receiver_id)
        for fault in self.active_faults():
            targets = fault.target_nodes
            if fault.fault_type is FaultType.LINK_DROP and set(edge) <= set(targets):
                reason = "link_drop"
            elif fault.fault_type is FaultType.ASYMMETRIC_LOSS and edge == targets[:2]:
                loss = max(loss, float(fault.params.get("loss_ratio", 1.0)))
            elif fault.fault_type is FaultType.LINK_DELAY and set(edge) <= set(targets):
                delay += float(fault.params.get("delay_ms", 0.0))
        return reason, delay, loss

    def _apply_drift(self, event: FaultEvent) -> None:
        if len(event.target_nodes) < 2:
            return
        source, target = event.target_nodes[:2]
        current = self.links.get((source, target), self.default_link)
        self.set_link(source, target,
                      float(event.params.get("mean_ms", current.mean_ms)),
                      float(event.params.get("sigma_ms", current.sigma_ms)),
                      float(event.params.get("loss_ratio", current.loss_ratio)))

    def _crash_targets(self, event: FaultEvent) -> None:
        for node_id in event.target_nodes:
            self.crash_node(node_id)

    def _reboot_targets(self, event: FaultEvent) -> None:
        for node_id in event.target_nodes:
            self.reboot_node(node_id)
