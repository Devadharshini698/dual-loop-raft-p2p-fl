"""Delayed protocol-evidence labels for crash-versus-link discrimination."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from src.core.clock_network import SimNetwork
from src.core.types import FaultType


class FailureLabel(IntEnum):
    UNRESOLVED = -1
    LINK_FAILURE = 0
    CRASH = 1
    HEALTHY_SLOW = 2


@dataclass(slots=True)
class WitnessEvidence:
    reachable: bool
    target_alive: bool
    boot_id: int | None
    timestamp_ms: float


@dataclass(slots=True)
class PendingSuspicion:
    started_ms: float
    peer_id: str
    features: np.ndarray
    known_boot_id: int
    known_log_index: int
    heartbeat_interval_ms: float
    expected_witnesses: int
    witnesses: dict[str, WitnessEvidence] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ResolvedLabel:
    started_ms: float
    resolved_ms: float
    peer_id: str
    features: np.ndarray
    label: FailureLabel
    confidence_weight: float


@dataclass(slots=True, frozen=True)
class LabelQuality:
    evaluated: int
    correct: int
    noise_rate: float

    @property
    def agreement(self) -> float:
        return 1.0 - self.noise_rate


class ProtocolSelfLabeler:
    """Freeze suspicions and resolve them from later protocol evidence."""

    def __init__(self, horizon_ms: float = 2_000.0) -> None:
        if not np.isfinite(horizon_ms) or horizon_ms <= 0:
            raise ValueError("horizon must be positive")
        self.horizon_ms = float(horizon_ms)
        self.pending_suspicions: dict[str, PendingSuspicion] = {}
        self.resolved: list[ResolvedLabel] = []
        self.training_labels: list[ResolvedLabel] = []

    def suspect(self, timestamp_ms: float, peer_id: str, features: np.ndarray,
                known_boot_id: int, known_log_index: int = -1,
                heartbeat_interval_ms: float = 100.0,
                expected_witnesses: int = 3) -> None:
        if peer_id in self.pending_suspicions:
            raise ValueError(f"suspicion already pending for {peer_id}")
        vector = np.asarray(features, dtype=np.float64)
        settings = (timestamp_ms, heartbeat_interval_ms)
        if (not peer_id or vector.shape != (10,) or not np.all(np.isfinite(vector))
                or not all(np.isfinite(settings)) or heartbeat_interval_ms <= 0
                or known_boot_id < 0 or known_log_index < -1 or expected_witnesses < 1):
            raise ValueError("invalid suspicion snapshot")
        vector = vector.copy()
        vector.setflags(write=False)
        self.pending_suspicions[peer_id] = PendingSuspicion(
            float(timestamp_ms), peer_id, vector, known_boot_id,
            known_log_index, float(heartbeat_interval_ms), expected_witnesses)

    def observe_direct(self, timestamp_ms: float, peer_id: str, boot_id: int,
                       log_index: int, heartbeat_delay_ms: float | None = None,
                       local_health_lag_ms: float = 0.0,
                       jitter_ms: float = 0.0) -> ResolvedLabel | None:
        pending = self.pending_suspicions.get(peer_id)
        if pending is None or timestamp_ms > pending.started_ms + self.horizon_ms:
            return None
        values = (timestamp_ms, local_health_lag_ms, jitter_ms)
        if (not all(np.isfinite(values)) or timestamp_ms < pending.started_ms
                or local_health_lag_ms < 0 or jitter_ms < 0 or boot_id < 0
                or log_index < -1):
            raise ValueError("invalid direct protocol evidence")
        if heartbeat_delay_ms is not None and (not np.isfinite(heartbeat_delay_ms)
                                               or heartbeat_delay_ms < 0):
            raise ValueError("heartbeat delay must be non-negative")
        if boot_id > pending.known_boot_id:
            return self._finish(pending, timestamp_ms, FailureLabel.CRASH, 1.0)
        if boot_id != pending.known_boot_id or log_index < pending.known_log_index:
            return None
        delay = heartbeat_delay_ms if heartbeat_delay_ms is not None else timestamp_ms - pending.started_ms
        interval = pending.heartbeat_interval_ms
        slowdown = local_health_lag_ms >= 0.25 * interval or jitter_ms >= 0.1 * interval
        label = (FailureLabel.HEALTHY_SLOW
                 if interval < delay <= 1.5 * interval and slowdown
                 else FailureLabel.LINK_FAILURE)
        confidence = self._confidence(1.0, timestamp_ms - pending.started_ms, direct=True)
        return self._finish(pending, timestamp_ms, label, confidence)

    def observe_witness(self, timestamp_ms: float, peer_id: str, witness_id: str,
                        reachable: bool, target_alive: bool,
                        boot_id: int | None) -> ResolvedLabel | None:
        pending = self.pending_suspicions.get(peer_id)
        if pending is None or timestamp_ms > pending.started_ms + self.horizon_ms:
            return None
        if (not witness_id or not np.isfinite(timestamp_ms)
                or timestamp_ms < pending.started_ms or (target_alive and not reachable)
                or (target_alive and boot_id is None) or (boot_id is not None and boot_id < 0)):
            raise ValueError("invalid witness protocol evidence")
        pending.witnesses[witness_id] = WitnessEvidence(
            reachable, target_alive, boot_id, float(timestamp_ms))
        matching = [item for item in pending.witnesses.values()
                    if item.reachable and item.target_alive
                    and item.boot_id == pending.known_boot_id]
        if not matching:
            return None
        reachable_count = sum(item.reachable for item in pending.witnesses.values())
        agreement = len(matching) / max(1, reachable_count)
        confidence = self._confidence(agreement, timestamp_ms - pending.started_ms)
        return self._finish(pending, timestamp_ms, FailureLabel.LINK_FAILURE, confidence)

    def expire(self, timestamp_ms: float) -> list[ResolvedLabel]:
        if not np.isfinite(timestamp_ms):
            raise ValueError("expiration timestamp must be finite")
        expired = [item for item in self.pending_suspicions.values()
                   if timestamp_ms >= item.started_ms + self.horizon_ms]
        results = []
        for pending in expired:
            reachable = [item for item in pending.witnesses.values() if item.reachable]
            complete = len(pending.witnesses) >= pending.expected_witnesses
            all_answered = len(reachable) >= pending.expected_witnesses
            all_failed = complete and all_answered and all(
                not item.target_alive for item in reachable)
            label = FailureLabel.CRASH if all_failed else FailureLabel.UNRESOLVED
            agreement = len(reachable) / max(1, pending.expected_witnesses)
            confidence = self._confidence(agreement, self.horizon_ms)
            results.append(self._finish(pending, timestamp_ms, label, confidence))
        return results

    def _confidence(self, agreement: float, latency_ms: float,
                    direct: bool = False) -> float:
        freshness = 1.0 - min(1.0, max(0.0, latency_ms) / self.horizon_ms)
        value = 0.1 + 0.6 * np.clip(agreement, 0.0, 1.0) + 0.3 * freshness
        if direct:
            value = max(value, 0.8)
        return float(np.clip(value, 0.1, 1.0))

    def _finish(self, pending: PendingSuspicion, resolved_ms: float,
                label: FailureLabel, confidence: float) -> ResolvedLabel:
        features = pending.features.copy()
        features.setflags(write=False)
        result = ResolvedLabel(pending.started_ms, float(resolved_ms), pending.peer_id,
                               features, label, confidence)
        self.pending_suspicions.pop(pending.peer_id, None)
        self.resolved.append(result)
        if label is not FailureLabel.UNRESOLVED:
            self.training_labels.append(result)
        return result


class LabelQualityEvaluator:
    """Compare protocol labels with the simulator's injected-fault history."""

    def evaluate(self, labels: list[ResolvedLabel], network: SimNetwork) -> LabelQuality:
        pairs = [(label, self.ground_truth(label, network)) for label in labels]
        evaluable = [(label, truth) for label, truth in pairs if truth is not None]
        correct = sum(label.label is truth for label, truth in evaluable)
        total = len(evaluable)
        return LabelQuality(total, correct, 1.0 - correct / total if total else 0.0)

    def ground_truth(self, label: ResolvedLabel,
                     network: SimNetwork) -> FailureLabel | None:
        faults = [fault for fault in network.faults
                  if label.peer_id in fault.target_nodes
                  and fault.start_time <= label.resolved_ms
                  and fault.end_time >= label.started_ms]
        if any(fault.fault_type is FaultType.NODE_CRASH for fault in faults):
            return FailureLabel.CRASH
        if any(fault.fault_type in (FaultType.LINK_DROP, FaultType.ASYMMETRIC_LOSS)
               for fault in faults):
            return FailureLabel.LINK_FAILURE
        if any(fault.fault_type is FaultType.LINK_DELAY for fault in faults):
            return FailureLabel.HEALTHY_SLOW
        return None
