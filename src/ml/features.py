"""Rolling, normalized telemetry features for peer failure discrimination."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class _Event:
    inter_arrival_ms: float | None
    rtt_ms: float
    received: bool
    missed_run: int
    term_delta: int
    log_lag_delta: int
    witness_fraction: float
    local_health_lag_ms: float


class TelemetryExtractor:
    """Maintain W observations per peer and emit ten bounded features."""

    def __init__(self, window_size: int = 20, heartbeat_interval_ms: float = 100.0,
                 log_lag_scale: float = 20.0) -> None:
        if window_size < 2 or heartbeat_interval_ms <= 0 or log_lag_scale <= 0:
            raise ValueError("invalid telemetry normalization settings")
        self.window_size = window_size
        self.heartbeat_interval_ms = float(heartbeat_interval_ms)
        self.log_lag_scale = float(log_lag_scale)
        self._events: dict[str, deque[_Event]] = defaultdict(
            lambda: deque(maxlen=self.window_size))
        self._last_received: dict[str, float] = {}
        self._missed_runs: dict[str, int] = defaultdict(int)

    def observe(self, peer_id: str, timestamp_ms: float, *, received: bool = True,
                rtt_ms: float = 0.0, term_delta: int = 0, log_lag_delta: int = 0,
                witnesses_alive: int = 0, witnesses_queried: int = 0,
                local_health_lag_ms: float = 0.0) -> np.ndarray:
        if min(rtt_ms, local_health_lag_ms) < 0:
            raise ValueError("latencies must be non-negative")
        previous = self._last_received.get(peer_id)
        interval = timestamp_ms - previous if received and previous is not None else None
        if received:
            self._last_received[peer_id] = float(timestamp_ms)
            self._missed_runs[peer_id] = 0
        else:
            self._missed_runs[peer_id] += 1
        witness_fraction = (witnesses_alive / witnesses_queried
                            if witnesses_queried else 0.0)
        event = _Event(interval, float(rtt_ms), received, self._missed_runs[peer_id],
                       max(0, term_delta), max(0, log_lag_delta),
                       float(np.clip(witness_fraction, 0.0, 1.0)),
                       float(local_health_lag_ms))
        self._events[peer_id].append(event)
        return self.vector(peer_id)

    def vector(self, peer_id: str) -> np.ndarray:
        events = tuple(self._events.get(peer_id, ()))
        if not events:
            return np.zeros(10, dtype=np.float64)
        intervals = np.asarray([event.inter_arrival_ms for event in events
                                if event.inter_arrival_ms is not None], dtype=float)
        rtts = np.asarray([event.rtt_ms for event in events if event.received], dtype=float)
        scale = self.heartbeat_interval_ms
        mean_interval = float(intervals.mean()) if intervals.size else 0.0
        std_interval = float(intervals.std()) if intervals.size else 0.0
        mean_rtt = float(rtts.mean()) if rtts.size else 0.0
        jitter = float(rtts.std()) if rtts.size else 0.0
        latest = events[-1]
        values = np.asarray([
            mean_interval / (2 * scale), std_interval / scale,
            mean_rtt / scale, jitter / scale,
            sum(not event.received for event in events) / len(events),
            latest.missed_run / self.window_size, latest.term_delta / 5.0,
            latest.log_lag_delta / self.log_lag_scale,
            latest.witness_fraction, latest.local_health_lag_ms / scale,
        ], dtype=np.float64)
        return np.clip(values, 0.0, 1.0)

    def reset_peer(self, peer_id: str) -> None:
        self._events.pop(peer_id, None)
        self._last_received.pop(peer_id, None)
        self._missed_runs.pop(peer_id, None)
