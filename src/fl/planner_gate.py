"""Safety-bounded Raft timeout planning and provisional FL aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True, frozen=True)
class TimeoutDecision:
    timeout_ms: float
    mode: str
    start_election: bool
    request_witness_probe: bool
    keep_current_leader: bool
    activate_provisional_leader: bool
    provisional_term: int | None


class SafeRaftPlanner:
    """Convert advisory probabilities into liveness-only Raft actions."""

    def __init__(self, heartbeat_interval_ms: float, mean_rtt_ms: float,
                 std_rtt_ms: float, seed: int = 42,
                 base_timeout_min: float = 150.0,
                 base_timeout_max: float = 300.0,
                 max_timeout: float = 2_000.0) -> None:
        values = (heartbeat_interval_ms, mean_rtt_ms, std_rtt_ms,
                  base_timeout_min, base_timeout_max, max_timeout)
        if (not all(np.isfinite(values)) or heartbeat_interval_ms <= 0
                or mean_rtt_ms < 0 or std_rtt_ms < 0 or base_timeout_min <= 0
                or base_timeout_min > base_timeout_max or max_timeout <= 0):
            raise ValueError("invalid Raft planner timing settings")
        self.heartbeat_interval_ms = float(heartbeat_interval_ms)
        self.mean_rtt_ms = float(mean_rtt_ms)
        self.std_rtt_ms = float(std_rtt_ms)
        self.base_timeout_min = float(base_timeout_min)
        self.base_timeout_max = float(base_timeout_max)
        self.max_timeout = float(max_timeout)
        self.rng = np.random.default_rng(seed)
        self.last_decision: TimeoutDecision | None = None

    @property
    def safety_floor_ms(self) -> float:
        return self.heartbeat_interval_ms + self.mean_rtt_ms + 3.0 * self.std_rtt_ms

    def plan(self, probs: np.ndarray, is_ready: bool, entropy: float,
             current_term: int, classifier_enabled: bool = True,
             witness_alive: bool | None = None,
             base_timeout: float | None = None) -> TimeoutDecision:
        probabilities = np.asarray(probs, dtype=float)
        if (probabilities.shape != (3,) or not np.all(np.isfinite(probabilities))
                or np.any(probabilities < 0)
                or not np.isclose(probabilities.sum(), 1.0)):
            raise ValueError("probs must be a normalized shape-(3,) vector")
        if (not np.isfinite(entropy) or not 0 <= entropy <= 1
                or current_term < 0):
            raise ValueError("entropy and current term are invalid")
        if base_timeout is not None and (not np.isfinite(base_timeout)
                                         or base_timeout <= 0):
            raise ValueError("base timeout must be positive")
        fallback = not is_ready or not classifier_enabled or entropy > 0.75
        if fallback:
            decision = self._fallback(current_term)
        elif probabilities[1] >= 0.60:
            timeout = max(self.safety_floor_ms, 0.75 * self.base_timeout_min)
            decision = TimeoutDecision(timeout, "likely_crash", True, False,
                                       False, True, current_term)
        elif probabilities[0] >= 0.60:
            baseline = self.base_timeout_min if base_timeout is None else base_timeout
            timeout = min(self.max_timeout, 2.5 * baseline)
            timeout = max(self.safety_floor_ms, timeout)
            decision = TimeoutDecision(timeout, "likely_link", False, True,
                                       witness_alive is True, False, None)
        else:
            decision = self._fallback(current_term)
        self.last_decision = decision
        return decision

    def get_timeout_decision(self, node: Any, leader_id: str | None) -> float:
        if self.last_decision is not None:
            return self.last_decision.timeout_ms
        decision = self._fallback(int(getattr(node, "current_term", 0)))
        self.last_decision = decision
        return decision.timeout_ms

    def _fallback(self, current_term: int) -> TimeoutDecision:
        sampled = float(self.rng.uniform(self.base_timeout_min, self.base_timeout_max))
        timeout = max(self.safety_floor_ms, sampled)
        return TimeoutDecision(timeout, "fallback", False, False,
                               False, True, current_term)


@dataclass(slots=True)
class ProvisionalAggregatorState:
    active: bool = False
    term: int | None = None
    buffered_updates: list[tuple[str, Any]] = field(default_factory=list)

    def activate(self, term: int) -> None:
        if self.term != term:
            self.buffered_updates.clear()
        self.active, self.term = True, term

    def buffer_update(self, client_id: str, update: Any,
                      term: int | None = None) -> bool:
        expected = self.term if term is None else term
        if not self.active or expected != self.term:
            return False
        self.buffered_updates.append((client_id, update))
        return True

    def handover(self, elected_term: int) -> list[tuple[str, Any]]:
        acceptable = self.term is not None and elected_term in (self.term, self.term + 1)
        updates = list(self.buffered_updates) if acceptable else []
        self.buffered_updates.clear()
        self.active, self.term = False, None
        return updates

    def discard(self) -> None:
        self.buffered_updates.clear()
        self.active, self.term = False, None


class ProvisionalLeaderGate:
    """Apply planner decisions to provisional aggregation state."""

    def __init__(self, planner: SafeRaftPlanner,
                 state: ProvisionalAggregatorState | None = None) -> None:
        self.planner = planner
        self.state = state or ProvisionalAggregatorState()
        self.activations = 0
        self.suppressed_handovers = 0

    def evaluate(self, probs: np.ndarray, is_ready: bool, entropy: float,
                 current_term: int, classifier_enabled: bool = True,
                 witness_alive: bool | None = None,
                 base_timeout: float | None = None) -> TimeoutDecision:
        decision = self.planner.plan(probs, is_ready, entropy, current_term,
                                     classifier_enabled, witness_alive, base_timeout)
        if decision.activate_provisional_leader:
            self.state.activate(current_term)
            self.activations += 1
        elif decision.mode == "likely_link":
            self.state.discard()
            self.suppressed_handovers += 1
        return decision
