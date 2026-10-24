"""Lightweight failure-detector baselines with a shared heartbeat API."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from src.ml.fast_loop import SplitFailureClassifier


class FixedTimeoutDetector:
    """Standard randomized Raft election timeout."""

    def __init__(self, seed: int = 42, timeout_min_ms: float = 150.0,
                 timeout_max_ms: float = 300.0) -> None:
        if (not np.isfinite(timeout_min_ms) or not np.isfinite(timeout_max_ms)
                or timeout_min_ms <= 0 or timeout_min_ms > timeout_max_ms):
            raise ValueError("invalid timeout range")
        self.rng = np.random.default_rng(seed)
        self.timeout_min_ms = float(timeout_min_ms)
        self.timeout_max_ms = float(timeout_max_ms)
        self.last_heartbeat_ms = 0.0
        self.current_timeout_ms = self._sample_timeout()

    def observe_heartbeat(self, timestamp_ms: float, **_: float) -> None:
        self.last_heartbeat_ms = float(timestamp_ms)
        self.current_timeout_ms = self._sample_timeout()

    def should_suspect(self, timestamp_ms: float, **_: float) -> bool:
        return timestamp_ms - self.last_heartbeat_ms >= self.current_timeout_ms

    def suspicion_score(self, timestamp_ms: float) -> float:
        elapsed = max(0.0, timestamp_ms - self.last_heartbeat_ms)
        return float(np.clip(elapsed / self.current_timeout_ms, 0.0, 1.0))

    def _sample_timeout(self) -> float:
        return float(self.rng.uniform(self.timeout_min_ms, self.timeout_max_ms))


class PhiAccrualDetector:
    """Normal-model phi accrual detector over recent heartbeat intervals."""

    def __init__(self, threshold: float = 3.0, window_size: int = 100,
                 minimum_std_ms: float = 10.0) -> None:
        if (not np.isfinite(threshold) or not np.isfinite(minimum_std_ms)
                or threshold <= 0 or window_size < 2 or minimum_std_ms <= 0):
            raise ValueError("invalid phi detector settings")
        self.threshold = float(threshold)
        self.minimum_std_ms = float(minimum_std_ms)
        self.intervals: deque[float] = deque(maxlen=window_size)
        self.last_heartbeat_ms: float | None = None

    def observe_heartbeat(self, timestamp_ms: float, **_: float) -> None:
        timestamp = float(timestamp_ms)
        if self.last_heartbeat_ms is not None:
            self.intervals.append(max(0.0, timestamp - self.last_heartbeat_ms))
        self.last_heartbeat_ms = timestamp

    def phi(self, timestamp_ms: float) -> float:
        if self.last_heartbeat_ms is None or len(self.intervals) < 2:
            return 0.0
        elapsed = max(0.0, timestamp_ms - self.last_heartbeat_ms)
        values = np.asarray(self.intervals)
        mean, std = float(values.mean()), max(float(values.std()), self.minimum_std_ms)
        survival = 0.5 * math.erfc((elapsed - mean) / (std * math.sqrt(2.0)))
        return -math.log10(max(survival, 1e-12))

    def should_suspect(self, timestamp_ms: float, **_: float) -> bool:
        return self.phi(timestamp_ms) >= self.threshold

    def suspicion_score(self, timestamp_ms: float) -> float:
        return float(np.clip(self.phi(timestamp_ms) / self.threshold, 0.0, 1.0))


class SWIMLifeguardDetector:
    """Timeout detector with witness refutation and local-health multiplier."""

    def __init__(self, base_timeout_ms: float = 200.0,
                 max_multiplier: float = 8.0) -> None:
        if (not np.isfinite(base_timeout_ms) or not np.isfinite(max_multiplier)
                or base_timeout_ms <= 0 or max_multiplier < 1):
            raise ValueError("invalid Lifeguard settings")
        self.base_timeout_ms = float(base_timeout_ms)
        self.max_multiplier = float(max_multiplier)
        self.last_heartbeat_ms = 0.0
        self.local_health_lag_ms = 0.0

    def observe_heartbeat(self, timestamp_ms: float,
                          local_health_lag_ms: float = 0.0, **_: float) -> None:
        self.last_heartbeat_ms = float(timestamp_ms)
        self.local_health_lag_ms = max(0.0, float(local_health_lag_ms))

    @property
    def timeout_ms(self) -> float:
        multiplier = 1.0 + self.local_health_lag_ms / self.base_timeout_ms
        return self.base_timeout_ms * min(self.max_multiplier, multiplier)

    def should_suspect(self, timestamp_ms: float, witnesses_alive: int = 0,
                       witnesses_queried: int = 0, **_: float) -> bool:
        timed_out = timestamp_ms - self.last_heartbeat_ms >= self.timeout_ms
        witness_majority = (witnesses_queried > 0
                            and witnesses_alive > witnesses_queried // 2)
        return timed_out and not witness_majority

    def suspicion_score(self, timestamp_ms: float) -> float:
        elapsed = max(0.0, timestamp_ms - self.last_heartbeat_ms)
        return float(np.clip(elapsed / self.timeout_ms, 0.0, 1.0))


class BanditTimeoutTuner:
    """LinUCB selector over conservative timeout multipliers."""

    def __init__(self, seed: int = 42, alpha: float = 0.5,
                 base_timeout_ms: float = 150.0) -> None:
        if (not np.isfinite(alpha) or not np.isfinite(base_timeout_ms)
                or alpha < 0 or base_timeout_ms <= 0):
            raise ValueError("invalid bandit settings")
        self.rng = np.random.default_rng(seed)
        self.alpha = float(alpha)
        self.base_timeout_ms = float(base_timeout_ms)
        self.multipliers = np.asarray([1.0, 1.5, 2.5])
        self.A = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
        self.b = np.zeros((3, 3))
        self.last_action = 0
        self.last_context = np.ones(3)
        self.last_heartbeat_ms = 0.0
        self.current_timeout_ms = base_timeout_ms

    def select_timeout(self, context: np.ndarray, safety_floor_ms: float) -> float:
        vector = np.asarray(context, dtype=float)
        if (vector.shape != (3,) or not np.all(np.isfinite(vector))
                or not np.isfinite(safety_floor_ms) or safety_floor_ms <= 0):
            raise ValueError("context and safety floor must be valid")
        scores = []
        for action in range(3):
            inverse = np.linalg.inv(self.A[action])
            theta = inverse @ self.b[action]
            uncertainty = self.alpha * np.sqrt(vector @ inverse @ vector)
            scores.append(float(theta @ vector + uncertainty))
        self.last_action = int(np.argmax(scores))
        self.last_context = vector.copy()
        chosen = self.base_timeout_ms * self.multipliers[self.last_action]
        self.current_timeout_ms = max(float(safety_floor_ms), float(chosen))
        return self.current_timeout_ms

    def update(self, reward: float) -> None:
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        action, vector = self.last_action, self.last_context
        self.A[action] += np.outer(vector, vector)
        self.b[action] += float(reward) * vector

    def observe_heartbeat(self, timestamp_ms: float, **_: float) -> None:
        self.last_heartbeat_ms = float(timestamp_ms)

    def should_suspect(self, timestamp_ms: float, **_: float) -> bool:
        return timestamp_ms - self.last_heartbeat_ms >= self.current_timeout_ms

    def suspicion_score(self, timestamp_ms: float) -> float:
        elapsed = max(0.0, timestamp_ms - self.last_heartbeat_ms)
        return float(np.clip(elapsed / self.current_timeout_ms, 0.0, 1.0))


class OfflineStaticMLDetector:
    """Classifier trained once on Regime A and frozen afterward."""

    def __init__(self, seed: int = 42) -> None:
        self.classifier = SplitFailureClassifier(seed=seed)
        self.fitted = False

    def fit(self, features: np.ndarray, labels: np.ndarray,
            epochs: int = 8, learning_rate: float = 0.05) -> None:
        x_values, y_values = np.asarray(features), np.asarray(labels, dtype=int)
        if (x_values.ndim != 2 or x_values.shape[1] != 10 or not len(x_values)
                or len(x_values) != len(y_values) or not np.all(np.isfinite(x_values))
                or np.any(y_values < 0) or np.any(y_values > 2)
                or epochs <= 0 or not np.isfinite(learning_rate) or learning_rate <= 0):
            raise ValueError("expected aligned feature matrix (n, 10) and labels")
        for _ in range(epochs):
            for vector, label in zip(x_values, y_values, strict=True):
                self.classifier.partial_fit(vector, int(label), 1.0, learning_rate)
        self.fitted = True

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("offline detector has not been fitted")
        values = np.asarray(features)
        if values.ndim == 1:
            return self.classifier.predict_proba(values)
        return np.vstack([self.classifier.predict_proba(vector) for vector in values])

    def predict(self, features: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(features)
        return np.argmax(probabilities, axis=-1)
