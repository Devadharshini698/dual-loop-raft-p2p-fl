"""Pure NumPy split classifier and fast local drift-adaptation loop."""

from __future__ import annotations

from collections import deque
from math import exp, isfinite, log

import numpy as np


class SplitFailureClassifier:
    """Two-layer shared encoder with a node-personalised softmax head."""

    def __init__(self, seed: int = 42, l2_reg: float = 1e-4,
                 default_lr: float = 0.05) -> None:
        if not np.isfinite(l2_reg) or l2_reg < 0 or not np.isfinite(default_lr) or default_lr <= 0:
            raise ValueError("regularization and learning rate settings are invalid")
        rng = np.random.default_rng(seed)
        self._weights, self._biases = np.empty(312), np.zeros(27)
        self.W1 = self._weights[:160].reshape(10, 16)
        self.W2 = self._weights[160:288].reshape(16, 8)
        self.Wh = self._weights[288:].reshape(8, 3)
        self.b1, self.b2, self.bh = self._biases[:16], self._biases[16:24], self._biases[24:]
        self.W1[:] = rng.normal(0, np.sqrt(2 / 10), self.W1.shape)
        self.W2[:] = rng.normal(0, np.sqrt(2 / 16), self.W2.shape)
        self.Wh[:] = rng.normal(0, np.sqrt(2 / 8), self.Wh.shape)
        self.l2_reg = float(l2_reg)
        self.default_lr = float(default_lr)
        self._weight_sq = self._weight_square_sum()
        self._cached_x: np.ndarray | None = None
        self._cached_forward: tuple[np.ndarray, ...] | None = None
        self._gradient_buffer = np.empty(339)
        self._d_w1 = self._gradient_buffer[:160].reshape(self.W1.shape)
        self._d_w2 = self._gradient_buffer[160:288].reshape(self.W2.shape)
        self._d_wh = self._gradient_buffer[288:312].reshape(self.Wh.shape)
        self._d_z1 = self._gradient_buffer[312:328]
        self._d_z2 = self._gradient_buffer[328:336]
        self._dlogits = self._gradient_buffer[336:]
        self._h1, self._encoded, self._probs = np.empty(16), np.empty(8), np.empty(3)
        self._mask1, self._mask2 = np.empty(16, dtype=bool), np.empty(8, dtype=bool)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self._cached_x = x
        self._cached_forward = self._forward(x)
        return self._cached_forward[-1].copy()

    def partial_fit(self, x: np.ndarray, label: int, confidence_weight: float,
                    lr: float | None = None) -> float:
        self._validate_target(label, confidence_weight)
        cached = self._cached_forward if self._cached_x is x else self._forward(x)
        rate = self.default_lr if lr is None else float(lr)
        if lr is not None and (not np.isfinite(rate) or rate <= 0):
            raise ValueError("learning rate must be positive")
        loss = self._update_from_forward(cached, label, confidence_weight, rate)
        self._cached_x, self._cached_forward = None, None
        return loss

    def get_encoder_weights(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name).copy()
                for name in ("W1", "b1", "W2", "b2")}

    def set_encoder_weights(self, weights: dict[str, np.ndarray]) -> None:
        required = ("W1", "b1", "W2", "b2")
        if set(weights) != set(required):
            raise ValueError("encoder weights must contain W1, b1, W2, and b2")
        for name in required:
            value = np.asarray(weights[name], dtype=float)
            if value.shape != getattr(self, name).shape or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid encoder parameter {name}")
        for name in required:
            getattr(self, name)[:] = np.asarray(weights[name], dtype=float)
        self._weight_sq = self._weight_square_sum()
        self._cached_x, self._cached_forward = None, None

    @staticmethod
    def predictive_entropy(probs: np.ndarray) -> float:
        values = np.asarray(probs, dtype=float)
        if values.shape != (3,) or np.any(values < 0) or not np.isclose(values.sum(), 1.0):
            raise ValueError("probabilities must be a normalized shape-(3,) vector")
        safe = np.clip(values, 1e-12, 1.0)
        return float(-np.sum(safe * np.log(safe)) / np.log(3.0))

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, ...]:
        vector = np.asarray(x, dtype=float)
        if vector.shape != (10,) or not all(map(isfinite, vector)):
            raise ValueError("x must be a finite shape-(10,) vector")
        h1, encoded, probabilities = self._h1, self._encoded, self._probs
        np.dot(vector, self.W1, out=h1)
        h1 += self.b1
        np.maximum(h1, 0.0, out=h1)
        np.dot(h1, self.W2, out=encoded)
        encoded += self.b2
        np.maximum(encoded, 0.0, out=encoded)
        np.dot(encoded, self.Wh, out=probabilities)
        probabilities += self.bh
        maximum = max(probabilities[0], probabilities[1], probabilities[2])
        first = exp(probabilities[0] - maximum)
        second = exp(probabilities[1] - maximum)
        third = exp(probabilities[2] - maximum)
        total = first + second + third
        probabilities[0] = first / total
        probabilities[1] = second / total
        probabilities[2] = third / total
        return vector, h1, encoded, probabilities

    def _loss_and_gradients(self, x: np.ndarray, label: int,
                            confidence_weight: float) -> tuple[float, dict[str, np.ndarray]]:
        self._validate_target(label, confidence_weight)
        return self._gradients_from_forward(self._forward(x), label, confidence_weight)

    def _gradients_from_forward(self, forward: tuple[np.ndarray, ...], label: int,
                                confidence_weight: float) -> tuple[float, dict[str, np.ndarray]]:
        vector, h1, encoded, probs = forward
        weight = float(confidence_weight)
        data_loss = -weight * np.log(max(probs[label], 1e-12))
        regularized = (self.W1, self.W2, self.Wh)
        loss = data_loss + 0.5 * self.l2_reg * sum(np.sum(item * item) for item in regularized)
        dlogits = probs.copy()
        dlogits[label] -= 1.0
        dlogits *= weight
        gradients = self._backward(vector, h1, encoded, dlogits)
        return float(loss), gradients

    def _update_from_forward(self, forward: tuple[np.ndarray, ...], label: int,
                             confidence_weight: float, rate: float) -> float:
        vector, h1, encoded, probs = forward
        weight = float(confidence_weight)
        dlogits = self._dlogits
        dlogits[:] = probs
        dlogits[label] -= 1.0
        dlogits *= weight
        np.multiply(encoded[:, None], dlogits, out=self._d_wh)
        np.dot(self.Wh, dlogits, out=self._d_z2)
        np.greater(encoded, 0.0, out=self._mask2)
        self._d_z2 *= self._mask2
        np.multiply(h1[:, None], self._d_z2, out=self._d_w2)
        np.dot(self.W2, self._d_z2, out=self._d_z1)
        np.greater(h1, 0.0, out=self._mask1)
        self._d_z1 *= self._mask1
        np.multiply(vector[:, None], self._d_z1, out=self._d_w1)
        loss = -weight * log(max(probs[label], 1e-12))
        loss += 0.5 * self.l2_reg * self._weight_sq
        decay = 1.0 - rate * self.l2_reg
        self._gradient_buffer *= rate
        self._weights *= decay
        self._weights -= self._gradient_buffer[:312]
        self._biases -= self._gradient_buffer[312:]
        self._weight_sq = self._weight_square_sum()
        return float(loss)

    def _weight_square_sum(self) -> float:
        return float(np.vdot(self._weights, self._weights))

    @staticmethod
    def _validate_target(label: int, confidence_weight: float) -> None:
        if label not in (0, 1, 2) or not 0.1 <= confidence_weight <= 1.0:
            raise ValueError("invalid label or confidence weight")

    def _backward(self, x: np.ndarray, h1: np.ndarray, encoded: np.ndarray,
                  dlogits: np.ndarray) -> dict[str, np.ndarray]:
        d_wh = encoded[:, None] * dlogits + self.l2_reg * self.Wh
        d_bh = dlogits
        d_z2 = (self.Wh @ dlogits) * (encoded > 0)
        d_w2 = h1[:, None] * d_z2 + self.l2_reg * self.W2
        d_b2 = d_z2
        d_z1 = (self.W2 @ d_z2) * (h1 > 0)
        return {
            "W1": x[:, None] * d_z1 + self.l2_reg * self.W1,
            "b1": d_z1, "W2": d_w2, "b2": d_b2,
            "Wh": d_wh, "bh": d_bh,
        }


class OnlineDriftDetector:
    """Adaptive two-window detector for error and feature-norm shifts."""
    def __init__(self, window_size: int = 20, threshold: float = 2.5) -> None:
        if window_size < 10 or threshold <= 0:
            raise ValueError("invalid drift detector settings")
        self.window_size = window_size
        self.threshold = float(threshold)
        self._errors: deque[float] = deque(maxlen=window_size)
        self._norms: deque[float] = deque(maxlen=window_size)
        self.drift_detected = False
        self.drift_score = 0.0

    def update(self, error: float, x: np.ndarray) -> float:
        if not np.isfinite(error) or not np.all(np.isfinite(x)):
            raise ValueError("drift observations must be finite")
        self._errors.append(float(np.clip(error, 0.0, 1.0)))
        self._norms.append(float(np.linalg.norm(x)))
        if len(self._errors) < self.window_size:
            self.drift_detected = False
            self.drift_score *= 0.9
            return self.drift_score
        error_shift = self._window_shift(np.asarray(self._errors))
        norm_shift = self._window_shift(np.asarray(self._norms))
        raw = max(error_shift, norm_shift) / self.threshold
        self.drift_score = float(np.clip(max(raw, self.drift_score * 0.85), 0.0, 1.0))
        self.drift_detected = raw >= 1.0
        if self.drift_detected:
            self._errors = deque(list(self._errors)[-5:], maxlen=self.window_size)
            self._norms = deque(list(self._norms)[-5:], maxlen=self.window_size)
        return self.drift_score

    @staticmethod
    def _window_shift(values: np.ndarray) -> float:
        split = len(values) * 3 // 4
        reference, recent = values[:split], values[split:]
        scale = max(float(reference.std()), 0.05)
        return abs(float(recent.mean() - reference.mean())) / scale
class FastLoopNodeEngine:
    """Train locally, detect drift, and temporarily accelerate adaptation."""
    def __init__(self, classifier: SplitFailureClassifier | None = None,
                 detector: OnlineDriftDetector | None = None,
                 base_lr: float = 0.05, boost_samples: int = 30) -> None:
        self.classifier = classifier or SplitFailureClassifier(default_lr=base_lr)
        self.detector = detector or OnlineDriftDetector()
        self.base_lr = float(base_lr)
        self.boost_samples = boost_samples
        self._boost_remaining = 0
        self.trained_labels = 0
        self.class_counts = np.zeros(3, dtype=int)

    @property
    def drift_detected(self) -> bool:
        return self.detector.drift_detected
    @property
    def drift_score(self) -> float:
        return self.detector.drift_score
    @property
    def is_ready(self) -> bool:
        return (self.trained_labels >= 15 and self.class_counts[0] >= 2
                and self.class_counts[1] >= 2)
    def train(self, x: np.ndarray, label: int,
              confidence_weight: float = 1.0) -> float:
        self.classifier._validate_target(label, confidence_weight)
        probabilities = self.classifier.predict_proba(x)
        self.detector.update(1.0 - probabilities[label], x)
        if self.detector.drift_detected:
            self._boost_remaining = self.boost_samples
        multiplier = 3.0 if self._boost_remaining > 0 else 1.0
        loss = self.classifier.partial_fit(
            x, label, confidence_weight, self.base_lr * multiplier)
        self._boost_remaining = max(0, self._boost_remaining - 1)
        self.trained_labels += 1
        self.class_counts[label] += 1
        return loss
