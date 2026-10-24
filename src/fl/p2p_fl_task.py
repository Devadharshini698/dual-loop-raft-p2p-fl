"""Small non-IID P2P federated-learning workload with a Flower-compatible API."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Any

import numpy as np

from src.fl.planner_gate import ProvisionalAggregatorState

try:
    from flwr.client import NumPyClient as NumPyClientBase
    from flwr.server.strategy import FedAvg as FedAvgBase
    from flwr.server.strategy.aggregate import aggregate as flower_aggregate

    FLOWER_AVAILABLE = True
    try:
        FLOWER_VERSION = package_version("flwr")
    except PackageNotFoundError:
        FLOWER_VERSION = "unknown"
except ModuleNotFoundError as error:
    if error.name != "flwr":
        raise
    FLOWER_AVAILABLE = False
    FLOWER_VERSION = None
    flower_aggregate = None

    class NumPyClientBase:
        """Minimal Flower NumPyClient protocol shim."""

    class FedAvgBase:
        """Minimal Flower FedAvg protocol shim."""


def flower_runtime_status() -> dict[str, Any]:
    """Describe the active SDK path without hiding fallback execution."""
    fallback = not FLOWER_AVAILABLE
    return {
        "backend": "compatibility_shim" if fallback else "flower",
        "mode": "fallback" if fallback else "native",
        "available": FLOWER_AVAILABLE,
        "fallback": fallback,
        "version": FLOWER_VERSION,
        "strategy": "flwr.server.strategy.FedAvg",
        "execution": "in_process",
        "message": (
            "Flower-ready integration with a zero-dependency fallback; "
            "this run is using the compatibility fallback because Flower is unavailable."
            if fallback else "Real Flower SDK classes and weighted aggregation are active."
        ),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class PeerFlowerClient(NumPyClientBase):
    """Linear softmax client matching Flower's NumPyClient signatures."""

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray,
                 classes: int = 3) -> None:
        self.x_train = np.asarray(x_train, dtype=float)
        self.y_train = np.asarray(y_train, dtype=int)
        if (self.x_train.ndim != 2 or not len(self.x_train)
                or self.y_train.shape != (len(self.x_train),) or classes < 2):
            raise ValueError("expected non-empty aligned training arrays")
        if (not np.all(np.isfinite(self.x_train)) or np.any(self.y_train < 0)
                or np.any(self.y_train >= classes)):
            raise ValueError("training data contains invalid values or labels")
        self.W = np.zeros((self.x_train.shape[1], classes))
        self.b = np.zeros(classes)

    def get_parameters(self, config: dict[str, Any] | None = None) -> list[np.ndarray]:
        return [self.W.copy(), self.b.copy()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        if len(parameters) != 2:
            raise ValueError("expected weight and bias parameters")
        weight, bias = (np.asarray(item, dtype=float) for item in parameters)
        if (weight.shape != self.W.shape or bias.shape != self.b.shape
                or not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias))):
            raise ValueError("parameter shapes or values are invalid")
        self.W, self.b = weight.copy(), bias.copy()

    def fit(self, parameters: list[np.ndarray], config: dict[str, Any]) -> tuple:
        self.set_parameters(parameters)
        learning_rate = float(config.get("learning_rate", 0.15))
        epochs = int(config.get("local_epochs", 2))
        if not np.isfinite(learning_rate) or learning_rate <= 0 or epochs <= 0:
            raise ValueError("learning rate and local epochs must be positive")
        classes = self.b.size
        targets = np.eye(classes)[self.y_train]
        for _ in range(epochs):
            probabilities = _softmax(self.x_train @ self.W + self.b)
            error = (probabilities - targets) / len(self.x_train)
            self.W -= learning_rate * (self.x_train.T @ error)
            self.b -= learning_rate * error.sum(axis=0)
        loss = -np.mean(np.log(np.clip(_softmax(self.x_train @ self.W + self.b)
                                       [np.arange(len(self.y_train)), self.y_train], 1e-12, 1)))
        return self.get_parameters(), len(self.x_train), {"loss": float(loss)}

    def evaluate(self, parameters: list[np.ndarray], config: dict[str, Any]) -> tuple:
        self.set_parameters(parameters)
        probabilities = _softmax(self.x_train @ self.W + self.b)
        loss = -np.mean(np.log(np.clip(probabilities[
            np.arange(len(self.y_train)), self.y_train], 1e-12, 1)))
        accuracy = float(np.mean(np.argmax(probabilities, axis=1) == self.y_train))
        return float(loss), len(self.x_train), {"accuracy": accuracy}


class RaftFedAvgStrategy(FedAvgBase):
    """Example-weighted FedAvg independent of Flower transport internals."""

    @staticmethod
    def aggregate_parameters(updates: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
        RaftFedAvgStrategy._validate_updates(updates)
        if flower_aggregate is not None:
            return [np.asarray(item).copy() for item in flower_aggregate(updates)]
        counts = np.asarray([count for _, count in updates], dtype=float)
        total = counts.sum()
        return [sum(parameters[index] * count for parameters, count in updates) / total
                for index in range(len(updates[0][0]))]

    @staticmethod
    def _validate_updates(updates: list[tuple[list[np.ndarray], int]]) -> None:
        if not updates:
            raise ValueError("at least one client update is required")
        reference = [np.asarray(item) for item in updates[0][0]]
        if not reference:
            raise ValueError("client updates must contain parameters")
        for parameters, count in updates:
            arrays = [np.asarray(item) for item in parameters]
            if count <= 0 or len(arrays) != len(reference):
                raise ValueError("sample counts and parameter lists must be valid")
            for value, expected in zip(arrays, reference, strict=True):
                if value.shape != expected.shape or not np.all(np.isfinite(value)):
                    raise ValueError("client parameters must be finite and shape-aligned")


@dataclass(slots=True)
class FLMetrics:
    round_accuracy: list[float] = field(default_factory=list)
    virtual_time_to_target_ms: float | None = None
    stalled_rounds: int = 0
    legitimate_handovers: int = 0
    spurious_handovers: int = 0
    message_bytes: int = 0


class P2PFLTask:
    """Leader-coordinated FedAvg workload with provisional buffering."""

    def __init__(self, clients: dict[str, PeerFlowerClient], test_data: tuple,
                 round_duration_ms: float = 1_000.0,
                 target_accuracy: float = 0.8) -> None:
        if not clients or round_duration_ms <= 0 or not 0 <= target_accuracy <= 1:
            raise ValueError("invalid FL task settings")
        self.clients = clients
        self.x_test = np.asarray(test_data[0], dtype=float)
        self.y_test = np.asarray(test_data[1], dtype=int)
        if self.x_test.ndim != 2 or self.y_test.shape != (len(self.x_test),):
            raise ValueError("test arrays must be aligned")
        sample = next(iter(clients.values()))
        self.parameters = sample.get_parameters()
        self.strategy = RaftFedAvgStrategy()
        self.round_duration_ms = float(round_duration_ms)
        self.target_accuracy = float(target_accuracy)
        self.virtual_time_ms = 0.0
        self.metrics = FLMetrics()

    def run_round(self, leader_available: bool = True,
                  provisional: ProvisionalAggregatorState | None = None,
                  term: int = 0) -> bool:
        updates, buffered_any = [], False
        for client_id, client in self.clients.items():
            parameters, count, _ = client.fit(
                self.parameters, {"learning_rate": 0.2, "local_epochs": 2})
            self.metrics.message_bytes += sum(item.nbytes for item in parameters)
            updates.append((parameters, count))
            if not leader_available and provisional is not None and provisional.active:
                buffered_any |= provisional.buffer_update(
                    client_id, (parameters, count), term)
        self.virtual_time_ms += self.round_duration_ms
        if not leader_available:
            if not buffered_any:
                self.metrics.stalled_rounds += 1
            return False
        self._apply_updates(updates)
        return True

    def handover(self, provisional: ProvisionalAggregatorState,
                 elected_term: int, legitimate: bool = True) -> bool:
        buffered = provisional.handover(elected_term)
        if not buffered:
            return False
        updates = [payload for _, payload in buffered]
        self._apply_updates(updates)
        if legitimate:
            self.metrics.legitimate_handovers += 1
        else:
            self.metrics.spurious_handovers += 1
        return True

    def accuracy(self) -> float:
        probabilities = _softmax(self.x_test @ self.parameters[0] + self.parameters[1])
        return float(np.mean(np.argmax(probabilities, axis=1) == self.y_test))

    @property
    def model_bytes(self) -> int:
        return sum(item.nbytes for item in self.parameters)

    def _apply_updates(self, updates: list[tuple[list[np.ndarray], int]]) -> None:
        self.parameters = self.strategy.aggregate_parameters(updates)
        accuracy = self.accuracy()
        self.metrics.round_accuracy.append(accuracy)
        if accuracy >= self.target_accuracy and self.metrics.virtual_time_to_target_ms is None:
            self.metrics.virtual_time_to_target_ms = self.virtual_time_ms


def make_non_iid_shards(num_peers: int = 5, samples_per_peer: int = 90,
                        feature_count: int = 6, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    centers = np.zeros((3, feature_count))
    centers[:, :3] = np.eye(3) * 3.0
    clients = {}
    for peer in range(num_peers):
        dominant = peer % 3
        labels = rng.choice(3, samples_per_peer, p=[0.7 if i == dominant else 0.15
                                                    for i in range(3)])
        features = centers[labels] + rng.normal(0, 0.55, (samples_per_peer, feature_count))
        clients[str(peer)] = PeerFlowerClient(features, labels)
    test_labels = np.repeat(np.arange(3), 100)
    test_features = centers[test_labels] + rng.normal(0, 0.55, (300, feature_count))
    return clients, (test_features, test_labels)
