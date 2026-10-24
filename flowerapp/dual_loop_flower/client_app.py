"""Physical Flower ClientApp with seeded non-IID data and Raft evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from flwr.app import Context
from flwr.client import Client, ClientApp, NumPyClient


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _shard(partition: int, peers: int, samples: int,
           features: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + partition)
    centers = np.zeros((3, features))
    centers[:, :3] = np.eye(3) * 3.0
    dominant = partition % 3
    probabilities = [0.7 if index == dominant else 0.15 for index in range(3)]
    labels = rng.choice(3, samples, p=probabilities)
    return centers[labels] + rng.normal(0, 0.55, (samples, features)), labels


class PhysicalFlowerClient(NumPyClient):
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray,
                 node_id: str, status_path: str) -> None:
        self.x_train, self.y_train = x_train, y_train
        self.node_id, self.status_path = node_id, Path(status_path)
        self.weights = np.zeros((x_train.shape[1], 3))
        self.bias = np.zeros(3)

    def get_parameters(self, config=None):
        return [self.weights.copy(), self.bias.copy()]

    def fit(self, parameters, config):
        self.weights, self.bias = [np.asarray(item).copy() for item in parameters]
        targets = np.eye(3)[self.y_train]
        rate = float(config.get("learning_rate", 0.2))
        for _ in range(int(config.get("local_epochs", 2))):
            error = (_softmax(self.x_train @ self.weights + self.bias) - targets) / len(targets)
            self.weights -= rate * self.x_train.T @ error
            self.bias -= rate * error.sum(axis=0)
        status = self._status()
        metrics = {"node_id": self.node_id, "raft_term": int(status.get("term", -1)),
                   "raft_leader": str(status.get("leader_id") or ""),
                   "raft_boot_id": int(status.get("boot_id", -1))}
        return self.get_parameters(), len(self.y_train), metrics

    def evaluate(self, parameters, config):
        self.weights, self.bias = [np.asarray(item).copy() for item in parameters]
        probabilities = _softmax(self.x_train @ self.weights + self.bias)
        selected = np.clip(probabilities[np.arange(len(self.y_train)), self.y_train], 1e-12, 1)
        accuracy = float(np.mean(probabilities.argmax(axis=1) == self.y_train))
        return float(-np.log(selected).mean()), len(self.y_train), {"accuracy": accuracy}

    def _status(self):
        try:
            return json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}


def client_fn(context: Context) -> Client:
    partition = int(context.node_config.get("partition-id", 0))
    node_id = str(context.node_config.get("node-id", f"node{partition + 1}"))
    status_path = str(context.node_config.get("raft-status-path", "missing-status.json"))
    run = context.run_config
    x_train, y_train = _shard(partition, int(run["num-partitions"]),
                              int(run["samples-per-peer"]),
                              int(run["feature-count"]), int(run["seed"]))
    return PhysicalFlowerClient(x_train, y_train, node_id, status_path).to_client()


app = ClientApp(client_fn=client_fn)
