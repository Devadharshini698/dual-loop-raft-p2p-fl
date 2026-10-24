"""Flower ClientApp co-located with each physical Raft peer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.fl.p2p_fl_task import FLOWER_AVAILABLE, PeerFlowerClient, make_non_iid_shards

FLOWER_CLIENT_AVAILABLE = False
_FLOWER_IMPORT_ERROR: ImportError | None = None
if FLOWER_AVAILABLE:
    try:
        from flwr.app import Context
        from flwr.client import Client, ClientApp
    except ImportError as error:  # Incompatible or incomplete Flower installation.
        _FLOWER_IMPORT_ERROR = error
    else:
        FLOWER_CLIENT_AVAILABLE = True


def flower_client_status() -> dict[str, Any]:
    """Report whether the real networked Flower ClientApp can run."""
    if FLOWER_CLIENT_AVAILABLE:
        return {
            "available": True, "mode": "native", "fallback": False,
            "message": "Real Flower ClientApp deployment runtime is active.",
        }
    detail = f" ({_FLOWER_IMPORT_ERROR})" if _FLOWER_IMPORT_ERROR else ""
    return {
        "available": False, "mode": "unavailable", "fallback": False,
        "message": (
            "Physical Flower ClientApp is unavailable; install the Flower SDK "
            f"to run the networked testbed{detail}. No fallback was started."
        ),
    }


def _require_flower() -> None:
    if not FLOWER_CLIENT_AVAILABLE:
        raise RuntimeError(flower_client_status()["message"])


class RaftAwareFlowerClient(PeerFlowerClient):
    def __init__(self, *args: Any, node_id: str, status_path: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.node_id, self.status_path = node_id, Path(status_path)

    def fit(self, parameters, config):
        parameters, count, metrics = super().fit(parameters, config)
        status = self._raft_status()
        metrics.update({
            "node_id": self.node_id,
            "raft_term": int(status.get("term", -1)),
            "raft_leader": str(status.get("leader_id") or ""),
            "raft_boot_id": int(status.get("boot_id", -1)),
        })
        return parameters, count, metrics

    def _raft_status(self) -> dict[str, Any]:
        try:
            return json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}


def client_fn(context: Context) -> Client:
    _require_flower()
    partition = int(context.node_config.get("partition-id", 0))
    node_id = str(context.node_config.get("node-id", f"node{partition + 1}"))
    status_path = str(context.node_config.get(
        "raft-status-path", f"runs/current/status-{node_id}.json"))
    run = context.run_config
    clients, _ = make_non_iid_shards(
        num_peers=int(run["num-partitions"]),
        samples_per_peer=int(run["samples-per-peer"]),
        feature_count=int(run["feature-count"]), seed=int(run["seed"]),
    )
    source = clients[str(partition)]
    client = RaftAwareFlowerClient(source.x_train, source.y_train,
                                   node_id=node_id, status_path=status_path)
    return client.to_client()


app = ClientApp(client_fn=client_fn) if FLOWER_CLIENT_AVAILABLE else None
