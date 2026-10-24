"""Flower ServerApp with Raft-term consensus gating."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.fl.p2p_fl_task import FLOWER_AVAILABLE, RaftFedAvgStrategy

FLOWER_DEPLOYMENT_AVAILABLE = False
_FLOWER_IMPORT_ERROR: ImportError | None = None
if FLOWER_AVAILABLE:
    try:
        from flwr.app import Context
        from flwr.compat.server import ServerAppComponents
        from flwr.server import ServerApp, ServerConfig
    except ImportError as error:  # Incompatible or incomplete Flower installation.
        _FLOWER_IMPORT_ERROR = error
    else:
        FLOWER_DEPLOYMENT_AVAILABLE = True


def flower_deployment_status() -> dict[str, Any]:
    """Report whether the real networked Flower ServerApp can run."""
    if FLOWER_DEPLOYMENT_AVAILABLE:
        return {
            "available": True, "mode": "native", "fallback": False,
            "message": "Real Flower ServerApp deployment runtime is active.",
        }
    detail = f" ({_FLOWER_IMPORT_ERROR})" if _FLOWER_IMPORT_ERROR else ""
    return {
        "available": False, "mode": "unavailable", "fallback": False,
        "message": (
            "Physical Flower ServerApp is unavailable; install the Flower SDK "
            f"to run the networked testbed{detail}. No fallback was started."
        ),
    }


def _require_flower() -> None:
    if not FLOWER_DEPLOYMENT_AVAILABLE:
        raise RuntimeError(flower_deployment_status()["message"])


def consensus_signature(metrics: list[dict[str, Any]], minimum: int) -> tuple[int, str] | None:
    signatures = [
        (int(item.get("raft_term", -1)), str(item.get("raft_leader", "")))
        for item in metrics
        if int(item.get("raft_term", -1)) >= 0 and item.get("raft_leader")
    ]
    if not signatures:
        return None
    signature, count = Counter(signatures).most_common(1)[0]
    return signature if count >= minimum else None


class RaftGatedFedAvg(RaftFedAvgStrategy):
    """Reject stale/minority client updates before standard Flower FedAvg."""

    def __init__(self, *args: Any, min_agreeing_peers: int = 3, **kwargs: Any) -> None:
        _require_flower()
        super().__init__(*args, **kwargs)
        self.min_agreeing_peers = int(min_agreeing_peers)
        self.accepted_signature: tuple[int, str] | None = None
        self.rejected_updates = 0

    def aggregate_fit(self, server_round, results, failures):
        signature = consensus_signature(
            [fit_res.metrics for _, fit_res in results], self.min_agreeing_peers)
        if signature is None:
            self.rejected_updates += len(results)
            return None, {"raft_gate": "no_consensus", "accepted_updates": 0}
        selected = [item for item in results if _signature(item[1].metrics) == signature]
        self.rejected_updates += len(results) - len(selected)
        self.accepted_signature = signature
        parameters, metrics = super().aggregate_fit(server_round, selected, failures)
        metrics.update({"raft_gate": "accepted", "raft_term": signature[0],
                        "raft_leader": signature[1], "accepted_updates": len(selected)})
        return parameters, metrics


def _signature(metrics: dict[str, Any]) -> tuple[int, str]:
    return int(metrics.get("raft_term", -1)), str(metrics.get("raft_leader", ""))


def server_fn(context: Context) -> ServerAppComponents:
    _require_flower()
    run = context.run_config
    minimum = int(run["min-agreeing-peers"])
    strategy = RaftGatedFedAvg(
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=minimum, min_evaluate_clients=minimum,
        min_available_clients=minimum, min_agreeing_peers=minimum,
        on_fit_config_fn=lambda _: {
            "learning_rate": float(run["learning-rate"]),
            "local_epochs": int(run["local-epochs"]),
        },
    )
    return ServerAppComponents(
        strategy=strategy, config=ServerConfig(num_rounds=int(run["num-rounds"])))


app = ServerApp(server_fn=server_fn) if FLOWER_DEPLOYMENT_AVAILABLE else None
