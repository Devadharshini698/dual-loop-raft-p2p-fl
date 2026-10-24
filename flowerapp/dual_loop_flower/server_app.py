"""Flower ServerApp rejecting stale or minority Raft-term updates."""

from __future__ import annotations

from collections import Counter

from flwr.app import Context
from flwr.compat.server import ServerAppComponents
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg


def _signature(metrics):
    try:
        signature = int(metrics.get("raft_term", -1)), str(metrics.get("raft_leader", ""))
    except (TypeError, ValueError):
        return None
    return signature if signature[0] >= 0 and signature[1] else None


class RaftGatedFedAvg(FedAvg):
    def __init__(self, *args, min_agreeing_peers=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_agreeing_peers = int(min_agreeing_peers)
        if self.min_agreeing_peers <= 0:
            raise ValueError("min_agreeing_peers must be positive")
        self.accepted_term = -1
        self.accepted_leader = ""

    def aggregate_fit(self, server_round, results, failures):
        signatures = [_signature(result.metrics) for _, result in results]
        signatures = [item for item in signatures if item is not None]
        if not signatures:
            return None, {"raft_gate": "no_status", "accepted_updates": 0}
        newest_term = max(item[0] for item in signatures)
        newest = [item for item in signatures if item[0] == newest_term]
        accepted, count = Counter(newest).most_common(1)[0]
        if count < self.min_agreeing_peers:
            return None, {"raft_gate": "no_consensus", "accepted_updates": 0}
        stale = accepted[0] < self.accepted_term
        conflict = (accepted[0] == self.accepted_term and self.accepted_leader
                    and accepted[1] != self.accepted_leader)
        if stale or conflict:
            reason = "stale_term" if stale else "leader_conflict"
            return None, {"raft_gate": reason, "accepted_updates": 0}
        selected = [item for item in results if _signature(item[1].metrics) == accepted]
        parameters, metrics = super().aggregate_fit(server_round, selected, failures)
        if parameters is None:
            return None, {"raft_gate": "aggregation_failed", "accepted_updates": 0}
        self.accepted_term, self.accepted_leader = accepted
        metrics = dict(metrics or {})
        metrics.update({"raft_gate": "accepted", "raft_term": accepted[0],
                        "raft_leader": accepted[1], "accepted_updates": len(selected)})
        return parameters, metrics


def server_fn(context: Context) -> ServerAppComponents:
    run = context.run_config
    minimum = int(run["min-agreeing-peers"])
    strategy = RaftGatedFedAvg(
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=minimum, min_evaluate_clients=minimum,
        min_available_clients=minimum, min_agreeing_peers=minimum,
        on_fit_config_fn=lambda _: {"learning_rate": float(run["learning-rate"]),
                                    "local_epochs": int(run["local-epochs"])},
    )
    return ServerAppComponents(strategy=strategy,
                               config=ServerConfig(num_rounds=int(run["num-rounds"])))


app = ServerApp(server_fn=server_fn)
