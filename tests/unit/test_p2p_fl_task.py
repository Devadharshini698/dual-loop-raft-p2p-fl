"""Flower adapter runtime and aggregation contract checks."""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from src.fl.p2p_fl_task import (
    FLOWER_AVAILABLE,
    FLOWER_VERSION,
    PeerFlowerClient,
    RaftFedAvgStrategy,
    flower_runtime_status,
)


def test_flower_runtime_is_explicit_and_consistent() -> None:
    runtime = flower_runtime_status()
    assert runtime["available"] is FLOWER_AVAILABLE
    assert runtime["fallback"] is (not FLOWER_AVAILABLE)
    assert runtime["version"] == FLOWER_VERSION
    assert runtime["execution"] == "in_process"
    if FLOWER_AVAILABLE:
        from flwr.client import NumPyClient
        from flwr.server.strategy import FedAvg

        assert issubclass(PeerFlowerClient, NumPyClient)
        assert isinstance(RaftFedAvgStrategy(), FedAvg)
        assert runtime["mode"] == "native"
    else:
        assert runtime["mode"] == "fallback"
        assert "compatibility fallback" in runtime["message"]


def test_flower_weighted_aggregation_contract() -> None:
    updates = [([np.array([1.0, 3.0])], 1), ([np.array([4.0, 9.0])], 3)]
    aggregate = RaftFedAvgStrategy.aggregate_parameters(updates)
    np.testing.assert_allclose(aggregate[0], np.array([3.25, 7.5]))


def test_real_flower_client_conversion_when_available() -> None:
    if not FLOWER_AVAILABLE:
        return
    from flwr.client import Client

    client = PeerFlowerClient(np.zeros((4, 2)), np.zeros(4, dtype=int))
    assert isinstance(client.to_client(), Client)


def test_zero_dependency_fallback_imports_without_flower() -> None:
    script = """
import builtins
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'flwr' or name.startswith('flwr.'):
        raise ModuleNotFoundError("No module named 'flwr'", name='flwr')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
from src.fl import p2p_fl_task as task
assert not task.FLOWER_AVAILABLE
assert task.flower_runtime_status()['fallback']
client = task.PeerFlowerClient([[0.0, 1.0]], [0])
assert len(client.get_parameters({})) == 2
"""
    subprocess.run([sys.executable, "-c", script], check=True)
