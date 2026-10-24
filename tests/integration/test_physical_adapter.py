import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from physical_testbed.clock import MonotonicClock
from physical_testbed.config import LabConfig, NodeSpec, validate_run_id
from physical_testbed.demo import run_local_socket_smoke
from physical_testbed.flower.client_app import (
    FLOWER_CLIENT_AVAILABLE,
    app as flower_client_app,
    flower_client_status,
)
from physical_testbed.flower.server_app import (
    FLOWER_DEPLOYMENT_AVAILABLE,
    app as flower_server_app,
    consensus_signature,
    flower_deployment_status,
)
from physical_testbed.store import StableStateStore
from physical_testbed.transport import decode_message, encode_message
from src.core.types import Message, MessageType


def test_physical_message_codec_and_stable_boot():
    message = Message("node1", "node2", 7, MessageType.HEARTBEAT, 3, 2, 6,
                      {"node3": 12.5}, {"leader_commit": 1})
    assert decode_message(encode_message(message)) == message
    with tempfile.TemporaryDirectory(dir="work") as directory:
        store = StableStateStore(Path(directory) / "node1.json")
        assert store.start_incarnation() == 1
        assert store.start_incarnation() == 2


@pytest.mark.parametrize("payload", [
    b"[]",
    b'{"sender_id":"n1","receiver_id":"n2","term":0,"msg_type":"HEARTBEAT","payload":7}',
    b'{"sender_id":"n1","receiver_id":"n2","term":0,"msg_type":"HEARTBEAT","peer_heard_map":{"n3":NaN}}',
])
def test_physical_message_codec_rejects_malformed_datagrams(payload):
    with pytest.raises(ValueError):
        decode_message(payload)


def test_monotonic_clock_serializes_callbacks():
    clock = MonotonicClock()
    events = []
    try:
        clock.call_later(30, lambda: events.append("later"))
        clock.call_later(5, lambda: events.append("first"))
        deadline = time.monotonic() + 1
        while len(events) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert events == ["first", "later"]
    finally:
        clock.close()


def test_flower_raft_consensus_gate():
    metrics = [
        {"raft_term": 4, "raft_leader": "node2"},
        {"raft_term": 4, "raft_leader": "node2"},
        {"raft_term": 4, "raft_leader": "node2"},
        {"raft_term": 3, "raft_leader": "node1"},
    ]
    assert consensus_signature(metrics, 3) == (4, "node2")
    assert consensus_signature(metrics, 4) is None


def test_physical_flower_runtime_is_explicit():
    server = flower_deployment_status()
    client = flower_client_status()
    assert server["available"] is FLOWER_DEPLOYMENT_AVAILABLE
    assert client["available"] is FLOWER_CLIENT_AVAILABLE
    assert server["fallback"] is False
    assert client["fallback"] is False
    assert (flower_server_app is not None) is FLOWER_DEPLOYMENT_AVAILABLE
    assert (flower_client_app is not None) is FLOWER_CLIENT_AVAILABLE
    if not FLOWER_DEPLOYMENT_AVAILABLE:
        assert server["mode"] == "unavailable"
        assert "No fallback was started" in server["message"]


def test_physical_flower_modules_import_without_sdk():
    script = """
import builtins
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'flwr' or name.startswith('flwr.'):
        raise ModuleNotFoundError("No module named 'flwr'", name='flwr')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
from physical_testbed.flower import client_app, server_app
assert client_app.app is None and server_app.app is None
assert client_app.flower_client_status()['mode'] == 'unavailable'
assert server_app.flower_deployment_status()['mode'] == 'unavailable'
assert not client_app.flower_client_status()['fallback']
assert not server_app.flower_deployment_status()['fallback']
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_lab_config_rejects_management_data_interface_collision():
    nodes = tuple(NodeSpec(f"n{i}", "127.0.0.1", 7200 + i, i, "eth0", "eth0")
                  for i in range(3))
    with pytest.raises(ValueError, match="separate"):
        LabConfig("bad", nodes).validate()


def test_lab_config_and_run_id_reject_unsafe_physical_inputs():
    nodes = (
        NodeSpec("../escape", "127.0.0.1", 7201, 0, "eth0"),
        NodeSpec("n2", "127.0.0.1", 7202, 1, "eth0"),
        NodeSpec("n3", "127.0.0.1", 7203, 2, "eth0"),
    )
    with pytest.raises(ValueError, match="node id"):
        LabConfig("safe", nodes).validate()
    with pytest.raises(ValueError, match="run_id"):
        validate_run_id("../escape")


def test_real_udp_cluster_commit_and_restart():
    with tempfile.TemporaryDirectory(dir="work") as directory:
        root = Path(directory)
        result = run_local_socket_smoke(root, timeout_s=8)
        assert all(result["checks"].values())
        assert result["physical_multi_machine_claim"] is False
        saved = json.loads((root / "physical_smoke.json").read_text())
        assert saved["run_id"] == result["run_id"]
