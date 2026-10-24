"""Single-host socket smoke test for the physical adapter."""

from __future__ import annotations

import json
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

from src.core.raft_node import Role
from src.ml.self_labeler import FailureLabel

from .config import LabConfig, NodeSpec
from .runtime import PhysicalNodeRuntime
from .store import EvidenceLogger
from .validate_evidence import validate_run


def reserve_udp_ports(count: int) -> list[int]:
    sockets = [socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in range(count)]
    try:
        for item in sockets:
            item.bind(("127.0.0.1", 0))
        return [int(item.getsockname()[1]) for item in sockets]
    finally:
        for item in sockets:
            item.close()


def run_local_socket_smoke(root: str | Path, timeout_s: float = 6.0) -> dict:
    root = Path(root)
    run_id = "socket-smoke-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = root / "runs" / run_id
    ports = reserve_udp_ports(3)
    nodes = tuple(NodeSpec(f"node{index + 1}", "127.0.0.1", port, index, "lo")
                  for index, port in enumerate(ports))
    config = LabConfig("local-socket-smoke", nodes, 50.0,
                       str(run_root / "state"), str(root / "runs"))
    runtimes = [PhysicalNodeRuntime(config, node.node_id, run_id, 100 + index)
                for index, node in enumerate(nodes)]
    controller_log = EvidenceLogger(run_root / "fault-events.jsonl", run_id, "controller")
    restarted: PhysicalNodeRuntime | None = None
    result = None
    try:
        for runtime in runtimes:
            runtime.start()
        leader = _wait_for_leader(runtimes, timeout_s)
        index = leader.node.propose({"type": "physical_smoke", "run_id": run_id})
        _wait_until(lambda: all(item.node.commit_index >= index for item in runtimes), timeout_s)
        time.sleep(0.25)
        no_false_labels = all(not item.learning.labeler.resolved for item in runtimes)
        follower_index = next(i for i, item in enumerate(runtimes) if item is not leader)
        old_boot = runtimes[follower_index].node.boot_id
        controller_log.write("fault_start", leader.clock.now_ms,
                             fault_type="process_crash",
                             target_node=nodes[follower_index].node_id,
                             boot_id_before=old_boot, applied=True)
        runtimes[follower_index].close()
        time.sleep(0.15)
        restarted = PhysicalNodeRuntime(config, nodes[follower_index].node_id,
                                        run_id, 200 + follower_index)
        runtimes[follower_index] = restarted
        restarted.start()
        controller_log.write("fault_end", leader.clock.now_ms,
                             fault_type="process_crash",
                             target_node=restarted.node.node_id,
                             boot_id_after=restarted.node.boot_id, applied=True)
        _wait_until(lambda: restarted.node.leader_id == leader.node.node_id, timeout_s)
        _wait_until(lambda: any(item.peer_id == restarted.node.node_id
                                and item.label is FailureLabel.CRASH
                                for item in leader.learning.labeler.resolved), timeout_s)
        crash_labels = [item for item in leader.learning.labeler.resolved
                        if item.peer_id == restarted.node.node_id
                        and item.label is FailureLabel.CRASH]
        result = {
            "mode": "single_host_real_udp_smoke",
            "physical_multi_machine_claim": False,
            "run_id": run_id,
            "leader": leader.node.node_id,
            "term": leader.node.current_term,
            "committed_index": leader.node.commit_index,
            "restart_node": restarted.node.node_id,
            "boot_before": old_boot,
            "boot_after": restarted.node.boot_id,
            "restart_observed_leader": restarted.node.leader_id,
            "checks": {
                "one_leader": True,
                "log_committed": leader.node.commit_index >= index,
                "no_false_labels_before_fault": no_false_labels,
                "restart_self_labeled_crash": bool(crash_labels),
                "boot_id_incremented": restarted.node.boot_id == old_boot + 1,
                "rejoined_cluster": restarted.node.leader_id == leader.node.node_id,
            },
        }
    finally:
        for runtime in runtimes:
            runtime.close()
    if result is None:
        raise RuntimeError("physical smoke test did not produce a result")
    validation = validate_run(run_root)
    result["checks"]["evidence_validation_passed"] = validation["passed"]
    result["evidence_directory"] = str(run_root)
    output = root / "physical_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _wait_for_leader(runtimes: list[PhysicalNodeRuntime], timeout_s: float):
    selected = None

    def stable_leader() -> bool:
        nonlocal selected
        leaders = [item for item in runtimes if item.node.role is Role.LEADER]
        if len(leaders) != 1:
            return False
        candidate = leaders[0]
        agreed = sum(item.node.leader_id == candidate.node.node_id for item in runtimes)
        selected = candidate
        return agreed >= 2

    _wait_until(stable_leader, timeout_s)
    return selected


def _wait_until(predicate, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise TimeoutError("physical adapter smoke test timed out")
