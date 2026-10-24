"""Physical node runtime reusing the tested Raft state machine over UDP."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.core.raft_node import RaftNode, Role
from src.core.raft_validation import valid_message
from src.core.types import LogEntry, MessageType

from .clock import MonotonicClock
from .config import LabConfig, validate_run_id
from .learning import PhysicalLearningLoop
from .store import EvidenceLogger, StableStateStore, log_checksum
from .transport import UdpRaftNetwork


class DurableRaftNode(RaftNode):
    """Persist Raft safety state before emitting dependent messages."""

    def __init__(self, *args: Any, state_store: StableStateStore, **kwargs: Any) -> None:
        self.state_store = state_store
        self.peer_boot_ids: dict[str, int] = {}
        self.peer_log_indices: dict[str, int] = {}
        self.physical_witness_reports: list[dict[str, Any]] = []
        self.protocol_observer = None
        super().__init__(*args, **kwargs)
        self._restore()

    def receive(self, message) -> None:
        if not valid_message(message, self.node_id, self.peers):
            return
        known_boot = self.peer_boot_ids.get(message.sender_id, -1)
        if message.boot_id < known_boot:
            return
        self.peer_boot_ids[message.sender_id] = message.boot_id
        self.peer_log_indices[message.sender_id] = message.log_index
        if self.protocol_observer is not None:
            self.protocol_observer(message)
        super().receive(message)

    def propose(self, value: Any) -> int:
        if self.role is not Role.LEADER:
            raise RuntimeError("only the leader can propose")
        json.dumps(value, allow_nan=False)
        self.log.append(LogEntry(self.current_term, value))
        index = len(self.log) - 1
        self._acks[index] = {self.node_id}
        self._persist()
        self._broadcast_log()
        return index

    def _on_election_timeout(self, epoch: int) -> None:
        if epoch != self._election_epoch or self.node_id in self.network.crashed:
            return
        if self.role is Role.LEADER:
            return
        self.role, self.leader_id = Role.CANDIDATE, None
        self.current_term += 1
        self.voted_for, self._votes = self.node_id, {self.node_id}
        self._persist()
        self._reset_election_timer()
        for peer in self.peers:
            self.network.send(self._message(peer, MessageType.VOTE_REQ))

    def _send_vote(self, target: str, granted: bool) -> None:
        self._persist()
        super()._send_vote(target, granted)

    def _become_follower(self, term: int) -> None:
        super()._become_follower(term)
        self._persist()

    def _merge_leader_log(self, message) -> None:
        super()._merge_leader_log(message)
        self._persist()

    def _on_heartbeat_ack(self, message) -> None:
        before = self.commit_index
        super()._on_heartbeat_ack(message)
        if self.commit_index != before:
            self._persist()

    def _on_witness_request(self, message) -> None:
        target = str(message.payload.get("target"))
        heard = self.peer_heard_map.get(target, float("-inf"))
        alive = self.clock.now_ms - heard <= 2 * self.heartbeat_interval_ms
        payload = {"target": target, "last_heard": heard if math.isfinite(heard) else None,
                   "target_alive": alive,
                   "target_boot_id": self.peer_boot_ids.get(target)}
        self.network.send(self._message(message.sender_id, MessageType.WITNESS_RESP, payload))

    def _on_witness_response(self, message) -> None:
        super()._on_witness_response(message)
        self.physical_witness_reports.append({
            "witness_id": message.sender_id, "target": str(message.payload.get("target")),
            "target_alive": bool(message.payload.get("target_alive", False)),
            "boot_id": message.payload.get("target_boot_id"),
            "timestamp_ms": self.clock.now_ms,
        })

    def _restore(self) -> None:
        saved = self.state_store.read()
        self.current_term = int(saved.get("current_term", 0))
        self.voted_for = saved.get("voted_for")
        self.log = [LogEntry(int(term), value) for term, value in saved.get("log", [])]
        self.commit_index = min(int(saved.get("commit_index", -1)), len(self.log) - 1)
        self._persist()

    def _persist(self) -> None:
        self.state_store.save_raft(self.current_term, self.voted_for, self.log,
                                   self.commit_index, self.boot_id)


class PhysicalNodeRuntime:
    """Lifecycle, evidence, and status publication for one physical peer."""

    def __init__(self, config: LabConfig, node_id: str, run_id: str,
                 seed: int = 42) -> None:
        config.validate()
        validate_run_id(run_id)
        spec = config.node(node_id)
        self.config, self.spec, self.run_id = config, spec, run_id
        run_dir = Path(config.evidence_root) / run_id
        self.logger = EvidenceLogger(run_dir / f"{node_id}.jsonl", run_id, node_id)
        self.status_path = run_dir / f"status-{node_id}.json"
        state_path = Path(config.state_root) / f"{node_id}.json"
        self.state_store = StableStateStore(state_path)
        boot_id = self.state_store.start_incarnation()
        self.clock = MonotonicClock(self._clock_error)
        addresses = {node.node_id: node.raft_address for node in config.nodes}
        self.network = UdpRaftNetwork(node_id, addresses, self.clock, boot_id)
        peers = [node.node_id for node in config.nodes if node.node_id != node_id]
        self.node = DurableRaftNode(
            node_id, peers, self.clock, self.network, np.random.default_rng(seed),
            heartbeat_interval_ms=config.heartbeat_interval_ms,
            state_store=self.state_store,
        )
        self.learning = PhysicalLearningLoop(
            self.node, self.clock, self.logger, config.heartbeat_interval_ms)
        self.node.protocol_observer = self.learning.observe_message
        self._closed = False

    def start(self) -> None:
        self.node.start()
        self.learning.start()
        self.logger.write("node_started", self.clock.now_ms, boot_id=self.node.boot_id,
                          raft_address=list(self.spec.raft_address), pid=os.getpid())
        self.publish_status()

    def run(self, stop: threading.Event, duration_s: float = 0.0,
            status_interval_s: float = 0.5) -> None:
        self.start()
        deadline = time.monotonic() + duration_s if duration_s > 0 else None
        while not stop.wait(status_interval_s):
            self.publish_status()
            if deadline is not None and time.monotonic() >= deadline:
                break

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": 1, "run_id": self.run_id, "node_id": self.node.node_id,
            "boot_id": self.node.boot_id, "term": self.node.current_term,
            "role": self.node.role.name, "leader_id": self.node.leader_id,
            "commit_index": self.node.commit_index, "log_length": len(self.node.log),
            "log_checksum": log_checksum(self.node.log),
            "committed_checksum": log_checksum(self.node.log[:self.node.commit_index + 1]),
            "monotonic_ms": round(self.clock.now_ms, 3),
            **self.learning.status(),
        }

    def publish_status(self) -> dict[str, Any]:
        status = self.snapshot()
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, self.status_path)
        self.logger.write("status", self.clock.now_ms, **{
            key: value for key, value in status.items()
            if key not in {"schema", "run_id", "node_id", "monotonic_ms"}
        })
        return status

    def close(self) -> None:
        if self._closed:
            return
        self.publish_status()
        self.logger.write("node_stopped", self.clock.now_ms, term=self.node.current_term,
                          role=self.node.role.name)
        self.learning.close()
        self.network.close()
        self.clock.close()
        self._closed = True

    def _clock_error(self, error: Exception) -> None:
        self.logger.write("runtime_error", self.clock.now_ms,
                          error_type=type(error).__name__, message=str(error))
