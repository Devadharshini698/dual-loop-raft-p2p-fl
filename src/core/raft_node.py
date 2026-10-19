"""Small deterministic Raft state machine with witness metadata."""

from __future__ import annotations

from enum import Enum, auto
from typing import Any, Protocol

import numpy as np

from .clock_network import SimNetwork, VirtualClock
from .types import LogEntry, Message, MessageType


class Role(Enum):
    FOLLOWER = auto()
    CANDIDATE = auto()
    LEADER = auto()


class TimeoutPlanner(Protocol):
    def get_timeout_decision(self, node: "RaftNode", leader_id: str | None) -> float: ...


class RaftNode:
    """Raft safety core; timing advice affects election scheduling only."""

    def __init__(self, node_id: str, peers: list[str], clock: VirtualClock,
                 network: SimNetwork, rng: np.random.Generator | None = None,
                 timeout_planner: TimeoutPlanner | None = None,
                 heartbeat_interval_ms: float = 100.0) -> None:
        self.node_id, self.peers = node_id, tuple(peers)
        self.clock, self.network = clock, network
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.timeout_planner = timeout_planner
        self.heartbeat_interval_ms = heartbeat_interval_ms
        self.role, self.current_term = Role.FOLLOWER, 0
        self.voted_for: str | None = None
        self.leader_id: str | None = None
        self.boot_id = network.boot_ids.get(node_id, 0)
        self.log: list[LogEntry] = []
        self.commit_index = -1
        self.peer_heard_map: dict[str, float] = {}
        self.witness_maps: dict[str, dict[str, float]] = {}
        self._votes: set[str] = set()
        self._acks: dict[int, set[str]] = {}
        self._election_epoch = 0
        self._heartbeat_epoch = 0
        network.register(node_id, self.receive, self.on_reboot)

    @property
    def majority(self) -> int:
        return (len(self.peers) + 1) // 2 + 1

    def start(self) -> None:
        self._reset_election_timer()

    def receive(self, message: Message) -> None:
        if message.term > self.current_term:
            self._become_follower(message.term)
        self.peer_heard_map[message.sender_id] = self.clock.now_ms
        self.witness_maps[message.sender_id] = dict(message.peer_heard_map)
        handlers = {
            MessageType.HEARTBEAT: self._on_heartbeat,
            MessageType.HEARTBEAT_ACK: self._on_heartbeat_ack,
            MessageType.VOTE_REQ: self._on_vote_request,
            MessageType.VOTE_RESP: self._on_vote_response,
            MessageType.WITNESS_REQ: self._on_witness_request,
            MessageType.WITNESS_RESP: self._on_witness_response,
            MessageType.MODEL_SYNC: self._on_model_sync,
        }
        handlers[message.msg_type](message)

    def propose(self, value: Any) -> int:
        if self.role is not Role.LEADER:
            raise RuntimeError("only the leader can propose")
        self.log.append(LogEntry(self.current_term, value))
        index = len(self.log) - 1
        self._acks[index] = {self.node_id}
        self._broadcast_log()
        return index

    def query_witnesses(self, target_peer_id: str, k: int = 3) -> bool:
        candidates = [peer for peer in self.peers if peer != target_peer_id]
        if not candidates:
            return False
        count = min(k, len(candidates))
        chosen = list(self.rng.choice(candidates, size=count, replace=False))
        for peer in chosen:
            request = self._message(str(peer), MessageType.WITNESS_REQ,
                                    {"target": target_peer_id})
            self.network.send(request)
        threshold = self.clock.now_ms - 2 * self.heartbeat_interval_ms
        votes = sum(self.witness_maps.get(str(peer), {}).get(target_peer_id, float("-inf"))
                    >= threshold for peer in chosen)
        return votes >= count // 2 + 1

    def on_reboot(self, boot_id: int) -> None:
        self.boot_id = boot_id
        self.role, self.leader_id = Role.FOLLOWER, None
        self._votes.clear()
        self.peer_heard_map.clear()
        self.witness_maps.clear()
        self._reset_election_timer()

    def _reset_election_timer(self) -> None:
        self._election_epoch += 1
        epoch = self._election_epoch
        timeout = (self.timeout_planner.get_timeout_decision(self, self.leader_id)
                   if self.timeout_planner else float(self.rng.uniform(150.0, 300.0)))
        timeout = max(1.0, float(timeout))
        self.clock.call_later(timeout, lambda e=epoch: self._on_election_timeout(e))

    def _on_election_timeout(self, epoch: int) -> None:
        if epoch != self._election_epoch or self.node_id in self.network.crashed:
            return
        if self.role is Role.LEADER:
            return
        self.role, self.leader_id = Role.CANDIDATE, None
        self.current_term += 1
        self.voted_for, self._votes = self.node_id, {self.node_id}
        self._reset_election_timer()
        for peer in self.peers:
            self.network.send(self._message(peer, MessageType.VOTE_REQ))

    def _on_vote_request(self, message: Message) -> None:
        if message.term < self.current_term:
            self._send_vote(message.sender_id, False)
            return
        candidate_log = (message.log_term, message.log_index)
        local_log = (self.log[-1].term if self.log else 0, len(self.log) - 1)
        grant = self.voted_for in (None, message.sender_id) and candidate_log >= local_log
        if grant:
            self.voted_for = message.sender_id
            self._reset_election_timer()
        self._send_vote(message.sender_id, grant)

    def _send_vote(self, target: str, granted: bool) -> None:
        reply = self._message(target, MessageType.VOTE_RESP, {"granted": granted})
        self.network.send(reply)

    def _on_vote_response(self, message: Message) -> None:
        if self.role is not Role.CANDIDATE or message.term != self.current_term:
            return
        if message.payload.get("granted"):
            self._votes.add(message.sender_id)
        if len(self._votes) >= self.majority:
            self._become_leader()

    def _become_leader(self) -> None:
        self.role, self.leader_id = Role.LEADER, self.node_id
        self._election_epoch += 1
        self._heartbeat_epoch += 1
        self._send_heartbeats(self._heartbeat_epoch)

    def _become_follower(self, term: int) -> None:
        self.current_term, self.role = term, Role.FOLLOWER
        self.voted_for, self.leader_id = None, None
        self._votes.clear()
        self._heartbeat_epoch += 1
        self._reset_election_timer()

    def _send_heartbeats(self, epoch: int) -> None:
        if epoch != self._heartbeat_epoch or self.role is not Role.LEADER:
            return
        for peer in self.peers:
            self.network.send(self._message(peer, MessageType.HEARTBEAT,
                                            self._log_payload()))
        self.clock.call_later(self.heartbeat_interval_ms,
                              lambda e=epoch: self._send_heartbeats(e))

    def _on_heartbeat(self, message: Message) -> None:
        if message.term < self.current_term:
            return
        if self.role is not Role.FOLLOWER or self.leader_id != message.sender_id:
            self.role = Role.FOLLOWER
        self.leader_id = message.sender_id
        self._reset_election_timer()
        self._merge_leader_log(message)
        ack = self._message(message.sender_id, MessageType.HEARTBEAT_ACK,
                            {"match_index": len(self.log) - 1})
        self.network.send(ack)

    def _on_heartbeat_ack(self, message: Message) -> None:
        if self.role is not Role.LEADER or message.term != self.current_term:
            return
        match_index = int(message.payload.get("match_index", -1))
        for index in range(self.commit_index + 1, match_index + 1):
            self._acks.setdefault(index, {self.node_id}).add(message.sender_id)
            if len(self._acks[index]) >= self.majority and self.log[index].term == self.current_term:
                self.commit_index = index

    def _broadcast_log(self) -> None:
        for peer in self.peers:
            self.network.send(self._message(peer, MessageType.MODEL_SYNC,
                                            self._log_payload()))

    def _on_model_sync(self, message: Message) -> None:
        self._on_heartbeat(message)

    def _merge_leader_log(self, message: Message) -> None:
        entries = [LogEntry(int(term), value) for term, value in message.payload.get("entries", [])]
        common = 0
        while common < min(len(self.log), len(entries)) and self.log[common] == entries[common]:
            common += 1
        if common < len(self.log) and common <= self.commit_index:
            return
        self.log[common:] = entries[common:]
        leader_commit = int(message.payload.get("leader_commit", -1))
        self.commit_index = min(leader_commit, len(self.log) - 1)

    def _on_witness_request(self, message: Message) -> None:
        target = str(message.payload.get("target"))
        heard = self.peer_heard_map.get(target, float("-inf"))
        reply = self._message(message.sender_id, MessageType.WITNESS_RESP,
                              {"target": target, "last_heard": heard})
        self.network.send(reply)

    def _on_witness_response(self, message: Message) -> None:
        target = str(message.payload.get("target"))
        self.witness_maps.setdefault(message.sender_id, {})[target] = float(
            message.payload.get("last_heard", float("-inf")))

    def _log_payload(self) -> dict[str, Any]:
        return {"entries": [(entry.term, entry.value) for entry in self.log],
                "leader_commit": self.commit_index}

    def _message(self, target: str, kind: MessageType,
                 payload: dict[str, Any] | None = None) -> Message:
        last_term = self.log[-1].term if self.log else 0
        return Message(self.node_id, target, self.current_term, kind, self.boot_id,
                       len(self.log) - 1, last_term, dict(self.peer_heard_map), payload or {})
