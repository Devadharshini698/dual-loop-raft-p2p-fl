"""Connect real protocol traffic to telemetry, self-labels, and the fast loop."""

from __future__ import annotations

from typing import Any

from src.core.types import Message
from src.ml.fast_loop import FastLoopNodeEngine
from src.ml.features import TelemetryExtractor
from src.ml.self_labeler import FailureLabel, ProtocolSelfLabeler, ResolvedLabel


class PhysicalLearningLoop:
    def __init__(self, node, clock, logger, heartbeat_ms: float,
                 horizon_ms: float = 2_000.0) -> None:
        self.node, self.clock, self.logger = node, clock, logger
        self.heartbeat_ms = float(heartbeat_ms)
        self.features = TelemetryExtractor(heartbeat_interval_ms=heartbeat_ms)
        self.labeler = ProtocolSelfLabeler(horizon_ms)
        self.fast_engine = FastLoopNodeEngine()
        self.started_ms = clock.now_ms
        self._processed_reports = 0
        self._active = False

    def start(self) -> None:
        self._active = True
        self.clock.call_later(self.heartbeat_ms, self._tick)

    def close(self) -> None:
        self._active = False

    def observe_message(self, message: Message) -> None:
        if message.sender_id not in self.node.peers:
            return
        now = self.clock.now_ms
        vector = self.features.observe(
            message.sender_id, now, received=True,
            term_delta=max(0, message.term - self.node.current_term),
            log_lag_delta=max(0, len(self.node.log) - 1 - message.log_index),
        )
        resolved = self.labeler.observe_direct(
            now, message.sender_id, message.boot_id, message.log_index,
            jitter_ms=float(vector[3] * self.heartbeat_ms))
        if resolved is not None:
            self._train_and_log(resolved, vector)

    def status(self) -> dict[str, Any]:
        return {
            "classifier_ready": self.fast_engine.is_ready,
            "drift_score": round(self.fast_engine.drift_score, 6),
            "pending_suspicions": len(self.labeler.pending_suspicions),
            "resolved_labels": len(self.labeler.resolved),
            "training_labels": len(self.labeler.training_labels),
        }

    def _tick(self) -> None:
        if not self._active:
            return
        now = self.clock.now_ms
        self._observe_silence(now)
        self._consume_witnesses()
        for resolved in self.labeler.expire(now):
            self._train_and_log(resolved, resolved.features)
        self.clock.call_later(self.heartbeat_ms, self._tick)

    def _observe_silence(self, now: float) -> None:
        if now - self.started_ms < 3 * self.heartbeat_ms:
            return
        targets = (self.node.peers if self.node.role.name == "LEADER"
                   else ([self.node.leader_id] if self.node.leader_id else []))
        for peer in targets:
            if peer not in self.node.peer_heard_map:
                continue
            heard = self.node.peer_heard_map.get(peer, float("-inf"))
            if now - heard <= 2 * self.heartbeat_ms:
                continue
            vector = self.features.observe(peer, now, received=False)
            if peer in self.labeler.pending_suspicions:
                continue
            self.labeler.suspect(
                now, peer, vector, self.node.peer_boot_ids.get(peer, 0),
                self.node.peer_log_indices.get(peer, -1), self.heartbeat_ms,
                min(3, max(1, len(self.node.peers) - 1)),
            )
            self.node.query_witnesses(peer, min(3, len(self.node.peers) - 1))
            self.logger.write("suspicion_started", now, peer_id=peer,
                              features=vector.tolist())

    def _consume_witnesses(self) -> None:
        reports = self.node.physical_witness_reports
        for report in reports[self._processed_reports:]:
            resolved = self.labeler.observe_witness(
                report["timestamp_ms"], report["target"], report["witness_id"],
                True, report["target_alive"], report["boot_id"])
            if resolved is not None:
                self._train_and_log(resolved, resolved.features)
        self._processed_reports = len(reports)

    def _train_and_log(self, resolved: ResolvedLabel, features) -> None:
        trained = resolved.label is not FailureLabel.UNRESOLVED
        loss = (self.fast_engine.train(resolved.features, int(resolved.label),
                                       resolved.confidence_weight)
                if trained else None)
        self.logger.write(
            "label_resolved", self.clock.now_ms, peer_id=resolved.peer_id,
            label=resolved.label.name, confidence=resolved.confidence_weight,
            trained=trained, loss=loss, features=features.tolist(),
        )
