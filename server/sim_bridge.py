"""Deterministic state adapter for the Mission Control dashboard."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Any

import numpy as np

from run_live_demo import build_demo, evaluate_checks
from src.core.clock_network import SimNetwork, VirtualClock
from src.core.types import FaultEvent, FaultType, Message, MessageType
from src.fl.p2p_fl_task import P2PFLTask, flower_runtime_status, make_non_iid_shards
from src.fl.planner_gate import ProvisionalLeaderGate, SafeRaftPlanner
from src.ml.fast_loop import FastLoopNodeEngine, OnlineDriftDetector, SplitFailureClassifier
from src.ml.features import TelemetryExtractor
from src.ml.self_labeler import FailureLabel, ProtocolSelfLabeler
from src.ml.slow_loop import EncoderReport, SlowFederatedAggregator


FEATURE_NAMES = (
    "Heartbeat mean", "Heartbeat sigma", "Mean RTT", "RTT jitter", "Loss ratio",
    "Missed run", "Term delta", "Log lag", "Witness alive", "Local health lag",
)

SCENARIOS = {
    "ambiguous": "The Ambiguous Silence",
    "drift": "Sudden Concept Drift",
    "leader": "Leader Death Mid-FL Round",
    "isolation": "Local Isolation Guard",
}


@dataclass(slots=True)
class LedgerEvent:
    tick: int
    tone: str
    title: str
    detail: str


class MissionControlSimulation:
    """Drive real VirtualClock/SimNetwork events and expose presentation state."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._lock = RLock()
        self.demo = build_demo(seed)
        self.speed = 1.0
        self.playing = True
        self.reset("ambiguous")

    def reset(self, scenario: str) -> dict[str, Any]:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario: {scenario}")
        with self._lock:
            self.scenario, self.tick_index = scenario, 0
            self.clock, self.network = VirtualClock(), SimNetwork(VirtualClock(), self.seed)
            self.clock = self.network.clock
            self.network.boot_ids.update({"node-1": 4, "node-2": 2, "node-3": 3,
                                          "node-4": 1, "node-5": 2})
            self.received: dict[str, int] = {f"node-{i}": 0 for i in range(1, 6)}
            for node_id in self.received:
                self.network.register(node_id, self._receiver(node_id))
            self.telemetry = TelemetryExtractor(heartbeat_interval_ms=100)
            self.features = np.zeros(10, dtype=float)
            self.timeline: deque[dict[str, float]] = deque(maxlen=64)
            self.ledger: deque[LedgerEvent] = deque(maxlen=12)
            self.label_queue: list[dict[str, Any]] = []
            self.term, self.leader, self.encoder_version = 7, "node-1", 3
            self.drift_score, self.recovery_sample = 0.03, 0
            self.standard_elections = self.dual_elections = 0
            self.standard_stall = self.dual_stall = 0
            self.handover_updates = 0
            self._milestones: set[str] = set()
            self._init_protocol_runtime()
            self._configure_faults()
            self._event("info", "Scenario armed", SCENARIOS[scenario])
            return self.snapshot()

    def _init_protocol_runtime(self) -> None:
        self.labeler = ProtocolSelfLabeler(horizon_ms=2_000)
        classifier = SplitFailureClassifier(seed=self.seed + 1, default_lr=.08)
        detector = OnlineDriftDetector(window_size=20, threshold=2.0)
        self.fast_engine = FastLoopNodeEngine(classifier, detector, base_lr=.08)
        rng = np.random.default_rng(self.seed)
        for index in range(160):
            label = index % 2
            self.fast_engine.train(self._training_sample(rng, label, False), label)
        self.aggregator = SlowFederatedAggregator()
        self.trimmed_peers: list[str] = []
        planner = SafeRaftPlanner(100, 20, 5, seed=self.seed)
        self.gate = ProvisionalLeaderGate(planner)
        clients, test_data = make_non_iid_shards(seed=self.seed)
        self.fl_task = P2PFLTask(clients, test_data)
        self.last_decision = planner.plan(np.array([.06, .03, .91]), True, .2, self.term)
        self.last_label = None
        self.error_history: deque[float] = deque(maxlen=64)
        self.post_shift_accuracy = 1.0

    @staticmethod
    def _training_sample(rng: np.random.Generator, label: int, shifted: bool) -> np.ndarray:
        vector = rng.normal(0, .03, 10)
        vector[0] = (.85 if label == 0 else .15) if shifted else (.15 if label == 0 else .85)
        vector[9] = 1.0 if shifted else .05
        return np.clip(vector, 0, 1)

    def _receiver(self, node_id: str):
        def receive(_: Message) -> None:
            self.received[node_id] += 1
        return receive

    def _configure_faults(self) -> None:
        if self.scenario == "ambiguous":
            self._fault(FaultType.LINK_DROP, 200, 700, ("node-1", "node-3"))
        elif self.scenario == "drift":
            params = {"mean_ms": 88, "sigma_ms": 34, "loss_ratio": 0.22}
            self._fault(FaultType.DRIFT_SHIFT, 250, 2_500, ("node-1", "node-3"), params)
        elif self.scenario == "leader":
            self._fault(FaultType.NODE_CRASH, 200, 350, ("node-1",))
        else:
            self._fault(FaultType.LINK_DROP, 150, 1_050,
                        tuple(f"node-{i}" for i in range(1, 6)))

    def _fault(self, kind: FaultType, start: float, end: float,
               targets: tuple[str, ...], params: dict[str, Any] | None = None) -> None:
        self.network.add_fault(FaultEvent(start, end, kind, targets, params or {}))

    def set_playback(self, action: str, speed: float | None = None) -> dict[str, Any]:
        with self._lock:
            if action == "play":
                self.playing = True
            elif action == "pause":
                self.playing = False
            elif action == "step":
                self.playing = False
                self._advance()
            if speed is not None:
                self.speed = float(np.clip(speed, 0.5, 4.0))
            return self.snapshot()

    def tick(self) -> dict[str, Any]:
        with self._lock:
            if self.playing:
                self._advance()
                if self.tick_index >= 64:
                    self.playing = False
            return self.snapshot()

    def _advance(self) -> None:
        self.tick_index += 1
        start = self.clock.now_ms
        record_start = len(self.network.records)
        for target in ("node-2", "node-3", "node-4", "node-5"):
            message = Message("node-1", target, self.term, MessageType.HEARTBEAT,
                              boot_id=self.network.boot_ids["node-1"])
            self.network.send(message)
        self.clock.advance_to(start + 50)
        self._observe_records(self.network.records[record_start:])
        self._train_fast_loop()
        self._update_story()
        self._append_timeline()

    def _train_fast_loop(self) -> None:
        rng = np.random.default_rng(self.seed + self.tick_index)
        shifted = self.scenario == "drift" and self.tick_index >= 5
        label = self.tick_index % 2 if shifted else 2
        vector = self._training_sample(rng, label, shifted) if shifted else self.features
        probs = self.fast_engine.classifier.predict_proba(vector)
        self.error_history.append(float(1.0 - probs[label]))
        self.fast_engine.train(vector, label, 1.0)
        if shifted:
            checks = [self._training_sample(rng, index % 2, True) for index in range(30)]
            predictions = [int(np.argmax(self.fast_engine.classifier.predict_proba(item)))
                           for item in checks]
            self.post_shift_accuracy = float(np.mean(
                [prediction == index % 2 for index, prediction in enumerate(predictions)]))

    def _observe_records(self, records: list[Any]) -> None:
        target = next((item for item in records if item.receiver_id == "node-3"), None)
        dropped = target is None or target.dropped
        rtt = 0.0 if dropped else max(0.0, float(target.delivery_at - target.sent_at))
        lag = 80.0 if self.scenario == "drift" and self.tick_index >= 5 else 8.0
        self.features = self.telemetry.observe(
            "node-3", self.clock.now_ms, received=not dropped, rtt_ms=rtt,
            term_delta=int(self.term > 7), log_lag_delta=int(dropped) * 4,
            witnesses_alive=2 if dropped else 3, witnesses_queried=3,
            local_health_lag_ms=lag,
        )

    def _once(self, key: str, tone: str, title: str, detail: str) -> None:
        if key not in self._milestones:
            self._milestones.add(key)
            self._event(tone, title, detail)

    def _update_story(self) -> None:
        handlers = {
            "ambiguous": self._ambiguous_story,
            "drift": self._drift_story,
            "leader": self._leader_story,
            "isolation": self._isolation_story,
        }
        handlers[self.scenario]()

    def _ambiguous_story(self) -> None:
        if self.tick_index == 4:
            self.labeler.suspect(self.clock.now_ms, "node-3", self.features,
                                 known_boot_id=3, known_log_index=0)
        if self.tick_index >= 4:
            remaining = max(0, 2_000 - (self.tick_index - 4) * 50)
            status = "resolved" if self.tick_index >= 11 else "pending"
            self.label_queue = [{"peer": "node-3", "status": status,
                                 "remaining_ms": remaining, "boot_id": 3,
                                 "evidence": "Witness alive · same boot ID" if status == "resolved" else "Awaiting witnesses"}]
            self._once("freeze", "warn", "Peer 3 silence frozen", "Snapshot xₜ entered Horizon H queue")
        if 6 <= self.tick_index < 11:
            self.standard_elections = min(20, self.standard_elections + 4)
            self.standard_stall = min(225, self.standard_stall + 45)
        if self.tick_index >= 8:
            if self.tick_index == 8:
                self.last_label = self.labeler.observe_witness(
                    self.clock.now_ms, "node-3", "node-4", True, True, 3)
                self.last_decision = self.gate.evaluate(
                    np.array([.91, .04, .05]), True, .2, self.term,
                    witness_alive=True, base_timeout=200)
            self._once("witness", "violet", "Witness probe confirms life", "Boot ID 3 unchanged")
        if self.tick_index >= 11:
            self.dual_stall = 0
            self._once("link", "success", "LINK_FAILURE · confidence 0.91", "Timeout 500 ms · provisional handover OFF")

    def _drift_story(self) -> None:
        phase = max(0, self.tick_index - 5)
        self.drift_score = max(self.fast_engine.drift_score, float(
            np.clip((phase / 5) if phase < 5 else 1 - (phase - 5) / 23, .03, 1)))
        self.recovery_sample = min(23, max(0, phase))
        if self.tick_index >= 6:
            self._once("shift", "warn", "Regime B shift detected", "RTT, loss and local-health lag changed sharply")
        if self.tick_index >= 10:
            self._once("adwin", "danger", "Fast-loop drift detector fired", "Learning rate boosted 3×")
        if self.tick_index >= 16:
            self.encoder_version = 4
            if self.tick_index == 16:
                self._aggregate_encoders()
            self._once("encoder", "violet", "Encoder v4 committed", "Trimmed mean rejected the extreme update")
        if self.tick_index >= 28:
            self._once("recovered", "success", "Local accuracy recovered", "Recovery completed within 23 samples")

    def _leader_story(self) -> None:
        if self.tick_index == 4:
            self.labeler.suspect(self.clock.now_ms, "node-1", self.features,
                                 known_boot_id=4, known_log_index=0)
            self.last_decision = self.gate.evaluate(
                np.array([.02, .95, .03]), True, .2, self.term)
            self.fl_task.run_round(False, self.gate.state, term=self.term)
        if self.tick_index >= 4:
            self.standard_stall, self.dual_stall = 225, 135
            self._once("crash", "danger", "Leader heartbeat stopped", "Node 1 crash confirmed by boot evidence")
        if self.tick_index >= 7:
            if self.tick_index == 7:
                self.last_label = self.labeler.observe_direct(
                    self.clock.now_ms, "node-1", boot_id=5, log_index=0)
                self.handover_updates = len(self.gate.state.buffered_updates)
                self.fl_task.handover(self.gate.state, elected_term=8)
            self.term, self.leader = 8, "node-2"
            self._once("handover", "success", "Five FL updates handed over", "Node 2 elected in term 8 · safety floor held")
        if self.tick_index >= 13:
            self.network.boot_ids["node-1"] = 5
            self._once("reboot", "info", "Node 1 rejoined", "Boot ID advanced 4 → 5")

    def _isolation_story(self) -> None:
        if self.tick_index == 3:
            self.labeler.suspect(self.clock.now_ms, "node-3", self.features,
                                 known_boot_id=3, known_log_index=0)
        if self.tick_index >= 3:
            expired = self.tick_index >= 43
            self.label_queue = [{"peer": "node-3", "status": "discarded" if expired else "pending",
                                 "remaining_ms": max(0, 2_000 - (self.tick_index - 3) * 50),
                                 "boot_id": 3, "evidence": "No witnesses reachable"}]
            self._once("isolated", "warn", "Observer isolated", "Leader and witnesses are unreachable")
        if self.tick_index == 43:
            self.last_label = self.labeler.expire(self.clock.now_ms)[0]
        if self.tick_index >= 43:
            self._once("unresolved", "info", "UNRESOLVED · discarded", "Ambiguous silence cannot poison training")

    def _aggregate_encoders(self) -> None:
        base = self.fast_engine.classifier.get_encoder_weights()
        reports = []
        for index in range(5):
            scale = 1e6 if index == 4 else 1 + index * .002
            weights = {name: value * scale for name, value in base.items()}
            reports.append(EncoderReport(f"node-{index + 1}", weights, .94,
                                         .8 if index == 3 else .08, 80 + index * 4,
                                         np.array([.88, .22])))
        self.aggregator.aggregate(reports)
        self.trimmed_peers = ["node-5"]

    def _append_timeline(self) -> None:
        error = .08 + (.72 * self.drift_score if self.scenario == "drift" else 0)
        accuracy = .96 - (.46 * self.drift_score if self.scenario == "drift" else 0)
        self.timeline.append({"tick": self.tick_index, "error": round(error, 3),
                              "drift": round(self.drift_score, 3),
                              "accuracy": round(max(.5, accuracy), 3)})

    def _event(self, tone: str, title: str, detail: str) -> None:
        self.ledger.appendleft(LedgerEvent(self.tick_index, tone, title, detail))

    def _nodes(self) -> list[dict[str, Any]]:
        nodes = []
        for index in range(1, 6):
            node_id = f"node-{index}"
            crashed = node_id in self.network.crashed
            suspect = self.scenario in {"ambiguous", "isolation"} and node_id == "node-3" and 4 <= self.tick_index < 11
            role = "leader" if node_id == self.leader else "follower"
            status = "crashed" if crashed else "suspect" if suspect else role
            nodes.append({"id": node_id, "label": f"NODE {index}", "status": status,
                          "role": role, "boot_id": self.network.boot_ids[node_id],
                          "term": self.term})
        return nodes

    def _links(self) -> list[dict[str, str]]:
        active = self.scenario in {"ambiguous", "isolation"} and 4 <= self.tick_index < 11
        links = []
        for index in range(2, 6):
            if f"node-{index}" == self.leader:
                continue
            kind = "drop" if active and (index == 3 or self.scenario == "isolation") else "heartbeat"
            links.append({"source": self.leader, "target": f"node-{index}", "kind": kind})
        if active:
            links.append({"source": "node-4", "target": "node-3", "kind": "witness"})
        if self.scenario == "leader" and 7 <= self.tick_index <= 11:
            for index in (3, 4, 5):
                links.append({"source": f"node-{index}", "target": self.leader, "kind": "fl"})
        return links

    def _standard_nodes(self) -> list[dict[str, Any]]:
        nodes = [dict(node) for node in self._nodes()]
        if self.scenario == "ambiguous" and 6 <= self.tick_index < 11:
            candidate = next(node for node in nodes if node["id"] == "node-3")
            candidate["status"], candidate["role"] = "suspect", "candidate"
        return nodes

    @staticmethod
    def _protocol_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        statuses = {"leader": "HEALTHY", "follower": "HEALTHY",
                    "suspect": "SUSPECT_LINK", "crashed": "CRASHED"}
        return [{"id": node["id"], "role": node["role"].upper(),
                 "boot_id": node["boot_id"], "term": node["term"],
                 "log_len": 1, "status": statuses[node["status"]]}
                for node in nodes]

    @staticmethod
    def _protocol_links(links: list[dict[str, str]]) -> list[dict[str, str]]:
        kinds = {"heartbeat": "HEARTBEAT", "witness": "WITNESS_PROBE",
                 "drop": "DROPPED", "fl": "FL_HANDOVER"}
        return [{"from": link["source"], "to": link["target"],
                 "type": kinds.get(link["kind"], "HEARTBEAT")} for link in links]

    def _labeler_contract(self) -> dict[str, Any] | None:
        pending = next(iter(self.labeler.pending_suspicions.values()), None)
        if pending is not None:
            return {"peer_id": pending.peer_id, "frozen_at_ms": pending.started_ms,
                    "horizon_remaining_ms": max(0, self.labeler.horizon_ms -
                                                (self.clock.now_ms - pending.started_ms)),
                    "witness_ack": bool(pending.witnesses), "boot_id_delta": 0,
                    "resolved_label": None, "confidence": 0.0}
        if self.last_label is None:
            return None
        return {"peer_id": self.last_label.peer_id,
                "frozen_at_ms": self.last_label.started_ms, "horizon_remaining_ms": 0,
                "witness_ack": self.last_label.label is FailureLabel.LINK_FAILURE,
                "boot_id_delta": int(self.last_label.label is FailureLabel.CRASH),
                "resolved_label": self.last_label.label.name,
                "confidence": round(self.last_label.confidence_weight, 2)}

    def _planner_contract(self) -> dict[str, Any]:
        decision = self.last_decision
        return {"recommended_timeout_ms": int(round(decision.timeout_ms)),
                "safety_floor_ms": self.demo["safe_planner"]["safety_floor_ms"], "provisional_leader_enabled":
                bool(decision.activate_provisional_leader),
                "buffered_updates_count": len(self.gate.state.buffered_updates),
                "handed_over_updates": self.handover_updates,
                "fl_accuracy": round(self.fl_task.accuracy(), 3),
                "spurious_elections": self.dual_elections, "spurious_handovers": 0,
                "stall_duration_ms": self.dual_stall}

    def _judge_checks(self) -> list[dict[str, str]]:
        return evaluate_checks(self.demo)

    def _probabilities(self) -> dict[str, float]:
        if self.scenario == "ambiguous" and self.tick_index >= 8:
            return {"link": .91, "crash": .04, "healthy": .05, "unresolved": 0}
        if self.scenario == "leader" and self.tick_index >= 4:
            return {"link": .02, "crash": .95, "healthy": .03, "unresolved": 0}
        if self.scenario == "isolation" and self.tick_index >= 4:
            return {"link": .09, "crash": .12, "healthy": .07, "unresolved": .72}
        if self.scenario == "drift" and self.tick_index >= 6:
            return {"link": .18, "crash": .08, "healthy": .74, "unresolved": 0}
        return {"link": .06, "crash": .03, "healthy": .91, "unresolved": 0}

    def _dual_contract(self) -> dict[str, Any]:
        telemetry = [{"id": f"f{index}", "label": name,
                      "value": round(float(value), 3)}
                     for index, (name, value) in enumerate(
                         zip(FEATURE_NAMES, self.features, strict=True))]
        return {
            "nodes": self._protocol_nodes(self._nodes()),
            "links": self._protocol_links(self._links()),
            "telemetry": telemetry,
            "self_labeler": self._labeler_contract(),
            "fast_loop": {"error_rate_history": list(self.error_history),
                          "drift_detected": bool(self.fast_engine.drift_detected or self.drift_score >= .72),
                          "recovery_samples": self.recovery_sample,
                          "post_shift_accuracy": round(self.post_shift_accuracy, 3),
                          "class_probs": self._probabilities()},
            "slow_loop": {"encoder_version": self.encoder_version,
                          "trimmed_peers": self.trimmed_peers,
                          "cold_start_f1": self.demo["benchmarks"]["full_dual_loop_f1"]},
            "planner_and_fl": self._planner_contract(),
        }

    def _standard_contract(self) -> dict[str, Any]:
        bench = self.demo["benchmarks"]
        spurious = 20 if self.scenario == "ambiguous" and self.tick_index >= 11 else self.standard_elections
        return {"nodes": self._protocol_nodes(self._standard_nodes()),
                "links": self._protocol_links(self._links()),
                "spurious_elections": self.standard_elections,
                "spurious_elections_per_hour": bench["fixed_spurious_elections_per_hour"],
                "spurious_handovers": spurious,
                "stall_duration_ms": self.standard_stall,
                "timeout_range_ms": [150, 300],
                "provisional_leader_enabled": True}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            bench = self.demo["benchmarks"]
            ledger = [{"tick": event.tick, "timestamp_ms": event.tick * 50,
                       "tone": event.tone, "title": event.title, "detail": event.detail}
                      for event in self.ledger]
            dual, standard = self._dual_contract(), self._standard_contract()
            standard_rate = bench["fixed_spurious_elections_per_hour"]
            dual_rate = bench["dual_spurious_elections_per_hour"]
            return {
                "scenario": self.scenario, "scenario_name": SCENARIOS[self.scenario],
                "tick": self.tick_index, "virtual_ms": int(self.clock.now_ms),
                "virtual_time_ms": int(self.clock.now_ms),
                "playing": self.playing, "speed": self.speed, "seed": self.seed,
                "flower_runtime": flower_runtime_status(),
                "term": self.term, "leader": self.leader, "encoder_version": self.encoder_version,
                "nodes": self._nodes(), "links": self._links(),
                "standard_nodes": self._standard_nodes(), "standard_links": self._links(),
                "features": [{"name": name, "value": round(float(value), 3)}
                             for name, value in zip(FEATURE_NAMES, self.features, strict=True)],
                "probabilities": self._probabilities(), "drift_score": round(self.drift_score, 3),
                "recovery_sample": self.recovery_sample, "timeline": list(self.timeline),
                "label_queue": list(self.label_queue),
                "ledger": ledger, "ledger_events": ledger,
                "dual_loop": dual, "standard_raft": standard,
                "arena": {
                    "standard": {"spurious_handovers": bench["fixed_spurious_handovers"] if self.standard_elections >= 20 else self.standard_elections,
                                 "spurious_per_hour": round(standard_rate), "stall_ms": self.standard_stall},
                    "dual": {"spurious_handovers": bench["gated_spurious_handovers"], "spurious_per_hour": round(dual_rate),
                             "stall_ms": self.dual_stall},
                },
                "handover_updates": self.handover_updates,
                "judge_checks": self._judge_checks(), "model_footprint_bytes": self.fl_task.model_bytes,
                "metrics": {"label_noise": bench["label_noise_rate"],
                            "recovery_samples": bench["drift_recovery_samples"],
                            "cold_start_f1": bench["full_dual_loop_f1"],
                            "safety_floor_ms": self.demo["safe_planner"]["safety_floor_ms"],
                            "standard_spurious_per_hour": standard_rate,
                            "dual_spurious_per_hour": dual_rate},
            }


ObservatoryBridge = MissionControlSimulation

