"""Judge-facing deterministic demonstration of the complete dual-loop pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.evaluation import run_experiments
from src.fl.p2p_fl_task import P2PFLTask, make_non_iid_shards
from src.fl.planner_gate import ProvisionalAggregatorState, ProvisionalLeaderGate, SafeRaftPlanner
from src.ml.fast_loop import FastLoopNodeEngine, OnlineDriftDetector, SplitFailureClassifier
from src.ml.features import TelemetryExtractor
from src.ml.self_labeler import FailureLabel, ProtocolSelfLabeler


def _telemetry_vector() -> np.ndarray:
    extractor = TelemetryExtractor(heartbeat_interval_ms=100)
    for index in range(20):
        extractor.observe("peer-b", index * 100, rtt_ms=20 + index % 3,
                          witnesses_alive=2, witnesses_queried=3)
    return extractor.vector("peer-b")


def _self_label_demo() -> dict:
    labeler = ProtocolSelfLabeler(horizon_ms=2_000)
    vector = _telemetry_vector()
    labeler.suspect(0, "link-peer", vector, known_boot_id=4, known_log_index=8)
    link = labeler.observe_direct(600, "link-peer", boot_id=4, log_index=8)
    labeler.suspect(1_000, "crash-peer", vector, known_boot_id=2)
    crash = labeler.observe_direct(1_250, "crash-peer", boot_id=3, log_index=0)
    labeler.suspect(2_000, "isolated-peer", vector, known_boot_id=1)
    unresolved = labeler.expire(4_001)[0]
    return {
        "link_failure": {"label": link.label.name, "confidence": link.confidence_weight},
        "crash": {"label": crash.label.name, "confidence": crash.confidence_weight},
        "local_isolation": {"label": unresolved.label.name,
                            "excluded_from_training": unresolved not in labeler.training_labels},
        "feature_count": int(vector.size),
    }


def _sample(rng: np.random.Generator, label: int, regime: str) -> np.ndarray:
    vector = rng.normal(0, 0.03, 10)
    vector[0] = (0.15 if label == 0 else 0.85) if regime == "A" else (
        0.85 if label == 0 else 0.15)
    vector[9] = 0.05 if regime == "A" else 1.0
    return vector


def _fast_loop_demo(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    engine = FastLoopNodeEngine(
        SplitFailureClassifier(seed=seed + 1, default_lr=0.08),
        OnlineDriftDetector(window_size=20, threshold=2.0), base_lr=0.08)
    for index in range(160):
        label = index % 2
        engine.train(_sample(rng, label, "A"), label)
    triggered = False
    for index in range(30):
        label = index % 2
        engine.train(_sample(rng, label, "B"), label)
        triggered |= engine.drift_detected
    correct = []
    for index in range(100):
        label = index % 2
        prediction = np.argmax(engine.classifier.predict_proba(_sample(rng, label, "B")))
        correct.append(prediction == label)
    return {"ready": bool(engine.is_ready), "drift_detected": bool(triggered),
            "post_shift_accuracy": float(np.mean(correct)),
            "trained_labels": int(engine.trained_labels)}


def _planner_demo(seed: int) -> dict:
    gate = ProvisionalLeaderGate(SafeRaftPlanner(100, 20, 5, seed=seed))
    link = gate.evaluate(np.array([.80, .10, .10]), True, .20, 7,
                         witness_alive=True, base_timeout=200)
    crash = gate.evaluate(np.array([.10, .80, .10]), True, .20, 8)
    return {
        "safety_floor_ms": gate.planner.safety_floor_ms,
        "link": {"timeout_ms": link.timeout_ms,
                 "witness_probe": link.request_witness_probe,
                 "provisional_leader": link.activate_provisional_leader},
        "crash": {"timeout_ms": crash.timeout_ms,
                  "start_election": crash.start_election,
                  "provisional_leader": crash.activate_provisional_leader},
    }


def _fl_demo(seed: int) -> dict:
    clients, test_data = make_non_iid_shards(seed=seed)
    task = P2PFLTask(clients, test_data)
    provisional = ProvisionalAggregatorState()
    provisional.activate(7)
    task.run_round(False, provisional, term=7)
    buffered = len(provisional.buffered_updates)
    handed_over = task.handover(provisional, elected_term=8)
    for _ in range(7):
        task.run_round()
    return {"buffered_client_updates": buffered, "handover_applied": bool(handed_over),
            "legitimate_handovers": task.metrics.legitimate_handovers,
            "final_accuracy": task.accuracy(),
            "time_to_target_ms": task.metrics.virtual_time_to_target_ms}


def _metric(rows: list[dict], rq: str, variant: str, metric: str) -> float:
    return next(row["value"] for row in rows if row["rq"] == rq
                and row["variant"] == variant and row["metric"] == metric)


def build_demo(seed: int = 42) -> dict:
    rows = run_experiments(seed)
    return {
        "self_labelling": _self_label_demo(),
        "fast_loop": _fast_loop_demo(seed),
        "safe_planner": _planner_demo(seed),
        "p2p_fl": _fl_demo(seed),
        "benchmarks": {
            "label_noise_rate": _metric(rows, "RQ1", "DualLoop", "label_noise_rate"),
            "drift_recovery_samples": _metric(rows, "RQ2", "DualLoop", "recovery_samples"),
            "full_dual_loop_f1": _metric(rows, "RQ3", "FullDualLoop", "cold_start_macro_f1"),
            "gated_spurious_handovers": _metric(
                rows, "RQ5", "DualLoop + GatedProvisionalLeader", "spurious_handovers"),
        },
    }


def _line(ok: bool, title: str, detail: str, color: bool) -> str:
    green, red, cyan, reset = (("\033[92m", "\033[91m", "\033[96m", "\033[0m")
                               if color else ("", "", "", ""))
    status = f"{green}PASS{reset}" if ok else f"{red}FAIL{reset}"
    return f"{status}  {cyan}{title:<24}{reset} {detail}"


def present(report: dict, pause: float = 0.15, color: bool = True) -> bool:
    labels, fast = report["self_labelling"], report["fast_loop"]
    planner, workload, bench = report["safe_planner"], report["p2p_fl"], report["benchmarks"]
    lines = [
        (labels["feature_count"] == 10, "Telemetry", f"{labels['feature_count']} normalized features per peer"),
        (labels["link_failure"]["label"] == "LINK_FAILURE", "Link evidence", f"{labels['link_failure']['label']}  confidence={labels['link_failure']['confidence']:.2f}"),
        (labels["crash"]["label"] == "CRASH", "Crash evidence", f"{labels['crash']['label']}  confidence={labels['crash']['confidence']:.2f}"),
        (labels["local_isolation"]["excluded_from_training"], "Isolation guard", "UNRESOLVED sample excluded from training"),
        (fast["ready"] and fast["drift_detected"] and fast["post_shift_accuracy"] > .85,
         "Fast local loop", f"ready={fast['ready']}  drift={fast['drift_detected']}  accuracy={fast['post_shift_accuracy']:.1%}"),
        (not planner["link"]["provisional_leader"], "Link decision", f"timeout={planner['link']['timeout_ms']:.0f} ms  provisional=False"),
        (planner["crash"]["provisional_leader"], "Crash decision", f"timeout={planner['crash']['timeout_ms']:.0f} ms  provisional=True"),
        (workload["handover_applied"], "Election handover", f"{workload['buffered_client_updates']} updates transferred to elected leader"),
        (workload["final_accuracy"] >= .8, "P2P FL", f"accuracy={workload['final_accuracy']:.1%}  target_time={workload['time_to_target_ms']:.0f} ms"),
        (bench["label_noise_rate"] < .1, "RQ validation", f"noise={bench['label_noise_rate']:.2%}  recovery={bench['drift_recovery_samples']:.0f} samples"),
    ]
    print("\nDual Loop Raft P2P FL Live Demonstration\n")
    for ok, title, detail in lines:
        print(_line(ok, title, detail, color))
        if pause:
            time.sleep(pause)
    passed = all(ok for ok, _, _ in lines)
    print("\nAll demonstration checks completed successfully." if passed
          else "\nOne or more demonstration checks failed.")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pause", type=float, default=0.15)
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/live_demo.json"))
    args = parser.parse_args()
    report = build_demo(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    passed = present(report, max(0.0, args.pause),
                     color=not args.no_color and sys.stdout.isatty())
    print(f"Evidence saved to {args.output}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
