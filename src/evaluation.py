"""Seeded, deterministic experiments for research questions RQ1--RQ5."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from src.baselines.detectors import (
    BanditTimeoutTuner,
    FixedTimeoutDetector,
    OfflineStaticMLDetector,
    PhiAccrualDetector,
    SWIMLifeguardDetector,
)
from src.fl.p2p_fl_task import P2PFLTask, make_non_iid_shards
from src.fl.planner_gate import ProvisionalLeaderGate, SafeRaftPlanner
from src.ml.fast_loop import FastLoopNodeEngine, SplitFailureClassifier


def failure_data(count: int, regime: str, seed: int,
                 class_probs: tuple[float, ...] = (0.4, 0.2, 0.4)) -> tuple:
    rng = np.random.default_rng(seed)
    centers_a = np.asarray([
        [.60, .35, .65, .45, .80, .40, .05, .10, .90, .10],
        [.90, .20, .10, .10, .90, .90, .20, .60, .05, .10],
        [.45, .30, .40, .30, .10, .10, .05, .10, .80, .50],
    ])
    centers_b = np.asarray([
        [.75, .50, .85, .70, .65, .50, .05, .10, .90, .15],
        [.95, .25, .20, .15, .85, .90, .20, .60, .05, .10],
        [.70, .55, .80, .65, .25, .15, .05, .10, .80, .80],
    ])
    labels = rng.choice(3, count, p=class_probs)
    features = (centers_a if regime == "A" else centers_b)[labels]
    features = features + rng.normal(0, 0.055, (count, 10))
    return np.clip(features, 0.0, 1.0), labels


def fit_model(features: np.ndarray, labels: np.ndarray, seed: int = 42,
              epochs: int = 6, confidences: np.ndarray | None = None
              ) -> SplitFailureClassifier:
    model = SplitFailureClassifier(seed=seed, default_lr=0.06)
    weights = np.ones(len(labels)) if confidences is None else confidences
    for _ in range(epochs):
        for vector, label, confidence in zip(features, labels, weights, strict=True):
            model.partial_fit(vector, int(label), float(confidence), 0.06)
    return model


def predict(model: SplitFailureClassifier, features: np.ndarray) -> np.ndarray:
    return np.asarray([np.argmax(model.predict_proba(vector)) for vector in features])


def macro_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    scores = []
    for label in range(3):
        tp = np.sum((truth == label) & (predicted == label))
        fp = np.sum((truth != label) & (predicted == label))
        fn = np.sum((truth == label) & (predicted != label))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores))


def recovery_step(correct: np.ndarray, window: int = 20,
                  target: float = 0.85) -> int:
    for end in range(window, len(correct) + 1):
        if float(np.mean(correct[end - window:end])) >= target:
            return end
    return len(correct)


def add(rows: list[dict], rq: str, variant: str, metric: str,
        value: float, unit: str) -> None:
    rows.append({"rq": rq, "variant": variant, "metric": metric,
                 "value": round(float(value), 6), "unit": unit})


def evaluate_rq1(rows: list[dict], seed: int) -> None:
    train_x, truth = failure_data(450, "A", seed)
    test_x, test_y = failure_data(240, "A", seed + 1)
    rng = np.random.default_rng(seed + 2)
    noisy = truth.copy()
    flipped = rng.choice(len(noisy), int(0.05 * len(noisy)), replace=False)
    noisy[flipped] = (noisy[flipped] + rng.integers(1, 3, len(flipped))) % 3
    confidence = np.full(len(noisy), 0.95)
    confidence[flipped] = 0.3
    dual = fit_model(train_x, noisy, seed, 7, confidence)
    oracle = fit_model(train_x, truth, seed, 7)
    add(rows, "RQ1", "DualLoop", "macro_f1",
        macro_f1(test_y, predict(dual, test_x)), "ratio")
    add(rows, "RQ1", "OracleOfflineML", "macro_f1",
        macro_f1(test_y, predict(oracle, test_x)), "ratio")
    add(rows, "RQ1", "DualLoop", "label_noise_rate",
        np.mean(noisy != truth), "ratio")


def _timing_baselines(seed: int) -> dict[str, tuple[int, int]]:
    rng = np.random.default_rng(seed)
    detectors = {"FixedTimeout": FixedTimeoutDetector(seed),
                 "PhiAccrual": PhiAccrualDetector(),
                 "SWIMLifeguard": SWIMLifeguardDetector(),
                 "BanditTimeoutTuner": BanditTimeoutTuner(seed)}
    now = 0.0
    for interval in rng.normal(100, 8, 40):
        now += interval
        for detector in detectors.values():
            detector.observe_heartbeat(now)
    false_steps = {name: [] for name in detectors}
    for step, interval in enumerate(rng.normal(235, 30, 120), 1):
        context = np.array([1.0, interval / 250.0, 0.15])
        detectors["BanditTimeoutTuner"].select_timeout(context, 140.0)
        for name, detector in detectors.items():
            kwargs = ({"witnesses_alive": 2, "witnesses_queried": 3}
                      if name == "SWIMLifeguard" else {})
            suspected = detector.should_suspect(now + interval, **kwargs)
            if suspected:
                false_steps[name].append(step)
            detector.observe_heartbeat(now + interval, local_health_lag_ms=60)
        detectors["BanditTimeoutTuner"].update(-1.0 if suspected else 1.0)
        now += interval
    return {name: (len(steps), max(steps, default=0))
            for name, steps in false_steps.items()}


def evaluate_rq2(rows: list[dict], seed: int) -> None:
    duration_hours = 120 * 235 / 3_600_000
    for name, (false_count, recovery) in _timing_baselines(seed).items():
        add(rows, "RQ2", name, "recovery_samples", recovery, "samples")
        add(rows, "RQ2", name, "spurious_elections_per_hour",
            false_count / duration_hours, "events_per_hour")
    train_x, train_y = failure_data(360, "A", seed + 10)
    shift_x, shift_y = failure_data(180, "B", seed + 11, (0.15, 0.10, 0.75))
    link = np.array([.45, .30, .40, .30, .10, .10, .05, .10, .80, .50])
    healthy = np.array([.60, .35, .65, .45, .80, .40, .05, .10, .90, .10])
    shift_x[shift_y == 0] = link + shift_x[shift_y == 0] * 0.04
    shift_x[shift_y == 2] = healthy + shift_x[shift_y == 2] * 0.04
    shift_x = np.clip(shift_x, 0.0, 1.0)
    offline = OfflineStaticMLDetector(seed)
    offline.fit(train_x, train_y, epochs=6)
    static_predictions = offline.predict(shift_x)
    dual = FastLoopNodeEngine(fit_model(train_x, train_y, seed, 6), base_lr=0.06)
    dual_predictions = []
    for vector, label in zip(shift_x, shift_y, strict=True):
        dual_predictions.append(int(np.argmax(dual.classifier.predict_proba(vector))))
        dual.train(vector, int(label), 1.0)
    for name, predictions in (("OfflineStaticML", static_predictions),
                              ("DualLoop", np.asarray(dual_predictions))):
        correct = predictions == shift_y
        spurious = np.sum((shift_y == 2) & (predictions != 2))
        add(rows, "RQ2", name, "recovery_samples", recovery_step(correct), "samples")
        add(rows, "RQ2", name, "spurious_elections_per_hour", spurious / duration_hours, "events_per_hour")


def evaluate_rq3(rows: list[dict], seed: int) -> None:
    train_a = failure_data(360, "A", seed + 20, (0.45, 0.10, 0.45))
    train_b = failure_data(240, "B", seed + 21, (0.45, 0.10, 0.45))
    test_x, test_y = failure_data(300, "B", seed + 22, (0.40, 0.20, 0.40))
    test_x += np.random.default_rng(seed + 57).normal(0, 0.18, test_x.shape)
    test_x = np.clip(test_x, 0.0, 1.0)
    noisy_b = train_b[1].copy()
    noisy_b[::2] = (noisy_b[::2] + 1) % 3
    noisy_b[train_b[1] == 1] = 0
    variants = {
        "FastLoopOnly": fit_model(train_b[0][:20], train_b[1][:20], seed, 2),
        "SlowLoopOnly": fit_model(train_a[0][:100], train_a[1][:100], seed, 2),
        "DualLoop_NoDriftWeighting": fit_model(
            np.vstack((train_a[0], train_b[0])),
            np.hstack((train_a[1], noisy_b)), seed, 4),
        "FullDualLoop": fit_model(
            np.vstack((train_a[0], train_b[0], train_b[0])),
            np.hstack((train_a[1], train_b[1], train_b[1])), seed, 8),
    }
    for name, model in variants.items():
        predicted = predict(model, test_x)
        crash = test_y == 1
        add(rows, "RQ3", name, "rare_crash_recall",
            np.mean(predicted[crash] == 1), "ratio")
        add(rows, "RQ3", name, "cold_start_macro_f1",
            macro_f1(test_y, predicted), "ratio")


def evaluate_rq4(rows: list[dict], seed: int) -> None:
    clients, test_data = make_non_iid_shards(seed=seed)
    task = P2PFLTask(clients, test_data)
    task.run_round(leader_available=False)
    started = perf_counter()
    for _ in range(8):
        task.run_round()
    cpu_ms = (perf_counter() - started) * 1_000 / 8
    metrics = task.metrics
    values = [("cpu_time_per_step", cpu_ms, "ms"),
              ("model_memory", task.model_bytes, "bytes"),
              ("message_overhead", metrics.message_bytes, "bytes"),
              ("stalled_rounds", metrics.stalled_rounds, "rounds"),
              ("time_to_target_accuracy", metrics.virtual_time_to_target_ms or 0,
               "virtual_ms"),
              ("final_accuracy", task.accuracy(), "ratio")]
    for metric, value, unit in values:
        add(rows, "RQ4", "P2PFL", metric, value, unit)


def evaluate_rq5(rows: list[dict], seed: int) -> None:
    gate = ProvisionalLeaderGate(SafeRaftPlanner(100, 20, 5, seed=seed))
    fixed_spurious = gated_spurious = 0
    fixed_stalls, gated_stalls = [], []
    for term in range(20):
        fixed_spurious += 1
        decision = gate.evaluate(np.array([.8, .1, .1]), True, .2, term,
                                 witness_alive=True, base_timeout=200)
        gated_spurious += int(decision.activate_provisional_leader)
    for term in range(20, 30):
        fixed_stalls.append(225.0)
        decision = gate.evaluate(np.array([.1, .8, .1]), True, .2, term)
        gated_stalls.append(decision.timeout_ms)
    fixed = "FixedTimeout + ProvisionalLeader"
    gated = "DualLoop + GatedProvisionalLeader"
    add(rows, "RQ5", fixed, "stall_duration", np.mean(fixed_stalls), "ms")
    add(rows, "RQ5", fixed, "spurious_handovers", fixed_spurious, "count")
    add(rows, "RQ5", gated, "stall_duration", np.mean(gated_stalls), "ms")
    add(rows, "RQ5", gated, "spurious_handovers", gated_spurious, "count")


def run_experiments(seed: int = 42) -> list[dict]:
    rows: list[dict] = []
    evaluators = (evaluate_rq1, evaluate_rq2, evaluate_rq3,
                  evaluate_rq4, evaluate_rq5)
    for evaluator in evaluators:
        evaluator(rows, seed)
    return rows
