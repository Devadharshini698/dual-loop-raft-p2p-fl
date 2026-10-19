import pytest

from run_evaluation import run_experiments, write_results


@pytest.fixture(scope="module")
def metrics():
    return run_experiments(seed=42)


def value(rows, rq, variant, metric):
    return next(row["value"] for row in rows
                if row["rq"] == rq and row["variant"] == variant
                and row["metric"] == metric)


def test_rq1_h1_self_labels_match_oracle(metrics):
    dual = value(metrics, "RQ1", "DualLoop", "macro_f1")
    oracle = value(metrics, "RQ1", "OracleOfflineML", "macro_f1")
    noise = value(metrics, "RQ1", "DualLoop", "label_noise_rate")
    assert oracle - dual <= 0.08
    assert dual >= 0.90
    assert noise < 0.10


def test_rq2_h2_dual_loop_recovers_after_drift(metrics):
    dual_recovery = value(metrics, "RQ2", "DualLoop", "recovery_samples")
    static_recovery = value(metrics, "RQ2", "OfflineStaticML", "recovery_samples")
    dual_spurious = value(metrics, "RQ2", "DualLoop", "spurious_elections_per_hour")
    fixed_spurious = value(metrics, "RQ2", "FixedTimeout", "spurious_elections_per_hour")
    assert dual_recovery < static_recovery
    assert dual_spurious < fixed_spurious


def test_rq3_h3_full_dual_loop_wins_ablations(metrics):
    variants = ["FastLoopOnly", "SlowLoopOnly", "DualLoop_NoDriftWeighting"]
    full_recall = value(metrics, "RQ3", "FullDualLoop", "rare_crash_recall")
    full_f1 = value(metrics, "RQ3", "FullDualLoop", "cold_start_macro_f1")
    assert full_recall > max(value(metrics, "RQ3", name, "rare_crash_recall")
                             for name in variants)
    assert full_f1 > max(value(metrics, "RQ3", name, "cold_start_macro_f1")
                         for name in variants)


def test_rq4_h4_low_overhead_and_convergence(metrics):
    assert value(metrics, "RQ4", "P2PFL", "final_accuracy") >= 0.80
    assert value(metrics, "RQ4", "P2PFL", "time_to_target_accuracy") > 0
    assert value(metrics, "RQ4", "P2PFL", "stalled_rounds") == 1
    assert value(metrics, "RQ4", "P2PFL", "model_memory") < 10_000


def test_rq5_h5_gating_reduces_stall_and_spurious_handover(metrics):
    fixed = "FixedTimeout + ProvisionalLeader"
    gated = "DualLoop + GatedProvisionalLeader"
    assert value(metrics, "RQ5", gated, "stall_duration") < value(
        metrics, "RQ5", fixed, "stall_duration")
    assert value(metrics, "RQ5", gated, "spurious_handovers") < value(
        metrics, "RQ5", fixed, "spurious_handovers")


def test_evaluation_writes_markdown_and_csv(metrics):
    summary, csv_path = write_results(metrics, "work/test-results")
    assert summary.stat().st_size > 200
    assert csv_path.stat().st_size > 200
    assert all(f"## RQ{index}" in summary.read_text(encoding="utf-8")
               for index in range(1, 6))
