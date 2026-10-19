from run_live_demo import build_demo


def test_live_demo_exercises_complete_pipeline():
    report = build_demo(seed=42)
    labels = report["self_labelling"]
    assert labels["feature_count"] == 10
    assert labels["link_failure"]["label"] == "LINK_FAILURE"
    assert labels["crash"]["label"] == "CRASH"
    assert labels["local_isolation"]["excluded_from_training"]

    fast = report["fast_loop"]
    assert fast["ready"]
    assert fast["drift_detected"]
    assert fast["post_shift_accuracy"] > 0.85

    planner = report["safe_planner"]
    assert planner["crash"]["timeout_ms"] >= planner["safety_floor_ms"]
    assert planner["crash"]["provisional_leader"]
    assert not planner["link"]["provisional_leader"]

    workload = report["p2p_fl"]
    assert workload["buffered_client_updates"] == 5
    assert workload["handover_applied"]
    assert workload["final_accuracy"] >= 0.80
