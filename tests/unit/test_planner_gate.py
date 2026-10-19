import numpy as np

from src.fl.planner_gate import ProvisionalLeaderGate, SafeRaftPlanner


def test_safety_floor_never_violated():
    planner = SafeRaftPlanner(heartbeat_interval_ms=100, mean_rtt_ms=80,
                              std_rtt_ms=30, seed=5)
    decision = planner.plan(np.array([0.0, 1.0, 0.0]), True, 0.0, current_term=4)
    assert planner.safety_floor_ms == 270
    assert decision.timeout_ms >= planner.safety_floor_ms
    assert decision.start_election


def test_provisional_leader_gating_link_vs_crash():
    planner = SafeRaftPlanner(100, 20, 5, seed=6)
    gate = ProvisionalLeaderGate(planner)
    link = gate.evaluate(np.array([0.8, 0.1, 0.1]), True, 0.3, 7,
                         witness_alive=True, base_timeout=200)
    assert link.request_witness_probe and link.keep_current_leader
    assert not link.activate_provisional_leader
    assert gate.suppressed_handovers == 1
    assert not gate.state.active

    crash = gate.evaluate(np.array([0.1, 0.8, 0.1]), True, 0.2, 7)
    assert crash.start_election and crash.activate_provisional_leader
    assert crash.provisional_term == 7
    assert gate.state.buffer_update("client-a", {"gradient": [1, 2]}, term=7)
    assert gate.state.buffer_update("client-b", {"gradient": [3]}, term=7)
    handed_over = gate.state.handover(elected_term=8)
    assert [client for client, _ in handed_over] == ["client-a", "client-b"]
    assert not gate.state.active
