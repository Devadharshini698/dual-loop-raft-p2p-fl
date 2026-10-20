"""Contract checks for the optional FastAPI Judge Observatory adapter."""

from __future__ import annotations

import pytest


pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from server.app import app
from server.sim_bridge import ObservatoryBridge


def test_observatory_health_endpoint_contract() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert {key: payload[key] for key in ("status", "seed", "test_count")} == {
        "status": "ok", "seed": 42, "test_count": 31}
    runtime = payload["flower_runtime"]
    assert runtime["mode"] in {"native", "fallback"}
    assert runtime["fallback"] is (runtime["mode"] == "fallback")


def test_observatory_rq_and_live_check_bridges() -> None:
    with TestClient(app) as client:
        rq = client.get("/api/rq-results").json()
        live = client.post("/api/run-live-checks").json()
    assert {row["rq"] for row in rq["metrics"]} == {"RQ1", "RQ2", "RQ3", "RQ4", "RQ5"}
    assert live["passed"] and len(live["checks"]) == 10
    assert all(check["status"] == "PASS" for check in live["checks"])


def test_observatory_frame_contains_real_protocol_state() -> None:
    bridge = ObservatoryBridge(seed=42)
    for _ in range(8):
        frame = bridge.set_playback("step")
    assert frame["virtual_time_ms"] == 400
    assert len(frame["dual_loop"]["nodes"]) == 5
    assert frame["flower_runtime"]["backend"] in {"flower", "compatibility_shim"}
    assert len(frame["dual_loop"]["telemetry"]) == 10
    assert frame["dual_loop"]["self_labeler"]["resolved_label"] == "LINK_FAILURE"
    assert frame["dual_loop"]["planner_and_fl"]["recommended_timeout_ms"] == 500


def test_observatory_websocket_accepts_bidirectional_commands() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws/observatory") as websocket:
            initial = websocket.receive_json()
            websocket.send_json({"action": "pause"})
            paused = next((frame for frame in (websocket.receive_json() for _ in range(4))
                           if frame.get("playing") is False), None)
    assert initial["seed"] == 42
    assert initial["flower_runtime"]["mode"] in {"native", "fallback"}
    assert paused is not None and "dual_loop" in paused and "standard_raft" in paused

