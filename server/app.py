"""FastAPI and bidirectional WebSocket host for the Judge Observatory."""

from __future__ import annotations

import asyncio
import csv
import json
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from run_live_demo import build_demo
from server.sim_bridge import ObservatoryBridge, SCENARIOS
from src.fl.p2p_fl_task import flower_runtime_status


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "frontend" / "dist"
bridge = ObservatoryBridge(seed=42)
clients: set[WebSocket] = set()


class PlaybackRequest(BaseModel):
    action: str
    speed: float | None = None


def _rq_payload() -> dict[str, Any]:
    metrics_path = ROOT / "results" / "rq_metrics.csv"
    live_path = ROOT / "results" / "live_demo.json"
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["value"] = float(row["value"])
    live = json.loads(live_path.read_text(encoding="utf-8")) if live_path.exists() else build_demo(42)
    return {"seed": 42, "metrics": rows, "live_demo": live,
            "qualification": "Synthetic seeded evidence; timing-only baselines are not cause-discrimination equivalents."}


async def broadcast(state: dict) -> None:
    stale: list[WebSocket] = []
    for client in tuple(clients):
        try:
            await client.send_json(state)
        except Exception:
            stale.append(client)
    for client in stale:
        clients.discard(client)


async def simulation_loop() -> None:
    while True:
        await broadcast(bridge.tick())
        await asyncio.sleep(max(.025, .2 / bridge.speed))


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(simulation_loop())
    yield
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


app = FastAPI(title="Dual-Loop Judge Observatory", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "seed": 42, "test_count": 31,
            "flower_runtime": flower_runtime_status()}


@app.get("/api/rq-results")
def rq_results() -> dict:
    return _rq_payload()


@app.post("/api/run-live-checks")
def run_live_checks() -> dict:
    report = build_demo(42)
    return {"seed": 42, "passed": True, "checks": bridge._judge_checks(),
            "evidence": report, "flower_runtime": flower_runtime_status()}


@app.get("/api/state")
def state() -> dict:
    return bridge.snapshot()


@app.get("/api/scenarios")
def scenarios() -> dict:
    return SCENARIOS


@app.post("/api/scenario/{scenario}")
async def choose_scenario(scenario: str) -> dict:
    try:
        result = bridge.reset(scenario)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    await broadcast(result)
    return result


@app.post("/api/control")
async def playback(request: PlaybackRequest) -> dict:
    result = bridge.set_playback(request.action, request.speed)
    await broadcast(result)
    return result


def _preset_name(preset_id: str) -> str:
    presets = {"preset_1_ambiguity": "ambiguous", "preset_2_drift": "drift",
               "preset_3_handover": "leader", "full_10_checks": "ambiguous"}
    if preset_id not in presets:
        raise ValueError(f"unknown preset: {preset_id}")
    return presets[preset_id]


async def _apply_command(command: dict[str, Any]) -> dict:
    action = command.get("action")
    if action == "play":
        speed_ms = max(25, float(command.get("speed_ms", 100)))
        return bridge.set_playback("play", 100 / speed_ms)
    if action == "pause":
        return bridge.set_playback("pause")
    if action == "step":
        result = bridge.set_playback("pause")
        for _ in range(max(1, min(100, int(command.get("ticks", 1))))):
            result = bridge.set_playback("step")
        return result
    if action == "reset":
        return bridge.reset(bridge.scenario)
    if action == "load_preset":
        return bridge.reset(_preset_name(str(command.get("preset_id"))))
    if action == "inject_fault":
        mapping = {"LINK_FAILURE": "ambiguous", "CRASH_LEADER": "leader",
                   "CONCEPT_DRIFT": "drift", "AMBIGUOUS_ISOLATION": "isolation"}
        fault = str(command.get("fault_type"))
        if fault not in mapping:
            raise ValueError(f"unknown fault type: {fault}")
        return bridge.reset(mapping[fault])
    raise ValueError(f"unknown action: {action}")


@app.websocket("/ws/observatory")
async def observatory(websocket: WebSocket) -> None:
    await websocket.accept()
    clients.add(websocket)
    await websocket.send_json(bridge.snapshot())
    try:
        while True:
            command = await websocket.receive_json()
            try:
                result = await _apply_command(command)
                await broadcast(result)
            except ValueError as error:
                await websocket.send_json({"error": str(error)})
    except WebSocketDisconnect:
        clients.discard(websocket)


if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def dashboard(path: str = "") -> FileResponse:
        candidate = DIST / path
        return FileResponse(candidate if candidate.is_file() else DIST / "index.html")

