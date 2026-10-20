# Dual Loop Failure Discrimination for Raft Coordinated P2P FL

This repository is a lightweight deterministic research harness for studying crash-versus-link failure discrimination in Raft-coordinated peer-to-peer federated learning. It combines protocol-derived labels, a local online classifier, drift-aware encoder aggregation, safe timeout planning, and provisional-leader gating.

## Requirements

- Python 3.11 or newer
- NumPy
- pytest for tests

Flower is optional. When `flwr` is unavailable, the project uses a built-in NumPyClient and FedAvg-compatible shim.

Install the lightweight test and runtime dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Verify the implementation

```bash
pytest -v
python run_evaluation.py
```

The evaluation command writes:

- `results/rq_summary.md`
- `results/rq_metrics.csv`

## Run the judge demonstration

```bash
python run_live_demo.py
```

The demonstration walks through protocol self-labelling, local drift detection,
safe link-versus-crash planning, provisional FL update handover, and final P2P
training accuracy. It runs the real implementation with seed 42 and saves its
machine-readable evidence to `results/live_demo.json`. For a quick automated
run without presentation pauses, use:

```bash
python run_live_demo.py --pause 0 --no-color
```

For a judge session, first run `pytest -q`, then run the live demonstration,
and finish with `python run_evaluation.py` to show the complete RQ tables.

## Launch the Option B Judge Observatory

The repository includes a polished FastAPI + WebSocket + React/Tailwind/Cytoscape
dashboard. It streams the real deterministic simulation bridge rather than a
static UI mock. The core research modules remain dependency-light; web packages
are isolated in optional adapter files.

```bash
python -m pip install -r requirements-showcase.txt
cd frontend
npm install
npm run build
cd ..
python run_observatory.py
```

Open `http://127.0.0.1:8501`. The launcher normally opens it automatically.
Use `--no-browser` when running headlessly.

Judge Mode provides synchronized Standard Raft and Dual-Loop topologies, four
fault presets, step-by-step playback, the ten telemetry features, the frozen
Horizon-H evidence queue, drift and convergence plots, safety-bound timeout
decisions, FL handover state, all ten fail-closed checks, and the seeded RQ1–RQ5
benchmark explorer. The interface labels the evidence as deterministic and
synthetic so timing results are not mistaken for physical-device measurements.

The backend contracts are available at:

- `GET /api/health`
- `GET /api/rq-results`
- `POST /api/run-live-checks`
- `WS /ws/observatory`

## Project structure

- `src/core`: virtual time, deterministic network faults, and Raft
- `src/ml`: telemetry, self-labelling, fast local learning, and slow federated aggregation
- `src/fl`: safe planning, provisional aggregation, and the synthetic P2P FL workload
- `src/baselines`: fixed-timeout, phi-accrual, SWIM/Lifeguard, bandit, and offline ML baselines
- `tests`: unit, safety-property, and seeded end-to-end RQ tests

## Reproducibility scope

The RQ tables are deterministic synthetic benchmarks for validating the implementation and experimental pipeline. They are not measurements from a physical deployment and should not be presented as such. Before publication, rerun the same metrics on the intended emulation and edge-device testbeds with repeated seeds, confidence intervals, and significance tests.

## License

MIT License. See `LICENSE`.
