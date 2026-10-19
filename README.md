# Dual Loop Failure Discrimination for Raft Coordinated P2P FL

This repository is a lightweight deterministic research harness for studying crash-versus-link failure discrimination in Raft-coordinated peer-to-peer federated learning. It combines protocol-derived labels, a local online classifier, drift-aware encoder aggregation, safe timeout planning, and provisional-leader gating.

## Requirements

- Python 3.11 or newer
- NumPy
- pytest for tests

Flower is optional. When `flwr` is unavailable, the project uses a built-in NumPyClient and FedAvg-compatible shim.

## Verify the implementation

```bash
pytest -v
python run_evaluation.py
```

The evaluation command writes:

- `results/rq_summary.md`
- `results/rq_metrics.csv`

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
