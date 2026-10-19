# AGENTS.md — Architectural & Engineering Contract

## Project Overview
Lightweight, deterministic simulation and execution framework for "Dual-Loop Federated Learning for Self-Adaptive Failure Discrimination in Raft-Coordinated P2P Federated Learning".

## Strict Complexity & Dependency Guardrails (NON-NEGOTIABLE)
1. **Ultra-Lightweight Stack:** Use ONLY Python 3.11+ standard library, `numpy`, and `pytest` for the core engine (`src/`). Do NOT install PyTorch, TensorFlow, Ray, gRPC, or heavy web frameworks unless explicitly instructed in an optional adapter file.
2. **File & Function Size Budgets:**
   - Every module in `src/` MUST stay under **250 lines of code (LOC)**.
   - Every function/method MUST stay under **45 LOC**.
   - Prefer flat data structures (`@dataclass(slots=True)`) over deep inheritance hierarchies.
3. **Deterministic by Design:**
   - NEVER call `time.sleep()`, `time.time()`, or `random.*` directly inside `src/`.
   - Always use the injected `Clock` protocol and a seeded `np.random.Generator` (`rng`).
4. **Raft Safety Invariant:**
   - The ML discriminator is strictly **advisory** (modifies election timeout bounds and provisional-leader gating only). It must NEVER alter Raft voting rules, term monotonicity, or log-matching safety.

## Target Directory Structure
```text
src/
  core/
    types.py          # Messages, LogEntry, PeerState, TelemetrySnapshot
    clock_network.py  # VirtualClock, SimNetwork, FaultInjector (delays, loss, partitions, crashes)
    raft_node.py      # Lightweight Raft state machine + boot_id + witness piggybacking
  ml/
    features.py       # Rolling window telemetry extractor (10 features + local health lag)
    self_labeler.py   # Horizon H suspicion resolver & confidence-weighted label generator
    fast_loop.py      # Pure NumPy split-MLP (shared encoder + local head) & ADWIN drift detector
    slow_loop.py      # Drift-aware, trimmed-mean federated encoder aggregator
  fl/
    planner_gate.py   # Safe Raft timeout planner + Provisional Leader gating (RQ5)
    p2p_fl_task.py    # Lightweight P2P FL task (distributed logistic/MLP training + Flower adapter)
  baselines/
    detectors.py      # Fixed-Timeout, Phi-Accrual, SWIM-Lifeguard, Bandit (BALLAST-style)
tests/
  unit/               # Fast per-module unit tests (< 1s total)
  property/           # Raft safety & invariant tests (election safety, log matching)
  integration/        # End-to-end drift & fault-injection scenario tests
```

## Verification Rule
No task is complete until `pytest -q` passes with 0 warnings, 100% of existing tests remain green, and module LOC budgets are verified via `wc -l src/**/*.py`.
