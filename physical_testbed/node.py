"""Command-line entry point for one physical Raft peer."""

from __future__ import annotations

import argparse
import signal
import threading

from .config import load_lab_config
from .runtime import PhysicalNodeRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one physical Dual-Loop Raft peer")
    parser.add_argument("--config", required=True, help="lab JSON configuration")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=0.0,
                        help="seconds to run; zero runs until interrupted")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    runtime = PhysicalNodeRuntime(load_lab_config(args.config), args.node_id,
                                  args.run_id, args.seed)
    try:
        runtime.run(stop, args.duration)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
