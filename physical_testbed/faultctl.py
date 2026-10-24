"""Fail-safe, evidence-producing controller for physical lab faults."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import LabConfig, NodeSpec, load_lab_config, validate_run_id
from .store import EvidenceLogger


class FaultController:
    def __init__(self, config: LabConfig, run_id: str, apply: bool = False) -> None:
        validate_run_id(run_id)
        self.config, self.run_id, self.apply = config, run_id, apply
        path = Path(config.evidence_root) / run_id / "fault-events.jsonl"
        self.logger = EvidenceLogger(path, run_id, "controller")

    def timed_netem(self, node_id: str, duration_s: float, seed: int = 42,
                    loss_pct: float | None = None,
                    delay_ms: float | None = None, jitter_ms: float = 0.0) -> None:
        node = self.config.node(node_id)
        _validate_duration(duration_s)
        args = _netem_args(loss_pct, delay_ms, jitter_ms, seed)
        command = ["tc", "qdisc", "replace", "dev", node.data_interface,
                   "root", "netem", *args]
        self._record("fault_start", node, "netem", command, duration_s)
        try:
            self._remote(node, command)
            if self.apply:
                time.sleep(duration_s)
        finally:
            self.clear(node_id)

    def timed_crash(self, node_id: str, duration_s: float) -> None:
        node = self.config.node(node_id)
        _validate_duration(duration_s)
        service = f"dual-loop-node@{node.node_id}.service"
        stop = ["systemctl", "stop", service]
        self._record("fault_start", node, "process_crash", stop, duration_s)
        try:
            self._remote(node, stop)
            if self.apply:
                time.sleep(duration_s)
        finally:
            self._remote(node, ["systemctl", "start", service])
            self._record("fault_end", node, "process_crash", ["start", service], 0)

    def clear(self, node_id: str) -> None:
        node = self.config.node(node_id)
        command = ["tc", "qdisc", "del", "dev", node.data_interface, "root"]
        self._remote(node, command, check=False)
        self._record("fault_end", node, "netem", command, 0)

    def _remote(self, node: NodeSpec, command: list[str], check: bool = True) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", node.data_interface):
            raise ValueError("unsafe network interface name")
        invocation = ["ssh", node.ssh_target, "--", "sudo", *command]
        if not self.apply:
            print(json.dumps({"dry_run": True, "command": invocation}))
            return
        subprocess.run(invocation, check=check)

    def _record(self, event: str, node: NodeSpec, fault_type: str,
                command: list[str], duration_s: float) -> None:
        self.logger.write(event, time.monotonic() * 1_000.0, target_node=node.node_id,
                          target_host=node.host, fault_type=fault_type,
                          duration_s=duration_s, command=command, applied=self.apply)


def _netem_args(loss_pct: float | None, delay_ms: float | None,
                 jitter_ms: float, seed: int) -> list[str]:
    args: list[str] = []
    if loss_pct is not None:
        if not 0 <= loss_pct <= 100:
            raise ValueError("loss must be between 0 and 100 percent")
        args += ["loss", f"{loss_pct:g}%"]
    if delay_ms is not None:
        if delay_ms < 0 or jitter_ms < 0:
            raise ValueError("delay and jitter must be non-negative")
        args += ["delay", f"{delay_ms:g}ms", f"{jitter_ms:g}ms"]
    if not args:
        raise ValueError("specify loss or delay")
    return [*args, "seed", str(seed)]


def _validate_duration(duration_s: float) -> None:
    if not 0.1 <= duration_s <= 300:
        raise ValueError("fault duration must be between 0.1 and 300 seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inject a recoverable physical-lab fault")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--apply", action="store_true",
                        help="execute commands; otherwise only print and log the plan")
    subparsers = parser.add_subparsers(dest="fault", required=True)
    loss = subparsers.add_parser("loss")
    loss.add_argument("--percent", type=float, required=True)
    loss.add_argument("--duration", type=float, required=True)
    loss.add_argument("--seed", type=int, default=42)
    delay = subparsers.add_parser("delay")
    delay.add_argument("--milliseconds", type=float, required=True)
    delay.add_argument("--jitter", type=float, default=0.0)
    delay.add_argument("--duration", type=float, required=True)
    delay.add_argument("--seed", type=int, default=42)
    crash = subparsers.add_parser("crash")
    crash.add_argument("--duration", type=float, required=True)
    subparsers.add_parser("clear")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller = FaultController(load_lab_config(args.config), args.run_id, args.apply)
    if args.fault == "loss":
        controller.timed_netem(args.node_id, args.duration, args.seed,
                               loss_pct=args.percent)
    elif args.fault == "delay":
        controller.timed_netem(args.node_id, args.duration, args.seed,
                               delay_ms=args.milliseconds, jitter_ms=args.jitter)
    elif args.fault == "crash":
        controller.timed_crash(args.node_id, args.duration)
    else:
        controller.clear(args.node_id)


if __name__ == "__main__":
    main()
