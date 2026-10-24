"""Collect node evidence onto the controller without extra dependencies."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from .config import load_lab_config, validate_run_id


def collect(config_path: str, run_id: str, output_root: str,
            apply: bool = False) -> list[list[str]]:
    config = load_lab_config(config_path)
    validate_run_id(run_id)
    output = Path(output_root) / run_id
    output.mkdir(parents=True, exist_ok=True)
    controller_run = Path(config.evidence_root) / run_id
    for name in ("fault-events.jsonl", "manifest.json", "flower-rounds.csv"):
        source = controller_run / name
        if source.exists() and source.resolve() != (output / name).resolve():
            shutil.copy2(source, output / name)
    commands = []
    for node in config.nodes:
        remote_root = f"{config.evidence_root}/{run_id}"
        for name in (f"{node.node_id}.jsonl", f"status-{node.node_id}.json"):
            command = ["scp", f"{node.ssh_target}:{remote_root}/{name}", str(output / name)]
            commands.append(command)
            if apply:
                subprocess.run(command, check=True)
            else:
                print(json.dumps({"dry_run": True, "command": command}))
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect physical-run evidence")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    collect(args.config, args.run_id, args.output_root, args.apply)


if __name__ == "__main__":
    main()
