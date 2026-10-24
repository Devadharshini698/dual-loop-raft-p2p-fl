"""Capture an immutable environment manifest for a physical experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .config import load_lab_config, validate_run_id


def build_manifest(config_path: str | Path, run_id: str) -> dict:
    validate_run_id(run_id)
    path = Path(config_path)
    config = load_lab_config(path)
    raw = path.read_bytes()
    return {
        "schema": 1, "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "cluster_id": config.cluster_id, "node_count": len(config.nodes),
        "node_addresses": {node.node_id: f"{node.host}:{node.raft_port}"
                           for node in config.nodes},
        "git_commit": _command(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_command(["git", "status", "--porcelain"])),
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "python": sys.version.split()[0], "flower": _package_version("flwr"),
        "numpy": _package_version("numpy"), "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
    }


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _command(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=True, capture_output=True,
                              text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a physical-run manifest")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.config, args.run_id)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
