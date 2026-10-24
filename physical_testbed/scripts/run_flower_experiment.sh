#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/dual-loop}"
cd "${REPO_DIR}"
exec .venv/bin/flwr run flowerapp college-lab --stream
