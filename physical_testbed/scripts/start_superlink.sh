#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/dual-loop}"
STATE_DB="${FLOWER_STATE_DB:-/var/lib/dual-loop/flower-state.db}"

exec "${REPO_DIR}/.venv/bin/flower-superlink" \
  --insecure --host 0.0.0.0 --port 8000 \
  --database "${STATE_DB}" --disable-runtime-dependency-installation
