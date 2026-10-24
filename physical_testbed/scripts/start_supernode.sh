#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <controller-host> <partition-id> <node-id> <raft-status-path>" >&2
  exit 2
fi

REPO_DIR="${REPO_DIR:-/opt/dual-loop}"
exec "${REPO_DIR}/.venv/bin/flower-supernode" \
  --insecure --superlink "$1:8000" \
  --node-config "partition-id=$2 node-id='$3' raft-status-path='$4'" \
  --max-retries 30 --max-wait-time 300
