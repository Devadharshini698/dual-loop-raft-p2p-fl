#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-/opt/dual-loop}"

if [[ ! -f "${REPO_DIR}/requirements.txt" ]]; then
  echo "Repository not found at ${REPO_DIR}" >&2
  exit 2
fi

sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv iproute2 chrony tcpdump sysstat git
python3.11 -m venv "${REPO_DIR}/.venv"
"${REPO_DIR}/.venv/bin/python" -m pip install --upgrade pip
"${REPO_DIR}/.venv/bin/python" -m pip install -r "${REPO_DIR}/requirements.txt"
"${REPO_DIR}/.venv/bin/python" -m pip install 'flwr==1.40.0'
if ! id dual-loop >/dev/null 2>&1; then
  sudo useradd --system --home-dir /var/lib/dual-loop --shell /usr/sbin/nologin dual-loop
fi
sudo install -d -m 0775 /var/lib/dual-loop /var/log/dual-loop/runs
sudo chown -R dual-loop:dual-loop /var/lib/dual-loop /var/log/dual-loop

echo "Host ready. Verify with: ${REPO_DIR}/.venv/bin/python -m pytest -q"
