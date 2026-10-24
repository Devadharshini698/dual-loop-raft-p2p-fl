#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <controller-host>" >&2
  exit 2
fi

CONFIG_DIR="${HOME}/.flwr"
CONFIG_FILE="${CONFIG_DIR}/config.toml"
mkdir -p "${CONFIG_DIR}"
if [[ -f "${CONFIG_FILE}" ]]; then
  cp "${CONFIG_FILE}" "${CONFIG_FILE}.backup.$(date +%Y%m%d%H%M%S)"
fi

cat >"${CONFIG_FILE}" <<EOF
[superlink]
default = "college-lab"

[superlink.college-lab]
address = "http://$1:8000"
insecure = true
EOF

echo "Flower CLI configured. This insecure profile is only for the isolated lab VLAN."
