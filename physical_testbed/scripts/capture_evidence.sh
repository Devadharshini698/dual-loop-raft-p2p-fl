#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <interface> <peer-pid> <duration-seconds> <output-directory>" >&2
  exit 2
fi

INTERFACE="$1"
PEER_PID="$2"
DURATION="$3"
OUTPUT="$4"
mkdir -p "${OUTPUT}"

timeout "${DURATION}" tcpdump -i "${INTERFACE}" -w "${OUTPUT}/network.pcap" &
TCPDUMP_PID=$!
pidstat -h -r -u -p "${PEER_PID}" 1 "${DURATION}" >"${OUTPUT}/resource-metrics.txt" &
PIDSTAT_PID=$!
wait "${TCPDUMP_PID}" || true
wait "${PIDSTAT_PID}" || true
tc -s qdisc show dev "${INTERFACE}" >"${OUTPUT}/tc-final.txt"
sha256sum "${OUTPUT}"/* >"${OUTPUT}/SHA256SUMS"
