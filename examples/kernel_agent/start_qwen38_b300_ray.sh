#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/qwen38_b300_env.sh"

NODE_IP=${NODE_IP:-192.168.99.151}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-$(python -c 'import os; print(len(os.sched_getaffinity(0)))')}
RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-17179869184}
ARGS=(
    --head --node-ip-address "$NODE_IP" --num-cpus "$RAY_NUM_CPUS" --num-gpus 8
    --port 6389 --dashboard-host 0.0.0.0 --dashboard-port 8269
    --node-manager-port 23911 --object-manager-port 23912
    --dashboard-agent-listen-port 23913 --dashboard-agent-grpc-port 23914
    --metrics-export-port 23915 --min-worker-port 24100 --max-worker-port 24999
    --object-store-memory "$RAY_OBJECT_STORE_MEMORY"
    --temp-dir /tmp/ray_qwen38_b300 --disable-usage-stats
)
if [[ "${CONFIG_DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' ray start "${ARGS[@]}"
    printf '\n'
    exit 0
fi
if ! hostname -I | tr ' ' '\n' | grep -Fxq "$NODE_IP"; then
    echo "NODE_IP=$NODE_IP is not assigned to this host" >&2
    exit 1
fi
if (( $(df -B1 --output=avail /dev/shm | tail -1) < RAY_OBJECT_STORE_MEMORY )); then
    echo 'Insufficient /dev/shm; use the dedicated container with --ipc=host' >&2
    exit 1
fi
python - "$NODE_IP" <<'PY'
import socket
import sys

for port in (6389, 8269, 23911, 23912, 23913, 23914, 23915):
    with socket.socket() as connection:
        if connection.connect_ex((sys.argv[1], port)) == 0:
            raise SystemExit(f"Port {port} is already in use; no existing Ray service was stopped")
PY
exec ray start "${ARGS[@]}"
