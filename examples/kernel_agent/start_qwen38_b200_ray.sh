#!/usr/bin/env bash
# Run in csl_slime_032: node14 head first, then node13 worker.
# This starts Ray only; it does not stop an existing cluster or submit training.
set -euo pipefail
case "${1:-}" in
   head) NODE_IP=10.1.17.14; RESOURCE='{"slime_actor":8}' ;;
   worker) NODE_IP=10.1.17.13; RESOURCE='{"slime_rollout":8}' ;;
   *) echo "Usage: $0 {head|worker}" >&2; exit 2 ;;
esac

source /data/ssd1/chenshuailin/b200_runtime/env.sh
# Applied to the raylet at startup, not to a submitted job's runtime_env.
# Linux OOM handling still applies if physical memory is exhausted.
export RAY_memory_monitor_refresh_ms=0

ARGS=(
   --node-ip-address "$NODE_IP" --num-cpus 288 --num-gpus 8
   --resources "$RESOURCE" --object-store-memory 200000000000
   --node-manager-port 23901 --object-manager-port 23902
   --dashboard-agent-listen-port 23903 --dashboard-agent-grpc-port 23904
   --metrics-export-port 23905 --min-worker-port 24000 --max-worker-port 24999
   --disable-usage-stats
)
if [[ "$1" == head ]]; then
   ARGS+=(--head --port 6388 --dashboard-host 0.0.0.0 --dashboard-port 8268
          --temp-dir /data/ssd1/csl_ray_b200)
else
   ARGS+=(--address 10.1.17.14:6388)
fi
exec ray start "${ARGS[@]}"
