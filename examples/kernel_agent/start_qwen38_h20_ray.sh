#!/usr/bin/env bash
# Node70 head + Node69 actor; Node53 and Node64 rollout. Does not stop existing Ray.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/qwen38_h20_env.sh"
case "${1:-}" in
   node70) NODE_IP=10.11.2.170; RESOURCE='{"slime_actor":8}' ;;
   node69) NODE_IP=10.11.2.169; RESOURCE='{"slime_actor":8}' ;;
   node53) NODE_IP=10.11.2.153; RESOURCE='{"slime_rollout":8}' ;;
   node64) NODE_IP=10.11.2.164; RESOURCE='{"slime_rollout":8}' ;;
   *) echo "Usage: $0 {node70|node69|node53|node64}" >&2; exit 2 ;;
esac
if ! hostname -I | tr ' ' '\n' | grep -Fxq "$NODE_IP"; then
   echo "Run $1 on $NODE_IP" >&2
   exit 1
fi

# Separate ports from the pre-existing Qwen3.8 Ray cluster.
# GNU nproc honors OMP_NUM_THREADS=4; use CPU affinity for Ray scheduling capacity.
RAY_NUM_CPUS=${RAY_NUM_CPUS:-$(python -c 'import os; print(len(os.sched_getaffinity(0)))')}
ARGS=(
   --node-ip-address "$NODE_IP" --num-cpus "$RAY_NUM_CPUS" --num-gpus 8
   --resources "$RESOURCE" --object-store-memory "${RAY_OBJECT_STORE_MEMORY:-17179869184}"
   --node-manager-port 25901 --object-manager-port 25902
   --dashboard-agent-listen-port 25903 --dashboard-agent-grpc-port 25904
   --metrics-export-port 25905 --min-worker-port 26000 --max-worker-port 26999
   --disable-usage-stats
)
# Check the live mount: Docker metadata may say 64 MiB even after a tmpfs remount.
# Fall back to the node-local SSD only when shared memory is too small.
if (( $(df -B1 --output=avail /dev/shm | tail -1) < ${RAY_OBJECT_STORE_MEMORY:-17179869184} )); then
   mkdir -p "$H20_RUNTIME/plasma"
   ARGS+=(--plasma-directory "$H20_RUNTIME/plasma")
fi
if [[ "$1" == node70 ]]; then
   ARGS+=(--head --port 6588 --dashboard-host 0.0.0.0 --dashboard-port 8468
          --temp-dir /tmp/ray_qwen38_h20)
else
   ARGS+=(--address 10.11.2.170:6588)
fi
if [[ "${CONFIG_DRY_RUN:-0}" == 1 ]]; then
   printf '%q ' ray start "${ARGS[@]}"
   printf '\n'
   exit 0
fi
exec ray start "${ARGS[@]}"
