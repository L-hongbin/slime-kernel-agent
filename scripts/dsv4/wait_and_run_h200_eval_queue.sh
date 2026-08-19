#!/usr/bin/env bash
# Wait for an idle 8xH200 host, then run requested curve points serially.
# Each point is delegated to run_eval_step.sh, which owns the disposable
# --init container, KernelGym limits, artifact checks, and cleanup.

set -Eeuo pipefail

if [[ $# -eq 0 ]]; then
  echo "usage: $0 STEP [STEP ...]" >&2
  exit 2
fi

for step in "$@"; do
  if [[ ! "${step}" =~ ^[0-9]+$ ]] || ((10#${step} % 20 != 0)); then
    echo "STEP must be a non-negative multiple of 20: ${step}" >&2
    exit 2
  fi
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LOCK_PATH=${H200_EVAL_QUEUE_LOCK:-/tmp/csl_v4_h200_eval_queue.lock}
POLL_SECONDS=${H200_EVAL_QUEUE_POLL_SECONDS:-60}
WAIT_FOR_LOCK=${H200_EVAL_QUEUE_WAIT_FOR_LOCK:-0}

exec 9>"${LOCK_PATH}"
if [[ "${WAIT_FOR_LOCK}" == "1" ]]; then
  echo "$(date --iso-8601=seconds) waiting_for_queue_lock=${LOCK_PATH}"
  flock 9
  echo "$(date --iso-8601=seconds) queue_lock_acquired=${LOCK_PATH}"
elif ! flock -n 9; then
  echo "another H200 evaluation queue already holds ${LOCK_PATH}" >&2
  exit 3
fi

host_is_idle() {
  python3 - <<'PY'
import subprocess

rows = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    text=True,
).strip().splitlines()
if len(rows) != 8:
    raise SystemExit(1)
for row in rows:
    _, memory, util = [int(item.strip()) for item in row.split(",")]
    if memory >= 4096 or util >= 20:
        raise SystemExit(1)
PY
}

wait_until_idle() {
  while true; do
    if host_is_idle && ! ss -ltn | grep -Eq ':(6386|8269|52465|52466|52467)\b'; then
      echo "$(date --iso-8601=seconds) h200_idle=PASS"
      return 0
    fi
    echo "$(date --iso-8601=seconds) waiting_for_h200_idle"
    sleep "${POLL_SECONDS}"
  done
}

echo "queue_steps=$*"
for step in "$@"; do
  wait_until_idle
  echo "$(date --iso-8601=seconds) queue_start_step=${step}"
  bash "${SCRIPT_DIR}/run_eval_step.sh" h200 curve "${step}"
  echo "$(date --iso-8601=seconds) queue_complete_step=${step}"
done
echo "$(date --iso-8601=seconds) h200_eval_queue=PASS"
