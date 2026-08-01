#!/bin/bash
# THE ONLY sanctioned way to (re)launch formal DS-V4 training. Run ON node69_slime.
# Does: preflight -> stop old -> ray-clean all nodes -> launch -> wait for the
# real train_async driver. Configuration belongs to the formal run script; this
# wrapper only owns process lifecycle and destructive-operation safety.
# Formal topology/source/horizon are baked in the formal run script. Ambient smoke
# and replay variables are intentionally ignored.  ``--fresh`` is the only
# zero-based formal entrypoint and targets the runner's baked, lineage-specific
# scratch root; ordinary invocation is native resume only.
set -euo pipefail
REPO=/nfs/FM/chenshuailin/projects/kernel_agents/slime-v4flash-lora
LOG=/tmp/dsv4_formal.out
MANAGED_HEAD_IP=10.11.2.169

managed_mode=resume
managed_run_args=()
if (( "$#" == 1 )) && [[ "$1" == "--fresh" ]]; then
  managed_mode=fresh
  managed_run_args=(--fresh)
elif (( "$#" != 0 )); then
  echo "Usage: $0 [--fresh]" >&2
  exit 2
fi

verify_managed_head() {
  local host_ips
  host_ips=" $(hostname -I 2>/dev/null || true) "
  if [[ "${host_ips}" != *" ${MANAGED_HEAD_IP} "* ]]; then
    echo "[managed] FATAL: must run inside node69_slime (${MANAGED_HEAD_IP}); local IPv4s:${host_ips}" >&2
    return 2
  fi
}

# This script runs inside node69_slime.  That container's SSH alias points at
# its externally published port, which is not reachable by hairpin from the
# container itself.  Execute the head command locally and retain SSH for every
# other node so cleanup and idle checks still fail closed fleet-wide.
on_node() {
  local host=$1
  local command=$2
  if [[ "${host}" == "node69_slime" ]]; then
    bash -lc "${command}"
  else
    ssh -o ConnectTimeout=10 "${host}" "${command}"
  fi
}

cleanup_node() {
  local host=$1
  # PID 1 in some rollout containers does not reap historical raylet zombies.
  # Zombies cannot hold GPU/Ray resources and cannot be killed, so reject only
  # non-zombie processes after cleanup while still requiring /tmp/ray removal.
  on_node "${host}" 'set -euo pipefail; ray stop --force >/dev/null 2>&1 || true; pkill -9 -x raylet 2>/dev/null || true; pkill -9 -x gcs_server 2>/dev/null || true; rm -rf -- /tmp/ray; test ! -e /tmp/ray; if ps -C raylet -o stat= 2>/dev/null | awk '\''$1 !~ /^Z/ {live=1} END {exit live ? 0 : 1}'\''; then exit 41; fi; if ps -C gcs_server -o stat= 2>/dev/null | awk '\''$1 !~ /^Z/ {live=1} END {exit live ? 0 : 1}'\''; then exit 42; fi'
}

gpu_inventory() {
  local host=$1
  on_node "${host}" 'timeout --kill-after=5s 20s nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits'
}

parse_gpu_inventory() {
  local raw=$1
  local line index memory extra
  local count=0
  local busy=0
  local -A seen=()
  while IFS= read -r line; do
    [[ -n "${line}" ]] || return 1
    IFS=',' read -r index memory extra <<<"${line}"
    index=${index//[[:space:]]/}
    memory=${memory//[[:space:]]/}
    extra=${extra//[[:space:]]/}
    [[ -z "${extra}" && "${index}" =~ ^[0-9]+$ && "${memory}" =~ ^[0-9]+$ ]] || return 1
    (( index >= 0 && index < 8 )) || return 1
    [[ -z "${seen[${index}]+x}" ]] || return 1
    seen[${index}]=1
    ((count += 1))
    if (( memory > 1024 )); then
      ((busy += 1))
    fi
  done <<<"${raw}"
  (( count == 8 )) || return 1
  for index in {0..7}; do
    [[ -n "${seen[${index}]+x}" ]] || return 1
  done
  printf '%s\n' "${busy}"
}

abort_launch() {
  local pid=$1
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  fi
}

# Allow CPU tests to source the helpers without cleaning a live cluster.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  return 0
fi

verify_managed_head

echo "[managed] stage 0: preflight ${managed_mode} configuration before stopping the current run"
preflight_output="$(bash "$REPO/scripts/dsv4/run.deepseek_v4_flash.fp4.formal.rl.sh" --prepare-only "${managed_run_args[@]}")" \
  || { echo "[managed] FATAL: pre-stop configuration preflight failed" >&2; exit 1; }
printf '%s\n' "${preflight_output}"

echo "[managed] stage 1: stop any existing driver"
mapfile -t oldpids < <(pgrep -f "python3 train_asy[n]c.py" || true)
if (( ${#oldpids[@]} > 0 )); then
  kill -TERM "${oldpids[@]}" 2>/dev/null || true
  for _ in $(seq 1 30); do
    any_alive=0
    for oldpid in "${oldpids[@]}"; do
      if kill -0 "${oldpid}" 2>/dev/null; then
        any_alive=1
        break
      fi
    done
    (( any_alive == 0 )) && break
    sleep 5
  done
  survivors=()
  for oldpid in "${oldpids[@]}"; do
    kill -0 "${oldpid}" 2>/dev/null && survivors+=("${oldpid}")
  done
  (( ${#survivors[@]} == 0 )) || { echo "[managed] FATAL: drivers survived TERM+150s: ${survivors[*]}"; exit 1; }
fi

echo "[managed] stage 2: ray-clean all nodes"
for h in node69_slime node64_slime node53_dspark node70_dspark; do
  cleanup_node "$h" \
    || { echo "[managed] FATAL: cleanup failed on $h"; exit 1; }
done

echo "[managed] stage 3: verify all GPUs idle"
busy=0
for h in node69_slime node64_slime node53_dspark node70_dspark; do
  raw_inventory=$(gpu_inventory "$h" 2>/dev/null) || { echo "[managed] FATAL: gpu check failed on $h (NVML wedge? check container)"; exit 1; }
  n=$(parse_gpu_inventory "${raw_inventory}") || { echo "[managed] FATAL: invalid GPU inventory on $h (expected indices 0..7 and integer memory)" >&2; exit 1; }
  busy=$((busy + n))
done
[ "$busy" -gt 0 ] && { echo "[managed] FATAL: $busy GPUs still busy"; exit 1; }

echo "[managed] stage 4: launch formal PP1xCP2 training (${managed_mode})"
rm -f "$REPO/local_artifacts/deepseek-v4/r2_logs/r6_full_loop.lock"
[ -f "$LOG" ] && mv "$LOG" "${LOG}.prev.$(date -u +%H%M%S)"
# The formal script scrubs every source/start/debug/scheduler override inherited
# from a replay shell before applying its baked configuration.
setsid nohup bash "$REPO/scripts/dsv4/run.deepseek_v4_flash.fp4.formal.rl.sh" "${managed_run_args[@]}" > "$LOG" 2>&1 &
lpid=$!

echo "[managed] stage 5: wait for train_async driver"
driver_pid=""
for _ in $(seq 1 60); do
  sleep 5
  driver_pid=$(pgrep -f "python3 train_asy[n]c.py" | head -1 || true)
  if [[ -n "${driver_pid}" ]]; then
    break
  fi
  kill -0 "$lpid" 2>/dev/null \
    || { echo "[managed] FATAL: launcher died before train_async started"; tail -20 "$LOG"; exit 1; }
done
[[ -n "${driver_pid}" ]] \
  || { abort_launch "$lpid"; echo "[managed] FATAL: train_async did not start within 300s"; tail -20 "$LOG"; exit 1; }
echo "[managed] LAUNCH COMPLETE — train_async pid ${driver_pid}; log: ${LOG}"
