# shellcheck shell=bash
# Stable cluster infrastructure for the V4 full-loop launcher, extracted from
# scripts/dsv4/full_loop_smoke.sh (2026-07-14 refactor). This file holds the
# rarely-changing bring-up/cleanup machinery; all volatile config, derived
# sizing, validation guards, arg assembly and orchestration stay in
# full_loop_smoke.sh.
#
# CONTRACT:
#   * FUNCTIONS ONLY — sourcing this file must not execute anything.
#   * Sourced AFTER the launcher's config section: functions read the caller's
#     globals (REPO, HEAD_HOST, WORKER_HOSTS[...], SSH_OPTS, LOG, RAY_*, ...)
#     at call time, not at source time.
#   * The function bodies were moved VERBATIM and encode production incident
#     fixes (see the inline comments); do not "improve" them.

remote() {
  local host=$1
  shift
  if [[ "${host}" == "${HEAD_HOST}" ]]; then
    "$@"
  else
    ssh ${SSH_OPTS} "${host}" "$*"
  fi
}

check_node() {
  local host=$1
  remote "${host}" test -d "${REPO}"
  remote "${host}" test -f "${RAY_DASHBOARD_AGENT_PATCHER}"
  remote "${host}" test -f "${HF_CKPT}/config.json"
  remote "${host}" test -f "${PROMPT_DATA}"
  # Checkpoint-FAMILY probe per node, BOTH directions (codex launcher review
  # 2026-07-16 finding 2): the official mixed ckpt and the secondary FP8 ckpt
  # have byte-identical config/index; only per-tensor dtype tells them apart,
  # engines load HF_CKPT from each node's LOCAL /nfs, and both families now
  # co-exist on the fleet. A wrong-family (or stale partial) copy on ANY node
  # must fail preflight, not bring-up — in FP8 mode too (packed nibbles read
  # as FP8 bytes would be silent corruption).
  local probed
  probed=$(remote "${host}" python3 "${REPO}/scripts/dsv4/probe_expert_dtype.py" "${HF_CKPT}") \
    || { echo "FATAL: ${host}: expert dtype probe failed for ${HF_CKPT}" >&2; return 1; }
  if [[ "${V4_FP4_FROZEN_EXPERTS:-0}" == "1" ]]; then
    if [[ "${probed}" != "I8" && "${probed}" != "U8" ]]; then
      echo "FATAL: ${host}: ${HF_CKPT} routed experts are ${probed}, expected packed I8 (V4_FP4_FROZEN_EXPERTS=1)" >&2
      return 1
    fi
  else
    if [[ "${probed}" == "I8" || "${probed}" == "U8" ]]; then
      echo "FATAL: ${host}: ${HF_CKPT} routed experts are packed ${probed} (official mixed ckpt) but V4_FP4_FROZEN_EXPERTS!=1" >&2
      return 1
    fi
  fi
}

check_actor_node() {
  local host=$1
  check_node "${host}"
  remote "${host}" test -f "${LOAD}/latest_checkpointed_iteration.txt"
  remote "${host}" test -f "${LOAD}/release/.metadata"
}

kill_old_processes() {
  local patterns=(
    "[1]d\\.sh"
    "[1]p\\.sh"
    "[s]glang"
    "[V]LLM::EngineCore"
    "[r]ay::SGLangEngine"
    "[r]ay::RolloutManager"
    "[r]ay::MegatronTrainRayActor"
    "[t]rain\\.py"
    # Orphaned ray satellites survive `ray stop` when their raylet crashed and
    # keep the FIXED dashboard-agent ports (52365-52367) bound — the next run's
    # agent then fails "Address already in use" and its raylet dies with
    # "node timed out during startup" (bit formal r8a/r8b 2026-07-11).
    "[r]ay/dashboard/agent\\.py"
    "[r]ay/dashboard/dashboard\\.py"
    "[r]untime_env_agent"
    "[g]cs_server"
    "[r]aylet"
  )
  local pattern
  for pattern in "${patterns[@]}"; do
    pkill -9 -f "${pattern}" >/dev/null 2>&1 || true
  done
  local host
  for host in "${WORKER_HOSTS[@]}"; do
    ssh ${SSH_OPTS} "${host}" "$(printf "pkill -9 -f '%s' >/dev/null 2>&1 || true; " "${patterns[@]}")" &
  done
  for host in "${PHYSICAL_CLEAN_HOSTS[@]}"; do
    ssh ${PHYSICAL_SSH_OPTS} "${host}" "$(printf "pkill -9 -f '%s' >/dev/null 2>&1 || true; " "${patterns[@]}")" &
  done
  wait
}

external_sglang_kill_cmd() {
  local patterns=(
    "[1]d\\.sh"
    "[1]p\\.sh"
    "/usr/local/bin/[s]glang serve"
  )
  printf "pkill -9 -f '%s' >/dev/null 2>&1 || true; " "${patterns[@]}"
}

start_external_sglang_guard() {
  local kill_cmd
  kill_cmd=$(external_sglang_kill_cmd)
  local host
  for host in "${PHYSICAL_CLEAN_HOSTS[@]}"; do
    ssh ${PHYSICAL_SSH_OPTS} "${host}" \
      "guard_file='${EXTERNAL_SGLANG_GUARD_FILE}'; touch \"\${guard_file}\"; trap 'rm -f \"\${guard_file}\"; exit 0' TERM INT HUP EXIT; end=\$((\$(date +%s) + ${EXTERNAL_SGLANG_GUARD_SECS})); while [ -e \"\${guard_file}\" ] && [ \$(date +%s) -lt \${end} ]; do ${kill_cmd} sleep ${EXTERNAL_SGLANG_GUARD_POLL_SECS}; done" &
    EXTERNAL_SGLANG_GUARD_PIDS+=("$!")
  done
}

stop_external_sglang_guard() {
  local stop_pids=()
  local host
  for host in "${PHYSICAL_CLEAN_HOSTS[@]}"; do
    ssh ${PHYSICAL_SSH_OPTS} "${host}" "rm -f '${EXTERNAL_SGLANG_GUARD_FILE}'" >/dev/null 2>&1 &
    stop_pids+=("$!")
  done
  local stop_pid
  for stop_pid in "${stop_pids[@]}"; do
    wait "${stop_pid}" >/dev/null 2>&1 || true
  done

  local pid
  for pid in "${EXTERNAL_SGLANG_GUARD_PIDS[@]}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
  for pid in "${EXTERNAL_SGLANG_GUARD_PIDS[@]}"; do
    wait "${pid}" >/dev/null 2>&1 || true
  done
  EXTERNAL_SGLANG_GUARD_PIDS=()
}

check_gpu_idle() {
  local -r max_mib=1024
  local host
  for host in "${ACTOR_PHYSICAL_HOSTS[@]}" "${ROLLOUT_PHYSICAL_HOSTS[@]}"; do
    ssh ${PHYSICAL_SSH_OPTS} "${host}" "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v host=${host} -v max=${max_mib} '{gsub(/[^0-9]/, \"\", \$1); gsub(/[^0-9.]/, \"\", \$2); if (\$2 + 0 > max) {printf(\"%s gpu %s uses %s MiB > %s MiB\\n\", host, \$1, \$2, max); bad=1}} END {exit bad ? 1 : 0}'"
  done
}

wait_gpu_idle() {
  local -r wait_secs=120
  local -r poll_secs=5
  local deadline=$((SECONDS + wait_secs))
  while true; do
    if check_gpu_idle; then
      return 0
    fi
    if (( SECONDS >= deadline )); then
      echo "GPUs did not become idle within ${wait_secs}s"
      check_gpu_idle
      return 1
    fi
    sleep "${poll_secs}"
  done
}

cleanup_ray_cluster() {
  local rc=$?
  trap - EXIT
  stop_external_sglang_guard || true
  if [[ "${CLEANUP_RAY_ON_EXIT}" == "1" && "${RAY_STARTED}" == "1" ]]; then
    echo "=== stopping Ray cluster on exit (rc=${rc}) ===" | tee -a "${LOG}" || true
    ray stop --force >/dev/null 2>&1 || true
    local host
    for host in "${WORKER_HOSTS[@]}"; do
      ssh ${SSH_OPTS} "${host}" "ray stop --force >/dev/null 2>&1 || true" &
    done
    wait || true
    # ray stop reaps Ray actors, but sglang spawns detached scheduler/detokenizer/
    # EngineCore children and the direct driver leaves train.py; pkill those on all
    # physical hosts so a hard failure does not strand orphan compute processes.
    # NOTE: kill_old_processes uses broad `pkill -9 -f` (sglang/ray::/train.py) with no
    # run-id/PGID scoping. On this shared multi-tenant box that can kill OTHER sessions'
    # processes, so it defaults OFF. Enable only when you own the nodes.
    # TODO: scope kills to this run (PGID / env marker) before defaulting on.
    if [[ "${CLEANUP_KILL_ORPHANS_ON_EXIT:-0}" == "1" ]]; then
      echo "=== killing orphan sglang/train processes on exit ===" | tee -a "${LOG}" || true
      kill_old_processes >/dev/null 2>&1 || true
    fi
  fi
  exit "${rc}"
}

patch_ray_dashboard_agent() {
  local host=$1
  remote "${host}" python3 "${RAY_DASHBOARD_AGENT_PATCHER}"
}

# --- Unified env transport ---------------------------------------------------
# Single source of truth for the env that must reach every remote execution
# context: the ray head process, the ssh'd ray workers, and --train-env-vars
# for direct-mode train actors. Replaces the previously hand-maintained
# overlapping env lists that kept drifting: a key added to one list but not the
# others silently starved a context.
#
# Forward wandb auth only when a key is exported (formal RL runs), else wandb
# logging silently fails to authenticate on the actors. NCCL transport settings
# MUST also reach the sglang engines: the Megatron->SGLang weight-sync NCCL
# group spans train ranks AND engine ranks, and mixed transports (train on TCP
# via NCCL_IB_DISABLE=1, engines on IB) fail instantly with ncclRemoteError.
#
# Emits KEY=VALUE pairs, one per line:
#   (a) the explicit CORE list below, each included only if the variable is set
#       (set-but-empty is still included), plus
#   (b) a prefix sweep of every EXPORTED variable matching V4_* SGLANG_*
#       SLIME_DEBUG_* SLIME_PATCH_* (compgen -e: exported scalars only — shell
#       functions, arrays and non-exported launcher-local config vars are
#       excluded).
# Values must be single-line: a newline in any value is a hard error (the
# line-based pair protocol and the ssh env-prefix string cannot carry it).
dsv4_transport_env_pairs() {
  local -a core_keys=(
    PYTHONPATH
    PATH
    CUDA_DEVICE_MAX_CONNECTIONS
    PYTORCH_CUDA_ALLOC_CONF
    TMPDIR
    XDG_CACHE_HOME
    TRITON_CACHE_DIR
    TORCHINDUCTOR_CACHE_DIR
    CUDA_CACHE_PATH
    GLOO_SOCKET_IFNAME
    NCCL_SOCKET_IFNAME
    NCCL_DEBUG
    NCCL_IB_DISABLE
    NCCL_IB_HCA
    NO_PROXY
    no_proxy
    WANDB_API_KEY
    TILELANG_CACHE_DIR
    TILELANG_TMP_DIR
    SLIME_RAY_DASHBOARD_AGENT_EARLY_PORT
    RAY_raylet_start_wait_time_s
    RAY_agent_register_timeout_ms
    # Not covered by the prefix sweep; keep the standard crash-site pinning knob:
    CUDA_LAUNCH_BLOCKING
  )
  local -A _dsv4_env_seen=()
  local key prefix
  for key in "${core_keys[@]}"; do
    if [[ -n "${!key+x}" && -z "${_dsv4_env_seen[${key}]+x}" ]]; then
      if [[ "${!key}" == *$'\n'* ]]; then
        echo "FATAL: dsv4_transport_env_pairs: ${key} value contains a newline" >&2
        return 1
      fi
      printf '%s=%s\n' "${key}" "${!key}"
      _dsv4_env_seen[${key}]=1
    fi
  done
  for prefix in V4_ SGLANG_ SLIME_DEBUG_ SLIME_PATCH_; do
    while IFS= read -r key; do
      if [[ -z "${key}" || -n "${_dsv4_env_seen[${key}]+x}" || -z "${!key+x}" ]]; then
        continue
      fi
      if [[ "${!key}" == *$'\n'* ]]; then
        echo "FATAL: dsv4_transport_env_pairs: ${key} value contains a newline" >&2
        return 1
      fi
      printf '%s=%s\n' "${key}" "${!key}"
      _dsv4_env_seen[${key}]=1
    done < <(compgen -e "${prefix}" || true)
  done
}

# Space-joined KEY=<shell-quoted VALUE> string (trailing space) for embedding
# in the ssh'd worker ray-start command line. Values are printf %q quoted so
# JSON configs etc. survive the remote shell parse.
# WANDB_API_KEY is deliberately excluded: it reaches actors via
# --train-env-vars, and an ssh command line is visible in `ps` on both ends —
# do not start leaking the key there.
dsv4_ray_env_prefix() {
  local pairs line key value out=""
  pairs="$(dsv4_transport_env_pairs)" || return 1
  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    key=${line%%=*}
    value=${line#*=}
    if [[ "${key}" == "WANDB_API_KEY" ]]; then
      continue
    fi
    out+="${key}=$(printf '%q' "${value}") "
  done <<<"${pairs}"
  printf '%s' "${out}"
}

# --train-env-vars JSON (direct-driver train actors): flat {KEY: VALUE} dict
# from the same transported pairs. Ray train actors need explicit env — the
# raylet env is not inherited reliably in direct-driver mode; PYTHONPATH makes
# megatron.training importable in actors, NCCL/proxy keys keep transport sane.
dsv4_build_train_env_vars_json() {
  dsv4_transport_env_pairs | python -c '
import json, sys

env = {}
for line in sys.stdin.read().splitlines():
    if not line:
        continue
    key, _, value = line.partition("=")
    env[key] = value
print(json.dumps(env))
'
}

# --- Ray bring-up -------------------------------------------------------------
# Head start (moved from the launcher). Env transport now via
# dsv4_transport_env_pairs: the head ray process receives the transported pairs
# (minus WANDB_API_KEY, see dsv4_ray_env_prefix) as real argv env assignments via
# env(1) — a value-identical superset of the old inline prefix list.
start_ray_head() {
  rm -rf "${RAY_TMP_ROOT}/head"
  mkdir -p "${RAY_TMP_ROOT}/head"

  echo "=== starting Ray head ${HEAD_IP} ===" | tee -a "${LOG}"
  local pairs line
  local -a head_env=()
  pairs="$(dsv4_transport_env_pairs)" || return 1
  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    if [[ "${line%%=*}" == "WANDB_API_KEY" ]]; then
      continue
    fi
    head_env+=("${line}")
  done <<<"${pairs}"
  env "${head_env[@]}" ray start \
    --head \
    --node-ip-address "${HEAD_IP}" \
    --port "${RAY_PORT}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port "${RAY_DASHBOARD_PORT}" \
    --dashboard-agent-listen-port "${RAY_DASHBOARD_AGENT_LISTEN_PORT}" \
    --dashboard-agent-grpc-port "${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
    --runtime-env-agent-port "${RAY_RUNTIME_ENV_AGENT_PORT}" \
    --system-config "${RAY_SYSTEM_CONFIG_JSON}" \
    --num-gpus "${ACTOR_GPUS_PER_NODE}" \
    --num-cpus "${ACTOR_CPUS_PER_NODE}" \
    --resources "${ACTOR_RESOURCE_JSON}" \
    --object-store-memory "${RAY_OBJECT_STORE_MEMORY}" \
    --disable-usage-stats \
    --temp-dir "${RAY_TMP_ROOT}/head" | tee -a "${LOG}"
  RAY_STARTED=1
}

start_ray_worker() {
  local host=$1
  local ip=$2
  local num_gpus=$3
  local num_cpus=$4
  local resource_json=$5
  local temp_dir="${RAY_TMP_ROOT}/${host}"
  local env_prefix
  env_prefix="$(dsv4_ray_env_prefix)" || return 1

  echo "ray_worker_start host=${host} ip=${ip} gpus=${num_gpus} cpus=${num_cpus} resources=${resource_json}"
  ssh ${SSH_OPTS} "${host}" \
    "ulimit -n 1048576 || true; rm -rf ${temp_dir}; mkdir -p ${temp_dir} ${TMPDIR} ${XDG_CACHE_HOME} ${TRITON_CACHE_DIR} ${TORCHINDUCTOR_CACHE_DIR} ${CUDA_CACHE_PATH}; cd ${REPO} && ${env_prefix}ray start --address ${RAY_HEAD_ADDR} --node-ip-address ${ip} --num-gpus ${num_gpus} --num-cpus ${num_cpus} --resources '${resource_json}' --object-store-memory ${RAY_OBJECT_STORE_MEMORY} --dashboard-agent-listen-port ${RAY_DASHBOARD_AGENT_LISTEN_PORT} --dashboard-agent-grpc-port ${RAY_DASHBOARD_AGENT_GRPC_PORT} --runtime-env-agent-port ${RAY_RUNTIME_ENV_AGENT_PORT} --disable-usage-stats --temp-dir ${temp_dir}" 2>&1
}

wait_for_worker_starts() {
  local rc=0
  local pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      rc=1
    fi
  done
  return "${rc}"
}

# Worker-start orchestration (moved verbatim from the launcher; worker_pids/
# host/ip/i intentionally stay global, as before).
start_ray_workers() {
  echo "=== starting Ray train workers ===" | tee -a "${LOG}"
  worker_pids=()
  for i in "${!TRAIN_WORKER_HOSTS[@]}"; do
    host=${TRAIN_WORKER_HOSTS[$i]}
    ip=${TRAIN_WORKER_IPS[$i]}
    start_ray_worker "${host}" "${ip}" "${ACTOR_GPUS_PER_NODE}" "${ACTOR_CPUS_PER_NODE}" "${ACTOR_RESOURCE_JSON}" | tee -a "${LOG}" &
    worker_pids+=("$!")
  done
  wait_for_worker_starts "${worker_pids[@]}"

  echo "=== starting Ray rollout workers ===" | tee -a "${LOG}"
  worker_pids=()
  for i in "${!ROLLOUT_WORKER_HOSTS[@]}"; do
    host=${ROLLOUT_WORKER_HOSTS[$i]}
    ip=${ROLLOUT_WORKER_IPS[$i]}
    start_ray_worker "${host}" "${ip}" "${ROLLOUT_GPUS_PER_NODE}" "${ROLLOUT_CPUS_PER_NODE}" "${ROLLOUT_RESOURCE_JSON}" | tee -a "${LOG}" &
    worker_pids+=("$!")
  done
  wait_for_worker_starts "${worker_pids[@]}"
}
