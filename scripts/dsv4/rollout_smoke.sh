#!/bin/bash
# R4 rollout-side smoke for DeepSeek-V4-Flash on node62 only.
#
# This is deliberately not a full RL run. It starts one SGLang TP4 engine,
# generates two short samples, saves debug rollout data, and verifies that
# routing-replay data (`rollout_routed_experts`) is present in the dump.
set -euo pipefail

REPO=${REPO:-/nfs/FM/chenshuailin/projects/kernel_agents/slime-v4flash-lora}
HF_CKPT=${HF_CKPT:-/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8}
PROMPT_DATA=${PROMPT_DATA:-${REPO}/Data/dsv4_rollout_smoke.jsonl}
SCRATCH=${SCRATCH:-/nfs/FM/csl_v4r4_rollout_smoke_node62}
DEBUG_DIR=${DEBUG_DIR:-${SCRATCH}/debug}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOG=${LOG:-${REPO}/local_artifacts/deepseek-v4/r2_logs/r4_node62_rollout_smoke_${RUN_ID}.log}

MASTER_ADDR=${MASTER_ADDR:-10.11.2.162}
RAY_PORT=${RAY_PORT:-6382}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8268}
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/v4r4ro/ray}
RAY_DASHBOARD_AGENT_LISTEN_PORT=${RAY_DASHBOARD_AGENT_LISTEN_PORT:-52366}
RAY_DASHBOARD_AGENT_GRPC_PORT=${RAY_DASHBOARD_AGENT_GRPC_PORT:-38190}
RAY_RUNTIME_ENV_AGENT_PORT=${RAY_RUNTIME_ENV_AGENT_PORT:-34874}
RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-20000000000}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-64}
RAY_JOB_POLL_INTERVAL_SECS=${RAY_JOB_POLL_INTERVAL_SECS:-30}
RAY_JOB_TIMEOUT_SECS=${RAY_JOB_TIMEOUT_SECS:-3600}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
GPUS_PER_ENGINE=${GPUS_PER_ENGINE:-4}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.8}
SGLANG_CHUNKED_PREFILL_SIZE=${SGLANG_CHUNKED_PREFILL_SIZE:-1024}
SGLANG_MAX_PREFILL_TOKENS=${SGLANG_MAX_PREFILL_TOKENS:-1024}
SGLANG_ROUTER_REGISTRATION_TIMEOUT_SECS=${SGLANG_ROUTER_REGISTRATION_TIMEOUT_SECS:-120}
USE_SGLANG_DEEPEP=${USE_SGLANG_DEEPEP:-0}
SGLANG_DP_SIZE=${SGLANG_DP_SIZE:-4}
SGLANG_DEEPEP_CONFIG=${SGLANG_DEEPEP_CONFIG:-'{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'}
CLEANUP_RAY_ON_EXIT=${CLEANUP_RAY_ON_EXIT:-1}
RAY_STARTED=0

export PYTHONUNBUFFERED=1
export RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s:-120}
export PATH="/usr/local/cuda/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
export PYTHONPATH="${REPO}:/root/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=${SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK:-false}
export SGLANG_MEMORY_SAVER_CUDA_GRAPH=${SGLANG_MEMORY_SAVER_CUDA_GRAPH:-true}
export SGLANG_DSV4_FP4_EXPERTS=${SGLANG_DSV4_FP4_EXPERTS:-0}
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-256}
export SGLANG_ROUTER_REGISTRATION_TIMEOUT_SECS
ulimit -n 1048576 || true

NO_PROXY_LIST="127.0.0.1,localhost,0.0.0.0,::1,${MASTER_ADDR},node62,node62_slime"
export no_proxy="${no_proxy:-},${NO_PROXY_LIST}"
export NO_PROXY="${NO_PROXY:-},${NO_PROXY_LIST}"

mkdir -p "${SCRATCH}" "${DEBUG_DIR}" "$(dirname "${LOG}")" "$(dirname "${RAY_TEMP_DIR}")"
cd "${REPO}"

source "${REPO}/scripts/models/deepseek-v4-flash.sh"

cleanup_ray() {
  local rc=$?
  trap - EXIT
  if [[ "${CLEANUP_RAY_ON_EXIT}" == "1" && "${RAY_STARTED}" == "1" ]]; then
    echo "=== stopping Ray on exit (rc=${rc}) ===" | tee -a "${LOG}" || true
    ray stop --force >/dev/null 2>&1 || true
    kill_old_sglang
  fi
  exit "${rc}"
}
trap cleanup_ray EXIT

kill_old_sglang() {
  local patterns=(
    "[s]glang"
    "[V]LLM::EngineCore"
    "[r]ay::SGLangEngine"
    "[r]ay::RolloutManager"
  )
  for pattern in "${patterns[@]}"; do
    pkill -9 -f "${pattern}" >/dev/null 2>&1 || true
  done
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    node62 "$(printf "pkill -9 -f '%s' >/dev/null 2>&1 || true; " "${patterns[@]}")" || true
}

check_gpu_idle() {
  local -r max_mib=1024
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, -v max="${max_mib}" '{gsub(/[^0-9]/, "", $1); gsub(/[^0-9.]/, "", $2); if ($2 + 0 > max) {printf("gpu %s uses %s MiB > %s MiB\n", $1, $2, max); bad=1}} END {exit bad ? 1 : 0}'
}

echo "=== R4 node62 rollout smoke sanity ===" | tee "${LOG}"
echo "repo=${REPO}" | tee -a "${LOG}"
echo "hf=${HF_CKPT}" | tee -a "${LOG}"
echo "prompt_data=${PROMPT_DATA}" | tee -a "${LOG}"
echo "scratch=${SCRATCH}" | tee -a "${LOG}"
echo "ray=${MASTER_ADDR}:${RAY_PORT} dashboard=${RAY_DASHBOARD_PORT}" | tee -a "${LOG}"
echo "raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s}" | tee -a "${LOG}"
echo "ray_job_poll_interval_secs=${RAY_JOB_POLL_INTERVAL_SECS}" | tee -a "${LOG}"
echo "ray_job_timeout_secs=${RAY_JOB_TIMEOUT_SECS}" | tee -a "${LOG}"
echo "sglang_mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC}" | tee -a "${LOG}"
echo "sglang_chunked_prefill_size=${SGLANG_CHUNKED_PREFILL_SIZE}" | tee -a "${LOG}"
echo "sglang_max_prefill_tokens=${SGLANG_MAX_PREFILL_TOKENS}" | tee -a "${LOG}"
echo "sglang_router_registration_timeout_secs=${SGLANG_ROUTER_REGISTRATION_TIMEOUT_SECS}" | tee -a "${LOG}"
echo "sglang_enable_tp_memory_imbalance_check=${SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK}" | tee -a "${LOG}"
echo "sglang_dsv4_fp4_experts=${SGLANG_DSV4_FP4_EXPERTS}" | tee -a "${LOG}"
echo "use_sglang_deepep=${USE_SGLANG_DEEPEP} sglang_dp_size=${SGLANG_DP_SIZE} deepep_config=${SGLANG_DEEPEP_CONFIG}" | tee -a "${LOG}"

test -f "${HF_CKPT}/config.json"
test -f "${PROMPT_DATA}"
count=$(ls -1 "${HF_CKPT}"/*.safetensors 2>/dev/null | wc -l)
if [[ "${count}" -lt 46 ]]; then
  echo "HF checkpoint incomplete: ${count} safetensors, expected 46" | tee -a "${LOG}"
  exit 1
fi
python3 - "${HF_CKPT}" <<'PY' | tee -a "${LOG}"
import json, os, sys
ckpt = sys.argv[1]
cfg = json.load(open(os.path.join(ckpt, "config.json")))
print("model_type=%s num_hidden_layers=%s num_experts=%s" % (
    cfg.get("model_type"), cfg.get("num_hidden_layers"), cfg.get("n_routed_experts")
))
assert cfg.get("model_type") == "deepseek_v4"
PY

DSV4_CHAT_TEMPLATE="${REPO}/examples/kernel_agent/prompt_config/deepseek_v4_chat_template.jinja"
if [[ ! -s "${HF_CKPT}/chat_template.jinja" && -f "${DSV4_CHAT_TEMPLATE}" ]]; then
  cp "${DSV4_CHAT_TEMPLATE}" "${HF_CKPT}/chat_template.jinja"
  echo "installed chat_template.jinja -> ${HF_CKPT}/chat_template.jinja" | tee -a "${LOG}"
fi

echo "=== stopping old Ray and sglang ===" | tee -a "${LOG}"
ray stop --force >/dev/null 2>&1 || true
kill_old_sglang
sleep 3
check_gpu_idle | tee -a "${LOG}"

rm -rf "${RAY_TEMP_DIR}"
mkdir -p "${RAY_TEMP_DIR}"

echo "=== starting Ray head on node62 ===" | tee -a "${LOG}"
ray_start_args=(
  --head
  --node-ip-address "${MASTER_ADDR}"
  --port "${RAY_PORT}"
  --dashboard-host=0.0.0.0
  --dashboard-port "${RAY_DASHBOARD_PORT}"
  --dashboard-agent-listen-port "${RAY_DASHBOARD_AGENT_LISTEN_PORT}"
  --dashboard-agent-grpc-port "${RAY_DASHBOARD_AGENT_GRPC_PORT}"
  --runtime-env-agent-port "${RAY_RUNTIME_ENV_AGENT_PORT}"
  --num-gpus "${ROLLOUT_GPUS}"
  --num-cpus "${RAY_NUM_CPUS}"
  --object-store-memory "${RAY_OBJECT_STORE_MEMORY}"
  --disable-usage-stats
  --temp-dir "${RAY_TEMP_DIR}"
)
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}" ray start "${ray_start_args[@]}" | tee -a "${LOG}"
RAY_STARTED=1

ray status --address "${MASTER_ADDR}:${RAY_PORT}" | tee -a "${LOG}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
  HAS_NVLINK="${HAS_NVLINK:-1}"
else
  HAS_NVLINK="${HAS_NVLINK:-0}"
fi

RUNTIME_ENV_JSON=$(python3 - <<PY
import json, os
env = {
    "PYTHONPATH": "${REPO}:/root/Megatron-LM",
    "PATH": os.environ["PATH"],
    "CUDA_DEVICE_MAX_CONNECTIONS": os.environ["CUDA_DEVICE_MAX_CONNECTIONS"],
    "PYTORCH_CUDA_ALLOC_CONF": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
    "NCCL_SOCKET_IFNAME": os.environ["NCCL_SOCKET_IFNAME"],
    "GLOO_SOCKET_IFNAME": os.environ["GLOO_SOCKET_IFNAME"],
    "MASTER_ADDR": "${MASTER_ADDR}",
    "NO_PROXY": os.environ["NO_PROXY"],
    "no_proxy": os.environ["no_proxy"],
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "NCCL_DEBUG": "WARN",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": os.environ["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"],
    "SGLANG_MEMORY_SAVER_CUDA_GRAPH": os.environ["SGLANG_MEMORY_SAVER_CUDA_GRAPH"],
    "SGLANG_DSV4_FP4_EXPERTS": os.environ["SGLANG_DSV4_FP4_EXPERTS"],
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": os.environ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"],
    "SGLANG_ROUTER_REGISTRATION_TIMEOUT_SECS": os.environ["SGLANG_ROUTER_REGISTRATION_TIMEOUT_SECS"],
}
print(json.dumps({"env_vars": env}))
PY
)

# Request/graph budgets are divided across DP ranks: with dp-attention enabled
# the per-rank request pool is max-running-requests/dp_size — below dp_size it
# floors to 0 and graph capture asserts (capture_bs=[0], req_to_token_pool.size
# feeds get_batch_sizes_to_capture). Keep >= 2 per DP rank when dp-attention is on.
if [[ "${V4_FP4_FROZEN_EXPERTS:-0}" == "1" || "${USE_SGLANG_DEEPEP}" == "1" ]]; then
  SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-$((SGLANG_DP_SIZE * 2))}
  SGLANG_CUDA_GRAPH_MAX_BS=${SGLANG_CUDA_GRAPH_MAX_BS:-$((SGLANG_DP_SIZE * 2))}
else
  SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-2}
  SGLANG_CUDA_GRAPH_MAX_BS=${SGLANG_CUDA_GRAPH_MAX_BS:-2}
fi

SGLANG_ARGS=(
  --rollout-num-gpus "${ROLLOUT_GPUS}"
  --rollout-num-gpus-per-engine "${GPUS_PER_ENGINE}"
  --sglang-context-length 512
  --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
  --sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE}"
  --sglang-max-prefill-tokens "${SGLANG_MAX_PREFILL_TOKENS}"
  --sglang-disable-custom-all-reduce
  --sglang-watchdog-timeout 2400
  --sglang-decode-log-interval 1
  --router-policy round_robin
  --router-queue-timeout-secs 2400
)

# NEXTN/EAGLE speculative decoding. Chain mode (eagle-topk=1) needs draft =
# steps + 1. DELIBERATELY UNGATED for FP4+EAGLE (unlike _dsv4_launch_core.sh's
# hard block): this rollout-only harness IS the investigation tool for the
# known FP4+EAGLE NCCL deadlock (fp4_w4a16_design.md) — launching the wedging
# combo on purpose is its job. Do not use this script for routine validation
# with spec enabled unless you are reproducing/fixing that deadlock.
if [[ "${SGLANG_SPECULATIVE_ALGORITHM:-}" == "DSPARK" ]]; then
  # DSPARK (new runtime): draft config auto-inferred from the -DSpark ckpt;
  # no EAGLE-style steps/topk/draft-tokens. dp-lm-head is a hard requirement
  # under dp-attention. Serve the -DSpark ckpt (V4_ROLLOUT_MODEL_PATH).
  SGLANG_ARGS+=(
    --sglang-speculative-algorithm DSPARK
    --sglang-enable-dp-lm-head
  )
  if [[ -n "${V4_ROLLOUT_MODEL_PATH:-}" ]]; then
    # Rollout serves the -DSpark ckpt variant; trainer stays on --hf-checkpoint.
    SGLANG_ARGS+=(--rollout-model-path "${V4_ROLLOUT_MODEL_PATH}")
  fi
elif [[ -n "${SGLANG_SPECULATIVE_ALGORITHM:-}" && "${SGLANG_SPECULATIVE_ALGORITHM}" != "none" ]]; then
  SGLANG_ARGS+=(
    --sglang-speculative-algorithm "${SGLANG_SPECULATIVE_ALGORITHM}"
    --sglang-speculative-num-steps "${SGLANG_SPECULATIVE_NUM_STEPS:-1}"
    --sglang-speculative-eagle-topk "${SGLANG_SPECULATIVE_EAGLE_TOPK:-1}"
    --sglang-speculative-num-draft-tokens "${SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS:-2}"
  )
  # Spec-stage MoE reroute (FP4+EAGLE NCCL-desync workaround, 2026-07-16): the
  # pre-#23906 fork makes per-rank graph-vs-eager decisions in the spec stages,
  # which desyncs the cross-DP MoE collectives that a2a=none requires. Routing
  # ONLY the draft/verify/draft-extend MoE onto deepep (the fork's
  # speculative_moe_a2a_backend_context) removes the count-sensitive collective.
  if [[ -n "${SGLANG_SPECULATIVE_MOE_A2A_BACKEND:-}" ]]; then
    SGLANG_ARGS+=(
      --sglang-speculative-moe-a2a-backend "${SGLANG_SPECULATIVE_MOE_A2A_BACKEND}"
      --sglang-speculative-moe-runner-backend "${SGLANG_SPECULATIVE_MOE_RUNNER_BACKEND:-deep_gemm}"
    )
  fi
  if [[ -n "${SGLANG_EP_SIZE:-}" ]]; then
    SGLANG_ARGS+=(--sglang-ep-size "${SGLANG_EP_SIZE}")
  fi
fi
if [[ "${V4_FP4_FROZEN_EXPERTS:-0}" == "1" ]]; then
  # Packed-MXFP4 W4A16 serving (official checkpoint): SM90 runners are a2a=none
  # only; runner must be explicit ('auto' falls into Fp8MoEMethod). Mirrors the
  # _dsv4_launch_core.sh FP4 block; design handoffs/deepseek-v4/fp4_w4a16_design.md.
  if [[ "${USE_SGLANG_DEEPEP}" == "1" ]]; then
    echo "FATAL: V4_FP4_FROZEN_EXPERTS=1 requires USE_SGLANG_DEEPEP=0" >&2
    exit 1
  fi
  if [[ "${SGLANG_DSV4_FP4_EXPERTS}" != "1" ]]; then
    echo "FATAL: V4_FP4_FROZEN_EXPERTS=1 requires SGLANG_DSV4_FP4_EXPERTS=1" >&2
    exit 1
  fi
  probed_dtype=$(python3 "${REPO}/scripts/dsv4/probe_expert_dtype.py" "${HF_CKPT}")
  if [[ "${probed_dtype}" != "I8" && "${probed_dtype}" != "U8" ]]; then
    echo "FATAL: HF_CKPT routed experts are ${probed_dtype}, expected packed I8: ${HF_CKPT}" >&2
    exit 1
  fi
  # Keep the shared expert TP1-replicated under a2a=none (production/DeepEP-era
  # semantics; also required for unsharded shared-expert LoRA adapters).
  export SGLANG_SHARED_EXPERT_TP1=${SGLANG_SHARED_EXPERT_TP1:-1}
  SGLANG_ARGS+=(
    --sglang-data-parallel-size "${SGLANG_DP_SIZE}"
    --sglang-enable-dp-attention
    --sglang-moe-a2a-backend none
    --sglang-moe-runner-backend "${SGLANG_MOE_RUNNER_BACKEND:-flashinfer_mxfp4}"
    --sglang-disable-flashinfer-autotune
  )
elif [[ "${USE_SGLANG_DEEPEP}" == "1" ]]; then
  SGLANG_ARGS+=(
    --sglang-data-parallel-size "${SGLANG_DP_SIZE}"
    --sglang-enable-dp-attention
    --sglang-moe-a2a-backend deepep
    --sglang-deepep-config "${SGLANG_DEEPEP_CONFIG}"
  )
fi

echo "=== submitting debug-rollout-only job ===" | tee -a "${LOG}"
JOB_ID=${RAY_JOB_ID:-r4_v4_rollout_smoke_$(date -u +%Y%m%d_%H%M%S)_$$}
JOB_LOG_CAPTURE="${SCRATCH}/ray_job_${JOB_ID}.log"
JOB_LOG_CAPTURE_TMP="${JOB_LOG_CAPTURE}.tmp"
LAST_JOB_LOG_LINES=0
echo "ray_job_id=${JOB_ID}" | tee -a "${LOG}"

append_new_job_logs() {
  ray job logs --address="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}" "${JOB_ID}" >"${JOB_LOG_CAPTURE_TMP}" 2>&1 || return 0
  local total_lines
  total_lines=$(wc -l <"${JOB_LOG_CAPTURE_TMP}")
  if (( total_lines > LAST_JOB_LOG_LINES )); then
    sed -n "$((LAST_JOB_LOG_LINES + 1)),${total_lines}p" "${JOB_LOG_CAPTURE_TMP}" | tee -a "${LOG}"
    LAST_JOB_LOG_LINES="${total_lines}"
  fi
  mv -f "${JOB_LOG_CAPTURE_TMP}" "${JOB_LOG_CAPTURE}"
}

ray job submit --address="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}" \
  --submission-id="${JOB_ID}" \
  --no-wait \
  --working-dir="${REPO}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --num-gpus-per-node 8 \
  --hf-checkpoint "${HF_CKPT}" \
  "${MODEL_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  --prompt-data "${PROMPT_DATA}" \
  --input-key input \
  --label-key label \
  --metadata-key metadata \
  --apply-chat-template \
  --rollout-batch-size 2 \
  --n-samples-per-prompt 1 \
  --global-batch-size 2 \
  --num-rollout 1 \
  --rollout-max-context-len 512 \
  --rollout-max-response-len 16 \
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-1}" \
  --rollout-top-p "${ROLLOUT_TOP_P:-1}" \
  --rm-type random \
  --use-rollout-routing-replay \
  --debug-rollout-only \
  --save-debug-rollout-data "${DEBUG_DIR}/rollout_{rollout_id}.pt" \
  --attention-backend flash 2>&1 | tee -a "${LOG}"

JOB_DEADLINE=$((SECONDS + RAY_JOB_TIMEOUT_SECS))
while true; do
  echo "=== ray job status ${JOB_ID} $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "${LOG}"
  STATUS_OUTPUT=$(ray job status --address="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}" "${JOB_ID}" 2>&1 || true)
  echo "${STATUS_OUTPUT}" | tee -a "${LOG}"
  append_new_job_logs

  if grep -Eiq "SUCCEEDED|succeeded" <<<"${STATUS_OUTPUT}"; then
    break
  fi
  if grep -Eiq "FAILED|STOPPED|failed|stopped" <<<"${STATUS_OUTPUT}"; then
    echo "Ray job ${JOB_ID} ended unsuccessfully" | tee -a "${LOG}"
    append_new_job_logs
    exit 1
  fi
  if (( SECONDS >= JOB_DEADLINE )); then
    echo "Ray job ${JOB_ID} timed out after ${RAY_JOB_TIMEOUT_SECS}s" | tee -a "${LOG}"
    ray job stop --address="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}" "${JOB_ID}" 2>&1 | tee -a "${LOG}" || true
    append_new_job_logs
    exit 1
  fi
  sleep "${RAY_JOB_POLL_INTERVAL_SECS}"
done

append_new_job_logs

python3 scripts/dsv4/verify_rollout_dump.py "${DEBUG_DIR}/rollout_0.pt" | tee -a "${LOG}"

echo "=== R4 node62 rollout smoke PASS ===" | tee -a "${LOG}"
