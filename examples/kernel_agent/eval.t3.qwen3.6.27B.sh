#!/bin/bash

set -Eeo pipefail
trap 'status=$?; echo "Script exiting with status ${status} at line ${LINENO}: ${BASH_COMMAND}"' EXIT
trap 'status=$?; echo "ERROR status ${status} at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

export PYTHONUNBUFFERED=1
ulimit -n 1048576 || true

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3.5-27B.sh"

DEFAULT_MODEL_PATH="/data/FM/checkpoints/KernelRl_ckpt/hf_ckpt/Kernel-FullAsync-TVMFFI-Qwen3.6-27B-DrkernelRlThinkingTVMV2-Contest36KResponse10K-iter104"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
HF_MODEL_PATH="${HF_MODEL_PATH:-${MODEL_PATH}}"
MODEL_TAG="$(basename "${HF_MODEL_PATH%/}")"
if [[ ! -f "${HF_MODEL_PATH}/config.json" ]]; then
   echo "HF_MODEL_PATH is not an HF checkpoint (no config.json): ${HF_MODEL_PATH}" >&2
   echo "Set MODEL_PATH=/path/to/hf_checkpoint or HF_MODEL_PATH=/path/to/hf_checkpoint." >&2
   exit 1
fi

EVAL_DATA="${EVAL_DATA:-${REPO_ROOT}/Data/kernelbench-level1-validation-tvm-v2/eval_with_entry_point.parquet}"
if [[ ! -f "${EVAL_DATA}" ]]; then
   echo "EVAL_DATA does not exist: ${EVAL_DATA}" >&2
   exit 1
fi

KERNEL_ENV_URL="${KERNEL_ENV_URL:-http://192.168.112.2:20111}"
KERNEL_BACKEND="${KERNEL_BACKEND:-tvm_ffi}"
REFERENCE_BACKEND="${REFERENCE_BACKEND:-torch}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-8}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-36000}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-10000}"
MAX_TURNS="${MAX_TURNS:-3}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-16}"
SGLANG_WATCHDOG_TIMEOUT="${SGLANG_WATCHDOG_TIMEOUT:-2400}"
ROUTER_QUEUE_TIMEOUT_SECS="${ROUTER_QUEUE_TIMEOUT_SECS:-2400}"
EAGLE_DRAFT_TOKENS="${EAGLE_DRAFT_TOKENS:-4}"
SGLANG_LINEAR_ATTN_BACKEND="${SGLANG_LINEAR_ATTN_BACKEND:-triton}"

MASTER_ADDR="${MASTER_ADDR:-192.168.112.24}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
LOCAL_GLOO_SOCKET_IFNAME="${LOCAL_GLOO_SOCKET_IFNAME:-$(ip -o -4 addr show 2>/dev/null | awk -v ip="${MASTER_ADDR}" '$4 ~ "^"ip"/" {print $2; exit}')}"
LOCAL_GLOO_SOCKET_IFNAME="${LOCAL_GLOO_SOCKET_IFNAME:-bond0}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
   HAS_NVLINK="${HAS_NVLINK:-1}"
else
   HAS_NVLINK="${HAS_NVLINK:-0}"
fi

EXP_NAME="Eval.TVMFFI.Qwen3.6-27B.${MODEL_TAG}.ctx${MAX_CONTEXT_LEN}.resp${MAX_RESPONSE_LEN}.turn${MAX_TURNS}.n${N_SAMPLES_PER_EVAL_PROMPT}"
EXP_ROOT="${EXP_ROOT:-${REPO_ROOT}/experiments/${EXP_NAME}}"
EVAL_TAG="${EVAL_TAG:-${MODEL_TAG}}"
EVAL_DIR="${EVAL_DIR:-${EXP_ROOT}/${EVAL_TAG}}"
DUMP_DIR="${EVAL_DIR}/dumps"
mkdir -p "${EVAL_DIR}"

LOG_STAMP="$(date +%Y%m%d.%H%M%S)"
LOG_PATH="${EVAL_DIR}/${LOG_STAMP}.log"
echo "Logging to ${LOG_PATH}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "HF_MODEL_PATH=${HF_MODEL_PATH}"
echo "EVAL_DATA=${EVAL_DATA}"
echo "KERNEL_ENV_URL=${KERNEL_ENV_URL}"
echo "DUMP_DIR=${DUMP_DIR}"
echo "MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN}"
echo "MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN}"
echo "MAX_TURNS=${MAX_TURNS}"
echo "N_SAMPLES_PER_EVAL_PROMPT=${N_SAMPLES_PER_EVAL_PROMPT}"
echo "HAS_NVLINK=${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

if ! python3 "${REPO_ROOT}/scripts/check_kernelgym_health.py" \
   --url "${KERNEL_ENV_URL}" \
   --timeout "${KERNELGYM_HEALTH_TIMEOUT:-5}" \
   --attempts "${KERNELGYM_HEALTH_ATTEMPTS:-3}" \
   --interval "${KERNELGYM_HEALTH_INTERVAL:-2}"; then
   echo "KernelGym health check failed at ${KERNEL_ENV_URL}" >&2
   exit 1
fi

ray stop --force || true
sleep 2

EVAL_ARGS=(
   --num-rollout 0
   --eval-interval 1
   --eval-prompt-data kb_l1_val "${EVAL_DATA}"
   --eval-input-key prompt
   --eval-label-key reward_model
   --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
   --debug-rollout-only
   --dump-details "${DUMP_DIR}"
)

# prompt-data is required by the loader even in eval-only; reuse the eval set.
ROLLOUT_ARGS=(
   --prompt-data "${EVAL_DATA}"
   --input-key prompt
   --label-key reward_model
   --metadata-key extra_info
   --rollout-batch-size 16
   --n-samples-per-prompt 1
   --rollout-max-response-len "${MAX_RESPONSE_LEN}"
   --rollout-max-context-len "${MAX_CONTEXT_LEN}"
   --apply-chat-template-kwargs '{"enable_thinking":true}'
   --rollout-temperature 1
)

CUSTOM_ARGS=(
   --custom-generate-function-path examples.kernel_agent.generate_with_cuda_agent.generate
   --custom-rm-path examples.kernel_agent.generate_with_cuda_agent.reward_func
   --multi-turn-prompt-config-path "${SCRIPT_DIR}/prompt_config/multi_turn_cuda_kernel.yaml"
)

KERNEL_AGENT_ARGS=(
   --kernel-env-url "${KERNEL_ENV_URL}"
   --kernel-backend "${KERNEL_BACKEND}"
   --reference-backend "${REFERENCE_BACKEND}"
   --do-precheck
   --use-reference-cache
   --use-multi-turn
   # Keep all generated turns for eval; otherwise the parser default is finalize-mode=positive.
   --finalize-mode none
   --max-turns "${MAX_TURNS}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-context-length "${MAX_CONTEXT_LEN}"
   --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
   --sglang-mem-fraction-static 0.7
   --sglang-decode-log-interval 400
   --router-policy round_robin
   --router-queue-timeout-secs "${ROUTER_QUEUE_TIMEOUT_SECS}"
   --sglang-cuda-graph-max-bs "${SGLANG_MAX_RUNNING_REQUESTS}"
   --sglang-disable-custom-all-reduce
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens "${EAGLE_DRAFT_TOKENS}"
   --sglang-linear-attn-backend "${SGLANG_LINEAR_ATTN_BACKEND}"
   --sglang-mamba-scheduler-strategy extra_buffer
   --sglang-watchdog-timeout "${SGLANG_WATCHDOG_TIMEOUT}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-backend flash
)

export MASTER_ADDR
GLOO_SOCKET_IFNAME="${LOCAL_GLOO_SOCKET_IFNAME}" ray start \
   --head \
   --node-ip-address "${MASTER_ADDR}" \
   --port "${RAY_PORT}" \
   --dashboard-host 0.0.0.0 \
   --dashboard-port "${RAY_DASHBOARD_PORT}" \
   --num-gpus "${GPUS_PER_NODE}" \
   --num-cpus "${RAY_NUM_CPUS}" \
   --disable-usage-stats \
   --temp-dir="${RAY_TEMP_DIR}"

NO_PROXY_LIST="localhost,127.0.0.1,0.0.0.0,::1,${MASTER_ADDR},192.168.112.2"
RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    "no_proxy": "${NO_PROXY_LIST}",
    "NO_PROXY": "${NO_PROXY_LIST}",
    "NCCL_SOCKET_IFNAME": "${NCCL_SOCKET_IFNAME}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "PYTHONPATH": ".:/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "CUDA_AGENT_LOG_MULTI_TURN_TEXT": "0",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "NCCL_DEBUG": "WARN"
  }
}
EOF_JSON
)

ray job submit --address="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}" \
   --working-dir="${REPO_ROOT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-gpus-per-node "${GPUS_PER_NODE}" \
   --colocate \
   --hf-checkpoint "${HF_MODEL_PATH}" \
   "${MODEL_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}" \
   "${KERNEL_AGENT_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"

SUMMARY_PATH="${EVAL_DIR}/summary.${LOG_STAMP}.txt"
python3 "${SCRIPT_DIR}/summarize_eval.py" "${EVAL_DIR}" --max-turns "${MAX_TURNS}" | tee "${SUMMARY_PATH}"
echo "=== eval complete; dumps -> ${DUMP_DIR} ==="
echo "summary -> ${SUMMARY_PATH}"
