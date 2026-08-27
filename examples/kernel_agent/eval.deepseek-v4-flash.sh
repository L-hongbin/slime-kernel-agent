#!/bin/bash
# Eval DeepSeek-V4-Flash on the KernelBench-L1 TVM-FFI agent harness (sglang-only,
# --debug-rollout-only). Defaults reproduce the eval-of-record on node164
# (8x H20/sm90, TP=4), KernelGym at 127.0.0.1:20211. REQUIRES Hopper (sm90): the
# deepseek_v4 MoE top-k cluster kernel + DeepGEMM HC-prenorm GEMM are sm90-only,
# and the default triton MoE runner crashes on V4 fp8 ("Hidden size mismatch") so
# we force `--moe-runner-backend marlin`. Megatron is never built in this mode;
# the sourced model config is parsed only. sglang loads the HF fp8 checkpoint.

set -Eeo pipefail
trap 'status=$?; echo "Script exiting with status ${status} at line ${LINENO}: ${BASH_COMMAND}"' EXIT
trap 'status=$?; echo "ERROR status ${status} at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/deepseek-v4-flash.sh"

DEFAULT_MODEL_PATH="/nfs/FM/chenshuailin/checkpoints/deepseek-ai/DeepSeek-V4-Flash"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
HF_MODEL_PATH="${HF_MODEL_PATH:-${MODEL_PATH}}"
MODEL_TAG="$(basename "${HF_MODEL_PATH%/}")"
if [[ ! -f "${HF_MODEL_PATH}/config.json" ]]; then
   echo "HF_MODEL_PATH is not an HF checkpoint (no config.json): ${HF_MODEL_PATH}" >&2
   exit 1
fi

# DeepSeek-V4 checkpoints must provide the tokenizer chat template themselves.
if [[ ! -s "${HF_MODEL_PATH}/chat_template.jinja" ]]; then
   echo "HF_MODEL_PATH does not provide chat_template.jinja: ${HF_MODEL_PATH}/chat_template.jinja" >&2
   exit 1
fi

EVAL_DATA="${EVAL_DATA:-${REPO_ROOT}/Data/kernelbench-level1-validation-tvm-v2/train.parquet}"
if [[ ! -f "${EVAL_DATA}" ]]; then
   echo "EVAL_DATA does not exist: ${EVAL_DATA}" >&2
   exit 1
fi

KERNEL_ENV_URL="${KERNEL_ENV_URL:-http://127.0.0.1:20211}"
KERNEL_BACKEND="${KERNEL_BACKEND:-tvm_ffi}"
REFERENCE_BACKEND="${REFERENCE_BACKEND:-torch}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-8}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-32768}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-32768}"
MAX_TURNS="${MAX_TURNS:-3}"
# TP=4 -> two engines on 8 GPUs. V4-Flash is ~55GB fp8 block-quant; on Ampere
# (no native fp8 on sm80) sglang dequantizes to bf16 (~110GB), which fits in TP=4
# (4x80=320GB) with ample headroom for 32k ctx, while giving 2x rollout concurrency.
GPUS_PER_ENGINE="${GPUS_PER_ENGINE:-4}"
# Conservative max batch for V4 fp8+MTP; bump after confirming headroom.
# cuda-graph-max-bs follows this.
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-32}"
SGLANG_WATCHDOG_TIMEOUT="${SGLANG_WATCHDOG_TIMEOUT:-2400}"
ROUTER_QUEUE_TIMEOUT_SECS="${ROUTER_QUEUE_TIMEOUT_SECS:-2400}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"

# MTP speculative decoding. V4-Flash ships an in-checkpoint MTP/nextn draft
# (mtp.0.*, num_nextn_predict_layers=1); sglang auto-loads it as
# DeepseekV4ForCausalLMNextN from --model-path (NO draft path needed). LOSSLESS:
# only speeds up decode, does NOT change eval results. sglang's deepseek_v4 hook
# REQUIRES algorithm==EAGLE (literal, not NEXTN), eagle-topk==1, and num-steps set;
# with topk==1 it forces num-draft-tokens==num-steps+1. (3,1,4) is the DeepSeek
# default. The hook also auto-sets dsv4 attn backend + fp8 KV + page_size 256.
# Do NOT set SGLANG_OPT_USE_ONLINE_COMPRESS (online-c128 KV is incompatible w/ MTP).
USE_MTP_SPEC="${USE_MTP_SPEC:-1}"
SPEC_ALGO="${SPEC_ALGO:-EAGLE}"
SPEC_NUM_STEPS="${SPEC_NUM_STEPS:-3}"
SPEC_EAGLE_TOPK="${SPEC_EAGLE_TOPK:-1}"
SPEC_NUM_DRAFT_TOKENS="${SPEC_NUM_DRAFT_TOKENS:-4}"
SPEC_DRAFT_PATH="${SPEC_DRAFT_PATH:-}"

# MoE runner backend. The default (triton fused_moe) crashes on V4 fp8 with
# AssertionError "Hidden size mismatch" (TP4/TP8/eager all fail); marlin (W4A16
# MoE, the sglang DeepSeek-V4 cookbook backend, Hopper-only) is required.
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-marlin}"

MASTER_ADDR="${MASTER_ADDR:-10.11.2.164}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
# node164 (head) holds MASTER_ADDR on bond0; gloo needs that iface.
LOCAL_GLOO_SOCKET_IFNAME="${LOCAL_GLOO_SOCKET_IFNAME:-bond0}"

# Bypass the cluster proxy for loopback + in-cluster traffic (KernelGym on loopback).
export no_proxy="127.0.0.1,localhost,0.0.0.0,::1,${MASTER_ADDR}"
export NO_PROXY="${no_proxy}"
ulimit -n 1048576 || true

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
   HAS_NVLINK="${HAS_NVLINK:-1}"
else
   HAS_NVLINK="${HAS_NVLINK:-0}"
fi

EXP_NAME="Eval.TVMFFI.deepseek-v4-flash.${MODEL_TAG}.ctx${MAX_CONTEXT_LEN}.resp${MAX_RESPONSE_LEN}.turn${MAX_TURNS}.n${N_SAMPLES_PER_EVAL_PROMPT}"
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
echo "MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN}  MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN}  MAX_TURNS=${MAX_TURNS}"
echo "N_SAMPLES_PER_EVAL_PROMPT=${N_SAMPLES_PER_EVAL_PROMPT}  GPUS_PER_ENGINE=${GPUS_PER_ENGINE}"
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
   --multi-turn-prompt-config-path "${SCRIPT_DIR}/prompt_config/response_prompt/tvm_ffi_short.yaml"
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
   --rollout-num-gpus-per-engine "${GPUS_PER_ENGINE}"
   --sglang-context-length "${MAX_CONTEXT_LEN}"
   --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
   --sglang-decode-log-interval 400
   --router-policy round_robin
   --router-queue-timeout-secs "${ROUTER_QUEUE_TIMEOUT_SECS}"
   --sglang-cuda-graph-max-bs "${SGLANG_MAX_RUNNING_REQUESTS}"
   --sglang-disable-custom-all-reduce
   --sglang-watchdog-timeout "${SGLANG_WATCHDOG_TIMEOUT}"
   ${MOE_RUNNER_BACKEND:+--sglang-moe-runner-backend ${MOE_RUNNER_BACKEND}}
)

if [[ "${USE_MTP_SPEC}" == "1" ]]; then
   SGLANG_ARGS+=(
      --sglang-speculative-algorithm "${SPEC_ALGO}"
      --sglang-speculative-num-steps "${SPEC_NUM_STEPS}"
      --sglang-speculative-eagle-topk "${SPEC_EAGLE_TOPK}"
      --sglang-speculative-num-draft-tokens "${SPEC_NUM_DRAFT_TOKENS}"
   )
   if [[ -n "${SPEC_DRAFT_PATH}" ]]; then
      SGLANG_ARGS+=(--sglang-speculative-draft-model-path "${SPEC_DRAFT_PATH}")
   fi
   echo "MTP speculative: algo=${SPEC_ALGO} num_steps=${SPEC_NUM_STEPS} topk=${SPEC_EAGLE_TOPK} draft_tokens=${SPEC_NUM_DRAFT_TOKENS} draft_path=${SPEC_DRAFT_PATH:-<auto>}"
fi

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

NO_PROXY_LIST="localhost,127.0.0.1,0.0.0.0,::1,${MASTER_ADDR}"
# Default 0 for Hopper (H20/H100/H200): use the native sm90 cluster top-k kernel.
# On Ampere (sm80, e.g. A800) the deepseek_v4 top-k JIT kernel (topk_v2.cuh uses
# __cluster_dims__ / cooperative_groups::this_cluster, sm90-only) crashes cuda-graph
# capture — set AMPERE_TOPK_FALLBACK=1 there to force the non-cluster fallback path
# (but the sm90-only DeepGEMM HC-prenorm GEMM still won't run on Ampere).
AMPERE_TOPK_FALLBACK="${AMPERE_TOPK_FALLBACK:-0}"
if [[ "${AMPERE_TOPK_FALLBACK}" == "1" ]]; then
   TOPK_ENV='"SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK": "0", "SGLANG_OPT_USE_TOPK_V2": "0", "SGLANG_OPT_USE_FUSED_HASH_TOPK": "0",'
else
   TOPK_ENV=''
fi

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    ${TOPK_ENV}
    "no_proxy": "${NO_PROXY_LIST}",
    "NO_PROXY": "${NO_PROXY_LIST}",
    "NCCL_SOCKET_IFNAME": "${NCCL_SOCKET_IFNAME}",
    "GLOO_SOCKET_IFNAME": "${LOCAL_GLOO_SOCKET_IFNAME}",
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
