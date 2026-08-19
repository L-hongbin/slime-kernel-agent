#!/bin/bash
# Eval DeepSeek-V4-Flash on the KernelBench TVM-FFI agent harness (sglang-only,
# --debug-rollout-only). The default dataset remains Level 1. Base-model defaults reproduce the eval-of-record on node164
# (8x H20/sm90, TP=4), KernelGym at 127.0.0.1:20211. REQUIRES Hopper (sm90): the
# deepseek_v4 MoE top-k cluster kernel + DeepGEMM HC-prenorm GEMM are sm90-only,
# and the default triton MoE runner crashes on V4 fp8 ("Hidden size mismatch") so
# we force `--moe-runner-backend marlin`. Megatron is never built in this mode;
# the sourced model config is parsed only. sglang loads the HF checkpoint.
#
# Optional V4_RUNTIME=dspark mode serves the production packed-MXFP4 DSpark
# checkpoint on one 8xH20 DP-attention/MoE-TP8 engine. LORA_ADAPTER_PATH enables
# static PEFT LoRA serving: SGLang preloads the adapter and every generation
# request explicitly carries LORA_NAME. The base weights are never merged or
# rewritten. EVAL_NUM_PROMPTS provides a bounded smoke subset.

set -Eeo pipefail
trap 'status=$?; echo "Script exiting with status ${status} at line ${LINENO}: ${BASH_COMMAND}"' EXIT
trap 'status=$?; echo "ERROR status ${status} at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/deepseek-v4-flash.sh"

V4_RUNTIME="${V4_RUNTIME:-fleet}"
RAY_WORKER_HOST="${RAY_WORKER_HOST:-}"
case "${V4_RUNTIME}" in
   fleet) DEFAULT_MODEL_PATH="/nfs/FM/chenshuailin/checkpoints/deepseek-ai/DeepSeek-V4-Flash" ;;
   dspark) DEFAULT_MODEL_PATH="/nfs/FM/chenshuailin/checkpoints/deepseek-ai/DeepSeek-V4-Flash-DSpark" ;;
   *) echo "V4_RUNTIME must be fleet or dspark, got: ${V4_RUNTIME}" >&2; exit 1 ;;
esac
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
HF_MODEL_PATH="${HF_MODEL_PATH:-${MODEL_PATH}}"
MODEL_TAG="$(basename "${HF_MODEL_PATH%/}")"
if [[ ! -f "${HF_MODEL_PATH}/config.json" ]]; then
   echo "HF_MODEL_PATH is not an HF checkpoint (no config.json): ${HF_MODEL_PATH}" >&2
   exit 1
fi

# DeepSeek-V4 ships no jinja chat_template; install the repo's byte-exact one so a
# default run reproduces the eval (AutoTokenizer auto-loads <ckpt>/chat_template.jinja).
DSV4_CHAT_TEMPLATE="${SCRIPT_DIR}/prompt_config/deepseek_v4_chat_template.jinja"
if [[ ! -s "${HF_MODEL_PATH}/chat_template.jinja" && -f "${DSV4_CHAT_TEMPLATE}" ]]; then
   cp "${DSV4_CHAT_TEMPLATE}" "${HF_MODEL_PATH}/chat_template.jinja"
   echo "installed chat_template.jinja -> ${HF_MODEL_PATH}/chat_template.jinja"
fi

EVAL_DATA="${EVAL_DATA:-${REPO_ROOT}/Data/kernelbench-level1-validation-tvm-v2/train.parquet}"
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-kb_l1_val}"
EVAL_CONFIG="${EVAL_CONFIG:-}"
EVAL_SUMMARY_GROUP_KEY="${EVAL_SUMMARY_GROUP_KEY:-}"
if [[ ! -f "${EVAL_DATA}" ]]; then
   echo "EVAL_DATA does not exist: ${EVAL_DATA}" >&2
   exit 1
fi
if [[ -n "${EVAL_CONFIG}" && ! -f "${EVAL_CONFIG}" ]]; then
   echo "EVAL_CONFIG does not exist: ${EVAL_CONFIG}" >&2
   exit 1
fi

KERNEL_ENV_URL="${KERNEL_ENV_URL:-http://127.0.0.1:20211}"
KERNEL_BACKEND="${KERNEL_BACKEND:-tvm_ffi}"
REFERENCE_BACKEND="${REFERENCE_BACKEND:-torch}"
KERNEL_EVAL_WORKER_MAX_CONCURRENCY="${KERNEL_EVAL_WORKER_MAX_CONCURRENCY:-32}"
KERNEL_EVAL_RATE_LIMIT="${KERNEL_EVAL_RATE_LIMIT:-32}"
KERNEL_EVAL_PRIORITY="${KERNEL_EVAL_PRIORITY:-normal}"
KERNEL_AGENT_GENERATE_GUARD_SEC="${KERNEL_AGENT_GENERATE_GUARD_SEC:-}"
if [[ -n "${KERNEL_AGENT_GENERATE_GUARD_SEC}" &&
      ! "${KERNEL_AGENT_GENERATE_GUARD_SEC}" =~ ^[1-9][0-9]*$ ]]; then
   echo "KERNEL_AGENT_GENERATE_GUARD_SEC must be a positive integer when set" >&2
   exit 2
fi
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-8}"
EVAL_NUM_PROMPTS="${EVAL_NUM_PROMPTS:-0}"
ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-32768}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-32768}"
MAX_TURNS="${MAX_TURNS:-3}"
APPLY_CHAT_TEMPLATE_KWARGS="${APPLY_CHAT_TEMPLATE_KWARGS:-}"
if [[ -z "${APPLY_CHAT_TEMPLATE_KWARGS}" ]]; then
   APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking":true}'
fi
# TP=4 -> two engines on 8 GPUs. V4-Flash is ~55GB fp8 block-quant; on Ampere
# (no native fp8 on sm80) sglang dequantizes to bf16 (~110GB), which fits in TP=4
# (4x80=320GB) with ample headroom for 32k ctx, while giving 2x rollout concurrency.
GPUS_PER_ENGINE="${GPUS_PER_ENGINE:-$([[ "${V4_RUNTIME}" == "dspark" ]] && echo 8 || echo 4)}"
# Conservative max batch for V4 fp8+MTP; bump after confirming headroom.
# cuda-graph-max-bs follows this.
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-$([[ "${V4_RUNTIME}" == "dspark" ]] && echo 128 || echo 32)}"
SGLANG_WATCHDOG_TIMEOUT="${SGLANG_WATCHDOG_TIMEOUT:-2400}"
ROUTER_QUEUE_TIMEOUT_SECS="${ROUTER_QUEUE_TIMEOUT_SECS:-2400}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-}"
SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-}"
if [[ "${V4_RUNTIME}" == "dspark" ]]; then
   SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-2048}"
   SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-${MAX_CONTEXT_LEN}}"
fi

# Static LoRA serving. Keep the server LoRA-enabled for the base control too,
# so base and adapter checkpoints use the same unfused/wrapped execution path.
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-}"
LORA_NAME="${LORA_NAME:-eval_lora}"
ENABLE_LORA_SERVER="${ENABLE_LORA_SERVER:-$([[ "${V4_RUNTIME}" == "dspark" ]] && echo 1 || echo 0)}"
SGLANG_MAX_LORA_RANK="${SGLANG_MAX_LORA_RANK:-32}"
SGLANG_MAX_LORAS_PER_BATCH="${SGLANG_MAX_LORAS_PER_BATCH:-2}"
SGLANG_LORA_TARGET_MODULES="${SGLANG_LORA_TARGET_MODULES:-wq_a wkv wq_b wo_b wkv_gate gate_up_proj down_proj}"
SGLANG_LORA_BACKEND="${SGLANG_LORA_BACKEND:-triton}"
if [[ -n "${LORA_ADAPTER_PATH}" ]]; then
   if [[ -n "${RAY_WORKER_HOST}" ]]; then
      LORA_FILES_OK=0
      if ssh "${RAY_WORKER_HOST}" \
         "test -s '${LORA_ADAPTER_PATH}/adapter_model.safetensors' && test -s '${LORA_ADAPTER_PATH}/adapter_config.json'"; then
         LORA_FILES_OK=1
      fi
   elif [[ -s "${LORA_ADAPTER_PATH}/adapter_model.safetensors" && -s "${LORA_ADAPTER_PATH}/adapter_config.json" ]]; then
      LORA_FILES_OK=1
   else
      LORA_FILES_OK=0
   fi
   if [[ "${LORA_FILES_OK}" != "1" ]]; then
      echo "Invalid PEFT adapter directory: ${LORA_ADAPTER_PATH}" >&2
      exit 1
   fi
   if [[ "${ENABLE_LORA_SERVER}" != "1" ]]; then
      echo "LORA_ADAPTER_PATH requires ENABLE_LORA_SERVER=1" >&2
      exit 1
   fi
fi

# MTP speculative decoding. V4-Flash ships an in-checkpoint MTP/nextn draft
# (mtp.0.*, num_nextn_predict_layers=1); sglang auto-loads it as
# DeepseekV4ForCausalLMNextN from --model-path (NO draft path needed). LOSSLESS:
# only speeds up decode, does NOT change eval results. sglang's deepseek_v4 hook
# REQUIRES algorithm==EAGLE (literal, not NEXTN), eagle-topk==1, and num-steps set;
# with topk==1 it forces num-draft-tokens==num-steps+1. (3,1,4) is the DeepSeek
# default. The hook also auto-sets dsv4 attn backend + fp8 KV + page_size 256.
# Do NOT set SGLANG_OPT_USE_ONLINE_COMPRESS (online-c128 KV is incompatible w/ MTP).
USE_MTP_SPEC="${USE_MTP_SPEC:-1}"
SPEC_ALGO="${SPEC_ALGO:-$([[ "${V4_RUNTIME}" == "dspark" ]] && echo DSPARK || echo EAGLE)}"
SPEC_NUM_STEPS="${SPEC_NUM_STEPS:-3}"
SPEC_EAGLE_TOPK="${SPEC_EAGLE_TOPK:-1}"
SPEC_NUM_DRAFT_TOKENS="${SPEC_NUM_DRAFT_TOKENS:-4}"
SPEC_DRAFT_PATH="${SPEC_DRAFT_PATH:-}"
SPEC_DSPARK_BLOCK_SIZE="${SPEC_DSPARK_BLOCK_SIZE:-3}"

# MoE runner backend. The default (triton fused_moe) crashes on V4 fp8 with
# AssertionError "Hidden size mismatch" (TP4/TP8/eager all fail); marlin (W4A16
# MoE, the sglang DeepSeek-V4 cookbook backend, Hopper-only) is required.
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-$([[ "${V4_RUNTIME}" == "dspark" ]] && echo flashinfer_mxfp4 || echo marlin)}"

MASTER_ADDR="${MASTER_ADDR:-10.11.2.164}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_AGENT_LISTEN_PORT="${RAY_DASHBOARD_AGENT_LISTEN_PORT:-52365}"
RAY_DASHBOARD_AGENT_GRPC_PORT="${RAY_DASHBOARD_AGENT_GRPC_PORT:-52366}"
RAY_RUNTIME_ENV_AGENT_PORT="${RAY_RUNTIME_ENV_AGENT_PORT:-52367}"
RAY_DASHBOARD_AGENT_READY_TIMEOUT_SECS="${RAY_DASHBOARD_AGENT_READY_TIMEOUT_SECS:-120}"
# The pinned eval image carries the same early-port patch used by the formal
# V4 cluster launcher.  H20 dashboard-module loading can take longer than
# raylet's fixed dashboard-agent port-file deadline; publishing the fixed
# listen port before loading ReporterAgent avoids a control-plane-only race.
SLIME_RAY_DASHBOARD_AGENT_EARLY_PORT="${SLIME_RAY_DASHBOARD_AGENT_EARLY_PORT:-1}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray}"
RAY_RESTART_SETTLE_SECS="${RAY_RESTART_SETTLE_SECS:-10}"
RAY_HEAD_GPUS="${RAY_HEAD_GPUS:-$([[ -n "${RAY_WORKER_HOST}" ]] && echo 0 || echo "${GPUS_PER_NODE}")}"
RAY_WORKER_IP="${RAY_WORKER_IP:-}"
RAY_WORKER_TEMP_DIR="${RAY_WORKER_TEMP_DIR:-${RAY_TEMP_DIR}}"
RAYLET_START_WAIT_TIME_SECS="${RAYLET_START_WAIT_TIME_SECS:-60}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-${GPUS_PER_NODE}}"
if [[ -n "${RAY_WORKER_HOST}" && -z "${RAY_WORKER_IP}" ]]; then
   echo "RAY_WORKER_IP is required when RAY_WORKER_HOST is set" >&2
   exit 1
fi
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

LORA_TAG="$([[ -n "${LORA_ADAPTER_PATH}" ]] && echo "lora-${LORA_NAME}" || echo base)"
EXP_NAME="Eval.TVMFFI.deepseek-v4-flash.${V4_RUNTIME}.${MODEL_TAG}.${LORA_TAG}.ctx${MAX_CONTEXT_LEN}.resp${MAX_RESPONSE_LEN}.turn${MAX_TURNS}.n${N_SAMPLES_PER_EVAL_PROMPT}"
EXP_ROOT="${EXP_ROOT:-${REPO_ROOT}/experiments/${EXP_NAME}}"
EVAL_TAG="${EVAL_TAG:-${MODEL_TAG}}"
if [[ "${EVAL_NUM_PROMPTS}" -gt 0 ]]; then
   EVAL_TAG="${EVAL_TAG}.first${EVAL_NUM_PROMPTS}"
fi
EVAL_DIR="${EVAL_DIR:-${EXP_ROOT}/${EVAL_TAG}}"
DUMP_DIR="${EVAL_DIR}/dumps"
mkdir -p "${EVAL_DIR}"

if [[ "${EVAL_NUM_PROMPTS}" -gt 0 ]]; then
   SUBSET_PARQUET="${EVAL_DIR}/eval_subset.first${EVAL_NUM_PROMPTS}.parquet"
   python3 - "${EVAL_DATA}" "${SUBSET_PARQUET}" "${EVAL_NUM_PROMPTS}" <<'PY'
import sys

import pandas as pd

src, dst, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
df = pd.read_parquet(src)
df.head(count).to_parquet(dst, index=False)
print(f"wrote eval subset: {dst} rows={min(count, len(df))}")
PY
   EVAL_DATA="${SUBSET_PARQUET}"
fi

LOG_STAMP="$(date +%Y%m%d.%H%M%S)"
LOG_PATH="${EVAL_DIR}/${LOG_STAMP}.log"
echo "Logging to ${LOG_PATH}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "HF_MODEL_PATH=${HF_MODEL_PATH}"
echo "V4_RUNTIME=${V4_RUNTIME}  LORA_SERVER=${ENABLE_LORA_SERVER}  LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-<base>}  LORA_NAME=${LORA_NAME}"
echo "EVAL_DATA=${EVAL_DATA}"
echo "EVAL_DATASET_NAME=${EVAL_DATASET_NAME}"
echo "EVAL_CONFIG=${EVAL_CONFIG:-<none>}  EVAL_SUMMARY_GROUP_KEY=${EVAL_SUMMARY_GROUP_KEY:-<none>}"
echo "KERNEL_ENV_URL=${KERNEL_ENV_URL}"
echo "KERNEL_EVAL_WORKER_MAX_CONCURRENCY=${KERNEL_EVAL_WORKER_MAX_CONCURRENCY}  KERNEL_EVAL_RATE_LIMIT=${KERNEL_EVAL_RATE_LIMIT}  KERNEL_EVAL_PRIORITY=${KERNEL_EVAL_PRIORITY}"
echo "DUMP_DIR=${DUMP_DIR}"
echo "MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN}  MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN}  MAX_TURNS=${MAX_TURNS}"
echo "APPLY_CHAT_TEMPLATE_KWARGS=${APPLY_CHAT_TEMPLATE_KWARGS}"
echo "N_SAMPLES_PER_EVAL_PROMPT=${N_SAMPLES_PER_EVAL_PROMPT}  EVAL_NUM_PROMPTS=${EVAL_NUM_PROMPTS}  ROLLOUT_SEED=${ROLLOUT_SEED}  TEMPERATURE=${ROLLOUT_TEMPERATURE}  TOP_P=${ROLLOUT_TOP_P}  GPUS_PER_ENGINE=${GPUS_PER_ENGINE}"
echo "RAY_HEAD=${MASTER_ADDR} head_gpus=${RAY_HEAD_GPUS}  RAY_WORKER=${RAY_WORKER_HOST:-<local>} worker_ip=${RAY_WORKER_IP:-<local>}  ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS}"
echo "RAYLET_START_WAIT_TIME_SECS=${RAYLET_START_WAIT_TIME_SECS}"
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
if [[ -n "${RAY_WORKER_HOST}" ]]; then
   ssh "${RAY_WORKER_HOST}" "ray stop --force || true"
fi
sleep "${RAY_RESTART_SETTLE_SECS}"

EVAL_ARGS=(
   --num-rollout 0
   --eval-interval 1
   --eval-input-key prompt
   --eval-label-key reward_model
   --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
   --debug-rollout-only
   --dump-details "${DUMP_DIR}"
)
if [[ -n "${EVAL_CONFIG}" ]]; then
   EVAL_ARGS+=(--eval-config "${EVAL_CONFIG}")
else
   EVAL_ARGS+=(--eval-prompt-data "${EVAL_DATASET_NAME}" "${EVAL_DATA}")
fi

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
   --rollout-seed "${ROLLOUT_SEED}"
   --apply-chat-template-kwargs "${APPLY_CHAT_TEMPLATE_KWARGS}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE}"
   --rollout-top-p "${ROLLOUT_TOP_P}"
)

CUSTOM_ARGS=(
   --custom-generate-function-path examples.kernel_agent.generate_with_cuda_agent.generate
   --custom-rm-path examples.kernel_agent.generate_with_cuda_agent.reward_func
   --multi-turn-prompt-config-path "${SCRIPT_DIR}/prompt_config/multi_turn_tvm_ffi_short.yaml"
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
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
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
if [[ -n "${SGLANG_CHUNKED_PREFILL_SIZE}" ]]; then
   SGLANG_ARGS+=(--sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE}")
fi
if [[ -n "${SGLANG_MAX_PREFILL_TOKENS}" ]]; then
   SGLANG_ARGS+=(--sglang-max-prefill-tokens "${SGLANG_MAX_PREFILL_TOKENS}")
fi

if [[ "${V4_RUNTIME}" == "dspark" ]]; then
   SGLANG_ARGS+=(
      --sglang-data-parallel-size "${GPUS_PER_ENGINE}"
      --sglang-enable-dp-attention
      --sglang-moe-a2a-backend none
      --sglang-disable-flashinfer-autotune
      --sglang-kv-cache-dtype fp8_e4m3
   )
fi

if [[ "${ENABLE_LORA_SERVER}" == "1" ]]; then
   # shellcheck disable=SC2206
   SGLANG_ARGS+=(
      --sglang-enable-lora
      --sglang-max-lora-rank "${SGLANG_MAX_LORA_RANK}"
      --sglang-max-loras-per-batch "${SGLANG_MAX_LORAS_PER_BATCH}"
      --sglang-lora-target-modules ${SGLANG_LORA_TARGET_MODULES}
      --sglang-lora-backend "${SGLANG_LORA_BACKEND}"
   )
   if [[ -n "${LORA_ADAPTER_PATH}" ]]; then
      SGLANG_ARGS+=(--sglang-lora-paths "${LORA_NAME}=${LORA_ADAPTER_PATH}")
      # Custom CUDA-agent rollout attaches this exact preloaded adapter name to
      # every /generate request; no merge or training-time hot sync is involved.
      KERNEL_AGENT_ARGS+=(--rollout-lora-name "${LORA_NAME}")
   fi
fi

if [[ "${USE_MTP_SPEC}" == "1" ]]; then
   if [[ "${SPEC_ALGO}" == "DSPARK" ]]; then
      SGLANG_ARGS+=(
         --sglang-speculative-algorithm DSPARK
         --sglang-speculative-dspark-block-size "${SPEC_DSPARK_BLOCK_SIZE}"
         --sglang-enable-dp-lm-head
      )
      echo "DSPARK speculative: gamma=${SPEC_DSPARK_BLOCK_SIZE} verify_width=$((SPEC_DSPARK_BLOCK_SIZE + 1))"
   else
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
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-backend flash
)

export MASTER_ADDR
export SLIME_RAY_DASHBOARD_AGENT_EARLY_PORT
RAY_raylet_start_wait_time_s="${RAYLET_START_WAIT_TIME_SECS}" \
GLOO_SOCKET_IFNAME="${LOCAL_GLOO_SOCKET_IFNAME}" ray start \
   --head \
   --node-ip-address "${MASTER_ADDR}" \
   --port "${RAY_PORT}" \
   --dashboard-host 0.0.0.0 \
   --dashboard-port "${RAY_DASHBOARD_PORT}" \
   --dashboard-agent-listen-port "${RAY_DASHBOARD_AGENT_LISTEN_PORT}" \
   --dashboard-agent-grpc-port "${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
   --runtime-env-agent-port "${RAY_RUNTIME_ENV_AGENT_PORT}" \
   --num-gpus "${RAY_HEAD_GPUS}" \
   --num-cpus "${RAY_NUM_CPUS}" \
   --disable-usage-stats \
   --temp-dir="${RAY_TEMP_DIR}"

if [[ -n "${RAY_WORKER_HOST}" ]]; then
   ssh "${RAY_WORKER_HOST}" \
      "ulimit -n 1048576 || true; RAY_raylet_start_wait_time_s='${RAYLET_START_WAIT_TIME_SECS}' GLOO_SOCKET_IFNAME='${LOCAL_GLOO_SOCKET_IFNAME}' NCCL_SOCKET_IFNAME='${NCCL_SOCKET_IFNAME}' ray start --address='${MASTER_ADDR}:${RAY_PORT}' --node-ip-address='${RAY_WORKER_IP}' --num-gpus='${GPUS_PER_NODE}' --num-cpus='${RAY_NUM_CPUS}' --disable-usage-stats --temp-dir='${RAY_WORKER_TEMP_DIR}'"
   python3 - "${MASTER_ADDR}:${RAY_PORT}" "${ROLLOUT_NUM_GPUS}" <<'PY'
import sys
import time

import ray

address, required_gpus = sys.argv[1], float(sys.argv[2])
deadline = time.monotonic() + 180
ray.init(address=address, ignore_reinit_error=True)
while time.monotonic() < deadline:
    resources = ray.cluster_resources()
    if resources.get("GPU", 0.0) >= required_gpus:
        print(f"Ray rollout worker ready: resources={resources}")
        break
    time.sleep(2)
else:
    raise SystemExit(f"Ray rollout worker did not expose {required_gpus:g} GPUs within 180s")
ray.shutdown()
PY
fi

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

if [[ "${V4_RUNTIME}" == "dspark" ]]; then
   DSV4_RUNTIME_ENV=$(cat <<'EOF_DSV4'
    "SGLANG_DSV4_FP4_EXPERTS": "1",
    "SGLANG_SHARED_EXPERT_TP1": "1",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "true",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "true",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "true",
    "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
EOF_DSV4
)
else
   DSV4_RUNTIME_ENV=''
fi

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    ${TOPK_ENV}
    ${DSV4_RUNTIME_ENV}
    "no_proxy": "${NO_PROXY_LIST}",
    "NO_PROXY": "${NO_PROXY_LIST}",
    "NCCL_SOCKET_IFNAME": "${NCCL_SOCKET_IFNAME}",
    "GLOO_SOCKET_IFNAME": "${LOCAL_GLOO_SOCKET_IFNAME}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "EVAL_REPO_ROOT": "${REPO_ROOT}",
    "KERNEL_EVAL_WORKER_MAX_CONCURRENCY": "${KERNEL_EVAL_WORKER_MAX_CONCURRENCY}",
    "KERNEL_EVAL_RATE_LIMIT": "${KERNEL_EVAL_RATE_LIMIT}",
    "KERNEL_EVAL_PRIORITY": "${KERNEL_EVAL_PRIORITY}",
    "KERNEL_AGENT_GENERATE_GUARD_SEC": "${KERNEL_AGENT_GENERATE_GUARD_SEC}",
    "PYTHONPATH": ".:/root/Megatron-LM/:/nfs/FM/chenshuailin/projects/kernel_agents/TileKernels",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "CUDA_AGENT_LOG_MULTI_TURN_TEXT": "0",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "NCCL_DEBUG": "WARN"
  }
}
EOF_JSON
)

# Early port publication lets raylet start before slow H20 NVML discovery, but
# the jobs API must not race the agent's remaining module initialization.  The
# health route is registered by the same completed module load as JobAgent.
echo "waiting for Ray dashboard agent on ${MASTER_ADDR}:${RAY_DASHBOARD_AGENT_LISTEN_PORT}"
dashboard_agent_deadline=$((SECONDS + RAY_DASHBOARD_AGENT_READY_TIMEOUT_SECS))
while ! python3 - "http://${MASTER_ADDR}:${RAY_DASHBOARD_AGENT_LISTEN_PORT}/api/healthz" <<'PY'
import sys
import urllib.request

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    with opener.open(sys.argv[1], timeout=2) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
do
   if (( SECONDS >= dashboard_agent_deadline )); then
      echo "Ray dashboard agent did not become ready within ${RAY_DASHBOARD_AGENT_READY_TIMEOUT_SECS}s" >&2
      exit 1
   fi
   sleep 2
done
echo "Ray dashboard agent ready"

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
SUMMARY_ARGS=("${EVAL_DIR}" --max-turns "${MAX_TURNS}")
if [[ -n "${EVAL_SUMMARY_GROUP_KEY}" ]]; then
   SUMMARY_ARGS+=(--group-by-metadata "${EVAL_SUMMARY_GROUP_KEY}")
fi
python3 "${SCRIPT_DIR}/summarize_eval.py" "${SUMMARY_ARGS[@]}" | tee "${SUMMARY_PATH}"
echo "=== eval complete; dumps -> ${DUMP_DIR} ==="
echo "summary -> ${SUMMARY_PATH}"
