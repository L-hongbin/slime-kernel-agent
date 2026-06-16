#!/bin/bash
#
# Single-node eval-only of a trained kernel-agent checkpoint, FAITHFUL to
# training: same generate fn (generate_with_cuda_agent.generate), tvm_ffi
# backend, multi_turn_cuda_kernel prompt. examples/kernel_agent/summarize_eval.py
# turns the dumped env_result.env_state into Compile / Correct / Fast@1.0 /
# Fast@1.2 (in_all). NOTE: the sibling summarize_kernelgym_eval.py does NOT read
# this dump schema (it expects metadata.kernelgym); use summarize_eval.py.
#
# Eval-only mechanism: train.py runs `rollout_manager.eval()` once when
# --num-rollout 0 and --eval-interval is set, then exits. --debug-rollout-only
# skips the megatron backend, so all 8 GPUs serve SGLang (2 engines, TP4).
#
# Usage:
#   EVAL_HF_CKPT=.../hf/iter_39 KERNELGYM_PORT=20211 \
#     bash examples/kernel_agent/eval.t1.qwen3.6.27B.sh
set -Eex
trap 'status=$?; echo "Script exiting with status ${status} at line ${LINENO}: ${BASH_COMMAND}"' EXIT

export PYTHONUNBUFFERED=1
ulimit -n 1048576 || true
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3.5-27B.sh"

# ---- eval config ----
EVAL_HF_CKPT=${EVAL_HF_CKPT:?set EVAL_HF_CKPT to the converted HF checkpoint dir}
if [ ! -f "${EVAL_HF_CKPT}/config.json" ]; then
   echo "error: EVAL_HF_CKPT is not an HF checkpoint (no config.json): ${EVAL_HF_CKPT}" >&2
   exit 1
fi
EVAL_DATA=${EVAL_DATA:-${REPO_ROOT}/Data/kernelbench-level1-validation-tvm-v2/train.parquet}
KERNELGYM_PORT=${KERNELGYM_PORT:-20211}
KERNEL_ENV_URL="http://127.0.0.1:${KERNELGYM_PORT}"
KERNEL_BACKEND="tvm_ffi"
N_SAMPLES_PER_EVAL_PROMPT=${N_SAMPLES_PER_EVAL_PROMPT:-8}
MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-16384}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-${MAX_CONTEXT_LEN}}
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-64}
EVAL_TAG=${EVAL_TAG:-$(basename "${EVAL_HF_CKPT}")}

MASTER_ADDR="${MASTER_ADDR:-10.11.2.164}"
HF_MODEL_PATH=${HF_MODEL_PATH:-${EVAL_HF_CKPT}}

EXP_NAME="EvalFAsync.${KERNEL_BACKEND}.Qwen3.6-27B.CTX${MAX_CONTEXT_LEN}"
EXP_ROOT="${REPO_ROOT}/experiments/${EXP_NAME}"
EVAL_DIR="${EXP_ROOT}/${EVAL_TAG}.port${KERNELGYM_PORT}"
DUMP_DIR="${EVAL_DIR}/dumps"
mkdir -p "${EVAL_DIR}"

RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8266}
RAY_PORT=${RAY_PORT:-6380}
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/ray_eval}
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
# Gloo needs the interface that actually holds MASTER_ADDR's IP — it differs per
# node (head/.170=bond0, .169=enp76s0f0np0). raylet + SGLang actors inherit
# GLOO_SOCKET_IFNAME from `ray start`; a wrong iface => "Unable to find address".
LOCAL_GLOO_SOCKET_IFNAME="${LOCAL_GLOO_SOCKET_IFNAME:-$(ip -o -4 addr show 2>/dev/null | awk -v ip="${MASTER_ADDR}" '$4 ~ "^"ip"/" {print $2; exit}')}"
LOCAL_GLOO_SOCKET_IFNAME="${LOCAL_GLOO_SOCKET_IFNAME:-bond0}"
HAS_NVLINK="${HAS_NVLINK:-1}"

LOG_STAMP="$(date +%Y%m%d.%H%M%S)"
LOG_PATH="${EVAL_DIR}/${LOG_STAMP}.log"
echo "Logging to ${LOG_PATH}"
exec >> "${LOG_PATH}" 2>&1

echo "EVAL_HF_CKPT=${EVAL_HF_CKPT}"
echo "EVAL_DATA=${EVAL_DATA}"
echo "KERNEL_ENV_URL=${KERNEL_ENV_URL}"
echo "DUMP_DIR=${DUMP_DIR}"

# preflight: KernelGym health on this node/port
if ! curl -s --max-time 5 "${KERNEL_ENV_URL}/health" -o /dev/null -w '%{http_code}' | grep -q 200; then
   echo "KernelGym health check failed at ${KERNEL_ENV_URL}" >&2
   exit 1
fi

# clean any prior ray on this eval cluster
ray stop --force || true
sleep 2

EVAL_ARGS=(
   --num-rollout 0
   --eval-interval 1
   --eval-prompt-data kb_l1_val "${EVAL_DATA}"
   --eval-input-key prompt
   --eval-label-key reward_model
   --n-samples-per-eval-prompt ${N_SAMPLES_PER_EVAL_PROMPT}
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
   --rollout-max-response-len ${MAX_RESPONSE_LEN}
   --rollout-max-context-len ${MAX_CONTEXT_LEN}
   --apply-chat-template-kwargs '{"enable_thinking":true}'
   --rollout-temperature 1
)

CUSTOM_ARGS=(
   --custom-generate-function-path examples.kernel_agent.generate_with_cuda_agent.generate
   --custom-rm-path examples.kernel_agent.generate_with_cuda_agent.reward_func
   --multi-turn-prompt-config-path "${SCRIPT_DIR}/prompt_config/multi_turn_cuda_kernel.yaml"
)

KERNEL_AGENT_ARGS=(
   --kernel-env-url ${KERNEL_ENV_URL}
   --kernel-backend ${KERNEL_BACKEND}
   --reference-backend torch
   --do-precheck
   --use-reference-cache
   --use-multi-turn
   --max-turns 1
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-context-length ${MAX_CONTEXT_LEN}
   --sglang-max-running-requests ${SGLANG_MAX_RUNNING_REQUESTS}
   --sglang-mem-fraction-static 0.85
   --sglang-decode-log-interval 400
   --router-policy round_robin
   --sglang-cuda-graph-max-bs ${SGLANG_MAX_RUNNING_REQUESTS}
   --sglang-disable-custom-all-reduce
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 4
   --sglang-linear-attn-backend flashinfer
   --sglang-mamba-scheduler-strategy extra_buffer
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-backend flash
)

export MASTER_ADDR
GLOO_SOCKET_IFNAME="${LOCAL_GLOO_SOCKET_IFNAME}" ray start \
   --head \
   --node-ip-address ${MASTER_ADDR} \
   --port ${RAY_PORT} \
   --dashboard-host 0.0.0.0 \
   --dashboard-port ${RAY_DASHBOARD_PORT} \
   --num-gpus 8 \
   --disable-usage-stats \
   --temp-dir="${RAY_TEMP_DIR}"

NO_PROXY_LIST="localhost,127.0.0.1,0.0.0.0,::1,${MASTER_ADDR}"
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
   --actor-num-gpus-per-node 8 \
   --colocate \
   --hf-checkpoint ${HF_MODEL_PATH} \
   "${MODEL_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}" \
   "${KERNEL_AGENT_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"

SUMMARY_PATH="${EVAL_DIR}/summary.${LOG_STAMP}.txt"
python3 "${REPO_ROOT}/examples/kernel_agent/summarize_eval.py" "${EVAL_DIR}" --max-turns 1 | tee "${SUMMARY_PATH}"
echo "=== eval complete; dumps -> ${DUMP_DIR} ==="
echo "summary -> ${SUMMARY_PATH}"
