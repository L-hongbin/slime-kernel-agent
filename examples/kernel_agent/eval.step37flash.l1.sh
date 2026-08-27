#!/bin/bash
#
# Eval-only of stepfun-ai/Step-3.7-Flash on KernelBench LEVEL 1, run on THIS node
# (node64 / 10.11.2.164). Adapted from eval.t3.qwen3.6.27B.sh:
#   - eval TASK config is kept identical to t3 (tvm_ffi backend, multi_turn
#     tvm_ffi_short prompt, do-precheck, reference-cache, finalize-mode none,
#     max-turns 3, n=8, level1 validation parquet, summarize_eval.py metrics).
#   - MODEL + SGLang SERVING are adapted to the step3p7 MoE VLM (see notes).
#
# Why the deviations from "其他不变":
#   * Step-3.7-Flash is a 198B sparse-MoE vision-language model (Step3p7), NOT a
#     Qwen3 dense/hybrid model, so the Qwen megatron MODEL_ARGS and the
#     Qwen-specific sglang flags (EAGLE eagle3 head, linear-attn, mamba) do not
#     apply. In --debug-rollout-only megatron is never built and hf-validate is
#     skipped, so MODEL_ARGS is simply empty.
#   * 198B weights need all 8 GPUs in one engine -> --rollout-num-gpus-per-engine 8.
#   * LANGUAGE ONLY: --enable-multimodal is NOT passed and KernelBench prompts are
#     pure text, so no image tokens are ever fed (vision tower is unused).
#   * Step3 reasoning is controlled by the chat-template kwarg `reasoning_effort`
#     (low|medium|high), NOT Qwen's `enable_thinking`; thinking is always opened
#     by add_generation_prompt. We pass reasoning_effort (default: high).
#   * Speculative decoding (step3 MTP) is DISABLED by default for robustness; see
#     the commented block in SGLANG_ARGS to enable it later for speed.
#
# Usage (the user-requested smoke -> 8 -> full staging is driven by EVAL_NUM_PROMPTS):
#   EVAL_NUM_PROMPTS=1 N_SAMPLES_PER_EVAL_PROMPT=1 bash examples/kernel_agent/eval.step37flash.l1.sh   # smoke
#   EVAL_NUM_PROMPTS=8                              bash examples/kernel_agent/eval.step37flash.l1.sh   # 8 prompts
#                                                   bash examples/kernel_agent/eval.step37flash.l1.sh   # full (all 100)

set -Eeo pipefail
trap 'status=$?; echo "Script exiting with status ${status} at line ${LINENO}: ${BASH_COMMAND}"' EXIT
trap 'status=$?; echo "ERROR status ${status} at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

export PYTHONUNBUFFERED=1
# Bypass the cluster clash proxy for loopback/in-cluster traffic; otherwise the
# preflight health check to 127.0.0.1 gets black-holed by the proxy and times out.
export no_proxy="127.0.0.1,localhost,0.0.0.0,::1,${MASTER_ADDR:-10.11.2.164}"
export NO_PROXY="${no_proxy}"
ulimit -n 1048576 || true

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Step-3.7-Flash is a custom step3p7 arch; megatron is not built in
# --debug-rollout-only, so no model-arch script is sourced (MODEL_ARGS empty).
MODEL_ARGS=()

DEFAULT_MODEL_PATH="/nfs/FM/chenshuailin/checkpoints/stepfun-ai/Step-3.7-Flash"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
HF_MODEL_PATH="${HF_MODEL_PATH:-${MODEL_PATH}}"
MODEL_TAG="$(basename "${HF_MODEL_PATH%/}")"
if [[ ! -f "${HF_MODEL_PATH}/config.json" ]]; then
   echo "HF_MODEL_PATH is not an HF checkpoint (no config.json): ${HF_MODEL_PATH}" >&2
   echo "Set MODEL_PATH=/path/to/hf_checkpoint or HF_MODEL_PATH=/path/to/hf_checkpoint." >&2
   exit 1
fi

EVAL_DATA="${EVAL_DATA:-${REPO_ROOT}/Data/kernelbench-level1-validation-tvm-v2/train.parquet}"
if [[ ! -f "${EVAL_DATA}" ]]; then
   echo "EVAL_DATA does not exist: ${EVAL_DATA}" >&2
   exit 1
fi

KERNELGYM_PORT="${KERNELGYM_PORT:-20211}"
KERNEL_ENV_URL="${KERNEL_ENV_URL:-http://127.0.0.1:${KERNELGYM_PORT}}"
KERNEL_BACKEND="${KERNEL_BACKEND:-tvm_ffi}"
REFERENCE_BACKEND="${REFERENCE_BACKEND:-torch}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-8}"
# step3 reasoning effort: low | medium | high (chat-template kwarg)
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
case "${REASONING_EFFORT}" in
   low|medium|high) ;;
   *) echo "REASONING_EFFORT must be one of low|medium|high, got: ${REASONING_EFFORT}" >&2; exit 1 ;;
esac
# optional smoke control: keep only the first N eval prompts (unset/0 => all)
EVAL_NUM_PROMPTS="${EVAL_NUM_PROMPTS:-0}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-32768}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-32768}"
MAX_TURNS="${MAX_TURNS:-3}"
# 64 is the KV sweet spot for 32K-context: 96 over-subscribes the KV pool
# (1.76M tokens) and triggers retraction once 96 long sequences fill it.
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-64}"
SGLANG_WATCHDOG_TIMEOUT="${SGLANG_WATCHDOG_TIMEOUT:-2400}"
ROUTER_QUEUE_TIMEOUT_SECS="${ROUTER_QUEUE_TIMEOUT_SECS:-2400}"
# step3 ships MTP (multi-token-prediction) weights -> EAGLE-style speculative
# decoding is lossless (distribution-preserving) and only speeds up decode.
ENABLE_MTP="${ENABLE_MTP:-1}"
MTP_NUM_STEPS="${MTP_NUM_STEPS:-3}"
MTP_EAGLE_TOPK="${MTP_EAGLE_TOPK:-1}"
MTP_NUM_DRAFT_TOKENS="${MTP_NUM_DRAFT_TOKENS:-4}"

MASTER_ADDR="${MASTER_ADDR:-10.11.2.164}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"
# Separate ray ports/dir from any training instance on this node.
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8266}"
RAY_PORT="${RAY_PORT:-6380}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_eval}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
# Gloo needs the interface that actually holds MASTER_ADDR's IP (differs per node).
LOCAL_GLOO_SOCKET_IFNAME="${LOCAL_GLOO_SOCKET_IFNAME:-$(ip -o -4 addr show 2>/dev/null | awk -v ip="${MASTER_ADDR}" '$4 ~ "^"ip"/" {print $2; exit}')}"
LOCAL_GLOO_SOCKET_IFNAME="${LOCAL_GLOO_SOCKET_IFNAME:-bond0}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
   HAS_NVLINK="${HAS_NVLINK:-1}"
else
   HAS_NVLINK="${HAS_NVLINK:-0}"
fi

MTP_TAG="$([[ "${ENABLE_MTP}" == "1" ]] && echo mtpON || echo mtpOFF)"
EXP_NAME="Eval.TVMFFI.Step3.7-Flash.LangOnly.${MODEL_TAG}.l1.ctx${MAX_CONTEXT_LEN}.resp${MAX_RESPONSE_LEN}.turn${MAX_TURNS}.n${N_SAMPLES_PER_EVAL_PROMPT}.re${REASONING_EFFORT}.${MTP_TAG}"
EXP_ROOT="${EXP_ROOT:-${REPO_ROOT}/experiments/${EXP_NAME}}"
EVAL_TAG="${EVAL_TAG:-${MODEL_TAG}}"
if [[ "${EVAL_NUM_PROMPTS}" -gt 0 ]]; then
   EVAL_TAG="${EVAL_TAG}.first${EVAL_NUM_PROMPTS}"
fi
EVAL_DIR="${EVAL_DIR:-${EXP_ROOT}/${EVAL_TAG}}"
DUMP_DIR="${EVAL_DIR}/dumps"
mkdir -p "${EVAL_DIR}"

# Optional smoke subset: slice the first EVAL_NUM_PROMPTS rows into a local parquet.
if [[ "${EVAL_NUM_PROMPTS}" -gt 0 ]]; then
   SUBSET_PARQUET="${EVAL_DIR}/eval_subset.first${EVAL_NUM_PROMPTS}.parquet"
   python3 - "${EVAL_DATA}" "${SUBSET_PARQUET}" "${EVAL_NUM_PROMPTS}" <<'PY'
import sys, pandas as pd
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
df = pd.read_parquet(src)
sub = df.head(n)
sub.to_parquet(dst, index=False)
print(f"[subset] wrote {len(sub)}/{len(df)} prompts -> {dst}")
PY
   EVAL_DATA="${SUBSET_PARQUET}"
fi

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
echo "REASONING_EFFORT=${REASONING_EFFORT}"
echo "EVAL_NUM_PROMPTS=${EVAL_NUM_PROMPTS} (0=all)"
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
   # step3p7 thinking is controlled by reasoning_effort (low|medium|high).
   --apply-chat-template-kwargs "{\"reasoning_effort\":\"${REASONING_EFFORT}\"}"
   # Step-3.x LlamaTokenizerFast drops whitespace under transformers v5 (Mistral-lineage
   # regex). fix_mistral_regex=True restores the correct ByteLevel tokenization (matches
   # sglang + the model's own token-space). Scoped to this run, NOT global.
   --tokenizer-load-kwargs "{\"fix_mistral_regex\": true}"
   --rollout-temperature 1
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

# step3 MTP speculative decoding (EAGLE multi-layer). Lossless; decode speedup only.
SPEC_ARGS=()
if [[ "${ENABLE_MTP}" == "1" ]]; then
   SPEC_ARGS=(
      --sglang-speculative-algorithm EAGLE
      --sglang-speculative-num-steps "${MTP_NUM_STEPS}"
      --sglang-speculative-eagle-topk "${MTP_EAGLE_TOPK}"
      --sglang-speculative-num-draft-tokens "${MTP_NUM_DRAFT_TOKENS}"
      --sglang-enable-multi-layer-eagle
   )
fi

SGLANG_ARGS=(
   # 198B MoE: one engine across all 8 GPUs (tp=8).
   --rollout-num-gpus-per-engine "${GPUS_PER_NODE}"
   --sglang-context-length "${MAX_CONTEXT_LEN}"
   --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
   --sglang-mem-fraction-static 0.85
   --sglang-decode-log-interval 400
   --router-policy round_robin
   --router-queue-timeout-secs "${ROUTER_QUEUE_TIMEOUT_SECS}"
   --sglang-cuda-graph-max-bs "${SGLANG_MAX_RUNNING_REQUESTS}"
   --sglang-disable-custom-all-reduce
   --sglang-watchdog-timeout "${SGLANG_WATCHDOG_TIMEOUT}"
   "${SPEC_ARGS[@]}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-backend flash
)

export MASTER_ADDR
# Raise the raylet's wait-for-dashboard-agent timeout (default 30s). On this node
# the dashboard agent's nvidia-smi probe can be slow when KernelGym is pegging the
# GPUs at 100% util, so the raylet otherwise crashes in WaitForDashboardAgentPorts.
export RAY_agent_register_timeout_ms="${RAY_AGENT_REGISTER_TIMEOUT_MS:-180000}"
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
