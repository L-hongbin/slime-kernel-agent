#!/bin/bash

set -eo pipefail

# W8A8 INT8 rollout smoke harness — 100x4 default.
#
# Differences from `scripts/debug/debug.27b.sh` (BF16 baseline):
# - `--hf-checkpoint` points at the RTN-quantized W8A8 INT8 ckpt
#   (`/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B-W8A8-RTN`). Slime reads
#   the `quantization_config` block from its `config.json` to drive the per-step
#   weight-sync quantization path (`quantize_layer_int8`). SGLang also boots
#   its engine from this same ckpt so the int8 layout matches before slime's
#   first weight push.
# - `--ref-load` still points at the original BF16 ckpt so the Megatron actor
#   stays BF16 (RTN happens online inside `slime/.../quantizer_compressed_tensors.py`).
# - Reward server (KernelGym) → 192.168.16.39:20111.
# - Default eval shape = 100 prompts × 4 samples per prompt.
#
# Override env vars to retune.

CTX_LEN=${CTX_LEN:-65536}
N_SAMPLES_PER_EVAL_PROMPT=${N_SAMPLES_PER_EVAL_PROMPT:-4}
KERNELGYM_ERROR_SUMMARY_CHARS=${KERNELGYM_ERROR_SUMMARY_CHARS:-1600}
EXPT_LABEL=${EXPT_LABEL:-w8a8-rtn}
ROLLOUT_MAX_PROMPT_LEN=$((CTX_LEN - 1))
ROLLOUT_MAX_RESPONSE_LEN=$((CTX_LEN - 1))

EVAL_CONFIG_PATH=scripts/eval_kernelbench_level1.yaml
# Original BF16 ckpt — used for Megatron ref-load + render_prompt_check tokenizer.
MODEL_DIR=${MODEL_DIR:-/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B}
# RTN W8A8 ckpt — sglang engine boot weights + slime quantization_config source.
HF_W8A8_DIR=${HF_W8A8_DIR:-/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B-W8A8-RTN}
KG_REWARD_HOST=${KG_REWARD_HOST:-192.168.16.39}
KG_REWARD_PORT=${KG_REWARD_PORT:-20111}

RUN_TS="$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="checkpoints/${MODEL_DIR##*/}/${RUN_TS}_ctx${CTX_LEN}_n${N_SAMPLES_PER_EVAL_PROMPT}_summ${KERNELGYM_ERROR_SUMMARY_CHARS}_${EXPT_LABEL}"
LOG_DIR="${SAVE_DIR}"
LOG_FILE="${LOG_DIR}/run_log"
mkdir -p "${LOG_DIR}"
touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Logging to ${LOG_FILE} (CTX_LEN=${CTX_LEN}, N=${N_SAMPLES_PER_EVAL_PROMPT})"
echo "HF_W8A8_DIR=${HF_W8A8_DIR}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "Reward server: http://${KG_REWARD_HOST}:${KG_REWARD_PORT}"

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/ray/start_cluster.sh"
source "${SCRIPT_DIR}/models/qwen3.5-27B.sh"

TP=2
SAVE_INTERVAL=${SAVE_INTERVAL:-1}

CKPT_ARGS=(
   --hf-checkpoint ${HF_W8A8_DIR}
   --ref-load ${MODEL_DIR}/torch_dist
   --save ${SAVE_DIR}/
   --load ${SAVE_DIR}/
   --save-interval ${SAVE_INTERVAL}
   --dist-ckpt-optim-fully-reshardable
   --distrib-optim-fully-reshardable-mem-efficient
)

ROLLOUT_ARGS=(
   --custom-rm-path slime_plugins.drkernel.kernelgym_rm.custom_rm
   --rollout-function-path slime_plugins.drkernel.rollout.generate_rollout
   --prompt-data data/drkernel-rl-data-0513/train.parquet
   --input-key ground_truth
   --label-key ground_truth
   --metadata-key extra_info
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 0
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --n-samples-per-eval-prompt ${N_SAMPLES_PER_EVAL_PROMPT}
   --rollout-max-prompt-len ${ROLLOUT_MAX_PROMPT_LEN}
   --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN}
   --rollout-max-context-len ${CTX_LEN}
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data
   --debug-rollout-only
)

EVAL_ARGS=(
   --eval-interval 20
   --skip-eval-before-train
   --eval-config "${EVAL_CONFIG_PATH}"
   --eval-max-prompt-len ${CTX_LEN}
   --eval-max-response-len ${CTX_LEN}
   --eval-max-context-len ${CTX_LEN}
   --rm-url http://${KG_REWARD_HOST}:${KG_REWARD_PORT}
   --dump-details ${SAVE_DIR}/dumps
)

# KernelGym compile/eval host. .39 / .40 are RTX 4090 boxes — same arch.
if [ "${DRKERNEL_NO_TARGET_ENV:-0}" = "1" ]; then
    DRKERNEL_GPU_NAME=""
    DRKERNEL_COMPILER_NAME=""
else
    DRKERNEL_GPU_NAME=${DRKERNEL_GPU_NAME:-"NVIDIA GeForce RTX 4090 (SM 8.9, Ada Lovelace)"}
    DRKERNEL_COMPILER_NAME=${DRKERNEL_COMPILER_NAME:-"CUDA 12.9 (nvcc, targeting sm_89)"}
fi

DRKERNEL_PLUGIN_ARGS=(
   --use-multi-turn
   --max-turns 3
   --kernelgym-error-summary-chars ${KERNELGYM_ERROR_SUMMARY_CHARS}
)
if [ -n "${DRKERNEL_GPU_NAME}" ]; then
   DRKERNEL_PLUGIN_ARGS+=(--drkernel-gpu-name "${DRKERNEL_GPU_NAME}")
fi
if [ -n "${DRKERNEL_COMPILER_NAME}" ]; then
   DRKERNEL_PLUGIN_ARGS+=(--drkernel-compiler-name "${DRKERNEL_COMPILER_NAME}")
fi

PERF_ARGS=(
   --tensor-model-parallel-size ${TP}
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${TP}
   --sglang-context-length ${CTX_LEN}
   --sglang-max-running-requests 64
   --sglang-mem-fraction-static 0.9
   --sglang-decode-log-interval 400
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

DRKERNEL_SMOKE_MAX_PROMPTS=${DRKERNEL_SMOKE_MAX_PROMPTS:-100}
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SLIME_TENSOR_BACKUP_PIN_MEMORY\": \"0\",
    \"DRKERNEL_SMOKE_MAX_PROMPTS\": \"${DRKERNEL_SMOKE_MAX_PROMPTS}\"
  }
}"

# Pre-launch render check uses the BF16 ckpt — only tokenizer + chat_template
# are read; avoids any risk of compressed-tensors loader quirks tripping the
# check before the run starts.
_RENDER_CHECK_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --drkernel-gpu-name "${DRKERNEL_GPU_NAME}"
   --drkernel-compiler-name "${DRKERNEL_COMPILER_NAME}"
)
if [ -n "${DRKERNEL_GPU_NAME}" ]; then
   _RENDER_CHECK_ARGS+=(--expected-gpu-words "${DRKERNEL_GPU_NAME}")
fi
PYTHONPATH="${SCRIPT_DIR}/.." python3 "${SCRIPT_DIR}/debug/render_prompt_check.py" "${_RENDER_CHECK_ARGS[@]}"

submit_ray_job --address="${RAY_JOB_ADDRESS}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   "${DRKERNEL_PLUGIN_ARGS[@]}"
