#!/bin/bash

set -eo pipefail

# Parametrized for sweep runs. Override e.g. CTX_LEN=65536 N_SAMPLES_PER_EVAL_PROMPT=8 bash debug.27b.sh.
CTX_LEN=${CTX_LEN:-65536}
N_SAMPLES_PER_EVAL_PROMPT=${N_SAMPLES_PER_EVAL_PROMPT:-8}
KERNELGYM_ERROR_SUMMARY_CHARS=${KERNELGYM_ERROR_SUMMARY_CHARS:-1600}
EXPT_LABEL=${EXPT_LABEL:-}
ROLLOUT_MAX_PROMPT_LEN=$((CTX_LEN - 1))
ROLLOUT_MAX_RESPONSE_LEN=$((CTX_LEN - 1))

EVAL_CONFIG_PATH=scripts/eval_kernelbench_level1.yaml
MODEL_DIR=${MODEL_DIR:-/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B}
RUN_TS="$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="checkpoints/${MODEL_DIR##*/}/${RUN_TS}_ctx${CTX_LEN}_n${N_SAMPLES_PER_EVAL_PROMPT}_summ${KERNELGYM_ERROR_SUMMARY_CHARS}${EXPT_LABEL:+_${EXPT_LABEL}}"
LOG_DIR="${SAVE_DIR}"
LOG_FILE="${LOG_DIR}/run_log"
mkdir -p "${LOG_DIR}"
touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Logging to ${LOG_FILE} (CTX_LEN=${CTX_LEN}, N_SAMPLES_PER_EVAL_PROMPT=${N_SAMPLES_PER_EVAL_PROMPT})"

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
source "${SCRIPT_DIR}/ray/start_cluster.sh"
source "${SCRIPT_DIR}/models/qwen3.5-27B.sh"

TP=2
SAVE_INTERVAL=${SAVE_INTERVAL:-1}

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}
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
   --rm-url http://${KG_REWARD_HOST:-192.168.16.40}:${KG_REWARD_PORT:-20111}
   --dump-details ${SAVE_DIR}/dumps
)

# Runtime environment hints for the prompt's "Target environment:" block.
# These describe the KernelGym compile/eval host (192.168.16.40), NOT the local
# training/inference box: KG compiles & benchmarks every kernel on .40 with
# /usr/local/cuda-12.9/bin/nvcc targeting -gencode=arch=compute_89,code=sm_89
# on 8x NVIDIA GeForce RTX 4090. Telling the model the wrong arch (e.g. the
# A800 on .22) would mislead its tensor-core / SM-specific optimizations.
# Set DRKERNEL_NO_TARGET_ENV=1 to suppress the entire block (ablation mode).
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

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-kl-loss
   # --kl-loss-coef 0.00
   # --kl-loss-type low_var_kl
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
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-27B-test
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${TP}
   --sglang-context-length ${CTX_LEN}
   --sglang-max-running-requests 64
   --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC:-0.9}
   --sglang-decode-log-interval 400
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# Build the runtime environment JSON with proper variable substitution.
# DRKERNEL_SMOKE_MAX_PROMPTS caps eval to first N prompts in eval_rollout_single_dataset.
# Unset (or set to 0) for full validation-set eval.
DRKERNEL_SMOKE_MAX_PROMPTS=${DRKERNEL_SMOKE_MAX_PROMPTS:-100}
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True avoids the fragmentation OOM
# we hit on sglang 0.5.12 / torch 2.11+cu129 — the alloc reserved ~10 GiB
# "but unallocated" and refused to satisfy a contiguous 7 GiB request.
# Mirrors the W8A8 launcher.
PYTORCH_CUDA_ALLOC_CONF_VALUE=${PYTORCH_CUDA_ALLOC_CONF_VALUE-expandable_segments:True}
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SLIME_TENSOR_BACKUP_PIN_MEMORY\": \"0\",
    \"DRKERNEL_SMOKE_MAX_PROMPTS\": \"${DRKERNEL_SMOKE_MAX_PROMPTS}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF_VALUE}\"
  }
}"

# Pre-launch sanity check: render the first-turn prompt with the current
# profile + the values we are about to pass via DRKERNEL_PLUGIN_ARGS. Catches
# (a) leaked jinja markers like `{# ... #}` or `{{ ... }}` and (b) bash
# array-quoting bugs that truncate multi-word values such as the GPU name
# from "NVIDIA A800-SXM4-80GB" to just "NVIDIA". Exits non-zero on failure so
# the run aborts before a 50-min eval that would produce confounded data.
_RENDER_CHECK_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --drkernel-gpu-name "${DRKERNEL_GPU_NAME}"
   --drkernel-compiler-name "${DRKERNEL_COMPILER_NAME}"
)
# Only enforce GPU-word presence when we're injecting the env block; otherwise
# skip the word check so render_prompt_check doesn't false-fail on no-env runs.
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

   # --debugpy \
