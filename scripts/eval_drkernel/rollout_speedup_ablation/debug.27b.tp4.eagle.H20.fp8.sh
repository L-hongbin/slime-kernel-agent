#!/bin/bash

set -eo pipefail

CTX_LEN=${CTX_LEN:-65536}

N_SAMPLES_PER_EVAL_PROMPT=${N_SAMPLES_PER_EVAL_PROMPT:-8}
KERNELGYM_ERROR_SUMMARY_CHARS=${KERNELGYM_ERROR_SUMMARY_CHARS:-1600}
EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-${CTX_LEN}}
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-96}

PYTORCH_CUDA_ALLOC_CONF_VALUE=${PYTORCH_CUDA_ALLOC_CONF_VALUE-expandable_segments:True}
EXPT_LABEL=newSlimeKG.tp4.eagle.rm16.C${SGLANG_MAX_RUNNING_REQUESTS}.H20.linear-fi.fp8
ROLLOUT_MAX_PROMPT_LEN=$((CTX_LEN - 1))
ROLLOUT_MAX_RESPONSE_LEN=$((CTX_LEN - 1))

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
# This worktree is intentionally sparse; reuse shared debug/ray/data assets from
# the main slime checkout unless the caller points at a different copy.
SCRIPT_HELPER_DIR=${SCRIPT_HELPER_DIR:-/nfs/FM/chenshuailin/projects/kernel_agents/slime/scripts}
DATA_ROOT=/nfs/FM/chenshuailin/projects/kernel_agents/slime
EVAL_CONFIG_PATH=${SCRIPT_HELPER_DIR}/eval_kernelbench_level1.yaml
PROMPT_DATA_PATH=${DATA_ROOT}/data/drkernel-rl-data-0513/train.parquet
# MODEL_DIR stays BF16: used only for SAVE_DIR naming, the pre-launch render
# check (tokenizer/chat_template), and --ref-load (ignored under
# --debug-rollout-only, see slime/backends/megatron_utils/actor.py:53,596).
MODEL_DIR=/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B
REF_LOAD_DIR=${MODEL_DIR}
# FP8 rollout: SGLang serves --hf-checkpoint directly (debug-rollout-only does no
# Megatron->SGLang weight sync). FP8 is auto-detected from config.json
# quantization_config (quant_method=fp8); MTP/EAGLE draft head is bundled.
HF_W8A8_DIR=/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B-FP8

RUN_TS="$(date +%Y%m%d_%H%M%S)"
SAVE_DIR=checkpoints/${HF_W8A8_DIR##*/}/${RUN_TS}_${EXPT_LABEL}_ctx${CTX_LEN}_n${N_SAMPLES_PER_EVAL_PROMPT}_summ${KERNELGYM_ERROR_SUMMARY_CHARS}
LOG_FILE="${SAVE_DIR}/run.log"
mkdir -p "${SAVE_DIR}"
RESOLVED_EVAL_CONFIG_PATH="${SAVE_DIR}/eval_config.resolved.yaml"
sed "s|path: data/|path: ${DATA_ROOT}/data/|g" "${EVAL_CONFIG_PATH}" >"${RESOLVED_EVAL_CONFIG_PATH}"
touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

export PYTHONUNBUFFERED=1

source "${SCRIPT_HELPER_DIR}/ray/start_cluster.sh"
source "${SCRIPT_HELPER_DIR}/models/qwen3.5-27B.sh"

TP=4
SAVE_INTERVAL=${SAVE_INTERVAL:-1}

CKPT_ARGS=(
   --hf-checkpoint ${HF_W8A8_DIR}
   --ref-load ${REF_LOAD_DIR}/torch_dist
   --save ${SAVE_DIR}/
   --load ${SAVE_DIR}/
   --save-interval ${SAVE_INTERVAL}
   --dist-ckpt-optim-fully-reshardable
   --distrib-optim-fully-reshardable-mem-efficient
)

ROLLOUT_ARGS=(
   --custom-rm-path slime_plugins.drkernel.kernelgym_rm.custom_rm
   --rollout-function-path slime_plugins.drkernel.rollout.generate_rollout
   --prompt-data ${PROMPT_DATA_PATH}
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
   --eval-config "${RESOLVED_EVAL_CONFIG_PATH}"
   --eval-max-prompt-len ${CTX_LEN}
   --eval-max-response-len ${EVAL_MAX_RESPONSE_LEN}
   --eval-max-context-len ${CTX_LEN}
   --rm-url http://127.0.0.1:20391
   --dump-details ${SAVE_DIR}/dumps
)

DRKERNEL_GPU_NAME=${DRKERNEL_GPU_NAME:-"NVIDIA GeForce RTX 4090 (SM 8.9, Ada Lovelace)"}
DRKERNEL_COMPILER_NAME=${DRKERNEL_COMPILER_NAME:-"CUDA 12.9 (nvcc, targeting sm_89)"}

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
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-27B-test
   # --wandb-key ${WANDB_KEY}
)

# for --sglang-mem-fraction-static, 0.9 will OOM
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${TP}
   --sglang-context-length ${CTX_LEN}
   --sglang-max-running-requests ${SGLANG_MAX_RUNNING_REQUESTS}
   --sglang-mem-fraction-static 0.85
   --sglang-decode-log-interval 400
   --sglang-mamba-scheduler-strategy extra_buffer
   --router-policy consistent_hashing
   --sglang-cuda-graph-max-bs ${SGLANG_MAX_RUNNING_REQUESTS}
   --sglang-disable-custom-all-reduce
   --sglang-linear-attn-backend flashinfer
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 4
   # --sglang-enable-hierarchical-cache
   # --sglang-page-size 64
   # --sglang-hicache-ratio 1.2
   # --sglang-hicache-io-backend kernel
   # --sglang-hicache-mem-layout page_first
   # --sglang-enable-cache-report
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   # --apply-chat-template-kwargs '{"preserve_thinking":true}'
)

RAY_LOCAL_IP=${RAY_LOCAL_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}
LOCAL_NO_PROXY="localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR},${NODE_ADDR}"
if [ -n "${RAY_LOCAL_IP}" ]; then
   LOCAL_NO_PROXY="${LOCAL_NO_PROXY},${RAY_LOCAL_IP}"
fi
LOCAL_NO_PROXY="${LOCAL_NO_PROXY}${no_proxy:+,${no_proxy}}${NO_PROXY:+,${NO_PROXY}}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${REPO_ROOT}:/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${LOCAL_NO_PROXY}\",
    \"NO_PROXY\": \"${LOCAL_NO_PROXY}\",
    \"SLIME_TENSOR_BACKUP_PIN_MEMORY\": \"0\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF_VALUE}\"
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
PYTHONPATH="${REPO_ROOT}:${SCRIPT_HELPER_DIR}/..:${PYTHONPATH:-}" \
   python3 "${SCRIPT_HELPER_DIR}/eval_drkernel/render_prompt_check.py" "${_RENDER_CHECK_ARGS[@]}"

submit_ray_job --address="${RAY_JOB_ADDRESS}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-gpus-per-node 8 \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${DRKERNEL_PLUGIN_ARGS[@]}"
