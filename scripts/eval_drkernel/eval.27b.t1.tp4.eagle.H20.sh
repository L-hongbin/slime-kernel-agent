#!/bin/bash

# Single-turn KernelBench L1 eval of a trained HF checkpoint on one H20 node.
# Derived from rollout_speedup_ablation/debug.27b.tp4.eagle.H20.sh, with:
#   - single-turn rollout (no --use-multi-turn / --max-turns), matching the
#     single-turn training runs (e.g. 20260609_134303 bf16+gradf32+TIS)
#   - the evaluated HF checkpoint parameterized via EVAL_HF_CKPT
#
# Usage:
#   EVAL_HF_CKPT=checkpoints/Qwen3.6-27B/<run>/hf/iter_9 \
#     bash scripts/eval_drkernel/eval.27b.t1.tp4.eagle.H20.sh

set -eo pipefail

EVAL_HF_CKPT=${EVAL_HF_CKPT:?set EVAL_HF_CKPT to the HF checkpoint dir to evaluate}
if [ ! -f "${EVAL_HF_CKPT}/config.json" ]; then
   echo "error: EVAL_HF_CKPT does not look like an HF checkpoint: ${EVAL_HF_CKPT}" >&2
   exit 1
fi
EVAL_TAG=${EVAL_TAG:-$(basename "${EVAL_HF_CKPT}")}

CTX_LEN=${CTX_LEN:-65536}

N_SAMPLES_PER_EVAL_PROMPT=${N_SAMPLES_PER_EVAL_PROMPT:-8}
KERNELGYM_ERROR_SUMMARY_CHARS=${KERNELGYM_ERROR_SUMMARY_CHARS:-1600}
EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-${CTX_LEN}}
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-96}
# Multi-node parallel evals share one KernelGym backend: raise the per-request
# client timeout (default 600s in kernelgym_rm.py) so backend queue wait does
# not turn into timeout->retry storms (each retry re-POSTs a duplicate task),
# and cap per-node eval fan-out (default would be max_running x engines x 2 = 384).
KERNELGYM_CLIENT_TIMEOUT_S=${KERNELGYM_CLIENT_TIMEOUT_S:-3600}
DRKERNEL_EVAL_MAX_CONCURRENCY=${DRKERNEL_EVAL_MAX_CONCURRENCY:-96}

PYTORCH_CUDA_ALLOC_CONF_VALUE=${PYTORCH_CUDA_ALLOC_CONF_VALUE-expandable_segments:True}
EXPT_LABEL=trainEval.t1.${EVAL_TAG}.tp4.eagle.C${SGLANG_MAX_RUNNING_REQUESTS}.H20
ROLLOUT_MAX_PROMPT_LEN=$((CTX_LEN - 1))
ROLLOUT_MAX_RESPONSE_LEN=$((CTX_LEN - 1))

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
SCRIPT_HELPER_DIR=${SCRIPT_HELPER_DIR:-${REPO_ROOT}/scripts}
DATA_ROOT=${DATA_ROOT:-${REPO_ROOT}}
EVAL_CONFIG_PATH=${SCRIPT_HELPER_DIR}/eval_kernelbench_level1.yaml
PROMPT_DATA_PATH=${DATA_ROOT}/data/drkernel-rl-data-0513/train.parquet
# Base model dir only feeds --ref-load (unused under --debug-rollout-only).
MODEL_DIR=/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B
REF_LOAD_DIR=${MODEL_DIR}

# Eval artifacts live under the training run dir that owns the evaluated
# checkpoint, named like the Megatron iteration dirs:
#   …/<train_run>/hf/iter_9 -> …/<train_run>/eval/iter_0000009
# Override the root with EVAL_OUT_ROOT if needed.
TRAIN_RUN_DIR="$(cd -- "$(dirname -- "${EVAL_HF_CKPT}")/.." &>/dev/null && pwd)"
EVAL_OUT_ROOT=${EVAL_OUT_ROOT:-${TRAIN_RUN_DIR}/eval}
EVAL_ITER_NUM=$(basename "${EVAL_HF_CKPT}" | grep -oE '[0-9]+$' || true)
if [ -n "${EVAL_ITER_NUM}" ]; then
   EVAL_SUBDIR=$(printf 'iter_%07d' "${EVAL_ITER_NUM}")
else
   EVAL_SUBDIR=${EVAL_TAG}
fi
SAVE_DIR=${EVAL_OUT_ROOT}/${EVAL_SUBDIR}
LOG_FILE="${SAVE_DIR}/run.log"
mkdir -p "${SAVE_DIR}"
RESOLVED_EVAL_CONFIG_PATH="${SAVE_DIR}/eval_config.resolved.yaml"
sed "s|path: data/|path: ${DATA_ROOT}/data/|g" "${EVAL_CONFIG_PATH}" >"${RESOLVED_EVAL_CONFIG_PATH}"
touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

export PYTHONUNBUFFERED=1

echo "EVAL_HF_CKPT: ${EVAL_HF_CKPT}"
echo "EVAL_TAG: ${EVAL_TAG}"
echo "EXPT_LABEL: ${EXPT_LABEL}"
echo "SAVE_DIR: ${SAVE_DIR}"

source "${SCRIPT_HELPER_DIR}/ray/start_cluster.sh"
source "${SCRIPT_HELPER_DIR}/models/qwen3.5-27B.sh"

TP=4
SAVE_INTERVAL=${SAVE_INTERVAL:-1}

CKPT_ARGS=(
   --hf-checkpoint ${EVAL_HF_CKPT}
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

# Single-turn: intentionally no --use-multi-turn / --max-turns.
DRKERNEL_PLUGIN_ARGS=(
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
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
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
    \"KERNELGYM_CLIENT_TIMEOUT_S\": \"${KERNELGYM_CLIENT_TIMEOUT_S}\",
    \"DRKERNEL_EVAL_MAX_CONCURRENCY\": \"${DRKERNEL_EVAL_MAX_CONCURRENCY}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF_VALUE}\"
  }
}"

# Pre-launch render check against the checkpoint actually being evaluated
# (single-turn prompt rendering; only tokenizer + chat_template are read).
_RENDER_CHECK_ARGS=(
   --hf-checkpoint "${EVAL_HF_CKPT}"
   --drkernel-gpu-name "${DRKERNEL_GPU_NAME}"
   --drkernel-compiler-name "${DRKERNEL_COMPILER_NAME}"
)
if [ -n "${DRKERNEL_GPU_NAME}" ]; then
   _RENDER_CHECK_ARGS+=(--expected-gpu-words "${DRKERNEL_GPU_NAME}")
fi
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
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
