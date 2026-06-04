#!/bin/bash
#
# Milestone A — single-turn DrKernel GRPO TRAINING smoke (Qwen3.6-27B BF16).
#
# Goal: validate the post-merge TRAINING plumbing (loss moves / advantage != 0 /
# checkpoint saves), NOT quality. Single-turn only (no --use-multi-turn).
#
# Derived from scripts/eval_drkernel/rollout_speedup_ablation/debug.27b.tp4.eagle.H20.sh
# with these deltas (each marked  # [A] ... below):
#   - drop --debug-rollout-only            -> actually train (ref weights load + backward)
#   - --num-rollout 0 -> ${NUM_ROLLOUT:-3} -> a few real training steps
#   - drop --use-multi-turn / --max-turns  -> single-turn (decision: A doesn't use multi-turn)
#   - --actor-num-gpus-per-node 8 -> 4     -> this host has 6xH20; TP4 needs a multiple of 4
#   - smaller batch (rollout 4 x n8 = gbs 32) + smaller CTX_LEN  -> fast smoke
#   - drop EAGLE speculative args           -> fewer moving parts during weight-sync
# Prereqs: torch_dist at ${MODEL_DIR}/torch_dist (done), KernelGym at --rm-url, Ray up.

set -eo pipefail

TP=4
CP=2
SAVE_INTERVAL=${SAVE_INTERVAL:-1}
CTX_LEN=${CTX_LEN:-16384}                          # [A] smoke: 16384 (eval used 65536)
NUM_ROLLOUT=${NUM_ROLLOUT:-3}                       # [A] a few training steps
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-4}         # [A] prompts/step (eval used 32)
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}     # GRPO group size
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
KERNELGYM_ERROR_SUMMARY_CHARS=${KERNELGYM_ERROR_SUMMARY_CHARS:-1600}
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-32}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-4}   # [A] TP4 fits in 6xH20

PYTORCH_CUDA_ALLOC_CONF_VALUE=${PYTORCH_CUDA_ALLOC_CONF_VALUE-expandable_segments:True}
EXPT_LABEL=t1.tp${TP}.bf16.H20
ROLLOUT_MAX_PROMPT_LEN=$((CTX_LEN - 1))
ROLLOUT_MAX_RESPONSE_LEN=$((CTX_LEN - 1))

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
SCRIPT_HELPER_DIR=${SCRIPT_HELPER_DIR:-/nfs/FM/chenshuailin/projects/kernel_agents/slime/scripts}
DATA_ROOT=/nfs/FM/chenshuailin/projects/kernel_agents/slime
PROMPT_DATA_PATH=${DATA_ROOT}/data/drkernel-rl-data-0513/train.parquet
MODEL_DIR=/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B
REF_LOAD_DIR=${MODEL_DIR}

RUN_TS="$(date +%Y%m%d_%H%M%S)"
SAVE_DIR=checkpoints/${MODEL_DIR##*/}/${RUN_TS}_${EXPT_LABEL}_ctx${CTX_LEN}
LOG_FILE="${SAVE_DIR}/run.log"
mkdir -p "${SAVE_DIR}"
touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

export PYTHONUNBUFFERED=1

source "${SCRIPT_HELPER_DIR}/ray/start_cluster.sh"
source "${SCRIPT_HELPER_DIR}/models/qwen3.5-27B.sh"


CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}
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
   --num-rollout ${NUM_ROLLOUT}                     # [A] was 0
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}       # [A] was 32
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
   --rollout-max-prompt-len ${ROLLOUT_MAX_PROMPT_LEN}
   --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN}
   --rollout-max-context-len ${CTX_LEN}
   --rollout-temperature 1

   # [A] DAPO dynamic sampling: drop zero-variance (all-pass/all-fail) groups.
   # Use slime's BUILT-IN filter (no custom code); dev_lhb's filter_cuda_kernel_group
   # only adds multi-turn/padding guards we don't need single-turn.
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --over-sampling-batch-size $((ROLLOUT_BATCH_SIZE * 2))

   --global-batch-size ${GLOBAL_BATCH_SIZE}         # [A] was 256
   --balance-data
   # [A] NO --debug-rollout-only  -> real training (ref load + backward + checkpoint)
)

EVAL_ARGS=(
   # --eval-interval 9999
   --skip-eval-before-train
   --rm-url http://127.0.0.1:20391
)

DRKERNEL_GPU_NAME=${DRKERNEL_GPU_NAME:-"NVIDIA H20 (SM 9.0, Hopper)"}
DRKERNEL_COMPILER_NAME=${DRKERNEL_COMPILER_NAME:-"CUDA 12.9 (nvcc)"}

# [A] single-turn: NO --use-multi-turn / --max-turns
DRKERNEL_PLUGIN_ARGS=(
   --kernelgym-error-summary-chars ${KERNELGYM_ERROR_SUMMARY_CHARS}
   --drkernel-gpu-name "${DRKERNEL_GPU_NAME}"
   --drkernel-compiler-name "${DRKERNEL_COMPILER_NAME}"
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TP}
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size ${CP}
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216

   # [A] DAPO token-level policy-gradient loss (normalize by total tokens, not
   # per-sample-mean) — pairs with clip-higher; aligns with dev_lhb.
   --calculate-per-token-loss
)

# [A] rloo + clip-higher. Single-turn rloo == dev_lhb trloo (gamma-fold of 1 turn is
# identity, per-(prompt,turn) grouping degenerates to per-prompt). rloo advantage =
# (r - group_mean) * g/(g-1) (no std-norm), computed in _post_process_rewards.
RL_ARGS=(
   --advantage-estimator rloo
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

   # [A] Memory: 27B at TP4/DP1 keeps full fp32 Adam states in-GPU (~120GB/GPU) -> OOM.
   # Offload optimizer to CPU (host has ample RAM) + precision-aware states. Matches dev_lhb.
   --use-distributed-optimizer
   --overlap-grad-reduce
   --overlap-param-gather
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
)

# [A] EAGLE speculative dropped for the smoke (one less thing to interact with weight-sync).
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${TP}
   --sglang-context-length ${CTX_LEN}
   --sglang-max-running-requests ${SGLANG_MAX_RUNNING_REQUESTS}
   --sglang-mem-fraction-static 0.75
   --sglang-decode-log-interval 400
   --sglang-mamba-scheduler-strategy extra_buffer
   --router-policy consistent_hashing
   --sglang-cuda-graph-max-bs ${SGLANG_MAX_RUNNING_REQUESTS}
   --sglang-disable-custom-all-reduce
   --sglang-linear-attn-backend flashinfer
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --log-probs-chunk-size 10000   # [A] chunk log-prob compute to cut peak memory (dev_lhb)
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

# Pre-launch render sanity (tokenizer + chat_template only).
_RENDER_CHECK_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --drkernel-gpu-name "${DRKERNEL_GPU_NAME}"
   --drkernel-compiler-name "${DRKERNEL_COMPILER_NAME}"
)
PYTHONPATH="${REPO_ROOT}:${SCRIPT_HELPER_DIR}/..:${PYTHONPATH:-}" \
   python3 "${SCRIPT_HELPER_DIR}/eval_drkernel/render_prompt_check.py" "${_RENDER_CHECK_ARGS[@]}"

submit_ray_job --address="${RAY_JOB_ADDRESS}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-gpus-per-node ${ACTOR_NUM_GPUS_PER_NODE} \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${RL_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${DRKERNEL_PLUGIN_ARGS[@]}"
