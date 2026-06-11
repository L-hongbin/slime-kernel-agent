#!/bin/bash
#
# Milestone A — single-turn DrKernel GRPO TRAINING smoke (Qwen3.5-9B BF16).
#
# Goal: validate the post-merge TRAINING plumbing (loss moves / advantage != 0 /
# checkpoint saves), NOT quality. Single-turn only (no --use-multi-turn).
#
# Derived from scripts/eval_drkernel/rollout_speedup_ablation/debug.27b.tp4.eagle.H20.sh
# with these deltas (each marked  # [A] ... below):
#   - drop --debug-rollout-only            -> actually train (ref weights load + backward)
#   - --num-rollout 0 -> ${NUM_ROLLOUT:-3} -> a few real training steps
#   - drop --use-multi-turn / --max-turns  -> single-turn (decision: A doesn't use multi-turn)
#   - smaller batch (rollout 4 x n8 = gbs 32) + smaller CTX_LEN  -> fast smoke
#   - keep EAGLE speculative args           -> avoid the non-spec FlashInfer GDN
#                                                decode CUDA-graph alignment failure
# Prereqs: torch_dist at ${MODEL_DIR}/torch_dist (done), KernelGym at --rm-url, Ray up.

set -eo pipefail

TP=4
CP=2
PP=1

CTX_LEN=${CTX_LEN:-16384}                          # [A] smoke: 16384 (eval used 65536)
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}        # [A] prompts/step (eval used 32)
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}     # GRPO group size
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
NUM_ROLLOUT=${NUM_ROLLOUT:-9999}
KERNELGYM_ERROR_SUMMARY_CHARS=${KERNELGYM_ERROR_SUMMARY_CHARS:-1600}
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-32}
# Single-node: TP4 x PP1 x CP2 = 8 GPUs = DP1 on slime's default 8 GPUs/node.
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}

EXPT_LABEL=debug.t1.9B.bf16.TP${TP}.PP${PP}.CP${CP}.tis.eagle.colocate.offload.ctx${CTX_LEN}.gradf32.H20
EXPT_SUFFIX=${EXPT_SUFFIX:-}
if [[ -n "${EXPT_SUFFIX}" ]]; then
   EXPT_LABEL="${EXPT_LABEL}.${EXPT_SUFFIX}"
fi
ROLLOUT_MAX_PROMPT_LEN=${ROLLOUT_MAX_PROMPT_LEN:-$((CTX_LEN - 1))}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-$((CTX_LEN - 1))}
MASTER_ADDR=${MASTER_ADDR:-10.11.2.164}              # node64 host-network IP
NODE_ADDR=${NODE_ADDR:-${MASTER_ADDR}}

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_HELPER_DIR=${SCRIPT_HELPER_DIR:-${REPO_ROOT}/scripts}
DATA_ROOT=/nfs/FM/chenshuailin/projects/kernel_agents/slime
PROMPT_DATA_PATH=${PROMPT_DATA_PATH:-${DATA_ROOT}/data/drkernel-rl-data-0513/train.parquet}
MODEL_DIR=/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.5-9B
REF_LOAD_DIR=/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.5-9B

RUN_TS="$(date +%Y%m%d_%H%M%S)"
SAVE_DIR=checkpoints/${MODEL_DIR##*/}/${RUN_TS}.${EXPT_LABEL}
LOG_FILE="${SAVE_DIR}/run.log"
mkdir -p "${SAVE_DIR}"
touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

ulimit -n 1048576
export PYTHONUNBUFFERED=1
export SLIME_DEBUG_TP_ATTRS=${SLIME_DEBUG_TP_ATTRS:-linear_qkv.weight}

bool_enabled() {
   case "${1:-}" in
      1|true|TRUE|yes|YES|on|ON) return 0 ;;
      *) return 1 ;;
   esac
}

SAVE_INTERVAL=${SAVE_INTERVAL:-10}
USE_FULLY_RESHARDABLE=${USE_FULLY_RESHARDABLE:-1}
USE_FULLY_RESHARDABLE_MEM_EFFICIENT=${USE_FULLY_RESHARDABLE_MEM_EFFICIENT:-${USE_FULLY_RESHARDABLE}}
USE_ASYNC_SAVE=${USE_ASYNC_SAVE:-1}
USE_PERSISTENT_CKPT_WORKER=${USE_PERSISTENT_CKPT_WORKER:-1}
CKPT_ASSUME_CONSTANT_STRUCTURE=${CKPT_ASSUME_CONSTANT_STRUCTURE:-0}
LOAD_DEBUG_ROLLOUT_DATA=${LOAD_DEBUG_ROLLOUT_DATA:-}
ENABLE_WANDB=${ENABLE_WANDB:-1}
DYNAMIC_SAMPLING_FILTER_PATH=${DYNAMIC_SAMPLING_FILTER_PATH-slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std}

# W&B: load the API key from a file OUTSIDE the repo (never hardcode the secret in
# this tracked script). The key reaches the primary Ray actor via runtime_env below,
# so it works regardless of which node the rank-0 process lands on. node70 reaches
# api.wandb.ai directly (verified ~0.5s); node64 also works (slower ~7s), no proxy.
WANDB_KEY_FILE=${WANDB_KEY_FILE:-${HOME}/.config/wandb/slime.key}
if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_KEY_FILE}" ]]; then
   WANDB_API_KEY="$(tr -d '[:space:]' < "${WANDB_KEY_FILE}")"
fi
export WANDB_API_KEY
WANDB_GROUP=${WANDB_GROUP:-${EXPT_LABEL}}

if [[ "${SLIME_SKIP_RAY_START:-0}" != "1" ]]; then
   source "${SCRIPT_HELPER_DIR}/ray/start_cluster.sh"
   if [[ "${RAY_ROLE:-head}" == "worker" ]]; then
      echo "ERROR: worker role is managed by multi_node_train.py; do not run this train script directly as RAY_ROLE=worker."
      exit 2
   fi
else
   SLIME_RAY_START_CLUSTER_ON_SOURCE=0 source "${SCRIPT_HELPER_DIR}/ray/start_cluster.sh"
   RAY_JOB_ADDRESS=${RAY_JOB_ADDRESS:-http://127.0.0.1:${RAY_DASHBOARD_PORT:-8265}}
   if [[ -z "${HAS_NVLINK:-}" ]]; then
      NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | { grep -o 'NV[0-9][0-9]*' || true; } | wc -l)
      if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
         HAS_NVLINK=1
      else
         HAS_NVLINK=0
      fi
      export HAS_NVLINK
   fi
   export RAY_JOB_ADDRESS
fi

source "${SCRIPT_HELPER_DIR}/models/qwen3.5-9B.sh"


CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}
   --ref-load ${REF_LOAD_DIR}/torch_dist_tp${TP}_pp${PP}
   --save ${SAVE_DIR}/
   --load ${SAVE_DIR}/
   --save-interval ${SAVE_INTERVAL}

   # --save-hf ${SAVE_DIR}/hf/iter_{rollout_id}
)
if bool_enabled "${USE_FULLY_RESHARDABLE}"; then
   CKPT_ARGS+=(--dist-ckpt-optim-fully-reshardable)
   if bool_enabled "${USE_FULLY_RESHARDABLE_MEM_EFFICIENT}"; then
      CKPT_ARGS+=(--distrib-optim-fully-reshardable-mem-efficient)
   fi
fi
if bool_enabled "${USE_ASYNC_SAVE}"; then
   CKPT_ARGS+=(--async-save)
fi
if bool_enabled "${USE_PERSISTENT_CKPT_WORKER}"; then
   CKPT_ARGS+=(--use-persistent-ckpt-worker)
fi
if bool_enabled "${CKPT_ASSUME_CONSTANT_STRUCTURE}"; then
   CKPT_ARGS+=(--ckpt-assume-constant-structure)
fi

ROLLOUT_ARGS=(
   --custom-rm-path slime_plugins.drkernel.kernelgym_rm.custom_rm
   --rollout-function-path slime_plugins.drkernel.rollout.generate_rollout
   --prompt-data ${PROMPT_DATA_PATH}
   --input-key ground_truth
   --label-key ground_truth
   --metadata-key extra_info
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}       # [A] was 32
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
   --rollout-max-prompt-len ${ROLLOUT_MAX_PROMPT_LEN}
   --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN}
   --rollout-max-context-len ${CTX_LEN}
   --rollout-temperature 1

   --over-sampling-batch-size $((ROLLOUT_BATCH_SIZE * 2))

   --global-batch-size ${GLOBAL_BATCH_SIZE}         # [A] was 256
   --balance-data
   # [A] NO --debug-rollout-only  -> real training (ref load + backward + checkpoint)
)
if [[ -n "${DYNAMIC_SAMPLING_FILTER_PATH}" ]]; then
   ROLLOUT_ARGS+=(--dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER_PATH}")
fi
if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA}" ]]; then
   ROLLOUT_ARGS+=(--load-debug-rollout-data "${LOAD_DEBUG_ROLLOUT_DATA}")
fi

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
   --pipeline-model-parallel-size ${PP}
   # PP imbalance fix: the last stage also carries the 248320-vocab output layer +
   # cross-entropy + MTP head, so give it fewer transformer layers. PP=2 only.
   # With block recompute=25, 64 layers -> first 33 / last 31 reduces the PP0
   # activation peak seen with first 34 / last 30.
   --context-parallel-size ${CP}
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   # --recompute-granularity selective
   # --recompute-modules core_attn layernorm mlp

   --recompute-granularity full
   --recompute-method block
   --recompute-num-layers 25

   # --recompute-granularity full
   # --recompute-method uniform
   # --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
   --log-probs-max-tokens-per-gpu 16384

   # [A] DAPO token-level policy-gradient loss (normalize by total tokens, not
   # per-sample-mean) — pairs with clip-higher; aligns with dev_lhb.
   --calculate-per-token-loss
   --init-model-with-meta-device
   --qwen-gdn-backend flashqla
)
if [[ "${PP}" -gt 1 ]]; then
   PERF_ARGS+=(--decoder-last-pipeline-num-layers 31)
fi


# [A] rloo + clip-higher. Single-turn rloo == dev_lhb trloo (gamma-fold of 1 turn is
# identity, per-(prompt,turn) grouping degenerates to per-prompt). rloo advantage =
# (r - group_mean) * g/(g-1) (no std-norm), computed in _post_process_rewards.
RL_ARGS=(
   --advantage-estimator rloo
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --mtp-num-layers 1
   --enable-mtp-training
   --mtp-loss-scaling-factor 0.2
   --use-tis
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
   --use-precision-aware-optimizer
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
)

# W&B online logging. API key comes from WANDB_API_KEY (injected into runtime_env),
# NOT --wandb-key, so the secret never appears in the actor argv / `ps` output.
WANDB_ARGS=()
if bool_enabled "${ENABLE_WANDB}"; then
   WANDB_ARGS+=(
      --use-wandb
      --wandb-project slime
      --wandb-group ${WANDB_GROUP}
      --disable-wandb-random-suffix
      --wandb-centralized
   )
fi

# [A] Keep EAGLE speculative decoding. Without it, FlashInfer GDN T=1 decode
# hit a 16-byte tensor-alignment failure during CUDA graph capture on H20.
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${TP}
   --sglang-context-length ${CTX_LEN}
   --sglang-max-running-requests ${SGLANG_MAX_RUNNING_REQUESTS}
   --sglang-mem-fraction-static 0.7
   --sglang-decode-log-interval 400
   --sglang-mamba-scheduler-strategy extra_buffer
   --router-policy round_robin
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
   # --grad-reduce-in-bf16
   --attention-softmax-in-fp32
   --attention-backend flash
   --log-probs-chunk-size 10000   # [A] chunk log-prob compute to cut peak memory (dev_lhb)
   --update-weight-buffer-size 1073741824
)

RAY_LOCAL_IP=${RAY_LOCAL_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}
LOCAL_NO_PROXY="localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR},${NODE_ADDR}"
if [ -n "${RAY_LOCAL_IP}" ]; then
   LOCAL_NO_PROXY="${LOCAL_NO_PROXY},${RAY_LOCAL_IP}"
fi
LOCAL_NO_PROXY="${LOCAL_NO_PROXY}${no_proxy:+,${no_proxy}}${NO_PROXY:+,${NO_PROXY}}"

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    "no_proxy": "${LOCAL_NO_PROXY}",
    "NO_PROXY": "${LOCAL_NO_PROXY}",
    "WANDB_API_KEY": "${WANDB_API_KEY}",
    "PYTHONPATH": "${REPO_ROOT}:/root/Megatron-LM/",
    "SLIME_DEBUG_TP_ATTRS": "${SLIME_DEBUG_TP_ATTRS}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "NCCL_IB_HCA": "${NCCL_IB_HCA:-mlx5_0,mlx5_4,mlx5_5}"
  }
}
EOF_JSON
)
   #  "SLIME_TENSOR_BACKUP_PIN_MEMORY": "0",

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
   --actor-num-nodes ${ACTOR_NUM_NODES} \
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
