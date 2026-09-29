#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# MODEL CONFIG
source "${SCRIPT_DIR}/../../scripts/models/qwen3.5-27B.sh"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
# NODE CONFIG
MASTER_ADDR="192.168.112.42"
case "${MASTER_ADDR}" in
    192.168.112.42)
      ACTOR_NUM_NODES=4
      CP_SIZE=2
      REMOTE_HOSTS=(
        "192.168.112.92" "192.168.112.11" "192.168.112.68" "192.168.112.67" "192.168.112.36"
      )
      REMOTE_PORTS=(
        14997 16921 16921 11090 11090
      )
      ;;
    *)
      echo "No config for addr $MASTER_ADDR"
      exit 1
      ;;
esac
if [ "${#REMOTE_PORTS[@]}" -ne "${#REMOTE_HOSTS[@]}" ]; then
   echo "REMOTE_PORTS length (${#REMOTE_PORTS[@]}) must match REMOTE_HOSTS length (${#REMOTE_HOSTS[@]})."
   exit 1
fi
NUM_NODES=$((1 + ${#REMOTE_HOSTS[@]}))
GPUS_PER_NODE=8
NUM_GPUS=$((NUM_NODES * GPUS_PER_NODE))
ACTOR_NUM_GPUS_PER_NODE=8
ACTOR_GPUS=$((ACTOR_NUM_NODES*ACTOR_NUM_GPUS_PER_NODE))
ROLLOUT_GPUS=$((NUM_GPUS-ACTOR_GPUS))

if (( ROLLOUT_GPUS <= 0 )); then
   echo "ROLLOUT_GPUS must be greater than 0, got ${ROLLOUT_GPUS} (NUM_GPUS=${NUM_GPUS}, ACTOR_GPUS=${ACTOR_GPUS})." >&2
   exit 1
fi


echo "ACTOR_GPUS ${ACTOR_GPUS} ROLLOUT_GPUS ${ROLLOUT_GPUS}"
# EXP CONFIG
NUM_ROLLOUT=500
ROLLOUT_BATCH_SIZE=16
CONTEXT_LEN=120000
TEMPERATURE=1.0
TOP_P=0.95
MAX_RESPONSE_LEN=32000


MODEL_NAME="Qwen3.8-27B"
REASONING_EFFORT="${REASONING_EFFORT:-"medium"}"
HF_MODEL_PATH="/ms/FM/checkpoints/Qwen-Zoo/Qwen3.8-27B/"
MEGATRON_MODEL_PATH="/ms/FM/lihongbin/dataset/CUDA_RL/megatron_ckpt/Qwen3.8-27B-TP4-PP2-GDN-Dist-Torch-Dist"
# "/ms/FM/lihongbin/kernel_rl/checkpoints/Kernel-FullAsyncWarmup-TVM-FFI-Qwen3.8-27B-TorchOpsV4_DifficultyLt18-DppoTVUseRolloutLogprobFp32HeadCalcPerSampleCTX40000Resp24000T1.0TopP0.95"
LOAD_PATH="$warmup_ckpt"
KERNEL_BACKEND="tvm_ffi"
KERNEL_ENV_URL="${KERNEL_ENV_URL:-"http://192.168.116.44:20111"}"
LOSS_MODE="${LOSS_MODE:-dppoTV}"
USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-"true"}"

USE_FP32HEAD="${USE_FP32HEAD:-"true"}"
CALC_LOSS_MODE="${CALC_LOSS_MODE:-"PerSample"}"
USE_CTM="${USE_CTM:-"false"}"

RECORD_MEMORY="${RECORD_MEMORY:-"false"}"
DATASET="${DATASET:-""}"
TURN_PROMPT_PATH="$REPO_ROOT/examples/kernel_agent/prompt_config/response_prompt/tvm_ffi_gepa_kimi_v2.jinja"
MAX_TOKENS_PER_GPU=9120
OFFLOAD_OPTIMIZE="${OFFLOAD_OPTIMIZE:-"false"}"
OPTIMIZER="${OPTIMIZER:-"adam"}"
QWEN_GDN_IMPLEMENTATION=${QWEN_GDN_IMPLEMENTATION:-distributed}
CP_PARTITION_MODE=${CP_PARTITION_MODE:-zigzag}
CUDA_AGENT_APPLY_FAILED_GROUP_REWARD=${CUDA_AGENT_APPLY_FAILED_GROUP_REWARD:-0}
CUDA_AGENT_APPLY_KERNEL_FAILED_SCORE=${CUDA_AGENT_APPLY_KERNEL_FAILED_SCORE:-0}
DYNAMIC_REWARD=${DYNAMIC_REWARD:-None}

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1
export CUDA_AGENT_NUM_WARMUP=10
export CUDA_AGENT_NUM_PERF_TRIALS=100
export CUDA_AGENT_USE_REFERENCE_CACHE=1
export CUDA_AGENT_REFER_NUM_PERF_TRIALS=100
export CUDA_AGENT_ENABLE_NCU=1
export CUDA_AGENT_ENABLE_COMPUTE_SANITIZER=1
export CUDA_AGENT_RETURN_DETAIL_CORRECTNESS=1
export CUDA_AGENT_LOG_ROLLOUT_STATS_ONLY=1
export CUDA_AGENT_ENABLE_CORRECTNESS_INPUT_PERTURBATIONS=0
export CUDA_AGENT_APPLY_FAILED_GROUP_REWARD=$CUDA_AGENT_APPLY_FAILED_GROUP_REWARD
export CUDA_AGENT_APPLY_KERNEL_FAILED_SCORE=$CUDA_AGENT_APPLY_KERNEL_FAILED_SCORE   

EXP_PARAM="${LOSS_MODE^}"
case "${DATASET}" in
    "TorchOpsV4_DifficultyLt18")
      RL_DATA="/ms/FM/lihongbin/dataset/CUDA_RL/cuda_rl/prompt_tvm_GEPA4o_v2/torch_ops_difficulty_lt18.parquet"
      ;;
    *)
      echo "Unknown DATASET: ${DATASET}" >&2
      exit 1
      ;;
esac

if [ ! -f "$TURN_PROMPT_PATH" ]; then
   echo "Turn Prompt don't exists: ${TURN_PROMPT_PATH}" >&2
   exit 1
fi

case "${LOSS_MODE}" in
    dppoTV)
      LOSS_MODE="dppo_binary_tv"
      EPS_CLIP=0.2
      EPS_CLIP_HIGH=0.2
      ;;
    ppo)
      LOSS_MODE="ppo"
      EPS_CLIP=0.2
      EPS_CLIP_HIGH=0.28
      ;;
    *)
      echo "Unknown policy loss mode: ${LOSS_MODE}" >&2
      exit 1
      ;;
esac

case "${REASONING_EFFORT}" in
    xhigh)
      CHAT_TEMPLATE_KWARGS='{"enable_thinking":true,"reasoning_effort":"xhigh"}'
      ;;
    medium)
      CHAT_TEMPLATE_KWARGS='{"enable_thinking":true,"reasoning_effort":"medium"}'
      ;;
    low)
      CHAT_TEMPLATE_KWARGS='{"enable_thinking":true,"reasoning_effort":"low"}'
      ;;
    *)
      CHAT_TEMPLATE_KWARGS='{"enable_thinking":true}'
      ;;
esac

EXP_ARGS=(
   --policy-loss-mode $LOSS_MODE
   --eps-clip $EPS_CLIP
   --eps-clip-high $EPS_CLIP_HIGH
   --advantage-estimator trloo
   --multi-turn-gamma 1.0
   --entropy-coef 0.00
)

MIS_ARGS=()

if [ "$USE_ROLLOUT_LOGPROBS" = "true" ]; then
   EXP_PARAM+="RolloutLogprob"
   EXP_ARGS+=(
      --use-rollout-logprobs
   )
fi

if [ "$USE_CTM" != "false" ]; then
   EXP_PARAM+="CTM$USE_CTM"
   EXP_ARGS+=(
      --use-conditional-truncation-mask
      --conditional-truncation-mask-prob $USE_CTM
   )
fi

if [[ $USE_FP32HEAD == "true" ]]; then
   EXP_PARAM+="Fp32Head"
   EXP_ARGS+=(
      --enable-fp32-lm-head
   )
fi

EXP_PARAM+="$CALC_LOSS_MODE"
if [[ $CALC_LOSS_MODE == "PerToken" ]]; then
   EXP_ARGS+=(
      --calculate-per-token-loss
   )
elif [[ $CALC_LOSS_MODE == "PerPrompt" ]]; then
   EXP_ARGS+=(
      --calculate-per-prompt-loss
   )
elif [[ $CALC_LOSS_MODE == "Tokensum" ]]; then
   EXP_ARGS+=(
      --calculate-token-sum-loss
   )
fi

if [[ $OPTIMIZER == "pion" ]];then
   OFFLOAD_OPTIMIZE="false"
   EXP_PARAM+="Pion"
   echo "disable optimize offload for muon type optimizer"
   EXP_ARGS+=(
      --optimizer dist_pion
      --pion-per-head
      --muon-tp-mode distributed
      --muon-scale-mode spectral
      --muon-extra-scale-factor 5.0
   )
else
   EXP_ARGS+=(
      --optimizer adam
      --use-distributed-optimizer
      --overlap-grad-reduce
      --overlap-param-gather
   )
fi

if [[ $CUDA_AGENT_ENABLE_CORRECTNESS_INPUT_PERTURBATIONS == 1 ]]; then
   EXP_PARAM+="InputPerturbations"
fi

if [[ $CUDA_AGENT_APPLY_FAILED_GROUP_REWARD == 1 ]]; then
   EXP_PARAM+="AddFailedReward"
elif [[ $CUDA_AGENT_APPLY_KERNEL_FAILED_SCORE == 1 ]]; then
   EXP_PARAM+="ApplyPenalty"
fi

case "${DYNAMIC_REWARD}" in
   None) EXP_PARAM+="FixedRW" ;;
   sqrt) EXP_PARAM+="DynamicRWSqrt" ;;
   piecewise) EXP_PARAM+="DynamicRWPiecewise" ;;
   piecewise-sqrt) EXP_PARAM+="DynamicRWPiecewiseSqrt" ;;
   *)
      echo "DYNAMIC_REWARD must be None, sqrt, piecewise, or piecewise-sqrt." >&2
      exit 1
      ;;
esac
EXP_ARGS+=(--dynamic-reward-gate "${DYNAMIC_REWARD}" --dynamic-reward-gate-range 0.8 1.2)

EXP_ARGS+=("${MIS_ARGS[@]}")
EXP_NAME="Kernel-FAsync-${KERNEL_BACKEND^^}"
EXP_NAME="${EXP_NAME//_/-}-$MODEL_NAME-$DATASET-${EXP_PARAM}-CTX${CONTEXT_LEN}Resp${MAX_RESPONSE_LEN}T${TEMPERATURE}TopP${TOP_P}"

TENSORBOARD_DIR="/ms/FM/lihongbin/kernel_rl/tensorboard_log/${EXP_NAME}"
SAVE_PATH="/ms/FM/lihongbin/kernel_rl/checkpoints/${EXP_NAME}"


if ! python3 "${REPO_ROOT}/scripts/check_kernelgym_health.py" \
   --url "${KERNEL_ENV_URL}" \
   --timeout "${KERNELGYM_HEALTH_TIMEOUT:-5}" \
   --attempts "${KERNELGYM_HEALTH_ATTEMPTS:-3}" \
   --interval "${KERNELGYM_HEALTH_INTERVAL:-2}"; then
   echo "KernelGym health check failed at ${KERNEL_ENV_URL}" >&2
   exit 1
fi

NCCL_SOCKET_IFNAME="front1"
GLOO_SOCKET_IFNAME="front1"

RAY_DASHBOARD_PORT=8265
RAY_PORT=6379
RAY_HEAD_ADDR="${MASTER_ADDR}:${RAY_PORT}"
RAY_TEMP_DIR="/tmp/ray"

PYTHON_BIN=${PYTHON_BIN:-python3}
RAY_WAIT_TIMEOUT=${RAY_WAIT_TIMEOUT:-300}

# LOG CONFIG
LOG_DATE="$(date +%Y%m%d)"
LOG_TIME="$(date +%H%M%S)"
LOG_DIR="/ms/FM/lihongbin/kernel_rl/logs/${EXP_NAME}/${LOG_DATE}"
LOG_PATH="${LOG_DIR}/${LOG_TIME}.log"
echo "Logging to ${LOG_PATH}"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

case "${QWEN_GDN_IMPLEMENTATION}" in
   replicated|distributed) ;;
   *)
      echo "QWEN_GDN_IMPLEMENTATION must be one of: replicated, distributed." >&2
      exit 1
      ;;
esac

case "${CP_PARTITION_MODE}" in
   zigzag|contiguous) ;;
   *)
      echo "CP_PARTITION_MODE must be one of: zigzag, contiguous." >&2
      exit 1
      ;;
esac

if [[ "${QWEN_GDN_IMPLEMENTATION}" == "distributed" \
   && "${CP_SIZE}" -gt 1 \
   && "${CP_PARTITION_MODE}" != "zigzag" ]]; then
   echo "distributed Qwen GDN supports only CP_PARTITION_MODE=zigzag when context parallelism is enabled." >&2
   exit 1
fi

if [[ $OFFLOAD_OPTIMIZE == "true" ]];then
   EXP_ARGS+=(
      --optimizer-cpu-offload
      --overlap-cpu-optimizer-d2h-h2d
      --use-precision-aware-optimizer
   )
fi

# No NVLink matches is valid; preserve topology-query failures under pipefail.
NVLINK_COUNT=$(
   nvidia-smi topo -m |
      awk '{for (i=1; i<=NF; i++) if ($i ~ /^NV[0-9]+$/) n++} END {print n+0}'
)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
echo "HF_MODEL_PATH: ${HF_MODEL_PATH}"
echo "MEGATRON_MODEL_PATH: ${MEGATRON_MODEL_PATH}"
echo "RL_DATA: ${RL_DATA}"
echo "Turn prompt path: ${TURN_PROMPT_PATH}"
echo "EXP_NAME: ${EXP_NAME}"
echo "EXP_ARGS: ${EXP_ARGS[@]}"
echo "TENSORBOARD_DIR ${TENSORBOARD_DIR}"

run_ssh() {
   local host="$1"
   local port="$2"
   shift 2
   ssh -p "${port}" "${host}" "$@"
}

wait_for_cluster() {
   echo "Waiting for Ray cluster: expected nodes=${NUM_NODES}, expected GPUs=${NUM_GPUS}"
   local deadline=$((SECONDS + RAY_WAIT_TIMEOUT))
   local ready=0

   while ((SECONDS < deadline)); do
      if "${PYTHON_BIN}" - "${RAY_HEAD_ADDR}" "${NUM_NODES}" "${NUM_GPUS}" <<'PY'
import sys

import ray

ray_address = sys.argv[1]
expected_nodes = int(sys.argv[2])
expected_gpus = float(sys.argv[3])

ray.init(address=ray_address, ignore_reinit_error=True, logging_level="ERROR")
alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
gpu_count = sum(float(node.get("Resources", {}).get("GPU", 0)) for node in alive_nodes)

print(f"Ray alive nodes={len(alive_nodes)}, GPUs={gpu_count:g}")
sys.exit(0 if len(alive_nodes) >= expected_nodes and gpu_count >= expected_gpus else 1)
PY
      then
         ready=1
         break
      fi
      sleep 5
   done

   ray status --address "${RAY_HEAD_ADDR}" || true

   if [[ "${ready}" != "1" ]]; then
      echo "Ray cluster did not become ready within ${RAY_WAIT_TIMEOUT}s." >&2
      exit 1
   fi
}

# for rerun the task
pkill -9 sglang || true
ray stop --force || true
pkill -9 -f ray || true
pkill -9 python || true
for i in "${!REMOTE_HOSTS[@]}"; do
   run_ssh "${REMOTE_HOSTS[$i]}" "${REMOTE_PORTS[$i]}" "pkill -9 sglang || true; ray stop --force || true; pkill -9 ray || true; pkill -9 python || true"
done
sleep 3


TENSORBOARD_ARGS=(
   --use-tensorboard
)

LOGGING_ARGS=(
   --log-multi-turn
   --log-memory-to-tensorboard
   --log-timers-to-tensorboard
   --log-throughput
   --log-throughput-to-tensorboard
   --log-progress
   --log-device-memory-used
)

CKPT_ARGS=(
   --hf-checkpoint ${HF_MODEL_PATH}
   --ref-load ${MEGATRON_MODEL_PATH}
   --load $LOAD_PATH
   --no-load-optim
   --save $SAVE_PATH
   --save-interval 1
   --no-save-optim
   --async-save
)

ROLLOUT_ARGS=(
   --update-weights-interval 1
   --rollout-function-path examples.kernel_agent.fully_async_rollout.generate_rollout_fully_async
   --prompt-data ${RL_DATA}
   --input-key prompt
   --label-key reward_model
   --metadata-key extra_info
   --rollout-shuffle
   --num-rollout $NUM_ROLLOUT
   --rollout-batch-size $ROLLOUT_BATCH_SIZE
   --n-samples-per-prompt 16
   --rollout-max-response-len $MAX_RESPONSE_LEN
   --rollout-max-context-len $CONTEXT_LEN
   --rollout-temperature $TEMPERATURE
   --rollout-top-p $TOP_P
   --apply-chat-template-kwargs $CHAT_TEMPLATE_KWARGS
   --global-batch-size 128
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size $CP_SIZE
   --cp-partition-mode "${CP_PARTITION_MODE}"
   --qwen-gdn-implementation "${QWEN_GDN_IMPLEMENTATION}"
   --qwen-gdn-a2a-implementation "fused"
   --qwen-gdn-cache-thd-permutation
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --seq-length $CONTEXT_LEN
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 2
   --use-dynamic-batch-size
   --max-tokens-per-gpu $MAX_TOKENS_PER_GPU
)

if [[ "${QWEN_GDN_IMPLEMENTATION}" == "distributed" ]]; then
   PERF_ARGS+=(--qwen-gdn-sp-disable-batch-p2p-comm)
fi

OPTIMIZER_ARGS=(
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.0
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group search-r1_qwen2.5-3B-test
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --router-policy consistent_hashing
   --sglang-mem-fraction-static 0.85
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 4
   --sglang-mamba-radix-cache-strategy extra_buffer
   --sglang-server-concurrency 22
   --sglang-context-length 120000
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --log-probs-chunk-size 12000
   # need to comment this when using model with MLA
   --attention-backend flash
   --no-pin-cpu-grads
   --no-pin-cpu-params
)

CUSTOM_ARGS=(
   --custom-generate-function-path examples.kernel_agent.generate_with_cuda_agent.generate
   --custom-rm-path examples.kernel_agent.generate_with_cuda_agent.reward_func
   --custom-reward-post-process-path examples.kernel_agent.kernel_reward.reward_post_process_by_group
   --dynamic-sampling-filter-path examples.kernel_agent.kernel_filter.filter_cuda_kernel_group
   --multi-turn-prompt-config-path $TURN_PROMPT_PATH
)

KERNEL_AGENT_ARGS=(
   --kernel-env-url ${KERNEL_ENV_URL}
   --kernel-backend $KERNEL_BACKEND
   --reference-backend torch
   --do-precheck
   --finalize-mode positive
   --use-multi-turn
   --filter-by-last-turn
   --padding-turns
   --max-turns 3
   --use-coverage-rs
   --coverage-rs-key time_coverage
   --coverage-rs-threshold 0.3
   --coverage-rs-factor 0.1
)

   # --verify-rollout-ratio 0
   # --capture-verify-data
   # --save-verify-data './examples/kernel_agent/verify_data/verify_{rollout_id}.pt'
   # --verify-data-limit 10000

# launch the master node of ray in container
export MASTER_ADDR
export TENSORBOARD_DIR
# NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}" \
ray start \
   --head \
   --node-ip-address ${MASTER_ADDR} \
   --port ${RAY_PORT} \
   --dashboard-host 0.0.0.0 \
   --dashboard-port $RAY_DASHBOARD_PORT \
   --num-gpus ${GPUS_PER_NODE} \
   --disable-usage-stats \
   --temp-dir=$RAY_TEMP_DIR

for i in "${!REMOTE_HOSTS[@]}"; do
   echo "Starting Ray worker on ${REMOTE_HOSTS[$i]}"
   run_ssh "${REMOTE_HOSTS[$i]}" "${REMOTE_PORTS[$i]}" \
      "ray start --address ${RAY_HEAD_ADDR} --num-gpus ${GPUS_PER_NODE} --disable-usage-stats"
done

wait_for_cluster

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
    "GLOO_SOCKET_IFNAME": "${GLOO_SOCKET_IFNAME}",
    "TP_SOCKET_IFNAME": "${NCCL_SOCKET_IFNAME}",
    "NCCL_SOCKET_IFNAME": "${NCCL_SOCKET_IFNAME}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "PYTHONPATH": "/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "TENSORBOARD_DIR": "${TENSORBOARD_DIR}",
    "CUDA_AGENT_NUM_WARMUP": "${CUDA_AGENT_NUM_WARMUP}",
    "CUDA_AGENT_NUM_PERF_TRIALS": "${CUDA_AGENT_NUM_PERF_TRIALS}",
    "CUDA_AGENT_USE_REFERENCE_CACHE": "${CUDA_AGENT_USE_REFERENCE_CACHE}",
    "CUDA_AGENT_REFER_NUM_PERF_TRIALS": "${CUDA_AGENT_REFER_NUM_PERF_TRIALS}",
    "CUDA_AGENT_ENABLE_NCU": "${CUDA_AGENT_ENABLE_NCU}",
    "CUDA_AGENT_ENABLE_COMPUTE_SANITIZER": "${CUDA_AGENT_ENABLE_COMPUTE_SANITIZER}",
    "CUDA_AGENT_RETURN_DETAIL_CORRECTNESS": "${CUDA_AGENT_RETURN_DETAIL_CORRECTNESS}",
    "CUDA_AGENT_ENABLE_CORRECTNESS_INPUT_PERTURBATIONS": "${CUDA_AGENT_ENABLE_CORRECTNESS_INPUT_PERTURBATIONS}",
    "CUDA_AGENT_APPLY_FAILED_GROUP_REWARD": "${CUDA_AGENT_APPLY_FAILED_GROUP_REWARD}",
    "CUDA_AGENT_APPLY_KERNEL_FAILED_SCORE":"${CUDA_AGENT_APPLY_KERNEL_FAILED_SCORE}",
    "CUDA_AGENT_LOG_ROLLOUT_STATS_ONLY": "${CUDA_AGENT_LOG_ROLLOUT_STATS_ONLY}"
  }
}
EOF_JSON
)
ray job submit --address="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes ${ACTOR_NUM_NODES} \
   --actor-num-gpus-per-node ${ACTOR_NUM_GPUS_PER_NODE} \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${CURRICULUM_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${EXP_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${TENSORBOARD_ARGS[@]} \
   ${LOGGING_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${KERNEL_AGENT_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
