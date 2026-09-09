#!/bin/bash

set -euo pipefail

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1
export CUDA_AGENT_NUM_WARMUP=20
export CUDA_AGENT_NUM_PERF_TRIALS=50
export CUDA_AGENT_USE_REFERENCE_CACHE=1
export CUDA_AGENT_REFER_NUM_PERF_TRIALS=150
export CUDA_AGENT_SPEEDUP_REWARD_MODE="${CUDA_AGENT_SPEEDUP_REWARD_MODE:-legacy}"
export CUDA_AGENT_SPEEDUP_UNCERTAINTY_Z_SCORE="${CUDA_AGENT_SPEEDUP_UNCERTAINTY_Z_SCORE:-1.96}"
export CUDA_AGENT_SPEEDUP_UNCERTAINTY_LOG_STD_FLOOR="${CUDA_AGENT_SPEEDUP_UNCERTAINTY_LOG_STD_FLOOR:-0.0}"
export CUDA_AGENT_PERFORMANCE_REWARD_REQUIRES_CORRECTNESS="${CUDA_AGENT_PERFORMANCE_REWARD_REQUIRES_CORRECTNESS:-1}"
export CUDA_AGENT_ENABLE_DYNAMIC_REWARD_WEIGHT="${CUDA_AGENT_ENABLE_DYNAMIC_REWARD_WEIGHT:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# MODEL CONFIG
source "${SCRIPT_DIR}/../../scripts/models/qwen3.5-27B.sh"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# NODE CONFIG
MASTER_ADDR="${MASTER_ADDR:-192.168.112.68}"
case "${MASTER_ADDR}" in
    192.168.112.68)
      REMOTE_HOSTS=(
      )
      REMOTE_PORTS=(
      )
      ;;
    192.168.112.36)
      REMOTE_HOSTS=(
      )
      REMOTE_PORTS=(
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
ACTOR_NUM_NODES=4
ACTOR_NUM_GPUS_PER_NODE=8
ACTOR_GPUS=$((ACTOR_NUM_NODES*ACTOR_NUM_GPUS_PER_NODE))
ROLLOUT_GPUS=$((NUM_GPUS-ACTOR_GPUS))
echo "ACTOR_GPUS ${ACTOR_GPUS} ROLLOUT_GPUS ${ROLLOUT_GPUS}"
# EXP CONFIG
CONTEXT_LEN=38000
MAX_RESPONSE_LEN=14000
MODEL_NAME="Qwen3.6-27B"
HF_MODEL_PATH="/ms/FM/checkpoints/Qwen-Zoo/Qwen3.6-27B/"
MEGATRON_MODEL_PATH="/ms/FM/lihongbin/dataset/CUDA_RL/megatron_ckpt/Qwen3.6-27B-TP4-PP2-Torch-Dist"

KERNEL_BACKEND="tvm_ffi"
KERNEL_ENV_URL="${KERNEL_ENV_URL:-"http://192.168.116.92:20111"}"
LOSS_MODE="${LOSS_MODE:-dppoTV}"
USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-"true"}"
KEEP_OLD_ACTOR="${KEEP_OLD_ACTOR:-"true"}"
USE_CTM="${USE_CTM:-"false"}"
CALC_LOSS_MODE="${CALC_LOSS_MODE:-"PerToken"}"
LOAD_PATH="${LOAD_PATH:-""}"

DATASET="${DATASET:-"Hard-18K"}"

EXP_PARAM="${LOSS_MODE}Fp32HeadCTX${CONTEXT_LEN}Resp$MAX_RESPONSE_LEN"
case "${DATASET}" in
    "Drkernel-rl-thinking-TVM-V2")
      RL_DATA="/ms/FM/lihongbin/dataset/CUDA_RL/cuda_rl/prompt_tvm_v2/drkernel_rl_thinking.parquet"
      ;;
    "Drkernel-RL-TVM-GEPA4o")
      RL_DATA="/ms/FM/lihongbin/dataset/CUDA_RL/cuda_rl/prompt_tvm_GEPA4o/drkernel_rl_thinking.parquet"
      ;;
    "Hard-23K")
      RL_DATA="/ms/FM/lihongbin/dataset/CUDA_RL/cuda_rl/prompt_tvm_GEPA4o/hard_torch_ops_23k.parquet"
      ;;
    "Hard-Syn-23K")
      RL_DATA="/ms/FM/lihongbin/dataset/CUDA_RL/cuda_rl/prompt_tvm_GEPA4o/hard_syn_23k.parquet"
      ;;
    "Hard-18K")
      RL_DATA="/ms/FM/lihongbin/dataset/CUDA_RL/cuda_rl/prompt_tvm_GEPA4o/hard_torch_ops_18k.parquet"
      ;;
    *)
      echo "Unknown DATASET: ${DATASET}" >&2
      exit 1
      ;;
esac
TURN_PROMPT_PATH="$REPO_ROOT/examples/kernel_agent/prompt_config/response_prompt/tvm_ffi_gepa_kimi_v1.jinja"

case "${LOSS_MODE}" in
    cispo)
      LOSS_MODE="cispo"
      EPS_CLIP=10
      EPS_CLIP_HIGH=0.2
      ;;
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
EXP_ARGS=(
   --enable-fp32-lm-head
   --policy-loss-mode $LOSS_MODE
   --eps-clip $EPS_CLIP
   --eps-clip-high $EPS_CLIP_HIGH
   --advantage-estimator trloo
   --multi-turn-gamma 1.0
   --entropy-coef 0.00
   --log-exp-metrics
)

MIS_ARGS=(
   --rollout-data-postprocess-path examples.kernel_agent.kernel_filter.sequence_mis
   --sequence-mis-config '{"aggregation":"turns_geometric","token_veto_threshold":1e-4,"lower":0.999,"upper":1.001,"use_advantage":false}'
   --enable-turns-dp-partitions
)

if [ "$USE_ROLLOUT_LOGPROBS" = "true" ]; then
   EXP_PARAM+="UseRolloutLogprob"
   EXP_ARGS+=(
      --use-rollout-logprobs
   )
   MIS_ARGS=()
elif [ "$KEEP_OLD_ACTOR" = "true" ]; then
   EXP_ARGS+=(
      --keep-old-actor
   )
fi

if [ "$USE_CTM" != "false" ]; then
   EXP_PARAM+="CTM$USE_CTM"
   EXP_ARGS+=(
      --use-conditional-truncation-mask
      --conditional-truncation-mask-prob $USE_CTM
   )
fi

EXP_PARAM+="Calc$CALC_LOSS_MODE"
if [[ "$CALC_LOSS_MODE" == "PerToken" ]]; then
   EXP_ARGS+=(
      --calculate-per-token-loss
   )
fi
EXP_ARGS+=("${MIS_ARGS[@]}")
EXP_NAME="Kernel-FullAsync-${KERNEL_BACKEND^^}"
EXP_NAME="${EXP_NAME//_/-}-$MODEL_NAME-$DATASET-TurnPromptGEPAKimi-$EXP_PARAM"

echo "HF_MODEL_PATH: ${HF_MODEL_PATH}"
echo "MEGATRON_MODEL_PATH: ${MEGATRON_MODEL_PATH}"
echo "RL_DATA: ${RL_DATA}"
echo "Turn prompt path: ${TURN_PROMPT_PATH}"
echo "EXP_NAME: ${EXP_NAME}"

TENSORBOARD_DIR="/ms/FM/lihongbin/kernel_rl/tensorboard_log/${EXP_NAME}"
echo "TENSORBOARD_DIR ${TENSORBOARD_DIR}"

SAVE_PATH="/ms/FM/lihongbin/kernel_rl/checkpoints/${EXP_NAME}"

if [[ -z "${LOAD_PATH}" ]]; then
   LOAD_PATH="${SAVE_PATH}"
fi

if ! python3 "${REPO_ROOT}/scripts/check_kernelgym_health.py" \
   --url "${KERNEL_ENV_URL}" \
   --timeout "${KERNELGYM_HEALTH_TIMEOUT:-5}" \
   --attempts "${KERNELGYM_HEALTH_ATTEMPTS:-3}" \
   --interval "${KERNELGYM_HEALTH_INTERVAL:-2}"; then
   echo "KernelGym health check failed at ${KERNEL_ENV_URL}" >&2
   exit 1
fi

if [[ -f "${SAVE_PATH}/latest_checkpointed_iteration.txt" && "${SAVE_PATH}" != "${LOAD_PATH}" ]]; then
   echo "ERROR: --save already contains latest_checkpointed_iteration.txt but --load is different." >&2
   echo "  --save: ${SAVE_PATH}" >&2
   echo "  --load: ${LOAD_PATH:-<unset>}" >&2
   echo "To resume this run, set LOAD_PATH to SAVE_PATH, or choose a fresh SAVE_PATH." >&2
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

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

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
   --save-interval 5
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
   --num-rollout 3000
   --rollout-batch-size 16
   --n-samples-per-prompt 16
   --rollout-max-response-len $MAX_RESPONSE_LEN
   --rollout-max-context-len $CONTEXT_LEN
   --rollout-temperature 1
   --global-batch-size 128
   --balance-data
)
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --num-virtual-stages-per-pipeline-rank 2
   --context-parallel-size 4
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9120
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
   --use-distributed-optimizer
   --overlap-grad-reduce
   --overlap-param-gather
)
# --overlap-grad-reduce
# --overlap-param-gather
# --optimizer-cpu-offload
#    --overlap-cpu-optimizer-d2h-h2d
#    --use-precision-aware-optimizer

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group search-r1_qwen2.5-3B-test
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --router-policy round_robin
   --sglang-mem-fraction-static 0.75
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 4
   --sglang-mamba-scheduler-strategy extra_buffer
   --sglang-server-concurrency 16
   --sglang-context-length $CONTEXT_LEN
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
   # TIS-related args, recommended to enable when using TIS
   # --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   # --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
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
    "CUDA_AGENT_SPEEDUP_REWARD_MODE": "${CUDA_AGENT_SPEEDUP_REWARD_MODE}",
    "CUDA_AGENT_SPEEDUP_UNCERTAINTY_Z_SCORE": "${CUDA_AGENT_SPEEDUP_UNCERTAINTY_Z_SCORE}",
    "CUDA_AGENT_SPEEDUP_UNCERTAINTY_LOG_STD_FLOOR": "${CUDA_AGENT_SPEEDUP_UNCERTAINTY_LOG_STD_FLOOR}",
    "CUDA_AGENT_PERFORMANCE_REWARD_REQUIRES_CORRECTNESS": "${CUDA_AGENT_PERFORMANCE_REWARD_REQUIRES_CORRECTNESS}",
    "CUDA_AGENT_ENABLE_DYNAMIC_REWARD_WEIGHT": "${CUDA_AGENT_ENABLE_DYNAMIC_REWARD_WEIGHT}"
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
