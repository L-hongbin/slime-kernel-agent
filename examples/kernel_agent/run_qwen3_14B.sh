#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

EXP_NAME="qwen3_14B_cuda_rl"
HF_MODEL_PATH="/nfs/FM/checkpoints/Qwen/Qwen3-14B/"
MEGATRON_MODEL_PATH="/nfs/FM/checkpoints/Qwen/Qwen3-14B_torch_dist/"
RL_DATA="/nfs/FM/lihongbin/datasets/CUDA_RL/RL_Data/prompt_v4/drkernel_rl_thinking.parquet"
KERNEL_ENV_URL="http://127.0.0.1:8002"
NCCL_SOCKET_IFNAME="ens22f0"
GLOO_SOCKET_IFNAME="ens22f0"

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
REMOTE_HOSTS=()
REMOTE_PORTS=()
NUM_NODES=$((1 + ${#REMOTE_HOSTS[@]}))
RAY_PORT=${RAY_PORT:-6379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
GPUS_PER_NODE=1
RAY_HEAD_ADDR="${MASTER_ADDR}:${RAY_PORT}"
PYTHON_BIN=${PYTHON_BIN:-python3}
RAY_WAIT_TIMEOUT=${RAY_WAIT_TIMEOUT:-300}
EXPECTED_GPUS=$((NUM_NODES * GPUS_PER_NODE))

if [ "${#REMOTE_PORTS[@]}" -ne "${#REMOTE_HOSTS[@]}" ]; then
   echo "REMOTE_PORTS length (${#REMOTE_PORTS[@]}) must match REMOTE_HOSTS length (${#REMOTE_HOSTS[@]})."
   exit 1
fi

LOG_DATE="$(date +%Y%m%d)"
LOG_TIME="$(date +%H%M%S)"
LOG_DIR="${SCRIPT_DIR}/logs/${EXP_NAME}/${LOG_DATE}"
LOG_PATH="${LOG_DIR}/${LOG_TIME}.log"
echo "Logging to ${LOG_PATH}"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

run_ssh() {
   local host="$1"
   local port="$2"
   shift 2
   ssh -p "${port}" "${host}" "$@"
}

wait_for_cluster() {
   echo "Waiting for Ray cluster: expected nodes=${NUM_NODES}, expected GPUs=${EXPECTED_GPUS}"
   local deadline=$((SECONDS + RAY_WAIT_TIMEOUT))
   local ready=0

   while ((SECONDS < deadline)); do
      if "${PYTHON_BIN}" - "${RAY_HEAD_ADDR}" "${NUM_NODES}" "${EXPECTED_GPUS}" <<'PY'
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
pkill -9 ray || true
pkill -9 python || true
for i in "${!REMOTE_HOSTS[@]}"; do
   run_ssh "${REMOTE_HOSTS[$i]}" "${REMOTE_PORTS[$i]}" "pkill -9 sglang || true; ray stop --force || true; pkill -9 ray || true; pkill -9 python || true"
done
sleep 3

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

source "${SCRIPT_DIR}/../../scripts/models/qwen3-14B.sh"
echo "Logging to ${LOG_PATH}"

TENSORBOARD_ARGS=(
   --use-tensorboard
   --tb-project-name kernel_agent
   --tb-experiment-name ${EXP_NAME}
)

LOGGING_ARGS=(
   --log-multi-turn
   --log-memory-to-tensorboard
   --log-timers-to-tensorboard
   --log-throughput
   --log-throughput-to-tensorboard
   --log-progress
   --log-device-memory-used
   --log-straggler
)

CKPT_ARGS=(
   --hf-checkpoint ${HF_MODEL_PATH}
   --ref-load ${MEGATRON_MODEL_PATH}
   # --load /root/Qwen2.5-3B_slime/
   # --save /root/Qwen2.5-3B_slime/
   # --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${RL_DATA}
   --input-key prompt
   --label-key reward_model
   --metadata-key extra_info
   --rollout-shuffle
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 16
   --rollout-max-response-len 8192
   --rollout-max-context-len 32768
   --rollout-temperature 1

   # eval args
   # --eval-interval 25
   # --eval-prompt-data nq_test /root/Search-R1/data/nq_hotpotqa_train/test.parquet@[0:3000]
   # # --eval-prompt-data nq_test /root/nq_search/test.parquet
   # --eval-input-key prompt
   # --eval-label-key reward_model
   # --n-samples-per-eval-prompt 1

   --global-batch-size 256
   --balance-data
)

CURRICULUM_ARGS=(
   --use-dynamic-curriculum
   --difficulty-level-key difficulty_level
   --difficulty-score-key difficulty_score
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
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

RL_ARGS=(
   --advantage-estimator trloo
   --multi-turn-gamma 1.0
   --use-opsm
   --opsm-config '{"aggregation":"turns_geometric","token_veto_threshold":1e-4,"lower":0.999,"upper":1.001}'
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   # whether enabling TIS
   # --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
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
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.5
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

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_cuda_agent.generate
   --custom-rm-path generate_with_cuda_agent.reward_func
   --custom-reward-post-process-path kernel_reward.reward_post_process_by_group
   --dynamic-sampling-filter-path kernel_filter.filter_cuda_kernel_group
   --multi-turn-prompt-config-path "${SCRIPT_DIR}/prompt_config/multi_turn_cuda_kernel.yaml"

   # TIS-related args, recommended to enable when using TIS
   # --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   # --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

KERNEL_AGENT_ARGS=(
   --kernel-env-url ${KERNEL_ENV_URL}
   --kernel-backend cuda_agent
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
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}" \
   ray start --head --node-ip-address ${MASTER_ADDR} --port ${RAY_PORT} --dashboard-port ${RAY_DASHBOARD_PORT} --num-gpus ${GPUS_PER_NODE} --disable-usage-stats

for i in "${!REMOTE_HOSTS[@]}"; do
   run_ssh "${REMOTE_HOSTS[$i]}" "${REMOTE_PORTS[$i]}" \
      "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} ray start --address ${RAY_HEAD_ADDR} --node-ip-address ${REMOTE_HOSTS[$i]} --num-gpus ${GPUS_PER_NODE} --disable-usage-stats"
done

wait_for_cluster

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME}\",
    \"GLOO_SOCKET_IFNAME\": \"${GLOO_SOCKET_IFNAME}\"
  }
}"

ray job submit --address="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes ${NUM_NODES} \
   --actor-num-gpus-per-node ${GPUS_PER_NODE} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${CURRICULUM_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${RL_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${TENSORBOARD_ARGS[@]} \
   ${LOGGING_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${KERNEL_AGENT_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
