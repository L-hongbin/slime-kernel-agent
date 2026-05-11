#!/bin/bash

set -exo pipefail

MODEL_DIR=/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.5-9B
SAVE_DIR="checkpoints/${MODEL_DIR##*/}"
LOG_DIR="${SAVE_DIR}"
LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${LOG_DIR}"
touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Logging to ${LOG_FILE}"

dump_ray_logs() {
    local ray_log_dir="/tmp/ray/session_latest/logs"
    local log_file

    echo "ray start failed; dumping Ray logs from ${ray_log_dir}"
    if [[ ! -d "${ray_log_dir}" ]]; then
        echo "Ray log directory does not exist: ${ray_log_dir}"
        return
    fi

    for log_file in \
        "${ray_log_dir}/gcs_server.out" \
        "${ray_log_dir}/gcs_server.err" \
        "${ray_log_dir}/raylet.out" \
        "${ray_log_dir}/raylet.err" \
        "${ray_log_dir}/dashboard.log" \
        "${ray_log_dir}/dashboard.err" \
        "${ray_log_dir}/dashboard_agent.log" \
        "${ray_log_dir}/dashboard_agent.err"; do
        if [[ -f "${log_file}" ]]; then
            echo "===== ${log_file} ====="
            tail -n 200 "${log_file}" || true
        fi
    done
}

# for rerun the task
pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3
pkill -9 ray || true
pkill -9 python || true

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

if command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
    DETECTED_GPUS=0
fi
NUM_GPUS=${NUM_GPUS:-${DETECTED_GPUS}}
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -le 0 ]; then
    NUM_GPUS=8
fi
echo "NUM_GPUS: $NUM_GPUS"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3.5-9B.sh"

TP=2

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}
   --ref-load ${MODEL_DIR}/torch_dist
   --save ${SAVE_DIR}/
   --load ${SAVE_DIR}/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ./data/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

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
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
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
   # --wandb-group qwen3-4B-test
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${TP}
   --sglang-mem-fraction-static 0.7
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

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s:-120}
NO_PROXY_EXTRA="localhost,127.0.0.1,${MASTER_ADDR}"
export no_proxy="${NO_PROXY_EXTRA}${no_proxy:+,${no_proxy}}"
export NO_PROXY="${NO_PROXY_EXTRA}${NO_PROXY:+,${NO_PROXY}}"

if ! ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265; then
    dump_ray_logs
    exit 1
fi

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
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
   ${MISC_ARGS[@]}
