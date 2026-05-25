#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/nfs/FM/lihongbin/CODE/ms-swift/output/Qwen3.6-27B-32k-sft-full-stage2/v0-20260503-141128/checkpoint-681/}
# /nfs/FM/lihongbin/CODE/ms-swift/output/Qwen3.6-27B-32k-sft-full-stage2/v0-20260503-141128/checkpoint-681/
# /nfs/FM/checkpoints/Qwen/Qwen3.6-27B/
SAMPLE_PATH=${SAMPLE_PATH:-/nfs/FM/lihongbin/datasets/CUDA_RL/SFT/prompt_v4/parallel_drkernel_minimax_results_sft.parquet}
# /nfs/FM/lihongbin/datasets/CUDA_RL/SFT/prompt_v4/parallel_drkernel_minimax_results_sft.parquet
# /nfs/FM/lihongbin/datasets/CUDA_RL/RL_Data/prompt_v4/drkernel_rl_thinking.parquet
# "/nfs/FM/lihongbin/datasets/CUDA_RL/RL_Data/prompt_tvm/drkernel_rl_thinking.parquet"
SAMPLE_INDEX=${SAMPLE_INDEX:-1}
PROMPT_KEY=${PROMPT_KEY:-'messages'}
GROUND_TRUTH_KEY=${GROUND_TRUTH_KEY:-reward_model.ground_truth}
ENTRY_POINT_KEY=${ENTRY_POINT_KEY:-extra_info.entry_point}
UUID_KEY=${UUID_KEY:-extra_info.uuid}
KERNEL_ENV_URL=${KERNEL_ENV_URL:-http://192.168.16.21:20111}
# http://192.168.207.229:8002
# http://192.168.16.21:20111
KERNEL_BACKEND=${KERNEL_BACKEND:-cuda_agent}
# tvm_ffi
# cuda_agent

ROLLOUT_HOST=${ROLLOUT_HOST:-127.0.0.1}
ROLLOUT_PORT=${ROLLOUT_PORT:-30000}
TP_SIZE=${TP_SIZE:-1}
MAX_TURNS=${MAX_TURNS:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-32768}
MAX_FEEDBACK_CHARS=${MAX_FEEDBACK_CHARS:-0}

KERNEL_EVAL_CLIENT_TIMEOUT=${KERNEL_EVAL_CLIENT_TIMEOUT:-300}
KERNEL_EVAL_TASK_TIMEOUT=${KERNEL_EVAL_TASK_TIMEOUT:-120}
KERNEL_EVAL_HEARTBEAT_INTERVAL=${KERNEL_EVAL_HEARTBEAT_INTERVAL:-5}
NUM_CORRECT_TRIALS=${NUM_CORRECT_TRIALS:-1}
NUM_PERF_TRIALS=${NUM_PERF_TRIALS:-1}
USE_MULTI_TURN=${USE_MULTI_TURN:-1}
PADDING_TURNS=${PADDING_TURNS:-1}
ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR:-trloo}
MULTI_TURN_GAMMA=${MULTI_TURN_GAMMA:-1.0}
FINALIZE_MODE=${FINALIZE_MODE:-positive}
USE_COVERAGE_RS=${USE_COVERAGE_RS:-0}
KEEP_HISTORY_THINKING=${KEEP_HISTORY_THINKING:-1}
DO_PRECHECK=${DO_PRECHECK:-1}

LOG_DIR=${LOG_DIR:-examples/kernel_agent/test/log}
mkdir -p "${LOG_DIR}"
LOG_FILE=${LOG_FILE:-${LOG_DIR}/run_generate_smoke_real_sample_$(date +%Y%m%d_%H%M%S)_sample${SAMPLE_INDEX}.log}
echo "[cuda_agent][generate_smoke] logging to ${LOG_FILE}"

python examples/kernel_agent/test/run_generate_smoke.py \
  --start-rollout \
  --real-env \
  --model-path "${MODEL_PATH}" \
  --hf-checkpoint "${MODEL_PATH}" \
  --sample-path "${SAMPLE_PATH}" \
  --sample-index "${SAMPLE_INDEX}" \
  --prompt-key "${PROMPT_KEY}" \
  --ground-truth-key "${GROUND_TRUTH_KEY}" \
  --entry-point-key "${ENTRY_POINT_KEY}" \
  --uuid-key "${UUID_KEY}" \
  --kernel-env-url "${KERNEL_ENV_URL}" \
  --kernel-backend "${KERNEL_BACKEND}" \
  --rollout-host "${ROLLOUT_HOST}" \
  --rollout-port "${ROLLOUT_PORT}" \
  --max-turns "${MAX_TURNS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}" \
  --max-feedback-chars "${MAX_FEEDBACK_CHARS}" \
  --kernel-eval-client-timeout "${KERNEL_EVAL_CLIENT_TIMEOUT}" \
  --kernel-eval-task-timeout "${KERNEL_EVAL_TASK_TIMEOUT}" \
  --kernel-eval-heartbeat-interval "${KERNEL_EVAL_HEARTBEAT_INTERVAL}" \
  --num-correct-trials "${NUM_CORRECT_TRIALS}" \
  --num-perf-trials "${NUM_PERF_TRIALS}" \
  --advantage-estimator "${ADVANTAGE_ESTIMATOR}" \
  --multi-turn-gamma "${MULTI_TURN_GAMMA}" \
  --finalize-mode "${FINALIZE_MODE}" \
  --log-level INFO \
  $([[ "${USE_MULTI_TURN}" == "1" ]] && echo --use-multi-turn || echo --no-use-multi-turn) \
  $([[ "${PADDING_TURNS}" == "1" ]] && echo --padding-turns || echo --no-padding-turns) \
  $([[ "${USE_COVERAGE_RS}" == "1" ]] && echo --use-coverage-rs || echo --no-use-coverage-rs) \
  $([[ "${KEEP_HISTORY_THINKING}" == "1" ]] && echo --keep-history-thinking || echo --no-keep-history-thinking) \
  $([[ "${DO_PRECHECK}" == "1" ]] && echo --do-precheck) \
  --rollout-extra-args \
  --tp-size "${TP_SIZE}" \
  --trust-remote-code 2>&1 | tee "${LOG_FILE}"
