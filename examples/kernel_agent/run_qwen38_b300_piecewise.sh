#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$REPO_ROOT/examples/kernel_agent/qwen38_b300_env.sh"
cd "$REPO_ROOT"
source scripts/models/qwen3.5-27B.sh

export HF_MODEL_PATH=${HF_MODEL_PATH:-/nfs/hw-data/ms/FM/checkpoints/Qwen-Zoo/Qwen3.8-27B}
export TRAIN_LOAD_PATH=${TRAIN_LOAD_PATH:-$HF_MODEL_PATH}
export RL_DATA=${RL_DATA:-/nfs/hw-data/ms/FM/lihongbin/dataset/CUDA_RL/cuda_rl/prompt_tvm_GEPA4o_v2/torch_ops_difficulty_lt18.parquet}
export EXP_ROOT=${EXP_ROOT:-$REPO_ROOT/local_artifacts/qwen38_b300_r9}
export NUM_ROLLOUT=${NUM_ROLLOUT:-80}
SAVE_INTERVAL_STEPS=${SAVE_INTERVAL_STEPS:-20}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-8}
export TRAIN_ATTENTION_BACKEND=${TRAIN_ATTENTION_BACKEND:-flash}
RAY_DASHBOARD=${RAY_DASHBOARD:-http://192.168.99.151:8269}
RAY_SUBMISSION_ID=${RAY_SUBMISSION_ID:-qwen38-piecewise-b300-4train4rollout-tp4-cp1-fa4-rc${RECOMPUTE_NUM_LAYERS}-mb16k-rtp1-mem85-m320-opt2-r9}
export KERNEL_ENV_URL=${KERNEL_ENV_URL:-http://192.168.112.55:20111}
export TENSORBOARD_DIR="$EXP_ROOT/tensorboard"
GLOBAL_BATCH_SIZE=128
if [[ ! "$RECOMPUTE_NUM_LAYERS" =~ ^(0|[1-9][0-9]*)$ ]] || (( RECOMPUTE_NUM_LAYERS > 64 )); then
   echo 'RECOMPUTE_NUM_LAYERS must be an integer between 0 and 64' >&2
   exit 1
fi
if [[ "${VALIDATION_NO_SAVE:-0}" != 0 && "${VALIDATION_NO_SAVE:-0}" != 1 ]]; then
   echo 'VALIDATION_NO_SAVE must be 0 or 1' >&2
   exit 1
fi
if [[ "${VALIDATION_NO_SAVE:-0}" == 1 && -z "${LOAD_DEBUG_ROLLOUT_DATA:-}" ]]; then
   echo 'VALIDATION_NO_SAVE requires explicit LOAD_DEBUG_ROLLOUT_DATA replay' >&2
   exit 1
fi
if [[ "${VALIDATION_ROLLOUT_ONLY:-0}" != 0 && "${VALIDATION_ROLLOUT_ONLY:-0}" != 1 ]]; then
   echo 'VALIDATION_ROLLOUT_ONLY must be 0 or 1' >&2
   exit 1
fi
if [[ "${VALIDATION_ROLLOUT_ONLY:-0}" == 1 && -n "${LOAD_DEBUG_ROLLOUT_DATA:-}" ]]; then
   echo 'Rollout-only validation cannot replay saved rollout data' >&2
   exit 1
fi
for value in "$NUM_ROLLOUT" "$ROLLOUT_BATCH_SIZE" "$N_SAMPLES_PER_PROMPT" "$SAVE_INTERVAL_STEPS" "$ROLLOUT_TP_SIZE"; do
   if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
      echo "Expected positive integer, got: $value" >&2
      exit 1
   fi
done
ROLLOUT_SAMPLES=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
if (( ROLLOUT_SAMPLES % GLOBAL_BATCH_SIZE != 0 )); then
   echo 'Rollout samples must be divisible by global batch size 128' >&2
   exit 1
fi
STEPS_PER_ROLLOUT=$((ROLLOUT_SAMPLES / GLOBAL_BATCH_SIZE))
if (( SAVE_INTERVAL_STEPS % STEPS_PER_ROLLOUT != 0 )); then
   echo 'SAVE_INTERVAL_STEPS must be divisible by optimizer steps per rollout' >&2
   exit 1
fi
if (( 4 % ROLLOUT_TP_SIZE != 0 )); then
   echo 'Rollout TP must divide the four rollout GPUs' >&2
   exit 1
fi
SAVE_INTERVAL=$((SAVE_INTERVAL_STEPS / STEPS_PER_ROLLOUT))
export CUDA_AGENT_COVERAGE_REWARD_TYPE=reference_time_coverage
export CUDA_AGENT_COVERAGE_REWARD_WEIGHT=${CUDA_AGENT_COVERAGE_REWARD_WEIGHT:-0.5}
export CUDA_AGENT_USE_REFERENCE_CACHE=1
export CUDA_AGENT_ENABLE_PROFILING=1
export CUDA_AGENT_APPLY_KERNEL_FAILED_SCORE=0
export CUDA_AGENT_APPLY_FAILED_GROUP_REWARD=0
export CUDA_AGENT_OUTPUT_MISMATCH_FAILED_SCORE=0
export CUDA_AGENT_SPEEDUP_SCORE_MODE=legacy
export CUDA_AGENT_PERFORMANCE_REWARD_REQUIRES_CORRECTNESS=1
export CUDA_AGENT_LOG_MULTI_TURN_TEXT=0
export SLIME_SAVE_DEBUG_ROLLOUT_MAX_ID=${SLIME_SAVE_DEBUG_ROLLOUT_MAX_ID:-2}
export PYTHONUNBUFFERED=1
TRAIN_ENV_JSON=$(python - <<'PYENV'
import json, os
print(json.dumps({
    'PYTHONPATH': os.environ['SLIME_TRAIN_PACKAGES'] + ':' + os.environ['PYTHONPATH'],
    'TILELANG_CACHE_DIR': os.environ['B300_RUNTIME'] + '/cache/train_tilelang019',
    'LD_LIBRARY_PATH': os.environ['SLIME_TRAIN_LD_LIBRARY_PATH'],
    'PYTORCH_ALLOC_CONF': 'expandable_segments:True',
    'SGLANG_ENABLE_JIT_DEEPGEMM': '0',
    'SLIME_COMM_MEMORY_CHECK_INTERVAL': '64',
}))
PYENV
)
run_rollout_python() {
   local rollout_pythonpath=${PYTHONPATH:-}
   if [[ -n "${SLIME_ROLLOUT_PYTHONPATH:-}" ]]; then
      rollout_pythonpath="$SLIME_ROLLOUT_PYTHONPATH${rollout_pythonpath:+:$rollout_pythonpath}"
   fi
   PYTHONPATH="$rollout_pythonpath" python "$@"
}

RUNTIME_ENV_JSON=$(python - <<'PY'
import json, os
names = [
    'PYTHONPATH', 'SLIME_REPO', 'SLIME_MEGATRON_LM_PATH', 'B300_RUNTIME', 'SLIME_TRAIN_PACKAGES', 'TMPDIR', 'XDG_CACHE_HOME',
    'TRITON_CACHE_DIR', 'TILELANG_CACHE_DIR', 'TORCH_EXTENSIONS_DIR', 'FLASHINFER_WORKSPACE_BASE',
    'HF_HOME', 'CUDA_CACHE_PATH', 'PYTHONDONTWRITEBYTECODE', 'CUDA_DEVICE_MAX_CONNECTIONS', 'CUDA_HOME',
    'NCCL_SOCKET_IFNAME', 'GLOO_SOCKET_IFNAME', 'NCCL_IB_HCA', 'OMP_NUM_THREADS', 'NO_PROXY', 'no_proxy',
    'SGLANG_CACHE_DIR', 'SGLANG_DG_CACHE_DIR', 'SLIME_ROLLOUT_PYTHONPATH', 'PYTHONUNBUFFERED', 'SLIME_SAVE_DEBUG_ROLLOUT_MAX_ID',
    'TENSORBOARD_DIR', 'NCCL_DEBUG', 'NCCL_DEBUG_SUBSYS', 'SLIME_TRAIN_LD_LIBRARY_PATH',
]
names += [k for k in os.environ if k.startswith('CUDA_AGENT_')]
print(json.dumps({
    'working_dir': os.environ['SLIME_REPO'],
    'excludes': ['.git/', 'Data/', 'local_artifacts/', 'handoffs/', '.claude/', '.github/', 'imgs/', 'docs/', 'tests/', '__pycache__/'],
    'env_vars': {k: os.environ[k] for k in names if k in os.environ},
}))
PY
)

ARGS=(
   "${MODEL_ARGS[@]}"
   --actor-num-nodes 1 --actor-num-gpus-per-node 4 --rollout-num-gpus 4
   --train-env-vars "$TRAIN_ENV_JSON"
   --hf-checkpoint "$HF_MODEL_PATH" --load "$TRAIN_LOAD_PATH"
   --rollout-function-path examples.kernel_agent.fully_async_rollout.generate_rollout_fully_async
   --update-weights-interval 1
   --prompt-data "$RL_DATA" --input-key prompt --label-key reward_model --metadata-key extra_info
   --rollout-shuffle --seed 1234
   --num-rollout "$NUM_ROLLOUT"
   --rollout-batch-size "$ROLLOUT_BATCH_SIZE" --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
   --global-batch-size "$GLOBAL_BATCH_SIZE"
   --rollout-max-context-len 120000 --rollout-max-response-len 32000
   --apply-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"medium"}'
   --rollout-temperature 1.0 --rollout-top-p 0.95 --rollout-top-k -1 --balance-data
   --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --context-parallel-size 1
   --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
   --cp-partition-mode zigzag --sequence-parallel
   --qwen-gdn-backend flashqla --qwen-gdn-implementation distributed
   --qwen-gdn-a2a-implementation fused --qwen-gdn-cache-thd-permutation
   --qwen-gdn-sp-disable-batch-p2p-comm
   --use-dynamic-batch-size --max-tokens-per-gpu 16384
   --log-probs-max-tokens-per-gpu 16384 --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-2048}"
   --advantage-estimator trloo --multi-turn-gamma 1.0
   --enable-mtp-training --mtp-num-layers 1 --mtp-loss-scaling-factor 0.2
   --policy-loss-mode dppo_binary_tv --use-rollout-logprobs
   --eps-clip 0.2 --eps-clip-high 0.2 --eps-clip-c 20
   --enable-fp32-lm-head
   --entropy-coef 0.0 --overlong-penalty None
   --dynamic-reward-gate piecewise
   --difficulty-thresholds 0.3333333333333333 0.6666666666666666
   --dynamic-reward-gate-range 0.8 1.2
   --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.0
   --adam-beta1 0.9 --adam-beta2 0.98 --use-distributed-optimizer
   --overlap-grad-reduce --overlap-param-gather --use-precision-aware-optimizer
   --rollout-num-gpus-per-engine "$ROLLOUT_TP_SIZE" --sglang-dtype bfloat16 --sglang-kv-cache-dtype bfloat16
   --sglang-context-length 120000 --sglang-max-running-requests 64
   --sglang-mem-fraction-static 0.85 --sglang-max-mamba-cache-size 320
   --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND:-trtllm_mha}" --sglang-linear-attn-backend triton --sglang-mamba-backend triton
   --sglang-mamba-radix-cache-strategy extra_buffer --sglang-disable-custom-all-reduce
   --sglang-chunked-prefill-size 4096 --router-policy round_robin
   --sglang-speculative-algorithm NEXTN --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1 --sglang-speculative-num-draft-tokens 4
   --bf16 --attention-backend "$TRAIN_ATTENTION_BACKEND" --attention-dropout 0.0 --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
   --update-weight-buffer-size 1073741824 --no-pin-cpu-grads --no-pin-cpu-params
   --custom-generate-function-path examples.kernel_agent.generate_with_cuda_agent.generate
   --custom-rm-path examples.kernel_agent.generate_with_cuda_agent.reward_func
   --custom-reward-post-process-path examples.kernel_agent.kernel_reward.reward_post_process_by_group
   --dynamic-sampling-filter-path examples.kernel_agent.kernel_filter.filter_cuda_kernel_group
   --multi-turn-prompt-config-path "$REPO_ROOT/examples/kernel_agent/prompt_config/response_prompt/tvm_ffi_short.yaml"
   --kernel-env-url "$KERNEL_ENV_URL" --kernel-backend tvm_ffi --reference-backend torch
   --do-precheck --use-reference-cache --finalize-mode positive --max-turns 1 --enable-turns-dp-partitions
   --use-coverage-rs --coverage-rs-key time_coverage --coverage-rs-threshold 0.3 --coverage-rs-factor 0.1
   --save-debug-rollout-data "$EXP_ROOT/rollout/rollout_{rollout_id}.pt"
   --use-tensorboard --tb-project-name qwen38_b300_r9 --tb-experiment-name "$RAY_SUBMISSION_ID"
   --wandb-always-use-train-step --wandb-centralized
   --log-throughput --log-progress --log-device-memory-used
)

SGLANG_VERSION=$(python - <<'PYVERSION'
import importlib.metadata
print(importlib.metadata.version("sglang"))
PYVERSION
)
case "$SGLANG_VERSION" in
   0.5.20)
      ARGS+=(--sglang-cuda-graph-max-bs-decode 64 --sglang-sampling-mask-max-tokens 32768)
      ;;
   *)
      ARGS+=(--sglang-cuda-graph-max-bs 64)
      ;;
esac

if (( RECOMPUTE_NUM_LAYERS > 0 )); then
   ARGS+=(--recompute-granularity full --recompute-method block --recompute-num-layers "$RECOMPUTE_NUM_LAYERS")
fi
if [[ "${USE_CHECKPOINT_OPT_PARAM_SCHEDULER:-0}" == 1 ]]; then
   ARGS+=(--use-checkpoint-opt-param-scheduler)
fi
if [[ "${ASYNC_SAVE:-0}" == 1 ]]; then
   ARGS+=(--async-save --use-persistent-ckpt-worker)
fi
if [[ "${VALIDATION_NO_SAVE:-0}" != 1 ]]; then
   ARGS+=(--save "$EXP_ROOT/checkpoints" --save-interval "$SAVE_INTERVAL")
fi
if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA:-}" ]]; then
   ARGS+=(--load-debug-rollout-data "$LOAD_DEBUG_ROLLOUT_DATA")
fi
if [[ "${VALIDATION_ROLLOUT_ONLY:-0}" == 1 ]]; then
   ARGS+=(--debug-rollout-only --start-rollout-id "${START_ROLLOUT_ID:-0}")
elif [[ -n "${START_ROLLOUT_ID:-}" ]]; then
   ARGS+=(--start-rollout-id "$START_ROLLOUT_ID")
fi

if [[ "${CONFIG_DRY_RUN:-0}" == 1 ]]; then
   printf '%q ' python "$REPO_ROOT/train_async.py" "${ARGS[@]}"
   printf '\n'
   exit 0
fi
python scripts/check_qwen38_b300_runtime.py
python scripts/check_kernelgym_health.py --url "$KERNEL_ENV_URL" --timeout 5 --attempts 3
run_rollout_python scripts/check_sglang_top_p_replay.py --check-sort-reuse
run_rollout_python scripts/check_sglang_fp32_lm_head_cache.py
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
   exit 0
fi
mkdir -p "$EXP_ROOT/logs" "$EXP_ROOT/rollout" "$EXP_ROOT/provenance"
printf '%q ' python "$REPO_ROOT/train_async.py" "${ARGS[@]}" > "$EXP_ROOT/provenance/command.sh"
printf '%s\n' "$RUNTIME_ENV_JSON" | python -c 'import json,sys; d=json.load(sys.stdin); json.dump(d,sys.stdout,indent=2)' > "$EXP_ROOT/provenance/runtime_env.json"
exec ray job submit --address "$RAY_DASHBOARD" --submission-id "$RAY_SUBMISSION_ID" \
   --runtime-env-json "$RUNTIME_ENV_JSON" -- python "$REPO_ROOT/train_async.py" "${ARGS[@]}"
