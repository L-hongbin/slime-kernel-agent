#!/usr/bin/env bash
# Run inside csl_slime_qwen38_b300_r9_sgl0520 after successful final HF export.
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
LEVEL=${LEVEL:-${1:-1}}
case "$LEVEL" in 1|2|3) ;; *) echo 'LEVEL must be 1, 2, or 3' >&2; exit 2;; esac
export EXP_ROOT="$REPO_ROOT/experiments/qwen38_b300_baseline_t1"
SUITE_ROOT="$EXP_ROOT/eval_kernelbench_gepav2"
export EVAL_ROOT="$SUITE_ROOT/level$LEVEL"
export HF_MODEL_PATH=${HF_MODEL_PATH:-$EXP_ROOT/hf/iter_0000079}
EVAL_DATA="$SUITE_ROOT/data/kernelbench_level${LEVEL}_val.parquet"
DUMP_DIR="$EVAL_ROOT/dumps"
KERNEL_ENV_URL=http://192.168.112.55:20111
RAY_DASHBOARD=${RAY_DASHBOARD:-http://127.0.0.3:8271}
RAY_SUBMISSION_ID=${RAY_SUBMISSION_ID:-qwen38-timecov-final-kb-l${LEVEL}-gepav2}
export LEVEL
export SLIME_TRAIN_PACKAGES="$REPO_ROOT/local_artifacts/qwen38_b300_r9/runtime/train_packages_sgl0520"
export SLIME_ROLLOUT_PYTHONPATH="$REPO_ROOT/local_artifacts/qwen38_b300_r9/runtime/rollout_sgl0520_packages"
source "$REPO_ROOT/examples/kernel_agent/qwen38_b300_env.sh"
# The container's SGLang is 0.5.20; keep Torch/TE train ABI in the driver.
# Engine actors prepend SLIME_ROLLOUT_PYTHONPATH independently.
export PYTHONPATH="$SLIME_TRAIN_PACKAGES:$PYTHONPATH"
cd "$REPO_ROOT"
source scripts/models/qwen3.5-27B.sh
export CUDA_AGENT_NUM_CORRECT_TRIALS=5
export CUDA_AGENT_NUM_WARMUP=10
export CUDA_AGENT_NUM_PERF_TRIALS=100
export CUDA_AGENT_REFER_NUM_PERF_TRIALS=150
export CUDA_AGENT_ADAPTIVE_PERF_TRIALS=0
export CUDA_AGENT_USE_REFERENCE_CACHE=1
export CUDA_AGENT_ENABLE_NCU=0
export CUDA_AGENT_ENABLE_COMPUTE_SANITIZER=0
export CUDA_AGENT_ENABLE_CORRECTNESS_INPUT_PERTURBATIONS=0
export CUDA_AGENT_VERBOSE_ERRORS=0
export CUDA_AGENT_ENABLE_PROFILING=1
export CUDA_AGENT_LOG_MULTI_TURN_TEXT=0
export CUDA_AGENT_LOG_ROLLOUT_STATS_ONLY=1
export CUDA_AGENT_OUTPUT_MISMATCH_FAILED_SCORE=0
export CUDA_AGENT_APPLY_KERNEL_FAILED_SCORE=0
export CUDA_AGENT_APPLY_FAILED_GROUP_REWARD=0
export PYTHONUNBUFFERED=1
ARGS=(
  "${MODEL_ARGS[@]}"
  --actor-num-nodes 1 --actor-num-gpus-per-node 8
  --rollout-num-gpus 8 --rollout-num-gpus-per-engine 1
  --hf-checkpoint "$HF_MODEL_PATH"
  --num-rollout 0 --start-rollout-id 0 --eval-interval 1 --debug-rollout-only
  --eval-prompt-data "kb_l${LEVEL}_val" "$EVAL_DATA"
  --eval-input-key prompt --eval-label-key reward_model --n-samples-per-eval-prompt 8
  --prompt-data "$EVAL_DATA" --input-key prompt --label-key reward_model --metadata-key extra_info
  --rollout-batch-size 16 --n-samples-per-prompt 1 --seed 1234
  --rollout-max-context-len 80000 --rollout-max-response-len 60000
  --eval-max-context-len 80000 --eval-max-response-len 60000
  --apply-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"medium"}'
  --eval-temperature 0.7 --eval-top-p 0.7 --eval-top-k -1
  --custom-generate-function-path examples.kernel_agent.generate_with_cuda_agent.generate
  --custom-rm-path examples.kernel_agent.generate_with_cuda_agent.reward_func
  --multi-turn-prompt-config-path "$REPO_ROOT/examples/kernel_agent/prompt_config/response_prompt/tvm_ffi_gepa_kimi_v2.jinja"
  --kernel-env-url "$KERNEL_ENV_URL" --kernel-backend tvm_ffi --reference-backend torch
  --do-precheck --use-reference-cache --use-multi-turn --finalize-mode none --max-turns 1
  --dump-details "$DUMP_DIR"
  --sglang-dtype bfloat16 --sglang-kv-cache-dtype bfloat16 --sglang-context-length 80000
  --sglang-max-running-requests 64 --sglang-server-concurrency 64 --sglang-mem-fraction-static 0.85
  --sglang-max-mamba-cache-size 320 --sglang-attention-backend trtllm_mha
  --sglang-linear-attn-backend triton --sglang-mamba-backend triton
  --sglang-mamba-radix-cache-strategy extra_buffer --sglang-disable-custom-all-reduce
  --sglang-chunked-prefill-size 4096 --sglang-decode-log-interval 400 --router-policy round_robin
  --router-queue-timeout-secs 2400 --sglang-watchdog-timeout 2400
  --sglang-speculative-algorithm EAGLE --sglang-speculative-num-steps 3
  --sglang-speculative-eagle-topk 1 --sglang-speculative-num-draft-tokens 4
  --sglang-cuda-graph-max-bs-decode 64 --sglang-sampling-mask-max-tokens 32768
  --enable-fp32-lm-head --attention-dropout 0.0 --hidden-dropout 0.0 --attention-backend flash --bf16
)
if [[ ${CONFIG_DRY_RUN:-0} == 1 ]]; then
  printf '%q ' python "$REPO_ROOT/train.py" "${ARGS[@]}"
  printf '\n'
  exit 0
fi
[[ -f "$SUITE_ROOT/provenance/training_complete.json" ]]
[[ -f "$SUITE_ROOT/provenance/hf_complete.json" ]]
if [[ -f "$EVAL_ROOT/validated_summary.json" ]]; then
  echo "Level $LEVEL already has validated results; skipping."
  exit 0
fi
mkdir -p "$DUMP_DIR" "$EVAL_ROOT/logs" "$EVAL_ROOT/provenance"
exec 9>"$EVAL_ROOT/.launch.lock"
flock -n 9 || { echo "Level $LEVEL launcher already running" >&2; exit 1; }
python scripts/qwen38_timecov_eval_artifacts.py audit-data
python scripts/qwen38_timecov_eval_artifacts.py validate-hf --path "$HF_MODEL_PATH"
python - <<'PY'
import sglang, torch
assert sglang.__version__ == '0.5.20'
assert torch.cuda.is_available() and torch.cuda.device_count() == 8
print('Verified SGLang 0.5.20 with eight visible CUDA GPUs')
PY
python scripts/check_kernelgym_health.py --url "$KERNEL_ENV_URL" --timeout 5 --attempts 3
PYTHONPATH="$SLIME_ROLLOUT_PYTHONPATH:$PYTHONPATH" python scripts/check_sglang_top_p_replay.py --check-sort-reuse
RUNTIME_ENV_JSON=$(python - <<'PY'
import json,os
names=['PYTHONPATH','SLIME_REPO','SLIME_MEGATRON_LM_PATH','B300_RUNTIME','SLIME_TRAIN_PACKAGES',
       'TMPDIR','XDG_CACHE_HOME','TRITON_CACHE_DIR','TILELANG_CACHE_DIR','TORCH_EXTENSIONS_DIR',
       'FLASHINFER_WORKSPACE_BASE','HF_HOME','CUDA_CACHE_PATH','PYTHONDONTWRITEBYTECODE',
       'CUDA_DEVICE_MAX_CONNECTIONS','CUDA_HOME','NCCL_SOCKET_IFNAME','GLOO_SOCKET_IFNAME','NCCL_IB_HCA',
       'OMP_NUM_THREADS','NO_PROXY','no_proxy','SGLANG_CACHE_DIR','SGLANG_DG_CACHE_DIR',
       'SLIME_ROLLOUT_PYTHONPATH','PYTHONUNBUFFERED','NCCL_DEBUG','NCCL_DEBUG_SUBSYS']
names += [k for k in os.environ if k.startswith('CUDA_AGENT_')]
print(json.dumps({'working_dir':os.environ['SLIME_REPO'],
 'excludes':['.git/','Data/','local_artifacts/','experiments/','handoffs/','.claude/','.github/','imgs/','docs/','tests/','__pycache__/'],
 'env_vars':{k:os.environ[k] for k in names if k in os.environ}}))
PY
)
printf '%s\n' "$RUNTIME_ENV_JSON" > "$EVAL_ROOT/provenance/runtime_env.json"
printf '%q ' python "$REPO_ROOT/train.py" "${ARGS[@]}" > "$EVAL_ROOT/provenance/command.sh"
printf '\n' >> "$EVAL_ROOT/provenance/command.sh"
python - <<'PY' > "$EVAL_ROOT/provenance/config.json"
import json,os
level=int(os.environ['LEVEL']); rows={1:100,2:100,3:50}[level]
print(json.dumps({'level':level,'dataset_rows':rows,'candidates_per_prompt':8,'expected_trajectories':rows*8,
 'initial_prompt_source':'KernelBench-TVMFFI-GEPA-V2 (complete source prompt messages)',
 'response_prompt':'tvm_ffi_gepa_kimi_v2.jinja','hf_model':os.environ['HF_MODEL_PATH'],
 'max_turns':1,'context':80000,'response':60000,'temperature':0.7,'top_p':0.7,
 'rollout_gpus':8,'rollout_tp':1,'kernelgym_warmup':10,'kernel_perf_trials':100,'reference_perf_trials':150,
 'ncu':False,'sanitizer':False},indent=2))
PY
ray job submit --address "$RAY_DASHBOARD" --submission-id "$RAY_SUBMISSION_ID" \
  --entrypoint-resources '{"node:127.0.0.3":0.001}' \
  --runtime-env-json "$RUNTIME_ENV_JSON" -- python "$REPO_ROOT/train.py" "${ARGS[@]}"
python scripts/qwen38_timecov_eval_artifacts.py check-results --level "$LEVEL"
python examples/kernel_agent/eval/summarize_eval.py "$EVAL_ROOT" --max-turns 1 | tee "$EVAL_ROOT/summary.txt"
