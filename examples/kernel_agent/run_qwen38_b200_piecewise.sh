#!/usr/bin/env bash
# Run inside node14's csl_slime_032 container, using the existing two-node Ray cluster.
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source /data/ssd1/chenshuailin/b200_runtime/env.sh
export SLIME_REPO="$REPO_ROOT"
export QWEN38_SGLANG_PACKAGES="$B200_RUNTIME/qwen38_top_p_0516"
export PYTHONPATH="$REPO_ROOT:$QWEN38_SGLANG_PACKAGES:$B200_RUNTIME/packages:$SLIME_MEGATRON_LM_PATH"
cd "$REPO_ROOT"
source scripts/models/qwen3.5-27B.sh

export HF_MODEL_PATH=${HF_MODEL_PATH:-/data/ssd1/chenshuailin/checkpoints/Qwen3.8-27B-BF16}
export RL_DATA=${RL_DATA:-$REPO_ROOT/Data/prompt_tvm_GEPA4o_v2/torch_ops_difficulty_lt18.parquet}
export EXP_ROOT=${EXP_ROOT:-/data/ssd1/chenshuailin/experiments/qwen38_gepav4_piecewise_bf16_mtp3_dppotv_fp32_rc56_80rollout_r7}
export NUM_ROLLOUT=${NUM_ROLLOUT:-80}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-56}
RAY_DASHBOARD=${RAY_DASHBOARD:-http://10.1.17.14:8268}
RAY_SUBMISSION_ID=${RAY_SUBMISSION_ID:-qwen38-gepav4-piecewise-bf16-mtp3-dppotv-fp32-rc56-80rollout-r7}
export CUDA_AGENT_COVERAGE_REWARD_TYPE=reference_time_coverage
export CUDA_AGENT_USE_REFERENCE_CACHE=1
export CUDA_AGENT_ENABLE_PROFILING=1
export CUDA_AGENT_APPLY_KERNEL_FAILED_SCORE=0
export CUDA_AGENT_APPLY_FAILED_GROUP_REWARD=0
export CUDA_AGENT_OUTPUT_MISMATCH_FAILED_SCORE=0
export CUDA_AGENT_SPEEDUP_SCORE_MODE=legacy
export CUDA_AGENT_PERFORMANCE_REWARD_REQUIRES_CORRECTNESS=1
export CUDA_AGENT_LOG_MULTI_TURN_TEXT=0
export SLIME_SAVE_DEBUG_ROLLOUT_MAX_ID=0
export PYTHONUNBUFFERED=1
export NCCL_IB_HCA=mlx5_0
export CUDA_HOME=/usr/local/cuda
export WANDB_API_KEY
WANDB_API_KEY=$(< /root/.config/wandb/b200_v42.key)

TRAIN_ENV_JSON=$(python - <<'PY'
import json, os
root, runtime = os.environ['SLIME_REPO'], os.environ['B200_RUNTIME']
print(json.dumps({
    'PYTHONPATH': f'{root}:{runtime}/train_packages:{runtime}/qwen38_top_p_0516:{runtime}/flashqla_pinned_packages:{runtime}/packages:'
                  + os.environ['SLIME_MEGATRON_LM_PATH'],
    'LD_LIBRARY_PATH': f'{runtime}/train_packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/z3/lib:/usr/local/cuda/lib64',
    'CUDNN_HOME': f'{runtime}/train_packages/nvidia/cudnn',
    'TILELANG_CACHE_DIR': f'{runtime}/cache/tilelang019',
    'PYTORCH_ALLOC_CONF': 'expandable_segments:True',
    'SGLANG_ENABLE_JIT_DEEPGEMM': '0',
}))
PY
)
RUNTIME_ENV_JSON=$(python - <<'PY'
import json, os
names = [
    'PYTHONPATH', 'SLIME_REPO', 'SLIME_MEGATRON_LM_PATH', 'B200_RUNTIME', 'QWEN38_SGLANG_PACKAGES', 'TMPDIR', 'XDG_CACHE_HOME',
    'TRITON_CACHE_DIR', 'TILELANG_CACHE_DIR', 'TORCH_EXTENSIONS_DIR', 'FLASHINFER_WORKSPACE_BASE',
    'HF_HOME', 'CUDA_CACHE_PATH', 'PYTHONDONTWRITEBYTECODE', 'CUDA_DEVICE_MAX_CONNECTIONS', 'CUDA_HOME',
    'NCCL_SOCKET_IFNAME', 'GLOO_SOCKET_IFNAME', 'NCCL_IB_HCA', 'OMP_NUM_THREADS', 'NO_PROXY', 'no_proxy',
    'SGLANG_CACHE_DIR', 'SGLANG_DG_CACHE_DIR', 'PYTHONUNBUFFERED', 'SLIME_SAVE_DEBUG_ROLLOUT_MAX_ID',
    'WANDB_API_KEY',
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
   --actor-num-nodes 1 --actor-num-gpus-per-node 8 --rollout-num-gpus 8
   --actor-placement-resource slime_actor --rollout-placement-resource slime_rollout
   --train-env-vars "$TRAIN_ENV_JSON"
   --hf-checkpoint "$HF_MODEL_PATH" --load "$HF_MODEL_PATH"
   --save "$EXP_ROOT/checkpoints" --save-hf "$EXP_ROOT/hf/rollout_{rollout_id}"
   --save-interval "$NUM_ROLLOUT"
   --rollout-function-path examples.kernel_agent.fully_async_rollout.generate_rollout_fully_async
   --update-weights-interval 1
   --prompt-data "$RL_DATA" --input-key prompt --label-key reward_model --metadata-key extra_info
   --rollout-shuffle --seed 1234
   --num-rollout "$NUM_ROLLOUT"
   --rollout-batch-size "$ROLLOUT_BATCH_SIZE" --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
   --global-batch-size 128
   --rollout-max-context-len 120000 --rollout-max-response-len 32000
   --apply-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"medium"}'
   --rollout-temperature 1.0 --rollout-top-p 0.95 --rollout-top-k -1 --balance-data
   --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 --context-parallel-size 4
   --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
   --cp-partition-mode zigzag --sequence-parallel
   --qwen-gdn-backend flashqla --qwen-gdn-implementation distributed
   --qwen-gdn-a2a-implementation fused --qwen-gdn-cache-thd-permutation
   --qwen-gdn-sp-disable-batch-p2p-comm
   --recompute-granularity full --recompute-method block --recompute-num-layers "$RECOMPUTE_NUM_LAYERS"
   --use-dynamic-batch-size --max-tokens-per-gpu 8192
   --log-probs-max-tokens-per-gpu 16384 --log-probs-chunk-size 512
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
   --sglang-context-length 120000 --sglang-max-running-requests 16
   --sglang-mem-fraction-static 0.75 --sglang-cuda-graph-max-bs 16
   --sglang-attention-backend trtllm_mha --sglang-linear-attn-backend triton --sglang-mamba-backend triton
   --sglang-mamba-radix-cache-strategy extra_buffer --sglang-disable-custom-all-reduce
   --sglang-chunked-prefill-size 4096 --router-policy round_robin
   --sglang-speculative-algorithm NEXTN --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1 --sglang-speculative-num-draft-tokens 4
   --bf16 --attention-backend fused --attention-dropout 0.0 --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
   --update-weight-buffer-size 1073741824 --no-pin-cpu-grads --no-pin-cpu-params
   --custom-generate-function-path examples.kernel_agent.generate_with_cuda_agent.generate
   --custom-rm-path examples.kernel_agent.generate_with_cuda_agent.reward_func
   --custom-reward-post-process-path examples.kernel_agent.kernel_reward.reward_post_process_by_group
   --dynamic-sampling-filter-path examples.kernel_agent.kernel_filter.filter_cuda_kernel_group
   --multi-turn-prompt-config-path "$REPO_ROOT/examples/kernel_agent/prompt_config/response_prompt/tvm_ffi_short.yaml"
   --kernel-env-url http://127.0.0.1:20211 --kernel-backend tvm_ffi --reference-backend torch
   --do-precheck --use-reference-cache --finalize-mode positive --max-turns 1 --enable-turns-dp-partitions
   --use-coverage-rs --coverage-rs-key time_coverage --coverage-rs-threshold 0.3 --coverage-rs-factor 0.1
   --save-debug-rollout-data "$EXP_ROOT/rollout/rollout_{rollout_id}.pt"
   --use-wandb --wandb-mode online --wandb-project slime --wandb-dir "$EXP_ROOT/wandb"
   --wandb-group "Qwen38.Piecewise.DPPOTV.FP32Head.BF16.MTP3.GEPAV4Lt18.SingleTurn.120000.32000.80Rollout.8GPU.TP${ROLLOUT_TP_SIZE}Rollout"
   --disable-wandb-random-suffix --wandb-always-use-train-step --wandb-centralized
   --log-throughput --log-progress --log-device-memory-used
)

if [[ "${CONFIG_DRY_RUN:-0}" == 1 ]]; then
   printf '%q ' python "$REPO_ROOT/train_async.py" "${ARGS[@]}"
   printf '\n'
   exit 0
fi
python scripts/patch_megatron_mtp_hidden_detach.py --check \
   --path "$SLIME_MEGATRON_LM_PATH/megatron/core/transformer/multi_token_prediction.py"
python - <<'PY'
import hashlib, inspect, json, os
from pathlib import Path
import sglang
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.logprob_processor import compute_spec_v2_logprobs
assert Path(sglang.__file__).resolve().is_relative_to(Path(os.environ['QWEN38_SGLANG_PACKAGES']).resolve()), 'Expected isolated patched SGLang'
assert 'accept_lens' in inspect.signature(compute_spec_v2_logprobs).parameters, 'Missing speculative top-p replay patch'
assert 'next_token_top_p_token_ids' in LogitsProcessorOutput.__dataclass_fields__, 'Missing top-p replay output field'
root = Path(os.environ['HF_MODEL_PATH'])
gpt_source = Path(os.environ['SLIME_MEGATRON_LM_PATH']) / 'megatron/core/models/gpt/gpt_model.py'
assert 'mtp_output_weight = mtp_output_weight.detach()' in gpt_source.read_text(), 'MTP must not train the shared output layer'
config = json.loads((root / 'config.json').read_text())
assert not config.get('quantization_config'), 'Rollout checkpoint must be unquantized BF16'
index = json.loads((root / 'model.safetensors.index.json').read_text())
assert all((root / name).is_file() for name in set(index['weight_map'].values()))
assert Path(os.environ['RL_DATA']).is_file()
assert hashlib.sha256(Path(os.environ['RL_DATA']).read_bytes()).hexdigest() == 'ca5cd825d33406de2f73245274be63617ebccf8160c46f34ffcafffca8d03f94', 'Expected user-supplied GEPA4o_v2 torch_ops_difficulty_lt18.parquet'
assert not (Path(os.environ['EXP_ROOT']) / 'checkpoints/latest_checkpointed_iteration.txt').exists(), 'Use a fresh experiment directory'
print('Verified unquantized checkpoint, complete weights, dataset, and fresh output directory.')
PY
mkdir -p "$EXP_ROOT/logs" "$EXP_ROOT/rollout" "$EXP_ROOT/provenance"
printf '%q ' python "$REPO_ROOT/train_async.py" "${ARGS[@]}" > "$EXP_ROOT/provenance/command.sh"
printf '%s\n' "$RUNTIME_ENV_JSON" | python -c 'import json,sys; d=json.load(sys.stdin); d["env_vars"]["WANDB_API_KEY"]="<redacted: /root/.config/wandb/b200_v42.key>"; json.dump(d,sys.stdout,indent=2)' > "$EXP_ROOT/provenance/runtime_env.json"
exec ray job submit --address "$RAY_DASHBOARD" --submission-id "$RAY_SUBMISSION_ID" \
   --runtime-env-json "$RUNTIME_ENV_JSON" -- python "$REPO_ROOT/train_async.py" "${ARGS[@]}"
