#!/usr/bin/env bash
set -euo pipefail

# Standalone H200 rollout endpoint for predictive_tail_probe.py.  The server
# deliberately reproduces the active r21 DSPARK/LoRA settings, including CUDA
# graphs.  It does not enable the diagnostic router/reduction stabilizers, so
# generation-versus-full-prefix differences remain observable instead of being
# hidden from the online-tail investigation.

PORT=${1:-31003}

export SGLANG_DSV4_FP4_EXPERTS=1
export SGLANG_SHARED_EXPERT_TP1=1
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_USE_TILELANG_MHC_PRE=true
export SGLANG_OPT_USE_TILELANG_MHC_POST=true
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=true
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false
export SGLANG_MEMORY_SAVER_CUDA_GRAPH=true
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=true
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=true
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT=true
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=false

cd /sgl-workspace/sglang
# DP-attention divides the chunked-prefill input by dp-size; 2048 reproduces
# the formal server's effective per-rank chunked_prefill_size=256.
exec python3 -m sglang.launch_server \
  --model-path /ssd/checkpoints/DeepSeek-V4-Flash-DSpark \
  --trust-remote-code \
  --host 0.0.0.0 --port "${PORT}" \
  --context-length 12288 --mem-fraction-static 0.85 \
  --max-running-requests 128 --chunked-prefill-size 2048 \
  --max-prefill-tokens 12288 --page-size 256 \
  --cuda-graph-max-bs 128 \
  --tp-size 8 --dp-size 8 --enable-dp-attention --enable-dp-lm-head \
  --attention-backend dsv4 --kv-cache-dtype fp8_e4m3 \
  --moe-runner-backend flashinfer_mxfp4 --moe-a2a-backend none \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 3 \
  --speculative-moe-runner-backend flashinfer_mxfp4 \
  --enable-return-routed-experts \
  --enable-lora --max-lora-rank 32 --max-loras-per-batch 2 \
  --lora-paths \
    iter74=/ssd/csl_v4_train_rollout_probe_20260723/tail_contraction_20260723/adapter_iter74 \
    iter94=/ssd/csl_v4_train_rollout_probe_20260723/tail_contraction_20260723/adapter_iter94 \
  --lora-target-modules wq_a wkv wq_b wo_b wkv_gate gate_up_proj down_proj \
  --lora-backend triton \
  --disable-custom-all-reduce --disable-flashinfer-autotune \
  --schedule-policy fcfs --random-seed 1234 --watchdog-timeout 2400 \
  --decode-log-interval 40 \
  --skip-server-warmup
