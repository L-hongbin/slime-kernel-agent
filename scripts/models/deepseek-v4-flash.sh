#!/bin/bash

# DeepSeek-V4-Flash model configuration (rollout-only placeholder)
#
# DeepSeek-V4-Flash (DeepseekV4ForCausalLM, model_type deepseek_v4) is a brand-new
# architecture (DeepSeek sparse attention indexer + hash layers + fp8 block-quant
# MoE) that Megatron-LM has NO model spec for. We only ever run it through sglang
# in `--debug-rollout-only` eval, where Megatron is never constructed and these args
# are merely parsed (see slime/backends/megatron_utils/actor.py:53 — init() returns
# immediately, and megatron_validate_args is skipped in arguments.py:1772).
#
# So the values below only need to be argparse-valid; they are NOT used to build a
# model. sglang loads the HF checkpoint directly and auto-detects the fp8 block
# quantization from config.json. Scalars mirror config.json where trivial, for
# readable logs. Do NOT use this file for actual Megatron training of V4.

NLAYERS=43

MODEL_ARGS=(
    --disable-bias-linear
    --num-layers ${NLAYERS}
    --hidden-size 4096
    --ffn-hidden-size 18432
    --num-attention-heads 64
    --normalization RMSNorm
    --position-embedding-type rope
    --norm-epsilon 1e-6
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 129280
    --tokenizer-type HuggingFaceTokenizer
    --bf16
    --rotary-base 10000
    # MoE (256 routed experts, top-6, 1 shared) — parsed only, not built.
    --num-experts 256
    --moe-ffn-hidden-size 2048
    --moe-router-topk 6
    --moe-router-score-function sigmoid
    --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0.0
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
)
