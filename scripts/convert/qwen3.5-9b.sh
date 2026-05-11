source scripts/models/qwen3.5-9B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.5-9B \
    --save /nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.5-9B/torch_dist