source scripts/models/qwen3.5-27B.sh

MODEL_DIR=/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint ${MODEL_DIR} \
    --save ${MODEL_DIR}/torch_dist