export CUDA_DEVICE_MAX_CONNECTIONS=1
souerce scripts/models/qwen3-14B.sh
PYTHONPATH="/root/Megatron-LM/ \
   python tools/convert_hf_to_torch_dist.py \
   --output-dir ${OUTPUT_DIR} \
   --dtype fp16 \
   --tensor-parallel-size ${TP_SIZE} \
   --pipeline-parallel-size 1 \
   --max-sequence-length 32768