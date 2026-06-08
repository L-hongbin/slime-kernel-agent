#!/bin/bash

set -eo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"

MODEL_DIR=${MODEL_DIR:-/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B}
TP=${TP:-4}
PP=${PP:-1}
# Qwen3.6-27B ships an MTP (Multi-Token Prediction) head (mtp_num_hidden_layers=1
# in config.json, 15 mtp.* weights in the HF checkpoint). Without --mtp-num-layers
# the Megatron model is built without the MTP block and those weights are silently
# dropped, producing an incomplete torch_dist checkpoint. Keep this at 1.
MTP_NUM_LAYERS=${MTP_NUM_LAYERS:-1}
SAVE_DIR=${SAVE_DIR:-${MODEL_DIR}/torch_dist_tp${TP}_pp${PP}}
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-/root/Megatron-LM}
NPROC_PER_NODE=${NPROC_PER_NODE:-$((TP * PP))}
MASTER_PORT=${MASTER_PORT:-12355}
FORCE=${FORCE:-0}

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

if [ ! -d "${MODEL_DIR}" ]; then
  echo "MODEL_DIR does not exist: ${MODEL_DIR}" >&2
  exit 1
fi

if [ ! -d "${MEGATRON_LM_PATH}" ]; then
  echo "MEGATRON_LM_PATH does not exist: ${MEGATRON_LM_PATH}" >&2
  exit 1
fi

if [ "${NPROC_PER_NODE}" -ne "$((TP * PP))" ]; then
  echo "NPROC_PER_NODE must equal TP * PP (${TP} * ${PP}); got ${NPROC_PER_NODE}" >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi -L | wc -l)
  if [ "${GPU_COUNT}" -lt "${NPROC_PER_NODE}" ]; then
    echo "Need ${NPROC_PER_NODE} GPUs for TP=${TP}, PP=${PP}; found ${GPU_COUNT}" >&2
    exit 1
  fi
fi

if [ -e "${SAVE_DIR}" ]; then
  if [ "${FORCE}" != "1" ]; then
    echo "SAVE_DIR already exists: ${SAVE_DIR}" >&2
    echo "Set FORCE=1 to remove and regenerate it." >&2
    exit 1
  fi
  rm -rf "${SAVE_DIR}"
fi

mkdir -p "$(dirname -- "${SAVE_DIR}")"
LOG_FILE="${SAVE_DIR}.convert.log"

echo "Converting HF checkpoint to Megatron torch_dist"
echo "  MODEL_DIR=${MODEL_DIR}"
echo "  SAVE_DIR=${SAVE_DIR}"
echo "  TP=${TP}"
echo "  PP=${PP}"
echo "  MTP_NUM_LAYERS=${MTP_NUM_LAYERS}"
echo "  NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "  MEGATRON_LM_PATH=${MEGATRON_LM_PATH}"
echo "  CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}"
echo "  LOG_FILE=${LOG_FILE}"

cd "${REPO_ROOT}"
source scripts/models/qwen3.5-27B.sh

PYTHONPATH="${MEGATRON_LM_PATH}" torchrun \
  --nproc-per-node "${NPROC_PER_NODE}" \
  --master-port "${MASTER_PORT}" \
  tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${MODEL_DIR}" \
  --save "${SAVE_DIR}" \
  --ckpt-format torch_dist \
  --tensor-model-parallel-size "${TP}" \
  --pipeline-model-parallel-size "${PP}" \
  --mtp-num-layers "${MTP_NUM_LAYERS}" \
  --no-save-rng \
  2>&1 | tee "${LOG_FILE}"

echo "Done: ${SAVE_DIR}"
