#!/bin/bash
# DeepSeek-V4-Flash actor-checkpoint conversion: native FP8 HF -> Megatron torch_dist.
#
# Parameterized over the PP x EP topology (world size = NNODES * NPROC_PER_NODE
# must equal PP_SIZE * EP_SIZE). Defaults are the verified PP2 x EP8 = 16 ranks
# on 2 H20 actor nodes (node64/node69; node62 stays reserved for rollout).
#
# The verified PP3 x EP8 = 24-rank profile (3 actor nodes) is:
#   NNODES=3 PP_SIZE=3 FIRST_LAYERS=15 LAST_LAYERS=14 MIN_FREE_GIB=350 \
#   SAVE=.../DeepSeek-V4-Flash-FP8-r2-pp3-ep8-torch_dist MASTER_PORT=29662 ...
#
# Raw HF/native checkpoint is conversion input only; Megatron training loads the
# converted torch_dist checkpoint. Run once per actor node with NODE_RANK set.
set -euo pipefail

REPO=${REPO:-/nfs/FM/chenshuailin/projects/kernel_agents/slime-v4flash-lora}
CHECKPOINT=${CHECKPOINT:-/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8}
SAVE=${SAVE:-/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8-r2-pp2-ep8-torch_dist}

NNODES=${NNODES:-2}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
PP_SIZE=${PP_SIZE:-2}
EP_SIZE=${EP_SIZE:-8}
NUM_LAYERS=${NUM_LAYERS:-43}
FIRST_LAYERS=${FIRST_LAYERS:-21}
LAST_LAYERS=${LAST_LAYERS:-22}
MASTER_PORT=${MASTER_PORT:-29664}
# Local rank-shard estimate before overhead: ~306-320 GiB/node at PP2_EP8,
# ~220 GiB/node at PP3_EP8. Keep a margin.
MIN_FREE_GIB=${MIN_FREE_GIB:-340}
CLUSTER_NO_PROXY=${CLUSTER_NO_PROXY:-127.0.0.1,localhost,10.11.2.164,10.11.2.169,10.11.2.170,10.11.2.162,node64,node69,node70,node62,node64_slime,node69_slime,node70_slime,node62_slime}

: "${MASTER_ADDR:?Set MASTER_ADDR to the rank-0 actor node IP/hostname}"
: "${NODE_RANK:?Set NODE_RANK to this actor node rank}"

if ! [[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || [[ "${NODE_RANK}" -lt 0 || "${NODE_RANK}" -ge "${NNODES}" ]]; then
  echo "Refusing invalid NODE_RANK=${NODE_RANK}; expected integer in [0, ${NNODES})" >&2
  exit 2
fi
if [[ $((NNODES * NPROC_PER_NODE)) -ne $((PP_SIZE * EP_SIZE)) ]]; then
  echo "World size mismatch: NNODES(${NNODES}) * NPROC_PER_NODE(${NPROC_PER_NODE}) != PP_SIZE(${PP_SIZE}) * EP_SIZE(${EP_SIZE})" >&2
  exit 2
fi
if [[ ! -d "${REPO}" ]]; then
  echo "Repo path missing on this node: ${REPO}" >&2
  exit 2
fi
if [[ ! -d "${CHECKPOINT}" || ! -f "${CHECKPOINT}/model.safetensors.index.json" ]]; then
  echo "Native checkpoint missing on this node: ${CHECKPOINT}" >&2
  exit 2
fi
if [[ -e "${SAVE}" ]]; then
  echo "Output already exists, refusing to overwrite: ${SAVE}" >&2
  exit 2
fi

save_parent=$(dirname "${SAVE}")
free_gib=$(df -BG "${save_parent}" | awk 'NR==2 {gsub(/G/, "", $4); print $4}')
if [[ "${free_gib}" -lt "${MIN_FREE_GIB}" ]]; then
  echo "Insufficient free space at ${save_parent}: ${free_gib} GiB < ${MIN_FREE_GIB} GiB" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PATH="/usr/local/cuda/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
export PYTHONPATH="${REPO}:/root/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${no_proxy:-}" ]]; then
  export no_proxy="${no_proxy},${CLUSTER_NO_PROXY},${MASTER_ADDR}"
else
  export no_proxy="${CLUSTER_NO_PROXY},${MASTER_ADDR}"
fi
if [[ -n "${NO_PROXY:-}" ]]; then
  export NO_PROXY="${NO_PROXY},${CLUSTER_NO_PROXY},${MASTER_ADDR}"
else
  export NO_PROXY="${CLUSTER_NO_PROXY},${MASTER_ADDR}"
fi

cd "${REPO}"

echo "V4 PP${PP_SIZE}_EP${EP_SIZE} actor checkpoint conversion"
echo "  node_rank=${NODE_RANK}/${NNODES} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "  checkpoint=${CHECKPOINT}"
echo "  save=${SAVE}"
echo "  free_gib=${free_gib}"

torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m custom_kernels.deepseek_v4.megatron.slice_torch_dist \
  --checkpoint "${CHECKPOINT}" \
  --save "${SAVE}" \
  --num-layers "${NUM_LAYERS}" \
  --pp-size "${PP_SIZE}" \
  --ep-size "${EP_SIZE}" \
  --plan-first-layers "${FIRST_LAYERS}" \
  --plan-last-layers "${LAST_LAYERS}" \
  --master-port "${MASTER_PORT}"

# Always verify what actually landed on disk: a fresh process cold-reads the
# saved torch_dist shards and diffs them against the native checkpoint.
# VERIFY=0 to skip; verify_torch_dist.py also runs standalone against any
# existing checkpoint (no conversion needed).
VERIFY=${VERIFY:-1}
VERIFY_OUTPUT=${VERIFY_OUTPUT:-${SAVE}.verify.txt}
VERIFY_MASTER_PORT=${VERIFY_MASTER_PORT:-$((MASTER_PORT + 1))}
if [[ "${VERIFY}" == "1" ]]; then
  echo "V4 verifying saved shards -> ${VERIFY_OUTPUT}"
  torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${VERIFY_MASTER_PORT}" \
    -m custom_kernels.deepseek_v4.megatron.verify_torch_dist \
    --checkpoint "${CHECKPOINT}" \
    --load "${SAVE}" \
    --output "${VERIFY_OUTPUT}" \
    --num-layers "${NUM_LAYERS}" \
    --pp-size "${PP_SIZE}" \
    --ep-size "${EP_SIZE}" \
    --plan-first-layers "${FIRST_LAYERS}" \
    --plan-last-layers "${LAST_LAYERS}" \
    --master-port "${VERIFY_MASTER_PORT}"
fi
