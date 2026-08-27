#!/usr/bin/env bash
# Gather a 2-node-sharded Megatron torch_dist checkpoint (this repo's
# dp_reshardable full-async run: 16 ranks = 8 on the head + 8 on .169) onto the
# convert node and convert it to HuggingFace format.
#
# Topology of this run (TP4 x PP2 x CP2 = 16 ranks, DP1):
#   head 10.11.2.164 -> ranks 0-7  (__0..__7_*.distcp) + .metadata + common.pt
#   .169 10.11.2.169 -> ranks 8-15 (__8..__15_*.distcp)
# /nfs/FM is per-node local disk, so shards must be rsynced into one dir on the
# convert node (default .170, the only node with room for the ~470G gather).
#
# Usage:  bash scripts/sync/gather_convert_iter.sh ITER_PADDED   # e.g. 0000039
# Env:    CONVERT_NODE (default 10.11.2.170), KEEP_SCRATCH=1 to keep the gather.
set -Eeo pipefail

ITER=${1:?usage: gather_convert_iter.sh ITER_PADDED (e.g. 0000039)}
REPO=/nfs/FM/chenshuailin/projects/kernel_agents/slime-dev-csl-2
EXP=${EXP:-${REPO}/experiments/FAsync.tvm_ffi.Qwen3.6-27B.CTX16384}
ORIGIN_HF=${ORIGIN_HF:-/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B}
VOCAB=${VOCAB:-248320}
CONVERT_NODE=${CONVERT_NODE:-10.11.2.170}
HEAD_IP=${HEAD_IP:-10.11.2.164}
SHARD_NODES=(${SHARD_NODES:-10.11.2.164 10.11.2.169})   # nodes holding distcp shards
SSH="ssh -p 23422 -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"

SRC="${EXP}/checkpoints/iter_${ITER}"
SCRATCH="${EXP}/_convert/iter_${ITER}"
ITER_SHORT=$((10#${ITER}))                # 0000039 -> 39
HF_OUT="${EXP}/_convert/hf/iter_${ITER_SHORT}"

echo "[gather] ITER=${ITER} -> HF_OUT=${HF_OUT} on ${CONVERT_NODE}"
${SSH} "${CONVERT_NODE}" "mkdir -p ${SCRATCH}"

# Push each shard-holder's local half into the same scratch dir on CONVERT_NODE.
# Each node rsyncs its OWN iter dir, so the union is all 16 .distcp pairs
# (32 files) plus .metadata + common.pt (which live only on the head/rank-0 node).
LOCAL_IPS=" $(hostname -I 2>/dev/null) "
declare -a GATHER_PIDS=()
for node in "${SHARD_NODES[@]}"; do
  echo "[gather] ${node} -> ${CONVERT_NODE}:${SCRATCH}"
  if printf '%s' "${LOCAL_IPS}" | grep -qw "${node}"; then
    # this host holds the shards: push directly
    rsync -a -e "${SSH}" "${SRC}/" "${CONVERT_NODE}:${SCRATCH}/" &
  else
    # remote shard holder: ssh in and push from there
    ${SSH} "${node}" "rsync -a -e '${SSH}' '${SRC}/' '${CONVERT_NODE}:${SCRATCH}/'" &
  fi
  GATHER_PIDS+=($!)
done
wait "${GATHER_PIDS[@]}"

# Verify the gather is complete before converting (32 distcp + metadata + common).
NDIST=$(${SSH} "${CONVERT_NODE}" "ls ${SCRATCH}/*.distcp 2>/dev/null | wc -l")
if [ "${NDIST}" -lt 32 ] || ! ${SSH} "${CONVERT_NODE}" "test -f ${SCRATCH}/.metadata && test -f ${SCRATCH}/common.pt"; then
  echo "[gather] ERROR: incomplete gather (distcp=${NDIST}, need 32 + .metadata + common.pt)" >&2
  exit 1
fi
echo "[gather] complete: ${NDIST} distcp + .metadata + common.pt"

echo "[convert] torch_dist -> HF on ${CONVERT_NODE} (CPU, no_dist load)"
${SSH} "${CONVERT_NODE}" "cd ${REPO} && PYTHONPATH=${REPO}:/root/Megatron-LM python3 tools/convert_torch_dist_to_hf.py \
  --input-dir ${SCRATCH} \
  --output-dir ${HF_OUT} \
  --origin-hf-dir ${ORIGIN_HF} \
  --vocab-size ${VOCAB} \
  -f"

if ! ${SSH} "${CONVERT_NODE}" "test -f ${HF_OUT}/config.json"; then
  echo "[convert] ERROR: HF output missing config.json" >&2
  exit 1
fi
echo "[convert] done -> ${CONVERT_NODE}:${HF_OUT}"
${SSH} "${CONVERT_NODE}" "du -sh ${HF_OUT}"

if [ "${KEEP_SCRATCH:-0}" != "1" ]; then
  echo "[cleanup] removing gather scratch ${SCRATCH} on ${CONVERT_NODE}"
  ${SSH} "${CONVERT_NODE}" "rm -rf ${SCRATCH}"
fi
echo "[done] iter_${ITER_SHORT} HF ready at ${CONVERT_NODE}:${HF_OUT}"
