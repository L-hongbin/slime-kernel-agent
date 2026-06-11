#!/usr/bin/env bash
# Gather a 4-node-sharded Megatron torch_dist checkpoint onto THIS node and
# convert it to HuggingFace format. The run 20260610_142800 saved a
# fully-reshardable checkpoint whose 32 ranks (64 .distcp files) + .metadata +
# common.pt are split across 4 nodes; HF conversion needs them all in one dir.
#
# Run this ON the convert node (e.g. node70). It hardlinks this node's local
# shards (free) and rsyncs the rest from the other source nodes.
#
# Usage:  bash gather_convert_ckpt.sh ITER_PADDED      # e.g. 0000049
# Env:    KEEP_SCRATCH=1 to keep the gathered torch_dist after conversion.
set -eo pipefail

ITER=${1:?usage: gather_convert_ckpt.sh ITER_PADDED (e.g. 0000049)}
REPO=/nfs/FM/chenshuailin/projects/kernel_agents/slime
RUN=${RUN:-${REPO}/checkpoints/Qwen3.6-27B/20260610_142800.t1.27B.bf16.TP4.PP2.CP2.tis.eagle.colocate.offload.ctx16384.gradf32.H20}
ORIGIN_HF=${ORIGIN_HF:-/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B}
VOCAB=${VOCAB:-248320}
SSH="ssh -p 23422 -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"

SRC_DIR="${RUN}/iter_${ITER}"           # this node's local shards
SCRATCH="${RUN}/_convert_scratch/iter_${ITER}"
ITER_SHORT=$((10#${ITER}))              # 0000049 -> 49
HF_OUT="${RUN}/hf/iter_${ITER_SHORT}"

# The full checkpoint's 4 shard-holders. This node contributes its own shards via
# the local hardlink above; pull the rest from the other three (auto-exclude self).
# node62=ranks0-7 (+ .metadata/common.pt/metadata.json), node64=8-15, node69=16-23, node70=24-31.
ALL_NODES=("10.11.2.162" "10.11.2.164" "10.11.2.169" "10.11.2.170")
mapfile -t LOCAL_IPS < <(hostname -I 2>/dev/null | tr ' ' '\n' | grep -v '^$')
REMOTES=()
for n in "${ALL_NODES[@]}"; do
  skip=0
  for lip in "${LOCAL_IPS[@]}"; do [ "$n" = "$lip" ] && skip=1; done
  [ "$skip" = 0 ] && REMOTES+=("$n")
done
echo "[gather] local IPs: ${LOCAL_IPS[*]}"
echo "[gather] pulling from remotes: ${REMOTES[*]}"
if [ "${#REMOTES[@]}" -ne 3 ]; then
  echo "[gather] ERROR: expected 3 remote shard-holders, got ${#REMOTES[@]} (local not in ALL_NODES?)" >&2
  exit 1
fi

echo "[gather] ITER=${ITER} -> HF_OUT=${HF_OUT}"
echo "[gather] scratch=${SCRATCH}"
mkdir -p "${SCRATCH}"

# 1) hardlink local shards (instant, no extra space; same filesystem)
echo "[gather] hardlinking local shards from ${SRC_DIR}"
cp -aln "${SRC_DIR}"/. "${SCRATCH}"/

# 2) pull the other nodes' shards (+ node62 metadata/common.pt) into scratch
for r in "${REMOTES[@]}"; do
  echo "[gather] rsync from ${r}:${SRC_DIR}/"
  rsync -aW --info=stats1 -e "${SSH}" "root@${r}:${SRC_DIR}/" "${SCRATCH}/"
done

# 3b) recover a missing .metadata from a sibling complete checkpoint.
# A torch_dist .metadata is a deterministic layout index (tensor -> file/offset/
# length + global shape). For the SAME model/parallelism/world_size/save-code it
# is identical across iterations (only the tensor VALUES differ). So if this save
# was never finalized (.metadata absent) we can borrow it from METADATA_FALLBACK_ITER
# ONLY AFTER confirming every .distcp file size matches that donor iteration.
if [ ! -f "${SCRATCH}/.metadata" ] && [ -n "${METADATA_FALLBACK_ITER:-}" ]; then
  DONOR="${RUN}/iter_${METADATA_FALLBACK_ITER}"
  echo "[meta-recover] .metadata missing; validating donor ${DONOR} (.distcp sizes must match exactly)"
  # donor .metadata + metadata.json live on node62 (rank0 holder)
  META_NODE=${META_NODE:-10.11.2.162}
  # size-match check: compare each scratch shard against donor on META_NODE.
  # NOTE: run the loop in the main shell via here-string (no pipeline subshell)
  # and never end an iteration on a failing test, or set -e/pipefail would kill
  # the script on the all-match (good) path.
  donor_sizes=$(${SSH} "root@${META_NODE}" "cd '${DONOR}' 2>/dev/null && for f in *.distcp; do stat -c '%n %s' \"\$f\"; done") || {
    echo "[meta-recover] ABORT: could not read donor sizes from ${META_NODE}" >&2; exit 1; }
  mismatch=""
  while read -r b sz; do
    [ -z "${b}" ] && continue
    local_sz=$(stat -c%s "${SCRATCH}/${b}" 2>/dev/null || echo -1)
    if [ "${local_sz}" != "${sz}" ]; then
      mismatch="${mismatch} ${b}(donor=${sz},local=${local_sz})"
    fi
  done <<< "${donor_sizes}"
  if [ -n "${mismatch}" ]; then
    echo "[meta-recover] ABORT: shard sizes differ from donor; .metadata NOT transferable" >&2
    echo "${mismatch}" >&2
    exit 1
  fi
  echo "[meta-recover] all shard sizes match donor; copying .metadata + metadata.json"
  rsync -a -e "${SSH}" "root@${META_NODE}:${DONOR}/.metadata" "${SCRATCH}/.metadata"
  rsync -a -e "${SSH}" "root@${META_NODE}:${DONOR}/metadata.json" "${SCRATCH}/metadata.json" 2>/dev/null || true
fi

# 3) verify completeness
ndist=$(ls "${SCRATCH}"/*.distcp 2>/dev/null | wc -l)
echo "[gather] distcp files = ${ndist} (expect 64); .metadata=$(test -f "${SCRATCH}/.metadata" && echo YES || echo NO); common.pt=$(test -f "${SCRATCH}/common.pt" && echo YES || echo NO)"
if [ "${ndist}" -ne 64 ] || [ ! -f "${SCRATCH}/.metadata" ] || [ ! -f "${SCRATCH}/common.pt" ]; then
  echo "[gather] ERROR: incomplete gather; aborting before convert" >&2
  exit 1
fi

# 4) convert torch_dist -> HF (model weights only; optimizer keys are filtered)
# CONVERTER=parallel (default, fast, multiproc) | single (convert_torch_dist_to_hf.py,
# simpler/serial — use to cross-check a suspected parallel-converter bug).
CONVERTER=${CONVERTER:-parallel}
if [ "${CONVERTER}" = "single" ]; then
  CONV_TOOL=tools/convert_torch_dist_to_hf.py
else
  CONV_TOOL=tools/convert_torch_dist_to_hf_parallel.py
fi
echo "[convert] (${CONVERTER}: ${CONV_TOOL}) -> ${HF_OUT}"
cd "${REPO}"
PYTHONPATH="${REPO}:/root/Megatron-LM" python3 "${CONV_TOOL}" \
  --input-dir "${SCRATCH}" \
  --output-dir "${HF_OUT}" \
  --origin-hf-dir "${ORIGIN_HF}" \
  --vocab-size "${VOCAB}" \
  -f

# 5) verify HF output
echo "[verify] HF output:"
ls "${HF_OUT}"/config.json "${HF_OUT}"/model.safetensors.index.json >/dev/null 2>&1 \
  && echo "  config.json + index OK" || { echo "  ERROR: HF output incomplete" >&2; exit 1; }
nshard=$(ls "${HF_OUT}"/*.safetensors 2>/dev/null | wc -l)
nmtp=$(python3 -c "import json;m=json.load(open('${HF_OUT}/model.safetensors.index.json'))['weight_map'];print(sum(1 for k in m if 'mtp' in k.lower() or 'nextn' in k.lower()))" 2>/dev/null || echo '?')
echo "  safetensors shards=${nshard}  mtp_weights=${nmtp}"

# 6) cleanup gathered torch_dist (keeps original per-node shards intact)
if [ "${KEEP_SCRATCH:-0}" = "1" ]; then
  echo "[done] KEEP_SCRATCH=1, leaving ${SCRATCH}"
else
  echo "[cleanup] removing scratch ${SCRATCH}"
  rm -rf "${SCRATCH}"
fi
echo "[done] iter_${ITER} -> ${HF_OUT}"
