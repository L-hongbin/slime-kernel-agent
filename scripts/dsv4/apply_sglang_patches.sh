#!/usr/bin/env bash
# Idempotent applier for slime's sglang-fork modifications (repo convention: sglang
# changes live as VERSIONED PATCH ARTIFACTS here, not naked edits in the fork tree).
# Covers: routed-experts capturer spec-decode fixes, V4 LoRA-serve port
# (get_hidden_dim/name maps/wkv_gate/replicated-wrap), LoRA cuda-graph padding
# tail-fill, dsv4 c4/c128 pad-page pins.
#
# Usage: bash scripts/dsv4/apply_sglang_patches.sh [SGLANG_DIR]
# Run on EVERY node that serves sglang (the fork tree is per-node-local).
set -euo pipefail
SGL=${1:-/sgl-workspace/sglang}
PATCH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/patches/sglang_dsv4_lora_serve_and_capturer.patch"

cd "${SGL}"
if git apply --reverse --check "${PATCH}" >/dev/null 2>&1; then
  echo "sglang patches: already applied (${SGL})"
  exit 0
fi
if git apply --check "${PATCH}" >/dev/null 2>&1; then
  git apply "${PATCH}"
  echo "sglang patches: APPLIED (${SGL})"
  exit 0
fi
echo "sglang patches: NEITHER applied nor cleanly applicable — tree diverged; inspect manually" >&2
exit 1
