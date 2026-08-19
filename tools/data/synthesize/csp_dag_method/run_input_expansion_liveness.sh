#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 dtype RUN_ROOT SHARD_INDEX SHARD_COUNT CUDA_INDEX [LIMIT]" >&2
  exit 2
fi

stage=$1
run_root=$2
shard_index=$3
shard_count=$4
cuda_index=$5
limit=${6:-}
launcher_sha256=$(sha256sum "$0" | awk '{print $1}')

case "$stage" in
  dtype)
    lane_root="$run_root/dtype"
    module=tools.data.synthesize.dtype_method.validate_dtype_liveness
    ;;
  *)
    echo "unsupported stage: $stage" >&2
    exit 2
    ;;
esac

mkdir -p "$lane_root/runtime_h20"
output="$lane_root/runtime_h20/shard_$(printf '%03d' "$shard_index")_of_$(printf '%03d' "$shard_count").records.jsonl"
args=(
  "$lane_root/parents.parquet"
  "$lane_root/candidates.parquet"
  "$lane_root/manifest.jsonl"
  "$output"
  --device "cuda:$cuda_index"
  --trials 3
  --seed 17
  --timeout-seconds 600
  --launcher-sha256 "$launcher_sha256"
  --shard-index "$shard_index"
  --shard-count "$shard_count"
)
if [[ -n "$limit" ]]; then
  args+=(--limit "$limit")
fi

exec python -m "$module" "${args[@]}"
