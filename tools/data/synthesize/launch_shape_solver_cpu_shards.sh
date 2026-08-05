#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
  echo "usage: $0 MACHINE_RANK SHARD_INPUT_ROOT SHARD_RUN_ROOT" >&2
  exit 2
fi

machine_rank=$1
input_root=$2
run_root=$3
machine_count=${SHAPE_CPU_MACHINE_COUNT:-4}
workers=${SHAPE_CPU_WORKERS:-32}
fake_timeout=${SHAPE_FAKE_TIMEOUT_SECONDS:-30}
solver_module=${SHAPE_SOLVER_MODULE:-tools.data.synthesize.solve_multidim_shape_coverage}
expected_solver_contract=${SHAPE_EXPECTED_SOLVER_CONTRACT:-}

for value in "${machine_rank}" "${machine_count}" "${workers}"; do
  [[ ${value} =~ ^[0-9]+$ ]] || {
    echo "machine rank/count and worker count must be integers" >&2
    exit 2
  }
done
(( machine_count > 0 && workers > 0 && machine_rank < machine_count )) || {
  echo "require machine_count/workers > 0 and 0 <= machine_rank < machine_count" >&2
  exit 2
}
[[ -f ${input_root}/shards.json ]] || {
  echo "missing shard manifest: ${input_root}/shards.json" >&2
  exit 2
}
command -v flock >/dev/null 2>&1 || {
  echo "flock is required" >&2
  exit 2
}

mkdir -p "${run_root}"
rank_lock=${run_root}/.machine-rank-${machine_rank}.lock
exec {rank_lock_fd}>"${rank_lock}"
flock -n "${rank_lock_fd}" || {
  echo "another CPU launcher owns machine rank ${machine_rank}: ${rank_lock}" >&2
  exit 2
}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${run_root}/logs"
mapfile -t shard_names < <(
  python - "${input_root}/shards.json" "${machine_rank}" "${machine_count}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
rank = int(sys.argv[2])
machine_count = int(sys.argv[3])
for shard in manifest["shards"]:
    if int(shard["shard_index"]) % machine_count == rank:
        print(shard["name"])
PY
)
(( ${#shard_names[@]} > 0 )) || {
  echo "no shards assigned to machine rank ${machine_rank}" >&2
  exit 2
}

run_shard() {
  local name=$1
  local input_dir=${input_root}/${name}
  local output_dir=${run_root}/${name}
  local log=${run_root}/logs/${name}.log
  if [[ -f ${output_dir}/static/manifest.json ]]; then
    if [[ -n ${expected_solver_contract} ]]; then
      python - "${output_dir}/static/manifest.json" "${expected_solver_contract}" <<'PY'
import json
import sys

observed = json.load(open(sys.argv[1], encoding="utf-8")).get("contract_version")
if observed != sys.argv[2]:
    raise SystemExit(f"solver contract mismatch: expected={sys.argv[2]} observed={observed}")
PY
    fi
    echo "resume: ${name}" >"${log}"
    return 0
  fi
  if [[ -e ${output_dir} ]]; then
    echo "incomplete existing shard output: ${output_dir}" >&2
    return 1
  fi
  python -m "${solver_module}" \
    "${output_dir}" \
    --selected "${input_dir}/selected.parquet" \
    --fake-gate-timeout-seconds "${fake_timeout}" \
    >"${log}" 2>&1
  if [[ -n ${expected_solver_contract} ]]; then
    python - "${output_dir}/static/manifest.json" "${expected_solver_contract}" <<'PY'
import json
import sys

observed = json.load(open(sys.argv[1], encoding="utf-8")).get("contract_version")
if observed != sys.argv[2]:
    raise SystemExit(f"solver contract mismatch: expected={sys.argv[2]} observed={observed}")
PY
  fi
}
export -f run_shard
export input_root run_root fake_timeout solver_module expected_solver_contract

printf '%s\n' "${shard_names[@]}" | xargs -r -n1 -P "${workers}" bash -c 'run_shard "$1"' _

missing=0
for name in "${shard_names[@]}"; do
  if [[ ! -f ${run_root}/${name}/static/manifest.json ]]; then
    echo "missing completed shard manifest: ${name}" >&2
    missing=$((missing + 1))
  fi
done
(( missing == 0 )) || exit 1
printf '{"machine_rank":%s,"machine_count":%s,"workers":%s,"completed_shards":%s}\n' \
  "${machine_rank}" "${machine_count}" "${workers}" "${#shard_names[@]}"
