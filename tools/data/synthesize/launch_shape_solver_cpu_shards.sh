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
expected_solver_generator=${SHAPE_EXPECTED_SOLVER_GENERATOR:-}
expected_solver_source_sha256=${SHAPE_EXPECTED_SOLVER_SOURCE_SHA256:-}
expected_multidim_source_sha256=${SHAPE_EXPECTED_MULTIDIM_SOURCE_SHA256:-}
expected_shape_helper_source_sha256=${SHAPE_EXPECTED_SHAPE_HELPER_SOURCE_SHA256:-}
expected_group_scope=${SHAPE_EXPECTED_GROUP_SCOPE:-}
supports_group_scope=${SHAPE_SOLVER_SUPPORTS_GROUP_SCOPE:-0}
group_scope=

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
case ${supports_group_scope} in
  0) ;;
  1) group_scope=${SHAPE_GROUP_SCOPE:-generic} ;;
  *)
    echo "SHAPE_SOLVER_SUPPORTS_GROUP_SCOPE must be 0 or 1" >&2
    exit 2
    ;;
esac
if [[ -n ${expected_group_scope} && ${supports_group_scope} != 1 ]]; then
  echo "SHAPE_EXPECTED_GROUP_SCOPE requires a scope-capable solver" >&2
  exit 2
fi
if [[ ${supports_group_scope} == 1 ]]; then
  case ${group_scope} in
    generic|nonleading_no_explicit_batch|balanced_nonleading_no_explicit_batch) ;;
    *)
      echo "unknown SHAPE_GROUP_SCOPE: ${group_scope}" >&2
      exit 2
      ;;
  esac
fi
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

validate_solver_manifest() {
  local manifest=$1
  if [[ -z ${expected_solver_contract} \
    && -z ${expected_solver_generator} \
    && -z ${expected_solver_source_sha256} \
    && -z ${expected_multidim_source_sha256} \
    && -z ${expected_shape_helper_source_sha256} \
    && -z ${expected_group_scope} ]]; then
    return 0
  fi
  python - \
    "${manifest}" \
    "${expected_solver_contract}" \
    "${expected_solver_generator}" \
    "${expected_solver_source_sha256}" \
    "${expected_group_scope}" \
    "${expected_multidim_source_sha256}" \
    "${expected_shape_helper_source_sha256}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "contract_version": sys.argv[2],
    "generator_version": sys.argv[3],
    "solver_source_sha256": sys.argv[4],
}
for field, value in expected.items():
    if value and manifest.get(field) != value:
        raise SystemExit(
            f"solver manifest mismatch for {field}: "
            f"expected={value} observed={manifest.get(field)}"
        )
expected_scope = sys.argv[5]
if expected_scope:
    scope_contract = manifest.get("scope_selection_contract")
    observed_scope = (
        scope_contract.get("mode") if isinstance(scope_contract, dict) else None
    )
    if observed_scope != expected_scope:
        raise SystemExit(
            "solver manifest mismatch for scope mode: "
            f"expected={expected_scope} observed={observed_scope}"
        )
dependency_contract = manifest.get("dependency_source_contract")
expected_multidim = sys.argv[6]
if expected_multidim:
    observed_multidim = (
        dependency_contract.get("solve_multidim_shape_coverage_sha256")
        if isinstance(dependency_contract, dict)
        else None
    )
    if observed_multidim != expected_multidim:
        raise SystemExit(
            "solver dependency mismatch for solve_multidim_shape_coverage: "
            f"expected={expected_multidim} observed={observed_multidim}"
        )
expected_shape_helper = sys.argv[7]
if expected_shape_helper:
    observed_shape_helper = (
        dependency_contract.get("solve_shape_coverage_sha256")
        if isinstance(dependency_contract, dict)
        else None
    )
    observed_v3_helper = manifest.get("v3_helper_source_sha256")
    if (
        observed_shape_helper != expected_shape_helper
        or observed_v3_helper != expected_shape_helper
    ):
        raise SystemExit(
            "solver dependency mismatch for solve_shape_coverage: "
            f"expected={expected_shape_helper} "
            f"observed_dependency={observed_shape_helper} "
            f"observed_v3_helper={observed_v3_helper}"
        )
PY
}

run_shard() {
  local name=$1
  local input_dir=${input_root}/${name}
  local output_dir=${run_root}/${name}
  local log=${run_root}/logs/${name}.log
  if [[ -f ${output_dir}/static/manifest.json ]]; then
    validate_solver_manifest "${output_dir}/static/manifest.json" || return 1
    echo "resume: ${name}" >"${log}"
    return 0
  fi
  if [[ -e ${output_dir} ]]; then
    echo "incomplete existing shard output: ${output_dir}" >&2
    return 1
  fi
  local -a solver_args=(
    "${output_dir}"
    --selected "${input_dir}/selected.parquet"
    --fake-gate-timeout-seconds "${fake_timeout}"
  )
  if [[ -n ${group_scope} ]]; then
    solver_args+=(--group-scope "${group_scope}")
  fi
  python -m "${solver_module}" "${solver_args[@]}" >"${log}" 2>&1 || return 1
  validate_solver_manifest "${output_dir}/static/manifest.json" || return 1
}
export -f validate_solver_manifest
export -f run_shard
export input_root run_root fake_timeout solver_module expected_solver_contract
export expected_solver_generator expected_solver_source_sha256 expected_group_scope
export expected_multidim_source_sha256 expected_shape_helper_source_sha256
export supports_group_scope group_scope

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
