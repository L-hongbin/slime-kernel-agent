#!/usr/bin/env bash
set -uo pipefail

if (( $# != 2 )); then
  echo "usage: $0 MACHINE_RANK RUN_DIR" >&2
  exit 2
fi

machine_rank=$1
run_dir=$2
machine_count=${SHAPE_REGION_MACHINE_COUNT:-4}
gpus_per_machine=${SHAPE_REGION_GPUS_PER_MACHINE:-8}
virtual_shards_per_gpu=${SHAPE_REGION_VIRTUAL_SHARDS_PER_GPU:-1}
timeout_seconds=${SHAPE_REGION_TIMEOUT_SECONDS:-600}
trials=${SHAPE_REGION_TRIALS:-3}
seed=${SHAPE_REGION_SEED:-17}
idle_memory_mib=${SHAPE_REGION_IDLE_MEMORY_MIB:-64}
launcher_sha256=$(sha256sum "$0" | awk '{print $1}')

selected=${run_dir}/selected.parquet
children=${run_dir}/static/children.parquet
manifest=${run_dir}/static/manifest.json
allowlist=${run_dir}/analysis/region_reference_both_pass_uuids.txt
output_dir=${run_dir}/h20/region

for value in "${machine_rank}" "${machine_count}" "${gpus_per_machine}" \
  "${virtual_shards_per_gpu}" "${trials}" "${seed}" "${idle_memory_mib}"; do
  [[ ${value} =~ ^[0-9]+$ ]] || {
    echo "rank/count/trials/seed/idle-memory values must be integers" >&2
    exit 2
  }
done
(( machine_count > 0 && gpus_per_machine > 0 && virtual_shards_per_gpu > 0 )) || {
  echo "machine/GPU/virtual-shard counts must be positive" >&2
  exit 2
}
(( machine_rank < machine_count && trials > 0 )) || {
  echo "require 0 <= machine_rank < machine_count and trials > 0" >&2
  exit 2
}
for path in "${selected}" "${children}" "${manifest}" "${allowlist}"; do
  [[ -f ${path} ]] || { echo "required input is missing: ${path}" >&2; exit 2; }
done
command -v flock >/dev/null 2>&1 || { echo "flock is required" >&2; exit 2; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required" >&2; exit 2; }

physical_shards=$((machine_count * gpus_per_machine))
shard_count=$((physical_shards * virtual_shards_per_gpu))
selected_sha256=$(sha256sum "${selected}" | awk '{print $1}')
children_sha256=$(sha256sum "${children}" | awk '{print $1}')
manifest_sha256=$(sha256sum "${manifest}" | awk '{print $1}')
allowlist_sha256=$(sha256sum "${allowlist}" | awk '{print $1}')
validator_path=tools/data/synthesize/validate_shape_region_liveness.py
runtime_validation_path=tools/data/cleaning/runtime_validation.py
augment_path=tools/data/synthesize/augment_prompt_tasks.py
shape_solver_helper_path=tools/data/synthesize/solve_shape_coverage.py
memory_guard_path=tools/data/synthesize/validate_train_mode_contract.py
for path in "${validator_path}" "${runtime_validation_path}" "${augment_path}" \
  "${shape_solver_helper_path}" "${memory_guard_path}"; do
  [[ -f ${path} ]] || { echo "required validation source is missing: ${path}" >&2; exit 2; }
done
validator_sha256=$(sha256sum "${validator_path}" | awk '{print $1}')
runtime_validation_sha256=$(sha256sum "${runtime_validation_path}" | awk '{print $1}')
augment_sha256=$(sha256sum "${augment_path}" | awk '{print $1}')
shape_solver_helper_sha256=$(sha256sum "${shape_solver_helper_path}" | awk '{print $1}')
memory_guard_sha256=$(sha256sum "${memory_guard_path}" | awk '{print $1}')

mkdir -p "${output_dir}"
exec {run_lock_fd}>"${output_dir}/.validation.lock"
flock -n "${run_lock_fd}" || {
  echo "another launcher owns ${output_dir}/.validation.lock" >&2
  exit 2
}
launcher_archive=${output_dir}/launcher_source.sh
if [[ -e ${launcher_archive} ]]; then
  [[ -f ${launcher_archive} ]] || {
    echo "launcher source archive is not a regular file: ${launcher_archive}" >&2
    exit 2
  }
  archived_launcher_sha256=$(sha256sum "${launcher_archive}" | awk '{print $1}')
  [[ ${archived_launcher_sha256} == "${launcher_sha256}" ]] || {
    echo "launcher source archive hash mismatch: expected ${launcher_sha256}, found ${archived_launcher_sha256}" >&2
    exit 2
  }
else
  launcher_archive_tmp=${launcher_archive}.tmp.$$
  if ! cp -- "$0" "${launcher_archive_tmp}" || ! mv -- "${launcher_archive_tmp}" "${launcher_archive}"; then
    echo "failed to archive launcher source" >&2
    exit 2
  fi
  archived_launcher_sha256=$(sha256sum "${launcher_archive}" | awk '{print $1}')
  [[ ${archived_launcher_sha256} == "${launcher_sha256}" ]] || {
    echo "failed to archive the exact launcher source" >&2
    exit 2
  }
fi

# Freeze the exact node-owned shard family and all run-defining artifacts.
# In particular, the Python validation binding does not include the allowlist,
# so the launcher contract must prevent a changed allowlist from being resumed
# into an existing evidence directory.
scheduler_contract=${output_dir}/scheduler-contract-rank-${machine_rank}.json
scheduler_contract_tmp=${scheduler_contract}.tmp.$$
python - "${scheduler_contract_tmp}" \
  "${launcher_sha256}" "${validator_sha256}" \
  "${runtime_validation_sha256}" "${augment_sha256}" \
  "${shape_solver_helper_sha256}" "${memory_guard_sha256}" \
  "${machine_rank}" "${machine_count}" \
  "${gpus_per_machine}" "${virtual_shards_per_gpu}" "${shard_count}" \
  "$(realpath "${selected}")" "${selected_sha256}" \
  "$(realpath "${children}")" "${children_sha256}" \
  "$(realpath "${manifest}")" "${manifest_sha256}" \
  "$(realpath "${allowlist}")" "${allowlist_sha256}" \
  "${timeout_seconds}" "${trials}" "${seed}" "${idle_memory_mib}" \
  <<'PY' || exit 2
from __future__ import annotations

import json
import sys
from pathlib import Path

(
    output,
    launcher_sha256,
    validator_sha256,
    runtime_validation_sha256,
    augment_sha256,
    shape_solver_helper_sha256,
    memory_guard_sha256,
    machine_rank,
    machine_count,
    gpus_per_machine,
    virtual_shards_per_gpu,
    shard_count,
    selected_path,
    selected_sha256,
    children_path,
    children_sha256,
    manifest_path,
    manifest_sha256,
    allowlist_path,
    allowlist_sha256,
    timeout_seconds,
    trials,
    seed,
    idle_memory_mib,
) = sys.argv[1:]
payload = {
    "contract_version": "shape_region_scheduler_contract_v1",
    "launcher_source_sha256": launcher_sha256,
    "validator_source_sha256": validator_sha256,
    "runtime_validation_source_sha256": runtime_validation_sha256,
    "augment_prompt_tasks_source_sha256": augment_sha256,
    "solve_shape_coverage_source_sha256": shape_solver_helper_sha256,
    "validate_train_mode_contract_source_sha256": memory_guard_sha256,
    "machine_rank": int(machine_rank),
    "machine_count": int(machine_count),
    "gpus_per_machine": int(gpus_per_machine),
    "virtual_shards_per_gpu": int(virtual_shards_per_gpu),
    "shard_count": int(shard_count),
    "selected_path": selected_path,
    "selected_sha256": selected_sha256,
    "children_path": children_path,
    "children_sha256": children_sha256,
    "manifest_path": manifest_path,
    "manifest_sha256": manifest_sha256,
    "allowlist_path": allowlist_path,
    "allowlist_sha256": allowlist_sha256,
    "timeout_seconds": float(timeout_seconds),
    "trials": int(trials),
    "seed": int(seed),
    "idle_memory_mib": int(idle_memory_mib),
}
Path(output).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
if [[ -e ${scheduler_contract} ]]; then
  if [[ ! -f ${scheduler_contract} ]] || ! cmp -s "${scheduler_contract_tmp}" "${scheduler_contract}"; then
    rm -f -- "${scheduler_contract_tmp}"
    echo "scheduler contract mismatch on resume: ${scheduler_contract}" >&2
    exit 2
  fi
  rm -f -- "${scheduler_contract_tmp}"
else
  if ! mv -- "${scheduler_contract_tmp}" "${scheduler_contract}"; then
    echo "failed to install scheduler contract: ${scheduler_contract}" >&2
    exit 2
  fi
fi
exec {gpu_lock_fd}>/tmp/prompt_tvm_v4_reference_validation_all_visible_gpus.lock
flock -n "${gpu_lock_fd}" || {
  echo "another validation launcher owns the node-visible GPU set" >&2
  exit 2
}

mapfile -t gpu_rows < <(
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
)
(( ${#gpu_rows[@]} == gpus_per_machine )) || {
  echo "expected ${gpus_per_machine} visible GPUs, found ${#gpu_rows[@]}" >&2
  exit 2
}
for row in "${gpu_rows[@]}"; do
  IFS=',' read -r gpu_index memory_used utilization <<<"${row}"
  gpu_index=${gpu_index//[[:space:]]/}
  memory_used=${memory_used//[[:space:]]/}
  utilization=${utilization//[[:space:]]/}
  [[ ${gpu_index} =~ ^[0-9]+$ && ${memory_used} =~ ^[0-9]+$ && ${utilization} =~ ^[0-9]+$ ]] || {
    echo "unparseable nvidia-smi row: ${row}" >&2
    exit 2
  }
  (( memory_used <= idle_memory_mib && utilization == 0 )) || {
    echo "GPU ${gpu_index} is not idle: memory=${memory_used} MiB util=${utilization}%" >&2
    exit 2
  }
done
compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) || exit 2
[[ -z ${compute_pids//[[:space:]]/} ]] || {
  echo "visible GPUs have active compute processes" >&2
  exit 2
}

CUDA_VISIBLE_DEVICES=0 python - <<'PY' || exit 2
import torch
x = torch.randn((2, 3, 16, 16), device="cuda")
w = torch.randn((4, 3, 3, 3), device="cuda")
torch.nn.functional.conv2d(x, w)
print(f"region runtime preflight: torch={torch.__version__} cudnn={torch.backends.cudnn.version()}")
PY

pids=()
if ! queue_dir=$(mktemp -d "/tmp/prompt_tvm_v4_region_queue.rank-${machine_rank}.XXXXXX"); then
  echo "failed to create the node-local region queue" >&2
  exit 2
fi

cleanup_queue_dir() {
  find "${queue_dir}" -mindepth 1 -maxdepth 1 -type d -name 'claim-*' \
    -exec rmdir -- {} + 2>/dev/null || true
  rmdir -- "${queue_dir}" 2>/dev/null || true
}

terminate_children() {
  local child_pid
  local -a active_children=("${pids[@]}")
  local -a job_children=()
  trap - INT TERM HUP
  mapfile -t job_children < <(jobs -pr)
  active_children+=("${job_children[@]}")
  for child_pid in "${active_children[@]}"; do
    kill -TERM "${child_pid}" 2>/dev/null || true
  done
  for child_pid in "${active_children[@]}"; do
    wait "${child_pid}" 2>/dev/null || true
  done
  cleanup_queue_dir
  exit 130
}
trap terminate_children INT TERM HUP

claim_next_shard() {
  local virtual_index local_slot shard_id
  for ((virtual_index = 0; virtual_index < virtual_shards_per_gpu; virtual_index++)); do
    for ((local_slot = 0; local_slot < gpus_per_machine; local_slot++)); do
      shard_id=$((
        machine_rank * gpus_per_machine
        + local_slot
        + virtual_index * physical_shards
      ))
      if mkdir "${queue_dir}/claim-${shard_id}" 2>/dev/null; then
        printf '%s\n' "${shard_id}"
        return 0
      fi
    done
  done
  return 1
}

validate_final_summary() {
  local log_path=$1
  local output_path=$2
  local shard_id=$3
  python - "${log_path}" "${output_path}" "${shard_id}" "${shard_count}" \
    "${launcher_sha256}" "${validator_sha256}" <<'PY'
from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path

log_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
expected_output = str(output_path.resolve())
expected_shard = int(sys.argv[3])
expected_count = int(sys.argv[4])
expected_launcher = sys.argv[5]
expected_validator = sys.argv[6]

try:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    summary = json.loads(lines[-1])
except (IndexError, OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"missing or invalid final JSON summary in {log_path}: {exc}")
if not isinstance(summary, Mapping):
    raise SystemExit(f"final JSON summary in {log_path} is not an object")
if summary.get("contract_version") != "shape_changed_region_liveness_v3":
    raise SystemExit("final JSON summary has the wrong region contract")
if summary.get("perturbation_contract_version") != "seeded_bounded_non_affine_mix_v1":
    raise SystemExit("final JSON summary has the wrong perturbation contract")
if summary.get("shard_index") != expected_shard or summary.get("shard_count") != expected_count:
    raise SystemExit("final JSON summary has the wrong shard identity")
if summary.get("output_path") != expected_output:
    raise SystemExit("final JSON summary has the wrong output path")
if summary.get("launcher_source_sha256") != expected_launcher:
    raise SystemExit("final JSON summary has the wrong launcher hash")
if summary.get("validator_source_sha256") != expected_validator:
    raise SystemExit("final JSON summary has the wrong validator hash")

counts = {}
for key in ("selected", "executed", "resumed", "passed", "failed"):
    value = summary.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SystemExit(f"invalid {key!r} in final JSON summary: {value!r}")
    counts[key] = value
if summary.get("rows") != counts["selected"]:
    raise SystemExit("final JSON summary rows/selected mismatch")
if counts["executed"] + counts["resumed"] != counts["selected"]:
    raise SystemExit("final JSON summary violates selected == executed + resumed")
if counts["passed"] + counts["failed"] != counts["selected"]:
    raise SystemExit("final JSON summary violates selected == passed + failed")
raw_counts = summary.get("counts")
if not isinstance(raw_counts, Mapping):
    raise SystemExit("final JSON summary has invalid raw status counts")
if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in raw_counts.values()):
    raise SystemExit("final JSON summary has invalid raw status count values")
if sum(raw_counts.values()) != counts["selected"]:
    raise SystemExit("final JSON summary raw status counts do not sum to selected")
if int(raw_counts.get("passed", 0)) != counts["passed"]:
    raise SystemExit("final JSON summary raw/passed counts differ")

try:
    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid region output {output_path}: {exc}")
if len(records) != counts["selected"]:
    raise SystemExit("region output row count differs from selected")
child_uuids = [record.get("child_uuid") for record in records]
if any(not isinstance(uuid, str) or not uuid for uuid in child_uuids):
    raise SystemExit("region output contains an invalid child UUID")
if len(set(child_uuids)) != len(child_uuids):
    raise SystemExit("region output contains duplicate child UUIDs")
if any(record.get("launcher_source_sha256") != expected_launcher for record in records):
    raise SystemExit("region output contains a mismatched launcher hash")
if any(record.get("validator_source_sha256") != expected_validator for record in records):
    raise SystemExit("region output contains a mismatched validator hash")
PY
}

run_gpu_queue() {
  local local_gpu=$1
  local validator_pid=''
  local shard_id output log validator_status

  terminate_gpu_queue() {
    local child_pid
    local -a active_children=()
    local -a job_children=()
    trap - INT TERM HUP
    if [[ -n ${validator_pid} ]] && kill -0 "${validator_pid}" 2>/dev/null; then
      active_children+=("${validator_pid}")
    fi
    mapfile -t job_children < <(jobs -pr)
    active_children+=("${job_children[@]}")
    for child_pid in "${active_children[@]}"; do
      kill -TERM "${child_pid}" 2>/dev/null || true
    done
    for child_pid in "${active_children[@]}"; do
      wait "${child_pid}" 2>/dev/null || true
    done
    exit 130
  }
  trap terminate_gpu_queue INT TERM HUP

  while shard_id=$(claim_next_shard); do
    echo "region queue claim: rank=${machine_rank} gpu=${local_gpu} shard=${shard_id}/${shard_count}"
    output=${output_dir}/shard-$(printf '%03d' "${shard_id}")-of-$(printf '%03d' "${shard_count}").jsonl
    log=${output%.jsonl}.log
    CUDA_VISIBLE_DEVICES=${local_gpu} python \
      tools/data/synthesize/validate_shape_region_liveness.py \
      "${selected}" "${children}" "${manifest}" "${output}" \
      --child-uuid-file "${allowlist}" \
      --device cuda:0 \
      --trials "${trials}" \
      --seed "${seed}" \
      --timeout-seconds "${timeout_seconds}" \
      --launcher-sha256 "${launcher_sha256}" \
      --shard-count "${shard_count}" \
      --shard-index "${shard_id}" \
      >"${log}" 2>&1 &
    validator_pid=$!
    if wait "${validator_pid}"; then
      validator_status=0
    else
      validator_status=$?
    fi
    validator_pid=''
    if (( validator_status != 0 )); then
      echo "region shard ${shard_id} had infrastructure exit ${validator_status}; see ${log}" >&2
      return 2
    fi
    if ! validate_final_summary "${log}" "${output}" "${shard_id}"; then
      echo "region shard ${shard_id} has no trustworthy final summary; see ${log}" >&2
      return 2
    fi
  done
  trap - INT TERM HUP
  return 0
}

for ((local_gpu = 0; local_gpu < gpus_per_machine; local_gpu++)); do
  run_gpu_queue "${local_gpu}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if wait "${pid}"; then
    continue
  else
    worker_status=$?
    (( worker_status > status )) && status=${worker_status}
  fi
done
trap - INT TERM HUP
cleanup_queue_dir

shard_suffix=$(printf '%03d' "${shard_count}")
record_count=$(find "${output_dir}" -name "shard-*-of-${shard_suffix}.jsonl" -type f -exec cat {} + | wc -l)
printf '{"machine_rank":%s,"shard_count":%s,"records":%s,"status":%s,"launcher_source_sha256":"%s"}\n' \
  "${machine_rank}" "${shard_count}" "${record_count}" "${status}" "${launcher_sha256}"
exit "${status}"
