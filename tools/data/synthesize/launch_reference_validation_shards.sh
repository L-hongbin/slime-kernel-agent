#!/usr/bin/env bash
set -uo pipefail

if (( $# < 1 || $# > 3 )); then
  echo "usage: $0 MACHINE_RANK [INPUT_PARQUET] [RUN_DIR]" >&2
  exit 2
fi

machine_rank=$1
input_path=${2:-Data/prompt_tvm_v4/synthesis/candidates/extreme_ops.parquet}
run_dir=${3:-local_artifacts/data_handoffs/prompt_tvm_v4_synth26_validation}
machine_count=${SYNTH_MACHINE_COUNT:-3}
gpus_per_machine=${SYNTH_GPUS_PER_MACHINE:-8}
virtual_shards_per_gpu=${SYNTH_VIRTUAL_SHARDS_PER_GPU:-1}
timeout_seconds=${SYNTH_TIMEOUT_SECONDS:-120}
expected_mode_class=${SYNTH_EXPECTED_MODE_CLASS:-any}
max_device_memory_gib=${SYNTH_MAX_DEVICE_MEMORY_GIB:-64}
required_max_device_memory_gib=64
idle_memory_mib=${SYNTH_IDLE_MEMORY_MIB:-64}
kernelgym_root=${SYNTH_KERNELGYM_ROOT:-/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-reward-only}
expected_input_sha256=${SYNTH_EXPECTED_INPUT_SHA256:-}
launcher_sha256=$(sha256sum "$0" | awk '{print $1}')

if ! [[ ${machine_rank} =~ ^[0-9]+$ ]] || (( machine_rank < 0 || machine_rank >= machine_count )); then
  echo "MACHINE_RANK must be an integer in [0, $((machine_count - 1))]" >&2
  exit 2
fi
if ! [[ ${gpus_per_machine} =~ ^[1-9][0-9]*$ ]]; then
  echo "SYNTH_GPUS_PER_MACHINE must be a positive integer" >&2
  exit 2
fi
if ! [[ ${virtual_shards_per_gpu} =~ ^[1-9][0-9]*$ ]]; then
  echo "SYNTH_VIRTUAL_SHARDS_PER_GPU must be a positive integer" >&2
  exit 2
fi
if ! [[ ${idle_memory_mib} =~ ^[0-9]+$ ]]; then
  echo "SYNTH_IDLE_MEMORY_MIB must be a non-negative integer" >&2
  exit 2
fi
if ! awk -v value="${max_device_memory_gib}" 'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0) }'; then
  echo "SYNTH_MAX_DEVICE_MEMORY_GIB must be a positive number" >&2
  exit 2
fi
if ! awk -v value="${max_device_memory_gib}" -v required="${required_max_device_memory_gib}" \
  'BEGIN { exit !(value + 0 == required + 0) }'; then
  echo "SYNTH_MAX_DEVICE_MEMORY_GIB must equal the fixed ${required_max_device_memory_gib}-GiB policy" >&2
  exit 2
fi
if [[ ! -f ${input_path} ]]; then
  echo "input parquet is missing: ${input_path}" >&2
  exit 2
fi
if [[ ! -f ${kernelgym_root}/kernelgym/toolkit/kernelbench/correctness.py ]]; then
  echo "KernelGym correctness implementation is missing under ${kernelgym_root}" >&2
  exit 2
fi
actual_input_sha256=$(sha256sum "${input_path}" | awk '{print $1}')
validator_path=tools/data/synthesize/validate_train_mode_contract.py
[[ -f ${validator_path} ]] || {
  echo "reference validator source is missing: ${validator_path}" >&2
  exit 2
}
validator_sha256=$(sha256sum "${validator_path}" | awk '{print $1}')
if [[ -n ${expected_input_sha256} ]]; then
  if [[ ${actual_input_sha256} != "${expected_input_sha256}" ]]; then
    echo "input SHA-256 mismatch: expected ${expected_input_sha256}, found ${actual_input_sha256}" >&2
    exit 2
  fi
fi

physical_shard_count=$((machine_count * gpus_per_machine))
shard_count=$((physical_shard_count * virtual_shards_per_gpu))

mkdir -p "${run_dir}"
if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required for exclusive validation launch" >&2
  exit 2
fi
lock_path="${run_dir}/.validation.lock"
exec {lock_fd}>"${lock_path}"
if ! flock -n "${lock_fd}"; then
  echo "another validation launcher holds ${lock_path}" >&2
  exit 2
fi

launcher_archive=${run_dir}/launcher_source.sh
if [[ -e ${launcher_archive} ]]; then
  if [[ ! -f ${launcher_archive} ]]; then
    echo "launcher source archive is not a regular file: ${launcher_archive}" >&2
    exit 2
  fi
  archived_launcher_sha256=$(sha256sum "${launcher_archive}" | awk '{print $1}')
  if [[ ${archived_launcher_sha256} != "${launcher_sha256}" ]]; then
    echo "launcher source archive hash mismatch: expected ${launcher_sha256}, found ${archived_launcher_sha256}" >&2
    exit 2
  fi
else
  launcher_archive_tmp=${launcher_archive}.tmp.$$
  cp -- "$0" "${launcher_archive_tmp}"
  mv -- "${launcher_archive_tmp}" "${launcher_archive}"
  archived_launcher_sha256=$(sha256sum "${launcher_archive}" | awk '{print $1}')
  if [[ ${archived_launcher_sha256} != "${launcher_sha256}" ]]; then
    echo "failed to archive the exact launcher source" >&2
    exit 2
  fi
fi

# Freeze scheduler inputs per machine rank.  A resume must replay the exact
# same shard family and validation policy; otherwise old and new evidence
# could coexist under one run directory.
scheduler_contract=${run_dir}/scheduler-contract-rank-${machine_rank}.json
scheduler_contract_tmp=${scheduler_contract}.tmp.$$
python - "${scheduler_contract_tmp}" \
  "${launcher_sha256}" "${validator_sha256}" \
  "${machine_rank}" "${machine_count}" \
  "${gpus_per_machine}" "${virtual_shards_per_gpu}" "${shard_count}" \
  "$(realpath "${input_path}")" "${actual_input_sha256}" \
  "${timeout_seconds}" "${expected_mode_class}" \
  "${max_device_memory_gib}" "${idle_memory_mib}" <<'PY' || exit 2
from __future__ import annotations

import json
import sys
from pathlib import Path

(
    output,
    launcher_sha256,
    validator_sha256,
    machine_rank,
    machine_count,
    gpus_per_machine,
    virtual_shards_per_gpu,
    shard_count,
    input_path,
    input_sha256,
    timeout_seconds,
    expected_mode_class,
    max_device_memory_gib,
    idle_memory_mib,
) = sys.argv[1:]
payload = {
    "contract_version": "reference_scheduler_contract_v1",
    "launcher_source_sha256": launcher_sha256,
    "validator_source_sha256": validator_sha256,
    "machine_rank": int(machine_rank),
    "machine_count": int(machine_count),
    "gpus_per_machine": int(gpus_per_machine),
    "virtual_shards_per_gpu": int(virtual_shards_per_gpu),
    "shard_count": int(shard_count),
    "input_path": input_path,
    "input_sha256": input_sha256,
    "timeout_seconds": float(timeout_seconds),
    "expected_mode_class": expected_mode_class,
    "max_device_memory_gib": float(max_device_memory_gib),
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

# The run-directory lock protects append-only evidence.  This node-local lock
# additionally protects the physical visible GPU set across different run
# directories, closing the idle-check-to-spawn race between launchers.
global_lock_path=/tmp/prompt_tvm_v4_reference_validation_all_visible_gpus.lock
exec {global_lock_fd}>"${global_lock_path}"
if ! flock -n "${global_lock_fd}"; then
  echo "another reference validation launcher owns the node-visible GPU set" >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required inside the validation container" >&2
  exit 2
fi
if ! gpu_snapshot=$(nvidia-smi \
  --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits); then
  echo "failed to query container-visible GPUs with nvidia-smi" >&2
  exit 2
fi
mapfile -t gpu_rows <<<"${gpu_snapshot}"
if (( ${#gpu_rows[@]} != gpus_per_machine )); then
  echo "expected exactly ${gpus_per_machine} container-visible GPUs, found ${#gpu_rows[@]}" >&2
  exit 2
fi
for ((local_gpu = 0; local_gpu < gpus_per_machine; local_gpu++)); do
  IFS=',' read -r gpu_index memory_used utilization <<<"${gpu_rows[$local_gpu]}"
  gpu_index=${gpu_index//[[:space:]]/}
  memory_used=${memory_used//[[:space:]]/}
  utilization=${utilization//[[:space:]]/}
  if ! [[ ${gpu_index} =~ ^[0-9]+$ && ${memory_used} =~ ^[0-9]+$ && ${utilization} =~ ^[0-9]+$ ]]; then
    echo "unparseable nvidia-smi row: ${gpu_rows[$local_gpu]}" >&2
    exit 2
  fi
  if (( memory_used > idle_memory_mib || utilization != 0 )); then
    echo "GPU ${gpu_index} is not idle: memory_used=${memory_used} MiB utilization=${utilization}%" >&2
    exit 2
  fi
done
if ! compute_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits); then
  echo "failed to query container-visible GPU compute processes" >&2
  exit 2
fi
if [[ -n ${compute_processes//[[:space:]]/} ]]; then
  echo "container-visible GPUs have active compute processes: ${compute_processes//$'\n'/,}" >&2
  exit 2
fi

# Importing torch alone does not initialize cuDNN. Exercise a real cuDNN op so
# a container whose runtime package drifted from the PyTorch build fails before
# it can turn individual references into misleading runtime failures.
if ! runtime_versions=$(CUDA_VISIBLE_DEVICES=0 python -c '
import torch
x = torch.randn((2, 3, 16, 16), device="cuda")
w = torch.randn((4, 3, 3, 3), device="cuda")
torch.nn.functional.conv2d(x, w)
print(f"torch={torch.__version__} cudnn={torch.backends.cudnn.version()}")
'); then
  echo "PyTorch/cuDNN validation preflight failed" >&2
  exit 2
fi
echo "validation runtime preflight: ${runtime_versions}"

pids=()
if ! queue_dir=$(mktemp -d "/tmp/prompt_tvm_v4_reference_queue.rank-${machine_rank}.XXXXXX"); then
  echo "failed to create the node-local reference queue" >&2
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

validate_final_summary() {
  local log_path=$1
  local output_path=$2
  local shard_id=$3
  local validator_status=$4
  python - "${log_path}" "${output_path}" "${shard_id}" "${shard_count}" "${validator_status}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
expected_output = str(output_path.resolve())
expected_shard = int(sys.argv[3])
expected_count = int(sys.argv[4])
validator_status = int(sys.argv[5])

try:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    summary = json.loads(lines[-1])
except (IndexError, OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"missing or invalid final JSON summary in {log_path}: {exc}")

if not isinstance(summary, dict):
    raise SystemExit(f"final JSON summary in {log_path} is not an object")
counts = {}
for key in ("selected", "executed", "resumed", "passed", "failed"):
    value = summary.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SystemExit(f"invalid {key!r} in final JSON summary: {value!r}")
    counts[key] = value
if counts["executed"] + counts["resumed"] != counts["selected"]:
    raise SystemExit("final JSON summary violates selected == executed + resumed")
if counts["passed"] + counts["failed"] != counts["selected"]:
    raise SystemExit("final JSON summary violates selected == passed + failed")
if summary.get("shard_index") != expected_shard or summary.get("shard_count") != expected_count:
    raise SystemExit("final JSON summary has the wrong shard identity")
if summary.get("output_path") != expected_output:
    raise SystemExit("final JSON summary has the wrong output path")
all_passed = summary.get("all_passed")
if not isinstance(all_passed, bool) or all_passed != (counts["failed"] == 0):
    raise SystemExit("final JSON summary has inconsistent all_passed/failed fields")
if validator_status == 0 and not all_passed:
    raise SystemExit("validator exited 0 with a failing final JSON summary")
if validator_status == 1 and all_passed:
    raise SystemExit("validator exited 1 with a passing final JSON summary")

# The row validator intentionally does not open an output file when a shard
# selects zero rows.  Materialize that one valid empty-shard case so the
# distributed verifier can require a complete JSONL shard family.  A missing
# output for any non-empty shard remains an infrastructure failure.
if counts["selected"] == 0 and not output_path.exists():
    output_path.touch(exist_ok=False)
try:
    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"missing or invalid reference output {output_path}: {exc}")
if len(records) != counts["selected"]:
    raise SystemExit("reference output row count differs from selected")
PY
}

claim_next_shard() {
  local virtual_offset local_slot shard_id
  for ((virtual_offset = 0; virtual_offset < virtual_shards_per_gpu; virtual_offset++)); do
    for ((local_slot = 0; local_slot < gpus_per_machine; local_slot++)); do
      shard_id=$((
        machine_rank * gpus_per_machine
        + local_slot
        + virtual_offset * physical_shard_count
      ))
      if mkdir "${queue_dir}/claim-${shard_id}" 2>/dev/null; then
        printf '%s\n' "${shard_id}"
        return 0
      fi
    done
  done
  return 1
}

run_gpu_queue() {
  local local_gpu=$1
  local queue_status=0
  local validator_pid=''
  local shard_id output_path log_path validator_status

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
    echo "reference queue claim: rank=${machine_rank} gpu=${local_gpu} shard=${shard_id}/${shard_count}"
    output_path="${run_dir}/shard-$(printf '%02d' "${shard_id}")-of-$(printf '%02d' "${shard_count}").jsonl"
    log_path="${run_dir}/shard-$(printf '%02d' "${shard_id}")-of-$(printf '%02d' "${shard_count}").log"
    CUDA_VISIBLE_DEVICES="${local_gpu}" python tools/data/synthesize/validate_train_mode_contract.py \
      "${input_path}" \
      --expected-mode-class "${expected_mode_class}" \
      --kernelgym-root "${kernelgym_root}" \
      --shard-count "${shard_count}" \
      --shard-index "${shard_id}" \
      --device cuda:0 \
      --max-device-memory-gib "${max_device_memory_gib}" \
      --timeout-seconds "${timeout_seconds}" \
      --launcher-sha256 "${launcher_sha256}" \
      --output "${output_path}" \
      >"${log_path}" 2>&1 &
    validator_pid=$!
    if wait "${validator_pid}"; then
      validator_status=0
    else
      validator_status=$?
    fi
    validator_pid=''

    if (( validator_status != 0 && validator_status != 1 )); then
      echo "validation shard ${shard_id} had infrastructure exit ${validator_status}; see ${log_path}" >&2
      return 2
    fi
    if ! validate_final_summary "${log_path}" "${output_path}" "${shard_id}" "${validator_status}"; then
      echo "validation shard ${shard_id} has no trustworthy final summary; see ${log_path}" >&2
      return 2
    fi
    if (( validator_status == 1 )); then
      echo "validation shard ${shard_id} contains row-level failures; continuing GPU queue" >&2
      queue_status=1
    fi
  done
  trap - INT TERM HUP
  return "${queue_status}"
}

for ((local_gpu = 0; local_gpu < gpus_per_machine; local_gpu++)); do
  run_gpu_queue "${local_gpu}" &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    continue
  else
    worker_status=$?
    echo "validation GPU worker ${index} failed with status ${worker_status}; see ${run_dir}" >&2
    if (( worker_status > status )); then
      status=${worker_status}
    fi
  fi
done
trap - INT TERM HUP
cleanup_queue_dir

python - "${run_dir}" "${machine_rank}" "${shard_count}" "${gpus_per_machine}" "${virtual_shards_per_gpu}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
machine_rank = int(sys.argv[2])
shard_count = int(sys.argv[3])
gpus_per_machine = int(sys.argv[4])
virtual_shards_per_gpu = int(sys.argv[5])
physical_shard_count = shard_count // virtual_shards_per_gpu
totals = {"selected": 0, "executed": 0, "resumed": 0, "passed": 0, "failed": 0}
summaries = []
for local_gpu in range(gpus_per_machine):
    base_shard = machine_rank * gpus_per_machine + local_gpu
    for virtual_offset in range(virtual_shards_per_gpu):
        shard_id = base_shard + virtual_offset * physical_shard_count
        log_path = run_dir / f"shard-{shard_id:02d}-of-{shard_count:02d}.log"
        lines = log_path.read_text(encoding="utf-8").splitlines() if log_path.is_file() else []
        try:
            summary = json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError):
            summary = {"shard_index": shard_id, "failed": 1, "summary_error": "missing_final_json_summary"}
        summaries.append(summary)
        for key in totals:
            totals[key] += int(summary.get(key, 0))
print(json.dumps({
    "machine_rank": machine_rank,
    "shard_count": shard_count,
    "virtual_shards_per_gpu": virtual_shards_per_gpu,
    **totals,
    "shards": summaries,
}, sort_keys=True))
PY

exit "${status}"
