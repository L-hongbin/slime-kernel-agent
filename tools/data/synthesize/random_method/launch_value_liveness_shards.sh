#!/usr/bin/env bash
set -uo pipefail

if (( $# != 5 )); then
  echo "usage: $0 PARENTS_PARQUET CHILDREN_PARQUET MANIFEST_JSONL CHILD_UUID_FILE RUN_DIR" >&2
  exit 2
fi

parents_path=$1
children_path=$2
manifest_path=$3
allowlist_path=$4
run_dir=$5
machine_rank=${SYNTH_MACHINE_RANK:-0}
machine_count=${SYNTH_MACHINE_COUNT:-1}
gpus_per_machine=${SYNTH_GPUS_PER_MACHINE:-8}
virtual_shards_per_gpu=${SYNTH_VIRTUAL_SHARDS_PER_GPU:-1}
trials=${SYNTH_VALUE_LIVENESS_TRIALS:-3}
seed=${SYNTH_VALUE_LIVENESS_SEED:-17}
timeout_seconds=${SYNTH_TIMEOUT_SECONDS:-600}
idle_memory_mib=${SYNTH_IDLE_MEMORY_MIB:-64}
validator_path=tools/data/synthesize/random_method/validate_value_liveness.py

for path in "${parents_path}" "${children_path}" "${manifest_path}" "${allowlist_path}" "${validator_path}"; do
  if [[ ! -f ${path} ]]; then
    echo "required input is missing: ${path}" >&2
    exit 2
  fi
done
if ! [[ ${gpus_per_machine} =~ ^[1-9][0-9]*$ ]]; then
  echo "SYNTH_GPUS_PER_MACHINE must be a positive integer" >&2
  exit 2
fi
if ! [[ ${machine_count} =~ ^[1-9][0-9]*$ ]]; then
  echo "SYNTH_MACHINE_COUNT must be a positive integer" >&2
  exit 2
fi
if ! [[ ${machine_rank} =~ ^[0-9]+$ ]] || (( machine_rank >= machine_count )); then
  echo "SYNTH_MACHINE_RANK must be an integer in [0, $((machine_count - 1))]" >&2
  exit 2
fi
if ! [[ ${virtual_shards_per_gpu} =~ ^[1-9][0-9]*$ ]]; then
  echo "SYNTH_VIRTUAL_SHARDS_PER_GPU must be a positive integer" >&2
  exit 2
fi
if ! [[ ${trials} =~ ^[1-9][0-9]*$ ]] || (( trials < 3 )); then
  echo "SYNTH_VALUE_LIVENESS_TRIALS must be an integer of at least 3" >&2
  exit 2
fi
if [[ ${seed} != 17 ]]; then
  echo "SYNTH_VALUE_LIVENESS_SEED must be the frozen value 17" >&2
  exit 2
fi
if ! [[ ${idle_memory_mib} =~ ^[0-9]+$ ]]; then
  echo "SYNTH_IDLE_MEMORY_MIB must be a non-negative integer" >&2
  exit 2
fi
if ! awk -v value="${timeout_seconds}" 'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0) }'; then
  echo "SYNTH_TIMEOUT_SECONDS must be positive" >&2
  exit 2
fi

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

parents_sha256=$(sha256_file "${parents_path}")
children_sha256=$(sha256_file "${children_path}")
manifest_sha256=$(sha256_file "${manifest_path}")
allowlist_sha256=$(sha256_file "${allowlist_path}")
validator_sha256=$(sha256_file "${validator_path}")
launcher_sha256=$(sha256_file "$0")

verify_expected_hash() {
  local label=$1
  local actual=$2
  local expected=$3
  if [[ -n ${expected} && ${actual} != "${expected}" ]]; then
    echo "${label} SHA-256 mismatch: expected ${expected}, found ${actual}" >&2
    exit 2
  fi
}

verify_expected_hash parents "${parents_sha256}" "${SYNTH_EXPECTED_PARENTS_SHA256:-}"
verify_expected_hash children "${children_sha256}" "${SYNTH_EXPECTED_CHILDREN_SHA256:-}"
verify_expected_hash manifest "${manifest_sha256}" "${SYNTH_EXPECTED_MANIFEST_SHA256:-}"
verify_expected_hash allowlist "${allowlist_sha256}" "${SYNTH_EXPECTED_ALLOWLIST_SHA256:-}"

physical_shard_count=$((machine_count * gpus_per_machine))
shard_count=$((physical_shard_count * virtual_shards_per_gpu))

mkdir -p "${run_dir}"
if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required for exclusive validation launch" >&2
  exit 2
fi
exec {run_lock_fd}>"${run_dir}/.validation.lock"
if ! flock -n "${run_lock_fd}"; then
  echo "another liveness launcher owns ${run_dir}" >&2
  exit 2
fi

# Share the reference launcher's node-wide lock: reference correctness and
# liveness evidence must never contend for memory on the same GPU set.
global_lock_path=/tmp/prompt_tvm_v4_reference_validation_all_visible_gpus.lock
exec {global_lock_fd}>"${global_lock_path}"
if ! flock -n "${global_lock_fd}"; then
  echo "another validation launcher owns the node-visible GPU set" >&2
  exit 2
fi

for source_path in "$0" "${validator_path}"; do
  if [[ ${source_path} == "$0" ]]; then
    archive_path=${run_dir}/launcher_source.sh
  else
    archive_path=${run_dir}/validator_source.py
  fi
  if [[ -e ${archive_path} ]]; then
    if [[ ! -f ${archive_path} ]] || ! cmp -s "${source_path}" "${archive_path}"; then
      echo "source archive mismatch: ${archive_path}" >&2
      exit 2
    fi
  else
    cp -- "${source_path}" "${archive_path}.tmp.$$"
    mv -- "${archive_path}.tmp.$$" "${archive_path}"
  fi
done

scheduler_contract=${run_dir}/scheduler-contract-rank-${machine_rank}.json
scheduler_contract_tmp=${scheduler_contract}.tmp.$$
python - "${scheduler_contract_tmp}" "${launcher_sha256}" "${validator_sha256}" \
  "$(realpath "${parents_path}")" "${parents_sha256}" \
  "$(realpath "${children_path}")" "${children_sha256}" \
  "$(realpath "${manifest_path}")" "${manifest_sha256}" \
  "$(realpath "${allowlist_path}")" "${allowlist_sha256}" \
  "${machine_rank}" "${machine_count}" "${gpus_per_machine}" \
  "${virtual_shards_per_gpu}" "${shard_count}" \
  "${trials}" "${seed}" "${timeout_seconds}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

(
    output,
    launcher_sha256,
    validator_sha256,
    parents_path,
    parents_sha256,
    children_path,
    children_sha256,
    manifest_path,
    manifest_sha256,
    allowlist_path,
    allowlist_sha256,
    machine_rank,
    machine_count,
    gpus_per_machine,
    virtual_shards_per_gpu,
    shard_count,
    trials,
    seed,
    timeout_seconds,
) = sys.argv[1:]
payload = {
    "contract_version": "random_value_runtime_liveness_scheduler_v3",
    "launcher_source_sha256": launcher_sha256,
    "validator_source_sha256": validator_sha256,
    "parents_path": parents_path,
    "parents_sha256": parents_sha256,
    "children_path": children_path,
    "children_sha256": children_sha256,
    "manifest_path": manifest_path,
    "manifest_sha256": manifest_sha256,
    "allowlist_path": allowlist_path,
    "allowlist_sha256": allowlist_sha256,
    "machine_rank": int(machine_rank),
    "machine_count": int(machine_count),
    "gpus_per_machine": int(gpus_per_machine),
    "virtual_shards_per_gpu": int(virtual_shards_per_gpu),
    "shard_count": int(shard_count),
    "trials": int(trials),
    "seed": int(seed),
    "timeout_seconds": float(timeout_seconds),
    "max_device_memory_gib": 64.0,
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
if [[ -e ${scheduler_contract} ]]; then
  if [[ ! -f ${scheduler_contract} ]] || ! cmp -s "${scheduler_contract_tmp}" "${scheduler_contract}"; then
    rm -f -- "${scheduler_contract_tmp}"
    echo "scheduler contract mismatch on resume: ${scheduler_contract}" >&2
    exit 2
  fi
  rm -f -- "${scheduler_contract_tmp}"
else
  mv -- "${scheduler_contract_tmp}" "${scheduler_contract}"
fi

if ! gpu_snapshot=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits); then
  echo "failed to query GPUs" >&2
  exit 2
fi
mapfile -t gpu_rows <<<"${gpu_snapshot}"
if (( ${#gpu_rows[@]} != gpus_per_machine )); then
  echo "expected ${gpus_per_machine} visible GPUs, found ${#gpu_rows[@]}" >&2
  exit 2
fi
for row in "${gpu_rows[@]}"; do
  IFS=',' read -r gpu_index memory_used utilization <<<"${row}"
  gpu_index=${gpu_index//[[:space:]]/}
  memory_used=${memory_used//[[:space:]]/}
  utilization=${utilization//[[:space:]]/}
  if ! [[ ${gpu_index} =~ ^[0-9]+$ && ${memory_used} =~ ^[0-9]+$ && ${utilization} =~ ^[0-9]+$ ]]; then
    echo "unparseable nvidia-smi row: ${row}" >&2
    exit 2
  fi
  if (( memory_used > idle_memory_mib || utilization != 0 )); then
    echo "GPU ${gpu_index} is not idle: memory_used=${memory_used} MiB utilization=${utilization}%" >&2
    exit 2
  fi
done
if ! compute_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits); then
  echo "failed to query GPU compute processes" >&2
  exit 2
fi
if [[ -n ${compute_processes//[[:space:]]/} ]]; then
  echo "container-visible GPUs have active compute processes: ${compute_processes//$'\n'/,}" >&2
  exit 2
fi
if ! runtime_versions=$(CUDA_VISIBLE_DEVICES=0 python -c '
import torch
x = torch.randn((2, 3, 16, 16), device="cuda")
w = torch.randn((4, 3, 3, 3), device="cuda")
torch.nn.functional.conv2d(x, w)
print(f"torch={torch.__version__} cudnn={torch.backends.cudnn.version()}")
'); then
  echo "PyTorch/cuDNN liveness preflight failed" >&2
  exit 2
fi
echo "value liveness runtime preflight: ${runtime_versions}"

pids=()
queue_dir=$(mktemp -d "/tmp/prompt_tvm_v4_value_liveness_queue.rank-${machine_rank}.XXXXXX") || exit 2

cleanup_queue_dir() {
  find "${queue_dir}" -mindepth 1 -maxdepth 1 -type d -name 'claim-*' \
    -exec rmdir -- {} + 2>/dev/null || true
  rmdir -- "${queue_dir}" 2>/dev/null || true
}

terminate_children() {
  local pid
  trap - INT TERM HUP
  for pid in "${pids[@]}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  cleanup_queue_dir
  exit 130
}
trap terminate_children INT TERM HUP

claim_next_shard() {
  local virtual_offset local_slot shard_id
  for ((virtual_offset = 0; virtual_offset < virtual_shards_per_gpu; virtual_offset++)); do
    for ((local_slot = 0; local_slot < gpus_per_machine; local_slot++)); do
      shard_id=$((machine_rank * gpus_per_machine + local_slot + virtual_offset * physical_shard_count))
      if mkdir "${queue_dir}/claim-${shard_id}" 2>/dev/null; then
        printf '%s\n' "${shard_id}"
        return 0
      fi
    done
  done
  return 1
}

run_gpu_queue() {
  local gpu=$1
  local shard_id output_path log_path validator_pid=''

  terminate_gpu_queue() {
    trap - INT TERM HUP
    if [[ -n ${validator_pid} ]] && kill -0 "${validator_pid}" 2>/dev/null; then
      kill -TERM "${validator_pid}" 2>/dev/null || true
      wait "${validator_pid}" 2>/dev/null || true
    fi
    exit 130
  }
  trap terminate_gpu_queue INT TERM HUP

  while shard_id=$(claim_next_shard); do
    echo "value-liveness queue claim: rank=${machine_rank} gpu=${gpu} shard=${shard_id}/${shard_count}"
    output_path=${run_dir}/shard-$(printf '%02d' "${shard_id}")-of-$(printf '%02d' "${shard_count}").jsonl
    log_path=${run_dir}/shard-$(printf '%02d' "${shard_id}")-of-$(printf '%02d' "${shard_count}").log
    CUDA_VISIBLE_DEVICES=${gpu} python "${validator_path}" \
      "${parents_path}" "${children_path}" "${manifest_path}" "${output_path}" \
      --child-uuid-file "${allowlist_path}" \
      --device cuda:0 \
      --trials "${trials}" \
      --seed "${seed}" \
      --timeout-seconds "${timeout_seconds}" \
      --launcher-sha256 "${launcher_sha256}" \
      --shard-index "${shard_id}" \
      --shard-count "${shard_count}" \
      >"${log_path}" 2>&1 &
    validator_pid=$!
    if ! wait "${validator_pid}"; then
      validator_pid=''
      echo "value-liveness shard ${shard_id} had an infrastructure failure; see ${log_path}" >&2
      return 2
    fi
    validator_pid=''
  done
  trap - INT TERM HUP
}

for ((gpu = 0; gpu < gpus_per_machine; gpu++)); do
  run_gpu_queue "${gpu}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=2
  fi
done
trap - INT TERM HUP
cleanup_queue_dir
if (( status != 0 )); then
  echo "one or more value-liveness shards had an infrastructure failure" >&2
  exit "${status}"
fi

python - "${run_dir}" "${machine_rank}" "${machine_count}" "${gpus_per_machine}" \
  "${virtual_shards_per_gpu}" "${shard_count}" "${launcher_sha256}" "${validator_sha256}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
machine_rank = int(sys.argv[2])
machine_count = int(sys.argv[3])
gpus_per_machine = int(sys.argv[4])
virtual_shards_per_gpu = int(sys.argv[5])
shard_count = int(sys.argv[6])
launcher_sha256 = sys.argv[7]
validator_sha256 = sys.argv[8]
physical_shard_count = machine_count * gpus_per_machine
summaries = []
for local_slot in range(gpus_per_machine):
    base_shard = machine_rank * gpus_per_machine + local_slot
    for virtual_offset in range(virtual_shards_per_gpu):
        shard_index = base_shard + virtual_offset * physical_shard_count
        log_path = run_dir / f"shard-{shard_index:02d}-of-{shard_count:02d}.log"
        lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            raise SystemExit(f"empty shard log: {log_path}")
        summary = json.loads(lines[-1])
        if summary.get("shard_index") != shard_index or summary.get("shard_count") != shard_count:
            raise SystemExit(f"invalid final shard summary: {log_path}")
        if summary.get("launcher_source_sha256") != launcher_sha256:
            raise SystemExit(f"launcher hash mismatch: {log_path}")
        if summary.get("validator_source_sha256") != validator_sha256:
            raise SystemExit(f"validator hash mismatch: {log_path}")
        summaries.append(summary)
payload = {
    "contract_version": "random_value_runtime_liveness_launcher_summary_v3",
    "launcher_source_sha256": launcher_sha256,
    "validator_source_sha256": validator_sha256,
    "machine_rank": machine_rank,
    "machine_count": machine_count,
    "gpus_per_machine": gpus_per_machine,
    "virtual_shards_per_gpu": virtual_shards_per_gpu,
    "shard_count": shard_count,
    "selected": sum(item["selected"] for item in summaries),
    "executed": sum(item["executed"] for item in summaries),
    "resumed": sum(item["resumed"] for item in summaries),
    "passed": sum(item["passed"] for item in summaries),
    "failed": sum(item["failed"] for item in summaries),
    "shards": summaries,
}
(run_dir / f"launcher-summary-rank-{machine_rank}.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, sort_keys=True))
PY
