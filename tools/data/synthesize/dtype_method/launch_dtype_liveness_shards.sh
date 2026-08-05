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
gpus_per_machine=${SYNTH_GPUS_PER_MACHINE:-8}
trials=${SYNTH_DTYPE_LIVENESS_TRIALS:-3}
seed=${SYNTH_DTYPE_LIVENESS_SEED:-17}
timeout_seconds=${SYNTH_TIMEOUT_SECONDS:-600}
idle_memory_mib=${SYNTH_IDLE_MEMORY_MIB:-64}
validator_path=tools/data/synthesize/dtype_method/validate_dtype_liveness.py

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
if ! [[ ${trials} =~ ^[1-9][0-9]*$ ]] || (( trials < 3 )); then
  echo "SYNTH_DTYPE_LIVENESS_TRIALS must be an integer of at least 3" >&2
  exit 2
fi
if [[ ${seed} != 17 ]]; then
  echo "SYNTH_DTYPE_LIVENESS_SEED must be the frozen value 17" >&2
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

scheduler_contract=${run_dir}/scheduler-contract.json
scheduler_contract_tmp=${scheduler_contract}.tmp.$$
python - "${scheduler_contract_tmp}" "${launcher_sha256}" "${validator_sha256}" \
  "$(realpath "${parents_path}")" "${parents_sha256}" \
  "$(realpath "${children_path}")" "${children_sha256}" \
  "$(realpath "${manifest_path}")" "${manifest_sha256}" \
  "$(realpath "${allowlist_path}")" "${allowlist_sha256}" \
  "${gpus_per_machine}" "${trials}" "${seed}" "${timeout_seconds}" <<'PY'
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
    shard_count,
    trials,
    seed,
    timeout_seconds,
) = sys.argv[1:]
payload = {
    "contract_version": "dtype_parameter_free_runtime_liveness_scheduler_v1",
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
echo "dtype liveness runtime preflight: ${runtime_versions}"

pids=()
for ((gpu = 0; gpu < gpus_per_machine; gpu++)); do
  output_path=${run_dir}/shard-$(printf '%02d' "${gpu}")-of-$(printf '%02d' "${gpus_per_machine}").jsonl
  log_path=${run_dir}/shard-$(printf '%02d' "${gpu}")-of-$(printf '%02d' "${gpus_per_machine}").log
  CUDA_VISIBLE_DEVICES=${gpu} python "${validator_path}" \
    "${parents_path}" "${children_path}" "${manifest_path}" "${output_path}" \
    --child-uuid-file "${allowlist_path}" \
    --device cuda:0 \
    --trials "${trials}" \
    --seed "${seed}" \
    --timeout-seconds "${timeout_seconds}" \
    --launcher-sha256 "${launcher_sha256}" \
    --shard-index "${gpu}" \
    --shard-count "${gpus_per_machine}" \
    >"${log_path}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=2
  fi
done
if (( status != 0 )); then
  echo "one or more dtype-liveness shards had an infrastructure failure" >&2
  exit "${status}"
fi

python - "${run_dir}" "${gpus_per_machine}" "${launcher_sha256}" "${validator_sha256}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
shard_count = int(sys.argv[2])
launcher_sha256 = sys.argv[3]
validator_sha256 = sys.argv[4]
summaries = []
for shard_index in range(shard_count):
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
    "contract_version": "dtype_parameter_free_runtime_liveness_launcher_summary_v1",
    "launcher_source_sha256": launcher_sha256,
    "validator_source_sha256": validator_sha256,
    "shard_count": shard_count,
    "selected": sum(item["selected"] for item in summaries),
    "executed": sum(item["executed"] for item in summaries),
    "resumed": sum(item["resumed"] for item in summaries),
    "passed": sum(item["passed"] for item in summaries),
    "failed": sum(item["failed"] for item in summaries),
    "shards": summaries,
}
(run_dir / "launcher-summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, sort_keys=True))
PY
