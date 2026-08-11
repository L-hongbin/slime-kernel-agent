#!/usr/bin/env bash
# Launch the exact-1k KernelBench-gap liveness proof on eight idle CUDA GPUs.
set -euo pipefail

if (( $# != 4 )); then
  echo "usage: $0 CANDIDATES_PARQUET MANIFEST_JSONL REFERENCE_PASSED_UUIDS RUN_DIR" >&2
  exit 2
fi

candidates_path=$1
manifest_path=$2
allowlist_path=$3
run_dir=$4
gpu_count=8
max_authorized_candidates=1000
trials=${KERNELBENCH_GAP_LIVENESS_TRIALS:-3}
seed=${KERNELBENCH_GAP_LIVENESS_SEED:-17}
timeout_seconds=${KERNELBENCH_GAP_LIVENESS_TIMEOUT_SECONDS:-600}
idle_memory_mib=${KERNELBENCH_GAP_IDLE_MEMORY_MIB:-128}
adapter_path=tools/data/synthesize/kernelbench_gap_method/validate_kernelbench_gap_liveness.py
generator_path=tools/data/synthesize/kernelbench_gap_method/generate_kernelbench_gap.py
runtime_core_path=tools/data/synthesize/semantic_operator_method/validate_semantic_liveness.py

die() { echo "$*" >&2; exit 2; }

[[ -f ${candidates_path} ]] || die "candidates artifact missing: ${candidates_path}"
[[ -f ${manifest_path} ]] || die "manifest artifact missing: ${manifest_path}"
[[ -f ${allowlist_path} ]] || die "reference-pass UUID allowlist missing: ${allowlist_path}"
[[ -f ${adapter_path} && -f ${generator_path} && -f ${runtime_core_path} ]] || die "kernelbench-gap source closure missing"
[[ ${trials} =~ ^[0-9]+$ ]] && (( trials == 3 )) || die "KERNELBENCH_GAP_LIVENESS_TRIALS must be exactly 3"
[[ ${seed} == 17 ]] || die "KERNELBENCH_GAP_LIVENESS_SEED is fixed to 17"
[[ ${idle_memory_mib} =~ ^[0-9]+$ ]] || die "KERNELBENCH_GAP_IDLE_MEMORY_MIB must be non-negative"

run_dir_real=$(realpath -m "${run_dir}")
case ${run_dir_real} in
  *kernelbench_gap*) ;;
  *) die "RUN_DIR must be an explicit kernelbench_gap lane: ${run_dir_real}" ;;
esac
case ${run_dir_real} in
  *frontier_operator*|*semantic_operator*|*random_method*|*dtype_method*|*layout_method*)
    die "RUN_DIR points at a different synthesis lane: ${run_dir_real}" ;;
esac

launcher_path=$(realpath "$0")
launcher_sha256=$(sha256sum "${launcher_path}" | awk '{print $1}')
adapter_sha256=$(sha256sum "${adapter_path}" | awk '{print $1}')
generator_sha256=$(sha256sum "${generator_path}" | awk '{print $1}')
runtime_core_sha256=$(sha256sum "${runtime_core_path}" | awk '{print $1}')
candidates_sha256=$(sha256sum "${candidates_path}" | awk '{print $1}')
manifest_sha256=$(sha256sum "${manifest_path}" | awk '{print $1}')
allowlist_sha256=$(sha256sum "${allowlist_path}" | awk '{print $1}')

# The source hashes below bind the launcher, adapter, generator and shared
# runtime core.  On NFS, an old repository-local __pycache__ can otherwise be
# selected between this binding and a worker import.  Use a fresh cache root
# for every invocation, matching the established frontier liveness launcher.
python_cache_root=$(mktemp -d /tmp/prompt_tvm_v4_kernelbench_gap_pycache.XXXXXX)
export PYTHONPYCACHEPREFIX=${python_cache_root}

python3 - "${candidates_path}" "${manifest_path}" "${allowlist_path}" "${generator_sha256}" "${gpu_count}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from tools.data.synthesize.kernelbench_gap_method import generate_kernelbench_gap as generator
from tools.data.synthesize.kernelbench_gap_method import validate_kernelbench_gap_liveness as adapter

candidates, manifest, allowlist, generator_sha, gpu_count = (
    Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], int(sys.argv[5])
)
rows = pq.read_table(candidates).to_pylist()
manifests = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
allowed = [line.strip() for line in allowlist.read_text(encoding="utf-8").splitlines() if line.strip()]
expected = int(generator.EXACT_CANARY_ROWS)
if expected != 1000 or int(generator.MAX_AUTHORIZED_CANDIDATES) != expected:
    raise SystemExit("generator exact-1k constants are not bound")
if len(rows) != expected or len(manifests) != expected:
    raise SystemExit(f"kernelbench-gap canary must contain exactly {expected} aligned rows:{len(rows)}:{len(manifests)}")
if hashlib.sha256(Path(generator.__file__).read_bytes()).hexdigest() != generator_sha:
    raise SystemExit("generator source hash changed during launcher setup")
known, row_index_by_uuid = set(), {}
for index, (row, item) in enumerate(zip(rows, manifests, strict=True)):
    uuid = row.get("extra_info", {}).get("uuid")
    code = row.get("reward_model", {}).get("ground_truth")
    if item.get("candidate_row_index") != index or item.get("uuid") != uuid:
        raise SystemExit(f"candidate/manifest identity mismatch:{index}")
    if item.get("method") != "kernelbench_gap_canary":
        raise SystemExit(f"kernelbench-gap method mismatch:{index}")
    if item.get("primary_intervention") != "semantic_operator" or item.get("lineage_kind") != "standalone_semantic_synthetic":
        raise SystemExit(f"semantic compatibility lineage mismatch:{index}")
    if item.get("generator_source_sha256") != generator_sha:
        raise SystemExit(f"generator source binding mismatch:{index}")
    if item.get("training_approved") is not False or item.get("structured_output_deferred") is not True:
        raise SystemExit(f"review-only/single-Tensor decision mismatch:{index}")
    if item.get("final_output_contract") != {"kind": "single_tensor", "finite_required": True}:
        raise SystemExit(f"final output contract mismatch:{index}")
    if not isinstance(code, str) or hashlib.sha256(code.encode()).hexdigest() != item.get("reference_sha256"):
        raise SystemExit(f"reference hash mismatch:{index}")
    if not isinstance(uuid, str) or not uuid or uuid in known:
        raise SystemExit(f"invalid/duplicate UUID:{index}")
    known.add(uuid)
    row_index_by_uuid[uuid] = index
if not allowed or len(allowed) != len(set(allowed)) or not set(allowed).issubset(known):
    raise SystemExit("allowlist must be a non-empty unique subset of candidate UUIDs")
if len(allowed) < gpu_count or len(allowed) > expected:
    raise SystemExit(f"allowlist count must be in [{gpu_count},{expected}]:{len(allowed)}")
missing_shards = [shard for shard in range(gpu_count) if not any(row_index_by_uuid[uuid] % gpu_count == shard for uuid in allowed)]
if missing_shards:
    raise SystemExit(f"allowlist would leave liveness shards empty:{missing_shards}")
# Run the adapter's full CPU-only manifest contract before acquiring the GPU
# lock or creating CUDA contexts.  This catches a generator/adapter schema
# drift (notably mode behavior) as a static launch refusal rather than an
# eight-GPU failed liveness run.
tasks = adapter._tasks(candidates, manifest)
if len(tasks) != expected or {task["uuid"] for task in tasks} != known:
    raise SystemExit("adapter static task replay does not exactly match candidates")
for row, item in zip(rows, manifests, strict=True):
    replay = generator.replay_manifest(item)
    code = row["reward_model"]["ground_truth"]
    if (
        code != replay["code"]
        or item.get("declared_ops") != replay["declared_ops"]
        or item.get("coverage_labels") != replay["coverage_labels"]
        or item.get("static_proof") != replay["static_proof"]
    ):
        raise SystemExit(f"generator static replay mismatch:{item['uuid']}")
    spec = generator.TEMPLATE_BY_ID[item["template_id"]]
    generator._enforce_spec(code, spec, replay["static_proof"])
print(json.dumps({"candidate_rows": len(rows), "allowlisted": len(allowed)}, sort_keys=True))
PY

mkdir -p "${run_dir}"
command -v flock >/dev/null 2>&1 || die "flock is required"
exec {run_lock_fd}>"${run_dir}/.validation.lock"
flock -n "${run_lock_fd}" || die "another kernelbench-gap launcher owns ${run_dir}"
exec {global_lock_fd}>/tmp/prompt_tvm_v4_reference_validation_all_visible_gpus.lock
flock -n "${global_lock_fd}" || die "another reference/liveness launcher owns the visible GPU set"

for source_path in "${launcher_path}" "${adapter_path}" "${generator_path}" "${runtime_core_path}"; do
  case ${source_path} in
    "${launcher_path}") archive_path=${run_dir}/launcher_source.sh ;;
    "${adapter_path}") archive_path=${run_dir}/adapter_source.py ;;
    "${generator_path}") archive_path=${run_dir}/generator_source.py ;;
    *) archive_path=${run_dir}/shared_runtime_core_source.py ;;
  esac
  if [[ -e ${archive_path} ]]; then
    [[ -f ${archive_path} ]] && cmp -s "${source_path}" "${archive_path}" || die "source archive mismatch: ${archive_path}"
  else
    cp -- "${source_path}" "${archive_path}.tmp.$$"
    mv -- "${archive_path}.tmp.$$" "${archive_path}"
  fi
done

scheduler_contract=${run_dir}/scheduler-contract.json
scheduler_tmp=${scheduler_contract}.tmp.$$
python3 - "${scheduler_tmp}" \
  "${launcher_path}" "${launcher_sha256}" "$(realpath "${adapter_path}")" "${adapter_sha256}" \
  "$(realpath "${runtime_core_path}")" "${runtime_core_sha256}" "$(realpath "${generator_path}")" "${generator_sha256}" \
  "$(realpath "${candidates_path}")" "${candidates_sha256}" "$(realpath "${manifest_path}")" "${manifest_sha256}" \
  "$(realpath "${allowlist_path}")" "${allowlist_sha256}" "${gpu_count}" "${trials}" "${seed}" "${timeout_seconds}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

(
    output, launcher_path, launcher_sha, adapter_path, adapter_sha, runtime_path, runtime_sha, generator_path, generator_sha,
    candidates_path, candidates_sha, manifest_path, manifest_sha, allowlist_path, allowlist_sha, shards, trials, seed, timeout,
) = sys.argv[1:]
payload = {
    "contract_version": "kernelbench_gap_liveness_scheduler_v1",
    "source_binding": {
        "adapter_source": {"path": adapter_path, "sha256": adapter_sha},
        "shared_runtime_core_source": {"path": runtime_path, "sha256": runtime_sha},
        "generator_source": {"path": generator_path, "sha256": generator_sha},
        "launcher_source": {"path": launcher_path, "sha256": launcher_sha},
    },
    "candidates": {"path": candidates_path, "sha256": candidates_sha},
    "manifest": {"path": manifest_path, "sha256": manifest_sha},
    "allowlist": {"path": allowlist_path, "sha256": allowlist_sha},
    "shard_count": int(shards),
    "trials": int(trials),
    "seed": int(seed),
    "timeout_seconds": float(timeout),
    "max_device_memory_gib": 64.0,
    "persistent_train_mode_models": True,
    "single_tensor_final_output": True,
    "execution_controls": {"control_trace_comparison": "exact", "cudnn_benchmark": False, "cudnn_deterministic": True},
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
if [[ -e ${scheduler_contract} ]]; then
  [[ -f ${scheduler_contract} ]] && cmp -s "${scheduler_tmp}" "${scheduler_contract}" || {
    rm -f -- "${scheduler_tmp}"
    die "scheduler contract mismatch on resume"
  }
  rm -f -- "${scheduler_tmp}"
else
  mv -- "${scheduler_tmp}" "${scheduler_contract}"
fi

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"
mapfile -t gpu_rows < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
(( ${#gpu_rows[@]} == gpu_count )) || die "expected exactly ${gpu_count} visible GPUs, found ${#gpu_rows[@]}"
for row in "${gpu_rows[@]}"; do
  IFS=',' read -r index memory utilization <<<"${row}"
  memory=${memory//[[:space:]]/}
  utilization=${utilization//[[:space:]]/}
  [[ ${memory} =~ ^[0-9]+$ && ${utilization} =~ ^[0-9]+$ ]] || die "unparseable nvidia-smi row:${row}"
  (( memory <= idle_memory_mib && utilization == 0 )) || die "GPU not idle:${row}"
done
compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z ${compute_pids//[[:space:]]/} ]] || die "visible GPUs have active compute processes"

cudnn_lib=/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-load-inline/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib
[[ -d ${cudnn_lib} ]] && export LD_LIBRARY_PATH="${cudnn_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
for (( gpu=0; gpu<gpu_count; gpu++ )); do
  CUDA_VISIBLE_DEVICES=${gpu} python3 -c 'import torch; torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False; x=torch.randn((2,3,16,16),device="cuda"); w=torch.randn((4,3,3,3),device="cuda"); torch.nn.functional.conv2d(x,w); torch.cuda.synchronize(); print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} cudnn={torch.backends.cudnn.version()} deterministic={torch.backends.cudnn.deterministic} benchmark={torch.backends.cudnn.benchmark}")' || die "PyTorch/cuDNN liveness preflight failed on physical GPU ${gpu}"
done

pids=()
cleanup_children() {
  local pid
  trap - INT TERM HUP
  for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
}
trap 'cleanup_children; exit 130' INT TERM HUP

for (( gpu=0; gpu<gpu_count; gpu++ )); do
  output=${run_dir}/shard-$(printf '%02d' "${gpu}")-of-$(printf '%02d' "${gpu_count}").jsonl
  log=${run_dir}/shard-$(printf '%02d' "${gpu}")-of-$(printf '%02d' "${gpu_count}").log
  CUDA_VISIBLE_DEVICES=${gpu} python3 "${adapter_path}" "${candidates_path}" "${manifest_path}" "${output}" \
    --uuid-file "${allowlist_path}" --device cuda:0 --trials "${trials}" --seed "${seed}" \
    --timeout-seconds "${timeout_seconds}" --launcher-sha256 "${launcher_sha256}" \
    --launcher-source-path "${launcher_path}" --shard-index "${gpu}" --shard-count "${gpu_count}" >"${log}" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=2
done
(( status == 0 )) || die "one or more kernelbench-gap liveness shards had an infrastructure failure; raw shard records were retained"

python3 - "${run_dir}" "${gpu_count}" "${launcher_sha256}" "${adapter_sha256}" "${runtime_core_sha256}" "${generator_sha256}" "${allowlist_path}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

run, count, launcher, adapter, runtime, generator, allowlist = (
    Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], Path(sys.argv[7])
)
summaries = []
for index in range(count):
    path = run / f"shard-{index:02d}-of-{count:02d}.log"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise SystemExit(f"empty shard log:{path}")
    record = json.loads(lines[-1])
    if record.get("contract_version") != "kernelbench_gap_runtime_liveness_v1":
        raise SystemExit(f"bad shard contract:{path}")
    if record.get("shard_index") != index or record.get("shard_count") != count:
        raise SystemExit(f"bad shard summary:{path}")
    source = record.get("source_binding")
    if not isinstance(source, dict) or source.get("adapter_source", {}).get("sha256") != adapter:
        raise SystemExit(f"adapter source binding mismatch:{path}")
    if source.get("shared_runtime_core_source", {}).get("sha256") != runtime:
        raise SystemExit(f"runtime-core source binding mismatch:{path}")
    if source.get("generator_source", {}).get("sha256") != generator:
        raise SystemExit(f"generator source binding mismatch:{path}")
    if source.get("launcher_source", {}).get("sha256") != launcher:
        raise SystemExit(f"launcher source binding mismatch:{path}")
    summaries.append(record)
bindings = {record.get("validation_binding_sha256") for record in summaries}
if len(bindings) != 1 or not isinstance(next(iter(bindings)), str):
    raise SystemExit("mixed validation bindings across shards")
payload = {
    "contract_version": "kernelbench_gap_liveness_launcher_summary_v1",
    "validation_binding_sha256": next(iter(bindings)),
    "source_binding": summaries[0]["source_binding"],
    "shard_count": count,
    "selected": sum(item["selected"] for item in summaries),
    "executed": sum(item["executed"] for item in summaries),
    "resumed": sum(item["resumed"] for item in summaries),
    "passed": sum(item["passed"] for item in summaries),
    "failed": sum(item["failed"] for item in summaries),
    "shards": summaries,
}
allowed = [line.strip() for line in allowlist.read_text(encoding="utf-8").splitlines() if line.strip()]
if payload["selected"] != len(allowed) or payload["selected"] != payload["executed"] + payload["resumed"] or payload["selected"] != payload["passed"] + payload["failed"]:
    raise SystemExit("aggregate shard totals do not match allowlist/resume accounting")
(run / "launcher-summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
