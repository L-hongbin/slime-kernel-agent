#!/usr/bin/env bash
set -euo pipefail

if (( $# != 4 )); then
  echo "usage: $0 CANDIDATES_PARQUET MANIFEST_JSONL UUID_ALLOWLIST RUN_DIR" >&2
  exit 2
fi

candidates_path=$1
manifest_path=$2
allowlist_path=$3
run_dir=$4
gpu_count=${SEMANTIC_GPUS_PER_MACHINE:-8}
trials=${SEMANTIC_LIVENESS_TRIALS:-3}
seed=${SEMANTIC_LIVENESS_SEED:-17}
timeout_seconds=${SEMANTIC_LIVENESS_TIMEOUT_SECONDS:-600}
idle_memory_mib=${SEMANTIC_IDLE_MEMORY_MIB:-128}
validator_path=tools/data/synthesize/semantic_operator_method/validate_semantic_liveness.py
generator_path=tools/data/synthesize/semantic_operator_method/generate_semantic_operator.py

die() {
  echo "$*" >&2
  exit 2
}

[[ -f ${candidates_path} ]] || die "candidates artifact missing: ${candidates_path}"
[[ -f ${manifest_path} ]] || die "manifest artifact missing: ${manifest_path}"
[[ -f ${allowlist_path} ]] || die "UUID allowlist missing: ${allowlist_path}"
[[ -f ${validator_path} && -f ${generator_path} ]] || die "semantic pipeline source missing"
[[ ${gpu_count} =~ ^[1-9][0-9]*$ ]] || die "SEMANTIC_GPUS_PER_MACHINE must be positive"
[[ ${trials} =~ ^[0-9]+$ ]] && (( trials >= 3 )) || die "SEMANTIC_LIVENESS_TRIALS must be at least 3"
[[ ${seed} == 17 ]] || die "SEMANTIC_LIVENESS_SEED is fixed to 17"
[[ ${idle_memory_mib} =~ ^[0-9]+$ ]] || die "SEMANTIC_IDLE_MEMORY_MIB must be non-negative"

launcher_sha256=$(sha256sum "$0" | awk '{print $1}')
validator_sha256=$(sha256sum "${validator_path}" | awk '{print $1}')
generator_sha256=$(sha256sum "${generator_path}" | awk '{print $1}')
candidates_sha256=$(sha256sum "${candidates_path}" | awk '{print $1}')
manifest_sha256=$(sha256sum "${manifest_path}" | awk '{print $1}')
allowlist_sha256=$(sha256sum "${allowlist_path}" | awk '{print $1}')

python3 - "${candidates_path}" "${manifest_path}" "${allowlist_path}" "${generator_sha256}" "${gpu_count}" <<'PY'
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import pyarrow.parquet as pq

candidates, manifest, allowlist, generator_sha, gpu_count = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], int(sys.argv[5])
rows = pq.read_table(candidates).to_pylist()
manifests = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
allowed = [line.strip() for line in allowlist.read_text(encoding="utf-8").splitlines() if line.strip()]
if not 1 <= len(rows) <= 1000 or len(rows) != len(manifests):
    raise SystemExit(f"semantic canary must contain aligned 1..1000 rows:{len(rows)}:{len(manifests)}")
known = set()
for index, (row, item) in enumerate(zip(rows, manifests, strict=True)):
    uuid = row.get("extra_info", {}).get("uuid")
    code = row.get("reward_model", {}).get("ground_truth")
    if item.get("candidate_row_index") != index or item.get("uuid") != uuid:
        raise SystemExit(f"candidate/manifest identity mismatch:{index}")
    if item.get("manifest_contract_version") != "semantic_operator_parentless_manifest_v1":
        raise SystemExit(f"manifest contract mismatch:{index}")
    if item.get("parent_uuid") is not None or item.get("lineage_kind") != "standalone_semantic_synthetic":
        raise SystemExit(f"semantic row is not parentless:{index}")
    if item.get("generator_source_sha256") != generator_sha:
        raise SystemExit(f"generator source binding mismatch:{index}")
    if item.get("training_approved") is not False or item.get("structured_output_deferred") is not True:
        raise SystemExit(f"review-only/single-Tensor decision mismatch:{index}")
    if item.get("final_output_contract") != {"kind":"single_tensor","finite_required":True}:
        raise SystemExit(f"final output contract mismatch:{index}")
    if not isinstance(code, str) or hashlib.sha256(code.encode()).hexdigest() != item.get("reference_sha256"):
        raise SystemExit(f"reference hash mismatch:{index}")
    known.add(uuid)
if not allowed or len(allowed) != len(set(allowed)) or not set(allowed).issubset(known):
    raise SystemExit("allowlist must be a non-empty unique subset of candidate UUIDs")
if len(allowed) > 1000 or len(allowed) < gpu_count:
    raise SystemExit(f"allowlist count must be in [{gpu_count},1000]:{len(allowed)}")
print(json.dumps({"candidate_rows":len(rows),"allowlisted":len(allowed)}, sort_keys=True))
PY

mkdir -p "${run_dir}"
command -v flock >/dev/null 2>&1 || die "flock is required"
exec {run_lock_fd}>"${run_dir}/.validation.lock"
flock -n "${run_lock_fd}" || die "another semantic launcher owns ${run_dir}"
exec {global_lock_fd}>/tmp/prompt_tvm_v4_reference_validation_all_visible_gpus.lock
flock -n "${global_lock_fd}" || die "another validation launcher owns the visible GPU set"

for source_path in "$0" "${validator_path}" "${generator_path}"; do
  case ${source_path} in
    "$0") archive_path=${run_dir}/launcher_source.sh ;;
    "${validator_path}") archive_path=${run_dir}/validator_source.py ;;
    *) archive_path=${run_dir}/generator_source.py ;;
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
python3 - "${scheduler_tmp}" "${launcher_sha256}" "${validator_sha256}" "${generator_sha256}" \
  "$(realpath "${candidates_path}")" "${candidates_sha256}" "$(realpath "${manifest_path}")" "${manifest_sha256}" \
  "$(realpath "${allowlist_path}")" "${allowlist_sha256}" "${gpu_count}" "${trials}" "${seed}" "${timeout_seconds}" <<'PY'
from __future__ import annotations
import json, sys
from pathlib import Path
(out, launcher, validator, generator, cp, ch, mp, mh, ap, ah, shards, trials, seed, timeout) = sys.argv[1:]
payload = {
    "contract_version":"semantic_operator_liveness_scheduler_v1",
    "launcher_source_sha256":launcher,
    "validator_source_sha256":validator,
    "generator_source_sha256":generator,
    "candidates_path":cp,
    "candidates_sha256":ch,
    "manifest_path":mp,
    "manifest_sha256":mh,
    "allowlist_path":ap,
    "allowlist_sha256":ah,
    "shard_count":int(shards),
    "trials":int(trials),
    "seed":int(seed),
    "timeout_seconds":float(timeout),
    "max_device_memory_gib":64.0,
    "persistent_train_mode_models":True,
    "single_tensor_final_output":True,
    "execution_controls":{
        "control_trace_comparison":"exact",
        "cudnn_benchmark":False,
        "cudnn_deterministic":True,
    },
}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n", encoding="utf-8")
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
(( ${#gpu_rows[@]} == gpu_count )) || die "expected ${gpu_count} visible GPUs, found ${#gpu_rows[@]}"
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
CUDA_VISIBLE_DEVICES=0 python3 -c 'import torch; torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False; x=torch.randn((2,3,16,16),device="cuda"); w=torch.randn((4,3,3,3),device="cuda"); torch.nn.functional.conv2d(x,w); print(f"torch={torch.__version__} cudnn={torch.backends.cudnn.version()} deterministic={torch.backends.cudnn.deterministic} benchmark={torch.backends.cudnn.benchmark}")' || die "PyTorch/cuDNN liveness preflight failed"

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
  CUDA_VISIBLE_DEVICES=${gpu} python3 "${validator_path}" "${candidates_path}" "${manifest_path}" "${output}" \
    --uuid-file "${allowlist_path}" --device cuda:0 --trials "${trials}" --seed "${seed}" \
    --timeout-seconds "${timeout_seconds}" --launcher-sha256 "${launcher_sha256}" \
    --shard-index "${gpu}" --shard-count "${gpu_count}" >"${log}" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=2
done
(( status == 0 )) || die "one or more semantic-liveness shards had an infrastructure failure"

python3 - "${run_dir}" "${gpu_count}" "${launcher_sha256}" "${validator_sha256}" "${allowlist_path}" <<'PY'
from __future__ import annotations
import json, sys
from pathlib import Path
run, count, launcher, validator, allowlist = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], Path(sys.argv[5])
summaries=[]
for index in range(count):
    path=run/f"shard-{index:02d}-of-{count:02d}.log"
    lines=[line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise SystemExit(f"empty shard log:{path}")
    record=json.loads(lines[-1])
    if record.get("shard_index") != index or record.get("shard_count") != count:
        raise SystemExit(f"bad shard summary:{path}")
    if record.get("launcher_source_sha256") != launcher or record.get("validator_source_sha256") != validator:
        raise SystemExit(f"source binding mismatch:{path}")
    summaries.append(record)
bindings={record.get("validation_binding_sha256") for record in summaries}
if len(bindings) != 1 or not isinstance(next(iter(bindings)), str):
    raise SystemExit("mixed validation bindings across shards")
payload={
    "contract_version":"semantic_operator_liveness_launcher_summary_v1",
    "launcher_source_sha256":launcher,
    "validator_source_sha256":validator,
    "validation_binding_sha256":next(iter(bindings)),
    "shard_count":count,
    "selected":sum(item["selected"] for item in summaries),
    "executed":sum(item["executed"] for item in summaries),
    "resumed":sum(item["resumed"] for item in summaries),
    "passed":sum(item["passed"] for item in summaries),
    "failed":sum(item["failed"] for item in summaries),
    "execution_controls":{
        "control_trace_comparison":"exact",
        "cudnn_benchmark":False,
        "cudnn_deterministic":True,
    },
    "shards":summaries,
}
allowed=[line.strip() for line in allowlist.read_text(encoding="utf-8").splitlines() if line.strip()]
if payload["selected"] != len(allowed) or payload["selected"] != payload["executed"] + payload["resumed"] or payload["selected"] != payload["passed"] + payload["failed"]:
    raise SystemExit("aggregate shard totals do not match allowlist/resume accounting")
(run/"launcher-summary.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps(payload,sort_keys=True))
PY
