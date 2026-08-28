#!/usr/bin/env bash
# Run one bounded, source-bound layout-liveness lane on the visible GPUs.
set -euo pipefail

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
trials=${SYNTH_LAYOUT_LIVENESS_TRIALS:-3}
seed=${SYNTH_LAYOUT_LIVENESS_SEED:-17}
timeout_seconds=${SYNTH_TIMEOUT_SECONDS:-600}
idle_memory_mib=${SYNTH_IDLE_MEMORY_MIB:-64}
validator_path=tools/data/synthesize/layout_method/validate_layout_liveness.py
solver_path=tools/data/synthesize/layout_method/solve_layout_coverage.py
dependency_path=tools/data/synthesize/augment_prompt_tasks.py
pids=()

die() { echo "$*" >&2; exit 2; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

for path in "${parents_path}" "${children_path}" "${manifest_path}" "${allowlist_path}" "${validator_path}" "${solver_path}" "${dependency_path}"; do
  [[ -f ${path} ]] || die "required input is missing: ${path}"
done
[[ ${gpus_per_machine} =~ ^[1-9][0-9]*$ ]] || die "SYNTH_GPUS_PER_MACHINE must be a positive integer"
[[ ${machine_count} =~ ^[1-9][0-9]*$ ]] || die "SYNTH_MACHINE_COUNT must be a positive integer"
[[ ${machine_rank} =~ ^[0-9]+$ ]] && (( machine_rank < machine_count )) || die "SYNTH_MACHINE_RANK must be in [0,$((machine_count - 1))]"
[[ ${virtual_shards_per_gpu} =~ ^[1-9][0-9]*$ ]] || die "SYNTH_VIRTUAL_SHARDS_PER_GPU must be a positive integer"
[[ ${trials} =~ ^[1-9][0-9]*$ ]] && (( trials >= 3 )) || die "SYNTH_LAYOUT_LIVENESS_TRIALS must be an integer of at least 3"
[[ ${seed} == 17 ]] || die "SYNTH_LAYOUT_LIVENESS_SEED must equal frozen value 17"
[[ ${idle_memory_mib} =~ ^[0-9]+$ ]] || die "SYNTH_IDLE_MEMORY_MIB must be a non-negative integer"
awk -v value="${timeout_seconds}" 'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0) }' || die "SYNTH_TIMEOUT_SECONDS must be positive"

parents_sha256=$(sha256_file "${parents_path}")
children_sha256=$(sha256_file "${children_path}")
manifest_sha256=$(sha256_file "${manifest_path}")
allowlist_sha256=$(sha256_file "${allowlist_path}")
validator_sha256=$(sha256_file "${validator_path}")
solver_sha256=$(sha256_file "${solver_path}")
dependency_sha256=$(sha256_file "${dependency_path}")
launcher_sha256=$(sha256_file "$0")
for label in PARENTS CHILDREN MANIFEST ALLOWLIST; do
  lower=${label,,}
  actual_var=${lower}_sha256
  expected_var=SYNTH_EXPECTED_${label}_SHA256
  actual=${!actual_var}
  expected=${!expected_var:-}
  [[ -z ${expected} || ${actual} == "${expected}" ]] || die "${lower} SHA-256 mismatch: expected ${expected}, found ${actual}"
done

physical_shard_count=$((machine_count * gpus_per_machine))
shard_count=$((physical_shard_count * virtual_shards_per_gpu))

# Refuse artifacts outside the immutable layout static contract before GPUs are
# touched.  Exact child replay remains the analyzer's gate after reference data
# exists; this preflight proves the lane's canonical and family binding.
if ! source_git_commit=$(python - "${parents_path}" "${children_path}" "${manifest_path}" "${allowlist_path}" "$0" "${validator_path}" <<'PY'
from __future__ import annotations
import json, re, sys
from pathlib import Path
import pyarrow.parquet as pq
from tools.data.synthesize.layout_method import solve_layout_coverage as solver
from tools.data.synthesize.serial_source_contract import resolve_lane_serial_source

parents, children, manifest, allowlist, launcher, validator = map(Path, sys.argv[1:])
parent_rows = pq.ParquetFile(parents).metadata.num_rows
child_rows = pq.ParquetFile(children).metadata.num_rows
if not 1 <= child_rows <= solver.MAX_AUTHORIZED_CANDIDATES or parent_rows != child_rows:
    raise SystemExit(f"invalid bounded parent/child rows:{parent_rows}:{child_rows}")
rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != child_rows:
    raise SystemExit("manifest row count differs from candidates")
serial_source = resolve_lane_serial_source(rows)
if serial_source is None or serial_source[0].get("stage") != "dtype_fallback_base":
    raise SystemExit("layout lane must bind a gated dtype_fallback_base")
serial_binding = serial_source[0]
families = set(solver.LAYOUT_FAMILIES)
generator = Path(solver.__file__).resolve()
dependency = generator.parent.parent / "augment_prompt_tasks.py"
generator_sha = solver._sha256_file(generator)
dependency_sha = solver._sha256_file(dependency)
commits = {row.get("git_commit") for row in rows}
if len(commits) != 1:
    raise SystemExit("manifest does not bind exactly one Git commit")
commit = next(iter(commits))
if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
    raise SystemExit("manifest Git commit is invalid")
sources = {
    "solver": generator,
    "dependency": dependency,
    "validator": validator.resolve(),
    "launcher": launcher.resolve(),
}
for index, row in enumerate(rows):
    if (row.get("candidate_row_index") != index or row.get("manifest_contract_version") != "layout_lane_manifest_v1"
            or row.get("primary_intervention") != "layout" or row.get("assigned_family") not in families
            or row.get("source_artifact_path") != serial_binding["source_artifact_path"]
            or row.get("source_artifact_sha256") != serial_binding["source_artifact_sha256"]
            or row.get("source_row_index") is None or row.get("training_approved") is not False):
        raise SystemExit(f"static layout binding failed at manifest row {index}")
    family = row["assigned_family"]
    total = row.get("factory_count")
    transformed = row.get("transformed_factory_count")
    targets = row.get("target_factory_indices")
    expected = row.get("expected_layout_metadata")
    scope = row.get("layout_application_scope")
    realized = row.get("realized_intervention")
    if (type(total) is not int or type(transformed) is not int or not 0 < transformed <= total
            or not isinstance(targets, list) or any(type(item) is not int for item in targets)
            or targets != sorted(set(targets)) or any(not 0 <= item < total for item in targets)
            or not isinstance(expected, list) or len(expected) != transformed
            or [item.get("factory_index") for item in expected] != targets
            or not isinstance(realized, dict) or realized.get("factory_count") != total
            or realized.get("transformed_factory_count") != transformed
            or realized.get("target_factory_indices") != targets
            or realized.get("layout_application_scope") != scope):
        raise SystemExit(f"layout target factory schema failed at manifest row {index}")
    if family == "expand_zero_stride":
        if scope != "eligible_constant_factory_leaves_only":
            raise SystemExit(f"expand partial-leaf scope is not declared at manifest row {index}")
    elif scope != "all_direct_factory_leaves" or transformed != total:
        raise SystemExit(f"non-expand family is not all-leaf at manifest row {index}")
    if (row.get("git_commit") != commit
            or row.get("generator_contract_version") != solver.SERIAL_CONTRACT_VERSION
            or row.get("generator_source_sha256") != generator_sha
            or row.get("dependency_source_sha256") != dependency_sha):
        raise SystemExit(f"source-bound solver contract failed at manifest row {index}")
uuids = [line.strip() for line in allowlist.read_text(encoding="utf-8").splitlines() if line.strip()]
if not uuids or len(uuids) != len(set(uuids)) or not set(uuids).issubset({row["child_uuid"] for row in rows}):
    raise SystemExit("allowlist is not a unique candidate-child subset")
print(commit)
PY
); then
  die "layout static/source preflight failed"
fi

mkdir -p "${run_dir}"
command -v flock >/dev/null 2>&1 || die "flock is required for exclusive validation launch"
exec {run_lock_fd}>"${run_dir}/.validation.rank-${machine_rank}.lock"
flock -n "${run_lock_fd}" || die "another layout liveness launcher owns rank ${machine_rank} under ${run_dir}"
exec {global_lock_fd}>/tmp/prompt_tvm_v4_reference_validation_all_visible_gpus.lock
flock -n "${global_lock_fd}" || die "another validation launcher owns the node-visible GPU set"
cleanup_children() {
  local pid
  trap - INT TERM HUP
  for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
}
trap 'cleanup_children; exit 130' INT TERM HUP

for source_path in "$0" "${validator_path}" "${solver_path}" "${dependency_path}"; do
  case ${source_path} in
    "$0") archive_path=${run_dir}/launcher_source.sh ;;
    "${validator_path}") archive_path=${run_dir}/validator_source.py ;;
    "${solver_path}") archive_path=${run_dir}/solver_source.py ;;
    *) archive_path=${run_dir}/dependency_source.py ;;
  esac
  if [[ -e ${archive_path} ]]; then
    [[ -f ${archive_path} ]] && cmp -s "${source_path}" "${archive_path}" || die "source archive mismatch: ${archive_path}"
  else
    cp -- "${source_path}" "${archive_path}.tmp.$$" && mv -- "${archive_path}.tmp.$$" "${archive_path}"
  fi
done

scheduler_contract=${run_dir}/scheduler-contract-rank-${machine_rank}.json
scheduler_tmp=${scheduler_contract}.tmp.$$
python - "${scheduler_tmp}" "${launcher_sha256}" "${validator_sha256}" "${solver_sha256}" "${dependency_sha256}" "${source_git_commit}" \
  "$(realpath "${parents_path}")" "${parents_sha256}" "$(realpath "${children_path}")" "${children_sha256}" \
  "$(realpath "${manifest_path}")" "${manifest_sha256}" "$(realpath "${allowlist_path}")" "${allowlist_sha256}" \
  "${machine_rank}" "${machine_count}" "${gpus_per_machine}" "${virtual_shards_per_gpu}" "${shard_count}" \
  "${trials}" "${seed}" "${timeout_seconds}" <<'PY'
from __future__ import annotations
import json, sys
from pathlib import Path
(out, launcher, validator, solver, dependency, source_git_commit, pp, ph, cp, ch, mp, mh, ap, ah,
 machine_rank, machine_count, gpus_per_machine, virtual_shards_per_gpu, shards, trials, seed, timeout) = sys.argv[1:]
payload = {"contract_version":"layout_runtime_liveness_scheduler_v2", "launcher_source_sha256":launcher,
 "validator_source_sha256":validator, "solver_source_sha256":solver, "dependency_source_sha256":dependency,
 "source_git_commit":source_git_commit, "parents_path":pp,"parents_sha256":ph,
 "children_path":cp,"children_sha256":ch,"manifest_path":mp,"manifest_sha256":mh,"allowlist_path":ap,
 "allowlist_sha256":ah,"machine_rank":int(machine_rank),"machine_count":int(machine_count),
 "gpus_per_machine":int(gpus_per_machine),"virtual_shards_per_gpu":int(virtual_shards_per_gpu),
 "shard_count":int(shards),"trials":int(trials),"seed":int(seed),
 "timeout_seconds":float(timeout),"max_device_memory_gib":64.0}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n", encoding="utf-8")
PY
if [[ -e ${scheduler_contract} ]]; then
  [[ -f ${scheduler_contract} ]] && cmp -s "${scheduler_tmp}" "${scheduler_contract}" || { rm -f -- "${scheduler_tmp}"; die "scheduler contract mismatch on resume"; }
  rm -f -- "${scheduler_tmp}"
else mv -- "${scheduler_tmp}" "${scheduler_contract}"; fi

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"
mapfile -t gpu_rows < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits) || die "failed to query GPUs"
(( ${#gpu_rows[@]} == gpus_per_machine )) || die "expected ${gpus_per_machine} visible GPUs, found ${#gpu_rows[@]}"
for row in "${gpu_rows[@]}"; do
  IFS=',' read -r index memory utilization <<<"${row}"
  memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
  [[ ${memory} =~ ^[0-9]+$ && ${utilization} =~ ^[0-9]+$ ]] || die "unparseable nvidia-smi row:${row}"
  (( memory <= idle_memory_mib && utilization == 0 )) || die "GPU not idle:${row}"
done
compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) || die "failed to query GPU compute processes"
[[ -z ${compute_pids//[[:space:]]/} ]] || die "visible GPUs have active compute processes:${compute_pids//$'\n'/,}"
# .22 system Python needs the matching cuDNN wheel directory for real kernels.
cudnn_lib=/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-load-inline/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib
[[ -d ${cudnn_lib} ]] && export LD_LIBRARY_PATH="${cudnn_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
CUDA_VISIBLE_DEVICES=0 python -c 'import torch; x=torch.randn((2,3,16,16),device="cuda"); w=torch.randn((4,3,3,3),device="cuda"); torch.nn.functional.conv2d(x,w); print(f"torch={torch.__version__} cudnn={torch.backends.cudnn.version()}")' || die "PyTorch/cuDNN liveness preflight failed"

queue_dir=$(mktemp -d "/tmp/prompt_tvm_v4_layout_liveness_queue.rank-${machine_rank}.XXXXXX") || exit 2
cleanup_queue_dir() {
  find "${queue_dir}" -mindepth 1 -maxdepth 1 -type d -name 'claim-*' -exec rmdir -- {} + 2>/dev/null || true
  rmdir -- "${queue_dir}" 2>/dev/null || true
}
claim_next_shard() {
  local virtual_offset local_slot shard_id
  for ((virtual_offset=0; virtual_offset<virtual_shards_per_gpu; virtual_offset++)); do
    for ((local_slot=0; local_slot<gpus_per_machine; local_slot++)); do
      shard_id=$((machine_rank * gpus_per_machine + local_slot + virtual_offset * physical_shard_count))
      if mkdir "${queue_dir}/claim-${shard_id}" 2>/dev/null; then printf '%s\n' "${shard_id}"; return 0; fi
    done
  done
  return 1
}
run_gpu_queue() {
  local gpu=$1 shard_id output log validator_pid=''
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
    output=${run_dir}/shard-$(printf '%02d' "${shard_id}")-of-$(printf '%02d' "${shard_count}").jsonl
    log=${run_dir}/shard-$(printf '%02d' "${shard_id}")-of-$(printf '%02d' "${shard_count}").log
    CUDA_VISIBLE_DEVICES=${gpu} python "${validator_path}" "${parents_path}" "${children_path}" "${manifest_path}" "${output}" \
      --child-uuid-file "${allowlist_path}" --device cuda:0 --trials "${trials}" --seed "${seed}" \
      --timeout-seconds "${timeout_seconds}" --launcher-sha256 "${launcher_sha256}" \
      --shard-index "${shard_id}" --shard-count "${shard_count}" >"${log}" 2>&1 &
    validator_pid=$!
    if ! wait "${validator_pid}"; then return 2; fi
    validator_pid=''
  done
  trap - INT TERM HUP
}
for ((gpu=0; gpu<gpus_per_machine; gpu++)); do run_gpu_queue "${gpu}" & pids+=("$!"); done
status=0; for pid in "${pids[@]}"; do wait "${pid}" || status=2; done
trap - INT TERM HUP
cleanup_queue_dir
(( status == 0 )) || die "one or more layout-liveness shards had an infrastructure failure"

python - "${run_dir}" "${machine_rank}" "${machine_count}" "${gpus_per_machine}" "${virtual_shards_per_gpu}" \
  "${shard_count}" "${launcher_sha256}" "${validator_sha256}" "${allowlist_path}" <<'PY'
from __future__ import annotations
import json, sys
from pathlib import Path
run = Path(sys.argv[1]); machine_rank = int(sys.argv[2]); machine_count = int(sys.argv[3])
gpus_per_machine = int(sys.argv[4]); virtual_shards_per_gpu = int(sys.argv[5]); count = int(sys.argv[6])
launcher, validator, allowlist = sys.argv[7], sys.argv[8], Path(sys.argv[9])
physical_shard_count = machine_count * gpus_per_machine
summaries=[]
for local_slot in range(gpus_per_machine):
    base = machine_rank * gpus_per_machine + local_slot
    for virtual_offset in range(virtual_shards_per_gpu):
        index = base + virtual_offset * physical_shard_count
        path=run/f"shard-{index:02d}-of-{count:02d}.log"
        lines=[line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines: raise SystemExit(f"empty shard log:{path}")
        record=json.loads(lines[-1])
        if record.get("shard_index") != index or record.get("shard_count") != count: raise SystemExit(f"bad shard summary:{path}")
        if record.get("launcher_source_sha256") != launcher or record.get("validator_source_sha256") != validator: raise SystemExit(f"source binding mismatch:{path}")
        summaries.append(record)
bindings={record.get("validation_binding_sha256") for record in summaries}
if len(bindings) != 1 or not isinstance(next(iter(bindings)), str): raise SystemExit("mixed validation bindings across shard summaries")
payload={"contract_version":"layout_runtime_liveness_launcher_summary_v2","launcher_source_sha256":launcher,"validator_source_sha256":validator,"validation_binding_sha256":next(iter(bindings)),"machine_rank":machine_rank,"machine_count":machine_count,"gpus_per_machine":gpus_per_machine,"virtual_shards_per_gpu":virtual_shards_per_gpu,"shard_count":count,"selected":sum(x["selected"] for x in summaries),"executed":sum(x["executed"] for x in summaries),"resumed":sum(x["resumed"] for x in summaries),"passed":sum(x["passed"] for x in summaries),"failed":sum(x["failed"] for x in summaries),"shards":summaries}
allowed=[line.strip() for line in allowlist.read_text(encoding="utf-8").splitlines() if line.strip()]
expected_rank_selected=sum(1 for index in range(len(allowed)) if index % count in {machine_rank*gpus_per_machine+slot+offset*physical_shard_count for slot in range(gpus_per_machine) for offset in range(virtual_shards_per_gpu)})
if payload["selected"] != expected_rank_selected or payload["selected"] != payload["executed"] + payload["resumed"] or payload["selected"] != payload["passed"] + payload["failed"]:
    raise SystemExit("aggregate layout shard totals do not match allowlist/resume accounting")
(run/f"launcher-summary-rank-{machine_rank}.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps(payload,sort_keys=True))
PY
