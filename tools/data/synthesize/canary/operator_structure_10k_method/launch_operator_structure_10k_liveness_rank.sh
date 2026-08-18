#!/usr/bin/env bash
# Fixed 1-host x 8-GPU operator/structure canary liveness run.
set -euo pipefail

if (( $# != 6 )); then
  echo "usage: $0 MACHINE_RANK CANDIDATES_PARQUET MANIFEST_JSONL REFERENCE_PASSED_UUIDS REFERENCE_PASSED_BINDING_JSON RANK_RUN_DIR" >&2
  exit 2
fi

rank=$1; candidates=$2; manifest=$3; allowlist=$4; reference_binding=$5; run_dir=$6
machines=1
gpus=${OPERATOR_STRUCTURE_10K_GPUS_PER_MACHINE:-8}
trials=${OPERATOR_STRUCTURE_10K_LIVENESS_TRIALS:-3}
seed=${OPERATOR_STRUCTURE_10K_LIVENESS_SEED:-17}
timeout=${OPERATOR_STRUCTURE_10K_LIVENESS_TIMEOUT_SECONDS:-600}
idle_mib=${OPERATOR_STRUCTURE_10K_IDLE_MEMORY_MIB:-128}
generator_module=${OPERATOR_STRUCTURE_10K_GENERATOR_MODULE:-tools.data.synthesize.canary.operator_structure_10k_method.generate_operator_structure_10k}
adapter=tools/data/synthesize/canary/operator_structure_10k_method/validate_operator_structure_10k_liveness.py
generator=tools/data/synthesize/canary/operator_structure_10k_method/generate_operator_structure_10k.py
core=tools/data/synthesize/semantic_operator_method/validate_semantic_liveness.py
runtime_validation=tools/data/cleaning/runtime_validation.py
train_mode_contract=tools/data/synthesize/validate_train_mode_contract.py
semantic_operator_generator=tools/data/synthesize/semantic_operator_method/generate_semantic_operator.py
die() { echo "operator-structure-10k liveness: $*" >&2; exit 2; }

[[ $rank =~ ^[0-9]+$ && $machines =~ ^[1-9][0-9]*$ && $gpus =~ ^[1-9][0-9]*$ ]] || die "invalid rank/machine/GPU count"
(( rank < machines )) || die "rank outside machine count"
(( rank == 0 && gpus == 8 )) || die "canary liveness topology must be 1x8"
[[ $trials =~ ^[0-9]+$ ]] && (( trials == 3 )) || die "trials must be exactly 3"
[[ $seed == 17 ]] || die "seed must be exactly 17"
[[ $timeout =~ ^[0-9]+([.][0-9]+)?$ && $idle_mib =~ ^[0-9]+$ ]] || die "invalid timeout or idle threshold"
[[ -f $candidates && -f $manifest && -f $allowlist && -f $reference_binding && -f $adapter && -f $generator && -f $core && -f $runtime_validation && -f $train_mode_contract && -f $semantic_operator_generator ]] || die "candidate/source closure missing"
total_shards=$((machines * gpus))
run_real=$(realpath -m "$run_dir")
case $run_real in *operator_structure_10k*) ;; *) die "RANK_RUN_DIR must be an explicit operator_structure_10k lane";; esac
case $run_real in *kernelbench_gap*|*semantic_operator*|*frontier_operator*|*random_method*|*dtype_method*|*layout_method*) die "RANK_RUN_DIR points at another lane";; esac

launcher=$(realpath "$0"); launcher_sha=$(sha256sum "$launcher" | awk '{print $1}')
adapter_sha=$(sha256sum "$adapter" | awk '{print $1}'); generator_sha=$(sha256sum "$generator" | awk '{print $1}'); core_sha=$(sha256sum "$core" | awk '{print $1}')
runtime_validation_sha=$(sha256sum "$runtime_validation" | awk '{print $1}'); train_mode_contract_sha=$(sha256sum "$train_mode_contract" | awk '{print $1}'); semantic_operator_generator_sha=$(sha256sum "$semantic_operator_generator" | awk '{print $1}')
candidates_sha=$(sha256sum "$candidates" | awk '{print $1}'); manifest_sha=$(sha256sum "$manifest" | awk '{print $1}'); allowlist_sha=$(sha256sum "$allowlist" | awk '{print $1}'); reference_binding_sha=$(sha256sum "$reference_binding" | awk '{print $1}')
execution_host=${OPERATOR_STRUCTURE_10K_EXECUTION_HOST:-"$(hostname):$(hostname -I | awk '{print $1}')"}
[[ -n $execution_host ]] || die "execution host is empty"

cache=$(mktemp -d /tmp/operator_structure_10k_liveness_pycache.XXXXXX); trap 'rm -rf -- "$cache"' EXIT
export PYTHONPYCACHEPREFIX=$cache

# Full static/replay check before locking or constructing a CUDA context.
python3 - "$candidates" "$manifest" "$allowlist" "$generator_module" "$total_shards" <<'PY'
import json, sys
from pathlib import Path
from tools.data.synthesize.canary.operator_structure_10k_method import validate_operator_structure_10k_liveness as a
candidates, manifest, allowlist, module, shard_count = Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4],int(sys.argv[5])
tasks=a.tasks(candidates, manifest, generator_module=module)
known={str(x['uuid']) for x in tasks}
allowed=[x.strip() for x in allowlist.read_text().splitlines() if x.strip()]
if not allowed or len(allowed)!=len(set(allowed)) or not set(allowed).issubset(known): raise SystemExit('invalid allowlist')
allowed_set=set(allowed)
empty=[i for i in range(shard_count) if not any(int(t['candidate_row_index'])%shard_count==i and str(t['uuid']) in allowed_set for t in tasks)]
if empty: raise SystemExit(f'allowlist leaves global shards empty:{empty}')
print(json.dumps({'candidate_rows':len(tasks),'allowlisted':len(allowed),'global_shards':shard_count},sort_keys=True))
PY

mkdir -p "$run_dir"
command -v flock >/dev/null || die "flock is required"
exec {rank_lock_fd}>"$run_dir/.validation.lock"; flock -n "$rank_lock_fd" || die "another rank launcher owns $run_dir"
exec {gpu_lock_fd}>/tmp/prompt_tvm_v4_reference_validation_all_visible_gpus.lock; flock -n "$gpu_lock_fd" || die "another reference/liveness launcher owns this node's GPUs"

for pair in "$launcher:launcher_source.sh" "$adapter:adapter_source.py" "$generator:generator_source.py" "$core:shared_runtime_core_source.py" "$runtime_validation:runtime_validation_source.py" "$train_mode_contract:train_mode_contract_source.py" "$semantic_operator_generator:semantic_operator_generator_source.py"; do
  source=${pair%%:*}; archive="$run_dir/${pair#*:}"
  if [[ -e $archive ]]; then [[ -f $archive ]] && cmp -s "$source" "$archive" || die "source archive mismatch:$archive"
  else cp -- "$source" "$archive.tmp.$$" && mv -- "$archive.tmp.$$" "$archive"; fi
done

command -v nvidia-smi >/dev/null || die "nvidia-smi is required"
mapfile -t gpu_rows < <(nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits)
(( ${#gpu_rows[@]} == gpus )) || die "expected exactly $gpus visible GPUs"
gpu_inventory=$(printf '%s\n' "${gpu_rows[@]}")
for row in "${gpu_rows[@]}"; do
  IFS=',' read -r index uuid name total memory util <<<"$row"; memory=${memory//[[:space:]]/}; util=${util//[[:space:]]/}
  [[ $name == *H20* && $memory =~ ^[0-9]+$ && $util =~ ^[0-9]+$ ]] || die "GPU must be an H20 with parseable state:$row"
  (( memory <= idle_mib && util == 0 )) || die "GPU not idle:$row"
done
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z ${apps//[[:space:]]/} ]] || die "visible GPUs have compute processes"

scheduler="$run_dir/scheduler-contract-rank-$rank.json"; tmp="$scheduler.tmp.$$.json"
python3 - "$tmp" "$rank" "$machines" "$gpus" "$total_shards" "$execution_host" "$gpu_inventory" "$launcher" "$launcher_sha" "$(realpath "$adapter")" "$adapter_sha" "$(realpath "$generator")" "$generator_sha" "$(realpath "$core")" "$core_sha" "$(realpath "$runtime_validation")" "$runtime_validation_sha" "$(realpath "$train_mode_contract")" "$train_mode_contract_sha" "$(realpath "$semantic_operator_generator")" "$semantic_operator_generator_sha" "$(realpath "$candidates")" "$candidates_sha" "$(realpath "$manifest")" "$manifest_sha" "$(realpath "$allowlist")" "$allowlist_sha" "$(realpath "$reference_binding")" "$reference_binding_sha" "$generator_module" "$trials" "$seed" "$timeout" <<'PY'
import json,sys
from pathlib import Path
(out,rank,machines,gpus,shards,host,inventory,launcher,launcher_sha,adapter,adapter_sha,generator,generator_sha,core,core_sha,runtime_validation,runtime_validation_sha,train_mode_contract,train_mode_contract_sha,semantic_operator_generator,semantic_operator_generator_sha,candidates,candidates_sha,manifest,manifest_sha,allowlist,allowlist_sha,reference_binding,reference_binding_sha,module,trials,seed,timeout)=sys.argv[1:]
payload={'contract_version':'operator_structure_10k_liveness_rank_scheduler_v1','machine_rank':int(rank),'machine_count':int(machines),'gpus_per_machine':int(gpus),'global_shard_count':int(shards),'global_shard_indices':list(range(int(rank)*int(gpus),(int(rank)+1)*int(gpus))),'execution_host':host,'gpu_inventory':[x for x in inventory.splitlines() if x],'source_binding':{'adapter_source':{'path':adapter,'sha256':adapter_sha},'shared_runtime_core_source':{'path':core,'sha256':core_sha},'runtime_validation_source':{'path':runtime_validation,'sha256':runtime_validation_sha},'train_mode_contract_source':{'path':train_mode_contract,'sha256':train_mode_contract_sha},'semantic_operator_generator_source':{'path':semantic_operator_generator,'sha256':semantic_operator_generator_sha},'generator_source':{'path':generator,'sha256':generator_sha},'launcher_source':{'path':launcher,'sha256':launcher_sha}},'candidates':{'path':candidates,'sha256':candidates_sha},'manifest':{'path':manifest,'sha256':manifest_sha},'allowlist':{'path':allowlist,'sha256':allowlist_sha},'reference_passed_binding':{'path':reference_binding,'sha256':reference_binding_sha},'generator_module':module,'trials':int(trials),'seed':int(seed),'timeout_seconds':float(timeout),'max_device_memory_gib':64.0,'persistent_train_mode_models':True,'single_tensor_final_output':True,'execution_controls':{'control_trace_comparison':'exact','cudnn_benchmark':False,'cudnn_deterministic':True}}
Path(out).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
if [[ -e $scheduler ]]; then cmp -s "$tmp" "$scheduler" || { rm -f -- "$tmp"; die "scheduler contract mismatch on resume"; }; rm -f -- "$tmp"; else mv -- "$tmp" "$scheduler"; fi

cudnn_lib=/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-load-inline/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib
[[ -d $cudnn_lib ]] && export LD_LIBRARY_PATH="$cudnn_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for ((gpu=0; gpu<gpus; gpu++)); do
  CUDA_VISIBLE_DEVICES=$gpu python3 -c 'import torch; torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False; x=torch.randn((2,3,16,16),device="cuda"); w=torch.randn((4,3,3,3),device="cuda"); torch.nn.functional.conv2d(x,w); torch.cuda.synchronize(); assert "H20" in torch.cuda.get_device_name(0); print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} cudnn={torch.backends.cudnn.version()}")' || die "PyTorch/cuDNN H20 preflight failed on GPU $gpu"
done
pids=(); cleanup() { trap - INT TERM HUP; for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; for p in "${pids[@]}"; do wait "$p" 2>/dev/null || true; done; exit 130; }; trap cleanup INT TERM HUP
for ((gpu=0; gpu<gpus; gpu++)); do
  shard=$((rank*gpus+gpu)); output="$run_dir/shard-$(printf '%02d' "$shard")-of-$(printf '%02d' "$total_shards").jsonl"; log="${output%.jsonl}.log"
  CUDA_VISIBLE_DEVICES=$gpu python3 "$adapter" "$candidates" "$manifest" "$output" --uuid-file "$allowlist" --reference-passed-binding "$reference_binding" --generator-module "$generator_module" --device cuda:0 --trials "$trials" --seed "$seed" --timeout-seconds "$timeout" --launcher-sha256 "$launcher_sha" --launcher-source-path "$launcher" --driver-cuda-visible-devices "$gpu" --execution-host "$execution_host" --shard-index "$shard" --shard-count "$total_shards" >"$log" 2>&1 & pids+=("$!")
done
status=0; for p in "${pids[@]}"; do wait "$p" || status=2; done; (( status == 0 )) || die "infrastructure failure; append-only raw records retained for resume"

python3 - "$run_dir" "$rank" "$machines" "$gpus" "$total_shards" "$execution_host" "$allowlist" <<'PY'
import json,sys
from pathlib import Path
run,rank,machines,gpus,shards,host,allowlist=Path(sys.argv[1]),*[int(x) for x in sys.argv[2:6]],sys.argv[6],Path(sys.argv[7])
summaries=[]
for shard in range(rank*gpus,(rank+1)*gpus):
 p=run/f'shard-{shard:02d}-of-{shards:02d}.log'; lines=p.read_text().splitlines() if p.is_file() else []
 if not lines: raise SystemExit(f'missing shard summary:{p}')
 x=json.loads(lines[-1])
 if x.get('contract_version')!='operator_structure_10k_runtime_liveness_v1' or x.get('execution_host')!=host or x.get('global_shard_index')!=shard or x.get('global_shard_count')!=shards: raise SystemExit(f'bad shard summary:{p}')
 summaries.append(x)
bindings={str(x['global_shard_index']):x.get('validation_binding_sha256') for x in summaries}
if set(bindings)!={str(x['global_shard_index']) for x in summaries} or not all(isinstance(value,str) and len(value)==64 for value in bindings.values()): raise SystemExit('invalid per-shard validation bindings')
payload={'contract_version':'operator_structure_10k_liveness_rank_summary_v1','machine_rank':rank,'machine_count':machines,'gpus_per_machine':gpus,'global_shard_count':shards,'execution_host':host,'validation_binding_by_shard':bindings,'source_binding':summaries[0]['source_binding'],'selected':sum(x['selected'] for x in summaries),'executed':sum(x['executed'] for x in summaries),'resumed':sum(x['resumed'] for x in summaries),'passed':sum(x['passed'] for x in summaries),'failed':sum(x['failed'] for x in summaries),'shards':summaries}
if payload['selected']!=payload['executed']+payload['resumed'] or payload['selected']!=payload['passed']+payload['failed']: raise SystemExit('rank total accounting mismatch')
(run/'launcher-summary.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,sort_keys=True))
PY
