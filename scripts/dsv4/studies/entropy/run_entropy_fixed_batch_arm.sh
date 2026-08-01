#!/bin/bash
# Fixed-batch entropy-causality arm for DeepSeek-V4-Flash.
#
# Usage:
#   DEBUG_DATA=/nfs/FM/.../rollout_0.pt \
#     bash scripts/dsv4/studies/entropy/run_entropy_fixed_batch_arm.sh \
#       token_predictive node64 [--prepare-only]
#
# node64 is paired with node53_dspark; node69 is paired with node70_dspark.
# The pairs are disjoint so two arms may run concurrently after the frozen
# dump has been copied to both nodes in each pair.
set -euo pipefail

ARM=${1:?usage: $0 ARM node64|node69 [--prepare-only] [--sequence-mis-config JSON]}
HEAD_PHYSICAL=${2:?usage: $0 ARM node64|node69 [--prepare-only] [--sequence-mis-config JSON]}
shift 2
MODE=
REQUESTED_SEQUENCE_MIS_CONFIG=
while (( "$#" > 0 )); do
  case "$1" in
    --prepare-only)
      MODE=--prepare-only
      shift
      ;;
    --sequence-mis-config)
      [[ "$#" -ge 2 ]] || { echo "FATAL: --sequence-mis-config requires a value" >&2; exit 2; }
      REQUESTED_SEQUENCE_MIS_CONFIG=$2
      shift 2
      ;;
    *)
      echo "FATAL: unknown argument '$1'." >&2
      exit 2
      ;;
  esac
done

REPO=${REPO:-/nfs/FM/chenshuailin/projects/kernel_agents/slime-v4flash-lora}
DEBUG_DATA=${DEBUG_DATA:?set DEBUG_DATA to the frozen rollout dump on both nodes of the selected pair}

# This diagnostic must not inherit a formal/smoke shell's hidden model or
# runtime variants. Clear every transported experiment-prefix variable; the
# resolved contract is rebuilt explicitly below.
while IFS= read -r inherited_name; do
  unset "${inherited_name}"
done < <(compgen -e | LC_ALL=C grep -E '^(V4_|SGLANG_|SLIME_)' || true)
unset NUM_ROLLOUT ROLLOUT_BATCH_SIZE LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE
unset RUN_ID SCRATCH LOG MAX_TURNS TIS_CLIP TIS_CLIP_LOW LR_DECAY_STYLE
unset INPUT_KEY LABEL_KEY METADATA_KEY KERNEL_ENV_URL KERNEL_BACKEND TILEKERNELS_DIR
unset DEBUG_ROLLOUT_ONLY LOAD_FORGE_ROLLOUT_DATA SAVE_DEBUG_ROLLOUT_DATA SAVE_DEBUG_TRAIN_DATA
unset ROLLOUT_DATASET_LOAD OVERRIDE_OPT_PARAM_SCHEDULER DISTRIBUTED_TIMEOUT_MINUTES
unset CLEAN_RAY KILL_OLD_PROCESSES RECHECK_GPU_IDLE_BEFORE_SUBMIT KILL_OLD_PROCESSES_BEFORE_SUBMIT

case "${HEAD_PHYSICAL}" in
  node64)
    HEAD_HOST=node64_slime
    HEAD_IP=10.11.2.164
    ROLLOUT_WORKER_HOSTS=node53_dspark
    ROLLOUT_WORKER_IPS=10.11.2.153
    ROLLOUT_PHYSICAL_HOSTS=node53
    ;;
  node69)
    HEAD_HOST=node69_slime
    HEAD_IP=10.11.2.169
    ROLLOUT_WORKER_HOSTS=node70_dspark
    ROLLOUT_WORKER_IPS=10.11.2.170
    ROLLOUT_PHYSICAL_HOSTS=node70
    ;;
  *)
    echo "FATAL: HEAD_PHYSICAL must be node64 or node69, got '${HEAD_PHYSICAL}'." >&2
    exit 2
    ;;
esac

# full_loop_smoke.sh starts the Ray head in its own process namespace.  Mapping
# HEAD_HOST is not enough: invoking a node69 arm from node64 would otherwise
# silently start the head on node64 while advertising node69's address.
LOCAL_IPV4S=$(hostname -I 2>/dev/null || true)
if ! tr ' ' '\n' <<<"${LOCAL_IPV4S}" | grep -Fxq "${HEAD_IP}"; then
  echo "FATAL: ${ARM}/${HEAD_PHYSICAL} must be launched inside ${HEAD_HOST} (${HEAD_IP}); local IPv4 addresses are: ${LOCAL_IPV4S:-<none>}" >&2
  exit 2
fi

CALCULATE_PER_TOKEN_LOSS=1
CUSTOM_PG_LOSS_REDUCER_FUNCTION_PATH=
USE_TIS=0
USE_KEEP_OLD_ACTOR=0
USE_ROLLOUT_LOGPROBS=1
DEBUG_FREEZE_OLD_ACTOR_SNAPSHOT=0
DEBUG_FORCE_OLD_ACTOR_LOGPROB_RECOMPUTE=0
PREPARE_FIXED_BEHAVIOR_DUMP=0
EPS_CLIP=0.20
EPS_CLIP_HIGH=0.20
EPS_CLIP_C=5
HISTORICAL_MIS_CONFIG='{"aggregation":"turns_geometric","token_veto_threshold":1e-4,"lower":0.99,"upper":1.01,"use_advantage":false}'
NOOP_MIS_CONFIG='{"aggregation":"turns_geometric"}'
# All nominal no-MIS arms still install the common postprocessor, but this
# canonical config has no bounds and no token veto, hence it cannot filter.
# Arms that recompute train log-probs retain mismatch telemetry; rollout-anchor
# predictive arms may skip that postprocessor because no train log-probs exist.
SEQUENCE_MIS_CONFIG=${NOOP_MIS_CONFIG}
case "${ARM}" in
  token_predictive)
    POLICY_LOSS_MODE=dppo_topk_kl_predictive
    ;;
  token_predictive_nomask)
    # Pure predictive-mask ablation: preserve the predictive surrogate, ratio
    # cap, support tensors, and diagnostics, but make the finite KL threshold
    # unreachable for finite model/rollout log probabilities.
    POLICY_LOSS_MODE=dppo_topk_kl_predictive
    EPS_CLIP=1e30
    EPS_CLIP_HIGH=1e30
    ;;
  token_ppo)
    POLICY_LOSS_MODE=ppo
    ;;
  ppo_rollout_denom)
    # Denominator-only A/B, rollout arm.  Both arms intentionally instantiate
    # and freeze the same Megatron old actor and recompute its log-probs; only
    # policy_loss_function's denominator selector differs.  TIS and strict MIS
    # stay off in both arms.
    SEQUENCE_MIS_CONFIG=${NOOP_MIS_CONFIG}
    POLICY_LOSS_MODE=ppo
    USE_ROLLOUT_LOGPROBS=1
    USE_TIS=0
    USE_KEEP_OLD_ACTOR=1
    DEBUG_FREEZE_OLD_ACTOR_SNAPSHOT=1
    DEBUG_FORCE_OLD_ACTOR_LOGPROB_RECOMPUTE=1
    PREPARE_FIXED_BEHAVIOR_DUMP=1
    ;;
  ppo_recompute_denom)
    # Exact mate of ppo_rollout_denom: the only changed loss input is the PPO
    # denominator, selected from the frozen Megatron old-actor recompute.
    SEQUENCE_MIS_CONFIG=${NOOP_MIS_CONFIG}
    POLICY_LOSS_MODE=ppo
    USE_ROLLOUT_LOGPROBS=0
    USE_TIS=0
    USE_KEEP_OLD_ACTOR=1
    DEBUG_FREEZE_OLD_ACTOR_SNAPSHOT=1
    DEBUG_FORCE_OLD_ACTOR_LOGPROB_RECOMPUTE=1
    PREPARE_FIXED_BEHAVIOR_DUMP=1
    ;;
  completion_predictive)
    POLICY_LOSS_MODE=dppo_topk_kl_predictive
    CALCULATE_PER_TOKEN_LOSS=0
    CUSTOM_PG_LOSS_REDUCER_FUNCTION_PATH=examples.kernel_agent.diagnostic_reducers.get_completion_mean_pg_loss_reducer
    ;;
  completion_ppo)
    POLICY_LOSS_MODE=ppo
    CALCULATE_PER_TOKEN_LOSS=0
    CUSTOM_PG_LOSS_REDUCER_FUNCTION_PATH=examples.kernel_agent.diagnostic_reducers.get_completion_mean_pg_loss_reducer
    ;;
  mis_tis)
    # Exact historical MIS/TIS bands are supplied by the caller after the
    # source run is identified.  Refuse a guessed configuration.
    : "${REQUESTED_SEQUENCE_MIS_CONFIG:?mis_tis requires --sequence-mis-config from the exact source run}"
    SEQUENCE_MIS_CONFIG=${REQUESTED_SEQUENCE_MIS_CONFIG}
    POLICY_LOSS_MODE=ppo
    USE_ROLLOUT_LOGPROBS=0
    USE_TIS=1
    USE_KEEP_OLD_ACTOR=1
    DEBUG_FREEZE_OLD_ACTOR_SNAPSHOT=1
    PREPARE_FIXED_BEHAVIOR_DUMP=1
    # Match historical donor ivbx2rz8 exactly.  No dual-clip C was supplied.
    EPS_CLIP=0.20
    EPS_CLIP_HIGH=0.28
    EPS_CLIP_C=
    ;;
  oldactor_tis)
    # Isolate the historical old-actor/TIS path from sequence filtering.  The
    # common DrKernel task always installs the sequence_mis postprocessor, so
    # an aggregation-only config is the explicit no-op: lower/upper resolve to
    # -inf/+inf and token veto remains disabled.  The hook still reports the
    # cross-engine mismatch distribution and MUST report mis_reject_rate=0.
    SEQUENCE_MIS_CONFIG=${NOOP_MIS_CONFIG}
    POLICY_LOSS_MODE=ppo
    USE_ROLLOUT_LOGPROBS=0
    USE_TIS=1
    USE_KEEP_OLD_ACTOR=1
    DEBUG_FREEZE_OLD_ACTOR_SNAPSHOT=1
    PREPARE_FIXED_BEHAVIOR_DUMP=1
    # Keep this ablation identical to mis_tis except for strict sequence MIS.
    EPS_CLIP=0.20
    EPS_CLIP_HIGH=0.28
    EPS_CLIP_C=
    ;;
  *)
    echo "FATAL: unsupported arm '${ARM}' (token_predictive, token_predictive_nomask, token_ppo, ppo_rollout_denom, ppo_recompute_denom, completion_predictive, completion_ppo, mis_tis, oldactor_tis)." >&2
    exit 2
    ;;
esac

EXPECTED_SEQUENCE_MIS_CONFIG=${NOOP_MIS_CONFIG}
[[ "${ARM}" == "mis_tis" ]] && EXPECTED_SEQUENCE_MIS_CONFIG=${HISTORICAL_MIS_CONFIG}
if ! python3 -c 'import json,sys; raise SystemExit(0 if json.loads(sys.argv[1]) == json.loads(sys.argv[2]) else 1)' \
  "${SEQUENCE_MIS_CONFIG}" "${EXPECTED_SEQUENCE_MIS_CONFIG}"; then
  echo "FATAL: ${ARM} --sequence-mis-config does not match its canonical experiment contract." >&2
  exit 2
fi

local_file_sha256() {
  local path=$1
  local digest
  if [[ ! -s "${path}" ]]; then
    echo "FATAL: frozen dump is missing on ${HEAD_HOST}: ${path}" >&2
    return 1
  fi
  digest=$(sha256sum -- "${path}" | awk '{print $1}') || {
    echo "FATAL: cannot hash frozen dump locally on ${HEAD_HOST}: ${path}" >&2
    return 1
  }
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "FATAL: invalid local SHA256 for ${path}: '${digest}'" >&2
    return 1
  fi
  printf '%s\n' "${digest}"
}

remote_file_sha256() {
  local host=$1
  local path=$2
  local digest
  digest=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "${host}" "sha256sum -- '${path}'" | awk '{print $1}') || {
    echo "FATAL: cannot hash frozen dump on ${host}: ${path}" >&2
    return 1
  }
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "FATAL: invalid SHA256 returned by ${host} for ${path}: '${digest}'" >&2
    return 1
  fi
  printf '%s\n' "${digest}"
}

shared_pair_file_sha256() {
  local path=$1
  local expected
  local host digest
  # The IP guard above proves this process already runs inside HEAD_HOST.  Do
  # not ssh back through that host's alias: container-local aliases may be
  # stale even though the selected head is correct and its files are present.
  expected=$(local_file_sha256 "${path}")
  for host in ${ROLLOUT_WORKER_HOSTS}; do
    if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "${host}" "test -s '${path}'"; then
      echo "FATAL: frozen dump is missing on ${host}: ${path}" >&2
      return 1
    fi
    digest=$(remote_file_sha256 "${host}" "${path}")
    if [[ "${digest}" != "${expected}" ]]; then
      echo "FATAL: frozen dump differs across the selected pair: path=${path} ${HEAD_HOST}=${expected} ${host}=${digest}" >&2
      return 1
    fi
  done
  printf '%s\n' "${expected}"
}

# Hash the immutable capture before an old-actor arm derives its v0-stamped
# replay copy.  This is the cross-arm frozen-batch identity; the derived file's
# bytes are separately recorded below.
SOURCE_DEBUG_DATA=${DEBUG_DATA}
SOURCE_DEBUG_DATA_SHA256=$(shared_pair_file_sha256 "${SOURCE_DEBUG_DATA}")

RUN_TAG=${ENTROPY_AB_RUN_TAG:-20260721}
if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "FATAL: ENTROPY_AB_RUN_TAG must contain only letters, digits, dot, underscore, or dash; got '${RUN_TAG}'." >&2
  exit 2
fi
RUN_ID=entropy_ab_${ARM}_${HEAD_PHYSICAL}_${RUN_TAG}
SCRATCH=/nfs/FM/csl_v4_entropy_ab_${ARM}_${HEAD_PHYSICAL}_${RUN_TAG}
LOG=/tmp/${RUN_ID}.out
HF_CKPT=/nfs/FM/chenshuailin/checkpoints/deepseek-ai/DeepSeek-V4-Flash
LOAD=/nfs/FM/chenshuailin/checkpoints/deepseek-ai/DeepSeek-V4-Flash-FP4-r21-pp1-ep8-torch_dist
PROMPT_DATA=/ms/FM/lihongbin/dataset/CUDA_RL/cuda_rl/prompt_tvm_v2/drkernel_rl_thinking.parquet
TILEKERNELS_DIR=/nfs/FM/chenshuailin/projects/kernel_agents/TileKernels

LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE=0.5
ROLLOUT_BATCH_SIZE=8
N_SAMPLES_PER_PROMPT=16
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))

if [[ "${PREPARE_FIXED_BEHAVIOR_DUMP}" == "1" ]]; then
  # /nfs is node-local on this cluster.  Never derive on only the invoking
  # container and assume the same path appeared on its worker.  Old-actor arms
  # accept only a separately audited v0-stamped sibling that has already been
  # copied to BOTH nodes, plus its explicit expected digest.  This also makes a
  # prepare-only pass non-mutating with respect to the frozen replay artifact.
  : "${PREPARED_DEBUG_DATA:?${ARM} requires PREPARED_DEBUG_DATA: the pre-audited v0-stamped dump path present on both selected nodes}"
  : "${PREPARED_DEBUG_DATA_SHA256:?${ARM} requires PREPARED_DEBUG_DATA_SHA256 from the CPU audit/copy step}"
  if [[ ! "${PREPARED_DEBUG_DATA_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "FATAL: PREPARED_DEBUG_DATA_SHA256 must be 64 lowercase hex characters, got '${PREPARED_DEBUG_DATA_SHA256}'." >&2
    exit 2
  fi
  if [[ "${PREPARED_DEBUG_DATA}" == "${SOURCE_DEBUG_DATA}" ]]; then
    echo "FATAL: PREPARED_DEBUG_DATA must be a distinct v0-stamped sibling, not the immutable source dump." >&2
    exit 2
  fi
  DEBUG_DATA=${PREPARED_DEBUG_DATA}
  DEBUG_DATA_SHA256=$(shared_pair_file_sha256 "${DEBUG_DATA}")
  if [[ "${DEBUG_DATA_SHA256}" != "${PREPARED_DEBUG_DATA_SHA256}" ]]; then
    echo "FATAL: pre-audited frozen dump SHA256 mismatch: expected=${PREPARED_DEBUG_DATA_SHA256} actual=${DEBUG_DATA_SHA256} path=${DEBUG_DATA}" >&2
    exit 2
  fi
fi

if [[ "${DEBUG_DATA}" == "${SOURCE_DEBUG_DATA}" ]]; then
  DEBUG_DATA_SHA256=${SOURCE_DEBUG_DATA_SHA256}
elif [[ -z "${DEBUG_DATA_SHA256:-}" ]]; then
  DEBUG_DATA_SHA256=$(shared_pair_file_sha256 "${DEBUG_DATA}")
fi

# A git commit alone is insufficient in this intentionally dirty experiment
# tree.  Hash a fixed, sorted-by-declaration manifest of every behavior-critical
# source used by these arms, and require both selected containers to agree.
CRITICAL_CODE_FILES=(
  custom_kernels/deepseek_v4/attention/kernel.py
  custom_kernels/deepseek_v4/megatron/attention.py
  custom_kernels/deepseek_v4/megatron/compressor.py
  custom_kernels/deepseek_v4/megatron/cp_utils.py
  custom_kernels/deepseek_v4/megatron/decoder.py
  custom_kernels/deepseek_v4/megatron/lora.py
  custom_kernels/deepseek_v4/megatron/mcore_model.py
  custom_kernels/deepseek_v4/megatron/model_provider.py
  custom_kernels/deepseek_v4/megatron/native_checkpoint.py
  custom_kernels/deepseek_v4/mhc/official.py
  examples/kernel_agent/config.py
  examples/kernel_agent/diagnostic_reducers.py
  examples/kernel_agent/generate_with_cuda_agent.py
  examples/kernel_agent/kernel_filter.py
  examples/kernel_agent/kernel_reward.py
  examples/kernel_agent/utils.py
  scripts/dsv4/_dsv4_cluster_lib.sh
  scripts/dsv4/_dsv4_task_args.sh
  scripts/dsv4/full_loop_smoke.sh
  scripts/dsv4/studies/entropy/prepare_entropy_fixed_rollout.py
  scripts/dsv4/studies/entropy/run_entropy_fixed_batch_arm.sh
  slime/backends/megatron_utils/actor.py
  slime/backends/megatron_utils/cp_utils.py
  slime/backends/megatron_utils/data.py
  slime/backends/megatron_utils/initialize.py
  slime/backends/megatron_utils/lora_old_actor.py
  slime/backends/megatron_utils/lora_zero_audit.py
  slime/backends/megatron_utils/loss.py
  slime/backends/megatron_utils/model.py
  slime/ray/actor_group.py
  slime/ray/placement_group.py
  slime/ray/rollout.py
  slime/utils/arguments.py
  slime/utils/ppo_utils.py
  slime/utils/routing_replay.py
  slime/utils/train_metric_utils.py
  slime/utils/types.py
  train.py
)
printf -v CRITICAL_CODE_FILE_ARGS ' %q' "${CRITICAL_CODE_FILES[@]}"

critical_manifest_sha256() {
  local host=$1
  local manifest=$2
  local manifest_lines critical_sha
  manifest_lines=$(printf '%s\n' "${manifest}" | awk 'END {print NR}')
  if [[ "${manifest_lines}" != "${#CRITICAL_CODE_FILES[@]}" ]]; then
    echo "FATAL: critical-code manifest from ${host} has ${manifest_lines} entries; expected ${#CRITICAL_CODE_FILES[@]}." >&2
    return 1
  fi
  critical_sha=$(printf '%s\n' "${manifest}" | sha256sum | awk '{print $1}')
  if [[ ! "${critical_sha}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "FATAL: malformed critical-code SHA256 from ${host}: '${critical_sha}'" >&2
    return 1
  fi
  printf '%s\n' "${critical_sha}"
}

local_code_identity() {
  local git_revision manifest critical_sha
  if git_revision=$(git -C "${REPO}" rev-parse --verify HEAD 2>/dev/null); then
    if [[ -n "${ENTROPY_AB_SOURCE_GIT_REVISION:-}" && "${git_revision}" != "${ENTROPY_AB_SOURCE_GIT_REVISION}" ]]; then
      echo "FATAL: local git revision ${git_revision} differs from synced source revision ${ENTROPY_AB_SOURCE_GIT_REVISION}" >&2
      return 1
    fi
  else
    git_revision=${ENTROPY_AB_SOURCE_GIT_REVISION:-}
    if [[ -z "${git_revision}" && -s "${REPO}/.slime-v4-source-provenance" ]]; then
      local provenance_format provenance_manifest_sha
      provenance_format=$(sed -n 's/^format=//p' "${REPO}/.slime-v4-source-provenance")
      git_revision=$(sed -n 's/^git_revision=//p' "${REPO}/.slime-v4-source-provenance")
      provenance_manifest_sha=$(sed -n 's/^runtime_source_manifest_sha256=//p' "${REPO}/.slime-v4-source-provenance")
      if [[ "${provenance_format}" != "slime-v4-source-provenance-v1" || ! "${provenance_manifest_sha}" =~ ^[0-9a-f]{64}$ ]]; then
        echo "FATAL: malformed source provenance on content-only head ${HEAD_HOST}" >&2
        return 1
      fi
    fi
    if [[ -z "${git_revision}" ]]; then
      echo "FATAL: content-only head ${HEAD_HOST} has no source revision provenance; run scripts/sync/sync_dsv4_active_nodes.sh from node64" >&2
      return 1
    fi
    echo "INFO: using source-node git provenance ${git_revision} for content-only head ${HEAD_HOST}" >&2
  fi
  manifest=$(cd "${REPO}" && LC_ALL=C sha256sum -- "${CRITICAL_CODE_FILES[@]}") || {
    echo "FATAL: cannot compute critical-code manifest locally on ${HEAD_HOST}" >&2
    return 1
  }
  critical_sha=$(critical_manifest_sha256 "${HEAD_HOST}" "${manifest}")
  if [[ ! "${git_revision}" =~ ^[0-9a-f]{40,64}$ ]]; then
    echo "FATAL: malformed head git revision on ${HEAD_HOST}: '${git_revision}'" >&2
    return 1
  fi
  printf '%s %s\n' "${git_revision}" "${critical_sha}"
}

remote_critical_code_sha256() {
  local host=$1
  local manifest
  # A deployed worker checkout may intentionally be a content-only rsync and
  # therefore have no .git directory.  Runtime equivalence is guarded by the
  # fail-closed critical-file manifest, while provenance comes from the head.
  manifest=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "${host}" \
    "cd '${REPO}' && LC_ALL=C sha256sum --${CRITICAL_CODE_FILE_ARGS}") || {
    echo "FATAL: cannot compute critical-code manifest on ${host}" >&2
    return 1
  }
  critical_manifest_sha256 "${host}" "${manifest}"
}

HEAD_CODE_IDENTITY=$(local_code_identity)
read -r GIT_REVISION CRITICAL_CODE_SHA256 <<<"${HEAD_CODE_IDENTITY}"
for host in ${ROLLOUT_WORKER_HOSTS}; do
  host_critical_sha256=$(remote_critical_code_sha256 "${host}")
  if [[ "${host_critical_sha256}" != "${CRITICAL_CODE_SHA256}" ]]; then
    echo "FATAL: critical code differs across the selected pair: ${HEAD_HOST}=git:${GIT_REVISION}/sha:${CRITICAL_CODE_SHA256} ${host}=sha:${host_critical_sha256}" >&2
    exit 2
  fi
done

# Cheap initial-checkpoint identity: torch-dist's .metadata describes the full
# tensor/shard layout, while metadata.json/common.pt/latest pin the release and
# loader state.  Hashing these avoids rereading the ~155 GB parameter payload.
BASE_IDENTITY_FILES=(
  "${LOAD}/latest_checkpointed_iteration.txt"
  "${LOAD}/release/.metadata"
  "${LOAD}/release/metadata.json"
  "${LOAD}/release/common.pt"
)

local_base_identity() {
  local manifest manifest_lines
  manifest=$(LC_ALL=C sha256sum -- "${BASE_IDENTITY_FILES[@]}") || {
    echo "FATAL: cannot compute base checkpoint metadata identity locally on ${HEAD_HOST}" >&2
    return 1
  }
  manifest_lines=$(printf '%s\n' "${manifest}" | awk 'END {print NR}')
  if [[ "${manifest_lines}" != "${#BASE_IDENTITY_FILES[@]}" ]]; then
    echo "FATAL: base metadata manifest from ${HEAD_HOST} has ${manifest_lines} entries; expected ${#BASE_IDENTITY_FILES[@]}." >&2
    return 1
  fi
  printf '%s\n' "${manifest}" | sha256sum | awk '{print $1}'
}

BASE_MANIFEST_SHA256=$(local_base_identity)

resolve_external_revision() {
  local label=$1 path=$2 provenance_var=$3 revision expected
  expected=${!provenance_var-}
  if revision=$(git -C "${path}" rev-parse --verify HEAD 2>/dev/null); then
    if [[ -n "${expected}" && "${revision}" != "${expected}" ]]; then
      echo "FATAL: ${label} revision ${revision} differs from synced source revision ${expected}" >&2
      return 1
    fi
  else
    revision=${expected}
    if [[ -z "${revision}" && -s "${path}/.slime-v4-source-provenance" ]]; then
      local provenance_format provenance_manifest_sha
      provenance_format=$(sed -n 's/^format=//p' "${path}/.slime-v4-source-provenance")
      revision=$(sed -n 's/^git_revision=//p' "${path}/.slime-v4-source-provenance")
      provenance_manifest_sha=$(sed -n 's/^runtime_source_manifest_sha256=//p' "${path}/.slime-v4-source-provenance")
      if [[ "${provenance_format}" != "slime-v4-source-provenance-v1" || ! "${provenance_manifest_sha}" =~ ^[0-9a-f]{64}$ ]]; then
        echo "FATAL: malformed ${label} source provenance on ${HEAD_HOST}" >&2
        return 1
      fi
    fi
    if [[ -z "${revision}" ]]; then
      echo "FATAL: content-only ${label} checkout on ${HEAD_HOST} has no source provenance; run scripts/sync/sync_dsv4_active_nodes.sh from node64" >&2
      return 1
    fi
    echo "INFO: using source-node ${label} provenance ${revision} for content-only checkout" >&2
  fi
  printf '%s\n' "${revision}"
}

local_external_revisions() {
  resolve_external_revision Megatron-LM /root/Megatron-LM ENTROPY_AB_SOURCE_MEGATRON_REVISION
  resolve_external_revision TileKernels "${TILEKERNELS_DIR}" ENTROPY_AB_SOURCE_TILEKERNELS_REVISION
}

HEAD_EXTERNAL_REVISIONS=$(local_external_revisions) || {
  echo "FATAL: cannot resolve Megatron-LM/TileKernels revisions locally on ${HEAD_HOST}" >&2
  exit 2
}
MEGATRON_REVISION=$(printf '%s\n' "${HEAD_EXTERNAL_REVISIONS}" | sed -n '1p')
TILEKERNELS_REVISION=$(printf '%s\n' "${HEAD_EXTERNAL_REVISIONS}" | sed -n '2p')
if [[ ! "${MEGATRON_REVISION}" =~ ^[0-9a-f]{40,64}$ || ! "${TILEKERNELS_REVISION}" =~ ^[0-9a-f]{40,64}$ ]]; then
  echo "FATAL: malformed external runtime revisions on ${HEAD_HOST}." >&2
  exit 2
fi

audit_fixed_dump_on_pair() {
  local path=$1
  local expected_gen_version=${2:-}
  local host
  local -a version_args=()
  [[ -n "${expected_gen_version}" ]] && version_args=(--expected-gen-weight-version "${expected_gen_version}")
  python3 "${REPO}/scripts/dsv4/studies/entropy/prepare_entropy_fixed_rollout.py" "${path}" \
    --audit-only \
    --expected-rollout-id 0 \
    --load-subsample-ratio "${LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE}" \
    --expected-samples "${GLOBAL_BATCH_SIZE}" \
    --expected-group-size "${N_SAMPLES_PER_PROMPT}" \
    --expected-turn 0 \
    --expected-layers 43 \
    --expected-routing-topk 6 \
    --expected-predictive-top-k 20 \
    "${version_args[@]}"
  for host in ${ROLLOUT_WORKER_HOSTS}; do
    ssh -o BatchMode=yes -o ConnectTimeout=8 "${host}" \
      python3 "${REPO}/scripts/dsv4/studies/entropy/prepare_entropy_fixed_rollout.py" "${path}" \
      --audit-only \
      --expected-rollout-id 0 \
      --load-subsample-ratio "${LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE}" \
      --expected-samples "${GLOBAL_BATCH_SIZE}" \
      --expected-group-size "${N_SAMPLES_PER_PROMPT}" \
      --expected-turn 0 \
      --expected-layers 43 \
      --expected-routing-topk 6 \
      --expected-predictive-top-k 20 \
      "${version_args[@]}"
  done
}

# SHA equality alone cannot prove the selected 128 rows contain eight complete
# groups or every tensor consumed by predictive/routing replay.  Audit both
# node-local copies; old-actor input additionally must carry the v0 stamp.
audit_fixed_dump_on_pair "${SOURCE_DEBUG_DATA}"
if [[ "${DEBUG_DATA}" != "${SOURCE_DEBUG_DATA}" ]]; then
  audit_fixed_dump_on_pair "${DEBUG_DATA}" 0
fi

export REPO ARM HEAD_PHYSICAL HEAD_HOST HEAD_IP ROLLOUT_WORKER_HOSTS ROLLOUT_WORKER_IPS
export ROLLOUT_PHYSICAL_HOSTS RUN_ID SCRATCH LOG
# A non-empty whitespace value expands to an empty bash array in
# full_loop_smoke.sh, giving this diagnostic exactly one actor node.
export TRAIN_WORKER_HOSTS=' '
export TRAIN_WORKER_IPS=' '
export ACTOR_PHYSICAL_HOSTS="${HEAD_PHYSICAL}"
export PHYSICAL_CLEAN_HOSTS="${HEAD_PHYSICAL} ${ROLLOUT_PHYSICAL_HOSTS}"
export ACTOR_NUM_NODES=1 ACTOR_GPUS_PER_NODE=8 ROLLOUT_GPUS=8 ROLLOUT_GPUS_PER_ENGINE=8
export ACTOR_CPUS_PER_NODE=64 ROLLOUT_CPUS_PER_NODE=0

export TASK_MODE=rl REWARD_MODE=drkernel TRAIN_SCRIPT=train.py DEBUG_TRAIN_ONLY=1
export LOAD_DEBUG_ROLLOUT_DATA="${DEBUG_DATA}"
export LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE
export DEBUG_FREEZE_OLD_ACTOR_SNAPSHOT
export DEBUG_FORCE_OLD_ACTOR_LOGPROB_RECOMPUTE
export START_ROLLOUT_ID=0 NUM_ROLLOUT=3
export ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE
export OVER_SAMPLING_BATCH_SIZE="${ROLLOUT_BATCH_SIZE}"
export OVER_SAMPLING_REFILL_FACTOR=

export HF_CKPT LOAD
export SAVE="${SCRATCH}/out" SAVE_MODEL=0 LOAD_OPTIM=0 LOAD_RNG=0 SAVE_OPTIM=0 SAVE_RNG=0 ASYNC_SAVE=0
export PROMPT_DATA

export PP_SIZE=1 CP_SIZE=2 EP_SIZE=8 FIRST_LAYERS=43 LAST_LAYERS=0
export MAX_CONTEXT_LEN=12288 MAX_RESPONSE_LEN=12288 ROLLOUT_MAX_PROMPT_LEN=8192
export RECOMPUTE=1 RECOMPUTE_NUM_LAYERS=1 RECOMPUTE_METHOD=uniform
export LOG_PROBS_CHUNK_SIZE=512
export V4_FP4_FROZEN_EXPERTS=1
export LR=1e-5 LR_DECAY_STYLE=constant WEIGHT_DECAY=0.01 ADVANTAGE_ESTIMATOR=trloo ENTROPY_COEF=0.0
export MAX_TURNS=1 TIS_CLIP=2.0 TIS_CLIP_LOW=0.0
export INPUT_KEY=prompt LABEL_KEY=reward_model METADATA_KEY=extra_info
export KERNEL_ENV_URL=http://127.0.0.1:20211 KERNEL_BACKEND=tvm_ffi
export TILEKERNELS_DIR DISTRIBUTED_TIMEOUT_MINUTES=120
export EPS_CLIP EPS_CLIP_HIGH EPS_CLIP_C
export POLICY_LOSS_MODE USE_ROLLOUT_LOGPROBS USE_TIS USE_KEEP_OLD_ACTOR
export CALCULATE_PER_TOKEN_LOSS CUSTOM_PG_LOSS_REDUCER_FUNCTION_PATH
export DPPO_PREDICTIVE_TOP_K=20 DPPO_PREDICTIVE_TAIL_ESTIMATOR=aggregated
export ROLLOUT_TEMPERATURE=1 ROLLOUT_TOP_P=1 USE_ROLLOUT_ROUTING_REPLAY=1

# SGLang is skipped by LOAD_DEBUG_ROLLOUT_DATA, but the common launcher still
# validates the rollout worker and constructs its resource labels.
export V4_ROLLOUT_MODEL_PATH=/nfs/FM/chenshuailin/checkpoints/deepseek-ai/DeepSeek-V4-Flash-DSpark
export SGLANG_DSV4_FP4_EXPERTS=1 SGLANG_ENABLE_DP_ATTENTION=1 SGLANG_SHARED_EXPERT_TP1=1
export SGLANG_DP_SIZE=8 SGLANG_CONTEXT_LENGTH=12288 SGLANG_MAX_PREFILL_TOKENS=12288
export SGLANG_CHUNKED_PREFILL_SIZE=2048 SGLANG_MAX_RUNNING_REQUESTS=128 SGLANG_CUDA_GRAPH_MAX_BS=128
export USE_SGLANG_DEEPEP=0

unset WANDB_API_KEY
export USE_WANDB=0 CLEAN_DEBUG_DIR=1 CLEAN_TILELANG_CACHE=0 CLEAN_RUNTIME_CACHE=0
export CLEAN_RAY=1 KILL_OLD_PROCESSES=1 RECHECK_GPU_IDLE_BEFORE_SUBMIT=1 KILL_OLD_PROCESSES_BEFORE_SUBMIT=0
export EXTERNAL_SGLANG_GUARD=0 CLEANUP_KILL_ORPHANS_ON_EXIT=0
export RUN_LOCK="/tmp/slime_entropy_ab_${HEAD_PHYSICAL}.lock"
[[ "${MODE}" == "--prepare-only" ]] && export PREPARE_ONLY=1

# full_loop_smoke.sh owns LOG and truncates it at startup, so pass provenance
# explicitly for that launcher to emit after opening the final log.
export ENTROPY_AB_ARM="${ARM}"
export ENTROPY_AB_SOURCE_DEBUG_DATA="${SOURCE_DEBUG_DATA}"
export ENTROPY_AB_SOURCE_DEBUG_DATA_SHA256="${SOURCE_DEBUG_DATA_SHA256}"
export ENTROPY_AB_DEBUG_DATA="${DEBUG_DATA}"
export ENTROPY_AB_DEBUG_DATA_SHA256="${DEBUG_DATA_SHA256}"
export ENTROPY_AB_GIT_REVISION="${GIT_REVISION}"
export ENTROPY_AB_CRITICAL_CODE_SHA256="${CRITICAL_CODE_SHA256}"
export ENTROPY_AB_BASE_MANIFEST_SHA256="${BASE_MANIFEST_SHA256}"
export ENTROPY_AB_MEGATRON_REVISION="${MEGATRON_REVISION}"
export ENTROPY_AB_TILEKERNELS_REVISION="${TILEKERNELS_REVISION}"
export ENTROPY_AB_REDUCTION=$([[ "${CALCULATE_PER_TOKEN_LOSS}" == 1 ]] && echo global_token || echo completion_equal)
export ENTROPY_AB_CONTRACT="repeats=3,batch=128,groups=8,group_size=16,subsample=0.5,max_turns=1,tis=[0.0,2.0],dynamic_batch=off,fp4_expert_gemm=0,rollout_worker_cpus=0,lora_out_zero_assert=1,forced_oldactor_forward=${DEBUG_FORCE_OLD_ACTOR_LOGPROB_RECOMPUTE},common_probe=original_mask_global_token,eps_clip=${EPS_CLIP},eps_clip_high=${EPS_CLIP_HIGH},eps_clip_c=${EPS_CLIP_C:-none}"

cat <<EOF
entropy_ab_arm=${ARM}
pair=${HEAD_PHYSICAL}+${ROLLOUT_PHYSICAL_HOSTS}
debug_data=${DEBUG_DATA} subsample=${LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE}
source_debug_data=${SOURCE_DEBUG_DATA}
source_debug_data_sha256=${SOURCE_DEBUG_DATA_SHA256}
debug_data_sha256=${DEBUG_DATA_SHA256}
git_revision=${GIT_REVISION}
critical_code_sha256=${CRITICAL_CODE_SHA256}
base_manifest_sha256=${BASE_MANIFEST_SHA256}
megatron_revision=${MEGATRON_REVISION}
tilekernels_revision=${TILEKERNELS_REVISION}
freeze_old_actor_snapshot=${DEBUG_FREEZE_OLD_ACTOR_SNAPSHOT}
force_old_actor_logprob_recompute=${DEBUG_FORCE_OLD_ACTOR_LOGPROB_RECOMPUTE}
policy=${POLICY_LOSS_MODE} rollout_logprobs=${USE_ROLLOUT_LOGPROBS} tis=${USE_TIS} keep_old=${USE_KEEP_OLD_ACTOR}
clip=eps_clip:${EPS_CLIP},eps_clip_high:${EPS_CLIP_HIGH},eps_clip_c:${EPS_CLIP_C:-none}
sequence_mis_config=${SEQUENCE_MIS_CONFIG}
resolved_contract=${ENTROPY_AB_CONTRACT}
reduction=$([[ "${CALCULATE_PER_TOKEN_LOSS}" == 1 ]] && echo global_token || echo completion_equal)
parallel=PP1/CP2/EP8 actor_nodes=1 batch=${GLOBAL_BATCH_SIZE} repeats=${NUM_ROLLOUT}
save_model=0 wandb=0 log=${LOG}
EOF

exec bash "${REPO}/scripts/dsv4/full_loop_smoke.sh" \
  --lora-dim 32 \
  --lora-alpha 32 \
  --lora-dropout 0.0 \
  --lora-rslora \
  --lora-plus-lambda 4 \
  --dsv4-lora-shared-expert \
  --lora-checkpoint-max-node-bytes 2147483648 \
  --data-pad-size-multiplier 1024 \
  --sequence-mis-config "${SEQUENCE_MIS_CONFIG}" \
  --entropy-common-probe \
  --assert-zero-lora-out
