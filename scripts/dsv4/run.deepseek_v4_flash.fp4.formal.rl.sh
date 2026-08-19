#!/bin/bash
# Formal DS-V4 packed-MXFP4/W4A16 PP1xCP2xDP8xEP8 run. The current
# generation resumes model/optimizer/RNG and the original prompt-dataset cursor
# from step 724. Later restarts restore all state natively from this lineage.
# Formal launches go through launch_formal_managed.sh only.
set -euo pipefail

_formal_prepare_only_arg=0
_formal_fresh_start_arg=0
while (( "$#" > 0 )); do
  case "$1" in
    --prepare-only)
      (( _formal_prepare_only_arg == 0 )) || { echo "FATAL: duplicate --prepare-only" >&2; exit 2; }
      _formal_prepare_only_arg=1
      ;;
    --fresh)
      (( _formal_fresh_start_arg == 0 )) || { echo "FATAL: duplicate --fresh" >&2; exit 2; }
      _formal_fresh_start_arg=1
      ;;
    *)
      echo "Usage: $0 [--prepare-only] [--fresh]" >&2
      exit 2
      ;;
  esac
  shift
done
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# --- packed-MXFP4 / W4A16 mode -------------------------------------------------
# Official mixed checkpoint (attention FP8, routed experts packed MXFP4) staged
# per node; torch_dist conversion produced with FP4_EXPERTS=1 convert_torch_dist.sh.
export V4_FP4_FROZEN_EXPERTS=1
export SGLANG_DSV4_FP4_EXPERTS=1
# W4A16 SM90 runners register a2a=none only; MoE runs TP8 inside each engine.
export USE_SGLANG_DEEPEP=0
export SGLANG_ENABLE_DP_ATTENTION=1
# Shared expert stays TP1-replicated (a2a=none is not in the fork's TP1
# predicate; sharding it breaks unsharded shared-expert LoRA adapters at init).
export SGLANG_SHARED_EXPERT_TP1=1
# Formal rollout uses the single pinned DSpark SGLang stack
# (sglang:dspark-r3 = 692c5f7d + 12-patch series). DSPARK speculative decoding
# has been full-loop validated on this stack
# (dspark_full_loop_smoke16 rc=0: logprobs, routing replay, LoRA sync,
# idle fix; train-rollout gap 0.007 nats; design doc "MIGRATION COMPLETE").
# Override the -DSpark checkpoint's gamma=5 with gamma=3 (verify window 4).
# fp8_e4m3 KV is forced on this runtime (G-M1). Rollback:
# SGLANG_SPECULATIVE_ALGORITHM=none still serves the -DSpark ckpt spec-off on
# the same containers.
export SGLANG_SPECULATIVE_ALGORITHM=${SGLANG_SPECULATIVE_ALGORITHM:-DSPARK}
export SGLANG_SPECULATIVE_DSPARK_BLOCK_SIZE=3
export V4_ROLLOUT_MODEL_PATH=${V4_ROLLOUT_MODEL_PATH:-/nfs/FM/chenshuailin/checkpoints/deepseek-ai/DeepSeek-V4-Flash-DSpark}

# Formal is deliberately immune to a caller shell previously used for a
# replay/smoke. A leaked NUM_ROLLOUT=2, source path, or debug flag must not turn
# the managed launch into a no-op or resume the wrong state.
unset TOPOLOGY PP_SIZE EP_SIZE CP_SIZE FIRST_LAYERS LAST_LAYERS
unset RECOMPUTE RECOMPUTE_NUM_LAYERS RECOMPUTE_METHOD NUM_ROLLOUT SAVE_INTERVAL ASYNC_SAVE
unset LOAD SCRATCH SAVE RESUME START_ROLLOUT_ID
unset LOAD_OPTIM LOAD_RNG OVERRIDE_OPT_PARAM_SCHEDULER
unset DEBUG_TRAIN_ONLY DEBUG_ROLLOUT_ONLY LOAD_DEBUG_ROLLOUT_DATA LOAD_FORGE_ROLLOUT_DATA
unset SAVE_DEBUG_ROLLOUT_DATA SAVE_DEBUG_TRAIN_DATA PREPARE_ONLY
unset RUN_ID WANDB_GROUP ROLLOUT_DATASET_LOAD PROMPT_DATA
unset MAX_CONTEXT_LEN MAX_RESPONSE_LEN SGLANG_CONTEXT_LENGTH SGLANG_MAX_PREFILL_TOKENS
unset WEIGHT_DECAY MUON_MOMENTUM MUON_USE_NESTEROV MUON_NUM_NS_STEPS
unset MUON_COEFFICIENT_TYPE MUON_SCALE_MODE MUON_EXTRA_SCALE_FACTOR MUON_FP32_MATMUL_PREC MUON_TP_MODE
unset POLICY_LOSS_MODE DIS_RATIO_LEVEL DPPO_PREDICTIVE_TOP_K DPPO_PREDICTIVE_TAIL_ESTIMATOR
unset USE_ROLLOUT_LOGPROBS USE_TIS USE_KEEP_OLD_ACTOR EPS_CLIP EPS_CLIP_HIGH EPS_CLIP_C
unset ROLLOUT_TEMPERATURE ROLLOUT_TOP_P
if [[ "${_formal_prepare_only_arg}" == "1" ]]; then
  export PREPARE_ONLY=1
fi
export TOPOLOGY=${TOPOLOGY:-pp2}
export PP_SIZE=${PP_SIZE:-1}
export EP_SIZE=${EP_SIZE:-8}
# Contiguous CP2 halves each rank's local sequence while DP remains 8:
# world16 / (TP1 * PP1 * CP2) = DP8, and EP8 divides DP*CP=16.
export CP_SIZE=${CP_SIZE:-2}
# Bucket BSHD widths at 1024-token boundaries.  This keeps the actual-length
# path's shape set small enough for production kernel-cache reuse while still
# avoiding rollout-wide 12k padding.
readonly FORMAL_DATA_PAD_SIZE_MULTIPLIER=1024
# PP1 ignores first/last-stage split flags in _dsv4_launch_core.sh.  Keep the
# banner truthful instead of advertising the retired 20/23 PP2 partition.
export FIRST_LAYERS=${FIRST_LAYERS:-43}
export LAST_LAYERS=${LAST_LAYERS:-0}
# Topology (user directive 2026-07-20): 2 train nodes = node69 (head) +
# node64; 2 rollout nodes = node53 + node70. node53 serves rollout.
# LAUNCH FROM node69_slime (HEAD_HOST executes locally in full_loop).
export TRAIN_WORKER_HOSTS="node64_slime"
export TRAIN_WORKER_IPS="10.11.2.164"
export HEAD_HOST="node69_slime"
export HEAD_IP="10.11.2.169"
export ACTOR_PHYSICAL_HOSTS="node69 node64"

# Two independent rollout engines (dp-attention DP8, MoE-TP8, W4A16).
export ROLLOUT_WORKER_HOSTS="node53_dspark node70_dspark"
export ROLLOUT_WORKER_IPS="10.11.2.153 10.11.2.170"
export ROLLOUT_PHYSICAL_HOSTS="node53 node70"
export ROLLOUT_GPUS=16
export ROLLOUT_GPUS_PER_ENGINE=8
export SGLANG_DP_SIZE=8
export SGLANG_MAX_RUNNING_REQUESTS=128
export SGLANG_CUDA_GRAPH_MAX_BS=128

# 12k context. MAX_CONTEXT_LEN is the total
# serving window; DrKernel clamps each turn's generation budget against the
# prompt length, so MAX_RESPONSE_LEN can match that total cap.
export MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-12288}
export MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-12288}
export SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-12288}
export SGLANG_MAX_PREFILL_TOKENS=${SGLANG_MAX_PREFILL_TOKENS:-12288}
export SGLANG_CHUNKED_PREFILL_SIZE=2048

# Soft overlong penalty (user semantics 2026-07-18): last 2048 response tokens
# ramp linearly to -0.5 at the cap, applied to TRAINING rewards only for
# groups that survive the low-variance filter — the filter judges on the
# PRE-penalty task reward (metadata task_reward), so lengthy all-fail groups
# are discarded as before. examples/kernel_agent/utils.py; 6 unit tests.
readonly OVERLONG_BUFFER_LEN=2048
readonly OVERLONG_PENALTY_FACTOR=0.5

# Per-layer, reentrant activation checkpointing (identical to r20).
# Overridable: RECOMPUTE=0 is the fp4-freed-memory variant (smoke first).
export RECOMPUTE=${RECOMPUTE:-1}
export RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}
# Keep the validated 512-token log-prob chunks: chunk 2048 showed cross-step
# fragmentation drift 88.7->94.0 GB over 4 steps at mean-9.8k lengths
# (memwatch 2026-07-18; r16 precedent died at step 4 the same way).
export LOG_PROBS_CHUNK_SIZE=512

# USER DIRECTIVE (2026-07-16): LoRA rank doubled 16 -> 32, fresh from iter0.
# Alpha stays 32: with rsLoRA (scale = alpha/sqrt(r)) the effective scale is
# rank-stable by design — changing alpha alongside rank would double-adjust.
# Resume from iter164 at half the previous learning rate; all other LoRA and
# optimizer settings remain unchanged.
export LR=5e-6
readonly FORMAL_LORA_DIM=32
readonly FORMAL_LORA_ALPHA=32
readonly FORMAL_LORA_DROPOUT=0.0
readonly FORMAL_LORA_PLUS_LAMBDA=4
readonly FORMAL_LORA_CHECKPOINT_MAX_NODE_BYTES=2147483648

# DeepSeek-V4 Muon recipe (arXiv:2606.19348, Algorithm 1 / training setups).
# Keep the formal RL learning rate unchanged; gamma=0.18 rescales the
# orthogonalized update RMS so that the existing Adam-style LR is reusable.
export WEIGHT_DECAY=0.1
export MUON_MOMENTUM=0.95
export MUON_USE_NESTEROV=1
export MUON_NUM_NS_STEPS=10
export MUON_COEFFICIENT_TYPE=deepseekv4
export MUON_SCALE_MODE=spectral
export MUON_EXTRA_SCALE_FACTOR=0.18
export MUON_FP32_MATMUL_PREC=medium
export MUON_TP_MODE=blockwise

# Resume the formal lineage with SAO's token-level Direct Double-Sided
# Importance Sampling (DIS).  Keep the historical coding-task trust interval
# (0.20, 4.0); all non-objective parameters and checkpointed state are
# unchanged from the predictive-DPPO segment through iter279.
export POLICY_LOSS_MODE=dis
export DIS_RATIO_LEVEL=token
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1}
export ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1}
export USE_ROLLOUT_LOGPROBS=${USE_ROLLOUT_LOGPROBS:-1}
export USE_TIS=0
export USE_KEEP_OLD_ACTOR=${USE_KEEP_OLD_ACTOR:-0}
export EPS_CLIP=0.80
export EPS_CLIP_HIGH=3.0
unset EPS_CLIP_C DPPO_PREDICTIVE_TOP_K DPPO_PREDICTIVE_TAIL_ESTIMATOR

# MIS GATE OFF (user directive 2026-07-17): DPPO owns the train/rollout mismatch
# handling. NOTE (codex-verified): under
# --use-rollout-logprobs without --get-mismatch-metrics the pre-train
# old-logprob recompute never runs, so kernel_filter.sequence_mis early-returns
# on EVERY rank ("log_probs unavailable") — the hook is fully INERT here: no
# rejection AND no mis_* stats. Do NOT expect mis_* wandb panels in this
# recipe; the direct rollout ratio is watched via train/ppo_kl, clip fractions,
# and train/dis_* metrics. The old fleet-runtime MIS calibration is
# not a launch requirement for this inert path; fp4-impl verification uses
# dedicated USE_ROLLOUT_LOGPROBS=0 MIS smokes (2026-07-17: reject 100% at [0.99,1.01],
# bias -0.039 nats/tok, mean |lr| 0.12, max |lr| 36-46 — see design doc).
# Intermediate variable REQUIRED: inline ${:-{...json...}} brace-matches the
# JSON's first } and corrupts the value (the _dsv4_task_args.sh footgun).
_formal_mis_default='{"aggregation":"turns_geometric"}'
readonly FORMAL_SEQUENCE_MIS_CONFIG="${_formal_mis_default}"

# Official checkpoint (per-node staged copy) + topology-matched packed
# torch_dist conversion. A variant may explicitly override LOAD; formal runs
# clear leaked LOAD above and always take the baked topology mapping here.
export HF_CKPT="/nfs/FM/chenshuailin/checkpoints/deepseek-ai/DeepSeek-V4-Flash"
case "${PP_SIZE}" in
  1)
    _formal_default_load="/nfs/FM/chenshuailin/checkpoints/deepseek-ai/DeepSeek-V4-Flash-FP4-r21-pp1-ep8-torch_dist"
    ;;
  2)
    _formal_default_load="/nfs/FM/chenshuailin/checkpoints/deepseek-ai/DeepSeek-V4-Flash-FP4-r21-pp2-ep8-20-23-torch_dist"
    ;;
  *)
    echo "FATAL: the formal DS-V4 recipe has no verified packed-FP4 conversion for PP_SIZE=${PP_SIZE}." >&2
    exit 2
    ;;
esac
export LOAD="${LOAD:-${_formal_default_load}}"
resume_mode="native"
resume_latest=""
native_minimum_iteration=0
# This lineage contains the native iter724 model/optimizer/RNG checkpoint and
# the exact iter724 cursor for the original prompt dataset. The scratch path
# retains its historical identifier because it names an existing checkpoint.
readonly DSV4_FORMAL_SCRATCH=/nfs/FM/csl_v4r21_fp4_pp1cp2_14k_step724_originaldata_20260731
readonly DSV4_FORMAL_PROMPT_DATA=/nfs/FM/chenshuailin/projects/kernel_agents/slime-v4flash-lora/Data/prompt_tvm_v3/drkernel_rl_thinking.parquet
readonly DSV4_FORMAL_PROMPT_SHA256=9e9ffca46022e74c0616f5e272871e76dfd000b6e7937b01685cd0bb11d521e5
readonly DSV4_FORMAL_PROMPT_ROWS=40307
export SCRATCH="${DSV4_FORMAL_SCRATCH}"
export SAVE="${SCRATCH}/out"
export PROMPT_DATA="${DSV4_FORMAL_PROMPT_DATA}"

if [[ "${_formal_fresh_start_arg}" == "1" ]]; then
  # A fresh launch loads only the immutable PP1 packed base. Adapter, Muon,
  # RNG, scheduler, and rollout-dataset state must all start empty/zero.
  RESUME=0
  resume_mode="fresh"
  resume_latest=""
  LORA_ADAPTER_RESUME_LOAD=""
  export START_ROLLOUT_ID=0
  export LOAD_OPTIM=0
  export LOAD_RNG=0
  unset ROLLOUT_DATASET_LOAD OVERRIDE_OPT_PARAM_SCHEDULER
  default_run_id="formal_dsv4_fp4_pp1cp2_12k_step724_originaldata_20260801_fresh"
else
  # Ordinary launches are native-only. An empty root is fatal instead of
  # silently falling back to a retired migration source.
  RESUME=1
  local_snapshot="$(python3 "${SCRIPT_DIR}/formal_resume_preflight.py" snapshot --root "${SAVE}")"
  remote_snapshot="$(ssh node64_slime "python3 '${SCRIPT_DIR}/formal_resume_preflight.py' snapshot --root '${SAVE}'")"
  resume_plan="$(python3 "${SCRIPT_DIR}/formal_resume_preflight.py" resolve \
    --native-only --minimum-native-iteration "${native_minimum_iteration}" \
    --local-snapshot "${local_snapshot}" --remote-snapshot "${remote_snapshot}")"
  read -r resume_mode resume_latest START_ROLLOUT_ID LOAD_RNG < <(
    python3 - "${resume_plan}" <<'PY'
import json
import sys

p = json.loads(sys.argv[1])
print(p["mode"], p["checkpoint_iteration"], p["start_rollout_id"], p["load_rng"])
PY
  )
  if [[ "${resume_mode}" != "native" ]]; then
    echo "FATAL: zero-based formal lineage resolved unexpected mode '${resume_mode}'." >&2
    exit 2
  fi
  export START_ROLLOUT_ID LOAD_RNG
  export LOAD_OPTIM=1
  # Keep the checkpointed Muon state, but use the launcher LR instead of
  # restoring the checkpoint scheduler's previous maximum.
  export OVERRIDE_OPT_PARAM_SCHEDULER=1
  LORA_ADAPTER_RESUME_LOAD="${SAVE}"
  export ROLLOUT_DATASET_LOAD="${SAVE}"
  default_run_id="formal_dsv4_fp4_pp1cp2_12k_step724_originaldata_20260801_native_iter${resume_latest}"
fi
export RUN_ID="${RUN_ID:-${default_run_id}}"
export SAVE_MODEL=1
export SAVE_INTERVAL=${SAVE_INTERVAL:-5}
# Revised two-phase async-save passed the full rollout-overlap gate at iter150
# and the final force-sync iter151 gate on 2026-07-20. Formal mode scrubs an
# ambient override above and therefore enables it for the next authorized run.
export ASYNC_SAVE=${ASYNC_SAVE:-1}
export SAVE_OPTIM=1
export SAVE_RNG=1
# Formal training was authorized after the bounded 60--64 canary passed.  Keep
# the established r20 horizon: NUM_ROLLOUT is the exclusive end, so native
# iter64 resumes at rollout 65 and can run through rollout 2999.
export NUM_ROLLOUT=${NUM_ROLLOUT:-3000}
export ROLLOUT_BATCH_SIZE=16
# Cap the candidate pool at 24 prompt groups while accepting 16.  Tail refill
# is adaptive: if k accepted groups are still missing, submit 2*k prompt
# groups (capped at 24) instead of another fixed prompt wave.
# N_SAMPLES_PER_PROMPT remains 16, so the training batch remains 256.
export OVER_SAMPLING_BATCH_SIZE=24
export OVER_SAMPLING_REFILL_FACTOR=2
export N_SAMPLES_PER_PROMPT=16
export GLOBAL_BATCH_SIZE=256
export WANDB_GROUP="${WANDB_GROUP:-v4flash_lora_rl_fp4_pp1cp2_12k_step724_originaldata_20260801}"

if [[ "${resume_mode}" == "fresh" ]]; then
  dataset_resume_display="<none;fresh-base-fallback-verified-empty>"
else
  dataset_resume_display="${ROLLOUT_DATASET_LOAD:-<legacy-load-fallback>}"
fi

print_critical_params() {
  cat <<EOF
run_id=${RUN_ID}
wandb_group=${WANDB_GROUP}
mode=packed-MXFP4/W4A16 (V4_FP4_FROZEN_EXPERTS=1, a2a=none, flashinfer_mxfp4, spec=${SGLANG_SPECULATIVE_ALGORITHM})
resume_mode=${resume_mode} adapter_resume=${LORA_ADAPTER_RESUME_LOAD:-<empty>} resume_iteration=${resume_latest:-none} start_rollout_id=${START_ROLLOUT_ID}
dataset_resume=${dataset_resume_display}
prompt_data=${PROMPT_DATA}
prompt_rows=${DSV4_FORMAL_PROMPT_ROWS} prompt_sha256=${DSV4_FORMAL_PROMPT_SHA256}
hf=${HF_CKPT}
load=${LOAD}
save=${SAVE}
train_nodes=${ACTOR_PHYSICAL_HOSTS} rollout_nodes=${ROLLOUT_WORKER_HOSTS}
mis_config=${FORMAL_SEQUENCE_MIS_CONFIG}
policy=mode=${POLICY_LOSS_MODE} dis_ratio_level=${DIS_RATIO_LEVEL:-token} rollout_logprobs=${USE_ROLLOUT_LOGPROBS} tis=${USE_TIS} keep_old=${USE_KEEP_OLD_ACTOR} delta=${EPS_CLIP}/${EPS_CLIP_HIGH} ratio_cap=${EPS_CLIP_C:-none} predictive_top_k=${DPPO_PREDICTIVE_TOP_K:-none} predictive_tail=${DPPO_PREDICTIVE_TAIL_ESTIMATOR:-none}
parallel=PP${PP_SIZE}/EP${EP_SIZE}/TP1/CP${CP_SIZE} pp_layers=$([[ "${PP_SIZE}" == "1" ]] && echo all43 || echo "${FIRST_LAYERS}/${LAST_LAYERS}")
context=${MAX_CONTEXT_LEN} response_cap=${MAX_RESPONSE_LEN} sglang_context=${SGLANG_CONTEXT_LENGTH} max_prefill_tokens=${SGLANG_MAX_PREFILL_TOKENS} prefill_chunk=${SGLANG_CHUNKED_PREFILL_SIZE}
data_pad_size_multiplier=${FORMAL_DATA_PAD_SIZE_MULTIPLIER}
overlong_penalty=enabled,buffer${OVERLONG_BUFFER_LEN},factor${OVERLONG_PENALTY_FACTOR}
recompute_method=${RECOMPUTE_METHOD:-uniform}
recompute=$([[ "${RECOMPUTE}" == "1" ]] && echo "reentrant,${RECOMPUTE_METHOD:-uniform}(${RECOMPUTE_NUM_LAYERS})" || echo OFF)
lr=${LR} lora=r${FORMAL_LORA_DIM},alpha${FORMAL_LORA_ALPHA},rslora1,loraplus${FORMAL_LORA_PLUS_LAMBDA},shared_expert1
optimizer=muon momentum=${MUON_MOMENTUM} nesterov=${MUON_USE_NESTEROV} weight_decay=${WEIGHT_DECAY} ns_steps=${MUON_NUM_NS_STEPS} coefficient=${MUON_COEFFICIENT_TYPE} scale_mode=${MUON_SCALE_MODE} update_rms_scale=${MUON_EXTRA_SCALE_FACTOR} matmul_precision=${MUON_FP32_MATMUL_PREC} tp_mode=${MUON_TP_MODE}
batch=rollout${ROLLOUT_BATCH_SIZE}x${N_SAMPLES_PER_PROMPT}=gbs${GLOBAL_BATCH_SIZE} oversampling_cap=${OVER_SAMPLING_BATCH_SIZE} refill_factor=${OVER_SAMPLING_REFILL_FACTOR} num_rollout=${NUM_ROLLOUT}
rollout=2x(DP8-attn,MoE-TP8,W4A16),max_running${SGLANG_MAX_RUNNING_REQUESTS},spec=${SGLANG_SPECULATIVE_ALGORITHM}
checkpoint=adapter+optimizer+rng,every${SAVE_INTERVAL},async${ASYNC_SAVE} load_optim=${LOAD_OPTIM} load_rng=${LOAD_RNG} max_node_bytes=${FORMAL_LORA_CHECKPOINT_MAX_NODE_BYTES}
scheduler_override=${OVERRIDE_OPT_PARAM_SCHEDULER:-0}
EOF
}

_expert_dtype() {
  # Per-tensor family probe — single shared implementation (codex finding 5).
  python3 "${SCRIPT_DIR}/probe_expert_dtype.py" "$1"
}

_probe_resume_checkpoint() {
  local host=$1
  local root=$2
  local iteration=$3
  local mode=$4
  local expected_distcp=$5
  local -a cmd=(
    python3 "${SCRIPT_DIR}/formal_resume_preflight.py" probe
    --root "${root}"
    --iteration "${iteration}"
    --mode "${mode}"
    --expected-distcp "${expected_distcp}"
    --minimum-native-iteration "${native_minimum_iteration}"
  )
  if [[ "${host}" == "local" ]]; then
    "${cmd[@]}"
  else
    ssh "${host}" "python3 '${SCRIPT_DIR}/formal_resume_preflight.py' probe --root '${root}' --iteration '${iteration}' --mode '${mode}' --expected-distcp '${expected_distcp}' --minimum-native-iteration '${native_minimum_iteration}'"
  fi
}

_verify_rollout_dataset_state() {
  local path=$1
  local mode=$2
  python3 - "${path}" "${mode}" <<'PY'
import sys

import torch

path, mode = sys.argv[1:]
state = torch.load(path, map_location="cpu", weights_only=False)
required = {"sample_offset", "epoch_id", "sample_group_index", "sample_index", "metadata"}
if not isinstance(state, dict) or set(state) != required:
    raise SystemExit(f"invalid rollout dataset state keys at {path}: {state!r}")
for key in ("sample_offset", "epoch_id", "sample_group_index", "sample_index"):
    if type(state[key]) is not int or state[key] < 0:
        raise SystemExit(f"invalid rollout dataset counter {key}={state[key]!r} at {path}")
if not isinstance(state["metadata"], dict):
    raise SystemExit(f"invalid rollout dataset metadata at {path}: {state['metadata']!r}")
if state["sample_index"] != state["sample_group_index"] * 16:
    raise SystemExit(f"rollout dataset sample/group counters disagree at {path}: {state!r}")
if mode == "migration":
    expected = {
        "sample_offset": 2640,
        "epoch_id": 0,
        "sample_group_index": 2640,
        "sample_index": 42240,
        "metadata": {},
    }
    if state != expected:
        raise SystemExit(f"migration rollout dataset state mismatch at {path}: {state!r}")
print(f"rollout dataset state PASS: mode={mode} path={path} state={state}")
PY
}

_verify_formal_prompt_artifact() {
  local actual_sha actual_rows host remote_sha
  if [[ ! -s "${DSV4_FORMAL_PROMPT_DATA}" ]]; then
    echo "NOT READY: formal prompt artifact is missing on node69: ${DSV4_FORMAL_PROMPT_DATA}" >&2
    return 1
  fi
  actual_sha=$(sha256sum -- "${DSV4_FORMAL_PROMPT_DATA}" | awk '{print $1}') || return 1
  if [[ "${actual_sha}" != "${DSV4_FORMAL_PROMPT_SHA256}" ]]; then
    echo "NOT READY: formal prompt SHA-256 mismatch on node69: expected=${DSV4_FORMAL_PROMPT_SHA256} actual=${actual_sha}" >&2
    return 1
  fi
  actual_rows=$(python3 - "${DSV4_FORMAL_PROMPT_DATA}" <<'PY'
import sys

import pyarrow.parquet as pq

print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)
PY
  ) || return 1
  if [[ "${actual_rows}" != "${DSV4_FORMAL_PROMPT_ROWS}" ]]; then
    echo "NOT READY: formal prompt row mismatch on node69: expected=${DSV4_FORMAL_PROMPT_ROWS} actual=${actual_rows}" >&2
    return 1
  fi
  for host in node64_slime node53_dspark node70_dspark; do
    remote_sha=$(ssh "${host}" "test -s '${DSV4_FORMAL_PROMPT_DATA}' && sha256sum -- '${DSV4_FORMAL_PROMPT_DATA}'" | awk '{print $1}') || {
      echo "NOT READY: formal prompt artifact is missing or unreadable on ${host}: ${DSV4_FORMAL_PROMPT_DATA}" >&2
      return 1
    }
    if [[ "${remote_sha}" != "${DSV4_FORMAL_PROMPT_SHA256}" ]]; then
      echo "NOT READY: formal prompt SHA-256 mismatch on ${host}: expected=${DSV4_FORMAL_PROMPT_SHA256} actual=${remote_sha}" >&2
      return 1
    fi
  done
  echo "formal prompt PASS: rows=${actual_rows} sha256=${actual_sha} nodes=node69,node64,node53,node70"
}

preflight() {
  local failed=0
  if ! _verify_formal_prompt_artifact; then
    failed=1
  fi
  # Official HF checkpoint present + genuinely packed on the head node.
  if [[ ! -f "${HF_CKPT}/model.safetensors.index.json" ]]; then
    echo "NOT READY: official checkpoint missing on this node: ${HF_CKPT}" >&2
    failed=1
  else
    local dt
    dt=$(_expert_dtype "${HF_CKPT}")
    if [[ "${dt}" != "I8" && "${dt}" != "U8" ]]; then
      echo "NOT READY: ${HF_CKPT} routed experts are ${dt}, not packed I8 — wrong checkpoint family" >&2
      failed=1
    fi
  fi
  # Rollout nodes need the staged HF checkpoint too (engines load from disk).
  for host in node53_slime node70_slime; do
    if ! ssh "${host}" "test -f '${HF_CKPT}/model.safetensors.index.json'"; then
      echo "NOT READY: official checkpoint missing on ${host}: ${HF_CKPT}" >&2
      failed=1
    fi
  done
  # Rollout engines serve the -DSpark variant (V4_ROLLOUT_MODEL_PATH) from the
  # dspark containers — check it there, plus container reachability.
  for host in node53_dspark node70_dspark; do
    if ! ssh "${host}" "test -f '${V4_ROLLOUT_MODEL_PATH}/model.safetensors.index.json'"; then
      echo "NOT READY: DSpark checkpoint missing/unreachable on ${host}: ${V4_ROLLOUT_MODEL_PATH}" >&2
      failed=1
    fi
  done
  # Packed torch_dist base on both train nodes.
  if [[ ! -s "${LOAD}/latest_checkpointed_iteration.txt" || ! -s "${LOAD}/release/.metadata" ]]; then
    echo "NOT READY: missing topology-matched packed base checkpoint on node69 (head): ${LOAD}" >&2
    failed=1
  fi
  if ! ssh node64_slime "test -s '${LOAD}/latest_checkpointed_iteration.txt' && test -s '${LOAD}/release/.metadata'"; then
    echo "NOT READY: missing topology-matched packed base checkpoint on node64: ${LOAD}" >&2
    failed=1
  fi
  if [[ "${RESUME}" == "1" ]]; then
    # DCP metadata references the global storage set.  Even native PP1/CP2
    # saves therefore need all 32 shards replicated into each node-local root.
    local expected_distcp=32
    if ! _probe_resume_checkpoint local "${LORA_ADAPTER_RESUME_LOAD}" "${resume_latest}" "${resume_mode}" "${expected_distcp}"; then
      echo "NOT READY: formal ${resume_mode} resume checkpoint failed on node69" >&2
      failed=1
    fi
    if ! _probe_resume_checkpoint node64_slime "${LORA_ADAPTER_RESUME_LOAD}" "${resume_latest}" "${resume_mode}" "${expected_distcp}"; then
      echo "NOT READY: formal ${resume_mode} resume checkpoint failed on node64" >&2
      failed=1
    fi
    local dataset_state="${ROLLOUT_DATASET_LOAD}/rollout/global_dataset_state_dict_${resume_latest}.pt"
    if [[ ! -s "${dataset_state}" ]]; then
      echo "NOT READY: rollout dataset resume state is missing on node69: ${dataset_state}" >&2
      failed=1
    elif ! _verify_rollout_dataset_state "${dataset_state}" "${resume_mode}"; then
      echo "NOT READY: rollout dataset resume state is invalid: ${dataset_state}" >&2
      failed=1
    fi
  fi
  if [[ "${resume_mode}" == "fresh" ]]; then
    # Fail closed on *any* pre-existing content, not merely a checkpoint
    # marker.  This protects dataset state, async-save staging directories,
    # and arbitrary leftovers from accidental reuse or overwrite.
    if [[ -e "${SCRATCH}" ]]; then
      echo "NOT READY: fresh formal root already exists on node69: ${SCRATCH}" >&2
      failed=1
    fi
    if ! ssh node64_slime "test ! -e '${SCRATCH}'"; then
      echo "NOT READY: fresh formal root already exists on node64: ${SCRATCH}" >&2
      failed=1
    fi
    # start_rollout_id=0 causes the dataset manager to probe rollout id -1.
    # With no explicit dataset-resume flag slime falls back to --load, so also
    # prove that the immutable base has not been polluted with such a state.
    local base_dataset_fallback="${LOAD}/rollout/global_dataset_state_dict_-1.pt"
    if [[ -e "${base_dataset_fallback}" ]]; then
      echo "NOT READY: fresh dataset fallback state exists on node69: ${base_dataset_fallback}" >&2
      failed=1
    fi
    if ! ssh node64_slime "test ! -e '${base_dataset_fallback}'"; then
      echo "NOT READY: fresh dataset fallback state exists on node64: ${base_dataset_fallback}" >&2
      failed=1
    fi
  fi
  if [[ "${RESUME}" != "1" && -e "${SAVE}/latest_checkpointed_iteration.txt" ]]; then
    echo "REFUSING RESUME: fresh save already has a checkpoint: ${SAVE}" >&2
    failed=1
  fi
  return "${failed}"
}

print_critical_params
if ! preflight; then
  exit 2
fi
echo "DSV4_FORMAL_PREFLIGHT_PASS resume_mode=${resume_mode} checkpoint_iteration=${resume_latest:-none} start_rollout_id=${START_ROLLOUT_ID}"

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "PREPARE_ONLY=1: configuration is ready; training was not launched."
  exit 0
fi

exec "${SCRIPT_DIR}/run.t1.deepseek_v4_flash.rl.sh" \
  --data-pad-size-multiplier "${FORMAL_DATA_PAD_SIZE_MULTIPLIER}" \
  --sequence-mis-config "${FORMAL_SEQUENCE_MIS_CONFIG}" \
  --lora-dim "${FORMAL_LORA_DIM}" \
  --lora-alpha "${FORMAL_LORA_ALPHA}" \
  --lora-dropout "${FORMAL_LORA_DROPOUT}" \
  --lora-rslora \
  --lora-plus-lambda "${FORMAL_LORA_PLUS_LAMBDA}" \
  --dsv4-lora-shared-expert \
  --lora-checkpoint-max-node-bytes "${FORMAL_LORA_CHECKPOINT_MAX_NODE_BYTES}" \
  --lora-adapter-resume-load "${LORA_ADAPTER_RESUME_LOAD}" \
  --overlong-penalty \
  --overlong-buffer-len "${OVERLONG_BUFFER_LEN}" \
  --overlong-penalty-factor "${OVERLONG_PENALTY_FACTOR}"
