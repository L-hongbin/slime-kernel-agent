#!/bin/bash

# Do not enable xtrace here: WANDB_API_KEY and other credentials are exported
# into Ray's runtime environment later in this script.
set -Ee
trap 'status=$?; echo "Script exiting with status ${status} at line ${LINENO}: ${BASH_COMMAND}"' EXIT
trap 'status=$?; echo "ERROR status ${status} at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1
ulimit -n 1048576 || true
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# MODEL CONFIG
source "${SCRIPT_DIR}/../../scripts/models/qwen3.5-27B.sh"
# NODE CONFIG
MASTER_ADDR="${MASTER_ADDR:-10.11.2.170}"
CONFIG_DRY_RUN="${CONFIG_DRY_RUN:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
REUSE_RAY_CLUSTER="${REUSE_RAY_CLUSTER:-0}"
FULL_LOOP_SMOKE="${FULL_LOOP_SMOKE:-0}"
DEBUG_ROLLOUT_ONLY="${DEBUG_ROLLOUT_ONLY:-0}"
DEBUG_ROLLOUT_TWO_NODE="${DEBUG_ROLLOUT_TWO_NODE:-0}"
USE_NODE64_ROLLOUT="${USE_NODE64_ROLLOUT:-1}"
LOAD_DEBUG_ROLLOUT_DATA="${LOAD_DEBUG_ROLLOUT_DATA:-}"
SAVE_FIRST_TRAIN_ROLLOUT="${SAVE_FIRST_TRAIN_ROLLOUT:-0}"
DISABLE_CHECKPOINT_SAVE="${DISABLE_CHECKPOINT_SAVE:-0}"
DISABLE_WANDB="${DISABLE_WANDB:-0}"
if [[ "${FULL_LOOP_SMOKE}" != "0" && "${FULL_LOOP_SMOKE}" != "1" ]]; then
   echo "FULL_LOOP_SMOKE must be 0 or 1." >&2
   exit 1
fi
if [[ "${REUSE_RAY_CLUSTER}" != "0" && "${REUSE_RAY_CLUSTER}" != "1" ]]; then
   echo "REUSE_RAY_CLUSTER must be 0 or 1." >&2
   exit 1
fi
if [[ "${DISABLE_WANDB}" != "0" && "${DISABLE_WANDB}" != "1" ]]; then
   echo "DISABLE_WANDB must be 0 or 1." >&2
   exit 1
fi
if [[ "${DEBUG_ROLLOUT_TWO_NODE}" != "0" && "${DEBUG_ROLLOUT_TWO_NODE}" != "1" ]]; then
   echo "DEBUG_ROLLOUT_TWO_NODE must be 0 or 1." >&2
   exit 1
fi
if [[ "${USE_NODE64_ROLLOUT}" != "0" && "${USE_NODE64_ROLLOUT}" != "1" ]]; then
   echo "USE_NODE64_ROLLOUT must be 0 or 1." >&2
   exit 1
fi
if [[ "${SAVE_FIRST_TRAIN_ROLLOUT}" != "0" && "${SAVE_FIRST_TRAIN_ROLLOUT}" != "1" ]]; then
   echo "SAVE_FIRST_TRAIN_ROLLOUT must be 0 or 1." >&2
   exit 1
fi
SAVE_DEBUG_ROLLOUT_MAX_ID=${SAVE_DEBUG_ROLLOUT_MAX_ID:-}
if [[ "${SAVE_FIRST_TRAIN_ROLLOUT}" == "1" ]]; then
   SAVE_DEBUG_ROLLOUT_MAX_ID=${SAVE_DEBUG_ROLLOUT_MAX_ID:-0}
fi
if [[ "${DEBUG_ROLLOUT_TWO_NODE}" == "1" && "${DEBUG_ROLLOUT_ONLY}" != "1" ]]; then
   echo "DEBUG_ROLLOUT_TWO_NODE is allowed only with DEBUG_ROLLOUT_ONLY=1." >&2
   exit 1
fi
# Use `hostname -I` (not `ip`, which is absent in some node containers, e.g. node62)
# to enumerate local IPv4s and confirm we are on the Ray head node.
if [[ "${CONFIG_DRY_RUN}" != "1" ]] && ! hostname -I 2>/dev/null | tr ' ' '\n' | grep -Fxq "${MASTER_ADDR}"; then
   echo "This script must run on the Ray head node (${MASTER_ADDR}); local node IPs are:"
   hostname -I 2>/dev/null
   exit 1
fi
# Qwen3.8 cluster: node70 (head + BF16 actor), node69 (BF16 actor), and
# node53/node64 (FP8 rollout). The dedicated containers listen on 23538 so this
# run does not reuse or stop the long-lived :23522 slime containers.
if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA}" ]]; then
   # Replay trains only the actor and never creates SGLang. Keep node53 and its
   # rollout-only services entirely out of this isolated training diagnostic.
   REMOTE_HOSTS=("10.11.2.169")
   REMOTE_PORTS=("23538")
   REMOTE_PLACEMENT_RESOURCES=("slime_actor")
elif [[ "${DEBUG_ROLLOUT_TWO_NODE}" == "1" ]]; then
   # node53 is optional for rollout-only diagnostics.  Split two TP4 engines
   # across node70 (head) and node69 (worker), and allocate no BF16 actor.
   REMOTE_HOSTS=("10.11.2.169")
   REMOTE_PORTS=("23538")
   REMOTE_PLACEMENT_RESOURCES=("slime_rollout")
elif [[ "${USE_NODE64_ROLLOUT}" == "1" ]]; then
   REMOTE_HOSTS=(
      "10.11.2.169"
      "10.11.2.153"
      "10.11.2.164"
   )
   REMOTE_PORTS=(
      "23538"
      "23538"
      "23538"
   )
   REMOTE_PLACEMENT_RESOURCES=(
      "slime_actor"
      "slime_rollout"
      "slime_rollout"
   )
else
   # Supported 24K three-node topology: node70+node69 host the BF16 actor;
   # node53 hosts two TP4 rollout engines. node64 remains outside the job.
   REMOTE_HOSTS=(
      "10.11.2.169"
      "10.11.2.153"
   )
   REMOTE_PORTS=(
      "23538"
      "23538"
   )
   REMOTE_PLACEMENT_RESOURCES=(
      "slime_actor"
      "slime_rollout"
   )
fi
if [ "${#REMOTE_PORTS[@]}" -ne "${#REMOTE_HOSTS[@]}" ]; then
   echo "REMOTE_PORTS length (${#REMOTE_PORTS[@]}) must match REMOTE_HOSTS length (${#REMOTE_HOSTS[@]})."
   exit 1
fi
if [ "${#REMOTE_PLACEMENT_RESOURCES[@]}" -ne "${#REMOTE_HOSTS[@]}" ]; then
   echo "REMOTE_PLACEMENT_RESOURCES length (${#REMOTE_PLACEMENT_RESOURCES[@]}) must match REMOTE_HOSTS length (${#REMOTE_HOSTS[@]})."
   exit 1
fi
NUM_NODES=$((1 + ${#REMOTE_HOSTS[@]}))
NUM_GPUS=$((NUM_NODES * 8))
if [[ "${DEBUG_ROLLOUT_TWO_NODE}" == "1" ]]; then
   ACTOR_NUM_NODES=0
else
   ACTOR_NUM_NODES=2
fi
ACTOR_GPUS_PER_NODE=8
ACTOR_GPUS=$((ACTOR_NUM_NODES*ACTOR_GPUS_PER_NODE))
DERIVED_ROLLOUT_GPUS=$((NUM_GPUS-ACTOR_GPUS))
if [[ "${DEBUG_ROLLOUT_TWO_NODE}" == "1" ]]; then
   ROLLOUT_GPUS=8
else
   ROLLOUT_GPUS=${DERIVED_ROLLOUT_GPUS}
fi
ACTOR_PLACEMENT_RESOURCE="slime_actor"
ROLLOUT_PLACEMENT_RESOURCE="slime_rollout"
ACTOR_RESOURCE_JSON="{\"${ACTOR_PLACEMENT_RESOURCE}\": ${ACTOR_GPUS_PER_NODE}}"
if [[ "${DEBUG_ROLLOUT_TWO_NODE}" == "1" ]]; then
   # Four custom-resource units per node force one TP4 engine onto each host.
   HEAD_RESOURCE_JSON="{\"${ROLLOUT_PLACEMENT_RESOURCE}\": 4}"
   ROLLOUT_RESOURCE_JSON="{\"${ROLLOUT_PLACEMENT_RESOURCE}\": 4}"
else
   HEAD_RESOURCE_JSON="${ACTOR_RESOURCE_JSON}"
   # Each rollout host contributes eight one-GPU placement bundles. Advertising
   # the cluster-wide total on every host would allow Ray to overpack all four
   # TP4 engines onto a single physical 8-GPU node.
   ROLLOUT_RESOURCE_JSON="{\"${ROLLOUT_PLACEMENT_RESOURCE}\": ${ACTOR_GPUS_PER_NODE}}"
fi
echo "ACTOR_GPUS ${ACTOR_GPUS} ROLLOUT_GPUS ${ROLLOUT_GPUS}"
# EXP CONFIG
MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-24576}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-${MAX_CONTEXT_LEN}}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
if ! [[ "${ROLLOUT_TEMPERATURE}" =~ ^[0-9]+([.][0-9]+)?$ ]] \
   || [[ "${ROLLOUT_TEMPERATURE}" =~ ^0+([.]0+)?$ ]]; then
   echo "ROLLOUT_TEMPERATURE must be a positive decimal number." >&2
   exit 1
fi
# Medium is the production policy. Qwen3.8's asymmetric template leaves medium
# unprompted, while xhigh explicitly encourages long alternative exploration
# that routinely outgrows this launcher's 24K response budget.
ROLLOUT_REASONING_EFFORT=${ROLLOUT_REASONING_EFFORT:-medium}
case "${ROLLOUT_REASONING_EFFORT}" in
   xhigh|medium|low) ;;
   *)
      echo "ROLLOUT_REASONING_EFFORT must be one of: xhigh, medium, low." >&2
      exit 1
      ;;
esac
CHAT_TEMPLATE_KWARGS="{\"enable_thinking\":true,\"reasoning_effort\":\"${ROLLOUT_REASONING_EFFORT}\"}"
# FP8 rollout and BF16 training are deliberately different numerical policies.
# Production uses the predictive DPPO Top-K KL mask directly against the
# behavior-policy probabilities returned by SGLang.  TIS and hard sequence MIS
# remain available only for isolated historical diagnostics; production and the
# full-loop gate fail closed unless predictive DPPO is selected.
ROLLOUT_CORRECTION_MODE=${ROLLOUT_CORRECTION_MODE:-dppo_predictive}
case "${ROLLOUT_CORRECTION_MODE}" in
   dppo_predictive|tis|hard_sequence_mis) ;;
   *)
      echo "ROLLOUT_CORRECTION_MODE must be one of: dppo_predictive, tis, hard_sequence_mis." >&2
      exit 1
      ;;
esac
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-128}
# Maintained Qwen3.8 training is no-spec only. DSpark was intentionally removed
# after failing the production-maturity gate; MTP remains outside this launcher.
SGLANG_SPECULATIVE_LABEL="NoSpec"

# Training kernel contract. FlashQLA is the production GDN backend. FLA remains
# an opt-in fixed-replay diagnostic for isolating recurrent-kernel regressions.
QWEN_GDN_BACKEND=${QWEN_GDN_BACKEND:-flashqla}
case "${QWEN_GDN_BACKEND}" in
   flashqla|fla) ;;
   *)
      echo "QWEN_GDN_BACKEND must be one of: flashqla, fla." >&2
      exit 1
      ;;
esac
QWEN_GDN_IMPLEMENTATION=${QWEN_GDN_IMPLEMENTATION:-distributed}
case "${QWEN_GDN_IMPLEMENTATION}" in
   replicated|distributed) ;;
   *)
      echo "QWEN_GDN_IMPLEMENTATION must be one of: replicated, distributed." >&2
      exit 1
      ;;
esac
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-2}
if ! [[ "${CONTEXT_PARALLEL_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
   echo "CONTEXT_PARALLEL_SIZE must be a positive integer." >&2
   exit 1
fi
CP_PARTITION_MODE=${CP_PARTITION_MODE:-zigzag}
case "${CP_PARTITION_MODE}" in
   zigzag|contiguous) ;;
   *)
      echo "CP_PARTITION_MODE must be one of: zigzag, contiguous." >&2
      exit 1
      ;;
esac
DEFAULT_ENABLE_SEQUENCE_PARALLEL=0
if [[ "${QWEN_GDN_IMPLEMENTATION}" == "distributed" ]]; then
   DEFAULT_ENABLE_SEQUENCE_PARALLEL=1
fi
ENABLE_SEQUENCE_PARALLEL=${ENABLE_SEQUENCE_PARALLEL:-${DEFAULT_ENABLE_SEQUENCE_PARALLEL}}
if [[ "${ENABLE_SEQUENCE_PARALLEL}" != "0" && "${ENABLE_SEQUENCE_PARALLEL}" != "1" ]]; then
   echo "ENABLE_SEQUENCE_PARALLEL must be 0 or 1." >&2
   exit 1
fi
if [[ "${QWEN_GDN_IMPLEMENTATION}" == "distributed" \
   && "${CONTEXT_PARALLEL_SIZE}" -gt 1 \
   && "${CP_PARTITION_MODE}" != "zigzag" ]]; then
   echo "distributed Qwen GDN supports only CP_PARTITION_MODE=zigzag when context parallelism is enabled." >&2
   exit 1
fi

# All local actor ranks share a stable node-local Triton cache. The FLA source
# gate below prevents this cache from hiding a missing varlen specialization fix.
TRAIN_TRITON_CACHE_DIR=${TRAIN_TRITON_CACHE_DIR:-/tmp/qwen38_train_triton_cache}
if [[ "${TRAIN_TRITON_CACHE_DIR}" != /* || "${TRAIN_TRITON_CACHE_DIR}" == "/" ]]; then
   echo "TRAIN_TRITON_CACHE_DIR must be an absolute non-root path." >&2
   exit 1
fi

# Profiling remains a train-only diagnostic.
TRAIN_PYTORCH_PROFILE=${TRAIN_PYTORCH_PROFILE:-0}
if [[ "${TRAIN_PYTORCH_PROFILE}" != "0" && "${TRAIN_PYTORCH_PROFILE}" != "1" ]]; then
   echo "TRAIN_PYTORCH_PROFILE must be 0 or 1." >&2
   exit 1
fi
# Fixed-data A/B disproved longest-first ordering as the training-stall fix.
# It remains opt-in, but may be selected for a formal run as an execution-order
# preference; it does not change DP assignment or microbatch contents.
SORT_TRAIN_MICROBATCHES_BY_PADDED_LENGTH_DESC=${SORT_TRAIN_MICROBATCHES_BY_PADDED_LENGTH_DESC:-0}
if [[ "${SORT_TRAIN_MICROBATCHES_BY_PADDED_LENGTH_DESC}" != "0" \
   && "${SORT_TRAIN_MICROBATCHES_BY_PADDED_LENGTH_DESC}" != "1" ]]; then
   echo "SORT_TRAIN_MICROBATCHES_BY_PADDED_LENGTH_DESC must be 0 or 1." >&2
   exit 1
fi

# `matched` retains the audited FP8 target settings. `cookbook` follows the
# Qwen3.8 serving recipe while keeping the GDN/linear-attention backend on
# Triton because FlashInfer GDN caused a measured train-rollout mismatch.
SGLANG_SERVING_PROFILE=${SGLANG_SERVING_PROFILE:-matched}
case "${SGLANG_SERVING_PROFILE}" in
   matched)
      SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.85}
      SGLANG_TARGET_ATTENTION_BACKEND=${SGLANG_TARGET_ATTENTION_BACKEND:-fa3}
      SGLANG_KV_CACHE_DTYPE=${SGLANG_KV_CACHE_DTYPE:-}
      SGLANG_CHUNKED_PREFILL_SIZE=${SGLANG_CHUNKED_PREFILL_SIZE:-}
      SGLANG_MAX_PREFILL_TOKENS=${SGLANG_MAX_PREFILL_TOKENS:-}
      SGLANG_MAMBA_FULL_MEMORY_RATIO=${SGLANG_MAMBA_FULL_MEMORY_RATIO:-}
      SGLANG_MAMBA_SSM_DTYPE=${SGLANG_MAMBA_SSM_DTYPE:-}
      SGLANG_MAMBA_RADIX_CACHE_STRATEGY=${SGLANG_MAMBA_RADIX_CACHE_STRATEGY:-extra_buffer}
      ;;
   cookbook)
      SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.85}
      SGLANG_TARGET_ATTENTION_BACKEND=${SGLANG_TARGET_ATTENTION_BACKEND:-flashinfer}
      SGLANG_KV_CACHE_DTYPE=${SGLANG_KV_CACHE_DTYPE:-fp8_e4m3}
      SGLANG_CHUNKED_PREFILL_SIZE=${SGLANG_CHUNKED_PREFILL_SIZE:-32768}
      SGLANG_MAX_PREFILL_TOKENS=${SGLANG_MAX_PREFILL_TOKENS:-32768}
      SGLANG_MAMBA_FULL_MEMORY_RATIO=${SGLANG_MAMBA_FULL_MEMORY_RATIO:-7.34}
      SGLANG_MAMBA_SSM_DTYPE=${SGLANG_MAMBA_SSM_DTYPE:-float32}
      SGLANG_MAMBA_RADIX_CACHE_STRATEGY=${SGLANG_MAMBA_RADIX_CACHE_STRATEGY:-extra_buffer_lazy}
      ;;
   *)
      echo "SGLANG_SERVING_PROFILE must be one of: matched, cookbook." >&2
      exit 1
      ;;
esac
if [[ "${FULL_LOOP_SMOKE}" == "1" && "${DEBUG_ROLLOUT_ONLY}" != "0" ]]; then
   echo "FULL_LOOP_SMOKE is a train+rollout gate and requires DEBUG_ROLLOUT_ONLY=0." >&2
   exit 1
fi
if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA}" \
   && ("${FULL_LOOP_SMOKE}" == "1" || "${DEBUG_ROLLOUT_ONLY}" == "1") ]]; then
   echo "LOAD_DEBUG_ROLLOUT_DATA cannot be combined with FULL_LOOP_SMOKE or DEBUG_ROLLOUT_ONLY." >&2
   exit 1
fi
if [[ "${SAVE_FIRST_TRAIN_ROLLOUT}" == "1" \
   && ("${FULL_LOOP_SMOKE}" == "1" || "${DEBUG_ROLLOUT_ONLY}" == "1" || -n "${LOAD_DEBUG_ROLLOUT_DATA}") ]]; then
   echo "SAVE_FIRST_TRAIN_ROLLOUT=1 is for a formal train+rollout run and cannot be combined with debug replay/smoke modes." >&2
   exit 1
fi
if [[ "${DISABLE_CHECKPOINT_SAVE}" != "0" && "${DISABLE_CHECKPOINT_SAVE}" != "1" ]]; then
   echo "DISABLE_CHECKPOINT_SAVE must be 0 or 1." >&2
   exit 1
fi
if [[ "${DISABLE_CHECKPOINT_SAVE}" == "1" \
   && -z "${LOAD_DEBUG_ROLLOUT_DATA}" \
   && "${FULL_LOOP_SMOKE}" != "1" ]]; then
   echo "DISABLE_CHECKPOINT_SAVE=1 is allowed only with LOAD_DEBUG_ROLLOUT_DATA or FULL_LOOP_SMOKE=1." >&2
   exit 1
fi
if [[ "${DISABLE_CHECKPOINT_SAVE}" == "1" && "${RESUME_FROM_SAVE:-0}" == "1" ]]; then
   echo "DISABLE_CHECKPOINT_SAVE=1 cannot be combined with RESUME_FROM_SAVE=1." >&2
   exit 1
fi
if [[ -n "${DEBUG_ROLLOUT_NUM_GPUS:-}" ]]; then
   if [[ "${DEBUG_ROLLOUT_ONLY}" != "1" ]]; then
      echo "DEBUG_ROLLOUT_NUM_GPUS is allowed only with DEBUG_ROLLOUT_ONLY=1." >&2
      exit 1
   fi
   if ! [[ "${DEBUG_ROLLOUT_NUM_GPUS}" =~ ^[0-9]+$ ]] \
      || (( DEBUG_ROLLOUT_NUM_GPUS < 4 )) \
      || (( DEBUG_ROLLOUT_NUM_GPUS > DERIVED_ROLLOUT_GPUS )) \
      || (( DEBUG_ROLLOUT_NUM_GPUS % 4 != 0 )); then
      echo "DEBUG_ROLLOUT_NUM_GPUS must be a multiple of 4 between 4 and ${DERIVED_ROLLOUT_GPUS}." >&2
      exit 1
   fi
   ROLLOUT_GPUS=${DEBUG_ROLLOUT_NUM_GPUS}
   if [[ "${DEBUG_ROLLOUT_TWO_NODE}" == "1" && "${DEBUG_ROLLOUT_NUM_GPUS}" != "8" ]]; then
      echo "DEBUG_ROLLOUT_TWO_NODE requires DEBUG_ROLLOUT_NUM_GPUS=8." >&2
      exit 1
   fi
   if [[ "${DEBUG_ROLLOUT_TWO_NODE}" != "1" ]]; then
      ROLLOUT_RESOURCE_JSON="{\"${ROLLOUT_PLACEMENT_RESOURCE}\": ${ROLLOUT_GPUS}}"
   fi
   echo "debug-rollout-only: limiting rollout allocation to ${ROLLOUT_GPUS} GPUs"
fi
if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA}" ]]; then
   NUM_ROLLOUT=${NUM_ROLLOUT:-1}
   ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-2}
   N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}
elif [[ "${FULL_LOOP_SMOKE}" == "1" ]]; then
   NUM_ROLLOUT=${NUM_ROLLOUT:-1}
   ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-2}
   N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}
   if [[ "${SGLANG_SERVING_PROFILE}" != "matched" \
      || "${ROLLOUT_REASONING_EFFORT}" != "medium" \
      || "${ROLLOUT_CORRECTION_MODE}" != "dppo_predictive" \
      || "${NUM_ROLLOUT}" != "1" \
      || "${ROLLOUT_BATCH_SIZE}" != "2" \
      || "${N_SAMPLES_PER_PROMPT}" != "16" ]]; then
      echo "FULL_LOOP_SMOKE requires no-spec/matched/medium/dppo_predictive, num_rollout=1, rollout_batch_size=2, n_samples_per_prompt=16." >&2
      exit 1
   fi
elif [[ "${DEBUG_ROLLOUT_ONLY}" == "1" ]]; then
   NUM_ROLLOUT=${NUM_ROLLOUT:-1}
   ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
   N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}
else
   NUM_ROLLOUT=${NUM_ROLLOUT:-3000}
   ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
   N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}
fi
if [[ "${FULL_LOOP_SMOKE}" == "0" \
   && "${DEBUG_ROLLOUT_ONLY}" == "0" \
   && -z "${LOAD_DEBUG_ROLLOUT_DATA}" \
   && "${ROLLOUT_CORRECTION_MODE}" != "dppo_predictive" ]]; then
   echo "Production training requires ROLLOUT_CORRECTION_MODE=dppo_predictive; TIS and hard_sequence_mis are diagnostic-only." >&2
   exit 1
fi
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
MODEL_NAME="Qwen3.8-27B"
KERNEL_BACKEND="tvm_ffi"
KERNEL_ENV_URL="${KERNEL_ENV_URL:-http://127.0.0.1:20211}"

# Training stays BF16 by loading the torch_dist conversion of the BF16 HF
# checkpoint.  --hf-checkpoint intentionally points at the FP8 checkpoint:
# slime reads its quantization_config when converting each BF16 actor update
# for the FP8 SGLang rollout engine.
BF16_MODEL_PATH="${BF16_MODEL_PATH:-/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.8-27B}"
HF_MODEL_PATH="${HF_MODEL_PATH:-/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.8-27B-FP8}"
# Keep the large immutable torch_dist input on /mnt/data (mounted as
# /nfs/LOCAL) so /mnt/md1 remains available for full-parameter optimizer saves.
MEGATRON_MODEL_PATH="${MEGATRON_MODEL_PATH:-/nfs/LOCAL/chenshuailin/checkpoints/Qwen/Qwen3.8-27B/torch_dist_tp4_pp2_distributed_gdn_flashqla}"
DEFAULT_RL_DATA="${REPO_ROOT}/Data/prompt_tvm_v4/release/train.parquet"
VERIFIED_CLUSTER_RL_DATA="/nfs/FM/chenshuailin/projects/kernel_agents/slime-dev-csl-2-qwen38-rl-20260819/Data/prompt_tvm_v4/release/train.parquet"
if [[ ! -f "${DEFAULT_RL_DATA}" && -f "${VERIFIED_CLUSTER_RL_DATA}" ]]; then
   DEFAULT_RL_DATA="${VERIFIED_CLUSTER_RL_DATA}"
fi
RL_DATA="${RL_DATA:-${DEFAULT_RL_DATA}}"
# Keep data revisions in the experiment identity. Multiple Qwen3.8 lineages can
# otherwise have identical hyperparameters and collide in logs/checkpoints even
# though their rollout distributions are not resumable across each other.
TRAIN_DATA_LABEL="${TRAIN_DATA_LABEL:-DataV4}"
if ! [[ "${TRAIN_DATA_LABEL}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
   echo "TRAIN_DATA_LABEL must contain only letters, digits, dot, underscore, or hyphen and start with an alphanumeric character." >&2
   exit 1
fi
MULTI_TURN_PROMPT_CONFIG="${MULTI_TURN_PROMPT_CONFIG:-${SCRIPT_DIR}/prompt_config/initial_prompt/multi_turn_cuda_kernel.yaml}"
OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-4096}"
OVERLONG_PENALTY_FACTOR="${OVERLONG_PENALTY_FACTOR:-0.2}"
OUTPUT_MISMATCH_PARTIAL_REWARD="${OUTPUT_MISMATCH_PARTIAL_REWARD:-0.25}"
if ! [[ "${OUTPUT_MISMATCH_PARTIAL_REWARD}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
   echo "OUTPUT_MISMATCH_PARTIAL_REWARD must be a non-negative decimal number." >&2
   exit 1
fi
if ! python3 - "${OUTPUT_MISMATCH_PARTIAL_REWARD}" <<'PY'
import sys

raise SystemExit(0 if float(sys.argv[1]) < 0.5 else 1)
PY
then
   echo "OUTPUT_MISMATCH_PARTIAL_REWARD must stay below the 0.5 correctness floor." >&2
   exit 1
fi
PARTIAL_REWARD_TAG="${OUTPUT_MISMATCH_PARTIAL_REWARD/./p}"
PENALTY_FACTOR_TAG="${OVERLONG_PENALTY_FACTOR/./p}"
REWARD_POLICY_LABEL="Mismatch${PARTIAL_REWARD_TAG}.NoPRS.Len${OVERLONG_BUFFER_LEN}Pen${PENALTY_FACTOR_TAG}"


case "${ROLLOUT_CORRECTION_MODE}" in
   dppo_predictive) POLICY_OPTIMIZATION_LABEL="DPPOPredictive" ;;
   tis) POLICY_OPTIMIZATION_LABEL="TISDiagnostic" ;;
   hard_sequence_mis) POLICY_OPTIMIZATION_LABEL="HardSequenceMISDiagnostic" ;;
esac
CUDA_GRAPH_LABEL="DefaultCG"
TRAIN_ORDER_LABEL=""
if [[ "${SORT_TRAIN_MICROBATCHES_BY_PADDED_LENGTH_DESC}" == "1" ]]; then
   TRAIN_ORDER_LABEL=".LongestFirst"
fi
EXP_NAME="FAsync.${SGLANG_SPECULATIVE_LABEL}.${CUDA_GRAPH_LABEL}.${POLICY_OPTIMIZATION_LABEL}${TRAIN_ORDER_LABEL}.${SGLANG_SERVING_PROFILE}.${ROLLOUT_REASONING_EFFORT}.Temp${ROLLOUT_TEMPERATURE}.${REWARD_POLICY_LABEL}.${TRAIN_DATA_LABEL}.${KERNEL_BACKEND}.${MODEL_NAME}.BF16Train.FP8Rollout.CTX${MAX_CONTEXT_LEN}"
EXP_ROOT="${REPO_ROOT}/experiments/${EXP_NAME}"
CHECKPOINT_SAVE_PATH="${CHECKPOINT_SAVE_PATH:-${EXP_ROOT}/checkpoints}"
MIN_CHECKPOINT_FREE_GIB="${MIN_CHECKPOINT_FREE_GIB:-180}"
if ! [[ "${MIN_CHECKPOINT_FREE_GIB}" =~ ^[1-9][0-9]*$ ]]; then
   echo "MIN_CHECKPOINT_FREE_GIB must be a positive integer." >&2
   exit 1
fi
DEFAULT_DEBUG_ROLLOUT_DATA="${EXP_ROOT}/debug_rollout/rollout_{rollout_id}.pt"
DEFAULT_FULL_LOOP_SMOKE_DATA="${EXP_ROOT}/full_loop_smoke/rollout_{rollout_id}.pt"
DEFAULT_FIRST_TRAIN_ROLLOUT_DATA="${EXP_ROOT}/train_rollout_capture/rollout_{rollout_id}.pt"
if [[ "${FULL_LOOP_SMOKE}" == "1" ]]; then
   DEBUG_ROLLOUT_DATA="${DEBUG_ROLLOUT_DATA:-${DEFAULT_FULL_LOOP_SMOKE_DATA}}"
   KERNEL_AGENT_GENERATE_GUARD_SEC="${KERNEL_AGENT_GENERATE_GUARD_SEC:-3600}"
   FULL_LOOP_SMOKE_TIMEOUT_SEC="${FULL_LOOP_SMOKE_TIMEOUT_SEC:-14400}"
   if ! [[ "${KERNEL_AGENT_GENERATE_GUARD_SEC}" =~ ^[1-9][0-9]*$ \
      && "${FULL_LOOP_SMOKE_TIMEOUT_SEC}" =~ ^[1-9][0-9]*$ ]]; then
      echo "Smoke guard and timeout values must be positive integers." >&2
      exit 1
   fi
elif [[ "${SAVE_FIRST_TRAIN_ROLLOUT}" == "1" ]]; then
   DEBUG_ROLLOUT_DATA="${DEBUG_ROLLOUT_DATA:-${DEFAULT_FIRST_TRAIN_ROLLOUT_DATA}}"
   KERNEL_AGENT_GENERATE_GUARD_SEC="${KERNEL_AGENT_GENERATE_GUARD_SEC:-}"
   FULL_LOOP_SMOKE_TIMEOUT_SEC="${FULL_LOOP_SMOKE_TIMEOUT_SEC:-}"
else
   DEBUG_ROLLOUT_DATA="${DEBUG_ROLLOUT_DATA:-${DEFAULT_DEBUG_ROLLOUT_DATA}}"
   KERNEL_AGENT_GENERATE_GUARD_SEC="${KERNEL_AGENT_GENERATE_GUARD_SEC:-}"
   FULL_LOOP_SMOKE_TIMEOUT_SEC="${FULL_LOOP_SMOKE_TIMEOUT_SEC:-}"
fi

WANDB_KEY_FILE=${WANDB_KEY_FILE:-${HOME}/.config/wandb/slime.key}
if [[ "${DISABLE_WANDB}" == "1" ]]; then
   # Keep credentials out of Ray's runtime_env when tracking is disabled.
   unset WANDB_API_KEY
else
   if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_KEY_FILE}" ]]; then
      WANDB_API_KEY="$(tr -d '[:space:]' < "${WANDB_KEY_FILE}")"
   fi
   export WANDB_API_KEY
fi
DEFAULT_WANDB_GROUP="Qwen38.${SGLANG_SPECULATIVE_LABEL}.${POLICY_OPTIMIZATION_LABEL}${TRAIN_ORDER_LABEL}.${SGLANG_SERVING_PROFILE}.${ROLLOUT_REASONING_EFFORT}.T${ROLLOUT_TEMPERATURE}.${REWARD_POLICY_LABEL}.${TRAIN_DATA_LABEL}"
WANDB_GROUP=${WANDB_GROUP:-${DEFAULT_WANDB_GROUP}}
if (( ${#WANDB_GROUP} > 128 )); then
   echo "WANDB_GROUP exceeds the W&B 128-character GroupName limit." >&2
   exit 1
fi

NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
# mlx5_5 is physically unstable on node69 (link-down events and saturated
# receive-error counters).  Keep every Qwen3.8 path, including train-only
# replay, on the audited healthy HCAs instead of relying on an outer shell to
# remember this production constraint.
NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_3,mlx5_4}"
if [[ -z "${NCCL_IB_HCA}" || ",${NCCL_IB_HCA}," == *",mlx5_5,"* ]]; then
   echo "NCCL_IB_HCA must be non-empty and must exclude unstable mlx5_5." >&2
   exit 1
fi
LOCAL_GLOO_SOCKET_IFNAME="${LOCAL_GLOO_SOCKET_IFNAME:-bond0}"
REMOTE_GLOO_SOCKET_IFNAMES=()
for _ in "${REMOTE_HOSTS[@]}"; do
   REMOTE_GLOO_SOCKET_IFNAMES+=("bond0")
done
if [ "${#REMOTE_GLOO_SOCKET_IFNAMES[@]}" -ne "${#REMOTE_HOSTS[@]}" ]; then
   echo "REMOTE_GLOO_SOCKET_IFNAMES length (${#REMOTE_GLOO_SOCKET_IFNAMES[@]}) must match REMOTE_HOSTS length (${#REMOTE_HOSTS[@]})."
   exit 1
fi

# Dedicated host-network ports: node53 already has another user's Ray on 6379.
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8268}"
RAY_PORT="${RAY_PORT:-6388}"
RAY_HEAD_ADDR="${MASTER_ADDR}:${RAY_PORT}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_qwen38_nospec}"
RAY_NODE_MANAGER_PORT="${RAY_NODE_MANAGER_PORT:-23901}"
RAY_OBJECT_MANAGER_PORT="${RAY_OBJECT_MANAGER_PORT:-23902}"
RAY_DASHBOARD_AGENT_LISTEN_PORT="${RAY_DASHBOARD_AGENT_LISTEN_PORT:-23903}"
RAY_DASHBOARD_AGENT_GRPC_PORT="${RAY_DASHBOARD_AGENT_GRPC_PORT:-23904}"
RAY_METRICS_EXPORT_PORT="${RAY_METRICS_EXPORT_PORT:-23905}"
RAY_CLIENT_SERVER_PORT="${RAY_CLIENT_SERVER_PORT:-23906}"
RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-24000}"
RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-24999}"
RAY_WORKER_PORT_COUNT=$((RAY_MAX_WORKER_PORT - RAY_MIN_WORKER_PORT + 1))
if ((RAY_WORKER_PORT_COUNT < 512)); then
   echo "Ray worker port range must provide at least 512 ports; got ${RAY_MIN_WORKER_PORT}-${RAY_MAX_WORKER_PORT} (${RAY_WORKER_PORT_COUNT})." >&2
   exit 1
fi

PYTHON_BIN=${PYTHON_BIN:-python3}
RAY_WAIT_TIMEOUT=${RAY_WAIT_TIMEOUT:-300}
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
CUDA_BIN_DIR="${CUDA_HOME}/bin"
CUDA_LIB_DIR="${CUDA_HOME}/lib64"
RUNTIME_PATH="${CUDA_BIN_DIR}:${PATH}"
RUNTIME_LD_LIBRARY_PATH="${CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}"

HAS_NVLINK="${HAS_NVLINK:-1}"


run_ssh() {
   local host="$1"
   local port="$2"
   shift 2
   # accept-new: auto-add unseen host keys (fresh head/remote pairs) without a
   # prompt, but still reject changed keys. Avoids "Host key verification failed".
   ssh -p "${port}" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "${host}" "$@"
}

shell_quote() {
   printf "%q" "$1"
}

read -r -d '' TILELANG_CUDA_ATOMIC_CHECK_PY <<'PY' || true
import os
import shutil
import subprocess
import sys
import tempfile

label = sys.argv[1]
nvcc = sys.argv[2]
cuda_home = sys.argv[3]

if not os.path.exists(nvcc):
    nvcc = shutil.which("nvcc") or nvcc
if not os.path.exists(nvcc):
    raise SystemExit(f"{label}: nvcc not found: {nvcc}")

cuda_include = os.path.join(cuda_home, "include")
required_headers = [
    os.path.join(cuda_include, "cuda", "atomic"),
    os.path.join(cuda_include, "nv", "target"),
]
missing = [path for path in required_headers if not os.path.exists(path)]
if missing:
    raise SystemExit(f"{label}: missing CUDA 12.9 CCCL/libcu++ headers: {missing}")

import tilelang.contrib.nvcc as tl_nvcc

tl_compiler = tl_nvcc.get_nvcc_compiler()
if os.path.realpath(tl_compiler) != os.path.realpath(nvcc):
    raise SystemExit(f"{label}: TileLang selected {tl_compiler}, expected {nvcc}")

source = (
    "#include <cuda/atomic>\n"
    "extern \"C\" __global__ void k(int* out) {\n"
    "  cuda::atomic_ref<int, cuda::thread_scope_device> r(*out);\n"
    "  r.store(1);\n"
    "}\n"
)
with tempfile.TemporaryDirectory(prefix="tilelang_cuda_atomic_") as tmpdir:
    src = os.path.join(tmpdir, "test.cu")
    cubin = os.path.join(tmpdir, "test.cubin")
    with open(src, "w", encoding="utf-8") as f:
        f.write(source)
    cmd = [
        nvcc,
        "--cubin",
        "-O3",
        "-arch=sm_90a",
        "-std=c++17",
        "-o",
        cubin,
        src,
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(f"{label}: TileLang CUDA atomic compile check failed")
    print(f"{label}: TileLang CUDA atomic compile check passed with {nvcc}")
PY

read -r -d '' QWEN38_RUNTIME_CHECK_PY <<'PY' || true
import importlib.metadata
import inspect
import os
import sys

import sglang
import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from scripts.patch_fla_varlen_autotune_nb import patch_files as check_fla_varlen_patch

label = sys.argv[1]
try:
    check_fla_varlen_patch(check_only=True)
except (OSError, RuntimeError) as exc:
    raise SystemExit(f"{label}: FLA varlen autotune patch gate failed: {exc}") from exc
gpt_model_source = os.path.realpath(inspect.getsourcefile(GPTModel) or "")
expected_gpt_model_source = os.path.realpath(
    "/root/Megatron-LM/megatron/core/models/gpt/gpt_model.py"
)
if gpt_model_source != expected_gpt_model_source:
    raise SystemExit(
        f"{label}: imported GPTModel from {gpt_model_source}, "
        f"expected training source {expected_gpt_model_source}"
    )
mbridge_version = importlib.metadata.version("mbridge")
if mbridge_version != "0.15.1":
    raise SystemExit(f"{label}: mbridge 0.15.1 is required, got {mbridge_version}")
transformers_version = importlib.metadata.version("transformers")
if transformers_version != "5.12.1":
    raise SystemExit(
        f"{label}: transformers 5.12.1 is required, got {transformers_version}"
    )
if torch.__version__ != "2.11.0+cu129":
    raise SystemExit(
        f"{label}: torch 2.11.0+cu129 is required, got {torch.__version__}"
    )
version = sglang.__version__
if version != "0.5.16":
    raise SystemExit(f"{label}: SGLang 0.5.16 is required, got {version}")
print(
    f"{label}: runtime capability check passed "
    f"(mbridge={mbridge_version}, transformers={transformers_version}, "
    f"torch={torch.__version__}, sglang={version}, speculative=disabled, "
    "fla_varlen_autotune=nb-removed)"
)
PY

check_local_tilelang_cuda_headers() {
   local label="$1"
   echo "Checking TileLang CUDA headers on ${label}"
   CUDA_HOME="${CUDA_HOME}" CUDA_PATH="${CUDA_PATH}" PATH="${RUNTIME_PATH}" LD_LIBRARY_PATH="${RUNTIME_LD_LIBRARY_PATH}" \
      "${PYTHON_BIN}" - "${label}" "${CUDA_BIN_DIR}/nvcc" "${CUDA_HOME}" \
      <<<"${TILELANG_CUDA_ATOMIC_CHECK_PY}"
}

check_remote_tilelang_cuda_headers() {
   local host="$1"
   local port="$2"
   local label="$3"
   local quoted_python quoted_label quoted_cuda_home quoted_cuda_path quoted_path quoted_ld_library_path

   quoted_python="$(shell_quote "${PYTHON_BIN}")"
   quoted_label="$(shell_quote "${label}")"
   quoted_cuda_home="$(shell_quote "${CUDA_HOME}")"
   quoted_cuda_path="$(shell_quote "${CUDA_PATH}")"
   quoted_path="$(shell_quote "${RUNTIME_PATH}")"
   quoted_ld_library_path="$(shell_quote "${RUNTIME_LD_LIBRARY_PATH}")"

   echo "Checking TileLang CUDA headers on ${label}"
   run_ssh "${host}" "${port}" \
      "CUDA_HOME=${quoted_cuda_home} CUDA_PATH=${quoted_cuda_path} PATH=${quoted_path} LD_LIBRARY_PATH=${quoted_ld_library_path} ${quoted_python} - ${quoted_label} ${quoted_cuda_home}/bin/nvcc ${quoted_cuda_home}" \
      <<<"${TILELANG_CUDA_ATOMIC_CHECK_PY}"
}

check_all_tilelang_cuda_headers() {
   check_local_tilelang_cuda_headers "head-${MASTER_ADDR}"
   for i in "${!REMOTE_HOSTS[@]}"; do
      check_remote_tilelang_cuda_headers \
         "${REMOTE_HOSTS[$i]}" \
         "${REMOTE_PORTS[$i]}" \
         "worker-${REMOTE_HOSTS[$i]}"
   done
}

check_local_qwen38_runtime() {
   PYTHONPATH="${REPO_ROOT}:/root/Megatron-LM:${PYTHONPATH:-}" \
      "${PYTHON_BIN}" - "head-${MASTER_ADDR}" <<<"${QWEN38_RUNTIME_CHECK_PY}"
}

check_remote_qwen38_runtime() {
   local host="$1"
   local port="$2"
   local label="$3"
   run_ssh "${host}" "${port}" \
      "PYTHONPATH=$(shell_quote "${REPO_ROOT}:/root/Megatron-LM") ${PYTHON_BIN} - $(shell_quote "${label}")" \
      <<<"${QWEN38_RUNTIME_CHECK_PY}"
}

check_all_qwen38_runtime() {
   check_local_qwen38_runtime
   for i in "${!REMOTE_HOSTS[@]}"; do
      check_remote_qwen38_runtime \
         "${REMOTE_HOSTS[$i]}" \
         "${REMOTE_PORTS[$i]}" \
         "worker-${REMOTE_HOSTS[$i]}"
   done
}

print_runtime_gate_plan() {
   local i
   for i in "${!REMOTE_HOSTS[@]}"; do
      printf 'RUNTIME_GATE host=%s resource=%s speculative=disabled\n' \
         "${REMOTE_HOSTS[$i]}" \
         "${REMOTE_PLACEMENT_RESOURCES[$i]}"
   done
}

# Every node has a local KernelGym entry on 127.0.0.1:20211, so the health
# check runs on all nodes. /nfs/FM (this repo) is per-node local disk, so the
# health script is streamed to remote nodes over ssh before the check.
host_resource_check_args() {
   local -n args_ref=$1
   local health_script_path="${2:-${REPO_ROOT}/scripts/check_kernelgym_health.py}"
   args_ref=(
      --expected-gpus "8"
      # Default CPU-idle gate relaxed 50 -> 40: the node62 host runs sustained
      # ~49% idle from other tenants outside this container (container-visible
      # R=1), which still leaves ~110 of 224 cores free for this run. Override
      # via RESOURCE_IDLE_MIN_PERCENT.
      --idle-min-percent "${RESOURCE_IDLE_MIN_PERCENT:-40}"
      --kernelgym-url "${KERNEL_ENV_URL}"
      --kernelgym-health-script "${health_script_path}"
      --python-bin "${PYTHON_BIN}"
   )
   if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA}" ]]; then
      args_ref+=(--skip-kernelgym-health)
   else
      # KernelGym may be intentionally restarting. Retry indefinitely with
      # 2,4,...,2048,3600,3600,... second backoff after CPU/GPU pass.
      args_ref+=(
         --kernelgym-health-attempts "${KERNELGYM_HEALTH_ATTEMPTS:-0}"
         --kernelgym-health-interval "${KERNELGYM_HEALTH_INTERVAL:-2}"
         --kernelgym-health-backoff "${KERNELGYM_HEALTH_BACKOFF:-2}"
         --kernelgym-health-max-interval "${KERNELGYM_HEALTH_MAX_INTERVAL:-3600}"
      )
   fi
}

check_local_host_resources() {
   local label="$1"
   local args=()
   host_resource_check_args args

   "${REPO_ROOT}/scripts/check_host_resources.sh" \
      --label "${label}" \
      "${args[@]}"
}

check_remote_host_resources() {
   local host="$1"
   local port="$2"
   local label="$3"
   local args=()
   local remote_health_script="/tmp/slime_check_kernelgym_health.py"

   run_ssh "${host}" "${port}" "cat > ${remote_health_script}" \
      < "${REPO_ROOT}/scripts/check_kernelgym_health.py"
   host_resource_check_args args "${remote_health_script}"
   run_ssh "${host}" "${port}" bash -s -- --label "${label}" "${args[@]}" \
      < "${REPO_ROOT}/scripts/check_host_resources.sh"
}

check_all_host_resources() {
   echo "Checking KernelGym and CPU/GPU resources before Ray start"
   check_local_host_resources "head-${MASTER_ADDR}"
   for i in "${!REMOTE_HOSTS[@]}"; do
      check_remote_host_resources \
         "${REMOTE_HOSTS[$i]}" \
         "${REMOTE_PORTS[$i]}" \
         "worker-${REMOTE_HOSTS[$i]}"
   done
}

check_checkpoint_free_space() {
   if [[ "${DISABLE_CHECKPOINT_SAVE}" == "1" || "${DEBUG_ROLLOUT_ONLY}" == "1" ]]; then
      return
   fi

   local checkpoint_parent
   checkpoint_parent="$(dirname "${CHECKPOINT_SAVE_PATH}")"
   mkdir -p "${checkpoint_parent}"

   local min_free_kib=$((MIN_CHECKPOINT_FREE_GIB * 1024 * 1024))
   local free_kib
   free_kib="$(df -Pk -- "${checkpoint_parent}" | awk 'NR == 2 {print $4}')"
   if ! [[ "${free_kib}" =~ ^[0-9]+$ ]] || ((free_kib < min_free_kib)); then
      echo "head-${MASTER_ADDR}: checkpoint filesystem needs at least ${MIN_CHECKPOINT_FREE_GIB} GiB free at ${checkpoint_parent}; found $((free_kib / 1024 / 1024)) GiB." >&2
      exit 1
   fi

   local i
   for i in "${!REMOTE_HOSTS[@]}"; do
      if [[ "${REMOTE_PLACEMENT_RESOURCES[$i]}" != "${ACTOR_PLACEMENT_RESOURCE}" ]]; then
         continue
      fi
      run_ssh "${REMOTE_HOSTS[$i]}" "${REMOTE_PORTS[$i]}" \
         "mkdir -p $(shell_quote "${checkpoint_parent}")"
      free_kib="$(run_ssh "${REMOTE_HOSTS[$i]}" "${REMOTE_PORTS[$i]}" \
         "df -Pk -- $(shell_quote "${checkpoint_parent}") | awk 'NR == 2 {print \$4}'")"
      if ! [[ "${free_kib}" =~ ^[0-9]+$ ]] || ((free_kib < min_free_kib)); then
         echo "worker-${REMOTE_HOSTS[$i]}: checkpoint filesystem needs at least ${MIN_CHECKPOINT_FREE_GIB} GiB free at ${checkpoint_parent}; found $((free_kib / 1024 / 1024)) GiB." >&2
         exit 1
      fi
   done
   echo "Checkpoint capacity gate passed: at least ${MIN_CHECKPOINT_FREE_GIB} GiB free on every actor node."
}

wait_for_cluster() {
   echo "Waiting for Ray cluster: expected nodes=${NUM_NODES}, expected GPUs=${NUM_GPUS}"
   local deadline=$((SECONDS + RAY_WAIT_TIMEOUT))
   local ready=0

   while ((SECONDS < deadline)); do
      if "${PYTHON_BIN}" - "${RAY_HEAD_ADDR}" "${NUM_NODES}" "${NUM_GPUS}" "${ACTOR_PLACEMENT_RESOURCE}" "${ACTOR_GPUS}" "${ROLLOUT_PLACEMENT_RESOURCE}" "${ROLLOUT_GPUS}" <<'PY'
import sys

import ray

ray_address = sys.argv[1]
expected_nodes = int(sys.argv[2])
expected_gpus = float(sys.argv[3])
actor_resource = sys.argv[4]
expected_actor = float(sys.argv[5])
rollout_resource = sys.argv[6]
expected_rollout = float(sys.argv[7])

ray.init(address=ray_address, ignore_reinit_error=True, logging_level="ERROR")
alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
gpu_count = sum(float(node.get("Resources", {}).get("GPU", 0)) for node in alive_nodes)
actor_count = sum(float(node.get("Resources", {}).get(actor_resource, 0)) for node in alive_nodes)
rollout_count = sum(float(node.get("Resources", {}).get(rollout_resource, 0)) for node in alive_nodes)

print(
    f"Ray alive nodes={len(alive_nodes)}, GPUs={gpu_count:g}, "
    f"{actor_resource}={actor_count:g}, {rollout_resource}={rollout_count:g}"
)
ready = (
    len(alive_nodes) >= expected_nodes
    and gpu_count >= expected_gpus
    and actor_count >= expected_actor
    and rollout_count >= expected_rollout
)
sys.exit(0 if ready else 1)
PY
      then
         ready=1
         break
      fi
      sleep 5
   done

   ray status --address "${RAY_HEAD_ADDR}" || true

   if [[ "${ready}" != "1" ]]; then
      echo "Ray cluster did not become ready within ${RAY_WAIT_TIMEOUT}s." >&2
      exit 1
   fi
}

WANDB_ARGS=()
if [[ "${DISABLE_WANDB}" == "0" ]]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project slime
      --wandb-group ${WANDB_GROUP}
      --disable-wandb-random-suffix
      --wandb-always-use-train-step
      --wandb-centralized
   )
fi

LOGGING_ARGS=(
   --log-throughput
   --log-progress
   --log-device-memory-used
)

CKPT_ARGS=(
   --hf-checkpoint ${HF_MODEL_PATH}
   --ref-load ${MEGATRON_MODEL_PATH}
)
if [[ "${DISABLE_CHECKPOINT_SAVE}" == "0" ]]; then
   CKPT_ARGS+=(
      # /nfs/FM is per-node local disk: ranks write shards to their own node;
      # gather with scripts/sync/gather_convert_ckpt.sh afterwards.
      --save ${CHECKPOINT_SAVE_PATH}
      --save-interval 20
      # async save overlaps disk writes with the next train step; the worker
      # flag is required or Megatron disables --async-save. Keep the default
      # dp_reshardable optimizer format (do NOT add fully-reshardable).
      --async-save
      --use-persistent-ckpt-worker
   )
   if [[ "${RESUME_FROM_SAVE:-0}" == "1" ]]; then
      # Slime derives start_rollout_id from the latest finalized checkpoint.
      # A resumed formal run normally extends --num-rollout. Keep the current
      # launcher's constant-LR scheduler bounds instead of requiring its total
      # iteration count to equal the shorter checkpointed run exactly.
      CKPT_ARGS+=(
         --load ${CHECKPOINT_SAVE_PATH}
         --override-opt-param-scheduler
      )
   fi
fi

ROLLOUT_ARGS=(
   --rollout-function-path examples.kernel_agent.fully_async_rollout.generate_rollout_fully_async
   --update-weights-interval 1
   --prompt-data ${RL_DATA}
   --input-key prompt
   --label-key reward_model
   --metadata-key extra_info
   --rollout-shuffle
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
   --rollout-max-response-len $MAX_RESPONSE_LEN
   --rollout-max-context-len $MAX_CONTEXT_LEN
   --apply-chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}"
   --rollout-temperature ${ROLLOUT_TEMPERATURE}
   --rollout-top-p 1.0
   --rollout-top-k -1

   # eval args
   # --eval-interval 25
   # --eval-prompt-data nq_test /root/Search-R1/data/nq_hotpotqa_train/test.parquet@[0:3000]
   # # --eval-prompt-data nq_test /root/nq_search/test.parquet
   # --eval-input-key prompt
   # --eval-label-key reward_model
   # --n-samples-per-eval-prompt 1

   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --balance-data
)

# Predictive DPPO is anchored directly to the behavior distribution stored in
# each rollout, so it must not instantiate/recompute an old actor.  The legacy
# diagnostic modes retain their historical old-actor contract.
if [[ "${ROLLOUT_CORRECTION_MODE}" != "dppo_predictive" ]]; then
   ROLLOUT_ARGS+=(--keep-old-actor)
fi

# CURRICULUM_ARGS=(
#    --use-dynamic-curriculum
#    --difficulty-level-key difficulty_level
#    --difficulty-score-key difficulty_score
# )
# --num-layers-per-virtual-pipeline-stage 16
# --num-virtual-stages-per-pipeline-rank 2
# --decoder-last-pipeline-num-layers 30

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --pipeline-model-parallel-size 2
   --decoder-last-pipeline-num-layers 31
   --context-parallel-size ${CONTEXT_PARALLEL_SIZE}
   --cp-partition-mode "${CP_PARTITION_MODE}"
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --qwen-gdn-backend "${QWEN_GDN_BACKEND}"
   --qwen-gdn-implementation "${QWEN_GDN_IMPLEMENTATION}"

   --recompute-granularity full
   --recompute-method block
   --recompute-num-layers 29

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --calculate-per-token-loss
   --max-tokens-per-gpu 8192
   --log-probs-max-tokens-per-gpu 16384
   # --init-model-with-meta-device
)
if [[ "${ENABLE_SEQUENCE_PARALLEL}" == "1" ]]; then
   PERF_ARGS+=(--sequence-parallel)
fi
if [[ "${QWEN_GDN_IMPLEMENTATION}" == "distributed" && "${ENABLE_SEQUENCE_PARALLEL}" == "1" ]]; then
   PERF_ARGS+=(--qwen-gdn-sp-disable-batch-p2p-comm)
fi

if [[ "${SORT_TRAIN_MICROBATCHES_BY_PADDED_LENGTH_DESC}" == "1" ]]; then
   PERF_ARGS+=(--sort-train-microbatches-by-padded-length-desc)
fi

RL_ARGS=(
   --advantage-estimator trloo
   --multi-turn-gamma 1.0
   --entropy-coef 0.00

   # DAPO-style linear reward subtraction over the final response-token window.
   --overlong-penalty
   --overlong-use-effective-response-cap
   --overlong-buffer-len ${OVERLONG_BUFFER_LEN}
   --overlong-penalty-factor ${OVERLONG_PENALTY_FACTOR}

)

case "${ROLLOUT_CORRECTION_MODE}" in
   dppo_predictive)
      RL_ARGS+=(
         --policy-loss-mode dppo_topk_kl_predictive
         --use-rollout-logprobs
         --dppo-predictive-top-k 20
         --dppo-predictive-tail-estimator aggregated
         # Predictive DPPO has one KL trust-region threshold, not PPO's
         # asymmetric ratio bounds.  The paper's recommended main setting is
         # delta=0.15; cap the detached sampled-token importance ratio at 5.
         --eps-clip 0.15
         --eps-clip-high 0.15
         --eps-clip-c 5
      )
      ;;
   tis)
      RL_ARGS+=(--eps-clip 0.2 --eps-clip-high 0.28 --use-tis)
      ;;
   hard_sequence_mis)
      RL_ARGS+=(--eps-clip 0.2 --eps-clip-high 0.28)
      ;;
esac

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
   --use-distributed-optimizer
   --overlap-grad-reduce
   --overlap-param-gather
   --use-precision-aware-optimizer
   # --optimizer-cpu-offload
   # --overlap-cpu-optimizer-d2h-h2d
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-context-length ${MAX_CONTEXT_LEN}
   --sglang-max-running-requests ${SGLANG_MAX_RUNNING_REQUESTS}
   --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC}
   --sglang-decode-log-interval 400
   --router-policy round_robin
   --sglang-cuda-graph-max-bs ${SGLANG_MAX_RUNNING_REQUESTS}
   --sglang-disable-custom-all-reduce
   --sglang-attention-backend ${SGLANG_TARGET_ATTENTION_BACKEND}
   # triton (NOT flashinfer): the flashinfer GDN decode kernel diverges ~2e-3/layer
   # from the megatron chunkwise recompute on real inputs, compounding across the 48
   # GDN layers into a per-token logprob mismatch that collapses sequence_mis
   # (reject 96.5% -> 0% after switching to triton). Root cause = sglang#20791
   # (flashinfer GDN in-place state-pool aliasing); triton is the correct reference.
   --sglang-linear-attn-backend triton
   --sglang-mamba-backend triton
   --sglang-mamba-radix-cache-strategy ${SGLANG_MAMBA_RADIX_CACHE_STRATEGY}
)

if [[ -n "${SGLANG_KV_CACHE_DTYPE}" ]]; then
   SGLANG_ARGS+=(--sglang-kv-cache-dtype "${SGLANG_KV_CACHE_DTYPE}")
fi
if [[ -n "${SGLANG_CHUNKED_PREFILL_SIZE}" ]]; then
   SGLANG_ARGS+=(--sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE}")
fi
if [[ -n "${SGLANG_MAX_PREFILL_TOKENS}" ]]; then
   SGLANG_ARGS+=(--sglang-max-prefill-tokens "${SGLANG_MAX_PREFILL_TOKENS}")
fi
if [[ -n "${SGLANG_MAMBA_FULL_MEMORY_RATIO}" ]]; then
   SGLANG_ARGS+=(--sglang-mamba-full-memory-ratio "${SGLANG_MAMBA_FULL_MEMORY_RATIO}")
fi
if [[ -n "${SGLANG_MAMBA_SSM_DTYPE}" ]]; then
   SGLANG_ARGS+=(--sglang-mamba-ssm-dtype "${SGLANG_MAMBA_SSM_DTYPE}")
fi

echo "speculative decoding disabled for FP8 target control"

MISC_ARGS=(
   # slime also defaults to BF16 whenever --fp16 is absent. Keep this explicit
   # because the rollout HF checkpoint is FP8 and must not imply FP8 training.
   --bf16
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --log-probs-chunk-size 10000
   --update-weight-buffer-size 1073741824
   # need to comment this when using model with MLA
   --attention-backend flash
   --no-pin-cpu-grads
   --no-pin-cpu-params
)

if [[ "${TRAIN_PYTORCH_PROFILE}" == "1" ]]; then
   TRAIN_PROFILE_DIR=${TRAIN_PROFILE_DIR:-${EXP_ROOT}/train_profile}
   MISC_ARGS+=(
      --use-pytorch-profiler
      --profile-step-start 0
      --profile-step-end 1
      --tensorboard-dir "${TRAIN_PROFILE_DIR}"
   )
fi

CUSTOM_ARGS=(
   --custom-generate-function-path examples.kernel_agent.generate_with_cuda_agent.generate
   --custom-rm-path examples.kernel_agent.generate_with_cuda_agent.reward_func
   --custom-reward-post-process-path examples.kernel_agent.kernel_reward.reward_post_process_by_group
   --multi-turn-prompt-config-path "${MULTI_TURN_PROMPT_CONFIG}"
)

case "${ROLLOUT_CORRECTION_MODE}" in
   dppo_predictive)
      # No MIS/TIS postprocessor: the policy loss consumes rollout logprobs and
      # Top-K behavior support directly.
      ;;
   tis)
      CUSTOM_ARGS+=(
         --custom-config-path "${REPO_ROOT}/examples/train_infer_mismatch_helper/mis.yaml"
         --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
      )
      ;;
   hard_sequence_mis)
      CUSTOM_ARGS+=(
         --rollout-data-postprocess-path examples.kernel_agent.kernel_filter.sequence_mis
      )
      ;;
esac

# A one-sample debug group has reward std=0 by construction.  Sending it
# through the production dynamic-sampling filter makes debug-rollout-only
# discard every completed group and replenish forever.  Keep the production
# n=16 filter unchanged, and bypass it only for the n=1 diagnostic path.
if [[ "${DEBUG_ROLLOUT_ONLY}" != "1" || "${N_SAMPLES_PER_PROMPT}" -gt 1 ]]; then
   CUSTOM_ARGS+=(
      --dynamic-sampling-filter-path examples.kernel_agent.kernel_filter.filter_cuda_kernel_group
   )
else
   echo "debug-rollout-only with n_samples_per_prompt=1: bypassing zero-variance dynamic sampling filter"
fi

DEBUG_ARGS=()
if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA}" ]]; then
   DEBUG_ARGS+=(--load-debug-rollout-data "${LOAD_DEBUG_ROLLOUT_DATA}")
elif [[ "${DEBUG_ROLLOUT_ONLY}" == "1" ]]; then
   DEBUG_ARGS+=(
      --debug-rollout-only
      --save-debug-rollout-data "${DEBUG_ROLLOUT_DATA}"
   )
elif [[ "${FULL_LOOP_SMOKE}" == "1" ]]; then
   # Save the real rollout that is consumed by the one optimizer step.  This is
   # intentionally not debug-rollout-only: the gate must exercise train loss
   # and both the initial and post-step full-weight synchronization.
   DEBUG_ARGS+=(--save-debug-rollout-data "${DEBUG_ROLLOUT_DATA}")
elif [[ "${SAVE_FIRST_TRAIN_ROLLOUT}" == "1" ]]; then
   # Preserve exactly rollout 0 from a formal run so its dynamic microbatch
   # shapes can be replayed without growing one large dump per optimizer step.
   DEBUG_ARGS+=(--save-debug-rollout-data "${DEBUG_ROLLOUT_DATA}")
fi

KERNEL_AGENT_ARGS=(
   --kernel-env-url ${KERNEL_ENV_URL}
   --kernel-backend $KERNEL_BACKEND
   --reference-backend torch
   --do-precheck
   --use-reference-cache
   --finalize-mode positive
   # True single-turn rollout. The generator still requires max-turns=1, but
   # multi-turn list/padding/filter semantics are deliberately disabled.
   --max-turns 1
   --enable-turns-dp-partitions
)

if [[ "${ROLLOUT_CORRECTION_MODE}" == "hard_sequence_mis" ]]; then
   KERNEL_AGENT_ARGS+=(
      --sequence-mis-config '{"aggregation":"turns_geometric","token_veto_threshold":1e-4,"lower":0.999,"upper":1.001,"use_advantage":false}'
   )
fi

validate_local_inputs() {
   local required_paths=(
      "${BF16_MODEL_PATH}/config.json"
      "${HF_MODEL_PATH}/config.json"
      "${MEGATRON_MODEL_PATH}/latest_checkpointed_iteration.txt"
      "${RL_DATA}"
      "${MULTI_TURN_PROMPT_CONFIG}"
   )
   local path

   if ((MAX_RESPONSE_LEN > MAX_CONTEXT_LEN)); then
      echo "MAX_RESPONSE_LEN (${MAX_RESPONSE_LEN}) must not exceed MAX_CONTEXT_LEN (${MAX_CONTEXT_LEN})." >&2
      exit 1
   fi
   if ((OVERLONG_BUFFER_LEN <= 0 || OVERLONG_BUFFER_LEN >= MAX_RESPONSE_LEN)); then
      echo "OVERLONG_BUFFER_LEN (${OVERLONG_BUFFER_LEN}) must be between 1 and MAX_RESPONSE_LEN-1." >&2
      exit 1
   fi
   if [[ "${BF16_MODEL_PATH}" == "${HF_MODEL_PATH}" ]]; then
      echo "BF16_MODEL_PATH and HF_MODEL_PATH must be different checkpoints." >&2
      exit 1
   fi
   if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA}" ]]; then
      local replay_probe="${LOAD_DEBUG_ROLLOUT_DATA//\{rollout_id\}/0}"
      if [[ ! -f "${replay_probe}" ]]; then
         echo "Debug train replay input is missing: ${replay_probe}" >&2
         exit 1
      fi
   fi
   for path in "${required_paths[@]}"; do
      if [[ ! -e "${path}" ]]; then
         echo "Required input is missing: ${path}" >&2
         exit 1
      fi
   done

   "${PYTHON_BIN}" - "${BF16_MODEL_PATH}/config.json" "${HF_MODEL_PATH}/config.json" "${HF_MODEL_PATH}/model.safetensors.index.json" <<'PY'
import json
import os
import sys

bf16_path, fp8_path, fp8_index_path = sys.argv[1:]
with open(bf16_path, encoding="utf-8") as f:
    bf16 = json.load(f)
with open(fp8_path, encoding="utf-8") as f:
    fp8 = json.load(f)

if bf16.get("quantization_config"):
    raise SystemExit(f"BF16 training checkpoint unexpectedly declares quantization: {bf16_path}")
quant = fp8.get("quantization_config") or {}
method = str(quant.get("quant_method", "")).lower()
if "fp8" not in method:
    raise SystemExit(f"Rollout checkpoint does not declare FP8 quantization: {fp8_path}")
with open(fp8_index_path, encoding="utf-8") as f:
    weight_map = json.load(f).get("weight_map", {})
weight_files = sorted(set(weight_map.values()))
missing_weight_files = [
    name for name in weight_files if not os.path.isfile(os.path.join(os.path.dirname(fp8_path), name))
]
if missing_weight_files:
    raise SystemExit(
        f"FP8 rollout checkpoint is missing {len(missing_weight_files)}/{len(weight_files)} indexed weight files; "
        f"first missing: {missing_weight_files[0]}"
    )
print(f"checkpoint roles verified: train=BF16 rollout={method} speculative=disabled")
PY
}

validate_remote_inputs() {
   local fp8_manifest_code
   fp8_manifest_code='import json, os, sys; root=sys.argv[1]; weight_map=json.load(open(os.path.join(root, "model.safetensors.index.json"), encoding="utf-8")).get("weight_map", {}); files=sorted(set(weight_map.values())); missing=[name for name in files if not os.path.isfile(os.path.join(root, name))]; assert not missing, f"FP8 rollout checkpoint is missing {len(missing)}/{len(files)} indexed weight files; first missing: {missing[0]}"; print(f"FP8 weight manifest verified: {len(files)} files")'
   if [[ "${DEBUG_ROLLOUT_TWO_NODE}" == "1" ]]; then
      # The only worker is a rollout host; no torch_dist actor input is needed.
      run_ssh "${REMOTE_HOSTS[0]}" "${REMOTE_PORTS[0]}" \
         "test -f $(shell_quote "${HF_MODEL_PATH}/config.json") && test -f $(shell_quote "${HF_MODEL_PATH}/model.safetensors.index.json") && test -f $(shell_quote "${RL_DATA}") && ${PYTHON_BIN} -c $(shell_quote "${fp8_manifest_code}") $(shell_quote "${HF_MODEL_PATH}")"
   elif [[ -n "${LOAD_DEBUG_ROLLOUT_DATA}" ]]; then
      # The only worker owns the second BF16 actor stage. Replay data is loaded
      # by the head; neither node needs rollout weights or KernelGym.
      run_ssh "${REMOTE_HOSTS[0]}" "${REMOTE_PORTS[0]}" \
         "test -f $(shell_quote "${HF_MODEL_PATH}/config.json") && test -f $(shell_quote "${MEGATRON_MODEL_PATH}/latest_checkpointed_iteration.txt")"
   else
      # node69 hosts the second BF16 actor; node53 and node64 host the FP8
      # rollout engines, two TP4 engines per host.
      run_ssh "${REMOTE_HOSTS[0]}" "${REMOTE_PORTS[0]}" \
         "test -f $(shell_quote "${HF_MODEL_PATH}/config.json") && test -f $(shell_quote "${MEGATRON_MODEL_PATH}/latest_checkpointed_iteration.txt") && test -f $(shell_quote "${RL_DATA}")"
      for ((i = 1; i < ${#REMOTE_HOSTS[@]}; i++)); do
         run_ssh "${REMOTE_HOSTS[$i]}" "${REMOTE_PORTS[$i]}" \
            "test -f $(shell_quote "${HF_MODEL_PATH}/config.json") && test -f $(shell_quote "${HF_MODEL_PATH}/model.safetensors.index.json") && test -f $(shell_quote "${RL_DATA}") && ${PYTHON_BIN} -c $(shell_quote "${fp8_manifest_code}") $(shell_quote "${HF_MODEL_PATH}")"
      done
   fi
}

prepare_node_local_resume_metadata() {
   if [[ "${RESUME_FROM_SAVE:-0}" != "1" ]]; then
      return
   fi
   if [[ "${CHECKPOINT_SAVE_PATH}" != /* || "${CHECKPOINT_SAVE_PATH}" == "/" ]]; then
      echo "CHECKPOINT_SAVE_PATH must be an absolute non-root path for resume." >&2
      exit 1
   fi

   # Checkpoint shards are intentionally node-local.  Megatron writes the
   # small global metadata only on rank 0, which need not be the Ray head where
   # argument validation runs.  Locate the newest finalized actor copy, then
   # replicate only its metadata; each actor node keeps and reads its own large
   # distcp shards.
   local marker_rel="latest_checkpointed_iteration.txt"
   local best_iteration=-1
   local source_host=""
   local source_port=""
   local candidate_iteration=""
   local i
   if [[ -f "${CHECKPOINT_SAVE_PATH}/${marker_rel}" ]]; then
      candidate_iteration="$(tr -d '[:space:]' < "${CHECKPOINT_SAVE_PATH}/${marker_rel}")"
      if [[ ! "${candidate_iteration}" =~ ^[0-9]+$ ]]; then
         echo "Invalid local resume marker: ${candidate_iteration@Q}" >&2
         exit 1
      fi
      best_iteration=${candidate_iteration}
      source_host="local"
   fi
   for i in "${!REMOTE_HOSTS[@]}"; do
      if [[ "${REMOTE_PLACEMENT_RESOURCES[$i]}" != "${ACTOR_PLACEMENT_RESOURCE}" ]]; then
         continue
      fi
      candidate_iteration="$(run_ssh "${REMOTE_HOSTS[$i]}" "${REMOTE_PORTS[$i]}" \
         "test -f $(shell_quote "${CHECKPOINT_SAVE_PATH}/${marker_rel}") && tr -d '[:space:]' < $(shell_quote "${CHECKPOINT_SAVE_PATH}/${marker_rel}") || true")"
      if [[ -n "${candidate_iteration}" && ! "${candidate_iteration}" =~ ^[0-9]+$ ]]; then
         echo "Invalid resume marker on ${REMOTE_HOSTS[$i]}: ${candidate_iteration@Q}" >&2
         exit 1
      fi
      if [[ -n "${candidate_iteration}" ]] && (( candidate_iteration > best_iteration )); then
         best_iteration=${candidate_iteration}
         source_host="${REMOTE_HOSTS[$i]}"
         source_port="${REMOTE_PORTS[$i]}"
      fi
   done
   if (( best_iteration < 0 )); then
      echo "No finalized checkpoint marker found for resume at ${CHECKPOINT_SAVE_PATH}." >&2
      exit 1
   fi

   local iteration_dir
   printf -v iteration_dir 'iter_%07d' "${best_iteration}"
   local metadata_files=(
      "latest_checkpointed_iteration.txt"
      "progress.txt"
      "${iteration_dir}/.metadata"
      "${iteration_dir}/common.pt"
      "${iteration_dir}/metadata.json"
   )
   local rel
   if [[ "${source_host}" != "local" ]]; then
      mkdir -p "${CHECKPOINT_SAVE_PATH}/${iteration_dir}"
      for rel in "${metadata_files[@]}"; do
         scp -P "${source_port}" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
            "${source_host}:${CHECKPOINT_SAVE_PATH}/${rel}" "${CHECKPOINT_SAVE_PATH}/${rel}"
      done
   fi
   for rel in "${metadata_files[@]}"; do
      if [[ ! -f "${CHECKPOINT_SAVE_PATH}/${rel}" ]]; then
         echo "Resume metadata is missing after synchronization: ${CHECKPOINT_SAVE_PATH}/${rel}" >&2
         exit 1
      fi
   done
   if ! compgen -G "${CHECKPOINT_SAVE_PATH}/${iteration_dir}/*.distcp" >/dev/null; then
      echo "Ray head has no local distcp shards for ${iteration_dir}." >&2
      exit 1
   fi

   for i in "${!REMOTE_HOSTS[@]}"; do
      if [[ "${REMOTE_PLACEMENT_RESOURCES[$i]}" != "${ACTOR_PLACEMENT_RESOURCE}" ]]; then
         continue
      fi
      run_ssh "${REMOTE_HOSTS[$i]}" "${REMOTE_PORTS[$i]}" \
         "mkdir -p $(shell_quote "${CHECKPOINT_SAVE_PATH}/${iteration_dir}")"
      for rel in "${metadata_files[@]}"; do
         scp -P "${REMOTE_PORTS[$i]}" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
            "${CHECKPOINT_SAVE_PATH}/${rel}" "${REMOTE_HOSTS[$i]}:${CHECKPOINT_SAVE_PATH}/${rel}"
      done
      run_ssh "${REMOTE_HOSTS[$i]}" "${REMOTE_PORTS[$i]}" \
         "compgen -G $(shell_quote "${CHECKPOINT_SAVE_PATH}/${iteration_dir}/*.distcp") >/dev/null"
   done
   echo "resume metadata synchronized for ${iteration_dir} across actor nodes"
}

if [[ "${CONFIG_DRY_RUN}" == "1" ]]; then
   printf 'TRAIN_DTYPE=bf16\nROLLOUT_CHECKPOINT=%s\nTRAIN_CHECKPOINT=%s\n' \
      "${HF_MODEL_PATH}" "${MEGATRON_MODEL_PATH}"
   printf 'MAX_CONTEXT_LEN=%s\nMAX_RESPONSE_LEN=%s\nOVERLONG_BUFFER_LEN=%s\nOVERLONG_PENALTY_FACTOR=%s\nOUTPUT_MISMATCH_PARTIAL_REWARD=%s\nREWARD_POLICY_LABEL=%s\n' \
      "${MAX_CONTEXT_LEN}" "${MAX_RESPONSE_LEN}" "${OVERLONG_BUFFER_LEN}" "${OVERLONG_PENALTY_FACTOR}" \
      "${OUTPUT_MISMATCH_PARTIAL_REWARD}" "${REWARD_POLICY_LABEL}"
   printf 'ROLLOUT_REASONING_EFFORT=%s\nCHAT_TEMPLATE_KWARGS=%s\n' \
      "${ROLLOUT_REASONING_EFFORT}" "${CHAT_TEMPLATE_KWARGS}"
   printf 'ROLLOUT_CORRECTION_MODE=%s\nROLLOUT_TEMPERATURE=%s\n' \
      "${ROLLOUT_CORRECTION_MODE}" "${ROLLOUT_TEMPERATURE}"
   printf 'RL_DATA=%s\nTRAIN_DATA_LABEL=%s\n' "${RL_DATA}" "${TRAIN_DATA_LABEL}"
   printf 'WANDB_GROUP=%s\n' "${WANDB_GROUP}"
   printf 'FULL_LOOP_SMOKE=%s\nLOAD_DEBUG_ROLLOUT_DATA=%s\nSAVE_FIRST_TRAIN_ROLLOUT=%s\nDEBUG_ROLLOUT_DATA=%s\nDISABLE_CHECKPOINT_SAVE=%s\nCHECKPOINT_SAVE_PATH=%s\nNUM_ROLLOUT=%s\nROLLOUT_BATCH_SIZE=%s\nN_SAMPLES_PER_PROMPT=%s\nGLOBAL_BATCH_SIZE=%s\n' \
      "${FULL_LOOP_SMOKE}" "${LOAD_DEBUG_ROLLOUT_DATA}" "${SAVE_FIRST_TRAIN_ROLLOUT}" "${DEBUG_ROLLOUT_DATA}" \
      "${DISABLE_CHECKPOINT_SAVE}" "${CHECKPOINT_SAVE_PATH}" \
      "${NUM_ROLLOUT}" "${ROLLOUT_BATCH_SIZE}" \
      "${N_SAMPLES_PER_PROMPT}" "${GLOBAL_BATCH_SIZE}"
   printf 'MIN_CHECKPOINT_FREE_GIB=%s\n' "${MIN_CHECKPOINT_FREE_GIB}"
   printf 'DEBUG_ROLLOUT_TWO_NODE=%s\nHEAD_RESOURCE_JSON=%s\nROLLOUT_RESOURCE_JSON=%s\n' \
      "${DEBUG_ROLLOUT_TWO_NODE}" "${HEAD_RESOURCE_JSON}" "${ROLLOUT_RESOURCE_JSON}"
   printf 'USE_NODE64_ROLLOUT=%s\n' "${USE_NODE64_ROLLOUT}"
   printf 'SGLANG_SPECULATIVE_MODE=none\nSGLANG_SERVING_PROFILE=%s\nSGLANG_MEM_FRACTION_STATIC=%s\nSGLANG_TARGET_ATTENTION_BACKEND=%s\nSGLANG_MAMBA_RADIX_CACHE_STRATEGY=%s\n' \
      "${SGLANG_SERVING_PROFILE}" \
      "${SGLANG_MEM_FRACTION_STATIC}" \
      "${SGLANG_TARGET_ATTENTION_BACKEND}" "${SGLANG_MAMBA_RADIX_CACHE_STRATEGY}"
   printf 'QWEN_GDN_BACKEND=%s\nQWEN_GDN_IMPLEMENTATION=%s\n' \
      "${QWEN_GDN_BACKEND}" "${QWEN_GDN_IMPLEMENTATION}"
   printf 'CONTEXT_PARALLEL_SIZE=%s\nCP_PARTITION_MODE=%s\nENABLE_SEQUENCE_PARALLEL=%s\n' \
      "${CONTEXT_PARALLEL_SIZE}" "${CP_PARTITION_MODE}" "${ENABLE_SEQUENCE_PARALLEL}"
   printf 'REUSE_RAY_CLUSTER=%s\n' "${REUSE_RAY_CLUSTER}"
   printf 'DISABLE_WANDB=%s\n' "${DISABLE_WANDB}"
   printf 'TRAIN_PYTORCH_PROFILE=%s\nTRAIN_TRITON_CACHE_DIR=%s\nSORT_TRAIN_MICROBATCHES_BY_PADDED_LENGTH_DESC=%s\n' \
      "${TRAIN_PYTORCH_PROFILE}" "${TRAIN_TRITON_CACHE_DIR}" \
      "${SORT_TRAIN_MICROBATCHES_BY_PADDED_LENGTH_DESC}"
   printf 'NCCL_IB_HCA=%s\n' "${NCCL_IB_HCA}"
   printf 'ACTOR_NUM_NODES=%s\nACTOR_GPUS=%s\nROLLOUT_GPUS=%s\nACTOR_PLACEMENT_RESOURCE=%s\nROLLOUT_PLACEMENT_RESOURCE=%s\n' \
      "${ACTOR_NUM_NODES}" "${ACTOR_GPUS}" "${ROLLOUT_GPUS}" "${ACTOR_PLACEMENT_RESOURCE}" "${ROLLOUT_PLACEMENT_RESOURCE}"
   print_runtime_gate_plan
   declare -p MODEL_ARGS CKPT_ARGS ROLLOUT_ARGS PERF_ARGS RL_ARGS OPTIMIZER_ARGS SGLANG_ARGS MISC_ARGS DEBUG_ARGS KERNEL_AGENT_ARGS CUSTOM_ARGS
   exit 0
fi

# Validate all immutable inputs before stopping any process or starting Ray.
prepare_node_local_resume_metadata
validate_local_inputs
validate_remote_inputs
check_all_qwen38_runtime
check_all_host_resources
check_all_tilelang_cuda_headers

if [[ "${PREPARE_ONLY}" == "1" ]]; then
   echo "Qwen3.8 no-spec RL prepare-only sanity PASS (no process cleanup, Ray start, or GPU job submit)."
   exit 0
fi

# A full model+distributed-optimizer iteration occupies about 155 GiB per
# actor node. Fail before process cleanup and GPU allocation instead of after
# an otherwise successful training run reaches async checkpoint finalization.
check_checkpoint_free_space

if [[ "${DEBUG_ROLLOUT_ONLY}" == "1" || "${FULL_LOOP_SMOKE}" == "1" \
   || "${SAVE_FIRST_TRAIN_ROLLOUT}" == "1" ]]; then
   mkdir -p "$(dirname "${DEBUG_ROLLOUT_DATA}")"
fi

# LOG CONFIG
LOG_STAMP="$(date +%Y%m%d.%H%M%S)"
LOG_DIR="${EXP_ROOT}/logs"
LOG_PATH="${LOG_DIR}/${LOG_STAMP}.log"
echo "Logging to ${LOG_PATH}"
mkdir -p "${LOG_DIR}"
exec >> "${LOG_PATH}" 2>&1

# Reusing a healthy cluster avoids disrupting unrelated head-node services.
# The default remains a clean dedicated-cluster launch for formal jobs.
if [[ "${REUSE_RAY_CLUSTER}" == "1" ]]; then
   echo "Reusing existing Ray cluster ${RAY_HEAD_ADDR}; no process cleanup or Ray restart."
   wait_for_cluster
else
# Clean only the dedicated containers' process namespaces before rerunning.
pkill -9 sglang || true
ray stop --force || true
pkill -9 -x raylet || true
pkill -9 -x gcs_server || true
pkill -9 -f "python3? (train|train_async)\\.py" || true
for i in "${!REMOTE_HOSTS[@]}"; do
   run_ssh "${REMOTE_HOSTS[$i]}" "${REMOTE_PORTS[$i]}" "pkill -9 sglang || true; ray stop --force || true; pkill -9 -x raylet || true; pkill -9 -x gcs_server || true; pkill -9 -f 'python3? (train|train_async)\\.py' || true"
done
sleep 3

# launch the master node of ray in container.
# `ray start --head` intermittently fails with "node timed out during startup /
# GCS overloaded" on these containers (GCS startup race; host is actually idle),
# so retry with a clean slate between attempts. Override count via RAY_HEAD_START_ATTEMPTS.
export MASTER_ADDR
ray_head_attempts="${RAY_HEAD_START_ATTEMPTS:-4}"
for attempt in $(seq 1 "${ray_head_attempts}"); do
   if GLOO_SOCKET_IFNAME="${LOCAL_GLOO_SOCKET_IFNAME}" ray start \
      --head \
      --node-ip-address ${MASTER_ADDR} \
      --port ${RAY_PORT} \
      --dashboard-host 0.0.0.0 \
      --dashboard-port $RAY_DASHBOARD_PORT \
      --resources "${HEAD_RESOURCE_JSON}" \
      --node-manager-port ${RAY_NODE_MANAGER_PORT} \
      --object-manager-port ${RAY_OBJECT_MANAGER_PORT} \
      --dashboard-agent-listen-port ${RAY_DASHBOARD_AGENT_LISTEN_PORT} \
      --dashboard-agent-grpc-port ${RAY_DASHBOARD_AGENT_GRPC_PORT} \
      --metrics-export-port ${RAY_METRICS_EXPORT_PORT} \
      --ray-client-server-port ${RAY_CLIENT_SERVER_PORT} \
      --min-worker-port ${RAY_MIN_WORKER_PORT} \
      --max-worker-port ${RAY_MAX_WORKER_PORT} \
      --num-gpus 8 \
      --disable-usage-stats \
      --temp-dir=$RAY_TEMP_DIR; then
      echo "ray head started on attempt ${attempt}/${ray_head_attempts}"
      break
   fi
   echo "ray head start failed (attempt ${attempt}/${ray_head_attempts}); cleaning up and retrying"
   ray stop --force || true
   pkill -9 -x gcs_server 2>/dev/null || true
   pkill -9 -x raylet 2>/dev/null || true
   rm -rf "${RAY_TEMP_DIR}"
   if [[ "${attempt}" -eq "${ray_head_attempts}" ]]; then
      echo "ray head failed to start after ${ray_head_attempts} attempts" >&2
      exit 1
   fi
   sleep 5
done

worker_attempts="${RAY_WORKER_START_ATTEMPTS:-4}"
for i in "${!REMOTE_HOSTS[@]}"; do
   echo "Starting Ray worker on ${REMOTE_HOSTS[$i]}"
   if [[ "${REMOTE_PLACEMENT_RESOURCES[$i]}" == "${ACTOR_PLACEMENT_RESOURCE}" ]]; then
      remote_resource_json="${ACTOR_RESOURCE_JSON}"
   else
      remote_resource_json="${ROLLOUT_RESOURCE_JSON}"
   fi
   # Worker `ray start --address` hits the same intermittent "node timed out
   # during startup / GCS overloaded" race as the head, so retry with cleanup.
   for wattempt in $(seq 1 "${worker_attempts}"); do
      if run_ssh "${REMOTE_HOSTS[$i]}" "${REMOTE_PORTS[$i]}" \
         "GLOO_SOCKET_IFNAME=${REMOTE_GLOO_SOCKET_IFNAMES[$i]} ray start --address ${RAY_HEAD_ADDR} --resources $(shell_quote "${remote_resource_json}") --node-manager-port ${RAY_NODE_MANAGER_PORT} --object-manager-port ${RAY_OBJECT_MANAGER_PORT} --dashboard-agent-listen-port ${RAY_DASHBOARD_AGENT_LISTEN_PORT} --dashboard-agent-grpc-port ${RAY_DASHBOARD_AGENT_GRPC_PORT} --metrics-export-port ${RAY_METRICS_EXPORT_PORT} --min-worker-port ${RAY_MIN_WORKER_PORT} --max-worker-port ${RAY_MAX_WORKER_PORT} --num-gpus 8 --disable-usage-stats"; then
         echo "ray worker ${REMOTE_HOSTS[$i]} joined on attempt ${wattempt}/${worker_attempts}"
         break
      fi
      echo "ray worker ${REMOTE_HOSTS[$i]} join failed (attempt ${wattempt}/${worker_attempts}); cleaning up and retrying"
      run_ssh "${REMOTE_HOSTS[$i]}" "${REMOTE_PORTS[$i]}" \
         "ray stop --force >/dev/null 2>&1 || true; pkill -9 -x raylet 2>/dev/null || true; rm -rf /tmp/ray" || true
      if [[ "${wattempt}" -eq "${worker_attempts}" ]]; then
         echo "ray worker ${REMOTE_HOSTS[$i]} failed to join after ${worker_attempts} attempts" >&2
         exit 1
      fi
      sleep 5
   done
done

wait_for_cluster
fi

# Cover every node IP in BOTH no_proxy spellings: reqwest (sglang router) and
# other HTTP clients must never reach in-cluster engines through the egress
# proxy — it kills connections that stay silent for ~60s, aborting long
# non-streaming /generate requests.
NO_PROXY_LIST="localhost,127.0.0.1,0.0.0.0,::1,${MASTER_ADDR}$(printf ',%s' "${REMOTE_HOSTS[@]}")"

# CUDA graphs remain enabled with SGLang's default padded buckets.
RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    "no_proxy": "${NO_PROXY_LIST}",
    "NO_PROXY": "${NO_PROXY_LIST}",
    "NCCL_SOCKET_IFNAME": "${NCCL_SOCKET_IFNAME}",
    "NCCL_IB_HCA": "${NCCL_IB_HCA}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "WANDB_API_KEY": "${WANDB_API_KEY}",
    "CUDA_HOME": "${CUDA_HOME}",
    "CUDA_PATH": "${CUDA_PATH}",
    "PATH": "${RUNTIME_PATH}",
    "LD_LIBRARY_PATH": "${RUNTIME_LD_LIBRARY_PATH}",
    "PYTHONPATH": ".:/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "CUDA_AGENT_LOG_MULTI_TURN_TEXT": "0",
    "CUDA_AGENT_LOG_ROLLOUT_STATS_ONLY": "1",
    "CUDA_AGENT_LOG_SLOWEST_INFO": "0",
    "CUDA_AGENT_OUTPUT_MISMATCH_PARTIAL_REWARD": "${OUTPUT_MISMATCH_PARTIAL_REWARD}",
    "CUDA_AGENT_PERFORMANCE_REWARD_REQUIRES_CORRECTNESS": "1",
    "SGLANG_RETURN_ORIGINAL_LOGPROB": "0",
    "SLIME_LOG_TRAIN_MICROBATCH_SHAPES": "1",
    "SLIME_SAVE_DEBUG_ROLLOUT_MAX_ID": "${SAVE_DEBUG_ROLLOUT_MAX_ID}",
    "TORCHDYNAMO_DISABLE": "${TRAIN_PYTORCH_PROFILE}",
    "TRITON_CACHE_DIR": "${TRAIN_TRITON_CACHE_DIR}",
    "KERNEL_AGENT_GENERATE_GUARD_SEC": "${KERNEL_AGENT_GENERATE_GUARD_SEC}",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "NCCL_DEBUG": "WARN"
  }
}
EOF_JSON
)
RAY_JOB_TIMEOUT_PREFIX=()
RAY_JOB_ID_ARGS=()
SMOKE_SUBMISSION_ID=""
if [[ "${FULL_LOOP_SMOKE}" == "1" ]]; then
   SMOKE_SUBMISSION_ID="qwen38-nospec-full-loop-smoke-${LOG_STAMP}"
   RAY_JOB_TIMEOUT_PREFIX=(timeout --signal=TERM --kill-after=30s "${FULL_LOOP_SMOKE_TIMEOUT_SEC}s")
   RAY_JOB_ID_ARGS=(--submission-id="${SMOKE_SUBMISSION_ID}")
   echo "Full-loop smoke hard timeout: ${FULL_LOOP_SMOKE_TIMEOUT_SEC}s; submission_id=${SMOKE_SUBMISSION_ID}"
fi

set +e
"${RAY_JOB_TIMEOUT_PREFIX[@]}" ray job submit --address="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}" \
   "${RAY_JOB_ID_ARGS[@]}" \
   --working-dir="${REPO_ROOT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes ${ACTOR_NUM_NODES} \
   --actor-num-gpus-per-node ${ACTOR_GPUS_PER_NODE} \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   --num-gpus-per-node 8 \
   --actor-placement-resource "${ACTOR_PLACEMENT_RESOURCE}" \
   --rollout-placement-resource "${ROLLOUT_PLACEMENT_RESOURCE}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${CURRICULUM_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${RL_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${LOGGING_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${DEBUG_ARGS[@]}" \
   "${KERNEL_AGENT_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}"
job_status=$?
set -e
if [[ "${FULL_LOOP_SMOKE}" == "1" && ("${job_status}" == "124" || "${job_status}" == "137") ]]; then
   echo "Full-loop smoke exceeded its hard timeout; stopping Ray job ${SMOKE_SUBMISSION_ID}." >&2
   timeout 60s ray job stop --address="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}" "${SMOKE_SUBMISSION_ID}" || true
fi
exit "${job_status}"
