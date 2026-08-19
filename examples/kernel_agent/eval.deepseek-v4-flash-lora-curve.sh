#!/usr/bin/env bash
# Serial KernelBench-L1 accuracy curve for the r21 DeepSeek-V4-Flash rsLoRA run.
# The base control keeps SGLang's LoRA wrappers enabled but selects no adapter;
# step20 onward preload and explicitly route to one immutable PEFT adapter each.

set -Eeo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval.deepseek-v4-flash.sh"

ADAPTER_ROOT="${ADAPTER_ROOT:-/nfs/FM/csl_v4r21_fp4_pp1cp2_12k_dppo_predictive_resume40_20260722/eval_adapters}"
EXP_ROOT="${EXP_ROOT:-${REPO_ROOT}/experiments/Eval.KernelBenchL1.DeepSeekV4FlashLoRA.12k.turn1.n8}"
RAY_WORKER_HOST="${RAY_WORKER_HOST-node53_dspark}"
RAY_WORKER_IP="${RAY_WORKER_IP-10.11.2.153}"
MASTER_ADDR="${MASTER_ADDR:-10.11.2.169}"
LOCAL_GLOO_SOCKET_IFNAME="${LOCAL_GLOO_SOCKET_IFNAME:-bond0}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
RAY_HEAD_GPUS="${RAY_HEAD_GPUS:-$([[ -n "${RAY_WORKER_HOST}" ]] && echo 0 || echo 8)}"
RAY_DASHBOARD_AGENT_LISTEN_PORT="${RAY_DASHBOARD_AGENT_LISTEN_PORT:-52465}"
RAY_DASHBOARD_AGENT_GRPC_PORT="${RAY_DASHBOARD_AGENT_GRPC_PORT:-52466}"
RAY_RUNTIME_ENV_AGENT_PORT="${RAY_RUNTIME_ENV_AGENT_PORT:-52467}"
KERNEL_ENV_URL="${KERNEL_ENV_URL:-http://127.0.0.1:20211}"
KERNEL_EVAL_WORKER_MAX_CONCURRENCY="${KERNEL_EVAL_WORKER_MAX_CONCURRENCY:-4}"
KERNEL_EVAL_RATE_LIMIT="${KERNEL_EVAL_RATE_LIMIT:-4}"
KERNEL_EVAL_PRIORITY="${KERNEL_EVAL_PRIORITY:-low}"
FORCE_RERUN="${FORCE_RERUN:-0}"
EVAL_ONLY_STEP="${EVAL_ONLY_STEP:-}"

mkdir -p "${EXP_ROOT}"
ORCHESTRATOR_LOG="${EXP_ROOT}/curve.$(date +%Y%m%d.%H%M%S).log"
exec > >(tee -a "${ORCHESTRATOR_LOG}") 2>&1

cleanup() {
   local status=$?
   trap - EXIT
   ray stop --force >/dev/null 2>&1 || true
   if [[ -n "${RAY_WORKER_HOST}" ]]; then
      ssh "${RAY_WORKER_HOST}" "ray stop --force >/dev/null 2>&1 || true"
   fi
   echo "curve_exit_status=${status}"
   exit "${status}"
}
trap cleanup EXIT

run_checkpoint() {
   local step=$1
   local adapter_path=$2
   local tag="step${step}"
   local eval_dir="${EXP_ROOT}/${tag}"

   if [[ -n "${EVAL_ONLY_STEP}" && "${step}" != "${EVAL_ONLY_STEP}" ]]; then
      return 0
   fi

   if [[ "${FORCE_RERUN}" != "1" ]]; then
      local summary_path
      for summary_path in "${eval_dir}"/summary.*.txt; do
         [[ -f "${summary_path}" ]] || continue
         if grep -q '^samples: 800  (missing env_result: 0)$' "${summary_path}"; then
            echo "=== ${tag}: complete 800-sample summary found; skip (FORCE_RERUN=1 to replace) ==="
            return 0
         fi
      done
   fi

   echo "=== ${tag}: start $(date --iso-8601=seconds) adapter=${adapter_path:-<base>} ==="
   env \
      V4_RUNTIME=dspark \
      MASTER_ADDR="${MASTER_ADDR}" \
      LOCAL_GLOO_SOCKET_IFNAME="${LOCAL_GLOO_SOCKET_IFNAME}" \
      NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
      RAY_WORKER_HOST="${RAY_WORKER_HOST}" \
      RAY_WORKER_IP="${RAY_WORKER_IP}" \
      RAY_HEAD_GPUS="${RAY_HEAD_GPUS}" \
      RAY_DASHBOARD_AGENT_LISTEN_PORT="${RAY_DASHBOARD_AGENT_LISTEN_PORT}" \
      RAY_DASHBOARD_AGENT_GRPC_PORT="${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
      RAY_RUNTIME_ENV_AGENT_PORT="${RAY_RUNTIME_ENV_AGENT_PORT}" \
      KERNEL_ENV_URL="${KERNEL_ENV_URL}" \
      MAX_CONTEXT_LEN=12288 \
      MAX_RESPONSE_LEN=12288 \
      MAX_TURNS=1 \
      EVAL_NUM_PROMPTS=0 \
      N_SAMPLES_PER_EVAL_PROMPT=8 \
      ROLLOUT_SEED=42 \
      ROLLOUT_TEMPERATURE=1 \
      ROLLOUT_TOP_P=1 \
      KERNEL_EVAL_WORKER_MAX_CONCURRENCY="${KERNEL_EVAL_WORKER_MAX_CONCURRENCY}" \
      KERNEL_EVAL_RATE_LIMIT="${KERNEL_EVAL_RATE_LIMIT}" \
      KERNEL_EVAL_PRIORITY="${KERNEL_EVAL_PRIORITY}" \
      SGLANG_MAX_RUNNING_REQUESTS=128 \
      RAY_PORT=6386 \
      RAY_DASHBOARD_PORT=8269 \
      RAY_TEMP_DIR="/dev/shm/ray_eval_dsv4_${tag}" \
      EXP_ROOT="${EXP_ROOT}" \
      EVAL_TAG="${tag}" \
      ENABLE_LORA_SERVER=1 \
      LORA_ADAPTER_PATH="${adapter_path}" \
      LORA_NAME="eval_${tag}" \
      bash "${EVAL_SCRIPT}"
   echo "=== ${tag}: complete $(date --iso-8601=seconds) ==="
}

echo "curve_log=${ORCHESTRATOR_LOG}"
echo "exp_root=${EXP_ROOT}"
echo "adapter_root=${ADAPTER_ROOT}"
echo "contract=KernelBench-L1 prompts=100 samples_per_prompt=8 context=12288 response=12288 turns=1 seed=42 temperature=1 top_p=1"
echo "serve=base_plus_named_peft_adapter no_merge=1 runtime=dspark_mxfp4_w4a16 dp_attention=8 moe_tp=8 cuda_graph=on"

# Single-point runs may target checkpoints created after this file's original
# fixed curve.  Dispatch the requested point directly so a valid future step
# cannot silently exit without evaluating anything.
if [[ -n "${EVAL_ONLY_STEP}" ]]; then
   if [[ ! "${EVAL_ONLY_STEP}" =~ ^[0-9]+$ ]] || ((10#${EVAL_ONLY_STEP} % 20 != 0)); then
      echo "EVAL_ONLY_STEP must be a non-negative multiple of 20: ${EVAL_ONLY_STEP}" >&2
      exit 2
   fi
   if [[ "${EVAL_ONLY_STEP}" == "0" ]]; then
      run_checkpoint 0 ""
   else
      run_checkpoint "${EVAL_ONLY_STEP}" "${ADAPTER_ROOT}/step${EVAL_ONLY_STEP}"
   fi
   echo "=== curve single-point complete $(date --iso-8601=seconds) ==="
   exit 0
fi

run_checkpoint 0 ""
run_checkpoint 20 "${ADAPTER_ROOT}/step20"
run_checkpoint 40 "${ADAPTER_ROOT}/step40"
run_checkpoint 60 "${ADAPTER_ROOT}/step60"
run_checkpoint 80 "${ADAPTER_ROOT}/step80"
run_checkpoint 100 "${ADAPTER_ROOT}/step100"
run_checkpoint 120 "${ADAPTER_ROOT}/step120"
run_checkpoint 140 "${ADAPTER_ROOT}/step140"
run_checkpoint 160 "${ADAPTER_ROOT}/step160"
run_checkpoint 180 "${ADAPTER_ROOT}/step180"
run_checkpoint 200 "${ADAPTER_ROOT}/step200"
run_checkpoint 220 "${ADAPTER_ROOT}/step220"
run_checkpoint 240 "${ADAPTER_ROOT}/step240"
run_checkpoint 260 "${ADAPTER_ROOT}/step260"
run_checkpoint 280 "${ADAPTER_ROOT}/step280"
run_checkpoint 300 "${ADAPTER_ROOT}/step300"
run_checkpoint 320 "${ADAPTER_ROOT}/step320"
run_checkpoint 340 "${ADAPTER_ROOT}/step340"
run_checkpoint 360 "${ADAPTER_ROOT}/step360"
run_checkpoint 380 "${ADAPTER_ROOT}/step380"
run_checkpoint 400 "${ADAPTER_ROOT}/step400"
run_checkpoint 420 "${ADAPTER_ROOT}/step420"
run_checkpoint 440 "${ADAPTER_ROOT}/step440"
run_checkpoint 460 "${ADAPTER_ROOT}/step460"
run_checkpoint 480 "${ADAPTER_ROOT}/step480"
run_checkpoint 500 "${ADAPTER_ROOT}/step500"
run_checkpoint 520 "${ADAPTER_ROOT}/step520"
run_checkpoint 540 "${ADAPTER_ROOT}/step540"
run_checkpoint 560 "${ADAPTER_ROOT}/step560"

echo "=== curve complete $(date --iso-8601=seconds) ==="
