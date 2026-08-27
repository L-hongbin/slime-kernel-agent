#!/bin/bash

set -Eex
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
# Use `hostname -I` (not `ip`, which is absent in some node containers, e.g. node62)
# to enumerate local IPv4s and confirm we are on the Ray head node.
if ! hostname -I 2>/dev/null | tr ' ' '\n' | grep -Fxq "${MASTER_ADDR}"; then
   echo "This script must run on the Ray head node (${MASTER_ADDR}); local node IPs are:"
   hostname -I 2>/dev/null
   exit 1
fi
# head node = 10.11.2.170 (quiet host; 162's container has a flaky ray-head GCS
# startup so it is demoted to a worker). Cluster = 170(head) + 169 + 162.
REMOTE_HOSTS=(
   "10.11.2.169"
   "10.11.2.162"
)
REMOTE_PORTS=(
   "23522"
   "23522"
)
if [ "${#REMOTE_PORTS[@]}" -ne "${#REMOTE_HOSTS[@]}" ]; then
   echo "REMOTE_PORTS length (${#REMOTE_PORTS[@]}) must match REMOTE_HOSTS length (${#REMOTE_HOSTS[@]})."
   exit 1
fi
NUM_NODES=$((1 + ${#REMOTE_HOSTS[@]}))
NUM_GPUS=$((NUM_NODES * 8))
ACTOR_NUM_NODES=2
ACTOR_GPUS=$((ACTOR_NUM_NODES*8))
ROLLOUT_GPUS=$((NUM_GPUS-ACTOR_GPUS))
echo "ACTOR_GPUS ${ACTOR_GPUS} ROLLOUT_GPUS ${ROLLOUT_GPUS}"
# EXP CONFIG
MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-16384}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-${MAX_CONTEXT_LEN}}
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-32}
DEBUG_ROLLOUT_ONLY=${DEBUG_ROLLOUT_ONLY:-0}
if [[ "${DEBUG_ROLLOUT_ONLY}" == "1" ]]; then
   NUM_ROLLOUT=${NUM_ROLLOUT:-1}
else
   NUM_ROLLOUT=${NUM_ROLLOUT:-3000}
fi
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
MODEL_NAME="Qwen3.6-27B"
KERNEL_BACKEND="tvm_ffi"
KERNEL_ENV_URL="http://127.0.0.1:20211"


HF_MODEL_PATH="/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B"
MEGATRON_MODEL_PATH="/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B/torch_dist_tp4_pp2"
RL_DATA="/ms/FM/lihongbin/dataset/CUDA_RL/cuda_rl/prompt_tvm_v2/drkernel_rl_thinking.parquet"


EXP_NAME="FAsync.${KERNEL_BACKEND}.${MODEL_NAME}.CTX${MAX_CONTEXT_LEN}"
EXP_ROOT="${REPO_ROOT}/experiments/${EXP_NAME}"

WANDB_KEY_FILE=${WANDB_KEY_FILE:-${HOME}/.config/wandb/slime.key}
if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_KEY_FILE}" ]]; then
   WANDB_API_KEY="$(tr -d '[:space:]' < "${WANDB_KEY_FILE}")"
fi
export WANDB_API_KEY
WANDB_GROUP=${WANDB_GROUP:-${EXP_NAME}}

NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
LOCAL_GLOO_SOCKET_IFNAME="${LOCAL_GLOO_SOCKET_IFNAME:-bond0}"
REMOTE_GLOO_SOCKET_IFNAMES=(
   # 10.11.2.169: enp76s0f0np0 is a bond slave of bond0 (no own IP); the IP lives
   # on bond0, so gloo must bind bond0 (matches the other nodes).
   "bond0"
   "bond0"
)
if [ "${#REMOTE_GLOO_SOCKET_IFNAMES[@]}" -ne "${#REMOTE_HOSTS[@]}" ]; then
   echo "REMOTE_GLOO_SOCKET_IFNAMES length (${#REMOTE_GLOO_SOCKET_IFNAMES[@]}) must match REMOTE_HOSTS length (${#REMOTE_HOSTS[@]})."
   exit 1
fi

RAY_DASHBOARD_PORT=8265
RAY_PORT=6379
RAY_HEAD_ADDR="${MASTER_ADDR}:${RAY_PORT}"
RAY_TEMP_DIR="/tmp/ray"

PYTHON_BIN=${PYTHON_BIN:-python3}
RAY_WAIT_TIMEOUT=${RAY_WAIT_TIMEOUT:-300}
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
CUDA_BIN_DIR="${CUDA_HOME}/bin"
CUDA_LIB_DIR="${CUDA_HOME}/lib64"
RUNTIME_PATH="${CUDA_BIN_DIR}:${PATH}"
RUNTIME_LD_LIBRARY_PATH="${CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}"

# LOG CONFIG
LOG_STAMP="$(date +%Y%m%d.%H%M%S)"
LOG_DIR="${EXP_ROOT}/logs"
LOG_PATH="${LOG_DIR}/${LOG_STAMP}.log"
echo "Logging to ${LOG_PATH}"

mkdir -p "${LOG_DIR}"
exec >> "${LOG_PATH}" 2>&1

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

wait_for_cluster() {
   echo "Waiting for Ray cluster: expected nodes=${NUM_NODES}, expected GPUs=${NUM_GPUS}"
   local deadline=$((SECONDS + RAY_WAIT_TIMEOUT))
   local ready=0

   while ((SECONDS < deadline)); do
      if "${PYTHON_BIN}" - "${RAY_HEAD_ADDR}" "${NUM_NODES}" "${NUM_GPUS}" <<'PY'
import sys

import ray

ray_address = sys.argv[1]
expected_nodes = int(sys.argv[2])
expected_gpus = float(sys.argv[3])

ray.init(address=ray_address, ignore_reinit_error=True, logging_level="ERROR")
alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
gpu_count = sum(float(node.get("Resources", {}).get("GPU", 0)) for node in alive_nodes)

print(f"Ray alive nodes={len(alive_nodes)}, GPUs={gpu_count:g}")
sys.exit(0 if len(alive_nodes) >= expected_nodes and gpu_count >= expected_gpus else 1)
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

# for rerun the task
pkill -9 sglang || true
ray stop --force || true
pkill -9 -x raylet || true
pkill -9 -x gcs_server || true
pkill -9 -f "python3? (train|train_async)\\.py" || true
for i in "${!REMOTE_HOSTS[@]}"; do
   run_ssh "${REMOTE_HOSTS[$i]}" "${REMOTE_PORTS[$i]}" "pkill -9 sglang || true; ray stop --force || true; pkill -9 -x raylet || true; pkill -9 -x gcs_server || true; pkill -9 -f 'python3? (train|train_async)\\.py' || true"
done
sleep 3

check_all_host_resources
check_all_tilelang_cuda_headers

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime
   --wandb-group ${WANDB_GROUP}
   --disable-wandb-random-suffix
   --wandb-always-use-train-step
   --wandb-centralized
)

LOGGING_ARGS=(
   --log-throughput
   --log-progress
   --log-device-memory-used
)

CKPT_ARGS=(
   --hf-checkpoint ${HF_MODEL_PATH}
   --ref-load ${MEGATRON_MODEL_PATH}
   # /nfs/FM is per-node local disk: ranks write shards to their own node;
   # gather with scripts/sync/gather_convert_ckpt.sh afterwards.
   --save ${EXP_ROOT}/checkpoints
   # Resume from the latest finalized checkpoint in --save (iter_119); slime
   # derives start_rollout_id from it when --start-rollout-id is unset. Loading
   # the same dir we save to makes restarts continue instead of restart at 0.
   --load ${EXP_ROOT}/checkpoints
   --save-interval 20
   # async save overlaps disk writes with the next train step; the worker
   # flag is required or Megatron disables --async-save. Keep the default
   # dp_reshardable optimizer format (do NOT add fully-reshardable).
   --async-save
   --use-persistent-ckpt-worker
)

ROLLOUT_ARGS=(
   --rollout-function-path examples.kernel_agent.fully_async_rollout.generate_rollout_fully_async
   --update-weights-interval 1
   --keep-old-actor
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
   --apply-chat-template-kwargs '{"enable_thinking":true}'
   --rollout-temperature 1

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
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --decoder-last-pipeline-num-layers 31
   --context-parallel-size 2
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --qwen-gdn-backend flashqla

   --recompute-granularity full
   --recompute-method block
   --recompute-num-layers 29

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --calculate-per-token-loss
   --max-tokens-per-gpu 8192
   --log-probs-max-tokens-per-gpu 16384
   # --init-model-with-meta-device
   --mtp-num-layers 1
   --enable-mtp-training
   --mtp-loss-scaling-factor 0.2
)

RL_ARGS=(
   --advantage-estimator trloo
   --multi-turn-gamma 1.0
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   # whether enabling TIS
   # --use-tis
)

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
   --sglang-mem-fraction-static 0.7
   --sglang-decode-log-interval 400
   --router-policy round_robin
   --sglang-cuda-graph-max-bs ${SGLANG_MAX_RUNNING_REQUESTS}
   --sglang-disable-custom-all-reduce
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 4
   # triton (NOT flashinfer): the flashinfer GDN decode kernel diverges ~2e-3/layer
   # from the megatron chunkwise recompute on real inputs, compounding across the 48
   # GDN layers into a per-token logprob mismatch that collapses sequence_mis
   # (reject 96.5% -> 0% after switching to triton). Root cause = sglang#20791
   # (flashinfer GDN in-place state-pool aliasing); triton is the correct reference.
   --sglang-linear-attn-backend triton
   --sglang-mamba-scheduler-strategy extra_buffer
)

MISC_ARGS=(
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

CUSTOM_ARGS=(
   --custom-generate-function-path examples.kernel_agent.generate_with_cuda_agent.generate
   --custom-rm-path examples.kernel_agent.generate_with_cuda_agent.reward_func
   --custom-reward-post-process-path examples.kernel_agent.kernel_reward.reward_post_process_by_group
   --dynamic-sampling-filter-path examples.kernel_agent.kernel_filter.filter_cuda_kernel_group
   --multi-turn-prompt-config-path "${SCRIPT_DIR}/prompt_config/response_prompt/cuda_kernel.yaml"
   --rollout-data-postprocess-path examples.kernel_agent.kernel_filter.sequence_mis

   # TIS-related args, recommended to enable when using TIS
   # --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   # --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

DEBUG_ARGS=()
if [[ "${DEBUG_ROLLOUT_ONLY}" == "1" ]]; then
   DEBUG_ARGS+=(--debug-rollout-only)
fi

KERNEL_AGENT_ARGS=(
   --kernel-env-url ${KERNEL_ENV_URL}
   --kernel-backend $KERNEL_BACKEND
   --reference-backend torch
   --do-precheck
   --use-reference-cache
   --finalize-mode positive
   --use-multi-turn
   --filter-by-last-turn
   --padding-turns
   --max-turns 1
   --sequence-mis-config '{"aggregation":"turns_geometric","token_veto_threshold":1e-4,"lower":0.999,"upper":1.001,"use_advantage":false}'
   --enable-turns-dp-partitions
   --use-coverage-rs
   --coverage-rs-key time_coverage
   --coverage-rs-threshold 0.3
   --coverage-rs-factor 0.1
)

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
   # Worker `ray start --address` hits the same intermittent "node timed out
   # during startup / GCS overloaded" race as the head, so retry with cleanup.
   for wattempt in $(seq 1 "${worker_attempts}"); do
      if run_ssh "${REMOTE_HOSTS[$i]}" "${REMOTE_PORTS[$i]}" \
         "GLOO_SOCKET_IFNAME=${REMOTE_GLOO_SOCKET_IFNAMES[$i]} ray start --address ${RAY_HEAD_ADDR} --num-gpus 8 --disable-usage-stats"; then
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

# Cover every node IP in BOTH no_proxy spellings: reqwest (sglang router) and
# other HTTP clients must never reach in-cluster engines through the egress
# proxy — it kills connections that stay silent for ~60s, aborting long
# non-streaming /generate requests.
NO_PROXY_LIST="localhost,127.0.0.1,0.0.0.0,::1,${MASTER_ADDR}$(printf ',%s' "${REMOTE_HOSTS[@]}")"

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    "no_proxy": "${NO_PROXY_LIST}",
    "NO_PROXY": "${NO_PROXY_LIST}",
    "NCCL_SOCKET_IFNAME": "${NCCL_SOCKET_IFNAME}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "WANDB_API_KEY": "${WANDB_API_KEY}",
    "CUDA_HOME": "${CUDA_HOME}",
    "CUDA_PATH": "${CUDA_PATH}",
    "PATH": "${RUNTIME_PATH}",
    "LD_LIBRARY_PATH": "${RUNTIME_LD_LIBRARY_PATH}",
    "PYTHONPATH": ".:/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "CUDA_AGENT_LOG_MULTI_TURN_TEXT": "0",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "NCCL_DEBUG": "WARN"
  }
}
EOF_JSON
)
ray job submit --address="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}" \
   --working-dir="${REPO_ROOT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes ${ACTOR_NUM_NODES} \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
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
