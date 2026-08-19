#!/usr/bin/env bash
# Run one DS-V4 evaluation point in a disposable 8-GPU container.
#
# Usage:
#   run_eval_step.sh h20 curve STEP
#   run_eval_step.sh h200 curve STEP
#   run_eval_step.sh h20 level LEVEL STEP

set -Eeuo pipefail

usage() {
  cat >&2 <<EOF
usage:
  $0 h20 curve STEP
  $0 h200 curve STEP
  $0 h20 level LEVEL STEP

STEP must be a non-negative multiple of 20; LEVEL must be 1, 2, or 3.
EOF
  exit 2
}

if [[ $# -lt 2 ]]; then
  usage
fi

PROFILE=$1
MODE=$2
LEVEL=""
case "${PROFILE}:${MODE}" in
  h20:curve | h200:curve)
    [[ $# -eq 3 ]] || usage
    STEP_INPUT=$3
    ;;
  h20:level)
    [[ $# -eq 4 ]] || usage
    LEVEL=$3
    STEP_INPUT=$4
    [[ "${LEVEL}" =~ ^[123]$ ]] || usage
    LEVEL=$((10#${LEVEL}))
    ;;
  *)
    usage
    ;;
esac

if [[ ! "${STEP_INPUT}" =~ ^[0-9]+$ ]] || ((10#${STEP_INPUT} % 20 != 0)); then
  usage
fi
STEP=$((10#${STEP_INPUT}))

STAMP=$(date +%Y%m%d.%H%M%S)
KERNEL_EVAL_WORKER_MAX_CONCURRENCY=${KERNEL_EVAL_WORKER_MAX_CONCURRENCY:-32}
KERNEL_EVAL_RATE_LIMIT=${KERNEL_EVAL_RATE_LIMIT:-32}
KERNEL_EVAL_PRIORITY=${KERNEL_EVAL_PRIORITY:-low}
KERNEL_ENV_URL=${KERNEL_ENV_URL:-http://127.0.0.1:20211}
if [[ ! "${KERNEL_EVAL_WORKER_MAX_CONCURRENCY}" =~ ^[1-9][0-9]*$ ||
      ! "${KERNEL_EVAL_RATE_LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "KernelGym concurrency and rate limit must be positive integers" >&2
  exit 2
fi

case "${PROFILE}" in
  h20)
    RUN_NAME=${H20_EVAL_RUN_NAME:-csl_v4r21_fp4_pp1cp2_12k_dppo_predictive_resume40_20260722}
    DISK_ROOT=${H20_EVAL_DISK_ROOT:-/mnt/md1}
    CONTAINER_DISK_ROOT=/nfs/FM
    HOST_REPO="${DISK_ROOT}/chenshuailin/projects/kernel_agents/slime-v4flash-lora-eval-20260729"
    CONTAINER_REPO="${CONTAINER_DISK_ROOT}/chenshuailin/projects/kernel_agents/slime-v4flash-lora-eval-20260729"
    HOST_RUN_ROOT="${DISK_ROOT}/${RUN_NAME}"
    CONTAINER_RUN_ROOT="${CONTAINER_DISK_ROOT}/${RUN_NAME}"
    HOST_EVAL_ROOT="${HOST_RUN_ROOT}/h20_eval_20260729"
    CONTAINER_EVAL_ROOT="${CONTAINER_RUN_ROOT}/h20_eval_20260729"
    HOST_ADAPTER_ROOT="${HOST_RUN_ROOT}/eval_adapters"
    CONTAINER_ADAPTER_ROOT="${CONTAINER_RUN_ROOT}/eval_adapters"
    HOST_MODEL="${DISK_ROOT}/chenshuailin/checkpoints/deepseek-ai/DeepSeek-V4-Flash-DSpark"
    CONTAINER_MODEL="${CONTAINER_DISK_ROOT}/chenshuailin/checkpoints/deepseek-ai/DeepSeek-V4-Flash-DSpark"
    HOST_MEGATRON="${DISK_ROOT}/chenshuailin/projects/kernel_agents/Megatron-LM-eval-20260729"
    IMAGE=${H20_EVAL_IMAGE:-csl/sglang:dspark-r1-eval-snapshot-20260729}
    IMAGE_SNAPSHOT_EXPECTED=2026-07-29
    LOCK_PATH=${H20_EVAL_LOCK_PATH:-${DISK_ROOT}/.csl_v4_h20_eval.lock}
    MASTER_ADDR=${H20_EVAL_MASTER_ADDR:-10.11.2.153}
    SOCKET_IFNAME=bond0
    GPU_NAME_MODE=exact
    GPU_NAME_EXPECTED="NVIDIA H20"
    HOST_LOG_DIR="${HOST_EVAL_ROOT}/host_logs"
    DOCKER_MOUNTS=(
      -v "${DISK_ROOT}:${CONTAINER_DISK_ROOT}"
      -v "${HOST_MEGATRON}:/root/Megatron-LM:ro"
    )
    ;;
  h200)
    ROOT=${H200_EVAL_ROOT:-/ssd/csl_v4_h200_eval_20260727}
    HOST_REPO="${ROOT}/repo"
    CONTAINER_REPO=/eval/repo
    HOST_EVAL_ROOT="${ROOT}"
    CONTAINER_EVAL_ROOT=/eval
    HOST_ADAPTER_ROOT="${ROOT}/adapters"
    CONTAINER_ADAPTER_ROOT=/eval/adapters
    HOST_MODEL="${ROOT}/model"
    CONTAINER_MODEL=/eval/model
    HOST_MEGATRON="${ROOT}/Megatron-LM"
    IMAGE=${H200_EVAL_IMAGE:-csl/sglang:dspark-r4-h200-20260727}
    IMAGE_SNAPSHOT_EXPECTED=${H200_EVAL_IMAGE_SNAPSHOT:-2026-07-27}
    LOCK_PATH=${H200_EVAL_LOCK_PATH:-${ROOT}/h200_eval.lock}
    MASTER_ADDR=127.0.0.1
    SOCKET_IFNAME=lo
    GPU_NAME_MODE=prefix
    GPU_NAME_EXPECTED="NVIDIA H200"
    HOST_LOG_DIR="${ROOT}/host_logs"
    DOCKER_MOUNTS=(
      -v "${ROOT}:/eval"
      -v "${HOST_MEGATRON}:/root/Megatron-LM:ro"
    )
    ;;
esac

if [[ "${MODE}" == "curve" ]]; then
  if [[ "${PROFILE}" == "h20" ]]; then
    CONTAINER="csl_v4_h20_eval_step${STEP}_${STAMP}"
  else
    CONTAINER="csl_v4_eval_step${STEP}_${STAMP}"
  fi
  HOST_LOG="${HOST_LOG_DIR}/step${STEP}.${STAMP}.log"
  EXP_ROOT="${CONTAINER_EVAL_ROOT}/experiments/Eval.KernelBenchL1.DeepSeekV4FlashLoRA.12k.turn1.n8.${PROFILE}"
  PORTS_DISPLAY=6386/8269/52465-52467
  PORTS_REGEX=':(6386|8269|52465|52466|52467)\b'
else
  CONTAINER="csl_v4_h20_eval_l${LEVEL}_step${STEP}_${STAMP}"
  HOST_LOG="${HOST_LOG_DIR}/l${LEVEL}.step${STEP}.${STAMP}.log"
  EXP_ROOT="${CONTAINER_EVAL_ROOT}/experiments/Eval.KernelBenchL${LEVEL}.DeepSeekV4FlashLoRA.12k.turn1.n8.h20"
  PORTS_DISPLAY=6386/8269/52365-52367
  PORTS_REGEX=':(6386|8269|52365|52366|52367)\b'
  if [[ "${LEVEL}" -eq 3 ]]; then
    DATASET_DIR=kernelbench-level3-validation-tvm-v2
    EXPECTED_PROMPTS=50
  else
    DATASET_DIR="kernelbench-level${LEVEL}-validation-tvm-v2"
    EXPECTED_PROMPTS=100
  fi
  EXPECTED_SAMPLES=$((EXPECTED_PROMPTS * 8))
  HOST_EVAL_DATA="${HOST_REPO}/Data/${DATASET_DIR}/train.parquet"
  CONTAINER_EVAL_DATA="${CONTAINER_REPO}/Data/${DATASET_DIR}/train.parquet"
  EVAL_DATASET_NAME="kb_l${LEVEL}_val"
fi

mkdir -p "${HOST_LOG_DIR}"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  echo "another ${PROFILE} evaluation owns ${LOCK_PATH}" >&2
  exit 1
fi
exec > >(tee -a "${HOST_LOG}") 2>&1

cleanup() {
  local status=$?
  local attempt
  trap - EXIT INT TERM
  for attempt in 1 2 3; do
    docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
    if ! docker inspect "${CONTAINER}" >/dev/null 2>&1; then
      echo "container_removed=${CONTAINER} exit_status=${status} attempt=${attempt}"
      exit "${status}"
    fi
    sleep 2
  done
  echo "failed to remove eval container after 3 attempts: ${CONTAINER}" >&2
  [[ "${status}" -ne 0 ]] || status=1
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "profile=${PROFILE} mode=${MODE} level=${LEVEL:-<curve>} step=${STEP}"
echo "image=${IMAGE} container=${CONTAINER} host_log=${HOST_LOG}"
echo "kernel_eval=${KERNEL_EVAL_PRIORITY}/${KERNEL_EVAL_WORKER_MAX_CONCURRENCY}/${KERNEL_EVAL_RATE_LIMIT}"

IMAGE_ID=$(docker image inspect "${IMAGE}" --format '{{.Id}}')
IMAGE_SNAPSHOT=$(docker image inspect "${IMAGE}" --format '{{index .Config.Labels "org.openai.kernel-eval.snapshot"}}')
if [[ "${IMAGE_SNAPSHOT}" != "${IMAGE_SNAPSHOT_EXPECTED}" ]]; then
  echo "unexpected ${PROFILE} evaluation image provenance: id=${IMAGE_ID} snapshot=${IMAGE_SNAPSHOT}" >&2
  exit 1
fi
echo "image_id=${IMAGE_ID} snapshot=${IMAGE_SNAPSHOT}"

test -s "${HOST_REPO}/train.py"
test -s "${HOST_MODEL}/config.json"
test -s "${HOST_MODEL}/chat_template.jinja"
test -s "${HOST_MEGATRON}/megatron/training/arguments.py"
if [[ "${MODE}" == "curve" ]]; then
  test -s "${HOST_REPO}/Data/kernelbench-level1-validation-tvm-v2/train.parquet"
  test -s "${HOST_REPO}/examples/kernel_agent/eval.deepseek-v4-flash-lora-curve.sh"
  if [[ "${PROFILE}" == "h200" ]]; then
    test -d "${ROOT}/cache/tvm-ffi"
  fi
else
  test -s "${HOST_REPO}/examples/kernel_agent/eval.deepseek-v4-flash.sh"
  test -s "${HOST_EVAL_DATA}"
fi
if [[ "${STEP}" -ne 0 ]]; then
  test -s "${HOST_ADAPTER_ROOT}/step${STEP}/adapter_model.safetensors"
  test -s "${HOST_ADAPTER_ROOT}/step${STEP}/adapter_config.json"
fi

if [[ "${MODE}" == "level" ]]; then
  docker run --rm -i \
    --network none \
    -v "${DISK_ROOT}:${CONTAINER_DISK_ROOT}:ro" \
    "${IMAGE}" \
    python3 - "${LEVEL}" "${EXPECTED_PROMPTS}" "${CONTAINER_EVAL_DATA}" <<'PY'
import pathlib
import sys

import pandas as pd

level, expected, eval_data = int(sys.argv[1]), int(sys.argv[2]), pathlib.Path(sys.argv[3])
df = pd.read_parquet(eval_data)
if len(df) != expected:
    raise SystemExit(f"level{level} needs {expected} prompts, saw {len(df)}")
ids = [int(item["problem_id"]) for item in df["extra_info"]]
if ids != list(range(1, expected + 1)):
    raise SystemExit(f"level{level} problem IDs are not ordered 1..{expected}: {ids}")
print("dataset_contract=PASS")
PY
fi

python3 - "${PROFILE}" "${GPU_NAME_MODE}" "${GPU_NAME_EXPECTED}" "${STEP}" "${HOST_ADAPTER_ROOT}" <<'PY'
import json
import pathlib
import subprocess
import sys

profile, name_mode, expected_name, step, adapter_root = (
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    int(sys.argv[4]),
    pathlib.Path(sys.argv[5]),
)
rows = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    text=True,
).strip().splitlines()
if len(rows) != 8:
    raise SystemExit(f"need exactly 8 {profile} GPUs, saw {len(rows)} rows")
busy = []
for row in rows:
    idx, name, memory, util = [item.strip() for item in row.split(",")]
    name_ok = name == expected_name if name_mode == "exact" else name.startswith(expected_name)
    if not name_ok:
        raise SystemExit(f"GPU {idx} is {name!r}, expected {expected_name}")
    idx, memory, util = int(idx), int(memory), int(util)
    if memory >= 4096 or util >= 20:
        busy.append((idx, memory, util))
if busy:
    raise SystemExit(f"{profile} GPUs are not idle: {busy}")

if step:
    cfg = json.loads((adapter_root / f"step{step}" / "adapter_config.json").read_text())
    if cfg.get("r") != 32 or abs(float(cfg.get("lora_alpha", 0)) - 181.01933598375615) > 1e-9:
        raise SystemExit(f"unexpected adapter config for step{step}: {cfg}")
print("gpu_idle_preflight=PASS")
print("adapter_contract=PASS" if step else "adapter_contract=SKIP base_control")
PY

if ss -ltn | grep -Eq "${PORTS_REGEX}"; then
  echo "Ray evaluation ports ${PORTS_DISPLAY} are already occupied" >&2
  exit 1
fi

kernelgym_healthy=0
for attempt in 1 2 3; do
  if python3 - "${KERNEL_ENV_URL}" <<'PY'
import json
import sys
import urllib.request

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(f"{sys.argv[1].rstrip('/')}/health", timeout=5) as response:
    payload = json.load(response)
if payload.get("status") != "healthy":
    raise SystemExit(f"KernelGym is not healthy: {payload}")
print("kernelgym_health=PASS")
PY
  then
    kernelgym_healthy=1
    break
  fi
  echo "KernelGym health attempt ${attempt}/3 failed" >&2
  sleep 2
done
if [[ "${kernelgym_healthy}" -ne 1 ]]; then
  echo "KernelGym failed all health checks" >&2
  exit 1
fi

docker run -d \
  --name "${CONTAINER}" \
  --init \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  "${DOCKER_MOUNTS[@]}" \
  -w "${CONTAINER_REPO}" \
  "${IMAGE}" sleep infinity >/dev/null

if [[ "${MODE}" == "curve" ]]; then
  docker exec \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e WANDB_MODE=offline \
    -e MODEL_PATH="${CONTAINER_MODEL}" \
    -e HF_MODEL_PATH="${CONTAINER_MODEL}" \
    -e ADAPTER_ROOT="${CONTAINER_ADAPTER_ROOT}" \
    -e EXP_ROOT="${EXP_ROOT}" \
    -e EVAL_ONLY_STEP="${STEP}" \
    -e FORCE_RERUN="${FORCE_RERUN:-0}" \
    -e MASTER_ADDR="${MASTER_ADDR}" \
    -e LOCAL_GLOO_SOCKET_IFNAME="${SOCKET_IFNAME}" \
    -e NCCL_SOCKET_IFNAME="${SOCKET_IFNAME}" \
    -e RAY_WORKER_HOST= \
    -e RAY_WORKER_IP= \
    -e RAY_HEAD_GPUS=8 \
    -e KERNEL_ENV_URL="${KERNEL_ENV_URL}" \
    -e KERNEL_EVAL_WORKER_MAX_CONCURRENCY="${KERNEL_EVAL_WORKER_MAX_CONCURRENCY}" \
    -e KERNEL_EVAL_RATE_LIMIT="${KERNEL_EVAL_RATE_LIMIT}" \
    -e KERNEL_EVAL_PRIORITY="${KERNEL_EVAL_PRIORITY}" \
    "${CONTAINER}" \
    bash "${CONTAINER_REPO}/examples/kernel_agent/eval.deepseek-v4-flash-lora-curve.sh"
  echo "step${STEP}_eval=PASS"
else
  ADAPTER_PATH=""
  if [[ "${STEP}" -ne 0 ]]; then
    ADAPTER_PATH="${CONTAINER_ADAPTER_ROOT}/step${STEP}"
  fi
  docker exec \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e WANDB_MODE=offline \
    -e V4_RUNTIME=dspark \
    -e MODEL_PATH="${CONTAINER_MODEL}" \
    -e HF_MODEL_PATH="${CONTAINER_MODEL}" \
    -e EVAL_DATA="${CONTAINER_EVAL_DATA}" \
    -e EVAL_DATASET_NAME="${EVAL_DATASET_NAME}" \
    -e EXP_ROOT="${EXP_ROOT}" \
    -e EVAL_TAG="step${STEP}" \
    -e MAX_CONTEXT_LEN=12288 \
    -e MAX_RESPONSE_LEN=12288 \
    -e MAX_TURNS=1 \
    -e EVAL_NUM_PROMPTS=0 \
    -e N_SAMPLES_PER_EVAL_PROMPT=8 \
    -e ROLLOUT_SEED=42 \
    -e ROLLOUT_TEMPERATURE=1 \
    -e ROLLOUT_TOP_P=1 \
    -e ENABLE_LORA_SERVER=1 \
    -e LORA_ADAPTER_PATH="${ADAPTER_PATH}" \
    -e LORA_NAME="eval_step${STEP}" \
    -e MASTER_ADDR="${MASTER_ADDR}" \
    -e LOCAL_GLOO_SOCKET_IFNAME="${SOCKET_IFNAME}" \
    -e NCCL_SOCKET_IFNAME="${SOCKET_IFNAME}" \
    -e RAY_WORKER_HOST= \
    -e RAY_WORKER_IP= \
    -e RAY_HEAD_GPUS=8 \
    -e KERNEL_ENV_URL="${KERNEL_ENV_URL}" \
    -e KERNEL_EVAL_WORKER_MAX_CONCURRENCY="${KERNEL_EVAL_WORKER_MAX_CONCURRENCY}" \
    -e KERNEL_EVAL_RATE_LIMIT="${KERNEL_EVAL_RATE_LIMIT}" \
    -e KERNEL_EVAL_PRIORITY="${KERNEL_EVAL_PRIORITY}" \
    -e SGLANG_MAX_RUNNING_REQUESTS=128 \
    -e RAY_PORT=6386 \
    -e RAY_DASHBOARD_PORT=8269 \
    -e RAY_TEMP_DIR="/dev/shm/ray_eval_dsv4_l${LEVEL}_step${STEP}" \
    "${CONTAINER}" \
    bash "${CONTAINER_REPO}/examples/kernel_agent/eval.deepseek-v4-flash.sh"

  SUMMARY_DIR="${HOST_EVAL_ROOT}/experiments/Eval.KernelBenchL${LEVEL}.DeepSeekV4FlashLoRA.12k.turn1.n8.h20/step${STEP}"
  SUMMARY_PATH=""
  if [[ -d "${SUMMARY_DIR}" ]]; then
    SUMMARY_PATH=$(find "${SUMMARY_DIR}" -maxdepth 1 -type f -name 'summary.*.txt' \
      -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
  fi
  if [[ -z "${SUMMARY_PATH}" ]] ||
      ! grep -q "^samples: ${EXPECTED_SAMPLES}  (missing env_result: 0)$" "${SUMMARY_PATH}"; then
    echo "evaluation summary is absent or incomplete: ${SUMMARY_PATH:-<missing>}" >&2
    exit 1
  fi
  echo "level${LEVEL}_step${STEP}_eval=PASS summary=${SUMMARY_PATH}"
fi
