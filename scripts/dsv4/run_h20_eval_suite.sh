#!/usr/bin/env bash
# Evaluate one DS-V4 model/adaptor target on KernelBench Level 1/2/3 in one
# disposable 8xH20 job.  The three datasets share one model load, but every
# sample carries eval_level metadata and is summarized independently.

set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 stepSTEP|model0731" >&2
  exit 2
fi

TARGET=$1
RESULT_TAG=${H20_EVAL_RESULT_TAG:-${TARGET}}
RAY_TEMP_TAG=${H20_EVAL_RAY_TEMP_TAG:-${TARGET}}
DISK_ROOT=${H20_EVAL_DISK_ROOT:-/mnt/md1}
CONTAINER_DISK_ROOT=${H20_EVAL_CONTAINER_DISK_ROOT:-/nfs/FM}
HOST_REPO=${H20_EVAL_REPO:-${DISK_ROOT}/chenshuailin/projects/kernel_agents/slime-v4flash-lora-eval-20260801}
HOST_MEGATRON=${H20_EVAL_MEGATRON:-${DISK_ROOT}/chenshuailin/projects/kernel_agents/Megatron-LM-eval-20260729}
MASTER_ADDR=${H20_EVAL_MASTER_ADDR:-10.11.2.153}
IMAGE=${H20_EVAL_IMAGE:-csl/sglang:dspark-r1-eval-snapshot-20260729}
V4_EXPERIMENTS_ROOT=${H20_EVAL_V4_EXPERIMENTS_ROOT:-${DISK_ROOT}/chenshuailin/projects/kernel_agents/slime-v4flash-lora/experiments}
OUTPUT_ROOT=${H20_EVAL_SUITE_ROOT:-${V4_EXPERIMENTS_ROOT}/csl_v4_kernelbench_3turn_20260801}
ADAPTER_ROOT=${H20_EVAL_ADAPTER_ROOT:-${V4_EXPERIMENTS_ROOT}/csl_v4r21_fp4_pp1cp2_12k_dppo_predictive_resume40_20260722/eval_adapters}
HOST_DSPARK_MODEL="${DISK_ROOT}/chenshuailin/checkpoints/deepseek-ai/DeepSeek-V4-Flash-DSpark"
HOST_0731_MODEL="${DISK_ROOT}/chenshuailin/checkpoints/deepseek-ai/DeepSeek-V4-Flash-0731"
H20_EVAL_LOCK_PATH=${H20_EVAL_LOCK_PATH:-${DISK_ROOT}/.csl_v4_h20_eval.lock}
KERNEL_EVAL_WORKER_MAX_CONCURRENCY=${KERNEL_EVAL_WORKER_MAX_CONCURRENCY:-32}
KERNEL_EVAL_RATE_LIMIT=${KERNEL_EVAL_RATE_LIMIT:-32}
KERNEL_EVAL_PRIORITY=${KERNEL_EVAL_PRIORITY:-low}
KERNEL_ENV_URL=${KERNEL_ENV_URL:-http://127.0.0.1:20211}
# The generate wrapper's default guard is sized for one KernelGym turn
# (client timeout + task timeout + cleanup margin).  This suite runs up to
# three turns, so scale that guard with the evaluation contract instead of
# aborting otherwise healthy trajectories at the single-turn boundary.
KERNEL_AGENT_GENERATE_GUARD_SEC=${KERNEL_AGENT_GENERATE_GUARD_SEC:-9000}
MAX_TURNS=3
SAMPLES_PER_PROMPT=8
MAX_CONTEXT_LEN=${EVAL_MAX_CONTEXT_LEN:-12288}
MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-${MAX_CONTEXT_LEN}}
HOST_CHAT_TEMPLATE_OVERRIDE=${H20_EVAL_CHAT_TEMPLATE_OVERRIDE:-}
HOST_LAUNCHER_OVERRIDE=${H20_EVAL_LAUNCHER_OVERRIDE:-}
APPLY_CHAT_TEMPLATE_KWARGS=${APPLY_CHAT_TEMPLATE_KWARGS:-}

map_to_container() {
  local host_path=$1
  case "${host_path}" in
    "${DISK_ROOT}" | "${DISK_ROOT}/"*)
      printf '%s%s' "${CONTAINER_DISK_ROOT}" "${host_path#${DISK_ROOT}}"
      ;;
    *)
      echo "path is outside H20_EVAL_DISK_ROOT and cannot be mounted: ${host_path}" >&2
      return 1
      ;;
  esac
}

CONTAINER_REPO=$(map_to_container "${HOST_REPO}")
CONTAINER_OUTPUT_ROOT=$(map_to_container "${OUTPUT_ROOT}")
CONTAINER_ADAPTER_ROOT=$(map_to_container "${ADAPTER_ROOT}")

if [[ ! "${MAX_CONTEXT_LEN}" =~ ^[1-9][0-9]*$ ||
      ! "${MAX_RESPONSE_LEN}" =~ ^[1-9][0-9]*$ ]]; then
  echo "evaluation context and response lengths must be positive integers" >&2
  exit 2
fi
if [[ ! "${RESULT_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "evaluation result tag contains unsupported characters: ${RESULT_TAG}" >&2
  exit 2
fi
if [[ ! "${RAY_TEMP_TAG}" =~ ^[A-Za-z0-9._-]+$ || ${#RAY_TEMP_TAG} -gt 24 ]]; then
  echo "Ray temp tag must use safe characters and be at most 24 bytes: ${RAY_TEMP_TAG}" >&2
  exit 2
fi

case "${MAX_CONTEXT_LEN}" in
  12288) CONTEXT_TAG=12k ;;
  32768) CONTEXT_TAG=32k ;;
  *) CONTEXT_TAG="ctx${MAX_CONTEXT_LEN}" ;;
esac

case "${TARGET}" in
  step[0-9]*)
    STEP=${TARGET#step}
    if [[ ! "${STEP}" =~ ^[0-9]+$ ]] || ((10#${STEP} % 20 != 0)); then
      echo "adapter step must be a non-negative multiple of 20: ${STEP}" >&2
      exit 2
    fi
    HOST_MODEL=${HOST_DSPARK_MODEL}
    HOST_ADAPTER="${ADAPTER_ROOT}/step${STEP}"
    CONTAINER_ADAPTER="${CONTAINER_ADAPTER_ROOT}/step${STEP}"
    ENABLE_LORA_SERVER=1
    LORA_NAME="eval_step${STEP}"
    ;;
  model0731)
    STEP=""
    HOST_MODEL=${HOST_0731_MODEL}
    HOST_ADAPTER=""
    CONTAINER_ADAPTER=""
    ENABLE_LORA_SERVER=0
    LORA_NAME=""
    ;;
  *)
    echo "unknown target ${TARGET}; expected stepSTEP or model0731" >&2
    exit 2
    ;;
esac

CONTAINER_MODEL=$(map_to_container "${HOST_MODEL}")
STAMP=$(date +%Y%m%d.%H%M%S)
CONTAINER="csl_v4_h20_eval_${RESULT_TAG}_turn3_${STAMP}"
EXP_ROOT="${CONTAINER_OUTPUT_ROOT}/experiments/Eval.KernelBenchL123.DeepSeekV4Flash.${RESULT_TAG}.${CONTEXT_TAG}.turn3.n8.h20"
HOST_EXP_ROOT="${OUTPUT_ROOT}/experiments/Eval.KernelBenchL123.DeepSeekV4Flash.${RESULT_TAG}.${CONTEXT_TAG}.turn3.n8.h20"
HOST_EVAL_DIR="${HOST_EXP_ROOT}/${RESULT_TAG}"
HOST_LOG_DIR="${OUTPUT_ROOT}/host_logs"
HOST_LOG="${HOST_LOG_DIR}/${RESULT_TAG}.${STAMP}.log"

DOCKER_OVERRIDE_MOUNTS=()
if [[ -n "${HOST_CHAT_TEMPLATE_OVERRIDE}" ]]; then
  test -s "${HOST_CHAT_TEMPLATE_OVERRIDE}"
  DOCKER_OVERRIDE_MOUNTS+=(
    -v "${HOST_CHAT_TEMPLATE_OVERRIDE}:${CONTAINER_MODEL}/chat_template.jinja:ro"
  )
fi
if [[ -n "${HOST_LAUNCHER_OVERRIDE}" ]]; then
  test -s "${HOST_LAUNCHER_OVERRIDE}"
  DOCKER_OVERRIDE_MOUNTS+=(
    -v "${HOST_LAUNCHER_OVERRIDE}:${CONTAINER_REPO}/examples/kernel_agent/eval.deepseek-v4-flash.sh:ro"
  )
fi

if [[ ! "${KERNEL_EVAL_WORKER_MAX_CONCURRENCY}" =~ ^[1-9][0-9]*$ ||
      ! "${KERNEL_EVAL_RATE_LIMIT}" =~ ^[1-9][0-9]*$ ||
      ! "${KERNEL_AGENT_GENERATE_GUARD_SEC}" =~ ^[1-9][0-9]*$ ]]; then
  echo "KernelGym concurrency, rate limit, and generate guard must be positive integers" >&2
  exit 2
fi

mkdir -p "${HOST_LOG_DIR}" "${OUTPUT_ROOT}"
exec 9>"${H20_EVAL_LOCK_PATH}"
if ! flock -n 9; then
  echo "another H20 evaluation owns ${H20_EVAL_LOCK_PATH}" >&2
  exit 1
fi
exec > >(tee -a "${HOST_LOG}") 2>&1

cleanup() {
  local status=$?
  local attempt
  trap - EXIT INT TERM
  if (( status != 0 )) && [[ -f "${HOST_LOG}" ]]; then
    python3 - "${HOST_LOG}" "${KERNEL_ENV_URL}" <<'PY' || true
import collections
import concurrent.futures
import re
import sys
import urllib.error
import urllib.request

log_path, base_url = sys.argv[1:]
task_ids = []
seen = set()
with open(log_path, errors="replace") as handle:
    for line in handle:
        match = re.search(r"POST /evaluate task_id=(\S+)", line)
        if match and match.group(1) not in seen:
            seen.add(match.group(1))
            task_ids.append(match.group(1))


def cancel(task_id):
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/tasks/{task_id}", method="DELETE"
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except Exception as error:  # best-effort cleanup must not hide the run status
        return type(error).__name__


owned_ids = [
    candidate
    for task_id in task_ids
    for candidate in (
        task_id,
        f"{task_id}_compile",
        f"{task_id}_kernel",
        f"{task_id}_ref",
    )
]
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    results = collections.Counter(executor.map(cancel, owned_ids))
print(
    "kernelgym_cleanup="
    f"attempted parent_ids={len(task_ids)} endpoint_ids={len(owned_ids)} "
    f"results={dict(results)}"
)
PY
  fi
  for attempt in 1 2 3; do
    docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
    if ! docker inspect "${CONTAINER}" >/dev/null 2>&1; then
      echo "container_removed=${CONTAINER} exit_status=${status} attempt=${attempt}"
      exit "${status}"
    fi
    sleep 2
  done
  echo "failed to remove eval container after 3 attempts: ${CONTAINER}" >&2
  exit 1
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "target=${TARGET} result_tag=${RESULT_TAG} model=${HOST_MODEL} adapter=${HOST_ADAPTER:-<none>}"
echo "host_log=${HOST_LOG} exp_root=${HOST_EXP_ROOT}"
echo "contract=levels1,2,3/prompts100,100,50/samples8/turns3/context${MAX_CONTEXT_LEN}/response${MAX_RESPONSE_LEN}"
echo "kernel_eval=${KERNEL_EVAL_PRIORITY}/${KERNEL_EVAL_WORKER_MAX_CONCURRENCY}/${KERNEL_EVAL_RATE_LIMIT}"
echo "chat_template_override=${HOST_CHAT_TEMPLATE_OVERRIDE:-<none>}"
echo "launcher_override=${HOST_LAUNCHER_OVERRIDE:-<none>}"
echo "apply_chat_template_kwargs=${APPLY_CHAT_TEMPLATE_KWARGS:-<launcher-default>}"
echo "ray_temp_tag=${RAY_TEMP_TAG}"

IMAGE_ID=$(docker image inspect "${IMAGE}" --format '{{.Id}}')
IMAGE_SNAPSHOT=$(docker image inspect "${IMAGE}" --format '{{index .Config.Labels "org.openai.kernel-eval.snapshot"}}')
if [[ "${IMAGE_SNAPSHOT}" != "2026-07-29" ]]; then
  echo "unexpected H20 evaluation image provenance: id=${IMAGE_ID} snapshot=${IMAGE_SNAPSHOT}" >&2
  exit 1
fi
echo "image_id=${IMAGE_ID} snapshot=${IMAGE_SNAPSHOT}"

test -s "${HOST_REPO}/train.py"
test -s "${HOST_REPO}/examples/kernel_agent/eval.deepseek-v4-flash.sh"
test -s "${HOST_REPO}/examples/kernel_agent/prompt_config/kernelbench_l123_eval.yaml"
test -s "${HOST_MODEL}/config.json"
test -s "${HOST_MODEL}/model.safetensors.index.json"
test -s "${HOST_MEGATRON}/megatron/training/arguments.py"
if find "${HOST_MODEL}" -maxdepth 1 -type f -name '*.incomplete' -print -quit | grep -q .; then
  echo "model download/copy is incomplete: ${HOST_MODEL}" >&2
  exit 1
fi
if [[ $(find "${HOST_MODEL}" -maxdepth 1 -type f -name 'model-*-of-00048.safetensors' | wc -l) -ne 48 ]]; then
  echo "model does not contain all 48 safetensor shards: ${HOST_MODEL}" >&2
  exit 1
fi
if [[ -n "${HOST_ADAPTER}" ]]; then
  test -s "${HOST_ADAPTER}/adapter_model.safetensors"
  test -s "${HOST_ADAPTER}/adapter_config.json"
fi

python3 - <<'PY'
import subprocess

rows = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
    text=True,
).strip().splitlines()
if len(rows) != 8:
    raise SystemExit(f"need exactly 8 H20 GPUs, saw {len(rows)} rows")
busy = []
for row in rows:
    idx, name, memory, util = [item.strip() for item in row.split(",")]
    if name != "NVIDIA H20":
        raise SystemExit(f"GPU {idx} is {name!r}, expected NVIDIA H20")
    if int(memory) >= 4096 or int(util) >= 20:
        busy.append((int(idx), int(memory), int(util)))
if busy:
    raise SystemExit(f"H20 GPUs are not idle: {busy}")
print("gpu_idle_preflight=PASS")
PY

docker run --rm -i \
  --network none \
  -v "${DISK_ROOT}:${CONTAINER_DISK_ROOT}:ro" \
  -w "${CONTAINER_REPO}" \
  "${IMAGE}" \
  python3 - "${CONTAINER_REPO}" "${CONTAINER_ADAPTER}" <<'PY'
import json
import pathlib
import sys

import pandas as pd

repo, adapter = pathlib.Path(sys.argv[1]), sys.argv[2]
expected = {1: 100, 2: 100, 3: 50}
for level, count in expected.items():
    path = repo / "Data" / f"kernelbench-level{level}-validation-tvm-v2" / "train.parquet"
    df = pd.read_parquet(path)
    if len(df) != count:
        raise SystemExit(f"level{level} needs {count} prompts, saw {len(df)}")
    ids = [int(item["problem_id"]) for item in df["extra_info"]]
    if ids != list(range(1, count + 1)):
        raise SystemExit(f"level{level} problem IDs are not ordered 1..{count}")

if adapter:
    cfg = json.loads((pathlib.Path(adapter) / "adapter_config.json").read_text())
    if cfg.get("r") != 32 or abs(float(cfg.get("lora_alpha", 0)) - 181.01933598375615) > 1e-9:
        raise SystemExit(f"unexpected adapter config: {cfg}")
print("dataset_contract=PASS")
print("adapter_contract=PASS")
PY

if ss -ltn | grep -Eq ':(6386|8269|52365|52366|52367)\b'; then
  echo "Ray evaluation ports 6386/8269/52365-52367 are already occupied" >&2
  exit 1
fi

python3 "${HOST_REPO}/scripts/check_kernelgym_health.py" \
  --url "${KERNEL_ENV_URL}" --timeout 5 --attempts 3 --interval 2

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
  -v "${DISK_ROOT}:${CONTAINER_DISK_ROOT}" \
  -v "${HOST_MEGATRON}:/root/Megatron-LM:ro" \
  "${DOCKER_OVERRIDE_MOUNTS[@]}" \
  -w "${CONTAINER_REPO}" \
  "${IMAGE}" sleep infinity >/dev/null

docker exec \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e WANDB_MODE=offline \
  -e V4_RUNTIME=dspark \
  -e MODEL_PATH="${CONTAINER_MODEL}" \
  -e HF_MODEL_PATH="${CONTAINER_MODEL}" \
  -e EVAL_DATA="${CONTAINER_REPO}/Data/kernelbench-level1-validation-tvm-v2/train.parquet" \
  -e EVAL_DATASET_NAME=kb_l1_val \
  -e EVAL_CONFIG="${CONTAINER_REPO}/examples/kernel_agent/prompt_config/kernelbench_l123_eval.yaml" \
  -e EVAL_SUMMARY_GROUP_KEY=eval_level \
  -e EXP_ROOT="${EXP_ROOT}" \
  -e EVAL_TAG="${RESULT_TAG}" \
  -e MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN}" \
  -e MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN}" \
  -e MAX_TURNS="${MAX_TURNS}" \
  -e EVAL_NUM_PROMPTS=0 \
  -e N_SAMPLES_PER_EVAL_PROMPT="${SAMPLES_PER_PROMPT}" \
  -e ROLLOUT_SEED=42 \
  -e ROLLOUT_TEMPERATURE=1 \
  -e ROLLOUT_TOP_P=1 \
  -e ENABLE_LORA_SERVER="${ENABLE_LORA_SERVER}" \
  -e LORA_ADAPTER_PATH="${CONTAINER_ADAPTER}" \
  -e LORA_NAME="${LORA_NAME}" \
  -e MASTER_ADDR="${MASTER_ADDR}" \
  -e LOCAL_GLOO_SOCKET_IFNAME=bond0 \
  -e NCCL_SOCKET_IFNAME=bond0 \
  -e RAY_WORKER_HOST= \
  -e RAY_WORKER_IP= \
  -e RAY_HEAD_GPUS=8 \
  -e KERNEL_ENV_URL="${KERNEL_ENV_URL}" \
  -e KERNEL_EVAL_WORKER_MAX_CONCURRENCY="${KERNEL_EVAL_WORKER_MAX_CONCURRENCY}" \
  -e KERNEL_EVAL_RATE_LIMIT="${KERNEL_EVAL_RATE_LIMIT}" \
  -e KERNEL_EVAL_PRIORITY="${KERNEL_EVAL_PRIORITY}" \
  -e KERNEL_AGENT_GENERATE_GUARD_SEC="${KERNEL_AGENT_GENERATE_GUARD_SEC}" \
  -e APPLY_CHAT_TEMPLATE_KWARGS="${APPLY_CHAT_TEMPLATE_KWARGS}" \
  -e SGLANG_MAX_RUNNING_REQUESTS=128 \
  -e RAY_PORT=6386 \
  -e RAY_DASHBOARD_PORT=8269 \
  -e RAY_TEMP_DIR="/dev/shm/ray_dsv4_${RAY_TEMP_TAG}" \
  "${CONTAINER}" \
  bash "${CONTAINER_REPO}/examples/kernel_agent/eval.deepseek-v4-flash.sh"

SUMMARY_PATH=""
DUMP_PATH=""
if [[ -d "${HOST_EVAL_DIR}" ]]; then
  SUMMARY_PATH=$(find "${HOST_EVAL_DIR}" -maxdepth 1 -type f -name 'summary.*.txt' \
    -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
fi
if [[ -d "${HOST_EVAL_DIR}/dumps/rollout_data" ]]; then
  DUMP_PATH=$(find "${HOST_EVAL_DIR}/dumps/rollout_data" -maxdepth 1 -type f -name 'eval_*.pt' \
    -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
fi
if [[ -z "${SUMMARY_PATH}" || -z "${DUMP_PATH}" ]]; then
  echo "evaluation summary or dump is missing" >&2
  exit 1
fi

python3 - "${DUMP_PATH}" <<'PY'
import collections
import sys
import torch

obj = torch.load(sys.argv[1], weights_only=False)
samples = obj.get("samples", [])
expected = {1: 800, 2: 800, 3: 400}
groups = collections.defaultdict(set)
real_records = collections.Counter()
missing_env = collections.Counter()
abort_sentinels = collections.Counter()
for sample in samples:
    meta = sample.get("metadata") or {}
    level = int(meta["eval_level"])
    group_id = sample.get("group_id")
    if group_id is not None:
        groups[level].add(group_id)
    if not meta.get("is_pad_turn"):
        real_records[level] += 1
        if not isinstance(meta.get("env_result"), dict):
            is_wall_clock_abort_sentinel = (
                sample.get("remove_sample") is True
                and sample.get("status") == "aborted"
                and sample.get("reward") == 0.0
                and sample.get("response_length") == 1
                and sample.get("response") == ""
                and meta.get("finish_reason") == "aborted"
                and meta.get("abort_reason") == "wall_clock_timeout"
                and meta.get("remove_reason") == "aborted"
                and isinstance(meta.get("turn_idx"), int)
            )
            if is_wall_clock_abort_sentinel:
                abort_sentinels[level] += 1
            else:
                missing_env[level] += 1
for level, count in expected.items():
    if len(groups[level]) != count:
        raise SystemExit(f"level{level} trajectory count {len(groups[level])} != {count}")
    if missing_env[level]:
        raise SystemExit(f"level{level} has {missing_env[level]} unexpected real records without env_result")
    max_abort_sentinels = max(1, count // 100)
    if abort_sentinels[level] > max_abort_sentinels:
        raise SystemExit(
            f"level{level} has {abort_sentinels[level]} wall-clock abort sentinels; "
            f"limit is {max_abort_sentinels} (1% of trajectories)"
        )
    if not (count <= real_records[level] <= count * 3):
        raise SystemExit(f"level{level} real record count is invalid: {real_records[level]}")
    print(
        f"level{level}_contract=PASS trajectories={count} real_records={real_records[level]} "
        f"wall_clock_abort_sentinels={abort_sentinels[level]}"
    )
PY

echo "eval_suite=PASS target=${TARGET} result_tag=${RESULT_TAG} summary=${SUMMARY_PATH} dump=${DUMP_PATH}"
