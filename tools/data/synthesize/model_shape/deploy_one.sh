#!/usr/bin/env bash
set -Eeuo pipefail

action=${1:-status}
port=${SHAPE_SERVER_PORT:-31053}
disk_root=${SHAPE_SERVER_DISK_ROOT:-/mnt/md1}
container_disk_root=${SHAPE_SERVER_CONTAINER_DISK_ROOT:-/nfs/FM}
image=${SHAPE_SERVER_IMAGE:-csl/sglang:dspark-r1-eval-snapshot-20260729}
container=${SHAPE_SERVER_CONTAINER:-csl_dsv4_0731_shape_tp8_low_v10}
speculative_algorithm=${SHAPE_SERVER_SPECULATIVE_ALGORITHM:-DSPARK}
model_host=${SHAPE_SERVER_MODEL:-${disk_root}/chenshuailin/checkpoints/deepseek-ai/DeepSeek-V4-Flash-0731}
model_container=${container_disk_root}${model_host#${disk_root}}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
reasoning_patch=${script_dir}/patches/sglang_dsv4_0731_reasoning_effort.patch
dspark_patch=${script_dir}/patches/sglang_dspark_swa_eviction.patch
runtime_preflight=${script_dir}/runtime_preflight.py

case "${speculative_algorithm}" in
  DSPARK|NONE) ;;
  *)
    echo "SHAPE_SERVER_SPECULATIVE_ALGORITHM must be DSPARK or NONE" >&2
    exit 2
    ;;
esac

status() {
  docker ps -a --filter "name=^/${container}$" --format '{{.Names}} {{.Status}}'
}

verify_command_contract() {
  docker inspect "${container}" --format '{{json .Args}}' | python3 -c '
import json, sys
args = " ".join(json.load(sys.stdin))
speculative_algorithm = sys.argv[1]
required = {
    "--tp-size": "8",
    "--max-running-requests": "64",
    "--mem-fraction-static": "0.90",
    "--context-length": "524288",
    "--chunked-prefill-size": "2048",
    "--max-prefill-tokens": "32768",
    "--swa-full-tokens-ratio": "0.1",
    "--moe-runner-backend": "flashinfer_mxfp4",
    "--attention-backend": "dsv4",
    "--kv-cache-dtype": "fp8_e4m3",
    "--served-model-name": "deepseek-v4-flash-0731",
    "--reasoning-parser": "deepseek-v4",
}
for flag, value in required.items():
    if f"{flag} {value}" not in args:
        raise SystemExit(f"missing runtime contract: {flag} {value}")
for forbidden in ("--dp-size", "--data-parallel-size", "--enable-dp-attention", "--disable-cuda-graph", "--enable-torch-compile", "--chat-template"):
    if forbidden in args:
        raise SystemExit(f"forbidden runtime option: {forbidden}")
if speculative_algorithm == "DSPARK":
    if "--speculative-algorithm DSPARK" not in args:
        raise SystemExit("missing runtime contract: --speculative-algorithm DSPARK")
else:
    for forbidden in ("--speculative-algorithm", "--speculative-moe-runner-backend"):
        if forbidden in args:
            raise SystemExit(f"no-DSPARK control contains {forbidden}")
print(f"PASS live command: pure TP8, concurrency 64, mem 0.90, SWA 0.1, speculative={speculative_algorithm}")
' "${speculative_algorithm}"
}

host_preflight() {
  test -s "${model_host}/config.json"
  test -s "${model_host}/model.safetensors.index.json"
  test -s "${reasoning_patch}"
  test -s "${dspark_patch}"
  test -s "${runtime_preflight}"
  if [[ $(find "${model_host}" -maxdepth 1 -type f -name 'model-*-of-00048.safetensors' | wc -l) -ne 48 ]]; then
    echo "model does not contain all 48 official shards: ${model_host}" >&2
    return 1
  fi
  if find "${model_host}" -maxdepth 1 -type f -name '*.incomplete' -print -quit | grep -q .; then
    echo "model copy is incomplete: ${model_host}" >&2
    return 1
  fi
  local snapshot
  snapshot=$(docker image inspect "${image}" --format '{{index .Config.Labels "org.openai.kernel-eval.snapshot"}}')
  [[ "${snapshot}" == 2026-07-29 ]] || {
    echo "unexpected frozen image provenance: ${image} (${snapshot})" >&2
    return 1
  }

  local gpu_rows
  gpu_rows=$(docker run --rm --gpus all --network none --entrypoint /usr/bin/nvidia-smi "${image}" \
    --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits)
  python3 - "${gpu_rows}" <<'PY'
import sys

rows = sys.argv[1].splitlines()
if len(rows) != 8:
    raise SystemExit(f"need exactly 8 H20 GPUs, saw {len(rows)}")
busy = []
for row in rows:
    idx, name, memory, util = [item.strip() for item in row.split(",")]
    if name != "NVIDIA H20":
        raise SystemExit(f"GPU {idx} is {name!r}, expected NVIDIA H20")
    if int(memory) >= 4096 or int(util) >= 20:
        busy.append((int(idx), int(memory), int(util)))
if busy:
    raise SystemExit(f"H20 GPUs are not idle: {busy}")
print("PASS eight idle H20 GPUs")
PY

  docker run --rm --network none --entrypoint bash \
    -v "${reasoning_patch}:/tmp/reasoning.patch:ro" \
    -v "${dspark_patch}:/tmp/dspark.patch:ro" \
    -v "${runtime_preflight}:/tmp/runtime_preflight.py:ro" \
    "${image}" -lc '
      set -Eeuo pipefail
      cd /sgl-workspace/sglang
      patch --batch --forward -p1 < /tmp/reasoning.patch
      patch --batch --forward -p1 < /tmp/dspark.patch
      python3 /tmp/runtime_preflight.py --sglang-root /sgl-workspace/sglang
    '
  echo "PASS official 0731 weights, frozen image, no custom chat template, and patch source contract"
}

case "${action}" in
  status)
    status
    ;;
  preflight)
    if docker inspect "${container}" >/dev/null 2>&1; then
      echo "refusing preflight over existing container: ${container}" >&2
      exit 1
    fi
    if ss -ltn | grep -Eq ":${port}\\b"; then
      echo "port is already occupied: ${port}" >&2
      exit 1
    fi
    host_preflight
    ;;
  start)
    if docker inspect "${container}" >/dev/null 2>&1; then
      echo "refusing to replace existing container: ${container}" >&2
      exit 1
    fi
    if ss -ltn | grep -Eq ":${port}\\b"; then
      echo "port is already occupied: ${port}" >&2
      exit 1
    fi
    host_preflight
    if [[ ${speculative_algorithm} == DSPARK ]]; then
      speculative_launch_args="--speculative-algorithm DSPARK --speculative-moe-runner-backend flashinfer_mxfp4"
    else
      speculative_launch_args=""
    fi
    docker run -d \
      --name "${container}" --init --gpus all --network host --ipc host \
      --label "org.openai.shape.speculative=${speculative_algorithm}" \
      --shm-size 64g --ulimit memlock=-1 --ulimit stack=67108864 \
      --ulimit nofile=1048576:1048576 \
      -v "${disk_root}:${container_disk_root}" \
      -v "${reasoning_patch}:/tmp/reasoning.patch:ro" \
      -v "${dspark_patch}:/tmp/dspark.patch:ro" \
      -v "${runtime_preflight}:/tmp/runtime_preflight.py:ro" \
      -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
      -e V4_RUNTIME=dspark -e SGLANG_DSV4_FP4_EXPERTS=1 \
      -e SGLANG_SHARED_EXPERT_TP1=1 -e SGLANG_OPT_FUSE_WQA_WKV=0 \
      -e SGLANG_OPT_USE_TILELANG_MHC_PRE=true \
      -e SGLANG_OPT_USE_TILELANG_MHC_POST=true \
      -e SGLANG_OPT_DEEPGEMM_HC_PRENORM=true \
      -e SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false \
      -e SGLANG_MEMORY_SAVER_CUDA_GRAPH=true \
      -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=true \
      -e SGLANG_JIT_DEEPGEMM_FAST_WARMUP=true \
      -e SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT=true \
      -e SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false \
      -e SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=false \
      "${image}" bash -lc "
        set -Eeuo pipefail
        cd /sgl-workspace/sglang
        patch --batch --forward -p1 < /tmp/reasoning.patch
        patch --batch --forward -p1 < /tmp/dspark.patch
        python3 /tmp/runtime_preflight.py --sglang-root /sgl-workspace/sglang
        exec python3 -m sglang.launch_server \
          --model-path '${model_container}' --trust-remote-code \
          --host 0.0.0.0 --port '${port}' \
          --served-model-name deepseek-v4-flash-0731 \
          --reasoning-parser deepseek-v4 \
          --context-length 524288 --mem-fraction-static 0.90 \
          --max-running-requests 64 --chunked-prefill-size 2048 \
          --max-prefill-tokens 32768 --page-size 256 --cuda-graph-max-bs 64 \
          --swa-full-tokens-ratio 0.1 --tp-size 8 \
          --attention-backend dsv4 --kv-cache-dtype fp8_e4m3 \
          --moe-runner-backend flashinfer_mxfp4 --moe-a2a-backend none \
          ${speculative_launch_args} \
          --disable-custom-all-reduce --disable-flashinfer-autotune \
          --schedule-policy fcfs --random-seed 1234 --watchdog-timeout 2400 \
          --decode-log-interval 20 --skip-server-warmup
      "
    echo "started ${container}; verify after load with: $0 verify"
    ;;
  verify)
    [[ $(docker inspect "${container}" --format '{{.State.Running}}') == true ]]
    verify_command_contract
    docker exec "${container}" python3 /tmp/runtime_preflight.py \
      --sglang-root /sgl-workspace/sglang
    curl --fail --silent --show-error "http://127.0.0.1:${port}/health" >/dev/null
    echo "PASS live health endpoint: http://127.0.0.1:${port}/health"
    ;;
  wait)
    for _ in $(seq 1 270); do
      if [[ $(docker inspect "${container}" --format '{{.State.Running}}' 2>/dev/null || true) != true ]]; then
        docker logs --tail 200 "${container}" >&2 || true
        exit 1
      fi
      if curl --fail --silent "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        exec "$0" verify
      fi
      sleep 10
    done
    docker logs --tail 200 "${container}" >&2 || true
    echo "server did not become healthy within 45 minutes" >&2
    exit 1
    ;;
  smoke)
    python3 - <<'PY' | curl --fail --silent --show-error \
      -H 'Content-Type: application/json' --data-binary @- \
      "http://127.0.0.1:${port}/v1/chat/completions" | python3 -c '
import json, sys
response = json.load(sys.stdin)
if not isinstance(response.get("choices"), list) or len(response["choices"]) != 1:
    raise SystemExit(f"invalid completion response: {response}")
print("PASS explicit-low thinking request accepted by live server")
'
import json
print(json.dumps({
    "model": "deepseek-v4-flash-0731",
    "messages": [{"role": "user", "content": "Reply with OK."}],
    "max_tokens": 32,
    "temperature": 0,
    "chat_template_kwargs": {"thinking": True},
    "reasoning_effort": "low",
}))
PY
    ;;
  stop)
    docker stop "${container}"
    ;;
  *)
    echo "usage: $0 {status|preflight|start|wait|verify|smoke|stop}" >&2
    exit 2
    ;;
esac
