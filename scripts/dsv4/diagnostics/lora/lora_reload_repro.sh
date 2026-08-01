#!/usr/bin/env bash
# Standalone repro for the DeepSeek-V4 LoRA-serve "reload crash":
# cudaErrorIllegalAddress on the first forward AFTER a per-step unload+reload of
# the LoRA adapter (the first load + generation runs clean). This launches ONLY a
# sglang rollout engine (no Ray / no slime / no training) with the production V4
# LoRA flags, then a Python driver POSTs load -> generate -> unload -> reload ->
# generate in a loop, exactly like slime's weight-sync does, so we get a fast
# fix-test loop.
#
# DO NOT run while another job holds the GPUs. Usage:
#   NUM_GPUS=8 TP=8 DP=8 EP=8 bash scripts/dsv4/diagnostics/lora/lora_reload_repro.sh
#   NUM_GPUS=2 TP=2 DP=2 EP=2 bash scripts/dsv4/diagnostics/lora/lora_reload_repro.sh
# Bisect toggles (env):
#   SGLANG_DISABLE_CUDA_GRAPH=1   # THE decisive split: graphs off
#   MAX_LORAS_PER_BATCH=2         # double-buffer slot
#   ITERS=4                       # number of load->gen->unload cycles
set -uo pipefail
# Crashed schedulers dump ~10.5G apport cores EACH on the physical host and
# have repeatedly filled the 876G root disk during crash-loop debugging.
ulimit -c 0 2>/dev/null || true

REPO=${REPO:-/nfs/FM/chenshuailin/projects/kernel_agents/slime-v4flash-lora}
HF_CKPT=${HF_CKPT:-/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-31000}
NUM_GPUS=${NUM_GPUS:-2}
TP=${TP:-2}
DP=${DP:-2}
EP=${EP:-2}
MEM_FRACTION=${MEM_FRACTION:-0.85}
CONTEXT_LEN=${CONTEXT_LEN:-2048}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-64}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-64}
MAX_LORAS_PER_BATCH=${MAX_LORAS_PER_BATCH:-1}
MAX_LORA_RANK=${MAX_LORA_RANK:-16}
# Shared-expert LoRA targets ride along when SHARED_EXPERT=1 (native
# adapter leaves w1/w3/w2 normalize to the served gate_up_proj/down_proj).
SHARED_EXPERT=${SHARED_EXPERT:-0}
if [[ "${SHARED_EXPERT}" == "1" ]]; then
  LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"wq_a wkv wq_b wo_b wkv_gate gate_up_proj down_proj"}
else
  LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"wq_a wkv wq_b wo_b wkv_gate"}
fi
# Truncated-model fast loop: serve only the first N layers via
# --json-model-override-args (no new checkpoint needed). N=6 keeps 2 plain +
# c4 + c128 + c4 + c128 compressor layers + MoE — every code path the LoRA
# reload bug class needs — and cuts bring-up from ~7 min to ~1 min. The bug is
# batch-state machinery, layer-count-independent.
NUM_LAYERS=${NUM_LAYERS:-0}
SGLANG_DISABLE_CUDA_GRAPH=${SGLANG_DISABLE_CUDA_GRAPH:-0}
USE_DEEPEP=${USE_DEEPEP:-1}
SGLANG_DEEPEP_CONFIG=${SGLANG_DEEPEP_CONFIG:-'{"normal_dispatch":{"num_sms":32},"normal_combine":{"num_sms":32}}'}
ITERS=${ITERS:-4}
# same-slot: reproduce the crash (unload+reload same name, needs MAX_LORAS_PER_BATCH=1).
# alternating: exercise the QeRL-style fix (load-new-before-unload-old, needs >=2).
MODE=${MODE:-same-slot}
if [[ "${MODE}" == "alternating" && "${MAX_LORAS_PER_BATCH}" -lt 2 ]]; then
  MAX_LORAS_PER_BATCH=2
fi
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}
LOG=${LOG:-${REPO}/local_artifacts/deepseek-v4/r2_logs/lora_reload_repro_${RUN_ID}.log}
SCRATCH=${SCRATCH:-/tmp/lora_reload_repro_${RUN_ID}}
mkdir -p "${SCRATCH}" "$(dirname "${LOG}")"

export PYTHONUNBUFFERED=1
export PATH="/usr/local/cuda/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS - 1)))}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
export PYTHONPATH="${REPO}:/root/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NO_PROXY="127.0.0.1,localhost,0.0.0.0,::1"
export no_proxy="${NO_PROXY}"
# V4 engine env (mirrors full_loop_smoke.sh). LoRA needs wq_a/wkv unfused.
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=${SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK:-false}
export SGLANG_MEMORY_SAVER_CUDA_GRAPH=${SGLANG_MEMORY_SAVER_CUDA_GRAPH:-true}
export SGLANG_DSV4_FP4_EXPERTS=${SGLANG_DSV4_FP4_EXPERTS:-0}
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-256}
export SGLANG_OPT_USE_TILELANG_MHC_PRE=${SGLANG_OPT_USE_TILELANG_MHC_PRE:-true}
export SGLANG_OPT_USE_TILELANG_MHC_POST=${SGLANG_OPT_USE_TILELANG_MHC_POST:-true}
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=${SGLANG_OPT_DEEPGEMM_HC_PRENORM:-true}
# CUDA_LAUNCH_BLOCKING makes the illegal-access surface at the true faulting op.
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1}

echo "=== lora_reload_repro sanity ===" | tee "${LOG}"
echo "gpus=${NUM_GPUS} tp=${TP} dp=${DP} ep=${EP} deepep=${USE_DEEPEP} disable_cuda_graph=${SGLANG_DISABLE_CUDA_GRAPH} max_loras_per_batch=${MAX_LORAS_PER_BATCH} iters=${ITERS}" | tee -a "${LOG}"
echo "fuse_wqa_wkv=${SGLANG_OPT_FUSE_WQA_WKV} lora_targets='${LORA_TARGET_MODULES}' cvd=${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG}"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tee -a "${LOG}"

# ---- build the sglang server args (raw ServerArgs names, no --sglang- prefix) ----
SERVER_ARGS=(
  --model-path "${HF_CKPT}"
  --trust-remote-code
  --host "${HOST}" --port "${PORT}"
  --tp-size "${TP}"
  --dp-size "${DP}"
  --ep-size "${EP}"
  --enable-dp-attention
  --attention-backend dsv4
  --kv-cache-dtype fp8_e4m3
  --context-length "${CONTEXT_LEN}"
  --mem-fraction-static "${MEM_FRACTION}"
  # dp-attention divides chunked_prefill_size by dp_size; dsv4 forces page_size=256,
  # and the result must be divisible by 256. So feed dp_size*256 (÷dp_size = 256).
  --chunked-prefill-size "$((DP * 256))"
  --max-prefill-tokens "${CONTEXT_LEN}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}"
  --page-size 256
  --disable-custom-all-reduce
  --watchdog-timeout 600
  --decode-log-interval 1
  --enable-lora
  --max-lora-rank "${MAX_LORA_RANK}"
  --max-loras-per-batch "${MAX_LORAS_PER_BATCH}"
  --lora-target-modules ${LORA_TARGET_MODULES}
  --trust-remote-code
)
if [[ "${USE_DEEPEP}" == "1" ]]; then
  SERVER_ARGS+=(--moe-a2a-backend deepep --deepep-config "${SGLANG_DEEPEP_CONFIG}")
fi
if [[ -n "${LORA_BACKEND:-}" ]]; then
  SERVER_ARGS+=(--lora-backend "${LORA_BACKEND}")
fi
if [[ "${SGLANG_DISABLE_CUDA_GRAPH}" == "1" ]]; then
  SERVER_ARGS+=(--disable-cuda-graph)
fi
if [[ "${NUM_LAYERS}" != "0" ]]; then
  SERVER_ARGS+=(--json-model-override-args "{\"num_hidden_layers\": ${NUM_LAYERS}}")
fi

echo "=== launching sglang server ===" | tee -a "${LOG}"
echo "python -m sglang.launch_server ${SERVER_ARGS[*]}" | tee -a "${LOG}"
setsid python -m sglang.launch_server "${SERVER_ARGS[@]}" >>"${LOG}" 2>&1 &
SERVER_PID=$!
echo "server_pid=${SERVER_PID}" | tee -a "${LOG}"

cleanup() {
  # Kill ONLY this run's server process group (setsid gave it its own pgid).
  # The old pattern-based pkill on "port ${PORT}" murdered the NEXT run's
  # server whenever a stale wrapper exited late (observed twice on 2026-07-09).
  echo "=== cleanup: killing server pgid ${SERVER_PID} ===" | tee -a "${LOG}"
  kill -9 -- "-${SERVER_PID}" 2>/dev/null || kill -9 "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# ---- drive the load/reload loop once the server is healthy ----
DRIVER_LORA_ARGS=()
if [[ "${SHARED_EXPERT}" == "1" ]]; then
  DRIVER_LORA_ARGS+=(--shared-expert)
fi
python3 "${REPO}/scripts/dsv4/diagnostics/lora/lora_reload_repro_driver.py" \
  --host "${HOST}" --port "${PORT}" \
  --hf-ckpt "${HF_CKPT}" \
  --num-layers "${NUM_LAYERS}" \
  --rank "${MAX_LORA_RANK}" \
  "${DRIVER_LORA_ARGS[@]}" \
  --iters "${ITERS}" \
  --b-scale "${B_SCALE:-0}" \
  --mode "${MODE}" \
  --sustain-rounds "${SUSTAIN_ROUNDS:-0}" \
  --sustain-max-new "${SUSTAIN_MAX_NEW:-256}" \
  --log "${LOG}" 2>&1 | tee -a "${LOG}"
DRIVER_RC=${PIPESTATUS[0]}
echo "=== driver exit rc=${DRIVER_RC} ===" | tee -a "${LOG}"
exit "${DRIVER_RC}"
