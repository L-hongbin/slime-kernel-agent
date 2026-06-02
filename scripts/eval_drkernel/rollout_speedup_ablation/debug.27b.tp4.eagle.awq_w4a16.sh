#!/bin/bash

set -eo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)"
SCRIPT_HELPER_DIR=${SCRIPT_HELPER_DIR:-${REPO_ROOT}/scripts}
MODEL_DIR=${MODEL_DIR:-/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B}
PYTHON_BIN=${PYTHON_BIN:-${REPO_ROOT}/.venv_llmcompressor/bin/python}

HF_W4A16_DIR=${HF_W4A16_DIR:-checkpoints/quantized/AWQ/Qwen3.6-27B-AWQ-W4A16-mlp-preservefix}
AWQ_MIN_QUANTIZED_MLP=${AWQ_MIN_QUANTIZED_MLP:-180}
AWQ_REQUIRE_SYMMETRIC=${AWQ_REQUIRE_SYMMETRIC:-1}
AWQ_REQUIRE_SGLANG_ASYM_SUPPORT=${AWQ_REQUIRE_SGLANG_ASYM_SUPPORT:-0}
SGLANG_SOURCE_ROOT=${SGLANG_SOURCE_ROOT:-/sgl-workspace/sglang/python}

if [ ! -x "${PYTHON_BIN}" ]; then
   echo "missing executable PYTHON_BIN=${PYTHON_BIN}" >&2
   exit 1
fi

CHECK_ARGS=(
   --checkpoint "${HF_W4A16_DIR}"
   --reference-checkpoint "${MODEL_DIR}"
   --require-mtp
   --min-quantized-mlp "${AWQ_MIN_QUANTIZED_MLP}"
)

if [ "${AWQ_REQUIRE_SYMMETRIC}" = "1" ]; then
   CHECK_ARGS+=(--require-symmetric)
fi

if [ "${AWQ_REQUIRE_SGLANG_ASYM_SUPPORT}" = "1" ]; then
   CHECK_ARGS+=(--require-sglang-asym-support --sglang-source-root "${SGLANG_SOURCE_ROOT}")
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/quantize/utils/check_awq_w4a16.py" "${CHECK_ARGS[@]}"

export HF_W8A8_DIR="${HF_W4A16_DIR}"
export EXPT_LABEL=${EXPT_LABEL:-awq.w4a16.mlp.100x8.eagle}
export EVAL_CONFIG_PATH=${EVAL_CONFIG_PATH:-${SCRIPT_HELPER_DIR}/eval_kernelbench_level1.yaml}

exec bash "${REPO_ROOT}/scripts/eval_drkernel/rollout_speedup_ablation/debug.27b.tp4.eagle.w8a8.sh"
