#!/usr/bin/env bash

export SLIME_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export SLIME_MEGATRON_LM_PATH=${SLIME_MEGATRON_LM_PATH:-/root/Megatron-LM}
export B300_RUNTIME=${B300_RUNTIME:-$SLIME_REPO/local_artifacts/qwen38_b300_r9/runtime}
export SLIME_TRAIN_PACKAGES=${SLIME_TRAIN_PACKAGES:-$B300_RUNTIME/train_packages}
export SLIME_TRAIN_LD_LIBRARY_PATH="${SLIME_TRAIN_LD_LIBRARY_PATH:-/usr/local/lib/python3.12/dist-packages/z3/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$SLIME_REPO:$SLIME_MEGATRON_LM_PATH${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export TMPDIR=${B300_TMPDIR:-/tmp/slime_qwen38_b300}
if (( ${#TMPDIR} > 80 )); then
    echo 'B300_TMPDIR must be at most 80 characters to leave room for Unix socket filenames' >&2
    return 1
fi
export XDG_CACHE_HOME="$B300_RUNTIME/cache"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TILELANG_CACHE_DIR="$XDG_CACHE_HOME/tilelang"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export FLASHINFER_WORKSPACE_BASE="$B300_RUNTIME"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export SGLANG_CACHE_DIR="$XDG_CACHE_HOME/sglang"
export SGLANG_DG_CACHE_DIR="$XDG_CACHE_HOME/deep_gemm"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens16f1np1}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export NO_PROXY="localhost,127.0.0.1,192.168.99.151,192.168.112.55${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"

if [[ "${CONFIG_DRY_RUN:-0}" != 1 ]]; then
    mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
    ulimit -Sn 131072 || { echo 'Cannot raise open-file limit to 131072' >&2; return 1; }
fi
