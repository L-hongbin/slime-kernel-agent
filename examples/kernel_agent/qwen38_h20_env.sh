#!/usr/bin/env bash
# Source inside the H20 Qwen3.8 containers (SSH port 23538).
export SLIME_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export SLIME_MEGATRON_LM_PATH=${SLIME_MEGATRON_LM_PATH:-/root/Megatron-LM}
export H20_RUNTIME=${H20_RUNTIME:-/nfs/FM/chenshuailin/runtime/qwen38_piecewise_h20}
export PYTHONPATH="$SLIME_REPO:$SLIME_MEGATRON_LM_PATH${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export TMPDIR="$H20_RUNTIME/tmp"
export XDG_CACHE_HOME="$H20_RUNTIME/cache"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TILELANG_CACHE_DIR="$XDG_CACHE_HOME/tilelang"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export FLASHINFER_WORKSPACE_BASE="$H20_RUNTIME"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export SGLANG_CACHE_DIR="$XDG_CACHE_HOME/sglang"
export SGLANG_DG_CACHE_DIR="$XDG_CACHE_HOME/deep_gemm"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond0}
# Exclude mlx5_5, which has a history of link errors on Node69.
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_0,mlx5_3,mlx5_4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export NO_PROXY="localhost,127.0.0.1,10.11.2.153,10.11.2.164,10.11.2.169,10.11.2.170${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
ulimit -Sn 131072 || { echo 'Cannot raise open-file limit to 131072' >&2; return 1; }
