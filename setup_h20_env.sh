#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Keep the local /nfs checkpoint path available inside containers that only
# mount the shared /ms checkpoint tree.
if [[ ! -d "/nfs/FM/chenshuailin/checkpoints" ]]; then
    mkdir -p /nfs/FM/chenshuailin/
    ln -s /ms/FM/checkpoints /nfs/FM/chenshuailin/checkpoints
fi

if [[ ! -d "/ms/FM/checkpoints/Qwen-Zoo/" ]]; then
    mkdir -p /ms/FM/checkpoints/
    ln -s /nfs/FM/chenshuailin/checkpoints/Qwen /ms/FM/checkpoints/Qwen-Zoo
fi

cd "${REPO_ROOT}"

# Some node container images (e.g. node62) ship without iproute2, so `ip` is
# missing and the run scripts' head-node IP check fails. Install it here. Kept
# non-fatal so the rest of setup still runs when apt is unavailable/offline
# (the run scripts also fall back to `hostname -I`).
if ! command -v ip >/dev/null 2>&1; then
    (apt-get update -qq && apt-get install -y --no-install-recommends iproute2) \
        || echo "[warn] iproute2 install failed; run scripts fall back to 'hostname -I'"
fi

CUTLASS_DSL_VERSION="4.5.2"

# Install the repo itself without disturbing the carefully pinned runtime deps.
python3 -m pip install -e . --no-deps --break-system-packages
python3 -m pip install debugpy --break-system-packages

# Muon (the mandatory V4 optimizer) hard-asserts on this package at actor init
# ("Emerging Optimizers is not installed" — bit the first node54 formal launch,
# 2026-07-11: it was pip-installed post-creation on the older containers and
# never part of the image). Pin the commit the fleet runs. Non-fatal when
# offline: rsync site-packages/emerging_optimizers{,-*.dist-info} from any
# working container instead.
if ! python3 -c "import emerging_optimizers" >/dev/null 2>&1; then
    python3 -m pip install --break-system-packages --no-deps \
        "emerging-optimizers @ git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@9ad154b5962e163c2fba67e5cb9f8d23d11e9165" \
        || echo "[warn] emerging-optimizers install failed (offline?); rsync it from a working container"
fi

verify_flashinfer_gdn() {
    python3 - <<'PY'
import sys

try:
    import cutlass  # noqa: F401
    import flashinfer.gdn_decode as gd
except Exception as exc:
    print(f"FlashInfer GDN dependency check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)

required = {
    "_PRETRANSPOSE_AVAILABLE": True,
    "_NONTRANSPOSE_AVAILABLE": True,
    "_MTP_AVAILABLE": True,
}
for name, expected in required.items():
    actual = getattr(gd, name, None)
    if actual is not expected:
        print(f"FlashInfer GDN dependency check failed: {name}={actual!r}", file=sys.stderr)
        raise SystemExit(1)

for name in ("run_pretranspose_decode", "run_nontranspose_decode", "run_mtp_decode"):
    if not callable(getattr(gd, name, None)):
        print(f"FlashInfer GDN dependency check failed: {name} is not callable", file=sys.stderr)
        raise SystemExit(1)

print("FlashInfer GDN dependency check passed")
PY
}

verify_cutlass_dsl_version() {
    python3 - "${CUTLASS_DSL_VERSION}" <<'PY'
import importlib.metadata
import sys

expected = sys.argv[1]
packages = ("nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-base")
for package in packages:
    try:
        actual = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        print(f"CUTLASS DSL version check failed: {package} is not installed", file=sys.stderr)
        raise SystemExit(1)
    if actual != expected:
        print(
            f"CUTLASS DSL version check failed: {package}={actual}, expected {expected}",
            file=sys.stderr,
        )
        raise SystemExit(1)

print(f"CUTLASS DSL version check passed: {expected}")
PY
}

# FlashInfer GDN kernels need the CUTLASS DSL package to expose the `cutlass`
# Python module. If the wheel metadata exists but files are missing, SGLang can
# later fail with `run_pretranspose_decode` being None during CUDA graph capture.
if ! verify_flashinfer_gdn || ! verify_cutlass_dsl_version; then
    python3 -m pip install \
        --break-system-packages \
        --force-reinstall \
        --no-cache-dir \
        --no-deps \
        "nvidia-cutlass-dsl==${CUTLASS_DSL_VERSION}" \
        "nvidia-cutlass-dsl-libs-base==${CUTLASS_DSL_VERSION}"
    verify_flashinfer_gdn
    verify_cutlass_dsl_version
fi

# Optional local path compatibility symlinks.
# These paths are convenient for CUDA_RL/H20 runs, but the setup should still
# succeed on machines where the source datasets or checkpoints are unavailable.
optional_symlink() {
    local src="$1"
    local dst="$2"

    if [[ ! -e "${src}" ]]; then
        echo "[optional] skip missing source: ${src}"
        return 0
    fi

    mkdir -p "$(dirname -- "${dst}")"

    if [[ -L "${dst}" ]]; then
        rm -f "${dst}"
    elif [[ -e "${dst}" ]]; then
        echo "[optional] skip existing non-symlink path: ${dst}"
        return 0
    fi

    ln -s "${src}" "${dst}"
    echo "[optional] created symlink: ${dst} -> ${src}"
}

optional_symlink \
    /nfs/FM/chenshuailin/checkpoints/Qwen \
    /ms/FM/checkpoints/Qwen-Zoo

optional_symlink \
    /nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B/torch_dist_tp4_pp2 \
    /ms/FM/lihongbin/dataset/CUDA_RL/megatron_ckpt/Qwen3.6-27B-TP4-PP2-Torch-Dist

optional_symlink \
    "${REPO_ROOT}/Data" \
    /ms/FM/lihongbin/dataset/CUDA_RL/cuda_rl
