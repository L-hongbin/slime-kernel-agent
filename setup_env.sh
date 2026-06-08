#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Keep the local /nfs checkpoint path available inside containers that only
# mount the shared /ms checkpoint tree.
if [[ ! -d "/nfs/FM/chenshuailin/checkpoints" ]]; then
    mkdir -p /nfs/FM/chenshuailin/
    ln -s /ms/FM/checkpoints /nfs/FM/chenshuailin/checkpoints
fi

cd "${REPO_ROOT}"

# Install the repo itself without disturbing the carefully pinned runtime deps.
python3 -m pip install -e . --no-deps --break-system-packages
python3 -m pip install debugpy --break-system-packages

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

# FlashInfer GDN kernels need the CUTLASS DSL package to expose the `cutlass`
# Python module. If the wheel metadata exists but files are missing, SGLang can
# later fail with `run_pretranspose_decode` being None during CUDA graph capture.
if ! verify_flashinfer_gdn; then
    python3 -m pip install \
        --break-system-packages \
        --force-reinstall \
        --no-cache-dir \
        --no-deps \
        "nvidia-cutlass-dsl==4.5.1" \
        "nvidia-cutlass-dsl-libs-base==4.5.1"
    verify_flashinfer_gdn
fi
