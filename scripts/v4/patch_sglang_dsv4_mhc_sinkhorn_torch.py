#!/usr/bin/env python3
"""Patch SGLang DeepSeek-V4 MHC split-sinkhorn with an opt-in torch fallback."""

from __future__ import annotations

from pathlib import Path


TARGET = Path("/sgl-workspace/sglang/python/sglang/srt/layers/mhc.py")
MARKER = "SLIME_TORCH_MHC_SPLIT_SINKHORN_PATCH"


HELPER = """

def _slime_env_bool(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "y")


def _slime_hc_split_sinkhorn_torch(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    # SLIME_TORCH_MHC_SPLIT_SINKHORN_PATCH
    b, s, _ = mixes.size()
    flat = mixes.float().reshape(-1, (2 + hc_mult) * hc_mult)
    pre_logits = flat[:, :hc_mult]
    post_logits = flat[:, hc_mult : 2 * hc_mult]
    comb_logits = flat[:, 2 * hc_mult :].reshape(-1, hc_mult, hc_mult)

    pre = (
        torch.sigmoid(pre_logits * hc_scale[0] + hc_base[:hc_mult])
        + eps
    )
    post = 2 * torch.sigmoid(
        post_logits * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    comb = (
        comb_logits * hc_scale[2]
        + hc_base[2 * hc_mult :].reshape(1, hc_mult, hc_mult)
    )

    row_max = comb.max(dim=2, keepdim=True).values
    comb = torch.exp(comb - row_max)
    comb = comb / comb.sum(dim=2, keepdim=True) + eps
    comb = comb / (comb.sum(dim=1, keepdim=True) + eps)

    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=2, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=1, keepdim=True) + eps)

    return (
        pre.reshape(b, s, hc_mult),
        post.reshape(b, s, hc_mult),
        comb.reshape(b, s, hc_mult, hc_mult),
    )
"""


def main() -> int:
    if not TARGET.exists():
        raise SystemExit(f"missing target: {TARGET}")

    text = TARGET.read_text()
    if MARKER in text:
        print(f"already patched: {TARGET}")
        return 0

    if "import os\n" not in text:
        text = text.replace("import math\n", "import math\nimport os\n", 1)

    anchor = "\n\ndef hc_split_sinkhorn(\n"
    if anchor not in text:
        raise SystemExit("could not find hc_split_sinkhorn anchor")
    text = text.replace(anchor, HELPER + anchor, 1)

    old = """def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    b, s, _ = mixes.size()
"""
    new = """def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    if not _slime_env_bool("SGLANG_OPT_USE_TILELANG_MHC_SPLIT_SINKHORN", True):
        return _slime_hc_split_sinkhorn_torch(
            mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps
        )

    b, s, _ = mixes.size()
"""
    if old not in text:
        raise SystemExit("could not find hc_split_sinkhorn function body")
    text = text.replace(old, new, 1)

    TARGET.write_text(text)
    print(f"patched: {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
