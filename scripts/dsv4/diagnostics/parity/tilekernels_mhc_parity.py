"""Offline fwd+bwd parity + micro-bench for the DeepSeek-V4-Flash mHC
HyperConnection ("pre") stage across three implementations:

  * ``ref``      : our fp32 torch ground truth
                   (custom_kernels/deepseek_v4/mhc/reference.py :: hyper_connection_forward)
  * ``ours``     : our TileLang kernel
                   (custom_kernels/deepseek_v4/mhc/kernel.py :: hyper_connection)
  * ``official`` : DeepSeek's official TileKernels
                   (tile_kernels.modeling.mhc :: mhc_pre)

All three compute the identical math (verified below). The three
parameterizations map 1:1 with these conventions:

    ref/ours  hyper_connection(residual, fn, base, scale)      -> (post, comb, collapsed)
    official  mhc_pre(residual, fn, scale, base, ...)          -> (layer_input, (post_mix, comb_mix))

    official post_mix == ref post   iff  post_mult_value = 2.0     (ref hardcodes 2*sigmoid)
    official comb_mix == ref comb   iff  sinkhorn_repeat  = 20     (= hc_sinkhorn_iters)
    pre_eps = sinkhorn_eps = norm_eps = 1e-6                       (= HC_EPS / RMS_NORM_EPS)
    fn [MIX=24, H*D=16384]   scale [3]   base [MIX=24]             (identical shapes)
    NOTE the (scale, base) argument ORDER is swapped between the two APIs.

The official ``mhc_pre`` grad-enabled path routes the residual gradient through a
Megatron main-grad fusion hook (``residual.untyped_storage().grad_from_mhc_post``,
via ``mhc_pre_norm_fn(..., fuse_grad_acc=True)``), so it cannot be ``.backward()``-ed
standalone. For an apples-to-apples autograd comparison we rebuild ``mhc_pre`` from
the SAME published ops with ``fuse_grad_acc=False`` so autograd accumulates the
residual grad normally (``official_pre_autograd`` below). We ALSO exercise the
shipped ``mhc_pre`` under inference mode (its fused ``mhc_pre_big_fuse`` kernel) for
forward-only parity.

Run (node with an idle GPU, GPU 0 only):
    CUDA_VISIBLE_DEVICES=0 \
    TILELANG_CACHE_DIR=/tmp/claude-0/tilelang_cache_eval \
    /tmp/claude-0/tilekernels_eval_venv/bin/python scripts/dsv4/diagnostics/parity/tilekernels_mhc_parity.py
"""

from __future__ import annotations

import os
import sys

import torch

# ----------------------------------------------------------------- import paths
_REPO = "/nfs/FM/chenshuailin/projects/kernel_agents/slime-v4flash-lora"
_OURS_MHC = os.path.join(_REPO, "custom_kernels/deepseek_v4/mhc")
sys.path.insert(0, _OURS_MHC)

import kernel as OURS  # noqa: E402  (our TileLang hyper_connection)
import reference as REF  # noqa: E402  (our fp32 torch ground truth)

# official TileKernels
from tile_kernels.modeling.mhc import mhc_pre  # noqa: E402
from tile_kernels.modeling.mhc.ops import (  # noqa: E402
    mhc_pre_apply_mix,
    mhc_pre_norm_fn,
    mhc_pre_split_mixes,
    sinkhorn_normalize,
)

DEV = "cuda"
H = REF.HC_MULT  # 4
D = REF.HIDDEN  # 4096
MIX = REF.MIX  # 24
M = H * D  # 16384
SINK_ITERS = REF.HC_SINKHORN_ITERS  # 20
EPS = REF.HC_EPS  # 1e-6
POST_MULT = 2.0  # ref hardcodes post = 2*sigmoid

# V4-Flash real (B,S) shapes to probe.
SHAPES = [(1, 128), (2, 1024), (1, 4096)]


# ----------------------------------------------------------------- official pre, autograd-friendly
def official_pre_autograd(residual, fn, scale, base):
    """Rebuild the official ``mhc_pre`` grad path from its published ops with
    ``fuse_grad_acc=False`` so autograd accumulates residual.grad normally
    (identical math to the shipped ``mhc_pre``; only the grad-plumbing differs).
    Returns (layer_input, post_mix, comb_mix)."""
    mixes = mhc_pre_norm_fn(residual, fn, None, EPS, fuse_grad_acc=False, n_splits=16)
    pre_mix, post_mix, comb_mix = mhc_pre_split_mixes(mixes, scale, base, H, POST_MULT, EPS)
    comb_mix = sinkhorn_normalize(comb_mix, repeat=SINK_ITERS, eps=EPS)
    layer_input = mhc_pre_apply_mix(residual, pre_mix)
    return layer_input, post_mix, comb_mix


# ----------------------------------------------------------------- diff helpers
def _diffs(a, b):
    """max abs and max rel diff of a vs b (b is the reference denominator)."""
    a = a.detach().float()
    b = b.detach().float()
    abs_d = (a - b).abs()
    max_abs = abs_d.max().item()
    denom = b.abs().max().item()
    max_rel = (max_abs / denom) if denom > 0 else float("nan")
    return max_abs, max_rel


def _mk_inputs(B, S, seed):
    """Same RNG inputs for every impl. bf16 residual + fp32 fn/base/scale
    (the dtypes the official norm_fn kernel asserts)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(B, S, H, D, device=DEV, dtype=torch.bfloat16, generator=g)
    fn = (torch.randn(MIX, M, device=DEV, generator=g) / M**0.5).to(torch.float32)
    base = (torch.randn(MIX, device=DEV, generator=g) * 0.1).to(torch.float32)
    scale = (torch.rand(3, device=DEV, generator=g) + 0.5).to(torch.float32)
    return x, fn, base, scale


def _run_grad(which, x, fn, base, scale, gp, gc, gl):
    """Run one impl fwd+bwd, return dict of tensors {post,comb,coll,dx,dfn,dbase,dscale}.

    ``which`` in {"ref","ours","official"} selects the call convention. Residual
    grad for the official path comes straight from autograd (fuse disabled)."""
    xr = x.clone().requires_grad_(True)
    fnr = fn.clone().requires_grad_(True)
    br = base.clone().requires_grad_(True)
    sr = scale.clone().requires_grad_(True)

    if which == "official":
        coll, post, comb = official_pre_autograd(xr, fnr, sr, br)
        post = post.squeeze(-1)  # [...,H,1] -> [...,H] to match ref/ours
    elif which == "ours":
        post, comb, coll = OURS.hyper_connection(xr, fnr, br, sr)
    elif which == "ref":
        post, comb, coll = REF.hyper_connection_forward(xr, fnr, br, sr)
    else:
        raise ValueError(which)

    loss = (post.float() * gp).sum() + (comb.float() * gc).sum() + (coll.float() * gl).sum()
    loss.backward()
    return {
        "post": post,
        "comb": comb,
        "coll": coll,
        "dx": xr.grad,
        "dfn": fnr.grad,
        "dbase": br.grad,
        "dscale": sr.grad,
    }


def _fwd_only_shipped(x, fn, base, scale):
    """Shipped official ``mhc_pre`` under inference mode -> exercises the fused
    ``mhc_pre_big_fuse`` kernel. Returns (layer_input, post, comb)."""
    with torch.no_grad():
        layer_input, (post_mix, comb_mix) = mhc_pre(
            x,
            fn,
            scale,
            base,
            norm_weight=None,
            norm_eps=EPS,
            mhc_mult=H,
            post_mult_value=POST_MULT,
            pre_eps=EPS,
            sinkhorn_eps=EPS,
            sinkhorn_repeat=SINK_ITERS,
            n_splits=16,
        )
    return layer_input, post_mix.squeeze(-1), comb_mix


# ----------------------------------------------------------------- timing
def _bench(fn_call, iters=20, warmup=5):
    for _ in range(warmup):
        fn_call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn_call()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]  # median (ms)


def _bench_fwd(which, x, fn, base, scale):
    if which == "official":
        return _bench(lambda: official_pre_autograd(x, fn, scale, base))
    if which == "ours":
        return _bench(lambda: OURS.hyper_connection(x, fn, base, scale))
    if which == "official_shipped":
        return _bench(lambda: _fwd_only_shipped(x, fn, base, scale))
    raise ValueError(which)


def _bench_fwdbwd(which, x, fn, base, scale, gp, gc, gl):
    def _step():
        xr = x.clone().requires_grad_(True)
        fnr = fn.clone().requires_grad_(True)
        br = base.clone().requires_grad_(True)
        sr = scale.clone().requires_grad_(True)
        if which == "official":
            coll, post, comb = official_pre_autograd(xr, fnr, sr, br)
            post = post.squeeze(-1)
        else:
            post, comb, coll = OURS.hyper_connection(xr, fnr, br, sr)
        ((post.float() * gp).sum() + (comb.float() * gc).sum() + (coll.float() * gl).sum()).backward()

    return _bench(_step)


# ----------------------------------------------------------------- main
def main():
    print(
        f"host={os.uname().nodename}  torch={torch.__version__}  "
        f"cuda={torch.version.cuda}  dev={torch.cuda.get_device_name(0)}"
    )
    import tilelang

    print(f"tilelang={tilelang.__version__}")
    print(f"mHC config: H={H} D={D} MIX={MIX} sinkhorn_iters={SINK_ITERS} " f"eps={EPS} post_mult={POST_MULT}")
    print("=" * 96)

    FWD_KEYS = ["post", "comb", "coll"]
    BWD_KEYS = ["dx", "dfn", "dbase", "dscale"]

    for B, S in SHAPES:
        print(f"\n############### shape B={B} S={S}  (N={B*S} tokens) ###############")
        x, fn, base, scale = _mk_inputs(B, S, seed=1234 + B * 7 + S)

        # shared cotangents for the scalar loss
        g = torch.Generator(device=DEV).manual_seed(99)
        gp = torch.randn(B, S, H, device=DEV, generator=g)
        gc = torch.randn(B, S, H, H, device=DEV, generator=g)
        gl = torch.randn(B, S, D, device=DEV, generator=g)

        outs = {}
        for which in ("ref", "ours", "official"):
            try:
                outs[which] = _run_grad(which, x, fn, base, scale, gp, gc, gl)
            except Exception as ex:  # noqa: BLE001
                print(f"  [ERROR] impl={which} raised: {type(ex).__name__}: {ex}")
                outs[which] = None

        # forward-only shipped fused inference path
        try:
            ship_li, ship_post, ship_comb = _fwd_only_shipped(x, fn, base, scale)
            ship = {"post": ship_post, "comb": ship_comb, "coll": ship_li}
        except Exception as ex:  # noqa: BLE001
            print(f"  [ERROR] official_shipped(inference) raised: {type(ex).__name__}: {ex}")
            ship = None

        # ---- parity tables ----
        pairs = [("official", "ref"), ("ours", "ref"), ("official", "ours")]
        print("\n  FORWARD parity  (max_abs / max_rel), denominator = 2nd impl")
        print(f"    {'pair':<20}{'post':>22}{'comb':>22}{'collapsed':>22}")
        for a, b in pairs:
            if outs.get(a) is None or outs.get(b) is None:
                print(f"    {a+' vs '+b:<20}{'(skipped)':>22}")
                continue
            row = f"    {a+' vs '+b:<20}"
            for k in FWD_KEYS:
                ma, mr = _diffs(outs[a][k], outs[b][k])
                row += f"{ma:>10.2e}/{mr:<10.2e} "
            print(row)
        if ship is not None and outs.get("ref") is not None:
            row = f"    {'shipped(inf) vs ref':<20}"
            for k in FWD_KEYS:
                ma, mr = _diffs(ship[k], outs["ref"][k])
                row += f"{ma:>10.2e}/{mr:<10.2e} "
            print(row)

        print("\n  BACKWARD parity  (max_abs / max_rel), denominator = 2nd impl")
        print(f"    {'pair':<20}{'d_x':>22}{'d_fn':>22}{'d_base':>22}{'d_scale':>22}")
        for a, b in pairs:
            if outs.get(a) is None or outs.get(b) is None:
                print(f"    {a+' vs '+b:<20}{'(skipped)':>22}")
                continue
            row = f"    {a+' vs '+b:<20}"
            for k in BWD_KEYS:
                ma, mr = _diffs(outs[a][k], outs[b][k])
                row += f"{ma:>10.2e}/{mr:<10.2e} "
            print(row)

        # ---- timing ----
        print("\n  TIMING  (median of 20 iters, ms)")
        try:
            t_off_f = _bench_fwd("official", x, fn, base, scale)
            t_our_f = _bench_fwd("ours", x, fn, base, scale)
            t_ship_f = _bench_fwd("official_shipped", x, fn, base, scale)
            t_off_fb = _bench_fwdbwd("official", x, fn, base, scale, gp, gc, gl)
            t_our_fb = _bench_fwdbwd("ours", x, fn, base, scale, gp, gc, gl)
            print(f"    {'impl':<24}{'fwd (ms)':>14}{'fwd+bwd (ms)':>16}")
            print(f"    {'official (autograd)':<24}{t_off_f:>14.3f}{t_off_fb:>16.3f}")
            print(f"    {'ours (hyper_connection)':<24}{t_our_f:>14.3f}{t_our_fb:>16.3f}")
            print(f"    {'official_shipped (inf)':<24}{t_ship_f:>14.3f}{'n/a':>16}")
            print(f"    speedup ours/official  fwd={t_off_f / t_our_f:.2f}x  " f"fwd+bwd={t_off_fb / t_our_fb:.2f}x")
        except Exception as ex:  # noqa: BLE001
            print(f"    [ERROR] timing raised: {type(ex).__name__}: {ex}")

    print("\n" + "=" * 96)
    print("done.")


if __name__ == "__main__":
    main()
