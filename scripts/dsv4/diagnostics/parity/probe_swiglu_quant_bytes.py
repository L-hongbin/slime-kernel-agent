#!/usr/bin/env python
"""Byte-level probe: routed-expert SwiGLU -> FP8 down-input quantization,
trainer vs serving (Target 1 of the train<->rollout kernel alignment plan).

Feeds IDENTICAL bf16 gate_up tensors through:

  (1) trainer path  : _apply_gate replica (bf16 clamp / torch silu / bf16 mul)
                      + deep_gemm.per_token_cast_to_fp8(use_ue8m0=False)
  (2) serving masked: sglang JIT silu_and_mul_masked_post_quant
                      ([E, max_m, 2I] + masked_m layout, the DeepEP low-latency
                      decode branch — the operative production kernel)
  (3) serving contig: sglang JIT silu_and_mul_contig_post_quant (prefill branch)
  (4) fp32 oracle   : fp32 clamp+silu+mul, group-128 quant in fp32
                      (amax.clamp(min=1e-4)/448 — per_token_cast_to_fp8 formula),
                      NO bf16 intermediate
  (4b) fp32 oracle with the serving amax floor (1e-10) instead of 1e-4
  (5) bf16-materialized emulation: like (4) but the activation is rounded to
                      bf16 once before quantization (amax on bf16 values)

and reports FP8 byte-diff rates + scale diffs per input battery.

Run (node53 dev container):
  CUDA_VISIBLE_DEVICES=0 python scripts/dsv4/diagnostics/parity/probe_swiglu_quant_bytes.py

Self-contained; small tensors; single GPU.
"""

import argparse
import sys

sys.path.insert(0, "/sgl-workspace/sglang/python")

import torch
import torch.nn.functional as F

LIMIT = 10.0  # config.swiglu_limit (serving asserts == 10)
GROUP = 128
FP8_MAX = 448.0


# --------------------------------------------------------------------------- #
# Path (1): trainer replica
# --------------------------------------------------------------------------- #
def _clamp_trainer(x, min=None, max=None):
    """Copy of mcore_model._clamp (straight-through clamp). Value-identical to
    torch.clamp: x - x.detach() == 0 exactly."""
    return x.clamp(min=min, max=max).detach() + (x - x.detach())


def trainer_apply_gate(gate_up_bf16):
    """Exact replica of V4GroupedExperts._apply_gate (mcore_model.py:470-474):
    chunk, clamp gate at max=+10 (one-sided), clamp up to [-10,10],
    ACT2FN['silu'](gate) * up — all on bf16 tensors."""
    gate, up = gate_up_bf16.chunk(2, dim=-1)
    gate = _clamp_trainer(gate, max=LIMIT)
    up = _clamp_trainer(up, min=-LIMIT, max=LIMIT)
    return F.silu(gate) * up  # ACT2FN["silu"] is nn.SiLU -> F.silu


def trainer_quant(gate_up_bf16):
    from deep_gemm import per_token_cast_to_fp8

    act = trainer_apply_gate(gate_up_bf16)
    q, s = per_token_cast_to_fp8(act, use_ue8m0=False)
    return q, s, act


# --------------------------------------------------------------------------- #
# Path (2): serving masked JIT kernel
# --------------------------------------------------------------------------- #
def serving_masked_quant(gate_up_bf16, split=(128,)):
    """Run silu_and_mul_masked_post_quant on the same rows, laid out as
    [E, max_m, 2I] + masked_m (DeepEP low-latency masked layout), then gather
    the valid rows back to [T, I]."""
    from sglang.jit_kernel.dsv4 import silu_and_mul_masked_post_quant

    T, twoI = gate_up_bf16.shape
    intermediate = twoI // 2
    G = intermediate // GROUP
    counts = list(split)
    assert sum(counts) == T
    E = len(counts)
    max_m = max(max(counts), 1)
    dev = gate_up_bf16.device

    inp = torch.zeros(E, max_m, twoI, device=dev, dtype=torch.bfloat16)
    # poison the padding region so "reads beyond masked_m" would be visible
    inp += 123.0
    off = 0
    for e, c in enumerate(counts):
        inp[e, :c] = gate_up_bf16[off : off + c]
        off += c
    masked_m = torch.tensor(counts, device=dev, dtype=torch.int32)

    out = torch.full((E, max_m, intermediate), 0x7F, device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
    scale = torch.full((E, max_m, G), -1.0, device=dev, dtype=torch.float32)
    # topk only sizes the grid (needs max_m*topk >= sum(masked_m)); production=8
    topk = (sum(counts) + max_m - 1) // max_m + 1
    silu_and_mul_masked_post_quant(
        inp,
        out,
        scale,
        GROUP,
        masked_m,
        scale_ue8m0=False,
        topk=max(topk, 8),
        transposed=False,
        swiglu_limit=LIMIT,
        swizzle=False,
    )
    torch.cuda.synchronize()

    q = torch.empty(T, intermediate, device=dev, dtype=torch.float8_e4m3fn)
    s = torch.empty(T, G, device=dev, dtype=torch.float32)
    off = 0
    for e, c in enumerate(counts):
        q[off : off + c] = out[e, :c]
        s[off : off + c] = scale[e, :c]
        off += c
    return q, s, (out, scale)


# --------------------------------------------------------------------------- #
# Path (3): serving contiguous JIT kernel
# --------------------------------------------------------------------------- #
def serving_contig_quant(gate_up_bf16):
    from sglang.jit_kernel.dsv4 import silu_and_mul_contig_post_quant

    T, twoI = gate_up_bf16.shape
    intermediate = twoI // 2
    G = intermediate // GROUP
    dev = gate_up_bf16.device
    out = torch.empty(T, intermediate, device=dev, dtype=torch.float8_e4m3fn)
    scale = torch.empty(T, G, device=dev, dtype=torch.float32)
    silu_and_mul_contig_post_quant(
        gate_up_bf16,
        out,
        scale,
        GROUP,
        scale_ue8m0=False,
        transposed=False,
        swiglu_limit=LIMIT,
        swizzle=False,
    )
    torch.cuda.synchronize()
    return out, scale


# --------------------------------------------------------------------------- #
# Paths (4)/(4b)/(5): torch oracles
# --------------------------------------------------------------------------- #
def oracle_act_fp32(gate_up_bf16):
    """fp32 clamp + silu + mul, no bf16 intermediate. Clamp on the bf16 values
    is exact in either precision (10 is bf16-representable)."""
    x = gate_up_bf16.float()
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(max=LIMIT)
    up = up.clamp(min=-LIMIT, max=LIMIT)
    return gate * torch.sigmoid(gate) * up  # precise silu, fp32 product


def group_quant_fp32(act_f32, amax_floor):
    """per_token_cast_to_fp8's formula on an fp32 activation:
    amax per group-128, clamp(min=floor), sf=amax/448, x*(1/sf) -> e4m3."""
    T, intermediate = act_f32.shape
    v = act_f32.view(T, intermediate // GROUP, GROUP)
    amax = v.abs().amax(dim=2).clamp(min=amax_floor)
    sf = amax / FP8_MAX
    q = (v * (1.0 / sf).unsqueeze(2)).to(torch.float8_e4m3fn).view(T, intermediate)
    return q, sf


def oracle_fused_fp32(gate_up_bf16, amax_floor=1e-4):
    act = oracle_act_fp32(gate_up_bf16)
    q, s = group_quant_fp32(act, amax_floor)
    return q, s


def oracle_bf16_materialized(gate_up_bf16, amax_floor=1e-4):
    """Like the fused fp32 oracle but the activation is stored to bf16 once
    before quantization (amax over the bf16 values, quant of bf16 values)."""
    from deep_gemm import per_token_cast_to_fp8

    act_bf16 = oracle_act_fp32(gate_up_bf16).to(torch.bfloat16)
    assert amax_floor == 1e-4  # per_token_cast_to_fp8 hardcodes clamp(1e-4)
    q, s = per_token_cast_to_fp8(act_bf16, use_ue8m0=False)
    return q, s


# --------------------------------------------------------------------------- #
# Input batteries
# --------------------------------------------------------------------------- #
def bf16_ulp(x):
    t = torch.tensor(x, dtype=torch.bfloat16)
    return (torch.nextafter(t, torch.tensor(float("inf"), dtype=torch.bfloat16)) - t).item()


def make_batteries(dev, intermediate=2048, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    twoI = 2 * intermediate
    batteries = {}

    def to_dev(x):
        return x.to(dtype=torch.bfloat16, device=dev)

    # A) ordinary randn scales (T=160: one full + one partial 128-row block)
    batteries["randn"] = to_dev(torch.randn(160, twoI, generator=g))

    # B) clamp edge: values within 1-2 ulp of +-10 sprinkled over randn rows
    u = bf16_ulp(10.0)
    edge_vals = torch.tensor([LIMIT - 2 * u, LIMIT - u, LIMIT, LIMIT + u, LIMIT + 2 * u, 12.0, 64.0])
    x = torch.randn(160, twoI, generator=g) * 2.0
    for r in range(x.shape[0]):
        idx = torch.randperm(intermediate, generator=g)[:64]
        signs = torch.where(torch.rand(64, generator=g) < 0.5, -1.0, 1.0)
        # gate half: clamp is one-sided (max=+10) -> also plant negatives
        x[r, idx] = edge_vals[torch.randint(len(edge_vals), (64,), generator=g)] * signs
        idx2 = torch.randperm(intermediate, generator=g)[:64]
        signs2 = torch.where(torch.rand(64, generator=g) < 0.5, -1.0, 1.0)
        x[r, intermediate + idx2] = edge_vals[torch.randint(len(edge_vals), (64,), generator=g)] * signs2
    batteries["clamp_edge"] = to_dev(x)

    # C) near-tied group amax: two elements per group with ~equal |silu*up|
    T = 128
    x = torch.randn(T, twoI, generator=g) * 0.05
    for r in range(T):
        for grp in range(intermediate // GROUP):
            p0, p1 = torch.randperm(GROUP, generator=g)[:2] + grp * GROUP
            g0 = 2.0 + 4.0 * torch.rand((), generator=g)
            g1 = 2.0 + 4.0 * torch.rand((), generator=g)
            prod = 0.5 + 3.5 * torch.rand((), generator=g)
            rel = 1.0 + (torch.rand((), generator=g) - 0.5) * 2e-3  # within ~1e-3
            x[r, p0] = g0
            x[r, intermediate + p0] = prod / (g0 * torch.sigmoid(g0))
            x[r, p1] = g1
            x[r, intermediate + p1] = prod * rel / (g1 * torch.sigmoid(g1))
    batteries["amax_tie"] = to_dev(x)

    # D) tiny-magnitude groups: amax near the trainer 1e-4 clamp and below it
    x = torch.randn(96, twoI, generator=g)
    x[:32] *= 1e-2  # products ~1e-4 (straddle the clamp)
    x[32:64] *= 1e-4  # products ~1e-8 (below trainer clamp, above serving 1e-10)
    x[64:] *= 3e-3  # products ~1e-5
    batteries["tiny_groups"] = to_dev(x)

    # E) mixed per-group magnitudes (log-uniform 1e-6..1e1)
    x = torch.randn(160, twoI, generator=g)
    mag = 10 ** (torch.rand(160, twoI // GROUP, 1, generator=g) * 7 - 6)
    x = (x.view(160, twoI // GROUP, GROUP) * mag).view(160, twoI)
    batteries["mixed_mag"] = to_dev(x)

    return batteries


# --------------------------------------------------------------------------- #
# Comparison helpers
# --------------------------------------------------------------------------- #
def byte_diff(qa, qb):
    a = qa.view(torch.uint8)
    b = qb.view(torch.uint8)
    n = a.numel()
    d = a != b
    rate = d.float().mean().item()
    # bin distance among differing bytes (dequant-space would need scales;
    # report raw code distance of e4m3 codes as a locality hint)
    if d.any():
        fa = qa.float()[d]
        fb = qb.float()[d]
        with torch.no_grad():
            rel = ((fa - fb).abs() / fa.abs().clamp(min=1e-30)).max().item()
    else:
        rel = 0.0
    return rate, int(d.sum().item()), n, rel


def scale_diff(sa, sb):
    exact = (sa == sb).float().mean().item()
    rel = ((sa - sb).abs() / sa.abs().clamp(min=1e-30)).max().item()
    return exact, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--intermediate", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = args.device
    intermediate = args.intermediate

    torch.manual_seed(args.seed)
    batteries = make_batteries(dev, intermediate=intermediate, seed=args.seed)

    # sanity: _clamp replica is value-identical to torch.clamp
    zz = torch.randn(64, 64, device=dev, dtype=torch.bfloat16) * 8
    assert torch.equal(_clamp_trainer(zz, max=LIMIT), zz.clamp(max=LIMIT))

    print(f"# SwiGLU->FP8 quant byte probe   I={intermediate} group={GROUP} limit={LIMIT}")
    print(f"# device={torch.cuda.get_device_name(dev)}  torch={torch.__version__}")
    print()

    pair_names = [
        ("(1)trainer", "(2)masked"),
        ("(2)masked", "(4)fp32-oracle"),
        ("(2)masked", "(4b)fp32-oracle-floor1e-10"),
        ("(2)masked", "(5)bf16-emul"),
        ("(1)trainer", "(5)bf16-emul"),
        ("(1)trainer", "(4)fp32-oracle"),
        ("(2)masked", "(3)contig"),
    ]

    all_results = {}
    for name, x in batteries.items():
        T = x.shape[0]
        # masked layout: one full expert block + one partial + one empty-ish
        split = (128, T - 128) if T > 128 else (T,)
        q1, s1, act1 = trainer_quant(x)
        q2, s2, raw2 = serving_masked_quant(x, split=split)
        q3, s3 = serving_contig_quant(x)
        q4, s4 = oracle_fused_fp32(x, amax_floor=1e-4)
        q4b, s4b = oracle_fused_fp32(x, amax_floor=1e-10)
        q5, s5 = oracle_bf16_materialized(x)

        qs = {
            "(1)trainer": (q1, s1),
            "(2)masked": (q2, s2),
            "(3)contig": (q3, s3),
            "(4)fp32-oracle": (q4, s4),
            "(4b)fp32-oracle-floor1e-10": (q4b, s4b),
            "(5)bf16-emul": (q5, s5),
        }

        print(f"## battery: {name}  T={T} split={split}")
        res = {}
        for a, b in pair_names:
            qa, sa = qs[a]
            qb, sb = qs[b]
            rate, nd, n, rel = byte_diff(qa, qb)
            sx, srel = scale_diff(sa, sb)
            res[(a, b)] = (rate, nd, n, rel, sx, srel)
            print(
                f"  {a:>10s} vs {b:<28s} byte-diff {rate*100:8.4f}% ({nd}/{n})"
                f"  maxrel(diff bytes) {rel:9.3e} | scale exact {sx*100:7.3f}%"
                f" maxrel {srel:9.3e}"
            )
        all_results[name] = res
        print()

    # scale-layout record
    x = batteries["randn"]
    from deep_gemm import get_mn_major_tma_aligned_tensor

    q1, s1, _ = trainer_quant(x)
    s1_tma = get_mn_major_tma_aligned_tensor(s1)
    _, _, (out_m, scale_m) = serving_masked_quant(x, split=(128, x.shape[0] - 128))
    print("## scale layouts")
    print(f"  trainer per_token_cast scale : shape {tuple(s1.shape)} strides {s1.stride()} dtype {s1.dtype}")
    print(
        f"  trainer after TMA-align      : shape {tuple(s1_tma.shape)} strides {s1_tma.stride()} dtype {s1_tma.dtype}"
    )
    print(
        f"  serving masked scale         : shape {tuple(scale_m.shape)} strides {scale_m.stride()} dtype {scale_m.dtype}"
    )
    print(f"  serving masked fp8 out       : shape {tuple(out_m.shape)} strides {out_m.stride()} dtype {out_m.dtype}")
    # serving also TMA-aligns before the down GEMM (moe_runner/deep_gemm.py:456-459)
    s_m_tma = get_mn_major_tma_aligned_tensor(scale_m)
    print(
        f"  serving masked after TMA     : shape {tuple(s_m_tma.shape)} strides {s_m_tma.stride()} dtype {s_m_tma.dtype}"
    )
    print()

    # verdict
    print("## verdict")
    keys = ["randn", "clamp_edge", "amax_tie", "mixed_mag"]
    d24 = sum(all_results[k][("(2)masked", "(4b)fp32-oracle-floor1e-10")][0] for k in keys) / len(keys)
    d25 = sum(all_results[k][("(2)masked", "(5)bf16-emul")][0] for k in keys) / len(keys)
    d12 = sum(all_results[k][("(1)trainer", "(2)masked")][0] for k in keys) / len(keys)
    print(f"  mean byte-diff over {{{','.join(keys)}}}:")
    print(f"    (2)masked vs (4b)fused-fp32-oracle : {d24*100:.4f}%")
    print(f"    (2)masked vs (5)bf16-materialized  : {d25*100:.4f}%")
    print(f"    (1)trainer vs (2)masked            : {d12*100:.4f}%")
    if d24 < d25:
        print("  => serving masked kernel behaves like the FUSED FP32 oracle (single")
        print("     rounding, no bf16 store). Trainer's bf16-materialize-then-quantize")
        print("     is a real double-rounding divergence. GATE: PASS (Target 1 stands).")
    else:
        print("  => serving masked kernel matches the bf16-materialized emulation —")
        print("     serving also double-rounds. GATE: FAIL (Target 1 loses rationale).")


if __name__ == "__main__":
    main()
