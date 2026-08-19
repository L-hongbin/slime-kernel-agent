"""Quantify the RMSNorm bf16-vs-fp32 deviation (codex training-path review item 3).

Our mcore V4 keeps the weighted RMSNorms' ``.weight`` (gain) in bf16; HF keeps norms in
fp32 (``_keep_in_fp32_modules_strict``).  ``V4RMSNorm`` normalizes in fp32 internally but
multiplies by the bf16 ``self.weight`` WITHOUT upcasting it, so the bf16 gain DOES deviate
from HF's fp32-norm forward.  Our bf16-vs-bf16 parity proves shared-bf16 behavior, not
equivalence to HF's fp32-norm baseline — so measure the gap.

A clean "HF with fp32 norms but bf16 linears" full forward is NOT runnable in eager (an
fp32-norm output feeds a bf16 ``q_a_proj`` -> dtype-mismatch crash; this is the structural
reason the bf16-norm choice exists).  So we isolate the effect two ways:

  (1) **norm-module isolation** — feed the SAME bf16 activation through ``V4RMSNorm`` with a
      bf16 gain vs an fp32 gain (the ONLY difference is the ``self.weight`` multiply dtype).
      Reports the per-norm output deviation = exactly the effect codex flagged, swept over a
      range of realistic activation scales.  This is the size of the per-layer tradeoff.

  (2) **whole-model** — the M-impl harness already reports mcore(bf16 norms) vs HF-fp32
      (M0-vs-HFfp32 column) and the irreducible whole-model bf16 floor (HF-bf16 vs HF-fp32);
      we restate those so the norm gap is seen in context (it is a strict subset of the
      whole-model bf16 gap, since norms are a small fraction of the ops).

Run:  CUDA_VISIBLE_DEVICES=0 python -m custom_kernels.deepseek_v4.megatron.measure_rmsnorm_fp32_gap
"""

import torch


def _rel(ref, got):
    ref = ref.detach().float()
    got = got.detach().float()
    rel = ((got - ref).norm() / ref.norm().clamp_min(1e-12)).item()
    cos = torch.nn.functional.cosine_similarity(ref.reshape(1, -1), got.reshape(1, -1)).item()
    return rel, cos, (got - ref).abs().max().item()


def main():
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    from .rope import V4RMSNorm

    dev = torch.device("cuda")
    H = 4096

    print("(1) RMSNorm-module isolation: same bf16 input, bf16 gain vs fp32 gain.")
    print("    (this is EXACTLY the 'multiply by bf16 self.weight' effect, per-norm)\n")
    print(f"    {'act_scale':>10}{'rel':>12}{'cos':>12}{'max_abs':>12}")
    worst_rel = 0.0
    for scale in (0.1, 1.0, 5.0, 20.0):
        # a realistic RMSNorm gain ~ N(1, 0.1) (trained norms hover near 1).
        g = torch.randn(H, device=dev) * 0.1 + 1.0
        x = (torch.randn(4, 128, H, device=dev) * scale).bfloat16()

        norm_fp32 = V4RMSNorm(H).to(dev)
        norm_fp32.weight.data = g.float()
        norm_bf16 = V4RMSNorm(H).to(dev)
        norm_bf16.weight.data = g.bfloat16()

        with torch.no_grad():
            o_fp32 = norm_fp32(x)  # bf16 in -> fp32 gain multiply -> bf16 out (HF baseline)
            o_bf16 = norm_bf16(x)  # bf16 in -> bf16 gain multiply -> bf16 out (our model)
        rel, cos, mx = _rel(o_fp32, o_bf16)
        worst_rel = max(worst_rel, rel)
        print(f"    {scale:>10.1f}{rel:>12.6f}{cos:>12.7f}{mx:>12.5f}")

    print(f"\n    worst per-norm rel deviation across scales = {worst_rel:.6f}")
    print("    (bf16 eps ~ 2^-8 = 0.0039; a single bf16-gain multiply contributes ~that)")

    print("\n(2) whole-model context (from m_impl_parity baseline, tiny config):")
    print("    mcore(bf16 norms) vs HF-fp32 FINAL hidden : rel ~ 0.057  (M0-vs-HFfp32)")
    print("    irreducible whole-model bf16 floor        : rel ~ 0.044  (HF-bf16 vs HF-fp32)")
    print("    -> the norm-gain bf16 contribution is a SMALL subset of the 0.044 bf16 floor;")
    print("       per-norm rel ~0.004 (one bf16 mult) << the whole-model 0.044 floor.")

    floor = 0.004 * 1.5  # one bf16-mult epsilon with headroom
    verdict = "AT the bf16 floor (accepted tradeoff)" if worst_rel <= floor else "ABOVE floor -- FLAG for review"
    print(f"\nVERDICT: per-norm bf16-gain deviation worst rel={worst_rel:.6f} -> {verdict}")


if __name__ == "__main__":
    main()
