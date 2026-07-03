"""Community-SOTA forward comparison for the V4-Flash hd=512 attention.

Baseline: FlashInfer 0.6.12 `BatchMLAPagedAttentionWrapper` — DeepSeek MLA paged
attention at head_dim_ckv=512 + head_dim_kpe=64, a single latent KV head
broadcast to all H query heads (i.e. shared-KV MQA, exactly our KV structure).
This is the closest available production-tuned hd=512 forward kernel on this box
(FlashMLA is NOT installed).

Caveats (documented in RESULTS.md):
  * FlashInfer MLA is a *weight-absorbed inference* kernel (paged decode/prefill);
    ours is a *training dense forward* with per-head sink + additive compressed
    bias. The math differs; this is a hardware-utilization ceiling, not identical
    work.
  * MLA does FULL causal attention over the latent KV; our production kernel is
    sliding-window + compressed (block-sparse). To remove the sparsity-pattern
    confound from the kernel-efficiency question we run OUR kernel in a matched
    DENSE-CAUSAL mode (window >= S, no compressed) for the head-to-head, and also
    report the production sparse numbers separately.

Metric: achieved bf16 TFLOP/s = useful_FLOPs / time. For dense causal at seqlen S,
pairs = S*(S+1)/2. FLOPs/pair: MLA = 2*(ckv+kpe) [QK] + 2*ckv [PV]; ours = 4*D.
"""

import sys
import torch

sys.path.insert(0, ".")
import kernel as K  # noqa: E402

DEV = "cuda"
H, D = 64, 512
CKV, KPE = 512, 64


from _bench_common import cuda_time  # noqa: E402


def build_flashinfer_mla():
    import flashinfer

    return flashinfer.mla.BatchMLAPagedAttentionWrapper(
        torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=DEV),
        backend="auto",
    )


def fi_mla_prefill_fn(wrapper, B, S):
    """Dense causal MLA prefill: B sequences, query len S, kv len S, page_size 1."""
    page_size = 1
    qo_indptr = torch.arange(0, B + 1, device=DEV, dtype=torch.int32) * S
    kv_lens = torch.full((B,), S, device=DEV, dtype=torch.int32)
    kv_indptr = torch.arange(0, B + 1, device=DEV, dtype=torch.int32) * S
    kv_indices = torch.arange(0, B * S, device=DEV, dtype=torch.int32)
    sm_scale = 1.0 / ((CKV + KPE) ** 0.5)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        H,
        CKV,
        KPE,
        page_size,
        True,
        sm_scale,
        torch.bfloat16,
        torch.bfloat16,
    )
    q_nope = torch.randn(B * S, H, CKV, dtype=torch.bfloat16, device=DEV) * 0.3
    q_pe = torch.randn(B * S, H, KPE, dtype=torch.bfloat16, device=DEV) * 0.3
    ckv = torch.randn(B * S, 1, CKV, dtype=torch.bfloat16, device=DEV) * 0.3
    kpe = torch.randn(B * S, 1, KPE, dtype=torch.bfloat16, device=DEV) * 0.3

    def run():
        return wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)

    return run


def main():
    B = 1
    shapes = [2048, 4096, 8192, 16384]
    wrapper = build_flashinfer_mla()

    print(f"{'S':>7} | {'FI-MLA ms':>10} {'FI TF/s':>8} | {'ours dense ms':>13} {'ours TF/s':>9} | {'util ratio':>10}")
    print("-" * 78)
    for S in shapes:
        pairs = S * (S + 1) / 2
        # --- FlashInfer MLA dense causal ---
        try:
            fi_run = fi_mla_prefill_fn(wrapper, B, S)
            fi_run()  # warm/compile
            fi_ms = cuda_time(fi_run)
            fi_flops = B * H * (2 * (CKV + KPE) + 2 * CKV) * pairs
            fi_tf = fi_flops / (fi_ms * 1e-3) / 1e12
        except Exception as e:
            print(f"{S:>7} | FlashInfer MLA failed: {type(e).__name__}: {str(e)[:80]}")
            fi_ms = fi_tf = float("nan")

        # --- our kernel, DENSE CAUSAL (window>=S, no compressed) ---
        torch.cuda.empty_cache()
        q = torch.randn(B, H, S, D, device=DEV, dtype=torch.bfloat16) * 0.3
        kr = torch.randn(B, 1, S, D, device=DEV, dtype=torch.bfloat16) * 0.3
        sinks = torch.randn(H, device=DEV, dtype=torch.float32)

        def k_run():
            return K.v4flash_attention(q, kr, None, sinks, S, 4)

        k_run()
        k_ms = cuda_time(k_run)
        k_flops = B * H * 4 * D * pairs
        k_tf = k_flops / (k_ms * 1e-3) / 1e12

        ratio = k_tf / fi_tf if fi_tf == fi_tf else float("nan")
        print(f"{S:>7} | {fi_ms:>10.3f} {fi_tf:>8.1f} | {k_ms:>13.3f} {k_tf:>9.1f} | {ratio:>9.2f}x")
        del q, kr
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
