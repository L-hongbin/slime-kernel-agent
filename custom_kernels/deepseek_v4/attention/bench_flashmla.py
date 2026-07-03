"""PRIMARY community-SOTA forward comparison: V4-Flash hd=512 attention vs the
REAL sglang V4 attention kernel, FlashMLA `flash_mla_sparse_fwd`.

`flash_mla_sparse_fwd` (`sgl_kernel.flash_mla`) is the kernel sglang's V4 backend
(`deepseek_v4_backend.py::_forward_prefill_sparse`) actually calls for prefill. It
is the production V4 attention. API:
    q:       [s_q, h_q, d_qk]  bf16          (d_qk = ckv 512 + rope 64 = 576)
    kv:      [s_kv, h_kv, d_qk] bf16         (h_kv = 1: shared latent KV / MQA)
    indices: [s_q, h_kv, topk]  int32        (per-query selected kv ids; -1/>=s_kv = invalid)
    sm_scale, d_v=512, attn_sink:[h_q] opt, topk_length:[s_q] int32 opt
returns (out[s_q,h_q,d_v], max_logits, lse). Forward / inference only.

This is run in TWO configs:
  (a) NATIVE sparse  (topk = 512, production setting): each query attends a fixed
      512-key budget (the indexer's job). Isolates nothing — it is the real cost.
  (b) DENSE causal   (topk grows to full causal length): query i attends keys 0..i.
      Removes the sparsity-algorithm advantage so the gap is closest to pure
      kernel efficiency (still confounded by MLA matrix-absorption, see caveats).

Apples-to-apples caveats (see RESULTS.md):
  * FlashMLA = MLA matrix-ABSORBED (works in latent space: QK over d_qk=576,
    PV over d_v=512 latent — algorithmically cheaper than materialized hd512),
    + top-k sparse + fp8/paged-capable, NO per-head sink contribution to values,
    NO backward (inference).
  * Ours = dense-over-compressed, dense hd=512 (QK and PV both over D=512),
    per-head sink, bf16. (Our kernel supports training fwd+bwd, but THIS bench
    times the FORWARD ONLY — both sides are forward-only; the reported TF/s are
    forward useful-matmul throughput, NOT fwd+bwd. Our fwd+bwd is ~3.4-4.1x the
    forward; see the torch.compile table in RESULTS.)
So the sparse-config gap is mostly the ALGORITHM (sparse + absorbed does far less
work); the dense-config gap is a directional "same workload size, different kernel"
number — NOT an identical-operation ratio (FlashMLA-dense is still the SPARSE kernel
`flash_mla_sparse_fwd` driven with full-causal indices, in MLA-absorbed latent space;
ours is standard hd512 K==V). sm_scale (1/sqrt(576)) is passed ONLY to FlashMLA; our
kernel uses its own internal 1/sqrt(512) (no scale leak). TF/s exclude softmax/exp/sink.

FLOP accounting (useful work, achieved bf16 TFLOP/s = flops/time):
  FlashMLA per (query, selected-key) per head: QK 2*d_qk + PV 2*d_v = 2*(576+512)=2176.
    total pairs = sum_i topk_length[i].
  Ours per (q,k) per head (dense causal): QK 2*D + PV 2*D = 4*D = 2048.
    total pairs = S*(S+1)/2.
"""

import sys
import torch

sys.path.insert(0, ".")
import kernel as K  # noqa: E402

from sgl_kernel.flash_mla import flash_mla_sparse_fwd  # noqa: E402

DEV = "cuda"
H_Q = 64
D_QK = 576  # ckv 512 + rope 64
D_V = 512  # latent value dim (kernel requires d_v == 512)
CKV, KPE = 512, 64
D_OURS = 512  # our dense head_dim
TOPK_SPARSE = 512  # production sparse budget


from _bench_common import cuda_time  # noqa: E402


def _round128(x):
    return ((x + 127) // 128) * 128


def build_indices(S, budget, dense):
    """indices [S,1,topk_alloc] int32 + topk_length [S] int32. The allocated topk
    dim must be a multiple of 128 (kernel asserts topk % (2*B_TOPK)==0, B_TOPK=64);
    extra slots are -1 (invalid). topk_length holds the real per-row valid count.
    dense:  query i attends keys 0..i (full causal), n = i+1.
    sparse: query i attends n = min(i+1, budget) most-recent causal keys
            (i-n+1 .. i) -- the realistic top-k/sliding budget. Cost tracks the
            COUNT per row, not which ids, so this is faithful for timing."""
    row = torch.arange(S, device=DEV)
    n = (row + 1) if dense else torch.clamp(row + 1, max=budget)  # [S] valid count
    topk_alloc = _round128(S if dense else budget)
    start = row + 1 - n  # max(0, i-budget+1)
    col = torch.arange(topk_alloc, device=DEV)
    valid = col[None, :] < n[:, None]  # [S, topk_alloc]
    idx = torch.where(valid, (start[:, None] + col[None, :]), torch.full_like(col[None, :], -1))
    idx = idx.to(torch.int32).unsqueeze(1).contiguous()  # [S,1,topk_alloc]
    return idx, n.to(torch.int32).contiguous()


def flashmla_fn(S, topk, dense):
    q = torch.randn(S, H_Q, D_QK, device=DEV, dtype=torch.bfloat16) * 0.3
    kv = torch.randn(S, 1, D_QK, device=DEV, dtype=torch.bfloat16) * 0.3
    sinks = torch.randn(H_Q, device=DEV, dtype=torch.float32)
    idx, lens = build_indices(S, topk, dense)
    sm_scale = 1.0 / (D_QK**0.5)
    pairs = int(lens.sum().item())

    def run():
        return flash_mla_sparse_fwd(
            q=q, kv=kv, indices=idx, sm_scale=sm_scale, d_v=D_V, attn_sink=sinks, topk_length=lens
        )

    return run, pairs


def ours_dense_fn(S):
    q = torch.randn(1, H_Q, S, D_OURS, device=DEV, dtype=torch.bfloat16) * 0.3
    kr = torch.randn(1, 1, S, D_OURS, device=DEV, dtype=torch.bfloat16) * 0.3
    sinks = torch.randn(H_Q, device=DEV, dtype=torch.float32)

    def run():
        return K.v4flash_attention(q, kr, None, sinks, S, 4)  # window>=S -> dense causal

    return run


def main():
    shapes = [2048, 4096, 8192, 16384, 32768]
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ---- (a) NATIVE SPARSE: FlashMLA topk=512 vs ours production sliding+comp ----
    # NOTE ours here uses the production sparse config (sliding W=128 + CSA m=4)
    # already reported in RESULTS bench.py; this table reports FlashMLA sparse only,
    # the per-S latency/throughput of the real production attention kernel.
    print("=== (a) FlashMLA NATIVE SPARSE (topk=512, production budget) ===")
    print(f"{'S':>7} | {'pairs':>12} | {'ms':>9} {'TF/s':>8}")
    print("-" * 48)
    fmla_sparse = {}
    for S in shapes:
        run, pairs = flashmla_fn(S, TOPK_SPARSE, dense=False)
        run()
        ms = cuda_time(run)
        flops = H_Q * (2 * (D_QK + D_V)) * pairs
        tf = flops / (ms * 1e-3) / 1e12
        fmla_sparse[S] = (ms, tf)
        print(f"{S:>7} | {pairs:>12} | {ms:>9.3f} {tf:>8.1f}")
        torch.cuda.empty_cache()

    # ---- (b) DENSE causal: FlashMLA dense vs ours dense -- kernel-efficiency bar ----
    print()
    print("=== (b) DENSE CAUSAL: FlashMLA vs ours (kernel-efficiency comparison) ===")
    print(f"{'S':>7} | {'FMLA ms':>9} {'FMLA TF/s':>10} | {'ours ms':>9} {'ours TF/s':>10} | {'ours/FMLA':>10}")
    print("-" * 78)
    for S in shapes:
        # FlashMLA dense
        try:
            run, pairs = flashmla_fn(S, S, dense=True)
            run()
            f_ms = cuda_time(run)
            f_flops = H_Q * (2 * (D_QK + D_V)) * pairs
            f_tf = f_flops / (f_ms * 1e-3) / 1e12
        except Exception as e:
            f_ms = f_tf = float("nan")
            print(f"{S:>7} | FlashMLA dense failed: {type(e).__name__}: {str(e)[:70]}")
            torch.cuda.empty_cache()
        # ours dense
        torch.cuda.empty_cache()
        try:
            run = ours_dense_fn(S)
            run()
            o_ms = cuda_time(run)
            o_pairs = S * (S + 1) / 2
            o_flops = H_Q * 4 * D_OURS * o_pairs
            o_tf = o_flops / (o_ms * 1e-3) / 1e12
        except Exception as e:
            o_ms = o_tf = float("nan")
            print(f"{S:>7} | ours dense failed: {type(e).__name__}: {str(e)[:70]}")
            torch.cuda.empty_cache()
            continue
        ratio = o_tf / f_tf if f_tf == f_tf else float("nan")
        print(f"{S:>7} | {f_ms:>9.3f} {f_tf:>10.1f} | {o_ms:>9.3f} {o_tf:>10.1f} | {ratio:>9.2f}x")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
