# DSpark runtime port notes

Re-port of the fleet-fork sglang patches (originally built against pin
`28b095c`) onto the upstream DSpark tree `692c5f7d` (commit "DSpark:
confidence-scheduled speculative decoding with semi-autoregressive drafting"),
branch `dspark-port` in `/nfs/FM/chenshuailin/scratch_sglang_692c5f7d`.

Cross-reference for intended final state: node64 fleet tree at
`/sgl-workspace/sglang` (READ ONLY; carries fleet base + all fleet patches, so
it differs from upstream on pre-existing fleet-only fixes — see drift notes).

Each artifact below is `git format-patch -1 --stdout` of its port commit.

---

## 1. `0002-port-fp32_partial_merge-re-port-from-fleet-artifact-.patch`  (commit cf7b11c)

Source: `scripts/dsv4/patches/sglang_dsv4_fp32_partial_merge.patch`
Target: `python/sglang/srt/layers/attention/flash_mla_sm120_triton.py`

VERDICT: **FULLY PORTED** (all 6 hunks). One drift adaptation at the conflict
site; nothing dropped.

The fp32 partial merge is the sole DS-V4 path and is fixed on; there is no
runtime environment gate.

Hunk-by-hunk vs the original patch:

| # | Original hunk | Ported? | Notes |
|---|---|---|---|
| 1 | fixed `_FP32_PARTIAL_MERGE = True` contract + module comment | yes | verbatim |
| 2 | store-output: `acc.to(tl.bfloat16)` -> `o_dt = O_ptr.dtype.element_ty; acc.to(o_dt)` + comment | yes | verbatim |
| 3 | `_run_triton_sparse_decode` gains `out_dtype: torch.dtype = torch.bfloat16` + docstring | yes | verbatim |
| 4 | `out = torch.zeros(..., dtype=torch.bfloat16)` -> `dtype=out_dtype` | yes* | **DRIFT** — see below |
| 5 | new `_merge_partials_and_sink_fp32(...)` fn | yes | verbatim |
| 6 | fixed fp32 branch in `flash_mla_sparse_decode_triton` | yes | verbatim |

DRIFT (hunk 4, the single conflict, ~orig line 272): the fleet base already
carried a SEPARATE sizing fix — `out = torch.zeros(B, H, _D, ...)` where `_D`
is the kernel's fixed latent (nope 448 + rope 64 = 512), plus a 3-line comment
explaining why `_D` and not q's `D_qk`. Upstream `692c5f7d` does NOT have that
fix: it sizes by `q.shape[-1]` -> `out = torch.zeros(B, H, D, ...)`, no comment.
That `_D` sizing is a DIFFERENT fleet patch (it appears only as CONTEXT in this
diff, not as one of its `+`/`-` lines), so it is intentionally NOT ported here.
Only the dtype thread (`torch.bfloat16 -> out_dtype`) was applied, on upstream's
`D` line. The new fp32 path therefore stays dimensionally consistent with
upstream's own legacy path (both size by `q.shape[-1]`). If the `_D` sizing fix
is wanted on upstream, it is a separate port.

The retained helper default remains bf16 for the internal legacy A/B oracle,
while production always selects the fp32 partial-merge branch. This preserves
the original patch's byte-identity claim for that oracle.

---
