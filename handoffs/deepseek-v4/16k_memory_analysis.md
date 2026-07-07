# ctx-16k train-OOM: theoretical memory analysis

Debugging why a 24×H20 (95 GB each) job OOMs in the train backward at ctx 16384,
which should not happen. Fast loop: `--debug-train-only` replaying saved 16k
rollout data (`/nfs/FM/csl_v4_dbg16k/rollout_0.pt`), ~3 min/cycle.

## Established facts (measured)

- **Rest state (weights+optimizer, fp8 experts): 16.6 GB/GPU** (freed-GB probe;
  fp8 quantize frees 11.3 GB — retention hypothesis disproven).
- OOM is on the **last PP stage** (node69, stage2 = output layer + logits), in the
  **V4DecoderLayer recompute during the backward** (surfaced as
  `SystemError: V4DecoderLayer.forward returned NULL` — a CUDA/segment alloc NULL).
- **gbs=32 also OOMs** → per-microbatch peak, not total-microbatch accumulation.
- Attention is the **A1 flash kernel** (flash fwd + hand-written flash bwd) — no
  `[S,S]` materialization. mHC `comb` is `[hc,hc]=[4,4]` per token — no `[S,S]`.

## Why V4 activations are large but BOUNDED (config-driven)

| tensor | shape | bf16 | vs a normal LLM |
| --- | --- | ---: | --- |
| mHC hidden stream | `[1,16384,4,4096]` | 0.54 GB | **4×** (`hc_mult=4`) |
| attention q / attn_out | `[1,64,16384,512]` | 1.07 GB ea | **8×** (`head_dim=512`) |
| last-stage logits | `[1,16384,129280]` | 4.24 GB | `vocab=129280` |

## Expected last-stage peak (act-ckpt, 1F1B last stage holds ~1–2 microbatches)

- STORED layer inputs (act-ckpt saves each layer's input hc-stream): 14 × 0.54 = **7.5 GB**
- ONE layer recompute transient (attn q+out ~2.1 + mHC ~2 + MoE ~2): **~6 GB**
- logits + loss: **~8 GB**
- rest state: **16.6 GB**
- **Expected total ≈ 38–46 GB — comfortably under 95 GB.**

## The discrepancy ⇒ bug

OBSERVED OOM at **~93 GB** ⇒ **~50 GB unexplained.** Two leading causes:

1. **Recompute not freeing per layer.** If all 14 layers' recompute transients
   coexist instead of freeing after each layer's backward: 14 × 6 = **85 GB** →
   exactly an OOM. (torch.utils.checkpoint should free per-segment; a custom
   autograd Function or the mHC-oracle `detach` inside the checkpoint could defeat
   that.)
2. **`expandable_segments:True` segment-growth failure.** The OOM surfaced as
   "returned NULL without setting an exception" — the signature of an expandable
   segment failing to grow contiguously *despite free memory elsewhere*. This can
   be a partly SPURIOUS OOM. Testing `expandable_segments:False` (classic
   caching allocator) — if it trains, this was the cause.

## ROOT CAUSE CONFIRMED (V4_MEM_PROBE per-layer alloc)

`expandable_segments:False` gave a clean OOM: **87.2 GB ALLOCATED** (not
reserved) → genuine allocation, NOT fragmentation. Per-layer cuda-allocated
probe (V4_MEM_PROBE=1) is decisive:

- **Forward (grad=False): FLAT ~17 GB** across all layers → act-checkpoint IS
  storing only inputs, working correctly.
- **Backward recompute (grad=True): MONOTONIC CLIMB** — layer 42→20.7 GB,
  34→48.3 GB, 31→68.8 GB. As the stage backward walks its layers (42→29), each
  layer's recompute transient (~4.4 GB = attn q/out 2.1 + mHC hidden ~2) is
  HELD instead of freed → ~4.4 GB/layer × 14 ≈ 62 GB accumulation → OOM.

This is `torch.utils.checkpoint(use_reentrant=False)` failing to free
per-segment recompute across the stage backward (pytorch#147449; the
non-reentrant variant is documented to not work well with DDP/FSDP — we use
Megatron DDP).

FIX under test: `use_reentrant=True` (V4_ACT_CKPT_REENTRANT=1, now default) — the
reentrant variant runs a nested backward per segment and frees each layer's
recompute immediately. Expected: backward-recompute alloc stays FLAT, 16k fits.
