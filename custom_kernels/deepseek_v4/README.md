# DeepSeek-V4-Flash custom kernels (tilelang)

V4-specific structures with **no existing Megatron implementation**, written as tilelang
kernels (hand-written forward + backward) for the V4-Flash LoRA-on-Megatron port.
Inventory + rationale: `handoffs/deepseek-v4/v4_kernel_inventory.md`.

## Confirmed environment (this container)

- **8× NVIDIA H20** (sm90 Hopper, 97 GB each, all free). `nvcc` 12.9, `torch` 2.11.0+cu129,
  `tilelang` 0.1.8. JIT compile→run→correctness verified. `ncu` + `nsys` available.
- Ground-truth math: `/usr/local/lib/python3.12/dist-packages/transformers/models/deepseek_v4/modeling_deepseek_v4.py`
  and `configuration_deepseek_v4.py`.

## V4-Flash dims (fix all kernel shapes)

hidden=4096, heads=64, **kv_heads=1 (shared-KV MQA, K≡V)**, **head_dim=512**,
qk_rope_head_dim=64 (interleaved partial RoPE on trailing 64), q_lora_rank=1024,
sliding_window=128, o_groups=8/o_lora_rank=1024, compress_rate CSA=4/HCA=128,
hc_mult=4/sinkhorn_iters=20/eps=1e-6, 256 experts/6 per tok, 43 layers. Train target: 32k ctx,
bf16 compute + fp32 accumulation. Base is **frozen** (LoRA only); kernels need grads w.r.t.
their tensor inputs so gradient propagates across layers, not w.r.t. frozen params.

## Kernels (one subagent each)

- `attention/` — **A1**: MQA + per-head sink + additive-bias flash attention, head_dim=512
  (no FA/SDPA kernel exists above hd 256). Serves all 43 layers via the additive bias.
- `mhc/` — **B1**: Manifold-Constrained Hyper-Connection (mHC) incl. 20-iter Sinkhorn-Knopp.
- `compression/` — **B2**: windowed gated-softmax compression pool + RMSNorm (CSA/HCA compressor core).

## Test + benchmark contract (every kernel)

1. **Forward correctness** vs the exact HF eager reference (port the `modeling_deepseek_v4.py`
   math as the torch reference). fp32 reference; bf16 kernel within tight tolerance on real shapes.
2. **Backward correctness** vs torch autograd through the reference: fp32 gradcheck on small shapes
   + bf16 grad tolerance on real shapes. Hand-written backward kernels (not autodiff).
3. **Efficiency — canonical evaluation policy (3 separate metrics, do not conflate):**
   - **Forward efficiency → compare ONLY against sglang** (the production SOTA kernel for the op):
     A1 → FlashMLA `flash_mla_sparse_fwd`; B1 → `srt.layers.mhc.mhc_pre`; B2 → `dsv4.compress_forward`.
     Forward-only (these are inference kernels). Report latency / TF/s / %peak and the ours/sglang ratio,
     with apples-to-apples caveats (MLA-absorbed/sparse vs our dense+sink+bias). **`torch.compile(reference)`
     is NOT the forward-efficiency metric** — it's a naive baseline; keep it only as background context.
   - **Backward efficiency → compare ONLY against our own forward** (sglang/torch.compile have no usable
     training backward): report bwd/fwd latency ratio + bwd %peak vs fwd %peak. Sanity target ~2–2.5×
     (flash ideal); a structural ~3.5× FLOP floor applies if the bwd does ~3.5× the fwd's GEMM work.
   - Keep the tilelang kernel where its forward is competitive with sglang and/or it enables a shape the
     baselines OOM on, and its backward efficiency is sensible.
4. **Codex (xhigh)** optimization pass + review after a satisfactory correct+benchmarked version;
   optionally `ncu` to locate bottlenecks. Record codex findings and what changed.

## Layout per kernel dir

`kernel.py` (tilelang fwd+bwd + autograd.Function wrapper), `reference.py` (exact torch reference),
`test_correctness.py` (fwd+bwd), `bench.py` (vs torch.compile), `RESULTS.md` (numbers + codex/ncu notes).
