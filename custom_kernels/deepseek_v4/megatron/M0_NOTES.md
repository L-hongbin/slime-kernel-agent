# M0 — V4-Flash mcore scaffold + 3-kernel wiring

**Status: PASS.** A self-contained slime-local V4-Flash model module builds from a
tiny config and runs a no-NaN forward on GPU 0, with all three validated tilelang
kernels (A1 attention / B2 compression / B1 mHC) live in the path. This is
scaffolding + a no-NaN forward, **not** HF parity (that is M1).

## Where the code lives

All under `custom_kernels/deepseek_v4/megatron/` (slime-local; see "why custom block" below):

| File | Contents |
|---|---|
| `_kernels.py` | Import shim: loads the three sibling kernel dirs (`../attention`, `../compression`, `../mhc`) by file path with their own dir on `sys.path` (they use sibling-relative imports), re-exports `v4flash_attention`, `hca_compress`, `csa_compress`, `hyper_connection`, `hyper_connection_sglang`. |
| `rope.py` | `V4RMSNorm`, `V4UnweightedRMSNorm`, and a 1:1 re-export of HF's `DeepseekV4RotaryEmbedding` / `apply_rotary_pos_emb` / `rotate_half` (interleaved partial RoPE — kept torch-side, group C). |
| `compressor.py` | `V4HCACompressor`, `V4CSACompressor`: own kv/gate proj + position_bias + kv_norm, call B2, apply RoPE at `w*compress_rate`, return `[B,1,T,head_dim]`. |
| `attention.py` | `V4Attention`: q/kv proj + norms + partial RoPE → A1 → conjugate `-sin` RoPE → grouped `o_a_proj` (HF `DeepseekV4GroupedLinear` bmm) → `o_b_proj`. |
| `decoder.py` | `V4HyperConnection` (owns mHC fn/base/scale, calls B1), `V4DecoderLayer` (two HC sites + `post·out + comb.T@stream` mix), `V4Model` (embed → layers carrying `[B,S,hc_mult,hidden]` → `hc_head` → norm). |
| `m0_smoke.py` | Tiny-config builder, scaffold weight init, the no-NaN forward gate. |

## Module tree

```
V4Model
├── embed_tokens (nn.Embedding)
├── rotary_emb (DeepseekV4RotaryEmbedding)         # builds {"main","compress"} cos/sin
├── layers[i] = V4DecoderLayer
│   ├── attn_hc / ffn_hc = V4HyperConnection        # B1 mHC  (fp32 params)
│   ├── input_layernorm / post_attention_layernorm = V4RMSNorm
│   ├── self_attn = V4Attention
│   │   ├── q_a_proj, q_a_norm, q_b_proj, q_b_norm, kv_proj, kv_norm
│   │   ├── compressor = V4{CSA,HCA}Compressor | None    # B2  (CSA also holds HF Indexer, unused in M0)
│   │   ├── sinks (per-head)
│   │   └── o_a_proj (DeepseekV4GroupedLinear), o_b_proj
│   └── mlp = DeepseekV4SparseMoeBlock (HF)          # TopK/Hash router + grouped experts + shared expert
├── hc_head (DeepseekV4HyperHead, HF)               # final hc-stream collapse
└── norm (V4RMSNorm)
```

## How each kernel is wired (kernel vs torch)

| Kernel | Call site | Kernel does | Torch does (around it) |
|---|---|---|---|
| **A1** `v4flash_attention(q, k_raw, k_comp, sinks, window, m)` | `V4Attention.forward` | MQA hd=512 flash attn, per-head sink (denom-only), **structural** sliding-window + causal-threshold compressed masks, fp32 softmax | q/kv proj+norms, interleaved partial RoPE on q/kv, output-side conjugate `-sin` RoPE, grouped `o_a`/`o_b` proj |
| **B2** `hca_compress` / `csa_compress` | `V4{HCA,CSA}Compressor.forward` | windowed gated-softmax pool + RMSNorm; CSA folds the Ca/Cb overlap + position_bias in-kernel | kv/gate proj, RoPE on compressed entries at `w*m`, `.unsqueeze(1)` to `[B,1,T,D]`, concat into A1's `k_comp` |
| **B1** `hyper_connection(streams, fn, base, scale)` | `V4HyperConnection.forward` | RMSNorm-rescale + fn-GEMM → sigmoid pre/post + softmax→20-iter Sinkhorn comb → stream-collapse | the `post·out + comb.T@stream` residual mix (two sites/layer), in `V4DecoderLayer` |

Group-C ops kept in torch (cheap / trivial / reuse Megatron): interleaved partial
RoPE, grouped `o_a_proj` (block-diagonal bmm via HF `DeepseekV4GroupedLinear`),
RMSNorms, MoE (routers + grouped experts + shared expert + clamped SwiGLU),
`hc_head`. The Lightning Indexer top-k is **dropped** (dense-over-compressed);
at tiny seq it is a no-op so A1's structural mask == HF `block_bias` exactly.

## Smoke result (`CUDA_VISIBLE_DEVICES=0 python -m custom_kernels.deepseek_v4.megatron.m0_smoke`)

Tiny config: `hidden=4096, head_dim=512, hc_mult=4, sinkhorn=20` (all FIXED);
shrunk `layers=3, heads=8, experts=8, vocab=512`; `layer_types =
[sliding, CSA, HCA]`; `mlp_layer_types = [hash_moe, hash_moe, moe]`; `B=2, S=128`.

```
[fwd] out shape = (2, 128, 4096) dtype=torch.bfloat16
[fwd] any NaN=False  any Inf=False  min=-4.88 max=4.66 mean=-0.0015 std=1.0000
M0 SMOKE PASS
```

Manual verification beyond the assert (instrumented run): the three kernels fire
exactly as designed — A1×3 (layer0 `k_comp=None`, layer1 `k_comp=[2,1,32,512]`,
layer2 `k_comp=[2,1,1,512]`), CSA×1→32 entries, HCA×1→1 entry, B1×6 (2 sites×3
layers). MoE output is non-trivial at all three layers (std≈0.39, nonzero≈1.0)
covering both hash_moe and moe routers.

## Things that fought the design

1. **Custom block vs mcore `TransformerBlock` (the friction codex flagged).** Not
   reused. mcore's `get_gpt_decoder_block_spec` / `TransformerLayer` carry a single
   `[S,B,H]` stream with ordinary BDA residuals and never pass `input_ids` to the
   MLP. V4 needs (a) a `[B,S,hc_mult,hidden]` stack threaded through every layer,
   (b) two mHC mix sites per layer with `comb` consumed transposed, (c) `input_ids`
   at the hash-MoE router, (d) `hc_head` collapse only before the final norm. These
   are structural, not configurable, so M0 is a from-scratch `V4Model`/`V4DecoderLayer`
   that mirrors HF math at every seam. **Consequence for later milestones:** this
   block is not yet a Megatron `ModuleSpec`/PP-aware block — M1 parity targets this
   torch module directly; the mcore `ModuleSpec` + `deepseek_v4_bridge.py` fork +
   PP/EP wiring is deferred (M5 in the revised plan handles the full bridge).

2. **HF-owned submodules build with `torch.empty`.** `DeepseekV4SparseMoeBlock`
   (router `weight`, `Experts.gate_up_proj`/`down_proj`), `DeepseekV4Indexer.position_bias`,
   and `DeepseekV4HyperHead` get sane values only from HF's `_init_weights`, which
   we don't run (we instantiate modules directly, not via `from_config`). Building
   the model without re-initialising those gave NaN out (uninit router weight). Fix:
   `m0_smoke.init_weights` reproduces HF's `_init_weights` for exactly those modules
   (normal weights, zero bias/position_bias, ones scale, random valid `tid2eid`).
   Our own V4* modules self-init (`sinks`/`position_bias`=0, mHC `fn~N(0,std)`,
   `base`=0, `scale`=1, norms=1).

3. **mHC params must stay fp32.** B1 expects fn/base/scale in fp32 (HF
   `_keep_in_fp32_modules`). `V4HyperConnection.forward` passes `.float()` copies; the
   smoke also `.float()`s the `attn_hc`/`ffn_hc`/`hc_head` modules after the bf16 cast.

4. **B1 path choice.** Default is the dtype-agnostic hybrid `hyper_connection` (no
   sglang coupling for the scaffold). `hyper_connection_sglang` (bf16-only, sglang
   `mhc_pre` forward) is verified importable and is the intended production fwd for
   later milestones; switch via `decoder._HC`.

## Deviations / blockers

- None blocking. A benign transformers warning prints about
  `rope_parameters['compress']['attention_factor']` on a `rope_type='default'`
  (V4 forces `attention_factor=1.0` to suppress the YaRN mscale; harmless) and about
  `ExpertsInterface ... _experts_implementation=None` (HF Experts used standalone →
  reference loop; correct for M0).
