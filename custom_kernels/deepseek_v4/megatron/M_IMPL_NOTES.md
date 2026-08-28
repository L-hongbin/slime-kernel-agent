# M-impl — V4-Flash as a Megatron-core model (TP=PP=EP=1)

**Status: PASS.** The mcore `V4LanguageModel` (a custom `LanguageModule`) FORWARD is
numerically equivalent to HF `DeepseekV4ForCausalLM` (eager) at TP=PP=EP=1 on the tiny
config — the hardened M1 parity, re-run against the mcore model, matches the M0 result
seam-for-seam (the mcore wrapping introduces zero numerical change, exactly as the
sharding contract predicts at 1/1/1). The model is slime-buildable (provider resolves,
forward returns fp32 `[1,T,V]` through slime's exact kwargs). LoRA + PP>1 + the EP>1
dispatcher are deferred.

Re-run the gate: `CUDA_VISIBLE_DEVICES=0 python -m custom_kernels.deepseek_v4.megatron.m_impl_parity`
Build/forward smoke: `CUDA_VISIBLE_DEVICES=0 python -m custom_kernels.deepseek_v4.megatron.mcore_smoke`

## Files

| File | Contents |
|---|---|
| `mcore_model.py` | `V4LanguageModel(LanguageModule)` + the custom MoE: `V4MoELayer`, `V4TopKRouter`, `V4HashRouter`, `V4GroupedExperts` (exact clamp+SwiGLU), `V4SharedExpertMLP`, and a `V4DecoderLayer` reusing the M1-validated `V4Attention`/compressor/`V4HyperConnection`. |
| `model_provider.py` | `v4_model_provider` (slime `--custom-model-provider-path` entry), `build_v4_mcore_model(hf_config)` (standalone), `make_transformer_config(hf_config)`. |
| `mcore_smoke.py` | 1-rank Megatron init + build + no-NaN fp32-`[B,S,V]` forward gate. |
| `m_impl_parity.py` | the hardened M1 parity adapted to the mcore model (HF→mcore weight remap + same 5-pass gate). |

## Module tree (what's mcore-parallel vs kernel vs reused)

```
V4LanguageModel(LanguageModule)                         # GPT-compatible surface slime drives
├── embedding = LanguageModelEmbedding                  # MCORE-PARALLEL (VocabParallel; TP=1 -> replicated)
├── rotary_emb = DeepseekV4RotaryEmbedding              # REUSED (HF; interleaved partial RoPE, group-C)
├── layers[i] = V4DecoderLayer
│   ├── attn_hc / ffn_hc = V4HyperConnection            # KERNEL B1 (mHC, fp32 params)  [REUSED from decoder.py]
│   ├── input_layernorm / post_attention_layernorm = V4RMSNorm
│   ├── self_attn = V4Attention                         # KERNEL A1 + REUSED torch seams (attention.py)
│   │   └── compressor = V4{CSA,HCA}Compressor | None    # KERNEL B2 (compressor.py)
│   └── mlp = V4MoELayer                                 # CUSTOM (mcore-structured; input_ids threaded)
│       ├── gate = V4TopKRouter | V4HashRouter           # CUSTOM router (sqrtsoftplus + bias/hash)
│       ├── experts = V4GroupedExperts                   # CUSTOM grouped experts (exact clamp+SwiGLU)
│       └── shared_experts = V4SharedExpertMLP           # REPLICATED shared MLP
├── hc_head = DeepseekV4HyperHead                        # REUSED (HF; final stream collapse) [post_process]
├── norm = V4RMSNorm                                     # [post_process]
└── output_layer = ColumnParallelLinear                 # MCORE-PARALLEL (TP=1 -> plain fp32 matmul; untied)
```

- **Attention / compressor / mHC are fully replicated** (the V4 EP plan FORBIDS TP-ing
  attention — MQA single-KV head; sharding contract §0/§1b). At TP=1 "replicated
  nn.Linear" == "ColumnParallelLinear", so they are kept as the byte-identical M1
  torch path; swapping to the parallel class is a TP>1 concern the EP plan never uses.
- **MoE is the only EP-shardable part.** Wired as custom V4 modules with HF attribute
  names, so at EP=1 the math is HF's exactly (M1-validated semantics).

## HF → mcore weight key map (the remap)

Verified programmatically: **0 shape mismatches, 0 unmapped HF params, 0 uncovered
mcore params** (only mcore extra = `output_layer._extra_state`, TE bookkeeping, skipped).
The map is the identity except two renames + stripping the `model.` prefix:

| HF key | mcore key |
|---|---|
| `model.embed_tokens.weight` | `embedding.word_embeddings.weight` |
| `lm_head.weight` | `output_layer.weight` |
| `model.norm.weight` | `norm.weight` |
| `model.hc_head.*` | `hc_head.*` |
| `model.layers.{i}.*` | `layers.{i}.*` (attention/compressor/mHC/norm + MoE all identical) |
| `model.rotary_emb.*` (buffers) | — dropped (mcore rebuilds; non-persistent) |

The MoE keys map **identity** (`...mlp.gate.weight`, `...mlp.gate.e_score_correction_bias`
or `...mlp.gate.tid2eid`, `...mlp.experts.gate_up_proj` `[E,2I,H]`, `...mlp.experts.down_proj`
`[E,H,I]`, `...mlp.shared_experts.{gate,up,down}_proj.weight`) because `V4MoELayer` keeps
HF's attribute names AND shapes — **at EP=1 there is no expert-axis slicing and no GLU
gate/up reorder** (those are the EP>1 / `GroupedMLP`-layout concerns, deferred). The grouped
`o_a` (`...self_attn.o_a_proj.weight` `[8192,4096]`) and single MQA `kv_proj` `[512,4096]`
also map identity (reused HF `DeepseekV4GroupedLinear` / replicated `nn.Linear`).

Implemented by `m_impl_parity.load_hf_into_mcore` (asserts full coverage; `strict=False`
only to tolerate the `_extra_state` placeholder).

## slime provider path

`--custom-model-provider-path custom_kernels.deepseek_v4.megatron.model_provider.v4_model_provider`
(reads the HF `DeepseekV4Config` from `--hf-checkpoint`, builds a minimal valid
`TransformerConfig` at TP=PP=EP=1, returns `V4LanguageModel`). slime's
`_get_model_provider_func` (model_provider.py:69) calls it as
`provider(pre_process, post_process, vp_stage)`; verified `load_function`-resolvable with
that signature. The forward accepts slime's kwargs
(`input_ids`, `position_ids=None`, `attention_mask=None`, `labels=None`,
`packed_seq_params`, `loss_mask`) and returns fp32 `[B,S,V]` — passing `loss.py`'s
preconditions (fp32, 3-dim, `size(0)==1`).  **NOTE:** `packed_seq_params is not None`
is hard-failed (see "Training-path correctness" below) — the SFT sanity must use
non-packed (bshd) `[B,S]` data.

## Parity gate result (mcore vs HF, tiny config)

The M-impl seam table is **identical to the M0 M1 table** (same tilelang compute, same
torch seams, mcore wrappers add no collectives at 1/1/1):

| pass | result |
|---|---|
| baseline (sinks=0, pos_bias=0) | **PASS** (every non-MoE seam rel ≤ 0.019, cos ≥ 0.9998; FINAL rel 0.0565) |
| nonzero sinks + compressor/indexer position_bias | **PASS** |
| nonzero `e_score_correction_bias` (top-k bias path, both models) | **PASS** (L2 6/256 bf16 flips, matched-MoE cos 0.99964) |
| clamp-stress `swiglu_limit=1.0` (clamp dominant, both models) | **PASS** (matched-MoE cos 0.99972 at normal scale) |
| neg-control: zero RoPE sin (reused seam) | BREAKS ✓ |
| neg-control: permute grouped o_a rows (reused seam) | BREAKS ✓ |
| neg-control: permute expert down_proj rows (**NEW MoE compute**) | BREAKS ✓ |
| neg-control: permute router gate rows (**NEW MoE router**) | BREAKS ✓ |
| neg-control: zero mcore correction-bias while HF keeps it (**bias path**) | BREAKS ✓ |
| neg-control: disable mcore swiglu clamp under clamp-stress (**clamp path**) | BREAKS ✓ |

Four must-pass variants + six negative controls. The last four controls give the
harness teeth on the NEW `V4GroupedExperts` / V4 routers / clamp / correction-bias —
M1's controls (RoPE, o_a) only touched code reused from M0. The nonzero-bias /
clamp-stress magnitudes are tuned to a *realistic* regime (bias std=0.02 on the
~0.9-mean sqrtsoftplus scores; `swiglu_limit=1.0` clamps ~44% of activations at normal
output scale); an over-large bias (std=0.5) or weight-scaling clamp-stress (×6) instead
manufactures a bf16-degenerate flip-storm / dynamic-range blowup that trips the bf16
gates as a *test artifact*, not a model bug (the disable-clamp / zero-bias neg-controls
confirm both paths are load-bearing).

## Training-path correctness (NOT covered by the .eval()/non-packed parity)

The parity runs in `.eval()` on non-packed `[B,S]` data; these gaps would bite a real
SFT step and are handled explicitly:

1. **Packed THD is hard-failed** (`mcore_model.py` forward): real slime SFT packs docs
   into `[1,T]` with `cu_seqlens`, but this forward synthesizes a GLOBAL `position_ids`
   and A1/compressor apply ONE window across the whole row → cross-document attention +
   compression leakage. **`raise ValueError` (NOT `assert`)** when `packed_seq_params is
   not None` — an `assert` is stripped under `python -O` and would silently train wrong;
   the hard-fail must be unconditional. The SFT sanity must use non-packed (bshd) `[B,S]`
   data (one doc/row). The per-document THD path (per-doc position_ids + A1/compressor
   masking from cu_seqlens + compression-window reset) is a documented TODO, deferred.
2. **Dropout pinned to 0** (`model_provider.make_transformer_config`): `hidden_dropout=0`,
   `attention_dropout=0` (Megatron defaults 0.1; HF V4 has none) — else a training-mode
   forward is silently stochastic/wrong.
3. **fp32-kept modules survive the bf16 cast** (`V4LanguageModel.bfloat16()/half()` →
   `restore_fp32_modules()`): slime wraps the model in `Float16Module`, which calls
   `module.bfloat16()`. The override restores to fp32: `attn_hc`/`ffn_hc` (B1/mHC kernel
   *requires* fp32 fn/base/scale) and the top-k `e_score_correction_bias` (added to scores
   before the argmax → a bf16 bias rounds and flips borderline experts) — both mirror HF
   `_keep_in_fp32_modules_strict`. It also keeps `hc_head` fp32, but that is a **local
   storage choice** matching the mHC sites (DeepseekV4HyperHead upcasts internally), NOT a
   HF-strict entry. Idempotent / re-callable after ckpt load.

   **RMSNorms kept bf16 — quantified, accepted tradeoff** (codex review item 3). HF keeps
   norms fp32 (`_keep_in_fp32_modules_strict`); we keep the gain bf16. `V4RMSNorm`
   normalizes in fp32 internally (`.to(torch.float32)` … `.to(input_dtype)`) but multiplies
   by the bf16 `self.weight` WITHOUT upcasting, so the bf16 gain DOES deviate from HF's
   fp32-norm forward. A clean "fp32 norms + bf16 linears" full HF forward is **unrunnable**
   in eager (fp32-norm output into a bf16 `q_a_proj` → dtype crash — the structural reason
   the bf16-norm choice exists). Measured the effect in isolation
   (`measure_rmsnorm_fp32_gap.py`: same bf16 input through `V4RMSNorm` with a bf16 vs fp32
   gain): **per-norm rel ≈ 0.0024, cos 0.99999, max_abs ~0.03**, stable across activation
   scales 0.1–20 — ~0.6× one bf16-mult epsilon (2⁻⁸=0.0039) and a small SUBSET of the
   whole-model bf16 floor (HF-bf16 vs HF-fp32 rel ~0.044). **Accepted**: at the bf16 floor,
   and restoring fp32 would (a) break the B2 kernel's bf16 norm-gain dtype assumption and
   (b) break Megatron's uniform-dtype DDP/optimizer bucketing (all params bf16-forward +
   fp32 master). The norms + base are frozen under LoRA, so this never even accrues a grad.

The only above-floor divergence in the baseline is the irreducible bf16 top-k router
boundary on the learned-router layer (L2: 5–6/256 tokens flip; matched-router MoE cos
0.99967; router input matches at the 1.7% bf16 floor; margin_med 0.016 = near-ties).
Hash routers (L0/L1): 0 flips (frozen `tid2eid`). Same standard bf16 MoE-routing effect
M1 documented.

## What is deferred (NOT in this milestone)

- **PP>1**: the `[B,S,hc_mult,H]` stream stack breaks the stock scheduler's `[S,B,H]`
  assumption (contract §4/§risk 3). `set_input_tensor`/`forward(not post_process)` carry
  the stream stub, but P2P shape inference + `hc_head`-on-last-stage are unwired.
- **EP>1**: the custom `V4GroupedExperts` runs the HF per-expert loop (exact at EP=1). The
  `MoEAlltoAllTokenDispatcher` all-to-all is a documented **no-op** at EP=1
  (`_AllToAll.forward` returns input when `group.size()==1`), so it is deferred to the
  EP>1 milestone where it is actually exercised + testable. EP>1 then needs: expert-axis
  weight slice, GLU gate/up reorder for the `GroupedMLP` flat layout, `weight.allreduce=
  False`, and the V4 `(indices,weights)`→`(probs[N,E],routing_map)` convention bridge.
- **LoRA**: not wired (per the order). When added: exclude `*.o_a_proj` (block-diagonal
  bmm, not a dense/parallel linear — bridge LoRA would mis-wrap it), explicit
  `target_modules` (no all-linear glob), hash router not auto-wrappable.
- **`sharded_state_dict`**: inherited from `LanguageModule` (handles embedding/output tie
  + bias padding). The PP-aware layer-offset + expert-sharding metadata is an EP/PP>1
  concern, deferred with those milestones.
- **Per-document THD path**: per-doc position_ids + A1/compressor masking from
  `cu_seqlens` + compression-window reset (hard-failed now; see Training-path #1).
- **HF→mcore PRODUCTION weight loading**: the parity uses an in-test remap of a fresh
  HF state dict; loading a real V4 checkpoint into the mcore model for actual training
  (vs the random-init sanity) is a separate task (the remap is the identity at EP=1, so
  it's a thin wrapper, but the real-checkpoint dtype/sharding path is untested here).
- **`build_schedule_plan` / combined-1f1b**: not implemented; combined-1f1b must be
  rejected at startup (the custom block has no schedule-plan path). Defer with PP>1.

## Deviations from the codex implementation order (reported)

The order (step 2/3) said "reuse `MoEAlltoAllTokenDispatcher` + `GroupedMLP`, minimal
`MoELayer` subclass". I instead built a custom `V4MoELayer` with custom grouped experts
**for this TP=PP=EP=1 milestone**, because:
- **Why the literal order couldn't be taken as-is at EP=1 without parity risk:** mcore's
  `MoEAlltoAllTokenDispatcher` consumes a `(probs[N,E], routing_map[N,E])` routing
  convention and `GroupedMLP` applies a *fused* SwiGLU with no clamp — neither matches
  V4's `(indices, weights)` + `_apply_gate` clamp. Wrapping them would require a
  routing-convention bridge + a custom activation hook + flat-weight GLU reorder, all of
  which only *change behavior* at EP>1 (the dispatcher is a literal no-op at EP=1), while
  adding numerical surface that could break the parity M0 already passes by reusing the
  exact HF MoE math.
- **What I changed:** the custom `V4MoELayer` reproduces HF `DeepseekV4SparseMoeBlock`
  exactly (input_ids-threaded V4 routers; `V4GroupedExperts` with the exact clamp+SwiGLU;
  replicated shared MLP) — honoring the *intent* of steps 2–4 (custom V4 routers with
  input_ids, custom V4 expert activation, no bypass state) while keeping the EP=1 numerics
  airtight.
- **Result:** parity PASSES identically to M0; the HF→mcore MoE key map is the identity.
- **Remaining gap:** the actual `MoEAlltoAllTokenDispatcher`/`GroupedMLP` integration is
  deferred to the EP>1 milestone (where it is exercised and the GLU/expert-slice mapping
  is testable). The custom layer is structured (separate router/experts/shared modules,
  HF-named) so that swap is localized to `V4MoELayer`.
- **Other adjustment:** `packed_seq_params` (thd) is accepted by the forward (slime always
  passes it) but A1 currently treats the packed `[1,T]` as one contiguous block — per-doc
  causal masking across `cu_seqlens` is the contract §9 open item, validated separately
  from this forward-parity gate.
```
