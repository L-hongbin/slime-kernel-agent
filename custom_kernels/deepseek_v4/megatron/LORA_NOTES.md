# M-LoRA — LoRA on the V4 mcore model (TP=PP=EP=1)

**Status: PASS.** LoRA (`megatron.bridge.peft.LoRA`) wraps the V4 attention + compressor
linears, freezes the base, and is forward-identity at init (zero-init `linear_out`). The
3 tilelang kernels' backward now skips frozen-base param grads under LoRA. Validated on
GPU 0: trainable = only LoRA adapters, base frozen, forward still matches HF (parity
PASS), backward grads only on LoRA params (no leak), `o_a_proj` NOT wrapped.

Re-run: `CUDA_VISIBLE_DEVICES=0 python -m custom_kernels.deepseek_v4.megatron.lora_validate`
Grad-gating unit check: `... -m custom_kernels.deepseek_v4.megatron.test_lora_grad_gating`
Forward parity (incl. a LoRA-applied pass): `... -m custom_kernels.deepseek_v4.megatron.m_impl_parity`

## Files

| File | Contents |
|---|---|
| `lora.py` | `apply_v4_lora(model, dim, alpha, dropout)` (build `LoRA` + apply), `DSV4_LORA_TARGET_MODULES`, `audit_lora(model)`. |
| `lora_validate.py` | The 4-check gate (trainable-only-LoRA / forward-identity / backward-only-LoRA / wrap-audit). |
| `test_lora_grad_gating.py` | Unit check: A1/B2 backward returns None for frozen params, present+unchanged for trained, input grads preserved. |
| `model_provider.py` | `v4_model_provider` applies LoRA when `--v4-lora-dim > 0`. |
| kernel edits | `attention/kernel.py` (A1 `dsink`), `compression/kernel.py` (B2 `_CSAFusedPool`/`_HCAFusedPool`/`_GatedPoolCore` `dpos`/`dweight`) — `needs_input_grad` gating. |

## Target modules (what's wrapped vs excluded)

`megatron.bridge.peft.LoRA` matches a module if its leaf `name == pattern` OR
`wildcard_match(pattern, full_name)` (anchored regex, `*`->`.*`). We use **full-name
wildcards** (not bare leaf names) because several leaf names collide:

```
DSV4_LORA_TARGET_MODULES = [
    "*.self_attn.q_a_proj", "*.self_attn.q_b_proj",
    "*.self_attn.kv_proj",  "*.self_attn.o_b_proj",
    "*.self_attn.compressor.kv_proj", "*.self_attn.compressor.gate_proj",
]
```

| pattern | matches | does NOT match (why scoping matters) |
|---|---|---|
| `*.self_attn.kv_proj` | attention MQA kv_proj | `*.self_attn.compressor.kv_proj` (regex ends `.self_attn.kv_proj$`, not `.compressor.kv_proj`) |
| `*.self_attn.compressor.gate_proj` | compressor gate | `*.mlp.shared_experts.gate_proj` (different leaf path → experts stay frozen) |
| — (absent) | — | **`*.self_attn.o_a_proj`** — never listed → never wrapped |

**Excluded (audited in `lora_validate`):** `o_a_proj` (grouped block-diagonal bmm — a
generic `LinearAdapter` would treat its `[8192,4096]` weight as one dense linear and inject
a rank-r adapter ignoring the 8-group structure → wrong math; contract §1f/§7), the routed
`experts.*`, the `gate` router, the `shared_experts.*` MLP, embedding, output_layer.

The V4 attention/compressor linears are plain `nn.Linear`, so `LoRA.transform` wraps them
with `LinearAdapter` (lora_layers.py:139,153) — needs no `module.config`, zero-inits
`linear_out`, so the wrapped forward is the IDENTITY at init (forward parity preserved).

## Freeze + apply order

`LoRA.__call__` (PEFT base) FREEZES the whole model first (all `requires_grad=False`), then
walks + swaps matched linears for adapters whose A/B (`linear_in`/`linear_out`) are
trainable — so the base is frozen automatically (no separate freeze needed). The provider
applies LoRA on the built fp32 model BEFORE slime's `Float16Module` wrap, so the adapters
cast to bf16 with the model; `restore_fp32_modules()` (mHC/hc_head/correction-bias) is
unaffected (LoRA does not wrap those).

## Kernel backward gating (needs_input_grad)

Under LoRA the base params the kernels hold gradients for are frozen, so the backward skips
those param-grad reductions (returns `None`) while still returning the INPUT grads that
carry cross-layer flow to the trainable adapters. B1 (mHC) already did this; added:

| kernel | frozen param (skipped grad) | input grads always returned |
|---|---|---|
| A1 `_V4FlashAttn` | `sinks` (`needs_input_grad[3]`) | dq, dk_raw, dk_comp |
| B2 `_CSAFusedPool` | `position_bias` (idx 2), `weight`=kv_norm gain (idx 3) | dkv, dgate |
| B2 `_HCAFusedPool` | `position_bias` (idx 2), `weight` (idx 3) | dkv, dgate |
| B2 `_GatedPoolCore` | `weight` (idx 2) | dkv, dgate |

`test_lora_grad_gating` confirms: frozen → grad None; trained → grad present AND the input
grads are unchanged from the ungated path (A1 input grads compared by relative norm because
A1's `dKV` is atomic-accumulated / run-to-run nondeterministic at ~1e-5, not the gating).

## Validation result (`lora_validate`, tiny config, dim=16 alpha=32)

- **(d) wrap audit**: 16 modules wrapped — layer 0 (sliding, no compressor) 4 + layers 1,2
  (CSA/HCA) 6 each. `o_a_proj` wrapped: none. experts/router/shared/embed/output: none.
- **(a) trainable**: 1,613,824 LoRA params trainable; 250,370,303 base frozen;
  `trainable_are_lora_only=True`; representative base params (sinks/position_bias/experts/
  embedding/output/hc_head/correction_bias) all `requires_grad=False`.
- **(b) forward**: LoRA-vs-noLoRA logits rel = 0.00 (zero-init identity). Also a dedicated
  `m_impl_parity` pass builds the model, loads HF weights, applies LoRA, and confirms it
  STILL matches HF at the bf16 floor (so the wrap preserved the math).
- **(c) backward**: nonzero grad on 16 LoRA modules, 0 base params — no grad leak on frozen
  sinks/etc.

## Deferred / notes

- **Grouped `o_a` LoRA**: excluded for this milestone. If ever wanted, needs a custom
  grouped-block adapter (per-group rank-r), not the generic `LinearAdapter`.
- **slime args**: `--v4-lora-dim` / `--v4-lora-alpha` / `--v4-lora-dropout` are read via
  `getattr(args, ...)` (slime has no built-in LoRA args; these are V4-provider-specific —
  add to the launcher's extra args, or set them on the args namespace).
- **Experts/router LoRA**: OFF (frozen). Enabling later would use the grouped-expert
  adapter path + the `LoRATopKRouter` (router) — both opt-in, not in this target list.
- **THD packing**: the model still hard-fails packed `packed_seq_params` (M-impl note);
  the SFT-LoRA sanity uses non-packed bshd `[B,S]` data.
