# M-SFT — SFT-LoRA overfit-a-batch sanity (single GPU, TP=PP=EP=1)

**Status: PASS.** The whole stack trains end-to-end — the V4 mcore model + LoRA + slime's
exact SFT loss + real backprop — and **overfits a fixed batch**: loss **13.98 → 0.0016**
(100% drop) over 40 steps, finite throughout, grads ONLY on the LoRA adapters, base frozen,
`o_a_proj` not wrapped, packed-THD hard-fail not tripped (non-packed bshd).

Re-run: `CUDA_VISIBLE_DEVICES=0 python -m custom_kernels.deepseek_v4.megatron.sft_sanity`

## Loss curve (fixed bshd batch, B=2, S=64, response=32, Adam lr=1e-3)

The reported `loss` is the **sum of per-sample mean NLL** over the batch
(`get_sum_of_sample_mean` sums each sample's masked-mean) — i.e. ~B× the conventional
per-sample-mean NLL (B=2 here), matching how Megatron's loss is normalized. The sanity
prints both the sum and the per-sample-mean (`loss/B`); the table below is the sum.

| step | loss | grad_norm (LoRA) |
|---|---|---|
| 0 | 13.98370 | 3.24e+01 |
| 5 | 0.20050 | 6.87e+00 |
| 10 | 0.03819 | 8.20e-01 |
| 15 | 0.01579 | 3.20e-01 |
| 20 | 0.00895 | 1.98e-01 |
| 25 | 0.00478 | 9.42e-02 |
| 30 | 0.00310 | 5.74e-02 |
| 35 | 0.00217 | 3.91e-02 |
| 39 | 0.00160 | 2.63e-02 |

Monotone decrease to near-zero NLL = the LoRA adapters successfully memorize the fixed
batch, i.e. gradient flows correctly through the adapters into the frozen base's
attention/compressor seams (and through the 3 tilelang kernels' input-grad path).

## Validation (the M-SFT gate)

- **(a) 1 step no-OOM, loss finite**: PASS — peak GPU mem 0.73 GB (tiny 3-layer cfg);
  step time ~67 ms/step (fwd+bwd+opt, post kernel-compile warmup).
- **(b) loss DECREASES on a FIXED batch**: PASS — 13.98 → 0.0016 (100% drop).
- **(c) grad-norm finite**: PASS — LoRA grad-norm finite and shrinking (32 → 0.026).
- **(d) base `requires_grad=False` throughout, only LoRA updates**: PASS — every step
  checks for a base-grad leak (a base param with nonzero grad) and found none; trainable =
  1,613,824 LoRA params, frozen base = 250,370,303.
- **(e) packed hard-fail does NOT trip**: PASS — bshd path passes `packed_seq_params=None`.

### Validity controls (the drop is GENUINE, not trivial)

- **memorization**: after overfit, the response-token argmax matches the targets exactly —
  `logits[b, S-rl-1:S-1].argmax == tokens[b, S-rl:]` = **64/64 = 100%**. Proves the loss is
  computed over the RIGHT tokens with the correct off-by-one (not predicting padding / a
  shifted region), and the LoRA adapters genuinely memorized the batch.
- **all-frozen control**: with EVERY param frozen (including LoRA), the loss is identical
  step-to-step (e.g. 14.31 → 14.31, flat) — no spurious learning; the real drop requires
  the trainable LoRA params. Rules out a data/loss artifact driving the loss down.

## The invocation (what's real vs stubbed)

`sft_sanity.py` is a self-contained driver (single GPU, no Ray) that exercises the REAL
slime training math:

| component | real / stubbed |
|---|---|
| V4 mcore model build | **REAL** — `model_provider.build_v4_mcore_model` (the slime provider path) |
| LoRA | **REAL** — `lora.apply_v4_lora` (freezes base, wraps attn/compressor, excludes o_a) |
| SFT loss | **REAL** — slime's `sft_loss_function` + `get_sum_of_sample_mean` (bshd path) |
| backprop / autograd | **REAL** — full graph through the 3 kernels + LoRA adapters |
| dataset | **STUBBED** — tokenizer-free synthetic random token ids (a gradient-flow sanity needs no real text); fixed batch |
| optimizer / launch | **STUBBED** — plain torch `Adam` over the trainable LoRA params, no Megatron distributed-optimizer / DDP wrap / Ray |

### What this sanity validates — and what it does NOT

This sanity is faithful for the **forward / loss / autograd overfit behavior**: the real V4
model + LoRA, slime's real `sft_loss_function`, real backprop through the 3 kernels +
adapters, loss decreasing on a fixed batch with grads only on LoRA. slime's real train step
(`model.py:train`/`forward_step`) drives the identical `sft_loss_function` over the
identical `model(input_ids=..., position_ids=None, attention_mask=None, labels=None,
packed_seq_params=None)` call.

It does **NOT** validate (these are the next milestone — the full slime `--debug-train-only`
Ray launch):
- the **Megatron distributed optimizer** — it keeps an **fp32 master copy** of the bf16 LoRA
  params and applies the update in fp32, which **does change the per-step weight update** vs
  the plain bf16 `Adam` used here (so "the update math is unchanged" is NOT claimed);
- the **DDP grad-buffer / `main_grad` accumulation** path (grads land in a bucketed buffer,
  not `.grad`) and any grad-reduce;
- the **Ray launcher** + the full slime args namespace (Megatron's parser needs ~50+ flags)
  + checkpoint load.

The team lead explicitly allowed a direct train-backend sanity for "does the loss decrease".
The slime-buildable provider path (`v4_model_provider`, with `--v4-lora-dim`) is separately
verified to be `load_function`-resolvable and to return the LoRA'd model (LORA_NOTES /
M_IMPL_NOTES).

### Why bshd, not THD

The model hard-fails packed THD (`packed_seq_params is not None` → `raise ValueError`)
because A1 + the compressors would leak attention/compression windows across packed
document boundaries (M_IMPL_NOTES "Training-path #1"). slime's `get_batch` bshd path
(`data.py:70`) stacks tokens to `[B,S]` and sets `packed_seq_params=None`, and the SFT loss
path (`sft_loss_function` → `get_log_probs_and_entropy` → `_build_shifted_tokens` +
`_extract_per_sample`, `loss.py:386/430/445`) extracts the per-sample response log-probs —
so non-packed bshd is the correct SFT-sanity layout.
A real SFT run must use `--qkv-format bshd --micro-batch-size N` (asserted megatron-only,
no dynamic batch) until the per-document THD path is implemented.

### LoRA launch-config gate: do NOT set `--only-train-params-name-list`

slime calls `freeze_model_params` AFTER the model provider returns (model_provider.py:288),
and `--only-train-params-name-list` sets matching base params back to `requires_grad=True`
— which would OVERRIDE the LoRA freeze and un-freeze the base. For the SFT-LoRA launch,
**do not pass `--only-train-params-name-list`** (LoRA already froze the base and made the
adapters trainable), or scope it to match only `linear_in`/`linear_out`. The sanity asserts
(after the full build+freeze chain) that every trainable param is a LoRA adapter — a
base-unfreeze leak fails it (`base-unfreeze leak=0 (expect 0)`).

## Remaining gaps / friction

- The full slime Ray launch (`setup_model_and_optimizer` → `get_megatron_optimizer` + DDP +
  the `train()` loop with scheduler/wandb) was NOT exercised — only the model+loss+autograd
  core. A real multi-step SFT run through slime's launcher is the next integration step
  (would also exercise the distributed optimizer's fp32 master-weight path for the bf16
  LoRA params).
- The loss-mask here is all-ones over the response (train every response token); a real SFT
  mask comes from the data. Immaterial for the overfit sanity.
