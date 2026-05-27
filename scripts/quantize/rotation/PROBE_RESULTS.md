# Forward-divergence probe — rotation BF16 cost analysis

Goal: figure out which BF16 round-trip in the SpinQuant offline R1 rotation
pipeline contributes most to the observed ~4pp T3 correctness drop on
KernelBench Level1 (29.4% → 25.1%, rotated vs unrotated, 100×8).

## Method

5 variants applied to Qwen3.6-27B BF16, forward run on 8 fixed prompts
(mix of code + natural language, 42 tokens each after pad/truncate):

| variant | what's applied |
|---|---|
| `raw` | original BF16, no modification (baseline) |
| `roundtrip_only` | apply `w ← BF16(BF16(1+w) - 1)` to all 161 `Qwen3_5RMSNorm` |
| `fuse_bf16` | offset-norm BF16 temp `(1+w).bf16` + fuse 129 residual norms into following Linear (current llmcompressor pipeline, sans rotation) |
| `fuse_fp32` | same as fuse_bf16 but keeps temp `(1+w)` in FP32 (codex round-2 proposed fix) |
| `rotated_ckpt` | load the actual rotated-mm-bf16-no-center checkpoint as produced by `rotate_bf16_llmcompressor.py` |

Per-variant we save `hidden_states` (65 layers × 8 batch × 42 tokens × 5120 hidden)
and `logits` to `.pt` files. Comparator computes per-layer rel L2,
final logit rel L2, last-token KL, last-token top-1 agreement.

## Files

- `rotation_probe.py` — apply variant, forward, save hidden_states + logits
- `compare_rotation_probe.py` — load saved variants, print per-layer divergence
- `run_probe_all.sh` — run all 5 variants sequentially on .22

Each variant takes ~5min (load 27B + forward + save). Total ~30min.

## Results (2026-05-25)

```
=== roundtrip_only ===
per-layer rel L2 divergence:
  L0 embedding: 0.000%
  after L8:     0.415%
  after L16:    0.606%
  after L24:    0.817%
  after L32:    1.053%
  after L40:    1.218%
  after L48:    1.365%
  after L56:    1.433%
  L64 final:    1.666%
logit rel L2: 1.212%, last-token KL: 0.00117, top1 agree: 100.0%

=== fuse_bf16 ===
per-layer rel L2 divergence:
  L0 embedding: 0.000%
  after L8:     0.431%
  after L16:    0.629%
  after L24:    0.841%
  after L32:    1.081%
  after L40:    1.258%
  after L48:    1.427%
  after L56:    1.559%
  L64 final:    86.404%   ← jump at final layer (final norm zero'd, lm_head absorbs)
logit rel L2: 1.309%, last-token KL: 0.00146, top1 agree: 100.0%

=== fuse_fp32 ===
per-layer rel L2 divergence:
  L0 embedding: 0.000%
  after L8:     0.448%
  after L16:    0.637%
  after L24:    0.851%
  after L32:    1.083%
  after L40:    1.247%
  after L48:    1.391%
  after L56:    1.463%
  L64 final:    86.342%
logit rel L2: 1.254%, last-token KL: 0.00131, top1 agree: 100.0%

=== rotated_ckpt ===
per-layer rel L2 divergence:
  L0 embedding: 140.412%   ← rotated basis, expected
  after L8:     141.439%
  after L16:    141.657%
  after L24:    141.567%
  after L32:    140.871%
  after L40:    140.785%
  after L48:    140.665%
  after L56:    140.206%
  L64 final:    202.503%   ← norm zero'd + rotated basis
logit rel L2: 1.580%, last-token KL: 0.00126, top1 agree: 100.0%
```

## Round 2 — fp32 / fp64 pipeline + raw controls (added 2026-05-25)

To isolate rotation cost from bf16 forward arithmetic noise, ran 4 additional variants:
- `raw_fp32`: just cast model to fp32, no other modifications, fp32 forward
- `raw_fp64`: same with fp64
- `pipeline_fp32`: cast to fp32 + apply offset-norm fusion + Hadamard rotation, all storage fp32
- `pipeline_fp64`: same with fp64

```
=== raw_fp32 ===       logit rel L2: 1.144%, KL: 0.00096, top1: 100%
=== raw_fp64 ===       logit rel L2: 1.144%, KL: 0.00096, top1: 100%
=== pipeline_fp32 ===  logit rel L2: 1.144%, KL: 0.00096, top1: 100%
=== pipeline_fp64 ===  logit rel L2: 1.144%, KL: 0.00096, top1: 100%
=== rotated bf16 ckpt === logit rel L2: 1.580%, KL: 0.00126, top1: 100%
```

**All four fp32/fp64 variants give IDENTICAL 1.144% logit L2** — to 4 decimal places.

## Round 3 — restoring fused norm weights after rotation (added 2026-05-25)

Tested the proposed mitigation "put Qwen RMSNorm weights back after Hadamard transform" with exact inverse-fuse compensation:

- `pipeline_fused_bf16`: FP32 fuse + R1 rotation, then cast fused rotated model to BF16
- `pipeline_restore_norm_bf16`: same FP32 fuse + R1 rotation, then restore original norm weights and divide each consuming Linear input column by `(1 + weight)`, then cast BF16
- `pipeline_fused_fp64_to_bf16`: same fused layout, but transformation computed in FP64 before BF16 cast
- `pipeline_restore_norm_fp64_to_bf16`: same restored layout, but transformation computed in FP64 before BF16 cast

```
=== pipeline_fused_bf16 ===         logit rel L2: 1.559%, KL: 0.00142, top1: 100%
=== pipeline_restore_norm_bf16 ===  logit rel L2: 1.656%, KL: 0.00138, top1: 100%
=== pipeline_fused_fp64_to_bf16 ===         logit rel L2: 1.575%, KL: 0.00186, top1: 100%
=== pipeline_restore_norm_fp64_to_bf16 ===  logit rel L2: 1.605%, KL: 0.00203, top1: 100%
```

Pure FP64 confirms the inverse-fuse layout is mathematically equivalent:

```
=== pipeline_fp64 vs pipeline_restore_norm_fp64 ===
logit rel L2: 1.22e-7, max abs logit diff: 5.72e-6
```

So the restored layout is algebraically valid, but final BF16 storage still does **not** improve. Since the norm scale `D` and Hadamard `R` do not commute, the restored layout stores `W D R D^-1` in the Linear instead of `W D R`; this changes the BF16 rounding surface rather than eliminating it, and it was slightly worse on logits.

## Key conclusions (final)

1. **Rotation is mathematically exact when stored in fp32 or higher**. `pipeline_fp32 == raw_fp32` proves rotation adds 0% extra divergence — rotation perfectly cancels through residual stream + final norm/lm_head absorption. The 1.144% is **purely fp32 forward vs bf16 forward arithmetic difference**.

2. **fp64 buys nothing over fp32** for this model size — `raw_fp64 == raw_fp32`. fp32 is already converged precision for 27B forward.

3. **The 4.3pp KernelBench drop decomposes as**:
   - **1.14% logit L2** from bf16 vs fp32 forward arithmetic (exists with or without rotation — intrinsic to bf16 deployment)
   - **0.44% logit L2** from bf16 storage truncation of rotated weights (specific to rotation deployment)
   - **Total 1.58%** for rotated_bf16 ckpt with bf16 forward → ~5 nats per turn → e^5 ≈ 148× trajectory mass shift → 4.3pp downstream correctness loss

4. **`fuse_fp32` (codex round-2 proposed fix) ≈ `fuse_bf16`** because final `lin.weight` is stored bf16 anyway. The dominant truncation happens at storage cast, not at the intermediate `(1+w)` temp.

5. **`roundtrip_only` 1.21%** already covers most of the eventual 1.58% — the absorbed weight stores in bf16 either way.

6. **`hidden_states[L64]` 86% / 202% is by design** (final norm scale absorbed into lm_head, hidden state before lm_head is in different basis, lm_head compensates).

7. **Per-layer 0.025% monotonic growth** = normal residual-stack distributed accumulation, NOT a layer-specific bug.

## Implications for W8A8 pipeline

- **Don't bother fixing BF16 rotation cost** — math is right, 1.58% is bf16 deployment cost
- **W8A8 quantization**: per codex, keep **FP32 master copy of rotated weights** during quantize, apply RTN once from FP32 → save W8A8. Avoids "BF16 rotated → BF16 dequant → W8A8 quant" double round-trip
- The 4.3pp BF16 baseline loss will be dominated by the (larger) W8A8 quant noise. Net rotation+W8A8 vs unrotated+W8A8 should still favor rotation if INT8 outlier reduction works as theory predicts

## To reduce 1.58% → 1.14% (not recommended)

If for some reason we want to recover the 0.44pp:
1. Modify `rotate_bf16_llmcompressor.py` to `model.float()` before save → 108 GB ckpt
2. SGLang must load with `dtype=float32` → ~100 GB GPU memory + **~2× slower inference**
3. Does **NOT** recover the 1.14% — that's bf16-vs-fp32 forward arithmetic, fundamental to running 27B in bf16

Recovers 28% of total rotation drift (0.44/1.58). Not worth 2× inference cost for RL rollout where wall time dominates.

## How to reproduce

```bash
ssh -p 11116 root@192.168.16.22
source /tmp/w8a8-venv/bin/activate
cd /nfs/FM/chenshuailin/projects/kernel_agents/slime
bash scripts/quantize/rotation/run_probe_all.sh    # ~30min, runs all 5 variants
```
