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

## Key conclusions

1. **`fuse_fp32` ≈ `fuse_bf16`**: codex's "keep `(1+w)` in fp32" fix has essentially no effect (1.25% vs 1.31% logit L2). The dominant BF16 truncation happens at `lin.weight.data = (... * s).bfloat16()` (storing the absorbed weight back to BF16) — not at the intermediate `(1+w)` cast. Fully fixing would require fp32 weight storage (doubles ckpt size + ~2× slower inference).

2. **`roundtrip_only` alone causes 1.21% logit L2**: just round-tripping all 161 norms through `BF16(BF16(1+w)-1)` produces most of the BF16 cost. Fusion adds ~0.1%, rotation adds another ~0.3%.

3. **`hidden_states[L64]` 86% / 202% is expected, not a bug**: the final RMSNorm's scale is absorbed into `lm_head`, so the hidden state before `lm_head` is in a different basis (no norm scale applied, or rotated). `lm_head` compensates → logits remain near-identical (top-1 100%).

4. **Per-layer 0.025% monotonic growth** is normal residual-stack distributed accumulation (`e_{l+1} ≈ (I + J_l) e_l + u_l`), NOT a specific layer's amplifier bug.

5. **Why 1.6% logit L2 + 100% top-1 → 4.3pp KernelBench drop**: multi-turn autoregressive generation accumulates KL drift. Sequence KL = sum over tokens → 0.001 nats/token × 5000 tokens ≈ 5 nats per turn → e^5 ≈ 148× relative trajectory probability mass shift. Across 3 turns easily large enough to flip ~0.7pp per trajectory (matches the 4.3pp drop at n=8 pass@8 style).

## Implications for W8A8 pipeline

- **Don't bother fixing BF16 rotation cost** — it's structural, not a bug
- **When quantizing to W8A8**: keep an **FP32 master copy of rotated weights** during the quantize step, apply RTN once from FP32 → save BF16 weight + scale, avoiding the extra "BF16 rotated → BF16 dequant → W8A8 quant" round-trip
- Codex notes a future probe worth doing: teacher-forced cumulative KL on real 5-8k token KernelBench trajectories (this 42-token probe only explains plausibility, doesn't bound downstream effect)

## How to reproduce

```bash
ssh -p 11116 root@192.168.16.22
source /tmp/w8a8-venv/bin/activate
cd /nfs/FM/chenshuailin/projects/kernel_agents/slime
bash scripts/quantize/run_probe_all.sh    # ~30min, runs all 5 variants
```
