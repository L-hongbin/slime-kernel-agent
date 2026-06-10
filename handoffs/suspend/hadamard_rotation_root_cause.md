# Hadamard Rotation Accuracy Drop Root Cause

Recorded on 2026-05-25. Scope: Qwen3.6-27B rotated BF16 eval drop on KernelBench L1.

## Verdict

The drop is **not caused by the Hadamard transform itself**. The root cause is the lossy BF16 handling of Qwen3.5/3.6 `apply_layernorm_1p` RMSNorm scales during SpinQuant norm fusion:

```python
Qwen3_5RMSNorm.forward: output = norm(x.float()) * (1.0 + self.weight.float())
```

SpinQuant must move the diagonal `(1 + weight)` scale out of RMSNorm because it does not commute with the hidden-state Hadamard basis. In the BF16 checkpoint, that scale is fused into downstream Linear weights and the residual/final norm weights are zeroed. This creates a systematic BF16 rounding perturbation before any quantization.

## Eval Evidence

Compared:

- Base: `checkpoints/Qwen3.6-27B/20260524_154851_ctx65536_n8_summ1600_v2_3_noenv_n8/dumps/rollout_data/eval_0.pt`
- Rotated no-center: `checkpoints/Qwen3.6-27B-rotated-mm-bf16/20260525_071901_ctx65536_n8_summ1600_rotated_mm_nocenter_bf16_n8/dumps/rollout_data/eval_0.pt`

The paired setup is valid:

```text
n 800 800
same_problem_id 800
same_slots 800
turn 0 same prompt_snapshot 800
```

Final prompt text differs after turn 0 because previous sampled responses differ, but the first-turn inputs are identical.

Paired T3 results for rotated no-center:

```text
T3 compile: both 234, base_only 167, rot_only 154, neither 245, net -13
T3 correct: both 104, base_only 131, rot_only 95, neither 470, net -36
T3 fast@1.0: both 27, base_only 61, rot_only 46, neither 666, net -15
```

So the observed T3 correct drop is a real paired net `-36/800 = -4.5pp`, not a sample-set mismatch.

## Loader Evidence

The rotated checkpoint currently has a `quantization_config`, but it is a transform-only compressed-tensors config, not W8A8:

```text
quantization_config = {
  "quant_method": "compressed-tensors",
  "transform_config": {"config_groups": {"R1": ...}},
}
```

The rotated safetensor index has no quantization scale/zero-point keys:

```text
nkeys 1184
quant-like keys []
```

SGLang logs show the compressed-tensors wrapper is active but no quantized kernel is used:

```text
Acceleration for non-quantized schemes is not supported by Compressed Tensors.
Falling back to UnquantizedLinearMethod
Load weight end ... quant=compressed-tensors
```

This makes the `quantization_config` confusing, but it is not evidence of INT8/W8A8 quantization in the BF16 eval.

## Forward Evidence

Existing probes on `.22` under `/tmp/probe_*.pt` were re-compared directly.

Hadamard/fusion algebra in high precision is effectively exact:

```text
pipeline_fp32 vs raw_fp32:
  logit rel L2 = 0.000274%
  last-token KL = 2.59e-7
  top1 agree = 100%

pipeline_fp64 vs raw_fp64:
  logit rel L2 = 0.000014%
  last-token KL = 2.09e-7
  top1 agree = 100%
```

But BF16 norm-scale handling alone reproduces most of the drift even without Hadamard:

```text
roundtrip_only vs raw:
  operation: w <- BF16(BF16(1 + w) - 1) for Qwen3_5RMSNorm
  logit rel L2 = 1.211%
  last-token KL = 0.00117

fuse_fp32 vs raw:
  operation: fuse RMSNorm scales into downstream Linear weights, store BF16
  logit rel L2 = 1.253%
  last-token KL = 0.00131

fuse_bf16 vs raw:
  operation: same fusion with BF16 temp
  logit rel L2 = 1.309%
  last-token KL = 0.00146

rotated_ckpt vs raw:
  operation: full rotated BF16 checkpoint
  logit rel L2 = 1.581%
  last-token KL = 0.00126
```

Therefore `~77-83%` of the full rotated BF16 logit drift appears in no-Hadamard ablations. The remaining gap is consistent with BF16 storage of the fully rotated/fused weights, not with a broken Hadamard mapping.

Tested the proposed "put norm weights back after transform" layout on 2026-05-25.

```text
pipeline_fused_bf16 vs raw:
  operation: FP32 fuse + Hadamard rotation, then cast fused rotated weights to BF16
  logit rel L2 = 1.559%
  last-token KL = 0.00142

pipeline_restore_norm_bf16 vs raw:
  operation: same FP32 fuse + rotation, then restore original Qwen RMSNorm weights
             and divide each consuming Linear input column by (1 + weight), then cast BF16
  logit rel L2 = 1.656%
  last-token KL = 0.00138

pipeline_fp64 vs pipeline_restore_norm_fp64:
  operation: both kept in FP64 through forward, one fused and one inverse-fused back into norms
  logit rel L2 = 1.22e-7
  max abs logit diff = 5.72e-6

pipeline_fused_fp64_to_bf16 vs raw:
  operation: FP64 fuse + Hadamard rotation, then cast fused rotated weights to BF16
  logit rel L2 = 1.575%
  last-token KL = 0.00186

pipeline_restore_norm_fp64_to_bf16 vs raw:
  operation: FP64 fuse + rotation + inverse-fuse norms, then cast BF16
  logit rel L2 = 1.605%
  last-token KL = 0.00203
```

So inverse-fuse restoration is mathematically valid: the FP64 fused and restored models produce identical logits to measurement precision. The failure mode is the final low-precision storage cast. Once cast to BF16, restoring the norm scale does **not** reduce rotated checkpoint drift, even with FP64 intermediate computation. The transformed Linear stores `W D R D^-1` instead of `W D R`; because `D` and `R` do not commute, this layout changes the BF16 rounding surface rather than eliminating it, and it was slightly worse in the probe.

## Weight Evidence

Qwen3.6-27B has 209 language-model norm weight tensors:

```text
base: total 209, zero 0, nonzero 209
rotated: total 209, zero 129, nonzero 80
rotated categories:
  residual norms: 128 zeroed
  final norm: 1 zeroed
  q/k norms: 32 nonzero
  linear_attn internal norms: 48 nonzero
```

That matches the expected SpinQuant fusion: 128 residual RMSNorms + final norm are absorbed into linears.

The BF16 `(1 + weight)` roundtrip is coarse:

```text
norm elements checked: 674816
changed fraction: 0.88485
abs diff mean: 0.001632
abs diff p50 / p95 / p99 / max: 0.001465 / 0.003906 / 0.003906 / 0.007812
median relative perturbation: 1.59%
p95 relative perturbation: 20.19%

BF16 step near 0.05: 0.000244
BF16 step near 0.2: 0.000977
BF16 step near 1.0: 0.007812
```

Example from `layers.0.input_layernorm.weight`:

```text
original:  [ 0.05834961, -0.04833984, -0.06152344, -0.04418945]
roundtrip: [ 0.05468750, -0.04687500, -0.06250000, -0.04296875]
delta:     [-0.00366211,  0.00146484, -0.00097656,  0.00122070]
```

The issue is structural: storing `weight` near zero preserves more BF16 detail than storing or fusing `1 + weight` near one.

Fusing into representative Linear weights and storing BF16 adds per-layer storage error:

```text
layer0 input_layernorm -> linear_attn.in_proj_qkv:
  fused weight storage rel L2 = 0.166%

layer0 post_attention_layernorm -> mlp.gate_proj:
  fused weight storage rel L2 = 0.166%

layer3 input_layernorm -> self_attn.q_proj:
  fused weight storage rel L2 = 0.166%

layer63 post_attention_layernorm -> mlp.up_proj:
  fused weight storage rel L2 = 0.166%
```

Small per-layer BF16 storage errors then accumulate through 64 layers and long multi-turn sampling.

## Conclusion

The precision loss should be attributed to **BF16 norm-scale fusion/storage in the rotated checkpoint**, not to the Hadamard transform itself.

For future W8A8 work, avoid `rotated BF16 -> dequant/re-quant` as the source of truth. Use an FP32 master during rotation/fusion and quantize directly from that master into W8A8. For a clean BF16-only ablation, run a no-Hadamard `fuse_fp32/fuse_bf16` eval; the forward probe already predicts it should show most of the same degradation.
