# scripts/quantize/

W8A8 INT8 quantization tooling for slime rollout acceleration. Scoped to
the Qwen3.5/3.6 hybrid (Mamba + softmax attention) family.

## Layout

```
scripts/quantize/
├── producers/                         # output a W8A8 ckpt
│   ├── rtn_w8a8.py                    # ★ production: pure-PyTorch RTN, no calibration
│   ├── gptq_w8a8.py                   # llmcompressor + GPTQ + calibration
│   └── smoothquant_w8a8.py            # llmcompressor + SmoothQuant (+optional GPTQ)
├── rotation/                          # Hadamard rotation pipeline + probes
│   ├── rotate_bf16.py                 # apply Hadamard rotation to BF16 ckpt
│   ├── probe.py                       # forward-divergence probe
│   ├── probe_compare.py               # diff two probe runs
│   ├── run_probe_all.sh               # sweep wrapper
│   └── PROBE_RESULTS.md               # rotation diagnostic writeup
├── utils/
│   ├── build_calibration.py           # extract calibration prompts from eval_0.pt
│   └── validate_checkpoint.py         # post-save sanity checker for INT8 ckpts
└── patches/
    └── sglang_qwen3_5_dense_entry.patch # legacy sglang patch (kept for reference)
```

## Producers — which to pick

| producer | algorithm | calibration needed | use when |
|---|---|---|---|
| **`rtn_w8a8.py`** | Round-To-Nearest (no calibration) | No | Default. Production-validated. Supports `--target {all-linear, mlp, non_linear_attn}`. |
| `gptq_w8a8.py` | GPTQ (Hessian-aware) | Yes (~512 prompts) | When RTN quality is insufficient. ~+1pp Correct T3 over RTN on quality-sensitive scopes. Slower (5–10× calibration time). |
| `smoothquant_w8a8.py` | SmoothQuant pre-shift + RTN/GPTQ | Yes (~128 prompts) | To suppress activation outliers (Qwen3.5 has known outlier channels). Produces both smoothed-BF16 and W8A8 outputs. |

All three emit `compressed-tensors` format (raw INT8 + per-channel FP32 scale)
loadable by sglang 0.5.10.post1+ via `quant_method=compressed-tensors`.

## Workflow

1. **(Optional) Build calibration data** — only needed for GPTQ / SmoothQuant:
   ```
   python scripts/quantize/utils/build_calibration.py \
       --eval-pt path/to/eval_0.pt --output /tmp/calib.jsonl
   ```

2. **(Optional) Hadamard rotation** — if you want rotation pre-treatment:
   ```
   python scripts/quantize/rotation/rotate_bf16.py \
       --model-path /path/Qwen3.6-27B --output-path /path/Qwen3.6-27B-rotated-mm-bf16
   ```

3. **Quantize**:
   ```
   # production RTN (no calibration):
   python scripts/quantize/producers/rtn_w8a8.py \
       --model-path /path/Qwen3.6-27B \
       --output-path /path/Qwen3.6-27B-w8a8-rtn \
       --target mlp

   # GPTQ (calibration required):
   source /tmp/w8a8-venv/bin/activate
   python scripts/quantize/producers/gptq_w8a8.py \
       --model-path /path/Qwen3.6-27B \
       --calibration-path /tmp/calib.jsonl \
       --output-path /path/Qwen3.6-27B-w8a8-gptq --multimodal

   # SmoothQuant (calibration required):
   source /tmp/w8a8-venv/bin/activate
   python scripts/quantize/producers/smoothquant_w8a8.py \
       --model-path /path/Qwen3.6-27B \
       --calibration-path /tmp/calib.jsonl \
       --bf16-output-path /path/Qwen3.6-27B-smooth-bf16 \
       --w8a8-output-path /path/Qwen3.6-27B-smooth-w8a8 \
       --multimodal
   ```

4. **Validate**:
   ```
   python scripts/quantize/utils/validate_checkpoint.py \
       --checkpoint /path/Qwen3.6-27B-w8a8-rtn \
       --reference-checkpoint /path/Qwen3.6-27B
   ```

## Quality ablation results (production v2.3_env_n8 100×8)

See `handoffs/in_progress/HANDOFF_DRKERNEL_W8A8_ROLLOUT.md` `### Per-turn
accuracy` for the full 6-variant rotation × scope ablation grid (BF16 → unrot
all-linear, Correct T3 0.279 → 0.210).

## Legacy notes

Removed 2026-05-27 in the refactor (no behavioral impact):
- `quantize_w8a8_rtn_llmcompressor.py` — llmcompressor RTN path, had a broken
  save flow; superseded by `producers/rtn_w8a8.py` (pure PyTorch).
- `quantize_w8_gptqmodel.py` — alternative W8-weights-only GPTQModel path;
  never used in production.
