# custom_kernels/deepseek_v4/megatron — DS-V4 module index

The custom V4-Flash mcore model + its harnesses. Status: **PROD** (imported by
training), **HARNESS** (standalone validation tool, run manually), and **NOTES**
(investigation records).

## Production modules (imported by slime training)

| File | Purpose |
|---|---|
| `model_provider.py` | slime `custom_model_provider_path` entry (`v4_model_provider`); `build_v4_mcore_model`; env/arg bridging (recompute, MTP, LoRA cfg) |
| `mcore_model.py` | `V4LanguageModel` (43-layer hand-written decoder loop, act-ckpt wrap, packed-MXFP4 frozen experts, EP sharded_state_dict) |
| `decoder.py` | `V4DecoderLayer`, `V4HyperConnection` (fixed to official TileKernels mHC), MoE blocks |
| `attention.py` / `compressor.py` / `rope.py` | V4Attention (TileLang kernel wrap), CSA/HCA compressors, RoPE/RMSNorm |
| `lora.py` | `apply_v4_lora` (freeze base, wrap attention+compressor linears) |
| `mtp.py` | Train-side MTP head (gated by `--mtp-num-layers 1`) |
| `native_checkpoint.py` | HF-native ↔ mcore key mapping + loader (incl. `include_mtp`) |
| `slice_torch_dist.py` | Checkpoint slicer/converter (torchrun; `--include-mtp`) |
| `_kernels.py` | Kernel re-export shim (TileLang attn/mHC/compressor entry points) |

## Standalone harnesses (HARNESS — need GPU, run manually)

| File | Purpose |
|---|---|
| `mcore_smoke.py` | Random-init forward smoke of the full stack |
| `sft_sanity.py` / `sft_rollout.py` | Tiny SFT overfit sanity |
| `lora_validate.py` | LoRA wrap/freeze audit + grad flow check |
| `m0_smoke.py` / `m1_parity.py` / `m_impl_parity.py` | Staged mcore-vs-HF parity (M0/M1/impl) |
| `real_weight_parity.py` | Parity on REAL checkpoint weights (strict load) |
| `verify_torch_dist.py` | Converted-checkpoint verification (chained) |
| `measure_rmsnorm_fp32_gap.py` | RMSNorm precision gap probe |

## Notes (NOTES)

`M0_NOTES.md`, `M1_RESULTS.md`, `M_IMPL_NOTES.md`, `CONVERSION_NOTES.md`,
`LORA_NOTES.md`, `SFT_SANITY_NOTES.md` — stage-by-stage
investigation records (superseded summaries live in
`handoffs/deepseek-v4/release/`).
