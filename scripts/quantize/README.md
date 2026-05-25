# scripts/quantize/

Quantization tooling for slime rollout acceleration. Currently scoped to
Qwen3.5/3.6 family + W8 / W8A8 INT.

## Files

| File | Purpose |
|---|---|
| `build_calibration.py` | Extract prompt strings from a prior `eval_0.pt` into a JSONL for GPTQ calibration |
| `quantize_w8a8_llmcompressor.py` | llmcompressor + GPTQ, W8A8 (weights + activations INT8). Vanilla GPTQ only — llmcompressor doesn't expose GPTQv2 / GPTAQ / activation-aware variants |
| `quantize_w8_gptqmodel.py` | GPTQModel + GPTQ_V2 + `act_group_aware=True`, W8 (weights-only INT8). Activation precision stays BF16 |
| `sglang_qwen3_5_dense_entry.patch` | sglang patch that registers `Qwen3_5ForCausalLM` and guards its expert-location method against dense configs. Needed when llmcompressor's CausalLM load path rewrites `architectures` to the dense entry |

## Algorithm comparison

| | llmcompressor | GPTQModel |
|---|---|---|
| Quant scheme | W8A8 INT, W4A16, FP8, etc. | W4/W8 INT weights-only |
| GPTQv2 (FORMAT.GPTQ_V2) | ❌ | ✅ |
| GPTAQ (asymmetric calibration) | ❌ | ✅ (experimental, `GPTAQConfig`) |
| `act_group_aware` | ❌ | ✅ (default when `desc_act=False`, ~16k× faster than `desc_act=True`) |
| FOEM (first-order error compensation) | ❌ | ✅ |
| Qwen3.5 explicit model def | ❌ (falls back to generic CausalLM, rewrites `architectures`) | ✅ (`Qwen3_5GPTQ` mirrors `Qwen3_5MoeGPTQ` with dense MLP, preserves multimodal layout) |
| Activation quantization | ✅ (W8A8 path) | ❌ |
| Selective per-module quant | ✅ via `targets` + `ignore` regex | ✅ via `QuantizeConfig.dynamic` negative match |

## Which to pick

- **Rollout speedup is the main goal + accuracy is forgiving** → llmcompressor W8A8 (activation INT8 → tensor-core int8 matmul → 2× decode speedup)
- **Accuracy is critical / multi-turn agent reasoning** → GPTQModel W8 GPTQv2 (memory bandwidth saved on weights, but math stays BF16; ~0.07% wikitext degradation per the public Qwen3.6-27B-GPTQ-8bit ckpt)
- **First time, smoke test** → start with GPTQModel W8 (the public ckpt at `btbtyler09/Qwen3.6-27B-GPTQ-8bit` confirms the recipe works end-to-end on this model family)

## Loading caveats

Both paths produce checkpoints that mainstream loaders may not handle out of the box:

- **llmcompressor W8A8 + CausalLM mode**: `architectures` gets rewritten to `Qwen3_5ForCausalLM`; sglang's dense entry is unregistered (Blocker 1) and the registered code path has a hardcoded `num_experts` access (Blocker 2). Apply `sglang_qwen3_5_dense_entry.patch` to unblock. Or use `--multimodal` mode to preserve the multimodal arch tag.
- **GPTQModel W8 GPTQ_V2**: vLLM works after a small config-loader patch (see public ckpt's model card for the one-liner). sglang support is unverified — may need analogous patching.

See `handoffs/in_progress/HANDOFF_DRKERNEL_W8A8_ROLLOUT.md` for full investigation log.

## Calibration data

`build_calibration.py` extracts rendered first-turn prompts from any
prior DrKernel eval dump (one with the current production template).
Both quantize scripts consume the same JSONL format (one line =
`{"text": <prompt>, ...}`).
