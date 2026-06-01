# INDEX

## Repo Guides

- `RUNTIME.md`: stable runtime facts for nodes, endpoints, shared paths, data paths, and common run entrypoints; experiment-specific settings stay in handoffs.
- `handoffs/in_progress/handoff_w4a16_awq.md`: standalone W4A16 AWQ handoff covering INT4 MLP-only scope, UltraChat calibration, producer/gate fixes, ASYM SGLang zero-point runtime patch/full eval, symmetric v2 probes/full eval, BF16/W8A8 baseline comparison, HF AWQ comparison, and completed `mlp-rmsfix` RMSNorm-offset audit; v2 score is 0.31750 and ASYM after runtime patch is 0.35375, both below target.

## Data Conversion

- `scripts/data/convert_verl_to_slime.py`: converts VERL parquet records to the `ground_truth` + `extra_info` schema used by `scripts/debug.sh`.
- `tests/utils/test_convert_verl_to_slime_data.py`: unit coverage for the converter output and slime `Dataset` loading with `prompt_key=ground_truth`, `metadata_key=extra_info`, and chat-template rendering.

## DrKernel Plugin

- `handoffs/in_progress/handoff_drkernel_slime_plan.md`: current migration plan and handoff for implementing DrKernel on slime, including current rollout status and next KernelGYM reward step.
- `slime_plugins/drkernel/design-docs/dynamic_prompt_templates.md`: design for fragment-based dynamic prompt templates in the DrKernel custom rollout.
- `slime_plugins/drkernel/prompt_templates/legacy/first_turn_tvm_ffi/tvm_ffi_module_v2_3.jinja`: expanded legacy reference for the active `drkernel_v1_tvm_ffi` v2.3 first-turn prompt.
- `slime_plugins/drkernel/extract.py`: extracts DrKernel/KernelGym-ready kernel submissions from assistant responses.
- `slime_plugins/drkernel/eval_throttle.py`: lightweight helpers for bounding DrKernel eval coroutine fan-out.
- `slime_plugins/drkernel/kernelgym_rm.py`: async KernelGym HTTP client and optional single-turn custom RM wrapper.
- `slime_plugins/drkernel/rollout.py`: DrKernel custom rollout and prompt renderer; template selection is restricted only by `sample.metadata["template_allowed"]`.
- `slime_plugins/drkernel/prompt_templates/single_turn_v1.yaml`: first dynamic prompt profile set for DrKernel single-turn rollout templates.
- `scripts/drkernel/summarize_kernelgym_eval.py`: CLI to summarize KernelGym eval artifacts (file or run dir); writes review `sample_*.txt` under `dumps/rollout_data/`.
- `tests/utils/test_drkernel_prompt_templates.py`: unit coverage and real-data formatted prompt dump for DrKernel prompt rendering.
- `tests/utils/test_drkernel_eval_summary.py`: unit coverage for KernelGym eval summary metric extraction.
- `checkpoints/drkernel_prompt_debug/formatted_prompt_examples.txt`: generated real-data prompt examples for manual review.

## Run Configs

- `scripts/debug.sh`: current debug training/eval configuration for converted drkernel/kernelbench parquet data.
- `scripts/debug/debug.27b.w8a8.sh`: W8A8 RTN rollout smoke harness with bounded eval response length and DrKernel eval concurrency.
- `scripts/debug/debug.27b.tp4.eagle.w8a8.sh`: W8A8 rollout + EAGLE harness (`HF_W8A8_DIR` env-overridable; default is RTN Non-LA+MTP). SGLang memory/prefill can be overridden with `SGLANG_MEM_FRACTION_STATIC`, `SGLANG_CHUNKED_PREFILL_SIZE`, and `SGLANG_MAX_PREFILL_TOKENS`; current numbers are in `handoffs/in_progress/handoff_drkernel_w8a8_rollout.md`.
- `scripts/debug/debug.27b.tp4.w8a8.nospec.sh`: W8A8 no-spec A/B control (`HF_W8A8_DIR` env-overridable); keep checkpoint scope aligned with the EAGLE harness.
- `scripts/debug/debug.27b.tp4.eagle.awq_w4a16.sh`: AWQ W4A16 EAGLE eval wrapper; runs `check_awq_w4a16.py --reference-checkpoint`, symmetric-only by default, and can require SGLang WNA16 ASYM zero-point support before delegating to the TP4 EAGLE harness.
- `scripts/debug/bench_sglang_wall.py`: fixed-shape SGLang HTTP wall benchmark used for W8A8 EAGLE/no-spec probes.
- `scripts/debug/analyze_g128_lengths.py`: one-off paired BF16/per-channel/G128 eval dump analyzer for the 2026-06-01 G128 wall-time root-cause; writes turn-level token CSV, summary, and real long-tail examples.
- `scripts/analysis/summarize_sglang_decode_log.py`: parses SGLang `Decode batch` rows and reports same-batch throughput / accept stats.
- `checkpoints/Qwen3.6-27B/w8a8_eagle_probe_20260530/`: `.22` pure-SGLang BF16/W8A8 EAGLE/no-spec probe logs; short-prompt 96x2048 shows EAGLE positive on BF16 (1.25x), W8A8 all-linear (1.12x), and W8A8 nonla-mtp (1.16x) fixed-request throughput.
- `checkpoints/Qwen3.6-27B/g128_sglang_audit_20260601/`: `.22` pure-SGLang fixed-output G128 audit after A800 blockwise-int8 config tuning; TP4 C32 O512 shows G128 faster than BF16 for no-spec (11.356s / 1443 tok/s vs 12.178s / 1345 tok/s) and EAGLE (8.390s / 1953 tok/s vs 9.996s / 1639 tok/s).
- `checkpoints/Qwen3.6-27B/g128_length_analysis_20260601/`: paired BF16/per-channel/G128 length audit from eval dumps; shows G128 full-eval wall inflation is dominated by pathological generated-token long tail, especially turn1 (`30.27M` total tokens vs per-channel `26.47M`, turn1 `16.02M` vs `13.21M`). Review `top_g128_turn1_longer_examples.md`, `length_pathology_examples.md`, and `weight_rel_l2_compare.txt`; the last shows G128 blockwise-int8 has higher BF16 reconstruction rel-L2 than per-channel across 263 tensors (`0.01348` mean vs `0.01037`).
- `checkpoints/Qwen3.6-27B/rtn_g64_length_analysis_20260601/`: `.22` RTN W8A8-G64 Non-LA+MTP tuned eval analysis; valid run score 0.38125, wall 1:39:02, req>=80 decode median 2611 tok/s, total response tokens 27.55M, turn1 `>32768=109` and `>60000=15`; review `summary.txt` and `top_long_examples.md`.
- `checkpoints/Qwen3.6-27B/g64_smooth_length_analysis_20260601/`: `.22` SmoothQuant W8A8-G64 Non-LA+MTP tuned eval analysis; valid run score 0.37500, wall 1:35:25, req>=80 decode median 2597 tok/s, total response tokens 27.23M, turn1 `>32768=104` and `>60000=14`; review `summary.txt` and `top_long_examples.md`.
- `scripts/quantize/producers/rtn_w8a8.py`: RTN W8A8 producer with MTP-aware `mlp_mtp` and `non_linear_attn_mtp` targets.
- `scripts/quantize/producers/rtn_w8a8_g128.py`: SGLang `blockwise_int8` W8A8-G128 RTN producer; default scope is Non-LA+MTP and keeps `linear_attn` / `mtp.fc` BF16.
- `scripts/quantize/sgl_engine_smoke.py`: minimal SGLang Engine load/generate smoke test for quantized checkpoints, including optional EAGLE load and a conditional text-only torchvision import fallback for broken quantization venvs.
- `scripts/quantize/producers/smoothquant_w8a8.py`: SmoothQuant BF16 transform followed by RTN W8A8, with MTP tensor restoration before RTN.
- `scripts/quantize/producers/awq_w4a16.py`: experimental AWQ W4A16 producer; MLP-only final quantization, MLP-only AWQ mappings, default `duo_scaling=both`, restores non-target BF16 drift and BF16 MTP tensors.
- `scripts/quantize/patches/llmcompressor_qwen3_5_awq.py`: defensive llmcompressor AWQ smoothing patch/test hook for Qwen3.5 offset RMSNorm; current oneshot path also reports llmcompressor's built-in offset-norm conversion.
- `scripts/quantize/patches/sglang_compressed_tensors_wna16_asym.patch`: SGLang WNA16 compressed-tensors patch for ASYM W4A16; passes `weight_quant.symmetric` into `CompressedTensorsWNA16` so `weight_zero_point` is used.
- `scripts/quantize/utils/check_awq_w4a16.py`: pre-eval static gate for AWQ W4A16 checkpoints; checks W4 metadata/scope, optional reference drift with sampled large tensors, optional symmetric-only enforcement, and optional SGLang WNA16 ASYM runtime support.
- `scripts/quantize/utils/compare_real_hidden_mlp_quant_loss.py`: no-SGLang pre-eval MLP loss probe on real BF16 hidden states captured from prompt forward passes.
- `checkpoints/quantized/AWQ/Qwen3.6-27B-AWQ-W4A16-mlp`: completed `.16` AWQ W4A16 MLP-only checkpoint; MTP/self-attn/linear-attn/lm-head stay BF16 and `check_awq_w4a16.py --require-mtp` passed.
- `checkpoints/quantized/AWQ/Qwen3.6-27B-AWQ-W4A16-mlp-preservefix`: fixed `.16` AWQ W4A16 MLP checkpoint; restores non-target BF16 drift and passes reference gate.
- `checkpoints/quantized/AWQ/Qwen3.6-27B-AWQ-W4A16-asym-mlp`: `.16` AWQ W4A16_ASYM MLP checkpoint; built with 256 UltraChat calibration rows, `duo_scaling=both`, `n_grid=40`; after WNA16 ASYM patch full eval passed with score 0.35375.
- `checkpoints/quantized/AWQ/Qwen3.6-27B-AWQ-W4A16-mlp-v2`: `.16` AWQ W4A16 symmetric MLP checkpoint; uses fixed recipe `duo_scaling=both`, `n_grid=40`, passes reference gate/probes, completed full eval with score 0.31750, and is not a production candidate.
- `checkpoints/quantized/AWQ/Qwen3.6-27B-AWQ-W4A16-mlp-rmsfix`: `.16` AWQ W4A16 symmetric MLP RMSNorm-offset audit checkpoint; gate/probes passed and probe loss matches v2, so RMSNorm offset is not the main W4 low-score cause.
- `checkpoints/Qwen3.6-27B-AWQ-W4A16-mlp-v2/20260531_150856_awq.w4a16.mlp_v2.100x8.eagle.rm39_ctx65536_n8_summ1600`: completed `.16` AWQ W4A16 MLP v2 100x8 EAGLE eval on `.39` reward; Ray job `raysubmit_NZzJFx7auYP355hN`, score 0.31750, wall 1:10:55, server decode tok/s median 2659, accept len 3.278, accept rate 76%, RM workers 8 GPU + 24 CPU.
- `checkpoints/Qwen3.6-27B-AWQ-W4A16-asym-mlp/20260531_233411_awq.w4a16.asym_mlp.sglzpfix.100x8.eagle.rm39_ctx65536_n8_summ1600`: completed `.16` AWQ W4A16_ASYM MLP 100x8 EAGLE eval after SGLang zero-point patch; Ray job `raysubmit_3RpzfvxksL2Lnmae`, score 0.35375, wall 1:25:49, server decode tok/s median 2607, accept len 3.25, accept rate 75%, RM workers 8 GPU + 24 CPU.
- `checkpoints/Qwen3.6-27B-AWQ-W4A16-mlp/20260531_093445_awq.w4a16.mlp.100x8.eagle_ctx65536_n8_summ1600`: invalid `.16` AWQ W4A16 MLP 100x8 EAGLE eval from the pre-gate checkpoint; score 0.32875, wall 1:04:04, srv tok/s 2967, final accept 3.288.
- `checkpoints/quantized/AWQ/logs/`: AWQ W4A16 quant/eval/real-hidden-probe logs, including v2 logs, ASYM logs, `eval_awq_w4a16_asym_sglzpfix_100x8_eagle_driver.log`, RMSNorm audit logs, and real-hidden probe json/log pairs.
- `scripts/quantize/utils/mtp_checkpoint.py`: restores `mtp.*` tensors lost by HF `save_pretrained`; supports Qwen3.5/3.6 MTP R1 rotation with separate `mtp.lm_head.weight`.
- `scripts/quantize/rotation/check_mtp_rotation.py`: static QuaRot+MTP gate; checks body embed/head, Gemma RMSNorm fused identity, separate MTP head, MTP rotation, and W8A8 MTP dequant rel-L2 before full eval.
- `scripts/quantize/patches/sglang_qwen3_5_mtp_separate_lm_head.patch`: SGLang patch for loading checkpoint-provided `mtp.lm_head.weight` and not overwriting it during EAGLE head sharing.
- `checkpoints/quantized/QuaRot/Qwen3.6-27B-QR-BF16-mtp`: `.22` QuaRot BF16 short-path artifact rebuilt with Qwen/Gemma RMSNorm `1+weight` fusion and separate MTP head.
- `checkpoints/quantized/QuaRot/Qwen3.6-27B-QR-W8A8-nonla-mtp`: `.22` QuaRot+RTN Non-LA+MTP W8A8 artifact; passed `check_mtp_rotation.py`.
- `checkpoints/Qwen3.6-27B/20260528_234734_ctx65536_n8_summ1600_nonla_emfrac09_newSlimeKG_tp4`: completed `.22` BF16 no-spec 100x8 baseline with 8 RM workers; score 0.37500, wall 1:58:37, srv tok/s 2320, `num draft tokens=0`.
- `checkpoints/Qwen3.6-27B-QR-W8A8-nonla-mtp/20260531_002353_quarot.nonla_mtp.gemmafix_ctx65536_n8_summ1600`: completed `.22` QuaRot Non-LA full eval, Ray job `raysubmit_u9kfu3ucjh2c1W43`; score 0.37250, wall 1:24:12, srv tok/s 3086, final accept 3.145.
- `checkpoints/Qwen3.6-27B-QR-W8A8-nonla-mtp/20260531_023637_quarot.nonla_mtp.gemmafix.nospec_ctx65536_n8_summ1600`: completed `.22` QuaRot Non-LA no-spec control, Ray job `raysubmit_5FpQdQqA3bCQcsLu`; score 0.35750, wall 1:45:44, srv tok/s 2560, no `--sglang-speculative-*`.
- `checkpoints/quantized/SmoothQuant/Qwen3.6-27B-SQ-{BF16,W8A8-RTN}-{nonla,mlp}-mtp-a0p5-ultrachat`: renamed current SmoothQuant alpha-0.5 UltraChat-calibrated BF16/W8A8 artifacts.
- `checkpoints/quantized/SmoothQuant/Qwen3.6-27B-SQ-W8A8-G64-RTN-nonla-mtp-a0p5-ultrachat`: `.22` SmoothQuant alpha-0.5 UltraChat W8A8-G64 RTN checkpoint; blockwise_int8 `[64,64]`, Non-LA+MTP scope, keeps `linear_attn` and `mtp.fc` BF16; quant log `checkpoints/quantized/SmoothQuant/logs/sq_w8a8_g64_nonla_mtp_20260601_081350.log`, tuned SGLang config log `g64_sglang_tune_20260601_011540.log`.
- `checkpoints/Qwen3.6-27B-SQ-W8A8-G64-RTN-nonla-mtp-a0p5-ultrachat/20260601_013236_smooth.g64.nonla_mtp.sglcfg64.mem82.cp4096.100x8.eagle.rm16_ctx65536_n8_summ1600`: completed `.22` SmoothQuant W8A8-G64 100x8 EAGLE eval; Ray job `raysubmit_3H6J6bwNb7V3CRhL`, score 0.37500, wall 1:35:25, server decode tok/s median 2597, accept len 3.244, no missing-config/OOM; not a production candidate.
- `checkpoints/Qwen3.6-27B/quarot_smoothquant_100x8_20260530/`: current `.22` QuaRot/SmoothQuant W8A8 100x8 experiment root with quant logs, eval summary, and driver logs.
- `scripts/debug/bench_w8a8_spec_components.py`: synthetic A800 microbench for Qwen3.6-27B TP4 W8A8+EAGLE component costs; raw runs in `checkpoints/Qwen3.6-27B/w8a8_spec_rootcause/bench_w8a8_spec_components_tp4_b80.json` and `bench_w8a8_spec_components_tp4_b96_v2.json`.
- `scripts/debug/debug.27b.hicache_ablation.sh`: Qwen3.6-27B HiCache ablation harness copied from `debug.27b.sh`; env-gates HiCache backend, ratio, Mamba scheduler strategy, overlap schedule, and `CUDA_LAUNCH_BLOCKING`.
- `scripts/debug/debug.27b.1p3d.sh`: Qwen3.6-27B 1P3D harness for `.22`; uses reward `.40`, TP=2, `--prefill-num-servers 1`, env-selectable PD backend, and no-overlap guard.
- `scripts/debug/debug.27b.next_ablation.sh`: shared harness for ordered `.22` Qwen3.6-27B ablations; wrappers are `debug.27b.tp4x2.sh`, `debug.27b.1p3d.nohicache.sh`, `debug.27b.1p3d.hicache.sh`, `debug.27b.1p3d.nixl.nohicache.sh`, and `debug.27b.1p3d.nixl.hicache.sh`.
- `checkpoints/Qwen3.6-27B/20260528_150010_ctx65536_n8_summ1600_emfrac09_newSlimeKG_hicachev2_tp4x2_c96/run.log`: TP4x2 HiCache attempt; failed during SGLang CUDA graph capture at custom all-reduce.
- `checkpoints/Qwen3.6-27B/20260528_151643_ctx65536_n8_summ1600_emfrac09_newSlimeKG_1p3d_nohicache/run.log`: 1P3D Mooncake no-HiCache attempt after port allocator fix; fails KV transfer with TCP CUDA-copy invalid argument.
- `checkpoints/Qwen3.6-27B/20260528_153819_ctx65536_n8_summ1600_emfrac09_newSlimeKG_hicachev2_1p3d/run.log`: 1P3D Mooncake+HiCache attempt; reaches eval with host cache allocated, then fails the same TCP CUDA-copy KV transfer path and was stopped.
- `checkpoints/Qwen3.6-27B/20260528_152354_ctx65536_n8_summ1600_emfrac09_newSlimeKG_1p3d_nixl_c16_noov/run.log`: 1P3D NIXL no-HiCache attempt; reaches eval, but 2/800 took 9:05 and was stopped.
- `checkpoints/Qwen3.6-27B/20260528_123220_ctx65536_n8_summ1600_nonla_emfrac09_newSlimeKG_hicache_1p3d_nixl_c16_noov_restart/run.log`: NIXL/UCX 1P3D run; transfer/reward path works, but eval was much slower than regular 4-engine and was stopped at 2/800.
- `handoffs/in_progress/handoff_rollout_speedup.md`: **hub/总览** for drkernel Qwen3.6-27B rollout speedup. Navigates the five attempt areas: (1) prefix cache — concluded ineffective (cache hit ≠ wall; decode + KernelGym dominate); (2) W8A8-INT8 quantization — in progress, ~1.07× wall; (3) Hadamard rotation accuracy root-cause (supports quant quality); (4) SpecDec — EAGLE measured ~1.23× wall, score-lossless; (5) reward-server concurrency — reward worker 8→16 gives a real score-lossless ~1.14× wall (decode stays near-saturated). Key takeaway: speedup levers are quantization + SpecDec + reward concurrency + the survey's untried directions (PD/async), not cache; judge by score, not wall.
- `handoffs/rollout_speedup/handoff_reward_server_concurrency.md`: reward worker 8→16 results plus 8-worker SGLang C96/C64/C32 efficiency comparison; plot generated by `scripts/analysis/plot_sglang_concurrency_c32_c64_c96.py` to `handoffs/images/sglang_concurrency_c32_c64_c96.png`.
- `handoffs/rollout_speedup/handoff_specdec_drkernel.md`: SpecDec direction. EAGLE chain measured ~1.23× wall (1:58:37→1:36:09) at lossless score; `accept_length≈3.23` saturates `num_draft_tokens=4`. Ranked untried routes: P0 EAGLE tree-ify + NGRAM/suffix (zero-train, NGRAM best fits the multi-turn code-reemit workload), P2 EAGLE3 + domain draft (needs training).
- `handoffs/complete/handoff_bf16_baseline_jump_root_cause.md`: resolved root cause for the May24→May28 BF16 baseline +10pp jump; SGLang Qwen3.5 GDN non-contiguous `a/b` stride bug fixed by `sgl-project/sglang#22312`.
- SGLang/vLLM references for BF16 baseline jump triage: likely root cause `sgl-project/sglang#21019` plus fix `sgl-project/sglang#22312` for Qwen3.5 GDN non-contiguous `a/b` stride accuracy regression; direct premature-stop symptom issue `sgl-project/sglang#20550`; adjacent strict-thinking/think-end references `sgl-project/sglang#23953`, `sgl-project/sglang#18246`; vLLM parser-boundary references `vllm-project/vllm#35230` and `vllm-project/vllm#38789`.
- `scripts/quantize/utils/build_ultrachat_calibration.py`: builds generic UltraChat calibration JSONL for SmoothQuant/GPTQ; prefer this over DrKernel eval dumps to avoid target-distribution leakage.
- `scripts/quantize/utils/build_calibration.py`: DrKernel eval-dump calibration extractor, diagnostic/domain-adapted only because it can leak KernelBench/DrKernel eval distribution into calibration.
- `checkpoints/quantized/RTN/Qwen3.6-27B-W8A8-RTN-nonla-mtp`: RTN W8A8 with `--target non_linear_attn_mtp`; accept length stays healthy (~3.22 real eval, ~2.95 fixed-request probe).
- `checkpoints/quantized/RTN/Qwen3.6-27B-W8A8-G128-RTN-nonla-mtp`: RTN W8A8-G128 `blockwise_int8` artifact for `.22`; 263 G128 INT8 tensors, no `linear_attn` quantization, `mtp.fc` BF16, with `static_scope_summary.txt` and `validation_32tensor.txt`.
- `checkpoints/quantized/RTN/Qwen3.6-27B-W8A8-G64-RTN-nonla-mtp`: RTN W8A8-G64 `blockwise_int8` artifact for `.22`; 263 G64 INT8 tensors, no `linear_attn` quantization, `mtp.fc` BF16, with `static_scope_summary.txt` and `validation_32tensor.txt`; quant log `checkpoints/quantized/RTN/logs/rtn_w8a8_g64_nonla_mtp_20260601_111700.log`.
- `checkpoints/Qwen3.6-27B-W8A8-G128-RTN-nonla-mtp/20260531_110823_w8a8.g128.nonla_mtp.100x8.eagle_ctx65536_n8_summ1600`: diagnostic old `.22` W8A8-G128 100x8 EAGLE eval; Ray job `raysubmit_A3uyjDkukCt8EgfV`, score 0.38250, wall 1:55:24, srv tok/s 2866, final accept 3.236, but logs contain OOM/router errors, so do not use for production comparison.
- `checkpoints/Qwen3.6-27B-W8A8-G128-RTN-nonla-mtp/20260531_150558_w8a8.g128.nonla_mtp.sglcfg.mem82.cp4096.100x8.eagle_ctx65536_n8_summ1600`: completed `.22` W8A8-G128 fixed SGLang full eval; Ray job `raysubmit_gFWnazLJUabafytt`, score 0.38875, wall 1:39:56, server decode tok/s median 2767, accept len 3.227, no OOM; uses tuned SGLang G128 configs plus `SGLANG_MEM_FRACTION_STATIC=0.82`, `SGLANG_CHUNKED_PREFILL_SIZE=4096`, `SGLANG_MAX_PREFILL_TOKENS=4096`.
- `checkpoints/Qwen3.6-27B-W8A8-G64-RTN-nonla-mtp/20260601_032830_rtn.g64.nonla_mtp.sglcfg64.mem82.cp4096.100x8.eagle.rm16_ctx65536_n8_summ1600`: completed `.22` W8A8-G64 RTN 100x8 EAGLE eval; Ray job `raysubmit_4prDqk8zqB8VFG2X`, score 0.38125, wall 1:39:02, server decode tok/s median 2611, accept len 3.239, no fallback/OOM; not a production candidate.
- `scripts/eval_kernelbench_level1.yaml`: slime eval config for converted KernelBench L1 validation data.
- `checkpoints/Qwen3.5-9B/run_20260515_122920.log`: successful Qwen3.5-9B DrKernel eval rerun on `.67`; `raysubmit_v7whRX3AfMxtL3YY`, KernelBench L1 score `0.035`.
- `tests/utils/test_eval_config.py`: unit coverage for eval config prompt/response/context length fields.
- `tests/utils/test_drkernel_multiturn_render.py`: unit coverage for DrKernel multi-turn prompt history.
- `tests/utils/test_drkernel_eval_throttle.py`: unit coverage for DrKernel eval coroutine throttling.
