# R1 — slime SFT-LoRA train loop (single GPU, TP=PP=EP=1)

**Status: PASS for the single-process slime `train()` wrapper path and for the Ray
`train.py --debug-train-only` tiny path with Megatron Muon.** After review flagged that the
first R1 driver called `train_one_step` directly, `r1_train.py` was patched to call
`slime.backends.megatron_utils.model.train(...)` once over 40 fixed steps and capture
`train/loss` from `logging_utils.log`. That wrapper run passes: fixed synthetic bshd loss
**7.02722 -> 0.000069 over 40 steps**, LoRA-only optimizer params, base frozen, and fp32
modules surviving Float16Module. The Ray Muon attempt6 run also passes: 39 Ray train steps,
Muon BF16 Newton-Schulz kernel, loss **7.027 -> 0.114**, and final model-only `torch_dist`
save succeeded.

## Run

```
# phase 1: bootstrap a random-init LoRA-wrapped torch_dist ckpt (build via provider + save)
python -m custom_kernels.deepseek_v4.megatron.r1_bootstrap <args> --save $CKPT --pretrained-checkpoint $CKPT
# phase 2: the wrapper-path train loop (loads $CKPT, 40 SFT steps on the fixed batch)
python -m custom_kernels.deepseek_v4.megatron.r1_train  <args> --load $CKPT --no-load-optim --no-load-rng
```
The full arg set + staging is in `scripts/run.r1.v4.sft.lora.sh`. Files:
`r1_bootstrap.py`, `r1_train.py`, `sft_rollout.py` (synthetic rollout +
`stage_hf_checkpoint`), `scripts/run.r1.v4.sft.lora.sh`.

Evidence scratch path:
`/tmp/claude-0/-nfs-FM-chenshuailin-projects-kernel-agents-slime-v4flash-lora/r1_wrapper_1782862339`

Ray Muon evidence:
`handoffs/deepseek-v4/r2_logs/r1_muon_sft_lora_node64_attempt6.log`

## Loss Curve

Fixed bshd batch, B=2 S=64 resp=32, slime `train()` wrapper + real dist-optimizer Adam
lr=1e-3:

| step | loss | grad_norm |
|---|---|---|
| 0 | 7.02722 | 1.48e+01 |
| 5 | 0.12592 | 4.28e+00 |
| 10 | 0.00395 | 9.88e-02 |
| 15 | 0.00130 | 2.90e-02 |
| 20 | 0.00056 | 1.31e-02 |
| 25 | 0.00027 | 5.55e-03 |
| 30 | 0.00015 | 3.25e-03 |
| 35 | 0.00009 | 2.00e-03 |
| 39 | 0.00007 | 1.43e-03 |

Cross-check vs the stub (`SFT_SANITY_NOTES`, per-sample-mean 6.99→0.0009): same overfit
signal, same scale. The single-process `train()` wrapper + distributed optimizer added no
math drift.

## Validation Status

- **wrapper-path non-OOM, loss finite, fixed-batch loss DECREASES**: PASS — 7.02722→0.00007
  (100%), finite.
- **Ray `train.py --debug-train-only` with Megatron Muon**: PASS — attempt6 loaded the
  bootstrap torch_dist checkpoint, used Emerging-Optimizers BF16 Newton-Schulz kernels,
  logged 39 training steps, and ended with `Job ... succeeded`.
- **model-only final save under Muon**: PASS — the Ray run uses
  `--no-save-optim --no-save-rng` and saved `/tmp/codex-r1-muon-20260701f/out/iter_0000039`.
  Muon optimizer-state checkpoint/resume is not proven by this sanity.
- **optimizer param groups = ONLY LoRA**: PASS — `optimizer params=1,613,824 ==
  model-LoRA == model-trainable` (numel) AND `identity(trainable-set == LoRA-set)=True`.
  The Megatron distributed optimizer's `_get_param_groups` skipped the frozen base and
  `_build_model_and_main_param_groups` built fp32 master copies for ONLY the 1.61M trainable
  LoRA params (the bf16-LoRA + fp32-master behavior).  Audited BOTH by numel AND by the
  SOURCE-param identity set (the optimizer is built from the model's `requires_grad=True`
  params; the trainable-id set == the LoRA-id set exactly — codex R1 P2; identity check
  added so an equal-numel swap can't pass).  The fp32 masters are distinct objects, so we
  check the SOURCES not the masters.
- **train-wrapper grad/finalize hooks**: PASS — current code calls `train()`, so
  `grad_scale_func`, no-sync hooks, param sync handling, and `finalize_model_grads_func`
  are configured by the same wrapper as actor training.
- **base `requires_grad=False` throughout**: PASS — `base_frozen=True`,
  `base_frozen_after=True`, `base_grad_hook_hits=0`.
- **Float16Module wrap kept fp32 modules fp32**: PASS — `mHC/hc_head/correction-bias
  fp32=True; a normal linear bf16=True`.  First exercise of the `.bfloat16()` override
  through the real Float16Module wrap.
- **save→load roundtrip of the LoRA-wrapped model**: PASS — bootstrap saved a torch_dist
  `iter_0000000` (with dist-optimizer sharded state); the run loaded it (`successfully
  loaded checkpoint ... at iteration 0`), 0 missing/unexpected MODEL keys.

## Bootstrap (why + how)

slime's `initialize_model_and_optimizer` always `load_checkpoint`s, and the wrapper asserts
`--load` exists+non-empty (no from-scratch path); the HF-load path needs a registered V4
`AutoBridge` (we don't have one). So R1 bootstraps a valid torch_dist ckpt once:
`r1_bootstrap.py` does the 1-rank `init(args)` + `setup_model_and_optimizer` (builds via the
SAME provider+LoRA path; BUILD only — load is `initialize_model_and_optimizer`'s job) +
slime `save(0, ...)`.  The saved keys are the adapter-wrapped layout, so the run's load is
self-consistent.  Base random + LoRA zero-init.

## Bugs found + fixed during R1 (real-run signal)

1. **uninit router/expert/hc_head → NaN loss** (the M0_NOTES trap, via the real provider):
   `V4{TopK,Hash}Router` / `V4GroupedExperts` / `DeepseekV4HyperHead` use `torch.empty` and
   HF's `_init_weights` never runs on the mcore path (Megatron only inits embedding/output
   via `init_method`).  First train run NaN'd from step 0.  **Fix:** `model_provider.
   init_v4_module_weights` (run by `build_v4_mcore_model(init_weights=True)`, default) inits
   exactly those, mirroring HF `_init_weights`.  Parity harness passes `init_weights=False`
   (it overwrites via load_state_dict).
2. **`hf_validate_args` mismatches**: needed `--moe-ffn-hidden-size 256` (==hf
   moe_intermediate_size), `--untie-embeddings-and-output-weights` (hf tie=False),
   `--rotary-base 10000` (==hf rope_theta).
3. **world_size**: slime sets `world_size = actor_num_nodes * actor_num_gpus_per_node`, so a
   1-GPU run needs `--actor-num-nodes 1 --actor-num-gpus-per-node 1` (else dp=8 and gbs%mbs
   asserts).
4. **scheduler-state mismatch on load/save**: the bootstrap saved scheduler state tied to its own
   `--num-rollout`; the run uses `--no-load-optim --no-load-rng` (standard finetune flow) so
   the optimizer/scheduler state load (which asserted on `lr_decay_steps`) is skipped — the
   dist-optimizer is still BUILT fresh with fp32 LoRA masters (build != load). The Ray script
   now passes these flags too. For Muon, the sanity also passes `--no-save-optim --no-save-rng`
   because Megatron's current Muon chained optimizer can train but optimizer-state checkpointing
   still needs a separate compatibility gate.
5. **PYTHONPATH gotcha**: the pip-editable `slime` points at `slime-dev-csl-2`; the repo
   root must be on PYTHONPATH so the worker imports THIS repo's `slime` + `custom_kernels/deepseek_v4`.
6. **`--qkv-format bshd` REQUIRED** (codex P0): slime defaults `thd` (packed) which the V4
   model hard-fails; bshd → `packed_seq_params=None`.

## What R1 does NOT cover (deferred)

- Real V4-Flash weights (R2), EP>1 (R3), rollout/RL (R4–R7).
- Production Muon optimizer-state checkpoint/resume. The tiny Ray sanity intentionally proves
  model training and model-only save, not optimizer-state recoverability.
