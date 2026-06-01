# Handoff: merge origin/main (THUDM upstream) into dev_csl — 2026-06-01

## TL;DR

`origin/main` (THUDM `slime`, tip `bf14dc21`, "[release] bump to v0.3.0") was
merged into `dev_csl`. The merge is **textually clean (0 conflicts)**, but it
pulls **53 upstream commits / 191 files / +18053 −2169** over an old base, so
this is a large upstream catch-up, not a small sync. The merge has **not been
validated at runtime** and **not been pushed**. Review the open risks below
before relying on it or launching any training/eval.

## What happened

- Updated local `main` to `origin/main` (fast-forward, `main` had 0 unique
  commits) → `main = bf14dc21`.
- `git merge --no-edit origin/main` on `dev_csl` → merge commit **`32acf8cf`**,
  parents `d00773dd` (dev_csl tip) + `bf14dc21` (origin/main tip).
- Zero conflicts; no `MERGE_HEAD` left, no unmerged paths.
- `dev_csl` is now **67 ahead / 0 behind** `csl/dev_csl` (the 53 upstream
  commits + merge commit are not yet pushed).
- Merge base used: `git merge-base d00773dd bf14dc21` = `79989380`.

## Which "main"

Chosen explicitly: **`origin/main` = THUDM upstream**, NOT `lhb/main`
(collaborator kernel-agent main, the lineage dev_csl actually descends from) and
NOT `csl/main` (own fork main). origin/main diverges over the older base
`79989380`; `lhb/main` shares the closer base `8ef1fb47`. If a future merge is
meant to track the kernel-agent project line, that is `lhb/main`, not this.

## Verification done (text-merge integrity only)

Only 3 files were edited on **both** sides (where auto-merge interleaves edits
and silent breakage hides):

| File | dev_csl edit | origin edit | Merged result |
|---|---|---|---|
| `slime/backends/megatron_utils/model.py` | eval-only `scheduler_train_iters = max(args.train_iters, 1)` block in `get_optimizer_param_scheduler` | expanded train_iters comment + variable-gbs `step_global_batch_size`/`group_mask_sums`/`reduce_train_step_metrics` refactor, R3 KL guard, `save_hf_model_direct` | both present & coherent; dev_csl block kept as context, origin layered around it |
| `slime/backends/sglang_utils/sglang_engine.py` | health-wait `attempt` counter + `timeout=5` + periodic "Waiting for SGLang server health" logging | (other regions) | dev_csl edits present at lines 86–102 |
| `scripts/run-qwen3-32B.sh` | (minor) | `pkill -9 python` region | merged; low risk (script) |

- `py_compile` passes for both merged `.py` files.
- All dev_csl deliverables intact: `RUNTIME.md`, `CLAUDE.md`,
  `handoffs/in_progress/handoff_w4a16_awq.md`,
  `scripts/quantize/producers/awq_w4a16.py`, `slime_plugins/drkernel/rollout.py`.

## Open risks

1. **group_id contract — LATENT, not currently triggered (investigated).**
   Upstream added `_validate_group_id_annotated()` (`slime/ray/rollout.py:603`,
   origin-only — 0 occurrences in dev_csl pre-merge), called on the custom
   rollout's training output before flattening. For a "compact" rollout (depth-3
   `list[list[list[Sample]]]`, multiple training samples from one rollout) it
   asserts every sibling has a non-None, shared `group_id`.
   - **Why it does NOT fire today:** DrKernel's *training* rollout
     (`generate_rollout_async`) is single-turn — its own docstring says so and
     it delegates to core `generate_and_rm_group` → `generate_and_rm` → core
     `generate`, which returns a single `Sample` (no `--custom-generate-function-path`
     is set in `debug.27b*.sh`). So the training output is `list[list[Sample]]`
     (depth-2 standard); the validator only asserts at depth ≥ 2 leaves, so it
     skips, and `group_id` falls back to `sample.index` (`rollout.py:673`).
   - **Multi-turn is eval-only:** `generate_multi_turn_eval_sample` runs only in
     the eval path, which returns before `_validate_group_id_annotated` and is
     not used for loss — so eval is safe regardless. `--use-multi-turn
     --max-turns 3` in the harness drives the eval loop, not a multi-turn
     training rollout. The `multi_turn_gamma` / `filter_by_last_turn` /
     `padding_turns` args are defined in `drkernel/args.py` but **not yet
     consumed** anywhere — multi-turn *training* folding is unimplemented.
   - **When it WOULD bite (future):** if/when a multi-turn *training* rollout is
     added that emits per-turn `list[Sample]` (the defensive
     `isinstance(group[0], list)` branches anticipate this), each leaf
     `list[Sample]` becomes a compact group and every sibling must carry a
     shared `group_id`. At that point set `Sample.group_id` per the upstream
     contract (one group = one rollout's turns) and decide the loss-denominator
     semantics (`group_mask_sums` → `reduce_train_step_metrics`) deliberately.
     Not needed now.
   - Precise validator condition (per Codex review): asserts when `depth >= 2`
     **and** `len(node) > 1`; a compact group of exactly 1 sample at depth 2 is
     also skipped.
2. **`Sample.rollout_id` is now deprecated/read-raising.** `types.py:131` raises
   `AttributeError` on **read** of `Sample.rollout_id` and warns on write
   (redirects to `group_id`). Checked: `drkernel/rollout.py` only uses
   `rollout_id` as a **function parameter** and reads/writes `sample.index`
   (still valid) — it never reads `sample.rollout_id`. So no breakage found in
   the DrKernel rollout, but any custom reward / filter / process plugin loaded
   via `--custom-rm-path`, `--rollout-sample-filter-path`,
   `--dynamic-sampling-filter-path`, or `--rollout-all-samples-process-path`
   that READS `sample.rollout_id` will now crash — a runtime trap for user
   plugin code beyond the DrKernel rollout (flagged by Codex review).
3. **`rename rollout_ids to group_ids (#1984)` / `Don't use sample.index as
   default rollout_id (#1965)` / `Fix trajectory merging logic (#1963)`.** Core
   rollout id/trajectory semantics shifted. The custom rollout won't auto-break
   (it's a copy) but will not benefit from these fixes and may diverge from what
   the merged training side expects.
4. **Large surface, no runtime test.** 191 files include a new agent RL
   framework (`slime/agent/` adapters + `examples/coding_agent_rl`), MiniMax-M2
   support, delta weight sync (disk+nccl), FlashQLA Qwen GDN backend, sglang
   v0.5.12.post1 docker patches, `cp_utils` context-parallel changes, and a
   large new test set. None exercised against the DrKernel path here. (The
   sglang runtime is the docker image, independent of this repo merge, so the
   sglang version bump is lower risk for the merge itself.)
5. **Loss-aggregation behavior may have changed (flagged by Codex review).**
   Upstream's `cp_utils.reduce_train_step_metrics` now replaces the old inline
   per-step loss reduction in `train_one_step` (`model.py:602`, imported at
   `model.py:34`). This is a behavioral change to the loss pipeline, not just a
   refactor — if its semantics differ from the old per-step
   `num_samples_or_tokens` reduction it could silently shift training dynamics.
   The 61 passing CPU tests (`loss_cp_invariance`, `metric_report`, `cp_utils`)
   cover it, but a Codex xhigh review of this specific path is warranted before
   relying on training loss/metrics.

## Validation performed (2026-06-01, on .16 new-image container)

Ran on `.16` (`ssh -p 23422 root@192.168.16.16`, new image matching the merged
upstream code; `source ./set_env.sh` first). Repo on shared `/nfs`, HEAD
`32acf8cf`. Use `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to avoid tokenizer
fetch hangs.

- **Merge-critical CPU tests (loss aggregation / group_mask_sums): PASS.**
  `cp_utils`, `metric_report`, `loss_cp_invariance`, `sample`,
  `megatron_argument_validation`, `dp_schedule` → 61 passed.
- **tests/utils (incl. dev_csl quant + DrKernel): mostly PASS.** mask_utils 2,
  megatron_role_config 6, drkernel_eval_throttle 2, eval_config 3,
  drkernel_multiturn_render 11, drkernel_cache_template 15,
  awq_w4a16_producer 16 (+1 skip), llmcompressor_smoothquant_patch 5 — all pass.
- **Quant export import works:** sglang 0.5.12.post1 (delta symbols present),
  `quantize_layer_int8` imports fine.
- **GPU eval smoke (end-to-end) on .16: PASS.** Qwen3.6-27B BF16 TP4 + EAGLE,
  full KernelBench L1 (100 prompts) at `N_SAMPLES_PER_EVAL_PROMPT=1`, 3-turn
  eval. Ray job `raysubmit_B42KfTE6QeDsc9ZR` **succeeded**, 100/100, 0 errors,
  20:09. `eval/kernelbench_level1 = 0.36`, `spec_accept_length 3.235`,
  `prefix_cache_hit_rate 0.395`, `truncated_ratio 0.02`. Score is consistent
  with the documented BF16 baseline (~0.375 at n=8/800) and accept length
  matches the known-good ~3.23, so the merged serving → multi-turn eval rollout
  → EAGLE → KernelGym reward path is not regressed. Confirms eval-only
  (`--num-rollout 0 --debug-rollout-only`) does NOT load megatron ref weights —
  the missing `…/Qwen3.6-27B/torch_dist` was a non-issue for eval. (Intended a
  3-prompt smoke, but the harness hardcodes `EVAL_CONFIG_PATH` at line 22 with
  no `:-` default, so the env override was ignored and the full 100-prompt set
  ran — a stronger check.) Training path NOT smoked (the latent group_id item
  applies only there if multi-turn training is later implemented).

Two failures, both PRE-EXISTING (not merge-induced):
1. `test_drkernel_prompt_templates`: 4 failed / 13 passed — root cause is the
   profile-name mismatch: the tests pass `"profile": "drkernel_v1"` but the
   renderer now defaults to `drkernel_v1_tvm_ffi`, whose backend candidate list
   is just `tvm_ffi_module` (so candidate ids like `lhb_v3` / `csl_cuda_agent`
   are no longer in the profile — note the legacy `.jinja` files still exist on
   disk; it is the profile config that changed, not the templates, per Codex
   review). Stale dev_csl tests vs dev_csl's own prompt-profile refactor; the
   merge never touches `slime_plugins/drkernel/`. (They skip on the login host
   for lack of transformers, which hid the staleness.) FIXED 2026-06-01: the 4
   tests + `_render_review_sample` now pin the explicit `drkernel_v1` profile
   via `_new_prompt_renderer_with_profile("drkernel_v1")`; all 17 pass on .16.
2. `test_sglang_config::test_update_weights_default_true`: 1 failed — the test
   calls only `SglangConfig.from_yaml()` and asserts `update_weights is True`,
   but `from_yaml` leaves it `None` (`sglang_config.py:177`); the default
   inference moved to `resolve(args)` (`sglang_config.py:90-100`, commit #1665)
   which the test never calls. Stale upstream test-vs-impl mismatch, pure
   origin/main (dev_csl never touched `sglang_config.py`). NOT in any CI gate
   (`tests/utils/test_sglang_config.py` is not in `pr-test.yml.j2`; only the GPU
   `test_*_sglang_config*.py` integration files are), which is why it can fail
   upstream and still be merged. Not a merge regression; will NOT show red in
   normal CI — only a manual run of that file surfaces it.

Note: earlier apparent "hangs" and `mask_utils`/`megatron_role_config`
"TypeError" failures were a harness artifact — a `pkill -f 'pytest...'` pattern
that matched and killed its own SSH shell, leaving orphaned pytest procs that
contended and stalled/poisoned subsequent runs. Clean isolated runs pass.

## Recommended validation before launch

- Run the CPU/unit test subset (incl. the new `tests/test_*` and
  `tests/utils/*`) to catch import/API breakage from the 191-file delta.
- Run the DrKernel prompt-template + eval-throttle unit tests (no GPU needed).
- A short DrKernel eval smoke on `.22` (e.g. `scripts/debug/debug.27b.sh`
  scaled down) to confirm rollout → reward → loss path still runs and
  loss/metric aggregation is sane under the new group_id path.
- Dispatch a codex xhigh review of the merge diff focused on the model.py
  loss-aggregation + group_id interaction (per AGENTS substantial-change rule).

## 可应用的上游新特性 / 新语义（我方实现的改造点）

合并进来的 origin/main 带来若干新能力，DrKernel / Qwen3.6-27B 现有实现可改造以采用（按对训练/提速的价值排序；均为可选改造，非合并必须项）：

1. **多轮训练 + `group_id` / `group_mask_sums` 损失聚合（最直接）。**
   现状：训练 rollout（`slime_plugins/drkernel/rollout.py` 的 `generate_rollout_async`）是单轮，多轮折叠（`multi_turn_gamma` / `filter_by_last_turn` / `padding_turns`）尚未实现，多轮只在 eval 路径。改造：实现多轮训练时按上游 compact 契约输出 per-turn `list[Sample]`，并给同一条 rollout 的各 turn 设同一 `Sample.group_id`；这样上游新的 `reduce_train_step_metrics` / `group_mask_sums`（`model.py:602`）会按组归一损失（一条 rollout 计一次而非 N 次），正确支撑多轮 RL。

2. **用 `group_id` / `group_index` 取代 `sample.index` 分组。**
   现状：`rollout.py:250` 用 `basis = sample.index ... else rollout_id` 做模板分配/分组。改造：迁移到上游 `group_index`（数据源按 prompt 设置，用于 GRPO 优势中心化）与 `group_id`（损失聚合）语义；不要再读 `Sample.rollout_id`（已废弃，读会抛 `AttributeError`）。

3. **FlashQLA GDN backend（Qwen GDN 加速，关联 rollout 提速）。**
   上游新增 `--qwen-gdn-backend {fla,flashqla}`（`arguments.py:118`，`slime_plugins/models/qwen3_next.py:145`）。Qwen3.6-27B 是 GDN/hybrid，训练侧可试 `flashqla` 加速 GDN 层（与 `handoffs/in_progress/handoff_rollout_speedup.md` 方向一致）。

4. **Delta 权重同步（disk + nccl）。**
   上游新增 `UpdateWeightFromDistributedDelta`（`actor.py:143`）。训练→rollout 引擎的权重刷新可改用 delta 同步以降低开销（同样关联 rollout 提速）。

5. **HF 直接导出 / bridge（可能顺带解决 `torch_dist`）。**
   上游 `--megatron-to-hf-mode`（非 bridge 走 `save_hf_model_direct`，`model.py:889`）+ `--save-hf`（`arguments.py:797`）。量化/导出链路可改用这套官方 HF 导出，并评估 bridge 模式能否免去单独的 `torch_dist` 转换（当前 eval 不需要 `torch_dist`，但训练 `--ref-load .../torch_dist` 需要它）。

6. **`forge_load` 回放 dump 的 rollout（加速 eval/调试迭代）。**
   上游 `slime.rollout.forge_load.generate_rollout` + `--load-forge-rollout-data`（`slime/rollout/forge_load.py`）。可用已 dump 的 `eval_0.pt` 在 SGLang 存活下快速复现/调试，无需重跑生成。

7. **（较大重构，可选）上游 agent RL 框架。**
   上游新增 `slime/agent/`（adapters / sandbox / trajectory）+ `examples/coding_agent_rl`。DrKernel 多轮自定义 rollout 长期可考虑迁移到该框架复用 sandbox/trajectory，但工作量较大。

## State / next steps

- [ ] Decide validation depth (see above) — **not done yet**.
- [ ] Resolve the group_id/group_mask_sums question for the custom rollout.
- [ ] Push `dev_csl` to `csl/dev_csl` only after validation (currently 67 ahead,
      unpushed). Merge commit is local-only and can be reset if rejected:
      `git reset --hard d00773dd` (dev_csl pre-merge tip).
