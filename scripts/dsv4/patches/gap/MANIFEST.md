# Config-management gap patches (Phase 2 snapshot)

Retained snapshot of the **19 unresolved fork edits on
`node53_slime:/sgl-workspace/sglang` that are captured in no release patch** —
the config-management gap from `handoffs/deepseek-v4/runtime_upgrade_eval.md`
section 3. The original snapshot covered 27 edits; the four-file FP4 group was
absorbed upstream and the four-file multimodal group was obsolete, so those two
patches were removed on 2026-08-01.

- Source tree: `node53_slime:/sgl-workspace/sglang`, HEAD `28b095c01005d4a3a2a5b637b7d028b07fba31b2`
  (= the fork pin, `v0.5.13` cherry-pick line). Read-only; not modified.
- Generation: `git diff HEAD -- <files>` per group. All 19 retained gap files were **staged**
  (`M ` in porcelain, no further worktree drift), so `git diff HEAD` captures each
  edit in full and is applies-clean against a pristine `28b095c` tree by construction.
- Retained total: **19 files, +798 / -82**.

## Validation status — ALL CLEAN

- **Pin base fidelity.** node53's HEAD blob for a 3-file retained sample
  (`disaggregation/utils.py`, `observability/req_time_stats.py`,
  `speculative/eagle_draft_cuda_graph_runner.py`)
  is **byte-identical (sha256)** to `github raw` at `28b095c` — the diff base is the
  true upstream pin, not a locally mutated copy.
- **apply-check.** Pristine `28b095c` blobs of all 19 retained files were reconstructed via
  `git archive HEAD` (the shallow upstream scratch clone lacks the pin object, so the
  pin blobs came straight from node53's HEAD). Every artifact is
  **`git apply --check` CLEAN** against that pristine tree, with a `git apply --stat`
  line count matching its `git diff HEAD --numstat`.
- **py_compile.** All 19 retained files, after applying all 3 artifacts, **py_compile OK**.

Reproduce:
```
# reconstruct pristine pin tree (read-only on node53)
ssh node53_slime "git -C /sgl-workspace/sglang archive HEAD -- <27 paths>" | tar -x -C pin/
cp -r pin/. patched/
cd patched && for p in ../scripts/dsv4/patches/gap/gap_*.patch; do git apply --check "$p"; done
```

## Artifacts

3-way verdict column = the eval-doc's assessment of the edit **vs upstream
`692c5f7d`** (the migration target), i.e. what Phase 2 should do with each group
after this snapshot. It is NOT the apply-check result above (which is vs the pin).

### gap_disaggregation.patch — 5 files, +322 / -60
Provenance category (eval §3): **Disaggregation (PD)**.
3-way verdict: **PORT-NEEDED (genuine 3-way conflict) — highest-value un-captured body.**
Fork-specific; `utils.py` fork edit 161 lines AND upstream changed 101 lines → must
port onto upstream's changed file. Real risk; the largest un-captured fork body.

| file | +/- |
|---|---|
| `python/sglang/srt/disaggregation/base/conn.py` | +1 / -0 |
| `python/sglang/srt/disaggregation/decode.py` | +75 / -2 |
| `python/sglang/srt/disaggregation/mooncake/conn.py` | +57 / -21 |
| `python/sglang/srt/disaggregation/prefill.py` | +64 / -1 |
| `python/sglang/srt/disaggregation/utils.py` | +125 / -36 |

### gap_scheduler_infra.patch — 12 files, +469 / -20
Provenance category (eval §3): **Scheduler / managers / mem-cache infra**.
3-way verdict: **MIXED — classify per-file during Phase 2 (port-needed, TBD).**
Mixed infra edits; the eval doc explicitly defers per-file classification to Phase 2.
Bulk of the body is `observability/req_time_stats.py` (+220) and
`scheduler_components/load_inquirer.py` (+83).

| file | +/- |
|---|---|
| `python/sglang/srt/entrypoints/engine.py` | +15 / -0 |
| `python/sglang/srt/entrypoints/http_server.py` | +29 / -7 |
| `python/sglang/srt/managers/io_struct.py` | +54 / -1 |
| `python/sglang/srt/managers/tp_worker.py` | +10 / -0 |
| `python/sglang/srt/managers/scheduler_components/load_inquirer.py` | +83 / -0 |
| `python/sglang/srt/managers/scheduler_components/output_streamer.py` | +1 / -1 |
| `python/sglang/srt/managers/scheduler_components/profiler_manager.py` | +1 / -1 |
| `python/sglang/srt/managers/scheduler_components/weight_updater.py` | +44 / -0 |
| `python/sglang/srt/mem_cache/hiradix_cache.py` | +2 / -3 |
| `python/sglang/srt/mem_cache/memory_pool.py` | +6 / -3 |
| `python/sglang/srt/mem_cache/radix_cache.py` | +4 / -3 |
| `python/sglang/srt/observability/req_time_stats.py` | +220 / -1 |

### gap_spec_infra.patch — 2 files, +7 / -2
Provenance category (eval §3): **Spec-decode infra**.
3-way verdict: **PORT-NEEDED with partial-obsolescence risk — diff against DSpark.**
EAGLE infra; overlaps upstream's DSpark rework. Small (+7/-2); diff intent against the
DSpark spec-decode changes before porting — parts may be superseded.

| file | +/- |
|---|---|
| `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py` | +6 / -2 |
| `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py` | +1 / -0 |

## Excluded: the two files the task named as "untracked" — investigated, NOT a gap

The Phase-2 task asked to fold `srt/configs/deepseek_v4.py` and
`srt/layers/quantization/mxfp4_marlin_moe.py` into the since-removed
`gap_fp4_w4a16_glue.patch` as
full-file additions (`git diff --no-index /dev/null`), on the premise they are
**untracked** new files. On node53's live tree that premise does not hold:

- Both are **tracked and clean at the pin `28b095c`** (last touched by the pin commit
  `28b095c` itself; `git status` shows no edit; the only untracked non-.bak entries in
  the whole tree are the 4 `.bak` files). They are **not** in the eval-doc's 27-file
  gap set, and a container rebuild that checks out `28b095c` **keeps them** — so they
  are not a config-management gap.
- Both also **exist in upstream `692c5f7d`** (`configs/deepseek_v4.py` identical
  110 lines; `mxfp4_marlin_moe.py` differs — pin 229 vs upstream 180 lines).

Emitting them as `/dev/null → file` additions was therefore **rejected**: it would
make `gap_fp4_w4a16_glue.patch` **fail** `git apply --check` against a pristine
`28b095c` tree (the files already exist there), and there is no uncommitted edit to
capture. The only migration-relevant item is the **pin↔upstream 49-line drift** in
`mxfp4_marlin_moe.py`, which is a `28b095c → 692c5f7d` rebase reconcile (handle during
the base-image swap), not a working-tree snapshot. Flagged for the team lead.
