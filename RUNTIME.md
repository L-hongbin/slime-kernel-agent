# RUNTIME

Stable runtime facts for this repo. Keep one-off checkpoints, Ray job IDs,
scores, ablation notes, and experiment-specific settings in handoffs instead
of this file.

## Shared Paths

- Repo root: `/nfs/FM/chenshuailin/projects/kernel_agents/slime`
- Shared filesystems: `/nfs` and `/ms`
- Slime-format train parquet: `data/drkernel-rl-data-0513/train.parquet`
- KernelBench L1 validation parquet: `data/kernelbench-level1-validation/train.parquet`

## Nodes

- A800 node: `ssh -p 17708 root@192.168.16.22`
- A100 node: `ssh -p 23422 root@192.168.16.16`

## SGLang

- `.22` SGLang checkout: `/sgl-workspace/sglang`; use
  `PYTHONPATH=/sgl-workspace/sglang/python` for direct SGLang commands.

## Reward Services

- `http://192.168.16.40:20111`
- `http://192.168.16.39:8111`

## Common Entrypoints

- Generic debug/eval harness: `scripts/debug.sh`
- KernelGym eval summarizer:
  `scripts/drkernel/summarize_kernelgym_eval.py`

## Placement Rules

- Store stable runtime facts here.
- Store experiment results, special settings, checkpoint paths, logs, and
  conclusions in `handoffs/`.
- Keep `INDEX.md` as the concise map to important scripts, handoffs, and
  reviewable artifacts.
