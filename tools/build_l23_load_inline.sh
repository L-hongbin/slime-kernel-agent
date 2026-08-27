#!/bin/bash
# Build level2/level3 musa-coder load_inline datasets from the raw validation parquets.
set -euo pipefail
REPO=/nfs/FM/chenshuailin/projects/kernel_agents/slime-dev-csl-2
cd "$REPO"
for lvl in 2 3; do
  out="Data/kernelbench-level${lvl}-validation-musa-coder-load-inline"
  mkdir -p "$out"
  python3 tools/convert_prompt_with_template.py \
    --raw-data "Data/kernelbench-level${lvl}-validation/train.parquet" \
    --target-data "$out/train.parquet" \
    --problem-field ground_truth \
    --template-path examples/kernel_agent/prompt_config/musa_coder.jinja \
    --template-var "backend_display=pybind load_inline" \
    --template-var-file one_shot_example=Data/kernelbench-level1-validation-musa-coder-load-inline/one_shot_example_load_inline.txt
  echo "level${lvl} converted OK -> $out/train.parquet"
done
