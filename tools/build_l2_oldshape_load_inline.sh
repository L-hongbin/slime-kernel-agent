#!/bin/bash
# Build the level2 OLD-shape (pre-2025-07-02 scale-up, commit 21fbe5a) musa-coder
# load_inline dataset from the extracted old reference files.
set -euo pipefail
REPO=/nfs/FM/chenshuailin/projects/kernel_agents/slime-dev-csl-2
OLD_DIR=/nfs/FM/chenshuailin/staging_oneshot_conv1x1/kernelbench_level2_oldshape
OUT="$REPO/Data/kernelbench-level2-validation-musa-coder-load-inline-oldshape"
cd "$REPO"
mkdir -p "$OUT"
python3 tools/build_oldsize_raw.py \
  --level1-dir "$OLD_DIR" \
  --target-data "$OUT/raw.parquet" \
  --data-source kernelbench_level2_validation_oldshape
python3 tools/convert_prompt_with_template.py \
  --raw-data "$OUT/raw.parquet" \
  --target-data "$OUT/train.parquet" \
  --problem-field ground_truth \
  --template-path examples/kernel_agent/prompt_config/musa_coder.jinja \
  --template-var "backend_display=pybind load_inline" \
  --template-var-file one_shot_example=Data/kernelbench-level1-validation-musa-coder-load-inline/one_shot_example_load_inline.txt
echo "L2 oldshape dataset OK -> $OUT/train.parquet"
