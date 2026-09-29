#!/usr/bin/env bash
# Invoke once in the rollout container after the final model and eval Ray cluster are ready.
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
SUITE_ROOT="$REPO_ROOT/experiments/qwen38_b300_baseline_t1_effrefcov/eval_kernelbench_gepav2"
[[ -f "$SUITE_ROOT/provenance/training_complete.json" ]]
[[ -f "$SUITE_ROOT/provenance/hf_complete.json" ]]
mkdir -p "$SUITE_ROOT/logs"
exec 9>"$SUITE_ROOT/.suite.lock"
flock -n 9 || { echo 'Evaluation suite is already running' >&2; exit 1; }
level=0
trap 'printf "failed at level %s\n" "$level" > "$SUITE_ROOT/status.txt"' ERR
for level in 1 2 3; do
  printf 'running level %s\n' "$level" > "$SUITE_ROOT/status.txt"
  mkdir -p "$SUITE_ROOT/level$level/logs"
  LEVEL="$level" bash "$REPO_ROOT/examples/kernel_agent/eval/qwen38_b300_baseline_t1_effrefcov_gepav2.sh" \
    >> "$SUITE_ROOT/level$level/logs/launcher.log" 2>&1
done
[[ -f "$SUITE_ROOT/validated_summary.json" ]]
printf 'complete\n' > "$SUITE_ROOT/status.txt"
