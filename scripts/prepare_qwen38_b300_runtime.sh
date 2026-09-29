#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO_ROOT/examples/kernel_agent/qwen38_b300_env.sh"
run_rollout_python() {
   local rollout_pythonpath=${PYTHONPATH:-}
   if [[ -n "${SLIME_ROLLOUT_PYTHONPATH:-}" ]]; then
      rollout_pythonpath="$SLIME_ROLLOUT_PYTHONPATH${rollout_pythonpath:+:$rollout_pythonpath}"
   fi
   PYTHONPATH="$rollout_pythonpath" python "$@"
}
read -r SGLANG_VERSION SGLANG_DIR < <(run_rollout_python - <<'PY'
from pathlib import Path
import sglang
print(sglang.__version__, Path(sglang.__file__).resolve().parent)
PY
)
case "$SGLANG_VERSION" in
   0.5.15.post1) SGLANG_REPLAY_MODE=legacy ;;
   0.5.20) SGLANG_REPLAY_MODE=native ;;
   *) echo "No verified replay patch for SGLang $SGLANG_VERSION ($SGLANG_DIR)" >&2; exit 1 ;;
esac
echo "SGLang $SGLANG_VERSION: $SGLANG_DIR"
cd "$(dirname "$SGLANG_DIR")"
if [[ "$SGLANG_REPLAY_MODE" == native ]]; then
   PATCH="$REPO_ROOT/docker/patch/v$SGLANG_VERSION/sglang-sampling-mask-replay.patch"
   if [[ ! -f "$PATCH" ]]; then
      echo "No verified native sampling-mask replay patch for SGLang $SGLANG_VERSION" >&2
      exit 1
   fi
   if patch --dry-run --batch -R -p2 < "$PATCH" >/dev/null 2>&1; then
      echo 'Native sampling-mask replay with EAGLE/NEXTN support is already installed.'
   else
      patch --dry-run --batch -p2 < "$PATCH" >/dev/null
      BACKUP="$B300_RUNTIME/backups/$(date +%Y%m%dT%H%M%S)"
      mkdir -p "$BACKUP"
      sed -n 's|^--- a/python/||p' "$PATCH" | tar -czf "$BACKUP/sglang-0.5.20-before-sampling-mask-replay.tar.gz" -T -
      patch --batch -p2 < "$PATCH"
   fi
else
   PATCH="$REPO_ROOT/docker/patch/v$SGLANG_VERSION/sglang-top_p.patch"
   REPLAY_REUSE_PATCH="$REPO_ROOT/docker/patch/v$SGLANG_VERSION/sglang-top_p-reuse.patch"
   if [[ ! -f "$REPLAY_REUSE_PATCH" ]]; then
      echo "No verified top-p sort reuse patch for SGLang $SGLANG_VERSION" >&2
      exit 1
   fi
   if git apply --reverse --check -p2 "$REPLAY_REUSE_PATCH" 2>/dev/null; then
      echo 'Top-p replay with shared sorting is already installed.'
   else
      if git apply --reverse --check -p2 "$PATCH" 2>/dev/null; then
         echo 'The matching top-p replay patch is already installed.'
      else
         git apply --check -p2 "$PATCH"
         BACKUP="$B300_RUNTIME/backups/$(date +%Y%m%dT%H%M%S)"
         mkdir -p "$BACKUP"
         sed -n 's|^+++ b/python/||p' "$PATCH" | tar -czf "$BACKUP/sglang-before-top-p.tar.gz" -T -
         git apply -p2 "$PATCH"
      fi
      git apply --check -p2 "$REPLAY_REUSE_PATCH"
      BACKUP="$B300_RUNTIME/backups/$(date +%Y%m%dT%H%M%S)"
      mkdir -p "$BACKUP"
      sed -n 's|^--- a/python/||p' "$REPLAY_REUSE_PATCH" | tar -czf "$BACKUP/sglang-before-top-p-reuse.tar.gz" -T -
      git apply -p2 "$REPLAY_REUSE_PATCH"
   fi
   FP32_CACHE_PATCH="$REPO_ROOT/docker/patch/v$SGLANG_VERSION/sglang-fp32-lm-head-cache.patch"
   if [[ ! -f "$FP32_CACHE_PATCH" ]]; then
      echo "No verified FP32 LM head cache patch for SGLang $SGLANG_VERSION" >&2
      exit 1
   fi
   if git apply --reverse --check -p2 "$FP32_CACHE_PATCH" 2>/dev/null; then
      echo 'The matching FP32 LM head cache patch is already installed.'
   else
      git apply --check -p2 "$FP32_CACHE_PATCH"
      BACKUP="$B300_RUNTIME/backups/$(date +%Y%m%dT%H%M%S)"
      mkdir -p "$BACKUP"
      sed -n 's|^--- a/python/||p' "$FP32_CACHE_PATCH" | tar -czf "$BACKUP/sglang-before-fp32-cache.tar.gz" -T -
      git apply -p2 "$FP32_CACHE_PATCH"
   fi
fi
MTP_PATH="$SLIME_MEGATRON_LM_PATH/megatron/core/transformer/multi_token_prediction.py"
if ! python "$REPO_ROOT/scripts/patch_megatron_mtp_hidden_detach.py" --check --path "$MTP_PATH"; then
   mkdir -p "$B300_RUNTIME/backups"
   if [[ ! -e "$B300_RUNTIME/backups/multi_token_prediction.before-hidden-detach.py" ]]; then
      cp "$MTP_PATH" "$B300_RUNTIME/backups/multi_token_prediction.before-hidden-detach.py"
   fi
   python "$REPO_ROOT/scripts/patch_megatron_mtp_hidden_detach.py" --path "$MTP_PATH"
   python "$REPO_ROOT/scripts/patch_megatron_mtp_hidden_detach.py" --check --path "$MTP_PATH"
fi
python - <<'PY'
import os
from pathlib import Path
path = Path(os.environ['SLIME_MEGATRON_LM_PATH']) / 'megatron/core/models/gpt/gpt_model.py'
source = path.read_text()
assert 'mtp_output_weight = mtp_output_weight.detach()' in source, f'Missing output weight detach: {path}'
assert 'weight=mtp_output_weight' in source, f'MTP does not use the detached output weight: {path}'
print(f'MTP output weight detach verified: {path}')
PY
cd "$REPO_ROOT"
python scripts/patch_flashqla_b300.py --path "$SLIME_TRAIN_PACKAGES/flash_qla"
python scripts/patch_flashqla_b300.py --check --path "$SLIME_TRAIN_PACKAGES/flash_qla"
python scripts/patch_flashqla_cpu_metadata.py --path "$SLIME_TRAIN_PACKAGES/flash_qla"
python scripts/patch_flashqla_cpu_metadata.py --check --path "$SLIME_TRAIN_PACKAGES/flash_qla"
python scripts/patch_megatron_mtp_empty_mask.py --path "$SLIME_MEGATRON_LM_PATH/megatron/core/models/gpt/gpt_model.py"
python scripts/patch_megatron_mtp_empty_mask.py --check --path "$SLIME_MEGATRON_LM_PATH/megatron/core/models/gpt/gpt_model.py"
run_rollout_python scripts/check_sglang_top_p_replay.py --check-sort-reuse
run_rollout_python scripts/check_sglang_fp32_lm_head_cache.py
