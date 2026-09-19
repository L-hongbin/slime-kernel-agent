#!/usr/bin/env bash
# Install the version-matched replay patch and verify MTP isolation in this container.
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO_ROOT/examples/kernel_agent/qwen38_h20_env.sh"
read -r SGLANG_VERSION SGLANG_DIR < <(python - <<'PY'
from pathlib import Path
import sglang
print(sglang.__version__, Path(sglang.__file__).resolve().parent)
PY
)
case "$SGLANG_VERSION" in
   0.5.15.post1|0.5.16) PATCH="$REPO_ROOT/docker/patch/v$SGLANG_VERSION/sglang-top_p.patch" ;;
   *) echo "No verified replay patch for SGLang $SGLANG_VERSION ($SGLANG_DIR)" >&2; exit 1 ;;
esac
echo "SGLang $SGLANG_VERSION: $SGLANG_DIR"
cd "$(dirname "$SGLANG_DIR")"
REPLAY_REUSE_PATCH="$REPO_ROOT/docker/patch/v$SGLANG_VERSION/sglang-top_p-reuse.patch"
if [[ ! -f "$REPLAY_REUSE_PATCH" ]]; then
   echo "No verified top-p sort reuse patch for SGLang $SGLANG_VERSION" >&2
   exit 1
fi
# The incremental patch changes context in the base replay patch. Check it
# first, so a second preparation is idempotent after both are installed.
if git apply --reverse --check -p2 "$REPLAY_REUSE_PATCH" 2>/dev/null; then
   echo 'Top-p replay with shared sorting is already installed.'
else
   if git apply --reverse --check -p2 "$PATCH" 2>/dev/null; then
      echo 'The matching top-p replay patch is already installed.'
   else
      git apply --check -p2 "$PATCH"
      BACKUP="$H20_RUNTIME/backups/$(date +%Y%m%dT%H%M%S)"
      mkdir -p "$BACKUP"
      # Preserve only files touched by this patch; paths are supplied by the trusted repo patch.
      sed -n 's|^+++ b/python/||p' "$PATCH" | tar -czf "$BACKUP/sglang-before-top-p.tar.gz" -T -
      git apply -p2 "$PATCH"
   fi
   git apply --check -p2 "$REPLAY_REUSE_PATCH"
   BACKUP="$H20_RUNTIME/backups/$(date +%Y%m%dT%H%M%S)"
   mkdir -p "$BACKUP"
   sed -n 's|^--- a/python/||p' "$REPLAY_REUSE_PATCH" | tar -czf "$BACKUP/sglang-before-top-p-reuse.tar.gz" -T -
   git apply -p2 "$REPLAY_REUSE_PATCH"
fi
# The FP32 cache patch is separate from replay and matched to the installed
# weight-updater layout. Never apply the 0.5.16 patch to a different version.
FP32_CACHE_PATCH="$REPO_ROOT/docker/patch/v$SGLANG_VERSION/sglang-fp32-lm-head-cache.patch"
if [[ ! -f "$FP32_CACHE_PATCH" ]]; then
   echo "No verified FP32 LM head cache patch for SGLang $SGLANG_VERSION" >&2
   exit 1
fi
if git apply --reverse --check -p2 "$FP32_CACHE_PATCH" 2>/dev/null; then
   echo 'The matching FP32 LM head cache patch is already installed.'
else
   git apply --check -p2 "$FP32_CACHE_PATCH"
   BACKUP="$H20_RUNTIME/backups/$(date +%Y%m%dT%H%M%S)"
   mkdir -p "$BACKUP"
   # Old paths omit the new cache module, which has no pre-patch file.
   sed -n 's|^--- a/python/||p' "$FP32_CACHE_PATCH" | tar -czf "$BACKUP/sglang-before-fp32-cache.tar.gz" -T -
   git apply -p2 "$FP32_CACHE_PATCH"
fi
MTP_PATH="$SLIME_MEGATRON_LM_PATH/megatron/core/transformer/multi_token_prediction.py"
if ! python "$REPO_ROOT/scripts/patch_megatron_mtp_hidden_detach.py" --check --path "$MTP_PATH"; then
   mkdir -p "$H20_RUNTIME/backups"
   if [[ ! -e "$H20_RUNTIME/backups/multi_token_prediction.before-hidden-detach.py" ]]; then
      cp "$MTP_PATH" "$H20_RUNTIME/backups/multi_token_prediction.before-hidden-detach.py"
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
python scripts/check_sglang_top_p_replay.py --check-sort-reuse
python scripts/check_sglang_fp32_lm_head_cache.py
