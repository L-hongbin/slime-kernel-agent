#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
EXP_ROOT=${EXP_ROOT:-$REPO_ROOT/local_artifacts/qwen38_b300_r9}
TENSORBOARD_PORT=${TENSORBOARD_PORT:-6006}
mkdir -p "$EXP_ROOT/logs" "$EXP_ROOT/tensorboard"
printf '%s\n' "$$" > "$EXP_ROOT/logs/tensorboard.pid"
exec tensorboard \
   --logdir_spec "training:$EXP_ROOT/tensorboard,validation:$EXP_ROOT/validation" \
   --host 0.0.0.0 --port "$TENSORBOARD_PORT" --reload_interval 30 \
   > "$EXP_ROOT/logs/tensorboard_server.log" 2>&1
