#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: scripts/sync/rsync_project.sh HOST [REMOTE_DIR]

Sync the current slime checkout to HOST:REMOTE_DIR using a blacklist. REMOTE_DIR
defaults to the same absolute path as the local checkout.

Environment:
  MULTI_NODE_SSH_OPTS       Extra ssh options, also used by rsync.
  RSYNC_PROJECT_DELETE=1    Add --delete for non-excluded remote files.
  RSYNC_PROJECT_DRY_RUN=1   Show what would be synced.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

host=$1
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
remote_dir=${2:-${repo_root}}

ssh_opts=${MULTI_NODE_SSH_OPTS:-}
rsync_rsh="ssh"
if [[ -n "${ssh_opts}" ]]; then
  rsync_rsh+=" ${ssh_opts}"
fi

rsync_args=(
  -az
  --info=stats1,progress2
  --human-readable
  --exclude='/.git'
  --exclude='/__pycache__/'
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='/.pytest_cache/'
  --exclude='/.mypy_cache/'
  --exclude='/.ruff_cache/'
  --exclude='/slime.egg-info/'
  --exclude='/checkpoints/'
  --exclude='/Data/'
  --exclude='/experiments/'
  --exclude='/examples/kernel_agent/logs/'
  --exclude='/wandb/'
  --exclude='/outputs/'
  --exclude='/logs/'
  --exclude='/tmp/'
  --exclude='/build/'
  --exclude='/dist/'
  --exclude='*.egg-info/'
  --exclude='*.log'
)

if [[ "${RSYNC_PROJECT_DELETE:-0}" == "1" ]]; then
  rsync_args+=(--delete)
fi

if [[ "${RSYNC_PROJECT_DRY_RUN:-0}" == "1" ]]; then
  rsync_args+=(--dry-run)
fi

# echo "rsync_project: ${repo_root}/ -> ${host}:${remote_dir}/"
ssh ${ssh_opts} "${host}" "mkdir -p $(printf '%q' "${remote_dir}")"
rsync "${rsync_args[@]}" -e "${rsync_rsh}" "${repo_root}/" "${host}:${remote_dir}/"
