#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: scripts/sync/rsync_project.sh HOST [REMOTE_DIR]

Sync the current slime checkout to HOST:REMOTE_DIR using a blacklist. REMOTE_DIR
defaults to the same absolute path as the local checkout. The script uses rsync
when it is installed at both ends and otherwise falls back to tar over ssh.

Environment:
  MULTI_NODE_SSH_OPTS       Extra ssh options, also used by rsync.
  RSYNC_PROJECT_SOURCE_DIR  Source tree (default: this slime checkout).
  RSYNC_PROJECT_DELETE=1    Add --delete. Requires rsync at both ends.
  RSYNC_PROJECT_DRY_RUN=1   Show what would be synced without modifying HOST.
  RSYNC_PROJECT_PROGRESS=1  Show per-file aggregate transfer progress.
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
default_repo_root=$(cd -- "${script_dir}/../.." && pwd)
repo_root=$(cd -- "${RSYNC_PROJECT_SOURCE_DIR:-${default_repo_root}}" && pwd)
remote_dir=${2:-${repo_root}}

ssh_cmd=(ssh)
if [[ -n "${MULTI_NODE_SSH_OPTS:-}" ]]; then
  # MULTI_NODE_SSH_OPTS is intentionally a conventional whitespace-separated
  # ssh option list (for example: "-p 23522 -o ConnectTimeout=10").
  read -r -a extra_ssh_opts <<<"${MULTI_NODE_SSH_OPTS}"
  ssh_cmd+=("${extra_ssh_opts[@]}")
fi

shell_quote() {
  local value=$1
  printf "'%s'" "${value//\'/\'\\\'\'}"
}

remote_dir_q=$(shell_quote "${remote_dir}")

# Keep the rsync and tar blacklists together. The tar patterns account for
# tar's leading "./" archive member names while preserving rsync's root-only
# exclusions.
rsync_excludes=(
  '/.git'
  '/__pycache__/'
  '__pycache__/'
  '*.pyc'
  '/.pytest_cache/'
  '/.mypy_cache/'
  '/.ruff_cache/'
  '/slime.egg-info/'
  '/checkpoints/'
  '/Data/'
  '/experiments/'
  '/local_artifacts/'
  '/examples/kernel_agent/logs/'
  '/wandb/'
  '/outputs/'
  '/logs/'
  '/tmp/'
  '/build/'
  '/dist/'
  '*.egg-info/'
  '*.log'
)

tar_excludes=(
  '--exclude=./.git'
  '--exclude=./__pycache__'
  '--exclude=*/__pycache__'
  '--exclude=*.pyc'
  '--exclude=./.pytest_cache'
  '--exclude=./.mypy_cache'
  '--exclude=./.ruff_cache'
  '--exclude=./slime.egg-info'
  '--exclude=./checkpoints'
  '--exclude=./Data'
  '--exclude=./experiments'
  '--exclude=./local_artifacts'
  '--exclude=./examples/kernel_agent/logs'
  '--exclude=./wandb'
  '--exclude=./outputs'
  '--exclude=./logs'
  '--exclude=./tmp'
  '--exclude=./build'
  '--exclude=./dist'
  '--exclude=*.egg-info'
  '--exclude=*.log'
)

# A marker makes a missing remote binary distinguishable from an ssh failure
# and tolerates login banners printed by the remote shell.
if ! probe_output=$("${ssh_cmd[@]}" "${host}" \
  'if command -v rsync >/dev/null 2>&1; then r=1; else r=0; fi; if command -v tar >/dev/null 2>&1; then t=1; else t=0; fi; printf "__SLIME_SYNC_PROBE__ rsync=%s tar=%s\n" "$r" "$t"'); then
  echo "rsync_project: cannot connect to ${host} for capability probe" >&2
  exit 1
fi
probe_marker=$(printf '%s\n' "${probe_output}" | sed -n 's/^__SLIME_SYNC_PROBE__ /__SLIME_SYNC_PROBE__ /p' | tail -n 1)
case "${probe_marker}" in
  '__SLIME_SYNC_PROBE__ rsync=1 tar=1') remote_rsync=1; remote_tar=1 ;;
  '__SLIME_SYNC_PROBE__ rsync=1 tar=0') remote_rsync=1; remote_tar=0 ;;
  '__SLIME_SYNC_PROBE__ rsync=0 tar=1') remote_rsync=0; remote_tar=1 ;;
  '__SLIME_SYNC_PROBE__ rsync=0 tar=0') remote_rsync=0; remote_tar=0 ;;
  *)
    echo "rsync_project: malformed capability response from ${host}" >&2
    exit 1
    ;;
esac

local_rsync=0
command -v rsync >/dev/null 2>&1 && local_rsync=1

if [[ "${local_rsync}" == 1 && "${remote_rsync}" == 1 ]]; then
  backend=rsync
else
  backend=tar
  command -v tar >/dev/null 2>&1 || {
    echo "rsync_project: neither rsync nor local tar is available" >&2
    exit 1
  }
  if [[ "${remote_tar}" != 1 ]]; then
    echo "rsync_project: ${host} has neither rsync nor tar" >&2
    exit 1
  fi
  if [[ "${RSYNC_PROJECT_DELETE:-0}" == 1 ]]; then
    echo "rsync_project: RSYNC_PROJECT_DELETE=1 requires rsync at both ends; refusing an inexact tar fallback" >&2
    exit 2
  fi
fi

echo "rsync_project: backend=${backend} source=${repo_root}/ target=${host}:${remote_dir}/"

if [[ "${backend}" == rsync ]]; then
  # Keep the default output suitable for the three-target fleet wrapper. A
  # progress2 stream contains carriage-return updates for every target and
  # becomes thousands of noisy characters even when only one tiny file moves.
  rsync_args=(-az --info=stats1 --human-readable)
  if [[ "${RSYNC_PROJECT_PROGRESS:-0}" == 1 ]]; then
    rsync_args+=(--info=progress2)
  fi
  for pattern in "${rsync_excludes[@]}"; do
    rsync_args+=(--exclude="${pattern}")
  done
  if [[ "${RSYNC_PROJECT_DELETE:-0}" == 1 ]]; then
    rsync_args+=(--delete)
  fi
  if [[ "${RSYNC_PROJECT_DRY_RUN:-0}" == 1 ]]; then
    rsync_args+=(--dry-run)
  else
    "${ssh_cmd[@]}" "${host}" "mkdir -p -- ${remote_dir_q}"
  fi

  printf -v rsync_rsh '%q ' "${ssh_cmd[@]}"
  rsync "${rsync_args[@]}" -e "${rsync_rsh% }" "${repo_root}/" "${host}:${remote_dir}/"
  exit 0
fi

if [[ "${RSYNC_PROJECT_DRY_RUN:-0}" == 1 ]]; then
  echo "rsync_project: tar fallback dry-run; archive members follow"
  tar -C "${repo_root}" "${tar_excludes[@]}" -cf - . | tar -tf -
  exit 0
fi

"${ssh_cmd[@]}" "${host}" "mkdir -p -- ${remote_dir_q}"
tar -C "${repo_root}" "${tar_excludes[@]}" -cf - . \
  | "${ssh_cmd[@]}" "${host}" "tar -C ${remote_dir_q} -xf -"
