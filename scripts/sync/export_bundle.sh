#!/usr/bin/env bash
set -euo pipefail

N=${1:-5}

if [[ ! "$N" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: $0 [positive_commit_count]" >&2
  exit 2
fi

git rev-parse --verify "HEAD~${N}" >/dev/null
git bundle create "latest_${N}.bundle" "HEAD~${N}..HEAD"
