#!/usr/bin/env bash
# Delete the SAME absolute path(s) on every node in the hostfile.
#
# Per-node local /nfs/FM means a checkpoint dir exists independently on each
# node; this batch-removes one path across all of them. DRY-RUN by default —
# it only prints sizes. Set APPLY=1 to actually delete.
#
# Usage:
#   scripts/sync/rm_across_hosts.sh PATH [PATH...]            # dry-run (preview)
#   APPLY=1 scripts/sync/rm_across_hosts.sh PATH [PATH...]    # really delete
#
# Env:
#   HOSTFILE              hostfile path (default: <repo>/hostfile)
#   SSH_PORT              ssh port for worker nodes (default: 23422)
#   MULTI_NODE_SSH_OPTS   extra ssh opts (appended after -p SSH_PORT)
#   APPLY=1               perform deletion (otherwise dry-run)
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
HOSTFILE=${HOSTFILE:-${repo_root}/hostfile}
SSH_PORT=${SSH_PORT:-23522}
APPLY=${APPLY:-0}
ssh_base=(-p "${SSH_PORT}" -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new ${MULTI_NODE_SSH_OPTS:-})

if [[ $# -lt 1 ]]; then
  echo "usage: [APPLY=1] $0 PATH [PATH...]" >&2
  exit 2
fi
[[ -f "${HOSTFILE}" ]] || { echo "error: hostfile not found: ${HOSTFILE}" >&2; exit 2; }

# --- safety: refuse obviously dangerous paths ---
for p in "$@"; do
  case "$p" in
    /|/root|/root/|/nfs|/nfs/|/nfs/FM|/nfs/FM/) echo "REFUSE dangerous path: $p" >&2; exit 3;;
    *..*) echo "REFUSE path containing '..': $p" >&2; exit 3;;
  esac
  [[ "$p" = /* ]] || { echo "REFUSE non-absolute path: $p (pass absolute paths)" >&2; exit 3; }
  [[ "$p" == *checkpoints/* || "$p" == *experiments/* ]] || { echo "REFUSE path not under checkpoints/ or experiments/: $p" >&2; exit 3; }
done

# --- local node IPs, to run locally instead of ssh ---
mapfile -t local_ips < <(hostname -I 2>/dev/null | tr ' ' '\n' | grep -v '^$'; echo 127.0.0.1)
is_local() { local a=$1; for ip in "${local_ips[@]}"; do [[ "$a" == "$ip" ]] && return 0; done; return 1; }

# --- parse hostfile: "ssh_target [addr=IP]" ; first non-comment line is head ---
hosts=()  # entries: "ssh_target|node_addr"
while IFS= read -r raw; do
  line=${raw%%#*}; line=$(echo "$line" | xargs || true)
  [[ -z "$line" ]] && continue
  ssh_target=${line%% *}
  node_addr=$ssh_target
  for tok in $line; do [[ "$tok" == addr=* ]] && node_addr=${tok#addr=}; done
  hosts+=("${ssh_target}|${node_addr}")
done < "${HOSTFILE}"

echo "hostfile : ${HOSTFILE}  (${#hosts[@]} nodes)"
echo "mode     : $([[ "${APPLY}" == 1 ]] && echo 'APPLY (will delete)' || echo 'DRY-RUN (preview only)')"
echo "paths    : $*"
echo

# build a remote snippet that, per path, prints size then optionally deletes
remote_cmd() {
  local apply=$1; shift
  local s="set -e; total_label='';"
  for p in "$@"; do
    s+="if [ -e '$p' ]; then sz=\$(du -sh '$p' 2>/dev/null | cut -f1); echo \"  [present \$sz] $p\";"
    if [[ "$apply" == 1 ]]; then
      s+=" rm -rf '$p' && echo \"    -> deleted\";"
    fi
    s+=" else echo \"  [absent] $p\"; fi;"
  done
  s+="echo \"  df: \$(df -h /nfs/FM | tail -1)\";"
  echo "$s"
}

rc=0
for entry in "${hosts[@]}"; do
  ssh_target=${entry%%|*}; node_addr=${entry##*|}
  if is_local "$node_addr" || is_local "$ssh_target"; then
    echo "== ${ssh_target} (local) =="
    bash -c "$(remote_cmd "${APPLY}" "$@")" || rc=1
  else
    echo "== ${ssh_target} (ssh ${node_addr}) =="
    ssh "${ssh_base[@]}" "$ssh_target" "$(remote_cmd "${APPLY}" "$@")" || rc=1
  fi
  echo
done

if [[ "${APPLY}" != 1 ]]; then
  echo "DRY-RUN only. Re-run with APPLY=1 to delete."
fi
exit $rc
