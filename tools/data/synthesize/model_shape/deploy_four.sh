#!/usr/bin/env bash
set -Eeuo pipefail

action=${1:-status}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bundle=/tmp/dsv4-shape-deploy-v10
container=${SHAPE_SERVER_CONTAINER:-csl_dsv4_0731_shape_tp8_low_v10}

hosts=(
  "${SHAPE_NODE53_SSH:-node53}|10.11.2.153|31053|/mnt/md1"
  "${SHAPE_NODE64_SSH:-node64}|10.11.2.164|31064|/mnt/md1"
  "${SHAPE_NODE69_SSH:-node69}|10.11.2.169|31069|/mnt/data"
  "${SHAPE_NODE70_SSH:-node70}|10.11.2.170|31070|/mnt/data"
)

sync_one() {
  local spec=$1 ssh_host=${spec%%|*}
  ssh "${ssh_host}" "mkdir -p '${bundle}/patches'"
  tar -C "${script_dir}" -cf - \
    deploy_one.sh runtime_preflight.py \
    patches/sglang_dsv4_0731_reasoning_effort.patch \
    patches/sglang_dspark_swa_eviction.patch \
    | ssh "${ssh_host}" "tar -C '${bundle}' -xf -"
}

run_one() {
  local spec=$1 remote_action=$2
  local ssh_host rest ip port disk_root
  ssh_host=${spec%%|*}
  rest=${spec#*|}; ip=${rest%%|*}
  rest=${rest#*|}; port=${rest%%|*}; disk_root=${rest##*|}
  echo "[${ssh_host} ${ip}:${port}] ${remote_action}"
  ssh "${ssh_host}" \
    "SHAPE_SERVER_PORT='${port}' SHAPE_SERVER_DISK_ROOT='${disk_root}' SHAPE_SERVER_CONTAINER='${container}' bash '${bundle}/deploy_one.sh' '${remote_action}'"
}

run_all() {
  local remote_action=$1 failed=0
  local pids=() specs=()
  for spec in "${hosts[@]}"; do
    run_one "${spec}" "${remote_action}" &
    pids+=("$!"); specs+=("${spec}")
  done
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "FAILED ${specs[$index]} ${remote_action}" >&2
      failed=1
    fi
  done
  return "${failed}"
}

case "${action}" in
  endpoints)
    for spec in "${hosts[@]}"; do
      rest=${spec#*|}; ip=${rest%%|*}; rest=${rest#*|}; port=${rest%%|*}
      echo "http://${ip}:${port}"
    done
    ;;
  sync)
    for spec in "${hosts[@]}"; do sync_one "${spec}"; done
    ;;
  status)
    for spec in "${hosts[@]}"; do sync_one "${spec}"; done
    run_all status
    ;;
  preflight)
    for spec in "${hosts[@]}"; do sync_one "${spec}"; done
    run_all preflight
    ;;
  start)
    for spec in "${hosts[@]}"; do sync_one "${spec}"; done
    run_all preflight
    run_all start
    ;;
  wait|verify|smoke|stop)
    run_all "${action}"
    ;;
  *)
    echo "usage: $0 {endpoints|sync|status|preflight|start|wait|verify|smoke|stop}" >&2
    exit 2
    ;;
esac
