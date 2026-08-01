#!/usr/bin/env bash
# One-command sync and verification for the active V4 fleet.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: scripts/sync/sync_v4_active_nodes.sh [--dry-run | --check] [--target HOST]...

With no --target flags, concurrently sync the node64 checkout to all active
targets: node69_slime, node53_dspark, and node70_dspark. After a real sync,
verify a deterministic runtime-code fingerprint on every target.

Options:
  --dry-run      Preview each sync; do not modify or verify remote checkouts.
  --check        Do not sync; only compare local and remote fingerprints.
  --target HOST  Operate on a whitelisted subset (may be repeated).
  -h, --help     Show this help.

Environment:
  MULTI_NODE_SSH_OPTS          Extra ssh options.
  SYNC_V4_EXPECTED_SOURCE_HOST Expected short hostname (default: node64).
  SYNC_V4_TILEKERNELS_DIR      TileKernels source/destination path.
EOF
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
sync_script=${SYNC_V4_RSYNC_SCRIPT:-${script_dir}/rsync_project.sh}
fingerprint_script=${SYNC_V4_FINGERPRINT_SCRIPT:-${script_dir}/runtime_fingerprint.sh}
tilekernels_dir=${SYNC_V4_TILEKERNELS_DIR:-/nfs/FM/chenshuailin/projects/kernel_agents/TileKernels}

mode=sync
targets=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      [[ "${mode}" == sync ]] || { echo "sync_v4: --dry-run and --check are mutually exclusive" >&2; exit 2; }
      mode=dry-run
      shift
      ;;
    --check)
      [[ "${mode}" == sync ]] || { echo "sync_v4: --dry-run and --check are mutually exclusive" >&2; exit 2; }
      mode=check
      shift
      ;;
    --target)
      [[ $# -ge 2 ]] || { echo "sync_v4: --target requires HOST" >&2; exit 2; }
      targets+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "sync_v4: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

expected_source=${SYNC_V4_EXPECTED_SOURCE_HOST:-node64}
source_host=$(hostname -s)
if [[ "${source_host}" != "${expected_source}" ]]; then
  echo "sync_v4: refusing source host ${source_host}; expected ${expected_source}" >&2
  exit 2
fi

[[ -x "${sync_script}" ]] || { echo "sync_v4: sync helper is not executable: ${sync_script}" >&2; exit 2; }
[[ -r "${fingerprint_script}" ]] || { echo "sync_v4: fingerprint helper is not readable: ${fingerprint_script}" >&2; exit 2; }

allowed_targets=(node69_slime node53_dspark node70_dspark)
if [[ ${#targets[@]} == 0 ]]; then
  targets=("${allowed_targets[@]}")
fi

is_allowed_target() {
  local candidate=$1 allowed
  for allowed in "${allowed_targets[@]}"; do
    [[ "${candidate}" == "${allowed}" ]] && return 0
  done
  return 1
}

declare -A seen=()
for target in "${targets[@]}"; do
  if ! is_allowed_target "${target}"; then
    echo "sync_v4: refusing non-active target: ${target}" >&2
    exit 2
  fi
  if [[ -n "${seen[${target}]:-}" ]]; then
    echo "sync_v4: duplicate target: ${target}" >&2
    exit 2
  fi
  seen[${target}]=1
done

ssh_cmd=(ssh)
if [[ -n "${MULTI_NODE_SSH_OPTS:-}" ]]; then
  read -r -a extra_ssh_opts <<<"${MULTI_NODE_SSH_OPTS}"
  ssh_cmd+=("${extra_ssh_opts[@]}")
fi

work_dir=$(mktemp -d)
source_provenance_path=${repo_root}/.slime-v4-source-provenance
tile_provenance_path=${tilekernels_dir}/.slime-v4-source-provenance
created_source_provenance=0
created_tile_provenance=0
cleanup() {
  [[ "${created_source_provenance}" == 0 ]] || rm -f -- "${source_provenance_path}"
  [[ "${created_tile_provenance}" == 0 ]] || rm -f -- "${tile_provenance_path}"
  rm -rf -- "${work_dir}"
}
trap cleanup EXIT
source_manifest=${work_dir}/source-runtime.manifest
bash "${fingerprint_script}" --manifest "${repo_root}" >"${source_manifest}"
local_fingerprint=$(sha256sum "${source_manifest}" | cut -d' ' -f1)
source_git_revision=$(git -C "${repo_root}" rev-parse --verify HEAD) || {
  echo "sync_v4: cannot resolve source git revision on ${source_host}" >&2
  exit 2
}
[[ "${source_git_revision}" =~ ^[0-9a-f]{40,64}$ ]] || {
  echo "sync_v4: malformed source git revision: ${source_git_revision}" >&2
  exit 2
}
provenance_file=${work_dir}/source-provenance
printf '%s\n' \
  'format=slime-v4-source-provenance-v1' \
  "git_revision=${source_git_revision}" \
  "runtime_source_manifest_sha256=${local_fingerprint}" \
  >"${provenance_file}"
provenance_sha256=$(sha256sum "${provenance_file}" | cut -d' ' -f1)

[[ -d "${tilekernels_dir}" ]] || {
  echo "sync_v4: TileKernels source is missing: ${tilekernels_dir}" >&2
  exit 2
}
tile_manifest=${work_dir}/tilekernels.manifest
bash "${fingerprint_script}" --tree-manifest "${tilekernels_dir}" >"${tile_manifest}"
tile_fingerprint=$(sha256sum "${tile_manifest}" | cut -d' ' -f1)
tile_git_revision=$(git -C "${tilekernels_dir}" rev-parse --verify HEAD) || {
  echo "sync_v4: cannot resolve source TileKernels git revision" >&2
  exit 2
}
[[ "${tile_git_revision}" =~ ^[0-9a-f]{40,64}$ ]] || {
  echo "sync_v4: malformed TileKernels git revision: ${tile_git_revision}" >&2
  exit 2
}
tile_provenance_file=${work_dir}/tilekernels-provenance
printf '%s\n' \
  'format=slime-v4-source-provenance-v1' \
  "git_revision=${tile_git_revision}" \
  "runtime_source_manifest_sha256=${tile_fingerprint}" \
  >"${tile_provenance_file}"
tile_provenance_sha256=$(sha256sum "${tile_provenance_file}" | cut -d' ' -f1)
if [[ "${mode}" != check ]]; then
  cp -- "${provenance_file}" "${source_provenance_path}"
  created_source_provenance=1
  cp -- "${tile_provenance_file}" "${tile_provenance_path}"
  created_tile_provenance=1
fi
echo "sync_v4: mode=${mode} source=${source_host}:${repo_root} git_revision=${source_git_revision} source_manifest=${local_fingerprint} tile_revision=${tile_git_revision} tile_manifest=${tile_fingerprint}"
echo "sync_v4: targets=${targets[*]}"

run_target() {
  local target=$1
  if [[ "${mode}" == dry-run ]]; then
    RSYNC_PROJECT_DRY_RUN=1 "${sync_script}" "${target}" "${repo_root}"
    RSYNC_PROJECT_DRY_RUN=1 RSYNC_PROJECT_SOURCE_DIR="${tilekernels_dir}" \
      "${sync_script}" "${target}" "${tilekernels_dir}"
    return
  fi

  if [[ "${mode}" == sync ]]; then
    "${sync_script}" "${target}" "${repo_root}"
    RSYNC_PROJECT_SOURCE_DIR="${tilekernels_dir}" \
      "${sync_script}" "${target}" "${tilekernels_dir}"
  fi

  local remote_fingerprint remote_helper_q remote_repo_q remote_digest_q
  remote_helper_q=$(printf '%q' "${repo_root}/scripts/sync/runtime_fingerprint.sh")
  remote_repo_q=$(printf '%q' "${repo_root}")
  remote_digest_q=$(printf '%q' "${local_fingerprint}")
  remote_fingerprint=$("${ssh_cmd[@]}" "${target}" \
    "bash ${remote_helper_q} --verify-manifest ${remote_repo_q} ${remote_digest_q}" \
    < "${source_manifest}")
  if [[ "${remote_fingerprint}" != "${local_fingerprint}" ]]; then
    echo "sync_v4: fingerprint mismatch on ${target}: local=${local_fingerprint} remote=${remote_fingerprint}" >&2
    return 1
  fi
  local remote_provenance_q remote_provenance_sha256
  remote_provenance_q=$(printf '%q' "${repo_root}/.slime-v4-source-provenance")
  remote_provenance_sha256=$("${ssh_cmd[@]}" "${target}" \
    "sha256sum -- ${remote_provenance_q}" | awk '{print $1}') || {
      echo "sync_v4: missing source provenance on ${target}" >&2
      return 1
    }
  if [[ "${remote_provenance_sha256}" != "${provenance_sha256}" ]]; then
    echo "sync_v4: source provenance mismatch on ${target}: local=${provenance_sha256} remote=${remote_provenance_sha256}" >&2
    return 1
  fi

  local remote_tile_fingerprint remote_tile_q remote_tile_digest_q
  remote_tile_q=$(printf '%q' "${tilekernels_dir}")
  remote_tile_digest_q=$(printf '%q' "${tile_fingerprint}")
  remote_tile_fingerprint=$("${ssh_cmd[@]}" "${target}" \
    "bash ${remote_helper_q} --verify-manifest ${remote_tile_q} ${remote_tile_digest_q}" \
    < "${tile_manifest}")
  if [[ "${remote_tile_fingerprint}" != "${tile_fingerprint}" ]]; then
    echo "sync_v4: TileKernels fingerprint mismatch on ${target}: local=${tile_fingerprint} remote=${remote_tile_fingerprint}" >&2
    return 1
  fi

  local remote_tile_provenance_q remote_tile_provenance_sha256
  remote_tile_provenance_q=$(printf '%q' "${tilekernels_dir}/.slime-v4-source-provenance")
  remote_tile_provenance_sha256=$("${ssh_cmd[@]}" "${target}" \
    "sha256sum -- ${remote_tile_provenance_q}" | awk '{print $1}') || {
      echo "sync_v4: missing TileKernels provenance on ${target}" >&2
      return 1
    }
  if [[ "${remote_tile_provenance_sha256}" != "${tile_provenance_sha256}" ]]; then
    echo "sync_v4: TileKernels provenance mismatch on ${target}" >&2
    return 1
  fi
  echo "sync_v4: verified ${target} source_manifest=${remote_fingerprint} git_revision=${source_git_revision} tile_manifest=${remote_tile_fingerprint} tile_revision=${tile_git_revision}"
}

pids=()
for target in "${targets[@]}"; do
  run_target "${target}" >"${work_dir}/${target}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!targets[@]}"; do
  target=${targets[${index}]}
  if wait "${pids[${index}]}"; then
    status=0
  else
    status=$?
    failed=1
  fi
  echo "== ${target} (exit ${status}) =="
  sed "s/^/[${target}] /" "${work_dir}/${target}.log"
done

if [[ "${failed}" != 0 ]]; then
  echo "sync_v4: one or more targets failed" >&2
  exit 1
fi

if [[ "${mode}" == dry-run ]]; then
  echo "sync_v4: dry-run complete; no remote files were changed"
else
  echo "sync_v4: all ${#targets[@]} target(s) match ${local_fingerprint}"
fi
