#!/usr/bin/env bash
# Build or verify a deterministic source manifest for code that can affect a V4
# runtime. Verification is intentionally source-manifest based: stale extra
# files on a destination do not cause a mismatch, while every source path must
# exist remotely with the same type and content.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/sync/runtime_fingerprint.sh [REPO_ROOT]
  scripts/sync/runtime_fingerprint.sh --manifest REPO_ROOT
  scripts/sync/runtime_fingerprint.sh --tree-manifest TREE_ROOT
  scripts/sync/runtime_fingerprint.sh --verify-manifest REPO_ROOT SHA256

The default mode prints the SHA256 of a newly generated manifest. --manifest
writes its NUL-delimited records. --verify-manifest reads those records from
stdin, verifies every listed path, and prints the verified SHA256.
EOF
}

mode=fingerprint
tree_layout=0
case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  --manifest)
    [[ $# == 2 ]] || { usage; exit 2; }
    mode=manifest
    repo_root=$2
    ;;
  --tree-manifest)
    [[ $# == 2 ]] || { usage; exit 2; }
    mode=manifest
    tree_layout=1
    repo_root=$2
    ;;
  --verify-manifest)
    [[ $# == 3 ]] || { usage; exit 2; }
    mode=verify
    repo_root=$2
    expected_manifest_digest=$3
    [[ "${expected_manifest_digest}" =~ ^[0-9a-f]{64}$ ]] || {
      echo "runtime_fingerprint: invalid expected manifest SHA256" >&2
      exit 2
    }
    ;;
  '')
    script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    repo_root=$(cd -- "${script_dir}/../.." && pwd)
    ;;
  *)
    [[ $# == 1 ]] || { usage; exit 2; }
    repo_root=$1
    ;;
esac

command -v sha256sum >/dev/null 2>&1 || {
  echo "runtime_fingerprint: sha256sum is required" >&2
  exit 1
}

cd -- "${repo_root}"
if [[ "${tree_layout}" == 1 ]]; then
  scopes=(.)
else
  scopes=(slime custom_kernels scripts tests examples train.py train_async.py)
fi
if [[ "${mode}" != verify && "${tree_layout}" != 1 ]]; then
  for scope in "${scopes[@]}"; do
    if [[ ! -e "${scope}" ]]; then
      echo "runtime_fingerprint: required scope is missing: ${repo_root}/${scope}" >&2
      exit 1
    fi
  done
fi

emit_manifest() {
  # The ignored paths mirror the project sync blacklist for files inside these
  # runtime scopes. Empty directories intentionally do not affect the manifest.
  if [[ "${tree_layout}" == 1 ]]; then
    LC_ALL=C find . \
      \( -type d \( -path ./.git -o -path ./__pycache__ -o -name __pycache__ -o -path ./.pytest_cache -o -path ./.mypy_cache -o -path ./.ruff_cache -o -path ./slime.egg-info -o -path ./checkpoints -o -path ./Data -o -path ./experiments -o -path ./wandb -o -path ./outputs -o -path ./logs -o -path ./tmp -o -path ./build -o -path ./dist -o -name '*.egg-info' -o -name '*.pyc' -o -name '*.log' \) -prune \) -o \
      \( -type f ! -name '*.pyc' ! -name '*.log' ! -name '.slime-v4-source-provenance' -print0 \) -o \
      \( -type l ! -name '*.pyc' ! -name '*.log' ! -name '.slime-v4-source-provenance' -print0 \) \
      | LC_ALL=C sort -z \
      | emit_paths
    return
  fi
  LC_ALL=C find "${scopes[@]}" \
    \( -type d \( -name __pycache__ -o -name '*.egg-info' -o -name '*.pyc' -o -name '*.log' -o -path examples/kernel_agent/logs \) -prune \) -o \
    \( -type f ! -name '*.pyc' ! -name '*.log' -print0 \) -o \
    \( -type l ! -name '*.pyc' ! -name '*.log' -print0 \) \
    | LC_ALL=C sort -z \
    | emit_paths
}

emit_paths() {
  while IFS= read -r -d '' path; do
      if [[ -L "${path}" ]]; then
        target_digest=$(printf '%s' "$(readlink -- "${path}")" | sha256sum | cut -d' ' -f1)
        printf 'L\0%s\0%s\0' "${path}" "${target_digest}"
      else
        content_digest=$(sha256sum -- "${path}" | cut -d' ' -f1)
        printf 'F\0%s\0%s\0' "${path}" "${content_digest}"
      fi
  done
}

verify_manifest() {
  local kind path expected_digest actual_digest
  while IFS= read -r -d '' kind; do
    IFS= read -r -d '' path || {
      echo "runtime_fingerprint: truncated manifest after record type" >&2
      return 1
    }
    IFS= read -r -d '' expected_digest || {
      echo "runtime_fingerprint: truncated manifest after path: ${path}" >&2
      return 1
    }
    case "${path}" in
      /*|..|../*|*/../*)
        echo "runtime_fingerprint: unsafe manifest path: ${path}" >&2
        return 1
        ;;
    esac
    case "${kind}" in
      F)
        if [[ ! -f "${path}" || -L "${path}" ]]; then
          echo "runtime_fingerprint: missing or wrong-type file: ${path}" >&2
          return 1
        fi
        actual_digest=$(sha256sum -- "${path}" | cut -d' ' -f1)
        ;;
      L)
        if [[ ! -L "${path}" ]]; then
          echo "runtime_fingerprint: missing or wrong-type symlink: ${path}" >&2
          return 1
        fi
        actual_digest=$(printf '%s' "$(readlink -- "${path}")" | sha256sum | cut -d' ' -f1)
        ;;
      *)
        echo "runtime_fingerprint: unknown manifest record type: ${kind}" >&2
        return 1
        ;;
    esac
    if [[ "${actual_digest}" != "${expected_digest}" ]]; then
      echo "runtime_fingerprint: content mismatch: ${path}" >&2
      return 1
    fi
    printf '%s\0%s\0%s\0' "${kind}" "${path}" "${actual_digest}"
  done
}

case "${mode}" in
  manifest)
    emit_manifest
    ;;
  fingerprint)
    emit_manifest | sha256sum | cut -d' ' -f1
    ;;
  verify)
    observed_manifest_digest=$(verify_manifest | sha256sum | cut -d' ' -f1)
    if [[ "${observed_manifest_digest}" != "${expected_manifest_digest}" ]]; then
      echo "runtime_fingerprint: manifest digest mismatch: expected=${expected_manifest_digest} observed=${observed_manifest_digest}" >&2
      exit 1
    fi
    printf '%s\n' "${observed_manifest_digest}"
    ;;
esac
