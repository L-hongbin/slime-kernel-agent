#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "error: $*" >&2
  exit 1
}

usage() {
  echo "Usage: $0 [bundle_file]" >&2
}

find_bundle_file() {
  shopt -s nullglob
  local bundle_files=(./*.bundle)
  shopt -u nullglob

  if ((${#bundle_files[@]} != 1)); then
    echo "error: expected exactly one ./*.bundle file, found ${#bundle_files[@]}" >&2
    if ((${#bundle_files[@]} > 0)); then
      printf '  %s\n' "${bundle_files[@]}" >&2
    fi
    exit 1
  fi

  printf '%s\n' "${bundle_files[0]}"
}

select_bundle_ref() {
  local bundle_file=$1
  local current_branch ref
  local refs=()
  local matches=()

  mapfile -t refs < <(git bundle list-heads "$bundle_file" | awk 'NF >= 2 {print $2}')

  if ((${#refs[@]} == 0)); then
    die "no refs found in bundle: $bundle_file"
  fi

  current_branch=$(git symbolic-ref --quiet --short HEAD 2>/dev/null || true)
  if [[ -n "$current_branch" ]]; then
    for ref in "${refs[@]}"; do
      case "$ref" in
        "$current_branch" | "refs/heads/$current_branch" | "refs/remotes/origin/$current_branch")
          matches+=("$ref")
          ;;
      esac
    done

    if ((${#matches[@]} == 1)); then
      printf '%s\n' "${matches[0]}"
      return
    fi

    if ((${#matches[@]} > 1)); then
      die "multiple refs in bundle match current branch '$current_branch': ${matches[*]}"
    fi
  fi

  if ((${#refs[@]} == 1)); then
    printf '%s\n' "${refs[0]}"
    return
  fi

  echo "error: cannot determine which bundle ref to pull" >&2
  if [[ -n "$current_branch" ]]; then
    echo "current branch: $current_branch" >&2
  fi
  echo "bundle refs:" >&2
  printf '  %s\n' "${refs[@]}" >&2
  exit 1
}

if (($# > 1)); then
  usage
  exit 2
fi

if (($# == 0)); then
  BUNDLE_FILE=$(find_bundle_file)
else
  BUNDLE_FILE=$1
fi

[[ -f "$BUNDLE_FILE" ]] || die "bundle file not found: $BUNDLE_FILE"

git bundle verify "$BUNDLE_FILE"
BUNDLE_REF=$(select_bundle_ref "$BUNDLE_FILE")

git pull "$BUNDLE_FILE" "$BUNDLE_REF"
rm -f -- "$BUNDLE_FILE"
