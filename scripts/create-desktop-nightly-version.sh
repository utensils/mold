#!/usr/bin/env bash
# Produce the deterministic SemVer used by one main commit's Nightly artifacts.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: create-desktop-nightly-version.sh <base-semver> <commit-count>" >&2
  exit 2
fi

base_version="$1"
commit_count="$2"
semver_re='^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$'

if [[ ! "$base_version" =~ $semver_re ]]; then
  echo "error: base-semver must be a plain three-part SemVer" >&2
  exit 1
fi
major="${BASH_REMATCH[1]}"
minor="${BASH_REMATCH[2]}"
patch="${BASH_REMATCH[3]}"
if [[ ! "$commit_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: commit-count must be a positive integer" >&2
  exit 1
fi

next_patch=$((10#$patch + 1))

printf '%s.%s.%s-nightly.%s\n' "$major" "$minor" "$next_patch" "$commit_count"
