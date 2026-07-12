#!/usr/bin/env bash
# Decide whether a stable candidate may advance GitHub's releases/latest pointer.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: select-desktop-stable-latest.sh <candidate-semver> <current-semver>" >&2
  exit 2
fi

candidate="$1"
current="$2"
semver_re='^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$'

if [[ ! "$candidate" =~ $semver_re ]]; then
  echo "error: invalid stable candidate SemVer: $candidate" >&2
  exit 1
fi
if [[ ! "$current" =~ $semver_re ]]; then
  echo "error: invalid current stable SemVer: $current" >&2
  exit 1
fi

IFS=. read -r candidate_major candidate_minor candidate_patch <<< "$candidate"
IFS=. read -r current_major current_minor current_patch <<< "$current"
candidate_parts=("$candidate_major" "$candidate_minor" "$candidate_patch")
current_parts=("$current_major" "$current_minor" "$current_patch")

for index in 0 1 2; do
  candidate_part=$((10#${candidate_parts[$index]}))
  current_part=$((10#${current_parts[$index]}))
  if ((candidate_part > current_part)); then
    echo true
    exit 0
  fi
  if ((candidate_part < current_part)); then
    echo false
    exit 0
  fi
done

# Republishing the current stable tag is safe and should preserve latest.
echo true
