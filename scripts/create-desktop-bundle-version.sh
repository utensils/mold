#!/usr/bin/env bash
# Encode the main-history commit count as an Apple-safe CFBundleVersion.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: create-desktop-bundle-version.sh <commit-count>" >&2
  exit 2
fi

commit_count="$1"

if [[ ! "$commit_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: commit-count must be a positive integer" >&2
  exit 1
fi
if ((10#$commit_count > 99989999)); then
  echo "error: commit-count exceeds CFBundleVersion 4.2.2 capacity" >&2
  exit 1
fi

count=$((10#$commit_count))
major=$((count / 10000 + 1))
minor=$(((count / 100) % 100))
patch=$((count % 100))

# Bias the high base-100 component by one because Apple's first component must
# be positive. The mapping remains strictly ordered and deterministic while all
# components stay within Apple's 4.2.2 digit limits.
printf '%s.%s.%s\n' "$major" "$minor" "$patch"
