#!/usr/bin/env bash
# Exit 10 when a completed nightly build no longer represents main HEAD.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: check-desktop-nightly-main-head.sh <expected-sha> [remote]" >&2
  exit 2
fi

expected_sha="$1"
remote="${2:-origin}"

if [[ ! "$expected_sha" =~ ^([0-9a-fA-F]{40}|[0-9a-fA-F]{64})$ ]]; then
  echo "error: expected-sha is not a full Git object ID" >&2
  exit 2
fi

git fetch --quiet --no-tags "$remote" refs/heads/main
main_sha="$(git rev-parse --verify 'FETCH_HEAD^{commit}')"
expected_sha="$(printf '%s' "$expected_sha" | tr '[:upper:]' '[:lower:]')"

if [[ "$main_sha" == "$expected_sha" ]]; then
  echo "Nightly source is still main HEAD: $main_sha"
  exit 0
fi

echo "Nightly source $expected_sha is stale; current main HEAD is $main_sha."
exit 10
