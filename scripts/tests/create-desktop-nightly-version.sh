#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
script="$repo_root/scripts/create-desktop-nightly-version.sh"

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

[[ "$($script 0.16.0 337)" == "0.16.1-nightly.337" ]] \
  || fail "next-patch nightly version is incorrect"
[[ "$($script 0.16.9 338)" == "0.16.10-nightly.338" ]] \
  || fail "patch carry is incorrect"
[[ "$($script 1.2.3 9999)" == "1.2.4-nightly.9999" ]] \
  || fail "major version nightly is incorrect"

if "$script" 0.16 337 > /dev/null 2>&1; then
  fail "invalid base SemVer was accepted"
fi
if "$script" 0.16.0 0 > /dev/null 2>&1; then
  fail "zero commit count was accepted"
fi
if "$script" 0.16.01 337 > /dev/null 2>&1; then
  fail "base SemVer with a leading zero was accepted"
fi

echo "PASS: create-desktop-nightly-version"
