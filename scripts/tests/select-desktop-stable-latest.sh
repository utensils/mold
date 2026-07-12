#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
script="$repo_root/scripts/select-desktop-stable-latest.sh"

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

assert_result() {
  local candidate="$1"
  local current="$2"
  local expected="$3"
  local actual
  actual="$($script "$candidate" "$current")"
  [[ "$actual" == "$expected" ]] \
    || fail "$candidate against $current returned $actual, expected $expected"
}

assert_result 0.16.0 0.15.9 true
assert_result 0.16.0 0.16.0 true
assert_result 0.15.9 0.16.0 false
assert_result 0.17.0 0.16.99 true
assert_result 1.0.0 0.99.99 true
assert_result 0.16.10 0.16.9 true

if "$script" 0.16 0.15.9 > /dev/null 2>&1; then
  fail "invalid candidate SemVer was accepted"
fi
if "$script" 0.16.0 0.15.01 > /dev/null 2>&1; then
  fail "current SemVer with a leading zero was accepted"
fi

echo "PASS: select-desktop-stable-latest"
