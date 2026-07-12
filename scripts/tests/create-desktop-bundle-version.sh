#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
script="$repo_root/scripts/create-desktop-bundle-version.sh"

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

assert_version() {
  local commit_count="$1"
  local expected="$2"
  local actual
  actual="$($script "$commit_count")"
  [[ "$actual" == "$expected" ]] \
    || fail "commit count $commit_count encoded as $actual, expected $expected"

  IFS=. read -r major minor patch <<< "$actual"
  [[ "$major" =~ ^[1-9][0-9]{0,3}$ ]] \
    || fail "major component is not Apple-safe: $actual"
  [[ "$minor" =~ ^[0-9]{1,2}$ && "$patch" =~ ^[0-9]{1,2}$ ]] \
    || fail "minor/patch components are not Apple-safe: $actual"
}

assert_version 1 "1.0.1"
assert_version 99 "1.0.99"
assert_version 100 "1.1.0"
assert_version 9999 "1.99.99"
assert_version 10000 "2.0.0"
assert_version 99989999 "9999.99.99"

# The base-100 carry boundaries must remain strictly ordered.
before="$($script 9999)"
after="$($script 10000)"
[[ "$before" == "1.99.99" && "$after" == "2.0.0" ]] \
  || fail "base-100 carry boundary is not ordered"

if "$script" 0 > /dev/null 2>&1; then
  fail "zero commit count was accepted"
fi
if "$script" not-numeric > /dev/null 2>&1; then
  fail "non-numeric commit count was accepted"
fi
if "$script" 99990000 > /dev/null 2>&1; then
  fail "commit count beyond the 4.2.2 capacity was accepted"
fi

echo "PASS: create-desktop-bundle-version"
