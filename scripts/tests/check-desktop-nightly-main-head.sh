#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
script="$repo_root/scripts/check-desktop-nightly-main-head.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

remote="$tmp/remote.git"
seed="$tmp/seed"
checkout="$tmp/checkout"

git init --bare -q "$remote"
git init -q "$seed"
git -C "$seed" config user.name "Mold CI Test"
git -C "$seed" config user.email "ci-test@example.invalid"
git -C "$seed" checkout -q -b main
printf 'first\n' > "$seed/history"
git -C "$seed" add history
git -C "$seed" commit -q -m "test: first main commit"
git -C "$seed" remote add origin "$remote"
git -C "$seed" push -q -u origin main
git --git-dir="$remote" symbolic-ref HEAD refs/heads/main
git clone -q "$remote" "$checkout"

current_sha="$(git -C "$checkout" rev-parse HEAD)"
(
  cd "$checkout"
  "$script" "$current_sha" "$remote" > /dev/null
) || fail "current main HEAD was rejected"

printf 'second\n' >> "$seed/history"
git -C "$seed" add history
git -C "$seed" commit -q -m "test: advance main"
git -C "$seed" push -q origin main

set +e
(
  cd "$checkout"
  "$script" "$current_sha" "$remote" > /dev/null 2>&1
)
stale_status=$?
set -e
[[ "$stale_status" -eq 10 ]] \
  || fail "stale main returned $stale_status instead of the clean-skip status 10"

set +e
(
  cd "$checkout"
  "$script" "$current_sha" "$tmp/missing.git" > /dev/null 2>&1
)
fetch_status=$?
set -e
[[ "$fetch_status" -ne 0 && "$fetch_status" -ne 10 ]] \
  || fail "fetch failure was mistaken for a stale-build skip"

echo "PASS: check-desktop-nightly-main-head"
