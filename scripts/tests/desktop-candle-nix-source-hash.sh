#!/usr/bin/env bash
# Verify the Nix fixed-output hash for the excluded desktop crate's Candle
# git source against the exact revision selected by its Cargo lockfile.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
desktop_lock="$repo_root/desktop/src-tauri/Cargo.lock"
flake="$repo_root/flake.nix"

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

command -v nix >/dev/null || fail "nix is required to verify the desktop Candle source hash"
command -v jq >/dev/null || fail "jq is required to read nix prefetch output"

read_lock_field() {
  local field="$1"
  awk -v field="$field" '
    $0 == "name = \"candle-core\"" {
      found = 1
      next
    }
    found && $1 == field && $2 == "=" {
      gsub(/"/, "", $3)
      print $3
      exit
    }
    found && /^\[\[package\]\]$/ {
      exit
    }
  ' "$desktop_lock"
}

version="$(read_lock_field version)"
source="$(read_lock_field source)"
test -n "$version" || fail "desktop lockfile is missing the candle-core version"
test -n "$source" || fail "desktop lockfile is missing the candle-core git source"

case "$source" in
  git+https://github.com/*\?*#*) ;;
  *) fail "unsupported candle-core source: $source" ;;
esac

repository="${source#git+}"
repository="${repository%%\?*}"
revision="${source##*#}"
archive_url="$repository/archive/$revision.tar.gz"
expected="$(
  sed -nE \
    "s|^[[:space:]]*\"candle-core-${version}\" = \"([^\"]+)\";.*$|\\1|p" \
    "$flake"
)"
test -n "$expected" || fail "flake.nix is missing the candle-core-${version} output hash"

actual="$(nix store prefetch-file --unpack --json "$archive_url" | jq -r .hash)"
test -n "$actual" && test "$actual" != "null" || fail "could not prefetch $archive_url"
test "$actual" = "$expected" ||
  fail "flake.nix has $expected for candle-core at $revision; expected $actual"

echo "PASS: desktop Candle Nix source hash matches $revision"
