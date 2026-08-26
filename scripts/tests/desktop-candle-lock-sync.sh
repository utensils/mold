#!/usr/bin/env bash
# Keep the excluded Tauri crate's lockfile aligned with the official-name
# Candle compatibility source inherited through mold-inference.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
manifest="$repo_root/crates/mold-inference/Cargo.toml"
desktop_lock="$repo_root/desktop/src-tauri/Cargo.lock"
root_lock="$repo_root/Cargo.lock"

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

manifest_package() {
  local dependency="$1"
  sed -nE \
    "s/^${dependency} = \{ package = \"([^\"]+)\".*$/\\1/p" \
    "$manifest"
}

lock_field() {
  local lockfile="$1"
  local package="$2"
  local field="$3"
  awk -v package="$package" -v field="$field" '
    $0 == "name = \"" package "\"" {
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
  ' "$lockfile"
}

for dependency in candle-core candle-nn candle-transformers; do
  package="$(manifest_package "$dependency")"
  root_version="$(lock_field "$root_lock" "$package" version)"
  desktop_version="$(lock_field "$desktop_lock" "$package" version)"
  root_source="$(lock_field "$root_lock" "$package" source)"
  desktop_source="$(lock_field "$desktop_lock" "$package" source)"

  test -n "$package" || fail "could not read $dependency package from mold-inference"
  test -n "$desktop_version" || fail "desktop lockfile is missing $package"
  test "$desktop_version" = "$root_version" ||
    fail "desktop lockfile has $package $desktop_version, but root lockfile has $root_version"
  test "$desktop_source" = "$root_source" ||
    fail "desktop and root lockfiles resolve $package from different sources; update both locks together"
done

echo "PASS: desktop Candle lockfile matches mold-inference"
