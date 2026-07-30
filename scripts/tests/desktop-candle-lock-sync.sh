#!/usr/bin/env bash
# Keep the excluded Tauri crate's lockfile aligned with the Candle fork
# requirements inherited through mold-inference.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
manifest="$repo_root/crates/mold-inference/Cargo.toml"
desktop_lock="$repo_root/desktop/src-tauri/Cargo.lock"

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

manifest_version() {
  local dependency="$1"
  sed -nE \
    "s/^${dependency} = \\{ package = \"${dependency}-mold\", version = \"([^\"]+)\".*$/\\1/p" \
    "$manifest"
}

lock_version() {
  local package="$1"
  awk -v package="$package" '
    $0 == "name = \"" package "\"" {
      getline
      if ($1 == "version" && $2 == "=") {
        gsub(/"/, "", $3)
        print $3
        exit
      }
    }
  ' "$desktop_lock"
}

for dependency in candle-core candle-nn candle-transformers; do
  required="$(manifest_version "$dependency")"
  locked="$(lock_version "${dependency}-mold")"

  test -n "$required" || fail "could not read $dependency requirement from mold-inference"
  test -n "$locked" || fail "desktop lockfile is missing ${dependency}-mold"
  test "$locked" = "$required" ||
    fail "desktop lockfile has ${dependency}-mold $locked, but mold-inference requires $required; run: cargo update --manifest-path desktop/src-tauri/Cargo.toml -p candle-core-mold --precise $required"
done

echo "PASS: desktop Candle lockfile matches mold-inference"
