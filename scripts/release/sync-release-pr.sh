#!/usr/bin/env bash
# Runs on the release-plz release-PR branch after release-plz has bumped the
# workspace version. Finishes the two things release-plz cannot do here:
#
#   1. CHANGELOG.md is hand-maintained (Keep a Changelog), so promote the
#      [Unreleased] section to the new version heading and refresh the
#      compare-link references at the bottom of the file.
#   2. The Tauri desktop app is its own cargo root (excluded from the
#      workspace), so mirror the workspace version into
#      desktop/src-tauri/Cargo.toml, its Cargo.lock, and desktop/package.json.
#
# Idempotent: a second run on an already-synced tree changes nothing.
#
# Usage: sync-release-pr.sh [repo-root]   (defaults to the script's repo)
set -euo pipefail

root="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$root"

version=$(sed -n '/^\[workspace\.package\]/,/^\[/s/^version = "\([^"]*\)"/\1/p' Cargo.toml | head -1)
if [ -z "$version" ]; then
  echo "error: could not read [workspace.package] version from Cargo.toml" >&2
  exit 1
fi
today=$(date -u +%Y-%m-%d)
repo_url="https://github.com/utensils/mold"

# --- 1. CHANGELOG.md -------------------------------------------------------
if ! grep -q "^## \[$version\]" CHANGELOG.md; then
  # Most recent released version (first "## [x.y.z]" heading) BEFORE we insert
  # the new one; used for the new compare link.
  prev=$(sed -n 's/^## \[\([0-9][^]]*\)\].*/\1/p' CHANGELOG.md | head -1)
  if [ -z "$prev" ]; then
    echo "error: could not find a previous version heading in CHANGELOG.md" >&2
    exit 1
  fi

  awk -v ver="$version" -v date="$today" '
    /^## \[Unreleased\]$/ { print; print ""; print "## [" ver "] - " date; next }
    { print }
  ' CHANGELOG.md > CHANGELOG.md.tmp && mv CHANGELOG.md.tmp CHANGELOG.md

  awk -v ver="$version" -v prev="$prev" -v url="$repo_url" '
    /^\[Unreleased\]: / {
      print "[Unreleased]: " url "/compare/v" ver "...HEAD"
      print "[" ver "]: " url "/compare/v" prev "...v" ver
      next
    }
    { print }
  ' CHANGELOG.md > CHANGELOG.md.tmp && mv CHANGELOG.md.tmp CHANGELOG.md
  echo "CHANGELOG.md: promoted [Unreleased] -> [$version] (previous: $prev)"
else
  echo "CHANGELOG.md: [$version] already present, skipping"
fi

# --- 2. Desktop app version ------------------------------------------------
# First `version = "..."` line is the [package] version; the path dependencies
# on workspace crates (package = "mold-ai-*") carry version requirements that
# must track the workspace version or cargo fails to select them.
awk -v ver="$version" '
  !done && /^version = "/ { print "version = \"" ver "\""; done = 1; next }
  /package = "mold-ai-/ { gsub(/version = "[^"]*"/, "version = \"" ver "\"") }
  { print }
' desktop/src-tauri/Cargo.toml > desktop/src-tauri/Cargo.toml.tmp \
  && mv desktop/src-tauri/Cargo.toml.tmp desktop/src-tauri/Cargo.toml

# mold-desktop itself plus every workspace crate resolved via path deps.
awk -v ver="$version" '
  /^name = "mold-desktop"$/ || /^name = "mold-ai-/ { print; sync = 1; next }
  sync && /^version = / { print "version = \"" ver "\""; sync = 0; next }
  { print }
' desktop/src-tauri/Cargo.lock > desktop/src-tauri/Cargo.lock.tmp \
  && mv desktop/src-tauri/Cargo.lock.tmp desktop/src-tauri/Cargo.lock

awk -v ver="$version" '
  !done && /^  "version": "/ { print "  \"version\": \"" ver "\","; done = 1; next }
  { print }
' desktop/package.json > desktop/package.json.tmp \
  && mv desktop/package.json.tmp desktop/package.json

echo "desktop: synced to $version"
