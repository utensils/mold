#!/usr/bin/env bash
# Runs on the release-plz release-PR branch after release-plz has bumped the
# workspace version. Finishes the two things release-plz cannot do here:
#
#   1. CHANGELOG.md is hand-maintained (Keep a Changelog). Every PR ships its
#      note as a fragment in changelog.d/<slug>.md (one file per PR, so two
#      in-flight PRs never edit the same line). Assemble those fragments under
#      [Unreleased], delete them, then promote the [Unreleased] section to the
#      new version heading and refresh the compare-link references at the
#      bottom of the file.
#   2. The Tauri desktop and mobile apps are standalone cargo roots, so mirror
#      the workspace version into their manifests, locks, and configs.
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

# --- 0. changelog.d fragments -> [Unreleased] -------------------------------
# Newest fragment first (matches the hand-maintained "newest on top" order);
# ordered by the commit that added each file, falling back to filename when
# the tree is not a git checkout (the test fixture). README.md is docs.
fragments=()
if [ -d changelog.d ]; then
  while IFS= read -r line; do
    fragments+=("${line#* }")
  done < <(
    for f in changelog.d/*.md; do
      [ -e "$f" ] || continue
      [ "$(basename "$f")" = "README.md" ] && continue
      ts=$(git log -1 --diff-filter=A --format=%ct -- "$f" 2>/dev/null || true)
      printf '%s %s\n' "${ts:-0}" "$f"
    done | sort -k1,1nr -k2,2
  )
fi
if [ "${#fragments[@]}" -gt 0 ]; then
  grep -q '^## \[Unreleased\]$' CHANGELOG.md || {
    echo "error: CHANGELOG.md has no [Unreleased] heading to assemble fragments under" >&2
    exit 1
  }
  assembled=$(mktemp)
  for f in "${fragments[@]}"; do
    # Normalise: strip leading/trailing blank lines, guarantee one trailing newline.
    awk 'NF{p=1} p{buf=buf $0 "\n"} END{sub(/\n+$/,"\n",buf); printf "%s", buf}' "$f" >> "$assembled"
  done
  awk -v file="$assembled" '
    /^## \[Unreleased\]$/ {
      print; print ""
      while ((getline line < file) > 0) print line
      close(file)
      # Swallow the blank line that followed the heading (we printed our own),
      # but keep a blank before whatever heading or text comes next so an
      # empty [Unreleased] section still separates from the version below.
      if ((getline nxt) > 0) {
        if (nxt != "") { print nxt }
        else if ((getline nxt2) > 0) { if (nxt2 ~ /^#/) print ""; print nxt2 }
      }
      next
    }
    { print }
  ' CHANGELOG.md > CHANGELOG.md.tmp && mv CHANGELOG.md.tmp CHANGELOG.md
  rm -f "$assembled"
  rm -f "${fragments[@]}"
  echo "CHANGELOG.md: assembled ${#fragments[@]} changelog.d fragment(s) into [Unreleased]"
fi

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

# --- 3. iOS app version ----------------------------------------------------
sed -E "0,/^version = \"[^\"]+\"/s//version = \"$version\"/" \
  apps/mobile/src-tauri/Cargo.toml > apps/mobile/src-tauri/Cargo.toml.tmp \
  && mv apps/mobile/src-tauri/Cargo.toml.tmp apps/mobile/src-tauri/Cargo.toml

awk -v ver="$version" '
  /^name = "mold-mobile"$/ { print; sync = 1; next }
  sync && /^version = / { print "version = \"" ver "\""; sync = 0; next }
  { print }
' apps/mobile/src-tauri/Cargo.lock > apps/mobile/src-tauri/Cargo.lock.tmp \
  && mv apps/mobile/src-tauri/Cargo.lock.tmp apps/mobile/src-tauri/Cargo.lock

python3 - "$version" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path("apps/mobile/src-tauri/tauri.conf.json")
data = json.loads(path.read_text())
data["version"] = sys.argv[1]
path.write_text(json.dumps(data, indent=2) + "\n")
PY

echo "ios: synced to $version"
