#!/usr/bin/env bash
# Pull-request changelog policy. Every PR ships its release note as a fragment
# in changelog.d/<slug>.md; CHANGELOG.md's [Unreleased] section is assembled
# from those fragments by scripts/release/sync-release-pr.sh on the release PR
# and is never edited by hand (two in-flight PRs inserting at the same line is
# what made every PR conflict on CHANGELOG.md).
#
# Checks, in order:
#   1. Every fragment present at HEAD is well-formed (a Keep-a-Changelog
#      bullet: first non-blank line starts with "- ", no conflict markers,
#      README.md exempt).
#   2. The [Unreleased] section of CHANGELOG.md is unchanged between BASE and
#      HEAD.
#   3. When the diff touches shipped source, the PR adds at least one fragment
#      — unless SKIP_CHANGELOG=true (the `skip-changelog` PR label).
#
# Usage: check-changelog-fragments.sh <base-sha> <head-sha>   (run at repo root,
# full history available — the checkout must not be shallow)
set -euo pipefail

base_ref=${1:?base sha}
head=${2:?head sha}
skip=${SKIP_CHANGELOG:-false}
status=0

# GitHub's pull_request.base.sha is the base branch TIP, not the fork point.
# A PR that branched before main moved (or before a release emptied
# [Unreleased]) must be judged against what it actually changed, so every
# comparison below uses the merge base.
base=$(git merge-base "$base_ref" "$head" 2>/dev/null || echo "$base_ref")

err() { echo "::error::$1" >&2; status=1; }

# 1. Fragment shape — only fragments this PR added or modified, so one bad
#    file already on main does not fail every unrelated PR. Paths are read
#    unquoted (core.quotePath=false) so non-ASCII slugs are still seen.
while IFS= read -r f; do
  [ -n "$f" ] || continue
  [ "$(basename "$f")" = "README.md" ] && continue
  case "$f" in
    changelog.d/*/*) err "$f is nested; fragments must live directly in changelog.d/ (the release script only assembles changelog.d/*.md)"; continue ;;
    *.md) ;;
    *) err "$f is not a .md fragment" ; continue ;;
  esac
  content=$(git show "$head:$f" 2>/dev/null || true)
  first=$(printf '%s\n' "$content" | awk 'NF{print; exit}')
  case "$first" in
    "- "*) ;;
    *) err "$f must start with a Keep-a-Changelog bullet ('- **Title.** body'); got: ${first:-<empty>}" ;;
  esac
  if printf '%s\n' "$content" | grep -Eq '^(<<<<<<<|=======|>>>>>>>)'; then
    err "$f contains merge-conflict markers"
  fi
  if printf '%s\n' "$content" | grep -q $'\r'; then
    err "$f has CRLF line endings; use LF"
  fi
done < <(git -c core.quotePath=false diff --name-status --find-renames "$base" "$head" -- changelog.d \
  | awk '$1 ~ /^(A|M|R)/ {print $NF}')

# 2. [Unreleased] is bot-owned.
unreleased() {
  git show "$1:CHANGELOG.md" 2>/dev/null \
    | awk '/^## \[Unreleased\]$/{s=1; next} s && /^## \[/{exit} s{print}'
}
if [ "$(unreleased "$base")" != "$(unreleased "$head")" ]; then
  err "CHANGELOG.md [Unreleased] was edited directly; add changelog.d/<slug>.md instead (the release PR assembles fragments, and hand edits conflict between PRs)"
fi

# 3. Shipped source changes carry a note.
changed=$(git -c core.quotePath=false diff --name-only "$base" "$head")
# Only a genuinely new fragment counts — renaming one that already lives on
# main would reuse another PR's note.
added_fragments=$(git -c core.quotePath=false diff --name-status --find-renames "$base" "$head" -- changelog.d \
  | awk '$1 == "A" && $NF ~ /^changelog\.d\/[^\/]+\.md$/ && $NF !~ /README\.md$/ {print $NF}')
if [ "$skip" != "true" ] \
  && printf '%s\n' "$changed" | grep -Eq '^(crates/|web/src/|desktop/src/|desktop/src-tauri/src/|apps/mobile/src/|apps/mobile/src-tauri/src/|studio/|ui/)' \
  && [ -z "$added_fragments" ]; then
  err "this PR changes shipped source but adds no changelog.d/<slug>.md fragment; add one (see changelog.d/README.md) or apply the skip-changelog label"
fi

if [ "$status" -eq 0 ]; then
  echo "changelog policy: OK"
fi
exit "$status"
