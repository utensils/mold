#!/usr/bin/env bash
# Replace release-plz's changelog placeholder with the curated notes from the
# promoted version section in CHANGELOG.md. The package summary and footer that
# release-plz generated remain untouched.
#
# Usage: render-release-pr-body.sh CHANGELOG VERSION CURRENT_BODY
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 CHANGELOG VERSION CURRENT_BODY" >&2
  exit 2
fi

changelog=$1
version=$2
current_body=$3
notes=$(mktemp)
trap 'rm -f "$notes"' EXIT

awk -v heading="## [$version]" '
  index($0, heading) == 1 { in_release = 1; next }
  in_release && /^## \[/ { exit }
  in_release && /^\[Unreleased\]: / { exit }
  in_release { print }
' "$changelog" > "$notes"

if ! grep -q '[^[:space:]]' "$notes"; then
  echo "error: CHANGELOG.md has no notes for [$version]" >&2
  exit 1
fi

awk -v notes="$notes" '
  function emit_notes(line) {
    while ((getline line < notes) > 0) print line
    close(notes)
  }

  /^<details><summary><i><b>Changelog<\/b><\/i><\/summary><p>$/ {
    print
    print ""
    emit_notes()
    replacing = 1
    next
  }

  replacing && /^<\/p><\/details>$/ {
    print ""
    print
    replacing = 0
    replaced = 1
    next
  }

  !replacing { print }

  END {
    if (replacing || !replaced) exit 42
  }
' "$current_body" || {
  status=$?
  if [ "$status" -eq 42 ]; then
    echo "error: release-plz changelog placeholder not found in PR body" >&2
  fi
  exit "$status"
}
