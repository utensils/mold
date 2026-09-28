#!/usr/bin/env bash
# The architecture lints both native apps share (apps/macos, apps/ios).
#
#   swift-lint.sh <check> <root>...
#
# checks:
#   color      no literal colors -- the system (or an asset catalog) owns the palette
#   a11y       a file that draws an SF Symbol must label it, or say why not
#   size       advisory: files over 150 lines
#   type-size  advisory: a type whose Type.swift + Type+*.swift exceed $TYPE_MAX
#
# Each app's Makefile keeps the rules only it has (the Mac's composition root,
# the phone's Dynamic Type and widget rules) and calls this for the rest, so a
# rule both apps follow is written once.
set -euo pipefail

check=${1:?usage: swift-lint.sh <color|a11y|size|type-size> <root>...}
shift
[ "$#" -gt 0 ] || { echo "swift-lint.sh: no roots given" >&2; exit 2; }
for root in "$@"; do
  [ -d "$root" ] || { echo "swift-lint.sh: $root is not a directory" >&2; exit 2; }
done

case "$check" in
color)
  if grep -rnE 'Color\((red|hue|white):|Color\(hex|UIColor\((red|white|hue):|NSColor\((red|white|calibratedRed|srgbRed):' \
    "$@" --include='*.swift'; then
    echo "COLOR VIOLATION: use a semantic color, a material, or an asset-catalog color"
    exit 1
  fi
  echo "  color ok"
  ;;
a11y)
  # A per-FILE floor: a file with a symbol in it must also carry a real label
  # somewhere, or say by name why the glyph needs none. Whether the modifier
  # reaches the right view is still a human call, not this rule's.
  fail=0
  while IFS= read -r f; do
    if ! grep -qE 'accessibilityLabel|\.help\(|Label\(|Label \{|accessibilityElement|// a11y:' "$f"; then
      echo "  no VoiceOver label and no // a11y: opt-out: $f"
      fail=1
    fi
  done < <(grep -rl 'Image(systemName:' "$@" --include='*.swift' || true)
  [ "$fail" = 0 ] || exit 1
  echo "  a11y ok"
  ;;
size)
  find "$@" -name '*.swift' -exec wc -l {} + \
    | awk '$1 > 150 && $2 != "total" { print "  large: " $2 " (" $1 " lines)" }'
  ;;
type-size)
  # A type's size is the sum of Type.swift and every Type+Concern.swift beside
  # it: splitting a 400-line type into three files does not make it smaller.
  find "$@" -name '*.swift' -exec wc -l {} + \
    | awk -v max="${TYPE_MAX:-600}" '$2 != "total" {
        name = $2; sub(/.*\//, "", name); sub(/\.swift$/, "", name); sub(/\+.*$/, "", name)
        total[name] += $1 }
      END { for (t in total) if (total[t] > max)
        printf "  large type: %s (%d lines across its files)\n", t, total[t] }' \
    | sort
  ;;
*)
  echo "swift-lint.sh: unknown check '$check'" >&2
  exit 2
  ;;
esac
