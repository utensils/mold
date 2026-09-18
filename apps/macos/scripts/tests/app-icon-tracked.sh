#!/usr/bin/env bash
# Every image the icon set names is a file GIT carries.
#
# **Fails today**: the root `.gitignore` says `*.png`, and the app's icon set
# was never exempted. Locally every build had the icon, because the files
# were on the disk; the runner cloned `Contents.json` and nothing else, actool
# WARNED about ten missing files, and the first nightly shipped with no
# `Assets.car`, no `AppIcon.icns` and a generic icon on the Dock, in the
# Finder and on the mounted disk image. Nothing turned red, because a
# missing icon is a warning to Xcode.
#
# `git ls-files` rather than `test -f`: the disk is exactly what fooled
# everyone, the index is what the runner gets.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
MACOS="$(cd "$HERE/.." && pwd)"
ICON_SET="Sources/Mold/Resources/Assets.xcassets/AppIcon.appiconset"

cd "$MACOS"
tracked="$(git ls-files "$ICON_SET")"
fail=0
count=0
for image in $(grep -o '"filename" *: *"[^"]*"' "$ICON_SET/Contents.json" | sed 's/.*: *"//; s/"$//'); do
  count=$((count + 1))
  if ! printf '%s\n' "$tracked" | grep -qx "$ICON_SET/$image"; then
    echo "FAIL: $ICON_SET/$image is named by Contents.json but not tracked by git (is it .gitignored?)" >&2
    fail=1
  fi
done
[ "$count" -gt 0 ] || { echo "FAIL: Contents.json names no images" >&2; exit 1; }
[ "$fail" = 0 ] || exit 1
echo "  app icon: $count images tracked"
