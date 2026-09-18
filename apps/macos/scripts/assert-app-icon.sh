#!/usr/bin/env bash
# Refuse to ship an app with no icon.
#
# actool treats a missing icon image as a WARNING, so an icon set whose files
# never reached the runner builds a green app with a generic Dock icon. The
# first nightly shipped exactly that. Two checks, because the plist alone
# says what was asked for: the bundle must name an icon file, and that file
# must be in the bundle.
set -euo pipefail

APP="${1:?usage: assert-app-icon.sh <path to .app>}"
PLIST="$APP/Contents/Info.plist"

no_icon() {
  cat >&2 <<EOF
$1

This build would ship with a generic icon on the Dock, in the Finder and on
the disk image. Check that every image in
Sources/Mold/Resources/Assets.xcassets/AppIcon.appiconset is tracked by git
(scripts/tests/app-icon-tracked.sh) and rebuild.
EOF
  exit 1
}

[[ -f "$PLIST" ]] || no_icon "missing Info.plist: $PLIST"
icon="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIconFile' "$PLIST" 2>/dev/null || true)"
[[ -n "$icon" ]] || no_icon "$PLIST names no CFBundleIconFile."
case "$icon" in *.icns) ;; *) icon="$icon.icns" ;; esac
[[ -s "$APP/Contents/Resources/$icon" ]] || no_icon "$APP carries no Contents/Resources/$icon."

echo "  app icon: $icon"
