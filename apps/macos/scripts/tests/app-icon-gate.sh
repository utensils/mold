#!/usr/bin/env bash
# Prove `assert-app-icon.sh` refuses a bundle with no icon and accepts one
# that has it.
#
# **Fails today**: there is no such script; `make signed` never asked whether
# the app it was about to sign had an icon, and the first nightly did not.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

bundle() {
  local name="$1" icon_key="$2" icon_file="$3"
  local app="$WORK/$name.app"
  mkdir -p "$app/Contents/Resources"
  {
    printf '<?xml version="1.0" encoding="UTF-8"?>\n<plist version="1.0"><dict>\n'
    printf '<key>CFBundleIdentifier</key><string>io.utensils.mold.native</string>\n'
    if [ -n "$icon_key" ]; then
      printf '<key>CFBundleIconFile</key><string>%s</string>\n' "$icon_key"
    fi
    printf '</dict></plist>\n'
  } > "$app/Contents/Info.plist"
  if [ -n "$icon_file" ]; then
    printf 'icns' > "$app/Contents/Resources/$icon_file"
  fi
  echo "$app"
}

expect_refused() {
  local app="$1" why="$2"
  if "$HERE/assert-app-icon.sh" "$app" > /dev/null 2>&1; then
    echo "FAIL: accepted a bundle $why" >&2
    exit 1
  fi
}

# The shape the first nightly shipped: no key, no file.
expect_refused "$(bundle bare "" "")" "with no icon at all"
# The plist promises a file the bundle does not carry.
expect_refused "$(bundle promised AppIcon "")" "whose CFBundleIconFile is missing from Resources"

# What Xcode writes for an icon set: the key without the extension.
"$HERE/assert-app-icon.sh" "$(bundle good AppIcon AppIcon.icns)" > /dev/null
# The key may carry the extension too.
"$HERE/assert-app-icon.sh" "$(bundle explicit AppIcon.icns AppIcon.icns)" > /dev/null

echo "  app icon gate ok"
