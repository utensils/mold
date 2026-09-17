#!/usr/bin/env bash
# Wrap a signed Mold.app in a DMG with an Applications symlink.
set -euo pipefail

APP="${1:?usage: create-dmg.sh <path to .app> <output.dmg>}"
OUT="${2:?usage: create-dmg.sh <path to .app> <output.dmg>}"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# `ditto` rather than `cp`: it preserves the extended attributes and the code
# signature, which a plain copy can strip.
ditto "$APP" "$STAGE/$(basename "$APP")"
ln -s /Applications "$STAGE/Applications"

rm -f "$OUT"
hdiutil create -volname "Mold" -srcfolder "$STAGE" -ov -format UDZO "$OUT" >/dev/null

# The disk image is signed too. Notarization and stapling work without it, but
# an unsigned DMG is an unsigned file until the ticket is checked, and signing
# it is one line (review 05-L4).
IDENTITY="${MOLD_SIGN_IDENTITY:-}"
if [ -n "$IDENTITY" ]; then
  codesign --force --timestamp --sign "$IDENTITY" "$OUT"
  codesign --verify --verbose=2 "$OUT"
else
  echo "  (MOLD_SIGN_IDENTITY unset -- the disk image itself is unsigned)" >&2
fi
echo "dmg: $OUT"
