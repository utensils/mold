#!/usr/bin/env bash
# Wrap a signed app in a DMG with an Applications symlink and the app's own
# icon on the mounted volume.
set -euo pipefail

APP="${1:?usage: create-dmg.sh <path to .app> <output.dmg>}"
OUT="${2:?usage: create-dmg.sh <path to .app> <output.dmg>}"
VOLNAME="${MOLD_DMG_VOLNAME:-Mold Studio}"

STAGE="$(mktemp -d)"
RW="$STAGE.rw.dmg"
MOUNT=""
cleanup() {
  if [ -n "$MOUNT" ]; then hdiutil detach "$MOUNT" -quiet 2>/dev/null || true; fi
  rm -rf "$STAGE" "$RW"
}
trap cleanup EXIT

# `ditto` rather than `cp`: it preserves the extended attributes and the code
# signature, which a plain copy can strip.
ditto "$APP" "$STAGE/$(basename "$APP")"
ln -s /Applications "$STAGE/Applications"

# The volume's icon is the app's. A DMG made straight from a folder gets the
# generic disk-image icon: `.VolumeIcon.icns` only counts once the volume's
# ROOT carries the Finder "custom icon" flag, and a source folder's own flags
# never reach the image. So the image is built writable, mounted, flagged,
# and only then compressed.
ICON_FILE="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIconFile' "$APP/Contents/Info.plist" 2>/dev/null || true)"
case "$ICON_FILE" in "" | *.icns) ;; *) ICON_FILE="$ICON_FILE.icns" ;; esac
ICON="$APP/Contents/Resources/$ICON_FILE"
if [ -n "$ICON_FILE" ] && [ -s "$ICON" ]; then
  cp "$ICON" "$STAGE/.VolumeIcon.icns"
else
  echo "  ($APP carries no icon -- the disk image gets the generic one)" >&2
fi

rm -f "$OUT" "$RW"
hdiutil create -volname "$VOLNAME" -srcfolder "$STAGE" -ov -format UDRW "$RW" >/dev/null
if [ -f "$STAGE/.VolumeIcon.icns" ]; then
  MOUNT="$(hdiutil attach -nobrowse -readwrite -noverify "$RW" | awk -F'\t' '/\/Volumes\// { print $NF; exit }')"
  [ -n "$MOUNT" ] || { echo "could not mount $RW" >&2; exit 1; }
  if command -v SetFile >/dev/null 2>&1; then
    SetFile -a C "$MOUNT"
  else
    # No Xcode tools: the same bit, by hand. Finder flags are bytes 8-9 of
    # the 32-byte FinderInfo; 0x0400 is kHasCustomIcon.
    # Exactly 32 bytes: type+creator (8), flags (2), the rest (22).
    xattr -wx com.apple.FinderInfo \
      "0000000000000000 0400 00000000000000000000000000000000000000000000" "$MOUNT"
  fi
  hdiutil detach "$MOUNT" -quiet
  MOUNT=""
fi
hdiutil convert "$RW" -format UDZO -ov -o "$OUT" >/dev/null

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
