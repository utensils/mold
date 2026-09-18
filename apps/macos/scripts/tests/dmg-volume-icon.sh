#!/usr/bin/env bash
# The disk image `create-dmg.sh` makes mounts with the app's icon.
#
# **Fails today**: the script built the image straight from the staging
# folder, so the volume carried no `.VolumeIcon.icns` and its root had no
# custom-icon flag -- the mounted nightly showed the generic disk-image icon
# beside a generic app icon.
#
# A real `hdiutil` round trip on a stub bundle: a few seconds, and the only
# way to ask the question the Finder asks.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
MOUNT=""
cleanup() {
  if [ -n "$MOUNT" ]; then hdiutil detach "$MOUNT" -quiet 2>/dev/null || true; fi
  rm -rf "$WORK"
}
trap cleanup EXIT

APP="$WORK/Mold Studio.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
printf 'icns-bytes' > "$APP/Contents/Resources/AppIcon.icns"
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>io.utensils.mold.native</string>
<key>CFBundleIconFile</key><string>AppIcon</string>
</dict></plist>
PLIST

# Twice: with SetFile, and on a PATH without it so the by-hand FinderInfo
# arm runs (its first draft wrote 31 bytes, which xattr refuses, and nothing
# had exercised it).
NO_SETFILE="$WORK/bin"
mkdir -p "$NO_SETFILE"
for tool in bash hdiutil ditto awk cp rm ln mktemp basename dirname cat xattr; do
  ln -s "$(command -v "$tool")" "$NO_SETFILE/$tool"
done

check() {
  local label="$1" path="$2"
  local OUT="$WORK/$label.dmg"
  MOLD_SIGN_IDENTITY="" PATH="$path" "$HERE/create-dmg.sh" "$APP" "$OUT" > /dev/null 2>&1 \
    || { echo "FAIL ($label): create-dmg.sh failed" >&2; exit 1; }
  [ -s "$OUT" ] || { echo "FAIL ($label): no DMG produced" >&2; exit 1; }
  MOUNT="$(hdiutil attach -nobrowse -readonly "$OUT" | awk -F'\t' '/\/Volumes\// { print $NF; exit }')"
  [ -n "$MOUNT" ] || { echo "FAIL ($label): could not mount $OUT" >&2; exit 1; }
  verify "$label"
  hdiutil detach "$MOUNT" -quiet
  MOUNT=""
}

verify() {
  local label="$1"

case "$(basename "$MOUNT")" in
  "Mold Studio"*) ;;
  *) echo "FAIL ($label): volume is named '$(basename "$MOUNT")', not 'Mold Studio'" >&2; exit 1 ;;
esac
[ -d "$MOUNT/Mold Studio.app" ] || { echo "FAIL ($label): the app is not on the volume" >&2; exit 1; }
[ -L "$MOUNT/Applications" ] || { echo "FAIL ($label): no Applications symlink" >&2; exit 1; }
cmp -s "$MOUNT/.VolumeIcon.icns" "$APP/Contents/Resources/AppIcon.icns" \
  || { echo "FAIL ($label): .VolumeIcon.icns is missing or is not the app's icon" >&2; exit 1; }

# The flag is what makes the Finder READ that file. Bytes 8-9 of FinderInfo
# carry the Finder flags; 0x0400 is kHasCustomIcon.
flags="$(xattr -px com.apple.FinderInfo "$MOUNT" 2>/dev/null | tr -d ' \n' | cut -c17-20)"
if [ -z "$flags" ] || [ $(( 0x$flags & 0x0400 )) -eq 0 ]; then
  echo "FAIL ($label): the volume root has no custom-icon flag (FinderInfo flags: '${flags:-none}')" >&2
  exit 1
fi

}

command -v SetFile > /dev/null || { echo "FAIL: SetFile not on PATH -- the primary arm cannot run" >&2; exit 1; }
check setfile "$PATH"
check xattr "$NO_SETFILE:/usr/libexec"
echo "  dmg volume icon ok (SetFile and FinderInfo arms)"
