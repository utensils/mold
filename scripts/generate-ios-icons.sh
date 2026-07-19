#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_ICON="$ROOT/desktop/icon-master.png"
MOBILE_ICON="$ROOT/apps/mobile/src-tauri/app-icon.png"
SOURCE_CHECKSUM="$ROOT/apps/mobile/src-tauri/app-icon-source.sha256"
APP_ICON_SET="$ROOT/apps/mobile/src-tauri/gen/apple/Assets.xcassets/AppIcon.appiconset"
CATALOG_CHECKSUM="$APP_ICON_SET/SHA256SUMS"
BACKGROUND="#12101d"

command -v magick >/dev/null || {
  echo "generate-ios-icons: ImageMagick is required" >&2
  exit 1
}
command -v cargo >/dev/null || {
  echo "generate-ios-icons: cargo is required" >&2
  exit 1
}
test -f "$SOURCE_ICON" || {
  echo "generate-ios-icons: missing $SOURCE_ICON" >&2
  exit 1
}
test -d "$APP_ICON_SET" || {
  echo "generate-ios-icons: initialize the Tauri iOS project first" >&2
  exit 1
}

work=$(mktemp -d "${TMPDIR:-/tmp}/mold-ios-icons.XXXXXX")
cleanup() {
  find "$work" -depth -delete
}
trap cleanup EXIT

# iOS applies its own rounded mask. Keep the source square, flatten Mold's
# transparent desktop artwork onto the mobile theme, and remove alpha so the
# App Store marketing icon is valid.
magick "$SOURCE_ICON" \
  -background "$BACKGROUND" \
  -alpha remove \
  -alpha off \
  "PNG24:$work/app-icon.png"

(
  cd "$ROOT/apps/mobile/src-tauri"
  cargo tauri icon "$work/app-icon.png" --ios-color "$BACKGROUND" --output "$work/generated"
)

for generated in "$work"/generated/ios/*.png; do
  filename=$(basename "$generated")
  magick "$generated" -alpha off "PNG24:$APP_ICON_SET/$filename"
done

# Keep the exact opaque 1024px marketing asset as the canonical iOS icon. The
# Tauri Rust context separately needs the RGBA desktop icon configured in
# tauri.conf.json, while Xcode and TestFlight consume this opaque catalog.
cp "$APP_ICON_SET/AppIcon-512@2x.png" "$MOBILE_ICON"

source_hash=$(shasum -a 256 "$SOURCE_ICON" | awk '{print $1}')
printf '%s\n' "$source_hash" > "$SOURCE_CHECKSUM"
(
  cd "$APP_ICON_SET"
  shasum -a 256 AppIcon-*.png | LC_ALL=C sort -k2 > "$CATALOG_CHECKSUM"
)

echo "generate-ios-icons: updated Mold iOS app icons"
