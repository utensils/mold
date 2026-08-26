#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MOBILE="$ROOT/apps/mobile/src-tauri"
SOURCE_ICON="$ROOT/desktop/icon-master.png"
ICON_MANIFEST="$MOBILE/android-icon.json"
SOURCE_CHECKSUM="$MOBILE/android-icon-source.sha256"
ANDROID_RES="$MOBILE/gen/android/app/src/main/res"
CATALOG_CHECKSUM="$ANDROID_RES/ANDROID_ICON_SHA256SUMS"

command -v cargo >/dev/null || {
  echo "generate-android-icons: cargo is required" >&2
  exit 1
}
test -f "$SOURCE_ICON" || {
  echo "generate-android-icons: missing $SOURCE_ICON" >&2
  exit 1
}
test -d "$ANDROID_RES" || {
  echo "generate-android-icons: initialize the Tauri Android project first" >&2
  exit 1
}

work=$(mktemp -d "${TMPDIR:-/tmp}/mold-android-icons.XXXXXX")
cleanup() {
  find "$work" -depth -delete
}
trap cleanup EXIT

(
  cd "$MOBILE"
  cargo tauri icon "$ICON_MANIFEST" --output "$work/generated"
)

find "$ANDROID_RES" -maxdepth 2 -type f \
  \( -path '*/mipmap-*/*launcher*' -o -path '*/drawable*/ic_launcher*' \) \
  -delete

while IFS= read -r generated; do
  relative=${generated#"$work/generated/android/"}
  mkdir -p "$ANDROID_RES/$(dirname "$relative")"
  cp "$generated" "$ANDROID_RES/$relative"
done < <(find "$work/generated/android" -type f | LC_ALL=C sort)

source_hash=$(shasum -a 256 "$SOURCE_ICON" | awk '{print $1}')
printf '%s\n' "$source_hash" > "$SOURCE_CHECKSUM"
(
  cd "$ANDROID_RES"
  find mipmap-* values -type f \
    \( -name 'ic_launcher*' -o -name 'ic_launcher_background.xml' \) \
    -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 shasum -a 256 > "$CATALOG_CHECKSUM"
)

echo "generate-android-icons: updated Mold Android launcher icons"
