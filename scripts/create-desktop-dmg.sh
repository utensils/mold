#!/usr/bin/env bash
set -euo pipefail

app_path="${1:?usage: create-desktop-dmg.sh <app> <output.dmg> <signing-identity>}"
output_path="${2:?usage: create-desktop-dmg.sh <app> <output.dmg> <signing-identity>}"
signing_identity="${3:?usage: create-desktop-dmg.sh <app> <output.dmg> <signing-identity>}"
max_attempts="${MOLD_DMG_MAX_ATTEMPTS:-3}"
retry_delay_seconds="${MOLD_DMG_RETRY_DELAY_SECONDS:-5}"

if [[ ! -d "$app_path" || "$app_path" != *.app ]]; then
  echo "desktop app bundle not found: $app_path" >&2
  exit 1
fi
if [[ "$output_path" != *.dmg ]]; then
  echo "desktop DMG output must end in .dmg: $output_path" >&2
  exit 1
fi
if [[ ! "$max_attempts" =~ ^[1-9][0-9]*$ ]]; then
  echo "MOLD_DMG_MAX_ATTEMPTS must be a positive integer" >&2
  exit 1
fi
if [[ ! "$retry_delay_seconds" =~ ^[0-9]+$ ]]; then
  echo "MOLD_DMG_RETRY_DELAY_SECONDS must be a non-negative integer" >&2
  exit 1
fi
if [[ -e "$output_path" && ! -f "$output_path" ]]; then
  echo "desktop DMG output exists but is not a regular file: $output_path" >&2
  exit 1
fi

output_dir="$(dirname "$output_path")"
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
output_path="$output_dir/$(basename "$output_path")"

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT
stage_dir="$work_dir/stage"
attempt_dmg="$work_dir/Mold.dmg"
mkdir -p "$stage_dir"

# ditto preserves the signed bundle's metadata and resource forks.
ditto "$app_path" "$stage_dir/Mold.app"
ln -s /Applications "$stage_dir/Applications"
volume_icon="$app_path/Contents/Resources/icon.icns"
if [[ -f "$volume_icon" ]]; then
  ditto "$volume_icon" "$stage_dir/.VolumeIcon.icns"
  SetFile -c icnC "$stage_dir/.VolumeIcon.icns"
  SetFile -a C "$stage_dir"
fi

created=false
for ((attempt = 1; attempt <= max_attempts; attempt++)); do
  rm -f "$attempt_dmg"
  echo "Creating desktop DMG (attempt $attempt/$max_attempts)..."
  if hdiutil create -verbose \
    -srcfolder "$stage_dir" \
    -volname Mold \
    -fs HFS+ \
    -format UDZO \
    -imagekey zlib-level=9 \
    "$attempt_dmg"; then
    created=true
    break
  fi

  if ((attempt < max_attempts)); then
    delay=$((retry_delay_seconds * attempt))
    echo "hdiutil failed; retrying in ${delay}s..." >&2
    sleep "$delay"
  fi
done

if [[ "$created" != true || ! -f "$attempt_dmg" ]]; then
  echo "hdiutil failed to create the desktop DMG after $max_attempts attempts" >&2
  exit 1
fi

codesign --force --sign "$signing_identity" --timestamp "$attempt_dmg"
mv -f "$attempt_dmg" "$output_path"
echo "Created signed desktop DMG: $output_path"
