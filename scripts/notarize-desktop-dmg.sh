#!/usr/bin/env bash
set -euo pipefail

dmg="${1:?usage: notarize-desktop-dmg.sh <dmg>}"
if [[ ! -f "$dmg" ]]; then
  echo "missing desktop DMG: $dmg" >&2
  exit 1
fi

for name in APPLE_API_ISSUER APPLE_API_KEY APPLE_API_KEY_PATH; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing $name for DMG notarization" >&2
    exit 1
  fi
done

xcrun notarytool submit "$dmg" \
  --key "$APPLE_API_KEY_PATH" \
  --key-id "$APPLE_API_KEY" \
  --issuer "$APPLE_API_ISSUER" \
  --wait
xcrun stapler staple "$dmg"
