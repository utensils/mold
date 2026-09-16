#!/usr/bin/env bash
# Sign Mold.app for distribution.
#
# Deliberately NOT `--deep`: Apple has discouraged it for years because it
# re-signs nested code with the OUTER bundle's entitlements, which is how a
# helper quietly gains permissions it was never reviewed for. Signing
# depth-first, innermost outwards, is the supported way.
set -euo pipefail

APP="${1:?usage: sign-release.sh <path to .app> [identity]}"
IDENTITY="${2:-${MOLD_SIGN_IDENTITY:-}}"

if [ -z "$IDENTITY" ]; then
  echo "No signing identity. Set MOLD_SIGN_IDENTITY or pass one." >&2
  exit 1
fi

ENTITLEMENTS="$(dirname "$0")/Mold.entitlements"

sign() {
  codesign --force --timestamp --options runtime \
    --generate-entitlement-der \
    ${ENTITLEMENTS:+--entitlements "$ENTITLEMENTS"} \
    --sign "$IDENTITY" "$1"
}

# Innermost first. `find -depth` gives exactly that order.
while IFS= read -r nested; do
  [ "$nested" = "$APP" ] && continue
  echo "  signing $(basename "$nested")"
  sign "$nested"
done < <(find "$APP" -depth \( -name '*.framework' -o -name '*.dylib' -o -name '*.app' \))

echo "  signing $(basename "$APP")"
sign "$APP"

codesign --verify --strict --verbose=2 "$APP"
echo "signed: $APP"
