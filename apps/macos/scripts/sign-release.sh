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

# Nested code gets NO entitlements. Passing the app's to every nested dylib
# and framework is precisely the `--deep` behaviour this script's own header
# criticises -- a helper quietly granted JIT, unsigned executable memory and
# library validation off, because the outer bundle needs them. Only the app
# carries them, and only the app is reviewed for them (review 05-M14, 05-L2).
sign_nested() {
  codesign --force --timestamp --options runtime --generate-entitlement-der \
    --sign "$IDENTITY" "$1"
}

sign_app() {
  codesign --force --timestamp --options runtime --generate-entitlement-der \
    --entitlements "$ENTITLEMENTS" --sign "$IDENTITY" "$1"
}

# Innermost first. `find -depth` gives exactly that order. `.bundle`, `.xpc`
# and `.appex` are here because a bundle is not only frameworks and dylibs;
# today SwiftPM links its dependencies statically and this loop is usually
# empty, which is not a reason to be wrong when it stops being.
while IFS= read -r nested; do
  [ "$nested" = "$APP" ] && continue
  echo "  signing $(basename "$nested")"
  sign_nested "$nested"
done < <(find "$APP" -depth \
  \( -name '*.framework' -o -name '*.dylib' -o -name '*.app' \
     -o -name '*.bundle' -o -name '*.xpc' -o -name '*.appex' \))

echo "  signing $(basename "$APP")"
sign_app "$APP"

# `--deep` on VERIFY is not the `--deep` on sign that Apple discourages: here
# it means "check the nested code too", which is the only way this loop's
# output is actually verified.
codesign --verify --deep --strict --verbose=2 "$APP"
# What Gatekeeper will say on the other Mac. Before notarization it reports
# the missing ticket, so a failure is only fatal once stapled.
spctl --assess --type execute --verbose=4 "$APP" || \
  echo "  (not yet notarized -- spctl will pass after 'make notarize')"
echo "signed: $APP"
