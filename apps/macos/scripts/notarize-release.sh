#!/usr/bin/env bash
# Notarize and staple a DMG.
#
# Needs a keychain profile created once with:
#   xcrun notarytool store-credentials mold-notary \
#     --apple-id <id> --team-id <team> --password <app-specific-password>
set -euo pipefail

DMG="${1:?usage: notarize-release.sh <path to .dmg> [keychain-profile]}"
PROFILE="${2:-${MOLD_NOTARY_PROFILE:-mold-notary}}"

xcrun notarytool submit "$DMG" --keychain-profile "$PROFILE" --wait

# Stapling lets the DMG validate with no network, which is the difference
# between "opens" and "opens on a machine that is offline or behind a
# captive portal".
xcrun stapler staple "$DMG"
xcrun stapler validate "$DMG"
echo "notarized: $DMG"
