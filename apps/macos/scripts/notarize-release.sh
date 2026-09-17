#!/usr/bin/env bash
# Notarize and staple a DMG.
#
# Needs a keychain profile created once with:
#   xcrun notarytool store-credentials mold-notary \
#     --apple-id <id> --team-id <team> --password <app-specific-password>
set -euo pipefail

DMG="${1:?usage: notarize-release.sh <path to .dmg> [keychain-profile]}"
PROFILE="${2:-${MOLD_NOTARY_PROFILE:-mold-notary}}"

# `--wait` returns successfully for a submission that was ACCEPTED and for one
# that was rejected alike; only `stapler` below then fails, with nothing to
# read. The status is checked here and the log is fetched on anything but
# Accepted, because "Invalid" with no reason is not a diagnosis (review 05-L3).
submission="$(xcrun notarytool submit "$DMG" --keychain-profile "$PROFILE" --wait --output-format json)"
echo "$submission"
id="$(printf '%s' "$submission" | sed -n 's/.*"id" *: *"\([^"]*\)".*/\1/p' | head -1)"
status="$(printf '%s' "$submission" | sed -n 's/.*"status" *: *"\([^"]*\)".*/\1/p' | head -1)"
if [ "$status" != "Accepted" ]; then
  echo "notarization was $status; the log follows" >&2
  [ -n "$id" ] && xcrun notarytool log "$id" --keychain-profile "$PROFILE" >&2 || true
  exit 1
fi

# Stapling lets the DMG validate with no network, which is the difference
# between "opens" and "opens on a machine that is offline or behind a
# captive portal".
xcrun stapler staple "$DMG"
xcrun stapler validate "$DMG"

# What Gatekeeper will actually say. `stapler validate` answers "a ticket is
# attached", which is close but not the same question, and nothing else in
# `make release` ever gets a positive assessment (review F10).
spctl --assess --type open --context context:primary-signature --verbose=4 "$DMG"
echo "notarized: $DMG"
