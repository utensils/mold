#!/usr/bin/env bash
# A Release that cannot verify an update must not ship.
#
# Sparkle authenticates every appcast item against `SUPublicEDKey`
# (https://sparkle-project.org/documentation -- "Configure EdDSA public key in
# Info.plist"). mold commits a PLACEHOLDER for that key, because the private
# half is the owner's and lives only in the `MOLD_NATIVE_SPARKLE_KEY` secret.
# A Release built before the real public key is in place would happily run an
# updater whose signature check can never pass -- or, worse, be pointed at a
# feed by someone who can serve one. So the placeholder fails the build here,
# BEFORE signing, rather than at a user's first update check.
#
# Debug is deliberately untouched: `make build`, `make uat` and the test scheme
# never call this, and a dev build has no updater at all (`UpdaterActivation`).
set -euo pipefail

APP="${1:?usage: assert-sparkle-key.sh <path to .app>}"
PLIST="$APP/Contents/Info.plist"

if [[ ! -f "$PLIST" ]]; then
  echo "no Info.plist in $APP -- nothing was built?" >&2
  exit 1
fi

read_key() {
  /usr/libexec/PlistBuddy -c "Print :$1" "$PLIST" 2>/dev/null || true
}

key="$(read_key SUPublicEDKey)"
feed="$(read_key SUFeedURL)"

if [[ -z "$key" || "$key" == 'REPLACE_WITH_SPARKLE_PUBLIC_KEY' || "$key" == '$('* ]]; then
  cat >&2 <<'MESSAGE'
SUPublicEDKey is still the placeholder.

Run Sparkle's `generate_keys` once, put the PRIVATE half in the
MOLD_NATIVE_SPARKLE_KEY repository secret, and paste the PUBLIC half into
apps/macos/project.yml as MOLD_SPARKLE_PUBLIC_ED_KEY. See the "Updates"
section of apps/macos/README.md.
MESSAGE
  exit 1
fi

# Valid base64 is not enough: an Ed25519 public key is exactly 32 bytes, and
# anything else is a key no signature will ever verify against.
bytes="$(printf '%s' "$key" | base64 --decode 2>/dev/null | wc -c | tr -d ' ')" || bytes=0
if [[ "$bytes" != "32" ]]; then
  echo "SUPublicEDKey does not decode to a 32-byte Ed25519 key (got ${bytes:-0} bytes)" >&2
  exit 1
fi

# The feed is the other half of the same promise, and it is pinned in exactly
# the shape `UpdateFeedTests` pins the runtime choice: https, this repository's
# releases, an .xml document. A signed appcast served from somewhere else is
# not the feed this app was reviewed for.
case "$feed" in
  https://github.com/utensils/mold/releases/*.xml) ;;
  *)
    echo "SUFeedURL is not an https appcast under this repository's releases: ${feed:-<missing>}" >&2
    exit 1
    ;;
esac

echo "  sparkle key ok: ${key:0:8}… (32 bytes), feed $feed"
