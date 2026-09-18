#!/usr/bin/env bash
# Prove `notarize-release.sh` hands notarytool the credentials it was given:
# an App Store Connect API key when the workflow exports one, the Keychain
# profile otherwise.
#
# **Fails today**: the script only knows the profile, so on a runner -- which
# has no Keychain profile at all -- nightly 1393 built, signed and packaged
# the DMG and then stopped at "No Keychain password item found".
#
# `xcrun` is stubbed: notarytool, stapler and spctl are Apple's and need a
# real submission; the argv is the surface this script decides.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK/bin"
cat > "$WORK/bin/xcrun" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$OUT_DIR/argv"
case "$1 $2" in
  "notarytool submit") printf '{"id":"sub-1","status":"Accepted"}\n' ;;
esac
STUB
cat > "$WORK/bin/spctl" <<'STUB'
#!/usr/bin/env bash
printf 'spctl %s\n' "$*" >> "$OUT_DIR/argv"
STUB
chmod +x "$WORK/bin/xcrun" "$WORK/bin/spctl"
touch "$WORK/fake.dmg" "$WORK/AuthKey_ABC123.p8"

run() {
  : > "$WORK/argv"
  env OUT_DIR="$WORK" PATH="$WORK/bin:$PATH" "$@" "$HERE/notarize-release.sh" "$WORK/fake.dmg" >/dev/null
}

# CI: the API key the workflow exports wins, and no profile is named.
run env APPLE_API_KEY_PATH="$WORK/AuthKey_ABC123.p8" APPLE_API_KEY=ABC123 APPLE_API_ISSUER=issuer-1
grep -q -- "notarytool submit .*--key $WORK/AuthKey_ABC123.p8 --key-id ABC123 --issuer issuer-1" "$WORK/argv" \
  || { echo "FAIL: the API key was not handed to notarytool"; cat "$WORK/argv"; exit 1; }
grep -q -- "--keychain-profile" "$WORK/argv" \
  && { echo "FAIL: a Keychain profile was named beside the API key"; exit 1; }

# A Mac: the profile, by its default name.
run env -u APPLE_API_KEY_PATH -u APPLE_API_KEY -u APPLE_API_ISSUER
grep -q -- "notarytool submit .*--keychain-profile mold-notary" "$WORK/argv" \
  || { echo "FAIL: the Keychain profile was not used without an API key"; cat "$WORK/argv"; exit 1; }

# Every arm still staples and assesses what it notarized.
grep -q "stapler staple" "$WORK/argv" && grep -q "spctl --assess" "$WORK/argv" \
  || { echo "FAIL: the DMG was not stapled and assessed"; exit 1; }

# Half an API key is a misconfiguration, not a silent fall-through to a
# profile the runner does not have.
if run env APPLE_API_KEY_PATH="$WORK/AuthKey_ABC123.p8" -u APPLE_API_KEY -u APPLE_API_ISSUER 2>/dev/null; then
  echo "FAIL: a key path with no id or issuer must refuse"; exit 1
fi

echo "notarize credentials ok"
