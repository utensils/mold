#!/usr/bin/env bash
# Prove `assert-sparkle-key.sh` refuses a Release that cannot verify updates.
#
# **Fails today**: there is no such script. The EdDSA public key is a build
# setting with a committed placeholder, and a Release built before the owner
# has generated the real key would ship an updater that accepts an appcast it
# cannot authenticate -- the one failure mode where "it still launches" is
# worse than not shipping.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

STABLE_FEED="https://github.com/utensils/mold/releases/latest/download/mold-native-appcast.xml"

# A real-shaped key: 32 bytes of Ed25519 public key, base64. Generated here so
# this fixture never carries anything that looks like mold's own.
GOOD_KEY="$(head -c 32 /dev/zero | tr '\0' 'A' | base64)"

bundle() {
  local name="$1" key="$2" feed="$3" app="$WORK/$name.app"
  mkdir -p "$app/Contents"
  cat > "$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleIdentifier</key>
	<string>io.utensils.mold.native</string>
	<key>SUFeedURL</key>
	<string>$feed</string>
	<key>SUPublicEDKey</key>
	<string>$key</string>
</dict>
</plist>
PLIST
  echo "$app"
}

accepts() {
  "$HERE/assert-sparkle-key.sh" "$1" >/dev/null 2>&1
}

good="$(bundle Good "$GOOD_KEY" "$STABLE_FEED")"
accepts "$good" || { echo "FAIL: a real 32-byte key and the stable feed were refused" >&2; exit 1; }

# The committed placeholder -- the whole reason this gate exists.
placeholder="$(bundle Placeholder 'REPLACE_WITH_SPARKLE_PUBLIC_KEY' "$STABLE_FEED")"
if accepts "$placeholder"; then
  echo "FAIL: the placeholder public key must stop a Release" >&2
  exit 1
fi

# An unexpanded build setting: what a bundle carries when the setting is not
# defined at all, which reads as "present" to a naive grep.
unexpanded="$(bundle Unexpanded '$(MOLD_SPARKLE_PUBLIC_ED_KEY)' "$STABLE_FEED")"
if accepts "$unexpanded"; then
  echo "FAIL: an unexpanded build setting must stop a Release" >&2
  exit 1
fi

empty="$(bundle Empty '' "$STABLE_FEED")"
if accepts "$empty"; then
  echo "FAIL: an empty public key must stop a Release" >&2
  exit 1
fi

# Valid base64 of the WRONG length. Sparkle wants 32 bytes; anything else is a
# key that will never verify, and it would sail past a length-blind check.
short="$(bundle Short "$(printf 'too short' | base64)" "$STABLE_FEED")"
if accepts "$short"; then
  echo "FAIL: base64 that is not 32 bytes must stop a Release" >&2
  exit 1
fi

# The feed is half of the same promise: a signed appcast fetched over http, or
# from somewhere else entirely, is not the feed this app was reviewed for.
for bad_feed in \
  "http://github.com/utensils/mold/releases/latest/download/mold-native-appcast.xml" \
  "https://example.com/mold-native-appcast.xml" \
  "https://github.com/utensils/mold/issues/mold-native-appcast.xml" \
  "https://github.com/utensils/mold/releases/latest/download/mold-native-appcast.json"
do
  feed="$(bundle Feed "$GOOD_KEY" "$bad_feed")"
  if accepts "$feed"; then
    echo "FAIL: the gate accepted an off-allowlist feed: $bad_feed" >&2
    exit 1
  fi
  rm -rf "$feed"
done

# And it must not pass by finding nothing: a bundle with no Info.plist at all
# is a broken build, not an acceptable one.
mkdir -p "$WORK/Bare.app/Contents"
if accepts "$WORK/Bare.app"; then
  echo "FAIL: a bundle with no Info.plist must stop a Release" >&2
  exit 1
fi

echo "sparkle key gate ok"
