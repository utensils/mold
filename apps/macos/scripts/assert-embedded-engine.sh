#!/usr/bin/env bash
# Refuse to ship an app that has no engine in it.
#
# `Engine.xcconfig` is generated and gitignored, so on a fresh clone it does
# not exist, `MOLD_EMBEDDED_ENGINE` is never defined, and the notarized build
# is a remote-only client that says "This build has no local engine." Nothing
# in `release: signed dmg notarize` depended on `engine`, so that shipped
# silently (review 05-H4).
#
# Two checks, because the xcconfig alone only says what was ASKED for: the
# flag has to be defined, and the built binary has to actually carry the
# engine's own bytes.
set -euo pipefail

APP="${1:?usage: assert-embedded-engine.sh <path to .app>}"
CONFIG="$(dirname "$0")/../Engine.xcconfig"
BINARY="$APP/Contents/MacOS/Mold"

# A string only the Rust staticlib contains: the CORS origin the embedded
# engine is started with (`rust/mold-macos-ffi/src/lib.rs`).
MARKER='mold-embedded-engine no browser origin'

remote_only() {
  cat >&2 <<EOF
$1

This build would ship as a remote-only client: Settings > This Mac would read
"This build has no local engine." Run \`make engine\` first, or pass
ALLOW_REMOTE_ONLY=1 to say that is what you meant.
EOF
  exit 1
}

if [[ ! -f "$CONFIG" ]] || ! grep -q 'MOLD_EMBEDDED_ENGINE' "$CONFIG"; then
  remote_only "Engine.xcconfig does not define MOLD_EMBEDDED_ENGINE."
fi
if [[ ! -f "$BINARY" ]]; then
  echo "missing binary: $BINARY" >&2
  exit 1
fi
if ! LC_ALL=C grep -a -q -F "$MARKER" "$BINARY"; then
  remote_only "$BINARY does not contain the engine."
fi

echo "  engine embedded: $BINARY"
