#!/usr/bin/env bash
# Generate and sign one channel's appcast.
#
# Sparkle ships `generate_appcast` inside the SPM binary artifact it
# downloads, so there is nothing to install: the tool beside the XCFramework
# mold links against is the one that signs its feed, which is how the two can
# never drift.
#
# THE PRIVATE KEY NEVER TOUCHES THE DISK, THE PROCESS TABLE OR THE LOG. It
# arrives in `MOLD_NATIVE_SPARKLE_KEY` and goes to the tool on STDIN, which is
# the documented way (`--ed-key-file -` reads one line from stdin; see
# https://github.com/sparkle-project/Sparkle `generate_appcast` options, and
# its own example `echo "$PRIVATE_KEY_SECRET" | generate_appcast --ed-key-file -`).
# The deprecated `-s <key>` form would put it in argv, where every `ps` on the
# machine can read it.
#
# Two FEEDS, not Sparkle channels, so `--channel` is deliberately NOT passed
# to the tool: each stream has its own directory, its own output file and its
# own release. See `Sources/Mold/Support/UpdateFeed.swift` for why.
set -euo pipefail

CHANNEL=""
UPDATES_DIR=""
OUTPUT=""
DOWNLOAD_URL_PREFIX=""
LINK="https://utensils.io/mold/"
MAXIMUM_VERSIONS="10"
SPARKLE_BIN_DIR="${SPARKLE_BIN_DIR:-}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --channel) CHANNEL="$2"; shift 2 ;;
    --updates-dir) UPDATES_DIR="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --download-url-prefix) DOWNLOAD_URL_PREFIX="$2"; shift 2 ;;
    --link) LINK="$2"; shift 2 ;;
    --maximum-versions) MAXIMUM_VERSIONS="$2"; shift 2 ;;
    --sparkle-bin-dir) SPARKLE_BIN_DIR="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$CHANNEL" in
  stable) DEFAULT_OUTPUT="mold-native-appcast.xml" ;;
  nightly) DEFAULT_OUTPUT="mold-native-appcast-nightly.xml" ;;
  *) echo "--channel must be stable or nightly (got '${CHANNEL}')" >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "$0")/.." && pwd)"
UPDATES_DIR="${UPDATES_DIR:-$HERE/dist/updates}"
OUTPUT="${OUTPUT:-$UPDATES_DIR/$DEFAULT_OUTPUT}"

if [ -z "${MOLD_NATIVE_SPARKLE_KEY:-}" ]; then
  cat >&2 <<'MESSAGE'
MOLD_NATIVE_SPARKLE_KEY is not set.

It holds the PRIVATE half of the EdDSA key Sparkle's `generate_keys` made.
In CI it is the repository secret of that name; locally, export it for the
one command and do not write it to a file. Never commit it, and never pass
it on the command line.
MESSAGE
  exit 1
fi

if [ ! -d "$UPDATES_DIR" ]; then
  echo "no updates directory: $UPDATES_DIR" >&2
  exit 1
fi

# The tool, from the artifact SwiftPM already resolved. `--sparkle-bin-dir`
# and `SPARKLE_BIN_DIR` are for a caller who keeps DerivedData elsewhere.
if [ -z "$SPARKLE_BIN_DIR" ]; then
  SPARKLE_BIN_DIR="$(dirname "$(find "$HERE/build/DerivedData/SourcePackages/artifacts" \
    -type f -name generate_appcast -print -quit 2>/dev/null || true)")"
fi
GENERATE="$SPARKLE_BIN_DIR/generate_appcast"
if [ ! -x "$GENERATE" ]; then
  # An unqualified name on PATH is the last resort, and it is also what the
  # test stubs.
  if command -v generate_appcast > /dev/null 2>&1; then
    GENERATE="$(command -v generate_appcast)"
  else
    cat >&2 <<'MESSAGE'
generate_appcast not found.

It lives beside the Sparkle XCFramework SwiftPM resolves, so a build is what
produces it:
  make gen && make build
then re-run. Or point at it with --sparkle-bin-dir.
MESSAGE
    exit 1
  fi
fi

arguments=("$UPDATES_DIR" -o "$OUTPUT" --ed-key-file - --maximum-versions "$MAXIMUM_VERSIONS" --link "$LINK")
if [ -n "$DOWNLOAD_URL_PREFIX" ]; then
  arguments+=(--download-url-prefix "$DOWNLOAD_URL_PREFIX")
fi

echo "  appcast: $CHANNEL -> $OUTPUT"
printf '%s\n' "$MOLD_NATIVE_SPARKLE_KEY" | "$GENERATE" "${arguments[@]}"

if [ ! -s "$OUTPUT" ]; then
  echo "generate_appcast produced nothing at $OUTPUT" >&2
  exit 1
fi
# A feed with no signed enclosure is a feed no Sparkle will install from, and
# it is what an unsigned run silently produces.
if ! grep -q 'sparkle:edSignature' "$OUTPUT"; then
  echo "appcast carries no EdDSA signature -- refusing to publish it" >&2
  exit 1
fi
echo "  appcast signed: $(basename "$OUTPUT")"
