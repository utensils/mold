#!/usr/bin/env bash
# Prove `sign-release.sh` signs Sparkle the way Sparkle says to.
#
# **Fails today**: the script's `find -depth` sweep matched `Sparkle.framework`
# by name and everything inside it by name too, so the wrapper was signed
# somewhere in the middle of its own helpers -- which invalidates every one of
# them -- and `Autoupdate`, a bare Mach-O with no bundle extension, was never
# signed at all. Downloader.xpc's own sandbox entitlements were dropped.
#
# A real signature needs a real identity, which CI does not have and a test
# should not want. `codesign` and `spctl` are shadowed on PATH and simply
# record their arguments, so what is asserted is the ORDER and the FLAGS --
# which is the whole of what this script decides.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

APP="$WORK/Mold.app"
FRAMEWORK="$APP/Contents/Frameworks/Sparkle.framework/Versions/B"
mkdir -p "$APP/Contents/MacOS" \
  "$FRAMEWORK/XPCServices/Installer.xpc" \
  "$FRAMEWORK/XPCServices/Downloader.xpc" \
  "$FRAMEWORK/Updater.app"
touch "$APP/Contents/MacOS/Mold" "$FRAMEWORK/Autoupdate" "$FRAMEWORK/Sparkle"
ln -s B "$APP/Contents/Frameworks/Sparkle.framework/Versions/Current"
ln -s Versions/Current/XPCServices "$APP/Contents/Frameworks/Sparkle.framework/XPCServices"
# A nested item that is NOT Sparkle's, so the ordinary sweep is exercised too.
mkdir -p "$APP/Contents/Frameworks/Other.bundle"

mkdir -p "$WORK/bin"
LOG="$WORK/codesign.log"
cat > "$WORK/bin/codesign" <<STUB
#!/usr/bin/env bash
printf '%s\n' "\$*" >> "$LOG"
exit 0
STUB
cat > "$WORK/bin/spctl" <<'STUB'
#!/usr/bin/env bash
exit 0
STUB
chmod +x "$WORK/bin/codesign" "$WORK/bin/spctl"

PATH="$WORK/bin:$PATH" "$HERE/sign-release.sh" "$APP" "Fake Identity" >/dev/null

# The `--verify` pass at the end is not a signing call.
signed=$(grep -v -- '--verify' "$LOG" | sed "s#$APP#APP#g")

position() {
  local needle="$1" n
  n=$(printf '%s\n' "$signed" | grep -n -- "$needle" | head -1 | cut -d: -f1)
  if [[ -z "$n" ]]; then
    echo "FAIL: nothing signed $needle" >&2
    printf '%s\n' "$signed" >&2
    exit 1
  fi
  echo "$n"
}

installer=$(position 'XPCServices/Installer.xpc')
downloader=$(position 'XPCServices/Downloader.xpc')
autoupdate=$(position 'Versions/B/Autoupdate')
updater=$(position 'Versions/B/Updater.app')
framework=$(position 'Sparkle.framework$')
app=$(position 'APP$')

# Innermost first: every helper before the framework, the framework before the
# app. A wrapper signed before its contents is the bug this pins.
for helper in "$installer" "$downloader" "$autoupdate" "$updater"; do
  if [[ "$helper" -ge "$framework" ]]; then
    echo "FAIL: a Sparkle helper was signed at or after its framework" >&2
    printf '%s\n' "$signed" >&2
    exit 1
  fi
done
if [[ "$framework" -ge "$app" ]]; then
  echo "FAIL: Sparkle.framework was signed at or after the app" >&2
  exit 1
fi

# Downloader.xpc keeps its own entitlements; nothing else does, and the app's
# are never handed to nested code.
downloader_line=$(printf '%s\n' "$signed" | grep -- 'XPCServices/Downloader.xpc')
case "$downloader_line" in
  *--preserve-metadata=entitlements*) ;;
  *) echo "FAIL: Downloader.xpc lost its own entitlements: $downloader_line" >&2; exit 1 ;;
esac
while IFS= read -r line; do
  case "$line" in
    *XPCServices/Downloader.xpc*) continue ;;
    *--preserve-metadata=entitlements*)
      echo "FAIL: entitlements preserved on something that is not Downloader.xpc: $line" >&2
      exit 1
      ;;
  esac
  case "$line" in
    *--entitlements*)
      case "$line" in
        *'APP'|*'APP ') ;;
        *) echo "FAIL: the app's entitlements were handed to nested code: $line" >&2; exit 1 ;;
      esac
      ;;
  esac
done <<< "$signed"

# And it must not pass by finding nothing: the fixture has to have produced a
# real sweep, Sparkle's five plus the bundle plus the app.
count=$(printf '%s\n' "$signed" | grep -c .)
if [[ "$count" -lt 7 ]]; then
  echo "FAIL: only $count signing calls -- the fixture did not reproduce a real bundle" >&2
  printf '%s\n' "$signed" >&2
  exit 1
fi

# Nothing twice. `Versions/Current` is a symlink to `Versions/B`, so a glob
# that follows it signs every helper again under a second path -- wasted work
# on the release runner, and a second name for the same code in the log
# anyone reads when a signature is wrong.
paths=$(printf '%s\n' "$signed" | awk '{ print $NF }' | sort)
if [[ "$(printf '%s\n' "$paths" | uniq -d)" != "" ]]; then
  echo "FAIL: something was signed twice:" >&2
  printf '%s\n' "$paths" | uniq -d >&2
  exit 1
fi
if printf '%s\n' "$signed" | grep -q 'Versions/Current'; then
  echo "FAIL: the Versions/Current symlink was followed" >&2
  exit 1
fi

echo "sparkle signing order ok ($count items)"
