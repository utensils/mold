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

# Sparkle's Downloader XPC service carries its OWN sandbox entitlements and is
# the one piece of nested code that must keep them. Sparkle's signing
# instructions say so explicitly -- `codesign … --preserve-metadata=entitlements
# Sparkle.framework/Versions/B/XPCServices/Downloader.xpc`
# (https://sparkle-project.org/documentation/sandboxing, "Manually Re-sign
# Sparkle XPC Services"). Every other nested item there, Installer.xpc and
# Updater.app and Autoupdate included, is signed with NO entitlements, which is
# what this script already does. Mold is NOT sandboxed
# (`ENABLE_APP_SANDBOX: "NO"`), so none of that page's other requirements apply
# to us: no `SUEnableInstallerLauncherService`, and no
# `com.apple.security.temporary-exception.mach-lookup.global-name` pair -- those
# exist so a SANDBOXED app can reach its own installer, and adding them here
# would be granting an exception for a sandbox we do not have.
sign_preserving_entitlements() {
  codesign --force --timestamp --options runtime --generate-entitlement-der \
    --preserve-metadata=entitlements --sign "$IDENTITY" "$1"
}

# Innermost first. `find -depth` gives exactly that order. `.bundle`, `.xpc`
# and `.appex` are here because a bundle is not only frameworks and dylibs;
# SwiftPM links most dependencies statically, but Sparkle is a real framework
# with an `Updater.app`, two XPC services and an `Autoupdate` tool inside it,
# and every one of them is code that Gatekeeper will assess.
#
# `Autoupdate` is a bare Mach-O executable, so no name pattern finds it: it is
# named here, by the path Sparkle's own instructions use.
sparkle_code() {
  local framework="$APP/Contents/Frameworks/Sparkle.framework" versions helper
  [ -d "$framework" ] || return 0
  # Its helpers first, in the order Sparkle's own instructions sign them, then
  # the framework that wraps them. Version `B` is Sparkle 2's; the glob keeps
  # this working if that ever changes.
  for versions in "$framework"/Versions/*/; do
    # `Versions/Current` is a symlink to `Versions/B`, and it matches this
    # glob. Following it signs every helper a second time under a second path.
    [ -d "$versions" ] && [ ! -L "${versions%/}" ] || continue
    for helper in XPCServices/Installer.xpc XPCServices/Downloader.xpc Autoupdate Updater.app; do
      if [ -e "$versions$helper" ]; then printf '%s\n' "$versions$helper"; fi
    done
  done
  printf '%s\n' "$framework"
}

while IFS= read -r nested; do
  [ "$nested" = "$APP" ] && continue
  echo "  signing $(basename "$nested")"
  case "$nested" in
    */XPCServices/Downloader.xpc) sign_preserving_entitlements "$nested" ;;
    *) sign_nested "$nested" ;;
  esac
done < <({
  # Everything else. Sparkle is cut out of this sweep entirely -- the framework
  # AND its contents -- because `find -depth` would otherwise sign the wrapper
  # before the helpers inside it, which invalidates them.
  find "$APP" -depth \
    \( -name '*.framework' -o -name '*.dylib' -o -name '*.app' \
       -o -name '*.bundle' -o -name '*.xpc' -o -name '*.appex' \) \
    ! -path '*/Sparkle.framework' ! -path '*/Sparkle.framework/*'
  sparkle_code
})

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
