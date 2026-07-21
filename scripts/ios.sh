#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOBILE="$ROOT/apps/mobile/src-tauri"
ACTION="${1:-dev}"
shift || true

# pick_device / device_udid (simulator selection, tested by
# scripts/tests/ios-sim-pick.sh).
source "$ROOT/scripts/ios-sim.sh"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Mold iOS commands require macOS and Xcode." >&2
  exit 1
fi

export PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
export CC=/usr/bin/cc
export CXX=/usr/bin/c++
export CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER=/usr/bin/cc
export CARGO_TARGET_AARCH64_APPLE_IOS_LINKER=/usr/bin/clang
export CARGO_TARGET_AARCH64_APPLE_IOS_SIM_LINKER=/usr/bin/clang

prepare_build() {
  if [[ -d gen/apple/build ]]; then
    find gen/apple/build -depth -delete
  fi
}

command -v xcodebuild >/dev/null || { echo "Xcode command-line tools are required." >&2; exit 1; }
command -v pod >/dev/null || { echo "CocoaPods is required (brew install cocoapods)." >&2; exit 1; }

cd "$MOBILE"
if [[ ! -d gen/apple && "$ACTION" != "init" ]]; then
  cargo tauri ios init --ci
  "$ROOT/scripts/generate-ios-icons.sh"
fi

case "$ACTION" in
  init)
    cargo tauri ios init --ci "$@"
    "$ROOT/scripts/generate-ios-icons.sh"
    ;;
  check) cargo check --target aarch64-apple-ios-sim "$@" ;;
  dev)
    # Zero-arg default: pick a simulator by name so the Tauri CLI never
    # grabs a plugged-in iPhone (a provisioning device build) or prompts.
    # Pass a device name to target hardware; MOLD_IOS_DEVICE overrides the
    # preferred simulator.
    if [[ $# -eq 0 ]]; then
      device="$(pick_device)"
      if [[ -z "$device" ]]; then
        echo "error: no available iPhone simulator (xcrun simctl list devices available)" >&2
        exit 1
      fi
      # The CLI installs without booting — a Shutdown target fails deploy
      # with SimError 405 ("Unable to lookup in current state"). Pre-boot
      # the exact UDID the CLI will resolve the name to.
      udid="$(device_udid "$device")"
      if [[ -n "$udid" ]]; then
        echo "==> booting $device ($udid)"
        xcrun simctl boot "$udid" 2>/dev/null || true # already booted is fine
        open -a Simulator
      fi
      set -- "$device"
    fi
    cargo tauri ios dev "$@"
    ;;
  run)
    prepare_build
    cargo tauri ios run "$@"
    ;;
  simulator)
    prepare_build
    # Keep Tauri/Xcode's local simulator signature. A --no-sign build is only
    # linker-signed and cannot read or write the iOS Keychain.
    cargo tauri ios build --debug --target aarch64-sim --ci "$@"
    test -x gen/apple/build/arm64-sim/Mold.app/Mold || {
      echo "iOS simulator build did not produce Mold.app" >&2
      exit 1
    }
    codesign --verify --deep --strict gen/apple/build/arm64-sim/Mold.app
    ;;
  build)
    prepare_build
    cargo tauri ios build --target aarch64 --export-method app-store-connect --ci "$@"
    test -x gen/apple/build/mold-mobile_iOS.xcarchive/Products/Applications/Mold.app/Mold || {
      echo "iOS archive did not produce a signed Mold.app" >&2
      exit 1
    }
    ;;
  *)
    echo "usage: scripts/ios.sh {init|check|dev|run|simulator|build} [args...]" >&2
    exit 2
    ;;
esac
