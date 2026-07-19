#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOBILE="$ROOT/apps/mobile/src-tauri"
ACTION="${1:-dev}"
shift || true

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
  touch src/lib.rs
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
  dev) cargo tauri ios dev "$@" ;;
  run)
    prepare_build
    cargo tauri ios run "$@"
    ;;
  simulator)
    prepare_build
    cargo tauri ios build --debug --target aarch64-sim --no-sign --ci "$@"
    test -x gen/apple/build/arm64-sim/Mold.app/Mold || {
      echo "iOS simulator build did not produce Mold.app" >&2
      exit 1
    }
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
