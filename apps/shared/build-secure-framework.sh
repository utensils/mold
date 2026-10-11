#!/usr/bin/env bash
# Build the same Rust crypto for macOS, iPhone, and Simulator. No Swift crypto fork.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
artifact="$repo/apps/shared/Packages/MoldClient/build/MoldSecure.xcframework"
staging="$repo/apps/shared/build/secure"
export CARGO_PROFILE_RELEASE_LTO=false
export CARGO_TARGET_DIR="${MOLD_SECURE_TARGET_DIR:-$repo/apps/shared/build/secure-cargo}"
toolchain="${MOLD_SECURE_TOOLCHAIN:-stable}"
export CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER="$(xcrun --find clang)"
export CARGO_TARGET_X86_64_APPLE_DARWIN_LINKER="$CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER"
export CARGO_TARGET_AARCH64_APPLE_IOS_LINKER="$CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER"
export CARGO_TARGET_AARCH64_APPLE_IOS_SIM_LINKER="$CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER"
export CARGO_TARGET_X86_64_APPLE_IOS_LINKER="$CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER"
mkdir -p "$(dirname "$staging")"
lock="$staging.lock"
for attempt in {1..180}; do
  if mkdir "$lock" 2>/dev/null; then break; fi
  if [[ "$attempt" == 180 ]]; then echo "Another secure framework build still owns $lock" >&2; exit 1; fi
  sleep 1
done
trap 'rmdir "$lock"' EXIT
mkdir -p "$staging/headers"
cp "$repo/crates/mold-secure/include/mold_secure.h" "$staging/headers/"
cp "$repo/crates/mold-secure/include/module.modulemap" "$staging/headers/"
targets=(aarch64-apple-darwin x86_64-apple-darwin aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios)
# Keep inventory failures outside the conditional: they must stop the build,
# not masquerade as a missing target. The host runner need not install ripgrep.
installed_targets=$(rustup target list --installed --toolchain "$toolchain")
for target in "${targets[@]}"; do
  if ! grep -Fx -- "$target" <<< "$installed_targets" >/dev/null; then
    rustup target add --toolchain "$toolchain" "$target"
  fi
  MACOSX_DEPLOYMENT_TARGET=26.0 IPHONEOS_DEPLOYMENT_TARGET=26.0 \
    cargo "+$toolchain" build --manifest-path "$repo/Cargo.toml" --locked --release -p mold-ai-secure --target "$target"
done
for target in "${targets[@]}"; do
  case "$target" in
    aarch64-apple-darwin) arch=arm64; platform=macos; sdk=macosx ;;
    x86_64-apple-darwin) arch=x86_64; platform=macos; sdk=macosx ;;
    aarch64-apple-ios) arch=arm64; platform=ios; sdk=iphoneos ;;
    aarch64-apple-ios-sim) arch=arm64; platform=ios-simulator; sdk=iphonesimulator ;;
    x86_64-apple-ios) arch=x86_64; platform=ios-simulator; sdk=iphonesimulator ;;
  esac
  "$repo/apps/shared/scripts/isolate-rust-staticlib.sh" \
    "$CARGO_TARGET_DIR/$target/release/libmold_secure.a" "$arch" "$platform" "$sdk" \
    "$staging/headers/mold_secure.h" "$staging/libmold_secure_$target.a"
done
digest=$(for target in "${targets[@]}"; do shasum -a 256 "$staging/libmold_secure_$target.a"; done
  shasum -a 256 "$staging/headers/mold_secure.h" "$staging/headers/module.modulemap")
if [[ -d "$artifact" && -f "$staging/input.sha256" && "$(cat "$staging/input.sha256")" == "$digest" ]]; then exit 0; fi
xcrun lipo -create "$staging/libmold_secure_aarch64-apple-darwin.a" \
  "$staging/libmold_secure_x86_64-apple-darwin.a" -output "$staging/libmold_secure_mac.a"
xcrun lipo -create "$staging/libmold_secure_aarch64-apple-ios-sim.a" \
  "$staging/libmold_secure_x86_64-apple-ios.a" -output "$staging/libmold_secure_sim.a"
rm -rf "$artifact"
xcodebuild -create-xcframework \
  -library "$staging/libmold_secure_mac.a" -headers "$staging/headers" \
  -library "$staging/libmold_secure_aarch64-apple-ios.a" -headers "$staging/headers" \
  -library "$staging/libmold_secure_sim.a" -headers "$staging/headers" \
  -output "$artifact"

printf '%s\n' "$digest" > "$staging/input.sha256"
