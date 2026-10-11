#!/usr/bin/env bash
# Build the same Rust crypto for macOS, iPhone, and Simulator. No Swift crypto fork.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
artifact="$repo/apps/shared/Packages/MoldClient/build/MoldSecure.xcframework"
staging="$repo/apps/shared/build/secure"
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
for target in "${targets[@]}"; do
  if ! rustup target list --installed --toolchain "$toolchain" | rg -qx "$target"; then
    rustup target add --toolchain "$toolchain" "$target"
  fi
  MACOSX_DEPLOYMENT_TARGET=26.0 IPHONEOS_DEPLOYMENT_TARGET=26.0 \
    cargo "+$toolchain" build --manifest-path "$repo/Cargo.toml" --locked --release -p mold-ai-secure --target "$target"
done
digest=$(for target in "${targets[@]}"; do shasum -a 256 "$CARGO_TARGET_DIR/$target/release/libmold_secure.a"; done
  shasum -a 256 "$staging/headers/mold_secure.h" "$staging/headers/module.modulemap")
if [[ -d "$artifact" && -f "$staging/input.sha256" && "$(cat "$staging/input.sha256")" == "$digest" ]]; then exit 0; fi
xcrun lipo -create "$CARGO_TARGET_DIR/aarch64-apple-darwin/release/libmold_secure.a" \
  "$CARGO_TARGET_DIR/x86_64-apple-darwin/release/libmold_secure.a" -output "$staging/libmold_secure_mac.a"
xcrun lipo -create "$CARGO_TARGET_DIR/aarch64-apple-ios-sim/release/libmold_secure.a" \
  "$CARGO_TARGET_DIR/x86_64-apple-ios/release/libmold_secure.a" -output "$staging/libmold_secure_sim.a"
rm -rf "$artifact"
xcodebuild -create-xcframework \
  -library "$staging/libmold_secure_mac.a" -headers "$staging/headers" \
  -library "$CARGO_TARGET_DIR/aarch64-apple-ios/release/libmold_secure.a" -headers "$staging/headers" \
  -library "$staging/libmold_secure_sim.a" -headers "$staging/headers" \
  -output "$artifact"

printf '%s\n' "$digest" > "$staging/input.sha256"
