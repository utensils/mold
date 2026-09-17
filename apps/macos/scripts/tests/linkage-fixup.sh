#!/usr/bin/env bash
# Prove `fix-macos-native-linkage.sh` on a bundle that really does load
# libc++ and libiconv from a /nix/store path.
#
# A remote-only build carries no such load command, so the fixup could
# otherwise only be exercised by a 40-minute engine build. This synthesises
# the exact shape instead: an install_name is recorded in the load command
# verbatim, so the dylibs need only be BUILT with those names, not installed
# at them.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

APP="$WORK/Mold.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Frameworks"

fake_dylib() {
  local name="$1" out="$2"
  echo 'int mold_fake(void) { return 0; }' > "$WORK/fake.c"
  clang -dynamiclib -install_name "$name" -o "$out" "$WORK/fake.c"
}

fake_dylib '/nix/store/00000000000000000000000000000000-libcxx-19.1.7/lib/libc++.1.0.dylib' \
  "$WORK/libcxx.dylib"
fake_dylib '/nix/store/11111111111111111111111111111111-libiconv-109/lib/libiconv.2.dylib' \
  "$WORK/libiconv.dylib"

echo 'int main(void) { return 0; }' > "$WORK/main.c"
clang -o "$APP/Contents/MacOS/Mold" "$WORK/main.c" "$WORK/libcxx.dylib" "$WORK/libiconv.dylib" \
  -Wl,-rpath,/nix/store/22222222222222222222222222222222-some-lib/lib
# A nested Mach-O too: the desktop script fixes ONE binary, and a bundle is
# not one binary (review 05-H3, 05-L2).
clang -dynamiclib -install_name '@rpath/Nested.dylib' \
  -o "$APP/Contents/Frameworks/Nested.dylib" "$WORK/fake.c" "$WORK/libcxx.dylib"

before="$(otool -L "$APP/Contents/MacOS/Mold" | grep -c /nix/store || true)"
if [[ "$before" -lt 2 ]]; then
  echo "FAIL: the fixture does not reproduce the hazard (found $before /nix/store loads)" >&2
  exit 1
fi

"$HERE/fix-macos-native-linkage.sh" "$APP"

for binary in "$APP/Contents/MacOS/Mold" "$APP/Contents/Frameworks/Nested.dylib"; do
  if otool -L "$binary" | grep -q /nix/store || otool -l "$binary" | grep -q /nix/store; then
    echo "FAIL: $binary still references /nix/store" >&2
    otool -L "$binary" >&2
    exit 1
  fi
done
otool -L "$APP/Contents/MacOS/Mold" | grep -q '/usr/lib/libc++.1.dylib' \
  || { echo "FAIL: libc++ was not retargeted to /usr/lib" >&2; exit 1; }
otool -L "$APP/Contents/MacOS/Mold" | grep -q '/usr/lib/libiconv.2.dylib' \
  || { echo "FAIL: libiconv was not retargeted to /usr/lib" >&2; exit 1; }

# And it must FAIL rather than pass on something it cannot fix.
mkdir -p "$WORK/Unfixable.app/Contents/MacOS"
fake_dylib '/nix/store/33333333333333333333333333333333-openssl-3/lib/libssl.3.dylib' \
  "$WORK/libssl.dylib"
clang -o "$WORK/Unfixable.app/Contents/MacOS/Mold" "$WORK/main.c" "$WORK/libssl.dylib"
if "$HERE/fix-macos-native-linkage.sh" "$WORK/Unfixable.app" >/dev/null 2>&1; then
  echo "FAIL: a /nix/store load it cannot rewrite must stop the release" >&2
  exit 1
fi

echo "linkage fixup ok"
