#!/usr/bin/env bash
# Make the app's Mach-O linkage portable, or fail.
#
# The engine is linked inside `nix develop`, where `-lc++ -liconv` resolve
# through LIBRARY_PATH to the Nix store. The resulting binary carries
# `LC_LOAD_DYLIB /nix/store/…/libc++.1.0.dylib`, and on any other Mac that path
# does not exist and the app dies in dyld before `main` -- a notarized DMG that
# cannot launch (review 05-H3). `scripts/fix-desktop-macos-linkage.sh` has
# solved exactly this for the Tauri app since day one; this is that script,
# applied to every Mach-O in the bundle rather than to one binary.
#
# It FAILS CLOSED: a surviving /nix/store reference stops the release rather
# than shipping.
set -euo pipefail

APP="${1:?usage: fix-macos-native-linkage.sh <path to .app>}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  exit 0
fi
if [[ ! -d "$APP" ]]; then
  echo "missing bundle: $APP" >&2
  exit 1
fi

retarget() {
  local binary="$1" dependency
  while IFS= read -r dependency; do
    case "$dependency" in
      /nix/store/*-libcxx-*/lib/libc++.1.0.dylib)
        install_name_tool -change "$dependency" /usr/lib/libc++.1.dylib "$binary"
        ;;
      /nix/store/*-libiconv-*/lib/libiconv.2.dylib)
        install_name_tool -change "$dependency" /usr/lib/libiconv.2.dylib "$binary"
        ;;
    esac
  done < <(otool -L "$binary" | tail -n +2 | awk '{ print $1 }')

  # An LC_RPATH into the store is not fatal by itself -- nothing resolves
  # through it once the loads above are rewritten -- but it is a build path
  # shipped to users, and the check below would refuse it anyway.
  while IFS= read -r dependency; do
    install_name_tool -delete_rpath "$dependency" "$binary" 2>/dev/null || true
  done < <(otool -l "$binary" | awk '/LC_RPATH/ { rpath = 1 } rpath && $1 == "path" { print $2; rpath = 0 }' \
    | grep '^/nix/store' || true)
}

mach_o=()
while IFS= read -r -d '' candidate; do
  if file -b "$candidate" | grep -q 'Mach-O'; then
    mach_o+=("$candidate")
  fi
done < <(find "$APP" -type f -print0)

if [[ ${#mach_o[@]} -eq 0 ]]; then
  echo "no Mach-O files under $APP -- nothing was built?" >&2
  exit 1
fi

for binary in "${mach_o[@]}"; do
  retarget "$binary"
done

failed=0
for binary in "${mach_o[@]}"; do
  if otool -L "$binary" | grep -q /nix/store || otool -l "$binary" | grep -q /nix/store; then
    echo "still references /nix/store after linkage fixup: $binary" >&2
    otool -L "$binary" >&2
    failed=1
  fi
done
if [[ $failed -ne 0 ]]; then
  exit 1
fi

echo "  linkage portable: ${#mach_o[@]} Mach-O file(s)"
