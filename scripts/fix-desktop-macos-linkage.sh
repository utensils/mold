#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  exit 0
fi

if [[ $# -gt 0 ]]; then
  binary="$1"
elif [[ "${TAURI_ENV_DEBUG:-false}" == "true" ]]; then
  binary="src-tauri/target/debug/mold-desktop"
else
  binary="src-tauri/target/release/mold-desktop"
fi
if [[ ! -f "$binary" ]]; then
  echo "missing desktop binary: $binary" >&2
  exit 1
fi

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

if otool -L "$binary" | grep -q /nix/store; then
  echo "desktop binary still references /nix/store after linkage fixup:" >&2
  otool -L "$binary" >&2
  exit 1
fi

echo "Desktop Mach-O linkage is portable."
