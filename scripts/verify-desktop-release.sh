#!/usr/bin/env bash
set -euo pipefail

app_path="${1:-desktop/src-tauri/target/release/bundle/macos/Mold.app}"
dmg_path="${2:-}"

if [[ ! -d "$app_path" ]]; then
  echo "Mold.app not found at $app_path" >&2
  exit 1
fi

echo "Verifying Developer ID signature: $app_path"
codesign --verify --deep --strict --verbose=2 "$app_path"
codesign -dvvv --entitlements :- "$app_path"

echo "Assessing app with Gatekeeper"
spctl --assess --type execute --verbose=2 "$app_path"

echo "Validating app notarization staple"
xcrun stapler validate "$app_path"

while IFS= read -r -d '' binary; do
  if file "$binary" | grep -q 'Mach-O' && otool -L "$binary" | tail -n +2 | grep -q '/nix/store'; then
    echo "Distribution binary references /nix/store: $binary" >&2
    otool -L "$binary" >&2
    exit 1
  fi
done < <(find "$app_path/Contents" -type f -print0)

if [[ -n "$dmg_path" ]]; then
  if [[ ! -f "$dmg_path" ]]; then
    echo "DMG not found at $dmg_path" >&2
    exit 1
  fi
  echo "Validating DMG notarization staple: $dmg_path"
  xcrun stapler validate "$dmg_path"
  spctl --assess --type open --context context:primary-signature --verbose=2 "$dmg_path"
fi

echo "Desktop distribution verification passed"
