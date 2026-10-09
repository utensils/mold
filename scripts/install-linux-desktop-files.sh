#!/usr/bin/env bash
# Install the Linux desktop app layout under a prefix: the `mold-desktop`
# binary, its .desktop entry, AppStream metainfo, and hicolor icons.
#
# Usage: scripts/install-linux-desktop-files.sh <binary> <prefix>
#
# The one layout authority for every Linux desktop distribution: the AUR
# source recipe (`mold-ai-desktop`, prefix "$pkgdir/usr") and the release
# archive packager (scripts/package-desktop-linux-archive.sh) both call it,
# so the binary package and the source package cannot drift apart. It never
# executes the binary — the AUR recipe runs it under fakeroot (#1742).
set -euo pipefail
[[ $# == 2 ]] || { echo "usage: $0 <binary> <prefix>" >&2; exit 64; }
binary="$1"
prefix="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
app_id="com.utensils.mold"
icons="$repo_root/desktop/src-tauri/icons"

[[ -f "$binary" ]] || { echo "error: $binary does not exist" >&2; exit 1; }

install -Dm755 "$binary" "$prefix/bin/mold-desktop"
install -Dm644 "$repo_root/packaging/linux/${app_id}.desktop" \
  "$prefix/share/applications/${app_id}.desktop"
install -Dm644 "$repo_root/packaging/linux/${app_id}.metainfo.xml" \
  "$prefix/share/metainfo/${app_id}.metainfo.xml"
# Tauri's icon set, keyed by the pixel size each file actually is.
for entry in 32x32.png:32 64x64.png:64 128x128.png:128 128x128@2x.png:256 icon.png:512; do
  size="${entry#*:}"
  install -Dm644 "$icons/${entry%%:*}" \
    "$prefix/share/icons/hicolor/${size}x${size}/apps/${app_id}.png"
done
