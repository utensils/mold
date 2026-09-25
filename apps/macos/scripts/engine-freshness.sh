#!/usr/bin/env bash
# Warns when the engine staticlib Engine.xcconfig links predates engine
# changes on this checkout.
#
# `make build` relinks whatever `libmold_macos_ffi.a` is on disk; only
# `make engine` rebuilds it. A Debug build therefore silently carried an
# engine a week behind the branch -- "This Mac · 0.30.0" on a tree at
# 0.31.0 -- and nothing said so. `make engine` stamps the commit it built
# from beside the library; this compares that stamp against HEAD and the
# working tree for the paths the engine is built from. A WARNING, never a
# failure: a UI-only iteration must stay fast.
#
# Usage: engine-freshness.sh <Engine.xcconfig>
set -euo pipefail

xcconfig="${1:?usage: engine-freshness.sh <Engine.xcconfig>}"
[ -f "$xcconfig" ] || exit 0
grep -q 'MOLD_EMBEDDED_ENGINE' "$xcconfig" || exit 0   # remote-only build
lib=$(sed -n 's/^MOLD_ENGINE_LIB = //p' "$xcconfig")
[ -n "$lib" ] || exit 0

paths=(crates apps/macos/rust Cargo.lock)
root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
stamp="$lib.commit"
if [ ! -f "$stamp" ]; then
  echo "warning: the embedded engine ($lib) records no commit; run \`make engine\` to rebuild it." >&2
  exit 0
fi
built=$(tr -d '[:space:]' <"$stamp")
if ! git -C "$root" cat-file -e "$built^{commit}" 2>/dev/null; then
  echo "warning: the embedded engine was built from ${built:0:7}, which this checkout does not have; run \`make engine\`." >&2
  exit 0
fi
if ! git -C "$root" diff --quiet "$built" -- "${paths[@]}"; then
  head=$(git -C "$root" rev-parse --short HEAD)
  echo "warning: the embedded engine was built at ${built:0:7}, but engine sources have changed since (HEAD is $head); run \`make engine\` to rebuild it." >&2
fi
