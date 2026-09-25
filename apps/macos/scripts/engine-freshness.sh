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
# The stamp is the commit AND a digest of the uncommitted engine diff the
# build included, so an engine built from local edits is fresh until those
# edits change -- and stashing them after the build is noticed.
#
# Usage: engine-freshness.sh <Engine.xcconfig>
#        engine-freshness.sh --stamp <file>     (written by `make engine`)
set -euo pipefail

paths=(crates apps/macos/rust Cargo.lock)

# The diff of the working tree against <commit>, for the engine's paths.
digest() {
  git -C "$root" diff --no-ext-diff --binary "$1" -- "${paths[@]}" | shasum -a 256 | cut -d' ' -f1
}

if [ "${1:-}" = "--stamp" ]; then
  root=$(git rev-parse --show-toplevel)
  head=$(git -C "$root" rev-parse HEAD)
  printf '%s\n%s\n' "$head" "$(digest "$head")" > "${2:?usage: --stamp <file>}"
  exit 0
fi

xcconfig="${1:?usage: engine-freshness.sh <Engine.xcconfig>}"
[ -f "$xcconfig" ] || exit 0
grep -q 'MOLD_EMBEDDED_ENGINE' "$xcconfig" || exit 0   # remote-only build
lib=$(sed -n 's/^MOLD_ENGINE_LIB = //p' "$xcconfig")
[ -n "$lib" ] || exit 0

root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
stamp="$lib.commit"
if [ ! -f "$stamp" ]; then
  echo "warning: the embedded engine ($lib) records no commit; run \`make engine\` to rebuild it." >&2
  exit 0
fi
built=$(sed -n 1p "$stamp" | tr -d '[:space:]')
recorded=$(sed -n 2p "$stamp" | tr -d '[:space:]')
if ! git -C "$root" cat-file -e "$built^{commit}" 2>/dev/null; then
  echo "warning: the embedded engine was built from ${built:0:7}, which this checkout does not have; run \`make engine\`." >&2
  exit 0
fi
# An older stamp carries no digest: it was a clean build of that commit.
[ -n "$recorded" ] || recorded=$(printf '' | shasum -a 256 | cut -d' ' -f1)
if [ "$(digest "$built")" != "$recorded" ]; then
  head=$(git -C "$root" rev-parse --short HEAD)
  echo "warning: the embedded engine was built at ${built:0:7}, but engine sources have changed since (HEAD is $head); run \`make engine\` to rebuild it." >&2
fi
