#!/usr/bin/env bash
# `make build` warns when the linked engine predates engine changes.
#
# A Debug build relinked a staticlib a week older than the branch and
# reported "This Mac · 0.30.0" on a 0.31.0 tree; nothing said so.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
check="$HERE/engine-freshness.sh"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

cd "$work"
git init -q . && git config user.email t@t && git config user.name t
mkdir -p crates/core && echo one > crates/core/lib.rs && echo readme > README.md
git add -A && git commit -qm one
lib="$work/libmold_macos_ffi.a"
touch "$lib"
printf 'MOLD_ENGINE_LIB = %s\nSWIFT_ACTIVE_COMPILATION_CONDITIONS = $(inherited) MOLD_EMBEDDED_ENGINE\n' \
  "$lib" > Engine.xcconfig

fail() { echo "FAIL: $*" >&2; exit 1; }
warns() { "$check" Engine.xcconfig 2>&1 | grep -q '^warning:'; }

warns || fail "an engine with no stamp is unaccounted for and must warn"

git rev-parse HEAD > "$lib.commit"
! warns || fail "an engine built at HEAD must not warn"

echo docs >> README.md && git commit -qam docs
! warns || fail "a change outside the engine's sources must not warn"

echo two >> crates/core/lib.rs
warns || fail "an uncommitted engine change must warn"
git commit -qam two
warns || fail "a committed engine change since the stamp must warn"

printf '// remote-only\n' > Engine.xcconfig
! warns || fail "a remote-only build has no engine to be stale"

echo "engine-freshness: ok"
