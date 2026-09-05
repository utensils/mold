#!/usr/bin/env bash
# Contract for `scripts/build-clean.sh` (the devshell's `build-clean`).
#
# The one behaviour worth a test is the symlinked target directory. On the
# development machines every cargo target lives on external storage behind a
# `target -> /Volumes/…` symlink, and `cargo clean` deletes the SYMLINK rather
# than following it — the next build then lands a fresh `target/` on the
# internal disk, which is exactly the disk pressure the symlink exists to
# avoid. The script must empty what the link points at and leave the link.
#
# Everything else is a list: the frontend bundles and caches, the tsbuildinfo
# stamps, the Tauri schema output, the nix `result` links. The test builds a
# fake repo with one of each, cleans it, and checks what survived.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
script="$repo_root/scripts/build-clean.sh"

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

[ -x "$script" ] || fail "$script is not executable"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

fake="$work/repo"
external="$work/external-target"
mkdir -p "$fake" "$external/dev-fast/deps"
touch "$external/.rustc_info.json" "$external/dev-fast/deps/libmold.rlib"

# A cargo root whose target is a symlink onto "external storage".
ln -s "$external" "$fake/target"
# Cargo roots with a plain target directory.
mkdir -p "$fake/desktop/src-tauri/target/debug" "$fake/apps/mobile/src-tauri/target/debug"
touch "$fake/desktop/src-tauri/target/debug/app" "$fake/apps/mobile/src-tauri/target/debug/app"
# Frontend bundles, caches and stamps.
mkdir -p "$fake/web/dist" "$fake/web/.vite" "$fake/desktop/dist" "$fake/desktop/dist-mobile" \
  "$fake/website/.vitepress/dist" "$fake/website/.vitepress/cache" \
  "$fake/node_modules/.vite/deps" "$fake/desktop/node_modules/.vite" "$fake/web/node_modules/.vite" \
  "$fake/apps/mobile/src-tauri/gen/schemas" "$fake/desktop/src-tauri/gen/schemas"
touch "$fake/web/dist/index.html" "$fake/web/tsconfig.app.tsbuildinfo" \
  "$fake/desktop/tsconfig.tsbuildinfo" "$fake/node_modules/.vite/deps/vue.js" \
  "$fake/apps/mobile/src-tauri/gen/schemas/acl.json"
ln -s /nix/store/nothing "$fake/result"
ln -s /nix/store/nothing "$fake/result-bin"
# Things that must survive: sources, dependencies, the committed gen tree.
mkdir -p "$fake/web/src" "$fake/node_modules/vue" "$fake/apps/mobile/src-tauri/gen/apple"
touch "$fake/web/src/main.ts" "$fake/node_modules/vue/package.json" \
  "$fake/apps/mobile/src-tauri/gen/apple/project.yml" "$fake/Cargo.toml"

# ── dry run removes nothing and names what it would remove ──────────────────
dry="$("$script" --root "$fake" --dry-run)"
[ -f "$fake/web/dist/index.html" ] || fail "dry run removed web/dist"
[ -f "$external/dev-fast/deps/libmold.rlib" ] || fail "dry run emptied the symlinked target"
echo "$dry" | grep -q "web/dist" || fail "dry run did not name web/dist"
echo "$dry" | grep -q "target" || fail "dry run did not name a target dir"

# ── the real thing ──────────────────────────────────────────────────────────
"$script" --root "$fake" >/dev/null

[ -L "$fake/target" ] || fail "the target symlink was deleted (cargo clean behaviour)"
[ -d "$external" ] || fail "the external target directory was deleted"
[ -z "$(ls -A "$external")" ] || fail "the external target directory was not emptied"
[ ! -e "$fake/desktop/src-tauri/target" ] || fail "desktop target survived"
[ ! -e "$fake/apps/mobile/src-tauri/target" ] || fail "mobile target survived"

for gone in web/dist web/.vite web/tsconfig.app.tsbuildinfo desktop/dist desktop/dist-mobile \
  desktop/tsconfig.tsbuildinfo website/.vitepress/dist website/.vitepress/cache \
  node_modules/.vite desktop/node_modules/.vite web/node_modules/.vite \
  apps/mobile/src-tauri/gen/schemas desktop/src-tauri/gen/schemas result result-bin; do
  [ ! -e "$fake/$gone" ] && [ ! -L "$fake/$gone" ] || fail "$gone survived"
done

for kept in web/src/main.ts node_modules/vue/package.json \
  apps/mobile/src-tauri/gen/apple/project.yml Cargo.toml; do
  [ -e "$fake/$kept" ] || fail "$kept was removed"
done

# ── --node-modules is opt-in and takes the dependency installs too ─────────
mkdir -p "$fake/web/node_modules/x" "$fake/website/node_modules/y"
"$script" --root "$fake" --node-modules >/dev/null
[ ! -e "$fake/node_modules" ] || fail "--node-modules kept the root install"
[ ! -e "$fake/web/node_modules" ] || fail "--node-modules kept web/node_modules"
[ ! -e "$fake/website/node_modules" ] || fail "--node-modules kept website/node_modules"
[ -f "$fake/web/src/main.ts" ] || fail "--node-modules touched sources"

# ── a CARGO_TARGET_DIR is emptied, never removed ───────────────────────────
shared="$work/shared-target"
mkdir -p "$shared/debug"
touch "$shared/debug/thing"
CARGO_TARGET_DIR="$shared" "$script" --root "$fake" >/dev/null
[ -d "$shared" ] || fail "CARGO_TARGET_DIR directory was removed"
[ -z "$(ls -A "$shared")" ] || fail "CARGO_TARGET_DIR was not emptied"

# ── a bad root is a refusal, not a rm -rf somewhere else ───────────────────
if "$script" --root "$work/does-not-exist" >/dev/null 2>&1; then
  fail "a missing --root was accepted"
fi
if "$script" --root "$work" >/dev/null 2>&1; then
  fail "a root with no Cargo.toml was accepted"
fi

echo "build-clean contract: ok"
