#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

fail() {
  echo "frontend-architecture: $*" >&2
  exit 1
}

test -f package.json || fail "missing root frontend workspace manifest"
test -f bun.lock || fail "missing unified root Bun lockfile"
test -f bun.nix || fail "missing unified root bun2nix dependency set"
test -f studio/package.json || fail "missing @mold/studio workspace package"

for legacy_lock in web/bun.lock web/bun.nix desktop/bun.lock desktop/bun.nix; do
  test ! -e "$legacy_lock" || fail "$legacy_lock must be replaced by the root lock"
done

if rg -n '"(vue|vue-router|pinia|vite|vitest|typescript|tailwindcss|prettier|happy-dom|smol-toml|@microsoft/fetch-event-source)"[[:space:]]*:' web/package.json desktop/package.json; then
  fail "shared browser dependencies must be owned and pinned at the workspace root"
fi

bun -e '
  const manifest = await Bun.file("package.json").json();
  for (const group of [manifest.dependencies, manifest.devDependencies]) {
    for (const [name, version] of Object.entries(group ?? {})) {
      if (typeof version !== "string" || !/^\d+\.\d+\.\d+/.test(version)) {
        throw new Error(`${name} must use an exact root version, got ${version}`);
      }
    }
  }
'

if rg -n "@tauri-apps|from [\"']\.\./(web|desktop)|from [\"'].*(web|desktop)/" studio; then
  fail "studio must remain browser-safe and application-shell independent"
fi

if rg -n "path: [\"']/(generate|catalog)[\"']|redirect:.*(generate|catalog)" web/src/router.ts; then
  fail "web legacy routes must not remain registered"
fi

for dead_file in web/src/components/Metadata.vue; do
  test ! -e "$dead_file" || fail "$dead_file is unreachable and must be removed"
done

for duplicate in \
  web/src/lib/promptCycler.ts \
  desktop/src/lib/promptCycler.ts \
  web/src/lib/sourceFit.ts \
  desktop/src/lib/sourceFit.ts; do
  test ! -e "$duplicate" || fail "$duplicate duplicates studio domain logic"
done

echo "frontend-architecture: ok"
