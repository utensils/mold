#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
web_dir="${MOLD_WEB_ROOT:-$repo_root/web}"
dist_dir="$web_dir/dist"
stamp_file="$dist_dir/.mold-build-stamp"
workspace_install_stamp="$repo_root/node_modules/.mold-install-stamp"

# A `web/node_modules` containing its own vite is a leftover from the
# pre-workspace layout (each package used to install independently).
# It shadows the repo-root workspace install and makes `vue-tsc` see two
# incompatible copies of vite/rollup ("Plugin<any> is not assignable to
# PluginOption"). Bun never hoists vite into web/ under the unified
# workspace, so its presence is always stale — remove it.
if [ -d "$web_dir/node_modules/vite" ]; then
    echo "ensure-web-dist: removing stale $web_dir/node_modules (pre-workspace install)" >&2
    rm -rf "$web_dir/node_modules"
fi

needs_build=0

if [ ! -f "$dist_dir/index.html" ] || [ ! -f "$stamp_file" ]; then
    needs_build=1
else
    while IFS= read -r path; do
        if [ "$path" -nt "$stamp_file" ]; then
            needs_build=1
            break
        fi
    done < <(
        find \
            "$web_dir/src" \
            "$web_dir/public" \
            "$repo_root/studio" \
            "$repo_root/ui" \
            \( -name node_modules -o -name dist \) -prune -o \
            -type f -print
        printf '%s\n' \
            "$web_dir/package.json" \
            "$repo_root/package.json" \
            "$repo_root/bun.lock" \
            "$repo_root/bun.nix" \
            "$repo_root/studio/package.json" \
            "$repo_root/ui/package.json" \
            "$web_dir/index.html" \
            "$web_dir/vite.config.ts" \
            "$web_dir/tsconfig.json" \
            "$web_dir/tsconfig.app.json" \
            "$web_dir/tsconfig.node.json" \
            "$web_dir/vitest.config.ts"
    )
fi

if [ "$needs_build" -eq 0 ]; then
    exit 0
fi

if [ ! -d "$repo_root/node_modules" ] \
    || [ ! -f "$workspace_install_stamp" ] \
    || [ "$repo_root/package.json" -nt "$workspace_install_stamp" ] \
    || [ "$repo_root/bun.lock" -nt "$workspace_install_stamp" ]; then
    (
        cd "$repo_root"
        bun install --frozen-lockfile
        touch "$workspace_install_stamp"
    )
fi

(
    cd "$web_dir"
    bun run build
    touch "$stamp_file"
)
