#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
web_dir="${MOLD_WEB_ROOT:-$repo_root/web}"
dist_dir="$web_dir/dist"
stamp_file="$dist_dir/.mold-build-stamp"
install_stamp="$web_dir/node_modules/.mold-install-stamp"

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
            -type f \
            -print
        printf '%s\n' \
            "$web_dir/package.json" \
            "$web_dir/bun.lock" \
            "$web_dir/bun.nix" \
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

if [ ! -d "$web_dir/node_modules" ] \
    || [ ! -f "$install_stamp" ] \
    || [ "$web_dir/package.json" -nt "$install_stamp" ] \
    || [ "$web_dir/bun.lock" -nt "$install_stamp" ]; then
    (
        cd "$web_dir"
        bun install --frozen-lockfile
        touch "$install_stamp"
    )
fi

(
    cd "$web_dir"
    bun run build
    touch "$stamp_file"
)
