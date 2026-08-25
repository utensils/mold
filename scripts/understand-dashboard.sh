#!/usr/bin/env bash

set -euo pipefail

script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
project_arg="$script_root"
if [ "$#" -gt 0 ] && [[ "$1" != -* ]]; then
    project_arg="$1"
    shift
fi

if ! project_dir="$(cd "$project_arg" 2>/dev/null && pwd -P)"; then
    echo "understand-dashboard: project directory not found: $project_arg" >&2
    exit 1
fi

# Prefer a graph committed in the current checkout. If an ephemeral worktree
# has no graph of its own, follow the shared git directory back to the durable
# graph in the main checkout.
if [ ! -f "$project_dir/.understand-anything/knowledge-graph.json" ] \
    && [ ! -f "$project_dir/.ua/knowledge-graph.json" ]; then
    common_dir="$(git -C "$project_dir" rev-parse --git-common-dir 2>/dev/null || true)"
    git_dir="$(git -C "$project_dir" rev-parse --git-dir 2>/dev/null || true)"
    if [ -n "$common_dir" ] && [ -n "$git_dir" ]; then
        common_abs="$(cd "$project_dir" && cd "$common_dir" 2>/dev/null && pwd -P || true)"
        git_abs="$(cd "$project_dir" && cd "$git_dir" 2>/dev/null && pwd -P || true)"
        if [ -n "$common_abs" ] && [ "$common_abs" != "$git_abs" ]; then
            project_dir="$(dirname "$common_abs")"
        fi
    fi
fi

if [ -d "$project_dir/.understand-anything" ]; then
    ua_dir="$project_dir/.understand-anything"
else
    ua_dir="$project_dir/.ua"
fi

if [ ! -f "$ua_dir/knowledge-graph.json" ]; then
    echo "understand-dashboard: no knowledge graph found at $ua_dir/knowledge-graph.json" >&2
    echo "Run the Understand Anything analysis first." >&2
    exit 1
fi

skill_real="$(realpath "${HOME}/.agents/skills/understand-dashboard" 2>/dev/null || true)"
self_relative=""
if [ -n "$skill_real" ]; then
    self_relative="$(cd "$skill_real/../.." 2>/dev/null && pwd -P || true)"
fi

plugin_root=""
for candidate in \
    "${CLAUDE_PLUGIN_ROOT:-}" \
    "${HOME}/.understand-anything-plugin" \
    "$self_relative" \
    "${HOME}/.codex/understand-anything/understand-anything-plugin" \
    "${HOME}/.opencode/understand-anything/understand-anything-plugin" \
    "${HOME}/.pi/understand-anything/understand-anything-plugin" \
    "${HOME}/understand-anything/understand-anything-plugin"; do
    if [ -n "$candidate" ] && [ -f "$candidate/package.json" ]; then
        plugin_root="$candidate"
        break
    fi
done

if [ -z "$plugin_root" ]; then
    echo "understand-dashboard: cannot find the installed Understand Anything plugin" >&2
    exit 1
fi

viewer="$plugin_root/packages/viewer/bin/viewer.mjs"
if [ -f "$viewer" ] && [ -d "$plugin_root/packages/viewer/dist" ]; then
    exec node "$viewer" "$project_dir" "$@"
fi

plugin_version="$(node -p "require('$plugin_root/package.json').version")"
viewer_url="https://github.com/Egonex-AI/Understand-Anything/releases/download/v${plugin_version}/understand-anything-viewer.tgz"
if curl --fail --silent --location --head "$viewer_url" >/dev/null; then
    exec npx --yes "$viewer_url" "$project_dir" "$@"
fi

echo "understand-dashboard: release viewer unavailable; building the installed viewer locally" >&2
(
    cd "$plugin_root"
    pnpm install --frozen-lockfile 2>/dev/null || pnpm install
    pnpm --filter @understand-anything/core build
    pnpm --filter understand-anything-viewer build
)
exec node "$viewer" "$project_dir" "$@"
