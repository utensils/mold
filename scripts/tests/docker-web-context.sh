#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dockerfile="$repo_root/Dockerfile"

ui_copy_line="$(grep -n -m1 '^COPY ui /ui$' "$dockerfile" | cut -d: -f1 || true)"
web_build_line="$(grep -n -m1 '^RUN bun run build$' "$dockerfile" | cut -d: -f1 || true)"

if [[ -z "$ui_copy_line" ]]; then
  echo "FAIL: Docker web-builder must copy repo-root ui/ to /ui for @ui imports" >&2
  exit 1
fi

if [[ -z "$web_build_line" || "$ui_copy_line" -ge "$web_build_line" ]]; then
  echo "FAIL: Docker web-builder must copy ui/ before running the web SPA build" >&2
  exit 1
fi

grep -Fq '"@ui/*": ["../ui/*"]' "$repo_root/web/tsconfig.app.json" || {
  echo "FAIL: web TypeScript @ui alias no longer resolves from /web to /ui" >&2
  exit 1
}

grep -Fq 'new URL("../ui", import.meta.url)' "$repo_root/web/vite.config.ts" || {
  echo "FAIL: web Vite @ui alias no longer resolves from /web to /ui" >&2
  exit 1
}

echo "PASS: Docker web-builder includes the shared ui/ source before build"
