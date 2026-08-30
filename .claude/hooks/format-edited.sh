#!/usr/bin/env bash
# PostToolUse Write|Edit: format the edited file with the project's formatter.
set -u
f=$(jq -r '.tool_response.filePath // .tool_input.file_path // empty')
[ -n "$f" ] && [ -f "$f" ] || exit 0
root=$(cd "$(dirname "$0")/../.." && pwd)
case "$f" in
  "$root"/apps/mobile/src-tauri/*) exit 0 ;;              # Rust 2024, own gate; excluded from treefmt
  *.rs) rustfmt --edition 2021 "$f" ;;
  "$root"/studio/*.ts|"$root"/studio/*.vue) "$root/node_modules/.bin/prettier" --write --log-level warn "$f" ;;
esac
