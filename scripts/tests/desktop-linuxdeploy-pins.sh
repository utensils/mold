#!/usr/bin/env bash
set -euo pipefail

# `prepare-desktop-linuxdeploy.sh` checksums everything it downloads, and the
# Linux AppImage job that runs it is push-only. A checksum on a URL that names
# a BRANCH therefore turns an upstream commit into a red `main` that no PR
# could have seen: tauri-apps/linuxdeploy-plugin-gtk moved `master` on
# 2026-08-26 and every Desktop push failed at the sha256 check. A raw GitHub
# download must name the 40-hex commit its checksum was taken from.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
script="$repo_root/scripts/prepare-desktop-linuxdeploy.sh"

raw_urls="$(grep -oE 'https://raw\.githubusercontent\.com/[^"[:space:]]+' "$script" || true)"
if [[ -z "$raw_urls" ]]; then
  echo "FAIL: found no raw.githubusercontent.com download in $script; update this contract" >&2
  exit 1
fi

# The other spellings of "a file from a branch" carry the ref somewhere else in
# the path; none is in use, so refuse them rather than parse them.
if grep -nE 'https://(github\.com/[^/"]+/[^/"]+/(raw|archive)/|codeload\.github\.com/)' "$script" >&2; then
  echo "FAIL: fetch repository files through raw.githubusercontent.com/<owner>/<repo>/<commit>/ so the pin is checkable" >&2
  exit 1
fi

failed=0
while IFS= read -r url; do
  # https://raw.githubusercontent.com/<owner>/<repo>/<ref>/<path>
  ref="$(printf '%s\n' "$url" | cut -d/ -f6)"
  if [[ ! "$ref" =~ ^[0-9a-f]{40}$ ]]; then
    echo "FAIL: $url names the mutable ref '$ref'; pin the 40-hex commit the sha256 was taken from" >&2
    failed=1
  fi
done <<<"$raw_urls"

# Every plugin download keeps its checksum: a pinned commit says WHICH bytes,
# the sha256 proves they are the bytes that arrived.
plugin_calls="$(grep -c '^prepare_plugin \\$' "$script" || true)"
plugin_hashes="$(grep -cE '^  "[0-9a-f]{64}" \\$' "$script" || true)"
if [[ "$plugin_calls" -eq 0 || "$plugin_calls" -ne "$plugin_hashes" ]]; then
  echo "FAIL: $plugin_calls prepare_plugin calls but $plugin_hashes sha256 pins" >&2
  failed=1
fi

[[ "$failed" -eq 0 ]] || exit 1
echo "PASS: desktop-linuxdeploy-pins"
