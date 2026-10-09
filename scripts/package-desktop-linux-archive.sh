#!/usr/bin/env bash
# GPU-free Linux desktop archive: a `usr/` tree (binary, .desktop entry,
# AppStream metainfo, hicolor icons) plus LICENSE, laid out by
# scripts/install-linux-desktop-files.sh so `mold-ai-desktop-bin` installs the
# same files as the `mold-ai-desktop` source recipe.
set -euo pipefail
[[ $# == 2 ]] || { echo "usage: $0 <binary> <archive.tar.gz>" >&2; exit 64; }
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$(realpath -m "$2")"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
"$repo_root/scripts/install-linux-desktop-files.sh" "$1" "$scratch/usr"
install -m644 "$repo_root/LICENSE" "$scratch/LICENSE"
tar -czf "$out" -C "$scratch" --owner=0 --group=0 --numeric-owner usr LICENSE
