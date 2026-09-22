#!/usr/bin/env bash
# GPU-free Linux archive with the same completion layout as CUDA releases.
set -euo pipefail
[[ $# == 2 ]] || { echo "usage: $0 <binary> <archive.tar.gz>" >&2; exit 64; }
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/cuda-release-archive.sh
source "$repo_root/scripts/lib/cuda-release-archive.sh"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/completions"
install -m755 "$1" "$scratch/mold"
for entry in "${MOLD_RELEASE_COMPLETIONS[@]}"; do
  mold_generate_release_completion "$scratch" "" "${entry%%:*}" "$scratch/${entry#*:}"
done
mapfile -t members < <(mold_release_archive_members)
tar -czf "$2" -C "$scratch" "${members[@]}"
