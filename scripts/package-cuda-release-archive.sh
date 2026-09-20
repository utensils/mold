#!/usr/bin/env bash
# Stage a Linux CUDA release archive: the verified binary as `mold` beside the
# shell completion scripts generated from THAT binary, laid out as
# scripts/lib/cuda-release-archive.sh describes. The completions ship inside
# the archive so no packager ever has to execute the CUDA-linked payload
# (#1742); the archive is then checked against the same layout by
# scripts/verify-cuda-release-binary.sh.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <binary> <archive.tar.gz>" >&2
  exit 64
fi

binary="$1"
archive="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/cuda-release-archive.sh
source "$repo_root/scripts/lib/cuda-release-archive.sh"

[[ -x "$binary" ]] \
  || { echo "CUDA release binary is missing or not executable: $binary" >&2; exit 1; }

scratch_dir="$(mktemp -d)"
trap 'rm -rf "$scratch_dir"' EXIT
stage_dir="$scratch_dir/stage"
mkdir -p "$stage_dir/completions"
install -m 0755 "$binary" "$stage_dir/mold"

loader_path="$(mold_cuda_gpuless_loader_path "$stage_dir/mold" "$scratch_dir")"
for entry in "${MOLD_RELEASE_COMPLETIONS[@]}"; do
  mold_generate_release_completion \
    "$stage_dir" "$loader_path" "${entry%%:*}" "$stage_dir/${entry#*:}"
done

# Members are named explicitly, binary first, so the listing is deterministic
# and carries no directory entry.
mapfile -t members < <(mold_release_archive_members)
tar -czf "$archive" -C "$stage_dir" "${members[@]}"
echo "packaged $archive: ${members[*]}"
