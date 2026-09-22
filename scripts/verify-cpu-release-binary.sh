#!/usr/bin/env bash
# Run inside a clean Linux runtime, with no driver/toolkit or loader overrides.
set -euo pipefail
[[ $# == 1 ]] || { echo "usage: $0 <archive.tar.gz>" >&2; exit 64; }
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
unset LD_LIBRARY_PATH
if ldconfig -p | grep -Ei 'lib(cuda|cudart|cublas|cudnn|nvrtc|nvidia)' ; then
  echo 'CPU smoke environment contains NVIDIA libraries' >&2
  exit 1
fi
tar -xzf "$1" -C "$scratch"
ldd "$scratch/mold" > "$scratch/ldd"
cat "$scratch/ldd"
if grep -Ei 'not found|lib(cuda|cudart|cublas|cudnn|nvrtc|nvidia)' "$scratch/ldd"; then
  echo 'CPU binary has unresolved or NVIDIA dependencies' >&2
  exit 1
fi
python3 "$repo_root/scripts/tests/cpu-release-smoke.py" "$scratch/mold"
# The archive must carry completions from this exact executable and argv[0].
# shellcheck source=scripts/lib/cuda-release-archive.sh
source "$repo_root/scripts/lib/cuda-release-archive.sh"
for entry in "${MOLD_RELEASE_COMPLETIONS[@]}"; do
  mold_generate_release_completion "$scratch" "" "${entry%%:*}" "$scratch/check"
  cmp "$scratch/check" "$scratch/${entry#*:}"
done
