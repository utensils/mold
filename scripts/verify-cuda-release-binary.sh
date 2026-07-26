#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <binary> <compute-capability> [archive.tar.gz]" >&2
  exit 64
fi

binary="$1"
compute_cap="$2"
archive="${3:-}"

[[ "$compute_cap" =~ ^[0-9]+$ ]] \
  || { echo "compute capability must be numeric: $compute_cap" >&2; exit 64; }
[[ -x "$binary" ]] \
  || { echo "CUDA release binary is missing or not executable: $binary" >&2; exit 1; }

scratch_dir="$(mktemp -d)"
trap 'rm -rf "$scratch_dir"' EXIT

elf_header="$(readelf -h "$binary")"
grep -Eq 'Machine:[[:space:]]+Advanced Micro Devices X86-64' <<<"$elf_header" \
  || { echo "CUDA release binary is not Linux x86_64: $binary" >&2; exit 1; }

# Candle/cudarc embeds generated PTX as ordinary strings in the final Mold
# executable. nvcc-created cubins can exist in intermediate archives but are
# dead-stripped when no final symbol references them, so cuobjdump correctly
# reports no device-code sections for the shipped artifact.
#
# Published packages currently have a strict single-target policy. A future
# fat-PTX package must add an explicit allowed target set here and corresponding
# release routing/tests; silently accepting any additional target would make
# the filename's compatibility claim untrue.
allowed_ptx_targets="sm_${compute_cap}"
embedded_strings="$scratch_dir/embedded-strings"
strings -a "$binary" >"$embedded_strings"
ptx_directives="$scratch_dir/ptx-directives"
# The embedded SPA can contain unrelated JavaScript such as
# `.target instanceof`. PTX architecture operands are the `sm_*` directives;
# enumerate all of those and reject any malformed operand below.
grep -Eo '\.target[[:space:]]+sm_[^[:space:][:cntrl:]]+' \
  "$embedded_strings" >"$ptx_directives" || true
if [[ ! -s "$ptx_directives" ]]; then
  echo "CUDA release binary does not embed any PTX target directives" >&2
  exit 1
fi
observed_ptx_targets="$scratch_dir/observed-ptx-targets"
while IFS= read -r directive; do
  target="$(awk '{ print $2 }' <<<"$directive")"
  if [[ ! "$target" =~ ^sm_[0-9]+$ ]]; then
    echo "CUDA release binary contains malformed PTX target directive: $directive" >&2
    exit 1
  fi
  printf '%s\n' "$target"
done <"$ptx_directives" | sort -u >"$observed_ptx_targets"
observed_set="$(paste -sd, "$observed_ptx_targets")"
if [[ "$observed_set" != "$allowed_ptx_targets" ]]; then
  echo "CUDA release binary PTX target set is {$observed_set}; allowed set is {$allowed_ptx_targets}" >&2
  exit 1
fi

# nvml-wrapper loads the driver API at runtime and retains its symbol table in
# the stripped executable. Check a stable required entry point rather than a
# library filename: the wrapper does not retain that filename in Mold's final
# link, while this symbol disappears when the NVML feature is absent.
grep -aFq 'nvmlDeviceGetCount_v2' "$binary" \
  || { echo "CUDA release binary does not include NVML loader support" >&2; exit 1; }

ldd_output="$(ldd "$binary")"
unresolved="$(
  printf '%s\n' "$ldd_output" \
    | awk '/not found/ { print $1 }' \
    | grep -Ev '^(libcuda|libnvidia-ml)\.so' \
    || true
)"
if [[ -n "$unresolved" ]]; then
  echo "CUDA release binary has unexpected unresolved libraries:" >&2
  echo "$unresolved" >&2
  exit 1
fi

loader_path="${LD_LIBRARY_PATH:-}"
if grep -Eq '^([[:space:]]*)libcuda\.so[^[:space:]]* => not found' <<<"$ldd_output"; then
  cuda_stub=""
  for candidate in \
    /usr/local/cuda/lib64/stubs/libcuda.so \
    /opt/cuda/lib64/stubs/libcuda.so; do
    if [[ -f "$candidate" ]]; then
      cuda_stub="$candidate"
      break
    fi
  done
  [[ -n "$cuda_stub" ]] \
    || { echo "libcuda is unavailable and no CUDA toolkit loader stub was found" >&2; exit 1; }
  ln -s "$cuda_stub" "$scratch_dir/libcuda.so.1"
  loader_path="$scratch_dir${loader_path:+:$loader_path}"
fi

LD_LIBRARY_PATH="$loader_path" "$binary" version >/dev/null \
  || { echo "CUDA release binary failed its GPU-less loader smoke" >&2; exit 1; }

if [[ -n "$archive" ]]; then
  [[ -f "$archive" ]] \
    || { echo "CUDA release archive is missing: $archive" >&2; exit 1; }
  [[ "$(tar -tzf "$archive")" == "mold" ]] \
    || { echo "CUDA release archive must contain exactly mold" >&2; exit 1; }

  extract_dir="$scratch_dir/archive"
  mkdir -p "$extract_dir"
  tar -xzf "$archive" -C "$extract_dir"
  cmp "$binary" "$extract_dir/mold" \
    || { echo "CUDA release archive does not contain the verified binary" >&2; exit 1; }
fi

echo "verified CUDA sm_${compute_cap} release binary${archive:+ and archive}"
