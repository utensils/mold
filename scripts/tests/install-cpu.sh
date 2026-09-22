#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/bin" "$scratch/payload"
# A closed PATH proves the missing-nvidia-smi case even on a GPU host.
for cmd in awk sed grep mkdir mktemp rm tar install sha256sum tr; do
  ln -s "$(command -v "$cmd")" "$scratch/bin/$cmd"
done
cat > "$scratch/bin/uname" <<'STUB'
#!/bin/sh
case "$1" in -s) echo "${TEST_OS:-Linux}" ;; -m) echo "${TEST_ARCH:-x86_64}" ;; esac
STUB
cat > "$scratch/payload/mold" <<'STUB'
#!/bin/sh
echo GPU-free-fixture
STUB
chmod +x "$scratch/payload/mold"
tar -czf "$scratch/archive.tar.gz" -C "$scratch/payload" mold
sha="$(sha256sum "$scratch/archive.tar.gz" | awk '{print $1}')"
printf '%s  mold-x86_64-unknown-linux-gnu-cpu.tar.gz\n' "$sha" > "$scratch/SHA256SUMS"
# cp is needed by the download stub; gzip is needed by tar.
for cmd in cp gzip; do ln -s "$(command -v "$cmd")" "$scratch/bin/$cmd"; done
cat > "$scratch/bin/curl" <<'STUB'
#!/bin/sh
out=""
url=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --output) out="$2"; shift 2 ;;
    --write-out|--retry|--retry-delay) shift 2 ;;
    -fsSILo) shift 2 ;;
    -w) shift 2 ;;
    -*) shift ;;
    *) url="$1"; shift ;;
  esac
done
printf '%s\n' "$url" >> "$TEST_ROOT/urls"
case "$url" in
  */releases/latest) printf 'https://github.com/utensils/mold/releases/tag/v-test' ;;
  */mold-x86_64-unknown-linux-gnu-cpu.tar.gz)
    if [ "${TEST_404:-0}" = 1 ]; then printf 404; else cp "$TEST_ROOT/archive.tar.gz" "$out"; printf 200; fi ;;
  */SHA256SUMS) cp "$TEST_ROOT/SHA256SUMS" "$out"; printf 200 ;;
  *) printf 500 ;;
esac
STUB
chmod +x "$scratch/bin/"{curl,uname}
run_install() {
  : > "$scratch/urls"
  env -u MOLD_CUDA_ARCH -u CUDA_VISIBLE_DEVICES -u MOLD_CHANNEL -u MOLD_VERSION \
    MOLD_BACKEND=auto MOLD_INSTALL_DIR="$scratch/install" \
    PATH="$scratch/bin" TEST_ROOT="$scratch" "$@" /bin/sh "$repo_root/install.sh" > "$scratch/output" 2>&1
}
# No probe command, latest stable resolution, real extraction and executable mode.
run_install
[[ "$("$scratch/install/mold")" == GPU-free-fixture ]]
grep -q '/download/v-test/.*-cpu.tar.gz' "$scratch/urls"
cat > "$scratch/bin/nvidia-smi" <<'STUB'
#!/bin/sh
printf 'probed\n' >> "$TEST_ROOT/probes"
case "${TEST_PROBE:-empty}" in
  fail) exit 1 ;;
  partial-fail) echo 'not a successful inventory'; exit 1 ;;
  gpu) [ "$1" = -L ] || echo '0, GPU-123, 8.9' ;;
  empty) exit 0 ;;
esac
STUB
chmod +x "$scratch/bin/nvidia-smi"
for probe in empty fail partial-fail; do
  run_install TEST_PROBE="$probe" MOLD_CHANNEL=nightly
  grep -q '/download/latest/.*-cpu.tar.gz' "$scratch/urls"
done
for visibility in '' -1 ' -1 '; do
  run_install TEST_PROBE=gpu CUDA_VISIBLE_DEVICES="$visibility" MOLD_VERSION=v-pinned
  grep -q '/download/v-pinned/.*-cpu.tar.gz' "$scratch/urls"
done
: > "$scratch/probes"
run_install TEST_PROBE=gpu MOLD_BACKEND=cpu MOLD_VERSION=v-pinned
[[ ! -s "$scratch/probes" ]]
for args in 'MOLD_BACKEND=invalid' 'MOLD_BACKEND=cpu MOLD_CUDA_ARCH=sm89' 'MOLD_BACKEND=cuda TEST_PROBE=fail' 'MOLD_BACKEND=cuda TEST_OS=Darwin TEST_ARCH=arm64' 'MOLD_BACKEND=cpu TEST_OS=Darwin TEST_ARCH=arm64'; do
  # Intentional word splitting: fixed test inputs, one assignment per word.
  # shellcheck disable=SC2086
  if run_install MOLD_VERSION=v-test $args; then echo "accepted $args" >&2; exit 1; fi
  [[ ! -s "$scratch/urls" ]]
done
if run_install MOLD_BACKEND=cpu MOLD_VERSION=v-old TEST_404=1; then exit 1; fi
grep -q 'no GPU-free Linux CLI archive' "$scratch/output"
[[ "$(wc -l < "$scratch/urls" | tr -d ' ')" == 1 ]]
grep -q '/download/v-old/.*-cpu.tar.gz' "$scratch/urls"
# A bad checksum must leave the installed binary untouched.
printf 'existing\n' > "$scratch/install/mold"
printf '%064d  mold-x86_64-unknown-linux-gnu-cpu.tar.gz\n' 0 > "$scratch/SHA256SUMS"
if run_install MOLD_BACKEND=cpu MOLD_VERSION=v-test; then exit 1; fi
grep -q 'SHA-256 mismatch' "$scratch/output"
[[ "$(cat "$scratch/install/mold")" == existing ]]
echo 'GPU-free installer: ok'
