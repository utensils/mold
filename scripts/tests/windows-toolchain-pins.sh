#!/usr/bin/env bash
set -euo pipefail

# The Windows build steps install protoc and nasm before compiling. Both used
# to come from Chocolatey, whose live package feed is a third party's uptime:
# protoc's copy 504'd once and `choco install` still exited 0 (desktop.yml has
# that story), and on 2026-09-22 the feed answered 503 for `nasm` and failed
# the v0.31.0 Windows release build after every other stage had passed. A
# release artifact's toolchain is fetched from the project's own release with
# a pinned sha256, exactly as protoc already is — never from a package feed.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
workflows=(
  "$repo_root/.github/workflows/release.yml"
  "$repo_root/.github/workflows/windows-nightly.yml"
  "$repo_root/.github/workflows/desktop.yml"
)

failed=0
for workflow in "${workflows[@]}"; do
  name="${workflow##*/}"
  if grep -nE '^\s*choco install' "$workflow"; then
    echo "FAIL: $name installs a build tool through Chocolatey's live feed; fetch a pinned release instead" >&2
    failed=1
  fi
  if ! grep -Fq 'NASM_VERSION:' "$workflow"; then
    echo "FAIL: $name does not pin a nasm version" >&2
    failed=1
  fi
  if ! grep -Fq 'NASM_SHA256:' "$workflow"; then
    echo "FAIL: $name does not pin the nasm archive sha256" >&2
    failed=1
  fi
  if ! grep -Fq 'PROTOC_VERSION:' "$workflow"; then
    echo "FAIL: $name does not pin a protoc version" >&2
    failed=1
  fi
  if ! grep -Fq 'nasm.exe" --version' "$workflow"; then
    echo "FAIL: $name does not prove nasm runs after installing it" >&2
    failed=1
  fi
done

# One pin, three copies: the version and digest must agree everywhere, or a
# bump to one workflow ships a different assembler in another.
for key in NASM_VERSION NASM_SHA256 PROTOC_VERSION; do
  values="$(grep -hoE "$key: \"[^\"]+\"" "${workflows[@]}" | sort -u)"
  if [[ "$(printf '%s\n' "$values" | grep -c .)" -ne 1 ]]; then
    echo "FAIL: $key differs across the Windows build steps:" >&2
    printf '%s\n' "$values" >&2
    failed=1
  fi
done

[[ "$failed" -eq 0 ]] || exit 1
echo "PASS: windows-toolchain-pins"
