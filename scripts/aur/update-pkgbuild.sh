#!/usr/bin/env bash
# Refresh `pkgver` + `sha256sums` in an AUR PKGBUILD against a
# specific upstream release tag. Designed to be called from CI on a
# tag push, but also runnable locally for hand-bumping.
#
# Usage:
#   scripts/aur/update-pkgbuild.sh <pkgname> <version>
#
#   <pkgname>  one of: mold-ai-bin, mold-ai
#   <version>  the release version without the leading `v` (e.g. 0.10.0)
#
# Side effects:
#   - Mutates packaging/aur/<pkgname>/PKGBUILD in place.
#   - Always resets `pkgrel` to 1 (a new upstream version implies a
#     fresh package release).
#
# Note: this script does NOT regenerate .SRCINFO. The deploy action
# (KSXGitHub/github-actions-deploy-aur) runs `makepkg --printsrcinfo`
# inside an Arch container after this script finishes.
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "usage: $0 <pkgname> <version>" >&2
  exit 64
fi

pkgname="$1"
version="$2"

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
pkgdir="${repo_root}/packaging/aur/${pkgname}"
pkgbuild="${pkgdir}/PKGBUILD"

if [ ! -f "${pkgbuild}" ]; then
  echo "error: ${pkgbuild} does not exist" >&2
  exit 1
fi

# `sha256sum` (GNU coreutils) ships on Linux but not on macOS by
# default — macOS ships `shasum -a 256`. Probe once at top of script
# so contributors hand-bumping on macOS don't need coreutils.
if command -v sha256sum >/dev/null 2>&1; then
  _hasher() { sha256sum; }
elif command -v shasum >/dev/null 2>&1; then
  _hasher() { shasum -a 256; }
else
  echo "error: neither sha256sum nor shasum is on PATH" >&2
  exit 1
fi

fetch_sha() {
  local url="$1"
  # `-fL` so curl follows redirects and fails on 404 instead of
  # silently writing an HTML error page that hashes to garbage.
  curl --silent --show-error --fail --location "${url}" \
    | _hasher \
    | awk '{print $1}'
}

# In-place sed that works on both GNU sed (CI) and BSD sed (local
# macOS hand-bumps). `sed -i ''` is GNU-incompatible and `sed -i` is
# BSD-incompatible, so we always write to a tempfile and rename.
rewrite() {
  local file="$1" pattern="$2"
  local tmp
  tmp="$(mktemp)"
  sed -E "${pattern}" "${file}" > "${tmp}"
  mv "${tmp}" "${file}"
}

case "${pkgname}" in
  mold-ai-bin)
    base="https://github.com/utensils/mold/releases/download/v${version}"
    license_url="https://github.com/utensils/mold/raw/v${version}/LICENSE"
    # mold ships two Linux variants per release: cuda-sm89 (Ada Lovelace,
    # RTX 40-series) and cuda-sm120 (Blackwell, RTX 50-series). The -bin
    # PKGBUILD defaults to sm_89 — Blackwell users build from source.
    echo "==> fetching sha256 for sm_89 tarball"
    sha_x86_64="$(fetch_sha "${base}/mold-x86_64-unknown-linux-gnu-cuda-sm89.tar.gz")"
    echo "==> fetching sha256 for LICENSE"
    sha_license="$(fetch_sha "${license_url}")"

    rewrite "${pkgbuild}" "s/^pkgver=.*/pkgver=${version}/"
    rewrite "${pkgbuild}" "s/^pkgrel=.*/pkgrel=1/"
    rewrite "${pkgbuild}" "s|^sha256sums=.*|sha256sums=('${sha_license}')|"
    rewrite "${pkgbuild}" "s|^sha256sums_x86_64=.*|sha256sums_x86_64=('${sha_x86_64}')|"
    ;;

  mold-ai)
    src="https://github.com/utensils/mold/archive/refs/tags/v${version}.tar.gz"
    echo "==> fetching sha256 for source tarball"
    sha="$(fetch_sha "${src}")"

    rewrite "${pkgbuild}" "s/^pkgver=.*/pkgver=${version}/"
    rewrite "${pkgbuild}" "s/^pkgrel=.*/pkgrel=1/"
    rewrite "${pkgbuild}" "s|^sha256sums=.*|sha256sums=('${sha}')|"
    ;;

  mold-ai-git)
    # -git packages derive pkgver from `git describe` at build time;
    # the PKGBUILD here only changes when the build recipe changes,
    # so this script intentionally refuses to touch it.
    echo "error: mold-ai-git is not auto-bumped; edit its PKGBUILD by hand" >&2
    exit 2
    ;;

  *)
    echo "error: unknown pkgname '${pkgname}'" >&2
    exit 64
    ;;
esac

echo "==> ${pkgbuild} updated to v${version}"
