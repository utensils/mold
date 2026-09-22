#!/usr/bin/env bash
# Build + install + smoke-test an in-tree PKGBUILD inside an Arch
# Linux container. See packaging/aur/test/Dockerfile.
#
# Usage:
#   scripts/aur/test-in-docker.sh                # mold-ai-bin (default)
#   scripts/aur/test-in-docker.sh mold-ai-bin
#   scripts/aur/test-in-docker.sh mold-ai
#   scripts/aur/test-in-docker.sh mold-ai-git
#   scripts/aur/test-in-docker.sh --version 0.31.0 [pkg]  # target a release
#                                                   # (default: the latest
#                                                   # GitHub release; the
#                                                   # in-tree pkgver is a
#                                                   # placeholder CI rewrites)
#   scripts/aur/test-in-docker.sh --archive /path/to/cpu.tar.gz mold-ai-bin
#   scripts/aur/test-in-docker.sh --as-is [pkg]     # keep the in-tree pkgver
#   scripts/aur/test-in-docker.sh --rebuild [pkg]   # force image rebuild
#   scripts/aur/test-in-docker.sh --shell [pkg]     # drop into a shell
#                                                   # after the build
#
# For mold-ai-bin the run has two phases. The package is CREATED first with
# `makepkg --nodeps`, before any dependency is installed, on an image that
# carries no CUDA library at all: a recipe that executes the prebuilt payload
# to generate its completions fails right there, the way 0.30.1 did with
# `libcudart.so.12: cannot open shared object file` (#1742). Only then are
# the runtime dependencies pulled in for `pacman -U` and the `mold --version`
# smoke, whose loader state is reported by `ldd` when it fails.
set -euo pipefail

IMAGE="mold-aur-test"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DOCKERFILE="${REPO_ROOT}/packaging/aur/test/Dockerfile"
CONTEXT="${REPO_ROOT}/packaging/aur/test"

# Pick whichever container runtime is on PATH. Linux contributors
# typically have podman (Arch default); macOS contributors usually
# have docker (Docker Desktop / OrbStack).
runtime=""
for candidate in docker podman; do
  if command -v "$candidate" >/dev/null 2>&1; then
    runtime="$candidate"
    break
  fi
done
if [ -z "$runtime" ]; then
  echo "error: neither docker nor podman is on PATH" >&2
  exit 1
fi

pkgname=""
version=""
archive=""
as_is=false
rebuild=false
shell_after=false

while [ "$#" -gt 0 ]; do
  case "$1" in
    --rebuild) rebuild=true ;;
    --shell)   shell_after=true ;;
    --as-is)   as_is=true ;;
    --archive)
      [ "$#" -ge 2 ] || { echo "error: --archive needs a path" >&2; exit 64; }
      archive="$2"
      shift
      ;;
    --version)
      [ "$#" -ge 2 ] || { echo "error: --version needs a value" >&2; exit 64; }
      version="${2#v}"
      shift
      ;;
    -h|--help)
      sed -n '2,27p' "$0" >&2
      exit 0
      ;;
    mold-ai-bin|mold-ai|mold-ai-git)
      pkgname="$1"
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      exit 64
      ;;
  esac
  shift
done

pkgname="${pkgname:-mold-ai-bin}"

# Stage the recipe on the host. mold-ai-bin and mold-ai are published per
# release with `pkgver` + checksums rewritten by CI, and the in-tree values
# are a placeholder — so unless told otherwise, point the copy at a real
# release the same way the publish job does. mold-ai-git tracks main and is
# never bumped.
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
cp -a "${REPO_ROOT}/packaging/aur/${pkgname}/." "$stage/"
if [ -n "$archive" ]; then
  [ "$pkgname" = mold-ai-bin ] || { echo '--archive requires mold-ai-bin' >&2; exit 64; }
  if [ -n "$version" ] || "$as_is"; then echo '--archive conflicts with --version/--as-is' >&2; exit 64; fi
  cp "$archive" "$stage/mold-x86_64-unknown-linux-gnu-cpu.tar.gz"
  cp "$REPO_ROOT/LICENSE" "$stage/LICENSE"
  python3 - "$stage" <<'PYARCH'
import hashlib
from pathlib import Path
import re
import sys
stage = Path(sys.argv[1])
s = (stage / "PKGBUILD").read_text()
for field, filename in (("source", "LICENSE"), ("source_x86_64", "mold-x86_64-unknown-linux-gnu-cpu.tar.gz")):
    s = re.sub(rf"^{field}=.*$", f"{field}=('{filename}')", s, flags=re.M)
    checksum = hashlib.sha256((stage / filename).read_bytes()).hexdigest()
    sums = "sha256sums" if field == "source" else "sha256sums_x86_64"
    s = re.sub(rf"^{sums}=.*$", f"{sums}=('{checksum}')", s, flags=re.M)
s = s.replace('LICENSE-${pkgver}', 'LICENSE')
(stage / "PKGBUILD").write_text(s)
PYARCH
elif [ "$pkgname" != mold-ai-git ] && ! "$as_is"; then
  if [ -z "$version" ]; then
    version="$(
      curl --silent --show-error --fail --location \
        https://api.github.com/repos/utensils/mold/releases/latest \
        | sed -n 's/.*"tag_name": *"v\([^"]*\)".*/\1/p'
    )"
    [ -n "$version" ] || { echo "error: could not resolve the latest release; pass --version" >&2; exit 1; }
  fi
  echo "==> targeting release v${version}"
  MOLD_AUR_PKGBUILD="$stage/PKGBUILD" \
    "${REPO_ROOT}/scripts/aur/update-pkgbuild.sh" "$pkgname" "$version"
fi

# (Re)build the image if asked or if it's missing.
if "$rebuild" || ! "$runtime" image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "==> building $IMAGE via $runtime"
  "$runtime" build -t "$IMAGE" -f "$DOCKERFILE" "$CONTEXT"
fi

echo "==> running $pkgname build + smoke test inside $IMAGE"

# The staged recipe is copied to a writable tmp dir so makepkg's build tree
# doesn't pollute the host. `$workdir` lives in the container shell, not the
# host shell — it is escaped so the host doesn't try (and fail) to
# interpolate it before the string ever reaches the container.
if [ "$pkgname" = mold-ai-bin ]; then
  build_cmd=$(cat <<INNER
set -euo pipefail
workdir=\$(mktemp -d)
cp -a /pkgbuild/. "\$workdir/"
cd "\$workdir"

echo "==> phase 1: package creation with no CUDA library on the box"
if ldconfig -p | grep -q 'libcudart\\.so'; then
  echo "error: the test image already carries a CUDA runtime; it cannot prove package() needs none" >&2
  exit 1
fi
unset LD_LIBRARY_PATH
echo "==> makepkg --noconfirm --nodeps (pkg: ${pkgname})"
makepkg --noconfirm --nodeps
pkgfile=\$(ls -1 ./*.pkg.tar.* | head -n 1)
echo "==> built \$pkgfile without a CUDA runtime present"
bsdtar -xOf "\$pkgfile" .INSTALL | grep -F 'GPU-free CLI' >/dev/null || { echo "error: missing CPU migration install notice" >&2; exit 1; }
for member in    usr/bin/mold    usr/share/bash-completion/completions/mold    usr/share/zsh/site-functions/_mold    usr/share/fish/vendor_completions.d/mold.fish; do
  bsdtar -tf "\$pkgfile" | grep -qx "\$member"      || { echo "error: \$pkgfile is missing \$member" >&2; exit 1; }
done
for member in    usr/share/bash-completion/completions/mold    usr/share/zsh/site-functions/_mold    usr/share/fish/vendor_completions.d/mold.fish; do
  [ -n "\$(bsdtar -xOf "\$pkgfile" "\$member")" ]      || { echo "error: \$member is empty" >&2; exit 1; }
  if bsdtar -xOf "\$pkgfile" "\$member" | grep -q "\$workdir"; then
    echo "error: \$member bakes in the build directory" >&2
    exit 1
  fi
done
echo "✓ ${pkgname} package created without executing the payload"

echo "==> phase 2: install with runtime dependencies, then smoke"
sudo pacman -Syu --noconfirm
sudo pacman -U --noconfirm --needed "\$pkgfile"
echo "==> smoke test: \$(which mold)"
if ! mold --version; then
  echo "error: the installed binary does not run; unresolved libraries:" >&2
  ldd /usr/bin/mold | grep 'not found' >&2 || true
  exit 1
fi
if sudo pacman -Qq | grep -Ei '^(cuda|cudnn|nvidia)(-|$)'; then
  echo "error: GPU-free package installed NVIDIA dependencies" >&2
  exit 1
fi
bash /workspace/scripts/verify-cpu-release-binary.sh "\$workdir/mold-x86_64-unknown-linux-gnu-cpu.tar.gz"
python3 /workspace/scripts/tests/cpu-release-smoke.py /usr/bin/mold
echo "✓ ${pkgname} builds, installs, runs without CUDA"
INNER
)
else
  build_cmd=$(cat <<INNER
set -euo pipefail
workdir=\$(mktemp -d)
cp -a /pkgbuild/. "\$workdir/"
cd "\$workdir"
echo "==> makepkg -si --noconfirm --needed (pkg: ${pkgname})"
makepkg -si --noconfirm --needed
echo "==> smoke test: \$(which mold)"
if ! mold --version; then
  echo "error: the installed binary does not run; unresolved libraries:" >&2
  ldd /usr/bin/mold | grep 'not found' >&2 || true
  exit 1
fi
echo "✓ ${pkgname} builds, installs, runs"
INNER
)
fi

if "$shell_after"; then
  build_cmd="${build_cmd}"$'\n''echo "==> dropping into shell (the built workdir is \$workdir)"; exec bash'
fi

tty_args=()
if [[ -t 0 && -t 1 ]]; then tty_args=(-it); fi
exec "$runtime" run --rm "${tty_args[@]}" --init \
  -v "${REPO_ROOT}:/workspace:rw" \
  -v "${stage}:/pkgbuild:ro" \
  "$IMAGE" \
  bash -lc "$build_cmd"
