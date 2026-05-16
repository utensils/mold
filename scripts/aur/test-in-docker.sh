#!/usr/bin/env bash
# Build + install + smoke-test an in-tree PKGBUILD inside an Arch
# Linux container. See packaging/aur/test/Dockerfile.
#
# Usage:
#   scripts/aur/test-in-docker.sh                # mold-ai-bin (default)
#   scripts/aur/test-in-docker.sh mold-ai-bin
#   scripts/aur/test-in-docker.sh mold-ai
#   scripts/aur/test-in-docker.sh mold-ai-git
#   scripts/aur/test-in-docker.sh --rebuild [pkg]   # force image rebuild
#   scripts/aur/test-in-docker.sh --shell [pkg]     # drop into a shell
#                                                   # after the build
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
rebuild=false
shell_after=false

while [ "$#" -gt 0 ]; do
  case "$1" in
    --rebuild) rebuild=true ;;
    --shell)   shell_after=true ;;
    -h|--help)
      sed -n '2,15p' "$0" >&2
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

# (Re)build the image if asked or if it's missing.
if "$rebuild" || ! "$runtime" image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "==> building $IMAGE via $runtime"
  "$runtime" build -t "$IMAGE" -f "$DOCKERFILE" "$CONTEXT"
fi

echo "==> running $pkgname build + smoke test inside $IMAGE"

# Copy the PKGBUILD to a writable tmp dir so makepkg's build tree
# doesn't pollute the host repo.
build_cmd=$(cat <<INNER
set -euo pipefail
workdir=\$(mktemp -d)
cp -a /workspace/packaging/aur/${pkgname}/. "\$workdir/"
cd "\$workdir"
echo "==> makepkg -si --noconfirm --needed (pkg: ${pkgname})"
makepkg -si --noconfirm --needed
echo "==> smoke test: \$(which mold)"
mold --version
echo "✓ ${pkgname} builds, installs, runs"
INNER
)

if "$shell_after"; then
  # `$workdir` lives in the container shell, not the host shell — escape
  # it so the host doesn't try (and fail) to interpolate it before the
  # string ever reaches the container. Same reason the main heredoc uses
  # `\$workdir` throughout.
  build_cmd="${build_cmd}"$'\n''echo "==> dropping into shell (the built workdir is \$workdir)"; exec bash'
fi

exec "$runtime" run --rm -it --init \
  -v "${REPO_ROOT}:/workspace:rw" \
  "$IMAGE" \
  bash -lc "$build_cmd"
