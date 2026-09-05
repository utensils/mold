#!/usr/bin/env bash
# Remove every build artifact the repo produces, backend and frontend, so the
# next build starts from nothing. The devshell's `build-clean` is a thin call
# into this script (the `ci-local` pattern), so it works outside the shell too.
#
#   build-clean                 clean everything below
#   build-clean --dry-run       print what would go, remove nothing
#   build-clean --node-modules  also remove the Bun dependency installs
#   build-clean --root DIR      operate on another checkout (the test uses it)
#
# Backend: the target directory of every cargo root — the workspace,
# desktop/src-tauri and apps/mobile/src-tauri (each has its own Cargo.toml),
# honouring CARGO_TARGET_DIR. A target directory that is a SYMLINK is emptied
# and the link kept: on the development machines every target lives on
# external storage behind `target -> /Volumes/…`, and `cargo clean` would
# delete the link itself, after which the next build lands a fresh `target/`
# on the internal disk — the disk pressure the link exists to avoid. A
# CARGO_TARGET_DIR is emptied rather than removed for the same reason: it may
# be a directory the user created and shares.
#
# Frontend: the web and desktop bundles, the VitePress site and cache, the
# Vite pre-bundle caches under every node_modules, the tsbuildinfo stamps,
# the Tauri-generated schema output, and nix `result*` links. Dependency
# installs (node_modules) are dependencies, not artifacts, so they stay
# unless asked for.
set -euo pipefail

usage() {
  sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dry_run=0
node_modules=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) dry_run=1 ;;
    --node-modules) node_modules=1 ;;
    --root)
      shift
      [ $# -gt 0 ] || { echo "build-clean: --root needs a directory" >&2; exit 2; }
      root="$1"
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "build-clean: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# Refuse anything that does not look like a mold checkout: every removal below
# is relative to this path, so a wrong root is a wrong `rm -rf`.
[ -d "$root" ] || { echo "build-clean: no such directory: $root" >&2; exit 2; }
root="$(cd "$root" && pwd)"
[ -f "$root/Cargo.toml" ] || { echo "build-clean: $root has no Cargo.toml; refusing" >&2; exit 2; }

removed=0

say() {
  if [ "$dry_run" = 1 ]; then
    echo "would remove $1"
  else
    echo "removed $1"
  fi
}

# Remove a path outright (file, directory, or symlink).
remove_path() {
  local path="$1"
  if [ -e "$path" ] || [ -L "$path" ]; then
    say "${path#"$root"/}"
    [ "$dry_run" = 1 ] || rm -rf "$path"
    removed=$((removed + 1))
  fi
}

# Empty a directory but keep it (and any symlink that points at it). Every
# directory that reaches here is SHARED — a symlinked target, or a
# CARGO_TARGET_DIR — and on the development machines that is one directory
# behind every worktree of the repo, so name it and say what else loses its
# build BEFORE removing anything.
empty_dir() {
  local dir="$1"
  local real
  real="$(cd "$dir" && pwd -P)"
  if [ -n "$(ls -A "$real")" ]; then
    echo "build-clean: shared build directory: $real"
    echo "build-clean: warning: every checkout and worktree pointing there loses its build"
    say "the contents of ${dir#"$root"/} ($real)"
    [ "$dry_run" = 1 ] || find "$real" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    removed=$((removed + 1))
  fi
}

# One cargo root's target directory: a symlink or a CARGO_TARGET_DIR is
# emptied, a plain directory is removed.
clean_target() {
  local cargo_root="$1"
  local target="${CARGO_TARGET_DIR:-$cargo_root/target}"
  if [ -L "$target" ]; then
    if [ -d "$target" ]; then
      empty_dir "$target"
    else
      # A dangling link has nothing behind it to clean.
      :
    fi
  elif [ -n "${CARGO_TARGET_DIR:-}" ]; then
    [ -d "$target" ] && empty_dir "$target"
  else
    remove_path "$target"
  fi
}

# ── backend ─────────────────────────────────────────────────────────────────
for cargo_root in "$root" "$root/desktop/src-tauri" "$root/apps/mobile/src-tauri"; do
  clean_target "$cargo_root"
done

# ── frontend ────────────────────────────────────────────────────────────────
for rel in \
  web/dist web/.vite \
  desktop/dist desktop/dist-mobile \
  website/.vitepress/dist website/.vitepress/cache \
  node_modules/.vite desktop/node_modules/.vite web/node_modules/.vite \
  website/node_modules/.vite ui/node_modules/.vite studio/node_modules/.vite \
  apps/mobile/src-tauri/gen/schemas desktop/src-tauri/gen/schemas; do
  remove_path "$root/$rel"
done

for stamp in "$root"/web/*.tsbuildinfo "$root"/desktop/*.tsbuildinfo \
  "$root"/ui/*.tsbuildinfo "$root"/studio/*.tsbuildinfo "$root"/website/*.tsbuildinfo; do
  remove_path "$stamp"
done

# nix build output links (`result`, `result-bin`, …).
for link in "$root"/result "$root"/result-*; do
  remove_path "$link"
done

if [ "$node_modules" = 1 ]; then
  for rel in node_modules web/node_modules desktop/node_modules website/node_modules \
    ui/node_modules studio/node_modules apps/mobile/node_modules; do
    remove_path "$root/$rel"
  done
fi

if [ "$removed" = 0 ]; then
  echo "build-clean: nothing to remove under $root"
elif [ "$dry_run" = 1 ]; then
  echo "build-clean: dry run, $removed item(s) would be removed under $root"
else
  echo "build-clean: $removed item(s) removed under $root"
fi
