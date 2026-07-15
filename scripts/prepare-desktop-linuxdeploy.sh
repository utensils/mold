#!/usr/bin/env bash
set -euo pipefail

arch=$(uname -m)
if [[ "$arch" != "x86_64" ]]; then
  echo "unsupported Linux AppImage architecture: $arch" >&2
  exit 1
fi

cache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/tauri"
tool="$cache_dir/linuxdeploy-$arch.AppImage"
real_tool="$cache_dir/linuxdeploy-$arch.real.AppImage"
stamp="$cache_dir/linuxdeploy-$arch.mold-wrapper"
linuxdeploy_sha256="e762bea85c8eb0d4b3508d46e5c1f037f717d0f9303ae3b4aafc8b04991fa1ef"

mkdir -p "$cache_dir"

if [[ -f "$tool" && ! -f "$stamp" ]]; then
  if [[ ! -x "$real_tool" && "$(head -c 4 "$tool")" == $'\x7fELF' ]]; then
    mv "$tool" "$real_tool"
  else
    rm -f "$tool"
  fi
fi

download_linuxdeploy() {
  curl --fail --location --retry 3 \
    "https://github.com/tauri-apps/binary-releases/releases/download/linuxdeploy/linuxdeploy-$arch.AppImage" \
    --output "$real_tool"
  chmod +x "$real_tool"
}

if [[ ! -x "$real_tool" ]]; then
  download_linuxdeploy
fi
if ! printf '%s  %s\n' "$linuxdeploy_sha256" "$real_tool" | sha256sum --check --status; then
  rm -f "$real_tool"
  download_linuxdeploy
fi
printf '%s  %s\n' "$linuxdeploy_sha256" "$real_tool" | sha256sum --check --status

wrapper=$(mktemp "$cache_dir/.linuxdeploy-wrapper.XXXXXX")
"${CC:-cc}" -O2 -x c -o "$wrapper" - <<'EOF'
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
  char real_tool[PATH_MAX];
  if (strlen(argv[0]) >= sizeof(real_tool)) return 126;
  strcpy(real_tool, argv[0]);
  char *name = strrchr(real_tool, '/');
  name = name == NULL ? real_tool : name + 1;
  static const char real_name[] = "linuxdeploy-x86_64.real.AppImage";
  size_t prefix_length = (size_t)(name - real_tool);
  if (prefix_length + sizeof(real_name) > sizeof(real_tool)) return 126;
  memcpy(name, real_name, sizeof(real_name));

  char **forwarded = calloc((size_t)argc + 3, sizeof(char *));
  if (forwarded == NULL) return 126;
  forwarded[0] = real_tool;
  int next = 1;
  int extracts = 0;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--appimage-extract-and-run") == 0) extracts = 1;
  }
  if (!extracts) forwarded[next++] = "--appimage-extract-and-run";
  for (int i = 1; i < argc; i++) forwarded[next++] = argv[i];
  forwarded[next] = "--exclude-library=libcuda.so*";

  execv(real_tool, forwarded);
  fprintf(stderr, "failed to run %s: %s\n", real_tool, strerror(errno));
  return 126;
}
EOF
mv "$wrapper" "$tool"
touch "$stamp"
