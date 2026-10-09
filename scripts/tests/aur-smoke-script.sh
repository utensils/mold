#!/usr/bin/env bash
# Execute the generated container checks; older Bash folds heredoc continuations
# differently inside command substitutions (the shipping builder uses Bash 5.1).
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
mkdir "$scratch/bin"
cat > "$scratch/bin/docker" <<'STUB'
#!/usr/bin/env bash
if [[ "$1" == run ]]; then printf '%s\n' "${@: -1}" > "$TEST_CAPTURE"; fi
STUB
chmod +x "$scratch/bin/docker"
PATH="$scratch/bin:$PATH" TEST_CAPTURE="$scratch/generated" \
  bash "$repo_root/scripts/aur/test-in-docker.sh" --as-is >/dev/null
bash -n "$scratch/generated"
awk '/^bsdtar -xOf .*\.INSTALL/ {emit=1} /^echo .*package created/ {exit} emit' \
  "$scratch/generated" > "$scratch/checks"
[[ -s "$scratch/checks" ]]
cat > "$scratch/fixture" <<'STUB'
set -euo pipefail
pkgfile=fixture.pkg.tar.zst
workdir=/tmp/build-fixture
bsdtar() {
  if [[ "$1" == -tf ]]; then
    for entry in usr/bin/mold usr/share/bash-completion/completions/mold usr/share/zsh/site-functions/_mold usr/share/fish/vendor_completions.d/mold.fish; do
      [[ "$entry" == "${TEST_MISSING:-}" ]] || echo "$entry"
    done
  elif [[ "$3" != "${TEST_EMPTY:-}" ]]; then
    if [[ "$3" == .INSTALL ]]; then echo 'GPU-free CLI'; else echo 'completion fixture'; fi
  fi
}
source "$1"
STUB
bash "$scratch/fixture" "$scratch/checks"
if TEST_MISSING=usr/bin/mold bash "$scratch/fixture" "$scratch/checks" > /dev/null 2>&1; then
  echo 'generated checks accepted a missing binary' >&2; exit 1
fi
for entry in .INSTALL usr/share/zsh/site-functions/_mold; do
  if TEST_EMPTY="$entry" bash "$scratch/fixture" "$scratch/checks" > /dev/null 2>&1; then
    echo "generated checks accepted empty $entry" >&2; exit 1
  fi
done

# The desktop recipes share one generated layout check; both must parse, and
# the check must refuse a package that lost any installed member.
for desktop_pkg in mold-ai-desktop mold-ai-desktop-bin; do
  PATH="$scratch/bin:$PATH" TEST_CAPTURE="$scratch/$desktop_pkg" \
    bash "$repo_root/scripts/aur/test-in-docker.sh" --as-is "$desktop_pkg" >/dev/null
  bash -n "$scratch/$desktop_pkg"
  grep -Fq 'desktop-file-validate /usr/share/applications/com.utensils.mold.desktop' \
    "$scratch/$desktop_pkg"
done
awk '/^for member in .*usr\/bin\/mold-desktop/ {emit=1} emit {print} emit && /^done$/ {exit}' \
  "$scratch/mold-ai-desktop-bin" > "$scratch/desktop-checks"
[[ -s "$scratch/desktop-checks" ]]
cat > "$scratch/desktop-fixture" <<'STUB'
set -euo pipefail
pkgfile=fixture.pkg.tar.zst
bsdtar() {
  for entry in usr/bin/mold-desktop usr/share/applications/com.utensils.mold.desktop \
    usr/share/metainfo/com.utensils.mold.metainfo.xml \
    usr/share/icons/hicolor/128x128/apps/com.utensils.mold.png \
    usr/share/icons/hicolor/512x512/apps/com.utensils.mold.png; do
    [[ "$entry" == "${TEST_MISSING:-}" ]] || echo "$entry"
  done
}
source "$1"
STUB
bash "$scratch/desktop-fixture" "$scratch/desktop-checks"
for entry in usr/bin/mold-desktop usr/share/applications/com.utensils.mold.desktop; do
  if TEST_MISSING="$entry" bash "$scratch/desktop-fixture" "$scratch/desktop-checks" > /dev/null 2>&1; then
    echo "generated desktop checks accepted a missing $entry" >&2; exit 1
  fi
done
echo 'generated Arch package smoke checks: ok'
