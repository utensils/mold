#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
packager="$repo_root/scripts/create-desktop-dmg.sh"
workflow="$repo_root/.github/workflows/desktop-distribution.yml"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

mkdir -p "$tmp/bin" "$tmp/Mold.app/Contents/MacOS" \
  "$tmp/Mold.app/Contents/Resources" "$tmp/out"
printf 'signed app fixture\n' > "$tmp/Mold.app/Contents/MacOS/mold-desktop"
printf 'icon fixture\n' > "$tmp/Mold.app/Contents/Resources/icon.icns"

cat > "$tmp/bin/ditto" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cp -R "$1" "$2"
EOF

cat > "$tmp/bin/hdiutil" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
count=0
[[ ! -f "$MOCK_HDIUTIL_COUNT" ]] || count="$(cat "$MOCK_HDIUTIL_COUNT")"
count=$((count + 1))
printf '%s\n' "$count" > "$MOCK_HDIUTIL_COUNT"

source_dir=''
for ((i = 1; i <= $#; i++)); do
  if [[ "${!i}" == '-srcfolder' ]]; then
    next=$((i + 1))
    source_dir="${!next}"
  fi
done
[[ -d "$source_dir/Mold.app" ]] || exit 40
[[ -L "$source_dir/Applications" ]] || exit 41
[[ -f "$source_dir/.VolumeIcon.icns" ]] || exit 42

if ((count < ${MOCK_HDIUTIL_SUCCEED_AT:-3})); then
  echo "mock transient DiskImages failure $count" >&2
  exit 1
fi

touch "${!#}"
EOF

cat > "$tmp/bin/SetFile" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$MOCK_SETFILE_ARGS"
EOF

cat > "$tmp/bin/codesign" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" > "$MOCK_CODESIGN_ARGS"
[[ -f "${!#}" ]]
EOF
chmod +x "$tmp/bin/ditto" "$tmp/bin/hdiutil" "$tmp/bin/SetFile" \
  "$tmp/bin/codesign"

PATH="$tmp/bin:$PATH" \
  MOCK_HDIUTIL_COUNT="$tmp/hdiutil-count" \
  MOCK_CODESIGN_ARGS="$tmp/codesign-args" \
  MOCK_SETFILE_ARGS="$tmp/setfile-args" \
  MOLD_DMG_MAX_ATTEMPTS=3 \
  MOLD_DMG_RETRY_DELAY_SECONDS=0 \
  "$packager" "$tmp/Mold.app" "$tmp/out/Mold.dmg" 'Developer ID Test'

[[ "$(cat "$tmp/hdiutil-count")" == 3 ]] \
  || fail "the packager did not retry hdiutil twice"
[[ -f "$tmp/out/Mold.dmg" ]] || fail "the successful DMG was not published"
grep -Fq -- '--sign Developer ID Test --timestamp' "$tmp/codesign-args" \
  || fail "the DMG was not signed with the requested identity"
grep -Fq -- '-c icnC' "$tmp/setfile-args" \
  || fail "the copied volume icon did not receive its file type"
grep -Fq -- '-a C' "$tmp/setfile-args" \
  || fail "the DMG root did not receive its custom-icon attribute"

printf 'known prior artifact\n' > "$tmp/out/prior.dmg"
if PATH="$tmp/bin:$PATH" \
  MOCK_HDIUTIL_COUNT="$tmp/failing-hdiutil-count" \
  MOCK_HDIUTIL_SUCCEED_AT=3 \
  MOCK_CODESIGN_ARGS="$tmp/unexpected-codesign-args" \
  MOCK_SETFILE_ARGS="$tmp/failing-setfile-args" \
  MOLD_DMG_MAX_ATTEMPTS=2 \
  MOLD_DMG_RETRY_DELAY_SECONDS=0 \
  "$packager" "$tmp/Mold.app" "$tmp/out/prior.dmg" 'Developer ID Test'; then
  fail "the packager succeeded after every hdiutil attempt failed"
fi
[[ "$(cat "$tmp/failing-hdiutil-count")" == 2 ]] \
  || fail "the packager did not stop at its configured retry bound"
grep -Fqx 'known prior artifact' "$tmp/out/prior.dmg" \
  || fail "a failed replacement damaged the prior DMG"
[[ ! -e "$tmp/unexpected-codesign-args" ]] \
  || fail "the packager tried to sign a failed DMG attempt"

distribution_step="$(sed -n '/name: Build, sign, notarize, and staple/,/name: Notarize and staple DMG/p' "$workflow")"
grep -Fq -- '--bundles app --ci' <<< "$distribution_step" \
  || fail "Tauri must build only the app/updater bundle"
grep -Fq '../scripts/create-desktop-dmg.sh' <<< "$distribution_step" \
  || fail "the distribution workflow does not call the retrying DMG packager"
if grep -Fq -- '--bundles app,dmg' <<< "$distribution_step"; then
  fail "the distribution workflow still invokes Tauri's fragile DMG helper"
fi

echo "PASS: desktop-dmg-packaging"
