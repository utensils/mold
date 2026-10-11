#!/usr/bin/env bash
# Exercise the real Apple build scripts with only portable tools, no rg or Xcode.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
repo="$scratch/repo"
bin="$scratch/bin"
mkdir -p "$bin" "$repo/apps/shared/scripts" "$repo/crates/mold-secure/include"
cp "$root/apps/shared/build-secure-framework.sh" "$repo/apps/shared/"
cp "$root/apps/shared/scripts/isolate-rust-staticlib.sh" "$repo/apps/shared/scripts/"
cp "$root/crates/mold-secure/include/"* "$repo/crates/mold-secure/include/"
for tool in bash dirname mkdir rmdir cp mktemp rm grep awk sort sed diff shasum cat sleep; do
  ln -s "$(command -v "$tool")" "$bin/$tool"
done
cat > "$bin/rustup" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
if [[ "$2" == list ]]; then
  [[ "${FAIL_LIST:-0}" == 0 ]] || exit 19
  printf '%s\n' aarch64-apple-darwin x86_64-apple-darwin aarch64-apple-ios aarch64-apple-ios-sim
else
  printf 'add %s\n' "$5" >> "$CALLS"
fi
SH
cat > "$bin/cargo" <<'SH'
#!/usr/bin/env bash
printf 'cargo\n' >> "$CALLS"
SH
cat > "$bin/xcrun" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
case "$1" in
  --find) printf '/usr/bin/clang\n' ;;
  --sdk) printf '26.0\n' ;;
  nmedit) cp "$3" "$EXPORTS" ;;
  nm)
    while IFS= read -r symbol; do printf '000 T %s\n' "$symbol"; done < "$EXPORTS"
    [[ "${EXTRA_EXPORT:-0}" == 0 ]] || printf '000 T _rust_implementation\n'
    [[ "${FAIL_NM:-0}" == 0 ]] || exit 23
    ;;
  ld|libtool|lipo)
    while [[ $# -gt 0 ]]; do
      if [[ "$1" == -o || "$1" == -output ]]; then : > "$2"; break; fi
      shift
    done
    ;;
  *) exit 29 ;;
esac
SH
cat > "$bin/xcodebuild" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
while [[ $# -gt 0 ]]; do
  if [[ "$1" == -output ]]; then mkdir -p "$2"; break; fi
  shift
done
SH
chmod +x "$bin/rustup" "$bin/cargo" "$bin/xcrun" "$bin/xcodebuild"
export PATH="$bin" CALLS="$scratch/calls" EXPORTS="$scratch/exports"
if command -v rg >/dev/null 2>&1; then echo 'test PATH unexpectedly contains rg' >&2; exit 1; fi

# Existing targets must not be reinstalled, and the exact ABI allowlist survives.
if ! bash "$repo/apps/shared/build-secure-framework.sh" > "$scratch/build.log" 2>&1; then
  cat "$scratch/build.log" >&2
  exit 1
fi
[[ $(grep -c '^cargo$' "$CALLS") == 5 ]]
[[ $(grep -c '^add ' "$CALLS") == 1 ]]
grep -Fx 'add x86_64-apple-ios' "$CALLS" >/dev/null
grep -Eo 'mold_secure_[a-z_]+' "$repo/crates/mold-secure/include/mold_secure.h" |
  sort -u | sed 's/^/_/' > "$scratch/expected"
diff -u "$scratch/expected" "$EXPORTS"

# A failed toolchain inventory must not be mistaken for a missing target.
: > "$CALLS"
if FAIL_LIST=1 bash "$repo/apps/shared/build-secure-framework.sh" > "$scratch/list.log" 2>&1; then
  echo 'failed rustup inventory was accepted' >&2; exit 1
fi
[[ ! -s "$CALLS" ]]

isolate=(bash "$repo/apps/shared/scripts/isolate-rust-staticlib.sh" ignored.a arm64 macos macosx
  "$repo/crates/mold-secure/include/mold_secure.h" "$scratch/result.a")
# Correct-looking downstream output cannot hide a producer's nonzero exit.
if FAIL_NM=1 "${isolate[@]}" > "$scratch/nm.log" 2>&1; then
  echo 'failed nm pipeline was accepted' >&2; exit 1
fi
[[ ! -e "$scratch/result.a" ]]
# An extra implementation symbol must still reject the complete archive.
if EXTRA_EXPORT=1 "${isolate[@]}" > "$scratch/extra.log" 2>&1; then
  echo 'unexpected exported symbol was accepted' >&2; exit 1
fi
[[ ! -e "$scratch/result.a" ]]
printf 'portable secure framework build contracts passed\n'
