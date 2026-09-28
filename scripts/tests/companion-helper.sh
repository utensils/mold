#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
mkdir "$scratch/bin"
cat > "$scratch/bin/make" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$@" > "$COMPANION_TEST_ARGS"
SH
chmod +x "$scratch/bin/make"
export PATH="$scratch/bin:$PATH" COMPANION_TEST_ARGS="$scratch/args"
for action in run build gen test uitest packages-test lint; do
  (cd /tmp && "$root/scripts/companion.sh" "$action" 'BUILD=/tmp/build with spaces' SIM=test-device)
  printf '%s\n' -C "$root/apps/ios" "$action" 'BUILD=/tmp/build with spaces' SIM=test-device > "$scratch/expected"
  diff -u "$scratch/expected" "$scratch/args"
done
if "$root/scripts/companion.sh" invalid > /dev/null 2>&1; then
  echo 'invalid action was accepted' >&2; exit 1
fi
"$root/scripts/companion.sh" --help | grep -q companion
echo 'native iOS helper contracts passed'
