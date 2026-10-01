#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir "$work/bin"
cat > "$work/bin/xcrun" <<'STUB'
#!/usr/bin/env bash
case "$*" in
  'simctl list devices') echo 'Fixture iPhone (fixture-device) (Booted)' ;;
  'simctl ui fixture-device appearance') echo light ;;
esac
STUB
cat > "$work/bin/xcodebuild" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$UITEST_CALLS"
case " $* " in
  *' -retry-tests-on-failure -test-iterations 2 -test-repetition-relaunch-enabled YES '*) ;;
  *) echo 'error: UI tests need exactly one bounded retry'; exit 3 ;;
esac
result=''
while (( $# )); do
  if [[ $1 == -resultBundlePath ]]; then result=$2; break; fi
  shift
done
[[ -n $result ]] || { echo 'error: missing result bundle'; exit 4; }
mkdir -p "$result"
echo 'Detailed test output retained for diagnosis'
if [[ ${UITEST_FAIL:-0} == 1 ]]; then echo '** TEST FAILED **'; exit 7; fi
echo '** TEST SUCCEEDED **'
STUB
chmod +x "$work/bin/"*
export PATH="$work/bin:$PATH" UITEST_CALLS="$work/calls"
run_audit() {
  make -s -C "$root/apps/ios" -o gen -o require-sim uitest \
    SIM=fixture-device UITEST_DEVICES=fixture-device BUILD="$work/build" \
    MARKETING_VERSION=0.0.0 BUILD_NUMBER=1
}
run_audit
[[ $(wc -l < "$UITEST_CALLS" | tr -d ' ') == 2 ]]
for mode in light dark; do
  [[ -d "$work/build/UITestResults/fixture-device-$mode.xcresult" ]]
  grep -q 'Detailed test output retained' "$work/build/UITestResults/fixture-device-$mode.log"
done
# Existing output must not prevent a fresh run; a failed xcodebuild must
# still fail make through tee/grep, and both appearance passes must run.
rm "$UITEST_CALLS"
if UITEST_FAIL=1 run_audit; then echo 'FAIL: repeated test failure was swallowed' >&2; exit 1; fi
[[ $(wc -l < "$UITEST_CALLS" | tr -d ' ') == 2 ]]
echo 'ios-uitest-runner: ok'
