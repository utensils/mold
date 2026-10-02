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
  'xcresulttool get test-results tests '* )
    [[ ${UITEST_FAIL:-} != infra ]] || exit 9
    cat "${@: -1}/tests.json" ;;
esac
STUB
cat > "$work/bin/xcodebuild" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$UITEST_CALLS"
case " $* " in
  *' -retry-tests-on-failure '*|*' -test-iterations '*|*' -test-repetition-relaunch-enabled '*)
    echo 'error: implicit full-target repetition is forbidden'; exit 3 ;;
esac
retry=false
[[ " $* " != *' -only-testing:MoldCompanionUITests/HiddenCollectionTests/testExtraSmall '* ]] || retry=true
result=''
while (( $# )); do
  if [[ $1 == -resultBundlePath ]]; then result=$2; break; fi
  shift
done
[[ -n $result ]] || { echo 'error: missing result bundle'; exit 4; }
mkdir -p "$result"
cat > "$result/tests.json" <<'JSON'
{"testNodes":[{"nodeType":"Test Case","result":"Failed","nodeIdentifierURL":"test://com.apple.xcode/MoldCompanion/MoldCompanionUITests/HiddenCollectionTests/testExtraSmall"}]}
JSON
echo 'Detailed test output retained for diagnosis'
if [[ ${UITEST_FAIL:-} == always || ( -n ${UITEST_FAIL:-} && $retry == false ) ]]; then
  echo '** TEST FAILED **'; exit 65
fi
if [[ $retry == true && ${UITEST_FAIL:-} == empty ]]; then
  echo '{"testNodes":[]}' > "$result/tests.json"
elif [[ $retry == true ]]; then
  python3 - "$result/tests.json" <<'PYRESULT'
import json, sys
p = sys.argv[1]
report = json.load(open(p))
report['testNodes'][0]['result'] = 'Passed'
with open(p, 'w') as output: json.dump(report, output)
PYRESULT
fi
echo '** TEST SUCCEEDED **'
STUB
chmod +x "$work/bin/"*
export PATH="$work/bin:$PATH" UITEST_CALLS="$work/calls"
run_audit() {
  make -s -C "$root/apps/ios" -o gen -o require-sim uitest \
    SIM=fixture-device UITEST_DEVICES=fixture-device BUILD="$work/build" \
    MARKETING_VERSION=0.0.0 BUILD_NUMBER=1 "$@"
}
run_audit
[[ $(wc -l < "$UITEST_CALLS" | tr -d ' ') == 2 ]]
grep -q -- '-only-testing:MoldCompanionUITests -resultBundlePath' "$UITEST_CALLS"
for mode in light dark; do
  [[ -d "$work/build/UITestResults/fixture-device-$mode.xcresult" ]]
  grep -q 'Detailed test output retained' "$work/build/UITestResults/fixture-device-$mode.log"
done
# One failed case must retry exactly once, with original and retry diagnostics.
: > "$UITEST_CALLS"
UITEST_FAIL=once run_audit UITEST_APPEARANCES=light
[[ $(wc -l < "$UITEST_CALLS" | tr -d ' ') == 2 ]]
grep -q -- '-only-testing:MoldCompanionUITests/HiddenCollectionTests/testExtraSmall ' "$UITEST_CALLS"
[[ -d "$work/build/UITestResults/fixture-device-light-retry.xcresult" ]]
[[ -f "$work/build/UITestResults/fixture-device-light-retry.log" ]]
# Successful reruns at the same result path remove prior retry diagnostics.
: > "$UITEST_CALLS"
run_audit UITEST_APPEARANCES=light
[[ ! -e "$work/build/UITestResults/fixture-device-light-retry.xcresult" ]]
[[ ! -e "$work/build/UITestResults/fixture-device-light-retry.log" ]]
# A zero-test retry may exit zero, but its public result proof must keep CI red.
: > "$UITEST_CALLS"
if UITEST_FAIL=empty run_audit UITEST_APPEARANCES=light; then exit 1; fi
[[ $(wc -l < "$UITEST_CALLS" | tr -d ' ') == 2 ]]
# Repeated failures remain red; an infra failure cannot launch a broad retry.
: > "$UITEST_CALLS"
if UITEST_FAIL=always run_audit UITEST_APPEARANCES=light; then exit 1; fi
[[ $(wc -l < "$UITEST_CALLS" | tr -d ' ') == 2 ]]
: > "$UITEST_CALLS"
if UITEST_FAIL=infra run_audit UITEST_APPEARANCES=light; then exit 1; fi
[[ $(wc -l < "$UITEST_CALLS" | tr -d ' ') == 1 ]]
: > "$UITEST_CALLS"
run_audit UITEST_APPEARANCES=dark UITEST_CLASSES='LibraryLongPressTests LibraryViewerTests'
[[ $(wc -l < "$UITEST_CALLS" | tr -d ' ') == 1 ]]
grep -q -- '-only-testing:MoldCompanionUITests/LibraryLongPressTests ' "$UITEST_CALLS"
grep -q -- '-only-testing:MoldCompanionUITests/LibraryViewerTests ' "$UITEST_CALLS"
if grep -q -- '-only-testing:MoldCompanionUITests ' "$UITEST_CALLS"; then exit 1; fi
echo 'ios-uitest-runner: ok'
