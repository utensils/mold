#!/usr/bin/env bash
# A disposable iPhone audits the populated Library picker in both appearances.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
audit_build=${BUILD:-$(mktemp -d "${TMPDIR:-/tmp}/mold-collection-picker.XXXXXX")}
audit_runtime=${SIM_RUNTIME:-$(xcrun simctl list runtimes --json | python3 -c '
import json, sys
rows = [r for r in json.load(sys.stdin)["runtimes"] if r.get("isAvailable") and r["name"].startswith("iOS ") and int(r["version"].split(".")[0]) >= 26]
print(max(rows, key=lambda r: tuple(map(int, r["version"].split("."))))["identifier"])
')}
audit_device_type=${SIM_DEVICE_TYPE:-com.apple.CoreSimulator.SimDeviceType.iPhone-16}
audit_sim=$(xcrun simctl create "Mold collection picker audit $$" "$audit_device_type" "$audit_runtime")
cleanup() {
  xcrun simctl shutdown "$audit_sim" >/dev/null 2>&1 || true
  xcrun simctl delete "$audit_sim" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM
mkdir -p "$audit_build"
xcrun simctl boot "$audit_sim"
xcrun simctl bootstatus "$audit_sim" -b
make -C "$repo_root/apps/ios" gen
xcodebuild -project "$repo_root/apps/ios/MoldCompanion.xcodeproj" -scheme MoldCompanion \
  -configuration Debug -derivedDataPath "$audit_build/DerivedData" \
  -destination "platform=iOS Simulator,id=$audit_sim" \
  -only-testing:MoldCompanionUITests/LibraryCollectionPickerTests \
  build-for-testing > "$audit_build/build.log" 2>&1
for audit_appearance in light dark; do
  xcrun simctl ui "$audit_sim" appearance "$audit_appearance"
  xcodebuild -project "$repo_root/apps/ios/MoldCompanion.xcodeproj" -scheme MoldCompanion \
    -configuration Debug -derivedDataPath "$audit_build/DerivedData" \
    -destination "platform=iOS Simulator,id=$audit_sim" \
    -only-testing:MoldCompanionUITests/LibraryCollectionPickerTests \
    -resultBundlePath "$audit_build/$audit_appearance.xcresult" \
    test-without-building > "$audit_build/$audit_appearance.log" 2>&1
  echo "Library collection picker: $audit_appearance passed"
done
echo "Audit results: $audit_build"
