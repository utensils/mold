#!/usr/bin/env bash
set -euo pipefail
# Only the isolated instrumentation package is installed/revoked here.
ADB="${ADB:-adb}"
serial="${ANDROID_SERIAL:-emulator-5554}"
test_package=com.utensils.mold.mobile_native.test
[[ "$("$ADB" -s "$serial" shell getprop ro.kernel.qemu | tr -d '\r')" == 1 ]]
[[ "$("$ADB" -s "$serial" shell getprop ro.build.version.sdk | tr -d '\r')" == 28 ]]
"$ADB" -s "$serial" install -r apps/mobile/plugins/android/build/outputs/apk/androidTest/debug/tauri-plugin-mold-mobile-native-debug-androidTest.apk
"$ADB" -s "$serial" shell pm revoke "$test_package" android.permission.WRITE_EXTERNAL_STORAGE
"$ADB" -s "$serial" shell pm revoke "$test_package" android.permission.READ_EXTERNAL_STORAGE
output="${MOLD_ANDROID_EVIDENCE:-/tmp/mold-android-28}"
mkdir -p "$output"
"$ADB" -s "$serial" shell am instrument -w -r \
  -e class com.utensils.mold.mobile_native.AndroidMediaInstrumentedTest#refusesSaveWithoutGrantThenWritesPublicDownloads \
  "$test_package/androidx.test.runner.AndroidJUnitRunner" | tee "$output/instrumentation.txt"
# am instrument may exit zero for a failed test; inspect JUnit's actual result.
grep -Fq 'OK (1 test)' "$output/instrumentation.txt"
! grep -Eq 'FAILURES!!!|INSTRUMENTATION_FAILED|Process crashed' "$output/instrumentation.txt"
