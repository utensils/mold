#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
release="$repo_root/.github/workflows/release.yml"
gradle="$repo_root/apps/mobile/src-tauri/gen/android/app/build.gradle.kts"
readme="$repo_root/README.md"
android_guide="$repo_root/website/guide/android.md"
installation_guide="$repo_root/website/guide/installation.md"

fail() {
  echo "android APK distribution contract: $*" >&2
  exit 1
}

grep -Fq 'build-android-apk:' "$release" || fail "release workflow does not build Android"
grep -Fq 'verify --verbose --print-certs' "$release" || fail "release APK signature is not verified"
grep -Fq 'artifacts/Mold-android.apk' "$release" || fail "raw APK is not attached to GitHub releases"
grep -Fq '*.apk > SHA256SUMS' "$release" || fail "APK is missing from release checksums"
grep -Fq 'ANDROID_KEY_BASE64' "$release" || fail "CI signing keystore secret is not configured"
grep -Fq 'signingConfig = signingConfigs.getByName("release")' "$gradle" || fail "Gradle release signing is not configured"

# The current stable release does not expose the APK, so user docs point to the
# live rolling release until a tagged APK is available there.
nightly='https://github.com/utensils/mold/releases/download/latest/Mold-android.apk'
for doc in "$readme" "$android_guide" "$installation_guide"; do
  grep -Fq "$nightly" "$doc" || fail "nightly raw APK link missing from ${doc#"$repo_root/"}"
done

echo "Android raw APK distribution contract is wired"
