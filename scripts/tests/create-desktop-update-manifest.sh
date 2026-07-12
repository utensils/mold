#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
script="$repo_root/scripts/create-desktop-update-manifest.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

artifact="$tmp/Mold_0.16.1-nightly.381_aarch64.app.tar.gz"
signature="$artifact.sig"
output="$tmp/mold-desktop-nightly.json"
printf 'signed updater payload\n' > "$artifact"
printf 'dGVzdC1zaWduYXR1cmU=\n' > "$signature"

"$script" \
  --channel nightly \
  --version 0.16.1-nightly.381 \
  --artifact "$artifact" \
  --signature "$signature" \
  --repository utensils/mold \
  --release-tag latest \
  --output "$output" \
  --pub-date 2026-07-12T19:00:00Z \
  --notes "Nightly from main abc1234"

jq -e '
  .version == "0.16.1-nightly.381" and
  .notes == "Nightly from main abc1234" and
  .pub_date == "2026-07-12T19:00:00Z" and
  (.platforms | keys == ["darwin-aarch64"]) and
  .platforms["darwin-aarch64"].url == "https://github.com/utensils/mold/releases/download/latest/Mold_0.16.1-nightly.381_aarch64.app.tar.gz" and
  .platforms["darwin-aarch64"].signature == "dGVzdC1zaWduYXR1cmU="
' "$output" > /dev/null || fail "nightly manifest content is incorrect"

stable_artifact="$tmp/Mold_0.16.0_aarch64.app.tar.gz"
stable_signature="$stable_artifact.sig"
stable_output="$tmp/mold-desktop-stable.json"
printf 'stable updater payload\n' > "$stable_artifact"
printf 'c3RhYmxlLXNpZw==' > "$stable_signature"

"$script" \
  --channel stable \
  --version 0.16.0 \
  --artifact "$stable_artifact" \
  --signature "$stable_signature" \
  --repository utensils/mold \
  --release-tag v0.16.0 \
  --output "$stable_output" \
  --pub-date 2026-07-12T19:00:00Z

jq -e '
  .version == "0.16.0" and
  .notes == "Mold desktop 0.16.0." and
  .platforms["darwin-aarch64"].url == "https://github.com/utensils/mold/releases/download/v0.16.0/Mold_0.16.0_aarch64.app.tar.gz" and
  .platforms["darwin-aarch64"].signature == "c3RhYmxlLXNpZw=="
' "$stable_output" > /dev/null || fail "stable manifest content is incorrect"

expect_failure() {
  local description="$1"
  shift
  if "$script" "$@" > /dev/null 2>&1; then
    fail "$description unexpectedly succeeded"
  fi
}

common_args=(
  --version 0.16.0
  --artifact "$stable_artifact"
  --signature "$stable_signature"
  --repository utensils/mold
  --pub-date 2026-07-12T19:00:00Z
)

expect_failure "unknown channel" \
  --channel edge "${common_args[@]}" \
  --release-tag latest --output "$tmp/mold-desktop-edge.json"

expect_failure "nightly manifest on a non-latest tag" \
  --channel nightly "${common_args[@]}" \
  --release-tag nightly --output "$tmp/mold-desktop-nightly.json"

expect_failure "stable manifest with a mismatched version tag" \
  --channel stable "${common_args[@]}" \
  --release-tag v0.15.0 --output "$tmp/mold-desktop-stable.json"

expect_failure "invalid SemVer" \
  --channel stable --version 0.16 \
  --artifact "$stable_artifact" --signature "$stable_signature" \
  --repository utensils/mold --release-tag v0.16 \
  --output "$tmp/mold-desktop-stable.json"

empty_signature="$tmp/empty.sig"
: > "$empty_signature"
expect_failure "empty signature" \
  --channel stable --version 0.16.0 \
  --artifact "$stable_artifact" --signature "$empty_signature" \
  --repository utensils/mold --release-tag v0.16.0 \
  --output "$tmp/mold-desktop-stable.json"

expect_failure "wrong output name" \
  --channel stable "${common_args[@]}" \
  --release-tag v0.16.0 --output "$tmp/latest.json"

mismatched_artifact="$tmp/Mold_0.16.9_aarch64.app.tar.gz"
printf 'wrong version payload\n' > "$mismatched_artifact"
printf 'd3JvbmctdmVyc2lvbi1zaWc=' > "$mismatched_artifact.sig"
expect_failure "artifact name with a mismatched version" \
  --channel stable --version 0.16.0 \
  --artifact "$mismatched_artifact" --signature "$mismatched_artifact.sig" \
  --repository utensils/mold --release-tag v0.16.0 \
  --output "$tmp/mold-desktop-stable.json"

echo "PASS: create-desktop-update-manifest"
