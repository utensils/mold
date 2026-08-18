#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORKFLOW="$ROOT/.github/workflows/testflight-ios.yml"

fail() {
  echo "ios-testflight-distribution: $*" >&2
  exit 1
}

grep -Fq '<key>method</key><string>app-store-connect</string>' "$WORKFLOW" ||
  fail "TestFlight export must target App Store Connect"

if grep -Fq '<key>testFlightInternalTestingOnly</key><true/>' "$WORKFLOW"; then
  fail "TestFlight builds must remain eligible for external testing"
fi

grep -Fq '<key>testFlightInternalTestingOnly</key><false/>' "$WORKFLOW" ||
  fail "TestFlight export must explicitly preserve external-testing eligibility"

grep -Fq 'path: ~/.cargo/bin/cargo-tauri' "$WORKFLOW" ||
  fail "TestFlight must cache the pinned Tauri CLI binary"

grep -Fq "if: steps.tauri-cli-cache.outputs.cache-hit != 'true'" "$WORKFLOW" ||
  fail "TestFlight must install Tauri CLI only on a cache miss"

grep -Fq "cargo tauri --version | grep -F '2.11.4'" "$WORKFLOW" ||
  fail "TestFlight must verify the cached Tauri CLI version"

echo "ios-testflight-distribution: ok"
