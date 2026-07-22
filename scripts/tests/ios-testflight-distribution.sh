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

echo "ios-testflight-distribution: ok"
