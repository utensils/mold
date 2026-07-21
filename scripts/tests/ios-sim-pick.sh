#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
lib="$repo_root/scripts/ios-sim.sh"

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

# Stub xcrun on PATH so pick_device/device_udid read a fixture instead of
# talking to CoreSimulator — the test must run on Linux CI too.
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
cat > "$workdir/xcrun" <<'STUB'
#!/usr/bin/env bash
cat "$XCRUN_FIXTURE"
STUB
chmod +x "$workdir/xcrun"
export PATH="$workdir:$PATH"

# shellcheck source=../ios-sim.sh
source "$lib"

# Typical machine: duplicate names across runtimes, Pro Max alongside Pro.
fixture_full="$workdir/full.txt"
cat > "$fixture_full" <<'EOF'
== Devices ==
-- iOS 26.0 --
    iPhone 16e (54C607C1-54BF-43A5-BAA0-F60CC29EE92E) (Shutdown)
    iPhone 17 Pro (89848BF1-F258-4AD0-AD24-9A9F02151419) (Shutdown)
    iPhone 17 Pro Max (11292F6F-4EDC-4961-9F89-66D02AF2D2B0) (Shutdown)
-- iOS 26.5 --
    iPhone 17 Pro (D947477F-BBD3-49AD-BDEF-5BF6D42AE851) (Shutdown)
EOF

# Only Pro Max — the preferred "iPhone 17 Pro" must not substring-match it.
fixture_max_only="$workdir/max-only.txt"
cat > "$fixture_max_only" <<'EOF'
== Devices ==
-- iOS 26.0 --
    iPhone 17 Pro Max (11292F6F-4EDC-4961-9F89-66D02AF2D2B0) (Shutdown)
EOF

# No iPhones at all (e.g. only watch/TV runtimes installed).
fixture_empty="$workdir/empty.txt"
cat > "$fixture_empty" <<'EOF'
== Devices ==
-- watchOS 26.0 --
    Apple Watch Ultra 3 (49mm) (0AA00000-0000-0000-0000-000000000AA0) (Shutdown)
EOF

export XCRUN_FIXTURE="$fixture_full"
[[ "$(pick_device)" == "iPhone 17 Pro" ]] \
  || fail "preferred simulator not picked from full fixture"
# Tauri resolves duplicate names to the FIRST simctl match (oldest runtime),
# so device_udid must return the iOS 26.0 UDID, not the 26.5 one.
[[ "$(device_udid "iPhone 17 Pro")" == "89848BF1-F258-4AD0-AD24-9A9F02151419" ]] \
  || fail "device_udid did not take the first (oldest-runtime) match"
[[ "$(MOLD_IOS_DEVICE="iPhone 16e" pick_device)" == "iPhone 16e" ]] \
  || fail "MOLD_IOS_DEVICE override was ignored"
# An override that doesn't exist falls back to the first available iPhone.
[[ "$(MOLD_IOS_DEVICE="iPhone 3G" pick_device)" == "iPhone 16e" ]] \
  || fail "missing override did not fall back to the first iPhone"

export XCRUN_FIXTURE="$fixture_max_only"
[[ "$(pick_device)" == "iPhone 17 Pro Max" ]] \
  || fail "Pro-Max-only fixture should fall back to the only iPhone"

export XCRUN_FIXTURE="$fixture_empty"
[[ -z "$(pick_device)" ]] \
  || fail "no-iPhone fixture should yield an empty pick"
[[ -z "$(device_udid "iPhone 17 Pro")" ]] \
  || fail "no-iPhone fixture should yield an empty udid"

echo "PASS: ios-sim-pick"
