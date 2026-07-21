#!/usr/bin/env bash
# Simulator selection shared by scripts/ios.sh — sourceable and side-effect
# free so scripts/tests/ios-sim-pick.sh can exercise it with a stubbed xcrun.

# Preferred simulator: MOLD_IOS_DEVICE override, else iPhone 17 Pro, else the
# first available iPhone. Echoes the resolved device name (empty if none).
pick_device() {
  local want="${MOLD_IOS_DEVICE:-iPhone 17 Pro}"
  local sims
  sims="$(xcrun simctl list devices available)"
  # "$want (" keeps the match exact — "iPhone 17 Pro (" can't hit
  # "iPhone 17 Pro Max (…" because Max breaks the "name (" adjacency.
  if grep -qF "$want (" <<<"$sims"; then
    echo "$want"
    return
  fi
  grep -oE '^ +iPhone [^(]+' <<<"$sims" | head -1 | sed 's/^ *//;s/ *$//' || true
}

# UDID for a device name. Several runtimes can carry the same name
# (iPhone 17 Pro exists for iOS 26.0…26.5); the Tauri CLI resolves a
# bare name to the FIRST simctl match (runtimes list oldest-first), so
# take head -1 to boot the same simulator it will deploy to.
device_udid() {
  xcrun simctl list devices available | grep -F "$1 (" |
    grep -oE '[0-9A-F-]{36}' | head -1 || true
}
