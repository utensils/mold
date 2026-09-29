#!/usr/bin/env bash
# Native SwiftUI iOS entrypoint. Tauri's scripts/ios.sh remains separate.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
action="${1:-dev}"
shift || true
case "$action" in
  --help|-h|help)
    echo 'usage: scripts/companion.sh {dev|run|build|gen|test|uitest|packages-test|lint} [SIM=<UDID>] [BUILD=<directory>]'
    echo 'companion dev rebuilds and relaunches after native iOS or shared Swift source changes; Ctrl-C stops watching.'
    exit 0 ;;
  dev|run|build|gen|test|uitest|packages-test|lint) ;;
  *) echo "unknown companion action: $action (use --help)" >&2; exit 2 ;;
esac
if [[ "$(uname -s)" != Darwin ]]; then
  echo 'Native iOS development requires macOS and Xcode.' >&2
  exit 1
fi
if [[ "$action" == dev ]]; then
  exec python3 "$root/apps/ios/scripts/watch.py" "$@"
fi
exec make -C "$root/apps/ios" "$action" "$@"
