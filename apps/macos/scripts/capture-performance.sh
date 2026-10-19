#!/bin/bash
# Read-only capture. Never restarts the app or sends process-control signals.
set -euo pipefail
umask 077
if [[ "${1:-}" == --help || $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 PID [seconds: 1-60, default 10]"
  if [[ "${1:-}" == --help ]]; then exit 0; fi
  exit 1
fi
pid=$1
seconds=${2:-10}
if [[ ! "$pid" =~ ^[0-9]+$ || ! "$seconds" =~ ^[0-9]+$ ]] || (( ${#seconds} > 2 )) || (( 10#$seconds < 1 || 10#$seconds > 60 )); then
  echo "Expected a numeric PID and a sample duration from 1 to 60 seconds." >&2
  exit 1
fi
process=$(/bin/ps -p "$pid" -o comm=)
if [[ "$process" != */"Mold Studio.app/Contents/MacOS/Mold" ]]; then
  echo "PID must identify the running Mold Studio app." >&2
  exit 1
fi
report=$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/mold-performance.XXXXXX")
echo "Collecting ${seconds}s sample for Mold Studio; app remains running."
/bin/ps -p "$pid" -o pid=,%cpu=,%mem=,rss=,etime= > "$report/process-before.txt"
/usr/bin/sample "$pid" "$((10#$seconds))" -file "$report/stacks.txt" > "$report/sample-status.txt" 2>&1
/bin/ps -p "$pid" -o pid=,%cpu=,%mem=,rss=,etime= > "$report/process-after.txt" || true
/usr/bin/log show --last 10m --style compact --info \
  --predicate "processIdentifier == $pid AND subsystem == 'io.mold.native' AND category == 'Performance'" \
  > "$report/performance.log" 2> "$report/log-status.txt" || true
cat > "$report/README.txt" <<'NOTE'
Private local diagnostic capture; nothing was uploaded.
process-*.txt: PID, CPU %, memory %, resident KiB, elapsed process time.
performance.log: bounded native operation duration and responsiveness events.
stacks.txt: sampled process stacks; may contain local usernames and paths.
Review stacks.txt before sharing. Logs can be empty on older builds, stable
builds without diagnostics enabled, or if unified-log access is unavailable.
NOTE
echo "Private report: $report"
