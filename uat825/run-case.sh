#!/usr/bin/env bash
# One #825 UAT render. Usage: run-case.sh <case-id> <prompt> [--reference ...]
#
# The production mold.service on :7680 shares this card, so the case waits for
# it to be idle before submitting and never stops or reconfigures it.
set -euo pipefail
root=/home/jamesbrink/Projects/utensils/mold/.claude/worktrees/agent-a81576867db752e34
out=/storage-fast/mold/uat-825
case_id="$1"; shift
prompt="$1"; shift

export MOLD_HOST=http://localhost:7681
export MOLD_API_KEY=uat825

mkdir -p "$out/logs" "$out/output"

# Wait for the production server to go idle and the card to drain.
waited=0
while [ "$waited" -lt 5400 ]; do
  busy="$(curl -s http://localhost:7680/api/status | grep -o '"busy":[a-z]*' || echo '"busy":false')"
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
  if [ "$busy" = '"busy":false' ] && [ "$used" -lt 2000 ]; then
    break
  fi
  sleep 20
  waited=$((waited + 20))
done
echo "waited_for_gpu=${waited}s"

# Page cache steals from the host admission sample, and the sample is taken
# after this server's own ~37 GB artifact pass. Drop it first so the run is
# measured against real headroom rather than against the previous attempt's.
sudo -n sync || true
sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' || true

serverpid="$(pgrep -f "$root/target/release/mold serve" | head -1)"
: >"$out/logs/$case_id.vram"
: >"$out/logs/$case_id.rss"
(
  while kill -0 "$serverpid" 2>/dev/null; do
    date +%s | tr '\n' ' ' >>"$out/logs/$case_id.vram"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >>"$out/logs/$case_id.vram"
    grep VmHWM "/proc/$serverpid/status" >>"$out/logs/$case_id.rss" 2>/dev/null || true
    sleep 1
  done
) &
sampler=$!

start=$(date +%s.%N)
set +e
"$root/target/release/mold" run minimax-h3-ref2va:comfy-pruned-int8 "$prompt" \
  --width 1344 --height 768 --frames 124 --fps 24 \
  --steps 21 --guidance 0 --strength 1.0 --format mp4 \
  --output "$out/output/$case_id.mp4" \
  "$@" >"$out/logs/$case_id.stdout" 2>"$out/logs/$case_id.stderr"
rc=$?
set -e
end=$(date +%s.%N)
kill "$sampler" 2>/dev/null || true
wait "$sampler" 2>/dev/null || true

echo "case=$case_id rc=$rc wall=$(awk "BEGIN{printf \"%.1f\", $end - $start}")s"
echo "peak_vram_mib=$(awk '{print $2}' "$out/logs/$case_id.vram" | sort -n | tail -1)"
echo "peak_rss_kb=$(awk '{print $2}' "$out/logs/$case_id.rss" | sort -n | tail -1)"
tail -4 "$out/logs/$case_id.stderr" || true
