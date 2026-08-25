#!/usr/bin/env bash
# One #825 UAT render. Usage: run-case.sh <case-id> <prompt> [--reference ...]
set -euo pipefail
root=/home/jamesbrink/Projects/utensils/mold/.claude/worktrees/agent-a81576867db752e34
out=/storage-fast/mold/uat-825
case_id="$1"; shift
prompt="$1"; shift

export MOLD_HOST=http://localhost:7681
export MOLD_API_KEY=uat825

mkdir -p "$out/logs" "$out/output"

# 1 Hz VRAM sampler and 1 Hz server RSS sampler for the whole render.
serverpid="$(pgrep -f 'target/release/mold serve' | head -1)"
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
echo "peak_rss=$(awk '{print $2}' "$out/logs/$case_id.rss" | sort -n | tail -1) kB"
tail -3 "$out/logs/$case_id.stderr" || true
