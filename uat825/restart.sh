#!/usr/bin/env bash
# Restart the #825 UAT scratch server with a wiped MOLD_HOME. Deliberately
# matches only this worktree's own release binary, so the production
# mold.service on :7680 (a different path) is never touched.
set -euo pipefail
root=/home/jamesbrink/Projects/utensils/mold/.claude/worktrees/agent-a81576867db752e34
pkill -f "$root/target/release/mold serve" || true
sleep 4
rm -rf /storage-fast/mold/uat-825-home
mkdir -p /storage-fast/mold/uat-825-home
nohup bash "$root/uat825/serve.sh" >/storage-fast/mold/uat-825/logs/server.log 2>&1 &
sleep 10
curl -s -H "x-api-key: uat825" http://localhost:7681/api/status | head -c 100
echo
