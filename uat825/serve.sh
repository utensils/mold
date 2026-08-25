#!/usr/bin/env bash
# Start the #825 UAT scratch server. Never touches the production mold.service
# on :7680; it binds :7681 with its own MOLD_HOME and output dir and shares the
# read-only model store.
set -euo pipefail
root=/home/jamesbrink/Projects/utensils/mold/.claude/worktrees/agent-a81576867db752e34
export MOLD_HOME=/storage-fast/mold/uat-825-home
export MOLD_MODELS_DIR=/storage-fast/mold/models
export MOLD_PORT=7681
export MOLD_API_KEY=uat825
export MOLD_OUTPUT_DIR=/storage-fast/mold/uat-825/output
# The private runtime-bound capture logs its observation at INFO on its own
# target; that record is what the measured Ref2VA figures are transcribed from.
export MOLD_LOG=${MOLD_LOG:-info}
mkdir -p "$MOLD_HOME" "$MOLD_OUTPUT_DIR"
exec "$root/target/release/mold" serve --bind 0.0.0.0
