#!/usr/bin/env bash
set -euo pipefail

# ── RunPod entrypoint for mold inference server ──
#
# RunPod convention: start services, then sleep infinity to keep the
# container alive. We run mold serve in the background and monitor it.

# If a network volume is mounted, use it for model storage and config
if [ -d "/workspace" ]; then
    export MOLD_HOME="${MOLD_HOME:-/workspace/.mold}"
    export MOLD_MODELS_DIR="${MOLD_MODELS_DIR:-/workspace/.mold/models}"
    # HuggingFace cache (shared with other tools on the volume)
    export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
fi

# Ensure directories exist
mkdir -p "${MOLD_HOME:-/root/.mold}" "${MOLD_MODELS_DIR:-/root/.mold/models}"

# Export environment for SSH sessions (RunPod convention)
env | grep -E '^(MOLD_|HF_|CUDA|LD_LIBRARY)' > /etc/rp_environment 2>/dev/null || true

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  mold inference server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MOLD_HOME:       ${MOLD_HOME:-/root/.mold}"
echo "  MOLD_MODELS_DIR: ${MOLD_MODELS_DIR:-/root/.mold/models}"
echo "  MOLD_PORT:       ${MOLD_PORT:-7680}"
echo "  MOLD_LOG:        ${MOLD_LOG:-info}"
echo "  GPU:             $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not detected')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Run pre_start hook if it exists (RunPod convention)
if [ -f /pre_start.sh ]; then
    echo "[mold] running /pre_start.sh ..."
    bash /pre_start.sh
fi

# Start mold server
exec mold serve \
    --bind 0.0.0.0 \
    --port "${MOLD_PORT:-7680}"
