#!/usr/bin/env bash
set -euo pipefail

HAL9000="hal9000.home.urandom.io"
REMOTE_USER="jamesbrink"
REMOTE_DIR="/home/jamesbrink/mold"
MOLD_PORT=7680

echo "🔨 Syncing source to hal9000..."
rsync -av --exclude target --exclude .git --exclude "*.png" \
  ./ "$REMOTE_USER@$HAL9000:$REMOTE_DIR/"

echo "🦀 Building on hal9000 (CUDA via nix develop)..."
ssh "$REMOTE_USER@$HAL9000" "
  cd $REMOTE_DIR
  nix develop --command cargo build --release -p mold-server --features cuda 2>&1
"

echo "🛑 Stopping existing mold-server (if running)..."
ssh "$REMOTE_USER@$HAL9000" "pkill -f mold-server || true"
sleep 1

echo "🚀 Starting mold-server on hal9000:$MOLD_PORT..."
ssh "$REMOTE_USER@$HAL9000" "
  cd $REMOTE_DIR
  MOLD_TRANSFORMER_PATH=/home/jamesbrink/AI/models/unet/flux1-dev.safetensors \
  MOLD_VAE_PATH=/home/jamesbrink/AI/models/vae/ae.safetensors \
  MOLD_T5_PATH=/home/jamesbrink/AI/models/text_encoders/t5xxl_fp16.safetensors \
  MOLD_CLIP_PATH=/home/jamesbrink/AI/models/clip/clip_l.safetensors \
  MOLD_T5_TOKENIZER_PATH=/home/jamesbrink/AI/models/tokenizers/t5-v1_1-xxl.tokenizer.json \
  MOLD_CLIP_TOKENIZER_PATH=/home/jamesbrink/AI/models/tokenizers/clip-vit-large-patch14.tokenizer.json \
  MOLD_PORT=$MOLD_PORT \
  nohup ./target/release/mold-server serve \
    --port $MOLD_PORT \
    > /tmp/mold-server.log 2>&1 &
  echo \$! > /tmp/mold-server.pid
  sleep 2
  if kill -0 \$(cat /tmp/mold-server.pid) 2>/dev/null; then
    echo \"✅ mold-server started (PID \$(cat /tmp/mold-server.pid))\"
  else
    echo \"❌ mold-server failed to start. Check /tmp/mold-server.log\"
    cat /tmp/mold-server.log
    exit 1
  fi
"

echo ""
echo "✅ Deployed! Test with:"
echo "   MOLD_HOST=http://hal9000.home.urandom.io:$MOLD_PORT mold generate \"a rusty robot on a beach\""
