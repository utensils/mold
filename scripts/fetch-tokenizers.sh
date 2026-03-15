#!/usr/bin/env bash
# Download tokenizer files required by mold.
# Run this on the inference host once.
set -euo pipefail

TOKENIZER_DIR="${1:-${MOLD_MODELS_DIR:-$HOME/.mold/models}/tokenizers}"
mkdir -p "$TOKENIZER_DIR"

echo "Downloading T5 tokenizer..."
curl -fSL -o "$TOKENIZER_DIR/t5-v1_1-xxl.tokenizer.json" \
  "https://huggingface.co/lmz/mt5-tokenizers/resolve/main/t5-v1_1-xxl.tokenizer.json"

echo "Downloading CLIP tokenizer..."
curl -fSL -o "$TOKENIZER_DIR/clip-vit-large-patch14.tokenizer.json" \
  "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/tokenizer.json"

echo "Done. Tokenizers saved to $TOKENIZER_DIR"
