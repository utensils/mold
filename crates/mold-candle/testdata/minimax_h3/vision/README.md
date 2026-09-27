# MiniMax H3 vision-tower oracle

`capture.py` runs transformers' `Qwen3VLVisionModel` (float32, CPU) on the
`visual.*` tensors of the installed H3 conditioner and writes the processor's
`pixel_values` / `image_grid_thw` plus the merger and the three DeepStack maps
for a deterministic 320x256 synthetic image. The capture is not committed; it
lives at `/storage/mold/fixtures/minimax_h3/vision_fp32.safetensors` on plato.

| File                      | SHA-256                                                            |
| ------------------------- | ------------------------------------------------------------------ |
| `vision_fp32.safetensors` | `3c5aa93a5846d07c94fbb8a106fa9d3a1220d29f230e6ebfc3fd8788aad1dae6` |

Captured with torch 2.11.0+cu128 and transformers 5.17.0 (recorded in the
file's `__metadata__`) from
`shared/minimax-h3/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
(the vision tower is stored BF16 in that file) and
`shared/minimax-h3/text_encoder/config.json`.

```bash
python capture.py \
  $MOLD_HOME/models/shared/minimax-h3/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  $MOLD_HOME/models/shared/minimax-h3/text_encoder/config.json \
  $MOLD_HOME/models/shared/minimax-h3/processor \
  vision_fp32.safetensors

MOLD_TEST_H3_VISION_CAPTURE=vision_fp32.safetensors \
MOLD_TEST_H3_SHARED_DIR=$MOLD_HOME/models/shared/minimax-h3 \
  cargo test --release -p mold-ai-candle --lib released_h3_tower -- --nocapture
```

The test (`minimax_h3/vision.rs`) is a no-op when either variable is unset.
