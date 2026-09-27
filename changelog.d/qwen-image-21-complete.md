- **Qwen Image 2.1: reference images, transparent backgrounds, LoRA, turbo,
  quantized tiers and native 2K.** Up to 10 ordered reference images (PNG,
  JPEG or WebP, never flattened); with no size chosen, the canvas takes the
  last reference's aspect ratio. A new **Transparent background** toggle
  (`transparent_background`, `mold run --transparent`, Discord `/transparent`,
  advertised as `capabilities.transparency`) applies the model card's RGBA
  prompt recipe inside the engine — the stored prompt stays yours — and keeps
  native alpha in PNG and WebP; JPEG is refused while it is on. A render keeps
  an alpha channel only when the toggle is on or a reference is itself
  transparent, so ordinary renders stay RGB and byte-identical. Prints record
  `has_alpha` and `transparent_background`. LoRA adapters now work on every
  tier (Qwen-Image / 2512 adapters are refused by name). New tiers:
  `int8-conv`, `fp8`, and GGUF `q8`–`q2`, plus the Viggle 6-step
  `qwen-image-2.1-turbo` (`bf16`, `int8-conv`, `q8`) with a pinned recipe. The
  Qwen3-VL text encoder can run from the official Q8_0 GGUF (auto) or Q4_K_M
  (`MOLD_QWEN3_VARIANT=q4`, explicit only). The model card's seven native 2K
  sizes (up to 2400x1792 and 2752x1536) and six ~1 MP aspect presets are
  offered. See the new [Qwen Image 2.1](https://utensils.io/mold/models/qwen-image-21)
  page.
- **Qwen Image 2.1 is 2.5x faster on CUDA at 1024² and over 6x faster at 2K.**
  It now takes the FastStill policy like FLUX: FlashAttention, fused
  projection/RoPE, compact modulation, fused adaLN, and a cuDNN VAE. On an
  L40S (BF16, 40 steps) 1024² went from 38.9 s to 15.2 s, 1344x768 with
  guidance 4 from 76.7 s to 31.0 s, 2048² from 525 s to 82.6 s, and 2752x1536
  from 537 s to 85.9 s; `int8-conv` runs at about the BF16 speed on half the
  memory. Pixels change for the same seed; `MOLD_ATTN=math MOLD_CONV=im2col`
  on the server restores v0.32's bytes exactly. `MOLD_QWEN_IMAGE21_KV_CACHE`
  (`auto`/`on`/`off`) controls the per-branch prefix cache.
- **Qwen Image 2.1 downloads now require accepting the Qwen Research License.**
  Every tier, turbo included, is non-commercial; earlier releases downloaded
  the weights without asking. Accept once in the apps' licence dialog or with
  `mold licenses accept qwen-research`.
- **WebP still output works for every image model.** WebP was advertised for
  stills but failed after the render; stills now encode as real single-frame
  WebP (lossy colour, lossless alpha). `mold run --format webp -o x.png` is
  reported as the still mismatch it is instead of naming APNG. Transparent
  images are no longer hidden from the Library as "solid black".
- **Transparent prints show a checkerboard** in the Library, lightbox, result
  canvas and recent prints on web, desktop and mobile. MCP `generate_image`
  gains `reference_images`, `transparent_background` and `webp`, and Discord
  routes `reference_1`/`reference_2` by the model's reference capability.
- **Fixes found along the way.** The web Create page offered the "Add-on
  looks" (LoRA) row for models that take no LoRA. `nix develop` failed on
  Linux with a CUDA `LICENSE` collision. Qwen Image 2.1 could never park its
  text encoder in host RAM on Linux (it read macOS-only memory probes), so a
  2K render dropped the transformer instead. INT8 ConvRot renders are now charged their
  quantized-activation workspace at admission, so a card that cannot hold it
  is refused up front rather than running out of memory mid-render.
