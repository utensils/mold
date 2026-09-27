- **Qwen Image 2.1: reference images, transparent backgrounds, LoRA, turbo,
  quantized tiers and native 2K.** Up to 10 ordered reference images (PNG,
  JPEG or WebP, never flattened). With no size chosen, the canvas takes the
  last reference's upright aspect ratio (EXIF orientation applied) at
  upstream's 1024² area, kept inside the model's size limits. A new
  **Transparent background** toggle (`transparent_background`,
  `mold run --transparent`, Discord `/transparent`, advertised as
  `capabilities.transparency`) applies the model card's RGBA prompt recipe
  inside the engine, so the stored prompt stays yours. It keeps native alpha
  in PNG and WebP, and JPEG is refused while it is on. A render keeps an alpha
  channel only when the toggle is on or a reference has transparent pixels,
  so ordinary renders stay RGB and byte-identical. Prints record `has_alpha`,
  `transparent_background` and `prefix_cache` (`retained` or `recomputed`,
  since the two are not bit-identical). LoRA adapters now work on every tier,
  and Qwen-Image / 2512 adapters are refused by name. New tiers: `int8-conv`,
  `fp8` (CUDA only, refused on Metal before it downloads), and GGUF
  `q8`–`q2`. The Viggle 6-step `qwen-image-2.1-turbo` (`bf16`, `int8-conv`,
  `q8`) has a pinned recipe; a negative prompt sent to it, or to any other
  recipe that hides the control, now draws a request warning instead of being
  dropped silently. The Qwen3-VL text encoder can run from the official Q8_0
  GGUF (`auto` picks it for a text-only prompt when BF16 does not fit; a
  reference prompt keeps BF16, on the CPU if need be, because Q8_0 loses
  precision on the image rows) or Q4_K_M (`MOLD_QWEN3_VARIANT=q4`, explicit
  only). The model
  card's seven native 2K sizes (up to 2400x1792 and 2752x1536) and six ~1 MP
  aspect presets are offered. See the new
  [Qwen Image 2.1](https://utensils.io/mold/models/qwen-image-21) page.
- **Qwen Image 2.1 is 2.5x faster on CUDA at 1024² and over 6x faster at 2K.**
  It now takes the FastStill policy like FLUX: FlashAttention, fused
  projection/RoPE, compact modulation, fused adaLN, and a cuDNN VAE. On an
  L40S (BF16, 40 steps) 1024² went from 38.9 s to 15.2 s, 1344x768 with
  guidance 4 from 76.7 s to 31.0 s, 2048² from 525 s to 82.6 s, and 2752x1536
  from 537 s to 85.9 s. `int8-conv` runs at about the BF16 speed (15.3 s) on
  half the transformer memory. Each guidance branch keeps its prefix cache
  whenever the card has room, so three references with guidance 4 went from
  167.4 s to 44.2 s. Pixels change for the same seed;
  `MOLD_ATTN=math MOLD_CONV=im2col` on the server restores v0.32's bytes
  exactly. `MOLD_QWEN_IMAGE21_KV_CACHE` (`auto`/`on`/`off`) controls the
  prefix cache.
- **Qwen Image 2.1 downloads now require accepting the Qwen Research License.**
  Every tier, turbo included, is non-commercial; earlier releases downloaded
  the weights without asking. Accept once in the apps' licence dialog or with
  `mold licenses accept qwen-research`. Re-pulling an installed tier is a
  no-op that does not ask, and pulling a turbo tier no longer leaves an empty
  model directory behind.
- **WebP still output works for every image model.** WebP was advertised for
  stills but failed after the render; stills now encode as real single-frame
  WebP (lossy colour, lossless alpha). `mold run --format webp -o x.png` is
  reported as the still mismatch it is instead of naming APNG. Transparent
  images are no longer hidden from the Library as "solid black", and an older
  animated WebP no longer reads as a still in `mold library show --preview`.
- **Transparent prints show a checkerboard** in the Library, lightbox, result
  canvas and recent prints on web, desktop and mobile. Every reference strip
  on web, desktop and phone now draws one numbered thumbnail per picture
  ("Image 1", "Image 2", as the prompt addresses them) with remove, reorder
  (drag with a mouse or pen, or the keyboard-reachable ‹ › buttons) and
  a "Sets canvas" mark on Qwen Image 2.1's last reference. MCP
  `generate_image` gains `reference_images`, `transparent_background` and
  `webp`, and Discord routes `reference_1`/`reference_2` by the model's
  reference capability. `--transparent` and MCP's `transparent_background`
  are refused with "update the server" against an older server, which would
  otherwise render opaque without a word.
- **A request no GPU can ever hold is refused at once instead of waiting
  forever.** When a render needs more memory than the largest eligible card
  could ever admit, the refusal names what it needs and what the card has,
  instead of leaving the job queued. Qwen Image 2.1's memory admission was
  corrected in the same pass. Its sequential plan is priced by its largest
  phase, VAE decode included, so `int8-conv` with a reference on a 24 GB card
  now renders rather than queueing for memory it could never get. A 2K
  sequential render is charged its decode peak. LoRA and turbo adapter bytes
  are charged in every phase, and a queued job prices all its references.
  On a 46–48 GB card the text encoder stays resident for one to three
  references (one or two with guidance and a negative prompt), saving about
  12 s a request.
- **Reference images are bounded and read as upstream reads them.** A
  reference over 16,384 px a side, 100 megapixels or a 200:1 aspect ratio is
  refused when the request is submitted, naming its position; before, a tiny
  PNG declaring a huge size could reach a multi-gigabyte decode. A 16-bit PNG
  is reduced to 8 bits as Pillow does (high byte), for Qwen Image 2.1
  references, Hunyuan3D source images and views, and MiniMax H3 endpoint and
  reference images alike, so a nearly opaque 16-bit alpha is no longer read
  as partly transparent. Qwen Image 2.1 references
  are also EXIF-oriented and converted from their ICC profile to sRGB.
- **MiniMax H3's vision mergers use exact erf GELU,** as every upstream does,
  instead of the tanh approximation. Image-conditioned H3 renders shift
  slightly for the same seed.
- **Fixes found along the way.** A single `/api/generate` refusal no longer
  carries a `requests[1]:` prefix (a real batch still names its row).
  Stopping the server during the startup artifact warm no longer hangs. The
  MCP server answers `initialize` with the client's protocol version when it
  supports it. The web Create page offered the "Add-on looks" (LoRA) row for
  models that take no LoRA. `nix develop` failed on Linux with a CUDA
  `LICENSE` collision. Qwen Image 2.1 could never park its text encoder in
  host RAM on Linux (it read macOS-only memory probes), so a 2K render dropped
  the transformer instead. A host-placed text encoder no longer reports
  "No GPU detected". INT8 ConvRot renders are now charged their
  quantized-activation workspace at admission, so a card that cannot hold it
  is refused up front rather than running out of memory mid-render.
