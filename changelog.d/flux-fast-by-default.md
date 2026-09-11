- **FLUX.1 and FLUX.2 are fast by default on CUDA.** Both families now render
  through FlashAttention-2 wherever the kernel is compiled in and their VAE
  convolutions take cuDNN wherever that feature is compiled in. Every shipped
  Linux CUDA build compiles both — the sm86, sm89, sm100 and sm120 release
  archives, the `mold`/`mold-sm86`/`mold-sm100`/`mold-sm120` Nix packages, the
  per-architecture container images, the AUR packages, and the Linux desktop
  builds. Before this, only the sm89 `h3-cuda` artifact carried the flash
  kernel, which would have left an RTX 3090/A40, an RTX 50-series card or a
  B200 with this release's byte change and none of its speedup: FLUX's math
  path folds the softmax scale into K whether or not the kernel is there. A
  binary you build yourself with `--features cuda` and no `flash-attn`, and
  every Metal build, are still in that position — correct, byte-changed, and
  on the math path — so add `flash-attn` to a source build's feature list.
  Every other still family — SD1.5, SDXL, SD3, Qwen-Image, Z-Image, LTX-Video,
  Hunyuan3D, MiniMax-H3 — keeps the byte-stable math/im2col defaults it has
  always had, unchanged in every build.
  `MOLD_ATTN=math` and `MOLD_CONV=im2col` remain the opt-outs and remain the
  cross-build determinism contract going forward.
- **A FLUX print archived before this release will not re-render byte-for-byte
  after it, under any setting.** Flash attention, cuDNN convolutions and the
  folded softmax scale all change reduction order, and restoring the old bytes
  would mean keeping the code path this release deletes. Renders made from here
  on are reproducible among themselves — same seed, same settings, same
  backend, same bytes — and the execution plan records which convolution
  backend and which attention kernel actually ran, so two renders that differ
  are never silently filed as the same execution. Dense BF16 FLUX.1 is the one
  exception: it attends through upstream Candle, which has no policy hook, so
  it stays on math.
- **FLUX.1 GGUF renders in BF16 on CUDA, through one transformer.** Every GGUF
  load — with or without a LoRA — now goes through mold's own transformer;
  the candle fork's quantized model, which the commonest no-LoRA render used
  to take, carried no attention-backend switch and F32 norm weights, so that
  render could reach neither FlashAttention nor half-precision activations
  however the binary was built. The two were verified bit-identical before the
  old path was deleted. Activations follow the working dtype instead of being
  pinned to F32, which halves the bandwidth every matmul moves; the weights
  stay quantized in VRAM exactly as before. `MOLD_WAN_FORCE_DMMV=1` still
  forces F32, because the fallback it selects reads activations as f32.
- **Flux.2 GGUF renders in BF16 and no longer scrubs NaN after every linear.**
  The GGUF transformer wrapped all eighteen of its linear sites in a full-tensor
  NaN compare, a zeros allocation and a `where_cond` — copied from SD3 without a
  Flux.2 NaN ever having been observed, and measured at about half a second per
  step. It is gone, and masking a non-finite value was the wrong shape anyway: a
  transformer emitting NaN has a bug, and zeroing the element turns a loud
  failure into a quietly wrong picture. `MOLD_FLUX_DEBUG_NONFINITE=1` replaces
  it with one check per denoise STEP that names the step and fails — off by
  default, and available for FLUX.1 too. Activations now follow the working
  dtype (BF16 on CUDA) instead of being cast to F32 at the transformer
  boundary; position ids stay F32, because the rotary embedding is built from
  them. `MOLD_FLUX2_QMATMUL=0` restores the per-forward dequantization arm if a
  render comes out wrong.
- **A `--lora` on a prompt-only render is no longer a silent no-op over the
  server.** Durable admission seals every request authority it later scrubs off
  the copy it hands the GPU worker, but the predicate deciding whether to seal
  anything at all counted only conditioning media — so a text-to-image render
  with an adapter sealed nothing, lost the adapter on its way to the worker,
  and produced pixels byte-identical to the same prompt with no LoRA while its
  print recorded none. Every LoRA-capable family was affected, FLUX.1 and
  FLUX.2 included; a LoRA beside an image, mask or video source always worked,
  and `--local` was never affected.
- **FLUX.2's FP8 tiers stop re-widening every weight on every forward.** An FP8
  layer cast its whole one-byte-per-parameter slab up to the working dtype on
  every call, so the tier chosen to save VRAM was paying full BF16 bandwidth
  for its weights and allocating a transient full-size copy hundreds of times a
  step. Where the card has the room the widening now happens ONCE at load and
  the packed slab is dropped, which is bit-for-bit the same arithmetic — the
  per-tensor scale still rides the matmul output, exactly where it did. The
  decision is a measured budget with the card's free VRAM on one side, so a
  24 GB card widens Klein-4B and leaves a 32 GB dev checkpoint alone;
  `MOLD_FLUX2_FP8_CACHE=1` or `=0` forces it either way.
- **An undistilled FLUX.2 [klein] base render guides in one forward per step
  instead of two.** Both branches denoise the same latent, so they ride one
  batch-2 forward and every weight is read once for the pair — Black Forest
  Labs' own sampler does this and mold was following diffusers, which does not.
  A guided base render is now much closer in cost to an unguided one rather
  than roughly double. mold falls back to the old two forwards when the doubled
  activations would not fit beside the weights on this card; the progress line
  names which shape ran — `one batched forward per step` or `two forwards per
  step` — and names no cause, because the budget is the only one a real render
  meets. (Both branches must also be the same length, but every Klein prompt is
  padded to a fixed 512 rows, so that gate survives only as a structural guard.)
  `--guidance 1` still skips the branch entirely.
- **FLUX renders spend far less time in norms, rotary embeddings and the VAE's
  attention.** Candle's fused normalization kernels were being missed
  everywhere in both families — the Q/K norms ran on a transposed view and the
  affine-less LayerNorms had no bias, and each miss cost about ten kernel
  launches and seven passes over the tensor instead of one. The rotary
  embedding now uses candle's fused interleaved kernel where the layout allows
  and falls back to the previous arithmetic where it does not. FLUX.2's double
  blocks issue one fused Q/K/V projection per stream rather than three, and the
  VAE's mid-block attention no longer materialises a full 16384x16384 score
  matrix during decode — the spike that used to push a loaded card into the
  much slower tiled-decode recovery.
- **Every FLUX.2 render's conditioning now matches Black Forest Labs.** Two
  things were wrong and both changed the picture. Text tokens were all given
  position zero, so the transformer could not tell the first word of a prompt
  from the last; they now carry a running index on their own axis, which is
  what BFL, diffusers and ComfyUI all do. And a FLUX.2 [klein] prompt was
  handed to the transformer at whatever length it happened to tokenize to,
  where upstream truncates and pads it to a fixed 512 rows and masks the
  padding out of the language model — mold's FLUX.2 [dev] path already did
  this, and now both tiers agree. **Klein and dev renders change**: the same
  seed and settings produce a different, better-conditioned picture than the
  same command did before this release. Prompt adherence improves most on long
  prompts, where the missing positions cost the most. A side effect is that
  every undistilled [klein] base render now takes the fast batched
  classifier-free-guidance path, since both branches are the same length by
  construction — the progress line reads "one batched forward per step".
- **FLUX.1 computes its rotary embedding in float32, as upstream does.** It
  previously built one in whatever dtype the render used, so a half-precision
  render computed every sine and cosine of every token position with eight bits
  of mantissa. FLUX.2 was already correct here.

- **The FLUX transformer stays on the card when it fits.** Both families used
  to drop it before every VAE decode and rebuild it on the next render,
  whatever the GPU had room for — 8.4 s per print for a FLUX.1 Q8 and 34 s for
  a FLUX.2 Q8 on an idle 46 GB card, and up to ~95 GB of host RAM for a LoRA
  rebuild. The decision is now a measurement taken per render: the resident
  checkpoint, the denoise workspace, the VAE decode workspace and an allocator
  margin against the card's usable free VRAM. FLUX renders are capped at 1.8
  megapixels (1328x1328 at the square), so across the whole range mold will
  actually render the answer comes down to the checkpoint and the card: a 24 GB
  card keeps a FLUX.1 Q8 tier (~12.6 GB) resident and never has room for the
  BF16 one (~23.8 GB), a 46 GB card keeps the BF16 tier as well, and for FLUX.2
  a 46 GB card keeps a 33 GB Q8 [dev] transformer where a 24 GB card never does
  though it does keep a Q8 Klein tier, 4B or 9B. The
  FLUX.2 sequential path — [dev], references, a LoRA, a source image — retains
  it across renders too, reusing it only when the LoRA stack, the working
  precision, the GPU and the resolved architecture all match, and releasing it
  before the text encoder streams whenever the two would not fit together. A
  prompt-cache hit runs no encoder at all, so repeated prompts and batches
  render with neither a reload nor an encode. `MOLD_FLUX_KEEP_TRANSFORMER`
  changes meaning: `0` forces the old drop, and `1` now means the same as the
  default, because an explicit keep has always had to yield to a card that
  cannot afford it. The execution fingerprint records the residency you ASKED
  for — `0` is its own execution class and unset and `1` share the other — so a
  forced drop is never filed with a budgeted render; the budget's own verdict is
  a per-render VRAM measurement and is reported in the server log rather than
  hashed into the plan. The GGUF activation width and the FLUX.2 CFG shape ARE
  the resolved answers, so two renders that differ in either are never filed as
  the same execution.
- **A FLUX identity render no longer fails at the first denoise step.** The
  eager `--id-image` path kept its own copy of the old rule that a quantized
  FLUX transformer runs its state tensors in F32. Once the GGUF path stopped
  pinning F32, the PuLID adapter met BF16 activations with an F32 weight and
  bias and every identity render died with "dtype mismatch in ternary op",
  while the same request rendered on the previous build. One function now
  answers what dtype a render's state tensors carry, and both the conditioning
  cast and the identity site ask it.
- **The desktop app's Speed & memory settings cover the new knobs.** Attention
  backend, convolution backend, FLUX transformer residency, the Flux.2
  quantized fast path, Flux.2 FP8 weight widening, PNG encoding and the
  graphics memory held back for the driver (`MOLD_RESERVE_VRAM_MB`) join live
  previews, text encoder parking, tiled VAE decode, block offloading and the
  queue window, and they say what their automatic setting actually does — the
  attention and convolution defaults are PER FAMILY, not one answer for every
  style. Parking text encoders becomes a three-way choice to match the engine.
  Each row still applies to this device's built-in engine and still needs an
  engine restart. A test now reads the Tauri side's allowlist and requires the
  two lists to be the SAME SET, so a control the app offers can never be one
  the engine never receives — and an engine knob can no longer sit copied but
  unoffered, which is exactly how the memory reserve went missing.

- **A large BF16 FLUX.1 checkpoint no longer streams its blocks on a card that
  can hold it.** The auto-offload decision was a file-size test with no
  availability arm, so a 23.8 GB `:bf16` tier paid the documented 3-5x
  streaming penalty on a 46 GB GPU with room for the whole thing. It now asks
  the same two-step question its FLUX.2 sibling already asked.
- **An XLabs-format FLUX.1 LoRA now fails loudly instead of rendering without
  the adapter.** mold's key matcher accepts the diffusers/PEFT
  (`lora_A`/`lora_B`), Kohya (`lora_down`/`lora_up`), OneTrainer and
  PEFT-default conventions, and has never accepted XLabs-AI's
  `double_blocks.N.processor.*_lora*.{down,up}.weight` layout. Such an adapter
  used to be discarded before it reached the parser, so the render succeeded
  with no LoRA applied at any scale; now it reaches the parser and the request
  fails, naming the layout it saw and pointing at the exports that do load.
  This is a behaviour change for anyone who was unknowingly rendering without
  their adapter — use a diffusers/PEFT or Kohya export of the same LoRA.

