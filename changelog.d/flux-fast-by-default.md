- **FLUX.1 and FLUX.2 are fast by default on CUDA.** Both families now render
  through FlashAttention-2 wherever the kernel is compiled in and their VAE
  convolutions take cuDNN wherever that feature is compiled in. Every shipped
  qualified Linux CUDA build compiles both — the sm86, sm89 and sm100 release
  archives, the `mold`/`mold-sm86`/`mold-sm100` Nix packages, their container
  and AUR builds, and the matching Linux desktop packages. Before this, only
  the sm89 `h3-cuda` artifact carried the flash kernel, which left an RTX
  3090/A40 and a B200 with this release's byte change and none of its speedup:
  FLUX's math path folds the softmax scale into K whether or not the kernel is
  there. **`mold-sm120` (RTX 50-series) deliberately stays on math attention**
  — FlashAttention picks its tile from a runtime test that reads consumer
  Blackwell as a datacenter part with far more shared memory than it has, and
  nobody has measured the result on that hardware — so it takes the byte change
  without the speedup until someone does. A binary you build yourself with
  `--features cuda` and no `flash-attn`, and every Metal build, are in the same
  position; add `flash-attn` to a source build's feature list on sm86, sm89 or
  sm100.
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
- **An LTX-2 built-in control render applies its control adapter again.** The
  server resolves the built-in IC-LoRA itself, after durable admission has
  already sealed the request's media set, so the publication scrub took it with
  the caller's adapters and nothing could hand it back: `--control depth` and
  its siblings rendered over the server with no control adapter at all. The
  adapter now travels with the job and is restored at dispatch, ahead of the
  caller's own stack rather than instead of it.
- **A `--lora` render now requires the encrypted request-media store.** This is
  the other side of the fix above: because the adapter is sealed like any other
  request authority, a host whose durable media store is unavailable answers
  `503 DURABLE_MEDIA_UNAVAILABLE` for a LoRA render instead of quietly
  rendering without the adapter. Correct, and a visible behaviour change on a
  degraded store — the render is refused rather than silently wrong.
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
  changes meaning: `0` (also `off`, `false`, `no`) forces the old drop, and `1`
  (also `on`, `true`, `yes`) now means the same as the default, because an
  explicit keep has always had to yield to a card that cannot afford it. **Both
  families read it through one function** — FLUX.2 resolved the budget directly
  and never read the variable at all, so on a card whose budget said "keep"
  there was no way to say "don't". The execution fingerprint records the residency you ASKED
  for — `0` is its own execution class and unset and `1` share the other — so a
  forced drop is never filed with a budgeted render; the budget's own verdict is
  a per-render VRAM measurement and is reported in the server log rather than
  hashed into the plan. The GGUF activation width and the FLUX.2 CFG shape ARE
  the resolved answers, so two renders that differ in either are never filed as
  the same execution. That budget charges what the card is HOLDING rather than
  what the checkpoint weighs on disk, on both families: off CUDA a dense
  checkpoint is materialized at F32 whatever the file stores, so a BF16
  Klein-9B is ~36 GB of weights on a Mac and not its ~18 GB file, and a FLUX.2
  FP8 tier widened once at load holds two bytes per parameter where the file
  holds one — the same figure the server's own estimates charge. A GGUF keeps
  its quantized bytes and is unchanged.
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

- **A FLUX.2 LoRA render is no longer refused on a card big enough to load the
  model eagerly.** The preload gate read the engine's configured load strategy
  while the render itself is chosen by the request — a LoRA is merged into the
  transformer as it is built, so a LoRA request always takes the sequential
  path whatever the strategy says. On a large card the two disagreed and a
  `flux2-klein` + `--lora` render failed outright with "Flux.2 LoRA requests
  require a sequential engine load plan", both on the first attempt and on the
  retry. The gate now asks the same question the render asks and simply defers
  the preload, so the adapter is applied by the sequential load that follows.
  Plain FLUX.2 renders and the FLUX.1 LoRA path are unchanged.

- **One model's numerical failure no longer takes the whole GPU out of
  service.** A worker that failed three times in a row was marked degraded and
  stopped scheduling anything for 60 seconds — the right answer for a card that
  is wedged or faulting, and the wrong one for a checkpoint that produces a
  NaN. Three non-finite `flux2-dev:q8` renders on a single-GPU host left
  `/api/devices` reporting `health: "degraded"` and answered the next twelve
  requests, for other models, with "no enabled, healthy GPU device is
  available". Failures the engine reports as belonging to the model or the
  request now hold that **model on that GPU** instead, with the same
  three-strike, 60-second shape: the device stays healthy and schedulable,
  every other model keeps rendering on it, a multi-GPU host routes the held
  model to another card, the refusal names the model rather than the GPU, and
  a successful render clears the strikes. Driver faults, CUDA errors and
  out-of-memory still count against the device exactly as before, as does any
  failure the engine has not classified.
- **FLUX.2 renders are planned against the transformer that runs them, so a
  FLUX.2 [dev] job no longer runs out of GPU memory two minutes into the
  denoise** ([#1707](https://github.com/utensils/mold/issues/1707)). The memory
  a FLUX.2 denoise needs was estimated with FLUX.1's per-pixel model, which
  knows nothing about the transformer's width: 273 MB charged at 1024x1024 for
  a working set three quantizations independently measure at ~3.0 GB. On a
  46 GB L40S that let `flux2-dev:q8` be admitted at ~38 GB and die in CUDA
  partway through the denoise, while the same shape had completed nine times
  before. Every FLUX.2 tier is now priced from its own geometry, and a render
  that cannot fit is refused at submit time with a reason instead of after a
  two-minute load. The same shape with a reference image — the one that failed
  — is now refused up front.
- **A failed render no longer teaches the planner that its shape needs the
  whole card.** The learned memory envelope absorbed the high-water mark of
  attempts that ran OUT of memory, which is a measurement of the GPU, not of
  the job. After two failures, `flux2-dev:q8` was re-planned at ~46.5 GB on a
  ~46.1 GB card and every retry was refused with a figure that could never fit,
  until the row aged out. Failures are still recorded, and a shape that has
  never succeeded still learns a floor from them; a shape with completed runs
  keeps the evidence those runs produced.
- **Out-of-memory messages say what actually happened.** A plan that exceeds
  the GPU's own capacity now says so and names both figures, instead of
  reporting "memory pressure changed after scheduler admission" on a card
  nothing else was using. A FLUX.2 job that could not stream its transformer
  says why — GGUF tiers have no block-streaming path — rather than leaving it
  to be guessed.
- **The scheduler and the model loader now agree on how much VRAM is
  available.** Admission planned against the raw driver reading while every
  pre-load check subtracted the reserve set by `MOLD_RESERVE_VRAM_MB`, so a job
  could be admitted and then refused at load with nothing having changed. The
  reserve is now subtracted once, where the scheduler's capacity is computed.
- A malformed or truncated `.safetensors` file no longer takes the server down.
  Probing a FLUX.2 checkpoint's header trusted the length the file declared and
  allocated it, so a placeholder or a half-finished download could abort the
  process; an unreadable header is now handled the same way an unrecognised one
  always was.
- **A retained transformer no longer wedges the queue behind itself.** A FLUX.2
  [dev] engine holds ~34 GB of weights on a card with room for them, and
  admission was offered raw free VRAM: the IDENTICAL next request — the one
  that would have reused those weights without loading anything — was reported
  `queued generation is blocked on memory` once a second, forever, and so was a
  request for any other model or family. Two things were wrong. The cache's
  credit was clipped to the host's per-process VRAM attribution, which reads as
  zero wherever that query cannot see mold's own pid; the engines are now asked
  directly and their answer is a floor under that clip, never a term added to
  it. And a generation whose plan resolver refused every device reached the
  planner with no placement to compare, which the plan pass read as "not
  blocked" and used to erase the block that had just been recorded — taking the
  idle reclaim and the bounded refusal with it. Now the same-model repeat is
  admitted immediately and reuses the weights, a different model's request
  releases them at dispatch and keeps the other engine's prompt cache, and
  anything genuinely too large is refused with numbers instead of waiting.

- **FLUX.2 [dev] renders again with the text-encoder park on, which is the
  default.** Every `flux2-dev` tier — `:q8`, `:q6`, `:q4` and `:fp8` alike —
  failed with `non-finite prediction at denoise step 0` as soon as the
  Mistral3 prefix was held in page-locked host RAM, because the streamed
  encoder builds the next decoder layer on a second thread and nothing made
  its uploads complete before the forward read them. A pageable copy blocks
  the calling thread, so every path that existed before the park could fire
  was serialising the two threads by accident; page-locking the source turned
  that upload into a real asynchronous transfer and the conditioning tensor
  came back entirely NaN. The prefetch now settles its own work before the
  layer is handed on. A parked render and an unparked one are byte-identical
  again (verified at the same sha256 on an L40S, both `:q8` and `:fp8`), and
  `MOLD_KEEP_TE_RAM=0` is no longer a workaround anybody needs.
- **That park now waits until there is a second render to pay for it.**
  Reading 34.7 GB of shards into host RAM costs about 29 s and saves about
  6 s per encode afterwards, so it only breaks even around the sixth render of
  one process — and a one-shot `mold run` could never collect any of it. The
  first encode of a process now streams from the mapping as it always did, and
  the park is taken from the second onwards, when the reuse it is buying is
  real. Nothing about the rendered pixels changes either way.
- **The non-finite bail no longer gives advice that cannot apply.** It used to
  end every failure with "re-run with `MOLD_FLUX2_QMATMUL=0`", including on
  FLUX.1 and on `flux2-dev:fp8`, which carries no quantized matmul at all. The
  suggestion now appears only when that fast path is the arm actually running.

- **A LoRA now reaches the planner on a server render, so the GPU memory plan
  and the load strategy describe the render that actually runs.** A durable
  job's adapter is sealed into the encrypted media set and removed from the
  request before the job reaches the scheduler, and the execution plan is built
  from that copy — so every `--lora` render over the server was planned as if
  it had none: the adapter's bytes were never charged against the card, and
  FLUX.2 and Z-Image, which merge a LoRA as the transformer is built, were
  given an eager load plan only their sequential path can honour. The sealed
  stack now travels with the job for planning, so those renders are planned
  sequentially with the adapter counted. Local renders (`--local`) were never
  affected.

- **The FLUX.2 [dev] tier descriptions now name the card each one actually
  needs.** `flux2-dev:q4` was described as the tier that "runs on a 24 GB GPU"
  and `:q6` as fitting a 32 GB one, on the strength of the checkpoint size
  alone. A GGUF tier has no block-streaming path, so every byte of it is
  resident and the render also holds a ~3 GB denoise working set, the VAE and
  the planner's safety headroom: q4 needs ~25 GB (a 32 GB-class card), q6 ~33 GB
  (40 GB-class) and q8 ~40 GB (46/48 GB-class). On 24 GB the [dev] tiers that
  run are the safetensors ones — `flux2-dev:fp8` and `:bf16`, which stream
  their transformer blocks from host RAM at the documented 3-5x slowdown — and
  every Klein tier. `mold list`, the model page and the API all say so, and an
  oversized GGUF request is refused at submit time naming both figures.
