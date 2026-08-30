- **Video families now default to FlashAttention.** Wan and LTX-2 resolve their
  attention backend through a new per-family policy: with the FlashAttention v2
  kernel compiled in, an unset `MOLD_ATTN` reaches a video DiT as `flash`
  instead of the hand-rolled math path. Measured **2.1x** on the Wan DiT
  (158.4 s to 75.3 s, `wan22-t2v-a14b:q5` 53 frames at 832x480 on an RTX 4090).
  Image families are unchanged and still default to `math`, so an archived
  still renders the same bytes it always did; `MOLD_ATTN=math` restores the old
  arithmetic everywhere.
- **Wan step caching is on by default.** `MOLD_WAN_STEP_CACHE` now defaults to
  `auto` rather than `off`. First-block residual reuse was already measured at
  **1.85x** on the non-distilled tiers (605.6 s to 327.4 s, `wan22-t2v-a14b:q8`
  33 frames at 832x480, 20 steps) and already refuses itself on distilled
  adapters and schedules under 12 steps, so the default only engages where it
  was qualified. `MOLD_WAN_STEP_CACHE=off` disables it.
- **Wan admission and block offload now price the attention backend that
  actually runs.** The activation model charged a math-attention score matrix
  — 44% of the per-token budget at A14B — on every render, including one using
  FlashAttention, which materializes no such tile. Since the engine's
  block-offload policy reads the same estimate, an 81-frame 832x480 render was
  parking all 40 transformer blocks against a shortfall it did not have. The
  math and flash calibrations are fitted separately, because the residual the
  slope stands in for does not shrink when attention stops writing its tile.
- **A Wan render that repeats a prompt no longer re-encodes it.** The engine
  keeps the ~4 MB prompt encoding it produced and reuses it when the prompt,
  negative, CFG arm, encoder weights, device, and dtype all match, skipping both
  the 11.37 GB UMT5-XXL load and the forward. That covers an auto-chained long
  video, a re-roll, and a batch child; an authored sequence gives every stage
  its own prompt and still pays one encoder load per stage. This caches the
  encoder's output, not the encoder, which is still dropped after use.
- **A chain stage that runs out of GPU memory now says so.** Chain stages
  bypassed every piece of CUDA error handling ordinary generations have, and
  surfaced a bare `DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")` with
  no shape advice, no device synchronize, and no reduced grant for the next
  attempt. They now classify the failure exactly as `process_job` does.
- **Chain stages are charged the host memory they actually need.** A stage
  whose worker already holds the engine was charged the full cold-load host
  increment, so a long sequence could be refused for host RAM partway through
  on a machine with plenty free. It now takes the warm-resident credit every
  ordinary generation already took, through the same accessor — which also
  fixes a Metal double-count, since the raw increment was being charged beside
  the unified device gate.
- **Cancelling a render no longer makes its shape look too big for the card.**
  A stopped generation or chain stage was recorded as a memory *failure*, which
  wrote the cancel-time VRAM high-water into that shape's estimate bucket; while
  the bucket had no successful sample, every later attempt at the same shape was
  then planned against that floor. Cancelling a slow sequence and re-queueing it
  is exactly how a render the card can hold came back as "not enough memory".
  Cancellations are now recorded as `invalidated` — an observation, not evidence.
- **A chain stage blocked on GPU memory now gets an answer instead of hanging.**
  An `InsufficientVram` from the plan resolver was silently discarded for owner
  work, so a blocked stage contributed no candidates, reported no reason, never
  asked mold's own idle model cache for the missing bytes, and was retried
  forever — indistinguishable from a hang, with the earlier stages of the
  sequence already rendered. Blocked owner work now records a typed memory
  block, triggers the same idle-scheduler cache reclaim queued generations get,
  and is bounded with the post-eviction numbers if the shortfall survives.
- **A Wan VAE decode that runs out of memory now tiles instead of failing.**
  Wan was the only family with no decode fallback at all — every other engine
  goes through `vae_tiling`, while an exhausted Wan decode failed the render
  after the whole denoise had already been paid for. The full decode is still
  attempted first, and only an OOM falls back to a spatially tiled decode with
  ComfyUI's own geometry: 256x256 pixel tiles, a quarter-tile overlap, and a
  linearly ramped blend mask.
- **The Wan DiT's RMS norms use candle's fused kernel, which is also the
  faithful one.** The hand-rolled version kept F32 across the weight multiply;
  upstream is `self._norm(x.float()).type_as(x) * self.weight`
  (`Wan2.2/wan/modules/model.py:82`), which casts back to the compute dtype
  first — exactly what `candle_nn::ops::rms_norm` does. It also stops
  materializing four full-size F32 temporaries per norm, per block, per step
  (~671 MB apiece at A14B over an 81-frame clip).
- **A finished render gives its GPU memory back, so the next one is not refused
  for it.** candle allocates through CUDA's stream-ordered memory pool, where
  freeing a tensor returns its bytes to the pool rather than to the driver — and
  `cuMemGetInfo`, which every admission decision reads, counts pool reservations
  as used. A completed render therefore left the card looking fuller than it
  was: on an RTX 4090, `wan22-t2v-a14b:q5` at 81 frames rendered on a fresh
  server and the immediate repeat of the same shape was refused at "requires
  23.20 GB, 22.19 GB available", for a render whose real peak is 22,475 MiB.
  Unused pool reservations are now returned when a render or chain stage ends,
  the device twin of the existing host-side `malloc_trim`.
