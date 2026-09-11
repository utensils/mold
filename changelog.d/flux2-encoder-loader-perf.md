- **FLUX.2 [dev]'s prompt encoder runs on the GPU again.** The placement planner
  priced the Mistral3 conditioner at its 36 GB file size, so an idle 46 GB card
  looked over-subscribed and the encoder was auto-parked to the CPU, where
  FLUX.2 selects F32 — a cache-miss prompt took 78.8 seconds with the GPU at
  0 % utilisation for the whole phase. The encoder streams: it memory-maps its
  shards and holds one decoder layer plus the one being prefetched, a peak near
  3.6 GB in bf16. mold now charges that on both the device and the host side, so
  the encoder stays on the GPU on a 24 GB card as well as a 46 GB one, and a
  64 GB desktop is no longer refused outright for host RAM it never needed. An
  explicit `--placement text-encoders=cpu` is still honoured.
- **The Mistral3 encoder no longer blocks on the GPU between layers.** It
  synchronized after the embedding lookup and after every one of its thirty
  decoder layers, which ordered nothing the CUDA stream did not already order
  and stopped the host from preparing the next layer. Each layer's weights are
  now loaded one layer ahead, and the render's progress bar advances through the
  encode instead of jumping once.
- **GGUF checkpoints load from a memory mapping.** Every quantized tensor used
  to be copied into a fresh host buffer before being uploaded, measured at
  0.96 GB/s on a 33 GB FLUX.2 checkpoint against 8.3 GB/s for
  stable-diffusion.cpp reading the same file. FLUX, FLUX.2, SD3, Z-Image,
  Qwen-Image, Wan, Hunyuan3D and the T5/UMT5/Qwen3 GGUF text encoders now upload
  straight from the mapping, report real byte progress rather than a tensor-count
  approximation, and log each file's measured throughput. Weights are unchanged,
  so renders are bit-identical.
- **Parking a text encoder in host RAM costs one copy, not two.** `MOLD_KEEP_TE_RAM=1`
  read the whole checkpoint into an anonymous buffer and then copied every tensor
  out of it, so parking FLUX's 9.79 GB T5 briefly needed twice that.
- **A GGUF checkpoint loads 8-10x faster on its first load.** Reading a whole
  22-35 GB checkpoint through its memory mapping is a page fault per 4 KiB, and
  on ZFS — which is where `$MOLD_HOME` lives on every qualified machine — the
  kernel serves those one page at a time out of ZFS's own cache, with no
  readahead: a 21.76 GB Qwen-Image checkpoint took 5,312,908 major faults and
  26.5 seconds, 0.82 GB/s, for bytes that were already in RAM. mold now reads
  the tensor payload in contiguous batches across eight threads into a reused
  page-locked staging buffer and uploads each tensor from there, with two
  buffers alternating so one is being read while the other is still in flight
  to the GPU. Measured on 4x L40S with the file's page cache dropped:
  `flux1-dev-Q8_0` 15.5 s -> 1.6 s, `qwen-image-Q8_0` 26.5 s -> 3.3 s, and
  `flux2-dev-Q8_0` 41.7 s -> 4.3 s (0.84 -> 8.1 GB/s); already-cached repeats
  went 6.6 s -> 2.7 s on the same 35 GB file. Host memory is bounded by the
  buffer pair rather than the checkpoint, macOS and CPU loading is unchanged,
  and the weights are byte-identical, so renders are too.
- **Text encoders park in host RAM when the machine can afford it, and
  `MOLD_KEEP_TE_RAM` becomes tri-state.** The old rule was a flag plus two
  carve-outs and asked nothing about the host. `auto` (the unset default) now
  measures it: a park is admitted only when the encoder, the transformer that
  loads beside it, and a `max(15 % of RAM, 8 GiB)` safety floor all fit in
  available memory, so a 64 GB desktop keeps streaming exactly as before while
  a 1.5 TB host parks and page-locks. `1` parks whenever the encoder alone
  clears the floor and remains the unchanged opt-in for FLUX/SD3's T5 and Wan's
  UMT5; `0` never parks; Metal never parks, because there the parked copy would
  sit in the pool the encoder already runs from. FLUX.2 [dev] parks the ~35 GB
  Mistral3 prefix it streams, filtered to the layers it actually runs — the
  vision tower, the projector and layers 30-39 the single-file republication
  also ships are never materialized — and the placement planner charges that
  park, so two queued [dev] prints on a 128 GB host cannot both be admitted
  against memory only one of them can have.
- **Quantized Qwen3 encoders park too.** Flux.2 Klein's and Z-Image's GGUF
  encoders used to be excluded from the host park and re-read from disk on
  every cache-miss prompt (3.9 s on Klein). Their `QTensor` bytes now move
  host-to-device losslessly through the same mechanism Qwen-Image's Qwen2
  encoder has used since #1044, verified byte-identical rather than merely
  close.

