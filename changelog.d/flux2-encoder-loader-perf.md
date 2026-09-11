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
