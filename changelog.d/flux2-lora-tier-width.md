- **A FLUX.2 LoRA trained for the wrong tier is refused at submit time, not
  after the checkpoint loads.** The three FLUX.2 tiers are different widths —
  3072 for [klein] 4B, 4096 for [klein] 9B, 6144 for [dev] — and nothing in an
  adapter's file name says which it was trained for. A 9B adapter handed to
  `flux2-klein:q8` used to report `32 layers, rank 32` and `32 patches on 16
  tensors, 0 skipped`, read and dequantise the whole GGUF checkpoint, and then
  fail with `shape mismatch in add, lhs: [3072, 3072], rhs: [4096, 4096]`. The
  adapter's safetensors header is now read up front, and the refusal names the
  adapter, its width, the tier's width, and the tier it fits — identically from
  the CLI, the HTTP API and the durable queue. An adapter that only partly
  matches is refused too, rather than half-merged into an image no adapter
  produced.
- **A FLUX.2 LoRA naming separate `to_q`/`to_k`/`to_v` projections now merges
  into a quantized checkpoint's fused attention weight.** Such an adapter was
  mapped onto the fused `qkv` tensor with the delta and the base the wrong way
  round, so on a GGUF tier it logged `Flux.2 LoRA Splat: base row count !=
  delta row count, skipping` and silently dropped three quarters of its layers
  while still reporting them applied. Relatedly, a merge that cannot be
  performed is now an error naming the tensor rather than a skipped patch or a
  message about two anonymous shapes.
