- **One weight index for every LTX-2 transformer file.**
  `mold_core::ltx2_weight_index::Ltx2TransformerWeightIndex` now answers what
  a safetensors or GGUF transformer weighs — per-block and non-block bytes at
  rest, widened, and packed, the block count, the storage format, and the
  AdaLN width — and both the scheduler's LTX-2 admission model and the
  engine's residency planner read it (pinned by a parity test). INT8 ConvRot
  packs are priced at the BF16 size the loader actually materializes on the
  device instead of their raw bytes (a 2× under-count that admitted plans
  which OOMed at the first denoise step), and the AdaLN width is read from the
  exact `adaln_single.linear.weight` key rather than whichever
  `*adaln_single.linear.weight` a hash map yielded last (a 4.5× swing in the
  conditioned activation term between runs). A `.sha256-verified` sidecar now
  records the byte length it hashed and vouches only for that length, and a
  sidecar from an older build is trusted only while the manifest's declared
  size still holds — so a file rewritten underneath its marker shows up as a
  repair instead of a "downloaded" model that fails qualification.
  LTX-2.5 packs no longer advertise the LTX-2.3-only x1.5 spatial upsampler,
  and the docs name the diffusion-VAE variants by their real `:bf16` tags
  ([#1398](https://github.com/utensils/mold/issues/1398),
  [#1414](https://github.com/utensils/mold/issues/1414)).
