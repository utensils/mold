- **The rest of the FLUX.2 family.** `mold pull flux2-dev:{q8,q6,q4,fp8}` brings
  FLUX.2 [dev] down from a 103 GB gated BF16 install to a 20 GB quantized one —
  and installs it without a Black Forest Labs license acceptance, because the
  quantized tiers pair ungated transformers with ungated mirrors of the same
  encoder and VAE bytes. `flux2-klein:fp8` and `flux2-klein-9b:fp8` add BFL's
  own FP8 conversions, and `flux2-klein-base{,-9b}:{bf16,q8,q6,q4}` add the
  undistilled base checkpoints, which sample with real classifier-free guidance
  and are the first Flux.2 tier to accept a negative prompt.
- **FLUX.2 [dev] quantized renders now honour the guidance scale.** The
  quantized transformer dropped the guidance conditioning a guidance-distilled
  checkpoint is trained to receive, so a GGUF dev render ignored `--guidance`
  entirely. Klein is distilled without guidance embedding and is unaffected.
- **FP8 Flux.2 checkpoints apply their dequantization scale.** A single-file
  FP8 checkpoint was loaded without its `weight_scale` sidecar, which does not
  soften a render — it multiplies every weight in the layer by ~500x.
