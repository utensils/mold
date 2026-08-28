- **MiniMax H3 Ref2VA Turbo 4-step renders instead of failing after the full
  clip.** The Ref2VA phase backend hard-coded the `comfy-res-multistep`
  integrator, so a Turbo tag rendered every phase with the wrong sampler and
  was then refused by the provenance guard; it now takes the integrator and
  video shift from the frozen quantization authority exactly as FL2VA does
  ([#1432](https://github.com/utensils/mold/issues/1432)).
