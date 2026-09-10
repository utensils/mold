- **Face identity now renders alongside a LoRA and img2img.** Both pairings
  were refused as "not yet qualified"; neither was ever a numerical
  prohibition, and both are qualified on FLUX.1 and SDXL. A LoRA merges into
  the base weights underneath the adapter, and an identity render simply
  starts further into the schedule.
- **Fixed `--id-start-step` on a FLUX img2img render.** The denoise loop gated
  identity on its index into the *truncated* schedule while the value was
  validated against the full `--steps`, so a start step meant the wrong step —
  and at any value at or past the remaining length it silently meant nothing
  at all, rendering an unconditioned print that still reported success.
- **IP-Adapter image prompting on SD 1.5 and SDXL.** Attach a reference picture
  with `--reference` and its appearance — subject, palette, setting — is
  transferred into the render alongside the text prompt. `--reference-weight`
  dials it from 0.0 to 2.0 (default 1.0; upstream suggests 0.6-0.8 as a
  starting range, since 1.0 lets the picture dominate the prompt). It composes
  with img2img, inpaint, ControlNet and a LoRA in the same pass rather than
  replacing any of them, and a weight of exactly 0.0 renders pixel-for-pixel
  what a request with no reference renders. Pull it with
  `mold pull ip-adapter-sd15` or `mold pull ip-adapter-sdxl`; both bundles
  share one OpenCLIP ViT-H/14 tower, so the second costs only its adapter
  ([#1573](https://github.com/utensils/mold/issues/1573)).
