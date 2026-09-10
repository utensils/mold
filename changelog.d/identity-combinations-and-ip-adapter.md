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
