- **Keep Wan full denoising correct by default.** Wan step-cache reuse is now
  explicit after a controlled Metal 1.3B run reproduced saturated output with
  the former automatic cache while the same request rendered correctly with
  full denoising ([#1059](https://github.com/utensils/mold/issues/1059)).
