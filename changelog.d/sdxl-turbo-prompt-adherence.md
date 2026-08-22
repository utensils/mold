- **SDXL Turbo follows the prompt again at its default guidance.** Its `0.0`
  guidance was mistakenly treated as active classifier-free guidance, which
  selected the unconditional prediction at every denoise step. Guidance at or
  below `1.0` now uses the intended single conditional pass.
