- **TUI image requests no longer carry video timing.** Image-only recipes such
  as SDXL now omit the TUI's hidden frame and FPS defaults, so identity and
  ordinary image generations reach the server without a video-field refusal
  ([#1309](https://github.com/utensils/mold/issues/1309)).
