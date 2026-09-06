- **Z-Image Metal VAE recovery.** Bounded Candle convolution workspaces replace
  the proactive tiling cap for whole 1024-pixel decoding. Memory errors can
  retry with tiles, and a repeated OOM during cleanup no longer prevents the
  eager CPU fallback. CPU and CUDA decode ordering is unchanged
  ([#1040](https://github.com/utensils/mold/issues/1040)).
