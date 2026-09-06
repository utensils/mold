- **Z-Image Metal VAE recovery.** Whole-decode memory errors can retry with
  tiles, and a repeated OOM during cleanup no longer prevents the eager CPU
  fallback. CPU and CUDA decode ordering is unchanged
  ([#1040](https://github.com/utensils/mold/issues/1040)).
