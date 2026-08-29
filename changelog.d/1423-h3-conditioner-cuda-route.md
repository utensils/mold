- **MiniMax H3 runs its Qwen3-VL conditioner on the CUDA device when it
  fits.** Every CUDA route had pinned the 32B conditioner to the host since
  #919, so each FL2VA and Ref2VA render paid its prefill on the CPU — a single
  2048-square Ref2VA image reference took ~40 minutes on an RTX 4090. The
  vision tower's per-image attention is now query-chunked (exact; a 16,384-patch
  reference no longer materializes a ~34 GB score matrix per layer), and
  admission places the conditioner on the device whenever the Qwen phase —
  fixed runtime, the 1.19 GB of dense-resident tensors, the request's
  activation demand, and the output state — fits the available VRAM, keeping
  the host route otherwise
  ([#1423](https://github.com/utensils/mold/issues/1423)).
