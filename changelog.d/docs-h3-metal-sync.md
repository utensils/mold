- **MiniMax H3 docs now describe the shipped Metal route.** The README, H3 model
  guide, qualification record, and agent skill said admission refused every
  backend but CUDA and that macOS builds omitted the `h3` feature entirely.
  Both stopped being true when the Apple Silicon route shipped: admission
  accepts a Metal device, the public runtime profile is
  `supported-compact-fl2va-cuda-sm89-or-metal`, and the released macOS builds
  carry `h3`. Metal remains correctness-only and unqualified — no H3 checkpoint
  has completed a render on it yet
  ([#1164](https://github.com/utensils/mold/issues/1164)). The qualification
  record also now carries the NVFP4 transformer no-go and its reopen criterion
  ([#1318](https://github.com/utensils/mold/issues/1318)).
