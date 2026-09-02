- **MiniMax H3 reuses its prompt conditioning.** A repeated prompt + first
  frame (or reference set) on the same conditioner route and device skips
  the 15.7 GB Qwen3-VL load (53.6 s in the measured CUDA row) and its encode,
  which scales with the conditioned patch count — 24.6 s on the CUDA route for
  that row's 2048-square Ref2VA reference, 2,405.6 s for the same reference on
  the host fallback route. The output is bit-identical, the hit is disclosed as
  `prompt conditioning [cache hit]`, and `MOLD_H3_CONDITIONER_CACHE=off|<MiB>`
  bounds or disables the 512 MiB in-process cache
  ([#814](https://github.com/utensils/mold/issues/814)).
