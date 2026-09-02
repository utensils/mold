- **MiniMax H3 reuses its prompt conditioning.** A repeated prompt + first
  frame (or reference set) on the same conditioner route and device skips
  the 15.7 GB Qwen3-VL load and encode (~80 s on the CUDA route, ~2,400 s on
  the host route); the output is bit-identical, the hit is disclosed as
  `prompt conditioning [cache hit]`, and `MOLD_H3_CONDITIONER_CACHE=off|<MiB>`
  bounds or disables the 512 MiB in-process cache
  ([#814](https://github.com/utensils/mold/issues/814)).
