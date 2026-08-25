- **MiniMax H3's reviewed Ref2VA Turbo tier ships as a model tag.**
  `minimax-h3-ref2va:comfy-pruned-int8-turbo-4step` pulls the Ref2VA compact
  stack plus one pinned 1.96 GB adapter and renders at 5 terminal-inclusive
  sampler grid points, the same way the two FL2VA Turbo tags do. A Turbo tag
  now collapses onto its own task's base checkpoint directory rather than
  FL2VA's, and each adapter stays reviewed for exactly one task partition
  ([#825](https://github.com/utensils/mold/issues/825)).
