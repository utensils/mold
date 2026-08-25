### Added

- MiniMax H3's reviewed Ref2VA Turbo 4-step tier ships as a first-class model tag, `minimax-h3-ref2va:comfy-pruned-int8-turbo-4step`: it pulls the Ref2VA compact stack plus one pinned 1.96 GB adapter and renders at 5 terminal-inclusive sampler grid points. A Turbo tag now collapses onto its own task's base checkpoint directory rather than FL2VA's, and each adapter stays reviewed for exactly one task partition.
