### Fixed

- MiniMax H3 Turbo tags now store their base stack in the base checkpoint's directory: pulling a Turbo tag on a machine with `minimax-h3-fl2va:comfy-pruned-int8` installed downloads only the ~1.96 GB adapter instead of re-downloading ~41 GB into a tag-named copy, and removal ref-counting protects the shared files in both directions.
