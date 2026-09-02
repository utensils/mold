- **Two more MiniMax H3 FL2VA Turbo tiers.**
  `minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p-v1.1` (5 grid points)
  and `minimax-h3-fl2va:comfy-pruned-int8-turbo-8step-768p` (9 grid points)
  pull the FL2VA compact stack plus one revision-pinned, SHA-256-verified
  adapter from `lightx2v/Minimax-h3-Turbo`, stored beside the Comfy-Org
  adapters under `shared/minimax-h3/loras/`; each Turbo manifest row now
  names its own adapter source and revision
  ([#814](https://github.com/utensils/mold/issues/814)).
