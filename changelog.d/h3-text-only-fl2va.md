- **MiniMax H3 text-only video (T2VA).** Every compact FL2VA model — the base
  `minimax-h3-fl2va:comfy-pruned-int8` and all of its Turbo tiers — now renders
  synchronized audio-video from a prompt alone, with no first or last frame.
  First, last, and first+last frames work exactly as before, the first frame is
  now advertised as optional so web, desktop, and iPhone stop blocking Generate
  without one, and a text-only prompt gets the whole 2,048-row conditioner
  budget. Text-only memory grants are derived from the measured first-frame
  ones and never exceed them. Ref2VA still requires references
  ([#1552](https://github.com/utensils/mold/issues/1552)).
