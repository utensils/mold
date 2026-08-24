- **MiniMax H3's compact envelope is a rule, not a pinned shape.** The compact
  FL2VA tags now render any canvas whose axes are multiples of 32, at least
  256 px each, totalling at most `1344x768`'s pixel count, with aspect between
  1:4 and 4:1 — plus 107 to 345 frames on the `17n+5` grid at 24 fps and 2 to
  50 sampler steps. The exact 124 frames, two-canvas set, and 21-step pin were
  the qualifying campaign's own shape read as a contract. The runtime envelope
  and its memory bounds are minted per request and scaled from those
  measurements, so a shape that does not fit is refused with real numbers
  instead of by a fixed list, and a source image renders its own aspect at the
  largest admitted size rather than being letterboxed. Reviewed Turbo tags keep
  their distilled adapter's exact step count.
