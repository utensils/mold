- **Wan's step cache is on by default.** `MOLD_WAN_STEP_CACHE=auto` now needs no
  opt-in on the non-distilled quality tiers, where it is measured at **1.85x**
  (`wan22-t2v-a14b:q8`, 33f at 832x480, 605.6 s to 327.4 s)
  ([#1482](https://github.com/utensils/mold/issues/1482)). It could not be the
  default while the memory it holds went undeclared: the retained residuals are
  invisible to the activation estimate that admission and the block-offload
  policy both read, so a near-capacity render parked too few blocks and ran out
  of device memory on bytes nothing had accounted for. They are charged now.
- **The step cache's distance check no longer costs more memory than the cache
  itself.** Comparing two residuals upcast both of them whole, holding three
  full-size float32 copies at once — roughly 1.3 GB at A14B 53f/832x480, against
  the ~450 MB the cache actually retains. The same comparison now reduces a slice
  at a time, so the check's cost no longer grows with the clip length, and the
  accumulation is more accurate than the one it replaces rather than less.
