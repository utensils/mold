- **Fixed the MiniMax H3 server build failing to compile against the PuLID
  identity work.** `DependencyPreparationContext` gained a `frozen_identity`
  field; the non-H3 construction sites were updated but their `h3`-gated twins
  in the placement-preview route and the scheduler were not, so every
  `h3-cuda` build — the feature set the Linux sm89 release artifact ships —
  failed to compile while CI stayed green. Those H3-only contexts now default
  additive optional inputs so the same shared-struct change cannot break them
  again.
