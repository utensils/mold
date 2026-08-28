- **Fixed the private MiniMax H3 UAT server build.** `cargo check -p mold-ai-server
--features h3-private-uat` compiles again — the presentation route still built the
  scheduler's `compute_capability` as a bare pair after it became optional — and CI now
  typechecks that edge, with its tests, so it cannot silently rot again
  ([#1350](https://github.com/utensils/mold/issues/1350)).
