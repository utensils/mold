- **CUDA builds compile again.** `crates/mold-candle` took `candle-flash-attn`
  from crates.io, whose copy depends on the upstream-named `candle-core`, so a
  `--features flash-attn` build carried that crate alongside the fork's
  `candle-core-mold` and every `flash_attn` call site failed to typecheck
  against two nominally distinct `Tensor` types. It now comes from the same
  fork revision as every other candle crate — the identical upstream 0.11.0
  kernel payload, so the MiniMax H3 kernel itself is unchanged
  ([#1399](https://github.com/utensils/mold/issues/1399)).
- **H3 FlashAttention qualification records name the payload that was
  compiled.** The release-candidate identity carried the crates.io archive
  checksum of a file no build reads once the dependency moved to the fork; it
  now carries the resolved git source, and the record schema is `v2` so a `v1`
  record cannot validate against this build
  ([#1399](https://github.com/utensils/mold/issues/1399)).
- **A second Candle in the dependency graph now fails on the pull request.**
  `scripts/tests/candle-single-identity.sh` joins the release-contract CI route
  and rejects any manifest or lockfile — in any cargo root, including ones that
  do not exist yet — that resolves a `candle-*` package from a registry, a
  path, or a second git revision. The `--features flash-attn` compile gate runs
  only on `main`, which is why this regression shipped four times before anyone
  read a red check ([#1399](https://github.com/utensils/mold/issues/1399)).
