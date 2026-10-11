- **Correct and bound LoRA reuse.** Keep fused-layer deltas tied to their source
  layers and adapter revisions, invalidate resident adapters after file changes,
  cap derived-delta caches at 256 MiB per engine, and honor integer and singleton
  SD3 alpha tensors.
- **Recover native background work reliably.** Cancel thumbnail downloads after
  their last consumer leaves, resume generation following after transient status
  failures, preserve submissions accepted while backgrounded, honor Stop for
  late admission responses, bound iOS queue preview memory and concurrency, and
  distinguish a missing transfer source from an unavailable or ineligible one.
- **Unify media input handling.** Web, desktop and mobile validate video/audio
  files against a 64 MiB limit before encoding, show read errors, preserve the
  prior attachment on refusal, and discard superseded selections. Mobile also
  enforces a 45 MiB combined media budget.
  Shared canvas transitions and wire contracts keep the clients consistent.
- **Simplify maintained code.** Remove superseded native/frontend helpers,
  consolidate LoRA and source-media policies, split large authoring and sync
  routines into tested stages, and check production frontend reachability
  independently of test imports.
