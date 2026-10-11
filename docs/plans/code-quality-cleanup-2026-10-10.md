# Code quality cleanup, 2026-10-10

Baseline: `cfdbec10f63fdd6b81fa34606cdeade1a6c8cb93`. Branch:
`fix/audit-quality-cleanup`. Scope: the 21 source-verified audit findings, their
regression tests, and closely related bugs exposed while consolidating contracts.

## Plan and review boundaries

1. Reproduce correctness failures before changing behavior. Divide ownership into
   Rust inference/core, frontend/shared UI, native/shared Swift, and scripts/docs.
2. Fix cache identities, resource bounds, cancellation, recovery and transfer
   evidence first. Keep model merge arithmetic and transaction ordering intact.
3. Consolidate proven duplicate policies and wire schemas. Remove private dead
   code only after checking all callers, feature gates and build roots. Preserve
   public compatibility surfaces unless their retirement is established.
4. Extract typed planning, execution and verification from large controllers.
   Keep shell presentation and platform-specific media decoding local. Raw file
   length is not a defect score; avoid moving code solely to meet a line limit.
5. Run focused regressions, full affected suites, type/build/lint gates, and a
   browser check of actual media controls. Do not launch model generations.
6. Have each area reviewed by a different author after implementation; resolve
   valid findings and rerun affected checks before opening the PR. Leave it
   unmerged and verify active CI on its exact head.

The implementation plans were cross-reviewed before edits. Final review swaps
ownership: Rust author reviews native, frontend author reviews Rust, native author
reviews frontend/scripts. The coordinator reviews cross-surface coverage.

## Implementation ledger

| Audit | Implemented change                                                                                                                                                        | Regression evidence                                                                                           |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| A01   | FLUX/SD3 delta keys use source-layer identity, not randomized patch position.                                                                                             | Reordered Q/K/V cache-on and cache-off comparisons.                                                           |
| A02   | Shared transfer-source result distinguishes eligible, ineligible, absent and failed lookup.                                                                               | Missing, ineligible and read-failure transfer cases.                                                          |
| A03   | Adapter revisions invalidate parsed, delta and resident caches; before/after load fences prevent reuse of an unstable observation.                                        | Same-path replacement through dense and CPU GGUF consumers; unavailable metadata and load-fence cases.        |
| A04   | Derived LoRA deltas have a 256 MiB per-engine FIFO bound and clear on unload.                                                                                             | Strength sweeps, eviction, oversized refusal before copying, and unload disposal.                             |
| A05   | Family-neutral alpha conversion/parser replaces divergent FLUX/SD3 copies.                                                                                                | Integer, singleton and alternate floating alpha tensors.                                                      |
| A06   | Thumbnail subscriptions carry identities; the last cancelled waiter cancels the download.                                                                                 | Shared-consumer and final-consumer cancellation.                                                              |
| A07   | Generation-following task ownership clears on failure; finite admissions survive suspension and retain late Stop intent.                                                  | Transient status recovery, background admission, late Stop/Stop Everything and suspended follower tests.      |
| A08   | iOS queue previews use a 32 MiB aggregate budget, four request lanes and bounded response accumulation.                                                                   | Aggregate eviction, lane limits, stale completion and oversized response cases.                               |
| A09   | Shared media-file reader/composable validates size, preserves prior values on failure and fences stale reads; mobile retains its aggregate budget.                        | Reader/composable tests, shell tests, browser empty/oversized/valid/remove checks.                            |
| A10   | Knip includes types; a separate production module graph follows Vue and workspace aliases without test roots.                                                             | Graph fixtures distinguish production reachability from test-only modules.                                    |
| A11   | Removed six modules/facades with no production consumers.                                                                                                                 | Useful facade tests retarget the shared authority; full suites and graph pass.                                |
| A12   | Removed retired model-filter/resolution functions and obsolete DTOs.                                                                                                      | Call-site search, Knip types/exports, production graph and shell tests.                                       |
| A13   | Removed the unreachable iOS GenerateCanvas/ResultPager subtree.                                                                                                           | Active media export/viewport contracts and full iOS compilation/tests.                                        |
| A14   | Removed superseded LibraryNewMedia, VideoPlaybackLayout, synchronous budget logging and durable-batch sentinel helpers.                                                   | Active replacement tests retained; full native/shared suites.                                                 |
| A15   | Removed unused cache parameters/keys from backends that do not consume them.                                                                                              | All-target workspace and Metal inference compilation; LoRA suites.                                            |
| A16   | Removed four obsolete private Rust helpers and replaced a dead shim with test-scoped imports.                                                                             | Clippy with warnings denied and inference tests.                                                              |
| A17   | Shared server DTOs replace duplicated shell schemas; desktop-only legacy rows have an explicit adapter.                                                                   | Wire-shape tests, both shell typechecks; corrected chain `stage`/numeric seed and preserved device placement. |
| A18   | Shared source-canvas controller owns desktop/mobile transitions.                                                                                                          | Controller and existing shell transition tests.                                                               |
| A19   | Shared Foundation-only source-media filtering, usage choice, inventory and publication fencing replace native copies.                                                     | Shared tests on macOS/iOS plus both app suites.                                                               |
| A20   | Extracted canvas/media policies, keyboard viewport lifecycle, retained-source restoration, and typed local-sync discovery/planning/copy/organization/verification stages. | Boundary tests plus sync receipt-before-organization and stable-final-checkpoint integration test.            |
| A21   | Removed the unused regression batch knob; retained intentional eager submission and made the concurrent retry fixture atomic.                                             | Concurrency, aggregate-failure and retry script contracts.                                                    |

Related fixes discovered during validation: the Mac library picker now uses the
existing row-action-menu contract; frontend chain consumers use the Rust wire
field `stage`; web placement retains an existing device pin. Each has a regression
test. SD15/SDXL key mapping and API/local prompt-operation selection also share
their existing policy without changing family-specific allowlists or messages.

## Evidence and limits

- Rust: full inference suite; final focused LoRA and core expansion suites;
  workspace all-target Clippy and explicit Metal/expand inference Clippy, warnings
  denied; workspace formatting and generated-contract checks.
- Frontend: complete Studio/web/desktop suites, web/desktop/mobile builds and
  typechecks, architecture, formatting, Knip and production graph.
- Native: full iOS and macOS app suites, shared MoldClient suites on macOS and
  iOS Simulator, MoldStyle/MoldMesh, and both native lints. Tests use isolated
  app state. The Mac test host uses a unique bundle ID to coexist with a running
  personal app.
- Browser: actual web LTX controls mounted with an isolated local fixture;
  empty/oversized rejection, valid audio acceptance/removal and preservation of
  accepted audio after video refusal. No console warnings/errors. This proves
  those controls, not the full desktop/mobile shell or physical-device behavior.
- Scripts: changed scripts pass ShellCheck and focused contracts. The broad
  contracts runner passes 41 of 49 steps. Unchanged failures include retained
  release/coverage expectations for owner-paused CI, Linux ELF/CUDA qualification
  assumptions on macOS, path-identity assumptions, and the H3 media scan's existing
  tracked iOS playback fixture. Do not reactivate paused CI to satisfy restoration
  tests; active routing/workflow checks pass.
- No CUDA build or checkpoint-generation performance qualification was performed
  on this Mac. Existing ignored inference tests remain ignored. Native size-lint
  advisories remain: responsibility extraction is incremental, not a wholesale
  rewrite. Public compatibility candidates remain intentionally preserved.
- The production graph proves module reachability. Named exports inside live
  modules still require review because Knip also includes tests. Exact test/tool
  exceptions are documented in the graph; passing it is not proof that every
  exported symbol ships.

Final peer-review results and exact-head CI belong in the PR validation record.
Passing tests reduce regression risk; they do not establish an absolute guarantee.
