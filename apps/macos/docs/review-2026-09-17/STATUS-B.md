# Lane B · Generate — ledger

Branch: `worktree-agent-a97cb5ed66f345ecc` (cut from `34cfa484`).
Findings from `02-generate.md`, plus `01-client.md` #3 #4 #7 #8 #13 #15 and its
`GenerateRequest.encode(to:)` test gap, plus `06-parity-matrix.md` §5.1.

| id | status | commit | test |
| --- | --- | --- | --- |
| 02#1 wells follow the relation | fixed | `ad315de3` | `sd15CombinesKeepsBothWellsAndParksNeither`, `anAdditiveRecipeDrawsBothWellsAndParksNeither`, `anExclusiveRequestCarriesOneWellAndNeverBoth` |
| 01#4 legacy reference fallback | fixed | `ad315de3` | `anAbsentBlockFallsBackToTheLegacyFamilyRule`, `anOlderHostStillDrawsQwensTargetStrip`, `aHiddenBlockNeverReachesTheLegacyRule` |
| 06 §5.1 finished clip never renders | fixed | `84385aef` | `aFinishedChildIsClassifiedByItsOwnContainer`, `aChildWithNoFileYetHasNoKind` |
| 02#2 Stop during `.submitting` | fixed | `a3edafc7` | `stopOnAFirstEverRenderCancelsTheBatchTheHostAdmits`, `stopDuringASecondSubmissionNeverCancelsTheBatchBefore`, `aSubmissionSupersededWhileInFlightQueuesInstead` |
| 02#9 result replaced before it is drawn | fixed | `a3edafc7` | `theNextBatchWaitsUntilTheCanvasHasTheResult`, `theQueueMovesOnAnywayWhenNobodyIsLooking` |
| 02#5 placement preview sends everything | fixed | `c0cc48f0` | `aPlacementPreviewCarriesNoBytesAndNoUserText`, `aPlacementPreviewSendsARedactedRequest` |
| 01#3 / 02#3 length ceiling | fixed | `94098808` | `ltx2sCeilingIsTheDurationCapAtTheChosenRate`, `aTextOnlyWanTierStopsAtItsOwnClip`, `theClipCeilingsAgreeWithTheSharedWanFixture` |
| 01#7 `off_bucket: warn` | fixed | `660793bf` | `aWarnedOffBucketSizeIsKeptRatherThanSnapped` |
| 01#8 alignment past `max_pixels` | fixed | `660793bf` | `alignmentNeverGrowsASizeBackPastThePixelBudget` |
| 02#7 identity photos (HEIC) | fixed | `86f737a6` | `anIdentityPhotoTheServerCannotReadIsTranscodedToPNG`, `theTwoContractsAreReallyDifferent` |
| 02#10 reads/base64/decodes off-main | fixed | `86f737a6` | (covered by `PictureImportTests`; the isolation itself is a compile-time property of `nonisolated` + `Task.detached`) |
| 02#11 `machineChoice` observable | fixed | `0c62659a` | — (a stored `@Observable` property; `MachineControlTests` cover the picker) |
| 01#13 / 02#12 Expand records the real task | fixed | `0c62659a` | `expandSendsAndRecordsTheRealTask`, `ExpandTaskTests` (7) |
| 02#13 expand/remix staleness fence | fixed | `0c62659a` | `aRewriteWhoseBoxMovedIsRefusedByName`, `theSnapshotNamesEachThingThatMoved` |
| 02#14 slider a11y labels | fixed | `0c62659a` | — (view modifier; `make lint`'s a11y rule is the floor) |
| 02#15 bare arrow keys | narrowed, see below | `0c62659a` | `theResultStripYieldsEveryArrowToACaret` |
| test gap: `GenerateRequest.encode(to:)` exhaustive | fixed | `cc95d6b9` | `everyStoredPropertyReachesTheWire`, `everyStoredPropertyIsPopulated` |

## Wave 3 (out of scope here, by the lane brief)

| id | why |
| --- | --- |
| 02#4 / 01#15 IP-Adapter weight control | F4 · Generate controls |
| 02#6 full Reuse | F2 · Reuse + retained source media |
| 02#8 `source_fit` / canvas follows a source | F4 |
| 02#16 draft persistence | F4 |
| 02#17 scheduler / CFG++ / wan recipe / LTX-2 overrides / pull-on-demand | F4 |
| auto-chaining a long clip | F4 — this pass CAPS the slider at the clip size and says why instead |
| the mesh result canvas | F1 · `MeshView`; the `.mesh` arm in `RunCanvas+Result` is the named seam |

## Notes for the integrator

- **02#15 is narrowed, not closed.** The reporter flagged it as "worth a manual
  check rather than asserting it", and the part that IS assertable was already
  right: an unmodified arrow is a window-scoped key equivalent, and the strip
  stands down for a caret because the prompt and the negative prompt publish
  `editingText`. That is now a pure function with a test. Whether a bare arrow
  out-competes a FOCUSED `Slider`/`Stepper` in the capsule cannot be settled by
  reading code; it needs a real keypress against the running app, so it belongs
  in Wave 4's HIG sweep. Nothing here changes a binding.
- **`GenerateController` grew 624 -> 655 lines across its files.** All of it is
  wiring: three collaborators declared (`SubmissionFence`, `ResultHandoff`,
  `PlacementProbe`), `machineChoice` becoming a stored property, and the `stop`
  / `submit` / expand landing branches. The behaviour itself went into new small
  types, and two extractions gave back 45 lines (`PlacementProbe`,
  `ExpansionOffer.resolve`).
- **Cross-lane overlap with 01#5 (Lane A).** `PrintKind(filename:)` in the new
  `ResultMedia.swift` classifies a batch child's stored FILENAME; Lane A is
  widening `GalleryPrint.isVideo`, which classifies a print's `format` field.
  They are two questions with two inputs and deliberately different answers for
  GIF/WebP -- a player cannot open one, so the canvas decodes it as a still --
  but an integrator should read both together.
- **Cross-lane edits.** `Tests/MoldTests/FakeBackend.swift` gained
  `cancelledBatchIds`, a holdable `submit`/`expand` and the recorded
  `placementRequests`/`expandRequests`/`remixRequests`; `RunQueueTests.swift`
  and `RunQueueTests+Machines.swift` acknowledge the handoff where a canvas
  would; `BatchOutcomeTests.swift` injects a zero debounce;
  `ExpansionTests.swift` calls `ExpansionOffer.resolve` where it used to call
  `controller.expansionOffer`. `RenderDraftTests` and `RenderDraftParkTests`
  had two assertions REVERSED on purpose -- the exclusive relation now keeps
  both wells and decides at request time (02#1).
- **Pre-existing flakes seen while running, not caused here.**
  `RunQueueTests.aSecondGenerateWhileOneRunsIsAdmittedAndQueued` and
  `ModelActionsTests.theMenuAndTheContextualMenuCallTheSameThing` each failed
  once under load and passed on a re-run: both assert after a `settle(until:)`
  that can time out rather than after the work itself.
- **Not-a-bug: none.** Every finding in this lane's list reproduced against the
  code as written.
