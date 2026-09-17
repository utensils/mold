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
| 02#15 bare arrow keys | fixed | `df190024` | `aFocusedControlKeepsItsOwnArrowKeys`, `theResultStripStandsDownWheneverTheArrowsAreClaimed` |
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

## Adversarial review of this lane (`REVIEW-B.md`), all addressed

| id | status | commit | test |
| --- | --- | --- | --- |
| HIGH 1 a Stop is discarded by the next Generate | fixed | `94828f19` | `aStoppedSubmissionIsStillCancelledWhenAnotherTakesTheCanvas`, `aSecondPressNeverAbortsAnUnansweredPost`, `aStoppedSubmissionThatFailsIsSilent`, `twoStoppedSubmissionsAreBothCancelled`, `theFenceRemembersAStopPerClientBatchId` |
| HIGH 2 `off_bucket: warn` ships refused sizes | fixed | `b55efd4b` | `aWarnedProfileStillHonoursItsGridAndItsBudget`, `aCoarserWarnedGridStillTakesACarriedSize`, `aWarnedSizeOutsideTheAspectBandFallsBackToTheLadder` |
| MED 3 placement still sent the title | fixed | `3fc887fd` | `aPlacementPreviewCarriesNoBytesAndNoUserText` |
| MED 4 the beat protected the chrome, not the picture | fixed | `2e96846c` | — (`show(_:)` is the single acknowledge site; `theNextBatchWaitsUntilTheCanvasHasTheResult` still pins the hold) |
| MED 5 a clip that cannot decode is a black rectangle | fixed | `2e96846c` | — (needs a real `AVPlayerItem`; the ticket re-mint and the sentence are one code path) |
| MED 6 imports race and can reorder Qwen's Target | fixed | `3518a826` | — (ordering is the awaited sequence; `PictureImportTests` still pin the conform) |
| MED 7 the fake's `submit` gate + the real flake | fixed | `94828f19` | the four held-submit tests above exercise the gate |
| LOW 8 stale `sourceMode` on a recipe-less model | fixed | `168957f2` | — |
| LOW 9 two classifiers sharing one name | fixed | `2e96846c` | `aFinishedChildIsClassifiedByItsOwnContainer` |
| LOW 10 stale placement answer after the await | fixed | `3fc887fd` | — |
| LOW 11 legacy Qwen strip capped at one | fixed | `3518a826` | `anAbsentReferenceCapIsUnbounded` |
| item 12 contextual menus | built | `df190024` | `GenerateMenusTests` (9) |
| type-size rule (624 -> 683) | fixed, now 601 | `168957f2` | `make lint` |

### The contextual menus, in full

| surface | ordinary | destructive | absent, and why |
| --- | --- | --- | --- |
| result canvas + each `ResultStrip` tile | Save a Copy…, Copy, Show in Library, Use as Source Image, Add as Reference | — | Quick Look: needs the Library's `PrintMaterializer` (Lane C's) to put a real file on disk, so it is not a trivial composition of anything in this pane |
| source well | Choose File…, Choose from Library…, Paste, Edit Mask… | Remove | — |
| reference item | Move Left, Move Right, Choose File… | Remove | — |
| reference strip background | Add…, Paste | Remove All | — |
| identity photograph | Choose File… | Remove | — |
| mask row | Edit Mask… | Clear Mask | Invert: no inline control exists to compose from |
| adapter row | its trained words, then Reset Strength | Remove | Reveal/Details: no route and no inline control |
| recent prompt | Use Prompt, Copy | — | Remove from History: the history route offers `clear` alone, no per-entry delete |

Every list is gated on a fact its surface already answers to draw its inline
control (room in the strip, a mask path on the recipe, a strength off its
default), and both surfaces render the SAME declaration. Menus appear on the
result tiles, the source well, each reference and the strip, each identity
photograph, the mask row, each adapter row and each recent prompt.

## Notes for the integrator

- **02#15 is closed, not deferred.** `ArrowKeyClaim` is the one gate and it
  asks the responder chain, so the strip's ←/→ stand down for a text view, a
  `Slider` and a `Stepper` alike; the strip re-reads it on
  `NSWindow.didUpdateNotification` and writes its state only when the answer
  CHANGES. No binding changed, and the gate is pure so the test builds real
  `NSSlider`/`NSStepper` instances.
- **`GenerateController` is 601 lines across its files**, below the 624 the
  lane found. `PendingRecovery`, `PreviewPoll` and `MachineChoiceStore` took
  the three concerns that were never the controller's.
- **`GalleryPrint.kind` fallback, for the integrator to wire.** Lane B exposes
  `PrintKind.playableClipExtensions` and `PrintKind(playbackOf:)`
  (`ResultMedia.swift`). The line to add in Lane A's `GalleryPrint.swift` is in
  `var kind: PrintKind`, as the answer when `format` is absent:
  `if format == nil { return PrintKind(playbackOf: filename) }` as its first
  statement. Lane B does not edit that file.
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
- **The flake diagnosis in the first pass was WRONG and is now fixed.**
  `RunQueueTests.aSecondGenerateWhileOneRunsIsAdmittedAndQueued` was not a
  `settle` timing out: two submit tasks raced an unguarded
  `submitAnswers.removeFirst()` from two threads, so the second press could
  become `batch-1`. The fake claims its answer under a lock now, in submission
  order. `ModelActionsTests.theMenuAndTheContextualMenuCallTheSameThing` is a
  separate suite's `settle` budget (Lane D raised `settle(until:)` to 2 s on its
  branch, which covers it); nothing here touches it.
- **Family sets in MoldClient, for the record.** The lane adds four:
  `ExpandTask`'s `h3Families`/`wanFamilies`/`videoFamilies`,
  `ClipLengthBounds`'s `"wan"` test, and `ReferenceImagesCapability.legacy`'s
  qwen/flux2 sniff. Each is a faithful port of an equivalent studio set
  (`expandTask.ts`, `chainRouting.ts`, `legacyRecipeRules.ts`) and 01#4
  explicitly sanctions the legacy one, but PLAN.md names CFG++ as the one
  sanctioned family set -- so they are listed here rather than left to be
  re-derived.
- **`ExpansionTests` coverage note.** Two tests now call
  `ExpansionOffer.resolve` where they called `controller.expansionOffer`; the
  controller's call-through is no longer exercised by them. Cheap to add back
  if the integrator wants both.
- **Not-a-bug: none.** Every finding in this lane's list reproduced against the
  code as written.
