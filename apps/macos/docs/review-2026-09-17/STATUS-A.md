# Lane A · MoldClient wire — ledger

Branch `worktree-agent-a2d5903403e02a402`, off `feat/macos-native-app` at `34cfa484`.
Package suite 417 → 491 tests, app bundle 380, `make lint` green.

| id | status | commit | test |
| --- | --- | --- | --- |
| 01#1 `QueueOrder.moves` index space | fixed | `ad283395` | `aBatchLandsContiguousWhereItWasDropped` (+ 2) |
| 01#2 `.unknown` re-encoded into `prompt_transform` | fixed | `3ea15843` | `PromptTransformWireTests` (6) |
| 01#3 clip length ignores `max_duration_seconds` | Lane B | — | — |
| 01#4 no legacy `reference_images` fallback | Lane B | — | — |
| 01#5 GIF/APNG/WebP clip classification | fixed | `1a0f75d5` | `GalleryPrintKindTests` (5) |
| 06 §5.2 `hunyuan3d` offered in the picker | fixed | `7cd123ff` | `ModelMeshFamilyTests` (3) |
| 01#6 query-value escaper | fixed | `d38a12c3` | `HTTPBackendQueryTests` (6) |
| 01#17 `escaped(_:)` on interpolated ids | fixed | `d38a12c3` | `everyDynamicRouteComponentGoesThroughAnEscaper` |
| 01#7 `fit()` ignores `off_bucket: warn` | Lane B | — | — |
| 01#8 alignment rounds up past `max_pixels` | Lane B | — | — |
| 01#9 / 04-M6 unstable `QueueListing.merged` | fixed | `a4ef57ad` | `QueueMergeTests` (5) |
| 01#10 SSE refusal loses the server's sentence | fixed | `9b0bbd7e` | `StreamRefusalTests` (4) |
| 01#11 `classifyAdmitFailure` blind to 401/licence | fixed | `220477ad` | `TransferAdmitFailureTests` (4) |
| 01#12 `playableURL` fails open on a keyed host | fixed | `3a273912` | `MediaTokenTests` (5) |
| 01#13 Expand records `task: .textToImage` always | Lane B | — | — |
| 01#14 no buffering policy | fixed | `3c7bb54f` | `everyStreamStatesItsBufferingPolicy` |
| 01#21 a Task-hop per byte | fixed | `3c7bb54f` | `theLineReaderReadsNoFurtherThanItsConsumerAsks` (+ 2) |
| 01#15 `reference_weight` unreachable | Lane B | — | — |
| 01#16 key rides a cross-origin redirect | fixed | `4c4c584a` | `RedirectGuardTests` (5) |
| 01#18 tag control characters | fixed | `71f55d5c` | `ClientTagControlTests` (5) |
| 01#19 `CollectionShelf.hidden` inverts studio | fixed | `430e9d91` | `aShelfIsHiddenWhenAnyMachineHidesIt` |
| 01#20 no reference-upload lease on transfer | deferred | — | see below |
| diagnosability: `os.Logger`, decode path | fixed | `b6da85a1` | `DecodingFailureTests` (5) |
| test gap 1 `GenerateRequest.encode` exhaustive | Lane B | — | — |
| test gap 2 `moves` with a neighbour | fixed | `ad283395` | `aBatchLandsContiguousWhereItWasDropped` |
| test gap 3 `VideoOnlyPolicy` untested | fixed | `672d1ac1` | `VideoOnlyPolicyTests` (5) |
| test gap 4 clip park/restore | Lane B | — | `DraftMedia*` is Lane B's |
| test gap 5 downloads `snapshot` frame | fixed | `672d1ac1` | `DownloadSnapshotTests` (4) |
| test gap 6 `MediaToken.playableURL` | fixed | `3a273912` | `MediaTokenTests` |
| test gap 7 `ExportOptions.forVideo`/`.forMesh` | fixed | `672d1ac1` | `ExportOptionsTests` (5) |
| — stub transport raced between suites | fixed | `466f18c9` | — |
| — keyboard batch move pinned the 01#1 bug | fixed | `7d029b3f` | `aBatchMovesFromTheKeyboardAsAscendingCalls` |

## Found while writing the missing tests

- **`DownloadsListing` threw on a host that omits `active_jobs`** (`672d1ac1`). The field is
  `#[serde(default)]` in Rust precisely because an older host sends only `active`, but a
  synthesized `init(from:)` requires every non-optional key — and `downloadEvents` decodes inside
  a `try?`, so the snapshot frame (the only way this app learns about a `mold pull` at a terminal)
  vanished silently and the popover stayed empty. Exactly the failure gap 5 predicted.
- **`Tests/MoldTests/QueuePaneTests.swift` asserted the 01#1 bug** (`7d029b3f`): `[1, 2]` is the
  plan that splits the batch around the row it moved past. Rewritten to assert the landed ORDER.

## Deferred

- **01#20** — a reference-upload lease path on transfer is the whole upload-session feature
  (`prepareReferenceUploadBatch` + the session routes), not a small fix, and the README already
  documents upload sessions as out of scope. Left as the stated parity gap the review calls it.

## Judged differently from the report

- **01#5, APNG.** The report calls APNG "worse-but-unfixable-here" because
  `OutputFormat::Apng.extension()` is `"png"`. That is the file extension; `GalleryPrint.format`
  carries the serialized `OutputFormat`, so `apng` IS answerable and is handled. The report's
  suggested set is also not quite right in the other direction: `webp` is offered for a STILL
  recipe and a temporal one (`generation_profile.rs:2140-2157`), so it cannot be classified by
  container at all and asks the print's own `frames` instead. `webm`/`mov` are dropped —
  `metadata_io.rs:36-56` is a closed set and mold has never written either.
- **01#19** was a design call, not a bug. Settled in studio's favour (`hidden` when ANY host copy
  is), because the hiding mutation fans out, so a mixed state is a half-landed edit rather than a
  disagreement. Nothing in `Sources/Mold` reads the property yet.
- **01#21, the remaining half.** The pipeline no longer hops per byte and no longer buffers, but
  the SOURCE is still `URLSession.AsyncBytes`, whose element is one byte: Foundation exposes no
  chunked body without a `URLSessionDataDelegate`, and a delegate-driven body stream is a
  separate piece of work. `LineAccumulator.consume(contentsOf:into:)` is there and tested for
  when it lands.

## Cross-lane edits

Two one-line changes in files Lane B owns; both are the call site of something added here.

- `Packages/MoldClient/Sources/MoldClient/GenerateRequest+Encoding.swift:58` —
  `promptTransform` → `promptTransform?.wireSafe` (01#2). The block cannot drop ITSELF.
- `Sources/Mold/Generate/ModelStore.swift:75` — `\.isGenerator` → `\.isPictureMaker` (06 §5.2).

Plus `Tests/MoldTests/QueuePaneTests.swift` (a test, allowed by the brief) for `7d029b3f`.
