# Lane A · MoldClient wire — ledger

Branch `worktree-agent-a2d5903403e02a402`, off `feat/macos-native-app` at `34cfa484`.
Package suite 417 → 507 tests, app bundle 380, `make lint` green.
Nothing deferred.

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
| 01#14 no buffering policy | fixed | `3c7bb54f` + `d9e9636c` | `EventDeliveryTests` (4), `everyStreamStatesItsBufferingPolicy` |
| 01#21 a Task-hop per byte | fixed | `3c7bb54f` | `theLineReaderReadsNoFurtherThanItsConsumerAsks` (+ 1) |
| 01#15 `reference_weight` unreachable | Lane B | — | — |
| 01#16 key rides a cross-origin redirect | fixed | `4c4c584a` + `60ee21f5` | `RedirectGuardTests` (6) |
| 01#18 tag control characters | fixed | `71f55d5c` | `ClientTagControlTests` (5) |
| 01#19 `CollectionShelf.hidden` inverts studio | fixed | `430e9d91` | `aShelfIsHiddenWhenAnyMachineHidesIt` |
| 01#20 no reference-upload lease on transfer | out of scope by README (upload sessions) | `412a0bc9` | `aTransferTooLargeToPostSaysWhatTheLimitIsAndWhereTheJobStayed` |
| diagnosability: `os.Logger`, decode path | fixed | `b6da85a1` + `9e8e8aa3` | `DecodingFailureTests` (5), `RouteRedactionTests` (6) |
| test gap 1 `GenerateRequest.encode` exhaustive | Lane B | — | — |
| test gap 2 `moves` with a neighbour | fixed | `ad283395` | `aBatchLandsContiguousWhereItWasDropped` |
| test gap 3 `VideoOnlyPolicy` untested | fixed | `672d1ac1` | `VideoOnlyPolicyTests` (5) |
| test gap 4 clip park/restore | Lane B | — | `DraftMedia*` is Lane B's |
| test gap 5 downloads `snapshot` frame | fixed | `672d1ac1` | `DownloadSnapshotTests` (4) |
| test gap 6 `MediaToken.playableURL` | fixed | `3a273912` | `MediaTokenTests` |
| test gap 7 `ExportOptions.forVideo`/`.forMesh` | fixed | `672d1ac1` | `ExportOptionsTests` (5) |
| — stub transport raced between suites | fixed | `466f18c9` | — |
| — keyboard batch move pinned the 01#1 bug | fixed | `7d029b3f` | `aBatchMovesFromTheKeyboardAsAscendingCalls` |

## Adversarial review, round 2

| id | status | commit | test |
| --- | --- | --- | --- |
| R1 `bufferingOldest` lost the NEWEST state; buffers stacked | fixed | `d9e9636c` | `EventDeliveryTests` (4), `everyStreamStatesItsBufferingPolicy` |
| R2 `RedirectGuard` unreachable by other key-setting callers | fixed | `60ee21f5` | `anyCallerThatSetsTheKeyCanAttachTheGuard` |
| R3 the log wrote filenames, ids and search text | fixed | `9e8e8aa3` | `RouteRedactionTests` (6) |
| R4 escaping contract accepted either escaper anywhere | fixed | `a3705497` | `everyDynamicRouteComponentGoesThroughAnEscaper` |
| R5 `refusalBody` had no deadline | fixed | `027fa6b7` | `RefusalBodyTests` (4) |
| R6 `consume(contentsOf:into:)` had no production caller | deleted | `412a0bc9` | — |
| R7 stale `ModelFamilyContractTests` header | fixed | `412a0bc9` | — |
| R8 imported animated WebP reads as a still | documented limit | `412a0bc9` | `anImportedAnimatedWebpIsAKnownMisclassification` |

**R1, what the fix is.** `.bufferingOldest(n)` keeps the OLDEST and refuses every new yield once
full, so on `/api/events` -- where each frame is state the client is never told again -- a burst
lost the most recent state permanently, with the stream still open and no resync trigger but a
changed instance id. Every policy is `bufferingNewest` now, so what survives a burst is CURRENT,
and `yieldOrResync` follows any drop with `.resyncRequired` (the server's own word for it, already
routed to every listener by `HostStore.deliver`; the marker is always kept because `newest`
evicts the oldest). `stream(_:timeout:)` was a SECOND buffer stacked under each route's; it is now
a lazy `SSEStream` like the two parsers under it, so a route's declared capacity is the only one
there is — pinned by naming the four files allowed to construct a buffer.

## Adversarial review, round 3

| id | status | commit | test |
| --- | --- | --- | --- |
| R9 one `.resyncRequired` per DROPPED FRAME | fixed | `lane-a-r3` | `EventDeliveryTests` (8) |

**R9.** `yieldOrResync` was stateless, and under `.bufferingNewest` EVERY yield reports `.dropped`
once the buffer is full — so a sustained burst produced a marker per frame: each took a slot and
evicted another real frame (measured: 5 markers and only 3 state frames left in an 8-slot buffer),
and each made every consumer fire its own re-read. The repair path amplified the overload.

`EventOverflow` bounds markers per overflow EPISODE. First drop → one marker, `overflowing` set;
further drops → nothing, just `droppedSinceMarker`. The episode ENDS on a yield reporting
`remaining >= max(1, capacity / 2)` — hysteresis, because a buffer that is exactly full enqueues
with `remaining == 0` and drops on the very next yield, so ending on the first successful enqueue
would flap once per frame and reproduce the storm; half the buffer is the consumer demonstrably
DRAINING rather than briefly keeping up, and it is never zero, so the closing marker always has
somewhere to land. At most two markers an episode, and the state read after the closing one is
current.

Two holes the red tests exposed, both fixed here: the OPENING marker is best-effort — a burst that
keeps going evicts it like anything else — so the closing one is the guarantee; and a stream that
ENDS mid-burst gets no episode end at all, which is a server closing the connection under load and
is NOT repaired by the reconnect (the instance id is unchanged). `hasUnannouncedLoss` is what
`events()` checks before either `finish`, yielding one last marker where it is newest and certain
to survive.

Thread safety: `EventOverflow` is a `struct` declared inside `events()`' own producing `Task`, so
that task is its only owner and there is nothing to lock. It constructs no stream, so the
`everyStreamStatesItsBufferingPolicy` file allowlist is unchanged.

**R2, the public surface for the integrator to route to Lane C:**

```swift
public final class RedirectGuard: NSObject, URLSessionTaskDelegate, Sendable {
    public init(origin: URL)                                   // MoldHost.baseURL
    public func sanitized(_ newRequest: URLRequest) -> URLRequest
    public static let keyHeader = "X-Api-Key"
}
```

Call site, one argument on the request `ThumbnailCache` already makes:

```swift
let (data, response) = try await session.data(
    for: request, delegate: RedirectGuard(origin: host.baseURL))
```

**R3, the rule.** `RouteTemplate.redacted` keeps `/api/<family>` and redacts the rest, dropping any
query whole. No table to rot: the first component after `/api` is fixed in every route this package
builds and the second is not, so exactly one is kept — pinned by a source test that fails if a route
literal ever interpolates into that position. An unrecognised path is redacted whole.

**01#21, why the delegate route was not taken.** Bridging `URLSessionDataDelegate` means an
`AsyncThrowingStream<Data, Error>`, which reintroduces exactly the unbounded buffer 01#14 exists to
remove — a redesign, not a follow-through. What the finding actually cost (a cross-task hop and an
`[String]` allocation per byte) is gone.

## Found while writing the missing tests

- **`DownloadsListing` threw on a host that omits `active_jobs`** (`672d1ac1`). The field is
  `#[serde(default)]` in Rust precisely because an older host sends only `active`, but a
  synthesized `init(from:)` requires every non-optional key — and `downloadEvents` decodes inside
  a `try?`, so the snapshot frame (the only way this app learns about a `mold pull` at a terminal)
  vanished silently and the popover stayed empty. Exactly the failure gap 5 predicted.
- **`Tests/MoldTests/QueuePaneTests.swift` asserted the 01#1 bug** (`7d029b3f`): `[1, 2]` is the
  plan that splits the batch around the row it moved past. Rewritten to assert the landed ORDER.

## Out of scope

- **01#20** — out of scope by README (upload sessions). A reference-upload lease on transfer is the
  whole upload-session feature (`prepareReferenceUploadBatch` + the session routes), which the
  README scopes out by design, so what has to hold instead is that the limit is EXPLAINED:
  `aTransferTooLargeToPostSaysWhatTheLimitIsAndWhereTheJobStayed` walks a raw 413 through to the
  plan's own sentence.

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
- **01#21** is fixed, not half-fixed: the harm the finding names (a `Task` hop and an `[String]`
  allocation per byte, and an unbounded buffer at that level) is gone. The source remains
  `URLSession.AsyncBytes` because the only way off it is a `URLSessionDataDelegate`, which means
  an `AsyncThrowingStream<Data, Error>` — reintroducing the very buffer 01#14 removes.

## Known flakes in the app bundle (NOT from this lane)

Four full `xcodebuild test` runs of the 380-test app bundle: three green, one failure each in
`QueueStoreLiveTests.aJobFrameReReadsThatMachineOnceForABurst` and
`ModelActionsTests.deleteAsksFirstAndTheFakeRecordsNothingUntilPerform`, in different runs.
Neither suite is touched by this lane's diff, neither involves a stream, an escaper, a log or a
redirect, and `QueueStoreLiveTests` passed three consecutive isolated runs. Both are the same
shape: an assertion immediately after an async action that a `settle` on a CALL COUNT does not
prove has landed. Reported rather than chased — they belong to the lanes that own those files.

## Cross-lane edits

Two one-line changes in files Lane B owns; both are the call site of something added here.

- `Packages/MoldClient/Sources/MoldClient/GenerateRequest+Encoding.swift:58` —
  `promptTransform` → `promptTransform?.wireSafe` (01#2). The block cannot drop ITSELF.
- `Sources/Mold/Generate/ModelStore.swift:75` — `\.isGenerator` → `\.isPictureMaker` (06 §5.2).

Plus `Tests/MoldTests/QueuePaneTests.swift` (a test, allowed by the brief) for `7d029b3f`.
