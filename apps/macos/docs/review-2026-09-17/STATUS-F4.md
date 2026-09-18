# Lane F4 · Generate controls

Branch `worktree-agent-a24ba166817b9671f`, on `feat/macos-native-app` at `a5172d65`.

| # | item | status | commit | test |
|---|---|---|---|---|
| 1 | IP-Adapter weight slider | fixed | `2f6c5e40` | `ReferenceWeightTests`, `ReferenceWeightParkTests` |
| 2 | scheduler picker | fixed | `2f6c5e40`, `79a3edd1` | `AdvancedControlsTests`, `AdvancedControlsParkTests` |
| 3 | CFG++ (+ TS contract test) | fixed | `2f6c5e40`, `79a3edd1` | `CfgPlusContractTests` |
| 4 | Wan recipe (`sample_shift`, distill high/low) | fixed | `2f6c5e40`, `79a3edd1` | `AdvancedControlsTests` |
| 5 | LTX-2 `guidance_overrides` | fixed | `2f6c5e40`, `79a3edd1` | `AdvancedControlsParkTests` |
| 6 | `source_fit` + canvas intent | fixed | `22d9100a` | `SourceFitTests`, `CanvasIntentTests` |
| 7 | long clips as an ephemeral chain | fixed | `b8219731`, `4cf27518` | `ChainRoutingTests`, `ChainLimitsTests`, `ChainRunTests` |
| 8 | draft persistence across launches | fixed | `2beba105` | `DraftPersistenceTests` |
| 9 | `RowAction` menus on every new row | fixed | `2f6c5e40`, `79a3edd1`, `22d9100a` | `ReferenceWeightTests` |

### Second round — the adversarial review's findings

| id | status | commit | test |
|---|---|---|---|
| S1 chain never survived a relaunch | fixed | `5eb8ed14` | `ChainRecoveryTests` (5) |
| S2 a chain preempted the run queue | fixed | `5ca73d0f`, `faa53598` | `ChainQueueTests` (4) |
| S3 a stale start clobbered the next | fixed | `d9cc66c6` | `ChainRunTests.aStaleStartNeverClobbersTheOneAfterIt` |
| S4 a dropped stream lost the job | fixed | `646b56ce` | `ChainRunTests.adroppedStreamReconnectsAndKeepsTheJob` |
| S5 `coercedForMaskless` unused | fixed | `2f47bc14` | `SourceFitAdoptTests` |
| S6 a painted mask was destroyed | fixed | `60e92e62` | `SourceFitMaskTests` (3, real pixels) |
| S7 canvas stopped following a source | fixed | `2f47bc14` | `SourceFitAdoptTests` |
| S8 unreachable `upscale-then-fit` selection | fixed | `76fdd5a5` | `SourceFitTests` round-trip + reconcile |
| S9 `batchSize` ignored by the chain | fixed | `76fdd5a5` | `ChainQueueTests.abatchOfLongClipsRendersEveryCopyItWasAskedFor` |
| S10 dead code + a wrong ledger claim | fixed | `76fdd5a5` | — (`ClipRouting.refusal` wired up rather than deleted) |
| S11 `onDisappear` is not a quit hook | fixed | `76fdd5a5` | — (UAT item 5) |

### Third round — a flaky gate on `macos-26`

`ChainRunTests.aStaleStartNeverClobbersTheOneAfterIt` failed on the GitHub
runner and passed everywhere else. The race was in `FakeBackend`, NOT in
`ChainRun`: `record(_:)` appends the route name before a create suspends on its
gate, so a settle on the call count returned with nothing parked, and
`releaseChainCreate()` BANKED the release for whichever create arrived first --
the second one, on a slow machine. Banking is gone (a release with nothing
waiting is an `Issue.record`), `chainCreatesWaiting` exposes what a release
acts on, and two other tests of the same shape were hardened the same way.
`f123917e`; 20/20 for `ChainRunTests`, `ChainQueueTests`, `ChainRecoveryTests`.

## Decisions worth knowing

- **The legacy-host scheduler heuristic was NOT ported.** Studio has a third
  branch for a host that hands it no recipe at all (`SCHEDULER_FAMILIES` →
  DDIM/Euler-a/UniPC, wan → its solvers). This app draws every control from the
  model's generation profile, a host with no profile is not a target it
  supports, and the coordinator ruled it out. A recipe in hand answers for
  itself; an absent or empty `schedulers` on a recipe IS "none" (the server
  omits an empty list), so there is no picker.
- **CFG++ is the one sanctioned family set**, named once in
  `AdvancedControlsOffered.cfgPlusFamilies` and pinned to
  `generationCapabilities.ts:282` by `CfgPlusContractTests`, which reads that
  file through `RepoFixtures.repoRoot`, asserts set EQUALITY and has a
  `count > 0` floor. It is sent as `cfg_plus: true` or omitted, never `false`.
- **LTX-2's guidance overrides are gated on the PROFILE's shape, not a family
  name**: LTX-2 is the family whose recipes carry a `request_selector.pipeline`
  (`generation_profile.rs:2222`), which is exactly what `require_ltx2_family`
  gates `guidance_overrides` on (`validation.rs:3295`). `modality_scale` is
  absent on an audio-only pipeline rather than offered and refused.
- **`source_fit`'s object keys are camelCase** (`alignX`, `alignY`,
  `upscalerModel`) because the server treats the value as opaque and studio's
  parser reads those spellings. A `CodingKeys` block would have been
  snake-cased by `MoldJSON.encoder`; it is built as a DICTIONARY, whose keys
  the key strategy leaves alone (verified empirically).
- **A chain id is not a batch id**: its own `PendingChain` store, its own cancel
  route, its own stream (and that stream is the one in the package that is NOT
  `latestOnly`, because its frames are deltas). Recovery re-attaches by reading
  the durable job and following it again (`PendingChainRecovery`), which
  `recoverPending()` calls at launch — in the first round that claim was FALSE:
  `reattach` had no caller and `PendingChain` was write-only (review S1).
- **A chain waits in the SAME run queue as a batch** (`QueuedRun`). It never
  preempts the canvas and never cancels a POST in flight; a press while
  something is showing is admitted and waits, which is M8 decision 8 for every
  press whichever door it came through (review S2).
- **A dropped stream is not a settlement.** The follow reconnects with the
  `2^n`/32 s backoff `HostStore+Events.watch` uses, re-reads the job each time,
  and forgets the record only on a TERMINAL state from the host (review S4).
  `paused` is shown and offers Resume: a host restart PARKS an ephemeral chain
  with everything it needs to continue.
- **A re-fit composes a painted mask, never replaces it** — `buildMask`'s rule
  (`sourceFitCanvas.ts:78-96`). The first round's "a rescaled mask is a
  plausible-looking lie" was wrong about what the other surfaces do, and the
  code it justified deleted an inpaint mask on a canvas nudge and on every
  re-appearance of the Source well (review S6).
- **`adopting` re-consults both the fit and the canvas intent**: `pad-repaint`
  is coerced onto a recipe with no mask path, and a canvas whose recorded
  intent still says "follow the source" keeps following it across a model
  switch — the other half of #1166 (review S5, S7).

## Metadata keys for the Reuse lane (F2)

Every new draft field is settable in one line. `OutputMetadata` will need these
keys (`types.rs` field names, snake_case on the wire):

| draft field | metadata key |
|---|---|
| `media.referenceWeight` | `reference_weight` |
| `advanced.scheduler` | `scheduler` |
| `advanced.cfgPlus` | `cfg_plus` |
| `advanced.sampleShift` | `sample_shift` |
| `advanced.distillStrengthHigh` / `Low` | `distill_strength_high` / `_low` |
| `advanced.stgScale` … `skipStep` | `guidance_overrides.{stg_scale, stg_blocks, rescale_scale, modality_scale, skip_step}` |
| `media.sourceFit` | `source_fit` (parse with `try? MoldJSON.decoder.decode(SourceFit.self, …)`) |
| `canvasIntent` | none — restore as `.manual`, and pass `preserveReplacement: true` to `attachSourceShape` so a restored canvas is not re-armed |

A reused `source_fit` of mode `upscale-then-fit` is normalised to its inner
policy by `DraftMedia.reconcile`: this app has no client-side upscale to run
first, and leaving it would bind the Fit picker to a selection matching no row.
Set `draft.media.sourceImagePixels` alongside any restored source image, or the
canvas cannot follow it across a model switch.

`AdvancedControls` has a `stgBlocks` free-text field; `Ltx2GuidanceOverrides`
round-trips it as `[Int]` (`guidanceOverridesFromWire`'s `join(", ")`).

## Deferred, and why

- **`upscale-then-fit`** is not authored: there is no client-side upscale to run
  first (that verb is lane F3's). The policy round-trips through the wire, so a
  print made elsewhere keeps its provenance rather than being rewritten.
- **Whole-queue pause** is listed under F4 in `PLAN.md` but belongs to the
  upscale/activity/queue-pause lane per my task; not touched.
- **Recovering a chain whose CREATE was lost to a relaunch.** The server uses
  the `x-mold-operation-id` AS the job id
  (`routes_chain_jobs.rs:142-146`), so replaying one is idempotent by
  construction — but nothing in the app retries a create, and across a relaunch
  the body is gone anyway. What IS recovered is every job the host named: the
  id is written the moment the create answers. Documented in `PendingChain`.

## Cross-lane edits (smallest possible, please sequence)

- `Packages/MoldClient/Sources/MoldClient/BatchStatus.swift` — a `public init`
  on `BatchResult`, so a chain's one result can be built with no batch behind it.
- `Packages/MoldClient/Sources/MoldClient/ClipLength.swift` — a `public init`
  on `ClipLengthBounds`, so the slider's ceiling can be widened for a chain.
- `Packages/MoldClient/Sources/MoldClient/MoldBackend.swift` — `MoldChainBackend`
  added to the composition.
- `Packages/MoldClient/Sources/MoldClient/CanvasIntent.swift` — `Codable` (new
  file this lane added; listed only because the draft descriptor needs it).
- `Sources/Mold/Generate/PromptPanel+Actions.swift` — the Resume button, and
  Generate's disabled state now also asks the routing.
- `Sources/Mold/Generate/RunCanvas.swift` — ONE case label:
  `case .running, .runningChain:`. The mesh lane owns `RunCanvas+Result.swift`,
  which is untouched.
- `Sources/Mold/Generate/GeneratePane+Models.swift` — adopts a restored model
  through the existing path.
- `Tests/MoldTests/FakeBackend.swift` — the four chain routes and
  `chainLimits`, plus a `holdsChainCreate` gate mirroring `holdsSubmit`.
- `Packages/MoldClient/Tests/MoldClientTests/StreamBufferingTests.swift` — the
  buffering inventory gained `HTTPBackend+Chain.swift`.
- `Packages/MoldClient/Tests/MoldClientTests/GenerateRequestExhaustiveTests.swift`
  — seven new fields in the reflective fixture.

## UAT owed

Nothing here has been exercised against a real machine. What needs a human:

1. **workstation (`100.105.134.43:7680`), an LTX-2 tier**: drag the Length slider past
   97 frames — the note under it should read "Rendered as N clips and stitched
   into one video" — press Generate, and watch the capsule count "Clip 2 of 3".
   One print lands, and `mold jobs list` on workstation should NOT show it as an
   authored sequence.
2. **Stop, twice**: once mid-clip (the job must disappear from workstation's queue),
   and once in the instant after pressing Generate, before the job id comes
   back. The second is the sequence the unit test simulates; it is worth seeing
   the GPU actually stop.
3. **A text-only wan tier** (`wan21-t2v-1.3b:bf16`) past 121 frames: the slider
   must stop there and say why, in the server's own sentence.
4. **`source_fit`**: attach a 16:9 photograph to an SD1.5 recipe on a 1:1
   canvas, switch the Fit control through all four modes, and check the well's
   preview changes and that `pad-repaint` leaves the bands repainted in the
   result. Then choose a size by hand and attach ANOTHER picture — the canvas
   should move (studio's rule) and the intent should re-arm.
5. **Draft persistence**: type a prompt, set a solver and a flow shift, quit
   with ⌘Q, relaunch — everything but the media comes back, and no picture is
   claimed. Then corrupt `~/Library/Application Support/io.utensils.mold.native/
   generate-draft.json` by hand and relaunch: the pane opens empty and the file
   is parked as `.corrupt`.
6. **A host too old to publish `/api/capabilities/chain-limits`** — the Length
   slider must still work off the ported constants, with no banner.
7. **Quit mid-chain and relaunch** — the same job is followed again at the clip
   the host says it reached. Then restart `mold serve` under a running chain:
   the capsule must read "Paused after clip N of M" and Resume must continue it
   rather than start it over.
8. **Generate an image, then immediately a long clip** — the picture must land
   on the canvas and the clip must follow it, with neither banner nor loss.
   Then a batch of three long clips: three prints, not one.
