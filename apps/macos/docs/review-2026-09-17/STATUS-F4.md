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
  the durable job and following it again (`ChainRun.reattach`).

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

`AdvancedControls` has a `stgBlocks` free-text field; `Ltx2GuidanceOverrides`
round-trips it as `[Int]` (`guidanceOverridesFromWire`'s `join(", ")`).

## Deferred, and why

- **`upscale-then-fit`** is not authored: there is no client-side upscale to run
  first (that verb is lane F3's). The policy round-trips through the wire, so a
  print made elsewhere keeps its provenance rather than being rewritten.
- **A mask painted over a previous fit is replaced, not rescaled.** A rescaled
  mask is a plausible-looking lie about which pixels somebody chose. What
  survives a re-fit is what the fit implies: the `pad-repaint` bands.
- **Whole-queue pause** is listed under F4 in `PLAN.md` but belongs to the
  upscale/activity/queue-pause lane per my task; not touched.
- **Recovering a chain whose CREATE was lost to a relaunch.** The operation id
  makes a retry safe inside one run, but there is no `by-operation-id` lookup
  route for a chain the way there is for a batch. Documented in `PendingChain`.

## Cross-lane edits (smallest possible, please sequence)

- `Packages/MoldClient/Sources/MoldClient/BatchStatus.swift` — a `public init`
  on `BatchResult`, so a chain's one result can be built with no batch behind it.
- `Packages/MoldClient/Sources/MoldClient/ClipLength.swift` — a `public init`
  on `ClipLengthBounds`, so the slider's ceiling can be widened for a chain.
- `Packages/MoldClient/Sources/MoldClient/MoldBackend.swift` — `MoldChainBackend`
  added to the composition.
- `Packages/MoldClient/Sources/MoldClient/CanvasIntent.swift` — `Codable` (new
  file this lane added; listed only because the draft descriptor needs it).
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

1. **plato (`100.105.134.43:7680`), an LTX-2 tier**: drag the Length slider past
   97 frames — the note under it should read "Rendered as N clips and stitched
   into one video" — press Generate, and watch the capsule count "Clip 2 of 3".
   One print lands, and `mold jobs list` on plato should NOT show it as an
   authored sequence.
2. **Stop, twice**: once mid-clip (the job must disappear from plato's queue),
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
