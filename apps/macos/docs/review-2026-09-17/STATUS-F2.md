# Lane F2 · Reuse + retained source media

Worktree `.claude/worktrees/agent-a6ef69c6b04a01fea`, branch
`feat/macos-native-app`, on `ddf6fc90`.

| # | item | status | commit | test |
|---|---|---|---|---|
| 1 | `OutputMetadata` widened to everything `applyMetadataToForm` reads | fixed | `ec31b545` | `ProvenanceTests` (11) |
| 2 | `RenderDraft(reusing:)` restores that table; media cleared | fixed | `419ccf6a` | `RenderDraftReuseTests` (13) |
| 3 | probe every copy; disclose only when the print's own markers say so | fixed | `9ea5a7da`, `79c07fd2` | `RetainedSourceMediaTests` (17), `ReuseTests` (10) |
| 4 | same-host reuse session on the one-child submit | fixed | `9ea5a7da`, `79c07fd2` | `ReuseTests.onTheMachineThatMadeItTheHostHydratesItself` |
| 5 | cross-host download-and-inline relay | fixed | `9ea5a7da`, `79c07fd2` | `ReuseTests.onAnotherMachineTheBytesTravelAndTheHandleDoesNot` |
| 6 | surfaces: Library tile + menu bar, mesh viewer | fixed | `79c07fd2` | `MeshViewMenuTests`, `MenuSurfaceTests` |

## What the port had to get right

- **`convertFromSnakeCase` runs `"sha256s".capitalized`, and `256` is a word
  boundary to Foundation.** `edit_image_sha256s` therefore arrives as
  `editImageSha256S`; spelled the way a person would spell it, it decodes as
  `nil` on every print and nothing says so. Both keys keep the odd storage
  name and are read through `editImageDigests` / `identityDigests`.
- **`members` is omitted, not empty**, on every unavailable inventory — which
  is the commonest answer there is. A required field would fail to decode
  hal9000's own reply for a text-to-image print.
- **Disclosure is decided by the PRINT, never by the answer.** A
  text-to-image print's archive entry has no pins either, so the host can only
  call it `unavailable_legacy`; without `disclosable` every picture ever made
  would be told to reattach a source it never had.
- **`pipeline` is restored only when `pipeline_requested` is true.**
  `pipeline` records what RAN; restoring it on a print that named nothing pins
  a choice nobody made.
- **The canvas intent is `.manual` after a reuse.** Left following a source,
  it would be re-derived the moment the retained source re-attaches and the
  print would come back a different shape.
- **Parking is the ADOPT's job, not the restore's.** Reuse writes into the
  live slots; `GenerateController.adopt(keepingDraft: true)` clamps, coerces
  the format and parks what the target recipe does not advertise. Pinned by
  `aControlTheTargetRecipeCannotTakeIsParkedRatherThanLost`, which walks a wan
  flow shift onto an SD recipe and back.
- **A handle is never held across an edit, because one is never held.** The
  host binds it to the sha256 of `target_request`, so the mint happens inside
  the submit against the exact request going out, and it is consumed by the
  next call. `BatchAdmission` excludes it from its coding keys, so it is
  structurally unable to reach a log, a persisted draft or a recovery record
  (`theHandleRidesAHeaderAndNeverTheBody`). An edit to a hydrated ROLE is
  handled by the vacancy check at mint time rather than by watching for it.
- **A batch of four takes the relay even at home.** A session binds exactly
  one child; refusing four siblings that want the same picture would be a rule
  with no reason behind it. `BatchAdmission.retainedMediaBatchRefusal` keeps
  the server's own sentence and code as the structural guard, pinned by
  `aBatchOfMoreThanOneIsRefusedInTheHostsOwnWords`.

## Second round — the adversarial review's findings

| id | status | commit | test |
|---|---|---|---|
| F2#1 the authority was never put down | fixed | `a4253d02` | `ReuseTests` (4 new) |
| F2#2 every refusal killed the render with raw prose | fixed | `bd0ac43d` | `RetainedSourceMediaTests` (2), `ReuseTests` (3) |
| F2#3 the long-clip warning told you to attach what you had | fixed | `a4253d02` | `aLongClipWithAPictureAlreadyAttachedIsToldNothing` |
| F2#4 the ceiling was checked after the bytes were resident | fixed | `b4d468d7` | `refusesARelayThatCouldNeverBeSentBeforeFetchingAByte` |
| F2#5 an `unknown` state erased a concrete answer | fixed | `c2a1aaac` | `aStateThisBuildCannotNameNeverErasesAConcreteAnswer` |
| F2#6 only one of the two `Sha256S` keys was pinned | fixed | `08c0f727` | `theOtherSha256SKeyIsPinnedToo` |
| F2#7 a third "Use These Settings" literal | fixed | `c2a1aaac` | `LibraryMenuPlanTests`, `MeshViewMenuTests` |
| deferral (a) the chain carried nothing | fixed | `82138ab9` | `aLongClipReusedFromAPrintGetsItsPictureInTheWell` (+2) |

### What the review got exactly right

- **`clear()` had zero callers.** Three seams close it: the draft the reuse
  landed in is recorded after the adopt and ANY edit puts the authority down;
  the submit TAKES it rather than reading it, so a print the machine can no
  longer honour refuses exactly one render instead of every one after it; and
  the pending attachment is now SAID (which print, which machine) with an ✕.
  The available path used to be silent, which is the disclosure inverted.
- **Only two refusal codes are worth asking twice** — `_INVALID` and
  `_ARCHIVE_CHANGED` describe the handle, not the archive. One re-mint, then
  the relay, then this app's own sentence with the way forward in it.
- **The chain deferral was avoidable.** `POST /api/chain-jobs` still redeems no
  session, but the chain WIRE carries the bytes per stage, so the relay needs
  no server support: `ChainRetainedSource` fetches the picture into the
  draft's own SOURCE WELL in `startRun`, before `controller.submit`, and the
  render goes out as an ordinary long clip. `ChainSubmission.take`'s
  synchronous ordering is untouched because the authority is taken BEFORE the
  await, so a second press finds none and takes the ordinary path. The well is
  a bonus: on that route the picture stops being invisible authority and
  becomes an attachment a person can see and remove. The warning survives for
  what a chain body genuinely cannot carry (a mask, a face), and only fires
  when something would actually have been hydrated.

## Deferred, with the reason

- **`reference_weight` is not restored**: `mold_core::OutputMetadata` records
  no such field (`types.rs:3127-3407`), so there is nothing to restore from.
- **`mesh`, `ic_lora_control`, `retake_range`, `spatial_upscale`,
  `temporal_upscale` are decoded but not applied**: this app has no mesh
  controls and no request fields for the four LTX-2 ones, so applying them
  would mean inventing wire fields. Decoding them keeps the metadata honest
  for the lane that adds the controls.
- **The negative prompt's explicit-empty marker is decodable but inert.**
  Absence and `""` are distinguishable on `OutputMetadata`; the draft carries
  no recipe negative default to tell them apart against (`adopting` never
  applies `recipe.defaults.negativePrompt`), so both restore as empty. Closing
  it means applying that default, which is a Generate-controls change, not a
  reuse one.
- **`references` is deliberately absent from the role table.** This app models
  no H3 reference descriptors, so there is nothing for retained reference
  bytes to attach to; studio hydrates them only when every descriptor is
  descriptor-only, and with no descriptors that can never be true. The relay
  refuses the role BY NAME rather than dropping it quietly.

## Cross-lane edits (all small and additive)

- `MoldApp.swift`: `ReuseStore` state + `.environment(reuse)`. There is no
  `AppStores.swift` on this base, so there was no `// NEW STORES GO HERE`
  marker to use. At integration take MAIN's file wholesale and move the two
  lines to `AppStores` (`let reuse: ReuseStore` in the let-list,
  `reuse = ReuseStore(hosts: hosts)` at the marker after `hosts`) plus
  `.environment(stores.reuse)` in `AppStores+Environment.swift`.
- `GenerateController+Run.swift`: one `retained:` parameter and one line
  calling `RetainedMedia.hydrated`. The type is now 603 lines against the
  advisory 600 budget; every piece of behaviour is in two new files.
- `GeneratePane.swift`: `@Environment(ReuseStore.self)`, `.reuseNotice(reuse)`,
  and the hydration passed to `submit`.
- `LibraryPane.swift` / `LibraryPane+Wiring.swift`: `reuseStore` environment
  and the probe in `reuse(_:)`.
- `MoldBackend.swift`: `MoldRetainedMediaBackend` added to the composition.
- `HTTPBackend+Transport.swift`: `post` takes optional headers.
- `HTTPBackend+Generation.swift`: `submit` sets the header and asks the batch
  refusal.
- `MeshViewMenu.swift` / `MeshCanvas*.swift` / `LibraryViewer+Mesh.swift`:
  the `reuse` case, its `canReuse` gate (defaulted, so no other caller moves),
  and the Library's handler. `LibraryMenuPlan.reuseTitle` is the one
  declaration all three surfaces read.
- `MoldBackend.swift`'s composition line and `LibraryPane.swift`'s
  `@Environment` block are the two the review names as certain conflicts with
  main's upscale/Sparkle lanes; both merges keep every property.

**Type budget.** `GenerateController` sits at 603 against the advisory 600.
The three lines this lane added are a parameter and one call; the reviewer's
suggested cut — moving `GenerateController+Expand.swift` (149 lines) into its
own store — is left to the integrator, since it touches the expansion
surfaces rather than anything here.

## Fixtures

Both captured read-only from **hal9000**, `http://100.123.198.98:7680`
(keyless), server **0.29.0 (b015496e 2026-09-16)**, on **2026-09-17**, and
each carries that provenance in its own `_captured` header:

- `Fixtures/provenance-hal9000.json` — 14 verbatim prints from
  `GET /api/gallery`, chosen so that between them they carry every field reuse
  restores, including a real sequence print whose prompt is three stages
  newline-joined.
- `Fixtures/source-media-hal9000.json` — three verbatim bodies from
  `GET /api/gallery/source-media/{filename}`: a `source_image` set, an
  `edit_images` set, and the `unavailable_legacy` reply with `members` omitted.

Synthetic metadata is used only for fields nothing on that host has ever
produced (CFG++, wan distill, guidance overrides, ControlNet, `video_only`)
and is labelled `Synthetic` at its one construction site.

## UAT owed (none of it can be done from here)

1. Reuse a print with a retained source image on the **keyless** hal9000:
   the recipe restores, no sentence appears, and the render comes back with
   the source applied (a session, so no download happens).
2. The same on a **keyed** host: same result, with the key set; then reach
   the same host with the key removed and confirm the API-key sentence.
3. **Cross-host relay**: reuse a hal9000 print with the render pointed at
   another machine, and confirm the picture rides along.
4. A **legacy** print (made before retained media): silence, because its
   metadata records no conditioning bytes.
5. A **text-to-image** print: silence.
6. A **batch of four** from a reused print: four renders, all with the source.
7. A **sequence** print: the first stage's prompt in the composer, never the
   newline-joined wall.
8. A **long clip** reused from a print: the picture appears in the source
   well before it renders, and the clip starts from it.
9. Reuse a print, then **retype the prompt**: the "Using the source media
   from…" line goes away, and the render is not conditioned on it.
10. Reuse a print, then **delete it on its machine** and press Develop: one
    sentence naming what happened, and the NEXT press renders normally.
11. Reuse a print and press Develop **twice**: only the first carries the
    picture.

## Gates

`make lint` green (advisory size notes only). Package `swift test`: 857 in 46
suites. App bundle `xcodebuild test` under the shared lock: 620 in 92 suites.
