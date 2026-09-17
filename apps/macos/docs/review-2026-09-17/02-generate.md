# Peer review — `apps/macos` Generate pane

Reviewed at `a9f42603` (the held-batch fix landed mid-review; everything below is against that
commit, and the held-spinner bug it fixes is **not** reported). Read-only review: nothing built or run.
Counterpart claims were checked on both sides (`desktop/`, `studio/`, `crates/`) unless a finding says
otherwise.

---

## HIGH

### 1. The source-image well is hidden for every recipe that advertises reference images — including `combines` and `exclusive`, where both wells must coexist

- **kind:** bug / parity-gap
- **where:** `apps/macos/Sources/Mold/Generate/PromptPanel.swift:84-89`
- **counterparts:** `studio/lib/sourceMediaPlan.ts:109-126`; `crates/mold-core/src/generation_profile.rs:905-921`
  (sd15/sdxl → `Combines`), `:872-888` (Klein → `Exclusive`); root `CLAUDE.md` "Additive-references
  invariant" and "Reference-images invariant"
- **confidence:** high (single call site, verified by grep — `SourceImageWell(` appears exactly once,
  in the `else if` branch)

`PromptPanel.prompt(_:)` renders `ReferenceStrip` **or** `SourceImageWell`, never both:

```swift
if let references = recipe.capabilities.referenceImages, references.mode.isVisible {
    ReferenceStrip(capability: references, draft: $draft)
} else if Self.showsSourceWell(for: recipe) {
    SourceImageWell(draft: $draft)
}
```

The server advertises `source_relation: Combines` with `mode: Adjustable` for **every sd15 and sdxl
recipe unconditionally** (`generation_profile.rs:905-921`) — `Combines` is precisely the relation that
"rides WITH img2img, inpaint, ControlNet and a LoRA rather than replacing any of them". So on the two
families most people actually use for img2img, the source well never renders: **img2img, denoise
strength and inpainting are unreachable in the native app.** The Refine group compounds it — with no
source attached, `RefineGroup.maskRow` returns `.needsSource` and the inspector prints "Add a source
picture first." next to a pane that offers no way to add one (`RefineGroup.swift:94-97`). FLUX.2
[klein] (`Exclusive`) has the same outcome: desktop renders both wells and parks the unused one with
its media kept (`resolveExclusiveWells`), macOS shows only the strip.

The draft layer is already correct — `DraftMedia+Reconcile.swift:48-52` parks the source only for
`exclusive`/`replaces` and deliberately keeps it for `combines` — so this is purely the view's
`else if`. Note that `GenerateInspectorTests.swift:77` asserts `PromptPanel.showsSourceWell(for:)`
is `true` and still passes: the predicate is right, the branch it feeds is not.

**Fix:** render the strip and the well from the relation, not from strip-presence: `replaces` → strip
only; `combines` → both, always; `exclusive` → both, with the last-written one active and the other
parked (port `resolveExclusiveWells`). Pin it with the existing `recipe-sd15.json` fixture, which
already carries `"source_relation": "combines"`.

---

## MED

### 2. Stop during `.submitting` cancels the *previous* batch, orphans the new one, and forgets its recovery id

- **kind:** bug / race
- **where:** `apps/macos/Sources/Mold/Generate/GenerateController+Queue.swift:74-83`,
  `GenerateController+Run.swift:32-62`
- **confidence:** high (code reading); the behaviour is untested — `RunQueueTests` always awaits the
  submit response before calling `stop()`

The Stop button renders whenever `run.isBusy`, which includes `.submitting`
(`RunState.swift:12-17`, `PromptPanel+Actions.swift:36`). During `.submitting`, `activeBatch` still
holds the *previous* batch (it is never cleared on settle) or is `nil`. So:

- **First-ever render:** `stop()`'s `guard let active = activeBatch else { return }` fires and Stop does
  **nothing at all** — no cancel, no state change, the button stays.
- **Any later render:** Stop cancels the previous, already-settled batch id on the host, `forget`s the
  *previous* client batch id, sets `.idle`, and cancels `runTask`. The in-flight `submit` then throws
  `CancellationError`, which runs `PendingBatch.forget(admission.clientBatchId)` (line 51) and
  `run = .failed(...)` (line 54) over the `.idle` Stop just wrote. If the POST already reached the
  host — the common case, since cancellation usually lands while awaiting the response — the batch is
  admitted, renders to completion, is never cancelled, and is no longer in `PendingBatch`, so launch
  recovery will never find it either. The user pressed Stop and sees an error banner while the GPU
  keeps going.

**Fix:** track the in-flight admission separately from `activeBatch` (e.g. a `pendingSubmission` id or
a `stopRequested` flag consulted in the `do` branch), and on Stop-during-submit either await the
response and cancel the id it returns, or leave the `PendingBatch` record so recovery reattaches. Clear
`activeBatch` when a batch settles.

### 3. The clip-length slider's ceiling is the family resource guard, not the requestable/renderable one

- **kind:** bug / parity-gap
- **where:** `apps/macos/Sources/Mold/Generate/NumberControls.swift:64`,
  `Packages/MoldClient/Sources/MoldClient/Temporal.swift:82-89` (`snap` clamps to `frames.max` and
  ignores `maxDurationSeconds`, which is decoded at `:71` and read nowhere — grepped)
- **counterparts:** `studio/lib/videoDuration.ts:138-169` (`maxVideoFrames`);
  `crates/mold-core/src/generation_profile.rs:1241-1253` (admission narrows `frames.max` by
  `max_duration_seconds × effective fps`); `crates/mold-core/src/generation_profile.rs:2558-2566`
  (LTX-2 deliberately advertises the 120 fps maximum); `studio/lib/chainRouting.ts:416-433`
- **confidence:** high for LTX-2 (arithmetic is explicit on both sides); medium-high for Wan (the
  engine consequence is inferred, the slider range is certain)

Two separate consequences of using `temporal.frames.max` raw:

- **LTX-2 sends a 422.** The server advertises the grid maximum *at 120 fps* on purpose ("Advertise the
  largest requestable value; admission applies the lower duration-derived cap for the selected FPS"),
  i.e. 601 frames. At the default 24 fps admission caps at `20 × 24 + 1 = 481`. Dragging Length to the
  right-hand end at 24 fps produces a request the host refuses. Web/desktop compute
  `max_runtime_seconds * fps + 4` themselves for exactly this reason.
- **Wan asks for one 257-frame denoise.** `max_frames_for_family_at_fps("wan")` is the flat
  `MAX_FRAMES_GLOBAL` = 257 memory guard, not the checkpoint's trained clip (81/121). Desktop clamps
  the slider to the clip size and either auto-chains or refuses by name
  (`textOnlyAutoChainSingleClipCeiling`, `videoDuration.ts:161-167`). macOS submits a single
  257-frame request, which validation accepts. Chain jobs being out of scope is fine (README); what is
  not fine is offering a length the family cannot render in one pass and saying nothing.

**Fix:** derive the effective ceiling in `TemporalProfile` from `maxDurationSeconds × fps` (clamped to
`frames.max`, snapped down onto the grid), and clamp further to the tier's clip size with an inline
sentence where the app cannot chain past it.

### 4. The IP-Adapter reference weight is decoded and never offered

- **kind:** parity-gap
- **where:** `apps/macos/Sources/Mold/Generate/ReferenceStrip.swift` (no use of `capability.weight`);
  `Packages/MoldClient/Sources/MoldClient/RecipeBlocks.swift:50`
- **counterpart:** root `CLAUDE.md` additive-references invariant — "the slider is gated on
  `weight != null` — never a family name"; `crates/mold-core/src/generation_profile.rs:913-920`
- **confidence:** high

`ReferenceImagesCapability.weight` is a `FloatControl` whose range travels with the capability, and
`RenderDraft+Request.swift:40` faithfully sends `reference_weight` only when it is set — but nothing
ever sets it, so an SD1.5/SDXL image prompt always renders at the server default with no way to dial
it. (Moot today because of finding 1, and it becomes user-visible the moment that is fixed.)

### 5. The placement preview POSTs the whole request — prompt, source image, references, identity photos, tags and collection — on every draft change

- **kind:** design / quality (privacy-adjacent)
- **where:** `apps/macos/Sources/Mold/Generate/GenerateController.swift:111-134`,
  `Packages/MoldClient/Sources/MoldClient/RenderDraft+Request.swift:111-113` (`placementRequest` is
  literally `request(...)`), fired from `GeneratePane.swift:62` on every `draft` mutation
- **counterpart:** `studio/api/generationPlacement.ts:357-402` (`redactGenerationForPlacement` blanks
  the prompt, negative prompt, every media field, every `edit_images` entry and each keyframe image,
  and **deletes** `tags` and `collection` — "a tag or a collection name ('Client X, unannounced') must
  not be fanned out to every candidate host just to price a render")
- **confidence:** high

With a 12 MB source photo attached, every keystroke in the prompt field (debounced 350 ms) re-uploads
~16 MB of base64 to price a render that hasn't been asked for. macOS previews against one host rather
than fanning out, so the privacy exposure is narrower than web's, but the filing text and media are
still sent to a machine for planning, contrary to the documented rule, and the bandwidth cost is real
on a LAN or Tailscale host.

**Fix:** port `redactGenerationForPlacement` into `RenderDraft.placementRequest` — blank the text and
media, drop `tags`/`collection`. Nothing in the planner reads them.

### 6. "Use These Settings" restores eight fields and silently drops every conditioning input

- **kind:** parity-gap
- **where:** `apps/macos/Packages/MoldClient/Sources/MoldClient/RenderDraft.swift:98-112`,
  `apps/macos/Sources/Mold/Library/LibraryPane.swift:139-146`
- **counterpart:** desktop/web Reuse settings + `studio/api/gallerySourceMedia.ts` (retained source
  media probe), root `CLAUDE.md` "Durable gallery source media" and the identity/file-under invariants
- **confidence:** high

`RenderDraft(reusing:)` restores prompt, negative prompt, size, steps, guidance, frames, fps and seed.
It does not restore strength, source image, references, mask, LoRAs, identity, ControlNet, output
format, upscaler, pipeline, audio flags, `save_to_gallery`, or the print's title/tags/collections —
several of which `OutputMetadata` (`GalleryPrint.swift:10-31`) does not even model. Reusing an img2img
print therefore renders a *text-to-image* picture from the same prompt with no indication that the
conditioning was dropped. There is also no retained-source-media probe at all: the server-side feature
exists (`GET /api/gallery/source-media/:filename`) and the root `CLAUDE.md` says "Every client always
asks, and the server is the only authority on what it retained".

**Fix (minimum):** model the remaining metadata fields, restore what is there, and state inline what
could not be restored. The retained-media probe can be a follow-up, but the silent drop should not be.

### 7. Identity photos accept HEIC/TIFF/WebP; the server reads PNG and JPEG only

- **kind:** bug
- **where:** `apps/macos/Sources/Mold/Generate/IdentityGroup.swift:70,77-80`
- **counterpart:** `crates/mold-core/src/identity.rs:831-880` (`header_dimensions`: PNG signature, else
  a JPEG marker walk, else "malformed JPEG")
- **confidence:** high

The open panel offers `.png, .jpeg, .webP, .heic, .tiff`, and `append` base64s whatever it gets with
no size or dimension check. HEIC is the default format of every photo taken on an iPhone and synced to
a Mac, so the likeliest possible identity photograph produces a 422 at submit time ("id_image is a
malformed JPEG: lost marker alignment"), after the whole request has been uploaded. The
identity-photo invariant also asks for the 16 MiB / 8192 px / 32 MP bounds to be refused **inline
beside the control**, not at submit.

**Fix:** restrict the panel to PNG/JPEG (or transcode on import), and validate bytes/pixels inline.

### 8. Source images are never fitted to the canvas, and attaching one never moves the canvas

- **kind:** parity-gap
- **where:** `apps/macos/Packages/MoldClient/Sources/MoldClient/RenderDraft+Request.swift:37-44`
  (raw bytes go straight out); no equivalent of a canvas intent anywhere in `Generate/`
- **counterpart:** `desktop/src/lib/sourceFitPreprocess.ts` (crop/fill/pad-repaint, mask refit,
  optional upscale-then-fit); `studio/lib/outputShape.ts` `CanvasIntent`
- **confidence:** high on the absence, medium on the exact rendered outcome (engine-side resize
  behaviour not traced end to end)

Attach a 3:2 photograph to a 1:1 recipe and the app sends both the photo and a 1024×1024 canvas; the
engine resizes to the latent grid and the render is distorted, with no fit policy, no padding repaint
and no offer to follow the source's shape. Masks stay internally consistent (they are stored in source
pixels, `MaskEditorSheet+Source.swift:29-50`) so this is a quality loss rather than a misaligned
repaint, but it is a control desktop has and this does not.

### 9. A finished batch's picture is replaced by the next queued batch in the same turn

- **kind:** bug / design
- **where:** `apps/macos/Sources/Mold/Generate/GenerateController+Queue.swift:35-39`
- **confidence:** medium-high (the ordering claim is unprovable either way at the SwiftUI level, which
  is the point)

The comment asserts that "the `.finished`/`.failed` write above lands first, so it is observed for at
least one beat before `followNext` replaces it". Nothing guarantees that: `followNext()` enqueues a
main-actor `Task` and SwiftUI coalesces `@Observable` invalidations into one render pass, so the
`.finished` state may never be drawn. Press Generate twice and the first batch's picture — including
its `ResultStrip`, `ResultBar` and any `failureSummary` — can vanish without ever appearing. `run` is
a single slot, so there is no way back to it from the pane.

**Fix:** keep the settled outcome beside `run` (a `lastFinished` the canvas falls back to, the way
web pins `pinnedDone`), or hold the next batch until the result is dismissed.

### 10. File reads, base64 encoding and image decodes run on the main actor

- **kind:** quality
- **where:** `SourceImageWell.swift:105,119,121`; `ReferenceStrip.swift:113,119`;
  `IdentityGroup.swift:77-79`; `PictureSource.swift:45` (inside a `Task` that inherits `@MainActor`)
- **confidence:** high

`Data(contentsOf:)` + `base64EncodedString()` + `NSImage(data:)` for a 50 MB image all happen on the
main thread, from a `View` method. The decode-in-`.task` pattern used by `ReferenceWell` and
`RunCanvas` shows the team already knows the rule; the import path missed it. The draft then holds the
full-size base64 (and a second copy as an `NSImage`) for the lifetime of the pane — a 4-reference
FLUX.2 draft can be hundreds of MB resident, and `onChange(of: controller.draft)` compares the whole
struct on every mutation.

**Fix:** move read/encode/decode to a detached task, and downscale to the recipe's advertised pixel
ceiling (`max_pixels_single`/`max_pixels_multi`, which `ReferenceImagesCapability` does not currently
decode) before encoding.

---

## LOW

### 11. `machineChoice` is invisible to `@Observable`

`GenerateController+Machine.swift:18-27` is a computed property reading `UserDefaults` directly, so
writing it registers no mutation. Views that read it (`GeneratePane.host:88-92`,
`MachineControl.body:16-18`) refresh only because `choose()` usually also writes `modelName`/`hostID`.
Picking **Auto** on a fleet where `hosts.preferredHost` is nil returns before any observable write
(`MachineControl.swift:53`), so the label keeps showing the old machine. **Fix:** back it with a
stored `@ObservationTracked` property mirrored to the suite.

### 12. Expand stamps `task: text-to-image` into every print's provenance

`GenerateController+Expand.swift:61` hardcodes `task: .textToImage` in the accepted offer, which rides
into `PromptTransformProvenance` (`accept`, `:114-120`) and onto the request. Expanding a prompt for an
LTX-2 clip or an img2img render records the wrong task in the saved metadata. Remix is correct — it
uses `response.task`. `/api/expand` returning no task is why (`types.rs:771-776`), but desktop derives
`ExpandTask` from the concrete request (root `CLAUDE.md`, conditioning-aware expansion invariant).
**Fix:** derive the task from the draft's own conditioning and both send it and record it.

### 13. No in-flight staleness check on expand/remix

`GenerateController+Expand.swift:50-63,77-99` installs whatever comes back, even if the model, family,
prompt or machine moved while the request was in flight; `accept` then writes a `sourcePrompt` that was
never in the box. Desktop/web refuse a landed rewrite **by name** when any of those moved
(`studio/lib/preparedExpansion.ts`). Low because the window is short and the damage is provenance +
a surprising prompt, not a wrong render.

### 14. Sliders are unlabelled for VoiceOver

`NumberControls.swift:5-19` builds a bare `Slider`, and `ControlLabel` (`ControlsRow.swift:84-89`)
puts the name in a *sibling* `Text` with no `accessibilityElement(children: .combine)` and no
`accessibilityLabel` on the control. VoiceOver reads "50 percent, slider" for Steps, Guidance, Length,
Strength, identity Weight and every LoRA scale. This is exactly the hole `make lint`'s a11y rule
admits it cannot see ("a tooltip on the row does not name the button inside it").
**Fix:** `.accessibilityLabel(title)` on the content inside `ControlLabel`, or combine the pair.

### 15. Bare arrow keys are bound as window-wide shortcuts

`ResultStrip.swift:63-68` attaches unmodified ←/→ `keyboardShortcut`s to the result thumbnails,
guarded only against a text caret. While a multi-result batch is showing, these plausibly out-compete
the focused `Slider`/`Stepper`'s own arrow-key adjustment in the capsule and inspector. I could not
verify the precedence without running the app — flagging it as worth a manual check rather than
asserting it.

### 16. Nothing is persisted across launches

The whole draft lives in `@State` in `MoldApp.swift:56`. Quitting loses the prompt, the attachments,
the filing and every control; desktop restores its Create form. The recovery path
(`GenerateController+Recover`) restores the *run* but not what made it.

### 17. Decoded-but-unused capability surface

`RecipeCapabilities.schedulers` (no scheduler picker), `WanRecipeCapabilities.supportsFirstLastFrame` /
`supportsDistillStrength` (no end-frame well, no distill slider),
`PlacementPreview.pendingDownloads` / `missingComponents` (no model pull-on-demand — desktop's
`classifyMissingModel` → pull + pull-resume). All are honest M-scope omissions; listing them so the
gap is enumerated rather than discovered.

### 18. Test gaps worth closing with the fixes above

- No test calls `stop()` while `run == .submitting` (finding 2) — `RunQueueTests` always awaits the
  admission first.
- No test asserts that a `combines` recipe draws both wells (finding 1); `recipe-sd15.json` already
  carries the fixture data.
- Nothing pins the length ceiling against `maxDurationSeconds` or a clip size (finding 3).
- `BatchOutcome` is well covered; `settle` → `followNext` ordering (finding 9) is not.

---

## Done notably well

1. **Capability absence rules are written down once and asked, never inlined.**
   `Capabilities+Reading.swift` and `RecipeCapabilities+Reading.swift` give each field its own
   documented absence semantics, and `readsSourceImage`'s "ABSENT MEANS YES" with the
   `manifest.rs:265-270` citation is the kind of thing most clients get wrong.
2. **Idempotent durable submission.** The client batch id is minted and persisted *before* the POST,
   recovery asks the host by that id, and re-submitting is explicitly refused as a way to find out what
   happened. `PendingBatch` + `GenerateController+Recover` is a better story than most of the fleet's.
3. **Batch N is N one-output siblings** sharing one `batch_id` and differing only by seed, with
   `randomBase` injected so the fan-out is a pure, testable function (`RenderDraft+Request.swift:83-103`).
4. **Parking rather than dropping conditioning** across a model or recipe switch
   (`DraftMedia+Reconcile.swift`), in a dependency order that is commented and justified — extend beats
   keyframes beats references beats source, mask needs a surviving source.
5. **Pure decision functions pulled out of the views** — `OutputGroup.Row.resolve`,
   `RefineGroup.maskRow`, `ShapeControl.resolve`, `MachineControl.rows`, `RecentGroup.Listing.resolve` —
   so the tests need no view host. (Finding 1 is the one place a correct predicate feeds a wrong branch,
   which is the argument for testing the branch too.)
