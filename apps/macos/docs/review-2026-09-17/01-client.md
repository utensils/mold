# MoldClient wire + domain review (`apps/macos/Packages/MoldClient`)

Scope: the 130 files under `Sources/MoldClient` and `Tests/MoldClientTests`, checked against
`crates/mold-core/src/{types,generation_profile,validation,organization}.rs`,
`crates/mold-server/src/{routes,auth}.rs`, `crates/mold-db/src/generation_queue.rs`, and the
`studio/lib`, `web/src`, `desktop/src` counterparts. Read-only; nothing was built or run.

Paths are absolute where they leave the package.

---

## HIGH

### 1. `QueueOrder.moves` plans a batch move in the wrong index space — children land split around the drop target

- **kind**: bug · **confidence**: high (replayed against the server's own algorithm)
- `/Users/jamesbrink/Projects/utensils/mold/apps/macos/Packages/MoldClient/Sources/MoldClient/QueueOrder.swift:43-56`
- server: `/Users/jamesbrink/Projects/utensils/mold/crates/mold-db/src/generation_queue.rs:1815-1836`

`moves` builds ONE candidate list with **every** moving id removed and then hands out
`base, base+1, base+2, …`. The server resolves each `PATCH /api/queue/:id`'s `position` against the
queued list minus **that one row only** (`order.remove(current_position); order.insert(position, job_id)`),
so the sibling children are still in the list when the second call is resolved.

Queue `[c1, c2, A, N, B]`, all queued; drag the `(c1, c2)` batch to sit after `N`. `candidates` is
`[A, N, B]`, `base = 2`, so the plan is `(c1, 2), (c2, 3)`. Replaying server-side: `c1` → `[c2, A, c1, N, B]`;
then `c2` → `[A, c1, N, c2, B]`. The two children end up on **either side of `N`**, and neither is where
the drop indicator was. The single-row `QueueOrder.move` is exactly right; only the multi-row path is wrong.
`QueueOrderTests.swift:45` (`aBatchMovesAsAscendingCallsSoItsChildrenLandTogether`) only exercises
`after: nil`, which is the one case where the two index spaces coincide, so it passes today.

**Fix**: plan against a working copy — for each id, remove only that id, take
`index(of: previousMovedId ?? neighbour) + 1`, then apply the move to the working copy before planning the next.

---

### 2. A prompt-transform provenance re-encodes an `.unknown` open enum, and the server refuses the whole render

- **kind**: bug (forward-compat) · **confidence**: high
- `/Users/jamesbrink/Projects/utensils/mold/apps/macos/Sources/Mold/Generate/GenerateController+Expand.swift:92` and `:114-120`
- `Sources/MoldClient/MoldJSON.swift:48-57` (`OpenWireEnum.encode` writes `rawValue`, and `.unknown`'s rawValue is the literal `"unknown"`)
- server: `/Users/jamesbrink/Projects/utensils/mold/crates/mold-core/src/types.rs:711-727` — `PromptTransformProvenance.task: ExpandTask` is REQUIRED and a strict Rust enum; `dimensions: Vec<RemixDimension>` likewise.

`remix` takes `response.task` straight off the wire (`:92`) and `response.variants[].dimensions` with it;
both decode leniently to `.unknown` when the host names a task or a dimension this build predates.
`accept(_:)` then writes them into `draft.promptTransform` (`:119-120`), and
`GenerateRequest+Encoding.swift:58` ships that as `prompt_transform`. Serde has no `unknown` variant for
either enum, so `POST /api/generation-batches` rejects the body outright.

The failure is sticky and opaque: the poisoned provenance lives in the draft, `revertExpansion()` only works
while the prompt is still byte-identical to what was just accepted, and every subsequent press of Generate
fails with a deserialization error that names `prompt_transform`, not the wand. `PromptTransform.swift:71-74`
already writes the rule down ("an `OpenWireEnum` may never be ENCODED as `.unknown`") and applies it to
`ExpandRequest.task` — the provenance path is the one place it is broken.

**Fix**: drop the whole `promptTransform` (or the offending field) when `task == .unknown`, and filter
`.unknown` out of `dimensions`, at the point the offer is accepted.

---

### 3. The clip-length control ignores `max_duration_seconds`, so the top of the slider is a guaranteed 422

- **kind**: bug · **confidence**: high
- `Sources/MoldClient/Temporal.swift:71` (`maxDurationSeconds` decoded) and `:82-89` (`snap` clamps to `frames.max` alone)
- `Sources/MoldClient/RenderDraft+Recipe.swift:43` snaps through it; `/Users/jamesbrink/Projects/utensils/mold/apps/macos/Sources/Mold/Generate/NumberControls.swift` builds the slider as `frames.min...frames.max`
- server: `/Users/jamesbrink/Projects/utensils/mold/crates/mold-core/src/generation_profile.rs:1243-1251` narrows `effective_frames.max` to `(seconds*fps − offset)/step*step + offset` BEFORE `validate_integer`; `:2558-2563` deliberately advertises the 120-fps ceiling with the comment "admission applies the lower duration-derived cap for the selected FPS"
- counterpart: `/Users/jamesbrink/Projects/utensils/mold/studio/lib/videoDuration.ts:138-169` does exactly that arithmetic

`maxDurationSeconds` is decoded and read nowhere in the app (grep confirms). For `ltx2`, `frames.max` is the
120-fps figure (601); at the default 24 fps admission's real cap is `20*24+1 = 481`. The Mac slider runs to
601 and reports "25.0s" against a 20-second budget, and every value from 489 to 601 is a hard 422 at submit.
Web and desktop cap the same slider at 481.

**Fix**: add `TemporalProfile.effectiveMaxFrames(fps:)` applying the duration cap on the grid, and use it in
`snap` and for the slider's upper bound.

---

## MED

### 4. No legacy fallback when `capabilities.reference_images` is absent — reference editing dies on an older host

- **kind**: parity-gap · **confidence**: high on the code paths, medium on how often an old host is targeted
- `Sources/MoldClient/DraftMedia+Reconcile.swift:34-37`: `let referencesVisible = references?.mode.isVisible == true` — a **nil** block and a `hidden` block are the same answer, so every staged reference is parked and `referenceWeight` cleared.
- server: `generation_profile.rs:583` makes the field `Option` precisely so absence is "OLDER SERVER, never a refusal"
- counterpart: `/Users/jamesbrink/Projects/utensils/mold/studio/lib/legacyRecipeRules.ts:104-136` (`legacyReferenceImages`) supplies Qwen-Image-Edit (required, target-first, `replaces`) and FLUX.2 [dev] (max 4, `replaces`); wired in at `studio/lib/generationCapabilities.ts`.

Point the Mac app at a `mold serve` predating the block and pick `qwen-image-edit-*`: the Target well never
draws and any staged reference is parked, so the one model whose recipe *requires* a reference can never be
submitted — while the browser on the same host works. Same host, flux2-dev: references are silently dropped
and the render comes back as a plain text-to-image.

**Fix**: port `legacyReferenceImages` and consult it **only** when the block is `nil`; keep `hidden` a refusal.

---

### 5. A clip rendered as GIF/APNG/WebP is classified as a picture

- **kind**: bug · **confidence**: high (verified on both sides)
- `Sources/MoldClient/GalleryPrint.swift:57`: `isVideo` is `["mp4", "webm", "mov"]`
- server: `generation_profile.rs:2140-2152` — every recipe with a `temporal` block advertises
  `formats: [Mp4, Gif, Apng, Webp]`, and `types.rs:3669-3671` serializes them lowercase.

The Generate inspector offers the format from `capabilities.output.formats`, so picking GIF for an LTX-2 or
Wan render is a supported, one-click choice. The resulting print carries `format: "gif"` and
`GalleryPrint.kind` answers `.picture`: no video player, `ExportOptions.forVideo` never offered, the `is:video`
search token misses it, and Quick Look opens it as a still. `webm`/`mov` in the list are formats mold does not
produce at all, while three that it does are missing. (APNG is worse-but-unfixable-here: `OutputFormat::Apng.extension()`
is `"png"`, so its print is indistinguishable from a still on the wire.)

**Fix**: add `gif` and `webp` to the video set, or better, derive `kind` from the recipe/metadata `frames`
rather than the container.

---

### 6. `escaped(_:)` is a PATH escaper and is used for QUERY values — an `&` or `+` in a catalog search silently changes the request

- **kind**: bug · **confidence**: high
- `Sources/MoldClient/HTTPBackend+Verbs.swift:51-55` escapes with `.urlPathAllowed` minus `/`. `.urlPathAllowed` **includes** `&`, `=`, `+`, `;`, `$`, `,`.
- callers that put its output in a query string: `Sources/MoldClient/Catalog.swift:36-40` (`q=`, `family=`, `kind=`, `source=`, `sort=`) and `Sources/MoldClient/HTTPBackend+Adapters.swift:22` (`model=`)
- `HTTPBackend+Transport.swift:51` then assigns the whole thing verbatim as `percentEncodedQuery`, so nothing fixes it downstream.

Searching Discover for `cats & dogs` sends `q=cats%20&%20dogs`: the host parses `q` as `"cats "` and an
unrelated empty parameter, and the results are for the wrong query with no error anywhere. `C++` sends a
literal `+`, which `serde_urlencoded` decodes as a space. The doc comment claims "the rest are server-defined
tokens, escaped the same way on principle" — but `text` is free user input and is escaped by the same
inadequate rule.

**Fix**: build query strings with `URLComponents.queryItems` (or a `.urlQueryValueAllowed`-style set that also
subtracts `&+=;$,`), and keep `escaped(_:)` for path components only.

---

### 7. `RenderDraft.fit(to:)` snaps every `.buckets` recipe, ignoring `off_bucket: warn`

- **kind**: bug · **confidence**: high
- `Sources/MoldClient/RenderDraft+Fit.swift:18-23` snaps unconditionally; `Sources/MoldClient/Resolution.swift:58` decodes `offBucket` and nothing reads it (grep: only tests construct it)
- server: `generation_profile.rs:2001` is the one family (`wan`) advertising `Buckets` with `OffBucketPolicy::Warn`; `validation.rs:1366-1367` refuses an off-bucket size only on `Reject`
- counterpart: `studio/lib/generationProfile.ts:1187-1189, 1246-1258` keeps the off-ladder size and warns

A Wan clip rendered at 1024×768 (admitted with a warning by any host) is silently rewritten to the nearest
advertised bucket when reused, so "reuse these settings" re-renders a different shape without saying so.

**Fix**: in the `.buckets` arm, `guard (resolution.offBucket ?? .reject) != .warn else { return }`.

---

### 8. Alignment rounds up *after* the pixel-budget scale, landing above `max_pixels`

- **kind**: bug · **confidence**: high (arithmetic)
- `Sources/MoldClient/RenderDraft+Fit.swift:33-41`

The `maxPixels` scale runs first, then `aligned()` rounds each axis to the **nearest** multiple, which can
grow both axes back past the budget it just enforced. Reusing a 2048×1152 print onto a FLUX recipe
(alignment 16, `max_pixels` 1,800,000): scale → 1788×1006 (1,798,728 ✓), align → 1792×1008 = **1,806,336** ✗.
`validate_resolution` refuses it, so a Reuse the app itself just "fitted" is rejected at submit with a
pixel-budget error the user cannot act on.

**Fix**: round down when the nearest-rounded pair exceeds `maxPixels`, or re-apply the pixel clamp after aligning.

---

### 9. `QueueListing.merged` uses a non-stable sort over positions that routinely tie

- **kind**: bug · **confidence**: high on the mechanism, medium on visible frequency
- `Sources/MoldClient/QueueEntry.swift:76-81`: `byID.values.sorted { ($0.position ?? .max) < ($1.position ?? .max) }`

Two independent sources of instability compound here. `byID.values` is a Dictionary value collection, whose
iteration order is unspecified and re-randomised per process; and Swift's `sorted(by:)` is an introsort with
no stability guarantee. `QueueOrder.swift:8-10` documents that a **held row inherits its `position` from the
next runnable one**, so ties are not a corner case — every held row ties with a queued one, and every row with
`position == nil` ties with all the others at `.max`. The result is that two consecutive `GET /api/queue`
refreshes of an unchanged queue can hand `QueueGroup.build` a different order, which then also changes the
group order (it keeps each group where its FIRST row sat). Rows visibly swap places while nothing happened.

**Fix**: make the comparator total — `(position ?? .max, id)` — so equal positions break deterministically.

---

### 10. An SSE route's refusal loses the machine's own sentence and can never be a licence refusal

- **kind**: bug · **confidence**: high
- `Sources/MoldClient/HTTPBackend+Transport.swift:118-121` drops `bytes` unconsumed on a non-2xx; `:137-141` (`streamFailure`) builds the error from the status code alone.

Every non-2xx on `/api/events`, `/api/downloads/stream`, `/api/resources/stream` and
`/api/generation-batches/{id}/events` becomes a bare `.http(status:code:nil,message:nil)`, while the identical
refusal on a plain GET goes through `check(_:_:)` (`:91-105`) and decodes `APIError`. A host refusing the event
stream with `503 SERVER_RESTARTING` surfaces as "The machine answered with an error (503)", and the reconnect
loop then retries it forever with no explanation.

**Fix**: read a bounded prefix of `bytes` on the failure path and route it through the existing `check(_:_:)`.

---

### 11. `TransferPlan.classifyAdmitFailure` cannot see `.unauthorized` or `.licenseRequired`

- **kind**: bug · **confidence**: high
- `Sources/MoldClient/TransferPlan+Guards.swift:39-50` matches only `MoldClientError.http`
- but `HTTPBackend+Transport.swift:93` throws `.unauthorized` for 401 and `:97` throws `.licenseRequired`

Moving a held job to a destination whose API key is wrong (or that needs a licence accepted) falls through to
`.ambiguous`. The plan then issues another authenticated lookup against the same machine, which also fails,
and the user is told "Acceptance by X is not confirmed. The original remains held. Retry this same destination
to check safely" — advice that can never succeed, for a failure that was definite and *pre-commit*.
`TransferPlanTests` covers 413, coded 503, 429 and `.unreachable`, but neither of these two.

**Fix**: classify both as `.rejected` with their own sentence, and add the two cases to `TransferPlanTests`.

---

### 12. `playableURL` hands an unauthenticated URL to the player on a keyed host

- **kind**: bug · **confidence**: high on the code path
- `Sources/MoldClient/MediaToken.swift:41-43`

The doc comment three lines above promises that "on a keyed host a ticket that fails to mint is a `throw`,
not a plain URL the player would send with no credential and get a 401 from" — and the `guard` does exactly
that: if the mint answers `authRequired == true` with a `nil` token, or `URLComponents` fails to rebuild,
it falls through to `return plain`. `AVPlayer` then requests the media with no `X-Api-Key`, gets a 401, and
shows a silent playback failure rather than an auth error. All three branches are untested —
`HTTPBackendURLTests.swift:80` is a compile-only protocol-reachability check that never invokes it.

**Fix**: throw `MoldClientError.unauthorized` on a keyed host when the ticket is unusable; add stubbed tests
for the keyless / keyed-with-token / keyed-without-token branches.

---

### 13. Expand records `task: .textToImage` on every print, including clips

- **kind**: bug (wrong provenance) · **confidence**: high
- `/Users/jamesbrink/Projects/utensils/mold/apps/macos/Sources/Mold/Generate/GenerateController+Expand.swift:61`

`ExpandResponse` carries no task, so the offer is hard-coded to `.textToImage` and that value ends up in the
print's `prompt_transform.task` (`:119`). Every clip whose prompt was expanded (rather than remixed) records
that it was written for a still. The server's `/api/expand` infers the real task from the family
(`routes.rs:4036-4045`) but does not report it back.

**Fix**: derive the task from the chosen recipe the way `studio/lib/expandTask.ts` does, and send it as
`ExpandRequest.task` so the server and the provenance agree.

---

### 14. Four nested `AsyncThrowingStream`s with no buffering policy

- **kind**: quality/race · **confidence**: high on mechanism, medium on observed impact
- `Sources/MoldClient/LineAccumulator.swift:41`, `Sources/MoldClient/ServerSentEvents.swift:52`,
  `Sources/MoldClient/HTTPBackend+Transport.swift:111`, `Sources/MoldClient/HTTPBackend+Events.swift:15`

All four default to `.unbounded`, so no level ever exerts backpressure on the socket. The consumer,
`Sources/Mold/Support/HostStore+Events.swift:90-102`, is `@MainActor` and fans out synchronously. During a
burst — a bulk import, `emptyTrash`, a batch settling — mold emits one `gallery_*` frame per print, each
carrying a whole `GalleryPrint`, and all of it queues in four unbounded buffers while the main thread is busy.
Memory then tracks the burst rather than what the app can process, and the UI spends seconds applying events
that are already stale.

**Fix**: give the frame-level streams an explicit `bufferingPolicy` — `.bufferingNewest(1)` for
`resourceStream`, a bounded `.bufferingOldest` for the event routes.

---

## LOW

### 15. `reference_weight` is modelled end to end and can never be set

`Sources/MoldClient/DraftMedia.swift:20` holds it, `RenderDraft+Request.swift:40` ships it, and
`RecipeBlocks.swift:50` decodes the `FloatControl` — but nothing in `Sources/Mold` ever writes
`media.referenceWeight` (grep: the only assignment is the `nil` in `DraftMedia+Reconcile.swift:37`). SD1.5 and
SDXL IP-Adapter renders therefore always run at the server default, and the one control the server puts
*inside* the block specifically so it would travel with the capability is invisible. **kind**: parity-gap.

### 16. The API key rides a redirect to whatever host answers

`HTTPBackend+Transport.swift:58-60` sets `X-Api-Key` on the request; the session is `.shared` with no
`URLSessionTaskDelegate` anywhere in the app (grep for `willPerformHTTPRedirection`: no hits). URLSession
forwards custom headers across a redirect, including cross-origin, and mold's default scheme is plain `http`.
A compromised or misconfigured proxy in front of a host can harvest the operator key with a single 302.
**kind**: security. Low likelihood, cheap fix: a delegate that drops `X-Api-Key` when the redirect target's
origin differs.

### 17. Nine routes interpolate an id into the path without `escaped(_:)`

`HTTPBackend+Work.swift:8,14,18,25,51`, `HTTPBackend+Generation.swift:11,15,21,25`. All of these carry
server- or client-minted UUIDs today, so it is not currently reachable — but `URLComponents.percentEncodedPath`
**raises** on an invalid character rather than returning nil, so the failure mode if an id shape ever changes
is a crash, not a bad request. Every comparable route (`queueJob`, `modelPath`, `transferExportPath`,
`revokePairedClient`) already escapes. **kind**: quality.

### 18. `ClientTags.normalize` does not strip control characters

`Sources/MoldClient/ClientTags.swift:28-41` vs `/Users/jamesbrink/Projects/utensils/mold/crates/mold-core/src/organization.rs:37-50`
(`normalize_tag_name` refuses non-whitespace control characters) and
`studio/lib/libraryOrganization.ts:70-72` (strips them client-side). A pasted tag containing an escape byte is
sent and the whole render is refused at admission. The length truncation is a documented, defensible
divergence; this one is not. **kind**: bug.

### 19. `CollectionShelf.hidden` inverts the studio rule

`Sources/MoldClient/CollectionShelf.swift:71` uses `allSatisfy`; `studio/lib/libraryOrganization.ts:167,205`
sets `hidden` when **any** host copy is hidden. It has a stated rationale and is contained (per-print hiding
goes through the per-host set, which does match studio), but two silently different rules for the same shelf
across a fleet deserves a deliberate decision. **kind**: design.

### 20. Transfer has no reference-upload lease path

`HTTPBackend+Transfer.swift:23-52` posts the export bytes inline; `studio/api/queueTransfer.ts:119-143` routes
them through `prepareReferenceUploadBatch`. A held job whose media exceeds 64 MiB can be moved from web/desktop
but never from the Mac app (`TransferPlan+Guards.swift:42` → "about 48 MB"). The README documents upload
sessions as out of scope; noting it as a real capability gap, not a defect. **kind**: parity-gap.

### 21. `moldLines()` iterates `URLSession.AsyncBytes` one byte at a time

`Sources/MoldClient/LineAccumulator.swift:45-47`. The accumulator itself is correct and well tested (including
a multi-byte character split across reads); it is the *source* that is expensive — one async suspension per
byte, and a `/api/events` gallery burst is hundreds of KB. **kind**: quality. Feed it from chunked data rather
than from `AsyncBytes` element by element.

---

## Test gaps on exported contracts

1. **`GenerateRequest.encode(to:)` is not pinned exhaustive** (`GenerateRequest+Encoding.swift:14-62`). A
   hand-written 45-line field list over a struct with a *synthesized* `CodingKeys`: adding a stored property
   compiles, gets a key for free, is happily set by `RenderDraft+Request`, and silently never reaches the wire.
   `GenerateRequestTests` asserts about twenty named keys and nothing about the set being complete. **One test
   that populates every optional and asserts the encoded key set equals an explicit expected set** would fail
   the build on the next new control. This is the highest-value missing test in the package.
2. **`QueueOrder.moves` with a non-nil neighbour** — the untested branch is finding 1.
3. **`VideoOnlyPolicy` has no tests at all** (`VideoOnlyPolicy.swift:39-60`), including the deliberate
   precedence of `audioOnlyPipeline` over `audioEnabled` and the `enabled && blocked → nil` case (returning
   `false` there would send the field and pin the server off its default multimodal path — an output change).
   `studio/lib/videoOnly.test.ts` is the oracle to port.
4. **The whole S6b clip park/restore slice** (`DraftMedia+ParkClip.swift:9-77`): `reconcileKeyframes`,
   `reconcileExtend` (three fields in lockstep, nothing pins that `extendOverlapFrames` survives a round trip),
   `reconcileAudioFile`, `reconcileSourceVideo`, and the extend-beats-keyframes rule at
   `DraftMedia+Reconcile.swift:26-29`. That last rule is only safe because `reconcileKeyframes` runs first and
   clears `parked.keyframes`; reorder the two statements and `parked.keyframes = keyframes` overwrites an
   already-parked set with no way back — precisely the loss this file exists to prevent.
5. **The downloads `snapshot` frame** (`HTTPBackend+Work.swift:61-68` decodes with `try?` and `continue`s).
   `StreamTests` feeds only a `job_progress`-shaped frame. A shape change to `DownloadsListing` makes every
   snapshot vanish and the popover permanently empty for jobs started by `mold pull` or the web app, silently.
6. **`MediaToken.playableURL`** — all three branches (finding 12).
7. **`ExportOptions.forVideo` / `.forMesh`** (`GalleryMutations.swift:113-125`) — the two sets deliberately
   overlap (`forMesh` is a union, `forVideo` is animated-only); two assertions over one mixed `formats` list
   would pin it.

---

## Done notably well

1. **`LineAccumulator` and the `no bytes.lines` lint.** The `URLSession.AsyncBytes.lines`-drops-empty-lines
   trap is real, silent and fatal for SSE, and it is fixed in one place, byte-wise so a multi-byte character
   split across two reads survives, tested, and enforced by a lint so it cannot come back.
2. **`OpenWireEnum`.** One protocol, one rule, applied to every open enum on the wire — a host that grows a
   new `ControlMode`, `JobStatus` or `OutputFormat` degrades one field instead of losing the whole listing.
   (Finding 2 is the one place the rule's own corollary was missed, which is a small blemish on a good design.)
3. **`Capabilities+Reading` / `RecipeCapabilities+Reading`.** Every absence is answered once, by name, with the
   server citation for *why* that particular absence means yes, no, or unknown — `persistsOutputs ?? true`,
   `readsSourceImage ?? true`, `trashRetentionDays` mapping 0 to nil. This is exactly the discipline the
   "absence means OLDER SERVER" rule needs, and it is the reason finding 4 stands out as the exception.
4. **`transferAdmissionBody`.** Keeping the export opaque and splicing it into a hand-built envelope — rather
   than decoding and re-encoding through this build's `GenerateRequest` — is the right call for exactly the
   reason the comment gives, and escaping the id through the encoder instead of interpolating it closes the
   injection at the same time.
5. **`HostAddress`.** Pure, table-driven, and deliberately written to agree with the two TypeScript ports so
   one box typed into three apps resolves to one origin. The IPv6-bracketing and default-port rules are the
   kind of thing that is usually wrong.
