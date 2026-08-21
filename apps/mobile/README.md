# Mold mobile

Mold is equally owned and maintained by core contributors James Brink and
Jeffrey Dilley.

Mold mobile is a remote-only Tauri 2 client. It never links or embeds the GPU
inference stack; a saved Mold server owns the models, queue, downloads,
generation work, and gallery media. The iPhone app is the shipped product. The
Android project is buildable groundwork that deliberately reuses this crate and
Vue frontend; secure API-key storage, Android NSD discovery, native media
actions, QR scanning, Android-specific polish, CI, and Play distribution remain
implementation work before an Android release.

The app is designed for iPhone first and supports iOS 17 or later. iPad is a
responsive secondary target.

For user setup and workflows, see the
[iPhone guide](../../website/guide/iphone.md). This file is the maintainer
reference for the native shell, development commands, and TestFlight release
path.

## Architecture

- Native shell: `apps/mobile/src-tauri`
- Shared Vue frontend: `desktop/src/mobile`
- Mobile HTML entry: `desktop/index.mobile.html`
- Shared desktop/mobile logic: `desktop/src/lib`, including API types, explicit
  host targets, SSE handling, generation capabilities, the form/request
  builder, source fitting, gallery media, catalog helpers, and themes
- Generated Apple project: `apps/mobile/src-tauri/gen/apple`
- Generated Android project: `apps/mobile/src-tauri/gen/android`
- Bundle identifier: `com.utensils.mold`
- Minimum iOS version: 17.0
- Android compile/target SDK: 36; minimum SDK: 24

The thin mobile crate is excluded from the root Cargo workspace so a phone
build never cross-compiles the desktop server or inference dependency tree.
Native commands are intentionally limited to platform responsibilities:

- Keychain storage for per-host API keys
- Bonjour/DNS-SD discovery of `_mold._tcp`
- UIKit appearance synchronization for readable system chrome

Android must provide those same contracts through Android Keystore, NSD,
MediaStore/share intents, system-bar appearance, and a barcode-scanner plugin.
Do not fork the product UI or remote HTTP/SSE logic to implement them.

Everything a remote Mold server can answer uses the same authenticated HTTP
and SSE contract as desktop. Do not import desktop stores that assume a local
**This device** engine; adapt shared pure helpers to an explicit `ApiTarget`
instead.

## Current product surface

The primary tabs are Create, Library, Models, and Machines. Settings is a
pushed screen opened from the header.

- **Create** picks where work lands. With one connected machine the Host
  control behaves exactly as before. Once two or more connected machines are
  reachable it also offers **Auto** (the least busy machine that already has
  the model) and **Most capable** (the strongest GPU that has it — CUDA before
  Metal, then VRAM, then queue depth), persisted at
  `mold.mobile.generate-target.v1` and resolved through
  `desktop/src/mobile/generateTarget.ts`. Under either policy the model picker
  is the union of every reachable machine's installed models, tagged with the
  machine that has one when they differ, and Develop fans
  `POST /api/generate/placement-preview` out to the candidate machines (each
  with its own Keychain key) before choosing, keeping slower machines in the
  race only until one of them answers with a plan — Auto by soonest predicted
  completion including round trip, Most capable by that machine's reported
  `gpu_info.backend`. The winner is frozen into the same immutable route record
  a pinned machine uses (host id, URL, Keychain key, instance id), so prepared
  and quick expansion keep their own machine, durable sequences restore on the
  exact machine that ran them, and a fleet split across incompatible major Mold
  versions stops automatic routing with the shared profile-conflict message
  instead of guessing. Nothing is queued when no machine can run the print; the
  failure names each machine. Create also supports model-aware image and video
  controls, Batch 1 quick
  expansion/undo, Batch N prepared-variation review, remote prompt history,
  local templates, independently cancellable siblings, source/edit images,
  masks, ControlNet, LoRA, scheduler and CFG++, post-generation upscaling,
  target-host estimates, proportional resolution choices, and explicit Random
  or Fixed seeds. Deeper options open in a full-screen **Advanced** sheet, and
  prompt **style** presets compose at submit without rewriting the prompt text.
  A **Title** field above the prompt names the print: the trimmed value rides
  every mobile-built `GenerateRequest` as additive `title` (batch siblings and
  prepared Batch N inherit it), an over-long or control-character title is
  refused inline before anything queues, and **Use as prompt** restores the
  print's saved title (a later Library rename wins over the metadata stamp).
  Title is a field of the whole Create stack, One shot and Sequence alike —
  a sequence's stitched print carries it on the chain wire.
  Below it, a capability-gated **File under** group
  (`desktop/src/mobile/MobileFileUnder.vue`) files the print as it is made:
  a **Tags** row with a dashed, removable `{slug} · from title` ghost chip and
  the tags you added, and a **Collection** row that pre-selects — never
  creates — the collection whose slug matches the title. Both open a bottom
  sheet; a mono line under them previews the filename the print lands as. The
  reducers are `@studio/lib/fileUnder`, shared verbatim with desktop and web,
  and `desktop/src/mobile/fileUnder.ts` owns only the phone's capability
  question: a pinned machine answers for itself, an automatic policy is
  satisfied by any reachable machine advertising
  `capabilities.gallery.organize`, and an unread snapshot hides the group and
  files nothing. Typed tag text loses a leading `#`; a suggestion a machine
  actually reported is added verbatim, so a real `#grain` files as `#grain`.
  An **Identity** photo well (`desktop/src/mobile/MobileIdentityWell.vue` over
  the shared `@studio/components/IdentityPhotoWell.vue`) sits in the primary
  Create stack beside the source wells, mounted only while the resolved recipe
  or the model row's additive `supports_identity` says yes — positive knowledge
  only, so an unread or older host renders nothing rather than a disabled
  control. Picking uses the well's own file input, which is the native
  photo/camera picker; the gallery escape hatch the source wells offer is
  deliberately absent, because a gallery print is a render, not a reference
  photograph. The bytes travel VERBATIM: an identity photo is never routed
  through source-fit preprocessing and carries no `source_fit` provenance. A
  photo staged before a capability-losing model switch is PARKED — retained in
  the form, kept off the wire by `buildRequest`, Develop still enabled — and
  the well returns with it when a qualified checkpoint is selected again.
  **Identity strength** (`0.0`–`3.0`, step `0.05`, default `1.0`) and
  **Identity start step** live in the Advanced sheet, count toward its badge,
  clear on its Reset (which keeps the attached face), and stay absent from the
  request until touched so the server's defaults remain authoritative. Every
  refusal — a photo with a LoRA or a source image, a knob with no photo, an
  oversized or unsupported file, a photo over the 45 MiB combined request-media
  budget — renders inline beside the control and blocks Develop, never as a
  toast. Prepared Batch N siblings inherit the whole partition — and because
  the reviewed card owns its own Develop, an identity refusal travels with the
  reviewed work as a named stale reason rather than relying on the composer's
  blocker. A changed photo stales reviewed prompt work through the shared
  conditioning fingerprint, exactly as a changed source image does. Routing is
  identity-aware: `automaticRoutingCandidates` narrows Auto / Most capable to
  the machines whose OWN `/api/models` row advertises `supports_identity` for
  that model (the picker row is the fleet union and cannot answer for the
  winner), the frozen route is re-checked before submission, and
  `requiresAuthoritativePlacement` now covers `id_image` so an identity request
  can never take the legacy 404/405 placement fallback on a server that would
  ignore the face. Every rule comes from
  `@studio/lib/identityConditioning`; `desktop/src/mobile/identity.ts` holds
  only the phone-shaped parts (budget, native ingest, Info rows, reuse
  outcome).
  A **↺ Reset** beside the Advanced trigger restores every generation setting
  to the selected model's defaults while preserving the prompt, model choice,
  and any prepared batch size; the Advanced sheet keeps its narrower
  advanced-only Reset.
  For LTX-2 checkpoints, the sheet honors additive `supports_audio` model
  metadata: video-only community checkpoints disable generated audio with an
  explanation while source-image video remains available. It also exposes the
  additive `guidance_overrides` contract (STG scale/blocks, CFG rescale,
  modality scale, and skip stride), validates before queueing, counts the group
  in its badge, and keeps empty fields absent so pipeline defaults remain exact.
  The **prompt is optional** for an `ltx2` / `ltx-video` model once the form
  carries visual conditioning (source image, keyframes, source video, or a
  continuation): Develop enables, the pre-submit guard stops requiring text, and
  the prompt placeholder says so. Every other model — including image families
  with a source image — still requires a prompt. This follows the shared
  `@studio/lib/promptRequirement` rule and its shared copy, so iPhone, desktop,
  and web cannot set different expectations; `MobileSequenceComposer` applies
  the same rule to clip rails through `SequenceLimits.promptOptional`. A blank
  prompt saves no VRAM and usually renders near-static motion — never imply
  otherwise in native copy.
  **Continue a video** appears only when the selected model advertises additive
  `supports_extend`, so a host that predates continuation shows nothing rather
  than offering a request it would reject. The attached clip counts toward the
  combined mobile request-media budget and the sheet states how many new frames
  the continuation appends before it is queued. The overlap picker offers only
  values below the clip length that the selected family's engine accepts: the
  LTX families re-encode an `8k+1` tail through their video VAE, while wan
  carries exactly the one frame its continuation was seeded with, so its picker
  has a single entry. That entry is submitted explicitly — the request always
  carries the overlap the picker is showing rather than leaving the field
  absent for the host to default.
  Source-fit and Upscale then fit preprocessing use a per-Create-form cache, so
  unchanged Batch siblings and repeat submissions share one host upscale and
  fitted source while keeping the editable original intact. When the host
  streams live latent previews (`preview` SSE events), the active print
  develops in a bed above the status line — the preview sharpens with denoise
  progress under the shared thinning Develop grain, matching the print's
  aspect ratio; without previews the plain status line remains.
- **Library** merges saved media from every configured host. Its full-screen
  viewer shows uncropped images, streams videos with native controls, plays
  audio-only prints (LTX-2 text-to-audio) as a waveform tile above a native
  transport, swipes
  horizontally between prints, explicitly copies or saves full-resolution
  stills through UIKit, saves original videos to Photos through a streaming
  native download, restores recorded prompt settings, and can use a still
  as the next source or Qwen edit target. On a print a sequence produced, **Use
  as prompt** reloads that sequence's recorded clips onto the Create clip rail
  as a new draft (raising any clip duration the selected model's motion tail no
  longer allows, and saying so); iPhone is reuse-only — **Edit sequence** stays
  a desktop/web action until mobile has a chain-detail recovery route.
  Generated stills open the same viewer on tap. Press and hold an image to keep
  the native iOS Share, Save to Photos, Copy, Copy Subject, and Look Up menu.
  Pinch the grid to resize thumbnails between two and five across — the iPhone
  counterpart to the web/desktop thumbnail-size slider. The choice persists at
  `mold.mobile.galleryColumns.v1`, separately from the shared pixel-target key
  so a phone's zoom never rewrites a Mac's grid, and defaults to three across.
  Tap the 44pt **Select** control to enter multi-select, then select all, clear,
  or delete the chosen prints. Delete removes every matching copy from
  reachable saved hosts; a host failure leaves that copy visible and reports
  the partial cleanup.
  Persistent New badges match desktop Library visits, and post-generation
  upscaled images carry the shared Upscaled badge.
  Library organization (V3) rides each host's advertised
  `capabilities.gallery`: when a connected host advertises `organize` or
  `trash`, a 44pt **Prints | Collections | Trash** scope row (with counts)
  appears under the header — plain buttons that never capture the grid's
  two-finger pinch. Prints gains a horizontally scrolling chip row (♥
  Favorites, the top tags with counts behind a **More…** sheet, and host chips
  when several machines are connected) filtering the grid client-side; tiles
  show a ♥ badge on favorites. Select mode adds **Add to collection** (checklist
  sheet with a New-collection input), **Tag** (chip editor with merged
  suggestions), ♥ toggle, and **Trash**, which replaces Delete on trash-capable
  hosts with the two-tap copy "Move N to trash?" → "Confirm" (hosts without a
  trash keep today's hard delete and its wording). Collections lists merged
  cross-host collection cards (cover, name, mono count, host labels) with a
  **New collection** row; the … menu offers Rename and an inline two-step
  Delete collection whose copy says the prints stay in the Library; tapping a
  card drills into its grid behind a back chevron, where Select offers
  **Remove from collection**. Trash lists trashed prints with a per-host
  retention banner (mono numbers, **Change · Machines** link), per-tile
  "Purges in N d" chips, Select-mode **Restore** (primary) and a two-step
  **Delete forever**, and a two-step **Empty trash** header button. The
  full-screen viewer titles itself with the print's `displayTitle` and gains a
  44pt **Info** control opening a bottom sheet: editable title (≥16px input;
  Done commits via `PATCH`, blank clears), ♥ toggle, a tags chip editor with
  suggestions, an "In collections" checklist with a New input, and — for
  trashed prints — the purge countdown with Restore / two-step Delete forever.
  A print developed with an identity photo also lists its provenance there
  (filename, short SHA-256, effective strength, and start step), which opens
  the Info sheet even on a host with no organization capability at all; **Use
  as prompt** restores both knobs and re-attaches the photo from this device's
  content-addressed stash, disclosing the miss in the persistent inline status
  line when the stash no longer holds it — saved metadata carries the digest,
  never the face bytes.
  Every mutation fans out to each physical copy's exact Keychain-authenticated
  host via the shared `planOrganizationFanout` plan
  (`desktop/src/mobile/libraryOrganization.ts` holds the mobile state
  helpers); failures are reported in a persistent inline banner, never a
  toast, and edits patch the offline IndexedDB cache behind its mutation
  fence.
- **Models** merges installed models with Hugging Face and Civitai results,
  supports host/media/source/family/kind filters with a downloads/rating/recent
  sort (the family list and a failed search reload themselves when the browsed
  host's address, key, or reachability changes; when the host taxonomy cannot
  be read the family list falls back to a session-sticky set accumulated from
  catalog results and installed inventories, which may include families absent
  from the current page and never collapses to the active filter), gives every card and detail sheet a model-kind badge, explicitly marks
  mature entries `18+ NSFW`, surfaces available description/source/license/tags/
  format/popularity and type-aware weights, and routes pull/load/unload/remove
  actions to the owning or selected host. Detail-sheet variant chips select an exact manifest
  `base:tag` target before pulling. Pull actions progress through `Connecting...`, `Starting...`,
  `Queued`, and `Pulling N%`; active downloads can be cancelled.
- **Machines** starts with native QR scanning for the recommended pairing path,
  then retains Bonjour discovery, manual IP/hostname/HTTPS entry, and Tailscale
  MagicDNS as fallbacks. Desktop and web Settings mint a random, one-use,
  two-minute ticket containing the reachable host address but never the durable
  API key. iPhone redeems it against the exact instance before host dedupe and
  Keychain storage. Host detail shows telemetry, models-disk usage, queue,
  downloads, loaded models, and installed models (all using catalog display
  names rather than opaque `cv:` / `hf:` ids), with rename, retry, select,
  unload, open-in-Models, and forget actions. Queued rows and running singleton
  generations have a 44pt two-tap **Cancel** action against that exact
  Keychain-authenticated host when it advertises cooperative cancellation;
  older hosts keep running work visible and read-only.
  When a host advertises `capabilities.gallery.trash`, host detail adds a
  **Library** card: a **Trash retention** select reading and writing that
  host's `gallery.trash_retention_days` through `GET`/`PUT /api/config/:key`
  (the first mobile `/api/config` client, `desktop/src/mobile/hostConfig.ts`;
  an env-pinned key renders read-only and names the variable) plus a
  "Prints in trash: N" row with a two-step **Empty trash**. Retention is a
  server setting — `mold.mobile.settings.v1` stays four local fields.
  Host detail's RAM card colors off the server's additive `host_memory`
  admission telemetry (headroom vs safety floor, mirroring the shared
  `studio/lib/hostMemory` levels; older servers keep the uncolored card), and
  queued Create activity rows show their live position in line, read over the
  existing 5-second activity tick from the host's authenticated `/api/queue`
  and never persisted (that endpoint carries full prompts). Those rows use the
  shared `studio/lib/queuePosition` vocabulary in this app's compact uppercase
  casing (`NEXT UP`, `QUEUED #2`, `WAITING FOR MEMORY`, plain `QUEUED` against
  a host that lists nothing); ordinary serialization on a busy GPU never
  overrides the position, and the queue header counts running and waiting
  separately ("1 active · 4 queued") rather than calling queued work active.
  Current V2 hosts also expose every GPU/MIG device and its queue lane. Device
  lifecycle controls are shown only when the host advertises
  `devices.lifecycle`; disabling a busy device leaves its current work running
  and shows the draining state until completion.
- **Settings** persists the Mold Studio theme families (Mold or Safelight) and
  System, Dark, or Light appearance. Fresh installs start with Safelight +
  System; valid saved choices remain authoritative. Its default-on Photos
  preference automatically fetches each completed still from its authenticated
  host gallery and saves it through UIKit; post-generation upscales save both
  images, while videos and audio-only prints remain in Mold Library. A
  **Library** section carries **Tag new prints with their title**
  (`autoTagTitle`, on by default), the mirror Create reads into
  `GenerateForm.fileUnderAutoTag`; it only decides whether the removable ghost
  chip is offered, so turning it off never touches prints already made.
  Settings also links to host
  management and shows the app version, remote-only processing policy, and
  TestFlight update channel. About opens the public privacy policy at
  `https://utensils.io/mold/privacy` through the native external-browser opener.

The app shell suppresses WebKit focus/double-tap page zoom and rubber-band
overscroll. The Library viewer keeps its scoped horizontal swipe gesture, and
the Library grid keeps a scoped two-finger pinch (`touch-action: pan-y`) that
resizes thumbnails while one-finger scrolling is unaffected.

Prepared expansion always snapshots the selected remote host ID, endpoint,
Keychain-provided key, and server instance. Batch is a directly editable
positive count. Batch N requires exactly N distinct non-empty prompts before
its inline review workspace appears; counts above eight use a compact
first-eight summary and bounded Review all pages. Edits and specifically named
stale work remain local until explicit approval, refresh, collapse, or discard.
One reviewed set is capped at 10,000 variations for memory safety; accepted
sets do not impose a cumulative queue limit.
The snapshot also freezes the conditioning-aware expansion task derived from
the actual request. T2V uses chronological shot language; I2V/V2V, retake,
keyframes, and audio-driven video preserve their source authority. Changing the
conditioning names the work as stale, and `/api/expand` receives only the task,
never source media bytes.
Once the host accepts the batch, the composer is immediately available to
prepare another while earlier siblings remain queued or running.
After source preprocessing, Create performs one read-only placement preview for
the finalized sibling shape (`batch_size: 1`, `copies: N`) on that exact frozen
route. A URL, Keychain key, or instance change, an authoritative infeasible
result, a malformed response, or any non-legacy HTTP failure preserves the
reviewed work and queues nothing. The UI names the server's infeasible reason,
temporary planner failure, malformed response, transport error, or host-identity
race instead of collapsing them into one route error. Additive
`missing_components` metadata is informational until Create owns a finalized
held request and the exact host's complete grouped repair pull; it must not
promise automatic resume before then. Additive `pending_downloads` and their
low-confidence estimate describe only devices selected by the candidate plan; cold
installed catalog IDs remain valid across server model-list refreshes. Only a
strictly valid version-1
non-authoritative `unsupported` result or a missing legacy endpoint
(`404`/`405`) may retain compatible routing without an authoritative plan.
Create generation queues and stale reasons resolve opaque catalog IDs through
the selected host's inventory for display, while requests and saved provenance
retain the stable ID.
`useMobileDownloadsStore` is the sole Models/Create pull authority and keeps
missing-model Connecting/Starting/Queued/Pulling/Ready/error recovery on that
frozen host. Create retains one immutable input/route recovery record and an
attempt-scoped lease that temporarily outranks ordinary Models credentials;
compatible Models and Create pulls already in `Starting` share one POST and
returned job ID. Terminal, failed, stale, superseded, and aborted attempts
release back to Models while preserving the record for an exact-route Retry.
Editing or removing reviewed work supersedes a pending replacement;
view-unique consumer IDs and unmount guards revoke deferred work without
touching a remounted view. Approved siblings preserve order, deterministic seeds,
`original_prompt`, and `batch_id`/`batch_index`/`batch_count` through normal and
long-video requests. Partial failures announce each one-based variation and its
reviewed prompt together with any separate unconfirmed-cancellation warning
while successful prints remain. Library shows the sibling
position and source prompt when those optional metadata fields are present.

iOS suspension marks interrupted generation streams structurally instead of
depending on localized WebKit error text. On resume, pre-ID queue and durable
chain joins are bounded to the original submission window, so a later
fixed-seed duplicate cannot be mistaken for the interrupted print or cancelled
as its zombie. Background failure notifications wait for this reconciliation.
Whether a still-queued row is cleared is decided per job, never per host: the
`/api/queue` row's additive `durable` says whether the host journalled THAT
job, and only then does reconciliation keep polling instead of deleting it
(deleting would destroy exactly the work the host kept). `queue.durable_queue`
is deliberately not consulted — a host that can promise durability still
reports `durable: false` for a job it excluded at admission (no gallery target,
reference-upload media, an oversized request), and waiting on one of those
would hang forever. An absent field is an older server, and the zombie row is
cleared as before. A `held` row is listed but never auto-run, so it settles
immediately with the host's `held_reason`. A terminal error frame carrying
`retained` is likewise treated as an interruption rather than a failure, so the
job is reconciled back to life instead of announcing a failure the host is
still going to finish, and it buys a much longer wait than a suspended socket
does — a restarting host is unreachable for longer than a cold model load.

Reconciliation lives in `desktop/src/lib/generationRecovery.ts` and is run by
the shared generation store for every surface, not by this shell alone; the
iPhone keeps its own call as the foreground-resume entry point. A host that
never answers produces no verdict: the row stays flagged as an interruption so
a later resume can try again, and only a host that answers "no such job, no
such print" settles it as a failure.

**Output** (One shot | Sequence) is a segmented field in the Create form
stack, directly above the model field — sequences are a setting of Create,
not a separate place, so there is no mode pair pinned above the form. The
clip list lives in the shared `@studio` sequence draft, which survives tab
switches and relaunches and carries the prompt across an Output switch (One
shot → Sequence seeds the opening clip; back again returns clip 1's prompt).
Sequence output narrows the model picker to chain-capable video models
through `modelsForOutput`, auto-picks one, and restores the previous single
pick on the way back; when the host reports `supports_sequence: false` its
`sequence_unsupported_reason` is shown inline rather than silently hiding the
model. With no chain-capable checkpoint installed, **Browse video models**
lands on Discover with the Video + Models filters already applied.

Clips are full-width cards with a 44pt **seam pill** between consecutive
cards. Tapping a seam opens a bottom sheet hosting the shared `SeamEditor` at
touch size — the iPhone's only fade-length control, clamped to the host's
`fade_frames_max` (32 when unknown). Seam wording always comes from
`transitionLabel()`, so LTX-Video's zero motion tail reads **Join**
rather than Smooth. New clip durations come from the model's own
server-advertised default, and duration choices stay strictly longer than the
active motion tail. Size, frame rate, steps, guidance, and seed are the SAME
form fields One shot uses, lent to the bench through its Sequence settings
disclosure and read live at submit time — there are no private copies to
drift.

The **Opening sequence image** — the still clip 1 is conditioned on, with its
source strength and fit-to-video-frame controls — is a primary-stack
disclosure, the seat One shot gives its own source media, not an Advanced
control; it is hidden entirely for a checkpoint whose `source_image` contract
is `unsupported`, and an older server that advertises no contract keeps it.
**Advanced sequence controls** therefore hosts only the active clip's negative
prompt and camera motion, and its Reset clears exactly those two — the staged
opening image, strength, and fit survive it, exactly as One shot's staged
media survives the Advanced sheet's Reset. The primary **↺ Reset** is the one
that discards the opening image, alongside every other generation setting.

**Validate plan** sends that live draft to the selected Keychain-authenticated
host's read-only `/api/generate/chain/validate` endpoint. The result names each
clip's normalized input/output frames, transition, conditioning inputs,
warnings, and VRAM estimate when available. It never creates a durable job or
starts downloads/inference, and any draft, shared-setting, model, or host edit
clears the result and fences an in-flight response.

Durable sequences stream over `/api/chain-jobs/:id/events` (SSE) with a 5s
snapshot-poll fallback when the stream fails and a forced re-sync when iOS
wakes the webview, and they appear in the SAME queue list as single prints:
Cancel while live, Resume and Dismiss once settled. Submission, watching,
cancellation, and relaunch recovery stay pinned to one immutable host ID,
URL, instance ID, and Keychain-supplied API key. Local storage retains only
the non-secret route identity and durable job ID. The initial mobile scope
does not include a local engine, the desktop TOML chain editor and full
durable-jobs administration workspace, RunPod provisioning, desktop engine
settings, or desktop self-update channels.

## Persistence and security

WebView local storage contains non-secret mobile state:

- `mold.mobile.hosts.v1` — host metadata with API keys removed
- `mold.mobile.selected-host.v1` — selected generation host
- `mold.sequence.draft.v1` — the shared Output mode, clip list, audio choice,
  and remembered single-print model (base64 clip source payloads are stripped
  before writing). Replaces the retired `mold.mobile.create-mode.v1`, which
  migrates into this draft once and is then removed
- `mold.mobile.sequence-job.v1` — non-secret exact-host identity and active
  durable sequence job ID for relaunch recovery; a saved instance UUID must
  exactly match the current host identity before Mold reattaches
- `mold.mobile.settings.v1` — appearance, color family, Photos auto-save, and
  the "tag new prints with their title" preference (absent on installs saved
  before File under; it migrates to the on default)
- `mold.mobile.generation.templates.v1` — mobile-local generation templates
- `mold.mobile.library-seen-at.v1` / `mold.mobile.library-visited.v1` — bounded
  per-host latest-print timestamps and the first-visit marker for New badges

Per-host API keys live in the iOS Keychain under
`com.utensils.mold.remote-api-key`. Android must use Keystore-backed encrypted
storage before authenticated hosts are enabled there. Never move keys into
local storage, query parameters, logs, generated project files, or plain Android
SharedPreferences.

Mobile pairing uses authenticated `POST /api/pairing/sessions` and the
one-time-token `POST /api/pairing/claim`. The claim route is intentionally the
only unauthenticated credential handoff: tokens are 256-bit random values,
stored server-side only as an HMAC, capped, single-use, and expire after two
minutes. Both responses are `no-store`; the QR must never contain the durable
API key. Pairing QR codes use the registered `mold://pair` scheme so the iOS
Camera app offers to open them directly in Mold; cold-launch and already-open
links share the same claim, instance-verification, and Keychain path as Mold's
in-app scanner.

An authenticated claim receives a distinct `mold_pair_...` credential, not the
host's operator key. The host stores only its digest in `mold.db`; web and
desktop Settings list grants with `GET /api/pairing/clients` and revoke one
with `DELETE /api/pairing/clients/:id`. Paired credentials can use normal APIs
but cannot create or manage other grants. When host authentication is disabled,
pairing remains credential-free and there is no grant to revoke.

Authenticated gallery media uses `POST /api/gallery/media-token` to exchange
the normal `X-Api-Key` request for a short-lived, read-only URL scoped to one
`/api/gallery/image/:filename` path. This allows native video Range requests and
seeking without exposing the long-lived API key. Keep the image-only fallback
for older hosts, but never buffer a whole video as that fallback.

## Local development

### Android

The default setup keeps Android Studio and the large SDK, NDK, emulator, AVD,
Gradle, Cargo, and Bun caches under `/Volumes/ExternalStorage/Android`. Override
the root with `MOLD_ANDROID_ROOT` when the volume is mounted elsewhere.

```bash
nix develop
./scripts/android.sh setup # first machine setup only
android-doctor             # print and verify every resolved path
android-emulator           # boot Mold_API_37 (Pixel 9 Pro / Android 17)
android-dev                # Tauri hot reload
android-check              # debug ARM64 APK build
android-run                # production-mode run
android-build              # ARM64/ARMv7 Google Play AAB
```

`android-studio` is installed as
`/Volumes/ExternalStorage/Android/Android Studio.app`. Open the generated
project with `./scripts/android.sh studio`. The helper defaults to NDK
`27.0.12077973`, which is pinned by the generated Tauri project; change it only
with a deliberate template/toolchain upgrade.

### iOS

Enter the Nix development shell on macOS. Xcode and CocoaPods are required.

```bash
nix develop
ios-dev        # Tauri hot reload — defaults to an iPhone simulator
ios-run        # production-mode run on a selected device or simulator
ios-check      # Rust check for aarch64-apple-ios-sim
ios-build      # signed App Store Connect archive/export
```

With no arguments `ios-dev` boots an iPhone simulator and deploys there, even
when a physical iPhone is plugged in (Tauri's own picker would otherwise grab
the phone and start a provisioning device build). It prefers `iPhone 17 Pro`,
falls back to the first available iPhone, and `MOLD_IOS_DEVICE` overrides the
preference. Pass a device name to target hardware: `ios-dev "James’s iPhone"`.

The underlying script also exposes setup and a deterministic simulator build:

```bash
./scripts/ios.sh init
./scripts/ios.sh simulator
```

Install the shared workspace at the repository root, then run the mobile
commands from there:

```bash
bun install --frozen-lockfile
bun --cwd desktop run dev:mobile
bun run build:mobile
bun --cwd desktop run test -- src/mobile
```

Tauri always boots a root `index.html`. The mobile Vite build starts from
`index.mobile.html` and renames the emitted file to `dist-mobile/index.html`.
Do not point Tauri at the desktop entry or remove this rename.

## Validation

Before publishing a mobile change, run the checks appropriate to the files
changed:

```bash
cd desktop
bun run fmt:check
bun run test
bun run build:mobile

cd ..
android-doctor
android-check
./scripts/tests/ios-release-assets.sh
./scripts/ios.sh check
./scripts/ios.sh simulator
```

`scripts/tests/ios-release-assets.sh` rebuilds the frontend and verifies that
the archive entry is `index.html`, the release marker is present, and the Apple
icon catalog contains the current opaque Mold artwork. If
`desktop/icon-master.png` changes, regenerate every iOS icon and its checksums:

```bash
./scripts/generate-ios-icons.sh
```

Keep `apps/mobile/src-tauri/Info.ios.plist`, the generated Apple plist, and
`gen/apple/project.yml` aligned when native capabilities change. Simulator
builds must retain Xcode's ad-hoc signature so Keychain access works.

## CI and TestFlight

`.github/workflows/ios.yml` runs for mobile-relevant pull requests and `main`
changes, including shared component changes imported by the mobile entry. It
verifies the frontend/release assets, runs the frontend tests and formatting
gate, and runs Rust fmt, check, and warning-denied Clippy for the Apple Silicon
simulator.

`.github/workflows/testflight-ios.yml` is the TestFlight pipeline. It runs after
a successful `iOS` workflow on `main`, or by manual dispatch. Uploads remain
eligible for both internal and external testing; never restore Apple's
`testFlightInternalTestingOnly` export restriction. The workflow:

1. resolves the marketing version from the mobile crate and uses the full Git
   commit count as the default build number;
2. rebuilds and validates the packaged mobile entry and icon catalog;
3. creates an unsigned archive, then uses App Store Connect automatic signing
   during export/upload;
4. uploads an external-testing-eligible build and waits until App Store Connect
   reports `processingState == VALID`; and
5. verifies the baseline internal release path by confirming
   `brink.james@gmail.com` has access through `Mold Internal`.

Required repository secrets are `APPLE_API_PRIVATE_KEY`, `APPLE_API_KEY`, and
`APPLE_API_ISSUER`. Never print or persist their values.

`testflight-ios-verify.yml` can resume validation for an exact uploaded bundle
version without rebuilding it. Upload success alone is not a completed release;
the automated finish line is a `VALID` build plus verified internal tester
access. External groups can then select that same build and submit it to Beta
App Review in App Store Connect.

Release PRs synchronize `apps/mobile/src-tauri/Cargo.toml`, its lockfile, and
`tauri.conf.json` through `scripts/release/sync-release-pr.sh`. Do not hand-bump
mobile versions independently of the workspace release.
