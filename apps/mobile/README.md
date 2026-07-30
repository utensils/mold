# Mold for iPhone

Mold is equally owned and maintained by core contributors James Brink and
Jeffrey Dilley.

Mold for iPhone is a remote-only Tauri 2 client. It never links or embeds the
GPU inference stack; a saved Mold server owns the models, queue, downloads,
generation work, and gallery media.

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
- Bundle identifier: `com.utensils.mold`
- Minimum iOS version: 17.0

The thin mobile crate is excluded from the root Cargo workspace so an iOS build
never cross-compiles the desktop server or inference dependency tree. Native
commands are intentionally limited to platform responsibilities:

- Keychain storage for per-host API keys
- Bonjour/DNS-SD discovery of `_mold._tcp`
- UIKit appearance synchronization for readable system chrome

Everything a remote Mold server can answer uses the same authenticated HTTP
and SSE contract as desktop. Do not import desktop stores that assume a local
**This device** engine; adapt shared pure helpers to an explicit `ApiTarget`
instead.

## Current product surface

The primary tabs are Create, Library, Models, and Machines. Settings is a
pushed screen opened from the header.

- **Create** supports model-aware image and video controls, Batch 1 quick
  expansion/undo, Batch N prepared-variation review, remote prompt history,
  local templates, independently cancellable siblings, source/edit images,
  masks, ControlNet, LoRA, scheduler and CFG++, post-generation upscaling,
  target-host estimates, proportional resolution choices, and explicit Random
  or Fixed seeds. Deeper options open in a full-screen **Advanced** sheet, and
  prompt **style** presets compose at submit without rewriting the prompt text.
  A **↺ Reset** beside the Advanced trigger restores every generation setting
  to the selected model's defaults while preserving the prompt, model choice,
  and any prepared batch size; the Advanced sheet keeps its narrower
  advanced-only Reset.
  For LTX-2 checkpoints, the sheet honors additive `supports_audio` model
  metadata: video-only community checkpoints disable generated audio with an
  explanation while source-image video remains available.
  Source-fit and Upscale then fit preprocessing use a per-Create-form cache, so
  unchanged Batch siblings and repeat submissions share one host upscale and
  fitted source while keeping the editable original intact. When the host
  streams live latent previews (`preview` SSE events), the active print
  develops in a bed above the status line — the preview sharpens with denoise
  progress under the shared thinning Develop grain, matching the print's
  aspect ratio; without previews the plain status line remains.
- **Library** merges saved media from every configured host. Its full-screen
  viewer shows uncropped images, streams videos with native controls, swipes
  horizontally between prints, explicitly copies or saves full-resolution
  stills through UIKit, restores recorded prompt settings, and can use a still
  as the next source or Qwen edit target. On a print a sequence produced, **Use
  as prompt** reloads that sequence's recorded clips onto the Create clip rail
  as a new draft (raising any clip duration the selected model's motion tail no
  longer allows, and saying so); iPhone is reuse-only — **Edit sequence** stays
  a desktop/web action until mobile has a chain-detail recovery route.
  Generated stills open the same viewer on tap. Persistent New badges match desktop Library visits, and post-generation
  upscaled images carry the shared Upscaled badge.
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
- **Machines** supports Bonjour discovery, manual IP/hostname/HTTPS entry, and
  Tailscale MagicDNS. Host detail shows telemetry, models-disk usage, queue,
  downloads, loaded models, and installed models (all using catalog display
  names rather than opaque `cv:` / `hf:` ids), with rename, retry, select,
  unload, open-in-Models, and forget actions. Still-queued generation rows have
  a 44pt two-tap **Cancel** action against that exact Keychain-authenticated
  host; running work remains visible but cannot be preempted.
  Current V2 hosts also expose every GPU/MIG device and its queue lane. Device
  lifecycle controls are shown only when the host advertises
  `devices.lifecycle`; disabling a busy device leaves its current work running
  and shows the draining state until completion.
- **Settings** persists the Mold Studio theme families (Mold or Safelight) and
  System, Dark, or Light appearance. Fresh installs start with Safelight +
  System; valid saved choices remain authoritative. Its default-on Photos
  preference automatically fetches each completed still from its authenticated
  host gallery and saves it through UIKit; post-generation upscales save both
  images, while videos remain in Mold Library. Settings also links to host
  management and shows the app version, remote-only processing policy, and
  TestFlight update channel. About opens the public privacy policy at
  `https://utensils.io/mold/privacy` through the native external-browser opener.

The app shell suppresses WebKit focus/double-tap page zoom and rubber-band
overscroll. The Library viewer keeps its scoped horizontal swipe gesture.

Prepared expansion always snapshots the selected remote host ID, endpoint,
Keychain-provided key, and server instance. Batch is a directly editable
positive count. Batch N requires exactly N distinct non-empty prompts before
its inline review workspace appears; counts above eight use a compact
first-eight summary and bounded Review all pages. Edits and specifically named
stale work remain local until explicit approval, refresh, collapse, or discard.
One reviewed set is capped at 10,000 variations for memory safety; accepted
sets do not impose a cumulative queue limit.
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
- `mold.mobile.settings.v1` — appearance, color family, and Photos auto-save preference
- `mold.mobile.generation.templates.v1` — mobile-local generation templates
- `mold.mobile.library-seen-at.v1` / `mold.mobile.library-visited.v1` — bounded
  per-host latest-print timestamps and the first-visit marker for New badges

Per-host API keys live in the iOS Keychain under
`com.utensils.mold.remote-api-key`. Never move them into local storage, query
parameters, logs, or generated project files.

Authenticated gallery media uses `POST /api/gallery/media-token` to exchange
the normal `X-Api-Key` request for a short-lived, read-only URL scoped to one
`/api/gallery/image/:filename` path. This allows native video Range requests and
seeking without exposing the long-lived API key. Keep the image-only fallback
for older hosts, but never buffer a whole video as that fallback.

## Local development

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
