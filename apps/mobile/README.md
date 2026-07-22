# Mold for iPhone

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
  Source-fit and Upscale then fit preprocessing use a per-Create-form cache, so
  unchanged Batch siblings and repeat submissions share one host upscale and
  fitted source while keeping the editable original intact.
- **Library** merges saved media from every configured host. Its full-screen
  viewer shows uncropped images, streams videos with native controls, swipes
  horizontally between prints, restores recorded prompt settings, and can use
  a still as the next source or Qwen edit target.
- **Models** merges installed models with Hugging Face and Civitai results,
  supports host/media/source/family filters, exposes model details and
  components, and routes pull/load/unload/remove actions to the owning or
  selected host. Pull actions progress through `Connecting...`, `Starting...`,
  `Queued`, and `Pulling N%`; active downloads can be cancelled.
- **Machines** supports Bonjour discovery, manual IP/hostname/HTTPS entry, and
  Tailscale MagicDNS. Host detail shows telemetry, models-disk usage, queue,
  downloads, loaded models, and installed models, with rename, retry, select,
  unload, open-in-Models, and forget actions.
- **Settings** persists the Mold Studio theme families (Mold or Safelight) and
  System, Dark, or Light appearance. Fresh installs start with Safelight +
  System; valid saved choices remain authoritative. Settings also links to host
  management and shows the app version, remote-only processing policy, and
  TestFlight update channel.

The app shell suppresses WebKit focus/double-tap page zoom and rubber-band
overscroll. The Library viewer keeps its scoped horizontal swipe gesture.

Prepared expansion always snapshots the selected remote host ID, endpoint,
Keychain-provided key, and server instance. Batch N requires exactly N non-empty
prompts before its inline review workspace appears; edits and specifically named
stale work remain local until explicit approval, refresh, collapse, or discard.
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

The initial mobile scope does not include a local engine, the desktop Chains
editor and durable-jobs workspace, RunPod provisioning, desktop engine
settings, or desktop self-update channels.

## Persistence and security

WebView local storage contains non-secret mobile state:

- `mold.mobile.hosts.v1` — host metadata with API keys removed
- `mold.mobile.selected-host.v1` — selected generation host
- `mold.mobile.settings.v1` — appearance and color family
- `mold.mobile.generation.templates.v1` — mobile-local generation templates

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
