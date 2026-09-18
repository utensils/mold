# Native macOS app: fix + polish + parity pass (branch `feat/macos-native-app`)

## Context

A six-reviewer audit of `apps/macos` (2026-09-17) found 15 HIGH, ~45 MED and ~25 LOW issues
plus a parity matrix against the Tauri desktop app. Headlines: finished clips never display,
img2img is unreachable on SD1.5/SDXL/Klein, a downward batch reorder splits the batch, pairing
codes always read "Expired", live state never reconciles after a reconnect, path traversal from
server-supplied filenames, a keyless CORS-permissive local engine, and a release path that
cannot produce a DMG that launches on another Mac (and `make engine` no longer resolves).

Decisions taken with James (2026-09-17):
- **Meshes**: native interactive GLB viewer, parity with `studio/components/MeshViewer.vue`.
- **Updater**: Sparkle 2 now.
- **Host API keys**: out of the Keychain into an owner-only 0600 file, as `.claude/rules/desktop.md` requires.
- **Parity in scope**: full Reuse + retained source media; Library upscale + `/api/activity`;
  Generate controls. **Out**: RunPod, mold-home relocation, profile editing, 3-D studio, chain
  *authoring* (README omissions stand; RunPod gets added to that list).

Everything lands on `feat/macos-native-app` — never merged, never force-pushed, one
Conventional Commit per finding-sized slice, pushed per slice. No changelog fragments (branch
never ships through release-plz); `apps/macos/README.md` is the doc that must stay true.

## Ground rules for every lane

- **TDD**: failing test first, doc-commented `**Fails today**: …` (house convention), then the fix.
  swift-testing only; store tests use `FakeBackend` + `FakeFixtures` + `settle(until:)`
  (`Tests/MoldTests/FakeBackend.swift`, `FakeFixtures.swift`); fixtures are built by decoding JSON,
  new capability axes get a new `FakeFixtures.capabilities(…)` overload; timing constants are
  constructor parameters, never sleeps. New wire fixtures are captured read-only from plato/hal9000
  with host + version + date + route in the header comment.
- **Lints stay green**: `make lint` (no `JSONDecoder()` outside `MoldJSON`, no `bytes.lines`, no UI
  imports in MoldClient, no literal colours, files ≤150 lines, a11y names). The three types already
  over the type-size budget (`HTTPBackend`, `LibraryStore`, `GenerateController`) must not grow —
  new behaviour goes in new small types.
- **Capabilities are the only authority**; absence = older server, never a refusal. The one
  sanctioned family set is CFG++ (`{"sd3","sd3.5"}`), mirrored from `studio/lib/generationCapabilities.ts:282`
  and pinned by a contract test that reads that file.
- **UX-changing review findings** (key bindings, menu layout) follow the reports' recommendations,
  which all restore what the README already promises; nothing new is invented.
- **Machine load**: at most 3 lanes at once; `swift test` in the package is free; app-bundle
  `xcodebuild test` is serialised through a `mkdir` lock (`/Volumes/ExternalStorage/.mold-macos-xcb.lock`);
  one cargo build at a time, `CARGO_BUILD_JOBS=8`, target dirs on ExternalStorage, deleted after.

## Execution mechanics

I am the integrator; implementation is delegated.
1. Each lane runs as an **Opus `Agent` in its own git worktree** (`isolation: "worktree"`), with a
   disjoint file-ownership list (below). In the worktree: `bun`-free; `make engine-clean` first so the
   build is remote-only (no Rust), then `make gen`.
2. A lane commits per finding on its lane branch and ticks `apps/macos/docs/review-2026-09-17/STATUS.md`
   rows it owns (one file per lane — `STATUS-<lane>.md` — so ledgers never conflict).
3. When a lane reports, a **second Opus agent adversarially reviews the lane diff** against the
   original finding text; confirmed problems go back to the lane agent (`SendMessage`).
4. I `git cherry-pick` the lane's commits onto the main checkout (kept on the feature branch so the
   dev app James watches stays current), run `make lint && make test`, push. Never rebase pushed
   history; conflicts are resolved forward.
5. Waves are strictly ordered; lanes inside a wave are file-disjoint.

## Step 0 — make the source material durable (me, first commit)

Copy the six scratchpad reports to `apps/macos/docs/review-2026-09-17/01…06-*.md`, add
`07-reference-notes.md` (the three exploration digests: desktop secrets/updater/release scripts;
mesh viewer pipeline + GLB writer bounds; reuse/upscale/activity/controls wire tables), and the
per-lane STATUS ledgers. Finding ids below (`01#3`, `05-H1`) refer to these files; each carries
file:line on both sides, a failure scenario and a fix. Commit `docs(macos): peer review reports`.

Then the one-line unblocker, because every later engine build needs it:
`rust/mold-macos-ffi/Cargo.toml` `0.29.0 → 0.30.0` (+ `Cargo.lock`), and teach
`scripts/release/sync-release-pr.sh` the new root (same `package = "mold-ai-` awk rule as desktop;
`sed` for `MARKETING_VERSION` in `project.yml`) with the fixture extended in
`scripts/tests/release-sync-pr.sh`. (05-H2)

## Wave 1 — correctness and security (5 lanes; run A,B,C then D,E)

**Lane A · MoldClient wire** — owns `Packages/MoldClient/**` EXCEPT `RenderDraft*`, `DraftMedia*`,
`GenerateRequest*`, `Temporal.swift`, `Recipe*.swift`, `GalleryPrint.swift`'s `OutputMetadata`.
- 01#1 `QueueOrder.moves`: plan each PATCH in the server's index space (only the moving row is
  removed per call); table test for up, down, across-neighbour, 3-child batches.
- 01#2 never re-encode an `.unknown` open enum into `prompt_transform` (drop the block).
- 01#5 animated GIF/APNG/WebP clip classification; add `hunyuan3d` to the non-picture families
  (`Model.swift`, contract test already reads `manifest.rs`).
- 01#6 a real query escaper; 01#17 `escaped()` on the nine interpolated ids.
- 01#9 stable sort in `QueueListing.merged` (also fixes 04-M6).
- 01#10/#11 SSE refusals keep the server's sentence and licence payload; `TransferPlan` classifies
  401/licence. 01#12 `playableURL` mints a media ticket on a keyed host or fails closed.
- 01#14/#21 one buffering policy for the stream stack (`.bufferingNewest` for previews), chunked
  line splitting. 01#16 strip `X-Api-Key` on cross-origin redirect (session delegate).
- 01#18/#19 tag control chars, `CollectionShelf.hidden`.
- Diagnosability: `os.Logger` categories in MoldClient; `get/post` log the `DecodingError` path
  before collapsing to `.malformedResponse`.
- Test gap: reflection test pinning `GenerateRequest.encode(to:)` exhaustive over its stored properties.

**Lane B · Generate** — owns `Sources/Mold/Generate/**` + the MoldClient files excluded from A.
- 02#1 `PromptPanel`: layout from `source_relation` via a Swift port of `sourceImageModeForReferences`
  (`replaces` → strip only; `exclusive` → both, the idle one parked; `combines` → both live);
  01#4 legacy fallback when the block is absent.
- Parity-HIGH: result canvas branches by kind — `AVPlayer` for clips (ticketed URL), image, mesh
  placeholder hook for Wave 3; failure summary and ResultBar move OUT of the `if let`.
- 02#2 Stop during `.submitting`; 02#9 a finished result is drawn before the next batch takes the canvas.
- 01#3/02#3 length ceiling = `min(max_duration_seconds·fps+4, max_frames, clip ceiling)`; port
  `text_only_auto_chain_refusal` wording from `tests/fixtures/wan/surface-parity-v1.json`.
- 02#5 placement preview sends the redacted request studio sends.
- 02#7 identity photos transcoded to PNG; 02#10 file read/base64/decode off the main actor.
- 01#7/#8 `fit()` honours `off_bucket: warn`; alignment rounds DOWN under `max_pixels`.
- 02#11–#15: `machineChoice` observable, Expand records the real task, staleness fence on
  expand/remix, slider a11y labels, arrow-key shortcuts scoped.

**Lane C · Library** — owns `Sources/Mold/Library/**`, `Support/PrintMaterializer*`, `QuickLook`.
- 03-H1 one `SafeFilename` (in MoldClient, tested: `..`, separators, NUL, leading dot, empty) applied
  at decode of `GalleryPrint.filename` AND at every write/remove site.
- 03-M1 gallery events queued (not dropped) while the outbox is pending; re-list after drain.
- 03-M2 day-grouping respects Sort By; viewer walks the same order. 03-M11 one index per data
  change (mirror `organizationIndex`), not per body pass.
- 03-M3 modifiers from the key event; 03-M4 ⌘⌫ only; 03-M5 Space yields to a first-responder text
  view (reuse the viewer's `editingText`); 03-M7 undo grouped by target, no `removeAllActions()`.
- 03-M8 budget never evicts the file just written or one in use; oversize print reported, not
  silently dropped. 03-M9 response size ceilings (512 MiB media, small JSON).
- 03-L1–L6.

**Lane D · Queue / Models / Machines / events** — owns `Sources/Mold/{Queue,Models,Machines}/**`,
`Support/HostStore+Events.swift`, `Support/HostStore+Reachability.swift`, notification/badge files.
- 04-H1 pairing `expires_at` is seconds (fix tests' unit too; fixture from a keyed host).
- 04-H2 every (re)connect emits `.resyncRequired`; `NSWorkspace.didWakeNotification` forces a
  reconnect; a 10 s poll for hosts where `wantsPoll`, wired in production.
- 04-M2 ignore `catalog_ready`; 04-M3 row Pause/Resume gated on `can_pause_job`; 04-M4 read
  `cooperative_cancellation`; 04-M5 single-flight `hydrate`; 04-M8 await authorization before the
  first `add`; 04-L1–L4 (64 MiB wording, settle loop cancellation, badge out of the view modifier).

**Lane E · Secrets + Settings** — owns `Support/{Keychain,HostPersistence,HostStore+Editing}.swift`,
`Sources/Mold/Settings/**`, `Shell/{HostEditor,MachinesSettings,AccountsSettings}.swift`.
- New `SecretStore`: `~/Library/Application Support/io.utensils.mold.native/secrets.json`, flat
  `{name: value}`, 0600 set on the temp file before an atomic rename, unparseable file parked as
  `.corrupt`, names `remote-api-key.<host-uuid>` and `local-engine-api-key` — a port of
  `desktop/src-tauri/src/secrets.rs` incl. its `secrets_file_is_owner_only` test.
- One-time migration: read each Keychain item → file → delete item; then delete `Keychain.swift`.
- 05-H5 persistence is explicit `set`/`clear` per host — saving the list never implies a delete.
- 05-H6 removing a machine uses a plain `ConfirmDialog` + danger button (no typed confirm).
- 05-M11 env-sourced rows read-only in curated panes; 05-M12 Return on an empty secret is a no-op;
  05-M13 Reset does what its sentence says (or the sentence changes).
- `#if DEBUG` around all eight `MOLD_NATIVE_*` hooks (`macos-uat` builds Debug already).

## Wave 2 — engine, FFI, release (1 lane, after E; needs `SecretStore`)

**Lane F** — owns `Sources/Mold/Engine/**`, `rust/mold-macos-ffi/**`, `Makefile`, `scripts/**`,
`project.yml`, `Info.plist`, `flake.nix` macos commands, `.github/workflows/macos-native.yml`.
- 05-H1 mint a UUID `local-engine-api-key` (precedence `MOLD_API_KEY` → stored → new, as
  `SecretStore::local_server_api_key`), pass it to `mold_engine_bootstrap`, and export
  `MOLD_CORS_ORIGIN` to a non-web origin so no browser page gets an ACAO. The local `MoldHost`
  carries the key. README's "cannot be paired" paragraph is rewritten accordingly.
- 05-M1/M9 FFI: `catch_unwind` + a drop guard that clears `ALIVE`; terminal error through
  `tracing::error!`; `join` bounded. Swift polls `mold_engine_is_alive` → `.failed(reason)`.
- 05-M2 Start works after a failure (or says relaunch is needed — it is, by the OnceLock; say so).
  05-M3 `.running` only after `/api/status` answers. 05-M4 bootstrap moved to `App.init`.
  05-M5 corrupt home pointer fails closed like Rust. 05-M6 second-engine interlock on the home.
  05-M7 quit honours the server's 45 s budget behind a "Finishing…" sheet with Quit Now; writer
  lease released. 05-M8 documented + SIGTERM forwarded to `NSApp.terminate`.
- 05-H3 `scripts/fix-macos-native-linkage.sh` — the desktop script's `install_name_tool -change`
  pair + its fail-closed `/nix/store` grep, run by `make signed` before signing.
  05-H4 `release` depends on `engine` and asserts `MOLD_EMBEDDED_ENGINE`. 05-M17 `ENGINE_TARGET`
  defaults to an in-repo ignored dir with the ExternalStorage path as James's override.
  05-M14 sign nested code WITHOUT the app entitlements; drop `allow-unsigned-executable-memory`
  if a signed Metal render still works. 05-M10 `pkill` by bundle path, not `-x Mold`.
- CI: `macos-native.yml`, path-filtered to `apps/macos/**`: `make lint`, `swift test`, remote-only
  `xcodebuild test`. If no hosted runner carries the macOS 26 SDK, the workflow is committed
  `workflow_dispatch`-only and the README says so — no fake green.

## Wave 3 — features (serial where they share the draft/request types)

**F1 · Mesh viewer** (parallel with F3). Bound: mold's writer emits one mesh/primitive, u32 indices,
float POSITION/NORMAL/TEXCOORD_0/COLOR_0(VEC3), buffer 0, embedded PNGs, no extensions, doubleSided.
- MoldClient (pure, tested): `GLB.swift` = port of `studio/lib/glb.ts` incl. every bounds check and
  generated normals; `GLBFixture` = port of `glbFixture.ts`; tests = port of `glb.test.ts`.
  `MeshViewerCamera.swift` = the four constants + `sweepProfile/sweepExtentOfProfile/orthographicScale`
  + column-major mat4, tests ported from `meshViewerCamera.test.ts`. A fifth arm in
  `poster.rs::the_viewer_mirrors_the_poster_camera` reads the Swift file (parser gains a `let` form).
- App: `MeshView` — `MTKView` via `NSViewRepresentable`, `nonisolated` delegate (the `QuickLook.swift`
  pattern), a Metal shader that is the GLSL line for line (orthographic, view-space key/fill, rim,
  back-face normal flip, no culling, wireframe pass with polygon offset, transparent clear over the
  media bed). Drag orbit, scroll/pinch zoom 0.25–6, arrows/±/0, double-click reset, auto-rotate
  0.25 rad/s that the first interaction ends and Reduce Motion disables, 256 MiB cap, every failure
  lands on the poster with one sentence. Home view == poster, pinned by the camera tests.
- Plugged into `LibraryViewer` (mesh arm before image) and the Generate result canvas; arrow keys
  belong to the mesh view while it has focus. Quick Look on a mesh shows its poster.
- Exports: list from `capabilities.mesh.export_formats` via a port of `splitMeshExportFormats`
  (delete `ExportOptions`' client constants, 03-L1); geometry options only when
  `export_geometry` is present, defaults from its table; turntable sheet (frames, fps,
  max_dimension, transparent); body = `meshExportRequest`.

**F3 · Library upscale + activity** (parallel with F1).
- Backend verbs: `POST /api/gallery/upscale`, `/api/video-upscale-jobs` (+ get/pause/resume/delete).
  Gate on `capabilities.video_upscale` (`gallery_image` for stills). Library menu "Make Bigger…",
  default upscaler per `studio/lib/upscale.ts::defaultUpscaler`, 750 ms poll behind an epoch fence,
  recovery of a non-terminal job on open, host gallery refresh on completion.
- `ActivityStore`: `GET /api/activity` every 5 s, reconcile rules ported from
  `studio/api/activity.ts` (wholesale replace; stale-retain on error; `unavailable_kinds`;
  `execution == "chain"` → chain authority). Queue pane gains an "Also running" section for rows
  with no queue entry (`can_cancel` respected).

**F4 · Generate controls** (after Lane B; before F2).
- IP-Adapter weight slider from `reference_images.weight`; scheduler picker from
  `capabilities.schedulers`; CFG++; Wan recipe (`sample_shift`, distill high/low) from
  `wan_recipe`; LTX-2 `guidance_overrides`; all in the inspector, each absent when unadvertised
  and parked on model switch through the existing `DraftMedia+Park` machinery.
- `source_fit`: port `studio/lib/sourceFit.ts` + `sourceResolution.ts` (crop-fill default, canvas
  follows an attached source only per the recorded canvas intent); preprocessing off-main.
- Whole-queue pause: `pauseQueue/resumeQueue` verbs, Queue menu item gated on `canPauseQueue`.
- Long clips: past the clip size a render becomes an **ephemeral** chain (`POST /api/chain-jobs`
  with `ephemeral: true`, follow `/api/chain-jobs/{id}/events`), port of `decideChainRouting` /
  `buildAutoChainRequest`; refused by name where `text_only_auto_chain_refusal` applies.
- Draft persistence: versioned descriptor in app support (explicit `CodingKeys`, no key strategy —
  the snake_case trap), bytes excluded exactly as `sanitizePersistedForm` excludes them.

**F2 · Reuse + retained source media** (after F4, so it restores the new fields).
- Widen `OutputMetadata` to everything `applyMetadataToForm` reads (`generateForm.ts:1455-1609`);
  `RenderDraft(reusing:)` restores that table; media explicitly cleared then re-attached.
- Always probe `GET /api/gallery/source-media/:filename` on every known copy of the print (port of
  `useReuseStillPrint.ts`); disclose only when `retainedSourceMediaDisclosable`; 401 → `unavailable_auth`.
- Same host: mint a reuse session and send `x-mold-retained-media-session` on the one-child submit
  (batch >1 refused with the server's sentence). Cross host: download-and-inline relay. Any draft
  edit to a hydrated role invalidates the session.

**F5 · Sparkle 2** (after Lane F).
- SPM dependency in `project.yml`; `SUPublicEDKey`; feeds on GitHub Releases mirroring desktop:
  stable `releases/latest/download/mold-native-appcast.xml`, nightly
  `releases/download/latest/mold-native-appcast-nightly.xml`; channel picker in Settings ▸ General
  (stable default), "Check for Updates…" in the app menu; feed URLs pinned by an allowlist test.
- `make appcast` (`generate_appcast`/`sign_update`), and `macos-native-distribution.yml` modelled on
  `desktop-distribution.yml` (ephemeral keychain, assets first, pointer last, anonymous re-verify).
- **Needs James once**: run Sparkle's `generate_keys`, put the private key in the
  `MOLD_NATIVE_SPARKLE_KEY` secret; I commit only the public key. Until then the updater is built
  and tested but has no feed to answer it — the README will say exactly that.

## Wave 4 — polish, docs, verification

- HIG/a11y sweep over what the waves touched (VoiceOver names, focus order, menu enablement,
  empty/error sentences through one `HostFailure` voice), then `apps/macos/README.md` rewritten to
  match: keys file, keyed engine, mesh viewer, updater, new controls, RunPod in "Not built yet".
  `.claude/rules/` gains `macos-native.md` with the invariants this pass establishes.
- A fresh Opus peer review of the whole range `a9f42603..HEAD`; its confirmed findings are fixed
  before I call the pass done. Reviewer UX suggestions that change bindings/layout come to James.

## Verification

- Per slice: the new failing test goes green; `make lint`; `swift test` (414 today — the count only rises).
- Per integration: `make test` (package + app bundle) on the main checkout; push.
- Engine (after Lane F): `make engine` resolves and links; `otool -L` on the Release binary shows
  no `/nix/store`; `curl` to the loopback port without the key is 401; a cross-origin `fetch`
  gets no ACAO; kill -TERM quits cleanly; a forced engine panic surfaces as `.failed`.
- UAT via `macos-uat` (throwaway prefs + home), driven by AX menu presses (never osascript
  keystrokes, never closing Ghostty), against hal9000 (keyless, 100.123.198.98) and plato (keyed):
  SD1.5 img2img with a reference; an LTX-2 clip that plays on the canvas; a clip past the clip size
  that chains as one print; drag a 2-child batch downward; sleep/wake then watch the queue resync;
  pairing sheet counts down; a hostile-filename fixture is refused; open a GLB — home view matches
  its thumbnail, orbit, export STL at 100 mm Z-up and check its bounds; Reuse a print with a source
  image on the same host and across hosts; upscale a still and a clip; pause the whole queue;
  quit and relaunch — the draft is back, the keys are in `secrets.json` (0600) and the Keychain is empty.
- Sparkle: `generate_appcast` output validates and the feed-allowlist test passes; an end-to-end
  update is verified only once James has installed the key.

## Deviations recorded during execution

- **Step 0 (05-H2)**: the plan said to teach `scripts/release/sync-release-pr.sh` about the FFI root.
  That script only ever runs on `main`'s release PR and this branch never merges, so it could never
  fire. Instead the `mold-ai-*` path dependencies carry no `version` requirement at all
  (`publish = false`), which cannot drift. Commit `34cfa484`.

- **2026-09-17 (late), owner decision**: the native app SHIPS ALONGSIDE the Tauri desktop app, both kept, for a
  handful of releases until one path is chosen (likely native). Consequences: the branch WILL merge (drop the
  "never merged" wording; add a `changelog.d/` fragment in Wave 4); the native DMG rides the SAME releases and
  channels as desktop — stable on the `v*` release cut by release-plz (`release.yml`), nightly on the rolling
  `latest` prerelease (`desktop.yml` order, verbatim) — never its own tags or Latest pointer (Sparkle review
  HIGH 2); side-by-side installs on one Mac sharing a `MOLD_HOME` are a real UAT case (distinct bundle ids,
  keyed engine, writer-lease advisory). The final step of the pass is that release CI, correct by reading,
  since it can only run on `main`.
