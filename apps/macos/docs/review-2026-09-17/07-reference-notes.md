# Reference notes for the fix + parity pass

Digests of three read-only explorations (2026-09-17). Line numbers are from the tree at
`a9f42603`; verify before relying on one. Everything here is a pointer to the reference
implementation a lane must mirror — never port from this file alone, open the source.

## 1. Secrets, updater, release plumbing (Tauri desktop as the reference)

### Secrets file (`desktop/src-tauri/src/secrets.rs`)
- Rule: `.claude/rules/desktop.md:12` — file-backed, owner-only `secrets.json`; "deliberately NOT the
  macOS Keychain, whose prompts users found obnoxious; don't reintroduce `keyring`".
- Path `<app_data_dir>/secrets.json`; flat `{"name": "value"}` pretty JSON (`:157-165`).
- `0600` set on `secrets.json.tmp` BEFORE the atomic rename (`:160-166`), pinned by
  `secrets_file_is_owner_only` (`:296-308`). Unparseable file is moved to `secrets.json.corrupt`,
  never clobbered (`:151-156`). Whole read-modify-write under one mutex (`:44-50`).
- Name allowlist (`:30-40`): `hf-token`, `civitai-token`, `remote-api-key`, `runpod-api-key`,
  `desktop-local-api-key`, and `remote-api-key.<slug>` with suffix `[A-Za-z0-9._-]+`.
- Local engine key: `local_server_api_key()` (`:103-119`) — env `MOLD_API_KEY` (non-empty) →
  persisted → `Uuid::new_v4()` persisted. Exported via `std::env::set_var("MOLD_API_KEY", …)` once
  at setup before threads (`desktop/src-tauri/src/lib.rs:158-168`); `run_server` takes no key argument.

### macOS app today
- `Support/Keychain.swift` (42 lines): `apiKey(for:)`, `setAPIKey(_:for:)` = delete-then-add, nil/empty
  = delete. Service `io.utensils.mold.native`, account = host UUID. All statuses swallowed.
- Only caller: `Support/HostPersistence.swift` — `load` (`:12-18`), `save` loops `setAPIKey(host.apiKey…)`
  for every host (`:20-26`) — THIS is the wipe bug: an absent in-memory key becomes a delete —
  `forget` (`:28-30`). Callers: `HostStore.seededHosts()` (`HostStore.swift:73`),
  `HostStore+Editing.swift` `persist()` `:112-116`, `remove(_:)` `:118-129`.
- `StoredHost` = `{id, name, base_url}` in `UserDefaults["hosts"]`; suite swaps to
  `io.utensils.mold.native.fresh` under `MOLD_NATIVE_FRESH` (`AppStorageSuite.swift:9-17`).
- Catalog tokens are NOT stored on this Mac at all (`AccountsSettings.swift:6-11`,
  `CatalogStore+Credentials.swift`). Nothing to migrate there.

### CORS
`crates/mold-server/src/lib.rs:1890-1934` `build_cors_layer`: a non-empty `MOLD_CORS_ORIGIN` env var
selects the restrictive arm; anything else is `CorsLayer::permissive()`. A malformed value is a hard
startup error. Desktop does not set it (it relies on its CSP + the key).

### Updater shape to mirror with Sparkle
- Endpoints (`desktop/src-tauri/src/updater.rs:23-26`): stable
  `https://github.com/utensils/mold/releases/latest/download/mold-desktop-stable.json`, nightly
  `https://github.com/utensils/mold/releases/download/latest/mold-desktop-nightly.json`.
  Pinned by `endpoints_are_fixed_https_allowlist` (`:717-724`).
- Channel enum `Stable|Nightly`, Stable default (`settings.rs:127-133, 333-335`).
- CI: reusable `.github/workflows/desktop-distribution.yml` (ephemeral keychain `:149-166`, ASC key
  0600 `:168-176`, build `:178-200`, notarize `:202-207`, verify `:209-234`, manifest `:246-254`).
  Callers: `desktop.yml:570-726` (nightly, push to main) and `release.yml:628-635` (stable, tags).
- Publish order worth copying verbatim (`desktop.yml:636-723`): upload immutable assets → poll the
  anonymous URL and compare SHA-256 → re-check `main` HEAD is still ours → THEN clobber the channel
  pointer → re-verify anonymously → prune to 10.
- Hosting is GitHub Releases only. No gh-pages, no utensils.io.

### Release scripts
- `scripts/fix-desktop-macos-linkage.sh` (65 lines): `install_name_tool -change` of
  `/nix/store/*-libcxx-*/lib/libc++.1.0.dylib → /usr/lib/libc++.1.dylib` and
  `/nix/store/*-libiconv-*/lib/libiconv.2.dylib → /usr/lib/libiconv.2.dylib` (`:42-51`), then
  fail-closed on any remaining `/nix/store` in `otool -L` (`:53-58`).
  NOTE: the native app DOES need this. `otool -L build/Debug/Mold.app/Contents/MacOS/Mold.debug.dylib`
  shows both Nix paths (verified 2026-09-17) because the link runs inside `nix develop`.
- `scripts/release/sync-release-pr.sh` (156 lines): truth = `[workspace.package] version` in root
  `Cargo.toml`. Block 2 (`:106-131`) bumps `desktop/src-tauri/Cargo.toml` (first `^version = "` and
  every line containing `package = "mold-ai-`), its `Cargo.lock`, `desktop/package.json`. Block 3
  (`:133-156`) does mobile. Test: `scripts/tests/release-sync-pr.sh` (fixture tree `:59-113`,
  exact-grep assertions `:144-154` incl. collateral-damage negatives).
- `flake.nix:1880-1937` darwin devshell commands: `macos-dev` (uses `pkill -x Mold` — also kills
  Mold Desktop), `macos-uat`, `macos-build`, `macos-test`, `macos-lint`, `macos-gen`.

## 2. Mesh viewer (reference: `studio/`)

| file | lines | role |
|---|---|---|
| `studio/components/MeshViewer.vue` | 1137 | GL state, events, shaders `:105-158` |
| `studio/lib/glb.ts` | 623 | container + accessor parser |
| `studio/lib/meshViewerCamera.ts` | 259 | constants, sweep fit, mat4 |
| `studio/lib/meshViewerMath.ts` | 106 | auto-rotate, edge list |
| `studio/lib/meshExport.ts` | 247 | export menu policy + request body |
| `studio/lib/glbFixture.ts` | 239 | synthetic GLB builder for tests |

### What mold's writer emits (`crates/mold-inference/src/hunyuan3d/glb.rs:128-423`)
One scene/node/mesh/primitive, NO node transform; indices always u32; POSITION VEC3 f32 with min/max;
optional TEXCOORD_0 (VEC2 f32), COLOR_0 (**VEC3** f32), NORMAL (VEC3 f32); tight stride, buffer 0
only; embedded PNG textures (baseColor, metallicRoughness, normal, optional occlusion sharing MR);
one sampler LINEAR/CLAMP_TO_EDGE; `doubleSided: true`; no extensions. Space: roughly unit cube, Y up,
centred on the query GRID not the model — so orbit the bounding-box centre, never the origin.

### Parser contract (`glb.ts`)
`splitContainer :165-227` (magic, version==2, declared length == byte length, 4-padded chunks);
`accessorLayout :241-330` (every bound checked before reading; `bufferView.buffer` must be 0);
reads mesh 0 primitive 0, `mode == 4`; indices u8/u16/u32 widened, absent → `0..<n`, every index
`< vertexCount`; NORMAL absent → area-weighted `generateNormals :417-461` (degenerate → +Y);
COLOR_0 VEC4 accepted, alpha dropped; only `pbrMetallicRoughness.baseColorTexture` is read, `uri`
images never fetched; non-finite position throws. Size cap 256 MiB (`MeshViewer.vue:225`).

### Shading (fragment, `MeshViewer.vue:127-158`) — port line for line
```
wireframe → vec4(0.16, 0.85, 0.98, 1)
normal = normalize(vNormal); if (!front_facing) normal = -normal
key  = normalize(0.45, 0.72, 0.85)   fill = normalize(-0.65, -0.15, 0.35)   // VIEW space, fixed
kd = max(dot(n,key),0)  fd = max(dot(n,fill),0)  rim = pow(1 - max(dot(n,eye),0), 2.5)
albedo = vColor (default 0.82,0.82,0.86) * texture.rgb (if any)
lit = albedo * (0.20 + 0.72*kd + 0.22*fd) + (0.16,0.19,0.24)*rim ;  clamp 0..1, alpha 1
```
No culling, depth test on, transparent clear over `--mold-media-bed` (#141110), no mipmaps,
no gamma/tonemap, wireframe pass with polygon offset (1,1) on the fill.

### Camera (`meshViewerCamera.ts`)
`POSTER_AZIMUTH_DEG = 30`, `POSTER_ELEVATION_DEG = 20`, `POSTER_MARGIN = 0.08`,
`TURNTABLE_AZIMUTH_STEP_SIGN = -1`; `yaw = -azimuth`, `pitch = elevation`, `zoom = 1` at home.
`sweepProfile :94-115`, `sweepExtentOfProfile :124-142`
(`max(radial, |cosE|*height + |sinE|*radial)`), `orthographicScale :153-167` (returns 0, never ∞),
`orthographic :231-244`. Draw (`MeshViewer.vue:277-345`):
`modelView = T(0,0,-3r) · RX(pitch) · RY(yaw) · T(-center)`;
`scale = orthographicScale(max(homeExtent, sweepExtent(pitch)), w, h, MARGIN) / zoom`;
`projection = orthographic(w/2/scale, h/2/scale, 3r-2r, 3r+2r)`; backing-store pixels, DPR ≤ 2.

### Interaction
Drag orbit `0.008 rad/px`, pitch clamp `±(π/2 − 0.01)`; wheel `exp(dy*0.0015)`; pinch; zoom 0.25–6;
NO pan; reset = `0`, double-click, button; arrows `0.12` (shift `0.30`), `+`/`-` ×1.15;
auto-rotate `0.25 rad/s`, step ≤ 100 ms, ends permanently on first interaction, off under Reduce Motion;
every failure lands on the poster with one sentence.

### Pins
`studio/lib/meshViewerCamera.test.ts` (348), `meshViewerMath.test.ts` (160), `glb.test.ts` (198).
Rust: `crates/mold-inference/src/hunyuan3d/poster.rs:1362 the_viewer_mirrors_the_poster_camera`
reads `studio/lib/meshViewerCamera.ts` with `ts_export_const` (`:1416-1443`). Add a Swift arm.

### Export
Body `GalleryExportRequest` (`crates/mold-server/src/routes.rs:9452-9490`): `format`, `playback`,
`repeat`, `max_dimension` (512, ≤2048), `fps` (10, ≤30), `frames` (36, 8–180), `size_mm`, `up_axis`,
`origin`, `transparent`. Geometry keys REFUSED on glb/turntable; turntable keys refused on geometry.
`MeshCapabilities` (`crates/mold-core/src/types.rs:11958-11996`), `export_geometry` optional
(`:12418-12425`) — absence is the only gate. Swift today: `GalleryMutations.swift:113-126` hard-codes
the sets and `HTTPBackend+Gallery.swift:54-61` posts `{"format": …}` only.

### Where it plugs in
`Library/LibraryViewer.swift:30-51` two-way branch (video | image) and `load() :120-139`; fetch mesh
bytes via `actions.data(for:)` (carries the key), not `playableURL`. Its arrow keys are window-scoped
key equivalents (`:116-118`) — the mesh view must own arrows while focused.
`Generate/RunCanvas+Result.swift` has no kind branch at all. Nothing on macOS previews `.glb`
(ModelIO/SceneKit/RealityKit/Quick Look: none). No 3-D framework is imported anywhere yet.
Delegates arrive off-main under `SWIFT_DEFAULT_ACTOR_ISOLATION: MainActor` — follow
`Support/QuickLook.swift:12-17` (`nonisolated`, `@unchecked Sendable`).

## 3. Reuse, upscale, activity, controls

### Retained source media (`crates/mold-server/src/gallery_source_media.rs`)
- `GET /api/gallery/source-media/:filename` → `{availability, members?[{member_id, role,
  display_name, size_bytes}]}`; availability ∈ `available | unavailable_legacy |
  unavailable_missing_or_corrupt | unavailable_auth`; `members` omitted when empty. Empty after a
  CLEAN resolve is `legacy`, not corruption (`:89-104`).
- `GET …/:member_id` → octet-stream, 512 MiB ceiling, hard 401 when unauthorized.
- `POST …/reuse-sessions` body `{target_request, member_ids}` → `{instance_id, expires_at,
  request_sha256, session_handle}`; TTL 120 s, ≤64 members; binds instance + credential + request
  digest + archive identity. Redeemed by header `x-mold-retained-media-session` on the generate
  call; exactly ONE child (`RETAINED_MEDIA_REUSE_BATCH_AMBIGUOUS` 422); target must not already
  carry authority for a selected role (`…_TARGET_CONFLICT` 409).
- Client reference: `studio/api/gallerySourceMedia.ts` — `retainedSourceMediaDisclosable :154-172`
  (disclosure only, never whether to probe), `retainedSourceMediaDisclosure :175-188`,
  `REQUEST_FIELD_FOR_ROLE :206-221`, `retainedSourceMediaMembersForRequest :226-248`,
  cross-host `relayRetainedSourceMedia :254-361`; 401 → `unavailable_auth` (`:60-62`).
  `desktop/src/composables/useReuseStillPrint.ts` probes EVERY known copy (`:45-72`).
- What studio restores: `desktop/src/lib/generateForm.ts:1455-1609` `applyMetadataToForm` (model,
  prompt first-stage, title, tags+collection, negative, generation w/h, steps, guidance, seed,
  scheduler, cfg_plus, strength, source_fit, loras, control_model/scale, upscale_model,
  output_format, frames/fps/audio, pipeline, guidance_overrides, wan recipe, identity weight/start).
- Swift today: `RenderDraft.swift:98-112` restores 7 things; `OutputMetadata`
  (`GalleryPrint.swift:9-32`) decodes 18 fields. Caller `Library/LibraryPane.swift:139-147` already
  resolves the print's origin host.

### Upscale
- Still: `POST /api/gallery/upscale {filename, model, tile_size?}` → `{filename, model, scale_factor}`,
  synchronous. Clip: `POST /api/video-upscale-jobs {source:{kind:"library", filename}, model,
  tile_size?}` → `VideoUpscaleJob {id, state, completed_frames, total_frames, output_filename?,
  error?, disclosure}`; states `queued|running|finalizing|paused|completed|failed|cancelled`;
  `GET` list / `GET :id` / `POST :id/pause|resume` / `DELETE :id`.
- Gate `capabilities.video_upscale` (`types.rs:12428-12444`); stills need `gallery_image == true`.
- Reference: `studio/api/videoUpscale.ts`, `studio/lib/upscale.ts` (`defaultUpscaler`,
  `framewiseProgress`, `shouldPollFramewiseJob`), `desktop/src/views/LibraryView.vue:494-643`
  (750 ms poll `:556`, epoch fence, recovery on open).
- Swift: `canUpscaleClips` exists (`Capabilities+Reading.swift:143`); no route is called.

### Activity
`GET /api/activity` (`crates/mold-server/src/routes_activity.rs:170-396`) →
`{instance_id, observed_at_unix_ms, items[], unavailable_kinds[]}`; item `{id, kind, execution?,
phase, model?, created_at_unix_ms, updated_at_unix_ms, position?, current?, total?, stage?,
preparation_progress?, can_cancel}`. An EPHEMERAL chain reports `kind: "generation"`,
`execution: "chain"`. Reference `studio/api/activity.ts`: `reconcileActivityHost :139-184`
(replace wholesale; on error keep last + `stale`; retain items of an `unavailable_kind`),
`mergeFleetActivity :196-229` (sorted by submission time only), `activeWorkPhaseLabel :31-47`.
Poll 5 s (`desktop/src/stores/liveActivity.ts:11`).

### Events / polling today
`Support/HostStore+Events.swift`: `watch :62-88` backoff `2^n` capped at 32 s; `deliver :90-102`
emits `.resyncRequired` ONLY when the instance id changes (`:97`) — an id that survives a restart.
`Queue/QueueStore.swift:86-88 wantsPoll` has no production caller; there is no timer anywhere.

### Generate controls
| control | gate | wire | studio | Swift today |
|---|---|---|---|---|
| IP-Adapter weight | `reference_images.weight` FloatControl (`generation_profile.rs:470-484`) | `reference_weight` | `SourceImageWell.vue:751-764` | decoded + encoded (`GenerateRequest.swift:38`), NO UI |
| source_fit | none (client provenance) | `source_fit` JSON | `studio/lib/sourceFit.ts`, `sourceResolution.ts:61-70`, `sourceFitCanvas.ts` | nothing |
| scheduler | `capabilities.schedulers` (`:558-559`) | `scheduler` | `generationCapabilities.ts:407-423` | decoded (`Recipe.swift:69`), unread, no field |
| CFG++ | none; client set `{"sd3","sd3.5"}` (`generationCapabilities.ts:282`) | `cfg_plus` | `AdvancedSettings.vue:538-540` | nothing |
| Wan recipe | `capabilities.wan_recipe` (`:526-534`) | `sample_shift`, `distill_strength_high/low` | `studio/lib/wanRecipe.ts` | decoded (`Recipe.swift:35-41`), unread |
| LTX-2 guidance | pipeline | `guidance_overrides {stg_scale, stg_blocks, rescale_scale, modality_scale, skip_step}` (`types.rs:2598-2625`) | `studio/lib/guidanceOverrides.ts` | nothing |
| queue pause | `queue.can_pause` | `POST /api/queue/pause|resume` → `{paused}` | `studio/api/queuePlan.ts:413-420` | `canPauseQueue` read, NO verb |
| length ceiling | `temporal.max_duration_seconds` (`:196-203`) | `frames`,`fps` | `studio/lib/videoDuration.ts:138-167` | decoded (`Temporal.swift:71`), unread |
| auto-chain | model `max_frames`, `source_image` | `POST /api/chain-jobs` + `ephemeral: true`, events `/api/chain-jobs/{id}/events` | `studio/lib/chainRouting.ts` (`decideChainRouting :400-470`, refusal `:224-258`, 97-frame clip, ≤16 stages, tail 17), `desktop/src/stores/generation.ts:2094-2140` | nothing |

Draft persistence reference: `web/src/composables/useGenerateForm.ts` — key `mold.generate.form`,
`FORM_VERSION = 3`, `sanitizePersistedForm :221-268` strips every byte-bearing root. Desktop does
not persist the draft across launches; web does.

## 4. Swift test conventions
swift-testing only (`@Test`, `#expect`, `try #require`), `@MainActor struct XxxTests`.
`Tests/MoldTests/FakeBackend.swift`: unplanted routes THROW; `callCount("route")`; `refuses` vs
`plantedErrors`; streams held open, push with `emit`; `settle(until:)` instead of sleeps.
`FakeFixtures.swift`: every value is built by decoding JSON through `MoldJSON.decoder`; one
`capabilities(…)` overload per axis. Model test: `QueueStoreLiveTests.swift`. Red-first tests carry a
`**Fails today**:` doc comment. Package fixtures load via `RepoFixtures.swift` (`#filePath`-relative;
`repoRoot` lets a contract test read Rust/TS). Captured fixtures name host, version, date and route.
