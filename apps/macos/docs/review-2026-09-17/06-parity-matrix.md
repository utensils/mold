# 06 — Feature parity matrix: Tauri `desktop/` vs native `apps/macos`

Reviewer: top-down completeness. READ-ONLY; nothing built or run.

## Method

**Desktop inventory** read from: `desktop/src/router.ts` (destinations), `desktop/src/views/*.vue`,
`desktop/src/components/{create,generate,gallery,settings,shell,machines,models,library,jobs,mesh}/`,
`desktop/src-tauri/src/lib.rs:172-224` (the registered Tauri command set),
`desktop/src-tauri/src/menu.rs:90-229` (native menu), `desktop/docs/feature-parity.md`,
`.claude/rules/desktop.md`, and `studio/api/*.ts` + `studio/lib/*.ts`.

**macOS inventory** read from: `apps/macos/README.md`, every file under `apps/macos/Sources/Mold/**`
(288 Swift files listed), `apps/macos/Packages/MoldClient/Sources/MoldClient/**`, `project.yml`, `Makefile`.

**Endpoint extraction** — both sides by regex over source (not tests):
`grep -rhoE '"(/)?api/[A-Za-z0-9_/:{}.%$@-]*' Packages/MoldClient/Sources Sources` on macOS;
the same over `studio/ web/src desktop/src ui/` on the web side, then per-file in `studio/api/*.ts`.

Every "missing on macOS" row below names the exact search that returned nothing.

---

## 1. Generate / Create

| Capability | Desktop | macOS | Evidence (desktop → macOS) | Impact |
| --- | --- | --- | --- | --- |
| Profile-driven controls (size/steps/guidance/seed/batch) | full | **full** | `desktop/src/components/create/InspectorPanel.vue` → `Sources/Mold/Generate/ControlsRow.swift`, `RenderDraft+Recipe.swift:17-40` | — |
| Prompt + negative, `prompt.mode` gating | full | full | `AdvancedSettings.vue` → `Sources/Mold/Generate/PromptPanel.swift:70-71`, `RenderDraft+Recipe.swift:37` | — |
| Expand / Remix in place | full | full | `components/generate/ExpandControl.vue` → `Generate/PromptWand*.swift`, `GenerateController+Expand.swift` | — |
| Source image + strength + inpaint mask | full | full | `SourceImageWell.vue`, `MaskEditorModal.vue` → `Generate/SourceImageWell.swift`, `MaskEditorSheet*.swift` | — |
| Ordered reference images + `source_relation` + weight | full | full | `@studio/lib/sourceMediaPlan` → `Generate/ReferenceStrip.swift`, `GenerateRequest.swift:37-38` | — |
| Identity (PuLID) single + multi-photo | full | full | `components/create/IdentityWell.vue` → `Generate/IdentityGroup.swift`, `IdentityConditioning.swift` | — |
| ControlNet picker | full | full | `AdvancedSettings.vue` controlnet-select → `Generate/RefineGroup+ControlNet.swift` | — |
| LoRA stack | full | full | `components/generate/LoraStack.vue` → `Generate/AdaptersGroup.swift`, `LoraStore.swift` | — |
| Video: frames/fps/audio/extend/keyframes/pipeline | full | full | `AdvancedSettings.vue` video section → `Generate/ClipGroup*.swift`, `ExtendRow.swift`, `KeyframeTable.swift` | — |
| **Finished video result on the canvas** | full | **BROKEN** | `desktop/src/views/GenerateView.vue` video branch → `Sources/Mold/Generate/RunCanvas+Result.swift:49` decodes every result with `NSImage(data:)`; searched `AVPlayer|VideoPlayer|isVideo|mp4` across `Sources/Mold/Generate/` → **zero hits** | **HIGH** — see §5.1 |
| **Scheduler picker** (`scheduler`) | full | **missing** | `desktop/src/components/create/AdvancedSettings.vue:517-528,577-583` → macOS decodes `Recipe.capabilities.schedulers` (`Recipe.swift:67-69`) but `grep -rn "\.schedulers" Sources Packages/*/Sources` returns hits only in `RecipeCapabilityTests.swift`; field absent from `GenerateRequest.swift:125-133` | MED — SD1.5/SDXL/wan lose solver choice |
| **CFG++** (`cfg_plus`) | full | **missing** | `AdvancedSettings.vue:538-540` → `grep -rn "cfg_plus\|cfgPlus" apps/macos` → 0 | MED |
| **Wan sampler recipe** (`sample_shift`, `distill_strength_high/low`) | full | **missing** | `studio/lib/wanRecipe.ts` + `AdvancedSettings.vue:548-583` → `grep -rn "sample_shift\|distill_strength"` → 0 | MED — wan's "primary character knob" |
| **LTX-2 guidance overrides** (STG/rescale/modality/skip) | full | **missing** | `desktop/docs/feature-parity.md:63,120` → `grep -rn "guidance_overrides"` → 0 | MED |
| **LTX-2.3 spatial/temporal upscale, retake range, IC-LoRA, camera motion** | full | **missing** | `studio/lib/ltx2Control.ts`, `studio/lib/cameraMotion.ts` → `grep -rn "spatial_upscale\|temporal_upscale\|retake\|ic_lora\|camera"` → 0 in request path | MED |
| **PuLID true-CFG** (`true_cfg`, `cfg_start_step`) | full | **missing** | `feature-parity.md:80-81` → `grep -rn "true_cfg"` → 0 | LOW-MED (identity-only) |
| **Source fit policy** (`source_fit` + client-side fit/crop/pad-repaint/pre-upscale) | full | **missing** | `studio/lib/sourceFit.ts`, `desktop/src/lib/sourceFitPreprocess.ts` → `grep -rn "source_fit\|sourceFit"` → 0 | MED — a mismatched source is sent raw; no pad-repaint mask, no crop control, no provenance for Reuse |
| **Device placement** (`placement`) | full | **missing** | `desktop/src/components/settings/PlacementSection.vue` → macOS reads `PlacementPreview` (`Placement.swift`) for the ETA hint only; no `placement` field in `GenerateRequest.swift` | LOW-MED (multi-GPU hosts) |
| `embed_metadata`, `gif_preview` | full | **missing** | `feature-parity.md:67,100` → grep → 0 | LOW |
| **Mesh controls** (octree / iso threshold / target faces / matting / delight / PBR) | full | **missing** | `AdvancedSettings.mesh.test.ts`, `InspectorPanel.mesh.test.ts`, `SourceImageWell.mesh.test.ts` → `grep -rni "mesh" apps/macos/Sources` finds only Library/export uses; no `mesh` block in `GenerateRequest.swift` | **HIGH** in combination — see §5.2 |
| **Starters / templates panel** | full | **missing** | `components/generate/TemplatesPanel.vue`, `StarterCards.vue`, `components/create/StarterList.vue` → `grep -rn "emplate\|tarter" Sources` → 2 unrelated comments | LOW |
| Recent prints tab | full | full | `components/create/RecentPrints.vue` → `Generate/RecentGroup.swift` | — |
| Prompt history | full | full | `/api/history` both | — |
| Batch N as N one-output children | full | full | `studio/api/generationAdmission.ts` → `RenderDraft+Request.swift:20-25` | — |
| Estimate / placement preview | full | full | `/api/generate/placement-preview` both | — |
| Machine pick (Auto / pinned / "Most capable") | full | partial | `desktop/src/components/create/HostChip.vue` (adds the `"capable"` sentinel) → `Generate/MachineControl.swift` (Auto + pin only; no capability-ranked target) | LOW |

## 2. Library

| Capability | Desktop | macOS | Evidence | Impact |
| --- | --- | --- | --- | --- |
| Merged multi-host grid, host badges, dedupe | full | full | `desktop/src/views/LibraryView.vue` → `Library/LibraryGrid.swift`, `LibraryStore*.swift` | — |
| Favourite / tag / title / collections / trash / restore / purge | full | full | `studio/api/galleryOrganization.ts` → `HTTPBackend+Gallery.swift:5-43`, `+Organize.swift` | — |
| Undo for organization | partial | **full (better)** | desktop has no undo stack; `Support/MoldUndo.swift` + `LibraryStore+Editing.swift` | macOS wins |
| Search tokens, sort, thumbnail size | full | full | → `LibraryQuery.swift`, `LibraryToken.swift` | — |
| Quick Look / drag to Finder / share / save | n/a (web) | full | → `Library/QuickLook.swift`, `DraggablePrint.swift`, `PrintMaterializer.swift` | macOS wins |
| Live SSE reflection of remote edits | full | full | → `LibraryStore+Live.swift` | — |
| **Opening a GLB print** | full (`MeshViewer`) | **BROKEN** | `desktop/src/views/GenerateView.vue:76,4448` imports `@studio/components/MeshViewer.vue`; Lightbox uses it too → `Sources/Mold/Library/LibraryViewer.swift:33-44,131-137`: not `isVideo`, so it takes the image branch, `NSImage(data:)` on GLB bytes returns nil, `full` stays nil and the 512 px poster renders **at `opacity 0.55` forever** | **HIGH** — see §5.2 |
| **Mesh export geometry options** (`size_mm`/`up_axis`/`origin`) | full | **missing** | `desktop/src/lib/mediaSave.ts:10-21` + `studio/lib/meshExport.ts:229` → `Packages/MoldClient/Sources/MoldClient/HTTPBackend+Gallery.swift:54-61` posts `{"format": format}` only | MED — every STL/PLY exports in Hunyuan3D unit-cube space; a slicer reads a few-millimetre object |
| **Video/turntable export options** (`playback`, `repeat`, `max_dimension`, `frames`, `fps`, `transparent`) | full | **missing** | `studio/lib/videoExport.ts:5-18` → same `HTTPBackend+Gallery.swift:54-61` | MED — server defaults only |
| **Image upscale from the Library** (`POST /api/gallery/upscale`) | full | **missing** | `studio/api/videoUpscale.ts:55`, `desktop/src/components/gallery/Lightbox.vue:1009` ("Make bigger…") → `grep -rn "gallery/upscale"` in macOS → 0 | MED |
| **Framewise video upscale** (`/api/video-upscale-jobs`, pause/resume/cancel/restart) | full | **missing** | `studio/api/videoUpscale.ts:26-110`, `LibraryView.vue:546,616` → macOS decodes `VideoUpscaleCapabilities` (`CapabilityBlocks.swift:70`) and exposes `canUpscaleClips` (`Capabilities+Reading.swift:143`) — `grep -rn "canUpscaleClips" Sources` → **0 call sites**. Dead code. | MED |
| **Retained source-media probe** (`GET /api/gallery/source-media/:filename`) | full | **missing** | `studio/api/gallerySourceMedia.ts` → `grep -rn "source-media\|sourceMedia" apps/macos` → 0 | MED, see next row |
| **"Use these settings" fidelity** | whole recipe + retained source bytes | **7 fields** | `desktop/src/views/GenerateView.vue` `useReuseStillPrint` + `reuseSettings` → `Sources/Mold/Library/LibraryPane.swift:140` → `RenderDraft.swift:98-112` restores only prompt, negative, w/h, steps, guidance, frames, fps, seed | **HIGH-ish MED** — LoRAs, strength, source image, mask, identity photo, ControlNet, tags/collection/title, format and upscaler are all silently dropped; the menu item says "Use These Settings" |
| "Make 4 variations" / repeat-print | full | **missing** | `desktop/src/lib/variations.ts`, ⌥↩ → no equivalent (`grep -rn "variation"` → 0) | LOW |
| Import into the gallery | **missing** | **full** | `PUT /api/gallery/import/:filename` exists in `crates/mold-server` and only macOS calls it (`Sources/Mold/Library/LibraryActions+Import.swift`); desktop's `import_source_image` (`lib/ipc.ts:399`) only stages a composer source | macOS wins |

## 3. Queue / Activity

| Capability | Desktop | macOS | Evidence | Impact |
| --- | --- | --- | --- | --- |
| Live queue per machine, batch grouping | full | full | `desktop/src/views/QueueView.vue` → `Queue/QueueStore+Live.swift`, `QueueBatchRow.swift` | — |
| Reorder, cancel, empty | full | full | → `QueuePane+Reorder.swift`, `HTTPBackend+Queue.swift:22-31` | — |
| Per-job pause / resume / retry | full | full | → `HTTPBackend+Work.swift:13-25` | — |
| **Whole-queue pause / resume** (`POST /api/queue/pause`, `/resume`) | full (Space is the shell chord) | **missing** | `studio/api/queuePlan.ts` `/api/queue/pause`, `/api/queue/resume`; `.claude/rules/desktop.md` shell Space → macOS routes end at `/api/queue/{id}/pause` (`HTTPBackend+Work.swift:14`); `grep -rn '"/api/queue/pause"'` → 0 | MED — cannot stop a machine from taking more work without emptying the queue |
| Held rows with typed cause + Move to… transfer | full | full | `studio/api/queueTransfer.ts` → `Queue/QueueHoldRow.swift`, `TransferStore+Steps.swift` | — |
| Denoise preview on a foreign job (`/api/queue/{id}/preview`) | full | full | `studio/api/ownPrintPreview.ts` → `HTTPBackend+Generation.swift:18-21` | — |
| Cross-client activity feed (`/api/activity`) | full | equivalent via `/api/events` | `studio/api/activity.ts` → macOS has no `/api/activity`; `HostStore+Events.swift` + `QueueStore+Live.swift` reconstruct from SSE | LOW |
| Dock badge for landed prints; completion/failure notifications | full | full | `desktop/src-tauri/src/lib.rs:185-187` → `MoldApp.swift:84-89`, `Support/MoldNotifications.swift` | — |
| Stop-everything confirm | full | partial | `useQueueCommands.askStopEverything` → `QueueCommands.swift:40` "Empty Queue…" only (running work keeps going, by design) | LOW |

## 4. Models, Machines, Settings, Shell

### Models
Parity is good. Installed table, Discover/catalog search, install/repair/cancel/load/unload/delete/components,
downloads popover, licence acceptance, catalog credentials all present
(`Sources/Mold/Models/*`, `HTTPBackend+Catalog.swift`, `+Licenses.swift`, `+Models.swift`).
Gaps: no catalog **detail drawer** equivalent depth is present (`CatalogDetailSheet.swift` exists — parity OK);
no `models_disk` per-family disk meter (desktop `StylesDiskSection.vue`) — macOS shows one footer figure
(`Models/ModelsFooter.swift:19`). LOW.

### Machines
| Capability | Desktop | macOS | Evidence | Impact |
| --- | --- | --- | --- | --- |
| Per-machine GPU cards + lifecycle switch, live RAM/CPU | full | full | `studio/components/DevicePanel.vue` → `Machines/DeviceRow.swift`, `DeviceControl.swift` | — |
| LAN discovery + add | full | full | `/api/discovery/peers` both | — |
| Paired phones / issue pairing | full | full | `studio/components/PairingAccessPanel.vue` → `Machines/PairingSection.swift`, `PairingSheet.swift` | — |
| **Per-host Storage card** (`gallery_storage` figures, that host's `gallery.trash_retention_days`, Empty trash) | full | **missing** | `.claude/rules/desktop.md` (Machines ▸ host ▸ Storage) → `grep -rn "gallery_storage\|galleryStorage" apps/macos` → **0**; the pane's sections are `MachinesPane+Sections.swift:24-113` (identity, memory, work, address, pairing) | MED — a remote's retention and trash size are unreachable |
| **Rent a GPU / RunPod** pod + network-volume provisioning | full | **missing** | `desktop/src/views/RunPodView.vue` + `desktop/src-tauri/src/runpod.rs` + `lib.rs:216-223` → `grep -rni "runpod\|lambda" apps/macos/Sources` → 1 hit, a comment | MED (deliberate?) — README does not list it as an omission |
| Rename / Forget / Open web UI from the pane | full | in Settings ▸ Machines | `HostsSection.vue` → `Shell/MachinesSettings.swift`, `HostEditor.swift` | LOW |

### Settings
Desktop sections (`desktop/src/views/SettingsView.vue:17-36`): Appearance, Updates, About, Hosts, Performance,
Generation, Media, Styles&Disk, Library, Expansion, Accounts, **Cloud**, **Per-style defaults**, **Profiles**,
Advanced, Pairing access, Licences.
macOS tabs (`Sources/Mold/Shell/SettingsView.swift` + `Settings/*`): General, Generation, Expansion, Library,
Performance, Accounts, Machines, This Mac, Advanced.

| Capability | Desktop | macOS | Evidence | Impact |
| --- | --- | --- | --- | --- |
| Curated engine keys pinned to `config_keys.rs` | full | full (+ a test that parses the Rust) | → `SettingKeys*.swift`, `SettingKeysContractTests.swift` | macOS wins |
| Raw `/api/config` escape hatch with source badges | full | full | `AdvancedSection.vue` → `Settings/AdvancedSettings*.swift` | — |
| **Profile switching** (`MOLD_PROFILE`, `POST /api/config/profile`) | full | **read-only** | `desktop/src/components/settings/ProfilesSection.vue` → `Settings/ProfileHeader.swift:4-11` explicitly states switching "is not a control"; macOS calls `/api/config/profiles` (list) only | MED — multi-profile users cannot switch |
| **Per-model defaults UI** (`models.<name>.<field>`) | curated rows | Advanced only | `PerStyleDefaultsSection.vue` → README line 18 ("nothing curated repeats them") | LOW (stated) |
| **Mold home relocation** | full (typed path + native picker + atomic migration + relaunch) | **read-only** | `desktop/src/components/settings/MoldHomeCard.vue` + `desktop/src-tauri/src/{mold_home,relocate}.rs` + `lib.rs:176-177` → `Packages/.../MoldHome.swift` resolves and *reports*; `grep -rn "change_mold_home\|relocate\|pickDirectory"` in macOS → 0 | MED |
| **App updater** (Stable/Nightly channels, signed, banner + notification) | full | **missing entirely** | `desktop/src/components/settings/UpdatesSection.vue`, `components/shell/UpdateBanner.vue`, `desktop-src-tauri/src/updater.rs`, `lib.rs:190-191`, `menu.rs:97` → `grep -rn "Sparkle\|updateChannel\|checkForUpdates\|updater" apps/macos/Sources project.yml` → **0** | **HIGH for replacement** — a shipped native app with no update path |
| About / version / licences panel | full | **missing** | `AboutSection.vue`, `LicenseSettingsPanel.vue` → macOS Help menu is one item, `Shell/MoldCommands.swift:93-95` ("Mold on the Web") | LOW |
| **Open Logs** | full | **missing** | `menu.rs:229` `help:logs`, `lib.rs:189` `open_logs_dir` → macOS computes `MoldEngine.logDirectory` (`Engine/MoldEngine.swift:61`) but exposes no way to open it | LOW-MED |
| Media cache cap + Empty Now | full | full | `MediaSection.vue` → `Shell/GeneralSettings.swift:58`, `CacheBudget.swift` | — |
| API keys storage | file-backed `secrets.json`, deliberately **not** Keychain | **Keychain** | `.claude/rules/desktop.md` ("deliberately NOT the macOS Keychain, whose prompts users found obnoxious; don't reintroduce `keyring`") → `Sources/Mold/Support/Keychain.swift`, README line 18 | see §4.3 |

### Shell / system integration
| Capability | Desktop | macOS | Evidence | Impact |
| --- | --- | --- | --- | --- |
| Native menu bar | full | full (richer: Library/Queue/Model/Machine menus) | `menu.rs` → `Shell/{MoldCommands,LibraryCommands,QueueCommands,ModelCommands,MachineCommands}.swift` | macOS wins |
| **Command palette ⌘K** | full | **missing** | `desktop/src/components/shell/CommandPalette.vue` → `grep -rn "alette" apps/macos/Sources` → 1 hit, a comment in `Shell/Appearance.swift:12` | LOW-MED |
| Themes (5 families × 2 tones) | full | **system light/dark only** (deliberate, README "no literal colors" lint) | `ui/tokens.css` → `Shell/Appearance.swift` | stated |
| Deep links `mold://` | **not registered** | **not registered** | `grep -rn "mold://" desktop/src-tauri` → 0; only `apps/mobile/src-tauri/tauri.conf.json:25` registers the scheme. macOS `project.yml` has no `CFBundleURLTypes`. | no gap, no collision |
| Tray icon | none | none | — | — |
| Drag & drop onto wells | full | full | `@studio/lib/imageDropRouting` → `Generate/MediaWell.swift`, `PictureSource.swift` | — |
| Local engine lifecycle (start/stop/reuse) | full (start/stop/ensure, reuses a running `mold serve` on :7680) | partial (**start once per process**, ephemeral loopback port, no key) | `desktop/src-tauri/src/lib.rs:178-180`, `server.rs` → `Sources/Mold/Engine/MoldEngine.swift:75-98`, `rust/mold-macos-ffi/src/lib.rs:106-131`; README lines 58-61 | see §4 |
| Onboarding / first run | dialog-driven connect flow | none found | `.claude/rules/desktop.md` ("Connect a machine is ONE dialog") → `grep -rn "onboard\|Onboarding\|Welcome\|firstLaunch" apps/macos/Sources` → 1 hit, a comment in `AppStorageSuite.swift:7` | LOW-MED — a first launch with no machines lands on an empty pane |
| Backup / export of settings | neither | neither | — | — |
| MCP / skill install | neither (CLI only) | neither | `desktop/docs/feature-parity.md:416` describes `mold mcp` as CLI | — |
| 3-D Studio (`/create/3d`, `/api/mesh-workflows`) | full | **absent (stated)** | `desktop/src/views/MeshWorkflowView.vue`, `studio/api/meshWorkflows.ts` → README line 72 | see §6 |
| Chain jobs | full | **absent (stated)** | `studio/lib/chainJobEvents.ts` → README line 72 | stated |
| Reference upload sessions (H3) | full | **absent (stated)** | `studio/api/referenceUploads.ts` → README line 73-75 | stated |

---

## 5. The three findings that are bugs, not gaps

### 5.1 A finished clip never appears on the Generate canvas (HIGH)

`Sources/Mold/Generate/RunCanvas.swift:29-30` routes a finished batch to `finishedView`, and
`Sources/Mold/Generate/RunCanvas+Result.swift:9` opens with `if let result` where `result: NSImage?`
is produced by `loadResult()` at line 45-50:

```swift
guard let data = try? await hosts.backend(for: host).media(filename, trashed: false) else { return }
result = NSImage(data: data)
```

There is no video branch anywhere in the pane — `grep -rn "AVPlayer|VideoPlayer|isVideo|mp4" Sources/Mold/Generate/`
returns **zero** hits (the only `VideoPlayer` in the app is `Library/LibraryViewer.swift:60`). `NSImage(data:)`
on MP4 bytes is nil, so `finishedView` falls to its `else` and the pane shows
`ProgressView("Fetching your picture…")` **permanently**. The whole result chrome — `ResultStrip`,
`ResultBar` (Save / Copy / Show in Library) and even `outcome.failureSummary` — is nested inside that
`if let`, so none of it renders either. Worse, the app first downloads the entire clip over HTTP
(`HTTPBackend+Gallery.swift:64-78`, 300 s timeout) on every `resultFilename` change and discards it.

Given the README's headline is "Stills and **clips**", and every clip control (length, fps, audio, extend,
keyframes, LTX-2 pipelines) is implemented, this is the largest single defect found.

**Fix:** branch on `entry.print.isVideo`/the result's container in `RunCanvas+Result.swift` and reuse
`LibraryViewer`'s ticketed `AVPlayer` path; hoist `ResultBar` and `failureSummary` out of the `if let result`.

### 5.2 A mesh print renders as a permanently half-transparent thumbnail (HIGH, and it is *caused* by the 3-D omission)

`Sources/Mold/Library/LibraryViewer.swift:33-44`: the only two branches are `isVideo` and "image". A GLB
takes the image branch, so `placeholder` gets the server's 512 px poster, `actions.data(for:)` returns GLB
bytes, `NSImage(data:)` returns nil, and line 42 keeps `.opacity(0.55)` because `full == nil`. The print
opens as a dim, low-resolution, half-faded square with no explanation, forever.

Quick Look is no better: `Library/PrintMaterializer.swift` writes the real `.glb` and macOS ships no
GLB previewer, so Space on a mesh opens a blank panel.

And `Packages/MoldClient/Sources/MoldClient/Model.swift:52-56` lists `hunyuan3d-paint` in
`auxiliaryFamilies` but **not `hunyuan3d`** — so a `hunyuan3d` checkpoint is `isGenerator` and *is offered*
in the macOS style picker (`Generate/ModelStore.swift:73-80`). Selecting it produces a request that is
structurally valid (`RenderDraft+Recipe.swift:19-20` takes the recipe's 0×0 defaults, line 37 clears the
prompt, `ShapeControl+Resolve.swift:33` hides the shape control) but carries **no `mesh` block**, so the
server silently uses `MESH_DEFAULT_*` — and when it finishes, §5.1's `NSImage(data:)` on the GLB gives a
blank canvas. The user spends several GPU minutes and sees nothing.

**Fix (minimum):** either add `"hunyuan3d"` to a "not offered here" set with an honest reason in the picker,
or render GLB. Do not leave it half-reachable.

### 5.3 "Use These Settings" restores 7 of ~30 fields (MED, arguably HIGH for data loss of intent)

`Sources/Mold/Library/LibraryPane.swift:140` → `RenderDraft.swift:98-112`. Restored: prompt (first stage),
negative, width/height, steps, guidance, frames, fps, seed. **Silently dropped:** `loras`, `strength`,
`source_image` (and the retained-source probe is not implemented at all —
`grep -rn "source-media|sourceMedia"` → 0), `mask_image`, identity photo(s) and weights, ControlNet
image/model/scale, `title`/`tags`/`collection`, `output_format`, `upscale_model`, `pipeline`,
`enable_audio`/`video_only`, `extend_*`, `keyframes`. Desktop restores the whole recipe including retained
authority bytes (`studio/api/gallerySourceMedia.ts`; CLAUDE.md's "Durable gallery source media" contract
says *every client always asks*). A user who reuses an img2img or LoRA print gets a text-to-image render at
the same size and calls it a regression in the model.

---

## 6. Endpoint families: who calls what

**Studio calls, MoldClient never does** (search: the macOS path regex above, full result set is 51 paths):

| Family | Studio site | macOS status |
| --- | --- | --- |
| `/api/activity` | `studio/api/activity.ts` | reconstructed from `/api/events` — acceptable |
| `/api/chain-jobs*`, `/api/capabilities/chain-limits` | `studio/lib/chainJobEvents.ts` | stated omission |
| `/api/mesh-workflows*` | `studio/api/meshWorkflows.ts` | stated omission |
| `/api/generate/reference-upload*` | `studio/api/referenceUploads.ts` | stated omission |
| **`/api/gallery/upscale`, `/api/video-upscale-jobs`, `/api/upscale/stream`** | `studio/api/videoUpscale.ts:55,66-110` | **missing** (capability decoded, never used) |
| **`/api/gallery/source-media/:filename`** | `studio/api/gallerySourceMedia.ts` | **missing** |
| **`/api/queue/pause`, `/api/queue/resume`** | `studio/api/queuePlan.ts` | **missing** |
| **`/api/config/profile` (setter)** | `studio/api/config.ts` | **missing** (list only) |
| `/api/gallery/organize`, `/api/gallery/trash/sweep` | `studio/api/galleryOrganization.ts` | covered by `/api/gallery/mutations` + `DELETE /api/gallery/trash` |
| `/api/generate/estimate`, `/api/generate/stream` | web/desktop | superseded by durable batches + placement-preview — fine |
| `/api/pairing/claim` | `studio/api/pairing.ts` | deliberate (never a claimant) |
| `/api/gallery/assets/:id` | `studio/api/generationAssets.ts` | not needed without H3 |

**MoldClient calls, studio never does:**

| Family | macOS site | Note |
| --- | --- | --- |
| `PUT /api/gallery/import/:filename` | `Library/LibraryActions+Import.swift` | **macOS-only**; the route exists in `crates/mold-server` (`routes_test.rs:16702+`) and no web surface uses it. A real capability the Tauri app lacks. |
| `/api/gallery/tags/:name` PATCH/DELETE | `HTTPBackend+Organize.swift:36-42` | studio has it too (`galleryOrganization.ts`) — parity, not macOS-only |

---

## 7. Coexistence on one Mac

1. **Bundle identifiers do not collide** — `apps/macos/project.yml:59-61` explicitly sets
   `io.utensils.mold.native` with a comment saying the Tauri app owns `com.utensils.mold`
   (`desktop/src-tauri/tauri.conf.json:4`). Good. Launch Services will keep them distinct.
2. **No URL-scheme collision** — neither desktop nor macOS registers `mold://`; only
   `apps/mobile/src-tauri/tauri.conf.json:25` does.
3. **Ports do not collide** — the macOS engine binds an ephemeral loopback port
   (`rust/mold-macos-ffi/src/lib.rs:106-109`, `Engine/MoldEngine.swift:75-85`); the Tauri app reuses or
   binds :7680. But that means *two engines on one GPU*: both are `mold_server::run_server` on Metal and
   both consult `MetalMemorySnapshot` independently, so each can believe it has headroom the other has
   already taken. Worth a line in the README at minimum.
4. **Same `MOLD_HOME` by construction** — `MoldHome.resolve` (`MoldHome.swift:37-47`) deliberately mirrors
   `Config::mold_dir`, including the `~/Library/Application Support/mold/home` pointer the Tauri app writes.
   Two writers on one gallery is a supported shape (shared writer lease), but it is worth stating that the
   **macOS app cannot move the home and the Tauri app can** — a relocation done in the Tauri app silently
   moves the native app too, which is correct but surprising.
5. **The macOS engine runs with no API key** — `mold_engine_start(bind, port, nil)`
   (`Engine/MoldEngine.swift:80-81`). On a keyless host `AuthState = None` means "open by policy", so any
   local process on the Mac can read the whole gallery, delete prints and queue renders on the loopback
   port. The Tauri app deliberately uses a persistent `desktop-local-api-key`. Loopback-only limits the
   blast radius, but this is a downgrade from the app it would replace. (security, LOW-MED)
6. **Credentials live in two places that never agree** — the Tauri app stores host API keys in an
   owner-only `secrets.json` (`.claude/rules/desktop.md` states the Keychain was *deliberately rejected*);
   the macOS app uses the Keychain (`Support/Keychain.swift`, README line 18). Neither reads the other, so
   a user running both re-enters every key, and the explicit no-Keychain decision is reversed without the
   README acknowledging it. Flag as a design decision to reconcile, not a bug.
7. **Preferences domains are independent** (`io.utensils.mold.native` UserDefaults vs the Tauri
   `settings.json`), so machine lists, default machine, thumbnail size and appearance all diverge.
   `MOLD_NATIVE_HOSTS` exists for dev seeding but there is no import of the Tauri host list.

---

## 8. Top 15 gaps blocking replacement (README omissions excluded)

1. **Finished clips never render on the Generate canvas** (§5.1) — HIGH, bug.
2. **No app updater at all** — HIGH. Every shipped Tauri build can self-update on two signed channels
   (`desktop-src-tauri/src/updater.rs`, `UpdatesSection.vue`, `UpdateBanner.vue`); `grep` for
   `Sparkle|updater|checkForUpdates` in `apps/macos` returns nothing. A replacement cannot ship without this.
3. **GLB prints open as a dimmed poster; Hunyuan3D is still offered in the picker** (§5.2) — HIGH.
4. **"Use These Settings" drops LoRAs, source image, mask, identity, ControlNet, filing and format** (§5.3) — HIGH-MED.
5. **No retained source-media probe** — MED-HIGH; the CLAUDE.md contract says every client always asks.
6. **No Library upscale (image) and no Framewise video upscale** — MED; capability decoded, zero call sites.
7. **No mold-home relocation** — MED; a user on an external drive cannot set it up from this app.
8. **No whole-queue pause/resume** — MED.
9. **Mesh and video exports post `{format}` only** — MED; STL/PLY come out at unit-cube scale.
10. **No `source_fit`** — MED; a mismatched source is shipped raw with no pad-repaint, crop control or provenance.
11. **No scheduler / CFG++ / wan sampler recipe / LTX-2 guidance overrides** — MED; four families lose their
    primary character knobs.
12. **No profile switching** — MED; `MOLD_PROFILE` users are read-only.
13. **No per-host Storage card** (retention + trash size + `gallery_storage`) — MED.
14. **No RunPod / cloud-pod provisioning** — MED, and *not* listed among the README's omissions, so it reads
    as an oversight rather than a decision. Decide and write it down.
15. **No command palette, no About/licences panel, no Open Logs, no first-run onboarding** — LOW-MED each,
    but together they are the "this feels unfinished" cluster.

## 9. Opinion: is omitting the 3-D studio tenable?

Omitting the **durable 3-D studio** — `/create/3d`, `/api/mesh-workflows`, the workflow host picker, the
stage graph, delete-workflow-data — is entirely tenable. It is a self-contained destination with its own
routes, its own host binding and its own provenance rules; nothing else in the app depends on it, and a user
who needs it has the web UI on the same host. What is **not** tenable is the current half-state, and that is
a different question from the studio. GLB prints already exist in the Library and in the queue: `LibraryToken`
has a `.mesh` case, `GalleryPrint.isMesh` keys off `format == "glb"`, `LibraryActions+Export.exportFormats`
branches to `options.forMesh`, and `hunyuan3d` is *not* in `Model.auxiliaryFamilies`, so the style picker
offers it and a render is submitted. So the app already claims mesh territory in three places while having
no way to draw a triangle — opening one gives a silent half-transparent thumbnail, Quick Look gives a blank
panel, and generating one gives a blank canvas after several GPU minutes. The honest resolutions are both
small: either port `studio/components/MeshViewer.vue` (it is raw WebGL over GLSL ES 1.00 — a SceneKit or
RealityKit viewer for a normalized unit-cube GLB is a day's work, and it also fixes the Library, Quick Look
and the Generate canvas at once), or close the door properly — hide `hunyuan3d` from the picker with a stated
reason, and give a mesh print in the Library a first-class "3-D object · Export…" card instead of pretending
it is a picture. Shipping the studio itself can wait indefinitely; shipping a Library that shows a mesh
badly cannot.

## 10. Five things done notably well

1. **`generation_profile` is genuinely the only authority.** `RenderDraft+Recipe.adopting` reconciles every
   control against the recipe and *parks* rather than discards conditioning the new recipe cannot read — the
   README's claim holds up in the code.
2. **The wire notes in the README are real, hard-won and encoded as tests** — `LineAccumulator` for the SSE
   empty-line bug, `DeviceControl.resolve`'s two-flag rule, `batch_size` always 1, the `accepted`-vs-`queued`
   state list. Each has a test beside it.
3. **`SettingKeysContractTests` parses `config_keys.rs`** so a bound that moves in the engine fails the Swift
   build. That is a stronger contract than the desktop's equivalent.
4. **Undo for organization** (`Support/MoldUndo.swift` + `LibraryStore+Editing.swift`) — favourite, tag, filing
   and tag-rename are all undoable from the Edit menu. The Tauri app has none of this.
5. **The lint set** (`MoldClient` cannot import UI, one file may construct a backend, 150-line file cap,
   600-line *type* cap, no `bytes.lines`, a11y floor) is the right response to `GenerateView.vue` reaching 4,885
   lines, and it is visibly working.
