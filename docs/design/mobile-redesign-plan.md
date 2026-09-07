# Mobile redesign plan — iOS first, shared with Android

Tracking: [#1628](https://github.com/utensils/mold/issues/1628).

Status: locked after independent peer review; implementation underway on `feat/mobile-studio-redesign`.
Baseline: `c38d569d5d2ef14ea60adb05cc81a2e95f126c13`, verified against remote main on 2026-09-06.

## Outcome

Make the phone feel like a simple creative tool: choose what to make, describe it or attach a photo, pick a style, and Generate. Keep the result prominent and make progress easy to find. Advanced controls remain available without turning the main screen into a form to complete.

This is the mobile redesign for both iOS and Android. Implement and accept the iPhone experience first. Share Vue screens, design tokens, state, and domain behavior with Android; retain platform adapters for navigation, permissions, media, secure credentials, and system chrome. Keep the existing remote-only Tauri architecture and iOS 17 minimum. iPad is a responsive secondary target.

The user explicitly directs best-judgment UX decisions where the mocks have gaps, with the completed desktop as the reference so moving between apps feels natural. Preserve the same vocabulary, output kinds, control meaning, defaults, style identity, result actions, and state explanations. Adapt placement and interaction for touch; do not introduce a competing mobile product model. The user also explicitly confirms that the scene editor was intentionally removed: scene sections, authored timelines, transitions, and scene editing are excluded from implementation.

## Evidence and precedence

The user's request is research and a plan, extended explicitly to Android with iOS first. The attached archive is design reference material, including historical proposals, rather than authorization to implement every feature it depicts.

Decision order: current user direction, current desktop behavior and shared capability contracts, then the archive's visual guidance. Where these leave a gap, choose the simplest usable phone interaction and record the reason; routine layout decisions do not need repeated user clarification.

Reviewed the archive's desktop, iPhone, and style-guide sources, and rendered the iPhone storyboard in a browser. Compared these against `docs/design/README.md`, current desktop/mobile source, recent commits, and live GitHub issue/PR state. The architecture graph at `.ua/knowledge-graph.json` predates the redesign (2026-08-28); it was useful for orientation only. Current source wins over its descriptions.

| Recent work                                      | Implication for mobile                                                                                                                      |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| #1565 and #1566: six shared themes               | Reuse the shipped theme contract; mobile already initializes it before mount.                                                               |
| #1595: desktop redesign                          | Adopt its visual language, plain wording, style picker, and result-first hierarchy.                                                         |
| #1598: output sections and remembered styles     | Use Still picture / Short clip / 3-D object and the shared last-used-style store.                                                           |
| #1600 and #1602: duration and backend follow-ups | Old claims that duration, storage totals, and save-to-gallery opt-out lack server support are stale. Gate presentation on actual host data. |
| #1619: retire interactive scene authoring        | Discard the archive's scenes screen. A short clip has one prompt and Length; internal splitting remains invisible.                          |
| #1623: long-clip recovery fixes                  | Preserve durable admission, progress, restart, and result recovery when changing the screen.                                                |
| #1620 and #1624: Hunyuan3D expansion             | Include current capability-driven mesh, texture, and named-view inputs, beyond the older storyboard.                                        |

Issues #1586 and #1597 are closed. Their checkboxes are not proof that phone adoption shipped: current mobile still renders Create/Library/Models, `meta.tone`, legacy CSS, and has no last-used-style-store consumer. Track these as mobile migration work instead of copying the closed issue's completion state.

Research boundary: this is a source/history and reference-design audit, not a simulator or physical-device UAT report. No inference, installation, or release was performed. Native runtime acceptance is an implementation gate below.

## Proposed experience

### Navigation

Use five stable tabs in this order: **Make, Queue, Images, Styles, Machines**. These short labels are an intentional mobile-only space adaptation of the desktop vocabulary, pinned by a lexicon test; screen headings use **New image / New clip / New 3-D object** and **My images**. Settings opens from a consistently positioned gear. Each destination retains its navigation, filter, and scroll state. An empty or disconnected state remains an explanation inside the destination, not a disappearing tab.

Queue deserves a destination because remote work persists when the user leaves Make. A compact progress card on Make opens the matching Queue item; it does not duplicate the entire queue. Preserve the current draft while inspecting another job or image.

Apple's [tab-bar guidance](https://developer.apple.com/design/human-interface-guidelines/tab-bars?changes=_5) supports stable destinations and preserving section state. Use sheets for focused temporary tasks, with a visible exit and one active sheet rather than stacked picker sheets, following [Apple's sheet guidance](https://developer.apple.com/design/human-interface-guidelines/sheets?changes=_3_3).

### Make

Read top to bottom: title and Settings; machine-routing chip; output-kind selector; result or compact empty guidance; prompt and required attachments; Style, Shape/Length, and Make count; More settings summary; compact progress; pinned Generate action above navigation.

- **One style picker.** Friendly description first, exact model ID second. Filter it by output kind and use the existing fleet availability/routing policy. Browse more opens Styles filtered to the same kind. Returning to each kind restores its remembered style through `studio/stores/lastUsedStyles.ts` without forgetting unavailable choices.
- **One main action.** Generate stays reachable; submission errors explain the fix immediately above it and beside the offending input. Preserve the existing cancellation of in-progress submission and prepared-batch review behavior.
- **Progressive disclosure.** Move Detail, guidance, seed, and secondary controls off the primary stack. Keep the current size or length and batch count visible. Avoid a large blank result area before the first print. During text entry, reduce the preview's space so the keyboard does not obscure the active editor or action.
- **Attachments are primary content.** Keep required target/reference images, first/last frames, identity photos, and named mesh views visible in the composer when relevant. An optional attachment can begin as an Add photo row, but attached media must remain visible. Do not bury these inputs in More settings.
- **A short clip has one flow.** Prompt, clip style, Length, optional supported sound/source controls. Honor the shared family frame grid and long-clip refusal rules. No scenes, timeline, or Simple/Scenes strip.
- **3-D gets a real entry point.** Prompt-ignored recipes show photo guidance instead of a misleading prompt requirement or expansion action. Use advertised named views, texture controls, and mesh defaults. Do not describe iso threshold as an image-adherence control; verify and reuse the current technical meaning.
- **Organization is optional.** Put Title and File under in a collapsed Name and organize group. A nonempty title, selected album, tags, or automatic title-tag must remain summarized before Generate. Reuse keeps the existing title/tag policy and parked attachments.
- **Style means the engine.** Remove the competing always-visible `MobileStyleChips` strip from the main composer. Preserve its existing prompt-modifier behavior behind an explicitly named secondary control until templates and saved drafts are migrated; never silently change an existing request.
- **The result owns its actions.** Save, share, reuse, upscale, export, and variations act on the displayed print and its frozen machine/request, not whichever model or prompt is currently in the editor. Show only supported actions; retain video streaming and mesh viewing.

### More settings

Use one sheet with groups, not a succession of stacked sheets. Quality appears first where the recipe offers it; then Detail, Stick to my words, size, Repeat this look, and capability-dependent media/mesh/add-on controls. Keep advanced video, negative prompt, identity tuning, and format available without showing irrelevant groups.

Reuse `desktop/src/lib/qualityPresets.ts` initially (it is a pure helper) or move it and its tests to Studio with a desktop re-export. Do not copy its ladder logic. Fixed-step recipes show their explanation; custom steps show Custom. Do not copy the mock's 8/28/50 values or its invented durations.

Keep machine routing in the header rather than repeating it in the sheet. Group-specific Reset must have a clear scope; the main Reset is separate. Close/Done preserves the existing draft, and any reset that discards authored media must remain explicit. More settings shows a meaningful active-setting summary.

Use inline expansion or a pushed detail within the same sheet for seed/resolution/add-on editing; use native photo/file selection where needed. Restore focus and scroll after returning. Essential validation must not depend on whether its control is currently mounted.

### Queue

Show work across connected machines: Being made, Waiting, Needs attention, and Finished. Identify the machine on every row, with sentence-first status and secondary technical details. Preserve the distinction between this phone's submissions and other work visible on a host. Reconcile host activity and local submissions by exact machine/job identity so they do not appear twice.

Promote `MobileApp.vue`'s existing `mobileActivityRows`, exact-host/job deduplication, and capability actions rather than reconstructing a queue. Full `/api/queue` entries remain memory-only because they contain prompts; offline restart display uses only the existing safe, prompt-free activity projection. Finished means bounded phone/session history plus correlated gallery results, never durable fleet-wide history. Promote existing host-detail controls rather than implementing another queue protocol. `MobileHostDetail.vue` already gates dispatch pause, per-job pause, reorder, and cooperative cancellation on capabilities. Use explicit row menus as well as swipe affordances. Per-job pause must never pause the machine queue. Pausing dispatch means “Pause after this print”; it does not suspend the running denoise. Running cancellation appears only where supported.

An offline machine retains its last-known rows with stale status and disabled mutations. Retry reconnects before acting. Use only server-backed ETA; never sum concurrent-machine work into a false fleet completion time. Fleet operations report partial failures persistently and name their scope. Finished items open the existing viewer; historical content can come from the gallery when available, without implying an infinite durable queue history.

### My images

Keep the fast cached grid and its 2–5-column pinch behavior. Add the new token styling, plain-language scopes, search, and compact filters. Preserve favorites, albums, tags, per-host copies, Trash, New badges, and upscaled provenance.

The viewer keeps horizontal navigation, native image long-press actions, video playback, mesh interaction, save/share, and a details sheet. Keep the visible Select action instead of replacing native long-press with selection as the mock suggests. Offline cached thumbnails stay visible; full-resolution actions explain when the source machine is required. Multi-host mutations preserve failed copies and show persistent errors.

### Styles, Machines, and Settings

Styles uses Ready to use / Browse more and phone-sized rows: purpose, friendly name, technical ID, availability, machine, size, and supported download progress. Reuse the catalog and license flows. Do not promise fixed speeds from storyboard examples; optional estimates need real samples and context.

Machines uses a concise status sentence and a push detail screen for queue, downloads, storage, and telemetry. Connect offers QR pairing, discovery, and manual address entry. Keep remote GPU rentals already reachable as ordinary machines, but do not add phone-side RunPod provisioning or billing controls solely because the mock includes them.

Settings uses grouped lists for appearance, Photos behavior, organization preferences, machine-scoped licenses, and About/privacy. Preserve Safelight as the current fresh-install default and valid saved choices. Use `toneLabel`. Trash retention belongs to a named machine, not an ambiguous global phone switch.

The mock's background-generation toggle, completion notifications, Wi-Fi-only style downloads, rental controls, and “nothing is uploaded” sentence do not match the current remote-client contract. Omit them from this migration. Explain instead that accepted work continues on the selected machine. Notification delivery would require separate design and infrastructure. Style downloads occur on that machine; source media is sent there too.

“Save every result” is now technically supported, but its real behavior is publication followed by Trash, not keeping unsaved pixels only on the phone. If exposed, explain that behavior and prove result retrieval and Photos auto-save for this path before enabling the control. Keep it distinct from Save to Photos.

## Implementation architecture

Maintain one `GenerateForm`: remembered styles select a model, not a separate per-kind draft. Preserve `resolveOutputShape`/`CanvasIntent`, source/reference/identity parking, restored unavailable style IDs, fixed recipe fields, and prepared-batch staleness checks when switching sections. Main navigation is shared Vue, with one shared navigation/transient-surface policy; native Back delegates to it. Test tab reselection, pushed host/queue detail, viewer, pairing scanner, top-sheet dismissal, and focus restoration.

The current `MobileApp.vue` is 12,928 lines and combines shell, orchestration, and page templates. Extract screen presentation incrementally while keeping one owner for hosts, draft, admission, live activity, and gallery state. Do not rebuild these systems in a new screen store.

Proposed boundaries: `MobileShell`, `MobileMakeView`, `MobileQueueView`, `MobileImagesView`, and a consolidated settings sheet. Existing Catalog, HostDetail, GalleryViewer, source wells, native bridges, and pure helpers remain useful. Component names are implementation suggestions, not a requirement to perform a wholesale file move.

| Layer                               | Responsibility                                                                                                           |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `ui/`                               | Shared theme maps, icons, low-level primitives, accessible semantics.                                                    |
| `studio/`                           | Shared capabilities, request policies, remembered styles, organization and queue presentation helpers. No Tauri imports. |
| `desktop/src/mobile/`               | Shared iOS/Android screens, mobile navigation and layout, orchestration with explicit targets.                           |
| `apps/mobile/` and platform bridges | Secure credentials, pairing/discovery, media, background admission lease, appearance and insets.                         |

Replace mobile legacy tokens with `--mold-*` and remove the shell's fixed-radius overrides. Build mobile sizing on top of shared primitives: at least 44pt targets on iOS, 48dp on Android, and at least 16px editable text on iOS. Do not import desktop control metrics that shrink touch controls to 26px. Do not remove the shared legacy bridge until the web surface no longer needs it.

Keep one mobile product UI. Android differences belong in the existing adapter layer and scoped platform metrics: system Back, keyboard/insets, picker/camera, sharing, secure storage, and permissions. Avoid separate iOS and Android copies of form or queue policy.

## Delivery sequence and acceptance

All milestones stay on the single long-running branch `feat/mobile-studio-redesign`, with conventional commits and incremental pushes. At every major milestone: fetch `origin`, inspect newly merged PRs and changes to mobile/shared code, merge `origin/main` when it has advanced (no history rewriting of the shared branch), rerun affected checks, commit and push, then record the exact pushed SHA. Final CI must pass on the exact final head. Each phase leaves a functioning app and updates the checklist below.

1. **Baseline and shell.** Capture current iPhone flows and a representative state matrix, then extract navigation and introduce five tabs together with a usable Queue destination backed by the existing activity/actions. Other tabs keep their functioning current views while adopting new tokens, phone typography, and vocabulary. No placeholder destinations. Keep request ownership stable. Acceptance: all destinations usable; no lost draft, gallery state, or pairing; all six themes readable; Android still builds.
2. **Make and settings.** Deliver the three output kinds, remembered style picker, result/composer hierarchy, consolidated settings, and optional organization. Acceptance: first still can be made without opening More settings; clip Length stays on the main screen; photo-driven 3-D makes no false prompt demand; old templates and reuse preserve request behavior.
3. **Queue.** Refine the already-functional Queue tab into status sections and machine-aware actions, with the compact Make card. Keep full queue snapshots memory-only and safe offline activity separate. Acceptance: no duplicate jobs, clear machine identity, exact-host actions, truthful pause/cancel semantics, and foreground recovery without resubmission.
4. **Images and remaining destinations.** Restyle grid/viewer, Styles, Machines, and Settings, preserving their current contracts. Acceptance: cached first paint, native media actions, downloads/license recovery, multi-host partial failures, and no dead settings copied from the mock.
5. **iOS acceptance, then Android acceptance.** Complete browser layout coverage, simulator checks, and physical iPhone UAT. Follow with Android emulator coverage and a physical Android smoke run for camera/FileProvider, sharing, MediaStore, credentials, gesture navigation, keyboard, and insets. If physical hardware is unavailable, keep that acceptance gate open and explicitly limit the Android completion claim. Update `docs/design`, mobile rules, maintainer docs, and user guides. Release through the existing workflows only in the implementation/delivery scope.

Use iOS acceptance feedback to settle layout before Android-specific polish, while running the Android shared-frontend/build checks throughout so it does not become a later port.

## Verification matrix

- Small iPhone (375px), standard iPhone (390/393px), large iPhone, landscape, and iPad. Check keyboard open/closed, long prompts, picker return, safe areas, scroll restoration, and Generate reachability.
- All six themes; light/dark system matching; VoiceOver names/order, focus restoration, large text, contrast, and reduced motion. Do not count a static mock screenshot as accessibility proof.
- Fresh install; one machine; multiple machines; offline; credential rejection; instance replacement; older host with missing additive capabilities; no installed style; missing style requiring download.
- Still, fixed-step style, image editing with ordered references, identity parking, short/long clip, audio where supported, and mesh with named views/texture where advertised. Fixed fields stay fixed, irrelevant fields stay off the wire.
- Batch submission, prepared expansion/review, partial admission failure, submission cancellation, app background/foreground, durable resume, and edits to the draft while a prior result is displayed.
- Library cold start with cached data, offline thumbnails, pinch, selection, native long-press, video seeking, mesh gestures, Trash/restore, organization fan-out, and failed-host copies.
- Android system Back dismisses the top transient surface first; 48dp controls; gesture/navigation-bar insets; native media and permissions remain correct.

Run focused behavioral Vitest coverage per slice, then the repo's frontend architecture/tests/build gates and applicable mobile release-asset/native checks in Nix. Run iOS simulator Rust check/build and Android validation for affected shared-mobile code. Exact-head CI and a physical iPhone run are required before claiming implementation complete; TestFlight delivery additionally requires a VALID processed build and tester access.

## Decisions and residual risks

The recommended direction is five tabs, a single More settings sheet, capability-driven output kinds, and the existing Tauri/Vue foundation. No SwiftUI rewrite or OS-26-only glass dependency is needed for this design.

The main engineering risks are unmounting controls that currently report validity, disturbing the long-lived mobile orchestrator, accidentally importing desktop-local state, losing native gestures, and making old-server behavior stricter during the visual migration. Address these with incremental extraction and request/outcome tests rather than a second implementation of the domain rules.

Before the first implementation slice, capture live baseline screens and resolve the small-phone keyboard layout in a working prototype. The static references do not prove that the result, editor, progress card, and five-tab bar fit while typing. Treat that as the first design validation, not as a late CSS adjustment.

## Locked plan review and milestone ledger

Independent review: GPT-5.6 Sol, medium reasoning, `plan_review`. All seven findings addressed: functional Queue at shell rollout; safe Queue lifetime; incremental single-branch sync/push; one form and conditioning authority; shared navigation/overlay policy; physical Android acceptance; explicit short-tab vocabulary exception. No scene authoring is in scope.

- [x] Research current history, desktop, and attached references.
- [x] Independent plan review; resolve findings and lock plan.
- [x] Baseline mobile tests: 52 files / 1,068 tests pass. Browser baseline at 393×852.
- [x] Milestone 1: working five-tab navigation, initial Queue destination, direct theme tokens and navigation vocabulary. Later screen-specific vocabulary remains in milestones 2–4.
- [x] Milestone 2: Make, three output kinds, style memory, consolidated settings.
- [x] Milestone 3: Queue sections, details, safe offline state, exact-machine actions.
- [x] Milestone 4: Images, Styles, Machines, Settings.
- [ ] Milestone 5: full regression, visual comparison, native iOS acceptance, Android acceptance.

Sync baseline: `origin/main` remains `c38d569d` at plan lock; no new main commits.

Milestone 1: functional five-tab navigation with existing queue rows/actions, compact Make link, per-destination scroll, direct mobile theme tokens, shared icons, My images/Styles headings, Generate wording. Mobile production build and frontend architecture pass. Mobile regression: 1,077 tests passed; one remaining obsolete CSS-token assertion corrected; focused regression passes (including Queue draft/scroll preservation). Browser: 393×852 navigation, empty Queue, Images, Styles, and return to Make verified. Native runtime acceptance remains open. Main checked again: still `c38d569d`. Plan push: `1c67ab3d`.

Milestone 2, first slice: shared output-kind helpers and remembered styles; prompt-first composer; collapsed prompt tools, organization and size; Detail/guidance/seed/mesh in More settings with changed-setting counts; focus restoration; explicit result actions and frozen-result reuse. Full mobile regression passed 1,073 tests before correcting one test-root selector; focused sheet tests then passed 4/4. Latest composer/sheet regression passes 332 tests, including result reuse after draft edits. Mobile production build and architecture pass. Browser fixture layout reviewed at 393×852; native acceptance remains open. New main PR #1627 identified for integration before continuing. Milestone 2 remains in progress.

Queue/keyboard slice: grouped active work into Being made, Waiting, Needs attention; bounded the phone's settled list to 20 jobs; added read-first Queue details with explicit reuse and result viewing, preserving the Make draft until requested. More settings, library sheets and the generated viewer use shared Android transient Back handling. Full mobile regression: 55 files / 1,079 tests pass; mobile production build passes. Signed iOS simulator build/install/launch succeeded on iPhone 17 Pro / iOS 26.5. Native UAT found and verified a keyboard fix: visual viewport sizing keeps Generate above the software keyboard and restores navigation on dismissal. More settings and quality controls visually verified. Physical iPhone devices are currently unavailable; that acceptance gate remains open. Main synchronized through #1627 at `8774bc50`, with no newer main commit at the next fetch.

The user authorizes real generation UAT on Plato through its Tailscale IP. Live discovery confirms it is reachable, version 0.28.0 at server commit `271fa93a`, and has installed still/video/mesh models. Test the advertised server capabilities as they stand; client-main features absent from this host require fixture coverage or a later server update, not an implicit deployment.

### First TestFlight milestone

The user now authorizes merging coherent milestones and will verify remotely through TestFlight. The first candidate delivers the shared navigation, Make and Queue foundation plus the initial Machines/Settings cleanup. This is an incremental delivery, not completion of the full acceptance matrix. Keep this branch for subsequent work, merge main after milestone integration, and track remaining physical-device and generation UAT explicitly.

Destination follow-up: saved machines lead their screen; Add a machine contains pairing/discovery/manual entry and starts expanded for an empty install. Settings uses shared vocabulary and platform-neutral appearance copy. Settings no longer renders Images or Machines underneath it. Library sheets contain keyboard focus and only the top sheet handles Escape; Settings participates in Android Back. Small-phone theme selection no longer scrolls the shell header out of view. Remove-background overrides now count in More settings.

Validation: 56 mobile files / 1,083 tests pass; architecture passes. The prior Android native build produced its debug APK successfully. Browser review at 375×667 covered all six themes, and 844×390 covered landscape navigation. Signed native iOS rebuild/install/launch passed. Full frontend gate passed: Studio 1,688 tests, web 1,757 tests, desktop/shared 6,295 tests, plus web/desktop production builds. Generation UAT has not been submitted. Plato is reserved strictly for tests requiring actual generation; layout and navigation use local fixtures.

### Keyboard acceptance follow-up

The user merged #1629 at `9537a835`; this branch merged that main commit and continues without a history rewrite. TestFlight 0.28.0 (0.28.0.1308) reached VALID and Mold Internal tester access was verified by run 34090548322. This is the foundation delivery, not closure of the entire acceptance matrix.

Expanded native keyboard testing found and fixed low numeric-field occlusion, rotated iPhone editor starvation, viewer viewport/safe-area sizing, and a scrolled-away sheet exit. All software-keyboard editors are revealed after viewport changes. Title and seed Done dismiss editing; manual machine setup traverses Name → Address → API key. Library sheets keep Done outside the scrolling body. Android transient history now handles nested viewer/details, catalog target selection, simultaneous close and reopening; the identity picker no longer double-registers its sheet.

Native iPhone 17 Pro / iOS 26.5 checks cover prompt open/dismiss/rotation, custom Width/Height traversal, title focus, viewer tag and lower collection editors, and persistent sheet exit. Local fixtures handle layout without inference. Full frontend validation passes (Studio 1,694; web 1,757; desktop/shared 6,332 tests and production builds). Native simulator build/install succeeds. Physical iPhone acceptance remains with the user through TestFlight; Android emulator and generation acceptance remain tracked separately.

### Primary controls and spacing follow-up

Color / PBR now sits directly below Style for a host advertising adjustable mesh texture, with its advertised texture sizes. Hidden and fixed texture controls remain unavailable; geometry stays in More settings. Older hosts that advertised a hidden matting placeholder retain their otherwise valid mesh profile, without accepting malformed current matting contracts. Mesh source images no longer carry a misleading strength hint. Repeat this look follows desktop order: Keep, then Surprise me.

My images searches all loaded metadata before thumbnail windowing: filenames, titles, prompts, styles, tags, and collection names, including every physical copy of a logical print. Queue group headings have consistent padding, and prompt-free mesh jobs use a model/machine title and accessible Actions label.

The complete frontend gate passes: Studio 1,695, web 1,757, desktop/shared 6,339 tests, plus production builds. Native iPhone layout checks confirm primary PBR and texture sizes, old-host prompt-free behavior, Queue group spacing, and live filtering against the cached library. Android's initial 21 instrumentation tests passed; a subsequent visual audit found duplicate system insets, now consumed by the native container after it applies them, with all 21 instrumentation tests passing again. The rebuilt Android emulator confirms the gap is gone and the active editor and Generate remain above the software keyboard. No generations were used for these layout checks. Milestones 2–4 describe implemented screens; milestone 5 remains open until its acceptance evidence is complete.

### Dynamic Type and native generation acceptance

The controls milestone #1630 merged at `8564ee56`. TestFlight **0.28.0 (0.28.0.1309)** reached VALID and internal tester access was verified by run 34095180972. This branch remains synchronized with that main head.

The native iOS shell now follows the system Dynamic Type body size, gated by the native iOS runtime. Shared text tokens and mobile-transitive component text use rem units with identical desktop values at its normal 16px root. Editors retain a 16px floor. Navigation chrome remains bounded, while content, headings, segments, shape choices, and actions reflow. Maximum accessibility text exposed and corrected clipping in Images, Styles, and Machines; the Library heading scrolls away instead of occupying the whole viewport. The persistent Generate area stays compact in both orientations so the form remains reachable. With the landscape software keyboard open, the prompt editor preserves one complete line, capped at 32px only in that short editing viewport; portrait content retains the requested Dynamic Type size.

Held Queue errors now use the full card width with a three-line preview, full details, and status/error in the accessible name. Recovered byte-free presentation stubs fall back to the model name without eagerly retrieving or persisting prompts.

Independent source review found no blockers. The full frontend gate passes (Studio 1,695; web 1,757; desktop/shared 6,341 tests and production builds), with 106 focused assertions passing after the additional screen reflow. Native iPhone simulator builds succeed. Android rebuild and all 21 instrumentation tests pass. Native maximum-text portrait checks verify Make, Queue and its detail actions, Images search/scope, Styles host picker, Machines addresses/status, and Settings. Maximum-text landscape checks also verify the navigation rail, wrapped output selector, populated VRAM footer, software keyboard, complete prompt line, Generate and Done. The remaining device matrix continues as an acceptance gate; physical acceptance is not implied by simulator evidence.

Actual generation UAT used the iPhone app pinned to Plato's Tailscale address, without changing the server or other jobs:

- Still: Z-Image Turbo Q8, 1024 square, nine steps. Batch `ff064997-057c-4bfe-91ac-fb1afb6c93c6`, job `7e8047de-fe4c-4d44-b90d-61671eefad31`, result `mold-z-image-turbo-q8-1788765418170.png`. The app recovered after relaunch during work, displayed completion and the correct image, and exercised Photos permission.
- Short clip: LTX-Video 0.9.8 distilled, 512 square, 17 frames at 30fps. Batch `52c0926d-7436-4e6e-95e8-9b543516f5b1`, job `2990cba0-553b-4ff3-a5c6-82d610260c32`, result `mold-ltx-video-0.9.8-2b-distilled-bf16-1788767068346.mp4`. Completed in 8,111ms and played in the native viewer. An earlier larger attempt was held for GPU memory; only that UAT job was cancelled before retrying smaller. The held state supplied the real error-layout regression.

- PBR: Hunyuan3D 2.1 FP16, 15 steps, octree 128, 1024px textures, native Photo Library chair source. Batch `700fa699-4bde-43f3-bf5b-e0067539e4bf`, job `3218906a-1c1a-4871-881d-7cf0ad55d0bc`, result `mold-hunyuan3d-2.1-fp16-1788769770804.glb`. Completed in 467,793ms. The native viewer displays the colored chair and responds to orbit and Reset view. The GLB contains UV coordinates plus embedded 1024-square base-color and metallic/roughness textures. Native Share → Save to Files completed into the Mold folder; the saved 7.2 MB GLB is byte-identical to the host result. Pinch remains unverified.

User physical iPhone verification remains open. Evidence screenshots are retained under the external volume's `build/mobile-redesign-evidence` directory.

Native iPad Pro 11-inch checks at standard and maximum accessibility text additionally caught a vertically stretched navigation rail. Tall tablet rails now group destinations near the top, while short phone rails keep their existing fit. Machine setup uses adaptive columns so enlarged fields stack; the normal iPad layout remains two columns with a full-width submit action. Settings and first-launch setup remain scrollable and their exit/navigation controls reachable.

### Viewer theme acceptance follow-up

Native PBR export revealed dark text on the fixed dark details sheet in light themes. The media canvas remains dark, while the details sheet now uses the selected theme’s background, text, labels, errors, and controls. Native renders were inspected in all six themes; Match phone was exercised through system light and dark changes. Sheet text, secondary text, labels, accent, and error colors all exceed 4.5:1 against each theme’s sheet background. The swatch test now asserts six parsed themes so selector changes cannot silently skip coverage.

TestFlight **0.28.0.1310** from main `0ac3dc9e` is VALID with Mold Internal tester access verified in run `34100408172`. It contains the accessibility milestone; the viewer theme follow-up is a subsequent change. The remaining native lifecycle, gesture, generation-mode, and physical-device acceptance rows remain open.

### Library selection acceptance follow-up

Native selection of the UAT still, confirmation to move it to Trash, and Restore completed successfully; the print returned to the Library and no other print was changed. Selection actions now wrap by their label widths, preserving “Add to collection” instead of splitting its last letter onto a separate line. Persistent action text is capped at 24px (status at 20px) so maximum Dynamic Type leaves the grid reachable. The favorite icon and selection check are bounded within their fixed targets. Native maximum-text scrolling, the selected-still action set, and the final selection-check glyph cap were verified in rebuilt simulator apps.

### Style picker completion

The native Style picker now presents the server’s friendly description before the runnable model ID, falling back to the shared display/family label for older or unavailable entries. A wrapping selected-ID line remains readable when the native select truncates its option. Browse more is always reachable and explicitly opens the current output kind; the separate failed-kind notice still browses the attempted unavailable kind.

Native iPhone checks verified the descriptions in the system menu, dismissal without selection changes, Browse more opening 3-D discovery, and returning to the same Make selection and scroll position. All 404 mobile app, style-label, and layout tests pass, including image/video/mesh browse routing and the stale attempted-kind regression. The simulator build and formatting checks pass; independent review found no behavioral blockers.
