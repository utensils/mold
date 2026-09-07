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

| Recent work | Implication for mobile |
| --- | --- |
| #1565 and #1566: six shared themes | Reuse the shipped theme contract; mobile already initializes it before mount. |
| #1595: desktop redesign | Adopt its visual language, plain wording, style picker, and result-first hierarchy. |
| #1598: output sections and remembered styles | Use Still picture / Short clip / 3-D object and the shared last-used-style store. |
| #1600 and #1602: duration and backend follow-ups | Old claims that duration, storage totals, and save-to-gallery opt-out lack server support are stale. Gate presentation on actual host data. |
| #1619: retire interactive scene authoring | Discard the archive's scenes screen. A short clip has one prompt and Length; internal splitting remains invisible. |
| #1623: long-clip recovery fixes | Preserve durable admission, progress, restart, and result recovery when changing the screen. |
| #1620 and #1624: Hunyuan3D expansion | Include current capability-driven mesh, texture, and named-view inputs, beyond the older storyboard. |

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

| Layer | Responsibility |
| --- | --- |
| `ui/` | Shared theme maps, icons, low-level primitives, accessible semantics. |
| `studio/` | Shared capabilities, request policies, remembered styles, organization and queue presentation helpers. No Tauri imports. |
| `desktop/src/mobile/` | Shared iOS/Android screens, mobile navigation and layout, orchestration with explicit targets. |
| `apps/mobile/` and platform bridges | Secure credentials, pairing/discovery, media, background admission lease, appearance and insets. |

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
- [ ] Milestone 2: Make, three output kinds, style memory, consolidated settings.
- [ ] Milestone 3: Queue sections, details, safe offline state, exact-machine actions.
- [ ] Milestone 4: Images, Styles, Machines, Settings.
- [ ] Milestone 5: full regression, visual comparison, native iOS acceptance, Android acceptance.

Sync baseline: `origin/main` remains `c38d569d` at plan lock; no new main commits.

Milestone 1: functional five-tab navigation with existing queue rows/actions, compact Make link, per-destination scroll, direct mobile theme tokens, shared icons, My images/Styles headings, Generate wording. Mobile production build and frontend architecture pass. Mobile regression: 1,077 tests passed; one remaining obsolete CSS-token assertion corrected; focused regression passes (including Queue draft/scroll preservation). Browser: 393×852 navigation, empty Queue, Images, Styles, and return to Make verified. Native runtime acceptance remains open. Main checked again: still `c38d569d`. Plan push: `1c67ab3d`.

Milestone 2, first slice: shared output-kind helpers and remembered styles; prompt-first composer; collapsed prompt tools, organization and size; Detail/guidance/seed/mesh in More settings with changed-setting counts; focus restoration; explicit result actions and frozen-result reuse. Full mobile regression passed 1,073 tests before correcting one test-root selector; focused sheet tests then passed 4/4. Latest composer/sheet regression passes 332 tests, including result reuse after draft edits. Mobile production build and architecture pass. Browser fixture layout reviewed at 393×852; native acceptance remains open. New main PR #1627 identified for integration before continuing. Milestone 2 remains in progress.
