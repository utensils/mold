# Generate Studio Console Follow-Up TODO

This is the consolidated, de-duplicated backlog from UAT feedback on the
Generate Studio Console in PR #326. Treat each top-level section as a likely
follow-up PR slice unless an item explicitly depends on another section.

**Status (2026-08-30):** closed historical backlog. Every checklist item here
is already `[x]`, and the web surface has since been rebuilt on the
five-workspace IA (`/create`, `/library`, `/models`, `/machines`,
`/settings`), so most of the "Likely files" pointers below name paths that no
longer exist. `GeneratePage.vue`, `GalleryPage.vue`, `ModelPicker.vue`,
`GenerateParamsPanel.vue`, `Composer.vue`, `RunningStrip.vue`,
`RunningJobCard.vue`, `TopBar.vue`, and `useHideMode.ts` are gone, replaced by
`web/src/pages/CreatePage.vue`, `web/src/pages/LibraryPage.vue`, and
`web/src/components/create/` (`CreateModelPicker.vue`, `ControlsAside.vue`,
`AdvancedDrawer.vue`, `ComposerCard.vue`, `ActivityStrip.vue`); hide/blur mode
and the top bar were removed outright. Read the pointers as historical
provenance, not as current locations.

Named templates are web-local and intentionally omit binary media payloads
while preserving safe path references; **Models ▸ Discover** (route `/models`)
is the install/repair surface for missing models and components.

## 1. Model Selection

- [x] Filter the Generate model picker to standalone generation models only.
  - Exclude upscalers, prompt-expansion utilities, companion/support models,
    ControlNet auxiliaries, and other non-primary generation assets from the
    main model list.
  - Keep `/api/models` broad for catalog/model management unless the backend
    contract is intentionally changed.
  - Likely files: `web/src/types.ts`, `web/src/components/ModelPicker.vue`,
    `web/src/pages/GeneratePage.vue`.

- [x] Default the Generate model picker to downloaded generation models only.
  - Remove `Show all` and `Download` actions from the picker.
  - If no downloaded generation models exist, show an empty state with a link
    to Models ▸ Discover.
  - Use the same downloaded-generation filter for first-run auto-selection.
  - Likely files: `web/src/components/ModelPicker.vue`,
    `web/src/pages/GeneratePage.vue`, `web/src/router.ts`.

- [x] Group models by family with collapsible sections.
  - Preserve stable ordering within each family.
  - Persist family collapsed/expanded state in localStorage.
  - Add readable family labels instead of raw family IDs.
  - Likely files: `web/src/components/ModelPicker.vue`, `web/src/types.ts`.

- [x] Add multi-sort support for models.
  - Support sorting by name, family, size, quantization, downloaded state, and
    variant quality/rank.
  - Preserve stable ordering when multiple sort keys are active.
  - Consider exposing sort-friendly fields on the wire instead of duplicating
    Rust model ordering logic in TypeScript.
  - Likely files: `web/src/components/ModelPicker.vue`,
    `crates/mold-core/src/catalog.rs`, `web/src/types.ts`.

- [x] Add advanced model filters.
  - Filter by name, family, size range, quantization, downloaded state, and
    other model metadata.
  - Support boolean modes for selected filters: `ANY`, `ALL`, and `NOT`.
  - Keep the UI compact for long model lists.
  - Likely files: `web/src/components/ModelPicker.vue`; possibly new helper
    module such as `web/src/lib/modelFilters.ts`.

- [x] Add focused model picker tests.
  - Cover generation-only filtering, downloaded-only default, empty catalog
    link, family collapse/expand, sorting, and boolean filters.
  - Likely files: new `web/src/components/ModelPicker.test.ts`.

## 2. Layout And Controls

- [x] Widen and de-cramp the Controls rail.
  - Current rail is fixed around `24rem`; increase desktop width and/or switch
    to a two-column controls layout on wide monitors.
  - Reduce empty left/right margins on large and 4K displays while preserving
    mobile stacking.
  - Likely files: `web/src/pages/GeneratePage.vue`,
    `web/src/components/GenerateParamsPanel.vue`.

- [x] Replace cramped pills with consistent segmented controls.
  - Fix `random` / `increment` seed controls so labels fit cleanly.
  - Normalize seed mode, size, batch, dirty/custom indicators, and other pills
    into reusable controls.
  - Likely files: `web/src/components/GenerateParamsPanel.vue`,
    `web/src/components/GenerateParamsPanel.test.ts`.

- [x] Centralize model-family generation capabilities.
  - Drive scheduler options, CFG++ visibility, negative prompt support, video
    controls, source/edit image behavior, and LoRA support from one frontend
    capability map.
  - Add tests to prevent frontend/backend capability drift.
  - Likely files: `web/src/types.ts`, new
    `studio/lib/generationCapabilities.ts`,
    `web/src/components/GenerateParamsPanel.vue`.

- [x] Show scheduler and other model-specific options only when valid.
  - Replace the current hard-coded scheduler list with options appropriate for
    the selected model/family.
  - Ensure flow models do not show controls that the backend ignores.
  - Depends on centralized generation capabilities.
  - Likely files: `web/src/components/GenerateParamsPanel.vue`,
    `studio/lib/generationCapabilities.ts`.

- [x] Move or mirror placement controls into the Controls rail.
  - `PlacementPanel` currently lives in the composer, while users expect CLIP,
    text encoder, transformer, VAE, and placement controls in Controls.
  - Decide whether to move the existing panel or create a Components section
    that includes placement plus component status.
  - Likely files: `web/src/components/PlacementPanel.vue`,
    `web/src/components/GenerateParamsPanel.vue`,
    `web/src/pages/GeneratePage.vue`.

- [x] Add component asset/status controls.
  - Show which CLIP/text encoder/VAE/transformer/etc. components are used.
  - For components that can be changed, show available options in dropdowns.
  - If required components are missing, highlight the field with an error and
    provide a path to repair/download from Models ▸ Discover.
  - Requires backend metadata such as `components: [{ kind, name, present,
    path?, repair_model? }]` or a dedicated model-components endpoint.
  - Likely files: `crates/mold-core/src/types.rs`,
    `crates/mold-server/src/routes.rs`, `web/src/types.ts`,
    `web/src/components/GenerateParamsPanel.vue`.

- [x] Replace free-text upscaler model input with a dropdown.
  - Derive options from installed/available upscaler models.
  - Include a `None` option and downloaded/missing status.
  - Keep upscalers excluded from the primary generation model picker.
  - Likely files: `web/src/components/GenerateParamsPanel.vue`,
    `web/src/pages/GeneratePage.vue`, `web/src/types.ts`.

- [x] Show estimated peak memory usage.
  - Use server-side preflight/memory estimation logic, not a static file-size
    guess.
  - Estimate should be request-sensitive: model, resolution, batch, frames,
    placement, LoRAs, and relevant offload/runtime settings.
  - Requires a new estimate endpoint or additional API metadata.
  - Likely files: `crates/mold-server/src/model_manager.rs`,
    `crates/mold-server/src/routes.rs`, `crates/mold-core/src/types.rs`,
    `web/src/api.ts`, `web/src/components/GenerateParamsPanel.vue`.

## 3. Generation Templates And Recreate

- [x] Add named generation templates.
  - Templates should save and load all generation configuration, including
    model, prompt, negative prompt, size, steps, guidance, seed mode/value,
    scheduler, LoRAs, source/mask/control settings, video settings, placement,
    and component selections.
  - If seed mode is random, template recreation is not 1:1; if static, it
    should be reproducible when all other inputs are available.
  - Decide whether templates are web-local localStorage, DB-backed and
    profile-scoped, or shared across CLI/TUI/web.
  - Likely files: `web/src/composables/useGenerateForm.ts`, `web/src/types.ts`,
    `web/src/components/GenerateParamsPanel.vue`; DB/API files if persisted
    server-side.

- [x] Add template browsing and management UI.
  - Save, load, rename, delete, search, sort, and scroll/browse templates.
  - Keep the UI compact enough for many templates.
  - Likely files: new template picker component, `GenerateParamsPanel.vue`,
    `GeneratePage.vue`.

- [x] Decide media handling for templates.
  - Current form persistence strips binary base64 image/video/audio payloads.
  - Options: omit binary media, store gallery references, store server-local
    media paths, or add DB-backed blobs/references.
  - This decision affects exact recreation for img2img, masks, audio, source
    video, and keyframes.

- [x] Unify form serialization helpers.
  - Gallery Recreate, template save/load, localStorage persistence, and
    request serialization should share helper functions instead of manual field
    copying in multiple places.
  - Add helpers such as `applyMetadataToForm()`, `cloneTemplateForm()`,
    `sanitizePersistedForm()`, and keep `toRequest()` as the wire serializer.
  - Do this before implementing templates to avoid drift.
  - Likely files: `web/src/composables/useGenerateForm.ts`,
    `web/src/pages/GeneratePage.vue`, `web/src/types.ts`.

## 4. LoRA Stack Ergonomics

- [x] Add drag-and-drop LoRA stack reordering.
  - Current reorder path is too cumbersome.
  - Preserve each row's path, scale, and trained words while reordering.
  - Include accessible keyboard/button fallback controls.
  - Likely files: `web/src/components/LoraPicker.vue`,
    `web/src/components/LoraPicker.test.ts`.

- [x] Lock LoRA request serialization order in tests.
  - `toRequest()` already serializes `loras` in array order; add tests so drag
    reorder cannot regress wire order.
  - Likely files: `web/src/composables/useGenerateForm.test.ts`.

## 5. Mask Editing And Media Workflows

- [x] Add a mask editor for img2img/inpaint workflows.
  - New UI should support drawing and editing masks on uploaded, source, or
    gallery images.
  - Needed controls: brush, erase, brush size, clear, invert, undo/redo, and
    apply/save.
  - Output should populate `form.state.maskImage`.
  - Likely new file: `web/src/components/MaskEditorModal.vue`.

- [x] Preserve direct uploaded-mask support.
  - Existing upload-mask behavior should remain available, either as a direct
    path or as an "Upload mask" action inside the mask editor.
  - Do not regress `mask_image` request serialization.
  - Likely files: `web/src/components/GenerateParamsPanel.vue`,
    `web/src/composables/useGenerateForm.ts`.

- [x] Add source/gallery entry points for mask editing.
  - Generalize `ImagePickerModal` or add a mask-specific launcher so users can
    edit masks from upload, current source image, or gallery image.
  - Fetch full gallery image before opening the editor.
  - Likely files: `web/src/components/ImagePickerModal.vue`,
    `web/src/components/GenerateParamsPanel.vue`, `web/src/api.ts`.

- [x] Validate mask/source combinations before submit.
  - For non-Qwen img2img/inpaint, a mask requires a source image.
  - Show a local form error matching backend behavior.
  - Keep Qwen Image Edit separate because the backend currently rejects
    `mask_image` for `qwen-image-edit`.
  - Likely files: `web/src/pages/GeneratePage.vue`,
    `web/src/composables/useGenerateForm.ts`,
    `web/src/components/GenerateParamsPanel.vue`.

- [x] Consider drag reorder for Qwen edit attachments.
  - Current target/reference images have button-based reordering.
  - Drag reorder could improve the workflow, but index `0` must remain clearly
    labeled as the target image.
  - Likely files: `web/src/components/Composer.vue`,
    `web/src/components/Composer.test.ts`.

- [x] Add mask editor tests.
  - Draw/apply emits a base64 mask.
  - Mask editor visible for non-edit image workflows and hidden/disabled for
    Qwen Image Edit.
  - `mask_image` serializes for non-edit and is omitted for Qwen edit.
  - Likely files: new `MaskEditorModal.test.ts`,
    `GenerateParamsPanel.test.ts`, `useGenerateForm.test.ts`.

## 6. Running Jobs, Queue, GPU Lanes

- [x] Keep running/queued items always visible and sorted left.
  - Replace or extend the current flat `RunningStrip`.
  - Active/running jobs should appear before queued/future jobs.
  - Likely files: `web/src/components/RunningStrip.vue`,
    `web/src/components/RunningJobCard.vue`, `web/src/pages/GeneratePage.vue`.

- [x] Use `/api/queue` as a reusable UI data source.
  - Current queue polling is primarily for zombie-card reconciliation.
  - Expose queue state in a composable that UI components can render directly.
  - Likely files: `web/src/api.ts`, `web/src/types.ts`,
    `web/src/composables/useQueueReconciler.ts`, new queue composable.

- [x] Add one queue lane per GPU.
  - Running jobs can use the existing `gpu` field from `/api/queue`.
  - Queued jobs currently have no GPU assignment; show them in an `Auto` lane
    until the server exposes target/preferred GPU.
  - Likely files: `web/src/components/RunningStrip.vue`,
    `web/src/pages/GeneratePage.vue`, `web/src/types.ts`.

- [x] Add queued-job lane assignment to the server contract.
  - The current queue API intentionally omits `gpu` for queued rows.
  - Add `target_gpu` / `preferred_gpu` metadata for queued jobs, populated
    from request placement or dispatcher assignment.
  - Likely files: `crates/mold-server/src/job_registry.rs`,
    `crates/mold-server/src/queue.rs`, `crates/mold-server/src/routes.rs`,
    `crates/mold-core/src/types.rs`, `web/src/types.ts`.

- [x] Support lane changes for queued jobs.
  - Add an endpoint such as `PATCH /api/queue/:id` with
    `{ target_gpu: number | null }`.
  - Reject changes for already-running jobs.
  - Dispatcher may need shared job metadata or lookup by job id because jobs
    currently flow through channels/lookahead buffers.
  - Depends on queued-job lane assignment.

- [x] Join local stream jobs to queue entries.
  - Capture server id from the first `queued` SSE event.
  - Join local `Job` rows to `/api/queue.entries` by id for lane display,
    position, and actual running GPU.
  - Likely files: `web/src/composables/useGenerateStream.ts`,
    `web/src/components/RunningStrip.vue`.

- [x] Add queue/lane tests.
  - Vue tests for lane grouping, running-left ordering, queued `Auto` lane,
    actual GPU lanes, and disabled lane changes for running jobs.
  - Rust route/registry tests for queued lane metadata and PATCH validation.

## 7. Gallery And NSFW Visibility

- [x] Remove hide/blur controls and default to showing all content.
  - Remove `useHideMode` wiring from Generate and Gallery.
  - Remove hide buttons from `TopBar`.
  - Remove reveal overlays and hide props from gallery and running job cards.
  - Likely files: `web/src/composables/useHideMode.ts`,
    `web/src/components/TopBar.vue`, `web/src/components/GalleryCard.vue`,
    `web/src/components/GalleryFeed.vue`,
    `web/src/components/RunningJobCard.vue`,
    `web/src/components/RunningStrip.vue`, `web/src/pages/GeneratePage.vue`,
    `web/src/pages/GalleryPage.vue`.

- [x] Keep the compact Generate gallery always visible.
  - After hide removal, recent thumbnails should always render normally.
  - Verify refresh-on-complete still works.
  - Likely files: `web/src/pages/GeneratePage.vue`,
    `web/src/components/GalleryFeed.vue`, `web/src/components/GalleryCard.vue`.

- [x] Add visibility tests.
  - No hide buttons/overlays remain.
  - Gallery and running previews are visible by default.
  - Likely files: existing gallery/running component tests.

## Suggested Implementation Order

1. Model list filtering/downloaded-only/empty state.
2. Layout and segmented-control spacing fixes.
3. Remove hide/blur controls.
4. LoRA drag reorder.
5. Centralize capabilities and model-specific controls.
6. Upscaler dropdown.
7. Shared recreate/template serialization helpers.
8. Named templates.
9. Estimated memory endpoint and display.
10. Component status/repair UX.
11. Queue UI lanes using current API, then server lane metadata and lane-change
    endpoint.
12. Mask editor and media workflow upgrades.

## Verification Checklist For Follow-Up PRs

- Web unit tests for each touched component/composable.
- The frontend is one repo-root Bun workspace, so run `bun install
  --frozen-lockfile` at the root, then `bun run check:architecture && bun run
  check:dead-code` and `bun run test:studio` (the shared `studio`/`ui` gates
  CI's `web` job runs), then the web-local
  `cd web && bun run fmt:check && bun run test && bun run build`.
- Rust tests for any API/server contract changes.
- `cargo fmt --all -- --check`.
- `cargo check -p mold-ai --features preview,discord,expand,tui,webp,mp4`.
- `cargo test --workspace` and `cargo clippy --workspace --all-targets -- -D warnings`
  for broad/backend slices.
- Website docs verification if API or user-facing behavior changes.
