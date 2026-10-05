# Native iOS media export parity

Status: implemented; independent code review fixes incorporated; full Simulator
UAT and final CI verification in progress. The checklist below separates
implemented behavior from completed validation.
Audited against commit `3ba145d9e` on 2026-10-05. “Tally” and “Towery” are
interpreted as the Tauri app (desktop and its iPhone shell).

## Goal

Give native iOS every applicable media export offered by Tauri, through native
menus, Forms and system destinations. Add the same GIF pause control to both
apps. Keep the stored print unchanged and perform conversions on its holding
machine through the existing authenticated export route.

The priority acceptance case is an already looping MP4: GIF, Loop, Forever,
Pause 0 ms. This preserves the ordinary frame cadence and adds no boundary
hold. It does not repair an existing discontinuity or duplicate boundary frame
in the input. Bounce is an independent choice, not a prerequisite for GIF.

## Evidence and gaps

| Capability | Tauri today | Native iOS today | Coverage |
| --- | --- | --- | --- |
| Share original | Native share sheet | Native share sheet | Preserve |
| Save original still/video to Photos | Available | Available, with permission recovery | Preserve |
| Copy still | Available | Available | Preserve |
| MP4 → GIF/APNG/WebP | Export options sheet; WebP depends on host build | No conversion action | Add |
| GIF Loop/Bounce | Available | No control | Add native parity |
| GIF Forever/Once | Available | No control | Add native parity |
| Longest side: Original/1080/720/480 | Available | No control | Add native parity |
| FPS: Original/24/12/8 | Available | No control | Add native parity |
| Extra GIF boundary pause, including zero | No wire field or control | No control | Add to encoder, server and both clients |
| Mesh OBJ/ZIP/STL/PLY conversion | Host-advertised formats; geometry controls | Original GLB sharing only | Add |
| Mesh physical size, up axis, origin | Capability-driven | Shared Swift types exist; no iOS UI | Reuse shared types and add native UI |
| Mesh GIF/APNG/WebP turntable | Export sheet; loop/bounce, repeat, size, FPS; transparent background | No action | Add |
| Turntable frames/budget guidance | Host supports it; Tauri sheet has no frames control and uses video size/FPS presets | Shared Swift budget policy exists; no iOS UI | Fix type-specific limits and offer shared coverage |
| Save exported file into an app-visible Mold folder | Tauri phone offers this for meshes/turntables; clip conversion currently shares only | No dedicated export destination | Add native Mold folder and Files destination; provide consistent clip/mesh destination choices |
| Download generation sidecar assets | Viewer offers host-provided assets | No sidecar download action | Add capability/data-driven asset actions |
| Entry points | Gallery viewer; desktop generation results | Library tile/menu, viewer and generation result menu lack export | One native action coordinator across all three |

Sources:

- `ui/components/VideoExportDialog.vue` and `studio/lib/videoExport.ts`: current
  animation controls, request fields and naming.
- `desktop/src/mobile/MobileGalleryViewer.vue`: MP4-only conversion gating,
  native share/folder routes, mesh geometry, turntables and generation assets.
- `desktop/src/components/gallery/Lightbox.vue`,
  `desktop/src/views/GenerateView.vue`: desktop gallery and result exports.
- `apps/ios/Sources/Companion/Library/PrintMenu.swift`, `PrintActions.swift`,
  `PrintActions+Files.swift`, `PrintViewer.swift`,
  `apps/ios/Sources/Companion/Generate/ResultPager.swift`: original-only actions
  and per-window sheet/file ownership.
- `apps/shared/Packages/MoldClient/Sources/MoldClient/GalleryMutations.swift`:
  `ExportOptions` currently decodes formats only, dropping playback/repeat.
- Shared `MeshExport.swift`, `MeshExportGeometry.swift`,
  `MeshExportRequest.swift`: reusable mesh policy; current Swift turntable
  request lacks playback/repeat fields.
- `crates/mold-server/src/routes.rs`: request/advertisement, MP4/GLB-only
  source contract, video bounds 240–2160 px and 1–60 FPS; turntable bounds
  240–2048 px, 1–30 FPS, 8–180 frames, 256 MiB frame budget.
- `crates/mold-inference/src/ltx_video/video_enc.rs`: RGB and RGBA GIF writers
  use uniform frame timing and avoid duplicate bounce endpoints.
- `crates/mold-inference/src/ltx2/media.rs` and
  `hunyuan3d/turntable.rs`: video and mesh paths share the GIF encoder.

## GIF timing contract

Use one additive `pause_ms` field for extra boundary dwell, independent of FPS.
The UI labels it **Pause between loops** for Loop and **Pause at turns** for
Bounce. Both apps allow 0–5000 ms in 10 ms increments, default 0. The server
advertises this range/step/default in an optional `gif_pause` block on
`GET /api/gallery/export-options`; validate it before decoding or rendering.
Reject unsupported non-GIF and geometry uses instead of silently ignoring them.

GIF stores frame delay in hundredths of a second; see the
[GIF89a Graphic Control Extension specification](https://www.w3.org/Graphics/GIF/spec-gif89a.txt).
Zero means **zero extra pause**, not zero frame duration. Keep normal positive
FPS-derived frame delays. Do not change speed or duplicate images to create a
hold. Add the pause to the existing boundary frame's encoded delay:

- Loop/Forever: last frame before restarting.
- Loop/Once: no additional terminal hold; expose no pause control because
  there is no loop boundary.
- Bounce/Forever: last forward frame and first frame at the other turn,
  once per cycle; retain the existing interior-only reverse traversal.
- Bounce/Once: last forward frame only; retain the final resting first frame.
- One-frame input: apply at most one dwell per repeating cycle; no double hold.

Omission keeps existing byte behavior. Explicit 0 must produce the same decoded
frames, delays and repeat metadata as omission. Old clients remain valid.
Clients hide the pause control and omit `pause_ms` when the selected machine
does not advertise it. Do not advertise this in a client fallback constant.
Hide/disable it for Loop/Once and omit the parked value when that combination,
APNG/WebP or a geometry format is selected; keep the draft for returning to a
compatible GIF choice. Reject positive pause on Loop/Once as ineffective;
accept omitted/explicit 0 without changing output. Reject malformed advertised
ranges client-side (non-finite, reversed, negative, invalid step/default), hiding
the new control rather than sending a guess. A capability/range change requires
revalidation of the parked draft before submit.
Mesh format authority remains `capabilities.mesh.export_formats`; fetch export
options from that same machine for timing support, rather than inheriting a
previous video's machine. Unknown advertised enum choices are skipped safely.

## Native architecture and UI

1. Add typed `VideoExportRequest` and playback/repeat enums in UI-free MoldClient.
   Decode host playback/repeat and optional pause capability in `ExportOptions`.
   Add a typed backend overload, using the existing authenticated POST route,
   extended conversion timeout, error handling and cancellation behavior.
   Retain existing format-only and mesh overloads for macOS compatibility.
2. Build a shared export availability policy from source filename/kind, trash
   state and the holding host. Only live MP4 offers video conversion; animated
   GIF/APNG/WebP originals remain shareable/saveable without an invalid MP4
   transcode offer. Only live GLB offers mesh conversion. Offline/removed hosts
   show a useful failure; never silently choose another machine for the request.
3. Extend the per-window `PrintActions` coordinator with an export presentation
   context capturing host ID, filename, export kind, capabilities and request.
   Library reads use the displayed lead copy. Generation results explicitly
   promote the render's host copy with `LibraryEntry.presented(onAnyOf:)` before
   opening export; `ResultPager` currently finds a merged row by `everyCopy`
   membership without promoting it. Capability reads, POST, asset downloads and
   reuse keys must all use that captured host plus filename identity. Never use
   the currently selected generation machine or a global capability cache.
   Revalidate the source/host at submit; serialize busy actions. Library tile,
   viewer More, and generation result More invoke this same path. Multi-selection
   keeps original sharing/saving; conversion starts as single-print only.
4. Native **Export…** uses a scrollable SwiftUI Form with labelled rows for
   Format, Playback, Repeat, Pause, Longest side, Frame rate and Destination.
   Offer only controls that apply. Use native pickers/steppers, semantic text,
   Dynamic Type and an explicit Cancel/Done path. Pause offers a labelled numeric
   field, slider and quick choices including **No pause (0 ms)**; the 10 ms step
   is precision, not a requirement to tap a stepper 500 times. Validate empty,
   non-finite, out-of-range and off-step edits before enabling Export. Loading/failure/retry live in
   the sheet. A failed host capability fetch cannot enable invented exports.
5. Separate geometry and turntable sheet bodies so video controls cannot leak
   into geometry requests. Reuse `MeshExportGeometry` capability/default policy
   and `MeshTurntableOptions` frame-budget logic. Expose turntable frame count,
   rate, dimension, GIF playback/repeat/pause and transparent background.
   Explain GIF's hard alpha edge versus APNG/WebP. Remember transparency using
   the existing native preference pattern; defaults otherwise match Tauri.
6. Destinations: **Share…**, **Save to Mold folder**, **Save to Files…**, and **Save to Photos** for
   verified supported image/video containers. Files uses the system document
   exporter. Add an app-visible Documents/Mold export folder with the required
   iOS document-sharing keys after verifying Documents contains only public
   exports; private state stays in Application Support/Keychain. Use atomic
   writes, explicit collision naming and a confirmation naming the saved file.
   Offer Share/Files/folder for original media and texture sidecars too, so the
   same destination vocabulary applies regardless of whether a conversion ran.
   A persistent folder export is user-owned and excluded from temporary-file
   cleanup. Geometry goes to Share/Files/folder, never Photos.
   Explicitly verify animation preservation in Photos; do not claim APNG/WebP
   support from their extensions alone. Tauri clips gain the existing folder
   option for destination consistency without changing existing share default.
7. Stage the response under a safe original-derived filename (`.png` for APNG),
   verify container identity, and retain caller-owned files through share/document
   exporter completion. Remove directories after completion, dismissal, errors,
   cancellation and scene teardown. Dismiss the options sheet before opening
   system delivery UI to avoid competing presentations. Repeated export actions
   cannot overwrite files currently owned by a system share sheet.
   Model the coordinator states explicitly: loading options → editing →
   converting → staged → delivering → finished/failed/cancelled. Use an operation
   identity to reject stale completions; Cancel cancels the network task and
   dismisses without allowing its late result to open another sheet. Transfer
   staged-file ownership only after the options sheet's actual dismissal. The
   existing `PrintSheets.onDismiss` calls `shareFinished()` indiscriminately;
   replace that with presentation-specific cleanup so closing options cannot
   delete the export about to be shared. Window teardown cancels tasks and
   cleans unclaimed files; system delivery dismissal cleans only its own files.
8. Add sidecar-asset download/share actions from the print's server-provided
   asset metadata and owning route. Extend shared wire/transport types only
   where absent: `GalleryPrint` currently has no `assets` field and MoldClient
   has no generation-asset transport. Decode the flat descriptor (`asset_id`,
   role, display_name, media_type, size_bytes, sha256, optional dimensions),
   authenticate `/api/gallery/assets/:filename/:asset_id`, and verify the
   advertised digest/size before delivery. Missing assets on older hosts means
   an empty list. Keep file identity, safe filename and cleanup rules identical.
   Audio and existing animated originals get original-file Share/Files coverage;
   new still conversion, video trimming and video speed editing are separate
   features, since Tauri does not currently offer them here.

**Existing format limits:** GIF has no audio, reduces colour precision and
quantizes frame time to 10 ms. The current decoder rounds source FPS and the
GIF writer uses a uniform rounded delay; Original FPS therefore does not
promise timestamp-exact timing, especially for fractional/VFR sources. Keep
the new pause feature independent of this pre-existing timing policy and
record total duration in UAT. Duration-preserving cadence is a separate change
if that evidence shows a user-visible problem. Do not introduce implicit
frame removal or retiming in a parity patch.

## Implementation sequence

### A. Host and encoder timing, then Tauri controls

- Failing RGB/RGBA tests first: numbered frames, traversal, per-frame delay,
  repeat extension, zero equivalence and single/two-frame edge cases.
- Add timing to the shared GIF writer and thread it through MP4 and mesh paths.
  Keep memory/decode limits and alpha/disposal unchanged.
- Route tests cover advertisement, omitted/zero/positive pause, invalid bounds,
  non-GIF rejection and video/mesh parity. Keep accepted input validation aligned
  with advertised 10 ms precision.
- Extend TS options and the shared dialog with capability-gated timing.
  Update every caller (desktop gallery/results, Tauri phone, web) to pass the
  selected host's capability. Correct turntable-specific preset limits; test
  source/format switches and preservation of 0 in serialization and native IPC.

### B. Native animation exports and delivery

- Failing MoldClient wire/availability tests first, including older host JSON,
  unknown values, missing/empty capability lists, snake_case, absent fields,
  filenames, no transparency on video and no geometry fields on animation.
- Add the backend request, native Form and coordinator routing.
- Native fake-backend tests cover original actions, request snapshot, selected
  host, offline/trash/animated-source gating, retries, cancellation, busy state,
  Photos denial recovery, container verification and file lifecycle.

### C. Remaining export gaps

- Native geometry and turntables, reuse shared mesh defaults/budget policy.
- Add Files/folder consistency and sidecar assets with the same delivery code.
- Add coverage for mesh formats, bounds/defaults, transparency budget changes,
  unknown formats and old hosts without geometry capability.

### D. UAT, documentation and delivery

- Use a deterministic read-only local HTTP fixture serving a small looping MP4,
  animated original, GLB, opaque/transparent GIF exports and sidecar asset.
  Record both request bodies and actual exported file metadata.
- Native Simulator UAT on iPhone and iPad: Library menu, viewer, generation
  result, Share/Files/Photos destinations, Cancel, retry, permission denial,
  source-host disconnect and window dismissal. Record Simulator evidence
  separately from any physical-device evidence.
- Sweep sheet controls at xSmall/Large/AX5 in light/dark with every row reachable,
  explicit dismissal, and no clipped controls or competing presentations.
- Tauri UAT exercises Loop/Forever/0 and Bounce with positive/zero pause,
  format switches, both destinations and owner-machine capability changes.
- Decode delivered GIF bytes and assert frame order/timing; visual playback
  alone is insufficient to prove zero extra dwell.
- Run focused Rust and Vue tests, shared Swift tests, native iOS unit/UI tests,
   iOS lint and macOS lint/build compatibility for changed shared code. Do not
  launch the macOS host-app tests locally.
- Update `apps/ios/docs/DESIGN.md`, `PLAN.md`, `IPHONE-UAT.md`, owning README,
  `.claude/rules/ios-native.md`, export API/website docs, Tauri parity docs and
  canonical CLI/MCP skill/docs only where the public timing contract affects
  them. Add one changelog fragment; never hand-edit generated output.
- Use conventional commits on `feat/ios-media-export-parity`. Publication,
  merge and TestFlight delivery are separate from local implementation and
  require the applicable user authorization.

## Independent review and disposition

The requested sub-agent independently checked the draft against the repository.
It found no blocking flaw in the GIF boundary timing semantics and rated the
plan ready after five amendments, all incorporated above:

| Priority | Finding | Disposition and required regression |
| --- | --- | --- |
| P1 | App-visible folder was conditional | Make Documents/Mold explicit, include originals/sidecars, distinguish persistent user files from staged files |
| P1 | Dismissal could delete the pending share output | Operation-scoped presentation/file ownership; test dismiss-then-share, Files cancel, stale completion and scene teardown |
| P2 | Merged result lead may differ from rendering host | Promote original render copy before capability read; two-host fixture with pause supported on only one host |
| P2 | Sidecar wire/transport work understated | Add typed asset descriptor and authenticated route; test snapshot round trips, missing/unsafe names, encoded components, unknown roles and old hosts |
| P2 | Pause stepper impractical; inactive values ambiguous | Numeric entry/slider/zero preset, parked-value omission, Loop/Once rejection and malformed/range-change tests |

The presentation regression suite must exercise both `RootView`'s normal
`PrintSheets` owner and `LinkedPrint`'s viewer owner, including export from a
deep-linked print. The final independent review recommendation is encoder and
route tests first, then delivery ownership, then native Forms; phases A/B follow
that order. No code execution/UAT result is claimed by this planning review.

## Completion checklist

- [x] Independent review incorporated with an explicit disposition of findings.
- [x] Native MP4 export matches Tauri formats, playback, repeat, size and FPS.
- [x] Both apps export Loop/Forever/0 with no added boundary dwell.
- [x] Both apps offer identical pause semantics and preserve explicit 0.
- [x] Mesh geometry and turntable controls are present with correct bounds.
- [x] Original files and sidecar assets have working native delivery paths.
- [x] Older hosts remain usable without ineffective new controls.
- [x] Every native entry point uses the same coordinator and policy.
- [ ] Encoded-output tests, native UAT, accessibility and focused checks pass.
- [ ] Documentation and changelog describe only verified shipped behavior.
