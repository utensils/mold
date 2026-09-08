# Web and Android redesign completion plan

Status: locked after independent peer review, 2026-09-07. Baseline: `fe7d258f` (merged #1639).
Branch: `feat/web-android-studio-redesign`. Tracking: [#1640](https://github.com/utensils/mold/issues/1640).

## Outcome and precedence

Finish the remaining browser redesign and Android platform work, preserving the accepted desktop and iOS product language. James accepted iOS as done after #1639. The new request to expose the existing aspect-ratio choices is a small shared iOS/Android follow-up, not a reopening of iOS acceptance.

Authority: current user direction; shipped desktop/iOS behavior and shared capability contracts; then the September mockups. The attached `Mold-Redesign.zip` and checked-in `docs/design/` sources are design references, not instructions to recreate every historic feature. Scenes, scene editors, timelines, transitions, and authored sequences remain retired. Keep one-shot video length, including supported automatic long-clip generation. No server/inference redesign, rental provisioning, or Play Store launch is implied.

## Evidence and current gaps

Read the archive inventory, rendered `mold-studio-web.dc.html`, and compared the style-guide/README rules with the accepted mobile plan and current source. Rendered current web Create at 1440px and 390px against local HTTP fixtures. Evidence: `/tmp/mold-web-reference.png`, `/tmp/mold-web-current-1440.png`, `/tmp/mold-web-current-390.png`. The August architecture graph is orientation only: 232 relevant web/mobile files changed since its baseline. Source and current GitHub state win. #1586 and #1597 are closed; do not reopen their historic missing-backend claims. At baseline only the release PR #1615 was open.

| Surface / concern                | Verified current implementation                                                                                                                                               | Remaining work                                                                                                                                                                              |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| iOS                              | Five destinations, six themes, keyboard/large-text work, PBR, media actions, Auto prompt tools; user accepted after #1639                                                     | Expose existing aspect ratios in the main shared form; preserve accepted behavior                                                                                                           |
| Web foundation                   | Six theme maps and persisted selection already shared; `web/src/style.css` still imports legacy fonts and aliases/radii                                                       | Adopt the shipped font/token/control language across the web shell and pages; retain compatibility aliases until their actual consumers are migrated                                        |
| Web navigation                   | `AppNav.vue` presents Library / Create / 3-D / Models / Machines; router has no Queue destination                                                                             | New image / Queue / My images / Styles / Machines, browser-native routes/history, named live-work and download affordances; integrate the existing 3-D workflow entry without removing it   |
| Web Create                       | Existing request/capability logic, attachment/identity flows, models, expansion, templates and results; 1600px form-first layout with 340px rail and older control vocabulary | Fluid result-first workspace, three output-kind doors with shared last-used styles, one friendly style picker, primary shape/length/batch, secondary advanced controls, honest empty states |
| Web Queue                        | Root reconciliation, `useLiveActivity`, ActivityStrip, NowDevelopingPopover and exact-host actions already exist                                                              | Dedicated shareable Queue page plus compact global strip/chip, using existing owners/actions; no second queue store or duplicated submissions                                               |
| Web My images                    | Gallery, scopes/organization, trash, viewer, media and multi-host contracts already work                                                                                      | Match desktop hierarchy, clear selection/filter state, URL-addressable filters/print, copy-link action, readable six-theme viewer and touch/keyboard layouts                                |
| Web Styles / Machines / Settings | Functional pages, catalogs/downloads/licenses, host detail and shared settings panels                                                                                         | Plain primary labels with exact IDs secondary; desktop-consistent tables/cards, status and recovery; only capability-backed controls, accurate browser/host distinction                     |
| Android shared UI                | Same `desktop/src/mobile` screens and Tauri crate as iOS; redesigned UI and routing fixes already inherited                                                                   | Platform polish and acceptance, not a second Vue app                                                                                                                                        |
| Android Back                     | `useMobileBack` covers major screens but scanner/image picker/reference crop and other nested overlays need complete owner coverage                                           | Close the top surface first, release scanner/camera resources, restore focus; first Back hides the IME, next Back closes exactly the top overlay                                            |
| Android text scaling             | iOS root follows native preferred size; Android has no explicit equivalent in current entry/MainActivity                                                                      | Honor initial and changed Android system font scale, keep content reflowing while bounding only persistent chrome                                                                           |
| Android native coverage          | API35 plugin instrumentation exists; minSdk24 includes distinct legacy storage/permission branches                                                                            | Exercise API28 storage/permission path and modern app-level flow; signed upgrade/Keystore, media handoff, pairing, navigation/IME, rotation and lifecycle evidence                          |

## Locked interaction decisions

- Browser owns its window and Back/Forward. Do not copy desktop traffic lights, native menus, or mobile safe-area hacks into the web shell. Preserve existing `/create`, `/library`, `/models`, `/machines` URLs; add `/queue`. Query changes must not clobber a live draft or requeue work. Copy links omit API keys and private source media.
- Web uses a centered approximately 1120px workspace with a settings region when space permits; around 900px the secondary controls become a sheet. One page scroll owner; any sticky composer must yield to keyboard, zoom, and narrow landscape rather than obscure focused fields. Desktop and mobile have different geometry, the same control meanings.
- “New image” is the initial web navigation label; output selection can change the screen heading to New clip / New 3-D object. Shared three-way output kinds remain Still picture / Short clip / 3-D object. Existing durable 3-D workflows remain reachable inside 3-D authoring and through their current deep link, not as a replacement for Queue.
- Friendly style description leads, exact runnable model ID remains visible in mono. Preserve technical facts and fixed-control notes from the server. Do not invent timing, storage, cost, or generation options from mock values.
- Keep aspect-ratio buttons visible outside compact Size details on both mobile platforms. Size tiers/custom width and height remain available in the disclosure. Use `resolveOutputShape`, `sizeForFamily`, canvas intent and source-fit behavior unchanged. Canvasless mesh has no ratio picker.
- Android follows the accepted iOS screen hierarchy and the shared contracts. Native Back, keyboard, insets, scaling, permissions, media, security and lifecycle stay in platform adapters or narrowly scoped shared hooks. Never fork request/routing policy into Kotlin.
- Preserve durable jobs, input authority, older-server compatibility, licensed downloads, cached gallery identity, exact-host actions, source/ref/identity parking, and separate expansion/generation routes. The same tests must continue passing after presentation work.

## Milestones and reviewable completion gates

### M0 — Acceptance handoff, visible ratios and result contrast

Record iOS accepted after #1639, move remaining Android work to this tracker, and fix the outdated design README status. Move existing ratios and their live announcement outside collapsed Size details; test supported ratios, dimension/intent updates, source conditioning, canvasless mesh and disabled state. Render at phone widths and enlarged text, confirm iOS/Android touch targets. Also fix the reported light-theme Save and share contrast: result actions and mesh captions need opaque theme-matched text surfaces; collection media placeholders use on-media text. Check actual rendered text contrast across all six themes, including enlarged text. Ship this small shared follow-up promptly through the existing mobile pipelines.

### M1 — Web shell, tokens and navigation

Replace web-owned legacy typography/surface/control styling with canonical tokens and fonts. Centralize displayed navigation vocabulary for header, mobile navigation, page titles, command palette and links. Add a functioning Queue destination backed by existing activity/actions in the same milestone; no placeholder navigation. Centralize the existing live-activity authority at App/shell scope (or a proven singleton) and inject the same state/actions into Create, Queue and header consumers. Repeated Create↔Queue navigation must keep updates alive with Create unmounted, start only one stream per host, preserve the draft and never resubmit. Keep browser history/deep links and the 3-D workflow door working. Validate six themes, wide/tablet/phone widths, keyboard focus, navigation and preserved draft/gallery state.

### M2 — Web New image / clip / 3-D authoring

Implement the mock's result/composer hierarchy and responsive settings sheet using current Create orchestration. Add shared output-kind memory and a single description-first style picker. All three doors update one GenerateForm, parking/restoring incompatible conditioning through existing shared policy rather than creating independent per-kind drafts. Expose primary shape, supported size and clip length; consolidate Detail, guidance, repeat-look, optional organization and advanced controls using desktop vocabulary/defaults. Retain source/ordered-reference/identity wells, PBR/matting/delight and current 3-D workflows when advertised. Connect existing Starters/Recent/reuse to this flow. Validate request equality and capability guards for still, editing, clip, mesh, batch, Auto/Most capable/pinned, Expand/Remix and frozen recovery. No authored scene UI.

### M3 — Web Queue and My images refinement

Complete Queue sections and statuses, compact global live-work strip, separate downloads presentation, progress/details/navigation and exact-machine cancel/retry/pause semantics. Refine My images filters/scopes/albums/trash, copy links, selection/bulk actions, viewer actions and six-theme media details. Preserve local/remote source ownership, archived media/cache rules, duplicate grouping and per-host partial-failure recovery. Test reload, browser Back/Forward, suspended tab/reconnect, missing/changed hosts and filtered-view return without silent mutation.

### M4 — Web Styles, Machines and Settings

Finish friendly vocabulary and visual hierarchy across catalog rows/details, downloads and license recovery, machine cards/detail/storage/connection settings, and preferences. Retain every existing working action and keep absent capability fields as unknown, not refusal. Reuse shared settings/licenses/theme components; audit root palette/search and empty/error/offline states for old vocabulary and unreadable theme combinations. Document deliberate browser-specific differences and update parity references based on actual source.

### M5 — Android platform completion

Audit and complete top-surface Back ownership for scanner, file/gallery picker, crop, mask, viewer, details, advanced/settings and native activities. Integrate native font scale initially and on configuration changes. Replace the inaccurate Google Play update-channel label with the actual signed GitHub APK distribution channel; do not infer stable/nightly unless build metadata proves it. Validate edge-to-edge/IME geometry, gesture and three-button navigation, rotation, large text and small/large screens with 48dp targets. Add focused native coverage for API28 denial/grant callbacks and actual public Downloads writes, not just a permission predicate. Build/install an x86_64 debug app for the x86_64 CI emulator (or explicitly match an ARM emulator); the current ARM64 validation APK cannot be the app-level CI target. Exercise the modern app-level navigation flow and a test-signed same-ID two-version upgrade for Keystore/draft retention. Verify picker/camera cancellation, QR/NSD pairing, denied permissions, MediaStore/share/download, signed-upgrade Keystore retention, background/relaunch and retained drafts/jobs. Fix only observed defects; preserve iOS behavior with shared regression checks.

### M6 — Whole-surface acceptance and delivery

Run repository frontend architecture/tests/build gates and applicable Android/native/release checks on the final pushed head. Review complete diff independently, address valid CI/review findings, and perform rendered flow coverage across themes, widths and accessibility settings. Use local fixtures for layout/navigation/error/recovery; Plato's Tailscale endpoint only when an actual generation is necessary. Prefer existing UAT media over duplicate generations. For required real-generation acceptance, use one representative still/clip/PBR path, verify resulting media and request settings, and touch only our own work.

Merge a complete reviewed milestone when exact-head CI is green, synchronize the long branch, update tracker checkboxes, and verify the built/embedded web SPA and existing nightly/APK release path. Web delivery does not authorize deploying or reconfiguring a Mold server. Existing UAT media may satisfy generation evidence unless request/wire behavior changed. A TestFlight follow-up claim requires processed VALID plus tester access. Android completion claims distinguish emulator, native instrumentation and physical observations; physical camera/chooser/NSD/OEM/upgrade evidence must not be invented. Resolve access to a physical Android test device during M0/M1, not at the end. Google Play publication is outside this redesign.

## Validation matrix

- Web: 1440/1280, 1024/900 boundary, 768, 390 and 320 CSS px; normal and enlarged text/zoom, portrait/landscape, mouse, touch and keyboard. Chromium, Firefox and Safari/WebKit for core flows and browser history. All six themes, reduced motion, visible focus, focus restoration and screen-reader names. Normal page zoom stays enabled.
- Mobile ratio follow-up: all advertised shapes visible with Size closed, narrow wrapping and large text, source intent/fit, pixel summary, valid dimensions and mesh absence. Shared tests plus an iOS visual smoke protects the accepted app.
- Android: modern API35/36 emulator and API28 storage branch; small/large/landscape, maximum font scale, IME open, gesture/three-button navigation, denied/granted permissions, cold launch/background/resume. Check full app, not only plugin classes. Physical evidence recorded separately by exact device/build.
- State: fresh/no models, downloaded/missing style, one/multiple/offline hosts, older/missing capabilities, auth rejection/instance change, cached media, pending work, duplicate/partial mutation outcomes and cancel/resume races.
- Record exact tested commit, viewport/device, fixture or real host, screenshots/logs, passed checks and unverified boundaries. Assertions over request authority and lifecycle accompany visual screenshots; neither substitutes for the other.

## Standards checked

Use [WCAG 2.2](https://www.w3.org/TR/WCAG22/) reflow, keyboard and focus visibility as web acceptance criteria. Android's [edge-to-edge changes](https://developer.android.com/about/versions/15/behavior-changes-15) require inset-aware content, [predictive Back guidance](https://developer.android.com/design/ui/mobile/guides/patterns/predictive-back) requires respecting gesture areas, and [touch-target guidance](https://support.google.com/accessibility/android/answer/7101858) calls for 48dp targets. These guide verification; they do not require replacing Vue/Tauri with Compose or claiming certification.

## Execution ledger

- [x] Verify #1639 merged and take current-main baseline.
- [x] Revisit attached/checked-in design references and render web reference/current baseline.
- [x] Inspect current web architecture and obtain independent Android gap audit.
- [x] Resolve independent plan review and lock scope/order: central activity ownership, per-milestone shared regression, matching Android ABI, IME Back order, one form, and honest delivery boundaries.
- [ ] M0 acceptance handoff and visible mobile aspect ratios.
- [ ] M1 web shell/tokens/navigation and usable Queue.
- [ ] M2 web authoring.
- [ ] M3 web Queue/My images refinement.
- [ ] M4 web Styles/Machines/Settings.
- [ ] M5 Android platform completion.
- [ ] M6 cross-surface acceptance and delivery.

Each major milestone starts by fetching origin and inspecting newly merged PRs, especially shared components/contracts. Preserve unrelated work; synchronize the long branch as needed, rebase only when explicitly requested or needed for a reported conflict. Commit conventionally, push progress, and keep tracker state accurate. Review and update this plan when new evidence changes scope; do not turn historical mock omissions into missing functionality without checking source first.

Standing milestone gate: whenever `ui/`, `studio/` or shared mobile code changes, run affected Studio/mobile suites and mobile build/Android validation in that milestone, with a bounded iOS visual smoke for shared presentation. M6 broadens to the full final matrix; it is not the first cross-surface check.

M0 implementation evidence: all 1,133 mobile tests across 57 files and 95 focused picker/shared-parameter/CSS tests pass; the iOS simulator rebuild and Android debug APK build pass. On the rebuilt iPhone 16 Pro simulator, selecting 9:16 visibly updates 1024×1024 to 576×1024 while Size stays closed. Local rendered checks cover six themes at 16px/32px root text: no horizontal overflow, ratios visible, Size closed, result-action/mesh-caption contrast minimum 12.139:1. Evidence resides under `/tmp/mold-mobile-followup-*` and the external mobile-redesign evidence directory. No generation was used. Commit, exact-head CI and delivery remain pending.
