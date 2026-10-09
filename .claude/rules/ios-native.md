---
paths:
  - "apps/ios/**"
  - "apps/shared/**"
---

# Mold Studio Companion (apps/ios) and the shared Swift packages (apps/shared)

**Temporary build-first delivery (2026-10-06, owner request).** Hosted native
accessibility CI is physically commented out in `ios-native.yml`; its matrix and
local audits remain available unchanged. The CI lane compiles the native app
and shared packages for iOS Simulator; TestFlight separately archives and signs
for devices. Restore the commented lint/unit/package and audit job blocks when the
owner resumes test gating.

**What it is.** The native SwiftUI iPhone/iPad app: `io.utensils.mold.companion`, Home Screen label "Mold Studio", iOS 26+, remote-only. It is the macOS Mold Studio app's (`apps/macos`) little sibling and installs BESIDE the Tauri iPhone app (`apps/mobile`, `com.utensils.mold`), never replacing it. `apps/ios/docs/DESIGN.md` is the binding spec, `apps/ios/docs/PLAN.md` the milestone plan. Follow the Mac app, never the Tauri app's idioms (custom tab bars, toasts, fixed px sizes).

**Shared code.** `apps/shared/Packages/MoldClient` (wire + transport, Foundation only; `lint-layers` bans SwiftUI/AppKit/UIKit), `MoldStyle` (tokens) and `MoldMesh` (the Metal mesh renderer; its shaders load from the resource bundle's `default.metallib` in Xcode builds -- Xcode compiles a package `.metal` even when declared `.copy` -- and from the copied source under `swift build`, which never compiles it; `MeshMetalStack.shaderLibrary` tries both, and `swift test` plus `make packages-test` cover one each) serve BOTH apps; `platforms` declares macOS 26 and iOS 26. A change there must keep `make -C apps/macos lint` green and compile for iOS (`xcodebuild -scheme MoldClient -destination 'generic/platform=iOS Simulator' build` inside the package). Mac-only API goes behind `#if os(macOS)` (as `MoldHome` is). Credentials go through `CredentialStore`, keyed by host UUID: the Mac's `SecretStore` file on macOS, the Keychain on iOS; an empty key is a clear. Decision logic longer than ~10 lines that both apps need lands in MoldClient with its test, not twice in two app targets.

**Never register `mold://`.** The Tauri app owns it and iOS resolves a shared scheme arbitrarily. The companion's scheme is `moldstudio://`. Pairing QRs are universal links, `https://utensils.io/mold/pair#<form-encoded payload>` (the payload in the FRAGMENT, never the query, so it never reaches the web server): the app claims `applinks:utensils.io` (the `apple-app-site-association` lives in `utensils/utensils.github.io` `public/.well-known/`), an opened link lands in `PairingLinkSheet` and is claimed only when the person taps Pair, and `utensils.io/mold/pair` (`website/pair.md`) is the page for a phone without the app. Producers and parsers on both sides (`studio/api/pairing.ts`, MoldClient's `MobilePairingPayload`) are held to `studio/api/pairing.fixtures.json`; change the format there first. Older `mold://pair?...` codes still parse. Claims go through `POST /api/pairing/claim`.

**Library offline and thumbnails.** Machines serve thumbnails at `size=256` or `512` only (anything else is a 422 and an empty tile): `ThumbnailLoader.bucket` never asks for more; the viewer reads the full print (`original(for:)`), not the thumbnail route. Listings are saved per machine (`LibrarySnapshots`, the app's strategy-free coders) and restored before any machine is asked; pictures live in Application Support/offline (excluded from backup), bounded by `OfflineLimit`. The grid's pinch is `PinchRecognizer` (UIKit), not `MagnifyGesture`.

**Library.** Every edit goes through `LibraryStore.apply` -> `PrintEdit.plan` over `everyCopy` -> the shared `MutationOutbox` (optimistic, retried with the same operation id, re-read on give-up); never a hand-rolled request per print. A recipe's output kind is `GenerationRecipe.makes` (GLB -> 3-D, time axis -> clip), never a family list.

**Dynamic Type is a gate, not a goal.** Text styles only (`make lint` bans `.font(.system(size:`, `.custom`, `minimumScaleFactor`). Label/value rows ask `RowAxis.for(_:)` (stack from AX1). Secondary text is `.secondaryText`, never `.secondary` (the system colour is ~4.4:1 on white and fails the audit); the one filled button per screen uses `.prominentAction()` (`ProminentFill`). `make uitest` runs `performAccessibilityAudit` on every destination at xSmall, Large and AX5 in light AND dark, on an iPhone, an iPhone SE when one is installed, and an iPad; CI (`ios-native.yml`) sweeps the iPhone only, because the hosted runner times iPad audits out, so iPad work needs a local `make uitest` before merge; it may skip contrast only for elements under the bottom chrome (the tab bar, its scroll-edge dimming band, or the composer -- chrome membership by tree descent, never frame containment) or outside a presented sheet / the open iPad sidebar (their dimmed backdrop), plus clipped composer scroll content only when the scroll-coverage pass requires every exported label/control to become fully visible and be contrast-audited, clipping only for the system search field and UIKit's single-line sidebar-row PREDICTION (the AX5 pass audits the open sidebar for real clipping), and Dynamic Type only for Settings' lazily laid-out Form and the Library grid's `day-header`s at xSmall/Large (AX5 proves them) and the `tile-badge`s. XCUITest lists views even under `accessibilityHidden`/`children: .ignore`, so an exemption keys on an `accessibilityIdentifier`. Models and the iPad sidebar's sections stay out of the floating tab bar (`.defaultVisibility(.hidden, for: .tabBar)`): this keeps the bar within the available width at large text sizes, but does not resolve the separate auditor-triggered UIKit pagination loop. On iPad, audit the Settings Form through its native sidebar page: the auditor's private text-size cycling over the Settings sheet can loop UIKit floating-tab pagination even though ordinary system text-size changes stay responsive. Keep all audit types on the Form, and separately test sheet Done/Add a Machine transitions; this is equivalent Form coverage, not sheet-chrome coverage. A new screen joins that audit. Never edit compiled sources while `make uitest` runs: every pass recompiles the working tree, so a half-written file fails the passes after it.

**Extensions.** Widgets read the App Group snapshot only (no MoldClient, no URLSession, no Keychain); the Share extension stages into the App Group and never networks. `make lint` enforces both. Because nothing outside the app reads a key, keys live in the app's OWN Keychain group (`KeychainCredentialStore`, service `io.utensils.mold.companion.remote-api-key`, `AfterFirstUnlockThisDeviceOnly`) -- there is no shared keychain-access-group entitlement.

**Stores and windows.** `CompanionStores` is the composition root; `CompanionStores+Backend.swift` is the ONLY file that may write `HTTPBackend(` or `HTTPBackend.claimPairing` (`make lint`), so every store test injects `MoldClientTesting.FakeBackend`. Stores are shared by every window; where a window IS (tab, sheets) is its own `AppRouter`, never shared. Event streams run only in the foreground (`ConnectionSupervisor`): stop on background, reconcile on return. A pairing code for a machine already listed (same instance or address) re-keys it, never adds a duplicate.

**Away from the app.** The server has no push. `ActivityCoordinator` drives the Live Activity from `GenerateController.run` (projection in `ActivityProjection`, pure and tested), `Notifier` posts once per batch and never in the foreground, and `CompanionStores+Background` reconciles `PendingLedger` from `BGAppRefreshTask` and on every return to the foreground. `StopRenderIntent` runs in the APP process through its static handler; the widget only draws the button. `Activity` is not Sendable: look it up again on the awaiting side (`ActivityCoordinator.find`). Widgets read `WidgetSnapshot` (written by `WidgetSnapshotWriter`, timelines reloaded only on change); the Share extension stages through `ShareInbox.stage` (ImageIO thumbnail, never a whole decode -- `ShareInboxTests` holds the 48 MP peak-memory ceiling). Links are `DeepLink` (`moldstudio://print|queue|generate`). Types in `Sources/Shared` are `nonisolated`: the extensions and ActivityKit use them off the main actor.

**Queue and Models.** Reorders go through `QueueOrder` (the machine's queued-only index), transfers through `TransferPlan`, held rows through `QueueHold`; downloads are reduced by MoldClient's `DownloadBoard`, which the Mac's `DownloadStore` also uses -- change it there, with its test, never in one app.

**TestFlight.** `testflight-ios-native.yml` uploads after `iOS native app` passes on main, gated on the repository variable `COMPANION_TESTFLIGHT=true` -- it skips with a notice until the owner creates the App Store Connect record (no API exists for that). It is NOT part of the release tag gate.

Root `Cargo.toml` changes trigger the native pipeline because its workspace
version supplies the marketing version. They run lint/unit builds without
repeating unchanged Swift UI audits; mixed UI changes still require the matrix.
Library audit settling allows 15 seconds for hosted AX snapshots while retaining
exact frame equality and all containment checks.

**Commands.** `make -C apps/ios gen|build|test|uitest|lint` (devshell: `companion-*`); pass `BUILD=/Volumes/ExternalStorage/...` locally to keep DerivedData off the internal disk. Simulator tests do not touch the desktop. Never run the MAC app's `make test` locally unasked: its host-app bundle launches Mold Studio on the user's desktop; CI (`macos-native.yml`) runs it. CI for this app: `.github/workflows/ios-native.yml`.

**Development loop.** `nix develop -c companion-dev` (`scripts/companion.sh dev`) watches iOS and shared Swift sources and rebuilds/relaunches after edits. `companion-run` launches once; `companion-build` builds only. All helpers accept `SIM=<UDID>` and `BUILD=<directory>`. Generated projects/build output are excluded from the watch set. `scripts/tests/companion-helper.sh` checks argument forwarding from outside the repository.

**Composer and playback.** Keep the prompt in one stable scroll-view hierarchy, capped using the window's available geometry (not `UIScreen.main`). Model selection is a labeled wrapping button that opens a searchable sheet; settle selection when model lists arrive, not only when reachability changes. Playback configures `.playback` / `.moviePlayback` without eagerly activating audio while browsing, and cancels ticket replacement when its task ends. Embed `AVPlayerViewController` for paged playback and let it own video gestures. Run contrast separately before Dynamic Type audits, which temporarily resize the hierarchy. `GenerationInteractionTests` exercises saved-machine layouts and prompt input in Simulator.

**Phone navigation and gallery performance.** Idle Generate on iPhone is one scrolling form with visible Kind and Machine choices; a render or result uses the canvas. At large text, the option chips collapse before they can make Generate wrap, and the action takes the full width. Give the form enough bottom travel to move Options above the pinned action. Model search needs a visible close action with the keyboard up. Keep the phone viewer navigation bar visible so a still image cannot trap the user after a chrome toggle. The Library exposes shelves and media types through one inline navigation title menu on phone; Settings is reachable from Generate, Library and Machines. Cache clearing includes saved listings and images, waits for pending listing writes, and drops offline-only rows. `LibraryView` uses `LibraryShowingCache` keyed by revision and query, and `LibraryGridProjectionCache` derives host badges, favorites and indexed print positions once per revision/query. Observe viewport geometry without invalidating view state; emit tile frames only during selection. Freeze the exact raw content offset before viewer navigation and restore it on dismissal, including partially visible tiles. Bound viewer pages to the selected print and two neighbors on each side; a different shelf or query gives the grid a new identity and top offset. Only the selected viewer clip may autoplay; pause it on page deselection and dismissal.

**Populated iPhone regressions.** Options uses separate form rows, never the composer's horizontal ChipRow inside the sheet. Model search has an explicit no-match state. Restore draft kind from the saved recipe after profiles arrive, keeping authored values. Observe raw model/reachability changes: a picture-filtered model list cannot wake a pending clip draft. Hide the main tab bar in PrintViewer so it cannot cover the bottom actions. `PopulatedGenerationTests` uses a loopback read-only fixture, not a live inference server; the iPhone UAT record is `apps/ios/docs/IPHONE-UAT.md`.

**Media exports.** `mediaFile` returns an extensionless caller-owned temporary file. Stage it under its safe original filename before handing it to UIKit/Photos; remove the export directory on share dismissal or after copy/save, including partial failures. Info always has an explicit Done action.

**Unavailable data.** Pending draft restoration must not fall back to the first responding machine; an explicit kind/model choice cancels it. Generate distinguishes offline machines from empty inventories. Queue marks unanswered/failed reads unavailable and reloads as machines reconnect. A missing export source fails the entire selection, never compresses the file list before Photos pairs it with entries.

**Notification presentation and refresh cancellation.** Use the system notification banner (the app bundle supplies AppIcon), concise completion copy, and no prompt or media attachment. Preserve print/queue links and per-batch deduplication. Notification delegates implement the explicit completion-handler API: extract payload strings off-actor, then route and invoke completion on MainActor. The nonisolated async delegate bridge completes on a cooperative thread and crashes UIKit notification activation/state restoration ("Call must be made on main thread"). Test the Objective-C callback from a background queue, including its completion thread, and actual Notification Center taps after backgrounding and cold launch; an onOpenURL-only test misses this boundary. Dismissed and unknown actions complete without navigation. A cancelled Library listing is routine lifecycle/event coalescing, not a host failure; keep loaded prints and never report it in FailureBanner. Genuine listing failures remain visible.

**Generation option parity.** BoundaryFramePolicy receives the recipe when attaching an endpoint: only the first frame re-arms source-driven closest-aspect sizing; the closing frame never changes the canvas. Test the real attachment path for Wan and H3, plus the iOS top-level aspect menu selection. Aspect menu icons draw the ratio of the offered dimensions rather than a generic rectangle. SourceFitOptions and SourceFitRender in MoldClient are shared native authorities. iOS fits a submission snapshot from original source bytes before admission; its painted mask is in source space and must follow the same transform (macOS masks remain canvas-space). Source-driven, parked reference-only and continuation inputs bypass fitting. Random seeds are the untouched default; Fixed is explicit and Reset restores Random plus centered crop-fill.

**Hidden collections.** Library queries include host-local hidden membership ids and check every merged copy, so a local lead cannot expose a hidden remote member. Hidden shelves remain directly browsable. Library View Options offers Manage Collections on phone and iPad; its Hide from All Prints toggle updates every machine holding that shelf and surfaces failures through HostStore.

**Library status notices.** Failure/offline notices use a top safe-area inset, never an overlay over date headings and prints. The saved-gallery notice uses primary text on an opaque semantic surface: material failed small-text contrast even with explicit primary text. Contrast coverage includes the populated date heading and offline notice at every audited text size; Library and Search share this layout. Keep pinned offline status independent of machine-name lengths: a compact count and saved-print context opens scrollable full-name details, with host IDs as row identity and a visible Done action. Library owns offline-details presentation and hides its presenting tab chrome while open, using the same visibility preference as Collections; restore it on Done. Test the actual native floating-tab owner on iPad rather than assuming it exports as TabBar. The details sheet uses an inline native title, an opaque semantic navigation background, a semantic secondary section heading and concise saved-print context. Its scalable Done action sits in an opaque bottom safe-area inset with a minimum 44-point target; the scrollable text viewport ends above that footer. Require the full button to fit its footer and presented sheet. Audit requested-size bounds and contrast before private Dynamic Type sweeps mutate the native List. Use a fresh native details presentation for each subsequent private sweep, requery its owner and prove the original settled paragraph size and title/action heights are restored. Preserve diagnostics and failure for audit issues without an element. Details contrast covers the fully visible native title, explanatory text, heading, every per-ID machine name and Done; require an exhaustive semantic inventory so new labels or actions cannot escape coverage. Do not sample a partly occluded lazy row beneath the pinned footer as though its full text were visible. At AX5, actual controls must fit the effective Library viewport; retained photos must fit when possible, or scroll to expose the maximum available image area with an unobscured hit region and a verified print action; never clamp Dynamic Type or truncate names to make the notice fit. Decorative tile badges stay one line within the proposed tile width; crowded symbol-and-text badges may fall back to the native symbol, while tile speech and Info retain complete metadata.

**Live Activity appearance.** Use `.activityBackgroundTint(nil)` for ActivityKit’s system material; never resolve a custom UIKit `systemBackground` separately from the Lock Screen foreground. Preserve semantic text and the system color scheme. Check running and stale cards in both appearances.

**Live Activity layout.** Keep the Lock Screen card within 160 pt including padding; `ActivityCard` offers a compact fallback before clipping status or Stop. The step figure and machine name are separate labels (strip only the exact legacy machine suffix in `ActivityCardContent`, without changing the ActivityKit wire payload). Preserve the stale refresh message, terminal print deep link, and 44 pt Stop target. `--live-activity-fixture` is Debug-only UAT, never a render or a distribution feature.

**UI test reports.** `make uitest` makes one full pass, then retries only identified failed methods once in a fresh `xcodebuild` process. Read public `xcresulttool get test-results tests` JSON; incomplete/unsupported failures never fall back to rerunning a complete target. Before accepting a retry, require exactly every requested method to appear as Passed in its result bundle. Preserve original and retry logs/result bundles under `build/UITestResults`, uploaded even on failure. CI partitions all XCTestCase classes into five groups (app, references, Library, Library interactions and media exports), each in light and dark; keep GenerationInteractionTests and PopulatedGenerationTests with ShellAccessibilityTests because Shell audits use the persisted machine fixture. Local `make uitest` defaults to the complete target and both appearances. The Git audit classifier only skips app audits for explicit Widget Swift/inert documentation/static routing or branding paths. App/shared/UI-test/build inputs, unknown paths, dispatch, and unavailable or stale-base diffs require full audits. The classifier job must succeed and explicitly emit `audit=false` to skip; classifier failure keeps the workflow red and runs all audits. Lint/unit/shared-package checks always run and compile Widgets. Widget-only changes require separate Lock Screen/Dynamic Island appearance UAT because app audits do not render extensions. The routing contract checks class coverage, while `scripts/tests/ios-uitest-runner.sh` and `scripts/tests/ios-uitest-retry.py` exercise bounded retries, exact selections, and failure propagation.

**Library selection and media filters.** Hide the presenting tab chrome while the Collections modal is open and restore automatic visibility on dismissal; this keeps iPad modal layout stable during Dynamic Type audits without dropping audit types. A one-finger horizontal start in Select mode acquires a UIKit range-selection pan; vertical starts remain native scrolling/refresh. Apply a baseline range, choosing select/deselect from the start tile, so reversal restores untouched selections. Cancel edge-scroll work on end, disable and coordinator destruction. `LibraryMediaFilter` in MoldClient applies the same kind tokens for both native apps without dropping other search tokens. The Mac toolbar offers the filter and preserves Command/Shift selection and external media drags; visible pointer selections must not recenter the grid.

Live Activity identity uses the bundled Mold logo on the Lock Screen and in every Dynamic Island state, including completion and failure; status remains available through text and accessibility labels. System notification banners use the app icon.

**Library previews and notification icons.** UIKit hosts context-menu and drag previews outside the grid environment; explicitly inject ThumbnailLoader into both preview roots. Exercise populated long-press menus and source-library selection on Simulator. The AppIcon catalog supplies the same authored image for Any and Dark appearances to preserve notification logo colors; Notification Center card backgrounds remain system controlled.

Library source selection prefers the merged print copy on a currently reachable machine (`presented(onAnyOf:)`), preserving that copy's filename and host together.

**Curated discovery and labels.** Native model headlines prefer the server's
additive `display_name`, then legacy descriptions; IDs remain request identity.
Discover searches the `/api/models` manifest inventory even without a live
catalog. Provider tokens are `hf` and `civitai`; logos use the same authored marks
as `ui/components/SourceGlyph.vue`. Curated Get always installs the exact model
ID through `startDownload`, not an aggregate repository catalog ID.

**Durable conditioning and queue inputs.** Reuse always probes retained media.
Restore source and paired source-space mask atomically before fitting, respecting
explicit attachments. Remaining private roles stay visibly disclosed until
cleared or a model/kind/recipe/reset supersedes them. Every submit mints fresh
same-host authority or relays bounded bytes through MoldClient's shared hydration
helper. A removed visible source/mask cannot be rehydrated from hidden authority.
Queue input thumbnails use the authenticated sealed-media route, never provenance
filenames or denoise previews. Cache by host, instance and job, fence late replies
against live row membership, and purge removed jobs/hosts.

**Model unloading.** Installed Models exposes a Server Memory section with visible per-model Unload and Unload All Models. Residency uses `isLoaded` independently of downloaded metadata. All-model unload sends nil model and GPU to the selected host. Coordinate load/unload/delete with a per-host in-flight guard held through refresh, preserve installed files, and display server refusals. Offline controls are disabled. Per-machine Models in regular-width layouts hides inherited floating tab chrome and restores it on Back; direct sidebar Models remains unchanged. Installed-family headings use ordinary section rows with semantic headline fonts and VoiceOver header traits; supplementary headings failed populated iPad Dynamic Type prediction.

Queue rows resolve headlines from the current host model catalog before the
additive queue label. A details sheet derives action state from the current
listing and loads full metadata separately. Per-job mutations revalidate state,
retry identity and reachability and stay guarded through refresh; cancelling
rows are read-only. Prompt History is machine-scoped server search, with fenced
requests and prompt-only recall. Search during Clear must reload the latest
query; reconnecting reloads without requiring a query edit.

**Reference parity.** Both native apps use shared `GenerationReference`/`DraftMedia` contracts for H3 image/MP4/PCM-WAV references and Hunyuan named views. Capability resolution owns older-host fallback; UI never matches model names. `BoundaryFramePolicy` owns Wan pairs and FL2VA endpoints. `DraftPictureAttachment` owns still-reference reorder/replace/remove and canvas refresh. Placement sends descriptors only; HTTP admission stages fresh upload V2 leases, canonicalizes exact media facts and cleans unused leases. Retained reuse requires matching descriptors, preserves new inputs, and never submits unresolved descriptors without a reuse session. Keep Photos/Library/Share routing in parity with primary attachment controls. Consume pending Library reuse on the actual Generate view's first mount and later changes; clear it synchronously so remounting never replays a handoff over user edits. Validation/UAT matrix: `docs/plans/native-reference-parity.md`.

**Permission recovery.** Native denial alerts use PermissionRecovery and public
UIApplication Settings URLs; restricted access explains Screen Time/MDM without
a Settings promise. Photos imports stay picker-only; saves request addOnly before
downloading, and auto-save never prompts. PhotosWriter's change callback is
nonisolated and Sendable, capturing immutable descriptors rather than app models;
PhotoKit invokes it off-main. Preserve files through completion. Camera capture
and supported QR scanning request authorization contextually. Nearby permission
recovery is only for Bonjour policy denial (-65570), never generic connectivity
errors. Refresh Settings/scanning/discovery after activation. See
`apps/ios/docs/PERMISSIONS.md` for the audit and hardware validation boundary.

**macOS retained recipe recall.** Mac typed-reference reuse is scoped to unchanged media and pipeline, with explicit model/recipe barriers; ordinary prompt/shape/seed edits and repeated admissions keep the visible attachment. Legacy hidden roles keep whole-draft fencing. The `SavedReuse` locator persists only provenance/origin identities, never bytes or scoped handles. Separate private Mac draft snapshots may preserve local authoring inputs, but never replace retained-source verification (see `macos-native.md`). Cold launch blocks admission until instance, archive, output and reference facts are freshly verified. Reset/explicit discard supersede recall; navigation and informational notice dismissal do not. Previews use the authenticated bounded gallery-member thumbnail route; older-host fallback is limited to small stills with a verified digest. Partial retained slot edits fail closed until originals are reattached.

**Export parity.** `MediaExportSession` captures the displayed print copy; generation results promote their rendering-host copy before opening actions. Every capability read, conversion and sidecar download uses that captured identity. Video request fields live in MoldClient; GIF `pause_ms` is extra boundary dwell and explicit zero is retained only when `gif_pause` is advertised and valid. Park/omit pause for non-GIF and Loop/Once. Mesh defaults and frame budgets use the shared authorities. `PrintSheets` releases only the dismissed delivery's files; pending delivery survives options dismissal. Documents/Mold contains user-owned exports only, with collision-safe names; private state stays in Application Support/Keychain. Texture sidecars verify digest and size before delivery. PhotoKit export destinations are conservative: GIF only for converted animation, with APNG/WebP routed to Files/Share.

Conversion/delivery failures remain visible beside Export, while loading and
unsupported-format explanations stay at the top. Audit text detection in every
settled export viewport before predictive clipping/Dynamic Type resize the Form.
Files-cancellation UAT follows the native picker to Cancel/Close and requires
its navigation bar to disappear before reopening export.

The export Form uses an opaque native navigation background to keep scrolled
text from bleeding behind its title and Cancel action.
Export audit scrolling uses the visible leading Form gutter, opposite UIKit's
trailing scroll-indicator hit region, and a held slow drag to
avoid skipping AX5 rows through momentum. Require complete label/value coverage
and keep all settled audit types strict. Only at maximum AX5, before recording a viewport, use
bounded measured gutter motion to settle text just crossing the navigation edge
when currently visible controls can remain fully contained. Requery the exact
text occurrence and original control IDs, keep gestures on screen, and fail
layout drift or displaced controls. Nonmoving adjustments with unchanged
geometry and unaffordable adjustments still face the strict audit.
The exact unnamed iOS 26.5 text-clipping prediction "Text of this element may
be clipped at larger Dynamic Type sizes." has retained inconsistent red/green
diagnostics at reviewed Large/XXXL sizes and at maximum AX5. Only those three
categories may use that specific disposition. Independently inspect every
retained actual-size GIF/geometry viewport in both appearances before accepting
UAT; frame containment alone does not detect internal truncation. Named clipping,
other descriptions, sizes, runtimes and audit types remain failures. Keep the
original red bundles and do not count a top-only lazy Form audit as full coverage.
Match the shell suite's bounded
framework-timeout recovery: retry only accessibilityAudit/-56 once, preserve
the first diagnostics, verify unchanged layout with fresh scope, and never
retry actual findings or accept a second timeout. Capture the immutable
viewport baseline with the existing inventory rather than repeatedly querying
the entire hierarchy. Hosted media-export jobs allow 90 minutes for the full
suite, build and report processing; other audit groups retain 60 minutes.
Photos export UAT scopes SpringBoard permission actions to Mold Studio's Photos
alert, requires that alert to dismiss, and waits up to 60 seconds for cold
authorization and PhotoKit completion. A generic Allow button can belong to
another permission request; its tap alone does not establish Photos permission.
For selected-method UAT, verify requested method names against the test source
and compare the executed method set/count with the intended selection.
XCTest can silently ignore an unknown selector while other selected tests pass.

Native CI splits media exports into delivery, video accessibility and mesh accessibility jobs per appearance. Method selectors cover every MediaExportUITests method exactly once; the routing contract refuses omissions and duplicates. The full local suite remains unchanged.

**Stable authoring and media.** Generate is a composer with a pinned Generate action and queue-count navigation; running/result media belongs in Queue/Library. Do not key or replace the composer/picker hierarchy on rendering state. Production Live Activities are disabled; keep completion/failure notifications. Source Library is merged across hosts with machine/search filters. Use as Source preserves unrelated draft edits, fences model/host/recipe/media changes, and routes through capability-defined boundary/reference/named-view inputs. New pictures conform through PictureImport (4096 axis/2 MiB, orientation/alpha), never through retained-media recovery. New source clears a stale mask. Selected-page 3-D loading uses mediaFile, checks size before parsing and removes temporary files. Video autoplay/repeat are user settings; background/deselection pauses.

Queue inputs use shared QueueInput descriptors and QueueInputPreview loading. Details independently load every ordered input, with role labels and explicit failed/nonimage previews; rows show an input image and additional-input count. Older hosts fall back only on missing additive routes. A failed member must not hide later images. Host/instance/job fencing and removal pruning apply to the entire set.

Library New badges compare filenames with the previous session visit, using the
whole active pool and a stable per-view snapshot. First visit establishes a
baseline; viewer navigation preserves the visit. Badge rows must fit machine
labels beside playback without overlap, including narrow tiles.

Use These Settings starts from fresh selected-print media, including parked wells.
Probe fences cover reuse identity and source authoring revisions. Unavailable
expected conditioning must block generation until restore, replacement or
explicit discard; retry after reconnection. Single/plural identity photos are one
authority when selecting retained members.

**Error presentation.** Both native apps use `MoldClient.UserFacingError` for
server diagnostics and old-server queue/batch/download replies. Local catches
use `UserFacingError.describe` (through `Error.sentence` on MainActor); background
workers call it directly. It records original local diagnostics in OSLog with
private details, then supplies device/app wording. Do not point a local file
error at a remote machine’s logs. A held Queue row shows its full-width reason
once, outside the thumbnail column; keep Retry/Move controls and Dynamic Type.

**Queue model recovery.** QueueDownloadRecovery in MoldClient owns the native
Download and Retry lifecycle. Show synchronous starting feedback, exact-ticket
progress and visible license/failure/reconnection outcomes in rows and details.
License dismissal cancels the waiting recovery, including a review over Job Details.
Fence every mutation by server instance and fresh held-job authority; never retry
on same-model historical success or after a failed/cancelled companion download.
Use concise held captions and one reason paragraph; narrow/AX action rows stack.

**Visible reuse media.** Restore supported legacy retained roles into ordinary authoring wells before allowing submission, including all endpoint/keyframe and reference-image inputs. Preserve list order, exact frame indices, manual canvas, continuation overlap and explicit reference strength. Retire each materialized or superseded legacy role, including the mask paired with a replaced source; removal must never revive an archived fallback. Fence every asynchronous operation by reuse identity, immutable origin route/instance, per-role monotonic attachment revisions and component lifetime. Scalar edits remain live. Keep restoration failures blocking until deliberate recovery/discard; descriptor-only typed references retain their exact-set authority.

**Shared library machine scope and visibility.** Library's Machine picker projects
copies before membership, counts and mutations. Shelf badges distinguish absent
(successful collection inventory) from unavailable (failed/offline inventory).
Hidden protection checks the full merged identity before machine projection; an
explicit collection still requires membership on the selected host. Use
`CollectionShelf.hiddenIDs` and `LibraryEntry.isFavorite`/`tags` for shared
attribute projections. CollectionVisibilityLedger persists deliberate hide/show
intent, fences superseded edits and replacement routes, and conservatively heals
mixed hidden replicas during fresh native inventory reconciliation. Successful
writes remain pending until a later listing confirms them. Never convert missing
selected-trash deletion support into live deletion; preserve the server refusal
in the operation result.

Queue Cancel remains visibly available for every currently actionable row, independently of batch metadata, retryability and transfer destinations. Failure Details preserves optional `error_detail` separately from the plain-English row explanation, falling back to the older machine’s reason. Copy includes machine and job identity. Cancel rechecks current state; Held actions always use the held-only endpoint and must never widen their intent after a state change. Current servers enforce this guard atomically; older servers may ignore the query, so do not promise that safeguard on older hosts.
