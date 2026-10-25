---
paths:
  - "apps/ios/**"
  - "apps/shared/**"
---

# Mold Studio Companion (apps/ios) and the shared Swift packages (apps/shared)

**Appearance.** Settings offers System, Light and Dark through the local
`appearance` AppStorage preference. The app root's AppearanceWindow probe applies
UIKit's window override to root content and every presented sheet. System explicitly
clears it to `.unspecified`, so iOS owns the palette again; never sample the overridden
window's traits to infer the device setting. SwiftUI `preferredColorScheme(nil)` can
leave a presented sheet holding its old explicit palette. Keep selection persistence,
live sheet/root changes and returning to System on both device palettes covered by native tests.

**Temporary build-first delivery (2026-10-06, owner request).** Hosted native
accessibility CI is physically commented out in `ios-native.yml`; its matrix and
local audits remain available unchanged. The CI lane compiles the native app
and shared packages for iOS Simulator; TestFlight separately archives and signs
for devices. Restore the commented lint/unit/package and audit job blocks when the
owner resumes test gating.

**What it is.** The native SwiftUI iPhone/iPad app: `io.utensils.mold.companion`, Home Screen label "Mold Studio", iOS 26+, remote-only. It is the macOS Mold Studio app's (`apps/macos`) little sibling and installs BESIDE the Tauri iPhone app (`apps/mobile`, `com.utensils.mold`), never replacing it. `apps/ios/docs/DESIGN.md` is the binding spec, `apps/ios/docs/PLAN.md` the milestone plan. Follow the Mac app, never the Tauri app's idioms (custom tab bars, toasts, fixed px sizes).

**Shared code.** `apps/shared/Packages/MoldClient` (wire + transport, Foundation only; `lint-layers` bans SwiftUI/AppKit/UIKit), `MoldStyle` (tokens) and `MoldMesh` (the Metal mesh renderer; its shaders load from the resource bundle's `default.metallib` in Xcode builds -- Xcode compiles a package `.metal` even when declared `.copy` -- and from the copied source under `swift build`, which never compiles it; `MeshMetalStack.shaderLibrary` tries both, and `swift test` plus `make packages-test` cover one each) serve BOTH apps; `platforms` declares macOS 26 and iOS 26. A change there must keep `make -C apps/macos lint` green and compile for iOS (`xcodebuild -scheme MoldClient -destination 'generic/platform=iOS Simulator' build` inside the package). Mac-only API goes behind `#if os(macOS)` (as `MoldHome` is). Credentials go through `CredentialStore`, keyed by host UUID: the Mac's `SecretStore` file on macOS, the Keychain on iOS; an empty key is a clear. Decision logic longer than ~10 lines that both apps need lands in MoldClient with its test, not twice in two app targets.

**Never register `mold://`.** The Tauri app owns it and iOS resolves a shared scheme arbitrarily. The companion's scheme is `moldstudio://`. The pairing webpage can hand its unchanged fragment to `moldstudio://pair#…`; accept only the exact `pair` authority with no credentials, port, path or query and reuse the universal-link payload validation. Never put invitations in a URL query. Bound the in-app paste field to three scrolling lines so long Secure codes do not hide Pair. Pairing QRs are universal links, `https://utensils.io/mold/pair#<form-encoded payload>` (the payload in the FRAGMENT, never the query, so it never reaches the web server): the app claims `applinks:utensils.io` (the `apple-app-site-association` lives in `utensils/utensils.github.io` `public/.well-known/`), an opened link lands in `PairingLinkSheet` and is claimed only when the person taps Pair, and `utensils.io/mold/pair` (`website/pair.md`) is the page for a phone without the app. Producers and parsers on both sides (`studio/api/pairing.ts`, MoldClient's `MobilePairingPayload`) are held to `studio/api/pairing.fixtures.json`; change the format there first. Only version 2 payloads are accepted. Absent `mode` means Secure; its claim signatures bind the one-use invitation to the completed Noise handshake and device keys through encrypted `POST /api/pairing/claim`. Explicit Regular payloads carry `mode: regular`, a direct HTTP(S) origin and `pairing_token`; redeem only against that approved origin through `/api/pairing/regular/claim`. Regular supports LAN, VPN and WAN, never Mold proxy. Camera-scanned Regular codes name their mode and direct address before an explicit Pair claim. Persist the selected mode in `MoldHost` and machine metadata so queued work and media retain it. Existing direct records without mode or pins remain Regular; new choices default Secure. Switching mode needs a fresh pairing or a separate Regular credential, never the previous Secure API key. Keep one physical machine row; regular requests never adopt discovered routes or fall back to proxy.

**Library offline and thumbnails.** Machines serve thumbnails at `size=256` or `512` only (anything else is a 422 and an empty tile): `ThumbnailLoader.bucket` never asks for more; the viewer reads the full print (`original(for:)`), not the thumbnail route. Each machine's rows are saved by its `GalleryReplica` (`LibrarySnapshots.replicas`, beside a collections-only `LibrarySnapshot` per machine, the app's strategy-free coders) and restored before any machine is asked (`GalleryReplica.restoreSaved`); pictures live in Application Support/offline (excluded from backup), bounded by `OfflineLimit` (including Unlimited). Opened originals use the
file-backed `OriginalMediaCache`, shared by viewer and export actions; download
one full video before local playback, with explicit download status. Never use
thumbnails as original export bytes. Every caller owns a separate temporary file;
cache eviction cannot interrupt playback or sharing. Key originals by print
version and machine authority; purge/removal epochs fence late downloads. The grid's pinch is `PinchRecognizer` (UIKit), not `MagnifyGesture`.

**Library replicas.** `LibraryStore` reads each machine through one MoldClient
`GalleryReplica` with the Mac's contract (`macos-native.md`): kept across route
changes, replaced only when the API key, connection mode or server instance
changes; every pull carries `Pull.origin` and `absorb` refuses a stale origin
with `redeliver()`; a Clear Cache epoch fences replicas, restores and listings
begun before it. The feed is used only when `hasGalleryChangesFeed`; otherwise
the conditional listings (trash only when `trashEnabled`). The replica's
`inTrash`, not `trashedAt`, decides a row's partition. Deltas reach `absorb`
before `pull()` returns; `absorb` only edits `live`/`trashed`, replays the
outbox's pending edits and notes ids, and whoever started the pull calls
`settleTimelines()` (`LibraryTimeline` patch, else a full merge). A nil delta
must not reassign `pool` or bump `revision`; shelves re-render only when
`refreshShelves` finds a difference. SSE `.gallery`/`.resyncRequired` frames
only `tickle` (a 300 ms pause, then one pull; any pull starting first answers
for them), never a full reload. Collections are re-read only when stale: first
listing this launch, `collectionsChanged`, resync, a reset or a delta touching
a filed row, the outbox's collection edits, and the fleet `reload()` reconcile
(pull to refresh, foreground, `BGAppRefreshTask`). An outbox give-up is a FULL
pull so a refused optimistic edit is undone. Replica files live under
Application Support/offline/library/replicas (not Caches: they are the offline
Library), are flushed (`flushIfDirty`, inside a background task) on entering
the background and after a background refresh, and deleted only for a machine
this session saw removed -- never by sweeping the directory against the host
list, which reads empty before first unlock. A replaced replica is `retire()`d
so its pending save cannot land over its successor's file. A stale-origin
delta `redeliver()`s and tickles; a cancelled pull re-arms the tickle it took
over. Older builds' whole-listing snapshots are migrated once
(`LibrarySnapshots.migrate` -> `GalleryReplicaStore.seedIfAbsent`, keeping
their validators); the old file is rewritten only once a replica file exists,
under the lock Clear Cache's purge takes, so a queued migration never writes
after a purge.

**Library.** Every edit goes through `LibraryStore.apply` -> `PrintEdit.plan` over `everyCopy` -> the shared `MutationOutbox` (optimistic, retried with the same operation id, re-read on give-up); never a hand-rolled request per print. A recipe's output kind is `GenerationRecipe.makes` (GLB -> 3-D, time axis -> clip), never a family list.

**Dynamic Type is a gate, not a goal.** Text styles only (`make lint` bans `.font(.system(size:`, `.custom`, `minimumScaleFactor`). Label/value rows ask `RowAxis.for(_:)` (stack from AX1). Secondary text is `.secondaryText`, never `.secondary` (the system colour is ~4.4:1 on white and fails the audit); the one filled button per screen uses `.prominentAction()` (`ProminentFill`). `make uitest` runs `performAccessibilityAudit` on every destination at xSmall, Large and AX5 in light AND dark, on an iPhone, an iPhone SE when one is installed, and an iPad; CI (`ios-native.yml`) sweeps the iPhone only, because the hosted runner times iPad audits out, so iPad work needs a local `make uitest` before merge; it may skip contrast only for elements under the bottom chrome (the tab bar, its scroll-edge dimming band, or the composer -- chrome membership by tree descent, never frame containment) or outside a presented sheet / the open iPad sidebar (their dimmed backdrop), plus clipped composer scroll content only when the scroll-coverage pass requires every exported label/control to become fully visible and be contrast-audited, clipping only for the system search field and UIKit's single-line sidebar-row PREDICTION (the AX5 pass audits the open sidebar for real clipping), and Dynamic Type only for Settings' lazily laid-out Form and the Library grid's `day-header`s at xSmall/Large (AX5 proves them) and the `tile-badge`s. XCUITest lists views even under `accessibilityHidden`/`children: .ignore`, so an exemption keys on an `accessibilityIdentifier`. Models and the iPad sidebar's sections stay out of the floating tab bar (`.defaultVisibility(.hidden, for: .tabBar)`): this keeps the bar within the available width at large text sizes, but does not resolve the separate auditor-triggered UIKit pagination loop. On iPad, audit the Settings Form through its native sidebar page: the auditor's private text-size cycling over the Settings sheet can loop UIKit floating-tab pagination even though ordinary system text-size changes stay responsive. Keep all audit types on the Form, and separately test sheet Done/Add a Machine transitions; this is equivalent Form coverage, not sheet-chrome coverage. A new screen joins that audit. Never edit compiled sources while `make uitest` runs: every pass recompiles the working tree, so a half-written file fails the passes after it.

**Extensions.** Widgets read the App Group snapshot only (no MoldClient, no URLSession, no Keychain); the Share extension stages into the App Group and never networks. `make lint` enforces both. Because nothing outside the app reads a key, keys live in the app's OWN Keychain group (`KeychainCredentialStore`, service `io.utensils.mold.companion.remote-api-key`, `AfterFirstUnlockThisDeviceOnly`) -- there is no shared keychain-access-group entitlement.

**Connection recovery.** Bonjour TXT identity is a hint, never credential authority. Match saved `connectionInstanceID` even while offline to avoid duplicate Nearby rows; accept a discovered address only after the Noise handshake verifies the pinned host signing and static keys. Preserve UUID, key and original origin; reject late proofs after editing/removal/rekey. Foreground recovery retries unavailable machines, refreshes read models on route/reachability recovery and cancels lookups/background work. Bound Nearby TCP resolution to five seconds. Stagger lower-priority probes and cancel them as soon as the best verified class answers; a slow proxy or same-class address must not delay LAN. A read failure describes the remote connection without mutation protocol instructions, and a successful listing clears its own old error.

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

**Composer and playback.** Keep the prompt in one stable scroll-view hierarchy, capped using the window's available geometry (not `UIScreen.main`). Model selection is a labeled wrapping button that opens a searchable sheet; settle selection when model lists arrive, not only when reachability changes. Playback configures `.playback` / `.moviePlayback` without eagerly activating audio while browsing, and cancels encrypted media reads when its task ends. Embed `AVPlayerViewController` for paged playback and let it own video gestures. Run contrast separately before Dynamic Type audits, which temporarily resize the hierarchy. `GenerationInteractionTests` exercises saved-machine layouts and prompt input in Simulator.

**Phone navigation and gallery performance.** Generate is one stable scrolling composer with visible Kind and Machine choices; progress belongs in Queue and results in Library. At large text, the option chips collapse before they can make Generate wrap, and the action takes the full width. Give the form enough bottom travel to move Options above the pinned action. Model search needs a visible close action with the keyboard up. Keep the phone viewer navigation bar visible so a still image cannot trap the user after a chrome toggle. The Library exposes shelves and media types through one inline navigation title menu on phone; Settings is reachable from Generate, Library and Machines. Cache clearing includes saved listings and images, waits for pending listing writes, and drops offline-only rows. `LibraryView` uses `LibraryShowingCache` keyed by revision and query, and `LibraryGridProjectionCache` derives host badges, favorites and indexed print positions once per revision/query. Observe viewport geometry without invalidating view state; emit tile frames only during selection. Freeze the exact raw content offset before viewer navigation and restore it on dismissal, including partially visible tiles. Bound viewer pages to the selected print and two neighbors on each side; a different shelf or query gives the grid a new identity and top offset. Only the selected viewer clip may autoplay; pause it on page deselection and dismissal.

**Populated iPhone regressions.** Options uses separate form rows, never the composer's horizontal ChipRow inside the sheet. Model search has an explicit no-match state. Restore draft kind from the saved recipe after profiles arrive, keeping authored values. Observe raw model/reachability changes: a picture-filtered model list cannot wake a pending clip draft. Hide the main tab bar in PrintViewer so it cannot cover the bottom actions. The grid (`LibraryView`) owns the tab-bar and bottom-bar visibility for its whole stack, keyed on `viewerOnScreen` (the destination's onAppear/onDisappear): viewer showing -> tab bar hidden, bottom bar the viewer's; otherwise tab bar visible (hidden only under its Collections / offline-details sheets) and bottom bar hidden. Never `.automatic` (it inherited a stale pop's state: the viewer's Share/Favourite/Info/Delete bar on the grid with no tab bar), and never a constant `.visible` (the grid's explicit value outranks the pushed viewer's, so it would show the tab bar inside the viewer) (`LibraryViewerChromeTests` leaves by Back, swipe down and edge swipe). `PopulatedGenerationTests` uses a loopback read-only fixture, not a live inference server; the iPhone UAT record is `apps/ios/docs/IPHONE-UAT.md`.

**Media exports.** `mediaFile` returns an extensionless caller-owned temporary file. Stage it under its safe original filename before handing it to UIKit/Photos; remove the export directory on share dismissal or after copy/save, including partial failures. Info always has an explicit Done action.

**Unavailable data.** Pending draft restoration must not fall back to the first responding machine; an explicit kind/model choice cancels it. Generate distinguishes offline machines from empty inventories. Offline is a non-issue, greyed out and nothing more: a recheck of a `.down` machine never flips it to `.checking` (only `.unknown` shows Checking -- the flip made the Queue flicker every 15 s; same on the Mac), `HostStore.report` turns `MoldClientError.unreachable` into `recordConnectionFailure` with no banner line, `.down` is a grey dot and "Offline" (the reason only in Machine details), and Queue draws a greyed "name · Offline" header (`QueueStore.offlineMachines`). Queue says unavailable only for an answering machine whose read was refused (`listFailed`), never for one offline or not yet read, and reloads as machines reconnect. A missing export source fails the entire selection, never compresses the file list before Photos pairs it with entries.

**Notification presentation and refresh cancellation.** Use the system notification banner (the app bundle supplies AppIcon), concise completion copy, and no prompt or media attachment. Preserve print/queue links and per-batch deduplication. Notification delegates implement the explicit completion-handler API: extract payload strings off-actor, then route and invoke completion on MainActor. The nonisolated async delegate bridge completes on a cooperative thread and crashes UIKit notification activation/state restoration ("Call must be made on main thread"). Test the Objective-C callback from a background queue, including its completion thread, and actual Notification Center taps after backgrounding and cold launch; an onOpenURL-only test misses this boundary. Dismissed and unknown actions complete without navigation. A cancelled Library listing is routine lifecycle/event coalescing, not a host failure; keep loaded prints and never report it in FailureBanner. Genuine listing failures remain visible.

**Generation option parity.** BoundaryFramePolicy receives the recipe when attaching an endpoint: the first frame owns source-driven closest-aspect sizing while present; otherwise the closing frame owns it. New driving endpoint attachments re-arm sizing; removing the first transfers automatic sizing to the last, while manual/reuse canvas choices survive removal and recipe adoption. Neither endpoint means no invented dimensions. Test the real attachment path for Wan and H3, plus the iOS top-level aspect menu selection. Aspect menu icons draw the ratio of the offered dimensions rather than a generic rectangle. SourceFitOptions and SourceFitRender in MoldClient are shared native authorities. iOS fits a submission snapshot from original source bytes before admission; its painted mask is in source space and must follow the same transform (macOS masks remain canvas-space). Source-driven, parked reference-only and continuation inputs bypass fitting. Random seeds are the untouched default; Fixed is explicit and Reset restores Random plus centered crop-fill.

**Hidden collections.** Library queries include host-local hidden membership ids and check every merged copy, so a local lead cannot expose a hidden remote member. Hidden shelves remain directly browsable. Library View Options offers Manage Collections on phone and iPad; its Hide from All Prints toggle updates every machine holding that shelf and surfaces failures through HostStore.

**Library status notices.** Failure/offline status occupies an always-present toolbar button; updates must never insert/remove a row above the grid or move its content origin. Full messages remain in the status details sheet. The saved-gallery notice uses primary text on an opaque semantic surface: material failed small-text contrast even with explicit primary text. Contrast coverage includes the populated date heading and offline notice at every audited text size; Library and Search share this layout. Keep pinned offline status independent of machine-name lengths: a compact count and saved-print context opens scrollable full-name details, with host IDs as row identity and a visible Done action. Library owns offline-details presentation and hides its presenting tab chrome while open, using the same visibility preference as Collections; restore it on Done. Test the actual native floating-tab owner on iPad rather than assuming it exports as TabBar. The details sheet uses an inline native title, an opaque semantic navigation background, a semantic secondary section heading and concise saved-print context. Its scalable Done action sits in an opaque bottom safe-area inset with a minimum 44-point target; the scrollable text viewport ends above that footer. Require the full button to fit its footer and presented sheet. Audit requested-size bounds and contrast before private Dynamic Type sweeps mutate the native List. Use a fresh native details presentation for each subsequent private sweep, requery its owner and prove the original settled paragraph size and title/action heights are restored. Preserve diagnostics and failure for audit issues without an element. Details contrast covers the fully visible native title, explanatory text, heading, every per-ID machine name and Done; require an exhaustive semantic inventory so new labels or actions cannot escape coverage. Do not sample a partly occluded lazy row beneath the pinned footer as though its full text were visible. At AX5, actual controls must fit the effective Library viewport; retained photos must fit when possible, or scroll to expose the maximum available image area with an unobscured hit region and a verified print action; never clamp Dynamic Type or truncate names to make the notice fit. Decorative tile badges stay one line within the proposed tile width; crowded symbol-and-text badges may fall back to the native symbol, while tile speech and Info retain complete metadata.

**Live Activity appearance.** Use `.activityBackgroundTint(nil)` for ActivityKit’s system material; never resolve a custom UIKit `systemBackground` separately from the Lock Screen foreground. Preserve semantic text and the system color scheme. Check running and stale cards in both appearances.

**Live Activity layout.** Keep the Lock Screen card within 160 pt including padding; `ActivityCard` offers a compact fallback before clipping status or Stop. The step figure and machine name are separate labels (strip only the exact legacy machine suffix in `ActivityCardContent`, without changing the ActivityKit wire payload). Preserve the stale refresh message, terminal print deep link, and 44 pt Stop target. `--live-activity-fixture` is Debug-only UAT, never a render or a distribution feature.

**UI test reports.** `make uitest` makes one full pass, then retries only identified failed methods once in a fresh `xcodebuild` process. Read public `xcresulttool get test-results tests` JSON; incomplete/unsupported failures never fall back to rerunning a complete target. Before accepting a retry, require exactly every requested method to appear as Passed in its result bundle. Preserve original and retry logs/result bundles under `build/UITestResults`, uploaded even on failure. CI partitions all XCTestCase classes into five groups (app, references, Library, Library interactions and media exports), each in light and dark; keep GenerationInteractionTests and PopulatedGenerationTests with ShellAccessibilityTests because Shell audits use the persisted machine fixture. Local `make uitest` defaults to the complete target and both appearances. The Git audit classifier only skips app audits for explicit Widget Swift/inert documentation/static routing or branding paths. App/shared/UI-test/build inputs, unknown paths, dispatch, and unavailable or stale-base diffs require full audits. The classifier job must succeed and explicitly emit `audit=false` to skip; classifier failure keeps the workflow red and runs all audits. Lint/unit/shared-package checks always run and compile Widgets. Widget-only changes require separate Lock Screen/Dynamic Island appearance UAT because app audits do not render extensions. The routing contract checks class coverage, while `scripts/tests/ios-uitest-runner.sh` and `scripts/tests/ios-uitest-retry.py` exercise bounded retries, exact selections, and failure propagation.

**Library selection and media filters.** Hide the presenting tab chrome while the Collections modal is open and restore automatic visibility on dismissal; this keeps iPad modal layout stable during Dynamic Type audits without dropping audit types. A one-finger horizontal start in Select mode acquires a UIKit range-selection pan; vertical starts remain native scrolling/refresh. Apply a baseline range, choosing select/deselect from the start tile, so reversal restores untouched selections. Cancel edge-scroll work on end, disable and coordinator destruction. `LibraryMediaFilter` in MoldClient applies the same kind tokens for both native apps without dropping other search tokens. The Photos-style Filter menu's model is MoldClient's `LibraryFilter`: ONE pick (All Items, Favourites, Photos/Videos/3D, In a / Not in a Collection, Made from a Source, Has Tags / Untagged, Portrait/Landscape/Square) that replaces only the pick tokens and keeps machines, families, collections, tags and text; `LibraryFamilies` lists only the families the Library holds (canonical slugs via `GalleryPrint.familySlug`, colours via `Model.familyColor(forFamily:)`) and toggles `.family` tokens, which widen each other. Family colour appears as Finder-tag-style dots in that submenu and beside the model in Info/viewer, never on grid tiles. On iOS the Filter is its own toolbar button beside View Options (`Library/LibraryFilterMenu.swift`, accessibility identifier `library-filter`, value = `LibraryFilter.summary`), holding the picks, Model Family, the Machine picker (moved out of View Options) and Clear Filters; menu dots are `FamilySwatch.menuImage(rgb:)` (`.alwaysOriginal`). The phone title menu keeps its media types beside the shelves. The Mac toolbar offers the filter and preserves Command/Shift selection and external media drags; visible pointer selections must not recenter the grid.

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

Library New badges and unread counts share persisted client-local viewing state.
Individual viewer display marks that logical print read immediately. A Library
visit captures its displayed eligible projections and marks that captured set
read when navigation leaves Library. Keep New labels during the visit; exclude
Trash and unvisited hidden/filtered media, and preserve arrivals after departure. Preserve current
read status on upgrade and establish an initial baseline for newly connected
hosts. Later arrivals remain unread until viewed or acknowledged by a Library visit. Count merged
visible active inventory, excluding hidden collections and Trash; retain offline
history. iOS navigation and native icon badges follow that same ledger, with the
existing permission/preference controls. Do not let activation-only arrival
trackers overwrite the persistent count.
Badge rows must fit machine
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
once, outside the thumbnail column; keep Move to controls and Dynamic Type. Cancel, Pause/Resume and Retry live in native swipe actions and Job Details. Download and Retry is also visible directly on a held card when its model recovery is supported.

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

Queue Cancel remains reachable through native swipe actions and Job Details for every currently actionable row, independently of batch metadata, retryability and transfer destinations. Retry and Download and Retry use leading swipes plus accessible context menus and detail controls; Download and Retry additionally stays visible on supported held cards; preserve recovery status/progress and on-card Move to. Disable full-swipe activation, suppress busy recovery actions and recheck recovery availability inside their closures. Offer a read-only Details swipe when no leading mutation is eligible; settled/unknown states stay excluded from the live queue. Failure Details preserves optional `error_detail` separately from the plain-English row explanation, falling back to the older machine’s reason. Copy includes machine and job identity. Cancel rechecks current state; Held actions always use the held-only endpoint and must never widen their intent after a state change. Current servers enforce this guard atomically; older servers may ignore the query, so do not promise that safeguard on older hosts.

Queue context-menu and Edit-mode reordering use row-aware `canMove`. Reserve every planned mover atomically through refresh, just as row Cancel/Pause does. Empty Queue reserves the currently affected queued, paused and Held IDs, refusing overlapping actions; keep existing bulk endpoint semantics, recheck host route/instance and current Held identity after its await, and retain the Held-only endpoint.

**Landscape video.** Phone clip pages hide gallery bars and the status bar in a landscape window, extending AVKit across the display with aspect-fit video. Keep a safe-area Close action; portrait and nonclip pages restore their ordinary chrome. Observe window geometry without replacing the paging/player hierarchy or reloading the stream. `LandscapePlaybackTests` covers both orientations, portrait restoration and dismissal using a local 16:9 clip.

Continuous Library rows preserve actual metadata aspect ratios with 2-point seams
and square corners. Missing/invalid dimensions use a square placeholder; decode
completion never changes geometry. Incomplete final rows stay left aligned at
target height. Valid extreme ratios remain uncropped. Keep ordered row geometry
shared (studio/lib/justifiedLayout.ts and MoldClient.JustifiedLayout), lazy/windowed
rendering, stable item identities, and a visible item anchor through resize/zoom.
Native macOS vertical arrows follow adjacent row centers, never a guessed column
count. Viewer return restores the covered viewport; native previews explicitly
inject thumbnail loaders. Native iOS omits the visual host-name thumbnail badge;
Info and host sorting/filtering retain their authority. Unseen semantics are unchanged.

**Keyboard dismissal.** The composer keyboard Done must clear its ordinary prompt FocusState and resign the actual UIKit first responder: shared clip rows do not own that local binding. Keep one keyboard action; expanded editors own their separate Done. Exercise scrolled clip rows, ordinary prompts and options/title/tag sheet dismissal without submitting a render.

**Prompt editor.** Keep quick prompt entry bounded to three lines; Edit prompt
opens a large native, keyboard-safe TextEditor sheet bound to the authoritative
draft. Done and ordinary dismissal retain and save edits. Native typing preserves
existing provenance semantics; explicit history replacement and Clear release
provenance. Reuse MoldClient PromptEditingSession for Clear/Undo clear. Disable
background Generate and Expand shortcuts while editing; editor arrows and Enter
remain text navigation/newline. Keep rewrite undo window-local across sheet
reopen and fence it against a different authored prompt.

**Library date separators.** Preference.showDateSeparators defaults on, persists
per client, and appears only in Settings Library. Pass it to LibraryShowingCache
for Library, Search and shelves; continuous sections retain sort, identities,
selection and the print anchor during reflow.

Library viewer removal preserves the previous filtered order and copy identities: keep any surviving copy of the current print, otherwise advance to the next survivor, then the previous survivor at the end. Close only when the list is empty. Web updates the print URL when advancing; explicit scope/filter changes retain their existing navigation behavior. Native Mac successful trash moves do not leave a completion notice; preserve progress and uncertain/interrupted outcomes.

**Source Media browser.** `SourceMediaCatalog` is a transient projection, never a
pin owner. The dedicated native Source Media section includes all live collections, including hidden ones; apply its selected collection and physical machine filters before enumerating every owner's members. Authenticated previews remain
bounded and separate from original attachment content. `SourceMediaAttachment`
offers current-recipe destinations, and card imports fence model, recipe, media
revisions, origin route/key/instance and view lifetime. Keep Source Media in the
phone Library navigation and iPad Shelves sidebar rather than adding a phone tab.

Native Queue navigation badges total `QueueState.isLive` across every host listing: queued, running, paused, held and cancelling. Count merged individual entries rather than batch groups; completed, failed and cancelled history does not contribute.

Source Media groups only verified reusable-byte digests and keeps every owning print/role selectable. Browser authority includes every current live owner, including hidden collections; Trash-only originals remain detail-only. Preview caches are bounded and memory-only, refresh on relevant source events or explicit Refresh, and evict revoked owners even when the browser is closed. Named gallery events invalidate only that owner; resync invalidates the host. Reuse must recheck the captured owner, route, instance and draft after downloading.

Native `HTTPBackend.queue()` consumes every returned cursor before publishing a queue snapshot. Merge repeated runtime/durable IDs within each host, reject malformed or repeated cursors, and never publish a partial backlog as a complete badge total. Legacy hosts without page metadata remain supported.

Native model recovery prefers the queue row’s `model_download` hint, requiring a nonblank exact match to the row model. A present malformed/null/mismatched hint blocks legacy fallback without dropping the row. Only absent hints permit typed batch-code fallback. Preserve actual missing-component prose and diagnostics; use the existing exact-host, license-aware download-and-retry coordinator.

Active curated/community model downloads reserve a full-width section beneath model identity; never put a growing cancel/progress column beside the title. Keep visible Cancel concise, accessibilityLabel Cancel Download explicit, and the button's label/hit area at least 44pt. RowAxis.model pins this layout for every Dynamic Type size.

The image Library picker keeps machine filters in a full-width ordinary content row below native navigation chrome. Do not use an opaque top safe-area inset: its intrinsic-width background can extend behind the title. Preserve native search, refresh, empty/error/loading states and Cancel; verify default and AX5 geometry in light and dark.

**Native read recovery and shared policies.** `QueueTransferSource` distinguishes eligible, present-but-ineligible, and authoritatively missing sources; only a 404 with a verified prior destination may report an original removed. Transport, authorization and server failures never become missing. Thumbnail coalescing tracks individual waiter tokens and releases each exactly once; only the final cancellation stops shared work. Generation followers clear their owned task on every exit, retain admitted batches through failed reads, and reread on foreground reconnect without resubmission; stale tasks cannot clear or replace a successor. Finite admission tasks are separate from progress followers: backgrounding must retain late accepted ownership without opening streams, and explicit Stop intent must cancel a late accepted batch without reviving it. Stop Everything includes all pending admissions; preserve the ledger if late cancellation fails. Queue source-preview bytes live in one 32 MiB LRU store; four read lanes and a per-response budget bound pending work, while ordered descriptors and retry state survive eviction. Fence keys by full host route/credential identity, server instance and job.

`SourceMediaDisplay` and `SourceMediaLoading` in Foundation-only MoldClient own shared filtering, preferred-copy selection, inventory/original publication checks and stale-error suppression. Both apps supply current route, instance, revision and access checks; keep platform image/player/navigation presentation local. Never publish a revoked or cancelled read, including its error. Library new/unread behavior uses `LibraryUnreadLedger`; active video geometry uses `VideoViewport`.

**Encrypted transport.** All remote HTTP, SSE, uploads, thumbnails and media use the shared Rust `mold-secure` C ABI and the pull-driven Swift WebSocket/HTTP adapter. Only an unpaired numeric loopback host may use direct HTTP for the embedded engine. Never restore relay v2, HTTP/S3 staging, media bearer URLs or mutation replay. AVPlayer uses `PlayableMedia` and its HEAD/Range resource loader. Private installation identity lives in the app Keychain (Mac: owner-only SecretStore), never preferences. Public pins and host-signed grants persist with the host; renew grants only through the authenticated channel. Bonjour identity remains a hint. `make gen`/archive builds the ignored XCFramework from source for all Apple targets, and secure-crate/lockfile changes trigger Apple builds. Protocol authority: `relay/aws/SECURE-PROTOCOL.md`.

Call `PlayableMedia.prepare(_:)` before publishing a streamed player for playback;
it awaits AVFoundation native preroll without starting the player, with a 30-second
timeout and cancellation/item fences. Do not require a fixed buffered duration:
a paused custom-loaded audio/video item may validly buffer only half a second.
For a resumed item, call `configure(_:)` before seeking, then prepare at the
new position. Apple's `AVAssetResourceLoader.setDelegate(_:queue:)`
contract requires predictive stall waiting to be disabled when the delegate
supplies media bytes. Local media keeps AVPlayer's default policy.
AVPlayer's resource loader receives bounded media chunks on the main actor;
never iterate individual response bytes there. Per-byte executor hops can stall
decoding even when a download works. Playback regression coverage must prove
decoded frames and sustained clock advancement, with a local-file control,
in addition to metadata and seeking.

**Input inspection.** The shared Well has a separate accessible View control;
typed references, masks, queue input details, Library picker tiles and From Share
use the same read-only InputMediaViewer. Viewing never attaches, consumes a share,
fits, reorders or acknowledges a Library print. Prefer authoring original bytes;
retained descriptors require their exact verified ordered member. Queue views
inspect full-sized submitted pixels, which may already be fitted. Private files
and players belong only to the selected presentation and are released/revoked on
close, cancellation, attachment replacement and machine authority changes.

Known Secure credential fingerprints are retained across native mode changes and aliases. Regular requests refuse matching host keys, X-API-Key and Bearer headers before transport; the Rust C ABI hashes the denial history, which never stores credential bytes or grants access.

A recognized gateway busy envelope during admission, or an HTTP 429/503 upgrade refusal before the WebSocket opens, may retry a fresh socket at most three times within the caller’s existing deadline, with cancellable short backoff. Authentication/protocol failures and every failure after admission remain terminal; never replay an HTTP request.

**Selection organization.** Expose Add to Collection directly in the selection bar and in PrintMenu, sharing one collection menu. Match macOS discoverability when capability inventory is unknown: only an explicit gallery.organize=false blocks edits. Empty selections are ineligible; consider every physical copy in the presented selection. Preserve selection and route mutations through LibraryStore.apply; the outbox refreshes collection inventory after writes settle.

Held Queue recovery actions use separate full-width rows with wrapping labels, including the Move to menu. Do not put native Menus in a horizontal ViewThatFits candidate: their ideal width can accept compressed labels. Keep input captions in the metadata column rather than widening the thumbnail column; cover card and Job Details geometry at normal, larger and accessibility text sizes.

**Batch parity.** OptionsControls always exposes Batch for every recipe kind; submission and displayed count must agree. Group cancellation reserves all eligible children and rechecks machine instance, batch/client identity and current state before each lifecycle-specific request, with one final refresh. Held uses only-held cancellation; older hosts without running cancellation leave running children alone.

Native Queue rows keep a fixed preview slot with MoldStyle’s neutral image
placeholder when no source image exists. Distinguish loading, empty, nonimage and
unavailable states accessibly; do not present an ordinary source-free job as an error.

Library source actions share MoldClient.SourceMediaActions/SourceMediaState and the exact-owner
bounded original loader. Metadata copies include owning-host commands and full member IDs;
file publication/sharing is fenced by route, instance, current owner revision and visibility.
Release private files on dismissal/revocation. Print actions use recipe-advertised endpoints
and keyframes; explicit first/last handoffs preserve interior LTX anchors and authored scalars.
Adding a last LTX endpoint migrates a current source image into frame 0 unless an explicit
first anchor already exists. Print menus retain their source handoff while recipe is absent.
Copy File Name is available for print selections and all Info file values are selectable.

Native clip length bounds use MoldClient’s `TemporalProfile.lengthSelection`/`lengthBounds` for an ordinary render (the Length slider's ceiling): resolve a valid rate before duration bounds, snap whole-second choices, deduplicate and include only the settled current custom frame count. Subsecond lengths must not format as zero. Rate/recipe changes reclamp frames without clearing the automatic-duration preference; output temporal upscaling never changes this inference-frame authority. Mac controls preserve `ClipRouting` chain bounds.


**Fresh typed-video preparation.** Keep ordered authoring inputs in `AuthoringReference`: canonical descriptors and immutable fresh original snapshots are distinct. Pending facts cannot encode through `GenerateRequest` or placement. Use only negotiated keyed upload V2 + nested normalization V1 for fresh typed `reference_video`; scalar source/continuation paths, retained archives and old leases stay on their existing contracts. Freeze the candidate POST bytes and domain-separated digest, validate original/canonical receipts and exact final descriptors, explicitly finalize, then price and admit the unchanged canonical request on the captured normalizer. Auto filters existing host order by health/model/capability; a hard machine selection never falls back. Full route/key/mode/secure identity/instance and cancellation checks fence every await; release every unused session. Persist originals, never ephemeral handles or candidate authority. Capture malformed advertised normalization V1 as a nonserialized client diagnostic without failing the whole capabilities response; reject it specifically before typed-video import fallback and normalization admission. Absence, an unversioned block and an unsupported future version retain canonical V2 compatibility.

**Playback endpoints.** `VideoPlaybackEndpoint` resolves the last displayed sample in asset presentation time, including edit lists and B-frame ordering; item duration minus guessed frame intervals is not an endpoint. Both native transports share `VideoPlaybackSession` for ended/replay/repeat/full-end seek behavior and item/seek-generation fencing. Preserve legitimate audio tails and native controls. Cancel asynchronous endpoint/seeks on replacement or dismissal, and prevent late resumes after pause/inactivity. Keep local and encrypted fixture coverage for CFR/VFR, offset timelines, one-frame video and audio overhang.

**Observational stalls.** Poll previews for running and cancelling jobs. The
shared optional stall diagnostic must suppress stale ETA, fractions and step
announcements, retain phase/cancel/device-held truth and clear on authoritative
null. Decode block/chunk counts belong to the current phase, not video-frame
completion. HTTP 202 is accepted intent, never terminal cancellation; retain the
live job and offer copyable diagnostics plus explicit operator recovery guidance.

Source Media retains its separate Library navigation and uses the Library View Options, tokenized search, tile sizes and pinch sizing. Source-free print details stay empty after a single cached inventory check; only a relevant owner/host invalidation or explicit Retry rechecks them. Refresh retains a same-authority preview until its replacement arrives.

Source Media uses shared `JustifiedLayout` geometry, oriented source ratios (never generated-output dimensions), stable source identities and a single bounded synchronous warm-pixel cache. Batch ratio discoveries and retain a visible anchor through reflow. Do not poll inventories/thumbnails or turn warm cache hits into loading states. Invalidate old pixel work before publishing the next content epoch; keep same-authority display rows read-only during revalidation, while revoked authority and server refusal still remove access. Display-only sync updates never invalidate source content; identical-content row events remain conservative because retained-source repairs can arrive independently of output media versions.

Source Media inventory caching is shared across scenes; filtered browser windows keep independent scans, authorized row projections, progress and failure counts. A scan in one scene must not cancel or blank a sibling scene.

**Multi-clip video.** In Short clip, choose **Add another clip** to enter ordered prompts and lengths, then **Generate video**. The app validates the draft automatically before submitting it. Move or remove rows without opening a separate editor. Library **Continue video** makes the next separate linked MP4 from the latest completed authored take on its original machine. Ordinary one-prompt generation and Reuse settings remain available. Shared wire/draft authority lives in MoldClient; SwiftUI session/editor composition lives in `apps/shared/NativeSequence`. Retained reference bytes stay in memory. Recovered drafts with missing references require explicit restoration or removal before publishing. Follow `chain-sequences.md` for owner, latest-take and operation recovery fences.

Queue discovery keys include the full current host authority, live installation and positive duration capability, never just machine UUIDs. Late capability/route metadata and offline/auth recovery relist parents while Queue stays visible; progress-only status updates do not. Fence queue snapshots, batch children and errors after every await by the captured owner/installation and task cancellation. Reconnection never resumes a paused parent.

Automatic Extend restoration compares the full current owner as well as the selected source. Coalesce only an active identical restore; a cancelled or changed-owner restore may be replaced. Token-owned completion cannot clear another inspection’s loading state, and automatic restoration cannot supersede a newer explicit Library action. Keep listing errors page-owned and preserve an otherwise-empty Queue for page Retry; use the shared coordinator content predicate in both shells.

Duration progress followers bind their slot to the full captured owner. A replacement owner supersedes a pending stale reader without waiting for its socket; token-owned cleanup cannot remove the successor. Cache-only status is explicitly last known whenever the row’s current-owner predicate refuses controls.

Completed-parent dismissal is durable, device-local and scoped to host UUID, installation and operation; captured recovery/page/activity reads must never recreate a removed row. Recheck the live row before accepting a completed-only gesture. Confirmed server deletion remains distinct, with immediate busy feedback; only the exact owner-fenced VIDEO_JOB_NOT_FOUND response may settle an already-removed job. Gallery navigation verifies the exact physical output, keeps unknown/offline separate from missing/Trash, and observes authenticated Gallery invalidations. Pending output reads coalesce only for the same full owner and cannot overwrite a removal or successor.

Video parents and authored sequences are ordinary Queue rows inside their machine's section (`Queue/VideoQueueEntryRow.swift`, built from `NativeVideoQueue` items): `QueueEntryRow`'s placeholder slot, headline, shared `VideoJobRow` caption and detail, a trailing destructive **Stop** swipe (Cancel's role), a leading **Resume** or **Start again…** swipe where offered, a `.contextMenu` with every action (Delete job… last behind a divider), and a tap opening Video Details. There is no Check queue button: `NativeVideoQueue.recheck` runs on pull-to-refresh and a 15-second poll. The Queue tab badge adds live video parents (`NativeVideoQueue.liveSymbols`), which never double-count ordinary rows.

Completed and stopped video parents and authored sequences leave the Queue on their own, like every other row (`ObservedVideoJob.belongsInQueue`, `ChainJobDetail.belongsInQueue` via `NativeSequenceSession.queueJobs`); they also stop holding an otherwise-empty Queue open, and Also Running excludes the same set. Failed and interrupted parents stay as recovery work, and so does a completion that reported no Library file: its row is the only place that warning appears.

Video Queue rows use an explicit independent button style inside List; a Delete gesture must not also run Dismiss, the owner re-check or Library navigation. Anchor its confirmation to the exact selected row and owner. Cover real app taps, including Cancel followed by one-press dismissal and relaunch.

**Length and seed (both native apps).** `apps/shared/NativeSequence/NativeLengthControl.swift` is the one Length control. On a Duration machine it is seconds in 0.1 s steps from 1.0 s up to one generation, read exactly ("14.0 s"; a restored off-grid value is shown and sent unchanged), because the server trims to the exact length; its window and overlap come from the video-job PLAN (`VideoLengthGrid.duration`, which is the recommended clip window for every family but H3), with chain limits only until a plan exists. An ordinary render's slider snaps to the model's frame grid (`VideoLengthGrid.resolve`: chain limits, else the recipe; never hard-coded). with **Automatic from prompt** hiding it (Automatic is an ordinary render; Extend always stays a video parent) and **Longer video** switching to the scenes editor, where each scene has a prompt and its own 0.1 s slider from 0.1 s up to the always-one-clip scene length `(single_clip_max_frames - overlap_frames) / fps` floored to 0.1 s, plus Add scene and "Total 38.6 s · 3 scenes". Off, a length within one generation still submits through the Duration flow as a one-clip `/api/video-jobs` parent. A saved length the sliders cannot show becomes equal scenes of the same total (`VideoDurationDraft.fit`), never a silent change. Plan warnings (repeat / shorter than the shortest generation) are shown. `NativeSeedControl` is the one seed control: "Seed", segmented `Random | Fixed`, a string-backed full-range UInt64 field (empty or partial text keeps Fixed and the last valid seed) and a dice New seed button; words and logic are MoldClient's `SeedChoice`. The Duration editor and the Length bounds read `VideoPlanController.shownPlan`, the last answer kept while only the length changes (a different machine, installation, model, fps or Extend source drops it at once, and so does a failed recalculation); submission checks (`openingConflict`, `usesSourceEnding`) read the strict `plan`. The clip line and plan total appear only when the plan joins more than one clip (or when extending), so dragging within one generation never resizes the panel.
