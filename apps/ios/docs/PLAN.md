# Mold Studio Companion — native iOS app plan

## Context

The iPhone app today is the Tauri/Vue app in `apps/mobile` (`com.utensils.mold`). It uses web-view idioms that aren't iOS-native: a custom tab bar, toasts, fixed px sizes, `user-scalable=no` and custom themes. The native macOS app **Mold Studio** (`apps/macos`, `io.utensils.mold.native`, SwiftUI, macOS 26) is now the recommended Mac client, with a clean architecture, a system-only palette and an Apple-style vocabulary.

The goal is a **fully native SwiftUI iOS/iPadOS app, "Mold Studio Companion"**, that feels like the Mac app's little sibling. It works flawlessly from xSmall to AX5 Dynamic Type and ships beside the Tauri app under its own identity. It is remote-only: no engine runs on the phone.

**Locked decisions (user)**

| Area | Decision |
|---|---|
| Platform | iOS 26 minimum (Liquid Glass) |
| Devices | iPhone + iPad adaptive (`.sidebarAdaptable`) |
| Home Screen label | "Mold Studio" |
| v1 scope | Generate (stills, clips, 3-D), Library V3, Queue, Machines + pairing, Models (browse, install, licences), native video + 3-D viewing, Live Activity, widgets, Share extension |
| Share extension | Stages the photo in the App Group; you finish in the app. The extension makes no network calls. |
| Networking | `NSAllowsArbitraryLoads`, with an App Review note that this is a client for self-hosted servers. Every address the Mac accepts works here too. |

**Facts that shape the design**

- `MoldClient` (261 files) is Foundation-only; a lint bans UI imports.
  - Only `MoldHome*` and `SecretStore*` use macOS-only APIs.
  - `playableURL(for:)` and media-token support already exist.
  - Only the pairing **claim** is missing.
- `MoldStyle` needs a single `nsColor` branch.
- The server has **no push, APNs or webhooks**. Live Activities and notifications are therefore driven locally: by the app in the foreground, then by `BGAppRefreshTask`.
- The Tauri app owns `mold://`. The companion must not register it, because when two apps claim the same scheme iOS picks between them unpredictably. Pairing is redeemed by an in-app scanner that parses the unchanged `mold://pair` QR.
- Users re-pair: the Tauri app's Keychain items are in a different access group.

---

## Phase D — Design deliverables (first, before code)

1. **`apps/ios/docs/DESIGN.md`** holds the binding spec. It is the content of §B–§E below, expanded, with a microcopy table.
2. **HTML mockup artifact** "Companion Mockups". Load the `artifact-design` skill first. Frames use iPhone/iPad device frames, system-like type scaled per Dynamic Type size, and light and dark.

| # | Frame |
|---|---|
| 1 | Generate: idle with result (Large, light) |
| 2 | Generate: running, with denoise preview and progress sentence (dark) |
| 3 | Generate: keyboard up, with source + image 1 / image 2 wells |
| 4 | Generate at AX5: collapsed Options, estimate stacked over a full-width Generate |
| 5 | Generate at xSmall (dark) |
| 6 | More options sheet at medium detent (Sampler, Output) |
| 7 | Library grid: day sections, title menu open |
| 8 | Library at AX3: 2 columns, `is:video` search token (dark) |
| 9 | Viewer with the Info sheet at the 35% detent |
| 10 | Queue: running batch parent/children, plus a held row with Pull / Retry |
| 11 | Queue held row at AX5: stacked full-width buttons |
| 12 | Machines: fleet cards, one offline (dimmed, with its reason), plus Nearby (dark) |
| 13 | Add a Machine: QR scanner, plus a manual address checked live |
| 14 | Models: a Discover row downloading, plus the licence sheet |
| 15 | iPad landscape: sidebar (Library shelves, Machines) with the Library grid (dark) |
| 16 | Live Activity (Lock Screen, Dynamic Island compact / minimal / expanded), widgets S/M, rectangular accessory |

Review the artifact with James before M1 code lands. Iterating on the mockups is cheap; iterating on SwiftUI is not.

---

## §A Architecture

**Layout**

- `git mv apps/macos/Packages/{MoldClient,MoldStyle} apps/shared/Packages/`
- Add the new `apps/shared/Packages/MoldMesh`.
- The app lives in `apps/ios/`. It is XcodeGen-generated from `project.yml`, and the `.xcodeproj` is gitignored.
- Settings mirror the Mac:
  - Swift 6
  - `SWIFT_STRICT_CONCURRENCY: complete`
  - `SWIFT_DEFAULT_ACTOR_ISOLATION: MainActor`
  - Swift Testing
  - `include: Version.yml`
  - Team `28X9H69QGE`

**Package changes (M0)**

- `platforms: [.macOS("26.0"), .iOS("26.0")]`.
- Put `#if os(macOS)` around `MoldHome*`, `SecretStore+File/+Local` and `SecretStore.applicationSupport`.
- New `CredentialStore` protocol:
  - The Mac's file `SecretStore` conforms to it.
  - New `KeychainCredentialStore`:
    - `AfterFirstUnlockThisDeviceOnly`
    - service `io.utensils.mold.companion.remote-api-key`
    - shared access group
    - every `OSStatus` becomes a typed error, so a locked keychain never reads as "no key"
- `MoldStyle/Chrome.swift:38` gets a `UIColor.separator` branch.
- `lint-layers` also bans `UIKit` in MoldClient.
- New `MoldClientTesting` library product: a public-API `FakeBackend` for both apps. The Mac's existing fake stays until it is migrated opportunistically.
- **MoldMesh**: move `MeshRenderer*`, `MeshScene`, `MeshPayload` and `MeshShaders.metal` out of `apps/macos/Sources/Mold/Mesh`.
  - The Mac keeps its `NSView`.
  - iOS adds an `MTKView` `UIViewRepresentable` with gestures: pan to orbit, pinch to zoom, two-finger pan to truck.
  - Fallback if SwiftPM metallib bundling fights XcodeGen: compile the files into both targets by path.

**Targets and IDs**

| Target | Bundle ID | Entitlements |
|---|---|---|
| MoldCompanion (app, `PRODUCT_NAME` Mold Studio) | `io.utensils.mold.companion` | App Group (keys in the app's own Keychain group) |
| MoldCompanionWidgets (WidgetKit + ActivityKit) | `…companion.widgets` | App Group only; no networking or Keychain (enforced by lint) |
| MoldCompanionShare | `…companion.share` | App Group only (staging; no network) |
| MoldCompanionTests (hosted) / MoldCompanionUITests | — | — |

- App Group `group.io.utensils.mold.companion` holds:
  - `hosts.json` (no secrets)
  - `recent-prints.json` plus downsampled JPEGs
  - `pending-batches.json`
  - `share-inbox/`
- `apps/ios/Sources/Shared/` holds the App Group paths, `RecentPrintsSnapshot` and `GenerationActivityAttributes`. It compiles into all targets.
- Info.plist:
  - `CFBundleDisplayName` = "Mold Studio"
  - `NSLocalNetworkUsageDescription`
  - `NSBonjourServices [_mold._tcp]`
  - `NSCameraUsageDescription`
  - `NSPhotoLibraryAddUsageDescription`
  - `NSAllowsArbitraryLoads`
  - `NSSupportsLiveActivities`
  - `BGTaskSchedulerPermittedIdentifiers [io.utensils.mold.companion.refresh]`
  - `UIBackgroundModes [fetch]`
  - `CFBundleURLTypes [moldstudio]`
  - multi-scene on iPad
- URL scheme `moldstudio://` is used for widget, notification and Live Activity links:
  - `moldstudio://print/<host>/<file>`
  - `moldstudio://queue/<job>`
  - `moldstudio://generate?inbox=<id>`
- Deferred: a universal-link pairing QR (`https://utensils.io/mold/pair#…`). It is cross-product work.

**MoldClient additions (TDD'd)**

- `MobilePairingPayload.parse(_:)` is a byte-level port of `parseMobilePairingPayload` in `studio/api/pairing.ts`. It accepts both the JSON and the `mold://pair` form, requires `version == 1` and an `http(s)` `base_url`, and round-trips with the existing `url` producer.
- `claim(token:clientName:clientKind:)` calls `POST /api/pairing/claim`. The request carries no key and goes through `RedirectGuard`. The contract is in `crates/mold-server/src/auth.rs:399`. The caller verifies `instance_id` and refuses a mismatch. `client_kind` is `iphone` or `ipad`, passed in by the caller.
- Thumbnails:
  - `ThumbnailKey(instanceID, filename, mediaVersion, size)`
  - a `DiskThumbnailStore` actor with LRU caps of 4,000 items, 256 MiB, and 2 MiB per item
  - evict by host or by file
  - prints with no `mediaVersion` are memory-only
  - the app-side `ThumbnailLoader` adds `NSCache<CGImage>`, in-flight dedupe, ImageIO downsampling at display scale, and visible-first priority
- Video: `AVPlayer(url: playableURL)`. Re-mint the token when the item fails after 15 minutes. Never buffer a whole file.
- Mesh: fetch the GLB with the auth header, then `GLBMesh` into MoldMesh.

**State and lifecycle**

- `CompanionStores` is the composition root and the only place allowed to call `HTTPBackend(` (enforced by lint). Its order:
  1. `HostStore`
  2. Library, Queue, Model, Download, Catalog and Licence stores
  3. `GenerateController`
  4. `PairingStore`
  5. `ActivityCoordinator`
  6. `WidgetSnapshotWriter`
- All stores are `@MainActor @Observable` and injected through `.environment`.
- **Phone stores are written fresh.** Any decision logic longer than about 10 lines that the Mac also has moves down into MoldClient as a pure public function, with its test, in the same PR. A MoldStores package is revisited after v1.
- Errors go to the `HostStore.failures` inline banner, never a toast or modal.
- `ConnectionSupervisor` follows `scenePhase`:
  - On `.active`: start `/api/events` watchers, reattach `batchEvents` for each pending batch, then reconcile: queue, then batch status, then library ETag relist, then widget snapshot.
  - On `.background`: cancel every SSE task, persist pending batches, and push a final Live Activity update.
- Submission runs inside `beginBackgroundTask`, covering preprocessing, upload and `POST /api/generation-batches`. The task always ends on admission, failure, cancellation or expiry.
- `BGAppRefreshTask` runs roughly every 15 minutes while any batch is pending. It polls status, posts deduped local notifications, updates or ends Live Activities, and refreshes the widget cache.
- Live Activity:
  - `ContentState` is under 4 KB: stage, step/total, position, machine name. The preview is referenced by App Group path, never embedded.
  - Updates are coalesced to about 1 per second.
  - On background, `staleDate` is set to the ETA plus 5 minutes, and the stale view says "Open Mold Studio to refresh".
  - Dismissed 15 minutes after the render finishes.
  - Gated at runtime on iPad.
- Share inbox: the extension writes the photo, downscaled to 2048 px with alpha kept (HEIC becomes PNG/JPEG), plus a small JSON manifest. On foreground, Generate shows a "From Share" card that offers Start from / Reference / Add to Library.

---

## §B Information architecture

**Principles**

- The Mac's words, SF Symbols and radii (panel 16, well 10, card 8, field 6, tile 5).
- System colours only, and the user's accent colour.
- Plain words in sans, technical detail in mono on the same row. At large sizes the mono detail wraps underneath.
- The model's capability block decides which controls appear. Switching model **parks** attachments rather than losing them.
- Say it or don't draw it: no dashes. An offline machine keeps its place, dimmed, with its reason.
- Generate never turns into Stop.

**iPhone tabs** (`TabView`, `.sidebarAdaptable`)

| Tab | SF Symbol | Notes |
|---|---|---|
| Generate | `wand.and.sparkles` | |
| Library | `photo.on.rectangle.angled` | |
| Queue | `list.bullet.indent` | Badge counts running and held jobs |
| Machines | `server.rack` | Also holds Models |
| Search | `Tab(role: .search)` | Library search |

- **Models** is reached from Machines: a "Models" row for the default machine, and Machine detail ▸ Models. On the Mac, the Models pane already follows the chosen machine.
- **Settings** is a sheet opened from a gear in the Machines toolbar.

**iPad sidebar** (`TabSection`s mirror the Mac sidebar)

- Generate
- Library: All Prints, Favourites, the collections plus New Collection…, Recently Deleted
- Queue
- Models
- Machines: one row per machine with a status dot, plus Add a Machine…
- Settings, in the sidebar footer

**State**

- One `NavigationStack(path:)` per tab.
- `@SceneStorage` keeps the tab, paths, shelf, sort and tile size.
- The Generate draft is also saved to disk.
- Handoff with the Mac viewer through `NSUserActivity`.

---

## §C Screens (summary; the full text goes in DESIGN.md)

### Generate

**Toolbar**

- Principal: a Model menu. The plain name is `.headline`; the ID is mono `.caption`.
- Trailing: a Machine menu (dot plus name; Auto follows the default machine).
- Kind switch (Still picture / Short clip / 3-D object): a menu on iPhone, segmented on iPad.

**Canvas.** Fills the space. It uses the AlphaBed checkerboard behind transparent prints. When empty it shows guidance.

**Composer.** A glass panel attached with `.safeAreaBar(edge: .bottom)`. It contains:

- A picture-well strip, shown only when the model uses it:
  - Source picture, captioned "Start from"
  - References numbered "image 1, image 2…", matching how the prompt refers to them
  - Each well's menu: Photos, Camera, Files, Choose from Library…, Paste
- The prompt: vertical `TextField`, 1 to 6 lines.
- Expand, with a "Suggest other ways" menu.
- A chip row: Shape, Steps, Batch, Length (clips only), More options.
- The final row: the estimate in mono at the leading edge; **Generate** at the trailing edge (`.glassProminent`, large, ⌘↩).

**More options sheet** (medium and large detents, a `Form` mirroring the Mac inspector)

- Adapters
- Identity
- Refine: ControlNet, and a full-screen PencilKit mask editor
- Clip
- Sampler: guidance, seed ("Repeat this look"), negative prompt
- Output: format, Transparent background, upscaler, Save
- File under: title, tags, collection
- Recent prompts

A section appears only when the model's capabilities allow it. Shape, Steps and Batch are repeated here.

**While running**

- The denoise preview shows under a glass plate with a sentence such as "Adding detail — about 12s left · `denoise 18/28`".
- A progress bar with step marks.
- A small Stop, with a menu item "Stop everything from here".
- "+2 waiting" when more batches are queued.

**Result**

- Batch results page horizontally.
- A glass action bar: Save to Photos, Share, Copy, Favourite, Show in Library. The overflow menu adds Use These Settings.
- A clip plays inline with `VideoPlayer`.
- A 3-D print opens in the MoldMesh viewer. It auto-turns unless Reduce Motion is on, and shows the poster plus one line if it can't be drawn.

**Other states**

- A gated model shows a large licence sheet before the pull.
- 3-D recipes that ignore the prompt say so in words.

### Library

**Grid**

- `LazyVGrid(.adaptive(minimum: @ScaledMetric))`; pinch snaps between 3 tile sizes.
- Pinned day headers.
- Tiles show a host badge (only when there's more than one machine), a duration for video, a cube for 3-D, and a star for favourites. Duplicates merge into one tile.

**Toolbar**

- `.toolbarTitleMenu` switches the shelf: All Prints, Favourites, Collections ▸, Recently Deleted.
- Select.
- A ⋯ menu with Sort By, Tile Size and Machine.

**Search.** `.searchable(text:tokens:)`. Tokens are Videos (`is:video`), 3-D (`is:mesh`), `tag:` and `on:`, suggested as you type.

**Selection.** A bottom toolbar with Share, Favourite, Add to Collection, Tags and Delete. In Recently Deleted: Put Back and Delete Immediately, the latter with a confirmation dialog.

**Viewer**

- Opens with a `.navigationTransition(.zoom)` push.
- Pages in grid order; pinch or double-tap to zoom; swipe down to close; tap toggles the chrome.
- Bar: Share, Favourite, Info, Delete.
- Menu: Use These Settings, Save to Photos, Copy, Add to Collection, Rename, Export (the formats this machine advertises).
- **Info sheet** (35% and large detents, background interaction on): title, prompt, Model · `id`, Machine, then Sampler, Conditioning, Clip, Mesh and File sections, each shown only when it has content. Every row has Copy.

**Context menus** use the Mac tile menu's items and order, with the destructive item last after a divider. There are no swipe actions in the grid. Recently Deleted tiles show a mono countdown.

### Queue

- A `List` grouped by machine.
- A **running row** shows a 52pt scaled thumbnail, the sentence, a progress bar and a mono ETA.
- A **batch** is a parent row with a `DisclosureGroup` of its children.
- A **held row** is written as a paragraph, with Pull (the licence comes first if the model is gated), Retry and Move to….
- Swipe actions: Cancel on the trailing edge; Pause / Resume on the leading edge, only where the machine supports it.
- Context menu: Move Up, Move Down, Move to…, Pause, divider, Cancel Job.
- Toolbar: Edit (reorder), plus a ⋯ menu with Pause, Resume and Empty Queue…, each also offered for All Machines.
- Pull-to-refresh reconciles.
- A row rendering on a machine that can't stop at a safe point has no Cancel.

### Machines

**List**

- Fleet cards: status dot, name, a Default badge, "Ready · 0.31.0", "4× NVIDIA L40S", the load figure, a VRAM `Gauge`, memory, "3 queued · 14 installed", and the mono address.
- An offline machine is dimmed and shows its reason.
- Card context menu: Open, Check Now, Set as Default, Copy Address, Edit…, divider, Remove….
- A **Nearby** section from `NWBrowser` on `_mold._tcp`.

**Detail.** Overview; GPUs, each with a VRAM gauge and a toggle where the machine honours one; Memory and CPU; Queue; Models; Address.

**Add a Machine** (a sheet)

1. **Scan a Pairing Code.** VisionKit `DataScannerViewController` for QR. Caption: "On your Mac: Machines ▸ your machine ▸ Pair a Phone…". Also a "Paste pairing link" field.
2. **Nearby.**
3. **Enter an Address.** Checked live, with the result as a sentence. Optional key in a `SecureField`, saved to the Keychain.
4. **Confirm.** Name, and a "Make this the Default machine" toggle.

**First run.** Machines shows a `ContentUnavailableView`: "No machines yet — Mold makes pictures on a computer you own." with Scan (prominent) and Enter an Address.

### Models (per machine)

- A segmented control switches Installed and Discover.
- **Installed:** grouped by family. Swipe to Delete; menu with Load, Unload, Repair, Components, then Delete…. The disk figure is in the footer, and active downloads are listed at the top.
- **Discover:** searchable, with Family and Sort filters. Each row shows Get, Installed, or Open Page.
- A download shows "2.1 / 11.8 GB · 42 MB/s" and Cancel Download. It re-attaches to the download stream when the app comes back to the foreground.
- A gated model opens the licence sheet.

### Settings (sheet)

- Machines
- Generation defaults
- Library: auto-save to Photos (off by default), cache size, Empty Now
- Notifications: Finished, Failed, Held
- Live Activities
- About, including the privacy link

There is no appearance setting; the app follows the system.

### Live Activity and Dynamic Island

| Placement | Content |
|---|---|
| Compact | `wand.and.sparkles`, plus a progress ring |
| Minimal | The progress ring |
| Expanded / Lock Screen | Preview thumbnail; the sentence; `Text(timerInterval:)` ETA; a 2-line prompt; a bar with a mono "denoise 18/28 · workstation"; a Stop button (`LiveActivityIntent`) |

When finished, it shows the thumbnail with "Finished on workstation" and View.

### Widgets

| Family | Content |
|---|---|
| Small | The latest print, full-bleed |
| Medium | 4 recent prints |
| Large | 3×3 grid |
| Accessory rectangular | "2 rendering · 1 held" |
| Accessory circular | Progress |
| Accessory inline | Queue summary |

- Configurable by machine and by shelf.
- Prints use `.widgetAccentedRenderingMode(.fullColor)`.

### Share extension

1. A SwiftUI sheet shows the preview.
2. "Use as" choice: Start from, Reference image, or Add to Library.
3. The photo is staged in the App Group. The sheet confirms: "Waiting in Mold Studio".

### Notifications

- Finished: a thumbnail attachment, with View and Favourite.
- Held: "Pull and Retry" and View.
- Failed: View.
- Threaded per machine.
- Suppressed in the foreground, where the inline banner covers it.

---

## §D Dynamic Type and accessibility (binding)

**Rules**

- **Text styles only.** A lint bans `.font(.system(size:` and literal point fonts, the same way the Mac lint bans literal colours. Figures use `.monospacedDigit()` or `.monospaced()`.
- `@ScaledMetric(relativeTo:)` for thumbnails (Queue 52), tile minimums (96/128/180), wells (72) and spacing. Clamp only with a maximum.
- When `dynamicTypeSize.isAccessibilitySize`, `AnyLayout` switches every label/value row, card header and the composer's final row from H to V.
- The chip row uses `ViewThatFits`: full chips, then icon plus short label, then one "Options" button.
- Never truncate meaning: `lineLimit(nil)` on every label and sentence. Only prompt previews in tiles and rows may truncate, and they stay readable in full to VoiceOver and the Large Content Viewer. Never use `minimumScaleFactor`.
- Custom glass buttons get `.accessibilityShowsLargeContentViewer`. Targets are at least 44pt.
- The composer is capped at 55% of the height and scrolls inside.

**VoiceOver**

- A tile is one element, e.g. "Print, a lighthouse at dusk, Today 14:02, workstation, favourite", with custom actions.
- Rotors for Day and Favourites.
- Progress: the accessibility value says "18 of 28, about 12 seconds left", with an announcement every 25%.

**Other settings**

- Reduce Motion: no auto-turn, cross-fade instead of zoom.
- Reduce Transparency: handled by the system glass.
- Increase Contrast: dimmed items use `.secondary`, never opacity below 0.6.
- Colour is never the only signal.

**Layout per size**

| Screen | xSmall | Large | xxxLarge | AX5 |
|---|---|---|---|---|
| Generate | 1 chip row | Chips wrap | Icon + short label, prompt ≤4 lines | One Options button; estimate above a full-width Generate; model ID moves into the menu |
| Library | 5 columns | 3 | 3 | 1–2; host badge only in VoiceOver and Info |
| Queue | Thumbnail beside text | same | same | Thumbnail above text; held buttons stacked full-width; ETA on its own line |
| Machines card | Dense | Dense | Figures wrap | Every pair in a VStack; gauge full width |
| Models row | 1 line | Size trails | Size trails | Name / sentence / size stacked; Get full width |

---

## §E Interactions

**Haptics**

| Event | Haptic |
|---|---|
| Finished, foreground | success |
| Accepted | light impact |
| Held or failed | warning |
| Stepper and pinch snaps | selection |

**iPad drag and drop:** drag prints out as `Transferable`; drop onto a well or the canvas; drop onto a sidebar collection to file; drop into Library to import; reorder the Queue.

**Keyboard shortcuts** (as on the Mac)

| Shortcut | Action |
|---|---|
| ⌘1–⌘5 | Tabs |
| ⌘↩ | Generate |
| ⌘F | Search |
| ⌘R | Refresh |
| ⌘, | Settings |
| ⌥⌘I | Info |
| ⌥⌘F | Favourite |
| ⌘⌫ | Delete |
| ⌘Z | Undo |
| ← / → | Previous / next print |
| Esc | Close the viewer |
| ⌘+ / ⌘− | Tile size |
| ⌘E | Expand |

---

## Milestones (serialized PRs; TDD: failing test first)

| # | PR | Verify |
|---|---|---|
| D | DESIGN.md plus the mockup artifact (16 frames); review with James | Artifact renders in light and dark, iPhone and iPad frames |
| M0 ✅ | Move packages to `apps/shared`; iOS 26 platform; `#if os(macOS)` on `MoldHome*` only (`SecretStore.applicationSupport` made portable instead, because `DraftStore` needs it on the phone); `CredentialStore` protocol keyed by host UUID, with `SecretStore` conforming; MoldStyle `hairline` branch; `lint-layers` also bans UIKit | Mac `make lint test` green; MoldClient (996 tests) and MoldStyle green on the iOS 26.5 sim. Two wall-clock `RefusalBodyTests` bounds widened: they measured parallel-run starvation on the sim |
| M1 ✅ | `apps/ios` skeleton: 5 targets (app, WidgetKit, Share, unit tests, UI tests), entitlements, Info.plist, `.sidebarAdaptable` shell (Models only at regular width: iPhone ignores `defaultVisibility` and pushed Machines into "More"), `EmptyState` + `RowAxis`, Settings sheet, Go menu ⌘1–⌘5, Makefile, devshell `companion-*`, `ios-native.yml`, `.claude/rules/ios-native.md`, `apps/shared/scripts/swift-lint.sh` (+ its own test) used by both Makefiles | 7 unit tests; `make packages-test` (997 + 11 on iOS); `make uitest` audit clean at xSmall/Large/AX5 in light and dark after three real fixes (AccentColor, ProminentFill, SecondaryText) |
| M2 ✅ | Machines: `KeychainCredentialStore` (app access group only: the Share extension never networks, so no shared Keychain group), `MoldClientTesting.FakeBackend` (97 routes, generated), `HostStore` (+ Editing, Reachability, Events, Pairing), `ConnectionSupervisor` (scene phase), `NearbyBrowser` (`NWBrowser`, TXT `id=` dedupe), fleet cards, detail with per-GPU switches, Add/Edit sheets, per-window `AppRouter`; `ServerStatus.hardware` and `DeviceWords` moved into MoldClient | FakeBackend + in-memory credential store tests; hosted Keychain tests; audit on the new screens |
| M3 ✅ | Pairing: `MobilePairingPayload.parse` (byte port of `pairing.ts`), `HTTPBackend.claimPairing` (no key sent, 401 → expired, instance check), DataScanner + paste, re-key instead of duplicate; `testflight-ios-native.yml` gated on `vars.COMPANION_TESTFLIGHT` until the App Store Connect record exists (Apple has no API to create one); `make archive` refuses an iconless archive | Parser + claim URLProtocol tests; round trip with the Mac `url` producer; shipped with M2 in one PR |
| M4 ✅ | Library read: `LibraryStore` (ETag listings per machine, `LibraryMerge`, collections by slug, Recently Deleted, event-driven reloads), `ThumbnailLoader` + `DiskThumbnailStore` (MoldClient; 4,000 / 256 MiB / 2 MiB LRU, media-version keys), day-sectioned grid with pinch tile sizes, shelf title menu, search tokens, zoom-transition viewer (UIScrollView zoom, `playableURL` clips, `MoldMesh` 3-D), Info sheet from `PrintDetails`; `MoldMesh` package extracted from the Mac (shaders compiled at runtime from source: `swift build` never makes a default metallib); `LibraryScope` + `GenerationRecipe.makes` moved into MoldClient | 1,044 MoldClient + MoldMesh tests; Mac `build-for-testing` green; LibraryStore tests; audit on the Library |
| M5 ✅ | Library V3: favourite / tags / title / collections through the shared `MutationOutbox` on EVERY copy (optimistic, 1s/2s/4s, operation-id fenced, re-read on give-up), one-step undo from `PrintEdit.inverse`, trash / Put Back / Delete Immediately / Empty (a machine without a trash deletes for good), selection bar, `PrintMenu` in the Mac order, Tags / Rename / New Collection sheets (`CollectionShelf.slug` ported from `collection_slug`), Share, Save to Photos (add-only), Copy | Fan-out + undo + refusal tests against FakeBackend |
| M6 ✅ | Generate stills: capability-driven picker (`GenerationRecipe`, `RenderDraft.adopting`, shared refusal sentences), picture wells, More options (Adapters / Identity / Refine with the PencilKit mask), finite background task around submission only, batch SSE + 700 ms preview poll, `PendingLedger` in the App Group, draft saved on Generate and on background | Admission and ledger tests; AX5 audit on Generate |
| M7 ✅ | Clips and 3-D generate (`GenerationRecipe.makes`), Camera / Photos / Files / Library sources, Use These Settings from the viewer and Library | Capability-gating tests |
| M8 ✅ | Queue: sections per machine, batch parents (`QueueGroup`), running rows with preview + `ProgressWords`, held rows in words (`QueueHold`) with Pull and Retry / Retry / Move to… (`TransferPlan`), swipe Cancel / Pause, Move Up/Down and Edit reorder via `QueueOrder`, Pause / Resume / Empty Queue… per machine and for all, Queue tab badge; `QueueStoreTests` | Supervisor phase tests |
| M9 ✅ | Models: Installed by family (Load / Unload / Repair / Components / Delete with the shared removal sentence), Discover (debounced catalog search, family + sort, paging), download stream reduced by the shared `DownloadBoard` (moved into MoldClient; the Mac uses it too), licence sheet then retry; Models row in Machines and machine detail; `DownloadBoardTests`, `ModelStoreTests` | Download reducer tests |
| M10 ✅ | Live Activity (`ActivityCoordinator` over `RunState`, ~1 update/s, stale date = estimate + 5 min, ended 15 min after, `StopRenderIntent` handled in the app process), local notifications (`Notifier`: once per batch, threaded per machine, never in the foreground, View / Favourite), `BGAppRefreshTask` over `PendingLedger`, `moldstudio://` deep links, Settings: Library (auto-save, cache size, Empty Now), Notifications, Live Activities; `ActivityProjectionTests`, `NotifierTests`, `DeepLinkTests` | `ActivityProjection` tests; Simulate Background Fetch UAT |
| M11 ✅ | Widgets from the App Group snapshot (`WidgetSnapshotWriter` → `WidgetSnapshot` + 360 px JPEGs, timelines reloaded only on change): Recent Prints S/M/L configurable by machine and favourites, Queue accessory rectangular / circular / inline, Live Activity UI; `WidgetSnapshotTests` | Encoder tests; previews of every family |
| M12 ✅ | Share extension (`ShareInbox.stage`: ImageIO thumbnail to 2048 px, alpha kept as PNG, no network) plus the "From Share" card in Generate (Start From / Reference / Add to Library / Discard); `ShareInboxTests` incl. the 48 MP peak-memory ceiling | Memory-ceiling test with a 48 MP photo |
| M13 ✅ | Docs: `apps/ios/README.md` (features, background behaviour, distribution), root README pointer, CLAUDE.md apps line, website `guide/companion.md` + nav, privacy policy (camera, local network, Photos, notifications, App Group, Keychain); the external TestFlight group is an owner step once the App Store Connect record exists | Doc review; `cd website && bun run build` |

Each user-visible PR gets a `changelog.d/companion-*.md` fragment. M0, M1 and the CI-only PRs use `skip-changelog`. PRs are serialized: one open at a time, rebased right before merge.

Goal (set by James, 2026-09-27): implement every milestone in full, with nothing deferred. The learning-mode contribution points were dropped under that goal; `RowAxis` shipped its tested rule (stack from AX1) in M1.

## Critical files

- `apps/shared/Packages/MoldClient/Package.swift`, `Sources/MoldClient/{SecretStore,CredentialStore,MobilePairingPayload,MediaToken}.swift`
- `apps/shared/Packages/MoldStyle/Sources/MoldStyle/Chrome.swift`
- `apps/macos/project.yml`, `apps/macos/Makefile`: package paths and lint extraction
- `apps/macos/Sources/Mold/Mesh/*`: MoldMesh extraction
- `apps/macos/Sources/Mold/AppStores.swift`: composition-root pattern to mirror
- `apps/macos/Sources/Mold/Shell/RootView.swift`, `Machines/DeviceWords.swift`, `Shell/HostStatus.swift`, `Queue/QueueHoldRow.swift`, `Generate/PromptPanel.swift`: vocabulary and behaviour to port
- `studio/api/pairing.ts`, `crates/mold-server/src/auth.rs`: the pairing contract
- `.github/workflows/testflight-ios.yml`, `.github/workflows/macos-native.yml`: CI templates and path filters

## Verification (end to end)

- `make -C apps/macos test lint` stays green after M0 and every later PR.
- `make -C apps/ios lint test uitest` runs:
  - Swift Testing units
  - an XCUITest sweep at `UICTContentSizeCategoryExtraSmall`, `Large` and `AccessibilityExtraExtraExtraLarge`, on the iPhone SE (3rd gen), iPhone 17 Pro Max and iPad Pro 13″ sims, calling `performAccessibilityAudit(for: [.dynamicType, .textClipped, .hitRegion, .contrast, .sufficientElementDescription])` on every main screen, with screenshots attached
- UAT on a real iPhone:
  1. Pair with the Mac app's QR.
  2. Generate a still, a clip and a 3-D object on hal9000 / workstation.
  3. Lock the phone mid-render and watch the Live Activity (it goes stale while backgrounded).
  4. Receive the finished notification.
  5. Widget shows the new print.
  6. Share a Photos picture, then find it waiting in Generate.
  7. Favourite, tag, trash and Put Back across two machines.
  8. Repeat the core flows at AX5 with VoiceOver on.
- CI: `ios-native.yml` green, and the TestFlight build reaches `VALID`.

## Risks

- MoldClient may contain more macOS-only APIs than found so far (likely `GalleryImport`, `DraftStore`). M0 finds them by compiling.
- Metal shaders in SwiftPM through XcodeGen: the by-path fallback is noted above.
- Live Activities go stale while backgrounded because there is no push. The copy says so honestly. APNs would need server work and is a possible later project.
- The phone and Mac stores can drift apart. Mitigation: the "logic moves down into MoldClient" rule, plus a MoldStores review after v1.
- A second App Store Connect app record, and the App Review justification for `NSAllowsArbitraryLoads`, are one-time owner steps.
