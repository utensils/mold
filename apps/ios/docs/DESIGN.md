# Mold Studio Companion — design specification

Status: draft for review (Phase D of [PLAN.md](PLAN.md)). This spec is binding
once accepted. Where it and a mockup disagree, the spec wins. Where the spec is
silent, follow the native Mac app's behaviour (`apps/macos/README.md`, "What
works today").

The visual reference is the "Mold Studio Companion Mockups" design canvas, 16
frames (<https://claude.ai/artifact/UaYj5f1YMaVuCYne4nLFRj>, private to the
owner until shared). They cover iPhone and iPad, light and dark, and Dynamic Type from
xSmall through AX5.

## 1. What this app is

Mold Studio Companion is the native iOS/iPadOS app for mold. It is SwiftUI, needs
iOS 26 or later, and runs in remote-only mode, driving the machines you already
own. It is the Mac app's little sibling: it has the same destinations, words,
symbols and behaviour, rearranged for a hand. It installs beside the Tauri iPhone
app (`com.utensils.mold`) and does not replace it.

| | |
| --- | --- |
| Product name | Mold Studio Companion |
| Home Screen label (`CFBundleDisplayName`) | Mold Studio |
| Bundle ID | `io.utensils.mold.companion` (`.widgets`, `.share` extensions) |
| URL scheme | `moldstudio://` — never `mold://`, which the Tauri app owns |
| Minimum OS | iOS / iPadOS 26.0 |
| Devices | iPhone and iPad; the iPad uses a sidebar and allows multiple windows |

## 2. Principles

1. **The Mac app's little sibling.** Use the Mac's words, SF Symbols and radii
   (panel 16, well 10, card 8, field 6, tile 5). Someone who knows Mold Studio on
   the Mac should find everything where they expect it.
2. **Plain words first, technical detail beside them.** A plain sentence is the
   primary text; the technical figure sits beside it in secondary monospaced
   text. Example: "Adding detail — about 12s left · `denoise 18/28`". When the row
   has to wrap at large text sizes, the mono part moves to its own line. It is
   never clipped.
3. **The model decides what is shown.** A control appears only when the model's
   generation profile (`recipes[].capabilities.*`) says the model reads it.
   Switching model **parks** attachments rather than dropping them. A parked well
   says "(not used)" in words. When a capability block is missing, that means an
   older server, not a refusal.
4. **Say it, or don't draw it.** A figure nothing has reported is not drawn; never
   show a dash. A machine that is not answering keeps its place, dimmed, and says
   why. Errors appear as a dismissable inline banner, never a toast or a modal
   alert.
5. **Everything is standard iOS.** Use the system `TabView`, `NavigationStack`,
   sheets with detents, `.searchable` with tokens, context menus, swipe actions,
   `ShareLink`, `PhotosPicker`, and Liquid Glass through system chrome and
   `glassEffect`. No custom tab bars, fixed point sizes, custom themes or literal
   colours. The app follows the system appearance and the user's accent colour.
6. **Every text size is designed.** Each screen has a stated layout for xSmall,
   Large, xxxLarge and AX5 (§6).
7. **The picture is the content.** Chrome is glass and floats; the canvas and the
   grid get the pixels.
8. **Generate is always Generate.** It never turns into Stop. Pressing it again
   queues another batch.

## 3. Lexicon

The native Mac app's vocabulary governs. `docs/design/README.md`'s web lexicon
("My images", "Styles") does **not** apply to this app.

| Say | Never say |
| --- | --- |
| Generate | Make, Create, Submit, Render |
| Library · Prints · All Prints | My images, Images, Gallery |
| Favourites | Favorites (the Mac spelling governs) |
| Collections | Albums |
| Recently Deleted · Put Back · Delete Immediately | Trash (in the UI), Restore, Purge |
| Queue · Waiting · Being made · Held · Finished | queued, active, blocked, done |
| Models · Installed · Discover · Get · Pull | Styles, checkpoints |
| Machines · Default · Nearby · Add a Machine… | Hosts, servers |
| Pair a Phone… (on the Mac) · Scan a Pairing Code (here) | QR login |
| Shape (aspect and size) · Steps · Batch · Length | Resolution, frames |
| Expand · Suggest other ways | Enhance, AI rewrite |
| Use These Settings | Remix, Reuse |
| Start from (source picture) · image 1, image 2 (references) | img2img, init image |

## 4. Information architecture

### iPhone: `TabView` + `.tabViewStyle(.sidebarAdaptable)`

| Tab | Symbol | Notes |
| --- | --- | --- |
| Generate | `wand.and.sparkles` | |
| Library | `photo.on.rectangle.angled` | The shelf is picked from the title menu |
| Queue | `list.bullet.indent` | Badge = running + held count; no badge at 0 |
| Machines | `server.rack` | Holds Models, and the Settings gear |
| Search | `Tab(role: .search)` | Library search; the separate glass button |

- **Models is not an iPhone tab.** On the Mac, "the machine picked here is the
  one the Models pane shows", so Models already belongs to a machine. The
  Machines list starts with a "Models" row for the Default machine, and Machine
  detail ▸ Models opens any other machine's.
- **Settings** is a sheet, opened from the Machines toolbar gear (and ⌘,); the
  iPad sidebar also lists it as a row that shows it as a page.

### iPad sidebar (mirrors the Mac sidebar through `TabSection`)

```
Generate
Library            (All Prints)
Queue
Models             (follows the Default machine; switch in its toolbar)
Machines
Search
Settings           (a page here; a sheet everywhere else)
Shelves            Favourites · each collection (drop prints on one to file them) · Recently Deleted
Your Machines      workstation · hal9000 "Offline" · studio-mini "Key"
```

The floating tab bar (the sidebar put away) carries only the five
destinations and Search. The sections live in the sidebar alone: listed in
the bar they made UIKit page and re-lay it out on every text-size change
until the app stopped answering. A machine that is not answering says so in
words beside its name, never by colour alone.

### State and links

- Each tab has its own `NavigationStack(path:)`. `@SceneStorage` keeps the
  selected tab, each tab's path, the Library shelf, sort, and tile size.
- The Generate draft is also written to disk, so it survives the app being
  terminated.
- iPad windows keep independent state.
- Deep links:

  | Link | Opens |
  | --- | --- |
  | `moldstudio://print/<host>/<filename>` | Viewer |
  | `moldstudio://queue/<job>` | Queue, scrolled to that row |
  | `moldstudio://generate?inbox=<id>` | Generate, with the shared photo |

  They are used by widgets, notifications, the Live Activity and App Intents.
- Handoff (`NSUserActivity`) continues a viewed print on the Mac and back.

## 5. Screens

### 5.1 Generate

**Model selection**

The composer has a full-width **Model** button with a wrapping model name and
chevron. It opens a searchable **Choose a Model** sheet. The sheet groups
installed models by family and includes Kind, Machine, Recipe, and a direct
**Get More Models…** navigation link. These controls no longer compete for a
fixed-height navigation title; the navigation bar says Generate. Kind uses a
menu on iPhone and at accessibility sizes, and a segmented control on a roomy
iPad. Auto follows the default online machine that holds the selected model.

**Canvas.** Fills the rest of the screen.

- Empty: a faint `wand.and.sparkles` and "Describe a picture below."
- Transparent prints sit on the AlphaBed checkerboard.
- Tapping the canvas while the keyboard is up dismisses it.

**Composer.** An opaque system-background panel (radius 16) above the tab bar
keeps text legible over the canvas. One stable scroll
view is capped to 55% of the current window's available height, including
keyboard avoidance and iPad resizing; its prompt never moves between
`ViewThatFits` alternatives. Top to bottom:

1. **Prompt:** `TextField(axis: .vertical)` with up to six lines (three at
   accessibility sizes). Keyboard toolbar: Expand, Done.
2. **Expand** (`text.badge.star`), beside the prompt. Tapping it rewrites the
   prompt in place, and the original is kept for undo. Its menu offers
   "Suggest other ways", which opens a list sheet with Use.
3. **Model** opens the chooser described above.
4. **Picture wells**, shown only when the recipe reads them:
   - **Start from** (the source picture). Strength lives in More options.
   - **image 1, image 2, …** (references). These numbers are how the prompt
     refers to them. On Qwen Image 2.1 the last reference sets the canvas
     shape, and its tile is marked.
   - Each well's menu: Photos, Camera, Files, Choose from Library…, Paste. A
     staged tile's menu adds Move Left, Move Right and Remove.
   - HEIC and WebP are converted on the phone. Alpha is never flattened.
5. **Chip row:** Shape · Steps · Batch · Length (clips only) · More options
   (`slider.horizontal.3`).
6. **Last row:** the estimate ("about 40s", mono) at the leading edge.
   **Generate** sits at the trailing edge: `.buttonStyle(.glassProminent)`,
   `.controlSize(.large)`, ⌘↩.

**More options.** A sheet with `.medium` and `.large` detents, containing a
`Form`. Its sections mirror the Mac inspector, and each is present only when the
capability block allows it:

- **Adapters:** LoRA rows with weight.
- **Identity:** photo wells, strength, start step.
- **Refine:** ControlNet well; mask editor, which opens a full-screen PencilKit
  canvas with undo.
- **Clip:** Length and Smoothness, with the mono frames · fps readout.
- **Sampler:** Stick to my words (guidance); "Repeat this look" (seed, Keep |
  Surprise me); negative prompt.
- **Output:** format; Transparent background; upscaler; Save to Library.
- **File under:** title, tags, collection.
- **Recent prompts.**

Shape, Steps and Batch are repeated at the top of the sheet so they are still
reachable when the chip row collapses at accessibility sizes.

**While running.**

- The canvas shows the denoise preview. On a glass plate over it: the sentence
  (e.g. "Adding detail — about 12s left · `denoise 18/28`"), a
  `ProgressView(value:)` with step marks, and a small **Stop**. Stop stops the
  batch on screen; its menu adds "Stop Everything from Here".
- Pressing Generate again queues another batch; the plate shows "+2 waiting".

**Result.**

- A batch pages horizontally, with a page indicator.
- The bottom glass bar has Save to Photos, Share, Copy, Favourite and Show in
  Library. The overflow menu has Use These Settings.
- A clip plays in place (AVKit, looped, muted until tapped).
- A mesh opens in the MoldMesh viewer: drag to turn, pinch to zoom, double-tap
  to reset, and a slow auto-turn unless Reduce Motion is on. If the mesh cannot
  be drawn, the poster shows with one line saying why.

**3-D object.** The source-picture well is the primary input. A recipe that
ignores the prompt hides the prompt field and says: "This model works from a
picture, not a description."

**Licence.** A gated model shows its licence in a `.large` sheet before any
pull, with "Accept and Download".

### 5.2 Library

**Grid.** `LazyVGrid(columns: [.adaptive(minimum: tileMin)])`, where `tileMin`
is an `@ScaledMetric` from the Tiny / Small / Medium / Large / Largest ladder
(52 / 76 / 112 / 170 / 300 pt at Large: about 7, 5, 3, 2 and 1 columns on an
iPhone).

- A pinch walks the ladder live, one size per ~35% of spread or squeeze, with
  `.selection` haptics, keeping the top print in place (UIKit's pinch
  recognizer beside the scroll view: SwiftUI's `MagnifyGesture` never saw a
  squeeze there). The View menu and ⌘+ / ⌘− offer the same choice.
- On the two smallest sizes badges shrink to a symbol; nothing wraps over a
  picture.
- Thumbnails are requested at 256 or 512 px only -- the sizes machines serve
  (any other is a 422) -- and decoded off the main thread.
- **Offline.** Each machine's listing is saved on the device and shown before
  any machine answers; a machine that is down keeps its prints in the grid,
  with a note saying so. Thumbnails and the prints opened in the viewer are
  kept too (Application Support, excluded from backup), within Settings'
  storage limit; the newest 200 prints' thumbnails are always saved.
- Day sections have pinned headers: "Today", "Thursday 24 September".

**Tile.** Square, radius 5.

- Badges: a machine badge (only when there is more than one machine, e.g. "This
  Mac +1"), duration for a clip, `cube` for a mesh, and ★ for a favourite.
- A print held on several machines is ONE tile, using the Mac's merge rule.

**Toolbar.**

- `.toolbarTitleMenu` switches the shelf: All Prints · Favourites ·
  Collections ▸ (each) · Recently Deleted. A collection exists only once a
  print is filed in it (no machine stores an empty one), so it is started
  from a print's Add to Collection ▸ New Collection…, never from the
  sidebar.
- Trailing: Select, and ⋯ (Sort By, Tile Size, Machine).

**Search.** The Search tab, or pulling Library down. Uses
`.searchable(text:tokens:)`.

- Suggested tokens: Videos (`is:video`), 3-D (`is:mesh`), every tag (`tag:`), and
  every machine (`on:`).
- Typing a known token and pressing Return makes it a token.

**Selection.**

- Tap Select, then tap or drag across tiles.
- Bottom toolbar: Share · Favourite · Add to Collection · Tags · Delete.
- In Recently Deleted the toolbar is Put Back · Delete Immediately (with a
  destructive confirmation dialog).

**Viewer.**

- Opens with a zoom navigation transition from the tile. Full screen, paging in
  the grid's order. Pinch or double-tap to zoom; swipe down to close; tap to hide
  or show the chrome.
- Bottom bar: Share · Favourite · Info · Delete.
- ⋯ menu: Use These Settings · Save to Photos · Copy · Add to Collection ·
  Rename · Export (whatever the machine advertises: OBJ, STL, PLY, Turntable).

**Info sheet.** Detents `.fraction(0.35)` and `.large`, with background
interaction.

- An editable title (Return commits; blank clears it).
- The prompt, selectable.
- Model · `id`; Machine.
- Then the Sampler, Conditioning, Clip, Mesh and File sections. Each appears only
  when it has something to show.
- Every row has a context menu with Copy.

**Context menu on a tile.** The preview is the print. Items follow the Mac's
tile menu in the same order, with Delete last after a divider.

**Recently Deleted.**

- Each tile carries a mono countdown ("12 days").
- Put Back and Delete Immediately are in the menu. The Library grid has no swipe
  actions.

### 5.3 Queue

A `List` with one section per machine; the section header carries its status dot
and name.

- **Being made:** 52 pt scaled live-preview thumbnail, the sentence, a progress
  bar and a mono ETA.
- **Batch:** a parent row ("Batch of 4 · 2 finished") with a `DisclosureGroup`
  of its children.
- **Waiting:** a placeholder thumbnail with the position in mono.
- **Held:** a warning glyph, then the machine's sentence in full, e.g. "Needs
  FLUX.2 [klein] on workstation before it can start." Bordered buttons follow:
  **Pull**, **Retry** and **Move to…**. Pull shows the licence first if the model
  is gated.
- **Swipe:** trailing Cancel (destructive). Leading Pause or Resume, only where
  the machine can pause a single job.
- **Context menu:** Move Up · Move Down · Move to… · Pause │ Cancel Job.
- **Toolbar:** Edit (drag to reorder), and ⋯ with Pause Queue · Resume Queue ·
  Empty Queue…, each also offered as "on All Machines".
- Pull to refresh does a full reconcile.
- A row that is rendering on a machine that cannot stop at a safe point has no
  Cancel at all, rather than a Cancel that fails.

### 5.4 Machines

**Header.** The list starts with a "Models" row for the Default machine.

**Card (radius 8).** One column on iPhone, adaptive columns on iPad. Contents:

- A status dot and the name, plus a **Default** capsule.
- "Ready · 0.31.0" and "4× NVIDIA L40S".
- The load figure and a linear VRAM `Gauge` ("14.9 / 24 GB").
- Memory.
- "3 queued · 14 installed".
- The address, in mono.

**A machine that is not answering** stays in place, uses `.secondary`
foreground, and shows its reason: "Not answering since 14:02 — it may be asleep
or off the network."

**Card context menu:** Open · Check Now · Set as Default · Copy Address · Edit…
│ Remove….

**Nearby.** A section fed by `NWBrowser` on `_mold._tcp`. Each row has an Add
button.

**Machine detail** (push):

- Overview
- GPUs: one card each, showing what it holds, a VRAM gauge, and a toggle where
  the scheduler honours one
- Memory and CPU
- Queue: opens the Queue tab filtered to this machine
- Models
- Address, with Copy
- Edit and Remove, at the bottom

**Add a Machine** is a sheet with its own navigation stack.

1. Three large rows: **Scan a Pairing Code** (`qrcode.viewfinder`), **Nearby**,
   **Enter an Address**.
2. **Scan:** VisionKit `DataScannerViewController` (`.barcode([.qr])`).
   - Caption: "On your Mac, open Machines ▸ your machine ▸ Pair a Phone…"
   - A "Paste a pairing link" row sits below it.
   - The code is a universal link (`https://utensils.io/mold/pair#…`), so the
     Camera app opens Mold Studio too. An opened link shows **Pair This
     Phone?** with the machine's name and address; nothing is claimed until
     **Pair**. An expired or unreadable code says so there.
   - The scanned text goes to `MobilePairingPayload.parse`. The app then claims
     the pairing ticket, checks that `instance_id` matches, and writes the
     returned key straight to the Keychain.
3. **Enter an Address:** a field with the `.URL` keyboard and no autocorrect.
   - It is checked live, and the result is shown as a sentence:
     "workstation · mold 0.31.0 · 4× NVIDIA L40S".
   - Optional API key in a `SecureField`, stored in the Keychain only.
4. **Confirm:** Name, and a "Make this the Default machine" toggle. Then Add.

**First run.** Machines shows a `ContentUnavailableView`.

- Symbol `server.rack`, title "No machines yet", and the line "Mold makes
  pictures on a computer you own. Add one to begin."
- A prominent **Scan a Pairing Code** button, and **Enter an Address**.
- Nearby results appear below as they arrive.

### 5.5 Models (per machine)

- The navigation title names the machine. A segmented control switches
  Installed and Discover.
- **Installed:** grouped by family. Each row shows the name, variant, the
  manifest's trade-off sentence and the size in mono.
  - Swipe: Delete.
  - Menu: Load · Unload · Repair · Components │ Delete….
  - Active downloads are listed first; the machine's disk figure is in the
    footer.
- **Discover:** `.searchable`, plus Family and Sort menus.
  - Each row ends in **Get**, **Installed**, or **Open Page** when the machine
    cannot take it.
  - Tapping a row opens its Details sheet.
- **Download row:** an inline progress bar, "2.1 / 11.8 GB · 42 MB/s" in mono,
  and Cancel Download.
  - The download runs on the machine, so it carries on while the app is
    suspended. The row reattaches to `/api/downloads/stream` when the app is
    back in the foreground.
- **Licence sheet:** the licence rendered from the machine's own payload, and
  "Accept and Download".

### 5.6 Settings (sheet, `Form`)

- **Machines:** add, edit, remove, Default.
- **Generation:** defaults.
- **Library:** Save new prints to Photos (off); Offline Storage limit
  (250 MB – 5 GB, 1 GB by default; thumbnails 30%, opened prints the rest);
  how much is used; Save All Thumbnails for Offline (with progress and Stop);
  Empty Now.
- **Notifications:** Finished · Failed · Held.
- **Live Activities.**
- **About:** version, acknowledgements, and a privacy policy link that opens
  `https://utensils.io/mold/privacy` in the browser.

There is no Appearance setting; the system decides. Advanced config editing
stays on the Mac.

### 5.7 Live Activity and Dynamic Island

One activity per running batch. It starts on iPhone only when activities are
enabled.

| Presentation | Content |
| --- | --- |
| Compact leading | `wand.and.sparkles`, tinted with the accent colour |
| Compact trailing | Progress ring |
| Minimal | Progress ring |
| Expanded | Leading: 44 pt preview thumbnail (App Group file). Trailing: `Text(timerInterval:)` ETA, in mono. Centre: the sentence. Bottom: the prompt (2 lines), a linear bar, "denoise 18/28 · workstation" in mono, and a Stop button (`LiveActivityIntent`). |
| Lock Screen | The same content as Expanded |
| Finished | Final thumbnail, "Finished on workstation" and View; dismissed after 15 minutes |

**Honesty rule: the server has no push.**

- While the app is in the foreground, updates arrive about once a second.
- When the app is backgrounded, `staleDate` is set to the ETA plus 5 minutes.
  The stale view says "Open Mold Studio to refresh", instead of pretending the
  progress is live.
- `BGAppRefreshTask` ends the activity once it sees the batch has finished.

### 5.8 Widgets

| Family | Content |
| --- | --- |
| systemSmall | Latest print, full-bleed, with its title on a glass plate |
| systemMedium | The 4 most recent prints |
| systemLarge | A 3×3 grid under a "Recent Prints" header |
| accessoryRectangular | "2 being made · 1 held" |
| accessoryCircular | Progress of the current render |
| accessoryInline | "Mold: 3 waiting" |

- Configured with an App Intent: All Machines or a named machine, and All
  Prints, Favourites or a collection.
- Prints use `.widgetAccentedRenderingMode(.fullColor)`.
- Tapping a print deep-links to it.
- The widget reads only the App Group snapshot; it has no network access.

### 5.9 Share extension

1. Share ▸ Mold Studio opens a SwiftUI sheet with a preview of the item.
2. "Use as": **Start from** · **Reference image** · **Add to Library**.
3. The photo is staged in the App Group inbox: downscaled to 2048 px, with alpha
   kept, and HEIC converted.
4. A confirmation reads: "Waiting in Mold Studio. Open the app to use it."

The extension makes **no network calls**. The next time the app is in the
foreground, Generate shows a **From Share** card with the same three choices.

### 5.10 Notifications

| Kind | Text | Actions |
| --- | --- | --- |
| Finished | "Finished — a lighthouse at dusk", with a thumbnail attachment | View, Favourite |
| Held | "Held — needs FLUX.2 [klein] on workstation" | Pull and Retry, View |
| Failed | "Failed — workstation ran out of video memory" | View |

- Notifications are threaded per machine.
- They are not presented while the app is in the foreground; the inline banner
  covers that case.

## 6. Dynamic Type and accessibility (binding)

### Rules

1. **Text styles only.** `apps/ios` lint rejects `.font(.system(size:`,
   `.font(.custom(`, and `UIFont(…size:`. Figures use `.monospacedDigit()`, or
   `.monospaced()` for identifiers.
2. **Scaled metrics.** Every non-text dimension that sits next to text is an
   `@ScaledMetric(relativeTo:)`: thumbnails (Queue 52), tile minimums
   (96/128/180), wells (72), badge insets and spacing. Clamp only with a maximum,
   for example a tile's minimum never exceeding the width available.
3. **Axis switching.** Read `@Environment(\.dynamicTypeSize)`. When
   `isAccessibilitySize` is true, every label/value row, card header, held-row
   button group and the composer's last row switches from
   `AnyLayout(HStackLayout())` to `AnyLayout(VStackLayout(alignment: .leading))`.
4. **Progressive collapse.** The composer chip row uses
   `ViewThatFits(in: .horizontal)` with three stages: full chips, then icon plus
   a short label, then a single **Options** button that opens More options.
5. **No truncation of meaning.** Every label and sentence has
   `lineLimit(nil)`. Only prompt previews on tiles and rows may truncate, and
   their full text is exposed to VoiceOver and the Large Content Viewer.
   `minimumScaleFactor` and `allowsTightening` are never used to make text fit.
6. **Large Content Viewer.** System bars get it for free. Custom glass buttons
   (Generate, chips, the viewer bar) add `.accessibilityShowsLargeContentViewer`.
7. **Targets** are at least 44×44 pt, via `.contentShape` and a scaled minimum
   frame.
8. **The composer** is capped at 55% of the screen height and scrolls inside that
   cap. The canvas never drops below 30% of the height; below that, the composer
   scrolls.
9. **Colour that the audit proved.** The palette is the system's, with three
   asset-catalog colours added after the shell's contrast audit failed on the
   system defaults:
   - `AccentColor` is light #0062CC and dark #4DA3FF. White on the stock light
     #007AFF is about 4.0:1, and the stock dark #0A84FF failed as tint text on
     a grouped row.
   - `ProminentFill` is light #0062CC and dark #1A66CC. It is the fill for the
     one filled button on a screen (`.prominentAction()`), giving 5.5:1 or
     better under white text. In dark mode no single blue passes as a fill AND
     as tint text on a grouped row, so the two are split.
   - `SecondaryText` is light #48484A and dark #C7C7CC. The system `.secondary`
     is about 4.4:1 on white, and #6C6C70 still failed on the grouped
     background; `make lint` rejects `.foregroundStyle(.secondary)`.
10. **An empty state's action is never under the glass.** At accessibility sizes
    it is pinned above the tab bar while the explanation scrolls.

### VoiceOver

- **A tile** is one element: "Print, a lighthouse at dusk, Today 14:02,
  workstation, favourite". Custom actions: Favourite, Share, Use These Settings,
  Delete.
- **Rotors:** Day (the section headers) and Favourites.
- **A Queue row** is combined into one element, with custom actions Cancel,
  Pause, Move Up, Move Down, Pull and Retry.
- **Progress:** the value is "18 of 28, about 12 seconds left". An announcement
  is posted at each 25% of progress, not at every step.

### Other settings

- **Reduce Motion:** no mesh auto-turn, a cross-fade instead of the zoom
  transition, no pulsing status dot.
- **Reduce Transparency:** system glass falls back on its own. Never hand-roll
  blur.
- **Increase Contrast:** dimmed states use `.secondary`, never opacity below 0.6.
  Status dots always come with words.
- **Colour** is never the only signal.

### Layout per size

| Screen | xSmall | Large (default) | xxxLarge | AX5 |
| --- | --- | --- | --- | --- |
| Generate | One chip row; large canvas | Chips wrap to two rows if needed | Chips are icon + short label; prompt shows 1…4 lines | One **Options** button; the estimate sits above a full-width Generate; the model id moves into the menu; the Model button wraps |
| Library | 5 columns | 3 columns | 3 columns, larger headers | 1–2 columns; headers wrap; machine badges hidden on tiles (still spoken, and shown in Info) |
| Viewer | Icon bar | Icon bar | Icon bar | Icon bar with the Large Content Viewer; Info opens straight to `.large` |
| Queue | Thumbnail beside text | same | same | Thumbnail above text; held-row buttons stacked full width; ETA on its own line |
| Machine card | Dense | Dense | Figures wrap | Every label/value pair stacks; gauge full width; address wraps |
| Models row | One line | Size trails | Size trails | Name, then sentence, then size, stacked; Get full width |
| Live Activity / widgets | System-capped | | | `ViewThatFits` drops the mono line first |

## 7. Microcopy

| State | Copy |
| --- | --- |
| No machine | **No machines yet** — Mold makes pictures on a computer you own. Add one to begin. |
| Generate, no machine | **Add a machine to start generating** [Add a Machine…] |
| Machine not answering | Not answering since 14:02 — it may be asleep or off the network. |
| Machine wants a key | This machine is there but wants an API key. [Add Key…] |
| Checking | Checking… |
| Held, missing model | Needs FLUX.2 [klein] on workstation before it can start. [Pull] [Retry] |
| Held, other | *The machine's own sentence.* [Try Again] |
| Gated pull | Qwen Image 2.1 is under the Qwen Research licence. [Accept and Download] |
| Generate refused | workstation couldn't start this: out of video memory. Try a smaller shape or fewer in the batch. [✕] |
| Pairing expired | This pairing code has expired. Make a new one from Pair a Phone… on your Mac. |
| Wrong machine | This code belongs to a different machine than the one that answered at that address. |
| Address silent | Nothing answered at 10.0.0.4:7680. |
| Duplicate | workstation already answers at this address. |
| Library empty | **No prints yet** — What you generate on any machine appears here. |
| Search empty | Nothing here matches what you're looking for. |
| Recently Deleted empty | **Nothing deleted** — Prints you delete stay here for 30 days. |
| Queue empty | **Nothing waiting** — Renders you start appear here. |
| Mesh can't draw | This 3-D object can't be drawn here — showing its poster instead. |
| Source gone | workstation no longer has the source picture for this print. |
| Parked well | Reference image (not used by this model) |
| Live Activity stale | Open Mold Studio to refresh. |
| Share staged | Waiting in Mold Studio. Open the app to use it. |

## 8. Interaction details

**Haptics** (`.sensoryFeedback`)

| Event | Feedback |
| --- | --- |
| A render finishes while the app is in the foreground | `.success` |
| Generate is accepted | `.impact(weight: .light)` |
| A job is held or fails | `.warning` |
| Stepper changes and tile-size pinch snaps | `.selection` |

There is no haptic on scroll.

**Context menus.** Every tile, row and card has one. Items follow the Mac's order,
and destructive items come last, after a divider.

**Swipe actions**

| Where | Actions |
| --- | --- |
| Queue | Cancel; Pause / Resume |
| Models | Delete |
| Recently Deleted list mode | Put Back; Delete Immediately |

**Pull to refresh** in Library, Queue, Machines and Models does a full reconcile,
not a delta.

**Drag and drop (iPad)**

- Drag prints out as `Transferable` files.
- Drop pictures onto wells or the canvas.
- Drop prints onto a sidebar collection to file them.
- Drop files onto Library to import them.
- Drag to reorder Queue rows.

**Keyboard shortcuts** (iPad, or iPhone with a hardware keyboard), shown in the ⌘
overlay:

| Keys | Action |
| --- | --- |
| ⌘1–⌘5 | Generate · Library · Queue · Models · Machines |
| ⌘↩ | Generate |
| ⌘E | Expand |
| ⌘F | Search |
| ⌘R | Refresh |
| ⌘, | Settings |
| ⌥⌘I | Info |
| ⌥⌘F | Favourite |
| ⌘⌫ | Delete / Cancel Job |
| ⌘Z | Undo |
| ← → | Previous / next print |
| Esc | Close the viewer |
| ⌘+ / ⌘− | Tile size |

## 9. Mockup frames

| # | Frame | Size / appearance |
| --- | --- | --- |
| 1 | Generate: idle with a result | Large, light |
| 2 | Generate: running (denoise preview, sentence) | Large, dark |
| 3 | Generate: keyboard up, source + image 1 / image 2 | Large, light |
| 4 | Generate: collapsed Options, stacked Generate | AX5, light |
| 5 | Generate | xSmall, dark |
| 6 | More options: medium detent | Large, light |
| 7 | Library: day sections, title menu open | Large, light |
| 8 | Library: 2 columns, `is:video` token | AX3, dark |
| 9 | Viewer with the Info sheet at 35% | Large, dark |
| 10 | Queue: batch parent/children, plus a held row | Large, light |
| 11 | Queue: held row with stacked buttons | AX5, light |
| 12 | Machines: offline card, Nearby | Large, dark |
| 13 | Add a Machine: scanner and live address check | Large, light |
| 14 | Models: Discover download, plus the licence sheet | Large, light |
| 15 | iPad landscape: sidebar and Library | Large, dark |
| 16 | Live Activity, Dynamic Island, widgets | Light and dark |
