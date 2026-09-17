# Peer review — `apps/macos` Library + Support (+ MoldClient library/gallery pieces)

Reviewer scope: `Sources/Mold/Library` (40 files), `Sources/Mold/Support` (15),
`Packages/MoldClient/Sources/MoldClient/{LibraryEntry,GalleryPrint,PrintEdit,MutationOutbox*,CollectionShelf,LibraryQuery,LibraryShowing,LibrarySection,LibraryCursor,CacheBudget,MediaURL,MediaToken,MoldEvent,HTTPBackend+Gallery,GalleryMutations}.swift`
and their tests. Counterparts read: `desktop/src/views/LibraryView.vue`,
`desktop/src/composables/useReuseStillPrint.ts`, `studio/lib/meshExport.ts`,
`studio/api/gallerySourceMedia.ts`, `crates/mold-server/src/{routes.rs,gallery_trash.rs,gallery_organization.rs}`.

Read-only review; nothing was built or run. Where a claim depends on AppKit/Foundation
runtime behaviour rather than on code I could read on both sides, I say so.

---

## HIGH

### H1 — security/bug: server-supplied filenames are written to disk unsanitised (path traversal)

`apps/macos/Sources/Mold/Library/PrintMaterializer.swift:49` and `:65`

```swift
let file = cacheRoot.appending(path: key).appending(path: entry.print.filename)
…
let written = folder.appending(path: entry.print.filename)
guard (try? data.write(to: written)) != nil else { return nil }
```

`entry.print.filename` comes straight off `GET /api/gallery` (`LibraryStore.swift:98`,
`LibraryStore+Live.swift:75`) and is never validated. `URL.appending(path:)` keeps `/`
and `..` literally (they are legal path characters), and `Data.write(to:)` /
`FileManager.fileExists(atPath:)` resolve them through POSIX. A host answering with
`"filename": "../../../../../../Users/you/Library/LaunchAgents/evil.plist"` makes the app
create directories and write attacker-controlled bytes anywhere the user can write —
the app is **not sandboxed** (`apps/macos/scripts/Mold.entitlements` sets
`com.apple.security.app-sandbox` to `false`, deliberately).

The same class exists on the multi-print save path, which additionally **deletes** at the
computed destination:

`apps/macos/Sources/Mold/Library/LibraryActions.swift:95-97`

```swift
let destination = folder.appending(path: entry.print.filename)
try? FileManager.default.removeItem(at: destination)
try? FileManager.default.copyItem(at: source.url, to: destination)
```

Here the user picked `folder` with an `NSOpenPanel`, so they consented to that directory
and nothing above it. The single-print branch is safe (an `NSSavePanel` URL).

Threat model is real rather than theoretical for this app: hosts are added by bare address
(`README.md` "Pointing it at a server"), `HostAddress` fills in **`http://`**, so any
machine on the LAN path can rewrite a gallery listing; and "a machine I added once" is a
weaker trust boundary than "code I shipped". An honest mold cannot produce such a name —
`clean_gallery_filename` and `render_gallery_thumbnail`'s `clean_name != filename` check
(`crates/mold-server/src/routes.rs:10679-10683`) reject them server-side — which is exactly
why the client has never noticed.

**Fix:** one validator in `MoldClient` — reject a `GalleryPrint` whose `filename` is not a
single safe path component (`URL(fileURLWithPath:).lastPathComponent == filename`, no `/`,
no leading `.`), applied where listings and `gallery_*` event rows are decoded, and reapply
`lastPathComponent` at both write sites as belt-and-braces. Confidence: high on the code
path; I did not execute it.

---

## MED

### M1 — bug/race: live gallery events are dropped for a whole host whenever any edit is pending, and nothing re-lists afterwards

`apps/macos/Sources/Mold/Library/LibraryStore+Live.swift:33-38`

```swift
private func apply(_ change: MoldEvent.Gallery, from host: MoldHost.ID) {
    guard outbox.chain(for: host).isEmpty else { return }
```

The comment reasons about *our own echo*, but the guard drops **every** gallery frame from
that host, including `gallery_added` (a render landing), and `gallery_trashed` /
`gallery_updated` produced by another client. Nothing repairs it afterwards: the drain loop
(`LibraryStore+Outbox.swift:28-51`) re-lists only on `giveUp`/non-transient failure, and on
success calls `reloadCollections()` at most. Nothing else in the app refreshes the Library
on job completion either — the only callers of `refresh`/`reload` are the pane's `.task`,
⌘R, the destructive actions, and the Generate picker sheet.

Concrete failure: you star a print on `plato` while a batch is finishing there. The star
takes 300 ms round-trip; a `gallery_added` arriving in that window is discarded, and the new
print is invisible until a manual ⌘R — contradicting the README's "a print favourited,
tagged or trashed somewhere else appears here without a refresh". On a flaky host the window
is the full backoff ladder (1 s + 2 s + 4 s, `MutationOutbox+Policy.swift:30`), and a
`gallery_trashed` lost there leaves a ghost row forever.

**Fix:** narrow the guard to an actual echo — skip only when a pending entry for that host
names the same filename *and* the same change — or, minimally, `relist(host)` once the
chain empties in `drain`'s tail (it already does this on the failure path, so the machinery
exists). Add a test: enqueue an edit, deliver `gallery_added` for a *different* filename,
assert the row appears.

### M2 — bug: the grid ignores the Sort By control, and the viewer's ← → disagrees with the grid

`Packages/MoldClient/Sources/MoldClient/LibrarySection.swift:21-31`

```swift
return buckets.keys.sorted(by: >).map { day in
    LibrarySection(…, items: buckets[day]!.sorted { $0.print.timestamp > $1.print.timestamp })
}
```

`LibraryShowing.init` (`LibraryShowing.swift:17`) builds `sections` by handing
`byDay` the already-sorted `visible`, and `byDay` re-sorts every bucket newest-first and the
days descending. `LibraryPane.content` draws `showing.sections`
(`LibraryPane.swift:119-124`), so **Oldest First / Largest First / Name are no-ops in the
grid**; only `.newest` ever renders. `LibraryQuery.sorted` is correct and tested
(`LibraryQueryTests.swift:108`) — it just never reaches the pixels.

Worse, the order is not merely ignored, it *diverges*: `LibraryCursor` is built from
`sections` (`LibraryGrid.swift:79`) while the viewer's Next/Previous walks
`showing.visible` (`LibraryPane+Wiring.swift:40-45`). Under "Oldest First", arrowing in the
grid and arrowing in the viewer move in opposite directions over the same list.

`LibraryShowingTests.swift` cannot catch this: it asserts
`showing.sections == byDay(query.apply(pool))`, i.e. it re-derives the bug.

**Fix:** have `byDay` preserve the incoming order within a bucket (it is already sorted) and
order the day keys by that order's first member, rather than always `>`; assert in a test
that `sections.flatMap(\.items) == visible` for every `LibrarySort` case.

### M3 — bug: arrow-key selection reads the last *mouse-down*'s modifiers

`apps/macos/Sources/Mold/Library/LibraryGrid.swift:125-128`

```swift
private func move(_ move: LibraryCursor.Move) -> KeyPress.Result {
    selection = cursor.moving(move, ClickModifiers.current, from: selection)
```

`ClickModifiers` is explicitly a record of the last `.leftMouseDown`
(`ClickModifiers.swift:17-27`), which is the right answer for a deferred tap and the wrong
one for a key press — `KeyPress` carries its own `.modifiers`, and the ⌘A handler four lines
above proves the author knows that (`press.modifiers.contains(.command)`).

Two symptoms: ⇧← / ⇧→ never extend a selection (the flags say whatever the last click said),
and after any shift-click every subsequent *bare* arrow extends the selection instead of
moving, until the next unmodified mouse-down.

**Fix:** `.onKeyPress(.leftArrow) { move(.left, $0.modifiers) }` and map
`EventModifiers` → `LibraryCursor.Modifier` in the grid; `LibraryCursorTests` already covers
the pure side.

### M4 — bug/HIG: a bare ⌫ trashes the selection

`apps/macos/Sources/Mold/Library/LibraryGrid.swift:71`

```swift
.onKeyPress(.delete) { trashSelection() }
```

`onKeyPress(_ key:)` matches the key **regardless of modifiers** (again, the ⌘A handler
filters manually because of this), so an unmodified Backspace moves the selection to the
trash. The menu (`Shell/LibraryCommands.swift:67-69`) and the README both promise ⌘⌫, and
in the Finder a bare Delete does nothing. In the Trash scope the same key routes to
`deleteForever`, which at least asks first.

It is also a second binding for a chord `LibraryCommands` already owns, which the README's
own rule forbids ("binding one twice queues the work twice"). In practice a main-menu key
equivalent is consumed first, so I do not claim a double-fire — but the bare-⌫ path is
unambiguous from the code.

**Fix:** require `press.modifiers.contains(.command)` in the grid handler, or drop the
handler and let the menu item own the chord.

### M5 — bug: `Space` as a bare menu key equivalent steals the space bar from the Library's text fields

`apps/macos/Sources/Mold/Shell/LibraryCommands.swift:39-41`

```swift
Button("Quick Look") { library?.quickLook() }
    .keyboardShortcut(.space, modifiers: [])
    .disabled(library?.isEmpty ?? true)
```

AppKit offers a key-down to the main menu's `performKeyEquivalent:` before the field editor
sees it, so an *enabled* menu item whose key equivalent is a bare space intercepts spaces
typed into any text field in the window. In the Library that is the `.searchable` field
(`LibraryPane.swift:101`), the inspector's **Title** field (`TitleField.swift:17`) and
**Add a tag** (`TagEditor.swift:39`) — and all three are only reachable with a selection,
which is precisely when the item is *not* disabled. Typing "my cat" as a title should fire
Quick Look at the space.

The author already solved this problem once: `LibraryViewer` stands its own key equivalents
down through `isSearching` / `@FocusedValue(\.editingText)` (`LibraryViewer.swift:114-118`),
and both text fields publish `editingText`. The menu item just doesn't consult it.

**Fix:** disable the Quick Look item while `editingText == true` (plumb the focused value
into `LibrarySelection`, the way `isEmpty` already is), and keep the grid's own
`.onKeyPress(.space)` as the real binding. Confidence: high on the AppKit ordering (it is
the standard reason apps avoid bare-space key equivalents), but not runtime-verified here.

### M6 — bug: a GLB print opens into a viewer that never finishes loading

`apps/macos/Sources/Mold/Library/LibraryViewer.swift:33-45,135-138`

The viewer branches on `entry.print.isVideo` and otherwise decodes the stored bytes with
`NSImage(data:)`. For a mesh, `actions.data(for:)` returns the GLB, `NSImage(data:)` returns
`nil`, `full` stays `nil` — so the poster thumbnail sits there at `opacity(0.55)` with
`.interpolation(.low)` **forever**, which is precisely the app's own visual language for
"still loading". There is no error, no "3-D object" affordance, and no way to tell it apart
from a slow download. `LibraryThumbnail`'s placeholder glyph is likewise `photo` for a mesh
(`LibraryThumbnail.swift:31`) even though `PrintKind.mesh` exists and the token row already
has a `cube` symbol (`LibraryToken.swift:44`).

`Return` / double-click / the context menu's **Open** all reach it in every scope, and
Quick Look does not rescue it either — macOS ships no GLB preview generator, so Space gives
a blank panel.

The README's "no 3-D studio" omission does not cover this: the preamble's own example is "a
mesh print that renders badly in the Library", and mold publishes mesh prints into the
ordinary gallery on every host that can make one.

**Fix:** at minimum branch on `entry.print.isMesh` and show the server-rendered poster at
full opacity with an explicit "3-D object · Export…" affordance instead of the loading
treatment; a `SceneKit`/`RealityKit` GLB view would be the parity answer
(`studio/components/MeshViewer.vue`), but is not required to stop this reading as a hang.

### M7 — bug: deleting a tag wipes the window's entire undo stack, including other apps' registrations

`apps/macos/Sources/Mold/Library/LibraryStore+Tags.swift:59` → `Support/MoldUndo.swift:61-63`

```swift
func forget() { manager?.removeAllActions() }
```

`manager` is deliberately the **window's** `UndoManager` (`MoldUndo.swift:19`, and the pane
hands it over at `LibraryPane.swift:70`), shared with every `NSTextField` field editor and
anything else in the window that registers. `removeAllActions()` clears all of it, so
"Delete Tag Everywhere…" also throws away the rename you were about to undo and whatever a
focused text field had recorded.

**Fix:** `manager?.removeAllActions(withTarget: self)` — `register` already registers
`withTarget: self` (`MoldUndo.swift:44`), so the targeted form removes exactly this store's
entries.

### M8 — bug: a print larger than the cache cap is downloaded and then immediately deleted, silently

`Packages/MoldClient/Sources/MoldClient/CacheBudget.swift:29` +
`apps/macos/Sources/Mold/Library/PrintMaterializer.swift:74`

```swift
var doomed = files.filter { $0.bytes > cap }.map(\.name)   // CacheBudget
…
let url = await task.value
inFlight[flightKey] = nil
enforceBudget()            // PrintMaterializer.url, after the write
return url
```

`evictions` correctly refuses to keep a file bigger than the whole cap, but
`PrintMaterializer.url` runs it *between writing the file and returning its URL*. With the
default 1 024 MB cap and a 1.5 GB clip (the README's own "hundreds of megabytes" case), the
whole clip is downloaded, written, deleted, and a URL to the deleted file is handed back.
Every consumer then fails with no message: `LibraryActions.save` uses `try?`
(`LibraryActions.swift:86,97`), `QuickLook` shows an empty panel, and the Finder drag
reports a generic failure. Setting the cap to 0 (documented as "a real setting") makes this
happen for *every* print.

Related, same function: eviction is unconditioned on what is in use, so a download can evict
the directory an open Quick Look panel or an in-flight drag promise is reading from
(`QuickLookItem` holds only a URL, and `DraggablePrint.file` resolves lazily).

**Fix:** exempt the just-written key from this pass (`enforceBudget(keeping: key)`), and
keep a small set of "in use" keys (Quick Look's current items, live drag promises) out of
`contents` while they are held. `CacheBudgetTests` already exists and can take both cases.

### M9 — design/robustness: whole responses are buffered in memory; a hostile host can OOM the app

`Packages/MoldClient/Sources/MoldClient/HTTPBackend+Transport.swift:65-69,73` —
every route including `media()` and `export()` goes through `session.data(for:)`, which
buffers the complete body. `LibraryActions.data(for:)` returns `Data`, so previewing,
saving, copying or dragging a clip holds the entire file in RAM before the materializer
writes it, and `copy(_:)` decodes up to ten of them into `NSImage`
(`LibraryActions.swift:59-68`). There is no `Content-Length` sanity check and no cap, so a
compromised or mis-behaving host can return an unbounded body and take the process down.

**Fix:** `session.download(for:)` (or `bytes(for:)` + streamed write) on the media/export
routes, writing straight into the materializer's file; and a size ceiling before buffering
on the JSON routes.

### M10 — parity gap: "Use These Settings" restores a fraction of the recipe and never asks about retained source media

`apps/macos/Sources/Mold/Library/LibraryPane.swift:139-146` →
`Packages/MoldClient/Sources/MoldClient/RenderDraft.swift:98-112`

The macOS reuse restores prompt (correctly reduced to the first stage for a sequence —
`RenderDraft.swift:114-118`, good), negative prompt, size, steps, guidance, seed, frames and
fps, and nothing else. `OutputMetadata` (`GalleryPrint.swift:9-32`) simply has no fields for
strength, mask, LoRAs, identity photo, reference images, scheduler or output format — all of
which the Generate pane supports per the README.

Two specific contract breaks against `CLAUDE.md`:

1. *"Every client always asks, and the server is the only authority on what it retained"* —
   nothing in the app calls `GET /api/gallery/source-media/:filename`. The desktop
   counterpart does it on this exact path
   (`desktop/src/composables/useReuseStillPrint.ts:54-84`, over
   `studio/api/gallerySourceMedia.ts`), asks every known copy, and discloses through
   `retainedSourceMediaDisclosable` when the print's own recorded conditioning bytes say
   there should have been something. Reusing an img2img print on macOS therefore silently
   produces a text-to-image render at the same size and seed.
2. There is no disclosure at all, so the user cannot tell a faithful restore from a lossy
   one.

**Fix:** decode the conditioning fields mold already records into `OutputMetadata`, and add
the source-media probe + inline disclosure behind the same one door
(`LibraryPane.reuse`), since it is the only reuse path in the app.

### M11 — performance: the whole library is re-filtered, re-sorted and re-grouped on every body pass

`apps/macos/Sources/Mold/Library/LibraryPane.swift:43`

```swift
let showing = LibraryShowing(pool: pool, query: resolved, selection: selection.items)
```

`LibraryShowing.init` runs `query.apply` (filter + sort over the merged pool) **and**
`LibraryGrouping.byDay` (a `Dictionary(grouping:)` plus a per-bucket sort, and one
`ISO8601DateFormatter()` allocated *per section* — `LibrarySection.swift:26`). `selection`
is `@State` on the pane, so every arrow key, every click, every inspector edit and every
character typed into search re-runs all of it over the entire library. `LibraryCursor` is
rebuilt per key press on top (`LibraryGrid.swift:78-80`).

The comment on `LibraryShowing` says it exists so the body computes this *once per pass*,
which it does — the missing half is that a pass happens on selection, not only on data.
The desktop's stated invariant is the opposite: `organizationIndex` / `bucketIndex` compute
"ONCE per data change" and tile models are immutable snapshots
(`.claude/rules/desktop.md`, "Desktop Library performance invariant", with
`LibraryView.perf.test.ts` at 2 000 prints).

At the brief's 10k prints this is a full sort + dictionary-group per keystroke on the main
actor. `LazyVGrid` itself is fine; the derivation above it is not.

**Fix:** cache `showing` keyed on `(pool identity, resolved query)` — an `@State` memo or a
small `@Observable` derived store — so a selection change reuses it; hoist the
`ISO8601DateFormatter` to a `static let`. Add a counting budget test in the shape of
`galleryPerfBudget.ts`.

---

## LOW

### L1 — parity: the export menu is built from client constants, not the host's advertised list

`Packages/MoldClient/Sources/MoldClient/GalleryMutations.swift:113-126` hard-codes
`animated = {gif, apng, webp}` and `geometry = {obj, stl, ply, zip}` and filters
`/api/gallery/export-options` through them
(`Sources/Mold/Library/LibraryActions+Export.swift:13-18`). `CLAUDE.md` names
`capabilities.mesh.export_formats` as the authority and says "never a client constant";
`studio/lib/meshExport.ts` reads exactly that and only knows the *stored* format (`glb`) as
a constant. A host that adds a container gets a menu entry on desktop/web and nothing here.

Two smaller consequences of the same shortcut: no turntable options (the shared surfaces
collapse the animated formats into one entry that opens the video export sheet with
playback/repeat/max-dimension/fps), and no `capabilities.mesh.export_geometry` support, so
`size_mm` / `up_axis` / `origin` are never offered and an STL always lands at the server
default. The server resolves sane defaults, so this is a missing affordance rather than a
wrong file.

### L2 — the thumbnail cache is a second, uncapped, never-purged on-disk copy

`apps/macos/Sources/Mold/Library/ThumbnailCache.swift:23-26` gives its private `URLSession`
a `URLCache` with `diskCapacity: 512 MB`. `PrintMaterializer.purge()` is what the quit path
calls (`MoldAppDelegate.swift:46`); nothing touches this one. So the README's "It is capped
(Settings ▸ Storage) and **emptied when Mold quits**" is true of the media cache and false
of the thumbnails, and Settings ▸ General's "Empty Now" leaves 512 MB behind.

Related: `purge()` runs only on `applicationShouldTerminate`, so a crash or a `SIGKILL`
leaves the media cache intact with no startup sweep — bounded by the cap, but the promise is
"emptied when Mold quits".

### L3 — a new collection is always created on `hosts.hosts.first`

`apps/macos/Sources/Mold/Shell/CollectionRow.swift:106-110`. If the first machine in the
list is down, creating a collection fails and reports, with no attempt at a reachable one —
even though the shelf would work fine made anywhere (the others get their copy on first
filing, which the comment says). Prefer the first *reachable* host, or the host of the
current selection.

### L4 — the LRU's `touch` may be overridden by the file system's own access date

`PrintMaterializer+Budget.swift:28` prefers `contentAccessDate` and falls back to
`contentModificationDate`, while `touch` (`:43-46`) writes only the *modification* date.
APFS does maintain access dates, so on the real file system the hand-maintained recency
signal is never read and eviction order is whatever the OS last recorded. Either set both
attributes or read `contentModificationDate` first.

### L5 — a failed edit leaves its undo entry on the stack

`LibraryStore+Editing.swift:29-36` registers the inverse before the outbox has sent
anything. If the machine refuses and `relist` repairs the row, the Edit menu still offers
"Undo Favorite" for a favourite that never happened; taking it applies a no-op locally and
sends a redundant mutation. Harmless today, but it is the kind of thing that stops being
harmless once a change is not idempotent.

### L6 — `TitleField` can commit the previous print's draft

`TitleField.swift:25,29`: `onChange(of: editing)` commits on focus loss and
`onChange(of: entry.id)` resets the draft; the ordering between the two when the selection
changes while the field is focused is not defined by SwiftUI. The view has no `.id(entry.id)`
to force a fresh instance. I could not construct the failing sequence by reading alone
(the grid normally holds focus), so: possible, unverified. `.id(entry.id)` on the field
removes the question.

---

## Test gaps

- Nothing covers `LibraryStore+Live`'s outbox gate (M1), the grid's sort (M2), the modifier
  source (M3), the bare-⌫ binding (M4), or `PrintMaterializer`'s key/path construction (H1).
  `LibraryStoreHostsTests` and `LibraryStoreFailureTests` are good tests of the *other*
  live-path bugs, which makes the absence of one for the gate conspicuous.
- `LibraryShowingTests.theThreeListsAgreeWithApplyingTheQueryByHand` asserts
  `sections == byDay(query.apply(pool))` — it re-derives the implementation and so can never
  fail for M2. The assertion that would catch it is
  `sections.flatMap(\.items).map(\.id) == visible.map(\.id)` for each `LibrarySort`.
- `ThumbnailCache` has no test at all (no injected session, no `URLProtocol` stub), so the
  `X-Api-Key`-on-thumbnails rule the README calls out is unpinned.
- `CacheBudgetTests` covers the arithmetic but not `PrintMaterializer`'s use of it, which is
  where M8 lives.

---

## Done notably well

1. **`PrintID` = (host, filename), threaded end to end.** The brief's first correctness
   worry is simply absent: identity, the drag transfer type
   (`PrintID+Transfer.swift`), the collection drop, the notification route, the cursor and
   the cache key are all host-qualified, and `LibraryEntry.swift:3-8` writes down why.
2. **`MutationOutbox` + `PrintEdit` as pure, tested values**, with the operation id minted
   once per intent and reused across attempts (`MutationOutbox.swift:16-18`,
   `MutationOutbox+Policy.swift:62-82`) — which is exactly the fence
   `mutate_gallery_bulk` implements (`gallery_organization.rs:412`, 409 on a reused id with
   a different payload). `failed(_:)` returning only the filenames no *later* entry speaks
   for is a genuinely subtle rule, correctly stated.
3. **Collections merged by slug, added by name, removed by slug.**
   `CollectionShelf.merge` + `PrintEdit.plan(collectionIDs:)` + `GalleryBulkMutation`
   reproduce mold's cross-machine contract precisely, including
   `CollectionShelf.count(in:)` refusing the host's own count because it includes trashed
   members — the same promise-about-what-opening-shows rule the desktop's
   `collectionCounts` makes.
4. **Per-host ETag listings with per-host failure isolation.** `LibraryStore.apply(_:for:)`
   keeps a failed host's rows, keeps a 304's rows, and scopes `succeeded(on:doing:)` to the
   verb so a passive re-list can't clear an action's failure — a pattern applied
   consistently across `+Mutations`, `+Organization` and `+Tags`.
5. **The undo funnel's synchronous registration.** `LibraryStore+Editing.swift:15-28`
   identifies that `UndoManager` only routes a re-registration to the redo stack from
   *inside* `undo()`, and keeps the local mutation and the registration on that side of the
   `await` while deferring only the network call. That is a real bug avoided, correctly
   reasoned, and written down.
6. (bonus) `QuickLook` using the real `QLPreviewPanel`, deliberately non-`@MainActor` with a
   lock and the `_swift_task_checkIsolatedSwift` trap recorded — the right call and the
   right note.
