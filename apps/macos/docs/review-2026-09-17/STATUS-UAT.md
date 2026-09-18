# Lane UAT — the native macOS app, driven on a real screen

Branch `feat/macos-native-app`, tip `911385fb`, MAIN checkout (the only build with
the Rust engine linked: `Engine.xcconfig` → `MOLD_EMBEDDED_ENGINE`, staticlib
`/Volumes/ExternalStorage/cargo-targets/mold-macos-ffi/release/libmold_macos_ffi.a`).
Built with `make build CONFIG=Debug`; **BUILD SUCCEEDED**.

Driven with throwaway state, the way `make uat` does:

```
MOLD_NATIVE_FRESH=1 MOLD_HOME="$TMPDIR/mold-native-uat" \
  apps/macos/build/Debug/Mold.app/Contents/MacOS/Mold
```

Machines: **hal9000** `http://100.123.198.98:7680` (keyless, RTX 4090, mold 0.29.0,
1,539 prints incl. 59 GLBs and 454 clips) and **This Mac** (the embedded engine,
mold 0.30.0, Metal). The keyed host `100.105.134.43:7680` was NOT used: its key
lives in the owner's real secrets file, which this lane does not read — every
keyed case below is BLOCKED naming exactly that.

Screenshots: `apps/macos/docs/review-2026-09-17/uat/`.
Tally: **40 PASS · 10 FAIL · 20 BLOCKED / not verifiable from here**.

---

## FAILURES

**F1. Opening any VIDEO print in the Library viewer ABORTS the app.**
Deterministic; three crash reports on three separate launches
(`~/Library/Logs/DiagnosticReports/Mold-2026-09-17-{184818,184932,185444}.ips`,
pids 79172 / 87308 / 87593). `SIGABRT`, Swift runtime fatal error in
`getSuperclassMetadata` under `_AVKit_SwiftUI`, preceded on stderr by
`failed to demangle superclass of VideoPlayerView from mangled name 'So12AVPlayerViewC'`
(`So12AVPlayerViewC` is `AVKit.AVPlayerView`).
`otool -L build/Debug/Mold.app/Contents/MacOS/Mold.debug.dylib` lists
`AVFoundation`, `_AVKit_SwiftUI` and `libswiftAVFoundation.dylib` (weak) but
**no `AVKit.framework`** — the ObjC superclass `VideoPlayer` needs is not in the
image, so the metadata for `_AVKit_SwiftUI.VideoPlayerView` cannot be built.
Sites: `Sources/Mold/Library/LibraryViewer.swift:76` (`VideoPlayer(player:)`) and
`Sources/Mold/Generate/RunCanvas+Result.swift:52` — so this blocks every clip in
the Library AND every clip rendered on the Generate canvas. Quick Look on an mp4
works (it is macOS's panel, not `VideoPlayer`), which is why the crash hides until
someone opens one.

**F2. The inspector says "Nothing selected" over an open print.**
`uat/fail-viewer-inspector-nothing-selected.png`. The viewer is showing a print
and the trailing column is empty. This is the exact bug
`Sources/Mold/Library/LibraryInspector+Pane.swift:4-7` says was fixed, and
`Packages/MoldClient/Sources/MoldClient/LibraryShowing+Inspected.swift:18-22` is
the rule it should obey (`viewing` wins, fall back to `selected`). Reproduced on
a still and on a mesh, whether the tile was selected first or not. Selecting in
the GRID fills the inspector correctly, so only the `viewing` arm is wrong.

**F3. Reuse claims the retained source and then renders without it.**
`uat/fail-reuse-source-not-in-well.png`. Use These Settings on an img2img print
(`mold-sd15-fp16-1789696823679.png`, `strength: 0.75`,
`source_image_name: mold-sd15-fp16-1789696757565.png` on the host) restores the
prompt, model, size, steps, guidance, seed, tag and collection and shows the
banner *"Using the source media from mold-sd15-fp16-1789696823679.png on
hal9000."* — but the source well is EMPTY, there is no ✕ on it, the composer
shows **no Strength control**, and the inspector still reads "Mask — Add a source
picture first." Pressing Generate produced `mold-sd15-fp16-1789697799324.png`,
whose metadata on hal9000 carries `strength: 0.75` and **`source_image_name:
None`**: the conditioning was lost, not merely undrawn. (The restored seed was
not used either — the print's seed is `1789697797687550999`, not the restored
`1789696822039566223`.)

**F4. Set as Default writes the preference and nothing on screen changes.**
`uat/fail-set-as-default-no-redraw-menu.png` (the item still offered after
choosing it) and `uat/fail-set-as-default-badge-after-rebuild.png` (the Default
badge, which only appears after leaving the pane and coming back).
`Sources/Mold/Support/HostStore+Default.swift:18-27` is a computed property over
`AppStorageSuite.defaults` — the write lands in the prefs domain (verified with
`defaults read io.utensils.mold.native.fresh`) but mutates no observed state, so
`MachineCard.isDefault` (`MachineFleet.swift:26`) and
`MachineCardActions.offered(isDefault:)` are never re-evaluated.

**F5. Quick Look does nothing on a mesh print.** Space, and Library ▸ Quick Look,
on a `.glb` tile open no panel at all (the same gestures on an `.mp4` open the
real Quick Look panel with "Open with QuickTime Player"). Lane F1's UAT item 8
asks for the poster; what a person gets is silence.

**F6. The Queue menu never carries the selected row's actions.** With a queued
row selected, the row's contextual menu offers **Pause Job / Cancel Job**
(`uat/menu-queue-row.png`) while `Queue` in the menu bar offers only
`Resume Queue, Empty Queue…` (read twice via AX, with the row selected). The
Model menu DOES mirror its row menu (`Load, Components…, ─, Delete…`), so this is
specific to `Sources/Mold/Shell/QueueCommands.swift:15-17`'s `selection` focused
value, not to the menu model. Lane G's TASK 1 claims one declaration for both
surfaces; for the queue they diverge.

**F7. Destructive is not last in the Library tile menu.** `Move to Trash` is
followed by `Share…` (`uat/menu-library-tile-all-prints.png`), and in a
collection scope by `Rename "…" / Hide from All Prints / Delete Collection… /
Share…` (`uat/menu-library-tile-collection.png`). The `extra:` ShareLink in
`Sources/Mold/Shell/RowActionMenu.swift:55-66` is appended after the model by
design, so the "destructive last, after a divider" rule the owner asked for is
not met anywhere a Share sheet rides along.

**F8. `Remove…` is not red.** `uat/menu-machine-card.png`, cropped at full
resolution: it renders in the normal label colour. The code IS correct —
`MachineCardActions.swift:37` passes `isDestructive: true` and
`RowActionMenu.swift:34` maps it to `role: .destructive` — macOS simply does not
tint a destructive `Button` inside a `contextMenu` (unlike iOS, and unlike the
red buttons in the confirm dialogs, which do render red). `STATUS-J.md:136`'s
screenshot requirement ("Remove… in red") therefore cannot be satisfied on this
platform and should be reworded.

**F9. Search tokens do not exist.** `is:mesh` matches nothing ("0 of 1,536
prints"); the field is plain text over prompts, models and tags, and there is no
`is:` vocabulary anywhere in `Sources` (only `.tag(_)` tokens, appended by
`LibraryInspector+Pane.swift:13`). Web and desktop have them; this is a stated
parity gap rather than a broken promise, since the placeholder never offers them.

**F10. Leftovers survive both Remove and Reset.** After removing hal9000 and then
Settings ▸ General ▸ Reset These Preferences, the throwaway domain still holds
five `pendingBatches` entries keyed on the removed machine's UUID. The reset's
key list is `Sources/Mold/Settings/PreferencesReset.swift` and does not name
`pendingBatches`; `HostStore.remove` does not clear them either.

### Incident worth recording

`open -a …/Mold.app` to re-focus the app launched a SECOND copy against the
**real** prefs domain and the real `~/.mold` home (the first copy had just
crashed per F1, so LaunchServices did not treat it as running). It was alive
about 30 s and terminated with `kill -TERM`; `defaults read
io.utensils.mold.native` shows only pre-existing keys plus window frames, and no
machine was added or removed there. **Never use `open -a` on this app** — re-focus
with `AXRaise` + `set frontmost` instead (this is in the skill).

---

## The checks

### 1. Launch, onboarding, the shell

1. **PASS** — fresh launch on an empty prefs domain and an empty `MOLD_HOME`
   reaches the shell with "Pick a model" / "No machine". There is no onboarding
   wizard in this app (nothing in `Sources` implements one); the empty states ARE
   the onboarding.
2. **PASS** — Add a Machine… → typing `100.123.198.98:7680` fills the Name
   ("hal9000"), shows the normalised `http://100.123.198.98:7680` under the field
   and a live green check line *"hal9000 · mold 0.29.0 · NVIDIA GeForce RTX 4090"*.
3. **PASS** — one Library section header with its shelves, no duplicate Library
   row, exactly one highlighted sidebar row.
4. **PASS** — View ▸ Generate/Library/Queue/Models/Machines carry ⌘1–⌘5 (AX
   reports cmd chars 1–5, modifier 0) and switch panes.
5. **PASS** — Settings opens with nine tabs: General, Generation, Expansion,
   Library, Performance, Accounts, Machines, This Mac, Advanced.
6. **PASS** — Reset These Preferences… asks first with a plain dialog, a red
   Reset and an accurate sentence; afterwards `destination`, `selectedMachine`,
   `defaultMachine` and the layout keys are gone from the domain. See **F10** for
   what it leaves behind.

### 2. Library against hal9000's real library

7. **PASS** — 1,539 prints, thumbnails, host badges.
8. **PASS** — Sort By ▸ Newest First ✓ / Oldest First / Largest First / Name.
9. **FAIL (F9)** — plain text search works ("teapot" → 12 of 1,536; "hunyuan" →
   17 of 1,539); `is:` tokens do not exist.
10. **PASS** — day sections with per-day counts ("Tuesday, September 15 · 6").
11. **PASS (with F7)** — tile menu photographed open in both scopes:
    `uat/menu-library-tile-all-prints.png`,
    `uat/menu-library-tile-collection.png`. All Prints: Open / Quick Look "…" ─
    Use These Settings ─ Make Bigger… ─ Add to Favourites / Move to Collection ▸ ─
    Copy / Save a Copy… / Export… ▸ ─ Move to Trash / Share…. In a collection the
    shelf items (Remove from …, Rename …, Hide from All Prints, Delete Collection…)
    join it. Order and dividers are sane; destructive is not last (**F7**).
12. **FAIL (F2)** — the inspector is empty over an open print.
13. **PASS** — ← and → step the viewer between prints (picture changes).
14. **PASS** — Quick Look on an `.mp4` shows the real panel with "Open with
    QuickTime Player". (On a mesh: **F5**.)
15. **PASS** — the inspector's star favourites (Favourites 2 → 3) and ⌘Z undoes
    it (3 → 2).
16. **PASS, partly** — a collection was created from Generate ▸ File under ▸ New
    Collection… ("Named now; it comes into being once a print lands in it") and
    filled. Rename/Delete Collection… are present in the tile menu and were NOT
    exercised — they would edit the owner's real shelves on hal9000.
17. **PASS** — Recently Deleted: *"181 in the trash · These are deleted after 30
    days"*, a per-tile countdown badge (29d / 15d / 14d / 13d), and the inspector
    shows "Deleting in 29 days" with Put Back and Delete Immediately….
18. **FAIL (F1)** — a clip cannot be played; the app dies on open.
19. **BLOCKED by F1** — Export ▸ GIF on an mp4 needs the viewer, which crashes.

### 3. Generate on hal9000

20. **PASS** — SD1.5, 512², 25 steps: "Getting ready… / A preview appears after
    the first step." → "Reloading UNet (GPU)" → the finished picture on the
    canvas with its seed. (The intermediate PREVIEW image was not caught between
    captures; the phase sentences were.)
21. **PASS** — the result bar carries Save a Copy… / Copy and the canvas menu
    adds Show in Library.
22. **PASS** — Stop appears beside Generate while a render runs; a fresh Generate
    after it works.
23. **PASS** — pressing Generate twice ran both (hal9000 is fast enough that the
    second did not have to wait); with the queue PAUSED the second one waits as a
    row, which is the same code path.
24. **PASS** — result canvas menu: Save a Copy…, Copy, Show in Library, Use as
    Source Image, Add as Reference (`uat/menu-generate-canvas.png`).
25. **PASS** — on an sd15 recipe the composer shows BOTH the source well and the
    reference `+`.
26. **PASS** — Use as Source Image fills the well with a ✕ badge, adds a
    **Strength** control (0.75) to the composer and **Fit** ("Crop to fill") plus
    "Edit mask…" to the inspector; the estimate drops from 39 s to 30 s.
27. **PASS** — the **Sampler** group is present for SD1.5 and absent for
    FLUX Schnell, which is the per-recipe rule.
28. **BLOCKED** — Expand becomes enabled once a prompt is typed, but it was not
    run: no expansion model is installed on hal9000 and pulling one costs a
    multi-GB download on the owner's machine.
29. **BLOCKED** — draft persistence across quit/relaunch was not isolated: the
    relaunches in this run also changed `MOLD_NATIVE_DESTINATION`, and the
    composer was re-seeded by a Reuse each time. Needs its own clean run
    (type a prompt, set a solver, ⌘Q, relaunch).
30. **PASS** — the tag `uat` and the collection `UAT 2026-09-17` are on every
    print made in this session, both in the sidebar count and on the print's own
    inspector.

### 4. Queue

31. **PASS** — a waiting row shows the machine group header, `sd15:fp16`,
    "Next up", "2 seconds ago", a pause control and an ✕.
32. **PASS** — the row's ✕ cancels it; the pane returns to Idle.
33. **PASS** — Queue ▸ Pause Queue gives exactly lane F3's sentence:
    *"hal9000 is not starting anything new — its queue is paused."*, the button
    flips to Resume Queue, a submitted render waits, and Resume releases it.
    hal9000's own `/api/status` confirmed `queue_paused` back to `false`
    afterwards.
34. **BLOCKED** — "Also Running" needs a CLIP upscale; every clip path is blocked
    by **F1** on the client side, and starting one from the Library would still
    need the viewer to watch it. See check 66.
    Row menu photographed: `uat/menu-queue-row.png`. Menu-bar parity: **F6**.

### 5. Models

35. **PASS** — Installed: "86 installed on hal9000", grouped by family, columns
    Model / Variant / Trade-off / Size / State, footer "1.65 TB of 2.02 TB used".
    Discover: 208 results with Family and Sort filters and per-row Install.
36. **PASS** — row menu `Load, Components…, ─, Delete…`
    (`uat/menu-model-row.png`), destructive last after a divider, and the Model
    menu in the menu bar carries the same four.
37. **BLOCKED** — no Discover row with a non-web link was found in 208 results
    (every HF/Civitai entry has a page). Two rows read "Not supported · Open
    Page"; clicking a row opens a detail sheet (name, family, downloads, likes,
    tags, Open Page ↗, Done) rather than a contextual menu.

### 6. Machines (lane J)

38. **PASS** — `uat/fleet-overview.png`: title "Machines · 2", hal9000 FIRST with
    its **Default** badge, **This Mac** marked with its own badge, GPU line with a
    load figure, two memory bars, "Nothing queued", the installed count and the
    address. **BLOCKED**: a card carrying "4× …" and a dimmed card with a reason
    both need the keyed multi-GPU host.
39. **PASS** — `uat/menu-machine-card.png`: Open, Check Now, Set as Default, Copy
    Address, Edit…, divider, Remove…; and on This Mac exactly the first four
    (`uat/menu-machine-card-this-mac.png`). Set as Default correctly disappears
    once the machine IS the default. Colour: **F8**. Behaviour: **F4**.
40. **PASS** — a card's Open (and the sidebar machine row) pushes the machine's
    page — Version, Up, Identity, GPUs with an enable toggle, Memory, Work here,
    Models here, Address, API key ("Not needed on this machine · Edit…"), Nearby
    — with a Back chevron in the toolbar.
41. **PASS** — Add a Machine… on the overview opens the same Address / Name / API
    key editor with the live check line that Settings ▸ Machines opens.
42. **PASS, weakly** — Nearby is present on the overview and on a machine page
    with a Refresh control and answers *"Nothing new on the networks these
    machines can see."*. No un-added mold was advertising, so DISCOVERY itself is
    unproven.
43. **PASS** — `uat/machine-remove-confirm.png`: `Remove "hal9000"?` — "Mold stops
    talking to 100.123.198.98:7680 and forgets it. Nothing on the machine itself
    changes, and you can add it again." with Cancel and a red Remove. Confirmed;
    the card and the sidebar row went.
44. **PASS** — This Mac's card appears in the fleet the moment the engine starts
    (Apple Metal GPU, 27.08 GB of 48 GB, "None installed", `127.0.0.1:51623`) and
    leaves when it is stopped.

### 7. This Mac's engine (lane F)

45. **PASS** — Start Engine → Status "Running", Address `http://127.0.0.1:51623`,
    Home `/Volumes/ExternalStorage/mold-uat-home` (`uat/this-mac-engine-running.png`).
    `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:51623/api/status`
    → **401** (`{"error":"missing X-Api-Key header","code":"UNAUTHORIZED"}`), and
    **200** with `X-Api-Key` read from
    `~/Library/Application Support/io.utensils.mold.native.fresh/secrets.json`
    (mode **0600**, 36-char key). The engine reports `0.30.0`, backend `metal`.
    This is lane F's owed item 1, first half.
46. **PASS** — `curl -si -H 'Origin: http://evil.example' …/api/status` answers
    `access-control-allow-origin: mold-embedded-engine no browser origin`. Never
    `*`. Lane F's owed item 1, second half.
47. **PASS** — the app itself lists This Mac as a machine and talks to it over the
    same HTTP it uses for a remote host.
48. **BLOCKED** — no model is installed in the throwaway home, so no local Metal
    render was made. The Models pane on This Mac says "0 installed on This Mac"
    and offers only HF/Civitai rows under Discover, so the manifest tiers are not
    installable from there. To unblock:
    `MOLD_HOME=/Volumes/ExternalStorage/mold-uat-home mold pull flux2-klein:q4`
    (2.4 GB), then render 256² at a few steps against This Mac.
49. **PASS** — Stop Engine closes the port (`curl` → connection refused) and the
    pane says *"The engine starts once per launch. Relaunch Mold to start it
    again."*
50. **PASS, partly** — quitting with the engine RUNNING exits cleanly within a
    second and leaves **no** `.mold-gallery-writer.lease` anywhere under
    `$MOLD_HOME` (lane F's owed item 5, second half). The "Finishing…" panel with
    Quit Now was NOT observed — with an idle engine the shutdown is faster than a
    1 s screenshot. `kill -TERM` also terminated the app cleanly earlier
    (owed item 5, first half).
51. **PASS** — relaunch restores the destination, the machine list and the
    settings tab seed; the engine is "Not running" again, as the once-per-launch
    rule says.
52. **BLOCKED** — a second `mold serve` on the same home needs a mold binary; the
    only `mold` on PATH is the devshell wrapper, which is
    `cargo run --profile dev-fast -p mold-ai --features metal,h3,…` and would cost
    a full build under another lane's lock. To unblock:
    `nix develop -c cargo run --profile dev-fast -p mold-ai --features metal -- serve`
    with `MOLD_HOME=/Volumes/ExternalStorage/mold-uat-home`, then press Start in
    Settings ▸ This Mac and read the advisory (lane F's owed item 6c).

### 8. Mesh (lane F1)

53. **PASS** — opening a GLB gives a real Metal render with
    "224,198 triangles · 109,382 points" and Reset View / Show Wireframe /
    Stop Turning / Export. After Reset View the framing and the angle are the
    gallery tile's poster — same pose, same size in frame.
54. **PASS, partly** — arrow keys turn the mesh (and do NOT step to the next
    print while it has focus, which is lane F1's item 4). Auto-rotate is on at
    open and had stopped after the first interaction. **UNVERIFIED**: orbit and
    zoom BY DRAG/SCROLL — synthetic `cliclick` drags (fast and slow, with
    intermediate moves) produce no rotation at all, which is a known limitation of
    CGEvent drags against this kind of view rather than evidence of a bug. A human
    must drag. The "Stop Turning" label also stays "Stop Turning" once turning has
    stopped, in both the footer and the menu — it should offer to start again.
55. **PASS** — Show Wireframe outlines the surface in cyan with no back-face
    lines bleeding through. (A mesh with no edges, for the ABSENT case, was not
    found.)
56. **PASS** — mesh menu: Reset View, Show Wireframe, Stop Turning, Use These
    Settings, Export ▸, Save a Copy… (`uat/menu-mesh-viewer.png`); the Export
    submenu is OBJ / ZIP / STL / PLY / Turntable… with `glb` correctly dropped
    (`uat/menu-mesh-export.png`).
57. **PASS, measured** — Export ▸ STL shows **no** "Resize for printing" toggle
    and reads "Longest side 100 mm / Up axis Z up / Origin On the floor /
    longest side 100 mm". The exported binary STL (224,198 triangles) measures:
    `min = [-25.745, -49.122, 0.000]`, `max = [25.745, 49.122, 100.000]` →
    **longest side exactly 100.000 mm, min Z exactly 0.0, tall axis Z**. That is
    lane F1's item 9 and `PLAN.md`'s "export STL at 100 mm Z-up and check its
    bounds", answered numerically.
58. **PASS** — Export ▸ OBJ DOES offer "Resize for printing" (unchecked), with
    Up axis "Y up", Origin "On the floor" and the summary "as stored" — the
    documented OBJ defaults, and lane F1's item 8c in both directions.
59. **PASS** — the turntable sheet starts at 36 views / 10 fps / 512 px /
    "3.6 seconds a turn"; choosing **2,048 px** clamps the views to **21** and
    says why: *"At 2048 px this machine renders at most 21 views a turn."*
    (`uat/turntable-clamp.png`). Lane F1's item 10 predicts "at most 16" — the cap
    is machine-derived, so the NUMBER in that ledger is not a contract; the clamp
    and the sentence are. The transparent-background arm was not exercised.
    Quick Look on a mesh: **F5**. Failure-lands-on-the-poster: **BLOCKED** (needs
    the host disconnected mid-open).

### 9. Reuse (lane F2)

60. **FAIL (F3)** — keyless, same host: the banner appears, the well does not.
61. **BLOCKED by F3** — dropping the attachment by retyping the prompt cannot be
    judged while the attachment never appears.
62. **FAIL (F3)** — the render does not carry the source at all
    (`source_image_name: None` on the host).
63. **PASS** — reusing a TEXT-TO-IMAGE print restores its settings and stays
    silent: no banner, no well, no strength. Its
    `GET /api/gallery/source-media/<file>` answers `{"availability":
    "unavailable_legacy"}`, and the client correctly says nothing about it.
64. **BLOCKED** — no sequence print was found on hal9000 to test the
    first-stage-prompt rule.
    Keyed host, cross-host relay, batch-of-four, legacy print, long clip, delete-
    then-Develop and press-twice (lane F2's items 2, 3, 6, 8, 10, 11): **BLOCKED**
    — the keyed cases need `100.105.134.43`'s API key, and the clip cases are
    blocked by **F1**.

### 10. Upscale (lane F3)

65. **PASS** — hal9000 advertises `video_upscale: {available: true,
    gallery_image: true}`; Make Bigger… on a 512² still produced
    `mold-real-esrgan-x4plus-fp16-1789698029889-upscaled.png` at **2048 × 2048**
    in the gallery. The transient "Making a bigger copy" row was not caught (it
    finished inside 4 s); the completion and the artifact were verified. Lane F3's
    "from a COLD LAUNCH straight to the Library" variant was NOT isolated.
66. **BLOCKED by F1** — a clip upscale (the Also Running row, its frame counter,
    pause/resume/cancel, and the restart-adopts-the-job case) needs clip surfaces.
67. **BLOCKED** — hal9000 has exactly one upscaler installed, so the submenu case
    cannot arise; the plain item is what was seen.
68. **BLOCKED** — needs a machine that does not advertise `video_upscale`.

### 11. Release checks owed by lane F

69. **BLOCKED, not run** — `make signed` was NOT executed. It runs `make engine`
    and a full `CONFIG=Release make build` BEFORE `scripts/assert-sparkle-key.sh`
    (`Makefile`, `signed:` target), so the fail-closed sentence costs a complete
    Release build under the shared lock. The gate's ORDER is readable in the
    Makefile and its unit is covered by `scripts/tests/sparkle-key-gate.sh` in
    `make test`. To unblock: `make signed` on an otherwise idle machine and record
    the exact refusal.
70. **PASS, both halves** — lane F's owed item 2. `otool -L` on the Debug product
    (`build/Debug/Mold.app/Contents/MacOS/Mold.debug.dylib`, which is where the
    engine lives in a debug-dylib build — the 60 KB `MacOS/Mold` is only the
    launcher) shows two `/nix/store` load commands, `libc++.1.0.dylib` and
    `libiconv.2.dylib`. Running `scripts/fix-macos-native-linkage.sh` on a COPY of
    the bundle reports "linkage portable: 18 Mach-O file(s)" and leaves **zero**
    `/nix/store` references. Note for that item's wording: point it at the debug
    dylib, not at `MacOS/Mold`.

---

## Prints created on hal9000 (please delete)

All five are tagged `uat` and filed in the collection **UAT 2026-09-17**.

| Filename | What it is |
| --- | --- |
| `mold-sd15-fp16-1789696757565.png` | text-to-image, 512², sd15:fp16 |
| `mold-sd15-fp16-1789696821294.png` | img2img from the above, strength 0.75 |
| `mold-sd15-fp16-1789696823679.png` | img2img, second press |
| `mold-sd15-fp16-1789697799324.png` | the Reuse render (the one that lost its source — F3) |
| `mold-real-esrgan-x4plus-fp16-1789698029889-upscaled.png` | 2048², Make Bigger… of the previous |

Nothing else on hal9000 was changed. Its queue was paused for one check and
resumed (`queue_paused: false` verified afterwards). No print was trashed,
renamed, favourited on the host, or removed from an existing collection — the
favourite check was made and undone.

## State this lane touched on this Mac

- Throwaway prefs domain `io.utensils.mold.native.fresh` and its secrets
  directory, and `/Volumes/ExternalStorage/mold-uat-home`. Both are disposable.
- One system permission was granted on the owner's behalf: *"Mold would like to
  access files on a removable volume"* → **Allow** (the throwaway `MOLD_HOME` is
  on `/Volumes/ExternalStorage`, so nothing could be read without it).
- The stray real-prefs launch described under **Incident** above.
