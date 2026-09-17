# Mold for macOS (native)

An experimental native Swift app for mold, on the long-running
`feat/macos-native-app` branch. **This branch is never merged.**

It is a candidate replacement for the Tauri `desktop/` app on macOS, scoped to
generation and the library. No 3-D studio.

## What works today

| | |
| --- | --- |
| **Generate** | Every control comes from the model's own generation profile, so a model added to mold tomorrow gets correct controls with no change here. Stills and clips (length in seconds, snapped to the family's frame grid), source images with strength, ordered reference images, batches, negative prompts. Durable submission, live step progress and denoise preview, then the picture with Save / Copy / Show in Library. Clicking the picture tucks the controls off the bottom edge, leaving a lip that still carries the step marks; clicking it again or pressing Escape brings them back. An Expand button under the prompt rewrites it or, from its menu or with ⌥, suggests other ways to say it, in place, with the original kept; a click picks a suggestion and Use, Return or a double-click takes it. The capsule's own controls are the machine (Auto follows the default machine, or the first that's up; any machine that's up and generates can be pinned, and the model follows it), the shape as aspect and size from the recipe's own ladder, steps, guidance, seed and batch, with the estimate and Generate riding the trailing edge of the last row. The source well takes a file, a print dragged out of the Library, or a picture chosen from the Library in a sheet; the reference strip takes the same three. Generate never turns into Stop: a second press while a render runs queues another batch on the machine, the pane follows them in turn, and Stop stops the one on screen or, from its menu, everything this pane queued. The inspector holds the format, an upscaler, whether it is saved at all, what to file it under, and the prompts this machine was last asked for. Batch N is N variations of one idea, not N copies. The inspector also holds the adapters a model can take, a face to keep, a mask to repaint through, a ControlNet, and everything a clip is made of -- each appearing only when the chosen model says it reads that thing, and each parked, never lost, when you switch to one that does not. |
| **Library** | Every machine's prints in one day-sectioned timeline, host-badged. Select with the mouse or the keyboard, open in place, play video, favourite, tag, trash, restore, save, copy, drag to the Finder, and export a clip or mesh into whatever the host will convert it to. Collections are sidebar rows, merged across the fleet by slug, and you file prints by dragging onto one. Search with real tokens (`tag:`, a machine, `is:video`), sort, and set the tile size. Recently Deleted carries each print's own countdown, Put Back and Delete Immediately. Name a print, tag it, file it, and rename or delete a tag across every machine at once. Favourite, tag, filing and renaming are all **undoable** from the Edit menu. Space is Quick Look, and every print can be shared, saved or dragged out. File ▸ Import to adds a picture, clip or mesh from this Mac to a machine. Refreshes by ETag, and follows each machine's live event stream — a print favourited, tagged or trashed somewhere else appears here without a refresh. |
| **Queue** | Every machine's work, live from its event stream: a batch as one row with its children beneath, drag or Move Up/Down to reorder where the machine will actually put the job, Empty Queue for what is waiting (anything rendering keeps going), and a held row that asks in words -- Pull the missing model then Retry, or the machine's own sentence and Try Again where it says trying again would help. Move to… sends a held job to another machine in the three calls the web app makes, idempotently. The Dock icon counts prints that landed while Mold was in the background, and a finished render or a failed job can notify you. |
| **Models** | Installed is a table grouped by family -- model, variant, the manifest's plain-English trade-off, size, state -- listing every installed model on the machine, with the machine's own disk figure underneath. Discover searches the catalog through the machine: family and sort come from what it advertises, a row installs, reads Installed, or offers its page when the machine cannot take it. Install, repair, cancel, load, unload, components and delete from the row, the Model menu or the keyboard; downloads in a toolbar popover; a gated model's licence rendered from the machine's own payload and accepted in place. Settings ▸ Accounts holds each machine's catalog tokens. |
| **Machines** | Every machine's page: its GPUs with what each is holding and how much memory is gone, a switch per card where the machine's scheduler will honour one, live memory and CPU, what is queued and installed there, and its address. Machines on the local network that this one can see are offered to add. The machine picked here is the one the Models pane shows. On a keyed machine this app already holds an operator key for, its page also lists **paired phones** -- name, when paired, when last used, Revoke -- and Pair a Phone… opens a sheet with a QR code and a `mold://pair` link that expires in under two minutes. The section is absent, not disabled, on a keyless machine, because there is nothing there to pair or revoke. |
| **Settings** | Nine tabs. **General** is this Mac's own preferences -- Appearance (System, Light or Dark, applied to every window; the colours are always the system's), Dock badge and notifications while Mold is in the background, the media cache and Empty Now (absorbed from the old Storage tab), Reset These Preferences. **Generation** and **Expansion** are the render defaults and the eight `expand.*` keys, each pinned to the engine's own `config_keys.rs` registry by a test that parses the Rust, so a bound that moves there is caught here. **Library** and **Performance** hold the trash/authority-log/held-retention and scheduler/port keys. **Accounts** and **Machines** are unchanged from M5 and M2 -- catalog tokens, and add/edit/remove. **This Mac** is unchanged, plus one sentence on why its engine can never be paired. **Advanced** is the escape hatch: every row `GET /api/config` returns for the picked machine, edited from its own value's JSON type (the wire declares no schema at all), a source badge telling env from db from file, a search field, and Reset wherever the source is `db`. Advanced is also the *only* way to reach `logging.*`, `runpod.*`, `lambda.*`, `models_dir`, `output_dir`, and the sixteen per-model `models.<name>.<field>` rows -- nothing curated repeats them. Adding a machine still normalizes its address the way the other apps do, checks it live while you type, refuses an address another machine already answers at, and puts its key in the Keychain, never a plist. |
| **This Mac** | mold's own Rust engine, running in-process on Metal. It joins the machine list like any other and is reached over the same HTTP. |

Shortcuts: ⌘1–⌘5 for the destinations, ⌘R to refresh, ⌘↩ to generate, ⌘, for
Settings, ⌥⌘I for the inspector (on Generate and on Library, each remembering its own), ⌥-click Expand to remix, ⌘[ and ⌘] for the brush and ⌘Z inside the mask editor, ⌃⌘S to hide or show the sidebar, ⌥⌘F to
favourite, ⌘⌫ to trash, ⌘Z to undo, Space for Quick Look, Escape to leave the
viewer. ⇧⌘R checks a selected machine right now, ⇧⌘E and ⇧⌘S export or save a
copy of the Library selection, ⌘F finds in the Library, and ⌘+ / ⌘− make the
grid's thumbnails larger or smaller. ⌘A is the grid's own, not a menu
command -- Edit ▸ Select All is the system's stock item, and this SDK never
routes it to a custom grid, so the grid answers ⌘A itself while Select All
stays present and disabled, the way Refresh does when there is nothing to
refresh.
Every shortcut is declared once in `MoldCommands` or `LibraryCommands` and only
*printed* elsewhere — binding one twice queues the work twice.

What a selection can do is in the **Library menu**, never in a bar that floats
over the grid: the menu bar is what macOS searches from Help, what the keyboard
reaches, and what VoiceOver reads.

## Where the bytes go

A print lives on the machine that made it, and Quick Look, sharing, saving and
dragging one to the Finder all need a real file here. They share one cache,
keyed on `(machine, filename, media_version)` — the same `media_version` the
server builds its ETag from, so a re-rendered poster invalidates cleanly. It
is capped (Settings ▸ Storage) and **emptied when Mold quits**: it exists so a
second look at a clip is free, not to be a copy of the library.

## The local engine

`make engine` builds `rust/mold-macos-ffi` (a staticlib around
`mold_server::run_server`) and rewrites `Engine.xcconfig` to link it. Without
it the app is a remote client and needs no Rust toolchain at all, which is the
point: UI work never costs a 40-minute build. `make engine-clean` goes back.

Five C functions, and nothing about a render crosses them — the app speaks HTTP
to loopback, exactly as it speaks to a machine on the network. Stopping is a
`POST /api/shutdown`, the only shutdown trigger an embedder can reach.

Two consequences worth knowing: the engine starts **at most once per process**
(mold's models-dir override is a process-lifetime `OnceLock`), and once it is
linked, `run_server` installs a process-wide SIGTERM handler — so `pkill` no
longer quits the app.

## Releasing

`make signed` (needs `MOLD_SIGN_IDENTITY`), then `make dmg`, then `make
notarize` — or `make release` for all three. Signing is depth-first and never
`--deep`, which re-signs nested code with the outer bundle's entitlements. The
entitlements allow JIT because candle compiles its Metal shaders at runtime.

## Not built yet

Chain jobs (scripted sequences are CLI and API only by design), the 3-D
studio, and large reference uploads -- mold's upload-session protocol is for
MiniMax H3 and 3-D meshes, neither of which this app makes, so reference
pictures always travel inline.

The app is never a *claimant*. It can issue a pairing for a keyed machine it
already holds an operator key for (Machines ▸ that machine ▸ Pair a Phone…),
but nothing here scans a code or asks to be paired itself. **This Mac's own
engine cannot be paired at all**: it binds `127.0.0.1` with no API key
(`rust/mold-macos-ffi`'s FFI carries a models directory, not a credential),
so there is no key to hand a phone, and a phone could not reach loopback on
this Mac regardless.

## Eight things about the wire that the docs do not say

The first two were found by reading frames off a live host, and both fail
silently; the third is a rule with two halves; the fourth is a refusal; the
fifth is an absence that means yes; the sixth is three small traps in one; the
seventh is about the queue; the eighth is that nothing on the wire says what
a setting is.

`GET /api/events` opens with `event: authority` and then sends **everything
else** as the literal `event: event`, with the real tag in the payload's
`type`. Routing on the SSE frame name finds `event` for every gallery change
and decodes none of them.

`URLSession.AsyncBytes.lines` **drops empty lines**, and in server-sent events
the blank line is the frame terminator. A parser fed by `.lines` sees `event:`
and `data:` arrive and is never told the frame ended: connected, receiving,
silent, no error anywhere. `LineAccumulator` in `MoldClient` is the answer, and
every SSE reader here goes through it.

A GPU's on/off switch is live only when **two** capability flags agree:
`devices.lifecycle` says `PATCH /api/devices/:id` exists, and
`dispatch.v2_authoritative` says the runtime answering it is the one that owns
dispatch. A legacy, observe or maintenance runtime can carry the first without
the second, and persisting a change it cannot enforce is a lie -- so the card
reads as read-only there. `DeviceControl.resolve` is that rule, tested.

A batch's size is the **length of `requests`**, never `batch_size`.
`POST /api/generation-batches` refuses any child whose `batch_size` is not 1,
so Batch N is N one-output requests sharing a prompt, a filing and a logical
batch id, differing only by seed -- the same shape the web app sends. And
per-model defaults are the eight `models.<name>.<field>` config keys and
nothing more; `model_prefs` has no route on any mold.

An absent `source_image` block on a recipe means an IMAGE family that reads
one, not a model that refuses one: the manifest omits the field for every
image family, and reading absence as "no source path" hid the source well on
every still model in the fleet and took inpainting with it. And
`GET /api/loras?model=<name>` is what decides which adapters a model can take,
so no client matches families itself.

Three things about models. "Needs repair" is `downloaded == true` AND
`remaining_download_bytes > 0` -- a model nobody has started also carries its
whole size as remaining, so the remainder alone calls every available model
broken. A licence acceptance names the terms (`id`, `url`, `sha256`), never a
bare id, and the 403 that refuses an install carries the whole payload the
accept route needs, so the flow is refuse, show, accept, retry the identical
install. And a catalog id like `hf:owner/repo` must be percent-encoded as part
of its whole path: encoded alone, Foundation turns the colon before the first
slash into `%3A` -- its guard against a leading segment that reads as a URI
scheme -- and the wildcard route never matches.

The queue. `GET /api/queue` has exactly four states -- `queued`, `running`,
`paused`, `held` -- and `accepted` belongs to the batch endpoint alone; this app
spelled the wrong one for four milestones and every waiting row decoded as
unknown. A reorder's `position` indexes the machine's `queued` rows alone, not
the row's listed `position` (which counts running rows) and not its place on
screen; a batch moves as ascending single-row calls. A held row's typed cause
lives only on the batch child, so `POST /api/generation-batches/status` -- a
read despite the verb -- is what names it, and only a missing or unknown model
is ever typed. A transfer is three calls this app makes -- export from the
source, admit on the destination, complete on the source -- with the export
kept as opaque bytes, because re-encoding it through this build's request type
would drop the media it carries.

**Nothing on the wire says what a setting is.** `GET /api/config` answers
`{key, value, source, env_var?, restart_required}` and the value is the only
type there is -- string, number, bool or null. There is no schema route, no
description, no bounds; the engine's own registry has a type it never ships
and bounds that are literal arguments at each setter. So the curated panes
carry their own knowledge and a test reads `config_keys.rs` to keep them
true, and Advanced renders whatever it is handed from the value's JSON type --
which is what makes a key newer than this build still appear and still work.
Two more traps in the same room: `runpod.api_key` and `lambda.api_key` read
back as the literal string `"<set>"`, so a field that writes back what it read
sets the key to `<set>`; and `restart_required` is true for exactly the three
`scheduler.*` keys, computed by string prefix, which means a key that DOES
need a restart (every `logging.*` one) reports false and the app does not
invent a second opinion. And a `DELETE /api/config/:key` answers
`source: "default"`, but the listing reports a key's storage SURFACE, so a
re-read after a reset says `db` again -- a listing cannot tell a stored value
from a default (`routes_config.rs:48-53`).

## Running it

From inside `nix develop`:

```bash
macos-dev      # build and run against your real settings, logs on the terminal
macos-uat      # the same build against a throwaway prefs domain and mold home
macos-test     # package tests
macos-lint     # architecture lints
macos-build    # release build
macos-gen      # regenerate Mold.xcodeproj, then open it in Xcode
```

Or directly: `make help` in this directory.

`macos-dev` is your setup: the machines you saved, and the mold home every other
mold on this Mac uses. `macos-uat` is disposable — it empties the
`io.utensils.mold.native.fresh` preferences domain and points `MOLD_HOME` at a
scratch directory under `$TMPDIR`, so a run can exercise onboarding, a first
launch and an empty library without costing you your machine list, your models
or your prints. Override the scratch home with `make uat UAT_HOME=/some/path`.
The engine started there has no models installed — that is the point.

## Which mold home it uses

The in-process engine resolves its home exactly the way `crates/mold-core`'s
`Config::mold_dir` does, and has to: it IS that engine, so if the two disagree
one Mac has two libraries. `MOLD_HOME` wins; otherwise the bootstrap pointer at
`~/Library/Application Support/mold/home` — the file Mold Desktop writes when
someone moves their home to another drive — and finally `~/.mold`. Settings ▸
This Mac prints the answer. A home named by that pointer but not currently
mounted is reported rather than recreated, because a fresh empty home in its
place is indistinguishable from having lost everything.

`Mold.xcodeproj` is **generated** from `project.yml` and is gitignored. Editing
it by hand creates a second source of truth that drifts; change `project.yml`
and re-run `make gen`.

## Pointing it at a server

The app ships with no machines. Add one in Settings: a name or an IP is enough,
because `HostAddress` fills in `http://` and port 7680 the way the Tauri app and
the browser build do — `plato`, `10.0.0.5:7680`, `https://box.ts.net` and a
pasted `http://box:7680/api/status` all resolve to the same one origin. The
sheet checks the address while you type and names the machine after the hostname
the server reports, so a box reached by IP still lists under its own name. A
keyless host needs no API key — that is a first-class state, not a degraded one.

To seed machines for a dev run without putting their addresses in the repo, set
`MOLD_NATIVE_HOSTS` to comma-separated `name=address` pairs, in the same
shorthand the sheet accepts:

```bash
MOLD_NATIVE_HOSTS='plato=plato,hal9000=10.0.0.6' macos-dev
```

`macos-dev` and `macos-uat` exec the binary rather than `open`ing it, so the
variable reaches the app and its stdout stays on your terminal.

`MOLD_NATIVE_DESTINATION` forces where the window opens: a destination name,
`settings`, `machines` to open on the machine page, or `add-machine` /
`edit-machine` to open the host sheet empty or on the first machine. `MOLD_NATIVE_QUEUE_FIXTURE=<file>` seeds the queue from a
machine-keyed JSON fixture and refuses every mutation, and
`MOLD_NATIVE_SOURCE_IMAGE=<png>` seeds a source picture, and
`MOLD_NATIVE_LIBRARY_PICKER=1` opens the source well's From Library… sheet at
launch -- all so a run can be photographed without a generation. The sheet ones exist so a UAT run can photograph it without
a script driving the mouse across the desktop.

`MOLD_NATIVE_SETTINGS_TAB=<general|generation|expansion|library|performance|accounts|machines|thisMac|advanced>`
opens the Settings window straight to that tab; an unknown or absent id opens
the first one rather than a blank window.
`MOLD_NATIVE_PAIRING_FIXTURE=<path>` seeds the Machines pane's Pairing
section from a JSON file keyed by machine NAME --
`{"hosts": {"<name>": <PairedClients> | {"clients": …, "session"?: …, "operator_required"?: true}}}`,
the bare shape or the keyed one that also seeds an in-flight session and the
403 state -- and refuses every mutation, the same class of hook as
`MOLD_NATIVE_QUEUE_FIXTURE`.

## Layout

| Path                  | What                                                              |
| --------------------- | ----------------------------------------------------------------- |
| `Sources/Mold/`        | The app. `MoldApp.swift` is the composition root                  |
| `Packages/MoldClient/` | Wire types and transport. **Never imports SwiftUI or AppKit**      |
| `Packages/MoldStyle/`  | Chrome tokens, panel surfaces, layouts                            |
| `rust/mold-macos-ffi/` | The C ABI around mold's engine. Its own cargo root                |
| `scripts/`             | Sign, DMG, notarize                                              |

## The rules `make lint` enforces

- `MoldClient` must not import a UI framework — it has to stay usable from
  tests and from anything that isn't this app.
- A concrete backend is built **only** in `HostStore+Reachability.swift`, so
  what the app is talking to is a decision in one file. That is what made
  running mold's engine in-process a new host in the list rather than a
  rewrite — the lint caught two attempts to construct one elsewhere while that
  was being built.
- No literal colors. The app is system light/dark only: semantic colors,
  materials, and the user's own accent. There is no palette to maintain.
- Files over 150 lines are flagged. The Tauri app's `GenerateView.vue` reached
  4,885; this is the guardrail against that.
- A **type's** size is flagged too: `lint-type-size` sums `Type.swift` and every
  `Type+Concern.swift` beside it and prints anything over 600, because slicing
  a type into files that each pass the rule above does not make it smaller.
  Today it prints two lines, `HTTPBackend` (1,130) and `LibraryStore` (834),
  and both are the honest remaining debt rather than a threshold to raise.
  `HTTPBackend`'s stands on purpose: Swift has no conformance delegation, so
  composing it out of sub-types would cost roughly 190 forwarder lines that
  this same rule sums straight back onto the total (M1.5 S8's rule, held
  again at M7).
- No `bytes.lines` in `MoldClient`. URLSession's splitter drops empty lines,
  and an empty line is what ends an SSE frame -- the download stream was
  silent for months because of it. `moldLines()` keeps them.
- `lint-a11y` is a hard gate: every file under `Sources/Mold` with an
  `Image(systemName:` must also carry `accessibilityLabel`, `.help(`,
  `Label(`, `Label {`, `accessibilityElement`, or the opt-out comment
  `// a11y:`. It is a per-FILE floor, not the audit -- it cannot see whether
  the modifier lands on the glyph that needs it rather than a sibling (a
  tooltip on the row does not name the button inside it), so a file can pass
  the rule and still be wrong. Use the opt-out only for a glyph that is
  genuinely decorative beside its own text.

`make test` runs two bundles: the `MoldClient` package (wire types, parsers,
the outbox policy, a stub `URLProtocol` for the transport) and the app's own
`MoldTests`, which launches the app against the same throwaway home `make
uat` uses and drives the stores with a `FakeBackend` that throws on any route
a test did not plant. Every store takes `HostStore` at init, so a test hands
it a fake and nothing else changes.

Every failure a machine reports goes through one funnel, `HostStore.report`,
and shows in one place -- a dismissable line above the pane, never a modal --
so one machine failing says nothing about the machines that worked.
