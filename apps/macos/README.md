# Mold for macOS (native)

An experimental native Swift app for mold, on the long-running
`feat/macos-native-app` branch. **This branch is never merged.**

It is a candidate replacement for the Tauri `desktop/` app on macOS, scoped to
generation and the library. No 3-D studio.

## What works today

| | |
| --- | --- |
| **Generate** | Every control comes from the model's own generation profile, so a model added to mold tomorrow gets correct controls with no change here. Stills and clips (length in seconds, snapped to the family's frame grid), source images with strength, ordered reference images, batches, negative prompts. Durable submission, live step progress and denoise preview, then the picture with Save / Copy / Show in Library. Clicking the picture tucks the controls off the bottom edge, leaving a lip that still carries the step marks; clicking it again or pressing Escape brings them back. A wand on the prompt rewrites it or suggests other ways to say it, in place, with the original kept. The inspector holds the format, an upscaler, whether it is saved at all, what to file it under, and the prompts this machine was last asked for. Batch N is N variations of one idea, not N copies. |
| **Library** | Every machine's prints in one day-sectioned timeline, host-badged. Select with the mouse or the keyboard, open in place, play video, favourite, tag, trash, restore, save, copy, drag to the Finder, and export a clip or mesh into whatever the host will convert it to. Collections are sidebar rows, merged across the fleet by slug, and you file prints by dragging onto one. Search with real tokens (`tag:`, a machine, `is:video`), sort, and set the tile size. Recently Deleted carries each print's own countdown, Put Back and Delete Immediately. Name a print, tag it, file it, and rename or delete a tag across every machine at once. Favourite, tag, filing and renaming are all **undoable** from the Edit menu. Space is Quick Look, and every print can be shared, saved or dragged out. File ▸ Import to adds a picture, clip or mesh from this Mac to a machine. Refreshes by ETag, and follows each machine's live event stream — a print favourited, tagged or trashed somewhere else appears here without a refresh. |
| **Queue** | Work in flight per machine, with the host's own actionable reason on each row, and retry / pause / resume / cancel. |
| **Models** | Variants grouped under the model they belong to, each with the manifest's plain-English trade-off, size and install state. Install and repair with live byte progress. |
| **Machines** | Every machine's page: its GPUs with what each is holding and how much memory is gone, a switch per card where the machine's scheduler will honour one, live memory and CPU, what is queued and installed there, and its address. Machines on the local network that this one can see are offered to add. The machine picked here is the one the Models pane shows. |
| **Settings** | Add, edit and remove machines. An address is normalized the way the other apps normalize it, checked live while you type, and refused when another machine already answers at it; keys go to the Keychain. Storage sets how much disk the media cache may use. |
| **This Mac** | mold's own Rust engine, running in-process on Metal. It joins the machine list like any other and is reached over the same HTTP. |

Shortcuts: ⌘1–⌘5 for the destinations, ⌘R to refresh, ⌘↩ to generate, ⌘, for
Settings, ⌥⌘I for the inspector (on Generate and on Library, each remembering its own), ⌥-click the wand to remix, ⌃⌘S to hide or show the sidebar, ⌥⌘F to
favourite, ⌘⌫ to trash, ⌘Z to undo, Space for Quick Look, Escape to leave the
viewer.
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

LoRAs and identity conditioning, inpainting, the model catalog, chain jobs (scripted sequences are CLI and API only by design), the
3-D studio, and pairing-based onboarding for keyed hosts.

## Four things about the wire that the docs do not say

The first two were found by reading frames off a live host, and both fail
silently; the third is a rule with two halves; the fourth is a refusal.

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
`edit-machine` to open the host sheet empty or on the first machine. The sheet ones exist so a UAT run can photograph it without
a script driving the mouse across the desktop.

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
  Today it prints exactly one line, `LibraryStore`, and that line is the
  honest remaining debt rather than a threshold to raise.
- No `bytes.lines` in `MoldClient`. URLSession's splitter drops empty lines,
  and an empty line is what ends an SSE frame -- the download stream was
  silent for months because of it. `moldLines()` keeps them.

`make test` runs two bundles: the `MoldClient` package (wire types, parsers,
the outbox policy, a stub `URLProtocol` for the transport) and the app's own
`MoldTests`, which launches the app against the same throwaway home `make
uat` uses and drives the stores with a `FakeBackend` that throws on any route
a test did not plant. Every store takes `HostStore` at init, so a test hands
it a fake and nothing else changes.

Every failure a machine reports goes through one funnel, `HostStore.report`,
and shows in one place -- a dismissable line above the pane, never a modal --
so one machine failing says nothing about the machines that worked.
