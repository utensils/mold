# Mold for macOS (native)

An experimental native Swift app for mold, on the long-running
`feat/macos-native-app` branch. **This branch is never merged.**

It is a candidate replacement for the Tauri `desktop/` app on macOS, scoped to
generation and the library. No 3-D studio.

## What works today

| | |
| --- | --- |
| **Generate** | Every control comes from the model's own generation profile, so a model added to mold tomorrow gets correct controls with no change here. Stills and clips (length in seconds, snapped to the family's frame grid), source images with strength, ordered reference images, batches, negative prompts. Durable submission, live step progress and denoise preview, then the picture with Save / Copy / Show in Library. Clicking the picture tucks the controls off the bottom edge, leaving a lip that still carries the step marks; clicking it again or pressing Escape brings them back. |
| **Library** | Every machine's prints in one day-sectioned timeline, host-badged. Select with the mouse or the keyboard, open in place, play video, favourite, tag, trash, restore, save, copy, drag to the Finder, and export a clip or mesh into whatever the host will convert it to. Collections are sidebar rows, merged across the fleet by slug, and you file prints by dragging onto one. Search with real tokens (`tag:`, a machine, `is:video`), sort, and set the tile size. Recently Deleted carries each print's own countdown, Put Back and Delete Immediately. Name a print, tag it, file it, and rename or delete a tag across every machine
at once. Favourite, tag, filing and renaming are all **undoable** from the Edit
menu. Space is Quick Look, and every print can be shared, saved or dragged out. Refreshes by ETag, and follows each machine's live event stream — a print
favourited, tagged or trashed somewhere else appears here without a refresh. |
| **Queue** | Work in flight per machine, with the host's own actionable reason on each row, and retry / pause / resume / cancel. |
| **Models** | Variants grouped under the model they belong to, each with the manifest's plain-English trade-off, size and install state. Install and repair with live byte progress. |
| **Settings** | Add, edit and remove machines. An address is normalized the way the other apps normalize it, checked live while you type, and refused when another machine already answers at it; keys go to the Keychain. Storage sets how much disk the media cache may use. |
| **This Mac** | mold's own Rust engine, running in-process on Metal. It joins the machine list like any other and is reached over the same HTTP. |

Shortcuts: ⌘1–⌘4 for the destinations, ⌘R to refresh, ⌘↩ to generate, ⌘, for
Settings, ⌥⌘I for the inspector, ⌥⌘F to favourite, ⌘⌫ to trash, ⌘Z to undo,
Space for Quick Look.
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

Prompt expansion, LoRAs and identity conditioning, inpainting, the model
catalog, per-machine GPU panels, chain jobs (scripted sequences are CLI and API
only by design), the 3-D studio, and pairing-based onboarding for keyed hosts.

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
`settings`, or `add-machine` / `edit-machine` to open the host sheet empty or on
the first machine. The sheet ones exist so a UAT run can photograph it without
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
