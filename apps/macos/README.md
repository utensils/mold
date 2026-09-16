# Mold for macOS (native)

An experimental native Swift app for mold, on the long-running
`feat/macos-native-app` branch. **This branch is never merged.**

It is a candidate replacement for the Tauri `desktop/` app on macOS, scoped to
generation and the library. No 3-D studio.

## What works today

| | |
| --- | --- |
| **Generate** | Text-to-image. Every control comes from the model's own generation profile, so a model added to mold tomorrow gets correct controls with no change here. Durable submission, live step progress and denoise preview, the finished picture. |
| **Library** | Every machine's prints in one day-sectioned timeline, host-badged, with search, a source filter, zoom and an inspector. Refreshes by ETag. |
| **Queue** | Work in flight per machine, with the host's own actionable reason on each row. |
| **Models** | Variants grouped under the model they belong to, each with the manifest's plain-English trade-off, size and install state. |
| **Settings** | Add, edit and remove machines. Keys go to the Keychain. |

Shortcuts: ⌘1–⌘4 for the destinations, ⌘R to refresh, ⌘↩ to generate, ⌘, for
Settings. Every shortcut is declared once in `MoldCommands` and only *printed*
elsewhere — binding one twice queues the work twice.

## Not built yet

Source images and img2img, video playback and length controls, tags and
collections, trash, export, and downloading a model from inside the app. The
larger remaining piece is running mold's own Rust engine in-process
(`mold_server::run_server` on a thread, reached over loopback) so the app can
render locally on Metal instead of only talking to a remote machine.

## Running it

From inside `nix develop`:

```bash
macos-dev      # build and run, logs on the terminal
macos-test     # package tests
macos-lint     # architecture lints
macos-build    # release build
macos-gen      # regenerate Mold.xcodeproj, then open it in Xcode
```

Or directly: `make help` in this directory.

`Mold.xcodeproj` is **generated** from `project.yml` and is gitignored. Editing
it by hand creates a second source of truth that drifts; change `project.yml`
and re-run `make gen`.

## Pointing it at a server

The app ships with one host, `http://localhost:7680`. To add machines for a dev
run without putting their addresses in the repo, set `MOLD_NATIVE_HOSTS` to
comma-separated `name=url` pairs:

```bash
MOLD_NATIVE_HOSTS='plato=http://10.0.0.5:7680,hal9000=http://10.0.0.6:7680' macos-dev
```

`macos-dev` execs the binary rather than `open`ing it, so the variable reaches
the app and its stdout stays on your terminal. A keyless host needs no API key —
that is a first-class state, not a degraded one.

## Layout

| Path                  | What                                                              |
| --------------------- | ----------------------------------------------------------------- |
| `Sources/Mold/`        | The app. `MoldApp.swift` is the composition root                  |
| `Packages/MoldClient/` | Wire types and transport. **Never imports SwiftUI or AppKit**      |
| `Packages/MoldStyle/`  | Chrome tokens and panel surfaces                                  |

## The rules `make lint` enforces

- `MoldClient` must not import a UI framework — it has to stay usable from
  tests and from anything that isn't this app.
- A concrete backend is built **only** in the composition root, so what the app
  is talking to is a decision in one file. That is what makes running mold's
  Rust engine in-process later a change to `MoldApp.swift` rather than a
  rewrite.
- No literal colors. The app is system light/dark only: semantic colors,
  materials, and the user's own accent. There is no palette to maintain.
- Files over 150 lines are flagged. The Tauri app's `GenerateView.vue` reached
  4,885; this is the guardrail against that.
