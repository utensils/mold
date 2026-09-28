# Mold Studio Companion (iOS / iPadOS)

The native SwiftUI app for mold on iPhone and iPad: remote-only, iOS 26+,
bundle `io.utensils.mold.companion`, "Mold Studio" on the Home Screen. It is
the macOS [Mold Studio](../macos/README.md) app's little sibling and installs
beside the Tauri iPhone app (`apps/mobile`), not instead of it.

- **Spec:** [docs/DESIGN.md](docs/DESIGN.md), binding.
- **Plan:** [docs/PLAN.md](docs/PLAN.md), milestones M0–M13.
- **Agent rules:** [`.claude/rules/ios-native.md`](../../.claude/rules/ios-native.md).
- **User guide:** [utensils.io/mold/guide/companion](https://utensils.io/mold/guide/companion).

## What it does

| Area | What |
| --- | --- |
| Machines | Fleet cards, Nearby (Bonjour), add by pairing QR, pasted link or address; keys in the Keychain |
| Generate | Stills, clips and 3-D objects with each model's own controls; picture wells from Photos, Camera, Files, Library or Share |
| Library | Every machine's prints as one grid; favourites, tags, collections, Recently Deleted; video and 3-D viewers |
| Queue | Every machine's work; held jobs in words with Pull and Retry, Retry and Move to…; reorder, pause, empty |
| Models | Installed per machine, Discover, downloads, licences |
| Away from the app | Live Activity with Stop, local notifications, background refresh, widgets, Share extension |

## Building

Everything goes through the Makefile. `make help` lists the targets, and inside
`nix develop` the `companion-*` commands wrap them.

| Target | What |
| --- | --- |
| `make gen` | Regenerate `MoldCompanion.xcodeproj` from `project.yml` (the project is generated and gitignored) |
| `make build` / `make run` | Build for the simulator, and install and launch it |
| `make test` | Unit tests (Swift Testing) on the simulator |
| `make packages-test` | MoldClient, MoldStyle and MoldMesh's own suites on the iOS simulator |
| `make uitest` | Accessibility audit: every destination at xSmall, Large and AX5, in light and dark, on an iPhone, an iPhone SE when one is installed (`xcrun simctl create "Companion SE" com.apple.CoreSimulator.SimDeviceType.iPhone-SE-3rd-generation <iOS 26 runtime>`), and an iPad |
| `make lint` | Architecture lints (shared ones via `../shared/scripts/swift-lint.sh`) |

On a disk that fills up, put build output elsewhere:
`make test BUILD=/Volumes/ExternalStorage/mold-ios-build`.
The simulator defaults to an iPhone 17 Pro when one exists (`scripts/pick-simulator.sh`).
To choose another, set `SIM="iPhone Air"`.

## Layout

| Path | What |
| --- | --- |
| `Sources/Companion/` | The app |
| `Sources/Shared/` | Compiled into the app and both extensions (App Group paths, snapshot types) |
| `Sources/Widgets/` | WidgetKit + Live Activity extension. Reads the App Group only |
| `Sources/Share/` | Share extension. Stages a photo in the App Group and never networks |
| `Tests/CompanionTests/` | Unit tests |
| `Tests/CompanionUITests/` | The accessibility audit |
| `../shared/Packages/` | `MoldClient` (wire + transport, with `MoldClientTesting`), `MoldStyle` (tokens) and `MoldMesh` (Metal mesh renderer), shared with the Mac app |

## Background behaviour

mold servers cannot push to a phone. The app follows a render while it is in
the foreground, keeps a ledger of pending batches in the App Group, and asks
about them again from `BGAppRefreshTask` (`io.utensils.mold.companion.refresh`)
and on every return to the foreground. Live Activities carry a stale date so
the Lock Screen says when it may be out of date. To try background refresh in
the simulator, use Xcode's Debug ▸ Simulate Background Fetch.

## Distribution

`.github/workflows/testflight-ios-native.yml` uploads to TestFlight after
`iOS native app` passes on `main`, once the repository variable
`COMPANION_TESTFLIGHT` is `true`. The App Store Connect record ("Mold Studio
Companion", `io.utensils.mold.companion`) has to be created by hand first:
there is no API for it. The App Review note explains `NSAllowsArbitraryLoads`:
this is a client for self-hosted servers at any address the owner chooses.
