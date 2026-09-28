# Mold Studio Companion (iOS / iPadOS)

The native SwiftUI app for mold on iPhone and iPad: remote-only, iOS 26+,
bundle `io.utensils.mold.companion`, "Mold Studio" on the Home Screen. It is
the macOS [Mold Studio](../macos/README.md) app's little sibling and installs
beside the Tauri iPhone app (`apps/mobile`), not instead of it.

- **Spec:** [docs/DESIGN.md](docs/DESIGN.md), binding.
- **Plan:** [docs/PLAN.md](docs/PLAN.md), milestones M0–M13.
- **Agent rules:** [`.claude/rules/ios-native.md`](../../.claude/rules/ios-native.md).

## Building

Everything goes through the Makefile. `make help` lists the targets, and inside
`nix develop` the `companion-*` commands wrap them.

| Target | What |
| --- | --- |
| `make gen` | Regenerate `MoldCompanion.xcodeproj` from `project.yml` (the project is generated and gitignored) |
| `make build` / `make run` | Build for the simulator, and install and launch it |
| `make test` | Unit tests (Swift Testing) on the simulator |
| `make packages-test` | MoldClient and MoldStyle's own suites on the iOS simulator |
| `make uitest` | Accessibility audit: every destination at xSmall, Large and AX5, in light and dark |
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
| `../shared/Packages/` | `MoldClient` (wire + transport) and `MoldStyle` (tokens), shared with the Mac app |
