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

| Area              | What                                                                                                                                                                                                                                    |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Machines          | Fleet cards, Nearby (Bonjour), add by pairing QR, pasted link or address; keys in the Keychain                                                                                                                                          |
| Generate          | Stills, clips and 3-D objects with each model's own controls; picture wells from Photos, Camera, Files, Library or Share                                                                                                                |
| Library           | Every machine's prints as one grid, browsable offline (saved listings, thumbnails and opened prints, within Settings' storage limit); five pinchable tile sizes; favourites, tags, collections, Recently Deleted; video and 3-D viewers |
| Queue             | Every machine's work; held jobs in words with Pull and Retry, Retry and Move to…; reorder, pause, empty                                                                                                                                 |
| Models            | Installed per machine, Discover, downloads, licences                                                                                                                                                                                    |
| Away from the app | Live Activity with Stop, local notifications, background refresh, widgets, Share extension                                                                                                                                              |

A render notification opens its finished print when the app is in the background
or closed. Notification activation and its system completion callback run on the
main actor; dismissing an alert does not navigate.

## Generation and playback

On iPhone, Generate opens as a scrolling form: choose the kind and machine,
describe the result, then choose a model. Tap **Model** to search installed
models and recipes; **Get More Models…** opens model management directly.
The prompt stays in one scrollable form as the keyboard and text size change;
Generate stays visible above the phone tabs.
The model search sheet keeps a Close action visible when its keyboard is open.
More Options gives Shape, Steps, Batch and Length their own rows, including at
large accessibility text sizes. Clip and 3-D drafts restore their kind after
machines reconnect. A model search with no matches offers Clear Search.
Video playback uses the media audio session, including in silent mode. The
viewer hides the main tabs so its Share, Favourite, Info and Delete controls
remain accessible, while keeping phone back navigation visible. Info has a Done button, and shared files retain their original
filename and media extension. Models uses a text-scaling Installed/Discover menu and
explains when an offline machine's inventory could not be read. Removing a
machine returns to the list. See the [iPhone UAT record](docs/IPHONE-UAT.md).

The iPhone Library uses one navigation title menu for All Prints, Favourites,
collections and Recently Deleted. Media Type filters All Media, Photos, Videos
and 3D within the current shelf. In Select mode, tap tiles or start a sideways
finger sweep to select a range; start on a selected tile to deselect a range.
Reverse the sweep to shorten it, or hold near a grid edge to scroll further.
Vertical swipes still scroll, and tapping selections keeps the viewport put. Its cached listing and images load before
the machines respond. Opening a print keeps the grid's scroll position when
you return; clips play automatically only while their page is selected in the
viewer. Settings is available from Generate, Library and Machines;
Library settings show image storage against its limit and saved listing size,
offer offline thumbnail saving, and
can clear both pictures and saved listings after explaining the offline effect.

## Building

Everything goes through the Makefile. `make help` lists the targets, and inside
`nix develop` the `companion-*` commands wrap them.

Run `nix develop -c companion-dev` to watch iOS and shared Swift sources,
rebuild, install and relaunch after edits. Ctrl-C stops watching.
`companion-run` launches once; `companion-build` only builds. The equivalent
script is `nix develop -c ./scripts/companion.sh dev` (also `run`, `build`,
`gen`, `test`, `uitest`, `packages-test`, and `lint`). These commands serve
the native app; `ios-dev` and `scripts/ios.sh` still serve Tauri.

Every helper accepts Make overrides, for example
`companion-dev SIM=<UDID> BUILD=/Volumes/ExternalStorage/mold-ios-build`.
Xcode must be installed and selected; the devshell provides XcodeGen and Python.

| Target                    | What                                                                                                                                                                                                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `make gen`                | Regenerate `MoldCompanion.xcodeproj` from `project.yml` (the project is generated and gitignored)                                                                                                                                                                        |
| `make build` / `make run` | Build for the simulator, and install and launch it                                                                                                                                                                                                                       |
| `make test`               | Unit tests (Swift Testing) on the simulator                                                                                                                                                                                                                              |
| `make packages-test`      | MoldClient, MoldStyle and MoldMesh's own suites on the iOS simulator                                                                                                                                                                                                     |
| `make uitest`             | Accessibility audit: every destination at xSmall, Large and AX5, in light and dark, on an iPhone, an iPhone SE when one is installed (`xcrun simctl create "Companion SE" com.apple.CoreSimulator.SimDeviceType.iPhone-SE-3rd-generation <iOS 26 runtime>`), and an iPad |
| `make lint`               | Architecture lints (shared ones via `../shared/scripts/swift-lint.sh`)                                                                                                                                                                                                   |

UI tests retry a failed test once in a fresh test process to tolerate hosted
Simulator timing failures. A repeated failure still fails the audit. Full logs
and result bundles are kept under `build/UITestResults/` for each device and
appearance; CI uploads them even when the audit fails. Run
`bash scripts/tests/ios-uitest-runner.sh` from the repository root to verify the
runner's retry limit, reports, and failure propagation without Xcode.

On iPad, the audit checks the Settings form through its sidebar page, while
interaction tests cover the sheet’s Done and Add a Machine actions. This avoids
a UIKit floating-tab loop triggered by the auditor’s private text-size cycling
over a sheet; ordinary system text-size changes are verified separately.

On a disk that fills up, put build output elsewhere:
`make test BUILD=/Volumes/ExternalStorage/mold-ios-build`.
The simulator defaults to an iPhone 17 Pro when one exists (`scripts/pick-simulator.sh`).
To choose another, pass `SIM=<UDID>` from `xcrun simctl list devices available`.
The Makefile accepts simulator UDIDs, not names.

For notification UAT, Debug builds accept
`--notification-fixture-link moldstudio://print/<host-UUID>/<filename>` and
schedule one real system notification after twenty seconds without rendering.
`NotificationTapTests` exercises background and cold-launch taps. Release and
TestFlight builds exclude the fixture.

The Live Activity uses a translucent card with a 48 pt preview, a compact status
and prompt, a full-width progress row, and a machine/queue footer. At large text
sizes it drops secondary content before clipping the status or Stop control.
Debug builds accept `--live-activity-fixture running` (also `waiting`, `finished`,
`failed`, `stale`, or `clear`) for Lock Screen UAT without submitting a render.
The fixture is excluded from Release/TestFlight builds.

## Layout

| Path                      | What                                                                                                                                          |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `Sources/Companion/`      | The app                                                                                                                                       |
| `Sources/Shared/`         | Compiled into the app and both extensions (App Group paths, snapshot types)                                                                   |
| `Sources/Widgets/`        | WidgetKit + Live Activity extension. Reads the App Group only                                                                                 |
| `Sources/Share/`          | Share extension. Stages a photo in the App Group and never networks                                                                           |
| `Tests/CompanionTests/`   | Unit tests                                                                                                                                    |
| `Tests/CompanionUITests/` | The accessibility audit                                                                                                                       |
| `../shared/Packages/`     | `MoldClient` (wire + transport, with `MoldClientTesting`), `MoldStyle` (tokens) and `MoldMesh` (Metal mesh renderer), shared with the Mac app |

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

When machines are offline, Generate and Queue explain that their data is unavailable instead of claiming no models or jobs exist. A saved draft keeps its model while that machine reconnects; choosing a different kind or model explicitly replaces the pending selection.

Render completion notifications use the native iOS banner and Mold Studio app icon, with a short readiness message instead of the full prompt or an image attachment. Tap to open the finished print; cancelled Library refreshes keep existing prints without an error banner.

Generate Options shows aspect-ratio icons in their actual proportions. With a source picture attached, Fit offers centered Crop to fill by default, Fit with borders, Stretch to fill, and Fit + repaint borders when masks are supported. Crop positioning offers horizontal and vertical alignment. Source pixels and painted masks are fitted together before submission. Seed defaults to Random; choose Fixed to reuse a seed. Reset returns fitting to centered Crop to fill and the seed to Random. Explicit saved settings and draft choices remain restorable.

In Library, **View Options → Manage Collections…** opens every collection,
including hidden ones, and offers **Hide from All Prints**. Hiding or showing a
collection updates every machine holding it. Hidden collections remain directly
accessible from the shelf picker and sidebar, while their prints stay out of
the general grid, including when another machine holds the tile’s leading copy.
While Manage Collections is open, the main tab chrome is hidden. Dismissing the sheet restores the Library and its scroll position.

## Optional remote HTTPS access

An authenticated machine can run `mold relay connect --transport aws` to expose its normal API
through a trusted HTTPS gateway. Add that HTTPS address through the existing
Machines flow or pair using a QR whose reachable URL is the public address.
Normal API-key storage, instance identity, device revocation and signed gallery
media tickets still apply; the relay enrollment token is only for the host.
The gateway can see credentials and media and publishes one machine per process.
See [the relay guide](https://utensils.io/mold/deployment/relay).

Hosting requires an explicitly running authenticated `mold serve` and connector.
Native macOS This Mac remains private; the GUI does not automatically open a
tunnel. Offline/sleeping hosts remain unavailable, and interrupted requests
are never replayed by the relay.
