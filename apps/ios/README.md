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
Vertical swipes still scroll, and tapping selections keeps the viewport put.
Status notices sit above the grid without covering date headings or prints.
Its cached listing and images load before
the machines respond. Closing a print restores the exact viewport, including
partially visible tiles. Scrolling uses cached gallery projections and the viewer
loads only nearby pages, so large libraries stay responsive. Clips play
automatically only while their page is selected in the viewer. Settings is available from Generate, Library and Machines;
Library settings show image storage against its limit and saved listing size,
offer offline thumbnail saving, and
can clear both pictures and saved listings after explaining the offline effect.

### Reusing retained source media

Use These Settings asks every known machine copy for the print's retained source media, including inputs that output metadata cannot describe. A retained source picture appears in the normal source well, where it can be inspected, replaced or removed. A paired retained mask enters the draft with its source picture, so aspect, crop and pad-repaint changes transform both together. Explicit attachments win. Other retained files are disclosed beside the composer and restored at admission: a single output on its original machine uses a one-use session; another machine or a batch relays the bounded file contents. Prompt, seed and size edits keep these disclosed files for repeated submissions; Remove retained sources, choosing a model/kind/recipe, Reset, or another reuse clears their authority. Removing the visible source image never restores it secretly. Unavailable media explains when it must be attached again.

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

UI tests make one complete pass, then retry only identified failed methods once
in a fresh `xcodebuild` process. The runner reads public `xcresulttool` test JSON
and verifies that every requested retry actually ran and passed; incomplete
reports and infrastructure failures fail the audit. Original and retry logs
and result bundles remain under `build/UITestResults/`, uploaded even on failure.
CI splits all test classes into app, references and Library groups, each in light and dark,
while unit/shared-package tests run independently. App UI, shared code, UI tests,
build inputs, and unknown paths require the full audit matrix. Widget Swift,
inert docs, and named static routing/branding contracts can skip app audits;
the always-on lint/unit lane still builds Widgets. Widget changes need their
own Lock Screen/Dynamic Island UAT in light and dark, since app audits never
render those surfaces. Dispatch and unavailable Git diffs default to full audits. `make uitest` locally still
runs the full target in both appearances; `UITEST_CLASSES` and
`UITEST_APPEARANCES` select a focused run. Run
`bash scripts/tests/ios-uitest-runner.sh` and
`python3 scripts/tests/ios-uitest-retry.py` from the repository root to verify
selection, retry bounds, reports, and failure propagation without Xcode.

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

The Live Activity uses ActivityKit’s default background material so the card and
its semantic text adapt together to light/dark Lock Screen appearances. It has a 48 pt preview, a compact status
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

Library keeps a compact offline-machine count above its saved prints. Tap the notice for scrollable details with every full machine name; long names and multiple offline machines do not consume the gallery viewport at large text sizes. Decorative Library badges fit the tile width; full names remain in the tile’s spoken label and Info.

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

Live Activity identity uses the bundled Mold logo on the Lock Screen and in every Dynamic Island state, including completion and failure; status remains available through text and accessibility labels. System notification banners use the app icon.

Hold a Library tile to preview it and open its actions. Preview and drag presentations share the grid's thumbnail loader. Source-image Library selection remains available after a long press. Notification Center uses the same Mold logo in light and dark appearances; iOS controls the notification card background. Native CI runs app, references and Library audit groups in both light and dark in parallel, and TestFlight follows successful native checks without waiting for the full nightly release.

When a Library print is saved on several machines, the source-image picker uses a currently reachable copy, including when the first listed machine is offline.

## Curated models and durable inputs

Discover includes **Mold Models**, the machine’s curated manifest inventory,
including models it has not downloaded. Search by model name, family or Hugging
Face repository, or filter by Hugging Face/Civitai. Source marks accompany the
provider’s name. Get installs the exact checkpoint using the machine’s
credentials; gated models still ask for their licence. Friendly model titles
are shared across surfaces, while runnable IDs and filenames stay compatible.

**Use These Settings** asks the print’s machine for its retained inputs. Source
images and paired repaint masks return together before fitting. Other supported
conditioning files remain disclosed beside the composer, including across
prompt edits and repeated renders; **Remove retained sources** clears them.
The same-machine request uses a fresh one-use session; another machine or a
multi-output batch receives bounded relayed bytes. Missing retained inputs are
explained when the print recorded conditioning; an ordinary text-only print
stays quiet.

Queue rows show the prompt and the sealed source image, labeled **Source**,
separately from a live **Rendering** preview. Durable images remain readable
after a machine restart. Wide iPad windows keep queue content centered at a
readable width; large accessibility text stacks the image and words.

### Unloading server models

Open Machines → Models → Installed. Server Memory lists loaded models with visible Unload controls. Unload All Models releases every resident model on the selected server while retaining downloaded files. Controls are unavailable offline or while another model operation is pending; server refusals appear in the failure banner.

### Queue details and prompt history

Queue cards show the source image, a short curated model title, prompt excerpt,
and current state. Tap a card for the full model ID, generation settings,
source, progress, and job controls. Pause applies only to waiting jobs; Resume
applies to paused jobs. Retry is offered only for a retryable held job with its
original batch identity. Cancelling jobs are read-only, and controls wait for
an in-flight change to finish.

Prompt History is available directly in Generate. Choose a machine and search
its saved prompts; selecting one changes only the prompt, preserving the model,
settings, and attached media. Loading, offline, unavailable history, and failed
requests have distinct messages. Clear asks for confirmation and removes that
machine's entire history, including prompts hidden by search.

### References and boundary frames

Both native apps expose the server's reference contracts. MiniMax H3 **Ref2VA** takes an ordered mixture of images, H.264 MP4 clips and mono/stereo PCM WAV audio; image references can come from Photos/Camera/Library/Share on iOS or Finder/Library/Paste on macOS, and movie/audio files use Files/Finder. Replace, remove and reorder attachments before generating. Use `image 1`, `video 1` and `audio 1` in the prompt (numbered within each media kind). Audio references need at least one visual reference. The limits are nine images, three videos, three audio files and twelve files total; each clip is 2–15 seconds, with at most 15 seconds of video and 15 seconds of audio including video soundtracks. Authenticated hosts use request-bound upload sessions; keyless hosts accept at most 32 MiB of inline reference media per render. Video clips with sound require authenticated uploads so the server can supply exact decoded soundtrack counts; on keyless hosts use a silent MP4 plus separate PCM WAV audio. Unsupported or oversized files report an error instead of silently disappearing.

Hunyuan3D multiview models offer named Front/Left/Back/Right wells from their recipe. Wan offers a first/last pair rather than arbitrary middle frames; MiniMax **FL2VA** offers separate optional first/last frames. Changing clip length updates the closing frame. Existing Qwen Edit, Qwen Image 2.1 and Flux.2 reference strips honor their source-image relation, count and pixel budgets; the last Qwen Image 2.1 reference updates the default canvas until you choose a size. SD1.5/SDXL reference weight comes from the model's own control. Model changes park unsupported attachments so they can return. Reuse restores retained typed references with fresh media authority while their original set and order stay unchanged. Changing retained slots requires reattaching the remaining originals; archived bytes never overwrite new attachments. Imported mesh texture/roundtrip workflows remain API/CLI-only.
