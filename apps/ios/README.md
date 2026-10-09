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
| Library           | Every machine's prints as one grid, browsable offline (saved listings, thumbnails and opened prints, within Settings' storage limit); five pinchable tile sizes; favourites, tags, collections, Trash; video and 3-D viewers |
| Queue             | Every machine's work; held jobs in words with Download and Retry, Retry and Move to…; reorder, pause, empty                                                                                                                             |
| Models            | Installed per machine, Discover, downloads, licences                                                                                                                                                                                    |
| Away from the app | Completion/failure notifications, background refresh, widgets, Share extension                                                                                                                                                          |

A render notification opens its finished print when the app is in the background
or closed. Notification activation and its system completion callback run on the
main actor; dismissing an alert does not navigate.

## Generation and playback

On iPhone, Generate opens as a scrolling form: choose the kind and machine,
describe the result, then choose a model. Tap **Model** to search installed
models and recipes; **Get More Models…** opens model management directly.
The prompt stays in one scrollable form as the keyboard and text size change;
Generate stays visible above the phone tabs. The composer never shows running or finished media: tap the queue count to open live progress and supported previews in Queue. Background queue updates keep attachment browsing open. The prompt has one Expand action.
The model search sheet keeps a Close action visible when its keyboard is open.
More Options gives Shape, Steps, Batch and Length their own rows, including at
large accessibility text sizes. Clip and 3-D drafts restore their kind after
machines reconnect. A model search with no matches offers Clear Search.
On iPhone, rotate a playing clip horizontally to use the full display. Video
keeps its proportions; narrow black bars may remain for a different aspect
ratio. Rotate upright to restore gallery actions, or tap Close to return to
the Library while horizontal. Native playback controls remain available. See the [landscape video UAT record](docs/LANDSCAPE-VIDEO-UAT.md).

Video playback uses the media audio session, including in silent mode. The
viewer hides the main tabs so its Share, Favourite, Info and Delete controls
remain accessible, while keeping phone back navigation visible. Info has a Done button, and shared files retain their original
filename and media extension. Models uses a text-scaling Installed/Discover menu and
explains when an offline machine's inventory could not be read. Removing a
machine returns to the list. See the [iPhone UAT record](docs/IPHONE-UAT.md).

The iPhone Library uses one navigation title menu for All Prints, Favourites,
collections and Trash. A machine search chip scopes Delete, Put Back,
Delete Immediately and Empty Trash to that machine; copies on other
machines stay intact. Without machine chips, these actions apply to all shown
copies. Permanent deletion asks for confirmation, including the print menu.
Media Type filters All Media, Photos, Videos
and 3D within the current shelf. In Select mode, tap tiles or start a sideways
finger sweep to select a range; start on a selected tile to deselect a range.
Reverse the sweep to shorten it, or hold near a grid edge to scroll further.
Vertical swipes still scroll, and tapping selections keeps the viewport put.
Status notices sit above the grid without covering date headings or prints.
Its cached listing and images load before
the machines respond. Closing a print restores the exact viewport, including
partially visible tiles. Scrolling uses cached gallery projections and the viewer
loads only nearby pages, so large libraries stay responsive. Clips play
only while their page is selected in the viewer. Settings → Video Playback controls autoplay (on initially) and repeat (off initially); leaving the viewer or backgrounding pauses playback. Settings is available from Generate, Library and Machines;
Library settings show image storage against its limit and saved listing size,
offer offline thumbnail saving, and
can clear both pictures and saved listings after explaining the offline effect.

### Sources and large pictures

Library tiles and the picture viewer’s More menu offer **Use as Source** without replacing the prompt or model. Reference-only recipes receive a reference; named camera views ask which role to fill. Choose from Library searches the merged gallery, with an All Machines or individual-machine filter. Switching models preserves explicit attachments; incompatible source/reference pictures are disclosed as parked until a compatible recipe is selected.

Oversized new still inputs are proportionally reduced to at most 4096 pixels per axis and 2 MiB, applying orientation and preserving transparency. Smaller compatible inputs keep their bytes. Replacing a source clears its old mask; retained source/mask pairs and exported originals remain unchanged. Fresh drafts use Random; a deliberately locked seed of zero remains valid.

Paired connections prefer a verified LAN endpoint over Tailscale and HTTPS relay, including after network changes. 3-D downloads use a temporary file and size check before decoding; only the selected viewer page loads a mesh, and failures offer Retry.

### Reusing retained source media

Use These Settings consumes a pending Library handoff when Generate first appears as well as when it is already open, then clears the handoff so returning to Generate preserves later edits. It asks every known machine copy for the print's retained source media, including inputs that output metadata cannot describe. A retained source picture appears in the normal source well, where it can be inspected, replaced or removed. A paired retained mask enters the draft with its source picture, so aspect, crop and pad-repaint changes transform both together. Explicit attachments win. Typed descriptor references are disclosed beside the composer and restored at admission: a single output on its original machine uses a one-use session; another machine or a batch relays the bounded file contents. Prompt, seed and size edits keep these disclosed files for repeated submissions; Remove retained sources, choosing a model/kind/recipe, Reset, or another reuse clears their authority. Removing the visible source image never restores it secretly. Unavailable media explains when it must be attached again.

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
<!-- Temporary owner-directed build-first delivery, 2026-10-06: hosted
accessibility matrix disabled; local tests remain available and CI compiles the
native app/shared packages for Simulator instead of running test/lint lanes. -->

in a fresh `xcodebuild` process. The runner reads public `xcresulttool` test JSON
and verifies that every requested retry actually ran and passed; incomplete
reports and infrastructure failures fail the audit. Original and retry logs
and result bundles remain under `build/UITestResults/`, uploaded even on failure.
Export audit drags use the leading Form gutter to avoid UIKit's trailing
scroll-indicator hit region. Library transitions allow 15 seconds for two
identical frame observations on hosted runners; geometry assertions remain strict.
CI splits all test classes into app, references, Library and Library interaction groups, each in light and dark,
while unit/shared-package tests run independently. App UI, shared code, UI tests,
build inputs, and unknown paths require the full audit matrix. Widget Swift,
inert docs, and named static routing/branding contracts can skip app audits;
the always-on lint/unit lane still builds Widgets. Widget changes need their
own Lock Screen/Dynamic Island UAT in light and dark, since app audits never
render those surfaces. Workspace `Cargo.toml` changes trigger lint/unit builds and
native TestFlight delivery; the manifest supplies the marketing version and
cannot change Swift UI. Mixed UI changes still require audits. Dispatch and
unavailable Git diffs default to full audits. `make uitest` locally still
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

Library keeps a compact offline-machine count above its saved prints. Tap the notice for scrollable details with every full machine name and a pinned Done action that scales with text size. Navigation tabs return when details close. Long names and multiple offline machines do not consume the gallery viewport at large text sizes. Decorative Library badges fit the tile width; full names remain in the tile’s spoken label and Info.

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

Hold a Library tile to preview it and open its actions. Preview and drag presentations share the grid's thumbnail loader. Source-image Library selection remains available after a long press. Notification Center uses the same Mold logo in light and dark appearances; iOS controls the notification card background. Native CI runs app, references, Library and Library interaction audit groups in both light and dark in parallel, and TestFlight follows successful native checks without waiting for the full nightly release.

When a Library print is saved on several machines, the source-image picker uses a currently reachable copy, including when the first listed machine is offline.

## Curated models and durable inputs

Discover includes **Mold Models**, the machine’s curated manifest inventory,
including models it has not downloaded. Search by model name, family or Hugging
Face repository, or filter by Hugging Face/Civitai. Source marks accompany the
provider’s name. Get installs the exact checkpoint using the machine’s
credentials; gated models still ask for their licence. Friendly model titles
are shared across surfaces, while runnable IDs and filenames stay compatible.

**Use These Settings** asks the print’s machine for its retained inputs. Source
images and paired repaint masks return together before fitting. Other supported legacy
conditioning files return to editable wells. Descriptor-only typed references
remain disclosed beside the composer across prompt edits and repeated renders; **Remove retained sources** clears them.
The same-machine request uses a fresh one-use session; another machine or a
multi-output batch receives bounded relayed bytes. Missing retained inputs are
explained when the print recorded conditioning; an ordinary text-only print
stays quiet.

Queue rows show the prompt and a sealed input image, labeled by role,
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

Hunyuan3D multiview models offer named Front/Left/Back/Right wells from their recipe. Wan offers a first/last pair rather than arbitrary middle frames; MiniMax **FL2VA** offers separate optional first/last frames. Changing clip length updates the closing frame. Existing Qwen Edit, Qwen Image 2.1 and Flux.2 reference strips honor their source-image relation and count limits; processing pixel budgets automatically resize references in the engine and never require a manual resize. The last Qwen Image 2.1 reference updates the default canvas until you choose a size. SD1.5/SDXL reference weight comes from the model's own control. Model changes park unsupported attachments so they can return. Reuse restores retained typed references with fresh media authority while their original set and order stay unchanged. Changing retained slots requires reattaching the remaining originals; archived bytes never overwrite new attachments. Imported mesh texture/roundtrip workflows remain API/CLI-only.

Attaching a first/start frame selects the closest supported aspect ratio from that image; adding a closing frame preserves it. Crop to fill is the default. The iOS aspect menu marks its current selection with a checkmark.

## Access and saving to Photos

Save to Photos requests add-only access. If access is denied, tap **Open
Settings**, allow adding photos, then return and save again. Camera and nearby
network failures offer the same recovery pattern; Notifications and Live
Activities have recovery actions in app Settings. Restricted access explains
Screen Time or device management. Photo imports use the system picker without
requesting access to your whole library. Auto-save requests Photos access when
you enable it, and saving videos uses PhotoKit's background callback safely.

## Exporting media

Library and result menus offer **Export…** for live MP4 clips and GLB meshes.
The holding machine advertises the formats: GIF/APNG and, where compiled,
WebP for clips; geometry files and animated turntables for meshes. GIF controls
include Loop/Bounce, Forever/Once and an extra pause in milliseconds. **0 ms**
adds no hold and is the choice for a continuous source loop. Frame rate remains
independent. Older hosts do not show the pause control.

Exports can be shared, saved through Files or copied into **On My iPhone/iPad ▸
Mold Studio ▸ Mold**. GIF exports can also be saved to Photos; APNG/WebP use
Share/Files because animation preservation in Photos is not promised. The
original print is unchanged. Texture sidecars appear under **Save Files**;
original media uses the same file destinations. Turntables expose frame count,
size, FPS and transparency; geometry controls follow the host's per-format
size/up-axis/origin defaults. Persistent filenames gain a numbered suffix on
collision. Temporary files stay alive until system delivery finishes.

Local validation and qualification limits: [native acceptance record](../../docs/uat/native-authoring-media.md).

Queue rows show the owning machine’s sealed conditioning images across models. Job Details shows every ordered reference separately with its role, including identity photos, named views, masks, control images and boundary frames. Audio/video references are listed by kind; unavailable previews are disclosed. Inputs remain separate from live denoise previews and are fetched through authenticated routes, including work submitted from another device. Older servers retain their singular source preview.

Native Library marks media added since the previous Library visit with a session-only New badge. Opening a picture, video or 3-D print removes its badge immediately during that visit; prepared neighboring pages and long-press previews do not count as viewing. The first visit still establishes a baseline, and returning for the next visit clears the remaining badges. Before the Library is opened, the iOS Home Screen and macOS Dock icons count new gallery media, once per merged print, excluding hidden collections and Trash. Opening Library clears that count using the existing seen behavior. Icon counts are saved locally across launches; paired machines establish an initial baseline. iOS updates while active and during opportunistic background refresh, subject to notification badge permission; the server has no push. Machine and playback labels share one fitted row on iOS so they never overlap.

On iOS, Use These Settings replaces all active and parked attachments with the selected print’s media. Retained archives restore from the owning machine or another available copy; unavailable conditioning blocks Generate until restored, reattached or explicitly removed. Retry retained media after reconnecting. Late replies cannot replace a newer reuse or revive a source added and removed during restoration.

## Error messages

Errors use concise summaries instead of internal traces. Memory refusals show
estimated requirements, the available budget and the shortfall in decimal GB/MB;
a shared-memory Mac identifies the shared pool. These are estimates on the
generation machine, not a promise that freeing exactly that amount guarantees
a render. Full server diagnostics remain in logs and durable records. Both
native apps share MoldClient error presentation, including older-server replies;
local failures are recorded in OSLog with privacy-protected details.

### Queue model downloads

A missing-model held job offers **Download and Retry** in Queue or Job Details.
The job shows Starting, download-queue status, live bytes/progress, license review,
reconnection and failures in place. Closing details does not stop recovery.
The download runs on the job’s owning machine; retry occurs only after every
returned download ticket succeeds and the original held job and server identity
are revalidated. Failed or cancelled downloads leave the job held. Global queue
pause stays in effect. Cancelling a job does not cancel a model download that
other jobs may need.

### Reuse input media

Reuse settings restores retained opening and closing frames into editable wells, preserving keyframe indices and ordered image references. Audio, source/continuation video, identity photos and control images restore with their saved settings where the selected recipe supports them. Continuation overlap and authored reference strength (including zero) are preserved. Older prints without recorded reference strength keep the server default. The original machine must retain the inputs; missing, damaged or oversized inputs are disclosed before generation. A new attachment or explicit removal wins over a late download, and restored wells never have a hidden archive fallback that can revive removed media.

The Library's View Options includes a Machine picker. All Machines shows the
shared library; choosing one machine shows and edits only its copies. Collection
counts follow that scope. A collection confirmed absent on that machine is marked
“Not on machine”; an unavailable inventory is marked “Unavailable”, rather than
being presented as an empty collection. Hidden collections remain directly
browsable, and Hide from All Prints applies to every same-slug collection across
machines. Pending visibility changes are saved locally and retried when this app
refreshes or reconnects; a closed app cannot reconcile other machines.

Actionable held queue jobs offer Cancel independently of Retry or Move availability. Failure Details shows the machine’s saved job diagnostic and supports copying it together with the machine, model and job identifiers; it does not imply access to the complete machine log. Queue explanations remain concise and use plain English. Current Mold servers enforce held-only cancellation, preventing stale Held actions from stopping a render. Update older machines for this safeguard.

Queue cards keep Cancel, Retry and Pause/Resume in native swipe actions and Job Details. Swipe left to reveal Cancel; swipe right for Pause/Resume, Retry or Download and Retry when the machine permits the action. Otherwise swipe right for Details. Revealing a swipe never activates it. Busy model recovery hides duplicate download/retry actions. Held explanations, model download progress, Move to and Failure Details remain on the card.

Reordering and Empty Queue reserve the jobs they affect until the machine listing refreshes. Pending row actions cannot overlap those operations; offline machines and changed job or machine identities refuse stale requests.
