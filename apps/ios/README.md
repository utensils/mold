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

| Area              | What                                                                                                                                                                                                                         |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Machines          | Fleet cards, Nearby (Bonjour), add by pairing QR, pasted link or address; keys in the Keychain                                                                                                                               |
| Generate          | Stills, clips and 3-D objects with each model's own controls; picture wells from Photos, Camera, Files, Library or Share                                                                                                     |
| Library           | Every machine's prints as one grid, browsable offline (saved listings, thumbnails and opened prints, within Settings' storage limit); five pinchable tile sizes; favourites, tags, collections, Trash; video and 3-D viewers |
| Queue             | Every machine's work; held jobs in words with Download and Retry, Retry and Move to…; reorder, pause, empty                                                                                                                  |
| Models            | Installed per machine, Discover, downloads, licences                                                                                                                                                                         |
| Away from the app | Completion/failure notifications, background refresh, widgets, Share extension                                                                                                                                               |

A render notification opens its finished print when the app is in the background
or closed. Notification activation and its system completion callback run on the
main actor; dismissing an alert does not navigate.

## Generation and playback

Secure connection startup retries brief proxy pressure up to three times within its existing deadline. Authentication failures and requests already sent are never retried by this startup recovery.

On iPhone, Generate opens as a scrolling form: choose the kind and machine,
describe the result, then choose a model. Tap **Model** to search installed
models and recipes; **Get More Models…** opens model management directly.
On updated machines advertising duration video jobs, clips use one prompt and **Duration**. The initial value follows the selected recipe; changing models preserves a chosen duration. The server reports the single-generation limit and actual number of clips joined into one movie. **Define scenes** optionally supplies separate prompts. Source fitting, opening/closing frames, typed references, audio, batch variations and filing settings use the ordinary preparation path. Generate creates durable parent jobs; Queue shows each movie once with the server’s progress, Stop, Resume and explicit deletion of recovery data. An unanswered submission remains a known attempt to check in Queue before starting another. Saved durations and scenes survive a cold launch in the private draft file. Older machines keep the existing single-generation controls and authored-sequence drafts keep their separate-part behavior.

Library **Extend Video…** prefills the original recipe and **Add duration**, preserving the original movie. Its machine and source identity stay pinned across a cold launch. Original typed references remain visible and are inherited unless replaced; historical opening and closing images are cleared. For continuous extension, the selected video supplies the first frame. An explicitly attached opening image remains visible with an actionable conflict instead of being discarded.

The keyboard’s **Done** dismisses the active ordinary or clip prompt without losing edits, including after scrolling. Options and the expanded prompt editor dismiss their keyboard when closed.
The prompt stays in one scrollable form as the keyboard and text size change;
Generate stays visible above the phone tabs. The composer never shows running or finished media: tap the queue count to open live progress and supported previews in Queue. Background queue updates keep attachment browsing open. The prompt has one Expand action.
After a dropped connection, foreground reconnection rereads admitted batches and resumes progress without submitting them again.
Backgrounding pauses progress streams while an in-flight upload/admission keeps its finite background allowance. Accepted batches remain tracked for foreground recovery. Stop before acceptance cancels the request and, if a late response confirms acceptance, sends cancellation for that admitted batch; an unconfirmed cancellation remains in the recovery ledger.
The model search sheet keeps a Close action visible when its keyboard is open.
More Options gives Shape, Steps, Batch and Length their own rows, including at
large accessibility text sizes. Clip and 3-D drafts restore their kind after
machines reconnect. A model search with no matches offers Clear Search.
On iPhone, rotate a playing clip horizontally to use the full display. Video
keeps its proportions; narrow black bars may remain for a different aspect
ratio. Rotate upright to restore gallery actions, or tap Close to return to
the Library while horizontal. Native playback controls remain available. See the [landscape video UAT record](docs/LANDSCAPE-VIDEO-UAT.md).

Video playback uses the media audio session, including in silent mode. Clips
stream in bounded chunks through the machine's authenticated media connection,
including when seeking. The
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
the machines respond. Each machine is followed through its gallery change feed
when it offers one (older machines through their conditional listings), so a
refresh, a new print or a burst of gallery events reads only what changed, and a
refresh that finds nothing new leaves the grid untouched. Closing a print restores the exact viewport, including
partially visible tiles. Scrolling uses cached gallery projections and the viewer
loads only nearby pages, so large libraries stay responsive. Clips play
only while their page is selected in the viewer. Settings → Video Playback controls autoplay (on initially) and repeat (off initially); leaving the viewer or backgrounding pauses playback. Settings is available from Generate, Library and Machines;
Library settings show image storage against its limit and saved listing size,
offer offline thumbnail saving, and
can clear both pictures and saved listings after explaining the offline effect.

### Sources and large pictures

Use **View** (native apps) or **View original** (web, desktop and mobile) beside an attached input to inspect it without replacing or removing it. References, source/target pictures, boundary frames, masks, control and identity pictures, and attached video/audio have separate viewing actions. Image previews fit the whole picture. Use **Actual size** on desktop and web, or pinch to zoom in the native iOS app, for close inspection. **Done** or Escape returns to the attachment or picker. Picker previews do not select or attach the picture. Queue viewing actions show the full submitted media, which may already have been fitted to the generation canvas; older machines without the original-input endpoint report it unavailable.

Library tiles and the picture viewer’s More menu offer capability-based image attachments without replacing the prompt or model. **Use as first frame** and **Use as last frame** use H3/Wan endpoints or LTX interpolation anchors; replacing an LTX endpoint preserves interior anchors. Reference-only recipes receive a reference; named camera views ask which role to fill. Choose from Library searches the merged gallery, with an All Machines or individual-machine filter. Its full-width filter row stays below native navigation chrome and above the pictures at every text size. Switching models preserves explicit attachments; incompatible source/reference pictures are disclosed as parked until a compatible recipe is selected.

Oversized new still inputs are proportionally reduced to at most 4096 pixels per axis and 2 MiB, applying orientation and preserving transparency. Smaller compatible inputs keep their bytes. Replacing a source clears its old mask; retained source/mask pairs and exported originals remain unchanged. Fresh drafts use Random; a deliberately locked seed of zero remains valid.

Paired connections prefer a pinned encrypted LAN endpoint over Tailscale and Lambda relay, including after network changes. A fast LAN handshake avoids a cloud probe; slow proxy or alternate-address checks do not delay the working local route. Nearby discovery also recovers a paired machine after its LAN address changes, preserving its saved identity and key. Unavailable machines are checked again while the app is open, and a Nearby lookup ends within five seconds with an actionable retry instead of an indefinite spinner. Saved Prints offers Retry to reconnect and refresh; connection messages use the available width at larger text sizes and disappear when their request succeeds. 3-D downloads use a temporary file and size check before decoding; only the selected viewer page loads a mesh, and failures offer Retry.

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

## Encrypted remote access

Pair with a fresh version 2 QR or pairing link. If a link opens the pairing webpage, choose **Open in Mold Studio Companion** to hand the same private invitation to the native app. The app asks before claiming it. In-app pasted links stay within a three-line scrolling field so Pair remains reachable. The link pins the host’s signing and Noise keys; each installation keeps its private device identity and API credential in the iOS Keychain. Secure is the default for a new connection. Explicit Regular pairing uses a separate one-use code and direct HTTP(S) API credential; existing direct credentials remain usable. Regular never falls back to Mold proxy, which always requires end-to-end encryption. Changing mode requires deliberate pairing or a separate Regular API key.

MoldClient’s Secure transport uses the same Rust Noise implementation as the server. Secure LAN, Tailscale and Lambda relay routes carry encrypted HTTP records over WebSocket, including API headers, uploads, live events, thumbnails, original files and retained media. The relay sees routing metadata and ciphertext. Video playback uses an AVFoundation resource loader with encrypted HEAD/Range requests, so seeking does not buffer an entire movie or expose a bearer URL. Requests are bounded to 64 MiB; responses stream with per-stream credit. Lost mutations are never replayed.

Secure LAN-only hosts can pair and reconnect without a cloud endpoint or cloud account. Their pairing codes pin the host keys and traffic uses the encrypted channel. Explicit Regular HTTP adds no encryption; HTTPS or Tailscale encrypts transport when used.

When both cloud peers support bulk delivery, consecutive Noise ciphertext frames may travel through short-lived S3 objects. Native clients pin the storage origin independently, reject redirects, and verify the full bounded ciphertext object before decrypting. The video and HTTP interfaces stay the same; LAN and small messages stay inline. The native carrier keeps at most four objects totaling at most 1 MiB in flight, with a 1 MiB body-credit window and coalesced paused-body buffers. WSS capability detection uses the selected route because Foundation reports its upgrade request as HTTPS. Gateway throttling retries exact pending ciphertext under the existing deadline, without reopening HTTP mutations. Retired proxy credentials are removed while existing direct Regular credentials remain in Keychain. Forget waits for credential removal; a locked store retains the machine and reports the failure.

Nearby routes become usable only after a handshake verifies the paired host keys. A fast LAN avoids cloud connections, lower-priority routes start after a short stagger, and stale checks cannot overwrite a recovered or removed machine. Backgrounding closes authenticated channels; foregrounding reconciles state. Device grants renew through the encrypted connection before expiry, and revocation is enforced by the host.

Builds require stable Rust as well as Xcode. `make gen`, `make build` and `make archive` automatically build the GPU-free `mold-secure` XCFramework for Mac, iPhone and Simulator. Generated binaries stay out of Git. See [the relay guide](https://utensils.io/mold/deployment/relay) and [the protocol contract](../../relay/aws/SECURE-PROTOCOL.md).

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

Queue cards reserve a preview square for every job. Without a source image, a
neutral image outline keeps the row balanced; loading and unavailable previews
have separate accessible descriptions. The Mac queue uses the same placeholder.

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

The **Boundary frames** fit controls remain available with only the last frame
attached. **Crop to fill** is the default; crop, pad and resize apply to the
submitted endpoints while preserving the original attachments for later edits.

Both native apps expose the server's reference contracts. MiniMax H3 **Ref2VA** takes an ordered mixture of images, H.264 MP4 clips and mono/stereo PCM WAV audio; image references can come from Photos/Camera/Library/Share on iOS or Finder/Library/Paste on macOS, and movie/audio files use Files/Finder. Replace, remove and reorder attachments before generating. Use `image 1`, `video 1` and `audio 1` in the prompt (numbered within each media kind). Audio references need at least one visual reference. The limits are nine images, three videos, three audio files and twelve files total; each clip is 2–15 seconds, with at most 15 seconds of video and 15 seconds of audio including video soundtracks. Authenticated hosts use request-bound upload sessions; keyless hosts accept at most 32 MiB of inline reference media per render. Video clips with sound require authenticated uploads so the server can supply exact decoded soundtrack counts; on keyless hosts use a silent MP4 plus separate PCM WAV audio. Unsupported or oversized files report an error instead of silently disappearing.

Hunyuan3D multiview models offer named Front/Left/Back/Right wells from their recipe. Wan offers a first/last pair rather than arbitrary middle frames; MiniMax **FL2VA** offers separate optional first/last frames. Changing clip length updates the closing frame. Existing Qwen Edit, Qwen Image 2.1 and Flux.2 reference strips honor their source-image relation and count limits; processing pixel budgets automatically resize references in the engine and never require a manual resize. The last Qwen Image 2.1 reference updates the default canvas until you choose a size. SD1.5/SDXL reference weight comes from the model's own control. Model changes park unsupported attachments so they can return. Reuse restores retained typed references with fresh media authority while their original set and order stay unchanged. Changing retained slots requires reattaching the remaining originals; archived bytes never overwrite new attachments. Imported mesh texture/roundtrip workflows remain API/CLI-only.

Boundary frames select the closest supported aspect ratio from the first frame while it is present, or from the last frame when the first is empty. Removing the first transfers automatic sizing to the last; replacing the driving endpoint updates it again. Manual and reused canvas sizes survive removal. Crop to fill is the default. The iOS aspect menu marks its current selection with a checkmark.

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

Queue rows show the owning machine’s sealed conditioning images across models. Job Details shows every ordered reference separately with its role, including identity photos, named views, masks, control images and boundary frames. Audio/video references are listed by kind; unavailable previews are disclosed. Inputs remain separate from live denoise previews and are fetched through authenticated routes, including work submitted from another device. Older servers retain their singular source preview. Source previews share a 32 MiB memory budget and at most four concurrent reads. Eviction keeps ordered input labels and allows previews to load again when needed.

Library New badges and unread counts are saved on each client independently. Newly discovered media stays new while browsing Library. Viewing an individual print clears its New label immediately; leaving Library acknowledges the eligible prints shown by the projections visited there. Hidden or filtered-out prints, Trash, and arrivals after departure remain unread. Counts and badges deduplicate merged copies and exclude hidden collections and Trash. Existing read status is preserved during upgrade, and a newly connected machine establishes a historical baseline; subsequent arrivals require viewing or a Library visit. Native iOS also badges its Images navigation. Native app-icon counts follow the same local history, subject to the existing badge preference or permission. iOS refreshes while active and opportunistically in the background; the server has no push.

On iOS, Use These Settings replaces all active and parked attachments with the selected print’s media. Retained archives restore from the owning machine or another available copy; unavailable conditioning blocks Generate until restored, reattached or explicitly removed. Retry retained media after reconnecting. Late replies cannot replace a newer reuse or revive a source added and removed during restoration.

## Appearance

Settings → Display → Appearance offers **System**, **Light** and **Dark**. System
is the default and follows iOS. The choice applies immediately to the app and
presented sheets, persists across launches, and is local to this device.

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

### Move queued work to another machine

When another connected machine can generate, **Move to…** is available on queued, paused, and held jobs until the source machine begins rendering. The prompt, seed, settings, and retained reference media move together. The destination must accept the job before the original is removed. Older servers support Held-only moves. Jobs that depend on machine-local LoRAs or workflows are refused without moving them.

If a connection is interrupted during a move, retry the same destination so Mold can check whether it already accepted the job. The source remains reserved while that result is unknown, preventing duplicate rendering. A confirmed destination rejection releases the original only after the destination records that this transfer cannot be admitted later. A failed source lookup keeps the move unresolved. Only a confirmed missing original with a verified destination counts as removed; a job that has already started on the source is reported as no longer eligible.

### Continuous Library rows

Library thumbnails on native macOS, native iOS, Tauri desktop and web use continuous justified rows: each picture keeps its aspect ratio, with narrow seams and square corners. Thumbnail size and window changes reflow the rows while keeping a visible print in place. The final incomplete row stays left aligned without enlarging a lone print. iOS thumbnails omit machine-name badges; image details and machine filtering still show the owning machines.

Long prompts use **Edit prompt** in Generate. Its large native editor keeps the
live draft, supports multiline text, Recent prompts, Clear/Undo clear and the
existing Expand menu; Done or dismissal retains edits. Settings ▸ Library has a
persisted **Show date separators** preference, on by default, for Library, Search
and collection shelves.

Deleting the media currently open in a Library viewer keeps the viewer open on the next surviving item in the current filtered order, or the previous item at the end. It returns to the grid only when no items remain. A surviving machine copy stays visible after a partial deletion failure. Native macOS successful moves to Trash finish without a persistent completion notice; progress, interrupted work and failures remain visible.

### Source Media

Open Source Media from Library (or the iPad Shelves sidebar) to browse original
inputs retained by all live library prints, including hidden collections. Each print's Info also lists its
retained inputs, including ordered references and boundary frames. Image previews
are authenticated thumbnails; Use source imports the original into a compatible
input of the current Generate model. Audio and video show their media icons and
can be attached when the selected recipe accepts them. Unknown types remain
listed with an explanation. Choose the owning print to return to its viewer.

Source Media creates no independent pins. Sources in Trash remain accessible in
that print's Info, and permanent deletion releases its ownership. Hidden
collections remain excluded from general source browsing. Older servers retain
per-print browsing; individual keyframe reuse requires the content endpoint.

Library connection status is always available in the toolbar. Offline and reconnecting updates change its icon and details without moving the gallery. Length choices use the selected frame rate and duration-aware frame bounds, keep a valid custom selection, and never offer a zero-second item. Changing rate or recipe clamps the authored frame count before display; LTX keeps its 20-second admission ceiling. Qualified models advertising duration prediction offer Automatic from prompt or an explicit clip length; the authored choice survives draft restoration and settings reuse, and parks on unsupported destinations.

The Queue navigation badge reads every server page and totals unfinished jobs across all machines: rendering, queued, paused, held, and cancelling jobs. Finished, failed, and cancelled history is excluded; batches contribute their individual unfinished children, with runtime and durable copies counted once per machine.

Source Media combines identical verified source bytes into one card while preserving every print, role and machine in **Details and roles**. Unknown content identities remain separate. Choose a copy there before **Use source** to select its exact authoring origin. The lazy thumbnail grid shows all discovered sources; the Library sizing controls resize it without reloading inventories. Inventories survive navigation, reordering and display-only edits until a relevant content/authority event or explicit Refresh revalidates them. Private previews stay in a 32 MiB decoded-memory cache, share repeated requests, and allow at most four downloads/decodes at once. They are never written to an offline source library; visibility and machine identity continue to fence preview and reuse.

Source previews and inventories do not poll or expire on a timer. Content changes expire only the affected owner; a reconnect resync expires that machine. Closing Sources keeps the bounded warm cache, while removed routes and removed owners are evicted across the app. Live owners from hidden collections appear in Source Media; Trash-only owners remain accessible through their print details. Source reuse rechecks the publication and destination draft after download.

Held jobs with a server-verified missing model component (including a missing adapter) offer **Download and Retry** directly in Queue and Job Details. The original explanation and Failure Details remain available alongside Move to. Downloads use that job’s exact model and machine, retain normal license review, and retry only after every returned download ticket succeeds. Malformed or mismatched recovery hints never redirect a download; older servers retain their typed missing-model recovery.

Active model downloads and repairs keep model names and IDs full width, with progress and a compact **Cancel** control underneath. The control remains labeled **Cancel Download** for VoiceOver and retains a 44-point touch target. Accessibility text sizes stack progress text and actions.

Source Media stays in its own Library section and reuses the Library machine, media-type, sort, thumbnail-size and tokenized-search controls, with a collection selector that includes hidden collections. Technical IDs, hashes and reuse actions are in source Details. Open a source to inspect its original image, video or audio, file details, and every eligible use (including multiple roles and keyframe positions in the same print). Choose an exact copy for reuse or open an owning print to manage retention. Native viewers stream only the selected original into a private temporary file, delete it on dismissal, and bound image decoding; originals larger than 128 MiB are not previewed. No originals are downloaded just by scrolling. Inventory discovery starts with four owners per machine, then batches up to 100 with two concurrent requests; fresh repeated visits make no new inventory requests.

Native source details use an indexed visibility projection and owner-specific
invalidation. Unrelated print updates do not restart the selected original.
Unchanged rows remain visible during refresh; refused inventories revoke open
source access. Legacy originals with unknown metadata are identified from their
contents only when opened, with bounded decoding and cancellable playback inspection.

Camera-scanned Regular pairing codes show the connection mode and direct machine address before Pair claims them. Secure remains the default connection choice.

In Library, tap **Select**, choose prints, then tap **Add to Collection** in the selection bar. Choose an existing collection or **New Collection…**, enter a name and tap **Create** to file the selection at once. The same action is available under **More** and a print’s context menu. Organization actions remain available while capability information is unknown; machines that explicitly refuse organization disable these edits. Changes use the ordinary per-machine mutation queue and retry after reconnection.

Held Queue actions use separate full-width rows for Download and Retry and Move to, including in Job Details, with wrapping labels at large text sizes. Input captions sit in the job text column so long reference labels do not squeeze the model and prompt.

## Finding models

Model search in Generate, Installed and curated Discover accepts partial names,
ordered abbreviations and small spelling mistakes. Case, accents and separators
such as `-`, `_`, `:` and `.` do not matter. Try `flx dev`, `flxu dev`, or
`fluxdevq4`; multiple words can match the model name, family or repository.
Direct name matches rank ahead of fuzzy matches, and clearing search restores
the existing list. Selection, machine routing and capability restrictions stay
unchanged. Search is available even when only a few models are installed.

Add-on look selectors and supported upscaler/adapter selectors use the same
matching behavior. Filtering a select keeps its current value visible.

Live Hugging Face/Civitai discovery uses the provider's normal search first.
After a successful zero-result query, an updated server searches a broader
prefix, filters at most 100 returned candidates with fuzzy matching, and paginates
that matching set. Source, family, kind and mature-content filters still apply.
Provider errors remain errors; this bounded fallback is not an exhaustive search
of either provider's index. Older servers retain their provider search behavior.

**Batch** in More Options applies to pictures and clips. Switching kind preserves the count; choose **1** to disable batching. A grouped Queue row exposes **Cancel Batch** even while collapsed. It cancels each eligible live child on its owning machine, preserves completed outputs, and respects older machines that cannot stop a running job.

Library print/selection/viewer menus offer **Copy File Name** (one filename per line for
selections), and Info file values are selectable. Source cards/details show the owning machine,
selectable `source:<print>#<member>` and optional SHA-256. **Source actions** offers Copy Source ID,
Copy CLI Command (shell-quoted with the owning host), original image/file clipboard, Save a Copy
and native Share. Downloads use the exact owner’s bounded authenticated original and are
revoked when the view or authority changes. The ID belongs to that host’s copy.
An authenticated empty source inventory, or a legacy no-retained-media answer for a print
without input provenance, displays **No source media.** without Retry. Missing/damaged,
unauthorized and failed reads remain distinct and retryable. No background sync is added on iOS.

Print menus offer explicit first/last frame reuse for supported recipes. LTX interpolation
keeps interior anchors and carries an existing first-frame source into keyframe 0 when a last
frame is added. Before an older machine supplies a recipe, **Use as source image** remains
available; an explicit unsupported recipe still controls which actions are offered.

Fresh MiniMax H3 typed video references can also use MOV/HEVC, M4V, WebM and Matroska when the keyed host advertises video normalization. The file picker preserves the bounded original bytes; **Preparing video on <machine>** probes or converts them on that machine before pricing and admission. Auto chooses an eligible connected host; an explicit machine choice stays fixed. Exact canonical facts and receipts must finalize before generating. Changing the host route, key, pairing or server instance cancels preparation and releases its session. Drafts keep original files, never spent upload handles. Older/keyless hosts retain the existing H.264 MP4 path and explain unsupported files. A malformed advertised normalization V1 contract surfaces a specific error when importing or submitting fresh video while preserving the host’s other capabilities. Absent, unversioned or unsupported future protocols follow the compatible canonical path. This does not add normalization for scalar source/continuation-video inputs or archived replay.

Video playback holds the last real presentation frame when it ends or is scrubbed to the full end. Play restarts at zero, and Repeat still loops. Audio may finish after the final video sample; its legitimate tail is preserved. Encrypted clips use AVFoundation’s native preroll to become ready without starting playback or waiting for an arbitrary buffer length. An unexplained native preroll failure is shown as a preparation error. Item changes, dismissal and inactivity fence pending seeks so an old clip cannot restart the next one.

The Queue continues polling running and cancelling jobs and shows the host's
observational stall diagnostic with its stage, elapsed gap, cancellation intent
and device ownership. Job Details includes copyable diagnostics and operator
recovery guidance. Stalled work suppresses stale ETA, completion fractions and
step announcements; real decode counters describe completed blocks or chunks
within their current phase, rather than completed video frames. A successful
202 cancellation response means the request is pending while the device remains
held. The app follows authoritative queue/preview snapshots until real worker
return, and a null diagnostic clears the stalled presentation when work resumes.

Opened images and videos keep their full original files in the device's offline
library. Video playback downloads one complete file before starting, then plays
locally; the loading state says “Downloading for offline playback…”. Opening,
sharing, saving the original to Photos or Files, and original-file export reuse
that cache without a machine connection. Remote generation and format conversions
still require their machine. Settings → Offline Storage includes Unlimited;
finite limits evict older files, and an image or video that cannot be retained
says “Not saved offline”. A thumbnail-only fallback instead says it is a preview.
Clearing the cache and removing a machine fence pending downloads. Cached originals
are versioned by the print and machine authority and excluded from device backup.

Source-free prints omit the Source Media details section after one inventory check. Empty answers are reused until that owner changes; unrelated sync activity does not restart the check or blank an existing preview.

Source Media uses the Library’s justified, aspect-preserving rows and two-point seams. Badges stay inside the image. Source ratios are learned from oriented thumbnails in short batches and retained separately from decoded pixels. A synchronous warm-preview cache prevents placeholder flashes when resizing moves a tile between rows. Display-only sync edits do not cancel source scans; legitimate revalidation keeps same-authority pixels visible. Transient preview misses retry a bounded three times without requiring a scroll.

## Multi-clip video

In Short clip, choose **Add another clip** to enter ordered prompts and lengths, then **Generate video**. The app validates the draft automatically before submitting it. Move or remove rows without opening a separate editor. Library **Continue video** makes the next separate linked MP4 from the latest completed authored take on its original machine. Ordinary one-prompt generation and Reuse settings remain available.

The editor uses the selected host’s frame grid and validates delivered duration before admission. Reconnect recovers the same operation; Queue shows parent progress and supports applicable resume/cancel actions. Reference restoration is explicit after relaunch; drafts do not persist input bytes or credentials.

Failed or interrupted video jobs offer **Start again…** when their machine supports recovery. Choose the same machine to start again, or another compatible machine to **Send video**. This creates a new attempt and keeps the original. Saved attempts remain in Queue after relaunch and can be checked or stopped using their original operation identity; unanswered submissions are never silently submitted again. **Delete job…** removes the confirmed job’s recovery data while keeping saved Library videos.

Extend preserves compatible original spatial and temporal upscaling, shows the inherited recipe, and starts new footage with a fresh random seed unless one is explicitly chosen.

Historical IC-LoRA/ControlNet recipes that this native editor cannot faithfully preserve show an explicit refusal; use the saved job’s **Start again** when available, or the CLI with the original retained controls. These settings are never silently removed.
