# Mold Studio Companion (iPhone and iPad)

Mold Studio Companion is the native SwiftUI app for mold on iPhone and iPad,
and the little sibling of [Mold Studio for Mac](/guide/macos). It is
**remote-only**: the phone never runs a model. It drives the `mold serve`
machines you own, and everything it makes stays in their libraries.

It appears as **Mold Studio** on the Home Screen, needs **iOS / iPadOS 26**,
and installs beside the older [iPhone app](/guide/iphone) rather than
replacing it. Builds go out through TestFlight while it is new.

## Add a machine

Open **Machines ▸ Add a Machine…** and pick one of three ways in:

- **Scan a Pairing Code.** On your Mac, choose **Pair a Phone…** from a
  machine's card menu or the Machine menu in Mold Studio (or Settings ▸
  Mobile pairing in the desktop or web app), and point the camera at the
  code. For a machine with an API key the code is single-use, expires after
  two minutes, and carries no key; the key the machine issues goes straight
  into this device's Keychain. For a machine without a key, the code simply
  carries its address. The code is a link, so the iPhone's own Camera opens
  Mold Studio too: it names the machine and pairs only when you tap **Pair**.
  You can paste a pairing link instead. On a phone without the app, the link
  opens a [page](/pair) that says what to install; the code itself never
  reaches the web.
- **Nearby.** Machines advertising `_mold._tcp` on your network are listed.
  Allow Local Network access when iOS asks.
- **Enter an Address.** An IP address, host name, Tailscale MagicDNS name or
  HTTPS URL. The address is checked as you type and the answer is shown as a
  sentence. Add the machine's API key if it has one.

A machine that is offline keeps its place, dimmed, with the reason.

## Generate

On iPhone, Generate is one scrolling form. Choose Still picture / Short clip /
3-D object and a machine (or leave it on Auto) above the prompt. Tap **Model**
to search installed models and recipes; **Get More Models…** opens model
management directly. The controls come from the model itself: a model that reads a
starting picture shows **Start from**; one that takes references shows
numbered wells (**image 1**, **image 2**) matching how the prompt names them;
clips add a length; 3-D objects say when the prompt is ignored. **More
options** holds the sampler, seed, adapters, identity photos, ControlNet and
the mask editor, and only what the model can use. Shape, Steps, Batch and clip
Length have separate rows that remain readable at large text sizes. At larger
text sizes Kind and Machine stack. On iPhone Generate stays above the tabs as a
full-width button, and the parameter chips become one **Options** button. Clip and
3-D drafts keep their kind when you reopen the app. A model search with no
matches offers **Clear Search**.
Model search also keeps a **Close Model Search** action above the keyboard at
large text sizes.

**Generate** never turns into Stop. Pressing it again queues another render;
Stop is its own small button beside the progress sentence, e.g.
"Adding detail — about 12s left" over `denoise 18/28`.

The full-screen viewer hides the main tabs so **Share**, **Favourite**, **Info**
and **Delete** remain reachable; use Back to return to the Library. Info has a
**Done** button. Clips play when their page appears and pause when you page
away. Returning to the Library keeps your place in the grid. Shared media
keeps its original filename and file type.

## Library, Queue and Models

- **Library** keeps working offline: it shows each machine's saved prints
  when the machine isn't answering, and the prints you have opened. Pinch to
  change the tile size (five sizes, from seven columns to one). Settings ▸
  Library sets how much space it may use and can save every thumbnail ahead
  of time. It reports image storage against its limit and lists saved listing
  storage separately; **Clear
  Library Cache** explains that prints on offline machines will disappear
  from this phone until they reconnect.
- **Library** shows every machine's prints as one grid, a print found on two
  machines once. Favourites, tags, collections and Recently Deleted work the
  way they do on the Mac, across every machine that holds a copy. On iPhone,
  tap the navigation title to change shelves; it stays readable as you scroll.
  Status notices reserve space above the grid so dates and prints stay visible.
  **Media Type** offers All Media, Photos, Videos and 3D within any shelf.
  Tap **Select**, then tap tiles or sweep sideways across them. Starting on
  a selected tile deselects the range; reverse to shorten it, or hold near
  an edge to scroll. Vertical swipes still scroll in Select mode. Search
  understands `is:video`, `is:mesh`, `tag:` and `on:`.
- **Queue** lists what each machine is rendering and waiting on. A held job
  says why in words, with **Pull and Retry** when a model is missing, **Retry**
  when the machine says it would help, and **Move to…** another machine.
- **Models** uses the **Show models** menu to switch Installed and Discover.
  An unreachable machine's unread inventory is shown as unavailable, not empty.
- **Models** (Machines ▸ Models on iPhone, the sidebar on iPad) lists what a
  machine has installed and lets you discover, download, load and remove
  models. A gated model shows its licence first.

## On the Lock Screen

While a render runs, a **Live Activity** shows its preview, the progress
sentence, the time left and a Stop button, on the Lock Screen and in the
Dynamic Island. The translucent card keeps the prompt compact, gives progress its own row, and places the machine and waiting count together in the footer. Large text uses a simpler layout to keep the status and Stop control readable. mold servers cannot push to a phone, so once the app is in the
background the activity is refreshed only when iOS lets Mold Studio run; it
says "Open Mold Studio to refresh" when it may be out of date.

When a render finishes, fails or is held while the app is away, a
**notification** says so; tap it to open the print. Each kind can be switched
off in Settings.

## Widgets

- **Recent Prints** (small, medium, large): your newest prints, from every
  machine or one, all of them or favourites only.
- **Queue** (Lock Screen): "2 rendering · 1 held", a progress ring, or one
  inline line.

Widgets show what the app last saw; they never contact a machine themselves.

## Share a photo to Mold Studio

In Photos (or any app), **Share ▸ Mold Studio** and choose **Start From**,
**Reference Image** or **Add to Library**. The share sheet only saves the
photo for the app, downscaled to 2048 px; open Mold Studio and the **From
Share** card in Generate finishes the job.

## Text size and accessibility

Every screen is audited from the smallest text size to the largest
accessibility size, in light and dark, on iPhone and iPad. Rows that put a
label beside a value stack at accessibility sizes instead of truncating, and
empty screens keep their one action above the tab bar.

## Privacy

The app talks only to the machines you add. Keys live in this device's
Keychain. See the [privacy policy](/privacy) for the camera, local network,
Photos and notification permissions and what each is used for.

## Native development

`nix develop -c companion-dev` watches Swift sources and rebuilds/relaunches
the native app in Simulator. Use `companion-run` to launch once and
`companion-build` to build only. Pass `SIM=<UDID>` and `BUILD=<directory>`
through any helper. These commands are separate from Tauri's `ios-dev`.

The prompt panel scrolls at large text sizes and stays above the keyboard.
Video playback uses the media audio session, so clips with audio can be heard
even when the phone's silent switch is on. Clips without an audio track remain silent.

When machines are offline, Generate and Queue explain that their data is unavailable instead of claiming no models or jobs exist. A saved draft keeps its model while that machine reconnects; choosing a different kind or model explicitly replaces the pending selection.

Render completion notifications use the native iOS banner and Mold Studio app icon, with a short readiness message instead of the full prompt or an image attachment. Tap to open the finished print, including after the app has been closed; cancelled Library refreshes keep existing prints without an error banner.

Generate Options shows aspect-ratio icons in their actual proportions. With a source picture attached, Fit offers centered Crop to fill by default, Fit with borders, Stretch to fill, and Fit + repaint borders when masks are supported. Crop positioning offers horizontal and vertical alignment. Source pixels and painted masks are fitted together before submission. Seed defaults to Random; choose Fixed to reuse a seed. Reset returns fitting to centered Crop to fill and the seed to Random. Explicit saved settings and draft choices remain restorable.

Hold a Library tile to preview it and open its actions. Source-image selection also remains usable after a long press. The notification icon uses the same Mold logo in light and dark appearances; the system controls the notification card background.

When a Library print is saved on several machines, the source-image picker uses a currently reachable copy, including when the first listed machine is offline.

Live Activity cards use ActivityKit’s default material so the background and text follow the Lock Screen appearance together, including the stale “Open Mold Studio to refresh” state.

## Find curated models and reuse their inputs

In **Models ▸ Discover**, **Mold Models** lists the curated checkpoints your
machine can fetch, including uninstalled variants. Search their readable name,
family, or Hugging Face repository. Hugging Face and Civitai marks identify the
source. **Get** installs one exact variant using the machine's configured
credentials; licence acceptance still applies. The same readable model names
appear across Mold's apps and CLI, with the stable model ID available separately.

**Use These Settings** restores a print's retained source image and repaint
mask into the composer before fitting. Other retained conditioning files are
named beside the composer and remain available across prompt edits and repeated
renders. **Remove retained sources** clears those attachments; replacing an
image keeps your choice. Unavailable retained files are explained inline.

The Queue shows the prompt and actual **Source** image separately from the live
**Rendering** preview, including when the source belongs to a durable job that
survived a machine restart. On iPad, jobs stay within a readable centered column.

### Server model memory

In Machines → Models → Installed, use Unload beside a resident model or Unload All Models for the selected server. Downloads remain installed. Offline and pending-operation controls are disabled, and server refusals remain visible.

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

Retained queue sources are also visible in the native Mac, web, desktop and Tauri
mobile queues, regardless of the submitting device. Library outputs retain their
source media after queue cleanup; trash preserves those references, and deleting
one output permanently does not remove sources still used by another output.

Before importing a source-bearing copy, clients check destination readiness. Windows local destinations currently cannot receive retained inputs, so those copies are refused before creating a local library output. Source-free copies remain supported, and Windows clients can recall retained sources from a supported remote machine.

### References and boundary frames

Both native apps expose the server's reference contracts. MiniMax H3 **Ref2VA** takes an ordered mixture of images, H.264 MP4 clips and mono/stereo PCM WAV audio; image references can come from Photos/Camera/Library/Share on iOS or Finder/Library/Paste on macOS, and movie/audio files use Files/Finder. Replace, remove and reorder attachments before generating. Use `image 1`, `video 1` and `audio 1` in the prompt (numbered within each media kind). Audio references need at least one visual reference. The limits are nine images, three videos, three audio files and twelve files total; each clip is 2–15 seconds, with at most 15 seconds of video and 15 seconds of audio including video soundtracks. Authenticated hosts use request-bound upload sessions; keyless hosts accept at most 32 MiB of inline reference media per render. Video clips with sound require authenticated uploads so the server can supply exact decoded soundtrack counts; on keyless hosts use a silent MP4 plus separate PCM WAV audio. Unsupported or oversized files report an error instead of silently disappearing.

Hunyuan3D multiview models offer named Front/Left/Back/Right wells from their recipe. Wan offers a first/last pair rather than arbitrary middle frames; MiniMax **FL2VA** offers separate optional first/last frames. Changing clip length updates the closing frame. Existing Qwen Edit, Qwen Image 2.1 and Flux.2 reference strips honor their source-image relation, count and pixel budgets; the last Qwen Image 2.1 reference updates the default canvas until you choose a size. SD1.5/SDXL reference weight comes from the model's own control. Model changes park unsupported attachments so they can return. Reuse restores retained typed references with fresh media authority while their original set and order stay unchanged. Changing retained slots requires reattaching the remaining originals; archived bytes never overwrite new attachments. Imported mesh texture/roundtrip workflows remain API/CLI-only.
