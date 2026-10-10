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

Use **Edit prompt** to open a spacious editor above the keyboard. Changes stay in
your draft when you tap **Done** or dismiss it. **Recent prompts** reuses just the
text; **Clear** offers **Undo clear**. Your model, sources and other settings stay
in place.

In **Settings → Library**, turn **Show date separators** off for continuous rows
without daily headings or breaks. The saved preference applies to Library views,
including search and collections, while preserving their sort order.

**Generate** never turns into Stop. Pressing it again queues another render;
Stop is its own small button beside the progress sentence, e.g.
"Adding detail — about 12s left" over `denoise 18/28`.

On iPhone, rotate a playing clip horizontally to use the full display. Video
keeps its proportions; narrow black bars may remain for a different aspect
ratio. Rotate upright to restore gallery actions, or tap Close to return to
the Library while horizontal. Native playback controls remain available.

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
  The offline notice shows a compact machine count; tap it for scrollable
  details with every full machine name and a pinned Done action, including at large text sizes. Navigation tabs return when details close.
  Decorative machine badges fit their tiles; full names remain in accessibility
  labels and Info.
  **Media Type** offers All Media, Photos, Videos and 3D within any shelf.
  Tap **Select**, then tap tiles or sweep sideways across them. Starting on
  a selected tile deselects the range; reverse to shorten it, or hold near
  an edge to scroll. Vertical swipes still scroll in Select mode. Search
  understands `is:video`, `is:mesh`, `tag:` and `on:`.
- **Queue** lists what each machine is rendering and waiting on. A held job
  says why in words, with **Download and Retry** when a model is missing, **Retry**
  when the machine says it would help, and **Move to…** another machine.
- **Models** uses the **Show models** menu to switch Installed and Discover.
  An unreachable machine's unread inventory is shown as unavailable, not empty.
- **Models** (Machines ▸ Models on iPhone, the sidebar on iPad) lists what a
  machine has installed and lets you discover, download, load and remove
  models. A gated model shows its licence first.

## On the Lock Screen

Live progress is available in Queue. Persistent generic Live Activities are disabled.

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

Completion notifications use the system notification appearance.

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

The Queue shows the prompt and actual conditioning images separately from the live
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

Swipe left on an actionable queue row for **Cancel**, even before its source images and batch settings load. Swipe right on a recoverable held row for **Retry**, or **Download and Retry** when its model is missing. Queued rows also offer **Pause** by swiping right, and paused rows offer **Resume**. When no recovery or pause action applies, swipe right for **Details**. Revealing the swipe controls does not change the job; tap an action to confirm it. Held cards keep **Move to…** available for transferring to another machine, and Job Details provides the controls without swiping. Running jobs offer cancellation only when their machine supports it. **Failure Details** opens the job's saved diagnostic with its machine and job ID, and **Copy Details** makes it easy to share for troubleshooting. These details remain available when a failed job cannot be retried. Cancelling a held job rechecks its machine and state; if it has started running, refresh the queue before choosing another action. Current Mold servers enforce held-only cancellation, preventing stale Held actions from stopping a render. Update older machines for this safeguard.

Prompt History is available directly in Generate. Choose a machine and search
its saved prompts; selecting one changes only the prompt, preserving the model,
settings, and attached media. Loading, offline, unavailable history, and failed
requests have distinct messages. Clear asks for confirmation and removes that
machine's entire history, including prompts hidden by search.

Retained queue sources are also visible in the native Mac, web, desktop and Tauri
mobile queues, regardless of the submitting device. Library outputs retain their
source media after queue cleanup; trash preserves those references, and deleting
one output permanently does not remove sources still used by another output.

Native Mac mirrors accept missing archive-only job IDs and generation durations, a short version matching the same version with a build suffix, and an alpha-channel fact derived from identical output bytes when absent on the source. The same compatibility check compares the gallery listing with its stable source archive, so completion bookkeeping does not falsely report a changed print. When an older archive omits scheduler or transparency settings, the source output is downloaded and its digest verified before its embedded recipe can corroborate those fields. Output bytes and all generation settings must still match; conflicting recorded provenance is refused. Retrying Sync All repairs retained inputs on compatible existing copies without duplicating their outputs.

Native Mac Save Locally and Sync All count a verified finished print as saved when its source explicitly reports legacy inputs as unavailable. The sync summary reports unavailable original inputs separately; it does not claim they were retained. Existing copies still require matching output bytes and recipes. Corrupt or unauthenticated input archives, changed outputs, and unsupported source transfer contracts remain failures.

Before importing a source-bearing copy, clients check destination readiness. Windows local destinations currently cannot receive retained inputs, so those copies are refused before creating a local library output. Source-free copies remain supported, and Windows clients can recall retained sources from a supported remote machine.

### References and boundary frames

Both native apps expose the server's reference contracts. MiniMax H3 **Ref2VA** takes an ordered mixture of images, H.264 MP4 clips and mono/stereo PCM WAV audio; image references can come from Photos/Camera/Library/Share on iOS or Finder/Library/Paste on macOS, and movie/audio files use Files/Finder. Replace, remove and reorder attachments before generating. Use `image 1`, `video 1` and `audio 1` in the prompt (numbered within each media kind). Audio references need at least one visual reference. The limits are nine images, three videos, three audio files and twelve files total; each clip is 2–15 seconds, with at most 15 seconds of video and 15 seconds of audio including video soundtracks. Authenticated hosts use request-bound upload sessions; keyless hosts accept at most 32 MiB of inline reference media per render. Video clips with sound require authenticated uploads so the server can supply exact decoded soundtrack counts; on keyless hosts use a silent MP4 plus separate PCM WAV audio. Unsupported or oversized files report an error instead of silently disappearing.

Hunyuan3D multiview models offer named Front/Left/Back/Right wells from their recipe. Wan offers a first/last pair rather than arbitrary middle frames; MiniMax **FL2VA** offers separate optional first/last frames. Changing clip length updates the closing frame. Existing Qwen Edit, Qwen Image 2.1 and Flux.2 reference strips honor their source-image relation and count limits; processing pixel budgets automatically resize references in the engine and never require a manual resize. The last Qwen Image 2.1 reference updates the default canvas until you choose a size. SD1.5/SDXL reference weight comes from the model's own control. Model changes park unsupported attachments so they can return. Reuse restores retained typed references with fresh media authority while their original set and order stay unchanged. Changing retained slots requires reattaching the remaining originals; archived bytes never overwrite new attachments. Imported mesh texture/roundtrip workflows remain API/CLI-only.

## Access and saving to Photos

Save to Photos requests add-only access. If access is denied, tap **Open
Settings**, allow adding photos, then return and save again. Camera and nearby
network failures offer the same recovery pattern; Notifications and Live
Activities have recovery actions in app Settings. Restricted access explains
Screen Time or device management. Photo imports use the system picker without
requesting access to your whole library. Auto-save requests Photos access when
you enable it, and saving videos uses PhotoKit's background callback safely.

## Focused generation and media playback

Generate keeps the prompt, attachments and Generate button stable while jobs run. Tap the queue count for job details and supported live previews; completed media belongs in Library. Completion/failure notifications remain available without a persistent generic Live Activity.

Picture menus in Library and its viewer offer **Use as Source**, retaining your prompt and model. The source picker searches all machines together and can filter to one machine. New oversized still inputs fit automatically with orientation and transparency preserved; original library files remain unchanged.

**Settings → Video Playback** offers autoplay and repeat. Only the selected video plays, and leaving the viewer or backgrounding pauses it. 3-D views load only when selected, use file-backed downloads and offer Retry on failure. Paired connections prefer a verified local-network route and reconsider routes on network changes.

On macOS, image and video viewers default to **Fit**, keeping the entire frame visible as the window changes size. **Actual Size** shows one media pixel per screen pixel and lets you scroll larger media. Video controls remain below the picture without dimming it on hover.

Queue rows show the owning machine’s sealed conditioning images across models. Job Details shows every ordered reference separately with its role, including identity photos, named views, masks, control images and boundary frames. Audio/video references are listed by kind; unavailable previews are disclosed. Inputs remain separate from live denoise previews and are fetched through authenticated routes, including work submitted from another device. Older servers retain their singular source preview.

Library New badges and unread counts are saved on each client independently. Newly discovered media stays new until that client displays it in the viewer; opening Library, changing filters, refreshing, and preparing neighboring pages do not clear it. Counts and badges deduplicate merged copies and exclude hidden collections and Trash. Existing read status is preserved during upgrade, and a newly connected machine establishes a historical baseline; subsequent arrivals require individual viewing. Native iOS also badges its Images navigation. Native app-icon counts follow the same local history, subject to the existing badge preference or permission. iOS refreshes while active and opportunistically in the background; the server has no push.

On iOS, Use These Settings replaces all active and parked attachments with the selected print’s media. Retained archives restore from the owning machine or another available copy; unavailable conditioning blocks Generate until restored, reattached or explicitly removed. Retry retained media after reconnecting. Late replies cannot replace a newer reuse or revive a source added and removed during restoration.

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

### Move queued work to another machine

When another connected machine can generate, **Move to…** is available on queued, paused, and held jobs until the source machine begins rendering. The prompt, seed, settings, and retained reference media move together. The destination must accept the job before the original is removed. Older servers support Held-only moves. Jobs that depend on machine-local LoRAs or workflows are refused without moving them.

If a connection is interrupted during a move, retry the same destination so Mold can check whether it already accepted the job. The source remains reserved while that result is unknown, preventing duplicate rendering. A confirmed destination rejection releases the original only after the destination records that this transfer cannot be admitted later.

### Editing long prompts

The compact prompt stays bounded while you type. **Edit prompt** opens a spacious live editor; **Done**, Escape and backdrop dismissal retain edits and return focus to the opener. Enter inserts a newline, and arrow keys move through text. **Recent prompts** searches full prompt text and replaces only the prompt, preserving model and other settings. **Clear** offers immediate **Undo clear**; a later edit or history choice ends that recovery. Expand and Remix use the existing rewrite flow and return to the editor. The editor never submits a generation shortcut. Phone presentation fills the visible viewport, with controls retained when the keyboard reduces the available space.
