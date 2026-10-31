# Mold Studio for Mac

Mold Studio is mold's app built natively for macOS. It is written in Swift, carries mold's own Rust engine running on
Metal, and talks to every other `mold serve` machine you use over the same
HTTP API.

<div class="platform-downloads">
  <a class="platform-download platform-download--primary" href="https://github.com/utensils/mold/releases/latest/download/Mold-Studio-macos-arm64.dmg">
    <img src="/icons/apple.svg" alt="" />
    <span><strong>Mold Studio for Mac · Latest Release</strong><small>Signed and notarized · Apple Silicon · macOS 26+</small></span>
  </a>
  <a class="platform-download" href="https://github.com/utensils/mold/releases/download/latest/Mold-Studio-macos-arm64.dmg">
    <img src="/icons/apple.svg" alt="" />
    <span><strong>Mold Studio for Mac · Latest Nightly</strong><small>Latest main build · Apple Silicon · macOS 26+</small></span>
  </a>
</div>

Open the DMG and drag **Mold Studio** to Applications. It is signed,
notarized, and stapled, so there is no quarantine step. Mold Studio updates
itself through Sparkle: **Mold ▸ Check for Updates…** sits under About Mold.
**Latest Release** always downloads the newest stable build; **Latest Nightly**
always downloads the newest build from `main`. Version-pinned DMGs and
`SHA256SUMS` remain on the [releases page](https://github.com/utensils/mold/releases).

::: tip Two desktop apps on the Mac
Mold Studio installs as **Mold Studio.app**. [Mold Desktop](/guide/desktop),
the same app that runs on Windows and Linux, installs as **Mold.app** and runs
on macOS 12 or newer. Both can sit in /Applications at once and connect to the
same machines. [Which one?](/download#mac)
:::

In the native Mac app, **Use source** is available in the print inspector and source viewer, including right-click menus. Choose a supported input role, or **Choose model…** to select an installed compatible model, recipe and machine without leaving the source. Successful reuse opens Generate; other draft settings are preserved where the selected recipe supports them.

## Requirements

- Apple Silicon (M1 or newer). There is no Intel build.
- macOS 26 or newer.

## What it does

- **Generate** — every control comes from the chosen model's own generation
  profile: stills and clips, source and reference images, batches, negative
  prompts, adapters, identity photos, masks, ControlNet, and prompt expansion.
  Live step progress and denoise preview; images, clips, and meshes are shown
  in place. The controls stay anchored above the bottom edge and scroll
  inside their capsule when the window is short.
- **Library** — every machine's prints in one day-sectioned timeline, with a
  print held on several machines shown once. Favourites, collections, tags,
  search tokens (`is:video`, `tag:name`, `on:machine`), Quick Look, undo,
  interactive 3-D mesh viewing and export, and **Use These Settings** to
  restore a print's full recipe. Select prints and choose **Move to Collection →
  New Collection…** to create a collection from the selection. Hidden collection
  members stay out of general browsing even when a local copy leads; the
  collection itself and Trash remain accessible. Context menus keep
  their rows stable while background transfer status changes.
- **Queue** — every machine's work live from its event stream: reorder,
  pause, resume, cancel, and move held jobs between machines. Source thumbnails
  reserve their image and caption space while loading; jobs without retained
  source media remain compact. Batch groups expand into individually selectable
  jobs, with disclosure separate from the batch actions.
- **Models** — installed models grouped by family, catalog discovery, and
  downloads with gated-licence acceptance in place.
- **Machines** — the fleet at a glance, per-GPU memory and load, nearby
  machines to add, and phone pairing for keyed hosts.
- **This Mac** — mold's engine starts in-process when the app opens (Settings
  ▸ This Mac turns that off) and joins the machine list like any other host.

To choose the port for This Mac, select it in **Settings ▸ Performance ▸ Port**
and relaunch Mold Studio. The address in Machines shows the port actually in
use. If another process already uses the chosen port, the engine reports the
conflict; choose a free port with `mold config set server_port <port>` before
starting it again.

Not in Mold Studio (by design or not yet): authoring scripted sequences (CLI
and API only), RunPod provisioning, and the 3-D authoring workflows.

## Working in Generate

Use **View** (native apps) or **View original** (web, desktop and mobile) beside an attached input to inspect it without replacing or removing it. References, source/target pictures, boundary frames, masks, control and identity pictures, and attached video/audio have separate viewing actions. Image previews fit the whole picture. Use **Actual size** on desktop and web, or pinch to zoom in the native iOS app, for close inspection. **Done** or Escape returns to the attachment or picker. Picker previews do not select or attach the picture. Queue viewing actions show the full submitted media, which may already have been fitted to the generation canvas; older machines without the original-input endpoint report it unavailable.

**Choose from Library** opens a spacious combined picture browser with machine
filtering, search, and previews. Library **Add as Reference** supports H3 while
keeping your current model and prompt. Oversized pictures are resized for the
selected input; keyframes share the same library, file, paste, and drop options.

Use the compact prompt field for quick edits, or choose **Edit prompt** for a
spacious text editor. Changes stay in your draft as you type; **Done** returns to
Generate. **Recent prompts** searches previous text without changing your model,
settings or source images. **Clear** offers **Undo clear**, and the existing prompt
rewrite tools remain available. In the editor, Return adds a line and arrow keys
move through your text.

Drag the settings inspector divider to change its width; click a section heading
to show or hide its controls. In **Settings → Library**, **Show date separators**
controls the day headings and daily row breaks. Turn it off for one continuous
grid in the current sort order. The preference is saved on this Mac.

**Random each time** chooses a new seed for every submission. **Keep this seed**
reuses the displayed starting number; **New seed** chooses another without
generating. A batch uses successive seeds to make variations.

Generate shows preparation and sending progress immediately, then confirms when
the machine accepts the job or explains why it could not. An accepted job may
start generating or wait in the queue. When reusing a print, **Stop restoring
original inputs** keeps your prompt, settings and attached files while removing
references that could not be restored. It does not delete Library files; attach
any required replacement inputs before generating.

Job Details groups progress, preview and settings, with copyable identifiers under
**Technical details**. Hover over controls for a plain-English explanation.

## Keyboard

⌘1–⌘5 switch destinations, ⌘↩ generates, ⌘R refreshes, ⌥⌘I toggles the
inspector, Space is Quick Look, ⌥⌘F favourites, ⌘⌫ trashes a print or cancels
the selected queue row, and ⌘, opens Settings.

## For contributors

The source lives in `apps/macos/`; its
[README](https://github.com/utensils/mold/blob/main/apps/macos/README.md)
covers building, the embedded engine, releasing, and the Sparkle feeds.
For distribution builds, `make engine` passes the app's resolved marketing
version as `MOLD_BUILD_VERSION` to Rust. This keeps the version reported by
This Mac and `/api/status` aligned with About Mold Studio on Nightly; ordinary
Rust builds use the workspace package version.

In Library, **Filter** offers Favourites, Photos, Videos, 3D, collection
membership, Made from a Source, tags and orientation, plus a Model Family
submenu with family colours. Combine one main filter with a family; Clear
Filters resets both. The button shows the current choices. Machine and search
filters remain available, and long machine names retain their full tooltip. Multi-select with
Command-click or Shift-click without recentering tiles already in view.
Arrow-key navigation still reveals offscreen selections.

### References and boundary frames

Both native apps expose the server's reference contracts. MiniMax H3 **Ref2VA** takes an ordered mixture of images, H.264 MP4 clips and mono/stereo PCM WAV audio; image references can come from Photos/Camera/Library/Share on iOS or Finder/Library/Paste on macOS, and movie/audio files use Files/Finder. Replace, remove and reorder attachments before generating. Use `image 1`, `video 1` and `audio 1` in the prompt (numbered within each media kind). Audio references need at least one visual reference. The limits are nine images, three videos, three audio files and twelve files total; each clip is 2–15 seconds, with at most 15 seconds of video and 15 seconds of audio including video soundtracks. Authenticated hosts use request-bound upload sessions; keyless hosts accept at most 32 MiB of inline reference media per render. Video clips with sound require authenticated uploads so the server can supply exact decoded soundtrack counts; on keyless hosts use a silent MP4 plus separate PCM WAV audio. Unsupported or oversized files report an error instead of silently disappearing. On authenticated hosts advertising fresh-video normalization, Files/Finder also accepts the host's advertised MOV/HEVC, WebM/VP9, MKV/AV1, ProRes and VFR MP4 inputs. Preparation verifies canonical facts before pricing/admission; original files stay available after a refusal. See [fresh reference video imports](./generating#fresh-reference-video-imports) for limits and host codec requirements.

Hunyuan3D multiview models offer named Front/Left/Back/Right wells from their recipe. Wan offers a first/last pair rather than arbitrary middle frames; MiniMax **FL2VA** offers separate optional first/last frames. Changing clip length updates the closing frame. Existing Qwen Edit, Qwen Image 2.1 and Flux.2 reference strips honor their source-image relation, count and pixel budgets; the last Qwen Image 2.1 reference updates the default canvas until you choose a size. SD1.5/SDXL reference weight comes from the model's own control. Model changes park unsupported attachments so they can return. Reuse restores retained typed references with fresh media authority while their original set and order stay unchanged. Changing retained slots requires reattaching the remaining originals; archived bytes never overwrite new attachments. Imported mesh texture/roundtrip workflows remain API/CLI-only.

### Recalling retained recipes

Reused ordered image references have authenticated previews. Prompt, shape and seed changes preserve the unchanged reference set, and each render gets fresh authority. If retained slots are reordered, removed or replaced, attach the remaining originals before generating. Preview failures offer Retry and do not discard the references. Shape menus show the actual aspect outline.

The app stores a byte-free origin locator for retained recipes. After relaunch it verifies the original machine instance, archive and output before enabling Generate. If verification fails, reconnect and reselect the source print, replace the attachments, or explicitly discard retained conditioning. Upload sessions and retained-source permissions are never saved as reusable authority.

Local authoring inputs are saved privately with your prompt and scalar settings. Attached images, masks, references, audio/video, first/last frames and parked inputs survive relaunch. Unresolved retained references still require verification on their original machine. A save failure asks you to keep Mold open; missing or corrupt saved inputs block Generate until you reattach the files you need and choose **Use current inputs**. Previously lost local attachments cannot be recovered by this change, though **Use These Settings** can recover originals retained with a Library print.

Native shipping builds include the reviewed Metal H3 engine; remote generation uses the selected host's capabilities.

Duration and extended-video copies preserve their complete captured recipe,
retained inputs, and separate provenance/audio archive files. Sync resumes
interrupted archive transfers and repairs missing assets on existing copies.
A movie download alone is an output export, not a full Library backup. Missing
legacy inputs are disclosed; a declared but incomplete video archive is a copy
failure until repaired.

**Save All to This Mac** copies retained inputs with every supported output type, independently of model family. The destination owns the source, identity, mask/control, edit/typed references, audio/video, keyframes, continuation and chain stage media, so reuse does not need the original machine. Incomplete source transfers are reported as failures. Use settings restores every recorded sequence stage and its retained inputs into a fresh editable draft. Missing historical settings or unavailable inputs are disclosed.

## Queue, discovery and video preferences

Click a job in Queue to see its source, settings and live progress or preview where the model supports it. The seed reads **Random** until a random job has an actual seed; explicitly locked zero remains valid.

Held jobs offer **Failure Details** beside their controls. Job Details also shows a selectable diagnostic and **Copy Details**, including the machine and job ID. The concise error explains known failures in plain English. Cancel rechecks the owning machine and current state. Current Mold servers enforce held-only cancellation, preventing stale Held actions from stopping a render. Update older machines for this safeguard.

Discover opens with manifest models from the selected machine. Choose **Browse Community Catalog**, search, or filter by family for community results. **Load More Models** shows the current result count in a prominent footer.

In **Settings → General → Video Playback**, choose autoplay and repeat. Videos use inline controls without the full-video hover overlay. Video and 3-D GIF export provide Loop/Bounce, Forever/Once and pause settings when applicable.

New oversized input pictures are fitted proportionally, with orientation and transparency preserved. Paired remote connections prefer verified LAN routes over Tailscale and relay and recheck when the network changes.

Queue rows show the owning machine’s sealed conditioning images across models. Job Details shows every ordered reference separately with its role, including identity photos, named views, masks, control images and boundary frames. Audio/video references are listed by kind; unavailable previews are disclosed. Inputs remain separate from live denoise previews and are fetched through authenticated routes, including work submitted from another device. Older servers retain their singular source preview.

Library New badges and unread counts are saved on each client independently. Media discovered after the initial machine baseline stays new during a Library visit. Individual viewer display clears one print immediately; leaving Library clears the eligible projections captured during that visit. Unvisited hidden or filtered-out media, Trash, and arrivals after departure remain unread. Merged copies count once; hidden collections and Trash do not count. Existing read history is preserved on upgrade. Native icon badges follow the same history, with the existing permission or preference controls.

### Queue model downloads

A held job with missing downloadable model components offers **Download and Retry**
on its queue card and in Job Details. This includes missing adapters for installed
recipes when the machine can identify a repairable download. The original error,
**Failure Details** and eligible **Move to…** action remain available during recovery.
The job shows Starting, download-queue status, live bytes/progress, license review,
reconnection and failures in place. Closing details does not stop recovery.
The download runs on the job’s owning machine; retry occurs only after every
returned download ticket succeeds and the original held job and server identity
are revalidated. Failed or cancelled downloads leave the job held. Global queue
pause stays in effect. Cancelling a job does not cancel a model download that
other jobs may need.

## Control copies on each machine

Library machine filters also scope trash, restore and permanent deletion to the selected machines, preserving copies elsewhere. Choose All Machines to act on every copy. Native Trash remains browsable for hidden collection members and offers Put Back, Delete Immediately and Empty Trash for the selected machines. Opening Trash clears the previous shelf’s search and media filters while keeping the machine selection. Trashed photos and videos can be inspected before restoring or deleting them.

Deleting selected Trash items requires an updated host with the trash-only deletion endpoint. If that host is older, Mold asks you to update it and preserves the files. Put Back and the dedicated Empty Trash operation remain available.

## Shared Library and session sync

The machine picker scopes collection counts, membership and media actions to the
selected machine. An absent collection is different from an unavailable machine
inventory. **All Machines** keeps the merged view available. Hidden collection
visibility is shared across same-slug replicas, and pending hide/show changes
retry when their original machines reconnect.

**Sync All to This Mac** copies once, when you press it. Turn on **Keep This Mac in Sync**
(the button's menu, or Settings ▸ Library; off by default) to copy new and changed prints as
soon as each machine announces them, plus one catch-up when a machine connects. There is
no timer. Completion appears in the Library status. During transfers, a single progress
banner offers **Stop Sync**; quitting Mold stops a running sync. **Done**, Return, and Escape close Sync Details;
reports without issues or source notices omit acknowledgment controls. **Details** offers
**Never show me this again for these sources.** for unchanged source issues, including
legacy prints whose original inputs are unavailable. Acknowledged notices disappear
from the banner but remain in Details. Acknowledgment keeps retries running; new or
changed errors and connection/authentication failures still appear. Removed local
copies stay removed during Sync; use explicit Save to copy one again. An older
server that cannot delete selected trash items must be updated; deletion reports
that refusal without falling back to deleting live media.

Reopening Sync Details shows the saved acknowledgment choice. New media issues leave the checkbox unchecked; Reset Acknowledgments clears the saved choice immediately.

Queue row buttons, contextual menus, the Queue menu and Job Details share the same action eligibility. Queued jobs offer Pause and paused jobs offer Resume when supported by the machine. Held jobs offer Retry or Download and Retry when recovery is supported. Move to is available for eligible queued, paused, and held jobs on current servers; it stops being available when rendering starts. Older servers retain Held-only moves. During a retry, a failed source read leaves the move unresolved. A verified destination plus a confirmed missing original establishes completion; an original that has started rendering is reported as no longer eligible. Running cancellation requires machine support. Offline, stopping, settled and currently updating jobs retain inspection without mutation controls. Recovery controls stack in narrow windows so their labels stay readable.

Also Running offers Cancel for reported long-clip chains when the machine permits it, and Resume for paused chains. Clip-upscale controls also check that the machine is connected and the displayed job is still current before acting. Durable authored sequences are managed through `mold jobs` or the HTTP API.

Library Settings remembers **Keep This Mac in Sync** between launches. Sync reports checking existing copies separately from copying outputs. Updated hosts provide batched retained-input evidence so unchanged verified copies can skip individual transfer probes; older hosts continue checking retained inputs so late repairs are not missed.

Sync reads every machine at once. Pressing **Sync All** checks every print
against each machine's own checkpoint; Keep This Mac in Sync only looks at what
changed since its last run, plus anything a previous run left unfinished, and
checks everything once a day. Either way, a sync with nothing new does not
check each print again.

Queue rows keep their height while previews and render progress load, preserving the scroll position. Interrupted Library syncs resume from acknowledged output copies and repair remaining retained-input or organization work.

Queue rows keep a neutral image placeholder when a job has no source image, matching
the native iOS queue. Successfully restored local input files remain usable after
prompt edits and relaunch without an obsolete archive warning. Unresolved remote
references and missing snapshots still require recovery.

## Video length, scenes and recovery

On updated machines, choose one prompt and the **Length** slider. Its initial
value follows the selected recipe; native duration controls use tenths of a
second up to one generation. **Automatic from prompt** lets supported models
choose the length instead. Turn on **Longer video** for ordered scene prompts
with a Length slider per scene and **Add scene**. Each scene is capped at the
host's always-one-clip limit; plan warnings explain repeated actions and scenes
shorter than the minimum generation. See [Video generation](./video#longer-videos)
for model-specific joins, CLI examples and extending a saved movie.

Library **Extend video** preserves the original and publishes the original plus
added footage as a new movie. **Use settings** creates a fresh editable draft:
Duration prints restore their original duration/scenes and exact seed; extended
prints restore the added-footage settings and disclose that the original movie
is not attached. Authored sequences restore every recorded stage and input.

Videos appear as ordinary rows in their machine's Queue section, with Waiting,
Rendering clip and Joining clips progress. Stop retains accepted Duration work
for explicit Resume. Start again and Send use a new attempt with the destination's
current recipe when offered. Completed and stopped videos leave the native Queue
automatically; failed or interrupted work stays available for recovery. Saved
movies remain in Library. An uncertain submission must be checked before making
another attempt; reconnecting never resumes paused work automatically.
