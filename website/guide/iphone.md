# iPhone App

Mold for iPhone is a first-class remote studio for a Mold server. The phone
does not run AI models itself: generation, model storage, downloads, queueing,
and gallery files stay on one or more remote Mold hosts while the app provides
an iPhone-native control surface.

Mold requires iOS 17 or later. The current build is distributed to invited
internal and external TestFlight groups; there is not yet a public App Store
listing.

## Connect a host

Start a reachable server on the GPU machine. Use an API key whenever the server
is reachable beyond localhost:

```bash
MOLD_API_KEY='choose-a-long-secret' mold serve --bind 0.0.0.0 --port 7680
```

Open **Machines** on the iPhone and use one of these paths:

- Recommended: open **Settings → Mobile pairing** in Mold desktop or web,
  confirm the LAN, MagicDNS, or HTTPS address the phone can reach, tap **Create
  pairing code**, then tap **Scan pairing code** on iPhone. The code is single-use,
  expires after two minutes, and contains no API key; the redeemed key goes
  directly into the iOS Keychain.
- Tap **Discover nearby** to browse `_mold._tcp` services on the current LAN.
  Allow Local Network access when iOS asks.
- Enter an IP address such as `192.168.1.10`. Mold adds `http://` and port
  `7680` when they are omitted.
- Enter a DNS hostname, HTTPS URL, or Tailscale MagicDNS name.

For manual entry, paste the same API key, then tap **Test and save**. Saved host
metadata remains in the app's local storage, while the API key is kept
separately in the iOS Keychain.

### Tailscale

Tailscale is the simplest way to reach a private GPU host away from its LAN:

1. Join the iPhone and GPU host to the same tailnet.
2. Keep `mold serve` listening on an address the Tailscale interface can reach.
3. Add the host by its MagicDNS name, for example `plato` or
   `plato.example-tailnet.ts.net`. Mold supplies port `7680` for a bare name.
4. Include `https://` and an explicit port only when your own reverse proxy
   provides TLS.

Bonjour discovery is LAN-local; Tailscale hosts are normally added by name.
Mold uses the installed Tailscale network and does not manage tailnet login,
ACLs, DNS, or certificates.

## Create

Choose where the print develops and one of the installed generation models. With
a single connected machine the Host control works as it always has. Once two or
more connected machines are reachable it also offers:

- **Auto**: the least busy machine that already has the selected model.
- **Most capable**: the strongest GPU that has it: CUDA before Metal, then
  VRAM, then queue depth.

An optional **Title** above the prompt names the print: it is embedded in the
saved metadata, folded into the output filename as a slug, and shown across
every Library. Batch siblings and prepared variations inherit it, and **Use as
prompt** restores it from a print.

Directly beneath it, **File under** files the print as you make it, as two
rows:

- **Tags**: a dashed chip offers the title's own slug (tap it away and it
  stays away), and the sheet takes typed names and suggests the tags your
  machines already use, with counts.
- **Collection**: the collection whose name matches the title is
  pre-selected, never created; the sheet offers None, every collection merged
  across your machines, and an inline **New collection…** that only records
  the name until the print develops.

A line under the rows previews the filename the print will land as. Both rows
disappear when the machine the print is routed to cannot organize its library,
and **Settings ▸ Library ▸ Tag new prints with their title** turns the title
chip off without touching prints you already made.

Under either policy the model list is the union of every reachable machine's
installed models, and a model that is not on all of them is tagged with the
machine that has it. Mold ranks the reachable machines from their cached
model, queue, and GPU telemetry and freezes the winner before queueing, so
recovery and prepared expansion both stay on the exact machine that ran the
work. If no machine can run the print, nothing is queued and the message
names each machine. Your choice is remembered; it falls back to the browsed
machine while only one is reachable.

The form adapts to
the selected model family and uses the same request contract and model defaults
as desktop. The primary controls stay on the main screen; deeper options open in
a full-screen **Advanced** sheet, and prompt **style** presets compose at submit
without rewriting your prompt text. A **↺ Reset** beside the Advanced trigger
restores every generation setting to the selected model's defaults, keeping
your prompt, model choice, and any prepared batch.

The mobile composer includes:

- a directly editable Batch count, Batch 1 quick prompt expansion with undo,
  compact/paged Batch N prepared-variation review, and remote prompt history;
- local templates that can be saved, searched, sorted, renamed, loaded, and
  deleted;
- batch generation as independent queued jobs, each with its own progress and
  Cancel action;
- source images and fit policies, Qwen edit target/reference images, masks,
  ControlNet, LoRA stacks and trigger words;
- scheduler, CFG++, steps, guidance, output format, and post-generation
  upscaling where the selected family supports them;
- validated video frames/FPS, audio, camera motion, source media, keyframes,
  retake, **Continue a video** (extending an existing clip on models that
  advertise it, including capable Wan checkpoints; resolution and fps stay
  locked to the source clip), LTX-2 pipeline/spatial/temporal controls, and
  optional STG, CFG-rescale, modality-scale, and guidance-skip overrides;
- a host-aware memory estimate before submission.

For video-only LTX-2 community checkpoints, **Generate audio** is disabled with
an explanation when the connected server reports that the installed files lack
an audio VAE or vocoder. Source-image video generation remains available.
Empty guidance-override fields preserve the pipeline defaults. Invalid block
lists or numeric ranges stay in the Advanced sheet with inline feedback and
cannot queue a request; **Reset advanced**, templates, and Library reuse share
the same saved override state as desktop.

### Identity photos (PuLID)

An identity photo conditions the print on a person's face: the render keeps
that likeness while the prompt decides everything else. The photo itself is
never composited into the output, and (unlike a source image) it is never
cropped or fitted to the canvas. It is a reference, not a composition input.

An **Identity** well sits in the Create form beside the source wells whenever
the selected model and the machine you are developing on both support it. When
they do not, the control is not there at all rather than present and disabled.
Tap it to pick a PNG or JPEG (at most 16 MiB, 8192 px per side, 32 MP) with the
usual iOS photo/camera picker. Identity conditioning is offered only for the
identity-qualified checkpoints on a server built with the feature, and it
cannot be combined with a LoRA or an img2img source image; see
[Identity Photos (PuLID)](/guide/generating#identity-photos-pulid) for the full
rule and the one-time InsightFace licence acceptance.

Switching to a model that cannot use an identity photo does not throw yours
away and does not stop you developing: the photo is parked, the request goes
out without it, and the well comes back with the photo still in it when you
select a qualified model again.

Two knobs live in the Advanced sheet, count toward its badge, and clear with
**Reset**. Both stay absent from the request until you touch them, so the
server's own defaults keep applying:

- **Identity strength**: how strongly the face is held, `0.0`–`3.0`
  (default `1.0`). Higher preserves the likeness; lower lets the prompt reshape
  it.
- **Identity start step**: the first denoise step the face is applied at
  (default `0`, always fewer than the print's step count). Delaying it lets the
  composition settle before the likeness is pinned.

If the combination cannot be submitted (a photo alongside a LoRA or a source
image, a knob set with no photo, an oversized or unsupported file) the reason
reads inline beside the control and Develop stays blocked. Prepared Batch N
siblings inherit the photo and both knobs (and the reviewed card names the same
reason on its own Develop), and changing the photo stales reviewed prompt work
exactly as changing a source image does.

Under **Auto** or **Most capable**, an identity print is only ever sent to a
machine that advertises identity support for that model itself; the model list
is merged across your machines, so the one you staged the photo against is not
necessarily the one that develops it. If the machine that was chosen cannot
hold the face, or is running a Mold old enough to ignore the photo, Mold says
so and queues nothing rather than returning a print of someone else.

Choose resolution through proportionally drawn shape tiles (the canonical
families (1:1, 5:4, 4:3, 3:2, 16:9, 21:9 and their portrait twins) the selected
model can actually express, plus **Source** when an image is attached) then a
**Size** row that lists that family's authored sizes as exact pixels with
their megapixels underneath. Explicit custom dimensions stay available. One
status line names the canvas and why it holds that size (`Matches source`,
`Follows source`, `Model default`, `Manual`), and the badge beside it always
agrees with that sentence. Seed has separate **Random** and **Fixed** modes,
including one-tap reuse of the last generated seed.

When several jobs are submitted, the Queue section keeps them visible while
you continue composing. Mold limits simultaneous streams so generation does
not starve gallery or download requests.

Queue rows on a machine's own screen swipe like any iOS list: drag right to
left to reveal **Cancel**, or keep going and a full swipe cancels the job.
Revealing the buttons is the first step and tapping one is the second, so
nothing is cancelled by a single flick. Every action is also on the **⋯**
button at the end of the row, so VoiceOver and a hardware keyboard reach it
without the gesture. Machines that support reordering also offer **To back**,
which sends the job to the end of the line. Tap the row itself to see the whole
job; its prompt, its settings, where it is in line, its live preview while it
runs, and the full reason if the machine has parked it.

For model families that stream live latent previews (FLUX.1, Flux.2,
Z-Image, and Wan 2.1/2.2), the active print develops right on the Create screen: the preview
sharpens as denoising progresses under a thinning film-grain wash, in a bed
matching the print's aspect ratio. Hosts without previews keep the plain
status line.

Batch is also the expansion count. Batch 1 freezes the selected host through
the next **Develop** while keeping quick undo. Batch 2 or greater first requests
exactly N distinct non-empty prompts from that host and opens an inline
workspace where
you can edit, intentionally remove, regenerate, refresh, or discard them before
anything is queued. Above eight, the workspace shows a compact first-eight
summary and bounded pages on demand. Queue acceptance immediately restores the
composer so another batch can be prepared while the earlier one runs. A single
reviewed set has a 10,000-variation memory-safety ceiling, but you can keep
queueing additional sets. Reducing
two prompts to one requires confirmation. Changes
to the source prompt, model, family, Batch count, host endpoint, Keychain key,
or server identity preserve the reviewed work with a specific stale reason and
block Develop until you choose Refresh or Discard.

The same frozen host is used for expansion-model pulls, source preprocessing,
and every sibling. Missing-model recovery stays inline with Connecting,
Starting, Queued, percentage/bytes/files/ETA Pulling, Ready, and failure or
cancellation retry states. The original inputs and route remain one immutable
recovery record, and an attempt lease prevents a newer Models credential from
redirecting the pull. Compatible Models work already in Starting is joined;
every terminal, stale, superseded, or aborted attempt releases the lease before
Retry reacquires the same frozen route. Editing/removing reviewed work cancels
a pending replacement. Siblings remain independently cancellable and keep
deterministic seeds, the source prompt, and durable batch position through
long-video chains. Every print is admitted through one durable
`/api/generation-batches` operation, chunked at the machine's advertised limit;
held children survive app/server restarts with an error and retry action. A
machine that cannot carry a request refuses it inline by name, and nothing is
queued. A partial result names each failed variation and reviewed
prompt, plus any separate unconfirmed-cancellation caveat, while keeping
successful prints. Library shows **Batch N of M** and the source prompt when
that provenance is present.

When iOS suspends the app, interrupted generation streams carry a structured
recovery marker rather than relying on localized WebKit error text. On resume,
Mold checks the frozen host and bounds any pre-ID queue or durable-chain join
to the original submission window, so a later fixed-seed duplicate is never
cancelled or reported as the interrupted print.

## Library

Library merges prints from every saved host, newest first. Unavailable hosts
are reported without hiding media from hosts that did respond.

Pinch the grid with two fingers to resize the thumbnails, between two across
(largest) and five across (smallest). This is the iPhone equivalent of the
thumbnail-size slider in the web and desktop Library. The size is remembered on
the device and starts at three across; it is stored separately from the desktop
and web setting, so resizing on your phone never changes the grid on your Mac.
One-finger scrolling through the grid is unaffected.

Press and hold an image to open the native iOS image menu for Share, Save to
Photos, Copy, Copy Subject, and Look Up. That menu also offers **Upscale…** for
images and **Framewise upscale…** for videos. Tap **Select** to select multiple
prints; with one image or video selected, the action bar offers the same
upscale flow, and it can also select all loaded prints, clear the selection, or
delete the selection. Deletion removes every matching copy from each reachable
saved host, including legacy copies whose auto-save filename differs. If a host
cannot complete the delete, its copy remains visible and Library reports the
partial cleanup.

Tap a tile to open the full-screen viewer:

- images are shown uncropped;
- generated images open this same viewer when tapped;
- videos stream from their owning host with native playback and seeking;
- **Upscale…** enlarges an image, while **Framewise upscale…** queues a durable
  video job with pause, resume, and cancel; missing Real-ESRGAN weights download
  on first use and temporal flicker may remain;
- use **Save video** to add the original MP4 to Photos;
- swipe left or right to move through the loaded prints;
- use **Copy image** or **Save photo** instead of the system long-press menu;
- use **Use as prompt** to restore recorded generation settings, or **Use as
  source** to attach a still to the next compatible generation.

New installs also save completed stills to the iPhone Photos library
automatically. Disable **Settings → Photos → Save to Photos automatically** if
you want outputs to remain only in Mold Library. Post-generation upscaling
saves both the original and upscaled image. Video generation offers the same
Framewise upscaler picker, and the just-generated viewer can queue it after the
original is published; videos remain on their Mold host.

Prints added since the prior Library visit carry a **New** badge. Images enlarged
by post-generation or standalone upscaling carry an **Upscaled** badge on both
iPhone and desktop.

### Collections, favorites, tags, and titles

When a connected host supports Library organization, a **Prints | Collections |
Trash** row appears under the Library header (each scope with its count; the
row only appears when a host supports it). In **Prints**, a scrolling chip row
filters the grid: ♥ Favorites, your most-used tags (the rest behind **More…**),
and a chip per machine when several are connected. Favorite prints carry a ♥
badge on their tiles.

In Select mode the action bar gains **Add to collection** (a checklist with a
New-collection input), **Tag** (add or remove tags, with suggestions from every
host), and a ♥ toggle. Edits apply to every copy of the print on every
reachable machine; if one machine cannot be updated, Library says so inline and
keeps the rest.

**Collections** lists your collections merged across machines (cover, name,
count, and which machines hold them) with a **New collection** row. The **…**
menu renames or deletes a collection (two taps; its prints stay in the
Library). Tap a collection to browse it; Select there offers **Remove from
collection**.

Tap a print and use **Info** to edit it in place: the title (Done saves; blank
clears it), ♥, tags, and its collections. The viewer's title line shows the
print's title, or its prompt while untitled.

A print developed with an identity photo also lists that provenance in **Info**:
the photo's filename, the first characters of its SHA-256, and the effective
strength and start step. Saved metadata records the digest, never the face
bytes, so **Use as prompt** restores the two knobs and re-attaches the photo
only when this device still holds it. When it does not, Mold says so instead of
rendering a different person.

### Trash

On hosts with the server-side trash, deleting moves prints to the **Trash**
scope instead of erasing them. The Trash grid shows how long each machine keeps
trashed prints (change it from the host's detail screen), and each tile counts
down to its purge ("Purges in N d"). Select offers **Restore** and a two-step
**Delete forever**; **Empty trash** in the header purges everything after a
confirming second tap. Machines without a trash keep the old immediate delete,
and Mold's wording never promises recoverability there.

Reusing settings switches to the print's host and restores the model when it is
installed there. If the original model is unavailable, Mold clearly identifies
the compatible fallback and removes non-portable adapter/component choices. On a
print made scene by scene by an older build, **Use as prompt** restores a plain
one-shot clip built from the first scene's prompt; the per-scene provenance
stays on the print.

Authenticated hosts issue a short-lived read-only media ticket for the selected
file. The app never puts a long-lived API key in a video URL or buffers an
entire video into phone memory.

## Models

Models combines installed models with live Hugging Face and Civitai results.
You can search, filter All/Images/Video, filter by source, family, or model
kind (Models, LoRAs, CLIP, text encoders, VAEs, tokenizers, ControlNet), sort
by downloads, rating, or recency, inspect download contents and installed
components, and include NSFW entries explicitly. Every card identifies the
model kind; mature results carry an **18+ NSFW** badge. The detail sheet repeats
those classifications, uses a kind-specific weights label, and shows available
description, source, license, tags, format, and popularity metadata. When a
manifest model has multiple quantizations, its detail sheet
shows variant chips with checkpoint sizes; the selected chip is the exact model
the Pull action downloads.

The family list comes from the browsed host. If that host is unreachable when
Models opens (or its saved key has not been read from the Keychain yet) the
list falls back to every family Mold has seen this session, from both catalog
results and the installed models on your hosts. That set only grows, so
choosing a family never shrinks the list to that one choice, and it can offer
families that are not on the current page. The real list, and any search that
failed, reload as soon as the host answers.

The host selector controls where Models browses. Pulling a model can target a
different ready host without changing the host selected in Create. With more
than one target available, Mold opens a host picker.

Pull buttons reflect the server state immediately:

- **Connecting...** while the download event stream opens
- **Starting...** while the request is accepted
- **Queued** while waiting for a download slot
- **Pulling N%** during transfer

Active downloads stay pinned above results and can be cancelled. Installed
model details offer Load on GPU, Unload from GPU, and a guarded Remove from
host action.

## Host details

Tap a saved host to inspect its current state. The detail screen shows:

- GPU/VRAM, CPU, and RAM telemetry;
- every GPU's utilization, VRAM, lifecycle state, and enable/disable control;
- free and used storage for the models filesystem;
- queued/running generation work, with a confirmed **Cancel** action for queued
  work and active singleton generations on current hosts, and loaded models;
- queued/active model downloads with progress;
- the models installed on that host.

Hosts with the server-side Library trash add a **Library** card: choose how
long that machine keeps trashed prints (**Trash retention**, `0` keeps them
forever; a value pinned by an environment variable is read-only and names the
variable) and see **Prints in trash: N** with a two-step **Empty trash**. The
setting lives on the host itself; it applies to every app that talks to it.

From the same screen you can rename or retry the host, select it for Create,
unload a model, open it in Models, or forget it. Forgetting a host also deletes
its API key from the iOS Keychain. Queue cancellation uses that host's
Keychain-authenticated route and refreshes the list after the server confirms
it. Running inference is revoked cooperatively at its next safe point.

## Settings and themes

Open Settings from the sliders button in the header. Mobile settings currently
cover:

- **Look:** the six Mold Studio themes — Mocha, Safelight, Blueprint,
  Graphite, Porcelain, Nebula
- **Appearance:** **Match phone**, which swaps to the theme's light or dark
  partner with iOS
- **Photos:** automatically save newly generated stills to the iPhone photo
  library (on by default)
- **Model licenses:** review and accept the selected host's model licenses
- **Library:** **Tag new prints with their title**; offer a titled print its
  own title slug as a tag in Create (on by default)
- **Remote hosts:** saved-host count and a shortcut to manage them
- **Compute devices:** enable or disable each GPU on the selected host; a busy
  GPU finishes its current stage before disabling
- **About:** app version, remote-only processing, TestFlight updates, and equal
  project-owner credit for core contributors James Brink and Jeffrey Dilley,
  plus an external link to the [Mold privacy policy](/privacy)

Fresh installs start on the Safelight theme with Match phone off, and with
Photos auto-save and title tagging enabled. Existing users keep any valid saved
choices, with auto-save and title tagging enabled when upgrading from a
settings record that predates either option.

The appearance choice updates both the WebView and native iOS system chrome so
status-bar content remains readable. Themes change the app chrome but never
recolor generated photos or videos.

The app intentionally prevents WebKit input-focus and double-tap page
magnification and suppresses rubber-band scrolling. The Library's horizontal
swipe navigation and its grid pinch-to-resize gesture remain enabled. iOS
system-level accessibility Zoom is separate from WebView page zoom.

## Updates and current boundaries

Internal TestFlight builds are created automatically after mobile-relevant
changes pass the iOS workflow on `main`. The pipeline validates the bundled
mobile `index.html`, Mold icon catalog, native archive, App Store Connect
processing, and internal tester access before it is considered complete.

The iPhone app focuses on remote Create, Library, Models, Machines, and
appearance settings. Video is a setting inside Create, not a separate screen:
pick a clip style, describe the shot, choose the duration, and generate while
Mold keeps progress and cancellation attached to the exact host. Ask for a clip
longer than one render and the machine splits, carries, and stitches the work
for you — it arrives as one video in one queue row. An optional opening image,
with its source strength and fit controls, is a disclosure in the primary
Create stack beside the other source media rather than inside the Advanced
sheet; the primary **↺ Reset** clears it while **Reset advanced** leaves it
alone. Use desktop or the CLI for a local engine, scene-by-scene scripted
sequences and the full jobs-administration workspace, RunPod provisioning,
engine configuration, and desktop Stable/Nightly self-update controls.

For server networking and deployment options, continue with
[Remote Workflows](/guide/remote-workflows). For supported model-family
controls, see [Feature Support](/guide/feature-matrix).

## Troubleshooting

### Discovery finds nothing

- Confirm Local Network access is enabled for Mold in iOS Settings.
- Confirm the phone and host are on the same LAN; Bonjour does not browse a
  remote tailnet.
- Confirm the server was not started with `--no-mdns` or `MOLD_MDNS=0`.
- Add the IP address or Tailscale MagicDNS name manually.

### The host test fails

- Open `http://host:7680/health` from another device on the same network.
- Confirm the server listens on `0.0.0.0`, not only `127.0.0.1`.
- Check the firewall, Tailscale ACL, port, scheme, and API key.

### A video will not play

Update the remote Mold host. Current authenticated video playback needs the
short-lived gallery media-ticket endpoint so iOS can make native Range requests
without exposing the API key.
