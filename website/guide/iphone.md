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

- Tap **Discover nearby** to browse `_mold._tcp` services on the current LAN.
  Allow Local Network access when iOS asks.
- Enter an IP address such as `192.168.1.10`. Mold adds `http://` and port
  `7680` when they are omitted.
- Enter a DNS hostname, HTTPS URL, or Tailscale MagicDNS name.

Paste the same API key, then tap **Test and save**. Saved host metadata remains
in the app's local storage, while the API key is kept separately in the iOS
Keychain.

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

Choose a host and one of its installed generation models. The form adapts to
the selected model family and uses the same request contract and model defaults
as desktop. The primary controls stay on the main screen; deeper options open in
a full-screen **Advanced** sheet, and prompt **style** presets compose at submit
without rewriting your prompt text.

The mobile composer includes:

- Batch 1 quick prompt expansion with undo, Batch N prepared-variation review,
  and remote prompt history;
- local templates that can be saved, searched, sorted, renamed, loaded, and
  deleted;
- batch generation as independent queued jobs, each with its own progress and
  Cancel action;
- source images and fit policies, Qwen edit target/reference images, masks,
  ControlNet, LoRA stacks and trigger words;
- scheduler, CFG++, steps, guidance, output format, and post-generation
  upscaling where the selected family supports them;
- validated video frames/FPS, audio, camera motion, source media, keyframes,
  retake, and LTX-2 pipeline/spatial/temporal controls; and
- a host-aware memory estimate before submission.

For video-only LTX-2 community checkpoints, **Generate audio** is disabled with
an explanation when the connected server reports that the installed files lack
an audio VAE or vocoder. Source-image video generation remains available.

Choose resolution through orientation, proportionally drawn aspect-ratio
tiles, a resolution tier, or explicit custom dimensions. Seed has separate
**Random** and **Fixed** modes, including one-tap reuse of the last generated
seed.

When several jobs are submitted, the Queue section keeps them visible while
you continue composing. Mold limits simultaneous streams so generation does
not starve gallery or download requests.

For model families that stream live latent previews (FLUX.1, Flux.2,
Z-Image), the active print develops right on the Create screen: the preview
sharpens as denoising progresses under a thinning film-grain wash, in a bed
matching the print's aspect ratio. Hosts without previews keep the plain
status line.

Batch is also the expansion count. Batch 1 freezes the selected host through
the next **Develop** while keeping quick undo. Batch 2 or greater first requests
exactly N non-empty prompts from that host and opens an inline workspace where
you can edit, intentionally remove, regenerate, refresh, or discard them before
anything is queued. Reducing two prompts to one requires confirmation. Changes
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
long-video chains. A partial result names each failed variation and reviewed
prompt, plus any separate unconfirmed-cancellation caveat, while keeping
successful prints. Library shows **Batch N of M** and the source prompt when
that provenance is present; prints from older servers remain unchanged.

When iOS suspends the app, interrupted generation streams carry a structured
recovery marker rather than relying on localized WebKit error text. On resume,
Mold checks the frozen host and bounds any pre-ID queue or durable-chain join
to the original submission window, so a later fixed-seed duplicate is never
cancelled or reported as the interrupted print.

## Library

Library merges prints from every saved host, newest first. Unavailable hosts
are reported without hiding media from hosts that did respond.

Tap a tile to open the full-screen viewer:

- images are shown uncropped;
- videos stream from their owning host with native playback and seeking;
- swipe left or right to move through the loaded prints; and
- use **Use as prompt** to restore recorded generation settings, or **Use as
  source** to attach a still to the next compatible generation.

Reusing settings switches to the print's host and restores the model when it is
installed there. If the original model is unavailable, Mold clearly identifies
the compatible fallback and removes non-portable adapter/component choices.

Authenticated hosts issue a short-lived read-only media ticket for the selected
file. The app never puts a long-lived API key in a video URL or buffers an
entire video into phone memory.

## Models

Models combines installed models with live Hugging Face and Civitai results.
You can search, filter All/Images/Video, filter by source, family, or model
kind (Models, LoRAs, CLIP, text encoders, VAEs, tokenizers, ControlNet), sort
by downloads, rating, or recency, inspect download contents and installed
components, and include NSFW entries explicitly. When a manifest model has multiple quantizations, its detail sheet
shows variant chips with checkpoint sizes; the selected chip is the exact model
the Pull action downloads.

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
- free and used storage for the models filesystem;
- queued/running generation work and loaded models;
- queued/active model downloads with progress; and
- the models installed on that host.

From the same screen you can rename or retry the host, select it for Create,
unload a model, open it in Models, or forget it. Forgetting a host also deletes
its API key from the iOS Keychain.

## Settings and themes

Open Settings from the sliders button in the header. Mobile settings currently
cover:

- **Color family:** the Mold Studio theme families, Mold or Safelight
- **Appearance:** System, Dark, or Light
- **Remote hosts:** saved-host count and a shortcut to manage them
- **About:** app version, remote-only processing, TestFlight updates, and equal
  project-owner credit for core contributors James Brink and Jeffrey Dilley,
  plus an external link to the [Mold privacy policy](/privacy)

Fresh installs start with the Safelight color family and System appearance.
Existing users keep any valid Mold or Safelight choice they already saved.

The appearance choice updates both the WebView and native iOS system chrome so
status-bar content remains readable. Themes change the app chrome but never
recolor generated photos or videos.

The app intentionally prevents WebKit input-focus and double-tap page
magnification and suppresses rubber-band scrolling. The Library's horizontal
swipe navigation remains enabled. iOS system-level accessibility Zoom is
separate from WebView page zoom.

## Updates and current boundaries

Internal TestFlight builds are created automatically after mobile-relevant
changes pass the iOS workflow on `main`. The pipeline validates the bundled
mobile `index.html`, Mold icon catalog, native archive, App Store Connect
processing, and internal tester access before it is considered complete.

The initial iPhone release focuses on remote Create, Library, Models, Machines,
and appearance settings. Use desktop or the CLI for a local engine, the full
Chains authoring/jobs workspace, RunPod provisioning, engine configuration, and
desktop Stable/Nightly self-update controls.

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
