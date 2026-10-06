# Mold Studio for Mac

Mold Studio is the native macOS app for mold and the **recommended download on
a Mac**. It is written in Swift, carries mold's own Rust engine running on
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

::: tip Coming from the Tauri desktop app?
Mold Studio installs as **Mold Studio.app**; the older cross-platform desktop
app is **Mold.app**. Both can sit in /Applications at once while you move
over. The Tauri app is still published as the
[legacy Mac download](https://github.com/utensils/mold/releases/latest/download/Mold-macos-arm64.dmg)
for Macs that cannot run macOS 26, and remains the default desktop app on
[Windows and Linux](/guide/desktop).
:::

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
  collection itself and Recently Deleted remain accessible. Context menus keep
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

In Library, **Media Type** filters the current shelf to All Media, Photos,
Videos or 3D while keeping search and machine filters. Multi-select with
Command-click or Shift-click without recentering tiles already in view.
Arrow-key navigation still reveals offscreen selections.

### References and boundary frames

Both native apps expose the server's reference contracts. MiniMax H3 **Ref2VA** takes an ordered mixture of images, H.264 MP4 clips and mono/stereo PCM WAV audio; image references can come from Photos/Camera/Library/Share on iOS or Finder/Library/Paste on macOS, and movie/audio files use Files/Finder. Replace, remove and reorder attachments before generating. Use `image 1`, `video 1` and `audio 1` in the prompt (numbered within each media kind). Audio references need at least one visual reference. The limits are nine images, three videos, three audio files and twelve files total; each clip is 2–15 seconds, with at most 15 seconds of video and 15 seconds of audio including video soundtracks. Authenticated hosts use request-bound upload sessions; keyless hosts accept at most 32 MiB of inline reference media per render. Video clips with sound require authenticated uploads so the server can supply exact decoded soundtrack counts; on keyless hosts use a silent MP4 plus separate PCM WAV audio. Unsupported or oversized files report an error instead of silently disappearing.

Hunyuan3D multiview models offer named Front/Left/Back/Right wells from their recipe. Wan offers a first/last pair rather than arbitrary middle frames; MiniMax **FL2VA** offers separate optional first/last frames. Changing clip length updates the closing frame. Existing Qwen Edit, Qwen Image 2.1 and Flux.2 reference strips honor their source-image relation, count and pixel budgets; the last Qwen Image 2.1 reference updates the default canvas until you choose a size. SD1.5/SDXL reference weight comes from the model's own control. Model changes park unsupported attachments so they can return. Reuse restores retained typed references with fresh media authority while their original set and order stay unchanged. Changing retained slots requires reattaching the remaining originals; archived bytes never overwrite new attachments. Imported mesh texture/roundtrip workflows remain API/CLI-only.

### Recalling retained recipes

Reused ordered image references have authenticated previews. Prompt, shape and seed changes preserve the unchanged reference set, and each render gets fresh authority. If retained slots are reordered, removed or replaced, attach the remaining originals before generating. Preview failures offer Retry and do not discard the references. Shape menus show the actual aspect outline.

The app stores a byte-free origin locator for retained recipes. After relaunch it verifies the original machine instance, archive and output before enabling Generate. If verification fails, reconnect and reselect the source print, replace the attachments, or explicitly discard retained conditioning. Ordinary inline media and session handles are not persisted. Native shipping builds include the reviewed Metal H3 engine; remote generation uses the selected host's capabilities.

**Save All to This Mac** copies retained inputs with every supported output type, independently of model family. The destination owns the source, identity, mask/control, edit/typed references, audio/video, keyframes, continuation and chain stage media, so reuse does not need the original machine. Incomplete source transfers are reported as failures. Reusing sequence settings restores the first stage's source; later stages remain in the copied archive. Rebuilding the complete authored sequence remains a separate workflow.
