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
  in place.
- **Library** — every machine's prints in one day-sectioned timeline, with a
  print held on several machines shown once. Favourites, collections, tags,
  search tokens (`is:video`, `tag:name`, `on:machine`), Quick Look, undo,
  interactive 3-D mesh viewing and export, and **Use These Settings** to
  restore a print's full recipe.
- **Queue** — every machine's work live from its event stream: reorder,
  pause, resume, cancel, and move held jobs between machines.
- **Models** — installed models grouped by family, catalog discovery, and
  downloads with gated-licence acceptance in place.
- **Machines** — the fleet at a glance, per-GPU memory and load, nearby
  machines to add, and phone pairing for keyed hosts.
- **This Mac** — mold's engine starts in-process when the app opens (Settings
  ▸ This Mac turns that off) and joins the machine list like any other host.

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
