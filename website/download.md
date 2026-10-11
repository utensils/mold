---
title: Download
description: Download Mold for Mac, Windows, Linux, iPhone, iPad and Android, plus the mold CLI.
---

# Download

Every app below is a client for the same mold engine. Desktop apps can make
pictures on the computer they run on and on any other `mold serve` machine you
connect; the phone apps drive those machines remotely. Links always point at
the newest stable release; nightly builds from `main` are listed
[at the end](#nightly-builds).

## Mac

There are two desktop apps for the Mac. Both are signed and notarized, both
connect to the same machines, and they install side by side, so you can keep
both.

<div class="platform-downloads">
  <a class="platform-download" href="https://github.com/utensils/mold/releases/latest/download/Mold-Studio-macos-arm64.dmg">
    <img src="/icons/apple.svg" alt="" />
    <span><strong>Mold Studio for Mac</strong><small>Built natively for macOS · Apple Silicon · macOS 26+</small></span>
  </a>
  <a class="platform-download" href="https://github.com/utensils/mold/releases/latest/download/Mold-macos-arm64.dmg">
    <img src="/icons/apple.svg" alt="" />
    <span><strong>Mold Desktop for Mac</strong><small>Same app as Windows and Linux · Apple Silicon · macOS 12+</small></span>
  </a>
</div>

|             | Mold Studio                              | Mold Desktop                           |
| ----------- | ---------------------------------------- | -------------------------------------- |
| Made for    | macOS only, written in Swift             | Mac, Windows and Linux, one shared app |
| Requires    | Apple Silicon, macOS 26 or newer         | Apple Silicon, macOS 12 or newer       |
| Installs as | **Mold Studio.app**                      | **Mold.app**                           |
| Updates     | Built in (**Mold ▸ Check for Updates…**) | Built in, Stable or Nightly channel    |
| Guide       | [Mold Studio for Mac](/guide/macos)      | [Desktop app](/guide/desktop)          |

**Which one?** If your Mac runs macOS 26, Mold Studio is the one that looks and
behaves like a Mac app. If it runs an earlier macOS, or you want the identical
app on every computer you use, take Mold Desktop.

## Windows

<div class="platform-downloads">
  <a class="platform-download" href="https://github.com/utensils/mold/releases/latest/download/Mold-windows-x64-self-signed.exe">
    <img src="/icons/windows.svg" alt="" />
    <span><strong>Mold Desktop for Windows</strong><small>Installer · x64</small></span>
  </a>
  <a class="platform-download" href="https://github.com/utensils/mold/releases/latest/download/mold-x86_64-pc-windows-msvc-cpu.zip">
    <img src="/icons/terminal.svg" alt="" />
    <span><strong>mold CLI for Windows</strong><small>CPU and remote-machine client · x64</small></span>
  </a>
</div>

The Windows builds are signed with mold's own certificate rather than a
publicly trusted one. Before running the installer, check and trust the
[public certificate](https://github.com/utensils/mold/releases/latest/download/mold-windows-self-signing.cert.cer)
as described in the [Windows instructions](/guide/desktop#windows). The
published builds use the CPU or a connected GPU machine; a CUDA build for a
local NVIDIA GPU is built from source. CLI setup is in
[Installation](/guide/installation#windows-cli).

## Linux

<div class="platform-downloads">
  <a class="platform-download" href="https://github.com/utensils/mold/releases/latest/download/mold-desktop-x86_64-unknown-linux-gnu-cpu.tar.gz">
    <img src="/icons/linux.svg" alt="" />
    <span><strong>Mold Desktop for Linux</strong><small>x86_64 · for remote GPU machines</small></span>
  </a>
  <a class="platform-download" href="/mold/guide/installation#one-line-install-recommended">
    <img src="/icons/terminal.svg" alt="" />
    <span><strong>mold CLI and server</strong><small>One-line install · picks the build for your NVIDIA GPU</small></span>
  </a>
</div>

The desktop download is a GPU-free build that drives your CUDA machines over
the network; on Arch it is the `mold-ai-desktop-bin` AUR package. For the
desktop app with local CUDA generation, install `mold-ai-desktop` from the AUR
or build it with Nix (`nix build .#mold-desktop`, or `.#mold-desktop-sm86` /
`.#mold-desktop-sm120` for RTX 30- and 50-series). The CLI and `mold serve`
also come as per-GPU release archives, Nix packages, AUR packages and Docker
images; see [Installation](/guide/installation).

## iPhone and iPad

Both phone apps are remote-only: they drive your mold machines and never run a
model on the phone. They are distributed through TestFlight; there is no public
App Store listing yet.

| App                                       | Runs on                  |
| ----------------------------------------- | ------------------------ |
| [Mold Studio Companion](/guide/companion) | iPhone and iPad, iOS 26+ |
| [Mold for iPhone](/guide/iphone)          | iPhone, iOS 17+          |

Mold Studio Companion is the native counterpart to Mold Studio for Mac; Mold
for iPhone shares its app with Android. Either one pairs with a machine by
scanning the pairing code your computer shows ([how pairing works](/guide/companion#add-a-machine)).

## Android

<div class="platform-downloads">
  <a class="platform-download" href="https://github.com/utensils/mold/releases/latest/download/Mold-android.apk">
    <img src="/icons/grid.svg" alt="" />
    <span><strong>Mold for Android</strong><small>Signed APK · Android 7.0+</small></span>
  </a>
</div>

Open the APK on the phone itself; no extraction is needed. The
[Android guide](/guide/android#download-and-install) covers the one-time
permission to install apps from your browser.

## mold CLI

mold is a command-line tool first, and every app above uses the same engine.
On macOS and Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/utensils/mold/main/install.sh | sh
```

[Installation](/guide/installation) covers Nix, Docker, the AUR, the
per-GPU archives and building from source. Checksums for every download are in
[`SHA256SUMS`](https://github.com/utensils/mold/releases/latest/download/SHA256SUMS).

## Nightly builds

Nightlies are built from `main` after its checks pass. They get new features
first and may change from one day to the next.

<div class="platform-downloads">
  <a class="platform-download" href="https://github.com/utensils/mold/releases/download/latest/Mold-Studio-macos-arm64.dmg">
    <img src="/icons/apple.svg" alt="" />
    <span><strong>Mold Studio for Mac · Nightly</strong><small>Apple Silicon · macOS 26+</small></span>
  </a>
  <a class="platform-download" href="https://github.com/utensils/mold/releases/download/latest/Mold-windows-x64-self-signed.exe">
    <img src="/icons/windows.svg" alt="" />
    <span><strong>Mold Desktop for Windows · Nightly</strong><small>Installer · x64</small></span>
  </a>
  <a class="platform-download" href="https://github.com/utensils/mold/releases/download/latest/Mold-android.apk">
    <img src="/icons/grid.svg" alt="" />
    <span><strong>Mold for Android · Nightly</strong><small>Signed APK · Android 7.0+</small></span>
  </a>
</div>

Mold Desktop for Mac switches to nightlies from inside the app (**Settings ▸
Updates & about ▸ Update channel ▸ Nightly**); its versioned nightly DMGs and every other nightly
file are on the [Latest Build](https://github.com/utensils/mold/releases/tag/latest)
page.
