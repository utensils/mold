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

Tap **Model** below the prompt to open **Choose a Model**. Search the installed
models, choose Still picture / Short clip / 3-D object, and select a machine
(or leave it on Auto). **Get More Models…** opens model management directly. The controls come from the model itself: a model that reads a
starting picture shows **Start from**; one that takes references shows
numbered wells (**image 1**, **image 2**) matching how the prompt names them;
clips add a length; 3-D objects say when the prompt is ignored. **More
options** holds the sampler, seed, adapters, identity photos, ControlNet and
the mask editor, and only what the model can use. Shape, Steps, Batch and clip
Length have separate rows that remain readable at large text sizes. Clip and
3-D drafts keep their kind when you reopen the app. A model search with no
matches offers **Clear Search**.

**Generate** never turns into Stop. Pressing it again queues another render;
Stop is its own small button beside the progress sentence, e.g.
"Adding detail — about 12s left" over `denoise 18/28`.

The full-screen viewer hides the main tabs so **Share**, **Favourite**, **Info**
and **Delete** remain reachable; use Back to return to the Library. Info has a
**Done** button. Shared media keeps its original filename and file type.

## Library, Queue and Models

- **Library** keeps working offline: it shows each machine's saved prints
  when the machine isn't answering, and the prints you have opened. Pinch to
  change the tile size (five sizes, from seven columns to one). Settings ▸
  Library sets how much space it may use and can save every thumbnail ahead
  of time.
- **Library** shows every machine's prints as one grid, a print found on two
  machines once. Favourites, tags, collections and Recently Deleted work the
  way they do on the Mac, across every machine that holds a copy. Search
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
Dynamic Island. mold servers cannot push to a phone, so once the app is in the
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
