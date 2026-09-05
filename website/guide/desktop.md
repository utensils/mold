# Desktop App

::: tip Looking for Mold on mobile?
The remote-only companions have dedicated [iPhone](/guide/iphone) and
[Android](/guide/android) guides with download, pairing, and setup instructions.
:::

Mold Studio is the native desktop app for macOS, Linux, and Windows. It puts
making a picture, the queue, everything you have made, your styles, your
machines, and settings in one focused window, in plain words, with six themes
that keep attention on your work.

::: info
The desktop app lives in `desktop/`. Local generation uses Metal on Apple
Silicon and CUDA on x86_64 Linux and Windows. On a machine with no supported
GPU (an ARM64 Windows laptop, for instance) the app still runs and connects
to a remote `mold serve` machine.
:::

![Mold Studio desktop app generating an owl](/screenshots/mold-studio-desktop.png)

_Mold Studio's New image workspace keeps the canvas, prompt, model-aware controls,
machine status, and live generation progress in one native window._

## Download

<div class="platform-downloads">
  <a class="platform-download" href="https://github.com/utensils/mold/releases/latest/download/Mold-macos-arm64.dmg">
    <img src="/icons/apple.svg" alt="" />
    <span><strong>macOS Desktop</strong><small>Signed and notarized · Apple Silicon</small></span>
  </a>
  <a class="platform-download" href="https://github.com/utensils/mold/releases/download/latest/Mold-windows-x64-self-signed.exe">
    <img src="/icons/windows.svg" alt="" />
    <span><strong>Windows Desktop (Nightly)</strong><small>Self-signed NSIS installer · x64</small></span>
  </a>
  <a class="platform-download" href="https://github.com/utensils/mold/releases/download/latest/mold-x86_64-pc-windows-msvc-cpu.zip">
    <img src="/icons/terminal.svg" alt="" />
    <span><strong>Windows CLI (Nightly)</strong><small>Self-signed CPU / remote client · x64</small></span>
  </a>
</div>

Every tagged release ships a signed, notarized, stapled macOS DMG. Open it and
drag **Mold** to Applications; no quarantine dance needed. Version-pinned
downloads and `SHA256SUMS` are on the
[releases page](https://github.com/utensils/mold/releases).

The current Windows downloads are nightly, self-signed CPU/remote-client builds.
Follow the [Windows trust instructions](#windows) before running them. The
[Windows CLI installation steps](/guide/installation#windows-cli) cover PATH
setup and connecting to a GPU host. You can also build from source with the
platform commands below.

Linux builds are currently source/CI distributions: `nix build
.#mold-desktop` produces the native sm_89 package, with
`.#mold-desktop-sm86` for RTX 3090/A40 and `.#mold-desktop-sm120` for RTX
50-series. B200/sm100 is server-only; desktop connects to it remotely.
`desktop-build` produces the native package on NixOS and a CUDA AppImage on
conventional Linux. Tagged releases do not publish the AppImage yet.

Windows currently ships through the rolling nightly release. The
`Windows Nightly` workflow (`.github/workflows/windows-nightly.yml`) publishes
the self-signed installer, the CLI zip, and the public certificate to the
rolling `latest` prerelease; the `Desktop` workflow additionally keeps a
14-day CI artifact of the same installer.
There is not yet a publicly trusted Windows installer; see [Windows](#windows)
below for the toolchain, trust steps, and the capabilities that are still
absent.

## What it is

A single native window that puts the full mold workflow behind a keyboard-driven
UI, instead of the CLI or the browser SPA. The same `mold-ai-server` HTTP + SSE
surface powers it, so anything the app does maps to a documented endpoint.

The window is one unified toolbar, a sidebar, the view, and a status bar. The
sidebar names its destinations in plain words — **New image**, **Queue**, **My
images**, **Styles**, **Machines** (⌘1–⌘5) and **Settings** (⌘,) — and keeps
the queue under the machine that is making images, so work in progress is
always in view without leaving the picture. Technical truth stays beside every
plain label in mono (`flux-dev:q4`, `28 passes`, `14.9 / 24 GB`), the status bar
always says which machine, how full, and how deep the queue is, and every
overlay — the ⌘K palette, the lightbox, dialogs, toasts — is built on the same
six-theme design system. The guide below uses those names; the web studio and
the phone keep Create / Library / Models until their own redesign.

## Features

- **New image** (⌘1): a capability-driven inspector that shows only the
  controls a model's family supports (negative prompt, scheduler, CFG++,
  add-on looks, img2img source/mask/control, video frames/fps/audio). It reads
  as groups rather than knobs: Start from a photo with **Paint a mask** and
  **Use a face** beneath it, a **Quality** row of Draft / Good / Best built
  from the ladder the model recommends (about half, its default, and one and a
  half times), the sliders, **Add-on looks**, a **3-D
  object** card offering Rough / Normal / Fine, a **Clip** card holding length
  and sound (and Smoothness in clip mode), **Repeat this look**, and **Where it
  runs**. What to make — Still picture,
  Short clip or 3-D object — is the one control in the view toolbar. **Starters**
  shows your saved starting points as cards, with the manager behind Edit…, and
  **Recent** lists the pictures you have made; clicking one brings back
  everything it was made with and puts the picture itself on the canvas, exactly
  as **Use these settings again** does from My images. A finished picture's caption names its file and
  size and offers Save, **Make 4 variations** (⌥↩), **Make bigger**, and the
  rest behind ⋯. Styles are
  selected and shown throughout the queue, downloads, print metadata, and
  machine details by their human-readable catalog names even when their stable
  internal generation ids are `cv:` or `hf:`. For video-only LTX-2 community
  checkpoints, Mold disables
  generated audio when the installed files lack an audio VAE or vocoder while
  keeping image-to-video available. Generation is visualized as a
  print _developing_: a deterministic grain field, seeded from the job's real
  seed, resolves in lockstep with `DenoiseStep` events. Batches
  run sequentially with `base_seed + i`, and a VRAM preflight forecasts fit
  before you press Generate. Drop a PNG or JPEG from the file manager onto New
  image and it lands on the well under the cursor; Mold images with
  embedded generation metadata also restore their complete saved settings.
  Prompt expansion follows the directly editable Batch count: Batch 1 is a
  quick rewrite with undo, while Batch N greater than 1 prepares exactly N
  distinct editable variations before generation. Counts above eight start
  with a compact first-eight review and bounded Review all pages. Mold shows
  and freezes the resolved host for expansion and every sibling. One reviewed
  set may contain up to 10,000 variations as a memory-safety boundary; the
  number of sets you can queue is not limited. Every print (Batch 1, Batch N,
  and each prepared variation) is admitted through one durable
  `/api/generation-batches` operation, chunked at the machine's advertised
  limit; held children remain visible and retryable. A machine that cannot
  carry a request refuses it by name and queues nothing. Once the host accepts a batch, the composer is immediately available to queue another
  while the earlier work continues in the queue. When a Batch 1 rewrite becomes
  stale, New image immediately
  offers to re-expand the original for the current model and generate, generate
  the visible rewrite anyway, or restore the original; generation errors use
  larger copy with a copy-to-clipboard control. Source, model, host, or count
  changes keep Batch N reviewed work visible but require refresh or discard
  before it can run. Generation chrome uses human-readable catalog names while
  retaining stable IDs internally. My images shows
  each prepared print's durable batch identity and sibling position. When the
  expansion model is missing, the same inline area follows its pull on that
  frozen host through connection, queue, byte/file/ETA progress, readiness, or
  retry; it never redirects away from the composer or hides prepared work. The
  right settings inspector resizes from its left divider across 280–480 px,
  persists committed widths, and opens at 300 px; double-click the divider to
  restore that default. The Settings tab sits beside **Starters** and
  **Recent**. The
  essentials-only inspector stays compact by default. Toggle **Advanced** to
  extend capability-gated, always-open icon sections below those essentials in the
  same scrolling inspector; the canvas remains visible and edits apply
  immediately. A **↺ Reset** beside the Settings header restores every
  generation setting to the selected model's defaults while keeping the
  prompt, the model choice, and any prepared batch size.
  For LTX-2, Advanced also exposes optional STG scale/blocks, CFG rescale,
  audio/video modality scale, and guidance skip stride. Empty fields preserve
  the selected pipeline's constants; invalid values block Generate inline, and
  starting points plus **Use these settings** restore recorded overrides.
- **Short clip** (inside New image): video is a setting, not a place. Switch
  the view toolbar's control to **Short clip** (File → New Clip and the ⌘K
  palette's **Make a short clip** land there too) and a strip appears under the
  toolbar: **Simple | Scenes**, with one sentence saying what the chosen way
  does. It has a row of its own so the toolbar's control never moves.

  **Simple** is where Short clip opens, and it is the plain render — describe
  the clip, pick a clip style, drag the **Length** chip on the composer, press
  Generate. The chip reads `97f · 4.0s` and snaps to the same frame grid the
  inspector's Clip card slider offers, because the two are one control shown
  twice; ask for more than one render's worth and Mold splits it into a
  sequence for you. **Make** is an ordinary batch count here, so you can ask for
  four clips at once wherever the style allows it.

  **Scenes** is the second way, and it is what raises the timeline above the
  composer. The composer stays where it is: it now carries the selected scene's
  words — "Scene 2 — describe what happens next" — **Make** is hidden, because a
  clip made scene by scene has no batch, and ⌘↩ makes the whole clip. Switching
  from Simple seeds scene 1 with the words and the length you had already
  written; switching back parks the scenes untouched, so nothing is lost either
  way, and Mold remembers which way you were working. The palette's **Edit the
  clip scene by scene** is the same door. The timeline is a transport (play the clip back once
  scenes have rendered, with the clock reading `0:06 / 0:20`, and **Add a
  scene**), a ruler in round seconds, and the scenes lane. Each block in the
  lane is as wide as the time it plays, so the lane always fits the window and
  never scrolls; a block wears the scene's own words, its rendered frame behind
  them, and its length in the corner. Drag the selected block's right edge to
  make that scene longer — it snaps to the family's own frame grid (`8n+1`, or
  `4n+1` for Wan, defaulted from the selected model and capped at that model's
  own clip size, so a scene is never longer than the clips a one-shot Duration
  would be split into) — or drag the block itself to reorder. The seam floating
  above each join names its transition in words; **Smooth**, **Cut**, or
  **Fade 8f** (zero-tail joins say **Join**); a click opens the seam editor's
  teaching rows and fade-length stepper. With a block focused, ←/→ move between
  scenes, Shift+←/→ change its length, ↩ opens the seam before it, and ⌫ removes
  it after a confirm. Right-click a block for Play (when a cached render
  exists), Duplicate, Insert before/after, Move, and Remove, or the lane
  background for Add clip, Check the plan, the TOML file tools, and Clear the
  clip. One mono line beneath the lane says what the clip adds up to — "3 scenes
  · 289 frames · 0:12 at 24fps", or the machine's own answer once you check the
  plan. A live fits/duration forecast runs against
  `/api/capabilities/chain-limits`, TOML import/export lives under File tools,
  and a running clip sits in the sidebar's queue beside prints, with
  watch and cancel. A finished clip leaves the queue: its video lands on
  the New image canvas with **Edit clip** and **Show in My images**, its print
  is in My images, and its job record is in **My images ▸ History ▸
  Clips**. Editing a finished clip reloads its scenes onto
  the lane, marks which stay cached versus re-render as you change
  things, and **Update sequence** re-renders only from the earliest changed
  clip; changing a transition type or a fade length re-stitches with no
  re-render at all. From a clip print in My images, **Edit clip**
  re-enters the original job with its cached clips and **Duplicate as new**
  starts a fresh sequence from the recorded clips. The picker shows
  clip styles from every connected
  machine (choosing Short clip auto-picks one that can join scenes and
  remembers your single-mode style, and a clip style that only renders one
  clip at a time is listed with that reason on its row; switching back to
  Still picture restores a picture style; with none installed the menu says
  so and Browse more deep-links to the video filter), and limits, creation,
  events, previews, and job actions
  stay routed to the style's machine. An optional **Opening image** well (with its
  source strength and fit-to-frame controls) sits in the inspector's primary
  form exactly where one-shot source media lives (the header ↺ Reset clears it;
  the Advanced reset does not), and Advanced keeps the per-clip negative
  prompt and camera motion. Job and action failures stay visible inline.

- **Queue** (⌘2): the same line the sidebar shows, at full width. Three counts
  — **Being made**, **Waiting**, **Done today** — one explainer, and a table
  with a sentence of status per row (Image · What's happening · Style ·
  Machine). Drag a row to reorder it, or **Jump the line** on the one you need
  first; a waiting row's ⋯ offers Pause and Resume where the machine supports
  holding one job. **Pause queue** and **Stop everything** sit in the view
  toolbar (Stop everything asks first and names what it will stop), and Space pauses or resumes the queue from anywhere outside a field —
  on a machine that offers it, which is also where the status bar shows the hint.
  Closing the window keeps the queue running.
- **My images** (⌘3): a justified, virtualized contact-sheet grid, with a
  Lightroom-style small-to-large slider in the view toolbar
  that resizes the contact sheet continuously and remembers its setting, NEW
  badges on fresh prints, a two-pane lightbox, and a History column holding
  Runs, Prompts and Sequences. **Space** opens
  Quick Look, ←/→ navigate, and **Use these settings** jumps back to New image
  with every parameter restored. On a print a clip produced, **Edit clip**
  is the primary action and re-enters the original job on the machine that made
  it so already-rendered clips stay cached. **Duplicate as new** loads the
  recorded clips as a fresh sequence (if the durable job is gone Mold takes
  this fallback and says so; if the machine is unreachable it says so and
  changes nothing). All merges every connected host without repeating
  matching saved prints, prefers the copy on **This device**, and labels every
  host where a print is available; source filters retain each host's full
  gallery. Still images offer full-resolution **Copy image** from tile and
  lightbox right-click menus. **Use as source** is offered wherever a print is
  shown — a tile, the lightbox, a run in the History column, and the
  finished render on the New image canvas — and loads that print back into the
  composer: a still becomes the source image (or the H3 first frame or ordered
  reference on those models), a clip becomes the source video, while audio and
  3-D meshes are refused because neither is conditioning. The My images header is segmented
  **Everything | Favourites | Albums | Trash**, each with its own count.
  Everything and Favourites keep the grid plus a filter-chip row (tag chips
  ending in a **＋ tag** picker, then **Made on** and one chip per machine, in
  every scope); Albums puts its cover cards in a strip above the grid, which
  stays where it is, and each card drills in; Trash holds deleted prints under
  a banner that carries how long they are kept and a red **Empty now** behind a
  plain confirm, with a per-tile "Purges in N d" countdown and **Restore** /
  **Delete forever**. A favourite wears a ★ in the top-right corner of its
  tile, and a clip, a 3-D model, or a sound says so in a word at the
  bottom-left ("clip 5s", "3-D", "audio"). Selecting pictures opens a bar
  reading how many are selected, then ★ **Favourite**, **Add tag**, **Add to
  album**, **Export…** — which saves every selected picture where a single
  **Save a copy** would put it — and **Delete**. **History** opens as a column
  beside the pictures rather than over them, so the grid stays usable, and each
  past run shows its style, size, seed and time over a **Use these settings**
  line. Titles, ♥, tags, and album membership are edited in the lightbox
  aside; the raw filename becomes a detail line. Everything lives on the machine that holds the print
  (its `mold.db`) and is merged across machines (collections by name, tags
  case-insensitively) and every change is applied to every copy of a print.
  Deleting moves a print to that machine's trash (the 6 s Undo stays); prints
  are purged after **Settings ▸ My images & trash ▸ Keep deleted pictures for**
  on this
  device (1 day … 1 year, or Forever) and after each remote machine's own
  setting in **Machines ▸ machine ▸ Storage**, which also shows "Prints in
  trash: N" and an **Empty trash** action. Naming a print starts in New image:
  the
  header's "Untitled print" is editable (click, Enter/blur commits, Escape
  reverts); the name travels with every sibling of that print, is restored by
  **Use these settings**, and leads the name suggested when you save or export
  (`{title-slug}__{model}__s{seed}.{ext}`; the stored file is never
  renamed). Filing starts in New image too: a **File under** group sits in the
  inspector between the essentials and **Advanced**, offering the print's own
  title as a removable tag chip, typed tags with suggestions drawn from every
  connected machine, and a collection row that pre-selects, but never creates,
  the collection whose name matches the title, with a picker for the fleet's
  collections and an inline **New album…**. A line beneath previews the
  filename the print will land as. The choice rides the one shot, every
  sibling of a batch, every prepared variation, and the single print a
  sequence stitches; **Use these settings** restores it, and
  **Settings ▸ My images & trash ▸ Tag new prints with their title** turns the title
  chip off without touching prints you already made. Older servers without
  organization simply hide these controls and keep the previous delete
  wording.
- **History** (a column inside My images, `?panel=history`): a fast,
  searchable list of past prompts from every ready machine, with **Runs**,
  **Prompts** and **Clips** as tabs in the column body; ↩ refills the
  composer, while Up/Down recalls the same merged history inline. It opens
  beside the grid rather than over it, so the pictures stay clickable. The
  **Clips** tab is the one place durable clip jobs are listed: open,
  edit, resume, or delete a job, jump to the print it produced, and run the
  machine-scoped **Clear inactive** and **Clean up disk** maintenance. It
  renders the 200 newest jobs and says so when
  there are more. Web has the same panel at `?panel=history`.
- **Styles** (⌘4): one searchable styles workspace whose view toolbar carries
  **Ready to use | Browse more**, the kind filter (**All / Pictures / Clips**)
  and **Filter…**. Ready to use opens with a disk-used-by-styles meter — one
  segment per family, counting each style's own weights so a shared helper is
  never counted twice — and is headed once by a mono column row — **NAME ·
  GOOD FOR · SIZE · MACHINE** — that every row below it shares, so each row's ⋯
  sits on the same line whether or not the row also offers **Get it** or
  **Unload**. Behind ⋯ sit the style's page and **Remove from disk…**.
  Browse more is the live Hugging Face/Civitai catalog merged with what you
  already have, source chips (All / HuggingFace / Civitai), a model-kind chip
  row (Models, LoRAs, CLIP, text encoders, VAEs, tokenizers, ControlNet), an
  18+ NSFW tag with its own include checkbox, and sorting by downloads, rating,
  or recency, in compact Grid and Table layouts. **Get it** is the one
  acquisition verb on both tabs.
  Active downloads pin above both tabs and stay put while the list scrolls,
  reading as a sentence with the CLI progress line — bytes, rate, eta — beside
  it, and naming the machine receiving the download. The desktop reuses cacheable 512 px Civitai thumbnails across both
  layouts, lazily decodes them, and contains each card's layout and paint
  work. Missing previews use a local model-family mark, with no additional
  image request. Grid cards and table rows carry the same kind badge, and
  mature entries use an explicit **18+ NSFW** label. The detail drawer repeats
  those classifications and surfaces available description, tags, license,
  source, format, and popularity metadata. The catalog renders **SIZE vs
  FETCH** honestly, with the primary weight label named for the actual kind
  instead of assuming every entry is a checkpoint. With several
  machines connected, Get it asks which one should store the style, and each
  machine's inventory refreshes when its download completes. Ready to use
  merges every ready machine with per-machine badges and routes its
  actions to the machine that owns the row. Primary model **weights** are
  labeled separately from the larger footprint **with shared runtime** (text
  encoders and VAEs), so shared dependencies are not mistaken for checkpoint
  size. Curated manifest variants take precedence over ambiguous
  multi-checkpoint Hugging Face repositories, preventing a whole repository
  from being presented as one oversized download. Live Hugging Face LoRA
  collections likewise select one runnable adapter variant instead of summing
  mutually exclusive adapters and fused checkpoints. A machine that accepts a
  download without returning any queued job is reported as an error. New
  image's clip and video empty states deep-link straight to Browse more with
  the video filters set.
- **Machines** (⌘5): a list beside the machine's own page — This device first,
  then connected machines, any rented pods with their cost meter, Remembered,
  On your network, and the Rent-a-GPU offer. Landing on Machines opens the one
  the app is already talking about. **Connect a machine** is a single dialog
  (the machines found on your network, or a typed address, an API key when the
  machine asks for one, and "Make images here from now on"), never a stepped
  wizard, and Settings keeps only a doorway to this workspace. See
  [How it connects](#how-it-connects) for what a machine's page shows.
- **Waiting on this machine** (inside each machine): running and waiting jobs,
  pause/resume, cancellation, and queue capacity live with the machine that
  owns them; the **Queue** view (⌘2) is the same line across every machine.
- **RunPod** (inside Machines): secure account setup, balance and live spend, GPU and
  datacenter discovery, pod launch/lifecycle/connection, and persistent network
  volume create/select/rename/grow/delete. A selected volume is remembered,
  forces Secure Cloud in its datacenter, replaces the ordinary workspace disk,
  and cannot be deleted while attached to a pod. Because RunPod cannot stop a
  network-volume Pod, the app hides Start/Stop for those rows and explains that
  deleting the compute instance preserves `/workspace` on the volume. Logs use
  a supported handoff to the RunPod console rather than a nonexistent REST
  endpoint. Production network volumes accept 10–3999 GB; the form and native
  validation enforce that live bound before launch. Region selectors show both
  the geographic location and RunPod ID, while the volume form limits choices
  to datacenters that currently support persistent volumes. While a picture is
  being made on a connected running pod, the sidebar's queue shows the same
  live accrued-cost and hourly-rate meter as Machines.
- **Settings** (⌘,): a 200px jump nav beside one scrolling page of always-open
  sections — **Look**, **Defaults for new images**, **Write more for me**,
  **Machines**, **Styles & disk**, **Style licences**, **My images & trash**,
  **Saving pictures & clips**, **Phone pairing**, **Speed & memory**,
  **Accounts & tokens**, **Profiles**, **Advanced**, and **Updates & about**.
  Typing in the search field narrows the nav and the page together, and nothing
  is hidden behind an accordion. Look holds the six themes and the **Match
  system appearance** toggle; Machines keeps this device, its API key, and the
  Mold home, with connecting and forgetting other machines living in the
  Machines workspace; **Styles & disk** holds where styles are kept, where
  finished pictures are written, and how full that disk is; Style licences
  lists each licence as one row with its state and a single action; Speed &
  memory exposes the `MOLD_*` engine knobs as real controls, applied on engine
  restart; Accounts & tokens keeps Hugging Face and Civitai keys in an
  owner-only local file under the app's data directory (no Keychain prompts;
  exported to the engine as `HF_TOKEN` / `CIVITAI_TOKEN`); and Advanced carries
  every remaining `/api/config` row with its provenance tag (⌂ db / ⛁ file /
  ⚿ env), where environment-overridden rows are locked with the variable that
  owns them.
  Settings also shows the effective **Mold home** (the shared root holding
  config, the SQLite DB, models, outputs, and logs) with a native folder picker
  or typed path. Changing it offers a recommended copy-everything migration.
  Mold validates and stages the copy without overwriting a non-empty
  destination, preserves the old root, and relaunches only after the new
  location is ready. You can instead use the selected location as-is, and an
  unavailable external drive appears as a recoverable offline state. The choice
  is stored outside the selected root, so the CLI, TUI, server, and desktop all
  resolve the same root (an explicit `MOLD_HOME` env override still wins).
  About credits core contributors James Brink and Jeffrey Dilley in both the
  Settings workspace and the native app menu.
- **Command palette**: **Cmd/Ctrl+K** for navigation, actions, style search, and
  prompt-history search in one field. Each row shows its shortcut on the right
  and its group on the left — make, queue, go, styles, machines — so
  **Generate from these words**, **Make 4 variations of the last picture**,
  **Pause the queue**, **Connect a machine…**, **Download a style…** and
  **Rent a GPU…** are all one search away. Style search covers the whole fleet:
  a style this machine has reads **Use `<name>`**, one that only another machine
  has reads **Use `<name>` · on `<machine>`** and repins generation there when
  you pick it, and a style nobody has yet appears from the live catalog as
  **Install `<name>` · not installed** and queues the download. The palette picks
  the target itself and names it in the toast; open **Styles** when you
  want to choose the machine explicitly.
- **Notifications bell**: in the title bar next to Search, with an unread
  badge. Toasts stay transient, but the bell opens the durable session history
  of every toast; complete untruncated messages and error bodies, per-host
  context where known, timestamps, and collapsed ×N repeats (newest first,
  capped at 100). Severity is color-coded (green for an ordinary notice or a
  success, yellow for a warning, red for an error) with the severity also named
  for screen readers and carried by its own glyph, and the unread badge takes
  the worst unread entry's color, so a bell holding only notices reads green. Each row has a **Copy**
  button that puts the message, its full body, and the machine/time line on the
  clipboard; the app chrome is not selectable, so that button is how a long
  server error leaves the app. Opening the panel marks everything read; Clear
  empties it.
- **Native desktop integration**: platform menus and shortcuts, Linux and
  Windows native window decorations, macOS overlay chrome, and background
  notifications on generation, chain, and pull completion. macOS uses
  UserNotifications so a signed release inherits Mold's bundle identity and app
  icon; Windows uses a WinRT toast whose click routes to the print, model, or
  update the alert names, exactly as the macOS and Linux notifications do.

## Updates

This section describes Mold's in-app desktop updater. Automatic signed desktop
updates are currently macOS-only; Linux Nix/AppImage builds and Windows
installer builds report updates as unsupported and are replaced manually. The iPhone app updates through
TestFlight after mobile-relevant `main` changes pass iOS CI, App Store Connect
reports the build `VALID`, and internal tester access is verified.

Signed desktop builds keep update checks separate from installation. Mold makes
a best-effort check after the app opens, and **Mold → Check for Updates…** plus
**Settings → Updates → Check for updates** run the same check manually. A
check only reports what is available: Mold does not download, install, or
restart until you explicitly choose **Update and restart**.

Choose the release stream in **Settings → Updates**:

- **Stable** (default) follows tagged, production releases.
- **Nightly** follows signed and notarized builds from desktop-relevant commits
  on `main`, after both desktop frontend and Rust CI gates pass. Nightlies expose
  changes sooner and may contain regressions.

Both channels use public, HTTPS-hosted manifests:

- [Stable manifest](https://github.com/utensils/mold/releases/latest/download/mold-desktop-stable.json)
- [Nightly manifest](https://github.com/utensils/mold/releases/download/latest/mold-desktop-nightly.json)

Startup checks are non-destructive. When an update is available, Mold shows a
persistent banner in the app; if Mold is backgrounded it also sends a native
notification. Download and installation begin only after you choose **Update
and restart**.

Tauri's updater signature check is mandatory. Before the installed app is
changed, the complete archive passes Minisign verification against Mold's
embedded public key and is fully extracted into temporary storage. Mold rejects
unsafe paths and extra app bundles, binds the bundle identifier and version to
the manifest, runs strict Apple code-signature verification and a Gatekeeper
assessment, validates the currently running bundle, rejects DMG or translocated
launches, and proves the install directory can be replaced. Downloads stop
cleanly after 15 minutes.

Only after every preflight check succeeds does Mold use macOS's atomic bundle
exchange and restart. Mold does not run a post-launch health watchdog or
roll back after a few seconds: the update either verifies and installs, or it
fails before installation and the running version remains in place.

Switching from Nightly to Stable changes which manifest Mold checks, but never
silently downgrades the installed app. If your nightly version is newer than the
latest stable version, Mold reports Stable as current until a newer tagged
release is published.

### Keyboard map

| Shortcut            | Action                                                       |
| ------------------- | ------------------------------------------------------------ |
| Cmd/Ctrl+1–5, comma | New image / Queue / My images / Styles / Machines / Settings |
| Cmd/Ctrl+K          | Command palette                                              |
| Cmd/Ctrl+N          | New image (clear composer, focus)                            |
| Cmd/Ctrl+Enter      | Generate (in Short clip, make the whole clip)                |
| ←/→                 | Short clip: move between scenes on the lane                  |
| Shift+←/→           | Short clip: shorten / lengthen the selected scene            |
| ↩ / ⌫               | Short clip: open the seam before a scene / remove it         |
| Alt/Option+Enter    | Make 4 variations of the finished picture (not in a field)   |
| Cmd/Ctrl+E          | Write more for me                                            |
| Cmd/Ctrl+R          | Surprise me (a new seed)                                     |
| Cmd/Ctrl+.          | Stop the image being made                                    |
| Cmd/Ctrl+\          | Toggle sidebar                                               |
| Space               | Pause / resume the queue (where the machine offers it)       |
| Space               | Quick Look in My images, which keeps Space for itself        |
| ←/→                 | My images: navigate                                          |
| Cmd/Ctrl+F          | My images: focus the search field                            |
| Cmd/Ctrl+A          | My images: select every picture the filters show             |
| ⌫                   | My images: move to trash (Undo for 6 s)                      |
| Cmd/Ctrl+⌫          | My images: delete forever (confirm)                          |
| F                   | My images: favourite / unfavourite                           |
| T                   | My images: tag the selected print                            |
| Shift+Cmd/Ctrl+N    | My images: new album                                         |
| Shift+Cmd/Ctrl+C    | Copy seed (lightbox)                                         |
| Cmd/Ctrl+0 / + / −  | Interface size reset/larger/smaller                          |

Interface scaling applies to the complete app, including fixed overlays and
right-click menus. Choose 80–130% from **Settings → Look → Interface size**, or
use the View menu and keyboard shortcuts. The selected level is restored on
the next launch.

**Look** offers six Mold Studio themes: Mocha (the default), Safelight,
Graphite, and Nebula are dark; Blueprint and Porcelain are light. Each card
shows a band of that theme's own surfaces above its name, the tone it strikes,
and the type pairing it brings. Each theme also brings its own corner radius,
and **Match system appearance** swaps a pick for its light or dark partner when
macOS changes. Saved choices
from earlier releases migrate to the nearest theme (Safelight stays
Safelight; the Mold palette becomes Mocha or Blueprint; System becomes the
match toggle). New iPhone installs start with Safelight (see
[iPhone → Settings](/guide/iphone)). Every theme keeps text and interactive
boundaries at WCAG AA contrast; an empty generation canvas follows the selected chrome, while actual
generated media remains on a color-stable viewing surface.

## Starting points

Save the settings you have now as a named, recallable starting point. Open the
inspector's **Starters** tab, click **Edit…** to reach the save/search/sort
manager, give the current settings a name, and it becomes a card you can load,
rename, or delete later. A card carries its style's family mark rather than a
picture, because a starting point's media is conditioning input, not a result.
Loading one restores every parameter (style, prompt, dimensions, detail,
guidance, scheduler, add-on looks, and the rest) in one click.

Starting points capture _parameters_, not media: source, mask, and control
images (and LTX-2 source video / keyframes) are referenced but never stored, so
after loading one that used them the app reminds you to re-select the files.
If the starting point's style isn't installed you still get its settings, with a
prompt to download it.

They are stored locally in the app and never shared with the browser SPA
or synced to the server. Desktop and iPhone maintain separate device-local
libraries; one saved on the desktop does not appear automatically on
the phone.

## Device placement

**Settings → Advanced → Device placement** saves a per-model default for _where_
a model's components run. Pick an installed model, then set its **Text
encoders** (the Tier-1 group knob covering T5, CLIP, and Qwen encoders) to
**Auto**, **CPU**, or a specific **GPU**. For Tier-2 families (FLUX, Flux.2,
Z-Image, Qwen-Image) an **Advanced** disclosure exposes per-component overrides
for the transformer, VAE, and each text encoder; any encoder can also be left to
follow the group knob.

**Save as default** persists the choice for that model and **Clear** removes it.
Placement is applied the next time the model loads, so save it before you
generate. GPU choices come from the connected engine's live device list.

This is the desktop surface for the same mechanism the CLI's `--device-*` flags
and `MOLD_PLACE_*` variables drive; see
[Configuration → Per-component device placement](./configuration.md#per-component-device-placement)
for the full component list and semantics.

## How it connects

The app talks to a `mold-ai-server` over localhost HTTP + SSE using the same
wire types as the CLI and web UI:

- **Built-in engine and LAN server**: embeds the server in-process and runs on
  Metal on macOS or CUDA on Linux, so no separate `mold serve` is required. It
  listens on port 7680, advertises itself over mDNS, and is always the app's
  own engine; **This device** in the machine list. The **Machines** workspace
  exposes the persistent per-device API
  key that another Mold client needs to connect. If an unrelated process owns
  7680, Mold uses and advertises an ephemeral port instead.
- **Existing server**: auto-detects a running `mold serve` on
  `localhost:7680`.
- **Machines** (remote GPU boxes (e.g. a Linux CUDA machine for LTX-2) are
  added in the **Machines** workspace: **Connect a machine** (the machines found
  on your network, or a typed address, plus "Make images here from now on"), a
  **Connected** list, **Remembered** hosts for one-click reconnect (each with
  its own API key, stored in an owner-only file under the app's data
  directory) never the macOS Keychain, so connecting never triggers Keychain
  prompts), and an **On your network** list of discovered servers. A bare
  hostname is enough: `hal9000` expands to `http://hal9000:7680`. One
  physical server is one entry: hosts are deduplicated by the server's stable
  instance id, so a box reached by hostname, mDNS name, and IP address
  collapses into a single row whose name follows the server's hostname unless
  you rename it. There is no separate remote "mode"; installs that
  previously used a remote primary migrate automatically: the old primary
  becomes a connected host, keeps its API key, and stays the generation
  target until you change it. The network list uses the operating
  system's native DNS-SD browser on macOS, so advertised `_mold._tcp` services
  share the same cache and interface handling as Finder and `dns-sd`.
- **Generation controls** (**Shape** and **Resolution** quick-select common,
  per-family model-native sizes (with manual width/height for anything else);
  **Repeat this look** is an explicit **Keep | Surprise me** pair with the seed
  in mono beside it and one-click "lock last"; the style picker offers only
  the styles the section you are in can make — picture styles under **Still
  picture**, clip styles under **Short clip**, 3-D styles under **3-D
  object** — names that section above the list, marks each
  style's source (Hugging Face / Civitai / local) and ends in **Browse more
  →**, which opens Styles already filtered to the same kind, ready-to-use
  styles first; each section remembers the style it was last used with, so
  switching back to Short clip returns to the clip style you were on and the
  app reopens on the style and section you left;
  the VRAM badge states plainly what fits ("VRAM · fits) est. 2.3 GB of
  64.0 GB").
- **Upscaling**: pick a Real-ESRGAN model in Advanced ▸ Upscale to upscale every
  print as it develops (the engine pulls the model on first use and retains
  both the original and `-upscaled` result), or
  right-click any picture → **Upscale**; the result lands in this Mac's
  gallery. **Make bigger** under a finished picture runs the same flow against
  the machine that made it. **Use these settings** always restores the
  generation canvas, not the upscaled file's physical dimensions.
- **A machine's own queue**: a queue console for every connected machine: the full
  server-side queue (other clients' jobs included), live thumbnails and step
  progress for this app's own jobs, per-job cancel, drag-to-reorder
  (`PATCH /api/queue/:id` with a new position), **Pause/Resume** of a
  host's queue (the running job finishes; nothing new starts), a two-step
  **Cancel all**, and a "Finished this session" list with one-click reuse.
  Reorder, Pause, and Cancel all are feature-detected via `/api/capabilities`
  (`queue.can_reorder` gates reorder), so older servers simply hide the
  controls. The same queue mirrors into the sidebar rail and the Queue view.
  Click a job to open its detail panel: the prompt, the settings it was
  submitted with, where it sits in line, when it was submitted, whether it
  survives a restart, and (for a job the machine has parked) the full reason
  and error with a **Copy details** button. A running job shows its live
  denoise preview there too. From the panel you can **Use these settings** (which
  opens New image with everything restored), **Cancel** the job, or **Retry** one
  this app submitted that the machine parked. A job that has only just been
  accepted may not show its settings yet (the machine lists it before it loads
  the request) and the panel says so rather than pretending they are missing.
- **History (a column in My images)** (three lenses: **Runs** (every finished
  generation with its thumbnail, style, size, seed, and clock) click to reuse
  the full settings including the seed), **Prompts** (the raw prompt log,
  searchable, for prompts whose outputs are gone), and **Clips** (every
  durable clip job on every connected machine, with open / edit / resume /
  delete, a jump to the print it produced, and the machine-scoped **Clear
  inactive** and **Clean up disk** maintenance). The tab is in the URL, so
  `?panel=history&tab=sequences` opens straight onto it.
- **Remote prints saved locally**: generations from remote machines and RunPod
  are also written into this Mac's output directory (Settings → Look → "Save
  remote prints locally", on by default), with embedded metadata
  intact, so My images stays the complete record even when the GPU lives
  elsewhere. The tile's right-click menu adds **Save to this Mac** for
  pulling any older remote print down on demand.
- **Several hosts at once**: alongside this device, any number of remote
  hosts can be live simultaneously (**Connect a machine** in Machines, or
  **Connect** beside a machine found on your network). With more than one live
  machine, the **Where it runs** chip at the right end of the New image
  toolbar opens a machine menu: pick one explicitly, leave it on **Auto** to route
  each batch to the least-busy machine by live queue depth, or choose **Most
  capable** to always target the strongest GPU (CUDA over Metal, then most
  VRAM, then shallowest queue). Both automatic modes prefer machines that
  already have the selected style, and the style picker lists
  every connected machine's styles; one that only lives on a remote machine is
  tagged with the machine that has it, and routing there just works. Jobs
  stream progress from (and cancel against) the machine they queued on, so a
  long LTX-2 render on a CUDA box never blocks quick local prints. Connections
  are remembered and restored on the next launch.

  New image waits for those remembered machines and their style
  inventories before deciding there is nothing installed. The “get your first
  style” screen appears only when every connected machine reports zero
  installed styles; a remote-only style is selected and routed without a
  local download.

- **A machine's own page**: click a machine in the Machines workspace to open it:
  live GPU, CPU, and RAM telemetry, disk usage for the filesystem holding its
  styles, every GPU's utilization, VRAM and lifecycle state, current queue
  state, active download progress, and a freshly fetched inventory of
  the styles on that machine. Its toolbar says the machine in one plain
  sentence — `RTX 4090 · CUDA · on your network · up 6 days` — beside **Make
  images here**, **Rename**, **Open web UI** and **Forget…**; the address,
  version and instance id live in the name's tooltip, and clicking the name
  copies the instance id. Each **Right now** tile keeps one short reading
  (`14.9 / 24.0 GB`, `16 cores`) with the card and its backend in the tooltip,
  and **Downloads here** is a compact readout — the style, its percent and
  eta, a meter, and one line saying what else is queued. **Storage** sits
  beside it with that machine's own trash retention and an **Empty trash**
  action. Each GPU can be enabled or disabled from the machine's page.
  This device also exposes the same controls under **Settings →
  Advanced**. A busy disable drains its current stage before the owner thread
  exits; enabling starts a fresh owner thread.
- **Launch reconnect**: every remembered machine is attempted immediately on
  every app launch, in parallel with This Mac. An unreachable machine stays in
  the Machines workspace as an errored row marked _reconnecting…_, and the
  10-second status poll keeps probing it so it self-heals without any action
  from you. A machine that drops raises a yellow warning notification saying it
  is retrying. When it answers again that warning is withdrawn and a green
  **Reconnected to `<machine>`** notification confirms it. The web UI behaves
  the same way.

## Windows

Windows is a first-class desktop target: the same Tauri crate and the same
Mold Studio frontend, rendered by WebView2 instead of WebKit. Both x64 and
ARM64 (Snapdragon Surface devices) build and run.

### Toolchain

`scripts\windows.ps1 doctor` checks the whole list and names anything missing
with the command that installs it; `scripts\windows.ps1 setup` installs what it
can. What it looks for:

| Component                                        | Why                                                            |
| ------------------------------------------------ | -------------------------------------------------------------- |
| Visual Studio Build Tools (C++)                  | the MSVC linker and the `cc` builds inside the dependency tree |
| Rust ≥ 1.93 (`aarch64`/`x86_64-pc-windows-msvc`) | the workspace MSRV                                             |
| Microsoft Edge WebView2 Runtime                  | the webview the app renders in                                 |
| Bun                                              | the frontend build and test runner                             |
| `tauri-cli`                                      | `cargo tauri dev` / `build` (setup installs it)                |
| protoc (optional)                                | required for the `pulid` face-identity feature                 |
| NASM (optional, x64)                             | `openh264` builds a faster assembly path when it is present    |

### Commands

`scripts\windows.ps1` is the Windows peer of the Nix devshell's `desktop-*`
commands; the devshell itself does not run on Windows:

```powershell
scripts\windows.ps1 doctor   # verify the toolchain, name what is missing
scripts\windows.ps1 setup    # install what doctor can install
scripts\windows.ps1 dev      # Tauri app with hot reload (Vite on :1430)
scripts\windows.ps1 ui       # frontend-only Vite server
scripts\windows.ps1 check    # rustfmt, clippy -D warnings, vue-tsc, prettier
scripts\windows.ps1 test     # cargo test (CPU) + vitest
scripts\windows.ps1 build    # NSIS installer plus the standalone Mold.exe
scripts\windows.ps1 bundle   # alias of build
scripts\windows.ps1 clean    # drop the desktop build outputs
scripts\windows.ps1 features # print the feature recipe resolved for this machine
```

The `main` artifact is `mold-desktop-windows-x64-self-signed`. It contains the
NSIS installer and `mold-windows-self-signing.cert.cer`. The signature proves
that an installer came from Mold's CI only after the public certificate has
been trusted on that Windows account; it does not establish a publicly trusted
publisher and does not suppress SmartScreen on a fresh machine. Inspect the
certificate thumbprint before trusting it:

```powershell
certutil -hashfile .\mold-windows-self-signing.cert.cer SHA1
# Expected: E8 DA 29 90 15 5C CC 6E 92 78 A8 31 90 08 A7 63 AC 5D FC 79
Import-Certificate -FilePath .\mold-windows-self-signing.cert.cer `
  -CertStoreLocation Cert:\CurrentUser\Root
Import-Certificate -FilePath .\mold-windows-self-signing.cert.cer `
  -CertStoreLocation Cert:\CurrentUser\TrustedPublisher
Get-AuthenticodeSignature .\Mold_*_x64-setup.exe | Format-List Status,SignerCertificate
```

Only install this certificate on machines where you explicitly trust Mold's
GitHub release process. Remove it from both stores when that trust is no longer
required. CI imports the password-protected PFX from the
`WINDOWS_CERTIFICATE` and `WINDOWS_CERTIFICATE_PASSWORD` repository secrets,
checks its pinned thumbprint, signs both `mold-desktop.exe` and the NSIS
installer through Tauri, and fails closed if either secret or signature is
missing. The retained private material must never enter the repository.

The feature recipe is resolved per machine and printed by `doctor`: CPU-only by
default, `cuda` added on an x64 host with a CUDA toolkit, and `pulid` added when
protoc is on PATH. `MOLD_WINDOWS_FEATURES` replaces the whole recipe;
`MOLD_WINDOWS_CUDA=1` and `MOLD_WINDOWS_NO_PULID=1` each move one axis.

### What is not there yet

These capabilities and gates are absent or intentionally excluded, and each
says so by name:

- **Generated AAC audio tracks.** The `mp4` feature pulls `fdk-aac-sys`, whose
  `FDK_archdef.h` recognises only GCC/Clang architecture macros and falls
  through to a `#warning`, which MSVC raises as the fatal error C1021 on x64
  and ARM64 alike. Video renders and muxes normally through the pure-Rust
  writer; only an explicitly requested audio track is refused.
- **In-app updates.** The updater's preflight is built around macOS bundle
  identity, `codesign`/Gatekeeper, and `RENAME_SWAP`. Windows reports updates
  as unsupported and is replaced by re-running the installer.
- **The `h3` / `h3-private-uat` features.** MiniMax-H3 is a CUDA/Metal surface
  whose private evidence capture is written against unix ownership semantics
  (`/proc/self/statm`, uid/mode identity), so those features do not compile for
  Windows at all. Nothing in the Windows recipe enables them.
- **`cargo test --workspace` is not the Windows gate**: `scripts\windows.ps1
test` is. Besides the `h3` compile above, a handful of `mold-core` tests
  assert unix path separators (`should end with .mold/output`) and fail on
  Windows on `main` today, independently of any Windows work. The desktop
  crate, which is what the Windows app actually builds, passes cleanly.

Publicly trusted Windows installers are still to come; current release and
`main` artifacts use the pinned self-signed certificate described above.

### Notes for contributors

- The repository pins LF line endings in the working tree via `.gitattributes`.
  Git's default `core.autocrlf` on Windows would otherwise check the tree out
  as CRLF, which prettier rejects for every file.
- On ARM64, `.cargo/config.toml` enables the `fullfp16` target feature for
  `aarch64-pc-windows-msvc`. `gemm-common`'s inline `fmla v.8h` requires it and
  it is not baseline on Windows, so without the flag the build type-checks and
  then fails during codegen; `cargo check` never sees it.
- Enable Windows long-path support (`LongPathsEnabled`). Cargo target paths in
  this tree get deep enough to matter.

## Development

Run inside `nix develop` (the devshell wires up Metal or CUDA, Bun, Tauri, and
Linux WebKitGTK/GStreamer dependencies):

On Windows the devshell is not available; use `scripts\windows.ps1` instead;
see [Windows](#windows) above.

```bash
desktop-dev        # Tauri app with hot reload (Vite on :1430)
desktop-build      # build Mold.app, a Linux AppImage, or the native NixOS package
desktop-release    # signed + notarized + stapled app and DMG, then verify
desktop-check      # CI gate: rustfmt, clippy, vue-tsc, prettier
desktop-test       # cargo test (CPU) + vitest
desktop-ui         # frontend-only Vite server (pair with a running `serve`)
frontend-bun-lock  # regenerate the repo-root bun.lock and bun.nix
ios-dev            # iPhone app with Tauri hot reload (Vite on :1431)
ios-run            # production-mode run on an iPhone or simulator
ios-check          # Rust check for the Apple Silicon simulator target
ios-build          # archive/export for App Store Connect
android-dev        # Android app with Tauri hot reload
android-run        # production-mode run on a device or emulator
android-check      # debug ARM64 APK from the shared mobile shell
android-test       # native instrumentation tests on an emulator
android-build      # ARM64/ARMv7 app bundles for Google Play
android-emulator   # boot the external-storage Mold_API_37 emulator
android-doctor     # verify Android Studio, SDK, NDK, AVD, and cache paths
```

On Linux, `nix build .#mold-desktop` builds the sm_89 native package,
`nix build .#mold-desktop-sm86` targets RTX 3090/A40, and
`nix build .#mold-desktop-sm120` targets RTX 50-series. `CUDA_COMPUTE_CAP`
controls local dev/AppImage compilation. `desktop-release` remains the macOS
sign/notarize path and intentionally exits on Linux.

The Rust crate under `desktop/src-tauri` is its own cargo root (excluded from
the workspace); the frontend lives in `desktop/src`. CI runs the `desktop-check`
and `desktop-test` gates via `.github/workflows/desktop.yml`, which also carries
a `windows-latest` job that runs clippy, the tests, and the signing contracts,
and builds the NSIS installer after merge on `main` (or on manual/nightly runs)
— never on pull requests; PR-native feedback comes from the macOS desktop gate.

The separate remote-only iOS crate lives under `apps/mobile/src-tauri`; its
shared frontend entry is `desktop/src/mobile`. Mobile CI runs through
`.github/workflows/ios.yml` and `.github/workflows/android.yml`. See the
repository's
[`apps/mobile/README.md`](https://github.com/utensils/mold/blob/main/apps/mobile/README.md)
for native setup, simulator validation, icon guards, and TestFlight maintenance.

## Signed distribution

`desktop-release` is the release-grade local path. It requires
`.secrets/signing.env` (gitignored) with `APPLE_SIGNING_IDENTITY` and App Store
Connect API credentials (`APPLE_API_ISSUER`, `APPLE_API_KEY`, and
`APPLE_API_KEY_PATH`). The command builds the Metal-enabled app and DMG, waits
for Apple notarization, staples the ticket, then verifies the hardened-runtime
signature, entitlements, Gatekeeper acceptance, and staple on both artifacts.

CI runs the same signed distribution job from
`.github/workflows/desktop-distribution.yml`. Tagged releases publish the Stable
DMG, updater archive, signature, and `mold-desktop-stable.json`; desktop-relevant
commits on `main` publish their signed Nightly counterparts to the rolling
`latest` prerelease only after desktop CI passes. Both publication paths verify
the archived app and updater signature against the exact public key embedded in
Mold, then prove that the public manifest points at an anonymously downloadable
payload before moving the channel pointer. Nightly publication prunes only old
desktop assets after that verification and retains ten generations; unrelated
CLI assets and the current manifest target are never selected.

Repository secrets hold the exported Developer ID certificate, App Store
Connect key, and the Tauri updater credentials `TAURI_SIGNING_PRIVATE_KEY` and
`TAURI_SIGNING_PRIVATE_KEY_PASSWORD`. Never print or commit the updater private
key. Keep a controlled offline backup: losing it prevents already installed
copies from trusting future updates. Key rotation must be staged by first
shipping the replacement public key in an update signed with the existing key.
Runner-only key material is written to temporary paths, and the temporary
signing keychain is removed even if the build fails.
