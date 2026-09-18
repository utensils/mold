# Mold for macOS (native)

An experimental native Swift app for mold, on the long-running
`feat/macos-native-app` branch. **This branch is never merged.**

It is a candidate replacement for the Tauri `desktop/` app on macOS, scoped to
generation and the library. No 3-D studio.

## What works today

| | |
| --- | --- |
| **Generate** | Every control comes from the model's own generation profile, so a model added to mold tomorrow gets correct controls with no change here. Stills and clips (length in seconds, snapped to the family's frame grid), source images with strength, ordered reference images, batches, negative prompts. Durable submission, live step progress and denoise preview, then what it made with Save / Copy / Show in Library -- a picture, a clip that plays in place, or a mesh you can turn, zoom and outline in place; anything that cannot be shown says so in a sentence rather than spinning. Length stops where the machine's own admission does: LTX-2's ceiling follows the chosen rate, and a text-to-video Wan tier stops at the clip it can render in one pass and says why. Clicking the picture tucks the controls off the bottom edge, leaving a lip that still carries the step marks; clicking it again or pressing Escape brings them back. An Expand button under the prompt rewrites it or, from its menu or with ⌥, suggests other ways to say it, in place, with the original kept; a click picks a suggestion and Use, Return or a double-click takes it. The capsule's own controls are the machine (Auto follows the default machine, or the first that's up; any machine that's up and generates can be pinned, and the model follows it), the shape as aspect and size from the recipe's own ladder, steps, guidance, seed and batch, wrapping as they need; under them a last row of their own carries the estimate at its leading edge and Generate at its trailing edge, where it stays whatever appears beside it. Every picture well is the same selector: the source still, each reference, the ControlNet still and every identity photograph take a file, a print dragged out of the Library, a picture chosen from the Library in a sheet, or a paste, through one menu that a click and a right-click both open -- and a staged picture is replaced from either door, in place. Each well is captioned with what it is for, so two empty squares side by side are never a guess, and a well the recipe has parked says "(not used)" in words rather than only fading. Which wells are drawn is the recipe's own `source_relation` -- SD1.5 and SDXL combine an image prompt WITH img2img, so both wells are live at once, while FLUX.2 [klein] keeps both and parks whichever the last drop did not claim. A picture the machine cannot read (HEIC, the format every iPhone syncs; WebP, which the identity encoder does not read) is converted here rather than refused after the upload, whichever door it came through. Generate never turns into Stop: a second press while a render runs queues another batch on the machine, the pane follows them in turn, and Stop stops the one on screen or, from its menu, everything this pane queued. The inspector holds the format, an upscaler, whether it is saved at all, what to file it under, and the prompts this machine was last asked for. Batch N is N variations of one idea, not N copies. The inspector also holds the adapters a model can take, a face to keep, a mask to repaint through, a ControlNet, and everything a clip is made of -- each appearing only when the chosen model says it reads that thing, and each parked, never lost, when you switch to one that does not. |
| **Library** | Every machine's prints in one day-sectioned timeline, host-badged. Select with the mouse or the keyboard, open in place, play video, favourite, tag, trash, restore, save, copy, drag to the Finder, and export a clip or mesh into whatever the host will convert it to. The sidebar's **Library** group is the way in, the way Photos and Music do it: All Prints, Favourites, every collection and Recently Deleted are rows, each opening the library on that shelf, with no duplicate Library row above them -- exactly one row in the whole sidebar is ever highlighted, and while the library is showing it is the shelf you are on. Collections are merged across the fleet by slug, and you file prints by dragging onto one. Search with real tokens -- type `is:video`, `is:mesh`, `tag:name` or `on:machine` and Return turns it into a chip; a machine, a tag and a kind are offered as you type -- sort, and set the tile size. Recently Deleted carries each print's own countdown, Put Back and Delete Immediately. Name a print, tag it, file it, and rename or delete a tag across every machine at once. Favourite, tag, filing and renaming are all **undoable** from the Edit menu. Space is Quick Look (it stands down while a field has the caret), and every print can be shared, saved or dragged out -- saving several into one folder never writes over what is already there. Every print action is declared once, so a tile's right-click menu and the Library menu offer the same things, in the same order, by the same names. The grid draws exactly the order Sort By asked for, and the viewer's ← → walk that same list; an order days cannot describe is drawn in one piece rather than under a day heading. A mesh opens into an interactive 3-D view: drag or use the arrow keys to turn it, scroll or pinch to zoom, `0` or a double-click to go back to the first view, and a control to outline it. It opens on exactly the camera the machine's own poster used, turns slowly on its own until you touch it (never under Reduce Motion), and if it cannot be drawn at all -- no Metal device, a file that will not read -- it falls back to that poster with one line saying why. Space previews a mesh by its poster, because macOS has no Quick Look generator for a GLB. Export offers what the machine itself advertises: OBJ, STL or PLY with the print size, up axis and origin a slicer wants, and one Turntable... that renders an animated spin. File ▸ Import to adds a picture, clip or mesh from this Mac to a machine. **Use These Settings** seeds Generate with a print's whole recipe on the machine that made it, and asks that machine for the print's own retained conditioning media: the source picture goes back into the well, where you can see it and set Strength, and whatever a well cannot hold -- a mask, an identity photograph, audio -- rides along at submit, under a one-use session on the same machine or relayed as bytes to another; when the machine no longer has it, one line says so. The inspector (⌥⌘I) is about whatever you are looking at: with a print open it describes that print and follows ← → to the next, and the menu bar's print actions act on it too. It shows everything the print recorded -- the prompt, the model and machine, the sampler, what conditioned it (the source picture, references, identity photographs, a ControlNet, adapters), a clip's rate and pipeline, a mesh's grid and faces, which 3-D workflow a stage came out of, how many clips a sequence was made of, and the file itself -- with a heading appearing only when there is something under it, never a column of dashes, and every row selectable and copyable from its own contextual menu. Refreshes by ETag, and follows each machine's live event stream — a print favourited, tagged or trashed somewhere else appears here without a refresh. |
| **Queue** | Every machine's work, live from its event stream: a batch as one row with its children beneath, drag or Move Up/Down to reorder where the machine will actually put the job, Empty Queue for what is waiting (anything rendering keeps going), and a held row that asks in words -- Pull the missing model then Retry, or the machine's own sentence and Try Again where it says trying again would help. Move to… sends a held job to another machine in the three calls the web app makes, idempotently. Every row, held ones included, has the same ✕ at its trailing edge and Cancel Job in its contextual menu and the Queue menu (⌘⌫), because a held row is cleared with the same DELETE that cancels a waiting one — except a row that is already RENDERING on a machine that cannot stop work at a safe point, where there is nothing to press rather than a button that fails. Pause and Resume are offered on a waiting or paused row, and only where the machine says it can pause one job. A row's contextual menu carries exactly the Queue menu's items, in its words and its order, with Cancel Job last behind a divider. A dropped stream reconciles itself: every reconnect asks each machine again rather than trusting the deltas it missed, waking the Mac reconnects the fleet on the spot, a machine that was off when Mold launched is probed again until it joins, and a machine too old to stream at all is read every ten seconds while Mold is the active app. The Dock icon counts prints that landed while Mold was in the background, and a finished render or a failed job can notify you. |
| **Models** | Installed is a table grouped by family -- model, variant, the manifest's plain-English trade-off, size, state -- listing every installed model on the machine, with the machine's own disk figure underneath. Discover searches the catalog through the machine: family and sort come from what it advertises, a row installs, reads Installed, or offers its page when the machine cannot take it. Install, repair, cancel, load, unload, components and delete from the row's contextual menu, the Model menu or the keyboard — one list, both menus, Delete… last behind a divider. A Discover row's own menu carries Details… (which is also its double-click), Install, and Open Page where the machine cannot take it. Downloads are in a toolbar popover, each live row offering Cancel Download; a gated model's licence is rendered from the machine's own payload and accepted in place. Settings ▸ Accounts holds each machine's catalog tokens. |
| **Machines** | **The fleet first**: one card per machine in a responsive grid -- status dot, version, a Default badge, This Mac marked as such, its GPUs said the way a person says them ("4× NVIDIA L40S") with one load figure and a video-memory bar, system memory, what is queued and installed there, and its address. A machine that is not answering keeps its place, dimmed, and says why in the app's own words; a figure nothing has reported is simply not drawn. Add a Machine… opens the same sheet Settings does, Nearby lists what the fleet can see that is not in it yet, and a card's contextual menu is Open, Check Now, Set as Default, Copy Address, Edit… and Remove… -- the same list the Machine menu carries, with ⇧⌘R and ⌘⌫. Arrow keys move between cards, Return opens one. Clicking a card opens **that machine's page**: its GPUs with what each is holding and how much memory is gone, a switch per card where the machine's scheduler will honour one, live memory and CPU, what is queued and installed there, and its address. Machines on the local network that this one can see are offered to add. The machine picked here is the one the Models pane shows. On a keyed machine this app already holds an operator key for, its page also lists **paired phones** -- name, when paired, when last used, Revoke -- and Pair a Phone… opens a sheet with a QR code and a `mold://pair` link that expires in under two minutes. The section is absent, not disabled, on a keyless machine, because there is nothing there to pair or revoke. Every row here right-clicks to what it already draws: a machine to Check Now and Set as Default (the Machine menu's own items and words) plus Show in Library, a card to the one switch its machine will honour, a paired phone to Revoke…, a nearby machine to Add. |
| **Settings** | Nine tabs. **General** is this Mac's own preferences -- Appearance (System, Light or Dark, applied to every window; the colours are always the system's), Dock badge and notifications while Mold is in the background, the media cache and Empty Now (absorbed from the old Storage tab), Reset These Preferences. **Generation** and **Expansion** are the render defaults and the eight `expand.*` keys, each pinned to the engine's own `config_keys.rs` registry by a test that parses the Rust, so a bound that moves there is caught here. **Library** and **Performance** hold the trash/authority-log/held-retention and scheduler/port keys. **Accounts** and **Machines** are unchanged from M5 and M2 -- catalog tokens, and add/edit/remove. **This Mac** is unchanged, plus one sentence on why its engine is reached with a key only this app holds and is not something a phone pairs with. **Advanced** is the escape hatch: every row `GET /api/config` returns for the picked machine, edited from its own value's JSON type (the wire declares no schema at all), a source badge telling env from db from file, a search field, and Reset wherever the source is `db`. Advanced is also the *only* way to reach `logging.*`, `runpod.*`, `lambda.*`, `models_dir`, `output_dir`, and the sixteen per-model `models.<name>.<field>` rows -- nothing curated repeats them. Adding a machine still normalizes its address the way the other apps do, checks it live while you type, refuses an address another machine already answers at, and puts its key in the owner-only secrets file, never a plist. Removing a machine asks first and says that its key goes with it. |
| **This Mac** | mold's own Rust engine, running in-process on Metal. It joins the machine list like any other and is reached over the same HTTP. |

Shortcuts: ⌘1–⌘5 for the destinations (⌘2 and View ▸ Library open the library
on the shelf you left it on), ⌘R to refresh, ⌘↩ to generate, ⌘, for
Settings, ⌥⌘I for the inspector (on Generate and on Library, each remembering its own), ⌥-click Expand to remix, ⌘[ and ⌘] for the brush and ⌘Z inside the mask editor, ⌃⌘S to hide or show the sidebar, ⌥⌘F to
favourite, ⌘⌫ to trash a print or cancel the selected queue row, ⌘Z to undo, Space for Quick Look, Escape to leave the
viewer. ⇧⌘R checks a selected machine right now, ⇧⌘E and ⇧⌘S export or save a
copy of the Library selection, ⌘F finds in the Library, and ⌘+ / ⌘− make the
grid's thumbnails larger or smaller. ⌘A is the grid's own, not a menu
command -- Edit ▸ Select All is the system's stock item, and this SDK never
routes it to a custom grid, so the grid answers ⌘A itself while Select All
stays present and disabled, the way Refresh does when there is nothing to
refresh.
Every shortcut is declared once in `MoldCommands` or `LibraryCommands` and only
*printed* elsewhere — binding one twice queues the work twice.

What a selection can do is in the **Library menu**, never in a bar that floats
over the grid: the menu bar is what macOS searches from Help, what the keyboard
reaches, and what VoiceOver reads.

## Where the bytes go

A print lives on the machine that made it, and Quick Look, sharing, saving and
dragging one to the Finder all need a real file here. They share one cache,
keyed on `(machine, filename, media_version)` — the same `media_version` the
server builds its ETag from, so a re-rendered poster invalidates cleanly. It
is capped (Settings ▸ General) and **emptied when Mold quits**: it exists so a
second look at a clip is free, not to be a copy of the library. Thumbnails are
a second, smaller cache and follow the same two doors -- quitting and Empty
Now -- rather than sitting on disk forever. A print larger than the whole cap
is still handed to whatever asked for it, with one line in the Library saying
Mold cannot keep a copy and where the cap lives; and eviction never takes the
file just written or one Quick Look is reading.

## The local engine

`make engine` builds `rust/mold-macos-ffi` (a staticlib around
`mold_server::run_server`) and rewrites `Engine.xcconfig` to link it. Without
it the app is a remote client and needs no Rust toolchain at all, which is the
point: UI work never costs a 40-minute build. `make engine-clean` goes back.

Five C functions, and nothing about a render crosses them — the app speaks HTTP
to loopback, exactly as it speaks to a machine on the network. Stopping is a
`POST /api/shutdown`, the only shutdown trigger an embedder can reach.

The engine starts **at most once per process** (mold's models-dir override is
a process-lifetime `OnceLock`), which is why Settings ▸ This Mac offers
Relaunch Mold rather than an inert Start after a failure that has spent it.
`run_server` installs a process-wide SIGTERM handler; the app installs its own
once the engine answers, so `kill` and `pkill` quit Mold through the same
drain the menu uses.

Its preamble runs from `MoldApp.init`, before any store exists, because it
writes `MOLD_HOME`, `MOLD_API_KEY` and `MOLD_CORS_ORIGIN` with `setenv` and
that is not safe beside a concurrent `getenv`. It runs on every launch of an
engine-linked build, whether or not you start the engine — so merely opening
the app does the same preamble every `mold` `main()` does, including the
one-shot `config.toml` → DB migration on a home that has never had one (it
renames `config.toml` to `config.toml.migrated`). It is idempotent and guarded
by a DB sentinel, so it happens once in the life of a home rather than once
per launch; it is still a write to a home you may be sharing with Mold Desktop
or `mold serve`. `.running` is published only
once `GET /api/status` answers on the chosen port — on a cold home with a big
gallery, "started" and "listening" are a long way apart — and while it runs,
its liveness is polled, so an engine that dies leaves the machine list instead
of pointing at a closed port. If another mold is already publishing into the same home — read from the
gallery writer lease, not from a port — Settings ▸ This Mac says so, naming
its pid. It is a warning and not a refusal, because mold supports two servers
on one home (`queue_journal.rs` is explicit about it, the writer lease is
shared by design, and an unadopted queue is reported as an orphan rather than
silently stranded); what was missing is that nobody was told.

Quitting gives the engine the **server's** budget — `MOLD_SHUTDOWN_ABORT_SECS`
or 45 s, plus what sits outside it — behind a small panel with a Quit Now, and
it waits for a startup or an in-flight drain as well as for a running engine.
The app used to allow 8 s and discard the answer. A drain that overruns even
that says so rather than reporting the engine stopped: its thread is still
writing, and this process can never start another.

`ENGINE_TARGET` is the cargo target directory; it defaults in-repo and
gitignored, so override it if this disk is the one you care about:
`make engine ENGINE_TARGET=/Volumes/Something/cargo-targets/mold-macos-ffi`.

## Releasing

`make signed` (needs `MOLD_SIGN_IDENTITY`), then `make dmg`, then `make
notarize` — or `make release` for all three. `signed` depends on `engine` and
refuses a bundle whose binary does not actually contain the engine, because
`Engine.xcconfig` is gitignored and a fresh clone would otherwise notarize a
remote-only client in silence; `ALLOW_REMOTE_ONLY=1` says you meant it. Before
signing, `scripts/fix-macos-native-linkage.sh` retargets the `/nix/store`
libc++ and libiconv loads the devshell link leaves behind and **fails the
release** on any that survive — that path does not exist on anyone else's Mac
and the app would die in dyld before `main`. Signing is depth-first and never
`--deep`; nested code is signed without the app's entitlements, which is what
`--deep` gets wrong. The entitlements are per configuration: Release does
**not** disable library validation (its only surviving reason was the Debug
path's own dylibs), and the two it keeps — `allow-jit` and
`allow-unsigned-executable-memory` — each record beside them the exact check
owed before they can go, which needs a signed build rendering on Metal.

**arm64 only, macOS 26.0 or newer.** The deployment target is macOS 26
(`project.yml`) and `ARCHS` is `arm64`, so every Intel Mac and every Mac not on
Tahoe is out — the Tauri app set no minimum and built for both architectures.
That one is deliberate for now, and it is in "Not built yet".

`CFBundleVersion` is `git rev-list --count HEAD`, stamped at every build by the
Makefile. Never bump it by hand: Sparkle decides which build is newer by
comparing it, and it compares **across channels** — a nightly and a stable are
the same app — so the number has to rise on its own.

CI is `.github/workflows/macos-native.yml`, on `macos-26` (the only hosted
image with the macOS 26 SDK), path-filtered to `apps/macos/**` plus the two
Rust files the contract tests parse. It runs `make lint` and `make test`
against a **remote-only** build, so the `#if MOLD_EMBEDDED_ENGINE` arm and the
Rust crate are not compiled there; `make engine` and `cargo test` in
`rust/mold-macos-ffi` remain a local gate.

## Updates

Sparkle 2, pinned to an exact version in `project.yml` — the one dependency
that can replace the whole app on someone's Mac, so which revision does that is
a reviewed decision rather than whatever resolved on a release runner. The app
is **not sandboxed**, so none of Sparkle's sandboxing requirements apply: no
`SUEnableInstallerLauncherService`, and no
`com.apple.security.temporary-exception.mach-lookup.global-name` pair — those
exist so a sandboxed app can reach its own installer, and adding them would be
an exception for a sandbox this app does not have.

Two channels, two feeds, both on GitHub Releases and nowhere else:

| Channel | Feed |
| --- | --- |
| Stable (default) | `https://github.com/utensils/mold/releases/latest/download/mold-native-appcast.xml` |
| Nightly | `https://github.com/utensils/mold/releases/download/latest/mold-native-appcast-nightly.xml` |

Two feeds rather than Sparkle's channel tagging inside one, because nightly is
a separate, far busier stream a stable user must never be offered by accident —
the same shape the Tauri app publishes. The stored channel is read by
`UpdateFeedDelegate` at **every** check, so the picker takes effect on the next
one with nothing persisted on Sparkle's side, and an unrecognised stored value
falls back to Stable alone. Both URLs are pinned by `UpdaterTests`, and the
stable one again, in the built bundle, by `scripts/assert-sparkle-key.sh`.

**Mold ▸ Check for Updates…** sits directly under About Mold, and
**Settings ▸ General ▸ Updates** has the channel picker, the two automatic
toggles (which read and write Sparkle's own settings, never shadow copies —
the scheduler reads those) and the last-checked date.

**A dev build never offers to replace itself.** `UpdaterActivation` is the one
gate: no updater in a Debug build, none under `MOLD_NATIVE_FRESH`, none in the
unit-test host. The menu item and the Settings group are then **absent**, not
greyed out.

### The one-time owner step (done 2026-09-17)

The EdDSA signing key is not in this repository and cannot be generated by
anyone but the owner. The pair exists: the public half is the
`MOLD_SPARKLE_PUBLIC_ED_KEY` in `project.yml`, the private half is the
`MOLD_NATIVE_SPARKLE_KEY` repository secret (Keychain account `mold-native` on
the owner's Mac). If it ever has to be re-made, a Release **fails to build**
until the new public half is committed, rather than shipping an updater that
can never verify anything; Debug is unaffected.

The tools live beside the XCFramework SwiftPM resolved, so `make gen && make
build` once is what puts them there:

```bash
cd apps/macos
KEYS=build/DerivedData/SourcePackages/artifacts/sparkle/Sparkle/bin/generate_keys

# 1. Generate the pair. The private half goes into your login Keychain and
#    the PUBLIC half is printed. (If a key already exists it is reused, not
#    overwritten.) Approve the Keychain prompt.
"$KEYS"

# 2. Export the private half just long enough to paste it into the repository
#    secret MOLD_NATIVE_SPARKLE_KEY (Settings ▸ Secrets and variables ▸
#    Actions), then destroy the file. `-x` is the only way out of the
#    Keychain; it writes a file, so the file is the thing to be careful with.
out=$(mktemp -d)/sparkle.key
"$KEYS" -x "$out"
pbcopy < "$out"          # paste into the secret, then:
rm -P "$out"
```

Paste the **public** half — step 1's output, or `"$KEYS" -p` at any time — into
`apps/macos/project.yml` as `MOLD_SPARKLE_PUBLIC_ED_KEY`. A public key is not a secret; it ships
inside every bundle, so one committed copy is correct.

### Publishing a build

`make appcast APPCAST_CHANNEL=stable|nightly` generates and signs one channel's
feed with Sparkle's own `generate_appcast`, taken from beside the XCFramework
SwiftPM resolved so the signer and the framework cannot drift. The private key
arrives in `MOLD_NATIVE_SPARKLE_KEY` and reaches the tool on **stdin**, never in
argv where every `ps` on the machine could read it, and never on disk.

CI does it for you. **The native app ships alongside the Tauri one, on the
channels that already exist — it never creates a release of its own.**

| | Built by | Published to | By |
| --- | --- | --- | --- |
| Stable | `macos-native-distribution.yml` (`channel: stable`) | the `v*` release release-plz cuts | `release.yml`'s `build-macos-native-dmg`, beside `build-desktop-dmg` |
| Nightly | the same, `channel: nightly` | the rolling `latest` prerelease | `macos-native.yml`'s `publish-macos-native-nightly` |

That is why `releases/latest/download/mold-native-appcast.xml` is the right
stable URL: release-plz owns the repository's Latest pointer, and this app
never touches it. Publishing to a tag of its own would have **taken** Latest
away from the CLI download links, the website, the Android APK and the Tauri
app's own stable updater — and then given it back at the next release, 404ing
the native feed. It could only ever be correct while breaking everything else.

The nightly publish follows the desktop nightly's order step for step
(`desktop.yml:598-726`): the immutable version-unique DMG first, proved
anonymously by SHA-256, a re-check that `main` has not moved, **then** the
appcast pointer, proved anonymously too, and only then a prune of this app's
own nightly DMGs — never the Tauri app's, on the release the two share. It
sits in the same concurrency group as the desktop publisher, because both
clobber assets on that one release.

`Mold-native-<version>.dmg` carries `native` in its name for the same reason:
three Molds on one release page is only confusing if they are not saying which.
The artifact name and the bundle's own `CFBundleShortVersionString` both come
from ONE variable — `MARKETING_VERSION`, exported by the workflow — so a
nightly's DMG and the version Sparkle shows in its update alert always agree.
`scripts/tests/release-names.sh` asks `make` what it would produce and greps
the workflow for the same expression, because they once disagreed and nightly
could not build at all.

## Not built yet

Chain jobs (scripted sequences are CLI and API only by design), the 3-D
studio, and large reference uploads -- mold's upload-session protocol is for
MiniMax H3 and 3-D meshes, neither of which this app makes, so reference
pictures always travel inline.

No Intel and nothing below macOS 26 — see "Releasing". (The updater is built:
see "Updates" -- the key is in place; what it still needs is a first run of
the publish workflow.)

The app is never a *claimant*. It can issue a pairing for a keyed machine it
already holds an operator key for (Machines ▸ that machine ▸ Pair a Phone…),
but nothing here scans a code or asks to be paired itself. **This Mac's own
engine is not paired either**, for a different reason: it binds `127.0.0.1`
and a phone cannot reach loopback on this Mac, so a pairing issued there
would be a credential nothing could use.

It is not keyless, though. The engine is started with an API key this app
mints once and keeps in the same owner-only `secrets.json` as every machine's
key (`local-engine-api-key`; `MOLD_API_KEY` in the environment overrides it),
and "This Mac" presents that key back on every call. Loopback is not a
boundary a browser respects: with no key, any page you opened could scan the
ephemeral port range and then read the gallery, delete prints, queue pulls and
`POST /api/shutdown`. For the same reason the engine is started with
`MOLD_CORS_ORIGIN` set to a value that is not a serialized origin at all, so
no page is ever handed a usable `Access-Control-Allow-Origin` — without it the
server falls back to `CorsLayer::permissive()`, which is `*`.

## Eight things about the wire that the docs do not say

The first two were found by reading frames off a live host, and both fail
silently; the third is a rule with two halves; the fourth is a refusal; the
fifth is an absence that means yes; the sixth is three small traps in one; the
seventh is about the queue; the eighth is that nothing on the wire says what
a setting is.

`GET /api/events` opens with `event: authority` and then sends **everything
else** as the literal `event: event`, with the real tag in the payload's
`type`. Routing on the SSE frame name finds `event` for every gallery change
and decodes none of them.

`URLSession.AsyncBytes.lines` **drops empty lines**, and in server-sent events
the blank line is the frame terminator. A parser fed by `.lines` sees `event:`
and `data:` arrive and is never told the frame ended: connected, receiving,
silent, no error anywhere. `LineAccumulator` in `MoldClient` is the answer, and
every SSE reader here goes through it.

A GPU's on/off switch is live only when **two** capability flags agree:
`devices.lifecycle` says `PATCH /api/devices/:id` exists, and
`dispatch.v2_authoritative` says the runtime answering it is the one that owns
dispatch. A legacy, observe or maintenance runtime can carry the first without
the second, and persisting a change it cannot enforce is a lie -- so the card
reads as read-only there. `DeviceControl.resolve` is that rule, tested.

A batch's size is the **length of `requests`**, never `batch_size`.
`POST /api/generation-batches` refuses any child whose `batch_size` is not 1,
so Batch N is N one-output requests sharing a prompt, a filing and a logical
batch id, differing only by seed -- the same shape the web app sends. And
per-model defaults are the eight `models.<name>.<field>` config keys and
nothing more; `model_prefs` has no route on any mold.

An absent `source_image` block on a recipe means an IMAGE family that reads
one, not a model that refuses one: the manifest omits the field for every
image family, and reading absence as "no source path" hid the source well on
every still model in the fleet and took inpainting with it. And
`GET /api/loras?model=<name>` is what decides which adapters a model can take,
so no client matches families itself.

Three things about models. "Needs repair" is `downloaded == true` AND
`remaining_download_bytes > 0` -- a model nobody has started also carries its
whole size as remaining, so the remainder alone calls every available model
broken. A licence acceptance names the terms (`id`, `url`, `sha256`), never a
bare id, and the 403 that refuses an install carries the whole payload the
accept route needs, so the flow is refuse, show, accept, retry the identical
install. And a catalog id like `hf:owner/repo` must be percent-encoded as part
of its whole path: encoded alone, Foundation turns the colon before the first
slash into `%3A` -- its guard against a leading segment that reads as a URI
scheme -- and the wildcard route never matches.

The queue. `GET /api/queue` has exactly four states -- `queued`, `running`,
`paused`, `held` -- and `accepted` belongs to the batch endpoint alone; this app
spelled the wrong one for four milestones and every waiting row decoded as
unknown. A reorder's `position` indexes the machine's `queued` rows alone, not
the row's listed `position` (which counts running rows) and not its place on
screen; a batch moves as ascending single-row calls. A held row's typed cause
lives only on the batch child, so `POST /api/generation-batches/status` -- a
read despite the verb -- is what names it, and only a missing or unknown model
is ever typed. A transfer is three calls this app makes -- export from the
source, admit on the destination, complete on the source -- with the export
kept as opaque bytes, because re-encoding it through this build's request type
would drop the media it carries.

**Nothing on the wire says what a setting is.** `GET /api/config` answers
`{key, value, source, env_var?, restart_required}` and the value is the only
type there is -- string, number, bool or null. There is no schema route, no
description, no bounds; the engine's own registry has a type it never ships
and bounds that are literal arguments at each setter. So the curated panes
carry their own knowledge and a test reads `config_keys.rs` to keep them
true, and Advanced renders whatever it is handed from the value's JSON type --
which is what makes a key newer than this build still appear and still work.
Two more traps in the same room: `runpod.api_key` and `lambda.api_key` read
back as the literal string `"<set>"`, so a field that writes back what it read
sets the key to `<set>`; and `restart_required` is true for exactly the three
`scheduler.*` keys, computed by string prefix, which means a key that DOES
need a restart (every `logging.*` one) reports false and the app does not
invent a second opinion. And a `DELETE /api/config/:key` answers
`source: "default"`, but the listing reports a key's storage SURFACE, so a
re-read after a reset says `db` again -- a listing cannot tell a stored value
from a default (`routes_config.rs:48-53`).

## Running it

From inside `nix develop`:

```bash
macos-dev      # build and run against your real settings, logs on the terminal
macos-uat      # the same build against a throwaway prefs domain and mold home
macos-test     # package tests
macos-lint     # architecture lints
macos-build    # release build
macos-gen      # regenerate Mold.xcodeproj, then open it in Xcode
```

Or directly: `make help` in this directory.

`macos-dev` is your setup: the machines you saved, and the mold home every other
mold on this Mac uses. `macos-uat` is disposable — it empties the
`io.utensils.mold.native.fresh` preferences domain and points `MOLD_HOME` at a
scratch directory under `$TMPDIR`, so a run can exercise onboarding, a first
launch and an empty library without costing you your machine list, your models
or your prints. Override the scratch home with `make uat UAT_HOME=/some/path`.
The engine started there has no models installed — that is the point.

## Which mold home it uses

The in-process engine resolves its home exactly the way `crates/mold-core`'s
`Config::mold_dir` does, and has to: it IS that engine, so if the two disagree
one Mac has two libraries. `MOLD_HOME` wins; otherwise the bootstrap pointer at
`~/Library/Application Support/mold/home` — the file Mold Desktop writes when
someone moves their home to another drive — and finally `~/.mold`. Settings ▸
This Mac prints the answer. A home named by that pointer but not currently
mounted is reported rather than recreated, because a fresh empty home in its
place is indistinguishable from having lost everything.

`Mold.xcodeproj` is **generated** from `project.yml` and is gitignored. Editing
it by hand creates a second source of truth that drifts; change `project.yml`
and re-run `make gen`.

## Pointing it at a server

The app ships with no machines. Add one in Settings: a name or an IP is enough,
because `HostAddress` fills in `http://` and port 7680 the way the Tauri app and
the browser build do — `workstation`, `10.0.0.5:7680`, `https://box.ts.net` and a
pasted `http://box:7680/api/status` all resolve to the same one origin. The
sheet checks the address while you type and names the machine after the hostname
the server reports, so a box reached by IP still lists under its own name. A
keyless host needs no API key — that is a first-class state, not a degraded one.

A key that is needed goes in `~/Library/Application Support/io.utensils.mold.native/secrets.json`,
a flat `{"name": "value"}` document this Mac's user alone can read (mode `0600`,
set before the file is moved into place). That is the Tauri app's own rule
(`.claude/rules/desktop.md`) and for its reason: the Keychain prompts on every
ad-hoc rebuild, and its statuses were being discarded, so one unreadable item at
launch used to take every stored key with it on the next save. An install that
already had Keychain items moves them into the file once, at launch, and deletes
them. Preferences hold the machine list; they have never held a credential.

To seed machines for a dev run without putting their addresses in the repo, set
`MOLD_NATIVE_HOSTS` to comma-separated `name=address` pairs, in the same
shorthand the sheet accepts:

```bash
MOLD_NATIVE_HOSTS='workstation=workstation,hal9000=10.0.0.6' macos-dev
```

`macos-dev` and `macos-uat` exec the binary rather than `open`ing it, so the
variable reaches the app and its stdout stays on your terminal.

`MOLD_NATIVE_DESTINATION` forces where the window opens: a destination name,
`settings`, `machines` to open on the fleet overview, or `add-machine` /
`edit-machine` to open the host sheet empty or on the first machine.
`MOLD_NATIVE_MACHINE=<name>` goes with `machines` and opens THAT machine's page
instead of the overview, matched on the name without regard to case; a name
matching nothing lands on the overview rather than on some other machine.
`MOLD_NATIVE_QUEUE_FIXTURE=<file>` seeds the queue from a
machine-keyed JSON fixture and refuses every mutation, and
`MOLD_NATIVE_SOURCE_IMAGE=<png>` seeds a source picture, and
`MOLD_NATIVE_LIBRARY_PICKER=1` opens the source well's From Library… sheet at
launch -- all so a run can be photographed without a generation. The sheet ones exist so a UAT run can photograph it without
a script driving the mouse across the desktop.

`MOLD_NATIVE_SETTINGS_TAB=<general|generation|expansion|library|performance|accounts|machines|thisMac|advanced>`
opens the Settings window straight to that tab; an unknown or absent id opens
the first one rather than a blank window.
`MOLD_NATIVE_PAIRING_FIXTURE=<path>` seeds the Machines pane's Pairing
section from a JSON file keyed by machine NAME --
`{"hosts": {"<name>": <PairedClients> | {"clients": …, "session"?: …, "operator_required"?: true}}}`,
the bare shape or the keyed one that also seeds an in-flight session and the
403 state -- and refuses every mutation, the same class of hook as
`MOLD_NATIVE_QUEUE_FIXTURE`.

## Layout

| Path                  | What                                                              |
| --------------------- | ----------------------------------------------------------------- |
| `Sources/Mold/`        | The app. `AppStores.swift` is the composition root; `MoldApp.swift` is the scenes and the menu bar |
| `Packages/MoldClient/` | Wire types and transport. **Never imports SwiftUI or AppKit**      |
| `Packages/MoldStyle/`  | Chrome tokens, panel surfaces, layouts                            |
| `rust/mold-macos-ffi/` | The C ABI around mold's engine. Its own cargo root                |
| `scripts/`             | Sign, DMG, notarize, appcast, and the release path's fail-closed gates |

## The rules `make lint` enforces

- `MoldClient` must not import a UI framework — it has to stay usable from
  tests and from anything that isn't this app.
- A concrete backend is built **only** in `HostStore+Reachability.swift`, so
  what the app is talking to is a decision in one file. That is what made
  running mold's engine in-process a new host in the list rather than a
  rewrite — the lint caught two attempts to construct one elsewhere while that
  was being built.
- No literal colors. The app is system light/dark only: semantic colors,
  materials, and the user's own accent. There is no palette to maintain.
- Files over 150 lines are flagged. The Tauri app's `GenerateView.vue` reached
  4,885; this is the guardrail against that.
- A **type's** size is flagged too: `lint-type-size` sums `Type.swift` and every
  `Type+Concern.swift` beside it and prints anything over 600, because slicing
  a type into files that each pass the rule above does not make it smaller.
  Today it prints ONE line, `HTTPBackend` (1,402), and that one stands on
  purpose: Swift has no conformance delegation, so composing it out of
  sub-types would cost roughly 190 forwarder lines that this same rule sums
  straight back onto the total (M1.5 S8's rule, held again at M7, and again
  in the consolidation pass below). It is the honest remaining debt rather
  than a threshold to raise.

  `GenerateController`, `LibraryActions`, `LibraryStore` and `RenderDraft`
  were over it too, and came back under by COMPOSING rather than by slicing:
  `ExpandStore` (a rewrite is a round trip with its own staleness fence, and
  a submit reads none of it), `PrintImport` (a batch with a policy about one
  unreadable file in the middle of ten), `GalleryLive` and `LibraryTags` (the
  live reconciler, and the tag index as distinct from the timeline),
  `CanvasFit` (a size and a `ResolutionProfile` -- it never touches a draft)
  and `RenderRequest` (the translation to the wire, which answers questions
  the draft holds no opinion on). Each takes the thing it serves as a
  parameter, the arrangement `LibraryMutations` already had with
  `LibraryStore`; none of them is a forwarder, and `TYPE_MAX` did not move.
  A type declared inside ANOTHER type's file is a miscount rather than debt,
  and is moved to its own (`EmptyBody`, `HostFailure`).
- No `bytes.lines` in `MoldClient`. URLSession's splitter drops empty lines,
  and an empty line is what ends an SSE frame -- the download stream was
  silent for months because of it. `moldLines()` keeps them.
- `lint-a11y` is a hard gate: every file under `Sources/Mold` with an
  `Image(systemName:` must also carry `accessibilityLabel`, `.help(`,
  `Label(`, `Label {`, `accessibilityElement`, or the opt-out comment
  `// a11y:`. It is a per-FILE floor, not the audit -- it cannot see whether
  the modifier lands on the glyph that needs it rather than a sibling (a
  tooltip on the row does not name the button inside it), so a file can pass
  the rule and still be wrong. Use the opt-out only for a glyph that is
  genuinely decorative beside its own text.

`make test` runs two bundles: the `MoldClient` package (wire types, parsers,
the outbox policy, a stub `URLProtocol` for the transport) and the app's own
`MoldTests`, which launches the app against the same throwaway home `make
uat` uses and drives the stores with a `FakeBackend` that throws on any route
a test did not plant. Every store takes `HostStore` at init, so a test hands
it a fake and nothing else changes.

Every failure a machine reports goes through one funnel, `HostStore.report`,
and shows in one place -- a dismissable line above the pane, never a modal --
so one machine failing says nothing about the machines that worked.
