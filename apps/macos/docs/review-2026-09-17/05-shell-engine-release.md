# Review 05 — Shell, Settings, Engine/FFI, Release (apps/macos)

Branch `feat/macos-native-app` @ `edc6555e`. Read-only; nothing built or run except
`scripts/tests/candle-single-identity.sh` (pure grep/awk, PASSES — see "Done well").
Settings and Release were reviewed by two Opus subagents against explicit question lists;
their findings are folded in and marked `[settings]` / `[release]`. Everything else I
verified myself on both sides.

## Answer to the lead's CHANGELOG question first

**Not a violation — local-`main` staleness.** `git rev-parse main` is `15715a9a`;
`origin/main` is `d056b05e`. Every out-of-`apps/macos` hunk in `git diff main...HEAD`
(CHANGELOG.md, the `changelog.d` deletions, the 0.29.0→0.30.0 bumps, `crates/mold-server/src/*`,
`Dockerfile`, `cuda-distribution-contract.sh`, both src-tauri roots) comes from three
commits that are already on `origin/main`: `d056b05e chore: release v0.30.0 (#1720)`,
`60eb5dc8`, `b015496e`. `git log origin/main..HEAD -- . ':!apps/macos'` returns only
`flake.nix` and `.gitignore`, from two branch commits. The CHANGELOG edit is the release
PR doing exactly what `release-plz.toml` says it does.

That staleness is, however, how I found HIGH-2.

---

## HIGH

### H1 · security · The embedded engine is keyless AND CORS-permissive — any web page can drive it
`Sources/Mold/Engine/MoldEngine.swift:68` passes `nil` for the FFI's `api_key` argument, so
`mold_engine_bootstrap` never sets `MOLD_API_KEY` (`rust/mold-macos-ffi/src/lib.rs:70-74`)
and the engine runs with `AuthState = None` — every route open. Meanwhile
`crates/mold-server/src/lib.rs:1890-1932`: with no `MOLD_CORS_ORIGIN` in the environment,
`build_cors_layer` falls through to `CorsLayer::permissive()` — `Access-Control-Allow-Origin: *`,
any method, any header. The reference implementation does not do this:
`desktop/src-tauri/src/lib.rs:158-161` mints a per-install UUID
(`desktop/src-tauri/src/secrets.rs:116`) and sets it as `MOLD_API_KEY` before starting the
same `run_server`, then sends it as `X-Api-Key` on every call.

Scenario: a user has Mold open with the local engine running. In any browser they visit a
page that loops `fetch("http://127.0.0.1:" + p + "/api/status", {mode:"cors"})` across the
ephemeral range. Permissive CORS means the page can *read* every response, not just fire
requests, so the scan succeeds in seconds and the page then has the whole API: enumerate and
download the gallery (`GET /api/gallery`, `/api/gallery/image/:f`), read prompts and
provenance, `DELETE /api/gallery/image/:f?permanent=true` to destroy prints, queue renders,
`POST /api/models/pull` to fill the disk, read `/api/config` (secrets are masked, but paths
and hostnames are not), and `POST /api/shutdown` — which `crates/mold-server/src/routes.rs:8605-8622`
explicitly permits unauthenticated *because* the caller is loopback. A local non-sandboxed
process obviously has the same access, but the browser vector is the one that needs no
malware. The random port is the only thing standing in the way and it is not a secret.

Fix: mint a key (Keychain, per install, like `secrets.rs`), pass it as the `api_key` argument
that the FFI already accepts, and put it on the `MoldHost` at `MoldEngine.swift:39`
(`apiKey:` is already a parameter). Separately, set `MOLD_CORS_ORIGIN` to something no browser
page can present (or add an embedder switch that installs a non-permissive layer) so the
loopback API is not readable cross-origin even if a key is later removed.
Confidence: high — read on both sides; not exercised at runtime.

### H2 · bug · `mold-macos-ffi` does not build on this branch (version requirement is stale)
`apps/macos/rust/mold-macos-ffi/Cargo.toml:22-28` requires `mold-ai-core`/`-db`/`-server`
at `version = "0.29.0"`. The workspace those path dependencies resolve to is now `0.30.0`
(`Cargo.toml:20`, and each crate is `version.workspace = true`, e.g.
`crates/mold-core/Cargo.toml:3`). A path dependency carrying a `version` req that the target
does not satisfy is a hard cargo resolution error, so `make engine` fails on a fresh checkout
of this branch. The crate's own `Cargo.lock:2621-2622` still pins `0.29.0`, confirming it has
not been resolved since the bump.

This is not a one-off. `scripts/release/sync-release-pr.sh:106-123` exists precisely for this
— its comment says "the path dependencies on workspace crates (`package = "mold-ai-*"`) carry
version requirements that must track the workspace version or cargo fails to select them" —
and it rewrites `desktop/src-tauri` and `apps/mobile/src-tauri`. It knows nothing about
`apps/macos/rust/mold-macos-ffi`, so **every future release PR will re-break it**. Combined
with `[release]`-R2 below (`make release` does not depend on `make engine`), the failure is
silent: the DMG builds, and the shipped app reports "This build has no local engine."

Fix: add the `apps/macos/rust/mold-macos-ffi` manifest and lock to
`scripts/release/sync-release-pr.sh` beside the two src-tauri roots, and bump this branch's
copy to `0.30.0` now. Confidence: high.

### H3 · bug · `/nix/store` linkage will ship in the DMG — nothing strips it `[release]`
`apps/macos/Engine.xcconfig:3` ends `OTHER_LDFLAGS` with `-lc++ -liconv`, resolved inside
`nix develop` via `LIBRARY_PATH`. `scripts/Mold.entitlements:21-25` *states* this ("The
embedded engine links libc++ from the Nix store") as the justification for
`disable-library-validation`. A `make release` DMG therefore carries
`LC_LOAD_DYLIB /nix/store/…/libc++.1.0.dylib`; on a user's Mac that path does not exist and
the app dies in `dyld` before `main`. The repo already solved this for Tauri —
`scripts/fix-desktop-macos-linkage.sh` rewrites exactly these two dylibs to `/usr/lib` and
**fails the build on a surviving `/nix/store` reference**. `apps/macos` has no equivalent and
no `otool -L` gate anywhere.
Fix: run that script against `Mold.app/Contents/MacOS/Mold` in `make signed` *before*
`sign-release.sh`, and hard-fail on any residual `/nix/store` ref.
Confidence: high that the hazard exists (the entitlement comment is direct evidence); a built
binary was not inspected.

### H4 · bug · `make release` can silently ship an engine-less app `[release]`
`Makefile:167-179`: `release: signed dmg notarize`, and `signed → build → gen`. Nothing in
that chain depends on `engine` or `engine-config`, and `Engine.xcconfig` is gitignored
(`.gitignore:118`). On a fresh clone the file does not exist, so `MOLD_EMBEDDED_ENGINE`
(`Sources/Mold/Engine/MoldEngine.swift:28,47,96`) is never defined and the notarized,
shipped build is a remote-only client. Fix: make `gen` depend on `engine-config`, and have
`signed` refuse unless `Engine.xcconfig` defines `MOLD_EMBEDDED_ENGINE` (or an explicit
`ALLOW_REMOTE_ONLY=1`). Confidence: high on the missing dependency.

### H5 · bug/security · A Keychain write mirrors an absent in-memory key as a delete `[settings]`
`Packages/…/HostPersistence.swift:20-26` writes the Keychain for *every* host on *every*
persist, and `Sources/Mold/Support/Keychain.swift:19-21` unconditionally `SecItemDelete`s
before adding; both OSStatus results are discarded. If `SecItemCopyMatching` returns anything
but `errSecSuccess` at load (`HostPersistence.swift:17`) — locked keychain, a re-signed dev
build whose ACL no longer matches, `errSecInteractionNotAllowed` — hosts load with
`apiKey == nil`, and the next `persist()` (adding, editing, or removing *any* machine) calls
`setAPIKey(nil, …)` for every host and permanently deletes all stored keys.
Fix: never delete from `save(_:)` — write only non-nil keys there, delete only from
`forget(_:)` and an explicit clear; check OSStatus and surface a failure.

### H6 · parity-gap · Removing a machine is unconfirmed and irreversibly destroys its API key `[settings]`
`Shell/MachinesSettings.swift:75,112-121` → `HostStore+Editing.swift:78-89` →
`HostPersistence.forget` → `SecItemDelete`. One click of a borderless icon-only "−" adjacent
to "+", no dialog, no undo. The web counterpart requires a danger confirm naming exactly this
consequence (`web/src/pages/MachinesPage.vue:203-217`, `HostDetailPage.vue:780-792`). The app
already has the primitive — `Destruction`/`.destructionDialog`, used for the far less
destructive preferences reset (`Shell/GeneralSettings.swift:75,98-109`).

---

## MED

### M1 · bug · A panic or error inside `run_server` leaves `ALIVE` true forever, and the app never asks anyway
`rust/mold-macos-ffi/src/lib.rs:163-177`: `ALIVE.store(false)` runs *after*
`runtime.block_on(...)` and `drop(runtime)`. A panic anywhere in the server unwinds the
thread and skips both stores, so `mold_engine_is_alive()` reports `true` for the life of the
process, and `mold_engine_join` (line 207) then always burns its full timeout and returns
false. Worse: **`mold_engine_is_alive` is never called from Swift at all** — grep over
`Sources/` and `Tests/` finds it only in `MoldEngineBridge.h:16`. So when the engine dies for
any reason (panic, `run_server` returning `Err` on the port race at
`lib.rs:99-111`, fatal CUDA/Metal), `MoldEngine.state` stays `.running(port:)`, "This Mac"
stays in the machine list, and every pane shows connection failures against a port nothing is
listening on, with no path back — `start()` refuses because the state is not `.stopped`.
Fix: set `ALIVE` from a `Drop` guard created at the top of the thread closure so it falls to
false on any unwind, and poll `mold_engine_is_alive()` (or the engine's own `/api/status`)
from `MoldEngine` so a dead engine transitions to `.failed`.

### M2 · bug · After a failure, "Start Engine" is enabled but does nothing
`Engine/LocalEngineSettings.swift:89-91` shows the Start button for `case .stopped, .failed`.
`Engine/MoldEngine.swift:48` is `guard case .stopped = state else { return }`. So every
recoverable failure the code itself raises — "Mold's home at … isn't available. Reconnect its
drive" (`MoldHome.unavailableReason`), "No free port on this Mac.", "The engine couldn't
start." — leaves an enabled button that is inert. The user reconnects the drive, presses
Start, and nothing happens, with no message. Fix: accept `.failed` in `start()`'s guard (the
bootstrap is already idempotent via `BOOTSTRAPPED`, `lib.rs:58-60`).

### M3 · bug/race · `.running` is published before the engine is listening
`MoldEngine.swift:80-87` sets `.running(port:)` as soon as `mold_engine_start` returns 0 — and
that returns the instant the *thread is spawned* (`lib.rs:180-186`), before the tokio runtime
exists, before DB migration, gallery-authority recovery, artifact-fact warming or the TCP
bind. `LocalEngineSettings.start()`'s poll (line 114-122) waits only for `engine.host`, which
exists the moment the state flips, so `hosts.adoptLocalEngine(host)` fires immediately and
every store begins polling a closed port. On a cold `MOLD_HOME` with a large gallery that
window is seconds, and the user sees the machine they just started reported as down.
Fix: probe `GET /api/status` (or `/health`) on the chosen port before publishing `.running`,
with a bounded retry; that also converts the documented alloc-port race into a detectable
failure rather than a silent one.

### M4 · bug · `setenv` is called with the app's threads already running
`lib.rs:62-74` sets `MOLD_HOME`/`MOLD_API_KEY` under a comment that says "`setenv` is not safe
once other threads are running" — but `mold_engine_bootstrap` is invoked from
`Task.detached` (`MoldEngine.swift:62-70`) in a fully launched SwiftUI app: main thread, the
Swift concurrency pool, URLSession's own threads, and `HostStore` already polling remote
machines. `setenv` reallocates `environ`, and a concurrent `getenv` (CFNetwork reads proxy
variables, among others) can read freed memory. Fix: call the bootstrap from
`applicationDidFinishLaunching` before any store starts, or have the FFI pass the home and
key through as parameters to `Config` rather than through the process environment.

### M5 · bug · A corrupt home pointer silently relocates the library (Rust fails closed here; Swift does not)
`crates/mold-core/src/config.rs:1339-1362` is explicit: `read_saved_mold_dir` returns an error
for an empty or non-absolute pointer, because "every other malformed/unreadable state must
fail closed at process startup". The Swift twin,
`Packages/MoldClient/Sources/MoldClient/MoldHome.swift:73-78`, returns `nil` for exactly those
cases and falls through to `~/.mold`. Because `MoldEngine.start()` then **forces** that answer
into the environment (`lib.rs:65-69` sets `MOLD_HOME`), the Rust guard can never fire. A user
whose pointer file got truncated (a crash mid-write, a backup restore) presses Start and the
engine creates a brand-new empty home in `~/.mold`: no models, no prints, no queue — visually
identical to total data loss, while the real library sits untouched on the other drive.
Fix: mirror the Rust rule — a pointer that exists but is unreadable or non-absolute must
produce an `unavailableReason`, not a fallback.

### M6 · design · No interlock against a second engine on the same `MOLD_HOME`
`MoldEngine.start()` picks a free port and starts `run_server` with no check for an engine
already serving this home. The Tauri app deliberately does the opposite:
`desktop/src-tauri/src/commands.rs:283-311,405-411` probes the well-known address first and
refuses with "A Mold server is already running at {base_url}, but it does not accept This Mac
API key." With Mold Desktop (or `mold serve`) running, the user starts the native app's engine
and now two `run_server` processes share one home. Per CLAUDE.md the gallery writer lease is
shared by design, but queue ownership is not: two owner records under
`$MOLD_HOME/queue-owners/` means "adopt none, mint fresh, report each orphan", so the other
app's queued work is stranded. Fix: before starting, probe the home for a live writer/owner
(or the desktop app's well-known port) and refuse with the same sentence the Tauri app uses.

### M7 · design · Quit gives the engine 8 s where the server's own budget is 45 s, and the lease is not released
`Support/MoldAppDelegate.swift:43-57` answers `.terminateLater`, then `MoldEngine.stop()`
(`MoldEngine.swift:95-111`) POSTs `/api/shutdown` and calls `mold_engine_join(8_000)` —
whose result is **discarded** — before replying `true`. The server's own GPU-owner join budget
is `DEFAULT_SHUTDOWN_ABORT_SECS = 45` (`crates/mold-server/src/lib.rs:1615`), and
`gallery_authority::release_gallery_writer_leases()` is the line *after* that join
(`lib.rs:1594-1599`). So quitting during any render terminates the process mid-drain and
leaves `<output_dir>/.mold-gallery-writer.lease` behind — the exact leftover that comment says
makes `mold system gallery-authority downgrade` believe a server is publishing. (Stale leases
are tolerated per CLAUDE.md, so this is friction rather than a lockout.) The `Info.plist`
`NSSupportsSuddenTermination=false` promise the delegate's own header cites is only kept for
8 s. Fix: use the join's return value — on `false`, either keep waiting up to
`resolve_shutdown_abort_secs()` or tell the user the engine is still finishing; and have the
FFI release the writer leases on the timeout path.

### M8 · design · The engine installs a process-wide SIGTERM handler; the app then ignores `kill`
`crates/mold-server/src/lib.rs:1247-1261` installs a tokio unix `SIGTERM` handler, which
replaces `SIG_DFL` process-wide. Once the engine is started, `kill <pid>` on Mold no longer
terminates the app — it begins a graceful *engine* shutdown while AppKit carries on running,
and since `allow_hard_shutdown_exit()` is deliberately not called (`lib.rs:143-146`) nothing
ends the process. macOS logout/restart uses Apple events rather than SIGTERM, so this is not a
shutdown-hang in normal use, but `pkill Mold`, a script, or a crash reporter's terminate now
leaves a zombie GUI app with a half-shut-down engine. Worth stating in the README at minimum;
better, have the FFI register a handler that also ends the app.

### M9 · bug · `mold_engine_join` can block with no timeout
`lib.rs:205-219`: the deadline loop only polls `ALIVE`. If `join` is called in the window
between `mold_engine_start` returning and the spawned thread executing
`ALIVE.store(true)` (line 151), the loop exits immediately and falls into a bare
`handle.join()` with no bound. Narrow, but it is on the quit path, where it would hang the
app instead of the 8 s it promises. Fix: set `ALIVE` in `mold_engine_start` before spawning,
and give the final `join` its own bound.

### M10 · quality/bug · The devshell's `macos-dev` kills the user's Tauri Mold app
`flake.nix` (new `macos-dev` and `macos-uat` commands) run `pkill -x Mold`. Tauri's
`productName` is also `"Mold"` (`desktop/src-tauri/tauri.conf.json:3`) and the native app's
`PRODUCT_NAME` is `Mold`, so the binaries are indistinguishable to `pkill -x`. A developer
running `macos-dev` SIGTERMs Mold Desktop — and with it an embedded engine that may be
mid-render. Fix: match on the bundle path (`pkill -f 'apps/macos/build/.*/Mold.app'`) instead.

### M11 · parity-gap · Curated Settings panes let you edit env-locked rows `[settings]`
`ConfigValueField.resolve` (`Settings/ConfigValueField.swift:33-36`) maps `entry.isEnvOwned`
to a read-only row naming the variable; `SettingRow.editor` (`Settings/SettingRow.swift:73-80`)
never consults it. 12 of the 24 curated keys carry an env var. On a host started with
`MOLD_EXPAND_TEMPERATURE=0.7`, Settings ▸ Expansion draws a live stepper; dragging it PUTs,
the server 403s `ENV_OVERRIDDEN` (`crates/mold-server/src/routes_config.rs:195-201`), and the
control snaps back. Fix: give `SettingRow` the same `isEnvOwned` branch.

### M12 · design · Return on an empty secret field silently clears a stored credential `[settings]`
`Settings/ConfigValueField.swift:104-107` skips only an *unchanged blur*; a Return always
commits, and `ConfigEntry.scalar(from:"",editor:.secret)` is `.null`
(`ConfigEntry+Editing.swift:69-74`) — "unset" to the server. A secret field always renders
empty (the `"<set>"` mask is stripped), so nothing distinguishes "nothing typed" from "clear
it". Tab into `runpod.api_key` in the Advanced table, hesitate, press Return: the credential
is gone. One pane over, `Shell/AccountsRow.swift:36` refuses exactly this and offers an
explicit Clear. Fix: make an empty commit on a `.secret` row a no-op.

### M13 · bug · "Reset These Preferences" does not keep its own promise `[settings]`
`Settings/PreferencesReset.swift:18-35` lists 16 keys and documents four deliberate
exclusions. Four persisted keys are in neither list: `libraryScope`, `libraryEdge`
(`Library/LibraryNavigation.swift:25-26`), `generateMachine`
(`Generate/GenerateController+Machine.swift:16`), `defaultMachine`
(`HostStore+Default.swift:16`). The button says it resets "every pane's own sort and scope …
and the remembered machine"; after it, the Library is still scoped to a collection and
Generate still targets the last machine. `SettingsPanesTests.swift:59-75` only asserts the
listed keys are removed, so it cannot catch an omission — pin the full set.

### M14 · security · Two hardened-runtime exemptions are broader than their stated reason `[release]`
`scripts/Mold.entitlements:14-27` justifies `allow-jit` (Metal shader compilation) but grants
`allow-unsigned-executable-memory` with no separate justification — strictly broader, removing
the `MAP_JIT` requirement process-wide, which is the mitigation `allow-jit` exists to preserve.
`disable-library-validation` then lets any unsigned dylib load into a process holding both,
and its stated reason is H3's linkage bug. Sandbox-off, `network.client` and `network.server`
are genuinely needed. Fix: drop `allow-unsigned-executable-memory`, test that candle's Metal
path still works; drop `disable-library-validation` once H3 is fixed.

### M15 · security · `NSAllowsArbitraryLoads` disables ATS globally `[release]`
`Sources/Mold/Resources/Info.plist:31-35`. Every `http://` URL the app fetches is permitted,
including a WAN host typed into Machines. `NSAllowsLocalNetworking` alone would not cover
Tailscale `100.64/10`, so the blanket switch is understandable — but the narrow form plus an
`NSExceptionDomains` entry for `ts.net`/the CGNAT range is the correct shape.

### M16 · design · arm64-only, macOS 26.0 minimum, and no auto-update `[release]`
`project.yml:6-14` sets `ARCHS: arm64` and `deploymentTarget macOS 26.0`; Tauri sets no
`minimumSystemVersion` (default 10.13) and `targets: "all"`. There is no Sparkle, no feed, no
updater (grep across `Sources/`, `project.yml`, `Makefile`), where the Tauri app ships a
minisign-signed `tauri-plugin-updater` with a channel selector. As a *replacement* for
`desktop/` this drops every Intel Mac and every Mac not on Tahoe, and removes any way to push
a security fix. Both are defensible decisions; neither is stated in the README.

### M17 · bug · `ENGINE_TARGET` hardcodes one machine's external volume `[release]`
`Makefile:35`: `ENGINE_TARGET ?= /Volumes/ExternalStorage/cargo-targets/mold-macos-ffi`. On any
Mac without that volume `make engine` fails, `engine-config` writes the remote-only xcconfig,
and H4 turns that into a shipped engine-less app. Default to `$(CURDIR)/rust/target`.

---

## LOW

- **L1 · test-gap · `apps/macos` is in no CI workflow.** `grep -rn 'apps/macos' .github/`
  returns nothing across all 15 workflows: no build, no `make test` (a real XCTest bundle,
  including the Rust-parsing config-key contract), no `make lint` (five architecture rules),
  no signing. `desktop/` has `desktop.yml` plus `desktop-distribution.yml`. A drift in
  `crates/mold-core/src/config_keys.rs` ships unnoticed. `[release]`/`[settings]`
- **L2 · quality/security · Signing applies the app's entitlements to nested code** —
  `scripts/sign-release.sh:20-32` passes `--entitlements Mold.entitlements` unconditionally,
  including to every nested `*.framework`/`*.dylib`/`*.app`, which is the `--deep` flaw the
  script's own header criticises. The loop also misses `*.bundle`/`*.xpc`/`*.appex`, and the
  verify is `--strict` without `--deep` plus no `spctl` assessment. Today probably a no-op
  (SwiftPM deps link statically). `set -euo pipefail` and quoting are correct throughout. `[release]`
- **L3 · quality · Notarization result is never inspected** — `scripts/notarize-release.sh:12`
  runs `notarytool submit --wait` with no JSON status check and no `notarytool log` on
  failure. `stapler` does fail without a ticket, so a bad DMG is not shipped; the gap is
  diagnosability. Credentials are handled correctly (keychain profile, nothing in `argv`). `[release]`
- **L4 · quality · Release version is a hardcoded, duplicated `0.1.0`** — `Makefile:163` and
  `project.yml:63`, while the repo is at 0.30.0 and releases are automated. `make release`
  produces `Mold-0.1.0.dmg` forever. `create-dmg.sh` also never signs the disk image. `[release]`
- **L5 · quality · An emptied machine list is re-seeded from `MOLD_NATIVE_HOSTS`** —
  `HostPersistence.load` returns `nil` for a decoded-empty array (`:15`), and `seededHosts()`
  treats `nil` as "never saved" (`HostStore.swift:73-88`) — the resurrection its comment says
  it prevents. A `try?` decode failure falls through the same way, then the next `persist()`
  overwrites the stored list while orphaned Keychain items remain. `[settings]`
- **L6 · quality · `nan`/`inf` in a number field reports the whole *machine* as failed** —
  `ConfigEntry.scalar` uses `Double(text)`, `JSONEncoder` throws, and
  `ConfigStore.swift:115-116`'s `default:` arm raises a fleet-wide host failure for a typo in
  one row. `[settings]`
- **L7 · quality · `keyboardShortcut` on a `Menu`** — `Shell/MoldCommands.swift:43` attaches
  ⇧⌘E to the `Menu("Export…")` container. A shortcut on a submenu is not honoured by AppKit's
  menu key-equivalent matching, so ⇧⌘E likely does nothing while the menu advertises it.
  Moderate confidence — not verified at runtime; worth one manual check.
- **L8 · quality · Repo-wide `.gitignore` patterns for one app** — the new block adds
  `*.xcodeproj`, `.build/`, `.swiftpm/`, `DerivedData/`, `Engine.xcconfig` with no path
  prefix, unlike the three sibling entries that do scope to `apps/macos/`.
- **L9 · unverified · `NSBonjourServices` absent while the FFI enables `mdns`** —
  `rust/mold-macos-ffi/Cargo.toml:26` turns on `mold-server/mdns`;
  `Info.plist` has `NSLocalNetworkUsageDescription` but no `NSBonjourServices`. No Swift source
  uses `NWBrowser`/`NetService`, and the Rust side is very likely raw multicast UDP (gated by
  the local-network permission, not the Bonjour allowlist), so this may be a non-issue.
  **Not verified** — worth one runtime check on a clean macOS 26 install that advertise and
  discover actually work from the signed bundle. `[release]`

---

## Done notably well

1. **The candle single-identity contract already covers the new cargo root.** I ran
   `scripts/tests/candle-single-identity.sh`: it discovers roots from `git ls-files` rather
   than a list, so `apps/macos/rust/mold-macos-ffi/Cargo.lock` was audited automatically and
   PASSES — 5 candle packages, all on `bf2cd29a`. The `[patch.crates-io]` block at
   `Cargo.toml:38-42` mirrors the workspace's, with the reason written down. This is exactly
   the failure mode #1393/#1399 cost four `main` merges, and it was got right first time.
2. **`MoldHome` is a faithful twin of `Config::mold_dir`.** The resolution order, the
   `MOLD_HOME_POINTER_PATH` override, and the `dirs::config_dir()` ≠ `~/.config` trap are all
   handled correctly (`MoldHome.swift:37-78` vs `crates/mold-core/src/config.rs:1312-1333`),
   and `install_config_post_load_hook()` is called by the embedder exactly as every `main()`
   must (`lib.rs:83`). Only the fail-closed half diverges (M5).
3. **The FFI's scope discipline.** Refusing to marshal `GenerateRequest` across C and talking
   to the in-process engine over the same HTTP used for a remote machine is the right call,
   and the `bound_http_drain`-not-`allow_hard_shutdown_exit` choice (`lib.rs:143-146`) is the
   correct reading of the embedded contract, with the reasoning recorded.
4. **The config-key contract tests genuinely parse the Rust registry** rather than comparing
   two Swift constants, including regex-parsing the `parse_u16|u32|f64(raw, MIN, MAX, key)`
   bounds; all 24 curated keys and both enum lists agree with `config_keys.rs`. And
   `theMaskedKeysAreTheOnlyTwoTheEngineMasks` fails the build if a *third* masked key ever
   appears upstream. `[settings]`
5. **The snake_case decoder trap is not merely avoided but pinned.**
   `MoldJSON.localEncoder/localDecoder` are strategy-free, `StoredHost` spells `base_url` out,
   and `StoredHostTests.swift:57-63` asserts the two wire strategies are *not* inverses, with a
   byte fixture of what is already on disk. Also: the bundle identifier collision was
   deliberately avoided — `io.utensils.mold.native` vs Tauri's `com.utensils.mold`, reason in a
   comment at `project.yml:59-61`, and neither app registers a `mold://` scheme. `[settings]`/`[release]`
