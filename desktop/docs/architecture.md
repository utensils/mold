# mold Desktop and mobile — Tauri 2 Architecture

Mold ships native macOS (Apple Silicon / Metal), x86_64 Linux (CUDA), and
Windows desktop apps plus a remote-only iPhone companion and an Android
foundation (`website/guide/desktop.md` and `website/guide/android.md` are the
canonical user-facing statements of what each platform ships). The backend and
shared frontend logic stay platform-neutral; Tauri platform configs and thin native
bridges own window chrome, device capabilities, and bundle details. The Mold
Studio Mold and Safelight theme families are shared without changing
generated-media color.

Design system: [`../../docs/design/mold-studio-spec.html`](../../docs/design/mold-studio-spec.html)
— the Mold Studio interface spec these surfaces implement. Its information
architecture is five workspaces on every surface: Create, Library, Models,
Machines, and Settings.

## Android foundation

Android reuses the same standalone `apps/mobile/src-tauri` crate, the generated
project at `apps/mobile/src-tauri/gen/android`, and the complete
`desktop/src/mobile` Vue surface. It is remote-only for the same reason as iOS:
no server or inference crates enter the phone build. The first scaffold targets
SDK 36 with minimum SDK 24 and builds through `scripts/android.sh`; Android
Studio, SDK/NDK, AVD, Gradle, Cargo, and Bun caches default to external storage.

The Android-native boundary lives in `apps/mobile/plugins`. It encrypts per-host
keys with a non-exportable Android Keystore AES-GCM key, retains only ciphertext
in private preferences, and uses Tauri's Android barcode scanner for the shared
one-time pairing flow. Android NSD for `_mold._tcp`, MediaStore/share intents,
and system-bar/viewport behavior remain behind the existing command contracts.

## iOS companion

The iPhone app is a separate thin Tauri crate at `apps/mobile/src-tauri`, with
its Vue entry at `desktop/src/mobile`. It is excluded from the root workspace
and never embeds an engine: every request targets a saved remote host. The app
shares API types, explicit-target HTTP/SSE helpers, capability and form logic,
source fitting, gallery media, catalog helpers, and themes; desktop stores that
assume **This device** remain desktop-only.

Host metadata, the selected host, mobile templates, and appearance preferences
live in WebView local storage. API keys never do: native Keychain commands store
them under `com.utensils.mold.remote-api-key`. Apple DNS-SD browses
`_mold._tcp` under Local Network permission; manual entry accepts IP/DNS/HTTPS
and Tailscale MagicDNS names. Tailscale support is network/DNS interoperability,
not an embedded SDK. The only other native command synchronizes System, Light,
or Dark appearance to UIKit so status-bar glyphs match the WebView.

Navigation covers Create, Library, Models, and Machines; Settings is a pushed
header destination. Create reuses the capability matrix and request builder
for prompt tools, templates, independent batch jobs, source/edit/mask/ControlNet
inputs, LoRA, resolution/seed controls, estimates, and video/LTX-2 controls.
Native prepared expansion is intentionally view-ephemeral. `GenerateView` and
`MobileApp` each own their input snapshot, monotonic request guards, frozen
`HostRoute`, preprocessing revalidation, and stale calculation; their platform
review components own inline editing, collapse confirmation, and conditional
focus restoration. `stores/generation.ts` maps the ordered prompts plus shared
source provenance and durable batch ID/position metadata to one independently
cancellable sibling each, including auto-chained video requests. Desktop
projects the route's `useDownloadsStore` bucket; iPhone shares
`useMobileDownloadsStore` with the Models view and holds an exact selected-host lease
without initializing desktop-primary state. Returned job IDs are authoritative;
a newly observed exact-model in-flight row covers older snapshot/response timing
without treating stale history, another host, or a competing job as completion.
The iPhone recovery record deep-freezes its original inputs and route-derived
host. Its attempt lease outranks later Models registration only while that pull
is active, joins a compatible Models POST already in Starting, and releases on
every terminal, error, stale, superseded, or aborted path; Retry reacquires the
exact route. Prepared edits/removals supersede pending replacement ownership.
Per-view consumer IDs and synchronous unmount invalidation keep late
expansion, preprocessing, pull, and retry callbacks from acting on a remount.
Capability discovery is advisory
per host and never participates in fallback routing.
Desktop Settings → About opens the public Mold privacy policy through the native
external-browser opener, matching the iPhone disclosure.
Library merges every saved remote host, fetches thumbnails and full-size
stills/audio through the native HTTP client (the webview's per-host connection
pool is shared with held-open generation/download streams, which can starve
media elements pointed straight at a busy host; a refused native read falls back
to the webview route), streams video through a short-lived path-scoped ticket
with native seeking, swipes between prints, exposes explicit native Copy image / Save photo / Save video and Use as
prompt / Use as source actions, and opens generated stills in the same viewer. Persistent
New visit state mirrors desktop, while both shells derive the Upscaled badge
from output provenance. Host detail
shows telemetry, storage, queue, downloads, and installed models. Models merges
installed/live entries and routes actions per host; Pull visibly progresses
through Connecting, Starting, Queued, and Pulling percentage states with
snapshot-before-POST reconciliation and duplicate prevention.

The shell is iPhone-first with safe areas, 44pt controls, 16px editable text,
document zoom disabled, and overscroll bounce suppressed. The Library viewer
keeps a narrowly scoped horizontal swipe gesture. Settings persists the Mold
Studio families (Mold or Safelight) with System/Dark/Light, host management,
version, the TestFlight update channel, and an external-browser link to the
public Mold privacy policy. Fresh installs start with Safelight + Dark; valid
persisted theme choices remain authoritative.

`.github/workflows/ios.yml` gates mobile-relevant pull requests and `main`
changes. A successful eligible `main` run triggers
`.github/workflows/testflight-ios.yml` (there is no wall-clock cron), which
uploads an archive eligible for internal and external TestFlight groups, waits
for App Store Connect `VALID`, and verifies membership in the Mold Internal
tester group. External groups can submit that build to Beta App Review. See
`apps/mobile/README.md` for commands, signing inputs, asset guards, and the
manual verification workflow.

---

## 1. Location & workspace strategy — DECISION: `desktop/` at repo root, own cargo root, root `[workspace] exclude`

**Pick:** New top-level `desktop/` directory (the iPhone/Android crate at `apps/mobile/src-tauri` is excluded the same way). The frontend lives at `desktop/` (package.json, src/), the Rust crate at `desktop/src-tauri/` with its **own `Cargo.toml` + `Cargo.lock`** (a standalone single-crate workspace). Add to the root `Cargo.toml`:

```toml
[workspace]
members = [ ... unchanged ... ]
exclude = ["desktop/src-tauri", "apps/mobile/src-tauri"]
resolver = "2"
```

**Why (rejecting workspace membership):**

- The root workspace is MSRV 1.93 / edition 2021 and every CI gate runs `--workspace` (`cargo check/clippy/test --workspace` with `-D warnings`). Joining would drag ~400 Tauri/objc2/wry crates into `clippy --workspace`, into crane's `buildDepsOnly` artifacts (invalidating the CUDA dep cache on every Tauri bump), and into `Cargo.lock` churn for a CUDA-heavy workspace. Exclusion means **zero risk to existing CI/builds** — the only main-tree edits are the one-line exclude, flake additions, and a new branch-gated workflow.
- This is exactly the proven Aethon pattern (`cargoRoot = "src-tauri"`, separate `Cargo.lock`, `rustPlatform.buildRustPackage` + `cargo-tauri.hook`).
- Path dependencies work fine across the boundary: `desktop/src-tauri` depends on `../../crates/mold-server` etc. by path; it compiles those crates under its own lock/profile.
- **Deliberate choice: edition 2021 for the desktop crate** (Tauri 2 does not require 2024). This keeps treefmt's `rustfmt { edition = "2021" }` correct for the whole tree and lets the desktop crate build with the devshell's existing stable toolchain. (Aethon used 2024; we don't need it and it buys friction.)

## 2. Backend integration — DECISION: embed `mold-ai-server` in-process, plus first-class remote mode; **all app data flows over HTTP+SSE in both modes**

**Pick:** The Tauri process links `mold-server` (package `mold-ai-server`) with `expand` and `mdns` always on, `mp4` on every non-Windows target (a `cfg(not(windows))` dependency entry — `fdk-aac-sys` cannot build with MSVC), and the GPU backends as opt-in features of `mold-desktop` itself (`metal`, `cuda`, `nvml`, `h3`/`h3-cuda`, `pulid`; `default = []` so CI runs CPU-only) and keeps a local server online as the app's permanent primary engine. It reuses a Mold server already answering on `localhost:7680`; otherwise it spawns `mold_server::run_server("0.0.0.0", port, models_dir, GpuSelection::All, queue_size)` on a dedicated thread with its own tokio runtime. The webview still uses `http://127.0.0.1:<port>`, while other machines reach the advertised LAN address. Local and remote hosts share the same HTTP + SSE wire contract.

**Why not sidecar / external server:**

- _Sidecar `mold serve`_: doubles the shipped binary (~each contains candle + all model pipelines), needs process supervision, orphan cleanup, version skew handling, and externalBin plumbing. No benefit — mold-server is already a clean library with one entry point (`crates/mold-server/src/lib.rs::run_server`, called the same way by `mold-cli/src/commands/serve.rs`).
- _Require external server_: fails "feels like a real desktop app"; double-click must just work.
- _Pure IPC (no HTTP)_: would force reimplementing the queue, SSE fan-out, chain-job runner, download driver, gallery reconciler, and would make remote mode a second code path. Embedding the server gives **one transport for local and remote** — the single most important architectural bet in this plan. The frontend never knows which mode it's in beyond a base URL + key.

**Runtime/threading:** do _not_ run the server on Tauri's async runtime. `run_server` is a long-lived `async fn` that installs global state (`Config::install_runtime_models_dir_override`, SIGPIPE handling) and blocks until shutdown; give it its own `tokio::runtime::Runtime` on a named thread (`mold-server`). Generation work already goes through `spawn_blocking` + per-GPU workers internally. Tauri's own commands stay on `tauri::async_runtime`.

**Port selection:** prefer `0.0.0.0:7680` so the desktop server has the conventional address. If an unrelated process occupies 7680, reserve an ephemeral wildcard port and advertise that real port over mDNS. The listener probe is dropped before `run_server` binds, leaving the existing small TOCTOU race; the upstream `run_server_with_listener` follow-up remains applicable.

**Auth:** resolve `MOLD_API_KEY` as an explicit override; otherwise reuse or generate `desktop-local-api-key` in the owner-only app secrets file. Export it before spawning the server thread (`auth::load_api_keys` reads env once), advertise `auth=1`, and expose a masked reveal/copy control in the Machines workspace. The frontend attaches `X-Api-Key`; remote hosts retain their own per-host keys. CORS stays permissive (default), with CSP constraining the frontend side.

**Shutdown:** the embedded handle is owned separately from host connections; app exit or an explicit local-engine restart POSTs `/api/shutdown` and joins the thread with a 5s timeout. User-run external servers are never shut down by the app.

**Machines UI (no modes):** there is no connection switcher — the built-in/local engine is permanently the internal primary (**This device**) and every remote server is a list entry managed in the Machines workspace (This-device card, Add host, Connected, Remembered, On your network). Hosts dedupe by the server's instance UUID (`/api/status.instance_id`, mDNS `id` TXT record) with display names from the server's hostname; a one-shot Rust boot migration (`settings::migrate_remote_primary`) re-homes old remote-primary installs into the host list, carrying the API key into the per-host secret slot and pinning the generation target. Routing is generation-time only: the Host selector's Auto / Most capable / sticky pick covers every connected host. LTX-2 is performance-qualified on CUDA and Apple Metal, and correctness-only on CPU; drive the distinction from the family capability map rather than disabling the family.

## 3. Frontend stack — one shared Vue workspace

Web, desktop, and iPhone use the private repo-root Bun workspace and its single
exact `bun.lock` / generated `bun.nix`. `ui/` contains design tokens and
low-level primitives. `studio/` is browser-safe and contains current-version
wire contracts, explicit `ApiTarget` transport, platform-adapter interfaces,
Pinia state, and reusable domain logic. `web/` and `desktop/` provide routing,
navigation, native bridges, and the few presentation differences required by
the Mold Studio spec. Tauri imports and legacy normalization may not enter
`studio/`; an architecture test enforces that boundary.

- **Bundler:** exactly pinned Vite 7 and `@vitejs/plugin-vue`, with desktop dev on **port 1430**.
- **Styling:** exactly pinned Tailwind v4 with the shared Mold Studio token layer. macOS uses overlay traffic-light chrome and Linux uses native decorations.
- **State:** Pinia is the common state model. Server state is reduced explicitly from HTTP snapshots and SSE events; TanStack Query is not part of the workspace.
- **Virtualized gallery:** `@tanstack/vue-virtual ^3` — virtualized rows of a justified layout, rendered as ONE flat print-keyed absolutely positioned tile layer so a slider drag or resize moves tiles instead of remounting their media. Per-tile facts come from a `TileModel` computed once per data change; the store's `organizationIndex` / `bucketIndex` getters make every organization and copy lookup O(1). Operation-count guards (`gallery.perf.test.ts`, `LibraryView.perf.test.ts`, the scheduler drain test) fail CI if a hot path regresses.
- **Persistent thumbnail cache:** `src-tauri/src/thumbnail_cache.rs` — `<app_data>/thumbnail-cache/v1/<aa>/<sha256>.bin`, keyed `sha256(origin \0 filename \0 media_version \0 size)` where `origin` is `local` or a digest of the host's base URL (never its API key). Bounded to 512 MB / 20 000 files, LRU by mtime, magic-sniffed on read. A tile is **prepared** through `prepare_gallery_thumbnail` (cache-first; a miss fetches `/api/gallery/thumbnail/:f?size=256|512&fmt=jpeg` natively or, with the engine Off, renders the file in-process via `mold_server::thumbnails`) and **displayed** through the async `mold-thumb://localhost/<origin>/<size>/<file>?v=<version>` protocol, which only reads the cache — no bytes, blobs, or object URLs pass through JS for a tile. `LibraryView` pre-warms the tiles around the viewport (`lib/gallery/thumbnailPrewarm.ts`, capped per host) through the shared scheduler at `near` / `background` priority. Clear it by deleting the folder; it rebuilds on the next Library visit.
- **Video:** native `<video>` pointed at `GET /api/gallery/image/:filename` — the endpoint already supports HTTP Range (206), which WKWebView requires for scrubbing. Thumbnails/GIF previews from the existing endpoints. Loading `http://127.0.0.1` media from the `tauri://` origin is permitted by the CSP below.
- **SSE:** `@microsoft/fetch-event-source ^2.0.1` for **everything** — required because (a) `/api/upscale/stream` is **POST**-SSE, and (b) native `EventSource` cannot send the `X-Api-Key` header even for the GET streams (`/api/resources/stream`, `/api/downloads/stream`, `/api/chain-jobs/:id/events`, `/api/events`). One `sse.ts` helper wraps auth, abort, retry-with-snapshot semantics, and the `/api/queue` polling reconciler (zombie-card dead-lettering, same trick the SPA uses via the `Queued{id}` correlation event).
- **Connection budget (HTTP/1.1, ~6 per host):** a print holds no stream at all — it is admitted through one `POST /api/generation-batches` and reconciled over the shared `/api/events` stream — so the budget is now spent by sequences alone. An auto-chained sequence holds a POST-SSE stream for its whole run, and downloads + resources + `/api/events` each hold one more; sequence siblings stay capped at **two concurrent streams** (`runWithConcurrency` in `stores/generation.ts`). Worst case: 2 sequence streams + downloads + resources + events = 5 < 6.
- **Cross-view state:** the Create form (model, prompt, params) lives in `stores/generateForm.ts`, not the view — `<router-view>` has no KeepAlive, so views unmount on navigation and component-local state would reset. The `events` store subscribes app-wide to `/api/events` (from `App.vue` on connection ready) and keeps the gallery store live; older servers fall back to a 5 s poll while jobs are pending.
- **TypeScript:** strict, `vue-tsc ^3` in CI. Wire types hand-written in `src/lib/api/types.ts` mirroring mold-core (source of truth: `/api/openapi.json` — validate drift with a vitest snapshot test that fetches the OpenAPI doc from a dev server, optional).

## 4. Tauri specifics

**Identifier:** `com.utensils.mold` — productName `Mold`.

**`desktop/src-tauri/tauri.conf.json` (key choices):**

```jsonc
{
  "$schema": "https://schema.tauri.app/config/2",
  "productName": "Mold",
  "identifier": "com.utensils.mold",
  "build": {
    "beforeDevCommand": "bun run dev",
    "devUrl": "http://localhost:1430",
    "beforeBuildCommand": "bun run build",
    "frontendDist": "../dist",
  },
  "app": {
    "windows": [
      {
        "title": "Mold",
        "width": 1360,
        "height": 860,
        "minWidth": 1080,
        "minHeight": 700,
        "center": true,
        "maximized": true,
        "fullscreen": false,
        "visible": false, // shown after frontend mounts (no white flash)
        // titleBarStyle/hiddenTitle live in tauri.macos.conf.json
      },
    ],
    "security": {
      "csp": "default-src 'self'; img-src 'self' blob: data: http: https: mold-local: mold-thumb:; media-src 'self' blob: http: https: mold-local:; connect-src 'self' ipc: http://ipc.localhost http: https: mold-local:; style-src 'self' 'unsafe-inline'; font-src 'self' data:",
    },
  },
  "bundle": {
    "active": true,
    "targets": "all",
    "category": "GraphicsAndDesign",
    "macOS": { "minimumSystemVersion": "12.0", "entitlements": "Entitlements.plist" },
    "createUpdaterArtifacts": false, // signed distribution CI overlays true
  },
  "plugins": {
    "updater": {
      "pubkey": "<embedded Minisign public key>",
      "endpoints": [
        "https://github.com/utensils/mold/releases/latest/download/mold-desktop-stable.json",
      ],
    },
  },
}
```

The sketch above is the shared base. Platform chrome and bundling are overlaid
by `tauri.{macos,linux,windows}.conf.json`: the macOS overlay owns
`titleBarStyle: "Overlay"` / `hiddenTitle`, `minimumSystemVersion` 12.0 and
`Entitlements.plist`; Windows bundles NSIS. `tauri.windows-self-signed.conf.json`
exists for unsigned proofs.

- **Not transparent, no `macOSPrivateApi`** — vibrancy via private API risks App Store/notarization pain and fights a color-accurate image app (translucent surfaces behind generated images distort perceived color). The dark, opaque custom chrome is the design.
- `minimumSystemVersion` 12.0 (Metal path + modern WKWebView; mold is Apple-Silicon-targeted anyway).

**Plugins (all `2.x`):** `dialog` (source image/mask/keyframe/TOML pickers, save-as export), `opener` (reveal in Finder, open URLs), `notification` (generation/chain/pull complete while unfocused; fallback outside bundled macOS builds — decided from the executable path, never `NSBundle.bundleIdentifier`, which the plugin's `mac-notification-sys` fallback swizzles to a fake id and which then made the next native notification abort `tauri dev`), `single-instance` (one engine, one GPU owner), `window-state` (restore geometry), `clipboard-manager` (copy image/prompt), `process` (relaunch), and `updater` (manifest parsing, download, and mandatory Minisign verification; Mold-owned Rust commands retain installation policy). Bundled macOS notifications use `send_native_notification` so the app's `icon.icns` is passed as the notification identity image instead of being lost in Tauri's bundle-ID lookup. Because macOS still resolves the notification icon by bundle identifier through Launch Services, `relocate::maybe_offer_relocation` runs at `setup` on macOS release builds: a launch from a mounted disk image or a Gatekeeper translocation path offers to `ditto`-copy Mold into `/Applications` and relaunch, so a transient bundle never registers `com.utensils.mold` and goes stale on eject (which is what left notifications with a generic placeholder icon). Native webview zoom supplies persistent whole-app scaling. **Deferred:** deep-link (no URL scheme use case yet), `shell` (not needed — no sidecars), `fs` (not needed — file reads happen in Rust commands after dialog returns paths).

The shared image picker opens the platform-native **Choose file** dialog for source images, masks, and keyframes. Its visible filter is PNG/JPEG; the native importer decodes the selected file to enforce that boundary, while drag-and-drop and the browser development surface retain their portable fallback.

**Updater channels and preflight:** the app-local settings store persists `stable` (default) or `nightly`. Startup performs a best-effort check only; menu/Settings checks do the same, an available update appears in persistent app chrome (plus a native notification while backgrounded), and the trusted Rust `Update` remains pending until the user explicitly chooses **Update and restart**. Stable reads public `mold-desktop-stable.json` from the latest tagged release. Nightly reads public `mold-desktop-nightly.json` from the rolling `latest` prerelease, built only for desktop-relevant `main` commits after both desktop CI jobs pass. Default SemVer comparison is intentional: selecting Stable from a newer Nightly waits for a newer stable release instead of silently downgrading data or code.

Tauri downloads the complete updater archive with a 15-minute hard timeout and verifies its Minisign signature before Mold changes the app. Mold then extracts every archive entry into temporary storage, rejects unsafe paths or multiple app bundles, requires `com.utensils.mold` plus the exact manifest version, and runs strict `codesign` plus Gatekeeper assessment against the staged bundle. The same preflight validates the running bundle's identity, rejects DMG/App Translocation launches, and proves the install directory can be replaced without falling into Tauri's destructive authorization path. Only after all checks pass does Mold atomically exchange the staged and installed bundles with macOS `renameatx_np(..., RENAME_SWAP)` and request a normal restart. There is no frontend health handshake, post-launch probation, persistent transaction, or Mold-owned rollback; a preflight failure leaves the installed app untouched, while a successful replacement is final. Because the immediately preceding release still launches its candidate under the removed supervisor, `main()` recognizes only that backup-binary parent plus its one-time health-token environment and terminates it before its first poll; subsequent updates never create a supervisor.

The checked-in config deliberately leaves `createUpdaterArtifacts` false so unsigned PR/dev bundle proofs do not need release credentials. `.github/workflows/desktop-distribution.yml` overlays it to true for Stable and Nightly, derives deterministic next-patch Nightly SemVer and an Apple-safe `CFBundleVersion` from the full main-history commit count, and publishes version-unique archives before updating a channel manifest. Tauri builds, signs, notarizes, and staples the app and creates the updater archive; `scripts/create-desktop-dmg.sh` then creates and signs the drag-to-Applications image with bounded, visible `hdiutil` retries instead of Tauri's opaque writable-image mount helper. The existing workflow separately notarizes, staples, and verifies that DMG. Distribution and publication independently decode the Tauri signature and verify it with the exact public key in `tauri.conf.json`. A Nightly job exits before upload if its commit is no longer `main` HEAD; older Stable-tag reruns may refresh their own release but cannot move `releases/latest` backward. The rolling release retains ten complete desktop generations and prunes only after the new public payload and manifest pass anonymous verification. CI requires `TAURI_SIGNING_PRIVATE_KEY` and `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` in addition to Apple signing/notarization secrets. The updater key needs controlled offline backup and must never enter source or logs; losing it strands installed clients, while rotation requires an old-key-signed bridge release containing the replacement public key.

**IPC vs HTTP split (the rule):** _HTTP+SSE for anything the remote server can also answer; IPC only for what must run in the app process._

- **Trash flow (Library organization):** the desktop never owns organization state. A delete anywhere in the Library is `DELETE /api/gallery/image/:filename` on the print's origin host (the server moves the file to `<output_dir>/.trash/`, writes a tombstone, flags the row, and publishes `gallery_trashed`); the 6 s undo toast is purely client-side limbo (undo = cancel, no server call). Restore is `POST /api/gallery/trash/restore`, permanent delete is `DELETE …?permanent=true`, and Empty trash is `DELETE /api/gallery/trash` behind the plain shared `ConfirmDialog`. Retention is the host's own `gallery.trash_retention_days` config key — Settings ▸ Library edits the primary's via the settings store, Machines ▸ host ▸ Storage edits a remote's via `lib/api/hostConfig.ts`. For **this device** the singular-authority rule applies: while the lifecycle mutex says the local server runs, `local_gallery_list/trash_list/delete/restore/delete_forever` in `src-tauri/src/gallery.rs` route over authenticated HTTP; only a proven `LocalServer::Off` performs the `.trash/` move + `mold_db::trash` tombstone/row flag on disk (and `mold-local:` media resolves trashed files into `.trash/`).
- **IPC commands:** `get_connection() -> {base_url, api_key, mode}`, `start_local_engine`, `stop_local_engine`, `test_remote_host` (probe returning version/auth plus `instance_id` + `hostname` from `/api/status`), `ensure_local_server` (resolve or start the local engine and report its target), `forget_remote_host`, `discover_servers`, `get_output_dir`, `get_mold_home` / `change_mold_home` (native bootstrap root selection with optional staged migration and process relaunch), `import_source_image` (feeding `source_image`/`edit_images`/`mask_image`/`control_image`/keyframes/audio as base64 into `GenerateRequest`) with `source_stash_put` / `source_stash_get` for content-addressed reuse, `save_output_bytes` / `save_media_bytes` / `save_gallery_media` plus `media_save_directory` (export), `reveal_output_file` / `reveal_saved_media` / `open_logs_dir`, `local_output_file_path`, `secret_get` / `secret_set` / `secret_clear` (per-host API keys in the owner-only secrets file), `set_dock_badge`, `take_notification_action`, `prepare_gallery_thumbnail` / `probe_gallery_thumbnails` / `cancel_gallery_thumbnail` / `forget_gallery_thumbnail`, local-gallery list/trash-list/delete (→ trash)/restore/delete-forever/read commands, `clipboard_write_image` (decode PNG/JPEG/GIF/WebP to RGBA before native clipboard write), `fetch_gallery_thumbnail` / `fetch_gallery_media` (bounded, semaphore-limited native reads of a host's gallery thumbnail or full-size still/audio bytes so WebKit's per-host pool cannot starve them; the webview HTTP route remains the fallback), `send_native_notification` (macOS notification identity image; returns false for the plugin fallback elsewhere), RunPod credential/account/inventory/pod/network-volume commands, `check_for_updates` / `install_pending_update`, and `app_settings_get/set` (window prefs, RunPod selections, update channel, UI scale, last mode — stored in a small `settings.json` under `app_data_dir`, **not** in mold's config.toml, which stays engine-owned).

The Mold-home selection is a tiny shared bootstrap pointer outside the selected root, resolved by `mold-core` before `Config`, the metadata DB hook, tracing, gallery paths, or the embedded server. Precedence is an explicit `MOLD_HOME` environment variable, then the saved local selection, then `~/.mold`. Changing it performs read-only validation before stopping the embedded server, rejects a separately owned `mold serve`, resolves symlink identities, optionally copies the complete current root through a sibling staging directory, persists the new bootstrap path only after success, and relaunches. Failed post-shutdown work restores the old embedded engine. If a saved external drive is absent, Desktop uses recovery logging outside that root, keeps This Mac offline, and leaves Settings available to retry or choose a replacement without creating a fresh tree at the missing mount. CLI, TUI, server/web, and desktop therefore share `Config::mold_dir()`; mobile continues to use its selected remote host's root.

- **HTTP (webview → server):** literally everything else — generate/stream, estimate, expand, upscale, gallery CRUD + Range video (thumbnails and full-size stills/audio go native-first, see above), models list/pull/rm/load/unload/components, loras, catalog families/search/installed/download, downloads queue + stream, chain-jobs full surface + events + stage previews, queue list/patch, resources stream, status, capabilities, chain-limits, placement PUT/DELETE.

**Capabilities file** (`capabilities/default.json`): main window; permissions: `core:default` plus `core:window:allow-show`, `core:window:allow-set-focus`, `core:window:allow-start-dragging`, `core:window:allow-internal-toggle-maximize`, `core:webview:allow-set-webview-zoom`, `dialog:default`, `opener:default` (+ `opener:allow-reveal-item-in-dir`), `notification:default`, `clipboard-manager:allow-write-text`, `clipboard-manager:allow-write-image`, `window-state:default`, `process:allow-restart`. No updater or fs scope is exposed to JavaScript: Mold's Rust commands own the trusted updater object, channel allowlist, and file IO.

**LAN discovery** (the Machines workspace "On your network"): the `discover_servers{timeout_ms?}` IPC command runs `mold_server::mdns::discover` (the `mdns` feature is enabled on the embedded `mold-ai-server` dep) inside `spawn_blocking`, then maps each hit to a camelCase `DiscoveredHost {name, url, host, port, version, authRequired, isThisMachine, instanceId}` (`instanceId` from the `id` TXT record; discovery lists dedupe against connected hosts by URL slug and instance id). `isThisMachine` is computed by intersecting the advertised addresses with the machine's own interface addresses (`if-addrs`), falling back to a hostname-prefix match — so the app's own embedded/local server is flagged rather than offered as a remote. The frontend wraps it as `ipc.discoverServers()` (browser fallback `[]`); pure sort/dedupe/label helpers live in `lib/discovery.ts`. Because the app is not sandboxed, no multicast entitlement is needed, but macOS 15 still gates the browse behind Local Network permission — `src-tauri/Info.plist` supplies `NSLocalNetworkUsageDescription` and `NSBonjourServices = ["_mold._tcp"]` (a browse silently returns nothing without the latter). `Entitlements.plist` is unchanged.

**Icons:** new icon set generated via `cargo tauri icon` from a 1024px master (new artwork, not the web favicon).

## 5. Dev workflow & Nix devshell

The shared devshell exposes cross-platform desktop helpers:

| Command             | Runs                                                                                                                                     |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `desktop-dev`       | Installs locked frontend dependencies, clears a stale Vite listener, and runs Tauri as `Mold - dev` with Metal on macOS or CUDA on Linux |
| `desktop-build`     | Builds the macOS application bundle or the native Linux desktop package and AppImage; optional signing secrets remain macOS-only         |
| `desktop-release`   | macOS only — builds, notarizes, staples, and verifies the signed app and DMG from `.secrets/signing.env`                                 |
| `desktop-check`     | Runs Rust formatting and warning-denied clippy plus frontend format and type checks                                                      |
| `desktop-test`      | Runs the CPU-only Rust test suite, embedded-engine boot test, and frontend Vitest suite                                                  |
| `desktop-ui`        | Runs the frontend-only Vite server against a running `mold serve`                                                                        |
| `frontend-bun-lock` | Refreshes the repo-root Bun lock and Nix dependency set for every frontend target                                                        |

The devshell does not run on Windows, so `scripts/windows.ps1` is the peer of
those commands: `doctor`, `setup`, `dev`, `ui`, `check`, `test`, `build`,
`bundle`, `clean`, and `features`. It resolves the feature recipe per machine —
CPU-only by default, `cuda` on an x64 host with a toolkit, `pulid` when
`protoc` is on PATH, overridable through `MOLD_WINDOWS_FEATURES` /
`MOLD_WINDOWS_CUDA` / `MOLD_WINDOWS_NO_PULID`. Windows bundles an NSIS
installer and has exactly two named absences rather than silent degradation:
in-app updates stay macOS-only, and the `mp4` feature is off (video renders and
muxes normally; only a generated AAC track is refused, by name). CI runs the
`desktop-windows` job in `.github/workflows/desktop.yml` after merge on `main`
or from a manual/nightly workflow — never on pull requests, where the macOS
desktop gate provides the feedback.

The devshell includes `cargo-tauri`, Bun tooling, `lsof`, and ImageMagick. Linux adds WebKitGTK, GTK, Soup, GStreamer, CUDA, and the runtime library paths required by the launched binary. macOS uses system WKWebView and scopes the system C compiler linker variables to desktop commands so the existing Apple build and signing flow is unchanged.

The flake exports `mold-desktop` for the platform default GPU target. Linux also exports `mold-desktop-sm86` for RTX 3090/A40 and `mold-desktop-sm120` for RTX 50-series, plus an AppImage through `desktop-build`. There is intentionally no B200/sm100 desktop package; desktop clients manage those servers remotely. macOS keeps the existing application bundle, signing, and bundle-verification path. Frontend dependencies are pinned through bun2nix on both platforms.

Desktop-created RunPod machines use the same centralized GPU-family table as
the CLI. Stable desktop releases fetch the exact release/source container
manifest and submit `ghcr.io/utensils/mold@sha256:…`; a missing or inconsistent
manifest fails provisioning without mutable fallback. Main-branch desktop
nightlies use the corresponding rolling `latest*` map.

**Rust toolchain:** the devshell's existing `rust-bin.stable.latest`. If a toolchain/Tauri transitive-dep breakage appears (Aethon had to pin 1.92 because 1.95 broke icu_provider/objc2), pin **only** in the desktop package derivation, never the shared devshell toolchain.

**treefmt:** rustfmt already walks all `.rs` (desktop crate is edition 2021, so config stays truthful); nixfmt covers flake edits. Frontend formatting follows the repo's web/ pattern: `bun run fmt` / `fmt:check` (prettier) wired into `desktop-check` — deliberately _not_ added to treefmt (matches existing repo convention).

**Hot reload:** `cargo tauri dev` + Vite HMR for UI; Rust changes rebuild only the thin desktop crate (mold-server & candle compile once, cached in `desktop/src-tauri/target` — recommend `desktop/.cargo/config.toml` reusing sccache via existing `RUSTC_WRAPPER`). `.envrc` untouched.

## 6. Testing strategy (repo TDD rule)

- **Rust (`desktop/src-tauri`):** unit tests for `settings.rs` (round-trip, migration), `server.rs` port allocation + API-key generation, connection-state machine (local/remote/off transitions), updater endpoint allowlisting, bounded requests, complete archive extraction, archive/bundle identity binding, unsafe install-location rejection, and replaceability preflight. One `#[tokio::test]` integration test boots the embedded server on an ephemeral port **without GPU features** (CPU build in CI) and asserts `/health`, `/api/capabilities`, and auth rejection without `X-Api-Key`. Feature-gate metal so `cargo test` runs CPU-only.
- **Frontend:** vitest 4 + `@vue/test-utils`, `happy-dom`. Priority coverage: capability-matrix-driven form logic (ported `generateCapabilities` + `GenerateForm` enable/disable per family), SSE reducer (progress event stream → job card states, including dropped-stream reconciliation via `/api/queue`), chain composer stage editing, updater startup/banner/manual-check/install/error states, and api client (mocked `fetch`, header injection, error envelope parsing). Tauri IPC mocked with `@tauri-apps/api/mocks`.
- **E2E (WebDriver on macOS is a dead end — pragmatic substitute):**
  1. _Browser-level E2E_ (primary): run the Vue app in a real browser (`bun run dev`) against a live `mold serve` (CPU, tiny model or a stub server built from `mold-server::test_support`), driven by **playwright-cli** — covers generate→progress→gallery→delete, model pull, chain job lifecycle. This tests 95% of the app (everything but the native shell).
  2. _Native smoke_ (manual/local, scripted): `desktop-build`, launch the .app, screenshot + basic interaction via the **computer-use** skill; a launch-smoke script asserting the window appears and `/health` of the embedded engine goes green.
- **CI:** `.github/workflows/desktop.yml` runs desktop frontend formatting/type/tests, updater publishing-script tests (including tamper/wrong-key rejection and retention selection), and the macOS Rust fmt/clippy/test gates for desktop-relevant pushes and pull requests. Pull requests and manual runs add a fast debug `.app` packaging smoke test that reuses the test build; `main` skips that redundant bundle and goes directly from the two CI jobs to the real signed/notarized Nightly distribution and public-manifest verification. Superseded commits cancel within their own PR, while main builds and publication remain non-canceling. Tagged Stable artifacts remain in `release.yml`. The desktop cargo root remains excluded from the workspace CI.

## 7. Branch & delivery plan

Desktop work follows the repository's normal feature-branch and pull-request workflow; `desktop/` is no longer isolated to a permanent experiment branch.

**Milestones (each ends runnable):**

- **M0 — Scaffold (skeleton window):** `desktop/` tree, tauri.conf, workspace exclude, devshell commands, CI workflow, blank three-pane shell with overlay title bar renders. Design tokens + typography locked.
- **M1 — Engine online:** embedded server boots on Metal, ephemeral port+key, `get_connection`/engine lifecycle IPC, remote-host mode + switcher, status footer (`/api/status`, `/api/resources/stream` sparklines). Upstream PR to main: `run_server_with_listener` + `ServerHandle` (small, reviewable independently).
- **M2 — Walking skeleton (the demo):** prompt → `/api/generate/stream` with live progress → result pane; gallery grid (virtualized, thumbnails, detail inspector with embedded metadata, delete, reveal-in-Finder); model picker from `/api/models`.
- **M3 — Full generation workspace:** capability-matrix form (all `GenerateRequest` params incl. img2img/mask/control/edit-images via dialog+drag-drop, LoRA stack picker from `/api/loras`, seed/batch, scheduler/CFG++, placement panel), estimate preflight, prompt expansion modal + inline expand, prompt history (recent/search), presets; queue strip + `/api/queue` reconciler + GPU re-lane (PATCH).
- **M4 — Models & catalog:** installed list w/ disk usage + components, pull with SSE progress, downloads drawer (`/api/downloads` + stream, cancel), rm/load/unload, HF+Civitai catalog search (family/kind/source/nsfw filters, trained_words display), install from `cv:`/`hf:` ids.
- **M5 — Video & chains:** video params (frames 8n+1 validation, fps, audio, pipeline modes, keyframes, retake range, spatial/temporal upscale), chain composer (stages, transitions, per-stage overrides, TOML import/export of `mold.chain.v1`), durable chain jobs panel (SSE events, stage previews, resume/retake cascade|splice/cancel/delete/gc), chain-limits + VRAM preflight; standalone upscale.
- **M6 — Polish & ship-readiness:** settings surface (config.toml + DB settings via existing endpoints/expand config), notifications, window-state, single-instance, drag-drop images from Finder, keyboard shortcuts, empty/error states, signed Stable/Nightly updater with complete preflight verification, `mold-desktop` Nix package + signing flow + verify-bundle, docs page.

**"Done enough to merge":** feature-parity checklist vs. the inventory in this plan complete; desktop CI green; `nix build .#mold-desktop` succeeds on aarch64-darwin; app runs signed-ad-hoc on a clean machine; main-tree diff limited to workspace exclude + flake + CI + the small mold-server API addition; no change in existing crate behavior.

## 8. Risks & upstream gaps

1. **`run_server` doesn't report its bound address or expose a shutdown handle** — the two real library-API gaps. Mitigate short-term (bind-probe port trick, POST `/api/shutdown`), fix upstream in M1: `run_server_with_listener(TcpListener, …) -> anyhow::Result<()>` (or return a `ServerHandle { local_addr, shutdown: CancellationToken }`).
2. **Queue/download wire structs live in mold-server, not mold-core** (`job_registry::{JobEntry, QueueListing}`, `LoadModelBody`, `CreateDownloadBody/Response`, `chain_limits::ChainLimits`) — harmless here (frontend consumes JSON; the desktop crate links mold-server anyway), but note for any future thin remote-only build: mirror ~5 small structs or upstream them to mold-core.
3. **Tauri + Metal feature unification:** desktop workspace compiles `mold-inference` with `metal` alongside tauri/wry. Separate lockfile isolates version conflicts from the main workspace; watch for duplicate `objc2`/`metal`-adjacent crates (candle's metal bindings vs wry's objc2) — build early in M1 to surface it. Use target-scoped features: `[target.'cfg(target_os = "macos")'.dependencies] mold-server = { …, features = ["metal"] }` so Linux later swaps to cuda cleanly.
4. **Binary size / build time:** the .app embeds all of candle + pipelines (release binary similar to `mold` CLI, hundreds of MB with fat LTO). Accept for experimental branch; use `lto = "thin"`, `codegen-units = 16` in the desktop release profile to keep CI < 30 min on macos-14.
5. **EventSource header limitation** — solved by fetch-event-source everywhere (decided in §3); do not regress to native EventSource.
6. **WKWebView loopback fetch/CSP:** `tauri://localhost` → `http://127.0.0.1` requires the explicit `connect-src`/`media-src`/`img-src` CSP above; verify Range-request video scrubbing in WKWebView in M2 (the server already supports 206).
7. **Resolved — LTX-2 Metal is performance-qualified** (#597, measured on the 19B/22B distilled FP8 tiers). CPU is the correctness-only backend; keep the qualification visible rather than disabling local execution.
8. **Global process state:** `Config::install_runtime_models_dir_override`, `MOLD_API_KEY` env, tracing init are process-global one-shots — enforce "engine starts at most once per process; restart = app relaunch" (plugin-process `restart`), which sidesteps re-init hazards.
9. **nixpkgs `cargo-tauri` version drift** vs `tauri-build` 2.x: pin check in `desktop-check` (`cargo tauri --version`); worst case override the hook's cargo-tauri like Aethon overrides its cargo.

## File tree — `desktop/`

```
desktop/
├── package.json  bun.lock  bun.nix          # bun2nix-generated
├── index.html  vite.config.ts  tsconfig.json  vitest.config.ts
├── src/
│   ├── main.ts  App.vue  router.ts
│   ├── styles/tokens.css  styles/base.css   # design system (CSS vars, light/dark)
│   ├── lib/
│   │   ├── api/client.ts                    # typed fetch wrapper (base URL + X-Api-Key from IPC)
│   │   ├── api/sse.ts                       # fetch-event-source helpers (POST-SSE, snapshots, reconnect)
│   │   ├── api/types.ts                     # mirrored mold-core wire types
│   │   ├── capabilities.ts                  # ported from web/src/lib/generateCapabilities.ts
│   │   └── ipc.ts                           # invoke() wrappers for all Tauri commands
│   ├── stores/  connection.ts generation.ts queue.ts composer.ts settings.ts
│   ├── views/   GenerateView.vue LibraryView.vue ModelsView.vue MachinesView.vue SettingsView.vue
│   └── components/
│       ├── shell/   TitleBar.vue NavRail.vue Inspector.vue StatusPopover.vue
│       ├── generate/ ParamPanel.vue LoraStack.vue SourceImageWell.vue ExpandSheet.vue EstimateBadge.vue VideoParams.vue PlacementPanel.vue
│       ├── gallery/ AuthedMedia.vue Lightbox.vue           # (planned: VirtualGrid/MediaCard/DetailPane/MetadataTable; the justified grid lives in LibraryView.vue)
│       ├── library/ HistoryDrawer.vue LibraryHeader.vue LibraryChipRow.vue CollectionsShelf.vue CollectionDrillIn.vue TrashBanner.vue TagEditor.vue CollectionPicker.vue RetentionSelect.vue
│       ├── create/  CreateHeader.vue (editable print title) HostChip.vue InspectorPanel.vue …
│       ├── settings/ LibrarySection.vue (trash retention for This device) GenerationSection.vue MediaSection.vue …
│       ├── jobs/    QueueStrip.vue JobCard.vue ChainComposer.vue StageCard.vue ChainJobDetail.vue DownloadsDrawer.vue
│       └── models/  ModelList.vue CatalogSearch.vue ComponentStatus.vue PullProgress.vue
└── src-tauri/
    ├── Cargo.toml  Cargo.lock  build.rs (tauri_build::build())
    ├── tauri.conf.json
    ├── capabilities/default.json
    ├── Entitlements.plist                   # network client; hardened-runtime ready
    ├── icons/…
    └── src/
        ├── main.rs                          # env setup (MOLD_API_KEY) → lib::run()
        ├── lib.rs                           # Builder, plugins, state, invoke_handler
        ├── server.rs                        # embedded engine thread + handle
        ├── commands.rs                      # IPC commands
        └── settings.rs                      # app-local settings.json store
```

## Key code sketches

> **Status (2026-08-30):** the sketches below are the original M0 design
> snapshot, kept for the reasoning. They are not the shipped code — read
> `desktop/src-tauri/Cargo.toml`, `src/lib.rs`, and `src/commands.rs` for that.
> Notably: the crate is at the workspace version (0.26.0) with `mold-core` /
> `mold-db` / `mold-server` pinned to it, `tauri` carries `devtools`,
> `tauri-plugin-updater` is a dependency, the macOS target block carries
> `mdns-sd-discovery` / `block2` / `objc2*` rather than a `mold-server` feature
> override, and a `[patch.crates-io]` block mirrors the root workspace's candle
> and cudarc revision pins because this is a standalone cargo root. Of the
> commands named in the `lib.rs` sketch, `set_remote_host`, `engine_status`,
> `pick_files`, `read_file_b64`, `save_bytes_as`, and `reveal_in_finder` were
> never shipped under those names — see the IPC list in §4.

**`src-tauri/Cargo.toml` (original M0 sketch):**

```toml
[package]
name = "mold-desktop"
version = "0.1.0"
edition = "2021"
rust-version = "1.93"

[lib]
name = "mold_desktop_lib"
crate-type = ["staticlib", "cdylib", "rlib"]

[build-dependencies]
tauri-build = { version = "2", features = [] }

[dependencies]
tauri = { version = "2", features = [] }
tauri-plugin-dialog = "2"
tauri-plugin-opener = "2"
tauri-plugin-notification = "2"
tauri-plugin-single-instance = "2"
tauri-plugin-window-state = "2"
tauri-plugin-clipboard-manager = "2"
tauri-plugin-process = "2"
mold-core   = { path = "../../crates/mold-core",   package = "mold-ai-core",   version = "0.14.0" }
mold-server = { path = "../../crates/mold-server", package = "mold-ai-server", version = "0.14.0",
                features = ["expand", "mp4"] }
tokio  = { version = "1", features = ["rt-multi-thread", "macros", "net", "time"] }
serde  = { version = "1", features = ["derive"] }
serde_json = "1"
anyhow = "1"
uuid   = { version = "1", features = ["v4"] }
reqwest = { version = "0.12", default-features = false, features = ["json", "rustls-tls"] }
tracing = "0.1"
base64  = "0.22"

[target.'cfg(target_os = "macos")'.dependencies]
mold-server = { path = "../../crates/mold-server", package = "mold-ai-server", version = "0.14.0",
                features = ["metal", "webp"] }

[profile.release]
lto = "thin"
codegen-units = 16
strip = true
```

**`src/server.rs` — embedded engine:**

```rust
pub struct EngineHandle {
    pub port: u16,
    pub api_key: String,
    thread: Option<std::thread::JoinHandle<()>>,
}

pub fn start_engine(models_dir: PathBuf, api_key: String) -> anyhow::Result<EngineHandle> {
    // TODO(M1 upstream): replace probe with mold_server::run_server_with_listener
    let port = {
        let probe = std::net::TcpListener::bind(("127.0.0.1", 0))?;
        probe.local_addr()?.port()
    };
    let thread = std::thread::Builder::new().name("mold-server".into()).spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all().thread_name("mold-engine").build().expect("engine runtime");
        if let Err(e) = rt.block_on(mold_server::run_server(
            "127.0.0.1", port, models_dir,
            mold_core::types::GpuSelection::All, 600,
        )) {
            tracing::error!("embedded mold engine exited: {e:#}");
        }
    })?;
    Ok(EngineHandle { port, api_key, thread: Some(thread) })
}
```

**`src/lib.rs` — app setup (abridged):**

```rust
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _, _| {
            if let Some(w) = app.get_webview_window("main") { let _ = w.set_focus(); }
        }))
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_process::init())
        .manage(AppState::default())            // Mutex<Connection>, Option<EngineHandle>
        .invoke_handler(tauri::generate_handler![
            commands::get_connection, commands::start_local_engine,
            commands::stop_local_engine, commands::set_remote_host,
            commands::test_remote_host, commands::engine_status,
            commands::pick_files, commands::read_file_b64,
            commands::save_bytes_as, commands::reveal_in_finder,
            commands::app_settings_get, commands::app_settings_set,
        ])
        .on_window_event(|w, e| { /* ExitRequested → POST /api/shutdown, join engine */ })
        .run(tauri::generate_context!())
        .expect("error while running mold desktop");
}
```

**One IPC command (`src/commands.rs`):**

```rust
#[derive(serde::Serialize)]
pub struct Connection { pub base_url: String, pub api_key: Option<String>, pub mode: String }

#[tauri::command]
pub async fn get_connection(state: tauri::State<'_, AppState>) -> Result<Connection, String> {
    let s = state.inner().connection.lock().await;
    Ok(match &*s {
        Conn::Local(engine) => Connection {
            base_url: format!("http://127.0.0.1:{}", engine.port),
            api_key: Some(engine.api_key.clone()), mode: "local".into(),
        },
        Conn::Remote { url, key } => Connection {
            base_url: url.clone(), api_key: key.clone(), mode: "remote".into(),
        },
        Conn::Off => Connection { base_url: String::new(), api_key: None, mode: "off".into() },
    })
}
```

**Frontend deps:** exact browser-safe versions are owned by the root `package.json` and `bun.lock`: Vue, Vue Router, Pinia, Vite, Tailwind, TypeScript, Vitest, Vue Test Utils, happy-dom, Prettier, `smol-toml`, fetch-event-source, and the retained shared UI packages. `desktop/package.json` owns only the Tauri API, plugins, and CLI required by the native shell. TanStack Query and unused Tauri plugins are intentionally absent.

### Critical files (repo-relative)

- `Cargo.toml` — `[workspace] exclude = ["desktop/src-tauri", "apps/mobile/src-tauri"]`
- `flake.nix` — devshell `desktop-*` commands, cargo-tauri, the `mold-desktop` packages
- `scripts/windows.ps1` — the Windows peer of the `desktop-*` commands
- `crates/mold-server/src/lib.rs` — embedding entry point (`run_server`)
- `studio/lib/generationCapabilities.ts` — the shared capability matrix (`web/src/lib/generateCapabilities.ts` is a thin re-export)
