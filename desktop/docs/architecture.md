# mold Desktop — Tauri 2 Implementation Plan

A native macOS (Apple Silicon / Metal) and x86_64 Linux (CUDA) desktop app for mold with full feature parity against the web SPA and a shared Safelight design. The backend and frontend stay platform-neutral; Tauri platform configs own window chrome and bundle details.

## iOS companion

The iPhone app is a separate thin Tauri crate at `apps/mobile/src-tauri`, with
its Vue entry at `desktop/src/mobile`. It never embeds an engine: every request
targets a saved remote host, so the typed HTTP/SSE client and pure generation
form builder are shared while desktop stores that assume **This device** remain
desktop-only. Native DNS-SD browses `_mold._tcp`; manual host entry also accepts
Tailscale MagicDNS and HTTPS names. The initial navigation is Generate,
Gallery, and Hosts, laid out for iPhone safe areas and touch targets.

---

## 1. Location & workspace strategy — DECISION: `desktop/` at repo root, own cargo root, root `[workspace] exclude`

**Pick:** New top-level `desktop/` directory. The frontend lives at `desktop/` (package.json, src/), the Rust crate at `desktop/src-tauri/` with its **own `Cargo.toml` + `Cargo.lock`** (a standalone single-crate workspace). Add to `/Users/jamesbrink/Projects/utensils/mold/Cargo.toml`:

```toml
[workspace]
members = [ ... unchanged ... ]
exclude = ["desktop/src-tauri"]
resolver = "2"
```

**Why (rejecting workspace membership):**

- The root workspace is MSRV 1.85 / edition 2021 and every CI gate runs `--workspace` (`cargo check/clippy/test --workspace` with `-D warnings`). Joining would drag ~400 Tauri/objc2/wry crates into `clippy --workspace`, into crane's `buildDepsOnly` artifacts (invalidating the CUDA dep cache on every Tauri bump), and into `Cargo.lock` churn for a CUDA-heavy workspace. Exclusion means **zero risk to existing CI/builds** — the only main-tree edits are the one-line exclude, flake additions, and a new branch-gated workflow.
- This is exactly the proven Aethon pattern (`cargoRoot = "src-tauri"`, separate `Cargo.lock`, `rustPlatform.buildRustPackage` + `cargo-tauri.hook`).
- Path dependencies work fine across the boundary: `desktop/src-tauri` depends on `../../crates/mold-server` etc. by path; it compiles those crates under its own lock/profile.
- **Deliberate choice: edition 2021 for the desktop crate** (Tauri 2 does not require 2024). This keeps treefmt's `rustfmt { edition = "2021" }` correct for the whole tree and lets the desktop crate build with the devshell's existing stable toolchain. (Aethon used 2024; we don't need it and it buys friction.)

## 2. Backend integration — DECISION: embed `mold-ai-server` in-process, plus first-class remote mode; **all app data flows over HTTP+SSE in both modes**

**Pick:** The Tauri process links `mold-server` (package `mold-ai-server`) with `metal` (+ `expand`, `webp`, `mp4`) features and keeps a local server online as the app's permanent primary engine. It reuses a Mold server already answering on `localhost:7680`; otherwise it spawns `mold_server::run_server("0.0.0.0", port, models_dir, GpuSelection::All, queue_size)` on a dedicated thread with its own tokio runtime. The webview still uses `http://127.0.0.1:<port>`, while other machines reach the advertised LAN address. Local and remote hosts share the same HTTP + SSE wire contract.

**Why not sidecar / external server:**

- _Sidecar `mold serve`_: doubles the shipped binary (~each contains candle + all model pipelines), needs process supervision, orphan cleanup, version skew handling, and externalBin plumbing. No benefit — mold-server is already a clean library with one entry point (`crates/mold-server/src/lib.rs::run_server`, called the same way by `mold-cli/src/commands/serve.rs`).
- _Require external server_: fails "feels like a real desktop app"; double-click must just work.
- _Pure IPC (no HTTP)_: would force reimplementing the queue, SSE fan-out, chain-job runner, download driver, gallery reconciler, and would make remote mode a second code path. Embedding the server gives **one transport for local and remote** — the single most important architectural bet in this plan. The frontend never knows which mode it's in beyond a base URL + key.

**Runtime/threading:** do _not_ run the server on Tauri's async runtime. `run_server` is a long-lived `async fn` that installs global state (`Config::install_runtime_models_dir_override`, SIGPIPE handling) and blocks until shutdown; give it its own `tokio::runtime::Runtime` on a named thread (`mold-server`). Generation work already goes through `spawn_blocking` + per-GPU workers internally. Tauri's own commands stay on `tauri::async_runtime`.

**Port selection:** prefer `0.0.0.0:7680` so the desktop server has the conventional address. If an unrelated process occupies 7680, reserve an ephemeral wildcard port and advertise that real port over mDNS. The listener probe is dropped before `run_server` binds, leaving the existing small TOCTOU race; the upstream `run_server_with_listener` follow-up remains applicable.

**Auth:** resolve `MOLD_API_KEY` as an explicit override; otherwise reuse or generate `desktop-local-api-key` in the owner-only app secrets file. Export it before spawning the server thread (`auth::load_api_keys` reads env once), advertise `auth=1`, and expose a masked reveal/copy control in Settings → Hosts. The frontend attaches `X-Api-Key`; remote hosts retain their own per-host keys. CORS stays permissive (default), with CSP constraining the frontend side.

**Shutdown:** the embedded handle is owned separately from host connections; app exit or an explicit local-engine restart POSTs `/api/shutdown` and joins the thread with a 5s timeout. User-run external servers are never shut down by the app.

**Hosts UI (no modes):** there is no connection switcher — the built-in/local engine is permanently the internal primary (**This device**) and every remote server is a list entry managed in Settings → Hosts (This-device card, Add host, Connected, Remembered, On your network). Hosts dedupe by the server's instance UUID (`/api/status.instance_id`, mDNS `id` TXT record) with display names from the server's hostname; a one-shot Rust boot migration (`settings::migrate_remote_primary`) re-homes old remote-primary installs into the host list, carrying the API key into the per-host secret slot and pinning the generation target. Routing is generation-time only: the Host selector's Auto / Most capable / sticky pick covers every connected host. LTX-2 is CUDA-only: the local Metal engine grays out `ltx2` (drive from `/api/status` gpus backend + family capability map); remote CUDA hosts get it enabled.

## 3. Frontend stack — DECISION: Vue 3.5 + TS strict + Vite 7 + Tailwind v4 + Pinia + TanStack Query/Virtual + fetch-event-source

**Pick (and why Vue again despite "don't copy the design"):** the mandate is new _design_, not new _framework_. Vue 3.5 lets us port two hard-won assets verbatim: the per-family capability matrix (`web/src/lib/generateCapabilities.ts` — the exact enable/disable logic for 11 families × 10 capabilities) and the typed API-layer knowledge in `web/src/lib/api.ts`/`useCatalog.ts`. It keeps one frontend language across the repo (Vue/Vite/Tailwind v4/vue-tsc/vitest all already proven in `web/` and in the bun2nix build). React would buy nothing here and cost a parallel toolchain. **No component library** — fully custom design system (tokens + primitives), per the "fully new, beautifully designed" mandate.

- **Bundler:** Vite 7 (`^7.1`), `@vitejs/plugin-vue ^6`, dev server on **port 1430** (avoid web/'s 5173 and Aethon's 1420).
- **Styling:** Tailwind v4 (`^4.2`, `@tailwindcss/vite`) with a custom token layer (CSS variables for surface/ink/accent, light+dark). The compact three-pane layout is shared; macOS uses overlay traffic-light chrome and Linux uses native decorations. Shortcut labels and primary modifiers are selected at build time.
- **State:** Pinia `^3` for app/session state (connection, generation form, queue mirror, composer drafts). **Server state via `@tanstack/vue-query ^5`** (gallery, models, catalog search with its 5-min server cache, chain jobs, settings) — gives retries, cache invalidation on SSE events, and stale-while-revalidate for the gallery.
- **Virtualized gallery:** `@tanstack/vue-virtual ^3` — virtualize rows of a CSS-grid (justified thumbnails, 256px server thumbs), `useVirtualizer` with dynamic row height. Thousands of items stay smooth.
- **Video:** native `<video>` pointed at `GET /api/gallery/image/:filename` — the endpoint already supports HTTP Range (206), which WKWebView requires for scrubbing. Thumbnails/GIF previews from the existing endpoints. Loading `http://127.0.0.1` media from the `tauri://` origin is permitted by the CSP below.
- **SSE:** `@microsoft/fetch-event-source ^2.0.1` for **everything** — required because (a) `/api/generate/stream`, `/api/upscale/stream`, `/api/generate/chain/stream` are **POST**-SSE, and (b) native `EventSource` cannot send the `X-Api-Key` header even for the GET streams (`/api/resources/stream`, `/api/downloads/stream`, `/api/chain-jobs/:id/events`, `/api/events`). One `sse.ts` helper wraps auth, abort, retry-with-snapshot semantics, and the `/api/queue` polling reconciler (zombie-card dead-lettering, same trick the SPA uses via the `Queued{id}` correlation event).
- **Connection budget (HTTP/1.1, ~6 per host):** every generate job holds a POST-SSE stream for its whole run, and downloads + resources + `/api/events` each hold one more. Batch siblings are therefore capped at **two concurrent streams** (`runWithConcurrency` in `stores/generation.ts`) so a big batch can't starve gallery/download requests behind held-open connections. Worst case: 2 job streams + downloads + resources + events = 5 < 6.
- **Cross-view state:** the Generate form (model, prompt, params) lives in `stores/generateForm.ts`, not the view — `<router-view>` has no KeepAlive, so views unmount on navigation and component-local state would reset. The `events` store subscribes app-wide to `/api/events` (from `App.vue` on connection ready) and keeps the gallery store live; older servers fall back to a 5 s poll while jobs are pending.
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
        "minWidth": 1024,
        "minHeight": 700,
        "center": true,
        "visible": false, // shown after frontend mounts (no white flash)
        "titleBarStyle": "Overlay",
        "hiddenTitle": true,
      },
    ],
    "security": {
      "csp": "default-src 'self'; img-src 'self' blob: data: http://127.0.0.1:* https:; media-src 'self' blob: http://127.0.0.1:* https:; connect-src 'self' http://127.0.0.1:* http://localhost:* https:; style-src 'self' 'unsafe-inline'",
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

- **Not transparent, no `macOSPrivateApi`** — vibrancy via private API risks App Store/notarization pain and fights a color-accurate image app (translucent surfaces behind generated images distort perceived color). The dark, opaque custom chrome is the design.
- `minimumSystemVersion` 12.0 (Metal path + modern WKWebView; mold is Apple-Silicon-targeted anyway).

**Plugins (all `2.x`):** `dialog` (source image/mask/keyframe/TOML pickers, save-as export), `opener` (reveal in Finder, open URLs), `notification` (generation/chain/pull complete while unfocused; fallback outside bundled macOS builds), `single-instance` (one engine, one GPU owner), `window-state` (restore geometry), `clipboard-manager` (copy image/prompt), `process` (relaunch), and `updater` (manifest parsing, download, and mandatory Minisign verification; Mold-owned Rust commands retain installation policy). Bundled macOS notifications use `send_native_notification` so the app's `icon.icns` is passed as the notification identity image instead of being lost in Tauri's bundle-ID lookup. Native webview zoom supplies persistent whole-app scaling. **Deferred:** deep-link (no URL scheme use case yet), `shell` (not needed — no sidecars), `fs` (not needed — file reads happen in Rust commands after dialog returns paths).

**Updater channels and preflight:** the app-local settings store persists `stable` (default) or `nightly`. Startup performs a best-effort check only; menu/Settings checks do the same, an available update appears in persistent app chrome (plus a native notification while backgrounded), and the trusted Rust `Update` remains pending until the user explicitly chooses **Update and restart**. Stable reads public `mold-desktop-stable.json` from the latest tagged release. Nightly reads public `mold-desktop-nightly.json` from the rolling `latest` prerelease, built only for desktop-relevant `main` commits after both desktop CI jobs pass. Default SemVer comparison is intentional: selecting Stable from a newer Nightly waits for a newer stable release instead of silently downgrading data or code.

Tauri downloads the complete updater archive with a 15-minute hard timeout and verifies its Minisign signature before Mold changes the app. Mold then extracts every archive entry into temporary storage, rejects unsafe paths or multiple app bundles, requires `com.utensils.mold` plus the exact manifest version, and runs strict `codesign` plus Gatekeeper assessment against the staged bundle. The same preflight validates the running bundle's identity, rejects DMG/App Translocation launches, and proves the install directory can be replaced without falling into Tauri's destructive authorization path. Only after all checks pass does Mold atomically exchange the staged and installed bundles with macOS `renameatx_np(..., RENAME_SWAP)` and request a normal restart. There is no frontend health handshake, post-launch probation, persistent transaction, or Mold-owned rollback; a preflight failure leaves the installed app untouched, while a successful replacement is final. Because the immediately preceding release still launches its candidate under the removed supervisor, `main()` recognizes only that backup-binary parent plus its one-time health-token environment and terminates it before its first poll; subsequent updates never create a supervisor.

The checked-in config deliberately leaves `createUpdaterArtifacts` false so unsigned PR/dev bundle proofs do not need release credentials. `.github/workflows/desktop-distribution.yml` overlays it to true for Stable and Nightly, derives deterministic next-patch Nightly SemVer and an Apple-safe `CFBundleVersion` from the full main-history commit count, and publishes version-unique archives before updating a channel manifest. Distribution and publication independently decode the Tauri signature and verify it with the exact public key in `tauri.conf.json`. A Nightly job exits before upload if its commit is no longer `main` HEAD; older Stable-tag reruns may refresh their own release but cannot move `releases/latest` backward. The rolling release retains ten complete desktop generations and prunes only after the new public payload and manifest pass anonymous verification. CI requires `TAURI_SIGNING_PRIVATE_KEY` and `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` in addition to Apple signing/notarization secrets. The updater key needs controlled offline backup and must never enter source or logs; losing it strands installed clients, while rotation requires an old-key-signed bridge release containing the replacement public key.

**IPC vs HTTP split (the rule):** _HTTP+SSE for anything the remote server can also answer; IPC only for what must run in the app process._

- **IPC commands:** `get_connection() -> {base_url, api_key, mode}`, `start_local_engine`, `stop_local_engine`, `test_remote_host` (probe returning version/auth plus `instance_id` + `hostname` from `/api/status`), `engine_status` (thread alive, port, models_dir), `pick_files{kind}` + `read_file_b64{path}` (feeding `source_image`/`edit_images`/`mask_image`/`control_image`/keyframes/audio as base64 into `GenerateRequest`), `save_bytes_as{path}` (export), `reveal_in_finder{path}`, local-gallery list/delete/read commands, `clipboard_write_image` (decode PNG/JPEG/GIF/WebP to RGBA before native clipboard write), `send_native_notification` (macOS notification identity image; returns false for the plugin fallback elsewhere), RunPod credential/account/inventory/pod/network-volume commands, `check_for_updates` / `install_pending_update`, and `app_settings_get/set` (window prefs, RunPod selections, update channel, UI scale, last mode — stored in a small `settings.json` under `app_data_dir`, _not_ in mold's config.toml, which stays engine-owned).
- **HTTP (webview → server):** literally everything else — generate/stream, estimate, expand, upscale, gallery CRUD + thumbnails + Range video, models list/pull/rm/load/unload/components, loras, catalog families/search/installed/download, downloads queue + stream, chain-jobs full surface + events + stage previews, queue list/patch, resources stream, status, capabilities, chain-limits, placement PUT/DELETE.

**Capabilities file** (`capabilities/default.json`): main window; permissions: `core:default` plus `core:webview:allow-set-webview-zoom`, `dialog:default`, `opener:default` (+ `opener:allow-reveal-item-in-dir`), `notification:default`, `clipboard-manager:allow-write-text`, `clipboard-manager:allow-write-image`, `window-state:default`, `process:allow-restart`. No updater or fs scope is exposed to JavaScript: Mold's Rust commands own the trusted updater object, channel allowlist, and file IO.

**LAN discovery** (Settings → Hosts "On your network"): the `discover_servers{timeout_ms?}` IPC command runs `mold_server::mdns::discover` (the `mdns` feature is enabled on the embedded `mold-ai-server` dep) inside `spawn_blocking`, then maps each hit to a camelCase `DiscoveredHost {name, url, host, port, version, authRequired, isThisMachine, instanceId}` (`instanceId` from the `id` TXT record; discovery lists dedupe against connected hosts by URL slug and instance id). `isThisMachine` is computed by intersecting the advertised addresses with the machine's own interface addresses (`if-addrs`), falling back to a hostname-prefix match — so the app's own embedded/local server is flagged rather than offered as a remote. The frontend wraps it as `ipc.discoverServers()` (browser fallback `[]`); pure sort/dedupe/label helpers live in `lib/discovery.ts`. Because the app is not sandboxed, no multicast entitlement is needed, but macOS 15 still gates the browse behind Local Network permission — `src-tauri/Info.plist` supplies `NSLocalNetworkUsageDescription` and `NSBonjourServices = ["_mold._tcp"]` (a browse silently returns nothing without the latter). `Entitlements.plist` is unchanged.

**Icons:** new icon set generated via `cargo tauri icon` from a 1024px master (new artwork, not the web favicon).

## 5. Dev workflow & Nix devshell

The shared devshell exposes cross-platform desktop helpers:

| Command            | Runs                                                                                                                                     |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `desktop-dev`      | Installs locked frontend dependencies, clears a stale Vite listener, and runs Tauri as `Mold - dev` with Metal on macOS or CUDA on Linux |
| `desktop-build`    | Builds the macOS application bundle or the native Linux desktop package and AppImage; optional signing secrets remain macOS-only         |
| `desktop-check`    | Runs Rust formatting and warning-denied clippy plus frontend format and type checks                                                      |
| `desktop-test`     | Runs the CPU-only Rust test suite, embedded-engine boot test, and frontend Vitest suite                                                  |
| `desktop-ui`       | Runs the frontend-only Vite server against a running `mold serve`                                                                        |
| `desktop-bun-lock` | Refreshes the Nix-pinned Bun dependency lock after `desktop/bun.lock` changes                                                            |

The devshell includes `cargo-tauri`, Bun tooling, `lsof`, and ImageMagick. Linux adds WebKitGTK, GTK, Soup, GStreamer, CUDA, and the runtime library paths required by the launched binary. macOS uses system WKWebView and scopes the system C compiler linker variables to desktop commands so the existing Apple build and signing flow is unchanged.

The flake exports `mold-desktop` for the platform default GPU target. Linux also exports `mold-desktop-sm120` for Blackwell and an AppImage through `desktop-build`; macOS keeps the existing application bundle, signing, and bundle-verification path. Frontend dependencies are pinned through bun2nix on both platforms.

**Rust toolchain:** the devshell's existing `rust-bin.stable.latest`. If a toolchain/Tauri transitive-dep breakage appears (Aethon had to pin 1.92 because 1.95 broke icu_provider/objc2), pin **only** in the desktop package derivation, never the shared devshell toolchain.

**treefmt:** rustfmt already walks all `.rs` (desktop crate is edition 2021, so config stays truthful); nixfmt covers flake edits. Frontend formatting follows the repo's web/ pattern: `bun run fmt` / `fmt:check` (prettier) wired into `desktop-check` — deliberately _not_ added to treefmt (matches existing repo convention).

**Hot reload:** `cargo tauri dev` + Vite HMR for UI; Rust changes rebuild only the thin desktop crate (mold-server & candle compile once, cached in `desktop/src-tauri/target` — recommend `desktop/.cargo/config.toml` reusing sccache via existing `RUSTC_WRAPPER`). `.envrc` untouched.

## 6. Testing strategy (repo TDD rule)

- **Rust (`desktop/src-tauri`):** unit tests for `settings.rs` (round-trip, migration), `server.rs` port allocation + API-key generation, connection-state machine (local/remote/off transitions), updater endpoint allowlisting, bounded requests, complete archive extraction, archive/bundle identity binding, unsafe install-location rejection, and replaceability preflight. One `#[tokio::test]` integration test boots the embedded server on an ephemeral port **without GPU features** (CPU build in CI) and asserts `/health`, `/api/capabilities`, and auth rejection without `X-Api-Key`. Feature-gate metal so `cargo test` runs CPU-only.
- **Frontend:** vitest 4 + `@vue/test-utils` + `@testing-library/vue`, `happy-dom`. Priority coverage: capability-matrix-driven form logic (ported `generateCapabilities` + `GenerateForm` enable/disable per family), SSE reducer (progress event stream → job card states, including dropped-stream reconciliation via `/api/queue`), chain composer stage editing, updater startup/banner/manual-check/install/error states, and api client (mocked `fetch`, header injection, error envelope parsing). Tauri IPC mocked with `@tauri-apps/api/mocks`.
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
7. **LTX-2 unusable on Metal:** must be a designed state (family card shows "Requires CUDA — connect a remote host"), not a runtime error.
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
│   ├── views/   GenerateView.vue GalleryView.vue JobsView.vue ModelsView.vue SettingsView.vue
│   └── components/
│       ├── shell/   TitleBar.vue NavRail.vue Inspector.vue StatusFooter.vue
│       ├── generate/ ParamPanel.vue LoraStack.vue SourceImageWell.vue ExpandSheet.vue EstimateBadge.vue VideoParams.vue PlacementPanel.vue
│       ├── gallery/ VirtualGrid.vue MediaCard.vue DetailPane.vue MetadataTable.vue
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

**`src-tauri/Cargo.toml` (exact deps):**

```toml
[package]
name = "mold-desktop"
version = "0.1.0"
edition = "2021"
rust-version = "1.85"

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

**Frontend deps (`desktop/package.json`):** vue `^3.5`, vue-router `^4.5`, pinia `^3`, `@tanstack/vue-query ^5`, `@tanstack/vue-virtual ^3`, `@microsoft/fetch-event-source ^2.0.1`, `@tauri-apps/api ^2`, `@tauri-apps/plugin-dialog ^2`, `-opener ^2`, `-notification ^2`, `-clipboard-manager ^2`, `-window-state ^2`, `-process ^2`; dev: `@tauri-apps/cli ^2`, vite `^7.1`, `@vitejs/plugin-vue ^6`, tailwindcss `^4.2` + `@tailwindcss/vite ^4.2`, typescript `^5.9`, vue-tsc `^3`, vitest `^4`, `@vue/test-utils ^2.4`, `@testing-library/vue ^8`, happy-dom, prettier.

### Critical Files for Implementation

- /Users/jamesbrink/Projects/utensils/mold/Cargo.toml — add `[workspace] exclude = ["desktop/src-tauri"]`
- /Users/jamesbrink/Projects/utensils/mold/flake.nix — devshell `desktop-*` commands, cargo-tauri, `mold-desktop` package
- /Users/jamesbrink/Projects/utensils/mold/crates/mold-server/src/lib.rs — embedding entry point; M1 upstream `run_server_with_listener`
- /Users/jamesbrink/Projects/utensils/mold/web/src/lib/generateCapabilities.ts — capability matrix to port verbatim
- /Users/jamesbrink/Projects/utensils/aethon/flake.nix — Tauri-on-Nix packaging recipe + Darwin linker pins to copy
