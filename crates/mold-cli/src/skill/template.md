# Mold media generation

Use `mold` to generate and transform images, video, and audio locally or on a
remote Mold server. Inspect the installed binary before relying on remembered
flags or model limits:

```bash
mold --help
mold info <model>
mold server status
```

## Native iPhone and iPad app

Mold Studio Companion (`apps/ios`) is remote-only. In Generate, tap **Model**
below the prompt to search installed models and choose the output kind, recipe
and machine. **Get More Models…** opens model management. Options uses separate
rows for large text; clip/3-D drafts restore after profiles arrive. The viewer
hides the main tabs to expose its media actions. On iPhone, landscape clips use
the full display with native controls and Close; portrait restores the actions.
Both native galleries immediately clear the selected media's New badge while
keeping the previous visit baseline and next-visit clearing. New media counts
appear on the iOS Home Screen/macOS Dock until Library opens; hidden collections
and Trash are excluded. Icon state is local and durable; iOS refresh is
opportunistic while backgrounded and needs notification badge permission.
Info has Done, and sharing
preserves media filenames. Offline Generate/Queue explain unavailable data;
saved drafts wait for their model profiles until an explicit new choice. For development,
`nix develop -c companion-dev` watches native/shared Swift sources and
rebuilds/relaunches in Simulator; `companion-run` launches once. Helpers accept
`SIM=<UDID>` and `BUILD=<directory>`. Tauri retains `ios-dev`.

## Route the request

- For generation, editing, or upscaling, confirm the intended model family,
  task, output type, source media, and destination. Always read the shared
  guide and exactly one family base below. Add a task leaf only when the
  selected H3, Wan, or LTX-2 task requires it.
- For a 3-D mesh (`hunyuan3d`), the input is one image or an advertised set of
  named front/left/back/right views and the output is a GLB. Use `--matting
auto` to preserve useful alpha and remove opaque backgrounds, `on` to
  recompute every supplied cutout, or `off` to keep the pixels unchanged. Add
  `--delight` when the profile advertises Hunyuan3D lighting and highlight
  removal; it runs after matting and before shape or paint.
  There is no prompt to write, and `mold expand` / `mold remix` answer
  with image advice instead of a rewrite. OBJ, OBJ+PBR ZIP, STL and PLY are gallery-side
  exports of the stored GLB, never generation targets, and take optional
  `--size-mm`/`--up-axis`/`--origin` to make the export print-ready; a
  turntable GIF, APNG or WebP (`mold library export <file> --format gif`) is
  a render of it for sharing where no viewer opens a GLB.
- For a multi-stage 3-D job, use `mold mesh-workflow` (or `/api/mesh-workflows`
  directly), after inspecting `/api/models` for the recipe's
  `capabilities.mesh.workflow_modes`. The `text_to_mesh` workflow durably
  chains an image request into Hunyuan3D; `mesh_texture` takes one GLB/OBJ
  mesh reference and an appearance image; `mesh_roundtrip` rebuilds a supplied
  mesh. The CLI infers which from what you give it — `--prompt`, `--mesh` with
  `--image`, or `--mesh` alone — and `--mode` names it outright. Matting and
  delight are independent retained stages when enabled. List or inspect jobs,
  follow their events, resume or cancel them, and delete settled workflow data
  when it is no longer needed. A workflow is durable on ONE machine, so there
  is no local form. A server restart parks an unfinished workflow without
  changing its attached child identity.
- For model selection and current capabilities, use `mold list`, `mold info
<model>`, or the selected server's `/api/models` data. These live surfaces are
  authoritative. `display_name` is a presentation alias; use the stable `name`
  in requests, commands, and saved settings. Curated aliases come from the
  manifest title and retain task/version/precision; never rename weight files.
  This skill intentionally does not duplicate changing model
  IDs, defaults, dimensions, frame grids, or runtime availability.
- For ordinary CLI, queue, server, library, and remote-host workflows, read
  [`{{reference_prefix}}/cli.md`]({{reference_prefix}}/cli.md).
- Before credentials, paid cloud resources, deletion, cancellation, purge, or
  service lifecycle changes, read
  [`{{reference_prefix}}/safety.md`]({{reference_prefix}}/safety.md).
- For small known-good invocations, read
  [`examples/quickstart.md`](examples/quickstart.md).

## Operating contract

1. Run read-only discovery first. Do not assume the local binary and a remote
   host expose the same models or features.
2. Preserve the user's exact prompt and requested model unless they ask for
   creative expansion or substitution.
3. For async work, retain the returned job or batch ID and poll that identity.
   A disconnected stream does not prove that accepted work stopped.
4. Treat command output as the authority on whether cancellation affected
   queued, running, or already-settled work; never report cancellation from the
   request alone.
5. Verify the output exists and is the requested media type before reporting
   success. For remote work, report which host owns the artifact.

{{agent_notes}}

## Direct prompting routes

{{prompt_routes}}

## Optional remote HTTPS relay

```bash
mold relay connect --transport aws --relay-url wss://API.execute-api.REGION.amazonaws.com/production --target 127.0.0.1:7680
```

This opens an outbound tunnel for an authenticated server. The enrollment token is
read from an owner-only `MOLD_RELAY_TOKEN_FILE`; clients use the normal HTTPS
`MOLD_HOST` and their Mold API key/pairing credential, never that token.
The production gateway uses API Gateway, Lambda, DynamoDB and private S3.
Large bodies/media use single-use staged transfers. Generation uses durable queue
recovery and requires saved output; `--no-save` is refused before admission.
For a direct local-development gateway:

```bash
mold relay serve
```

Legacy gateway access publishes one explicitly enrolled machine. Managed
enrollment isolates machines behind dedicated HTTPS root origins. Native macOS
Settings ▸ Remote Access ▸ **Pair your phone** prepares This Mac’s outbound
connector and verifies its managed proxy route before displaying a QR; gateway
configuration and wildcard DNS/TLS deployment must be in place first. The engine
stays authenticated on loopback. Stop Remote Access withdraws the route.
Phone clients prefer advertised LAN, then Tailscale, then HTTPS proxy routes
after credential-free instance proof; never invent routes a listener cannot serve.
The proxy can see credentials/media; do not claim end-to-end encryption or a
cloud GPU fallback.
Use `MOLD_RELAY_DIAGNOSTICS=1` for safe connection/reconnect categories.
See the deployment guide for safe TLS, streaming and metrics routing.

Automatic roaming requires a server-minted paired credential. Operator API keys
remain tied to the explicitly saved address; pair once to enable route learning.
Probes expose a stable digest tag, so arbitrary operator keys never participate
in anonymous route proofs. Plain HTTP still requires a trusted network.

Gallery GUI exports on native iOS, Tauri and web support an optional GIF extra
boundary pause through `pause_ms`; inspect the holding host's
`/api/gallery/export-options` `gif_pause` advertisement first. Zero adds no
pause while preserving FPS. This is an export API/GUI control, not a generation
setting or a new CLI flag. Non-GIF requests omit it.

API-only deployments: the bootstrap `web_ui_enabled` setting defaults to true.
Run `mold config set web_ui_enabled false` and restart the server, or use
`MOLD_WEB_UI_ENABLED=false`. NixOS exposes `services.mold.webUi.enable = false;`.
Only browser pages and SPA assets are disabled (404); all API routes remain,
including `/api/docs`, media, queue and generation. `MOLD_WEB_DIR` does not enable
a disabled interface. Queue source previews keep their media across unchanged
polls; Create's Recent excludes hidden collection members just like the Library.

### Native authoring and GUI inputs

Native iOS Generate is a stable composer; inspect live rendering in Queue and results in Library. Library **Use as Source** keeps prompt/model choices, and its source picker searches all hosts with a machine filter. Mac Queue rows open job details; Discover starts with manifest models. Native Settings offers video autoplay/repeat; Mac GIF export includes playback/repeat/pause controls. GUI still-input imports fit oversized images while preserving orientation and alpha; original exports and retained source/mask pairs remain authoritative. These GUI conveniences do not change CLI/API admission limits or explicit fixed seeds.

Queue GUI rows and Job Details show ordered, role-labeled sealed input images for every model. `/api/queue/{id}/inputs` lists path-free descriptors (`index`, `label`, `preview`); `/api/queue/{id}/input-thumbnail?index=N` reads that exact sealed member. Omit the index for the legacy scalar source image. Audio/video inputs are disclosed without a still preview. These private authenticated routes never resolve provenance filenames. Older servers retain the singular source-image fallback.

## Error diagnostics

Server errors are concise presentation summaries; inspect the server logs for
technical diagnostics. Memory summaries use decimal GB/MB and distinguish
graphics, system and shared memory, with estimated need, available budget and
shortfall. Free memory on the machine doing the render. Do not treat that estimate
as a guaranteed allocation target or decide retryability from prose: use error
codes and the queue’s retryable flag. Preserve required restart/cooldown advice.

GUI queue recovery uses **Download and Retry** for typed missing-model holds,
with inline starting/queued/progress/license/reconnecting/failure feedback.
It follows exact returned download tickets, including companions, and revalidates
server instance and held-job batch identity before retrying. Download failure or
cancellation never retries a generation. Cancelling a generation leaves shared
model downloads server-owned. CLI `mold queue retry` remains an explicit retry;
install a missing model on that same host first with `mold pull`.

GUI library lifecycle actions follow the explicit machine filter: trash, restore and permanent delete affect only selected hosts; All Machines includes every known copy. Empty Trash follows the same host scope. Native Trash ignores collection hiding so deleted prints remain recoverable. CLI and API lifecycle operations continue to target the requested server only.

Trash selections use `POST /api/gallery/trash/delete-selected` with a filenames array. The endpoint checks current trash membership under the publication writer; a restored live print is preserved with `409 GALLERY_NOT_TRASHED`. An older host returning 404 needs an update; never retry through the live-or-trash `delete-forever` endpoint.
