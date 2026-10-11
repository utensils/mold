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
Both native galleries keep new media unread until that client successfully
displays it; entering Library does not clear other items. The iOS Library badge
and native app-icon counts share this viewing history; hidden collections
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
   Typed H3 references use authenticated upload sessions when a configured key
   and advertised upload support are present; otherwise they use bounded inline
   data, with the server still enforcing authentication. Never reinterpret a 401
   or license refusal as a reason to retry without authentication. Remote license acceptance belongs to that host.
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

## Encrypted remote connections

```bash
mold relay connect --relay-url wss://API.execute-api.REGION.amazonaws.com/production --target 127.0.0.1:7680
mold pairing create --output "$MOLD_HOME/secure/invitation.json"
mold pair --invite-file /private/invitation.json --name "My client"
```

Start an authenticated host first. Its owner-only secure identity stays on the
host; authorize its public signing key through the relay operator's IAM
provisioning command. Configure MOLD_PUBLIC_URL and MOLD_SECURE_WEBSOCKET_URL
on headless hosts; mold serve then owns its outbound connector. The separate
relay connect command reads the identity, never an old relay token.
The operator creates a one-use, two-minute invitation using MOLD_API_KEY (or
@owner-only-key-file). Transfer the invitation privately. Each paired client
stores its own key and revocable grant. MOLD_HOST selects a paired origin.

All remote API traffic, uploads, progress and media use pinned Noise XX over
LAN or the Lambda WebSocket relay. The cloud sees metadata and ciphertext, not
API credentials or media. Bounded S3 objects may carry only the same encrypted
wire bytes; object keys and cloud URLs never replace host identity verification.
Old proxy credentials cannot be upgraded implicitly;
upgrade and pair every client again. Plain HTTP is only a private loopback
engine interface. Do not use raw remote curl or fall back to a cloud-provided
host key. Uploads are bounded to 64 MiB; responses stream with backpressure.
Interrupted mutations are never automatically replayed. RunPod generation requires an
already paired, running pod; explicitly create, pair, and select a replacement
when its GPU or datacenter differs. Provider readiness alone is not trusted
Mold readiness. Native Remote Access
opens the encrypted LAN listener and outbound connector; stopping it closes
active sessions. HTTPS browsers may require WSS because of mixed-content rules.

## Error diagnostics

Server errors are concise presentation summaries; inspect the server logs for
technical diagnostics. Memory summaries use decimal GB/MB and distinguish
graphics, system and shared memory, with estimated need, available budget and
shortfall. Free memory on the machine doing the render. Do not treat that estimate
as a guaranteed allocation target or decide retryability from prose: use error
codes and the queue’s retryable flag. Preserve required restart/cooldown advice.

GUI queue recovery uses **Download and Retry** for repairable missing-model components,
including missing adapters of installed recipes. Queue listing/detail rows may
include `model_download: { model }` for explicitly retryable held jobs. Clients
validate the exact model and preserve the original error, Failure Details and
eligible Move to action. Explicit `model_download: null` vetoes model recovery;
absent hints retain typed missing-model error fallback for older servers and
uninspected catalog recipes,
with inline starting/queued/progress/license/reconnecting/failure feedback.
It follows exact returned download tickets, including companions, and revalidates
server instance and held-job batch identity before retrying. Download failure or
cancellation never retries a generation. Cancelling a generation leaves shared
model downloads server-owned. CLI `mold queue retry` remains an explicit retry;
install a missing model on that same host first with `mold pull`.

GUI library lifecycle actions follow the explicit machine filter: trash, restore and permanent delete affect only selected hosts; All Machines includes every known copy. Empty Trash follows the same host scope. Native Trash ignores collection hiding so deleted prints remain recoverable. CLI and API lifecycle operations continue to target the requested server only.

Trash selections use `POST /api/gallery/trash/delete-selected` with a filenames array. The endpoint checks current trash membership under the publication writer; a restored live print is preserved with `409 GALLERY_NOT_TRASHED`. An older host returning 404 needs an update; never retry through the live-or-trash `delete-forever` endpoint.

GUI Library thumbnails preserve media aspect ratios in continuous justified rows on native macOS/iOS, Tauri desktop and web. Zoom changes target row height. iOS omits machine-name thumbnail badges; details and machine filters retain host information. This changes no CLI/API or unseen-media behavior.

GUI **Edit prompt** opens a spacious live draft editor with recent-prompt search, Clear and Undo clear. Dismissal keeps edits; history recall changes prompt text rather than the model, source media or other settings. Native iOS and macOS Library Settings persist **Show date separators**; turning it off removes daily row breaks while preserving sort order. These are client preferences and do not change CLI/API generation contracts.

GUI print details and **Source Media** expose publication-owned retained inputs with role labels and authenticated thumbnails. Native Mac inspectors and original viewers offer **Use source** and matching right-click actions; **Choose model…** selects a connected compatible model, recipe and input role when the current recipe cannot use the source. Individual reuse attaches original content to a new draft; it is separate from restoring a print's complete settings. Browse only owners allowed by the current library visibility and machine scope. Identical verified reusable bytes share one card while every owning print and role remains selectable; missing content digests keep entries separate. Incremental inventory and bounded memory-only previews speed up return visits; refresh rechecks retained inputs. Source browsing never creates independent pins or changes Trash/permanent-delete retention. Batch inventory accepts at most 100 exact owner filenames at `POST /api/gallery/source-media/inventory`; `GET /api/gallery/source-media/:filename/:member_id/content` extracts independently reusable content (keyframe image bytes), while the existing raw member download preserves its serialized replay format.

Deleting the media currently open in a Library viewer keeps the viewer open on the next surviving item in the current filtered order, or the previous item at the end. It returns to the grid only when no items remain. A surviving machine copy stays visible after a partial deletion failure. Native macOS successful moves to Trash finish without a persistent completion notice; progress, interrupted work and failures remain visible. These GUI behaviors do not alter CLI/API lifecycle contracts.

All GUI surfaces expose duration prediction only when the selected model advertises
`supports_duration_prediction` and is not runtime-disabled. Automatic mode omits
`frames`; explicit duration sends its frame count. Native persisted auto preference
parks on unsupported models without suppressing their explicit frames.

For convolutional LTX video VAE decoding, spatial Auto uses aspect-coupled 768px long-side tiles and 64px overlap independently of the temporal memory estimate. This is not a guarantee that long clips fit: temporal context and full output accumulation still consume memory. Explicit `--spatial-tile` overrides retain their existing geometry. After an observed OOM, smaller retry suggestions are estimates, not a fit guarantee.
