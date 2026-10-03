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
hides the main tabs to expose its media actions. Info has Done, and sharing
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

One gateway publishes one machine. The proxy can see credentials/media; do not
claim end-to-end encryption, automatic GUI hosting or a cloud GPU fallback.
Use `MOLD_RELAY_DIAGNOSTICS=1` for safe connection/reconnect categories.
See the deployment guide for safe TLS, streaming and metrics routing.

Automatic roaming requires a server-minted paired credential. Operator API keys
remain tied to the explicitly saved address; pair once to enable route learning.
Probes expose a stable digest tag, so arbitrary operator keys never participate
in anonymous route proofs. Plain HTTP still requires a trusted network.
