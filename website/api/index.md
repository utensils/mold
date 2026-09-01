# Server API

When running `mold serve`, the same engine behind Mold's CLI becomes a REST API
with SSE progress. This keeps shell scripts, agents, native apps, browser
clients, and custom integrations on one generation contract.

## Endpoints

| Method   | Path                                             | Auth          | Description                                                                                                                                                                                                                                                         |
| -------- | ------------------------------------------------ | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST`   | `/api/generate`                                  | key           | Generate images from prompt                                                                                                                                                                                                                                         |
| `POST`   | `/api/generate/stream`                           | key           | Generate with SSE progress streaming                                                                                                                                                                                                                                |
| `POST`   | `/api/generate/placement-preview`                | key           | Read-only authoritative placement probe; reserves nothing and queues nothing                                                                                                                                                                                        |
| `POST`   | `/api/generation-batches`                        | key           | Durably admit an idempotent ordered batch of singleton generations                                                                                                                                                                                                  |
| `GET`    | `/api/generation-batches/:id`                    | key           | Read one durable generation batch by server batch ID                                                                                                                                                                                                                |
| `DELETE` | `/api/generation-batches/:id`                    | key           | Cancel every non-terminal child of one durable generation batch                                                                                                                                                                                                     |
| `GET`    | `/api/generation-batches/:id/events`             | key           | Durable batch SSE stream; first frame is always a snapshot                                                                                                                                                                                                          |
| `GET`    | `/api/generation-batches/by-client/:id`          | key           | Recover a durable generation batch by its client idempotency ID                                                                                                                                                                                                     |
| `POST`   | `/api/generation-batches/status`                 | key           | Reconcile a bounded set of durable generation batches                                                                                                                                                                                                               |
| `POST`   | `/api/generation-batches/sweep`                  | key           | Purge fully settled batch summaries past `queue.held_retention_days`                                                                                                                                                                                                |
| `POST`   | `/api/generate/estimate`                         | key           | Estimate request-sensitive peak memory for a generation request                                                                                                                                                                                                     |
| `POST`   | `/api/generate/reference-upload-sessions`        | key           | Open a request-bound MiniMax H3 reference-upload session                                                                                                                                                                                                            |
| `DELETE` | `/api/generate/reference-upload-sessions`        | key           | Cancel an open upload session and release its staged bytes                                                                                                                                                                                                          |
| `PUT`    | `/api/generate/reference-upload`                 | key           | Stream one reference's bytes into an open session                                                                                                                                                                                                                   |
| `GET`    | `/api/capabilities/ltx2-control-adapters`        | key           | Compatible official IC-LoRA controls for an installed LTX-2 model                                                                                                                                                                                                   |
| `GET`    | `/api/capabilities/ltx2-camera-controls`         | key           | Compatible built-in camera controls; `?detail=1` returns the availability envelope instead of a bare array                                                                                                                                                          |
| `POST`   | `/api/generate/chain/validate`                   | key           | Normalize and validate a chain without queueing work                                                                                                                                                                                                                |
| `POST`   | `/api/chain-jobs`                                | key           | Create a durable async chain job                                                                                                                                                                                                                                    |
| `GET`    | `/api/chain-jobs`                                | key           | List durable chain jobs                                                                                                                                                                                                                                             |
| `POST`   | `/api/chain-jobs/placement-preview`              | key           | Per-stage placement preview for a chain job                                                                                                                                                                                                                         |
| `GET`    | `/api/chain-jobs/:id`                            | key           | Get durable chain-job detail                                                                                                                                                                                                                                        |
| `GET`    | `/api/chain-jobs/:id/events`                     | key           | Durable chain-job SSE events                                                                                                                                                                                                                                        |
| `POST`   | `/api/chain-jobs/:id/resume`                     | key           | Resume a paused, interrupted, failed, or cancelled chain job                                                                                                                                                                                                        |
| `POST`   | `/api/chain-jobs/:id/retake`                     | key           | Retake one chain-job stage                                                                                                                                                                                                                                          |
| `POST`   | `/api/chain-jobs/:id/amend`                      | key           | Replace a chain job's stage list in place, reusing cached clips                                                                                                                                                                                                     |
| `POST`   | `/api/chain-jobs/:id/cancel`                     | key           | Cancel a queued or running chain job                                                                                                                                                                                                                                |
| `POST`   | `/api/chain-jobs/:id/operations/:op_id/cancel`   | key           | Cancel one in-flight chain-job mutation by its operation ID                                                                                                                                                                                                         |
| `DELETE` | `/api/chain-jobs/:id`                            | key           | Delete a non-running chain job                                                                                                                                                                                                                                      |
| `POST`   | `/api/chain-jobs/gc`                             | key           | Run chain-job artifact GC                                                                                                                                                                                                                                           |
| `GET`    | `/api/chain-jobs/:id/stages/:idx/preview`        | key           | Fetch a stage preview JPEG                                                                                                                                                                                                                                          |
| `GET`    | `/api/chain-jobs/:id/stages/:idx/media`          | key or ticket | Stream a completed stage MP4; HEAD and byte ranges are supported                                                                                                                                                                                                    |
| `POST`   | `/api/chain-jobs/:id/stages/:idx/media-token`    | key           | Mint a short-lived ticket scoped to that exact stage-media path                                                                                                                                                                                                     |
| `POST`   | `/api/expand`                                    | key           | Expand a prompt using LLM, optionally absorbing a visual `style` directive                                                                                                                                                                                          |
| `POST`   | `/api/remix`                                     | key           | Generate exact-count, subject-preserving prompt alternatives with structured provenance                                                                                                                                                                             |
| `GET`    | `/api/models`                                    | key           | List available models                                                                                                                                                                                                                                               |
| `GET`    | `/api/models/:model/components`                  | key           | List required model component readiness and paths                                                                                                                                                                                                                   |
| `GET`    | `/api/loras`                                     | key           | List installed LoRAs, optionally filtered by `?model=` compatibility                                                                                                                                                                                                |
| `POST`   | `/api/models/load`                               | key           | Load/swap the active model                                                                                                                                                                                                                                          |
| `POST`   | `/api/models/pull`                               | key           | Pull/download a model                                                                                                                                                                                                                                               |
| `DELETE` | `/api/models/unload`                             | key           | Unload model to free GPU memory                                                                                                                                                                                                                                     |
| `DELETE` | `/api/models/:model`                             | key           | Remove a downloaded model (keeps components shared with other models)                                                                                                                                                                                               |
| `GET`    | `/api/gallery`                                   | key           | List saved images (`?view=library\|trash`, `?filename=` narrows to one print); supports conditional GET                                                                                                                                                             |
| `POST`   | `/api/gallery/media-token`                       | key           | Mint a short-lived, read-only ticket for one full-size gallery path                                                                                                                                                                                                 |
| `GET`    | `/api/gallery/source-media/:name`                | key           | List opaque retained source-media members, or an explicit legacy, missing/corrupt, or authentication-unavailable state                                                                                                                                              |
| `GET`    | `/api/gallery/source-media/:name/:member`        | key           | Download one exact retained source-media member without exposing a server path or queue-media identity                                                                                                                                                              |
| `POST`   | `/api/gallery/source-media/:name/reuse-sessions` | key           | Mint a one-time two-minute same-host hydration handle bound to the credential, server instance, exact gallery identity, selected members, and canonical target request; pass it only on the next singleton generation admission as `X-Mold-Retained-Media-Session`  |
| `GET`    | `/api/gallery/export-options`                    | key           | Animation export formats and options this build can transcode into                                                                                                                                                                                                  |
| `POST`   | `/api/gallery/export/:name`                      | key           | Transcode one gallery MP4 into GIF, APNG, or WebP                                                                                                                                                                                                                   |
| `PUT`    | `/api/gallery/import/:name`                      | key           | Stream an already-encoded print plus its metadata into this host's gallery                                                                                                                                                                                          |
| `POST`   | `/api/gallery/organize`                          | key           | Apply one organization edit (title, favorite, tags, collection) to many prints                                                                                                                                                                                      |
| `POST`   | `/api/gallery/mutations`                         | key           | Replay-safe bulk organization mutation, deduped by operation ID                                                                                                                                                                                                     |
| `GET`    | `/api/gallery/collections`                       | key           | List collections with their item counts                                                                                                                                                                                                                             |
| `POST`   | `/api/gallery/collections`                       | key           | Create a collection                                                                                                                                                                                                                                                 |
| `GET`    | `/api/gallery/collections/:id`                   | key           | One collection plus its member filenames in order                                                                                                                                                                                                                   |
| `PATCH`  | `/api/gallery/collections/:id`                   | key           | Rename, describe, or re-cover a collection                                                                                                                                                                                                                          |
| `DELETE` | `/api/gallery/collections/:id`                   | key           | Delete a collection; its prints are untouched                                                                                                                                                                                                                       |
| `PUT`    | `/api/gallery/collections/:id/items`             | key           | Add or remove prints in a collection                                                                                                                                                                                                                                |
| `GET`    | `/api/gallery/tags`                              | key           | Every tag with its use count                                                                                                                                                                                                                                        |
| `PATCH`  | `/api/gallery/tags/:name`                        | key           | Rename a tag, merging into an existing one when the new name is taken                                                                                                                                                                                               |
| `DELETE` | `/api/gallery/tags/:name`                        | key           | Delete a tag from every print                                                                                                                                                                                                                                       |
| `POST`   | `/api/gallery/trash`                             | key           | Move several prints to the trash                                                                                                                                                                                                                                    |
| `DELETE` | `/api/gallery/trash`                             | key           | Empty the trash now                                                                                                                                                                                                                                                 |
| `POST`   | `/api/gallery/trash/restore`                     | key           | Restore trashed prints to the live gallery                                                                                                                                                                                                                          |
| `POST`   | `/api/gallery/trash/delete-forever`              | key           | Permanently delete live or trashed prints                                                                                                                                                                                                                           |
| `POST`   | `/api/gallery/trash/sweep`                       | key           | Run one `gallery.trash_retention_days` retention pass now                                                                                                                                                                                                           |
| `POST`   | `/api/pairing/sessions`                          | key           | Mint an authenticated, one-use, two-minute mobile pairing ticket                                                                                                                                                                                                    |
| `POST`   | `/api/pairing/claim`                             | none          | Redeem a pairing ticket once; the durable key is never present in the QR                                                                                                                                                                                            |
| `GET`    | `/api/pairing/clients`                           | key           | List the clients paired with this host                                                                                                                                                                                                                              |
| `DELETE` | `/api/pairing/clients/:id`                       | key           | Revoke one paired client                                                                                                                                                                                                                                            |
| `GET`    | `/api/gallery/image/:name`                       | key or ticket | Fetch a saved image                                                                                                                                                                                                                                                 |
| `PATCH`  | `/api/gallery/image/:name`                       | key           | Edit one print's title, favorite flag, and tags                                                                                                                                                                                                                     |
| `DELETE` | `/api/gallery/image/:name`                       | key           | Move a saved image to the trash (`?permanent=true` deletes it for good)                                                                                                                                                                                             |
| `GET`    | `/api/gallery/thumbnail/:name`                   | key           | Fetch a cached thumbnail; `?size=256\|512` and `?fmt=png\|jpeg` select a rendition (default 256 px PNG, unchanged); the reply names it in `x-mold-thumbnail-rendition` (`512-jpg`) so clients can tell an older server, which ignores the query, from a small print |
| `GET`    | `/api/gallery/preview/:name`                     | key           | Fetch a cached GIF preview for video gallery rows                                                                                                                                                                                                                   |
| `GET`    | `/api/downloads`                                 | key           | List up to two active downloads (`active_jobs`), plus queued, failed, and completed jobs                                                                                                                                                                            |
| `POST`   | `/api/downloads`                                 | key           | Queue a manifest model download                                                                                                                                                                                                                                     |
| `DELETE` | `/api/downloads/:id`                             | key           | Cancel a queued or active download                                                                                                                                                                                                                                  |
| `GET`    | `/api/downloads/stream`                          | key           | Download queue updates as SSE                                                                                                                                                                                                                                       |
| `GET`    | `/api/licenses`                                  | key           | Third-party model licenses and whether this server has accepted them                                                                                                                                                                                                |
| `POST`   | `/api/licenses/accept`                           | key           | Record acceptance of pinned terms on this server without downloading                                                                                                                                                                                                |
| `GET`    | `/api/catalog/families`                          | key           | Live catalog family/kind metadata                                                                                                                                                                                                                                   |
| `GET`    | `/api/catalog/search`                            | key           | Search the live HF/Civitai catalog; sort by downloads, recent additions, or rating                                                                                                                                                                                  |
| `GET`    | `/api/catalog/installed`                         | key           | List installed catalog entries and LoRAs                                                                                                                                                                                                                            |
| `GET`    | `/api/catalog/credentials`                       | key           | Return masked status for server-owned HF/Civitai credentials                                                                                                                                                                                                        |
| `PUT`    | `/api/catalog/credentials/:provider`             | key           | Save an `hf` or `civitai` credential on the serving host                                                                                                                                                                                                            |
| `DELETE` | `/api/catalog/credentials/:provider`             | key           | Remove a saved host credential and fall back to its environment-provided default                                                                                                                                                                                    |
| `GET`    | `/api/catalog/:id`                               | key           | Resolve one `hf:` or `cv:` catalog entry                                                                                                                                                                                                                            |
| `POST`   | `/api/catalog/:id/download`                      | key           | Queue a catalog entry plus missing companions                                                                                                                                                                                                                       |
| `POST`   | `/api/upscale`                                   | key           | Upscale image with Real-ESRGAN                                                                                                                                                                                                                                      |
| `POST`   | `/api/upscale/stream`                            | key           | Upscale with SSE tile progress                                                                                                                                                                                                                                      |
| `GET`    | `/api/resources`                                 | key           | Latest RAM/GPU resource snapshot                                                                                                                                                                                                                                    |
| `GET`    | `/api/resources/stream`                          | key           | Resource snapshots as SSE                                                                                                                                                                                                                                           |
| `GET`    | `/api/devices`                                   | key           | Stable runtime-visible device inventory with nullable cached telemetry                                                                                                                                                                                              |
| `PATCH`  | `/api/devices/:id`                               | key           | Persist and apply a scheduler-V2 GPU enable/disable lifecycle request                                                                                                                                                                                               |
| `GET`    | `/api/events`                                    | key           | Server-wide lifecycle events (job + gallery) as SSE                                                                                                                                                                                                                 |
| `GET`    | `/api/queue`                                     | key           | Server-authoritative job listing (queued, running, and held; UUIDv4 ids), paged by `?limit=` / `?cursor=`; used by the SPA to reconcile dropped SSE streams                                                                                                         |
| `DELETE` | `/api/queue`                                     | key           | Cancel every queued or restart-paused job; running work is left alone                                                                                                                                                                                               |
| `POST`   | `/api/queue/pause`                               | key           | Pause new-job dispatch                                                                                                                                                                                                                                              |
| `POST`   | `/api/queue/:id/pause`                           | key           | Pause one waiting generation without pausing its siblings                                                                                                                                                                                                           |
| `POST`   | `/api/queue/:id/resume`                          | key           | Resume one paused generation without changing host-wide dispatch                                                                                                                                                                                                    |
| `POST`   | `/api/queue/resume`                              | key           | Resume new-job dispatch                                                                                                                                                                                                                                             |
| `GET`    | `/api/queue/:id`                                 | key           | Read one job in full, its submitted settings included                                                                                                                                                                                                               |
| `PATCH`  | `/api/queue/:id`                                 | key           | Update the preferred GPU lane and/or dispatch position for a queued job                                                                                                                                                                                             |
| `DELETE` | `/api/queue/:id`                                 | key           | Cancel a queued job, or cooperatively cancel a running one                                                                                                                                                                                                          |
| `GET`    | `/api/queue/:id/preview`                         | key           | Latest folded progress snapshot for one live job (step, stage, weights, download, denoise preview)                                                                                                                                                                  |
| `POST`   | `/api/queue/:id/retry`                           | key           | Retry a held child using its complete fenced batch authority                                                                                                                                                                                                        |
| `POST`   | `/api/queue/held/sweep`                          | key           | Purge held rows past `queue.held_retention_days` and release their staged media                                                                                                                                                                                     |
| `GET`    | `/api/activity`                                  | key           | Host-owned nonterminal work snapshot (prints + sequences) for reconciling Now developing                                                                                                                                                                            |
| `GET`    | `/api/history`                                   | key           | Prompt history, newest first (`?query=` substring filter, `?limit=` up to 500)                                                                                                                                                                                      |
| `DELETE` | `/api/history`                                   | key           | Clear prompt history (`?keep=N` trims to the most recent N)                                                                                                                                                                                                         |
| `GET`    | `/api/capabilities`                              | key           | Feature capabilities, including optional per-host expansion and LAN-discovery state                                                                                                                                                                                 |
| `GET`    | `/api/discovery/peers`                           | key           | DNS-SD peers visible on the serving host's LAN (when `discovery.can_browse`)                                                                                                                                                                                        |
| `GET`    | `/api/capabilities/chain-limits`                 | key           | Chain-generation request limits                                                                                                                                                                                                                                     |
| `GET`    | `/api/config`                                    | key           | List every effective config row with its source (`db`/`file`/`env`)                                                                                                                                                                                                 |
| `GET`    | `/api/config/:key`                               | key           | Read one config key (value + owning source)                                                                                                                                                                                                                         |
| `PUT`    | `/api/config/:key`                               | key           | Set a config key, routed by surface like `mold config set`                                                                                                                                                                                                          |
| `DELETE` | `/api/config/:key`                               | key           | Reset a DB-backed key like `mold config reset`                                                                                                                                                                                                                      |
| `GET`    | `/api/config/profiles`                           | key           | List settings profiles and the active one                                                                                                                                                                                                                           |
| `PUT`    | `/api/config/profile`                            | key           | Switch the active settings profile                                                                                                                                                                                                                                  |
| `GET`    | `/api/config/model/:name/placement`              | key           | Read model-specific device placement defaults                                                                                                                                                                                                                       |
| `PUT`    | `/api/config/model/:name/placement`              | key           | Save model-specific device placement defaults                                                                                                                                                                                                                       |
| `DELETE` | `/api/config/model/:name/placement`              | key           | Clear model-specific device placement defaults                                                                                                                                                                                                                      |
| `POST`   | `/api/shutdown`                                  | key           | Trigger graceful server shutdown                                                                                                                                                                                                                                    |
| `GET`    | `/api/status`                                    | key           | Server health + status                                                                                                                                                                                                                                              |
| `GET`    | `/health`                                        | none          | Always-200 liveness check; names degraded subsystems                                                                                                                                                                                                                |
| `GET`    | `/api/openapi.json`                              | none          | OpenAPI spec                                                                                                                                                                                                                                                        |
| `GET`    | `/api/docs`                                      | none          | Interactive API docs (Scalar)                                                                                                                                                                                                                                       |
| `GET`    | `/metrics`                                       | none          | Prometheus metrics (feature-gated; mounted outside the auth layer)                                                                                                                                                                                                  |

The **Auth** column is the tier enforced when `MOLD_API_KEY` is set: `key`
needs an `X-Api-Key` header, `none` is exempt, and `key or ticket` also accepts
a short-lived media ticket on `GET`/`HEAD` (see
[Gallery media tickets](#gallery-media-tickets)). With `MOLD_API_KEY` unset no
route requires a credential.

### Capability discovery

`GET /api/capabilities` is additive and safe to feature-detect. Current
servers advertise the accepted catalog sort values in
`catalog.sort` (`downloads`, `recent`, `rating`), queue controls as
`queue.can_pause`, `queue.can_pause_job`, `queue.can_cancel_all`, and
`queue.can_reorder`, and
server-assisted DNS-SD as `discovery.can_browse`, and the read-only device
resource as `devices.available`. `devices.lifecycle` is true only when
scheduler V2 is the authoritative runtime; legacy, observe, maintenance, and
unavailable runtimes report false. Those runtimes advertise
`devices.restart_enable` instead: clients may offer **Enable on restart** only
for a device whose persisted preference is disabled. Live controls must also
require `dispatch.v2_authoritative`. Stable pin support is advertised as
`devices.stable_pins`; versioned lanes and learned ETA are advertised as
`devices.planned_lanes` and `devices.learned_eta` only while V2 is
authoritative. Dispatch rollout is exposed as `dispatch.active_mode`,
`dispatch.v2_authoritative`, and `dispatch.observes_v2_decisions`. Clients must
only request
`GET /api/discovery/peers` when that discovery flag is true. Older servers may
omit these fields; clients must treat missing arrays as empty and missing
booleans as `false`.

`licenses` is `true` on a server that exposes `GET /api/licenses` and honours
`accept_licenses` on its download routes. Absent (`false`) means acceptance can
only be recorded by running `mold pull --accept-license` in a shell on that
host, so a client must not offer an in-app acceptance flow it cannot deliver.

`identity: { multi_photo, max_photos, true_cfg }` describes which face-identity
request shapes this server understands beyond the singular `id_image`. `multi_photo`
means the server accepts `id_images` / `id_image_names` and averages the set
into one identity (bounded by `max_photos`, always `mold_core::identity::ID_IMAGES_MAX`
when true); `true_cfg` means it accepts `true_cfg` / `cfg_start_step` and runs
PuLID's real negative branch. True CFG is FLUX-only. An SDXL identity request refuses
these fields outright rather than reading this block. The block is absent (all fields read as `false`)
on servers that predate it, and absence means no: a client that wants several
photographs or true CFG must refuse rather than send fields an older server
would silently drop and render without.

`gallery` describes what this host's Library can do: `can_delete` (always
true), `trash: { enabled, retention_days }`, `organize`, `bulk_mutations`,
`media_version`, `conditional_get`, and `row_events`. `trash.enabled` and
`organize` are false whenever the metadata DB is disabled (`MOLD_DB_DISABLE=1`)
or gallery output is off, and a client that sees them false must hide every
organization control and keep the hard-delete wording.

`durable_media` is present only while restart-safe encrypted request media is
actually live (`protocol_version`, `encrypted_at_rest`, `generate_request_media`,
`identity`, `private_h3`). Absence means unavailable, so a request carrying
conditioning media is refused with `503 DURABLE_MEDIA_UNAVAILABLE`.

`expand` reports `configured`, `backend`, `remix`, the manifest `model` local
expansion resolves (`qwen3-expand` today, so clients stop hard-coding it), and
`model_present` — the one field the shared expansion-routing policy reads
before moving a rewrite off the generation route.

`reference_uploads` describes the request-bound MiniMax H3 upload protocol:
`available` (only while API-key auth is on), `protocol_version` (2),
`session_path`, `upload_path`, the two handle headers, `max_file_bytes`,
`max_session_bytes`, `max_active_sessions`, and `session_ttl_ms`. It advertises
the ingress protocol, never model activation.

`model_access` names explicit server-enforced family restrictions, and
`minimax_h3` is present only on an authenticated host whose exact H3 runtime
partition carries a reviewed qualification record.

`generation_profile_v1` is true on a server whose `/api/models` rows carry the
complete version-1 generation profile; false or absent identifies a legacy host
whose flattened fields need the client-side adapter.

`queue` additionally carries `durable_queue`, `stable_device_pins`,
`cooperative_cancellation`, and `heterogeneous_batch_max_outputs`, and
`dispatch` carries `modes` (the values `MOLD_DISPATCH_MODE` accepts) and
`request_placement_preview`.

`GET /api/capabilities/ltx2-control-adapters?model=<id>` returns only the
controls compatible with that host's effective installed model profile. Each
row includes `id`, `label`, guide-video `guide`, `size_bytes`, `installed`,
the additive `gated` flag (Hugging Face licence acceptance required first;
absent reads as not gated), and the exact `download_model`, `download_repo`,
`download_filename`, and `download_sha256` identity. Dev checkpoints, unknown
catalog architectures, and unsupported profiles return `422`.

`GET /api/capabilities/ltx2-camera-controls?model=<id>` answers with the same
kind of bare array. Add `?detail=1` (or `true`, or the bare flag) to get the
`Ltx2CameraControlAvailability` envelope instead — `controls`, `supported`, and
an `unsupported_reason` naming why this host has none — which is the only way
to tell "no camera controls for this checkpoint" from "none installed".

## Authentication

When `MOLD_API_KEY` is set, all API requests must include an `X-Api-Key`
header. The exemptions are `/health`, `/api/docs`, `/api/openapi.json`, and
`/api/pairing/claim` — a phone redeems a pairing ticket before it holds a key,
so that route authenticates on the one-use ticket instead — plus `/metrics`,
which is mounted outside the auth layer. `GET` and `HEAD` reads of
`/api/gallery/image/:name` and `/api/chain-jobs/:id/stages/:idx/media` also
accept a short-lived signed media ticket in place of the header (see below).

```bash
curl -H "X-Api-Key: your-secret-key" http://localhost:7680/api/status
```

Without the header (or with an invalid key), the server returns
`401 Unauthorized`:

```json
{ "error": "missing X-Api-Key header", "code": "UNAUTHORIZED" }
```

The `MOLD_API_KEY` variable supports multiple formats:

- **Single key**: `MOLD_API_KEY=my-secret`
- **Multiple keys**: `MOLD_API_KEY=key1,key2,key3`
- **File reference**: `MOLD_API_KEY=@/path/to/keys.txt` (one key per line, `#`
  comments supported)

When `MOLD_API_KEY` is unset, no authentication is required (backward
compatible).

The `mold` CLI reads `MOLD_API_KEY` from the environment and sends the header
automatically.

### Gallery media tickets

Browser and native `<video>` elements cannot attach an `X-Api-Key` header to
their own streaming and Range requests. An authenticated client can exchange
normal API-key authentication for a short-lived credential scoped to one
full-size gallery path:

```bash
curl -X POST \
  -H 'Content-Type: application/json' \
  -H 'X-Api-Key: your-secret-key' \
  -d '{"path":"/api/gallery/image/clip.mp4"}' \
  http://localhost:7680/api/gallery/media-token
```

The response is
`{"token":"...","expires_at":1234567890,"auth_required":true}`. For the
next 15 minutes, append the token as `media_token` and the Unix expiry as
`expires` to that exact gallery-image URL. It authorizes only `GET` or `HEAD`
reads, including HTTP Range requests; it cannot access another file, delete
media, or call any other endpoint. The signing secret is per server process,
the response is `no-store`, and request tracing omits the bearer query. When API
authentication is disabled, the endpoint returns `auth_required: false` and
the direct media URL needs no ticket.

### iPhone pairing tickets

An authenticated desktop or web Settings client calls
`POST /api/pairing/sessions`. Its `no-store` response includes a 256-bit random
token, two-minute Unix expiry, server instance ID, and hostname. The QR adds
the operator-confirmed reachable base URL; it does not include the API key.
Mold for iPhone posts the token to `/api/pairing/claim`, verifies that the
returned instance ID matches the scanned envelope, and stores the returned key
in the iOS Keychain. A token is removed before a successful response, so a
second or concurrent redemption fails with `PAIRING_TOKEN_INVALID`. When API
authentication is disabled, both token and returned key are `null` because the
host requires no credential.

## Rate Limiting

When `MOLD_RATE_LIMIT` is set, per-IP rate limiting is enforced with two tiers:

- **Generation tier** (configured rate): `POST /api/generate`,
  `/api/generate/stream`, `/api/generation-batches`,
  `/api/generate/placement-preview`, `/api/chain-jobs/placement-preview`,
  `/api/expand`, `/api/upscale`, `/api/upscale/stream`, `/api/models/load`,
  `/api/models/pull`, `/api/queue/:id/retry`, `/api/queue/held/sweep`, plus
  `DELETE /api/models/unload`, `PATCH /api/devices/*`, and every
  `DELETE /api/gallery/*` — a trash or purge takes DB write transactions and
  filesystem work
- **Read tier** (10x the configured rate, capped at 1000 per period): every
  `GET`, plus any route not listed above. `POST /api/generation-batches/status`
  is deliberately here: every Studio surface polls it on reconnect, and on the
  generation tier it would drain the bucket new admissions need

Health, docs, and `/metrics` endpoints are exempt from rate limiting.

Example: `MOLD_RATE_LIMIT=10/min` allows 10 generation requests per minute per
IP, and 100 read requests per minute per IP.

Supported period formats: `sec` (or `s`), `min` (or `m`), `hour` (or `h`).

Override burst size with `MOLD_RATE_LIMIT_BURST` (defaults to 2x the rate,
capped at 100).

When rate limited, the server returns `429 Too Many Requests` with a
`Retry-After` header:

```json
{ "error": "rate limit exceeded", "code": "RATE_LIMITED" }
```

## Request IDs

Every response includes an `X-Request-ID` header for correlation. If the client
sends one, it is preserved; otherwise the server generates a UUID v4.

## Quick Examples

```bash
# Generate an image
curl -X POST http://localhost:7680/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a glowing robot"}' \
  -o robot.png

# Generate with API key authentication
curl -X POST http://localhost:7680/api/generate \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: your-secret-key" \
  -d '{"prompt": "a glowing robot"}' \
  -o robot.png

# Check status
curl http://localhost:7680/api/status

# List models
curl http://localhost:7680/api/models

# List installed LoRAs compatible with a model
curl "http://localhost:7680/api/loras?model=flux-dev:q8"

# Load a specific model
curl -X POST http://localhost:7680/api/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "flux-dev:q4"}'

# Upscale an image (base64 input, raw image output)
curl -X POST http://localhost:7680/api/upscale \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"real-esrgan-x4plus:fp16\",\"image\":\"$(base64 < photo.png)\"}" \
  -o photo_4x.png

# Interactive docs
open http://localhost:7680/api/docs
```

## `/api/generate`

`POST /api/generate` returns raw image bytes for exactly one output.
`batch_size > 1` fails with HTTP 422 and `DIRECT_BATCH_UNSUPPORTED`; use
`POST /api/generation-batches` for Batch N.
The server includes an `x-mold-seed-used` header with the effective seed on
singleton responses.

A print that fails while the caller is still attached is the caller's error,
in the shape this route always had: HTTP 404 with the held child's typed code
(`MODEL_NOT_FOUND`, `UNKNOWN_MODEL`) for a model the host cannot resolve,
503 `QUEUE_FULL` for a saturated queue, otherwise 500 `INFERENCE_ERROR`
carrying the engine's own sentence. The durable row behind it is held, not
lost, so the error names the job to resume (`POST /api/queue/{job_id}/retry`)
and the batch to reconcile. SSE delivers the same failure as its terminal
`error` event, with the same code.

Only when the attached observer disappears after commit does the route return
HTTP 202 with its `GenerationBatchStatus` instead of an opaque 500; reconcile
by batch or client operation ID and do not resubmit. SSE emits
`retained: true`, code `durable_observer_detached`, after its queued job ID so
clients can enter the same reconciliation path. Queued cancellation emits the
terminal code `queued_cancelled`.

Both facades accept an optional `X-Mold-Client-Batch-Id` header carrying a
caller-chosen UUID. It is the batch's `client_batch_id`, so a retry of a lost
response under the same value is answered with the batch the first attempt
admitted (HTTP 200 with the `GenerationBatchStatus` as JSON, naming the
gallery `result.filename` once the print is done) never a second render; a
changed request under the same id is HTTP 409. Without the header every POST
is a new print.

Durable admission is the only admission. `POST /api/generate`,
`POST /api/generate/stream`, and `POST /api/generation-batches` are one path
with three delivery shapes, so a host that cannot admit durably does not
generate: it returns HTTP 503 `DURABLE_ADMISSION_UNAVAILABLE` on all three,
with a message naming the unmet requirement; a claimed queue owner, gallery
output, an authoritative Scheduler V2 dispatcher, and a usable admission
service. A degraded encrypted-media store is a separate axis and refuses only
the requests that need it, with HTTP 503 `DURABLE_MEDIA_UNAVAILABLE`; a
media-free request is unaffected. There is no `X-Mold-Operation-Id` header and
no attached, non-durable fallback.

A LoRA combined with conditioning media is an ordinary durable request: the
adapter's path and scale are sealed in the encrypted media set beside the media,
restored before the print is planned, and re-validated when the job is
dispatched, so an adapter that was moved or deleted in the meantime holds its
row with a reason naming the file rather than rendering without it.

Ordered MiniMax H3 references are durable in the same way: each reference's
descriptor (kind, probed shape, content `sha256`) stays on the queued request,
its media (whether it arrived inline, through a request-bound upload session,
or as a server path) is sealed into the encrypted media set at admission, and
the row survives a restart and replays. One-use upload handles are consumed
inside admission and never written to `mold.db`.

One request trait is refused with HTTP 422, for a reason that is not about
protocol versions: `hdr_exr_dir` names an output directory on the machine doing
inference, which an HTTP client may not choose; re-run the CLI with `--local`.
`POST /api/generate` additionally refuses `batch_size != 1` with
`DIRECT_BATCH_UNSUPPORTED`; submit siblings through
`POST /api/generation-batches`.

`POST /api/generation-batches/status` is rate-limited as a read operation;
`POST /api/queue/{id}/retry` is rate-limited with generation because it
re-queues GPU work. A retry restores the job's dispatch budget but not its
replay budget, which bounds a boot crash loop and is not an operator's to
spend.

### `GET /api/generation-batches/{id}/events`

Server-sent events carrying the authoritative state of one durable batch.
Every frame is a complete `GenerationBatchStatus` under the event name
`generation_batch`, not a delta: the stream opens with one, emits another
whenever a child commits a new authoritative state, and closes once every
child is `complete`, `failed`, `cancelled` or `held`. A client that connects
late, reconnects, or misses a frame is therefore correct from the first event
it receives, and a lagged subscriber re-reads rather than resynchronising.

This is the state channel, not a progress channel. Per-step progress and
denoise previews ride the single observer a job's own admission registered;
`POST /api/generate/stream` is that observer for a singleton, and
`GET /api/queue/{id}/preview` is the snapshot every other surface polls.

### `GET /api/queue/{id}/preview`

One live job's folded progress snapshot, or `null` before it has reported
any. `404` means the row has left the queue.

```json
{
  "step": 4,
  "total": 20,
  "stage": "Denoising",
  "weight_load": {
    "bytes_loaded": 1,
    "bytes_total": 2,
    "component": "transformer"
  },
  "download": {
    "filename": "t5xxl_fp16.safetensors",
    "file_index": 1,
    "total_files": 3,
    "bytes_downloaded": 10,
    "bytes_total": 100
  },
  "queue_position": 0,
  "preview_image": "<base64 PNG>",
  "updated_at_ms": 1756200000000
}
```

Every field except `updated_at_ms` is omitted until the job reports it. In
particular `preview_image` is absent for the whole render on a host started
with `MOLD_STEP_PREVIEW=0`, while `step`, `total` and `stage` still advance;
so read the counter from those fields rather than from the image.

### Child results

A completed child carries a `result`:

```json
{
  "filename": "mold-flux-dev-1756200000000.png",
  "original_filename": "mold-flux-dev-1756200000000-original.png",
  "seed": 4242,
  "generation_time_ms": 7500,
  "gpu": 1
}
```

`filename` and `original_filename` name the gallery rows the render published;
the second is present only when a pre-upscale original was saved separately.
`seed`, `generation_time_ms` and `gpu` are the terminal facts the settlement
recorded, and are the same values the SSE complete event carries; the seed
matters when the server chose it. All five fields are additive: a child
settled before they existed, or replayed from the committed archive, omits
what it does not know rather than reporting a zero.

### Child revisions

Every child in a `GenerationBatchStatus` carries a monotonic `revision`,
incremented by each authoritative state transition and by nothing else. Order
snapshots and events by it rather than by `updated_at_ms`: several transitions
routinely commit inside one millisecond, and a retry moves a child backward
from `held` to `accepted`, so a client that breaks that tie by timestamp can
drop the retry entirely.

`revision` is additive. A server that predates it omits the field, and `0`
also means "not yet transitioned since the migration"; treat both as absent
and fall back to the timestamp.

A retry whose response was lost is reconciled by comparing the child's current
revision against the one observed before the POST. A child that is still
`held` at a HIGHER revision was retried and held again for a new reason; one
still at the submitted revision was never retried, and the caller should keep
its retry fence rather than treat the stale snapshot as fresh.

### Held-row retention

A held row survives `queue.held_retention_days` (default 30; `0` keeps held
rows forever; env `MOLD_QUEUE_HELD_RETENTION_DAYS`) measured from when it was
held. The server sweeps hourly, and `POST /api/queue/held/sweep` runs one pass
on demand, returning `{ "purged", "remaining", "media_deferred" }`. Purging a
row releases the encrypted request media it pinned and settles its batch child
as `failed`, so a reconnecting client still sees a terminal outcome rather
than a missing print. A retry or cancel that lands before the purge wins.

The same horizon bounds settled batch summaries: a batch whose every child is
`complete`, `failed`, or `cancelled` is purged once its newest child settlement
is older than `queue.held_retention_days`, and `POST /api/generation-batches/sweep`
runs that pass on demand, returning `{ "purged", "remaining" }`. A purged batch
answers `404 GENERATION_BATCH_NOT_FOUND` from `GET /api/generation-batches/:id`
and appears under `missing.batch_ids` in the bulk status lookup; clients read
that as missing and never reopen a job they already saw settle.

```bash
curl -i -X POST http://localhost:7680/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a glowing robot in a rainy alley",
    "model": "flux-schnell:q8",
    "width": 1024,
    "height": 1024,
    "steps": 4,
    "guidance": 0.0,
    "output_format": "png"
  }' \
  -o robot.png
```

Representative headers:

```http
HTTP/1.1 200 OK
content-type: image/png
x-mold-seed-used: 42
x-mold-dimension-warning: dimensions adjusted from 1000x1000 to 1024x1024
x-mold-request-warning: dimensions adjusted from 1000x1000 to 1024x1024
```

The `x-mold-dimension-warning` header is present when the requested dimensions
were adjusted to fit model constraints (e.g. multiples of 16, pixel cap). It
carries dimension adjustments only.

The `x-mold-request-warning` header carries **every** advisory about a request
that was still accepted; dimension adjustments plus anything else, such as a
lip-dub render taking its frame count and frame rate from the reference clip
instead of the values you sent, or a filing the host could not apply. Prefer it
over the dimension header if you want to surface all of them; the streaming
endpoints deliver the same list as `info` progress events, and the header is
sent before the first SSE frame.

::: warning Do not split this header on `; `
Several advisories are joined with `; `, but the advisory text itself contains
that sequence; "…were not applied; the print was generated and saved
normally". Splitting on it turns one advisory into two dangling half-sentences.
Show the value whole: the semicolons read as ordinary punctuation. `mold`'s own
client does exactly that. Browser Fetch also combines repeated response fields
and cannot recover their original boundaries, so servers must continue sending
this as one joined field until a future structured warning encoding replaces it.
:::

### Mesh, video and audio responses

A video render returns the encoded clip itself, with `x-mold-video-frames`,
`x-mold-video-fps`, `x-mold-video-width`, `x-mold-video-height`, and (when
the container carries a soundtrack) `x-mold-video-has-audio`,
`x-mold-video-duration-ms`, `x-mold-video-audio-sample-rate`, and
`x-mold-video-audio-channels`.

Runtime provenance rides the same response, and a client-side save must record
it rather than infer it from the request: `x-mold-video-pipeline` is the recipe
that actually completed (including an implicit `auto` choice),
`x-mold-video-pipeline-provenance-sha256` pins it,
`x-mold-video-source-preprocessing` is the executed source round-trip as JSON,
and `x-mold-video-attention-path` / `x-mold-video-int8-arm` name the backend
arms that ran. `x-mold-video-video-only` is `1` when the render deliberately
carried no audio track. On a multi-GPU host every generation response also
carries `x-mold-gpu`, the ordinal that produced it.

An audio-only render (`pipeline: "t2a"`) returns the WAV itself as
`content-type: audio/wav`, never its waveform tile:

```http
HTTP/1.1 200 OK
content-type: audio/wav
x-mold-seed-used: 42
x-mold-audio-format: wav
x-mold-audio-sample-rate: 24000
x-mold-audio-channels: 2
x-mold-audio-duration-ms: 5010
x-mold-audio-thumbnail-width: 640
x-mold-audio-thumbnail-height: 360
```

`x-mold-audio-format` states the container the server actually produced; a
request may omit `output_format` and let the server normalise an audio-only
pipeline to `wav`, so the request is not evidence of what came back.

The two `x-mold-audio-thumbnail-*` headers describe the waveform PNG the
server rendered for gallery grids; audio has no dimensions of its own, and
the tile's bytes cannot ride along in a body that is already the WAV. Probe
`x-mold-audio-sample-rate` before the video headers: an audio print has no
frames, so a video-shaped probe falls through and mislabels the response.

A 3-D render returns the binary glTF itself as
`content-type: model/gltf-binary`, never its poster tile:

```http
HTTP/1.1 200 OK
content-type: model/gltf-binary
x-mold-seed-used: 42
x-mold-mesh-format: glb
x-mold-mesh-vertices: 24576
x-mold-mesh-faces: 49152
x-mold-mesh-textured: false
x-mold-mesh-poster-width: 512
x-mold-mesh-poster-height: 512
```

The `x-mold-mesh-poster-*` headers describe the PNG the server rendered for
gallery grids, exactly as the audio ones describe the waveform tile: a mesh
has no dimensions of its own, and the tile cannot ride along in a body that
is already the GLB.

**Probe order is mesh, then audio, then video — narrowest first.** Each of
these artifacts is missing whatever the next probe keys on: a mesh has no
sample rate and no frames, an audio print has no frames. A client that probes
in any other order falls through to the image branch and hands its caller
glTF or WAV bytes labelled as a picture.

### MiniMax H3 reference uploads

For remote clients, MiniMax H3 Ref2VA media should use authenticated,
request-bound streaming uploads so reference bytes stay out of the final
generation JSON, URLs, logs, and durable metadata. Read `GET /api/capabilities`
→ `reference_uploads` before using the protocol. It advertises the endpoint
paths, secret header names, file/session byte limits, the number of sessions
one identity may hold open (`max_active_sessions`), and session TTL.

1. `POST /api/generate/reference-upload-sessions` with the complete
   `GenerateRequest`, using `{ "authority": "descriptor" }` for each ordered
   reference, plus one-based `upload_references` indices.
2. `PUT /api/generate/reference-upload` once per returned slot. Send its
   one-use handle in `X-Mold-Reference-Upload`, with exact `Content-Length` and
   `Content-Type`; the response returns content-probed canonical metadata.
3. Submit the same request through `/api/generation-batches`,
   `/api/generate`, or `/api/generate/stream`, replacing each descriptor with
   `{ "authority": "upload", "handle": "…" }` and the canonical metadata. A
   session binds ONE request, so a batch of siblings needs one session per
   sibling.

All calls require the same API-key identity. Handles are bearer secrets,
expire after 30 minutes, are bound to the exact request scope and server
instance, and can be consumed once; inside admission, before the request is
journaled, so a retry of a lost `POST` under the same `client_batch_id` is
answered from the journal rather than refused for a spent handle. Cancel an abandoned session with
`DELETE /api/generate/reference-upload-sessions` using the
`X-Mold-Reference-Upload-Session` header. Each file is capped at 256 MiB and a
session at 1 GiB; use the live capability values rather than hardcoding those
limits. The Mold CLI implements this flow for repeatable
`--reference KIND=PATH` inputs, where `KIND` is `image`, `video`, or `audio`.
Small references may instead use the inline authority (capped at 32 MiB), and
trusted server-side callers may use `server_path` within the configured input
roots. Streaming uploads are the portable choice when client and server do not
share a filesystem.

## Generate Request Shape

```json
{
  "prompt": "a cat on a skateboard",
  "model": "flux-schnell:q8",
  "width": 1024,
  "height": 1024,
  "steps": 4,
  "seed": 42,
  "guidance": 0.0,
  "batch_size": 1,
  "negative_prompt": "",
  "source_image": "<base64>",
  "edit_images": ["<base64>", "<base64 reference>"],
  "strength": 0.75,
  "mask_image": "<base64>",
  "control_image": "<base64>",
  "control_model": "controlnet-canny-sd15",
  "control_scale": 1.0,
  "loras": [
    { "path": "/path/to/style.safetensors", "scale": 0.8 },
    { "path": "/path/to/detail.safetensors", "scale": 0.4 }
  ],
  "frames": 97,
  "fps": 24,
  "enable_audio": true,
  "audio_file": "<base64 wav>",
  "audio_file_path": "/srv/mold-media/voice.wav",
  "source_video": "<base64 mp4>",
  "source_video_path": "/srv/mold-media/clip.mp4",
  "keyframes": [{ "frame": 0, "image": "<base64 png>" }],
  "references": [
    {
      "kind": "image",
      "media": { "authority": "upload", "handle": "<one-use handle>" },
      "provenance": {
        "name": "hero.png",
        "sha256": "<sha256>",
        "crop": {
          "x": 0,
          "y": 224,
          "width": 2048,
          "height": 2048,
          "source_width": 2048,
          "source_height": 2496,
          "source_sha256": "<sha256 of the uncropped original>"
        }
      },
      "mime_type": "image/png",
      "width": 2048,
      "height": 2048
    }
  ],
  "pipeline": "keyframe",
  "retake_range": { "start_seconds": 1.5, "end_seconds": 3.5 },
  "spatial_upscale": "x2",
  "temporal_upscale": "x2",
  "guidance_overrides": { "stg_scale": 1.5, "stg_blocks": [28, 29] },
  "placement": { "text_encoders": { "kind": "cpu" } },
  "cfg_plus": true,
  "embed_metadata": true,
  "upscale_model": "real-esrgan-x4plus:fp16",
  "expand": false,
  "output_format": "png",
  "title": "Smurf Village at Dusk",
  "tags": ["smurfs", "blue hour"],
  "collection": { "name": "Blue Period" }
}
```

`prompt` is the only field without a default, and it is required in every case
but one. All other fields have defaults or model-specific validation.

::: tip Optional prompt (LTX-2 / LTX-Video image-to-video)
An empty or whitespace-only `prompt` is accepted **only** when both of these
hold:

1. the resolved model family is `ltx2` or `ltx-video`, and
2. the request carries visual conditioning; `source_image`, a non-empty
   `keyframes[]`, `source_video` / `source_video_path`, or `extend_video` /
   `extend_video_path`.

Anything else (pure text-to-video, or any image family even with a
`source_image`) still fails with `prompt must not be empty`. For `cv:` / `hf:`
catalog IDs the family comes from the server's catalog resolution, so a
promptless request works only on a host that can resolve the model to one of
those two families.

LTX-2's Gemma encoder pads to a fixed 1,024-token context and replaces padded
positions with learned register embeddings, so `""` is a trained context rather
than a degenerate one. Two consequences worth passing on to your users: it
**does not reduce VRAM use** (the prompt context is a fixed-size tensor), and it
usually yields near-static output because nothing describes the motion. An
empty prompt also suppresses prompt expansion (`expand` is forced to `false`
rather than letting the expander invent a prompt) and is not written to prompt
history.
:::

A host that admits generation advertises `queue.heterogeneous_batch_max_outputs`,
the per-operation child limit and the single bit that says this host
generates at all. Its absence means every generation route returns
`503 DURABLE_ADMISSION_UNAVAILABLE`. Clients must also honor the exact
`durable_media` capability for requests carrying source or identity media.

## `/api/generation-batches`

`POST /api/generation-batches` durably commits 1–64 ordered singleton
`GenerateRequest` children before model resolution, downloads, or inference.
Every child must set `batch_size: 1`; clients chunk larger Batch N requests.
The body is `{ "client_batch_id": "<uuid>", "requests": [...] }`. A new
operation returns HTTP 202; replaying the same client ID and identical requests
returns the existing status, while changed requests return HTTP 409.

If the admission response is lost, recover it with
`GET /api/generation-batches/by-client/{client_batch_id}`. Poll one batch with
`GET /api/generation-batches/{id}` or reconcile a bounded set with
`POST /api/generation-batches/status`. That bulk request accepts at most 256
unique UUID identities across `client_batch_ids` and `batch_ids`; clients must
chunk larger recovery sets. Children expose `accepted`, `paused`, `held`,
`running`, `cancelling`, `complete`, `failed`, or `cancelled`; completed
children name their gallery `result.filename`. A `held` child carries the
machine's sentence in `error` and, when the hold has a typed cause, its code in
additive `error_code` (`MODEL_NOT_FOUND`, `UNKNOWN_MODEL`, …), which is the field a
client's missing-model pull offer classifies on. Held retryable work remains in
the durable queue with its error and can be resumed with
`POST /api/queue/{job_id}/retry`. Its JSON body must repeat the complete
authority captured from the admitted batch status:
`{ "instance_id": "...", "batch_id": "...", "client_batch_id": "...", "job_id": "..." }`.
The path and body job IDs must match; the server transactionally fences the
serving instance and batch/client/job identity before returning HTTP 202. Cancel queued work with
`DELETE /api/queue/{job_id}`, or the whole print run with
`DELETE /api/generation-batches/{id}`. This applies the same per-child cancel
applied to every non-terminal child under one durable transition, returning
the authoritative `GenerationBatchStatus` as of the revocation. A child that
had already settled keeps its outcome, and running inference stops at the next
model safe point.

Important fields:

| Field                                                                                                   | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `source_image`, `mask_image`                                                                            | img2img/inpainting source media as base64 PNG/JPEG bytes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `edit_images`                                                                                           | ordered Qwen-Image-Edit target/reference images; use this instead of `source_image` for `qwen-image-edit`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `references`                                                                                            | ordered MiniMax H3 Ref2VA image/video/audio references, each a descriptor (kind, probed shape, content `sha256`) plus its media as `inline` bytes, a request-bound `upload` handle, or a trusted `server_path`. Admission resolves every media authority, keeps the descriptor on the queued request, and seals the bytes into the encrypted queue-media store; only the descriptors and digests survive into saved metadata. An image reference may also carry an additive `provenance.crop` (`x`, `y`, `width`, `height`, `source_width`, `source_height`, `source_sha256`): the client already cropped the bytes, so this is provenance the server validates as a non-degenerate rectangle inside its source whose size equals the reference's own — a projection that reached the server uncropped is refused as `MINIMAX_H3_REFERENCE_CROP` — and then retains verbatim into saved metadata so Reuse settings can restore it.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `id_image`, `id_images`, `id_image_name(s)`, `id_weight`, `id_start_step`, `true_cfg`, `cfg_start_step` | Face-identity conditioning (PuLID). `id_image` is base64 PNG/JPEG bytes, bounds-checked from its header alone (≤ 16 MiB encoded, ≤ 8192 px per axis, ≤ 32 MP) before any decode; `id_images` is its plural shape, up to `ID_IMAGES_MAX` (4) references of one person averaged post-IDFormer into one identity, mutually exclusive with `id_image` (whole-set byte/pixel caps sit below the per-image limits times the count; a photograph with no detectable face refuses the whole request and names its one-based position). `id_weight` is `0.0..=3.0` (absent = `1.0`); `id_start_step` is the first identity-conditioned denoise step and must be `< steps` (absent = `0`); both family-blind. `true_cfg` and `cfg_start_step` are FLUX only: `true_cfg` is `1.0..=10.0` (absent or `1.0` = off) and restores a real negative branch on FLUX's guidance-distilled model, reusing `negative_prompt`; `cfg_start_step` is the first step the branch runs at (absent = `1`) and requires `true_cfg`. An SDXL identity request refuses both fields outright; its ordinary `guidance` is already the classifier-free scale, so `negative_prompt` works unconditionally with no flag needed. Qualification is family-wide and derived from the manifest, so a new FLUX.1 or SDXL entry inherits it: every FLUX.1 checkpoint takes PuLID-FLUX v0.9.1, and every SDXL checkpoint except `sdxl-turbo:fp16` takes PuLID v1.1 — see [Identity photos](/guide/identity#which-models). It refuses combinations with a LoRA or an img2img `source_image` on either family. Any companion field without its parent (`id_weight`/`id_start_step`/`id_image_name` without `id_image`, `id_image_names` without `id_images`, `cfg_start_step` without `true_cfg`) is an error, not an ignored field; sending both `id_image` and `id_images`, or `true_cfg` without an active identity, is likewise an error. A server that cannot execute identity conditioning refuses any request carrying an identity field rather than rendering without the face, with two distinct messages: a build without the `pulid` feature says it was built without PuLID support (a differently compiled binary is needed), and a build that links `pulid` while the runtime adapter for that family is still pending says identity conditioning is not available in this build yet (a newer one is needed). No release advertises the capability until the matching adapter lands, so check `/api/models[].supports_identity` before offering the control and never infer support from the build feature or the checkpoint's family alone. `id_images` and `true_cfg` are additionally gated by `GET /api/capabilities` → `identity` (`{ multi_photo, max_photos, true_cfg }`); absence reads as no, so a client that needs either shape probes first and refuses by name rather than sending fields an older server would silently drop. An identity request materializes its family's asset bundle itself; `pulid-flux` for FLUX, `pulid-sdxl` for SDXL, sharing four of their five files: `POST /api/generate/placement-preview` reports anything missing under `pending_downloads` (kinds `identity_adapter`, `identity_vision_encoder`, `face_detector`, `face_recognizer`, `face_parser`) without fetching it, and admission downloads it after the InsightFace license gate. `id_weight` `0` is inert; it plans and downloads nothing, on either family. |
| `control_image`, `control_model`, `control_scale`                                                       | SD1.5 ControlNet conditioning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `lora`, `loras`                                                                                         | singular legacy adapter or repeatable stack; `loras[]` wins when both are set                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `frames`, `fps`, `output_format`                                                                        | video/animation length and encoder selection                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `enable_audio`, `audio_file`, `audio_file_path`                                                         | LTX-2 synchronized audio toggle and audio-to-video input. Path input is server-local and requires configured `media_roots` / `MOLD_MEDIA_ROOTS`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `source_video`, `source_video_path`, `retake_range`                                                     | LTX-2 retake/video-conditioning source and seconds range. Path input is server-local and cannot be combined with inline base64 bytes.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `extend_video`, `extend_video_path`, `extend_overlap_frames`                                            | Continue an existing clip in one request: the last `extend_overlap_frames` frames are re-encoded as conditioning and the stitched result drops the duplicated overlap. Mutually exclusive with `source_video`, `source_image`, and `keyframes`. The overlap must sit on the family's frame grid and be strictly below `frames`. When omitted, the server materializes the family default (`extend_default_overlap_frames`: 17 for LTX-2, 1 for Wan) into the request at admission so saved provenance records the real overlap. Resolution and fps must match the source clip. Path input is server-local and requires configured `media_roots` / `MOLD_MEDIA_ROOTS`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `keyframes`, `pipeline`                                                                                 | Keyframe conditioning and explicit LTX-2 pipeline selection (`one-stage`, `two-stage`, `two-stage-hq`, `distilled`, `ic-lora`, `keyframe`, `a2-vid`, `retake`, `lip-dub`, `t2a`). Wan first/last-frame interpolation (FLF-capable checkpoints) uses a two-entry `keyframes` list anchoring pixel frames 0 and F-1; any other keyframe layout is refused at admission for Wan.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `ic_lora_control`                                                                                       | Canonical official control ID. LTX-2.0 checkpoints accept `union`, `pose`, `detailer`; LTX-2.3 accepts `union`, `motion-track`, `lipdub`, `hdr`. Every ID implies `pipeline=ic-lora` except `lipdub`, which selects the dedicated `lip-dub` pipeline. Requires source video and precedes custom `loras[]`. Use `GET /api/capabilities/ltx2-control-adapters?model=<id>` for the controls compatible with an installed model.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `spatial_upscale`, `temporal_upscale`                                                                   | LTX-2 latent upscaling modes such as `x1-5` and `x2`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `guidance_overrides`                                                                                    | Additive LTX-2 multimodal-guider overrides: `stg_scale`, `stg_blocks[]`, `rescale_scale`, `modality_scale`, `skip_step`. Each omitted field keeps the pipeline default.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `scheduler`                                                                                             | Denoise solver. Wan accepts `uni-pc` (default), `euler` (the lightx2v 4-step Lightning recipe's solver), and `dpm-pp`; the UNet schedulers `ddim` / `euler-ancestral` are rejected for Wan and the Wan solvers are rejected for every other family.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `sample_shift`                                                                                          | Wan flow shift, the family's primary quality/character knob. Additive; absent keeps the tier default. Precedence: request > `MOLD_WAN_SHIFT` > per-tier default. Rejected for non-Wan families and recorded in saved metadata.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `distill_strength_high`, `distill_strength_low`                                                         | Per-expert scale for Wan A14B manifest Lightning adapters (absent = 1.0). Refused, not ignored, on tiers that ship no distill in the addressed slot.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `placement`                                                                                             | per-request device placement override; persisted defaults use `/api/config/model/:name/placement`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `cfg_plus`                                                                                              | CFG++ guidance for supported SD-family scheduler paths                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `embed_metadata`                                                                                        | override config/env metadata embedding for this request                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `batch_id`, `batch_index`, `batch_count`                                                                | optional native prepared-batch identity plus one-based sibling position/total; copied unchanged into complete-event and Gallery metadata                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `source_fit`                                                                                            | additive, engine-ignored provenance recording the client-side source-image resize/crop policy; echoed verbatim into gallery `OutputMetadata.source_fit` so Reuse settings and running-job selection can restore the crop choice                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `title`, `tags`, `collection`                                                                           | creation-time filing: the print's name plus the tags and collection it lands in. `title` is ≤ 120 characters, embedded in metadata, seeded into the gallery row, and folded into the default filename as `~slug`. `tags` holds at most 20 distinct names of 1–64 characters each. `collection` is `{ "id": "…" }` or `{ "name": "…" }`; a name resolves by slug and is created when absent. See [Creation-time filing](#creation-time-filing).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `upscale_model`                                                                                         | post-generation Real-ESRGAN model applied before returning images                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

`guidance_overrides` is optional and every field inside it is optional. An
absent field keeps the engine's per-(pipeline, stage) constant, so a request
without the object reproduces pre-override outputs exactly. The server rejects
the object for non-LTX-2 families and bounds each value (`rescale_scale` is
`0..=1`; `stg_scale` and `modality_scale` are `0..=10`; `skip_step` is `0..=8`;
`stg_blocks` holds up to 8 distinct in-range indices) with HTTP 422 before any
queue work. Only pipelines that run the multimodal guider (`two-stage`,
`two-stage-hq`, `keyframe`, `a2-vid`) read the overrides, and a guider the
pipeline disables entirely stays disabled. Accepted values are recorded in
gallery metadata under the same key.

When `upscale_model` is set, the server gallery retains both artifacts as
`-original` and `-upscaled` files. The SSE `complete` event returns the
upscaled image in `image` and includes additive `original_image`,
`original_width`, and `original_height` fields so remote clients can mirror
the pair. Gallery metadata exposes the saved file size in `width` / `height`
and the reusable pre-upscale canvas in `generation_width` /
`generation_height`.

### Creation-time filing

`title`, `tags`, and `collection` let a print arrive already organized instead
of being filed afterwards. They are additive and optional (a request that
omits them behaves exactly as before) and the same three fields are accepted
on the chain body, where they describe the **stitched** print a sequence
produces; intermediate clips never reach the gallery and are never filed.
Batch and prepared siblings inherit their parent's filing.

Validation happens at admission, before any queue work:

- at most **20** distinct tags after normalization, each **1–64** Unicode
  scalars. Runs of whitespace and whitespace-like controls collapse to a
  single space; any other control character is a `422`. A leading `#` is an
  ordinary character; the server never strips it, so `#smurfs` and `smurfs`
  are different tags.
- tags match case-insensitively, so `Smurfs` and `smurfs` are one tag.
- `collection` is either `{ "id": "…" }` (resolved at admission) or
  `{ "name": "…" }` (resolved by slug on the serving host, created when
  absent). Creating by name is what makes one name mean one collection across
  a fleet.

Filing is applied **once**, when the print is published, and only as the
gallery row is inserted. Organization is user-owned from that moment: a
reconcile, an import, or a re-publication never resurrects a tag someone
removed.

The server never auto-tags. mold's own clients optionally add the title's slug
as a tag before sending (`generate.auto_tag_title` on the CLI and TUI, **Tag
new prints with their title** in the web, desktop, and iPhone apps), which is
why the tag is always visible in the request rather than invented downstream.

Nothing about filing can fail a render. A host running with
`MOLD_DB_DISABLE=1`, or a `{ "id": … }` collection deleted between listing and
generation, drops the filing, publishes the print anyway, and says so through the
`x-mold-request-warning` response header documented above and the additive
`request_warnings` list on `GenerateResponse` / `ChainResponse`.

The exhaustive schema for enums and nested objects is served by the running
server at `/api/docs` and `/api/openapi.json`.

## `/api/generate/estimate`

`POST /api/generate/estimate` accepts the same JSON shape as
`/api/generate` and returns the server's current peak-memory estimate for that
request. The estimate accounts for model files, resolution, batch, frames,
placement, and runtime load strategy.

```bash
curl -X POST http://localhost:7680/api/generate/estimate \
  -H "Content-Type: application/json" \
  -d '{"model":"flux-dev:q8","prompt":"a cat","width":1024,"height":1024}'
```

The response includes `peak_memory_bytes`, `activation_memory_bytes`,
`load_strategy`, and optional available-memory fit fields.

## `/api/generate/placement-preview`

`POST /api/generate/placement-preview` runs the authoritative scheduler against
an exact request without reserving, downloading, or enqueueing anything. The
body is `{ "request": <GenerateRequest>, "copies": 1 }`; `copies` previews a
batch's siblings. `POST /api/chain-jobs/placement-preview` is the sequence
peer, returning one `stage_candidates[]` entry per stage (and `copy_index` for
repeated sibling work). Feature-detect both with
`capabilities.dispatch.request_placement_preview`.

The response is versioned and carries `authoritative`, `state_version`,
`plan_version`, and an `outcome`:

- `planned` — `candidate` names the `device_id`, the frozen
  `execution_fingerprint`, `predicted_start_after_ms` /
  `predicted_completion_after_ms`, `setup_ms` / `setup_kind`, and
  `estimate_confidence`. `pending_downloads[]` lists dependencies admission
  will materialize first (`kind`, `name`, `repo`, `bytes`, the
  `install_model` to pull, and any `licenses` still blocking it); the preview
  never starts a download itself.
- `infeasible` — `reason` says why, and `missing_components[]` names the
  concrete absent model components with the `repair_model` that installs them.
  This is what a client classifies to tell "nobody has this model" apart from a
  capacity refusal or a policy block; only the first may be answered with a
  pull. Queue nothing on `infeasible`.
- `unsupported` — a non-authoritative answer (today: local prompt-expansion and
  post-generation-upscale utility work). A strictly valid version-1
  `unsupported`, and legacy `404`/`405`, may retain compatible routing; every
  other HTTP status, malformed response, or exhausted transient retry must
  queue nothing.

Probes redact prompts and media but retain server-local LoRA paths the
feasibility answer needs.

## `/api/models/:model/components`

`GET /api/models/:model/components` reports the component assets the server
expects for a model and whether each one is present. The Generate UI uses this
to highlight missing text encoders, VAEs, transformers, and companion files
with a path back to the model catalog.

```bash
curl "http://localhost:7680/api/models/flux-dev:q8/components"
```

## `DELETE /api/models/:model`

`DELETE /api/models/:model` removes a downloaded model; the HTTP counterpart
of `mold rm`. Several models share components (T5/CLIP/Qwen encoders, VAEs)
under the models directory, so removal ref-counts every file across all
installed models and deletes only files exclusively owned by the target;
shared components still referenced by another downloaded model are kept.
Hardlinked hf-hub cache blobs are cleaned up too, so `freed_bytes` reflects
real disk savings.

```bash
curl -X DELETE http://localhost:7680/api/models/flux-schnell:q8
```

```json
{
  "removed": ["/models/flux-schnell-q8/flux1-schnell-Q8_0.gguf"],
  "kept": [
    {
      "component": "/models/shared/flux/ae.safetensors",
      "used_by": ["flux-dev:q8"]
    }
  ],
  "freed_bytes": 12726374912
}
```

Returns `404` (`UNKNOWN_MODEL`) when the model isn't installed, and `409`
(`MODEL_LOADED`) while the model is GPU-resident; unload it first via
`DELETE /api/models/unload`. This is a destructive endpoint; pair with
`MOLD_API_KEY` when the server is exposed beyond localhost.

## `/api/gallery`

`GET /api/gallery` lists this host's saved prints. `?view=library` (the
default) hides trashed rows and `?view=trash` shows only them; `?filename=`
narrows to one print. The listing supports conditional GET — it sends an
`ETag` and answers `304` to a matching `If-None-Match`, so a polling client
does no work on an unchanged gallery — and every row carries `media_version`,
the identity clients key a thumbnail or media cache on. Both are advertised as
`capabilities.gallery.conditional_get` and `.media_version`. Thumbnails carry
their own per-rendition ETag.

### Library organization

Organization state lives in each host's own metadata DB, so a client holding
several machines merges it itself: collections by `slug`, tags
case-insensitively. Every route below returns
`501 GALLERY_ORGANIZATION_UNAVAILABLE` when the metadata DB is disabled, and
the whole surface is gated on `capabilities.gallery.organize` —
a host that reports it false hides these controls and keeps hard-delete
wording.

- `PATCH /api/gallery/image/:name` edits one print's `title`, `favorite`, and
  `tags`. A blank title clears it; renaming never renames the file.
- `POST /api/gallery/organize` applies one edit to many prints in a single
  transaction. `POST /api/gallery/mutations` is the replay-safe bulk form:
  it folds bulk title edits and portable collection slugs into one request and
  a retry with the same operation id returns the retained result instead of
  duplicating work.
- `GET|POST /api/gallery/collections` lists collections (`id`, `name`, `slug`,
  optional `description` and `cover_filename`, `hidden`, `count`,
  `created_at`, `updated_at`) and creates one.
  `GET|PATCH|DELETE /api/gallery/collections/:id` reads one with its ordered
  `filenames`, renames/describes/re-covers it, or deletes it — deleting a
  collection leaves its prints alone. `PUT /api/gallery/collections/:id/items`
  adds and removes members. A `hidden` collection stays visible and openable
  in Collections, but its members are excluded from the default grid and from
  text search until the collection is drilled into.
- `GET /api/gallery/tags` returns every tag with its use count (trashed prints
  included). `PATCH /api/gallery/tags/:name` renames a tag, merging into an
  existing one when the new name is taken; `DELETE /api/gallery/tags/:name`
  removes it from every print.

### Trash

`DELETE /api/gallery/image/:name` moves the print to `<output_dir>/.trash/`
and flags its row, emitting `gallery_trashed`. `?permanent=true` — and every
delete on a host with the metadata DB disabled — unlinks the bytes, cached
sidecars, and row for good, emitting `gallery_removed`. Trash support is
advertised as `capabilities.gallery.trash` (`enabled`, `retention_days`).

- `POST /api/gallery/trash` trashes several prints in one call, and
  `POST /api/gallery/trash/restore` returns them to the live gallery. Each
  stops at the first failure, names the filename, and leaves earlier work
  applied. Restoring onto a filename a live print already holds is
  `409 GALLERY_RESTORE_CONFLICT`.
- `POST /api/gallery/trash/delete-forever` permanently removes live or trashed
  prints; `DELETE /api/gallery/trash` empties the trash now.
- `POST /api/gallery/trash/sweep` runs one retention pass immediately. The
  server also runs it at startup and hourly against
  `gallery.trash_retention_days`, read fresh from the live config (`0` keeps
  forever, default 30). Edit it like any other config key through
  `PUT /api/config/gallery.trash_retention_days`.

### Export and import

`GET /api/gallery/export-options` reports the animation formats this build can
transcode a gallery MP4 into (`gif`, `apng`, and `webp` when the `webp` feature
is on) plus the GIF playback and repeat options.
`POST /api/gallery/export/:name` performs one such transcode; one export runs
at a time per server process. `PUT /api/gallery/import/:name` streams an
already-encoded print into the gallery using a fixed binary envelope —
`u32 metadata_len`, `u64 file_len`, the metadata JSON, then exactly `file_len`
bytes — so a native client can mirror a print with its metadata and
cross-host filename identity without the server buffering a large video.

## `/api/config`

The HTTP counterpart of the `mold config` CLI verbs. Config values live in
two stores (`config.toml` for bootstrap/paths/credentials and the settings
DB for user preferences) with `MOLD_*` environment variables overriding
both at runtime. Every row carries a `source` tag saying which surface owns
it; rows with `source: "env"` also carry the overriding variable name.

`GET /api/config` lists every effective row (like `mold config list --json`):

```json
{
  "profile": "default",
  "entries": [
    { "key": "models_dir", "value": "~/.mold/models", "source": "file" },
    { "key": "expand.enabled", "value": false, "source": "db" },
    {
      "key": "embed_metadata",
      "value": true,
      "source": "env",
      "env_var": "MOLD_EMBED_METADATA"
    }
  ]
}
```

`GET /api/config/:key` reads one row. `PUT /api/config/:key` sets it,
routed by surface exactly like `mold config set`; DB-backed keys
(`expand.*`, generation defaults, `models.<name>.<pref>`) land in the
settings DB for the active profile, file keys rewrite `config.toml`:

```bash
curl -X PUT http://localhost:7680/api/config/default_steps \
  -H "Content-Type: application/json" -d '{"value": 12}'
```

Env-overridden keys reject writes with `403` (`ENV_OVERRIDDEN`) naming the
variable to unset; unknown keys and out-of-range values return `422`.
`PUT /api/config/output_dir` is refused with `409` (`RESTART_REQUIRED`) for the
life of the process: the chain runner, queued attempts, the gallery publication
gate, and the HTTP routes all agree on the directory captured at boot, so the
supported editor is an offline `mold config set output_dir <path>` followed by
a restart. `GET /api/config/:key` answers `404` for a key it does not know.

Every `ConfigEntry` row (from `GET /api/config` and `GET /api/config/:key`)
carries `key`, the typed `value`, `source` (`db`, `file`, `env`, or `default`),
`env_var` when `source` is `env`, and the additive `restart_required` flag —
true for `scheduler.*`, whose persisted value is read when the coordinator
starts and does not change the running process.

`DELETE /api/config/:key` resets a DB-backed key like `mold config reset`
(drops the row for the active profile) and responds with the fallback value
(`source: "default"`). File-backed keys return `422` (`FILE_BACKED_KEY`);
edit those via `PUT` instead.

`GET /api/config/profiles` lists settings profiles and the active one;
`PUT /api/config/profile` with `{"name":"dev"}` switches the stored active
profile (a `MOLD_PROFILE` env var still wins at runtime):

```bash
curl -X PUT http://localhost:7680/api/config/profile \
  -H "Content-Type: application/json" -d '{"name":"dev"}'
```

DB-requiring operations return `503` (`CONFIG_UNAVAILABLE`) when the
metadata DB is disabled (`MOLD_DB_DISABLE=1`).

## `/api/queue`

`GET /api/queue` returns `entries`: the queued, running, **and held** rows.
Running jobs carry their actual `gpu`; queued jobs carry an
optional `target_gpu` so UI clients can render one lane per GPU plus an
automatic lane. Current authoritative V2 servers also return a nullable,
additive `plan` snapshot with versioned stable-device lanes, ordinary
generation plus scheduler-owned utility and durable-chain work items, estimated
start/finish times, confidence, blocked reasons, and the next tentative replan
deadline. Clients must treat it as advisory: the server revalidates the exact
execution fingerprint and frozen artifacts before CUDA.

A held row is durable work that exceeded an attempt cap or whose recorded
request could not be reconciled: listed so an operator can see it, never
auto-run. It carries `held_reason` and its clearer alias `error` (the
preparation refusal's own sentence), `retryable` — whether
`POST /api/queue/:id/retry` may safely resume it — and `dispatch_attempts`, how
many times a worker claimed it for execution. `position` answers "how many jobs
are ahead of me", and a held row is ahead of nobody, so numbering skips it and
it keeps the position of the next schedulable row; read `state` first and never
render a held row's position.

The durable listing is paged: `?limit=` is a positive page size bounded by the
runtime `queue_capacity` (omit it to use `queue_capacity`) and `?cursor=` is
the opaque exclusive cursor the preceding page returned. An explicitly bounded
request also returns `page` (`limit`, `offset`, `returned`, and `next_cursor`
while more remain) and `live_only_entries` — active jobs that intentionally
have no durable row, such as MiniMax H3, identity, reference-authority, and
oversized requests. That set is repeated on every explicit page and bounded by
the runtime queue capacity, so clients merge both arrays **by job id** before
reconciling local work. Invalid pagination is `400`.

Every row (in the listing and in the single-job read below) also carries its
durable batch identity when it has one: `batch_id`, the client-minted
`client_batch_id`, and the one-based `batch_index`. `POST /api/queue/:id/retry`
requires the whole authority (`instance_id`, `batch_id`, `client_batch_id`,
`job_id`) and only `instance_id` belongs to the server, so these three are what
let a client holding a bare job id compose a retry. A row that was admitted
outside a batch omits all three.

Use `GET /api/queue/:id` to read ONE job in full, settings included:

```bash
curl http://localhost:7680/api/queue/00000000-0000-0000-0000-000000000000
```

```json
{
  "job": {
    "id": "00000000-0000-0000-0000-000000000000",
    "model": "flux-dev:q8",
    "state": "queued",
    "position": 3,
    "durable": true,
    "metadata": {
      "prompt": "a lighthouse in a storm",
      "width": 1024,
      "steps": 28
    }
  },
  "work_item": { "work_id": "00000000-...", "blocked_reason": "preparing" }
}
```

The listing is deliberately payload-free (it never reads a request body per
row) so a durably admitted job carries no `metadata` there until it is
dispatched. This endpoint reads that one body and returns the same
metadata shape a replayed job describes itself with; media payloads are not
part of it. `work_item` is the planner's own entry for the job when it has
placed one. Unknown ids return `404` with `QUEUE_JOB_NOT_FOUND`. `position`
comes from the same bounded durable window the listing pages by default; a row
beyond that window reports the window's length, exactly as it would be absent
from the listing's first page.

Use `PATCH /api/queue/:id` to update a queued job's preferred lane and/or its
0-based position among queued jobs:

```bash
curl -X PATCH http://localhost:7680/api/queue/00000000-0000-0000-0000-000000000000 \
  -H "Content-Type: application/json" \
  -d '{"hard_pinned_device_id":"cuda:0123...","position":1}'
```

Set `target_gpu` to `null` to return the queued job to automatic placement.
`hard_pinned_device_id` accepts the opaque ID from `/api/devices`; send `null`
to return to Auto. If both ordinal and stable-ID pins are supplied, they must
name the same device.
Omitting either field leaves it unchanged. `position` is clamped to the current
queued range, so a large value sends a job to the back. Reordering changes real
dispatch priority, not only the listing returned by `GET /api/queue`.
Already-running jobs reject lane and position changes.

Use `DELETE /api/queue/:id` to cancel a queued or running singleton job:

```bash
curl -X DELETE http://localhost:7680/api/queue/00000000-0000-0000-0000-000000000000
```

Returns `204 No Content` as soon as cancellation authority is revoked and `404`
for unknown ids. Queued work is removed immediately; running work stops at the
next model safe point. Feature-detect active support through
`queue.cooperative_cancellation`. The waiting client observes a terminal
`CANCELLED` error and a streaming connection receives an `error` event before
it closes.

## `/api/history`

`GET /api/history` returns recent prompt history from the metadata DB, newest
first. `?query=` filters by case-insensitive prompt substring; `?limit=`
bounds the row count (default 50, max 500). `used_at` is Unix epoch
milliseconds.

The server records history automatically for accepted `POST /api/generate`,
`POST /api/generate/stream`, and `POST /api/generation-batches` requests. It
stores the typed prompt before expansion, negative prompt, and model.
Consecutive identical rows are collapsed, so retries do not duplicate history.

```bash
curl "http://localhost:7680/api/history?query=sunset&limit=10"
```

```json
{
  "entries": [
    {
      "prompt": "sunset over sea",
      "model": "flux-dev:q8",
      "used_at": 1700000000000
    }
  ]
}
```

`DELETE /api/history` clears the history (`204 No Content`). Pass `?keep=N`
to trim to the most recent N entries instead:

```bash
curl -X DELETE "http://localhost:7680/api/history?keep=100"
```

Both endpoints return `503` (`HISTORY_UNAVAILABLE`) when the metadata DB is
disabled (`MOLD_DB_DISABLE=1`).

## `/api/loras`

`GET /api/loras` returns installed LoRA adapters. Add `?model=<name>` to
restrict the list to the model family's compatible LoRAs. Use the returned
`path` values in `loras[].path` on `/api/generate` or `/api/generate/stream`.

```bash
curl "http://localhost:7680/api/loras?model=realistic-vision-v5:fp16"
```

## `/api/generate/stream`

The `/api/generate/stream` endpoint sends Server-Sent Events for progress:

```text
event: progress
data: {"type":"queued","position":1}

event: progress
data: {"type":"stage_start","name":"Loading model weights"}

event: progress
data: {"type":"denoise_step","step":1,"total":25,"elapsed_ms":640}

event: progress
data: {"type":"preview","image":"<base64 PNG>","step":1,"total":25}

event: complete
data: {"image":"<base64 PNG>","format":"png","width":1024,"height":1024,"seed_used":42,"generation_time_ms":12345,"model":"flux-dev:q4"}
```

`preview` events are live latent previews for FLUX.1, Flux.2, Z-Image, and Wan (video previews project the clip's middle latent frame):
a small PNG at latent resolution (~width/8 × height/8 for most families;
Wan 2.2 TI2V's VAE compresses 16×, so ~width/16) produced by a linear
latent→RGB projection; no VAE involved, so the cost per step is negligible.
Emitted at most every ~700 ms plus always on the final step; clients upscale
and blur it. Disable with `MOLD_STEP_PREVIEW=0` on the server.

Typical terminal usage:

```bash
curl -N http://localhost:7680/api/generate/stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a glowing robot",
    "model": "flux-dev:q4",
    "steps": 25,
    "width": 1024,
    "height": 1024
  }'
```

The final `complete` event is `SseCompleteEvent`, a flattened single-artifact
shape — **not** `GenerateResponse`, which nests an `images[]` array. It carries
one base64 `image` (video bytes for a video render), `format`, `width`,
`height`, `seed_used`, `generation_time_ms`, and `model`, plus the optional
`original_image` / `original_width` / `original_height` of a post-generation
upscale, the `video_*` block whose `video_frames` marks a video response, and
`request_warnings` — the same advisories the JSON path returns on
`x-mold-request-warning`, which a streaming client has no response headers to
read. Streaming is singleton-only; Batch N uses the durable, pollable
`/api/generation-batches` lifecycle above.

::: tip RunPod Note
RunPod's proxy can close a long-lived generation response. For reliable delivery,
submit through `POST /api/generation-batches` and reconcile the returned durable
batch status; the accepted work survives a client or proxy disconnect. Treat the
SSE streaming endpoint as live progress, not as the durability boundary.
:::

## `/api/events`

`GET /api/events` is a single SSE stream of **server-wide** lifecycle events:
generation jobs, gallery changes, versioned queue replans, and semantic device
lifecycle/health transitions. Raw utilization and memory telemetry remain on
`GET /api/resources/stream`. Frames use the event name `event` with an
internally tagged JSON payload:

```text
event: event
data: {"type":"job_queued","id":"6f9c…","model":"flux-dev:q4"}

event: event
data: {"type":"job_started","id":"6f9c…","model":"flux-dev:q4","gpu":0}

event: event
data: {"type":"gallery_added","filename":"mold-flux-dev-q4-1752300000000.png","image":{"filename":"…","metadata":{…},"timestamp":1752300000,"format":"png","size_bytes":1830421}}

event: event
data: {"type":"job_ended","id":"6f9c…"}

event: event
data: {"type":"gallery_removed","filename":"mold-flux-dev-q4-1752300000000.png"}

event: event
data: {"type":"chain_job_queued","id":"550e8400-…","model":"ltx-2-19b-distilled:fp8","stage_count":3}

event: event
data: {"type":"chain_job_started","id":"550e8400-…","model":"ltx-2-19b-distilled:fp8"}

event: event
data: {"type":"chain_job_ended","id":"550e8400-…","state":"completed"}
```

Event semantics:

| `type`                        | Meaning                                                                                                                                                                                                                                                                                          |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `job_queued`                  | A generation was accepted into the queue (`id`, `model`).                                                                                                                                                                                                                                        |
| `job_started`                 | A worker began the job. `gpu` is the ordinal on multi-GPU servers, omitted on single-GPU.                                                                                                                                                                                                        |
| `job_ended`                   | The job left the queue for **any** reason; completed, errored, or cancelled. Use the per-job stream for outcomes; `gallery_added` is the durable success signal.                                                                                                                                 |
| `gallery_added`               | A new output landed on disk. `image` carries the full gallery row when the metadata DB recorded it, including any tags and collection the request filed it under, so a client can insert it in place without refetching. When the DB is disabled `image` is omitted; refetch `GET /api/gallery`. |
| `gallery_removed`             | An output was deleted **permanently** — `DELETE /api/gallery/image/:name?permanent=true`, `POST /api/gallery/trash/delete-forever`, or a trashed print reaching the end of its retention. The default `DELETE` emits `gallery_trashed` instead.                                                  |
| `gallery_trashed`             | An output moved to the trash. It leaves the live listing and appears under `GET /api/gallery?view=trash`.                                                                                                                                                                                        |
| `gallery_restored`            | A trashed output returned to the live gallery. `image` carries the restored row when the DB recorded it; absent means refetch.                                                                                                                                                                   |
| `gallery_updated`             | A row's organization changed (title, favorite, tags, or collection membership). `image` carries the refreshed row when the DB recorded it; absent means refetch.                                                                                                                                 |
| `gallery_collections_changed` | A collection was created, renamed, re-covered, deleted, or had its membership changed. Refetch `GET /api/gallery/collections`. Carries no fields.                                                                                                                                                |
| `job_state_committed`         | One durable batch child committed a new authoritative state (`id`). Emitted only **after** the SQLite transaction, so a reconnecting client may safely reconcile that child through `/api/generation-batches/status`.                                                                            |
| `generation_states_committed` | One transaction committed authoritative state for several durable children. Reconcile the host once; a per-child event would turn bulk cancellation into an event storm. Carries no fields.                                                                                                      |
| `queue_paused`                | New-job dispatch was paused via `POST /api/queue/pause`. Emitted only on the resumed → paused edge; an idempotent pause is silent.                                                                                                                                                               |
| `queue_resumed`               | New-job dispatch resumed via `POST /api/queue/resume`. Emitted only on the paused → resumed edge.                                                                                                                                                                                                |
| `queue_plan_changed`          | The scheduler published a newer versioned queue plan. Replace tentative lanes only when `plan_version` advances.                                                                                                                                                                                 |
| `device_state_changed`        | Device administration, worker health, or activity changed. Treat the payload as a hint and refetch `GET /api/devices`; telemetry-only samples do not emit this event.                                                                                                                            |
| `chain_job_queued`            | A durable chain job entered the queue; created, resumed, retaken, or amended. Carries `id`, `model`, and `stage_count`.                                                                                                                                                                          |
| `chain_job_started`           | The chain runner claimed the job and began rendering stages (`id`, `model`).                                                                                                                                                                                                                     |
| `chain_job_ended`             | The job settled. `state` is `completed`, `failed`, or `cancelled`. Terminal chain jobs stay listed on `/api/chain-jobs`; this only says the runner is done with it.                                                                                                                              |

The three `chain_job_*` events are additive and deliberately distinct from
`job_queued` / `job_started` / `job_ended`: chain jobs do not support the
print-queue affordances (`PATCH`/`DELETE /api/queue/:id`), and older clients
ignore unknown `type` tags. The ephemeral jobs backing
chain planning stay silent; only durable `/api/chain-jobs` work is
announced. Clients that render sequences in a unified activity surface can
use these instead of polling `GET /api/chain-jobs`.

The stream carries **deltas only**; there is no initial snapshot. Subscribe
first, then bootstrap current state from `GET /api/queue`, `GET /api/devices`,
and `GET /api/gallery`. Refetch those authoritative snapshots after every
reconnect because lagged broadcast frames are intentionally not replayed.
Feature-detect with
`GET /api/capabilities` (`"events": {"available": true}`); servers older than
this endpoint omit the field. Keep-alive pings arrive every 15 s.

```bash
curl -N http://localhost:7680/api/events
```

## Chained video generation

Chained video for the LTX-2, LTX-Video, and Wan families (including installed
catalog checkpoints with opaque `cv:` / `hf:` IDs) splits a long video into N
per-clip renders and returns a single stitched MP4. The seam is family- and
checkpoint-specific: LTX-2 threads a motion tail of latents across each
boundary (default 17 frames), Wan continues via last-frame image conditioning
on image-conditioned checkpoints (the overlap is always 1 frame; text-to-video
checkpoints concatenate independent clips), and LTX-Video joins independently
rendered clips. See the
[LTX-2 chained video output guide](/models/ltx2#chained-video-output) for the
user-facing story.

A sequence is a **durable chain job** on every surface:
[`POST /api/chain-jobs`](#api-chain-jobs) creates one and
`GET /api/chain-jobs/{id}/events` streams its stage progress to settlement.
The synchronous `POST /api/generate/chain` and SSE
`POST /api/generate/chain/stream` endpoints, which ran a chain as a hidden
ephemeral job and deleted its artifacts after answering, have been removed:
they could not be resumed, retaken, or reattached after a dropped connection,
and every client now creates a real job instead.

## `/api/generate/chain/validate`

Accepts the same `ChainRequest` body as `POST /api/chain-jobs`, but performs
only build-feature, model-family, and structural normalization. It does not
create a durable job, start a download, lease a device, or touch inference.
The response reports normalized stage transitions, each stage's contributed
output frames, source/negative-prompt presence, warnings, and the optional
`vram_estimate` field:

```json
{
  "model": "ltx-2-19b-distilled:fp8",
  "width": 1216,
  "height": 704,
  "fps": 24,
  "motion_tail_frames": 17,
  "stage_count": 2,
  "estimated_total_frames": 177,
  "estimated_duration_ms": 7375,
  "stages": [
    {
      "prompt": "a cat enters the forest",
      "frames": 97,
      "output_frames": 97,
      "transition": "smooth",
      "has_source_image": true,
      "has_negative_prompt": false
    },
    {
      "prompt": "the forest opens to a clearing",
      "frames": 97,
      "output_frames": 80,
      "transition": "smooth",
      "has_source_image": false,
      "has_negative_prompt": false
    }
  ],
  "warnings": [],
  "vram_estimate": null
}
```

Media and negative-prompt contents are not echoed. HTTP `422` uses the normal
structured `VALIDATION_ERROR` response. `vram_estimate`, when present, reports
`worst_case_bytes` (the max over stages, never their sum; stages run
serially) and an advisory `fits` verdict computed against stable device
capacity. It is `null` when the server cannot price the run (model not
downloaded, or no device sample); it is advisory and never gates submission.

## `/api/chain-jobs`

Durable async chain jobs persist the request, per-stage state, retakes, and
final outputs under `MOLD_HOME/jobs/<job_id>` and mirror query state in
`mold.db`. They use the same `mold_core::chain::ChainRequest` body as
a chain plan, but return immediately with `202 Accepted`:

```json
{ "job_id": "550e8400-e29b-41d4-a716-446655440000" }
```

### Ephemeral jobs

`ChainRequest.ephemeral` (additive; absent means `false`) marks a chain that is
ONE print's implementation detail rather than a sequence the user authored.
`mold run --frames 200` splits a long video into clips because the model cannot
render it in one pass (the user asked for a video, not for a chain) so the CLI
sets this and browser surfaces should set it for the same auto-chained case.

An ephemeral job renders identically to an authored one and publishes the same
stitched print, with full per-clip provenance. What differs:

- it is absent from `GET /api/chain-jobs`, so it never appears in History ▸
  Sequences;
- it emits no authored-sequence `chain_job_queued` event, while `/api/activity`
  still exposes the live job in "Now developing" on every client;
- graceful shutdown parks it as `paused` with its manifest, source media,
  completed clips, and tail cache intact; `resume` requeues it explicitly;
- its working directory is swept only after the job settles;
- **its print records no `chain_job_id`**, so "Reuse settings" restores a
  one-shot rather than opening the clip rail for a job that no longer exists.

Chain jobs accept every video `output_format`. Stitching and its audio mux are
MP4-native, so the job's own artifact is always MP4 (amend and retake decode
it) and the gallery print is transcoded to the requested mp4, gif, webp, or
apng at finalization; the `finalized` event names it in `gallery_filename`.

Endpoints:

- `POST /api/chain-jobs`: create a queued job.
- `GET /api/chain-jobs`: list summaries, newest first. Ephemeral jobs are
  omitted; pass `?include_ephemeral=true` only to recover an id lost during
  suspension.
- `GET /api/chain-jobs/:id`: detail including stages, retakes, finalizes, and effective script.
- `GET /api/chain-jobs/:id/events`: SSE stream; first frame is always a snapshot.
- `POST /api/chain-jobs/:id/resume`: requeue `paused`, `interrupted`, `failed`, or `cancelled`.
- `POST /api/chain-jobs/:id/retake`: body is `RetakeRequest` (`stage_idx`, `mode`, optional `seed_offset`, optional `prompt`).
- `POST /api/chain-jobs/:id/amend`: replace the whole stage list in place, reusing cached clips. See below.
- `POST /api/chain-jobs/:id/cancel`: queued jobs settle as `cancelled`; an accepted running cancellation returns `202`, exposes `summary.cancelling: true`, and cannot publish a completed stage/job after that barrier.
- `DELETE /api/chain-jobs/:id`: remove a non-running job and its job directory.
- `POST /api/chain-jobs/gc`: explicitly sweep eligible ephemeral jobs and discard completed durable stage caches while retaining final outputs and job metadata. Automatic maintenance retains durable caches.
- `GET /api/chain-jobs/:id/stages/:idx/preview`: returns `image/jpeg` when that stage has a preview.
- `GET` or `HEAD /api/chain-jobs/:id/stages/:idx/media`; streams a completed raw stage MP4, including byte-range `206` and unsatisfiable-range `416` behavior.
- `POST /api/chain-jobs/:id/stages/:idx/media-token`: mints a short-lived ticket restricted to the exact corresponding media path.

Common errors: `503 CHAIN_JOBS_UNAVAILABLE` when the metadata DB is disabled,
`404 CHAIN_JOB_NOT_FOUND`, and `409 CHAIN_JOB_RUNNING` for mutations that
cannot safely run while the job is active.

### `POST /api/chain-jobs/:id/amend`

Edits a settled or queued sequence in place instead of creating a new job, so
clips that did not change are never re-rendered. This is what the Studio
surfaces call behind **Update sequence**.

The body maps to `mold_core::chain_job::AmendRequest`. `stages` is the
**complete** edited stage list in canonical order (not a patch); the same
`ChainStage` shape `POST /api/chain-jobs` accepts. Everything else is an
optional chain-level overlay applied over the job's current effective
request:

```json
{
  "stages": [
    { "prompt": "a cat walks into the autumn forest", "frames": 97 },
    { "prompt": "the forest opens to a clearing", "frames": 49 },
    {
      "prompt": "a spaceship lands",
      "frames": 97,
      "transition": "fade",
      "fade_frames": 12
    }
  ],
  "motion_tail_frames": 8,
  "fps": 24,
  "seed": "42",
  "steps": 8,
  "guidance": 3.0,
  "enable_audio": false
}
```

`seed` is a full-range `u64` encoded as a decimal **string**, matching the
rest of the chain wire format. Omitted overlays keep the job's current value.

`strength` is a chain-level overlay too, alongside `steps` and `guidance`.

**Not amendable.** `AmendRequest` carries no other fields, so `model`,
`width`, `height`, `output_format`, GPU `placement`, and the
`batch_id` / `batch_index` / `batch_count` provenance are inherited from the
original request and cannot be changed; create a fresh job for those. The
amended candidate must still pass every create-time gate: `normalise()`,
the family/audio check, and the video-format rule (`mp4`, `gif`, `webp`, or
`apng`; the job's own artifact is always stitched as MP4 and the gallery print
is transcoded to the requested format at finalization).

**Response**: `202 Accepted`. The body is the updated `ChainJobSummary`
flattened, plus `preserved_stages`:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "state": "queued",
  "model": "ltx-2-19b-distilled:fp8",
  "stage_count": 3,
  "current_stage": 2,
  "created_at_unix_ms": 1752300000000,
  "updated_at_unix_ms": 1752300600000,
  "error": null,
  "ephemeral": false,
  "preserved_stages": 2
}
```

`preserved_stages` is the count of leading stages whose cached artifacts were
kept; rendering requeues from that index. The job is requeued and a
`chain_job_queued` event is published on `/api/events`.

**Invalidation semantics.** The preserved prefix is the longest run of leading
stages whose render identity is unchanged, clamped to the leading run of
already-completed stages. A chain-level change to `seed`, `steps`, `guidance`,
`strength`, `fps`, or `motion_tail_frames`, or turning `enable_audio` **on**,
invalidates everything (turning it off preserves every clip; finalize simply
ignores the audio sidecars). Otherwise a stage is dirty when its `prompt`,
`frames`, `negative_prompt`, `source_image`, LoRA stack, effective per-stage
seed, or its _smooth-carry_ status changes, where carry means "not the first
stage and
`transition == "smooth"`". Because stage artifacts are stored as **raw,
untrimmed** segments with every boundary trim and crossfade applied at
finalize time, `cut` ↔ `fade` toggles and `fade_frames` edits break no prefix
at all and re-finalize with zero re-renders; `smooth` ↔ (`cut`|`fade`) changes
the rendered pixels and does break it. Appending clips renders only the new
ones, and removing trailing clips renders nothing. Jobs written by older
versions carry baked-in boundaries, so a preserved legacy stage is dropped
from the prefix when its artifacts cannot serve the new boundary plan.

Each amend is appended to the manifest with the pre-amend **effective**
request (retakes folded in) and its `preserved_stages`; any pending retakes
are folded and cleared. `GET /api/chain-jobs/:id` exposes the additive
`amends` array (`at_unix_ms`, `previous_request_json`, `preserved_stages`).

**Errors:**

- `409 CHAIN_JOB_RUNNING`: the job is rendering; cancel it first.
- `409 CHAIN_JOB_EPHEMERAL`: the job is ephemeral and owns no retained artifacts.
- `409 CHAIN_JOB_NOT_AMENDABLE`: the job left an amendable state mid-request. Amendable states are `queued`, `interrupted`, `failed`, `cancelled`, and `completed`.
- `422`: the amended request failed validation (bad frame counts, motion tail ≥ clip frames, too many stages, a non-video output format, an unsupported family, audio on a checkpoint without an audio path).
- `404 CHAIN_JOB_NOT_FOUND`: unknown id.

## `/api/models`

`GET /api/models` lists every known model. Each row is a flattened
`ModelInfoExtended`: identity (`name`, `family`, `hf_repo`, …), the
`ModelDefaults` block (`default_steps`, `default_guidance`, `default_width`,
`default_height`, `description`, …), and installation state, all at the top
level of the object; the defaults are not nested under a `defaults` key.
Video models additionally advertise their frame semantics there, so clients
stop hardcoding a frame count that ignores the selected checkpoint:

| Field                           | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `default_frames`                | Default frame count for one clip. LTX-2 defaults to 97, LTX-Video to its shipped 25.                                                                                                                                                                                                                                                                                                                                                                                     |
| `default_fps`                   | Default frames per second.                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `max_frames`                    | Ceiling for a single request **at `default_fps`**, already snapped onto the family's frame grid so the advertised value is itself submittable; 481 for LTX-2 at 24 fps, 257 for LTX-Video.                                                                                                                                                                                                                                                                               |
| `supports_extend`               | Whether this model can continue an existing video in one request. Every LTX-2 checkpoint can; Wan answers per checkpoint, from the same `source_image` contract its seam reads. Absent on servers that predate continuation; read absence as "no".                                                                                                                                                                                                                       |
| `supports_identity`             | Whether this model accepts a face-identity reference (`id_image`). True only for an identity-qualified checkpoint on a build that can actually execute identity conditioning; the `pulid` feature AND the landed runtime adapter, so the capability is advertised only once that adapter is present; derived from the same generation-profile authority as `capabilities.supports_identity`. Absent on servers that predate identity conditioning; read absence as "no". |
| `extend_default_overlap_frames` | Overlap applied when a continuation omits `extend_overlap_frames`. Per family, because it is the family's carryover: 17 on LTX-2, 1 on Wan.                                                                                                                                                                                                                                                                                                                              |
| `max_runtime_seconds`           | Present when the family's real ceiling is a duration rather than a frame count (LTX-2: 20). Recompute `max_frames` at another fps as `min(max_runtime_seconds · fps + 4, max_frames_absolute)`, then round **down** onto `k · frame_step + frame_offset`; the raw duration value sits off the grid at essentially every fps and is rejected at admission.                                                                                                                |
| `max_frames_absolute`           | fps-independent frame guard paired with `max_runtime_seconds` (LTX-2: 604).                                                                                                                                                                                                                                                                                                                                                                                              |
| `frame_step`, `frame_offset`    | Valid counts are `k · frame_step + frame_offset`; the offset is 1 for LTX/Wan and 5 for MiniMax H3 (`17n+5`). Older servers omit `frame_offset`; only then should clients use 1.                                                                                                                                                                                                                                                                                         |
| `source_image`                  | Per-checkpoint image-conditioning contract: `"unsupported"` (an attached image is rejected at admission), `"optional"`, or `"required"` (the checkpoint cannot generate without one). Omitted = unknown; clients must treat absence as unknown, not as a heuristic license. Derived from manifest task structure or the installed checkpoint's own tensor shapes.                                                                                                        |
| `dimension_alignment`           | Pixel grid both dimensions must sit on. Most Wan checkpoints use 16; `wan22-ti2v-5b` advertises 32 (its 16× VAE stride times the 2×2 DiT patch). Off-grid canvases are rejected at admission on the generate and chain routes.                                                                                                                                                                                                                                           |
| `min_frames`                    | Minimum requestable frame count, when the family has one above the historical one-frame floor (MiniMax H3: 107).                                                                                                                                                                                                                                                                                                                                                         |
| `max_pixels`, `max_axis_pixels` | Server-authoritative total-pixel and per-axis ceilings. The axis ceiling is per **model**, not per family: an LTX-2 checkpoint that ships the spatial upsampler composes stage 1 at half size plus a tiled stage 2 and reaches twice the trained RoPE span, one that does not cannot.                                                                                                                                                                                    |
| `recommended_dimensions`        | Runnable, family-appropriate size buckets every Studio surface renders; a client picks from these instead of inventing a canvas.                                                                                                                                                                                                                                                                                                                                         |
| `supports_audio`                | Whether this concrete checkpoint can generate a synchronized audio track. Per model, because a runnable LTX-2 video checkpoint may omit the audio VAE and vocoder; absent on older servers, which only advertised family support.                                                                                                                                                                                                                                        |
| `supports_sequence`             | Whether this model's effective runtime pipeline can render sequence clips. Absent on servers that predate per-model advertisement, where clients fall back to their own conservative name heuristic.                                                                                                                                                                                                                                                                     |
| `supports_duration_prediction`  | Whether omitting `frames` asks this model to run its qualified prompt-conditioned duration head. Absent on older servers and false without that exact component contract.                                                                                                                                                                                                                                                                                                |
| `guidance_capabilities`         | Guidance controls for this model's default resolved recipe. Clients refine it with an explicitly selected pipeline when applicable.                                                                                                                                                                                                                                                                                                                                      |
| `generation_profile`            | The complete, versioned generation-control contract for this model and every selectable recipe — alignment, ceilings, authored presets and their `tier`, and per-control `note` text. New clients read this instead of reconstructing policy from family names and the legacy flattened fields; `capabilities.generation_profile_v1` says whether the host publishes it.                                                                                                 |
| `runtime_ready`                 | Whether every component this concrete split pack requires is present and header-qualified on this host. Paired with `runtime_readiness_error`. LTX-2.5 publishes it even for incomplete rows so automatic routing can refuse them before queueing; absence passes.                                                                                                                                                                                                       |
| `runtime_available`             | Whether this exact model can execute on this build and host. `false` still permits download, verification, inventory, repair, and removal. Absent on older servers means unknown, not confirmed runnable.                                                                                                                                                                                                                                                                |
| `runtime_unavailable_reason`    | Present exactly when `runtime_available` is false; names an unsupported layout/task or a build missing the required engine, including MiniMax H3 download-only rows.                                                                                                                                                                                                                                                                                                     |

Models with a tuned default negative prompt (wan today) additionally
advertise `default_negative_prompt`: the negative the engine applies when a
request omits `negative_prompt` entirely. An explicit `""` in a request stays
a real empty uncond; clients prefill the advertised value, keep an untouched
field absent, and send `""` to opt out.

::: tip LTX-2's ceiling is a duration, not a frame count
The LTX-2 checkpoints ship `pos_embed_max_pos = 20`, and the temporal RoPE axis
is normalized in **seconds** (the pixel-frame coordinate is divided by fps
before `max_pos` normalization). So the raw budget is 20 seconds of runtime;
`seconds · fps + 4`, clamped by `max_frames_absolute`. What `/api/models`
advertises is that value snapped **down** onto the family's `8n+1` frame grid,
because the raw one is off-grid and a slider clamped to it produced a 422: 481
at 24 fps, 121 at 6 fps. Clients that let the user change fps must recompute
`max_frames` from `max_runtime_seconds` and re-snap it; treating the advertised
scalar as fixed will be wrong in both directions.

`--temporal-upscale x2` does **not** extend this budget: it halves the stage-1
frame count _and_ the stage-1 fps, so stage 1 renders the same runtime at half
the frame rate.
:::

All of these are additive and omitted entirely on image models; clients must
treat their absence as "not a video model" rather than substituting a
constant. They come from the same manifest defaults and validator constants
the server enforces, and the server-side validator stays authoritative.

`GET /api/capabilities/chain-limits?model=<name>&fps=<n>` reports
`frames_per_clip_recommended` from the same per-model default, and its
`frames_per_clip_cap` is the model's own clip size; the number of frames one
generation renders when a long request is chained automatically (97 for
LTX-2, or the checkpoint's own manifest default for Wan over a 53-frame A14B /
121-frame floor, e.g. 121 for TI2V-5B),
bounded above by the family's single-request ceiling at `fps` when that is
smaller. Every sequence picker locks its per-clip choices to this value. The
response echoes the `fps` it was computed at and, for families with a duration
budget, `frames_per_clip_runtime_seconds`. Chain admission itself enforces the
family's single-request ceiling, which is what an explicit CLI `--clip-frames`
may reach. Pass `fps` when the user is editing that control so the advertised
cap matches what the server will hold the request to.

## `/api/status`

Example response:

```json
{
  "version": "0.26.0",
  "git_sha": "da039e1",
  "build_date": "2026-08-30",
  "instance_id": "0b5c1a4e-9f3d-4c8a-b2e7-6d1f0a9c3e58",
  "models_loaded": ["flux-schnell:q8", "ltx-2-19b-distilled:fp8"],
  "busy": true,
  "queue_paused": false,
  "gpu_info": null,
  "models_disk": { "total_bytes": 994662584320, "free_bytes": 213909504000 },
  "host_memory": {
    "total_bytes": 67191394304,
    "available_bytes": 28173926400,
    "headroom_bytes": 18059427840,
    "safety_floor_bytes": 10114498560,
    "reclaimable_zfs_arc_bytes": 15081432704
  },
  "gpus": [
    {
      "ordinal": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "vram_total_bytes": 25757220864,
      "vram_used_bytes": 12918456320,
      "loaded_model": "flux-schnell:q8",
      "state": "idle"
    },
    {
      "ordinal": 1,
      "name": "NVIDIA GeForce RTX 4090",
      "vram_total_bytes": 25757220864,
      "vram_used_bytes": 21474836480,
      "loaded_model": "ltx-2-19b-distilled:fp8",
      "state": "generating"
    }
  ],
  "queue_depth": 1,
  "queue_capacity": 200,
  "uptime_secs": 3600,
  "hostname": "gpu-box",
  "durable_media": {
    "available": false,
    "reasons": [
      "owner media store unavailable: /srv/mold/queue-media must be a current-user-owned 0700 directory: found mode 0770 (expected 0700); repair with: chmod -- 0700 '/srv/mold/queue-media'"
    ]
  }
}
```

Older single-GPU clients can still read `gpu_info`; multi-GPU-aware clients
should prefer `gpus[]`, `queue_depth`, and `queue_capacity`.

`instance_id` is the stable UUID identifying this server installation,
persisted in the metadata DB on first boot and ephemeral per process when the
DB is unavailable. It is the field desktop, web, and iPhone dedupe a host on —
one box reached by hostname, mDNS, and IP collapses to one row — and the one a
phone verifies against the scanned pairing envelope. `models_disk` reports the
filesystem holding the models directory; `host_memory` is the scheduler
admission ledger's own host-RAM reading, which on ZFS also carries the additive
`reclaimable_zfs_arc_bytes` beside `available_bytes`. `queue_paused` mirrors
`POST /api/queue/pause`, `current_generation` describes the running job, and
`memory_status` is a human-readable one-liner. Every one of these is additive
and absent on older servers.

`durable_media` explains restart-safe encrypted request media to an operator.
It is absent whenever this server never offers the feature; no durable
generation queue, gallery output disabled, a non-authoritative scheduler;
which is a configuration rather than a degradation. When present, `available`
mirrors the presence of `capabilities.durable_media` exactly; `reasons` is
empty while available and otherwise lists every reason the feature is off,
retained for the life of the process so the startup log is not the only record.
A large held backlog is summarized with a count and a short sample rather than
enumerated. Reasons name host filesystem paths, so they appear only here,
behind authentication.

## `/health`

`GET /health` is auth-exempt and **always answers `200` while the process is
serving**, including when a subsystem is degraded: a health check that failed
here would pull a server out of a load balancer over a degradation that
generation survives. The additive body names which subsystems are off, never
why.

```json
{ "status": "degraded", "degraded": ["durable_media"] }
```

A healthy server answers `{ "status": "ok" }`. Read `/api/status`
`durable_media.reasons` for the diagnosis and the repair command. The probe
never waits on a lock: if it lands during a `/api/config` write it reports
healthy rather than blocking, so `/api/status` is the authority for the
degraded state.

`GET /api/resources` and `GET /api/resources/stream` expose only the GPU
inventory that CUDA made visible when the server started. CUDA builds sample
NVML by the raw CUDA/NVIDIA UUID rather than a physical ordinal, so numeric
`CUDA_VISIBLE_DEVICES` reordering and `GPU-...` selectors preserve the correct
process-local ordinal and hidden physical cards are not published. The
`nvidia-smi` fallback applies the same UUID filter and converts its MiB values
to binary bytes. A MIG worker accepts only telemetry carrying its matching
`MIG-...` UUID; Mold does not substitute the parent GPU's full-memory sample.
When the installed NVML adapter cannot prove a MIG parent UUID or profile,
`mig_parent_uuid` and `mig_profile` remain `null`.

## `/api/devices`

`GET /api/devices` is the stable multi-device resource. Device `id` values are
opaque and must be URL-encoded rather than parsed; CUDA ordinals are
process-local display hints. Operational values that the active sampler cannot
provide are JSON `null`, not zero.

```json
{
  "devices": [
    {
      "id": "cuda:0123456789abcdef0123456789abcdef",
      "backend": "cuda",
      "ordinal": 0,
      "device_kind": "full_gpu",
      "nvml_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
      "physical_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
      "mig_uuid": null,
      "mig_parent_uuid": null,
      "mig_profile": null,
      "name": "NVIDIA GeForce RTX 3090",
      "pci_bus_id": "00000000:01:00.0",
      "compute_capability": "8.6",
      "memory": {
        "total_bytes": 25769803776,
        "used_bytes": 8589934592,
        "mold_used_bytes": null,
        "other_used_bytes": null
      },
      "telemetry": {
        "utilization_percent": 41,
        "temperature_c": null,
        "power_w": null
      },
      "desired_enabled": true,
      "restart_required": false,
      "admin_state": "enabled",
      "health": "healthy",
      "activity": "idle",
      "schedulable": true,
      "unschedulable_reason": null,
      "loaded_models": [],
      "active_work_id": null,
      "planned_work_ids": []
    }
  ],
  "plan_version": 0
}
```

`plan_version` remains `0` until the versioned scheduler plan is active.
Desired enablement is machine-wide; a newly seen device with no explicit
preference defaults to enabled.

`PATCH /api/devices/{url-encoded-stable-id}` accepts
`{"enabled":false}` or `{"enabled":true}`. Disabling removes the device from
new scheduling immediately and returns `202` while an active lease drains.
Re-enabling starts a fresh owner lifetime and returns `202` with
`admin_state:"starting"` without waiting for CUDA context creation. The first
epoch-qualified ready event changes it to enabled. If context creation fails,
the desired preference remains enabled, health is unavailable, and
`unschedulable_reason` reports `device_start_failed: ...`; retry the PATCH or
restart after correcting the driver/device fault. A delayed ready, stopped, or
completion event from the predecessor cannot mutate or reap the replacement.

Runtime mutation requires scheduler V2. In legacy, observe, or maintenance
mode, disabling still returns `409`, but enabling a persistently-disabled,
startup-selected device records the preference for the next boot. The first
such PATCH returns `202` with `restart_required:true`; repeated identical
PATCHes return `200`, and subsequent device polls retain that flag until a
restart creates the owner. Startup-excluded devices cannot use this recovery
path and return `409`.

## `/api/models/pull`

Plain blocking response:

```bash
curl -X POST http://localhost:7680/api/models/pull \
  -H "Content-Type: application/json" \
  -d '{"model":"flux-schnell:q8"}'
```

Example text response:

```text
model 'flux-schnell:q8' pulled successfully
```

SSE streaming response:

```bash
curl -N http://localhost:7680/api/models/pull \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{"model":"flux-schnell:q8"}'
```

Representative events:

```text
event: progress
data: {"type":"download_progress","filename":"flux1-schnell-Q8_0.gguf","file_index":1,"total_files":6,"bytes_downloaded":1048576,"bytes_total":12714452256}

event: progress
data: {"type":"pull_complete","model":"flux-schnell:q8"}
```

The body also accepts an additive `accept_licenses` array; see
[`/api/licenses`](#api-licenses). A model whose files need an unaccepted
license is refused with `403` / `LICENSE_NOT_ACCEPTED` before the pull is
enqueued.

## `/api/licenses`

Some auxiliary weights carry terms Mold's own license does not cover; the
InsightFace antelopev2 face models that PuLID identity conditioning needs are
licensed for non-commercial research only. Mold refuses to download them until
an acceptance is on record.

**Acceptance is per Mold data root, and the root that matters is the one on the
machine doing the downloading.** A client that records acceptance locally and
then asks a remote server to pull has told the wrong machine, which is why the
ids travel on the request instead.

```bash
curl http://localhost:7680/api/licenses
```

```json
{
  "licenses": [
    {
      "id": "insightface-antelopev2",
      "name": "InsightFace pretrained models (antelopev2)",
      "url": "https://raw.githubusercontent.com/deepinsight/insightface/7fadd420c2351d0ffa8cac403421c1a3ed733365/README.md",
      "canonical": "https://github.com/deepinsight/insightface#license",
      "sha256": "84606d9ab37a38606b12c10d96172c6343768d2ef72c802a16482e476f8baf22",
      "summary": "InsightFace pretrained models (antelopev2: scrfd_10g_bnkps, glintr100) are licensed for non-commercial research purposes only.",
      "accepted": false,
      "required_by": ["pulid-flux", "pulid-sdxl"]
    }
  ]
}
```

`url` is a commit-pinned, immutable link to the exact text; `canonical` is the
browsable project page and is presentation only. An acceptance is bound to the
`(url, sha256)` pair, so `accepted` reads `false` again after a Mold release
re-pins the license to a newer upstream revision. `accepted` describes only the
server that answered; a multi-host client must ask each one.

### Accepting without downloading

`POST /api/licenses/accept` records consent on its own. Consent and acquisition
are different acts: before this route existed the only way to accept was to
start a download, so agreeing to terms always meant transferring the weights,
and a license no installed manifest required could not be accepted at all.

```bash
curl -X POST http://localhost:7680/api/licenses/accept \
  -H 'Content-Type: application/json' \
  -d '{"accept_licenses":[{"id":"tencent-hunyuan3d-2.0","url":"...","sha256":"..."}]}'
```

It answers with the same body as `GET /api/licenses`, refreshed, so a client
needs no second round trip. `400 UNKNOWN_LICENSE` for an id this server does
not register; `409 LICENSE_TERMS_MISMATCH`, carrying the server's own terms,
when the entry does not match what this server pins.

### Accepting a license

`POST /api/downloads` and `POST /api/models/pull` take an additive
`accept_licenses` array. Each entry carries the **exact terms the user was
shown**, not just an id:

```bash
curl -X POST http://localhost:7680/api/downloads \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pulid-flux",
    "accept_licenses": [{
      "id": "insightface-antelopev2",
      "url": "https://raw.githubusercontent.com/deepinsight/insightface/7fadd420c2351d0ffa8cac403421c1a3ed733365/README.md",
      "sha256": "84606d9ab37a38606b12c10d96172c6343768d2ef72c802a16482e476f8baf22"
    }]
  }'
```

An id alone would not be safe. The client and the server may be on different
Mold releases pinning different revisions of the same license, and a bare id
would let the server resolve terms of its own choosing and record consent for
text the user never read. So the server compares the submitted `(url, sha256)`
against its own pin and only records a match. Read `GET /api/licenses` first,
display what it returned, and send back exactly that.

The server records into **its own** `$MOLD_HOME/license-acceptances.json`
(owner-only, `0600`) before the pull starts. Omitting the field means "accept
nothing", which is what every existing client sends.

Nothing is written unless every entry passes, so a rejected request leaves the
root untouched:

| Condition                                | Status | Code                     |
| ---------------------------------------- | ------ | ------------------------ |
| Id this server does not know             | `400`  | `UNKNOWN_LICENSE`        |
| Known id, terms this server does not pin | `409`  | `LICENSE_TERMS_MISMATCH` |

A `409` carries the server's own `url`, `sha256`, and `canonical` in the
`license` object, so a client can display the server's terms and retry without
a second round trip.

### Refusals

A download whose model still needs an unaccepted license is refused with `403`
and code `LICENSE_NOT_ACCEPTED`, before any bytes move:

```json
{
  "error": "pulid-flux includes files under a license that must be accepted before download. …",
  "code": "LICENSE_NOT_ACCEPTED",
  "license": {
    "id": "insightface-antelopev2",
    "name": "InsightFace pretrained models (antelopev2)",
    "url": "https://raw.githubusercontent.com/deepinsight/insightface/7fadd420c2351d0ffa8cac403421c1a3ed733365/README.md",
    "canonical": "https://github.com/deepinsight/insightface#license",
    "sha256": "84606d9ab37a38606b12c10d96172c6343768d2ef72c802a16482e476f8baf22",
    "summary": "InsightFace pretrained models (antelopev2: scrfd_10g_bnkps, glintr100) are licensed for non-commercial research purposes only."
  }
}
```

`error` is the human message (it names the exact CLI command); the additive
`license` object is the machine-readable half, so a UI can render its own
acceptance prompt and retry with `accept_licenses` rather than parsing prose.
`license` is absent on every other error. There is no environment-variable
bypass.

## `/api/expand`

Expand a short prompt into one or more detailed generation prompts using the
configured expansion LLM.

```bash
curl -X POST http://localhost:7680/api/expand \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cat",
    "model_family": "flux",
    "variations": 3,
    "style": "gritty film noir"
  }'
```

**Request fields:**

| Field          | Type   | Required | Description                                                                                                                                                                                                                                                                                                                                                                                           |
| -------------- | ------ | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prompt`       | string | yes      | Short prompt to expand                                                                                                                                                                                                                                                                                                                                                                                |
| `model_family` | string | no       | Model family for prompt style (`flux` default; `sdxl`, `sd15`, `sd3`, ...)                                                                                                                                                                                                                                                                                                                            |
| `variations`   | number | no       | Number of prompt variations to generate (default 1, max 10,000). The ceiling is a per-request safety bound on one reviewed set, not a limit on total prints — clients queue further prepared batches.                                                                                                                                                                                                 |
| `style`        | string | no       | Visual style to absorb into the expansion (e.g. a style preset label). Passed to the LLM as a natural-language directive, never a literal suffix.                                                                                                                                                                                                                                                     |
| `task`         | string | no       | Resolved generation/conditioning task, additive: `text-to-image` (default), `text-to-video`, `image-to-video`, `video-to-video`, `retake`, `keyframe-interpolation`, `audio-driven-video`, `reference-to-audio-video`, `text-to-audio`. When omitted, the server infers text-to-video for known video families and text-to-image otherwise. Carries only the semantic task; never source media bytes. |
| `context`      | object | no       | Generation facts, additive: `model` (exact identity, selects a per-checkpoint guide), `width`, `height`, `frames`, `fps`, `clip_frames`, `negative_prompt_supported`, `audio`, `references` (ordered `{kind: image\|video\|audio, has_audio, role}` with roles `first-frame`, `last-frame`, `keyframe`, `source`, `identity`, `edit`, `reference`), and `loras` (adapter names). Duration is derived as frames / fps and is never sent. Structure only; media bytes stay on the generation request. |

The system prompt the LLM receives is the target family's prompting guide from
the [prompting corpus](/guide/prompting) (every section except `CLI` and
`Sources`) followed by a generation-context block rendered from `context`. For
MiniMax H3 the references are named `<Picture n>`, `<Video n>`, and `<Audio n>`
in the conditioner's order, so the expansion can be valid Context-IR. Inline
expansion (`expand: true` on `POST /api/generate`) derives the same context from
the generation request and the resolved model profile.

**Response:**

```json
{
  "original": "a cat",
  "expanded": ["a sleek black cat prowling a rain-slicked alley, ..."]
}
```

## `/api/remix`

Remix is separate from Expand so an older host returns `404` instead of
silently applying expansion semantics. The server preserves the source subject
and explicit constraints, assigns each result a deterministic creative
dimension, and rejects dimensions that contradict image/video/audio
conditioning authority.

```bash
curl -X POST http://localhost:7680/api/remix \
  -H 'Content-Type: application/json' \
  -d '{
    "source_prompt": "a red lighthouse with exactly three windows",
    "root_prompt": "a red lighthouse",
    "source_kind": "current",
    "model_family": "flux",
    "variations": 3,
    "dimensions": ["camera", "lighting"]
  }'
```

Dimensions are `composition`, `camera`, `lighting`, `setting`, `mood`,
`movement`, and `style`. When `style` is supplied it is a locked constraint and
cannot also appear in `dimensions`. An omitted dimension list uses task-aware
defaults. The response returns `variants[]` with both `prompt` and the exact
`dimensions` label used for that alternative. Remix accepts the same additive
`task` and `context` fields as `/api/expand`, with the same semantics, and
applies the same prompting guide; custom expansion templates never apply.

## `/api/upscale`

Upscale an image using Real-ESRGAN super-resolution models.

```bash
curl -X POST http://localhost:7680/api/upscale \
  -H "Content-Type: application/json" \
  -d '{
    "model": "real-esrgan-x4plus:fp16",
    "image": "<base64-encoded PNG or JPEG>",
    "output_format": "png",
    "tile_size": 512
  }' \
  --output upscaled.png
```

**Request fields:**

| Field           | Type   | Required | Description                                               |
| --------------- | ------ | -------- | --------------------------------------------------------- |
| `model`         | string | yes      | Upscaler model name (e.g. `real-esrgan-x4plus:fp16`)      |
| `image`         | string | yes      | Base64-encoded input image (PNG or JPEG)                  |
| `output_format` | string | no       | `png` (default) or `jpeg`                                 |
| `metadata`      | object | no       | Generation metadata to embed in the upscaled output       |
| `tile_size`     | number | no       | Tile size for memory-efficient processing (0 = no tiling) |

**Response:** Raw image bytes (PNG or JPEG) with `Content-Type` header.

## `/api/upscale/stream`

Same request format as `/api/upscale`, but returns SSE events for tile-by-tile progress:

```bash
curl -N -X POST http://localhost:7680/api/upscale/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "real-esrgan-x4plus:fp16",
    "image": "<base64-encoded PNG or JPEG>"
  }'
```

Representative events (tile progress reuses the `denoise_step` event type):

```text
event: progress
data: {"type":"denoise_step","step":1,"total":9,"elapsed_ms":1200}

event: complete
data: {"image":"<base64>","format":"png","model":"real-esrgan-x4plus:fp16","scale_factor":4,"original_width":512,"original_height":512,"upscale_time_ms":450}
```

The server caches the upscaler engine between requests; repeated upscales with the same model skip weight loading.

## Saved output metadata

Gallery rows (`GET /api/gallery`, the `image.metadata` object on
`gallery_added`, and the embedded `mold:parameters` chunk) map to
`mold_core::OutputMetadata`. The request's engine-ignored `source_fit`
provenance, when sent, is echoed verbatim here as `source_fit`. Two additive
fields record sequence provenance:

- `chain_job_id`: the durable chain job this output was finalized from.
  Absent for single generations and legacy rows.
- `chain`: structured per-clip provenance, so a sequence is never recorded
  under clip 1's prompt alone. Absent for single generations and legacy rows.

```json
{
  "chain_job_id": "550e8400-e29b-41d4-a716-446655440000",
  "chain": {
    "stage_count": 3,
    "motion_tail_frames": 8,
    "stages": [
      {
        "prompt": "a cat walks into the autumn forest",
        "frames": 97,
        "transition": "smooth",
        "seed": "42"
      },
      {
        "prompt": "the forest opens to a clearing",
        "frames": 49,
        "transition": "cut",
        "seed": "43"
      },
      {
        "prompt": "a spaceship lands",
        "frames": 97,
        "transition": "fade",
        "fade_frames": 12,
        "seed": "44"
      }
    ]
  }
}
```

Each stage carries `prompt`, `frames`, and `transition`; `fade_frames` and the
effective per-stage `seed` (a full-range `u64` as a decimal string) appear only
when known. The top-level `prompt` of a multi-clip row holds every distinct
clip prompt joined one per line, so gallery search matches any clip. The CLI's
local chain saves write the same `chain` block without a `chain_job_id`.

Studio clients read this block to reload a finished sequence's clips onto the
Create clip rail (**Reuse settings**); `chain_job_id` is what lets **Edit
sequence** re-enter the original durable job with its cached clips.

## Image Output

Generated images are saved to `~/.mold/output/` by default. Override with a
custom path:

```bash
MOLD_OUTPUT_DIR=/srv/mold/output mold serve
```

To disable image persistence (TUI gallery will not function):

```bash
MOLD_OUTPUT_DIR="" mold serve
```
