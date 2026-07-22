# Model Discovery Catalog

mold's catalog is a live discovery proxy over Hugging Face and Civitai, backed
by a short in-process cache and the installed-state sidecars in your models
directory. It is not an offline shard database and there is no separate catalog
CLI surface.

Use it from the web UI, desktop app, or iPhone app's **Catalog** view, or call
the `/api/catalog/*` routes from a running `mold serve` instance.

## What You Can Install

Catalog entries use stable install IDs:

| Prefix | Source       | Example                           |
| ------ | ------------ | --------------------------------- |
| `hf:`  | Hugging Face | `hf:black-forest-labs/FLUX.1-dev` |
| `cv:`  | Civitai      | `cv:618692`                       |

The catalog clients can install base checkpoints, fine-tunes, LoRAs, and
supported single-file checkpoints. When an entry needs shared components, mold
queues the primary file and any missing companions in the same catalog group.
Examples include text encoders, tokenizers, VAEs, and LTX-2 companion assets.

## Client flows

### Web UI

1. Start the server with `mold serve`.
2. Open `http://localhost:7680/models`.
3. Search or filter by family, kind, source, rating, and downloads. Catalog
   searches include NSFW rows by default.
4. Click **Install** on the entry you want.

The downloads drawer tracks queued jobs and up to two concurrent transfers.
Cancelling a queued or running job remains visible until the server confirms it
has stopped. Installed LoRAs then appear in compatible LoRA pickers in Generate,
and installed models appear in the model picker and `mold list`.

### Desktop and iPhone

Both native apps merge installed models with the live Hugging Face/Civitai
catalog, keep installed entries first, expose All/Images/Video and source
filters, and show download contents before a Pull or Repair.

The iPhone Catalog browses one selected host while allowing Pull to target any
ready saved host. Choosing a pull target does not change the host selected in
Generate. Its card, detail sheet, and host picker share one download state:
**Connecting...**, **Starting...**, **Queued**, then **Pulling N%**. Active
downloads pin above the results and can be cancelled. Installed details route
Load on GPU, Unload from GPU, guarded Remove, and Repair actions to the host
that owns the model.

## API

| Method | Path                        | Purpose                                     |
| ------ | --------------------------- | ------------------------------------------- |
| `GET`  | `/api/catalog/families`     | Families/kinds surfaced by the live catalog |
| `GET`  | `/api/catalog/search`       | Search HF/Civitai with filters              |
| `GET`  | `/api/catalog/installed`    | Installed catalog entries and LoRAs         |
| `GET`  | `/api/catalog/:id`          | Resolve one `hf:` or `cv:` entry            |
| `POST` | `/api/catalog/:id/download` | Queue the entry and missing companions      |
| `GET`  | `/api/downloads`            | Current download queue/listing              |
| `GET`  | `/api/downloads/stream`     | SSE stream for download queue updates       |

Examples:

```bash
# Search installable FLUX LoRAs from Civitai
curl "http://localhost:7680/api/catalog/search?q=cinematic&family=flux&kind=lora&source=civitai&page=1&page_size=48"

# Show installed FLUX LoRAs
curl "http://localhost:7680/api/catalog/installed?kind=lora&family=flux"

# Queue a catalog install
curl -X POST "http://localhost:7680/api/catalog/cv:618692/download"
```

Use the returned installed `path` or catalog `id` in generation clients that
accept LoRA references. The MCP `list_loras` tool also reads the installed
catalog route when the legacy LoRA endpoint is not available.

Search and single-entry responses include a `page_url` field pointing at the
entry's human-facing model page — `https://huggingface.co/{repo}` for Hugging
Face entries, `https://civitai.com/models/{modelId}?modelVersionId={versionId}`
for Civitai entries. It is additive (older clients can ignore it) and may be
`null` when the upstream response doesn't carry enough context to compose it.

Catalog search accepts exactly `q`, `family`, `kind`, `source`, `include_nsfw`,
`sort`, `page`, and `page_size`. The response's `entries` are bounded by
`page_size`; `page` identifies the returned window and `total` describes the
complete filtered feed (or a conservative lower bound when an upstream does
not publish an exact count). Modality is derived on every returned entry, so
clients can apply Image/Video filtering without sending an unsupported query.

## Tokens

| Variable        | Purpose                                                     |
| --------------- | ----------------------------------------------------------- |
| `HF_TOKEN`      | Hugging Face gated/private repository access                |
| `CIVITAI_TOKEN` | Civitai bearer token for gated, early-access, and NSFW rows |

Set tokens in the server environment before `mold serve`, or save them in the
web Settings panel. The web settings persist to `mold.db` as
`huggingface.token` and `civitai.token`.

The desktop app can forward its locally stored Hugging Face or Civitai token as
a request-scoped fallback to a remote host; the host's own credential remains
first. The iPhone stores only per-host Mold API keys, not upstream catalog
tokens, so its selected remote host must be configured for gated content.

## Disabling Catalog

Set `MOLD_CATALOG_DISABLE=1` to report catalog functionality as unavailable in
`/api/capabilities` and disable catalog-backed UI flows.

The test-only upstream override variables (`MOLD_CATALOG_HF_BASE` and
`MOLD_CATALOG_CIVITAI_BASE`) are intentionally not normal user configuration.
