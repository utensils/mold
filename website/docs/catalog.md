# Model Discovery Catalog

mold's catalog is a live discovery proxy over Hugging Face and Civitai, backed
by a short in-process cache and the installed-state sidecars in your models
directory. It is not an offline shard database and there is no separate catalog
CLI surface.

Use it from the web UI's **Catalog** tab, or call the `/api/catalog/*` routes
from a running `mold serve` instance.

## What You Can Install

Catalog entries use stable install IDs:

| Prefix | Source       | Example                           |
| ------ | ------------ | --------------------------------- |
| `hf:`  | Hugging Face | `hf:black-forest-labs/FLUX.1-dev` |
| `cv:`  | Civitai      | `cv:618692`                       |

The web UI can install base checkpoints, fine-tunes, LoRAs, and supported
single-file checkpoints. When an entry needs shared components, mold queues the
primary file and any missing companions in the same catalog group. Examples
include text encoders, tokenizers, VAEs, and LTX-2 companion assets.

## Web UI Flow

1. Start the server with `mold serve`.
2. Open `http://localhost:7680/catalog`.
3. Search or filter by family, kind, source, rating, and downloads. Catalog
   searches include NSFW rows by default.
4. Click **Install** on the entry you want.

The downloads drawer tracks queued jobs and up to two concurrent transfers. Cancelling a queued or running job remains visible until the server confirms it has stopped. Installed LoRAs then appear
in compatible LoRA pickers in Generate, and installed models appear in the model
picker and `mold list`.

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
curl "http://localhost:7680/api/catalog/search?q=cinematic&family=flux&kind=lora&source=civitai"

# Show installed FLUX LoRAs
curl "http://localhost:7680/api/catalog/installed?kind=lora&family=flux"

# Queue a catalog install
curl -X POST "http://localhost:7680/api/catalog/cv:618692/download"
```

Use the returned installed `path` or catalog `id` in generation clients that
accept LoRA references. The MCP `list_loras` tool also reads the installed
catalog route when the legacy LoRA endpoint is not available.

## Tokens

| Variable        | Purpose                                                     |
| --------------- | ----------------------------------------------------------- |
| `HF_TOKEN`      | Hugging Face gated/private repository access                |
| `CIVITAI_TOKEN` | Civitai bearer token for gated, early-access, and NSFW rows |

Set tokens in the server environment before `mold serve`, or save them in the
web Settings panel. The web settings persist to `mold.db` as
`huggingface.token` and `civitai.token`.

## Disabling Catalog

Set `MOLD_CATALOG_DISABLE=1` to report catalog functionality as unavailable in
`/api/capabilities` and disable catalog-backed UI flows.

The test-only upstream override variables (`MOLD_CATALOG_HF_BASE` and
`MOLD_CATALOG_CIVITAI_BASE`) are intentionally not normal user configuration.
