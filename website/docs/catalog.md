# Model Discovery Catalog

mold ships a built-in catalog of every model family it can run, scanned from
Hugging Face and Civitai. Browse it through the web UI at `/catalog`, the
`mold catalog` CLI, or directly via `mold pull <catalog-id>`.

## CLI

```bash
mold catalog list --family flux --limit 10
mold catalog show hf:black-forest-labs/FLUX.1-dev
mold catalog refresh --family flux        # re-scan one family
mold catalog where cv:618692              # path on disk if downloaded
```

## Web UI

Visit `/catalog` to browse the full catalog with filters by family,
modality, source, sub-family, and FTS5-backed search. The detail drawer
shows the download recipe; the Download button is disabled with a phase
badge for entries that need a single-file loader (mold v0.10+).

## Auth

Set `HF_TOKEN` and `CIVITAI_TOKEN` as env vars, or paste them in
`Settings → Model Discovery`. Both are stored in `mold.db` `settings`
(`huggingface.token`, `civitai.token`).

## Refresh

The catalog is global per mold install. Run `mold catalog refresh` weekly
(or whenever you want fresh discovery); the scanner is incremental and
deterministic — no-op refreshes produce byte-identical shard files.

## Environment Variables

| Variable                    | Default              | Purpose                                                           |
| --------------------------- | -------------------- | ----------------------------------------------------------------- |
| `CIVITAI_TOKEN`             | unset                | Civitai bearer token for early-access and NSFW model access       |
| `MOLD_CATALOG_DIR`          | `$MOLD_HOME/catalog` | Override the directory where catalog shards are stored on disk    |
| `MOLD_CATALOG_DISABLE`      | unset                | Set `1` to flag the catalog as unavailable in `/api/capabilities` |
| `MOLD_CATALOG_HF_BASE`      | unset                | Override the Hugging Face base URL (test-only)                    |
| `MOLD_CATALOG_CIVITAI_BASE` | unset                | Override the Civitai base URL (test-only)                         |
