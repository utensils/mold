# Catalog Expansion Design

> Sub-project A of a four-part model-discovery initiative. Replaces the hand-curated `manifest.rs` registry with a scanned, searchable catalog covering every supported model on Hugging Face and Civitai. Ships a new web `/catalog` route, a `mold catalog` CLI, single-file checkpoint loaders for SD1.5/SDXL/FLUX/Z-Image/LTX, and the data plumbing those need.

## Goals

- Discover and catalog **every** Hugging Face and Civitai model that maps to a mold-supported family (FLUX, Flux.2/Klein, SD1.5, SDXL incl. Pony/Illustrious/NoobAI, Z-Image, LTX-Video, LTX-2, Qwen-Image, Wuerstchen).
- Treat the catalog as a first-class, queryable, paginated, searchable browser — not an expansion of the existing model picker.
- Allow safetensors-format single-file Civitai checkpoints to actually run on mold (phases 2–5) — this is what makes the catalog useful, not just decorative.
- Surface the catalog identically through the web SPA, the CLI, and the existing `mold pull` command (catalog IDs become a first-class `mold pull` argument).
- Persist catalog state in `mold.db` for indexed, sub-millisecond filter/sort/search at 12 k+ entries.
- Keep the catalog refreshable without a binary upgrade (`mold catalog refresh` re-runs the scanner against a user's tokens).
- Default to a SFW catalog with one toggle to unlock the full Civitai inventory.

## Non-Goals

- **TUI parity.** The TUI keeps its current downloaded-only picker. A ratatui browser is its own future sub-project. Power users compose with `mold catalog list --json | jq` from the TUI's `!` shell escape.
- **Civitai LoRA / embeddings / hypernetworks / controlnets.** Out of A's scope. The `kind` column reserves space; sub-project D handles LoRAs.
- **CI cron auto-refresh.** Manual `mold catalog refresh` only — no scheduled bot opening PRs against the repo.
- **Showing models mold cannot run.** Architectures with no mold engine (AuraFlow, Chroma, Hunyuan, HiDream, Kolors, Lumina, Mochi, PixArt, Wan, etc.) are dropped at scanner time. The catalog never teases.
- **Civitai's legacy unsafe binary format ("PT").** Hard-blocked at the scanner — only safetensors and gguf entries reach the catalog. Arbitrary-code-exec risk on deserialization is not worth catalog completeness.
- **Per-checkpoint custom companions.** Single-file entries that demand exotic text encoders are dropped (`engine_phase: 99`). The supported-companion set is curated and finite.
- **Live HF Hub keyword search.** The catalog is *all matching supported models we discovered*, not a passthrough to HF's full search.
- **CDN-hosted shard distribution.** Embedded shards on first install + on-demand local rescan covers freshness without infrastructure.

## Delivery Shape

- New umbrella branch `feat/catalog-expansion`, cut from `main`. Draft PR opened on day zero so CI runs against incremental work.
- **Five sequential phases**, each independently mergeable into the umbrella:
  - **Phase 1 — Catalog plumbing + scanner + web UI**. Self-contained slice that delivers the catalog for already-supported HF formats.
  - **Phase 2 — SD1.5 + SDXL single-file loaders**. Unlocks ~95% of Civitai inventory.
  - **Phase 3 — FLUX single-file (all versions)**.
  - **Phase 4 — Z-Image single-file (Turbo + Base)**.
  - **Phase 5 — LTX single-file (LTXV / LTXV2 / LTXV 2.3)**.
- Phase 1 ships the catalog as a usable product. Phases 2–5 progressively unlock `engine_phase`-gated entries — same UI, more clickable Download buttons.
- **UAT cadence**: <gpu-host> (the dual-GPU GPU host) pulls and exercises after each phase merges into the umbrella.

---

## 1. Phase 1 — Catalog plumbing + scanner + web UI

### 1.1 New crate — `crates/mold-catalog/`

```
crates/mold-catalog/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── entry.rs          # CatalogEntry + Source + Family + Companion types
│   ├── companions.rs     # Curated canonical-companion registry
│   ├── families.rs       # mold's supported family taxonomy (single source of truth)
│   ├── civitai_map.rs    # Civitai baseModel → mold family mapping
│   ├── hf_seeds.rs       # Curated HF foundation repos to walk from
│   ├── scanner.rs        # Top-level orchestration; per-family scan task pool
│   ├── stages/
│   │   ├── mod.rs
│   │   ├── hf.rs         # HF stage
│   │   └── civitai.rs    # Civitai stage
│   ├── normalizer.rs     # Source-specific → CatalogEntry
│   ├── filter.rs         # Quality + safety filters
│   ├── sink.rs           # Atomic shard write + DB seed
│   ├── shards.rs         # Embedded-shard reader (rust-embed)
│   └── tests/
│       └── fixtures/     # Canned API responses
├── data/
│   └── catalog/          # Committed shards (one JSON per family)
│       ├── flux.json
│       ├── flux2.json
│       ├── sd15.json
│       ├── sdxl.json
│       ├── z-image.json
│       ├── ltx-video.json
│       ├── ltx-2.json
│       ├── qwen-image.json
│       └── wuerstchen.json
└── build.rs              # rust-embed manifest with stub fallback
```

**Crate dependencies:**
- `mold-ai-core` for shared `Family` enum (mold's existing taxonomy lives there; `families.rs` re-exports).
- `mold-ai-db` for the `catalog` table SQL surface.
- `reqwest` (already in tree), `serde`, `serde_json`, `tokio`, `tracing`.
- `rust-embed` for embedded shards (already used by `mold-server` for web/dist).

**Why a new crate:** scanner pulls heavy HF + Civitai client surface area that `mold-cli`, `mold-discord`, and `mold-tui` should not inherit transitively. `mold-cli` adds it as a dep; others stay clean.

### 1.2 Core types — `entry.rs`

```rust
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct CatalogEntry {
    pub id: CatalogId,                 // canonical "hf:repo" or "cv:modelVersionId"
    pub source: Source,                // Hf | Civitai
    pub source_id: String,
    pub name: String,
    pub author: Option<String>,
    pub family: Family,                // re-exported from mold-core
    pub family_role: FamilyRole,       // Foundation | Finetune
    pub sub_family: Option<String>,    // "pony", "illustrious", "noobai", etc.
    pub modality: Modality,            // Image | Video
    pub kind: Kind,                    // Checkpoint (reserved: Lora, Vae, TextEncoder, ...)
    pub file_format: FileFormat,       // Safetensors | Gguf | Diffusers
    pub bundling: Bundling,            // Separated | SingleFile
    pub size_bytes: Option<u64>,
    pub download_count: u64,
    pub rating: Option<f32>,           // Civitai only
    pub likes: u64,
    pub nsfw: bool,
    pub thumbnail_url: Option<String>,
    pub description: Option<String>,
    pub license: Option<String>,
    pub license_flags: LicenseFlags,
    pub tags: Vec<String>,
    pub companions: Vec<CompanionRef>, // canonical names from companions.rs
    pub download_recipe: DownloadRecipe,
    pub engine_phase: u8,              // 1..5 = phase that unlocks runnability; 99 = unsupported
    pub created_at: Option<i64>,       // unix seconds
    pub updated_at: Option<i64>,
    pub added_at: i64,                 // when scanner added to catalog
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum Source { Hf, Civitai }

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum FamilyRole { Foundation, Finetune }

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum Modality { Image, Video }

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum Kind {
    Checkpoint,
    // reserved for sub-project D + future
    Lora,
    Vae,
    TextEncoder,
    ControlNet,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum FileFormat { Safetensors, Gguf, Diffusers }

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum Bundling { Separated, SingleFile }

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Default, PartialEq)]
pub struct LicenseFlags {
    pub commercial: Option<bool>,
    pub derivatives: Option<bool>,
    pub different_license: Option<bool>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct DownloadRecipe {
    pub files: Vec<RecipeFile>,
    pub needs_token: Option<TokenKind>,   // None | Some(Hf) | Some(Civitai)
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct RecipeFile {
    pub url: String,
    pub dest: String,                     // template; substitutes {family}, {author}, {name}
    pub sha256: Option<String>,
    pub size_bytes: Option<u64>,
}

pub type CompanionRef = String;            // canonical name; resolved via companions.rs
```

**Design notes:**
- `engine_phase: u8` rather than `enum Phase { P1, P2, ... }` so future phases don't break shard schema. `99` is the sentinel "unsupported in this build".
- `companions` are `String` references, not embedded `CatalogEntry` — companions are a finite curated set defined separately, and we don't want shards storing their own copy of the T5 entry.
- `sub_family` carries Civitai's tribal distinctions (Pony, Illustrious, NoobAI) that matter for prompting even though they're SDXL-architecture under the hood.

### 1.3 Companion registry — `companions.rs`

```rust
pub struct Companion {
    pub canonical_name: &'static str,        // "t5-v1_1-xxl"
    pub kind: Kind,                          // TextEncoder | Vae | ...
    pub family_scope: &'static [Family],     // which families consume this
    pub source: Source,                      // always Hf for canonical companions
    pub repo: &'static str,                  // "google/t5-v1_1-xxl"
    pub files: &'static [&'static str],      // glob patterns within the repo
    pub size_bytes: u64,                     // approximate, for VRAM budgeting in D
}

pub static COMPANIONS: &[Companion] = &[
    Companion {
        canonical_name: "t5-v1_1-xxl",
        kind: Kind::TextEncoder,
        family_scope: &[Family::Flux, Family::Flux2, Family::LtxVideo, Family::Ltx2],
        source: Source::Hf,
        repo: "city96/t5-v1_1-xxl-encoder-bf16",
        files: &["t5xxl_*.safetensors"],
        size_bytes: 9_500_000_000,
    },
    Companion {
        canonical_name: "clip-l",
        kind: Kind::TextEncoder,
        family_scope: &[Family::Flux, Family::Flux2, Family::Sd15, Family::Sdxl],
        source: Source::Hf,
        repo: "openai/clip-vit-large-patch14",
        files: &["model.safetensors", "config.json", "tokenizer*.json", "vocab.json", "merges.txt"],
        size_bytes: 1_700_000_000,
    },
    Companion {
        canonical_name: "clip-g",
        kind: Kind::TextEncoder,
        family_scope: &[Family::Sdxl],
        source: Source::Hf,
        repo: "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        files: &["open_clip_pytorch_model.safetensors", "open_clip_config.json"],
        size_bytes: 5_700_000_000,
    },
    Companion {
        canonical_name: "sdxl-vae",
        kind: Kind::Vae,
        family_scope: &[Family::Sdxl],
        source: Source::Hf,
        repo: "madebyollin/sdxl-vae-fp16-fix",
        files: &["sdxl_vae.safetensors"],
        size_bytes: 335_000_000,
    },
    Companion {
        canonical_name: "sd-vae-ft-mse",
        kind: Kind::Vae,
        family_scope: &[Family::Sd15],
        source: Source::Hf,
        repo: "stabilityai/sd-vae-ft-mse",
        files: &["diffusion_pytorch_model.safetensors", "config.json"],
        size_bytes: 335_000_000,
    },
    Companion {
        canonical_name: "flux-vae",
        kind: Kind::Vae,
        family_scope: &[Family::Flux, Family::Flux2],
        source: Source::Hf,
        repo: "black-forest-labs/FLUX.1-dev",
        files: &["ae.safetensors"],
        size_bytes: 335_000_000,
    },
    // Reserved canonical for Z-Image. The exact text-encoder repo is finalized
    // when the phase 4 single-file loader lands — Z-Image's text-encoder choice
    // is still consolidating upstream. The canonical name "z-image-te" is
    // committed now so phase 1 entries can reference it without rewrites.
    Companion {
        canonical_name: "z-image-te",
        kind: Kind::TextEncoder,
        family_scope: &[Family::ZImage],
        source: Source::Hf,
        repo: "Tongyi-MAI/Z-Image-Turbo",          // text encoder bundled here for now
        files: &["text_encoder/*"],
        size_bytes: 4_400_000_000,
    },
];
```

**Why canonical companions:** Civitai single-file FLUX checkpoints don't bundle T5 (too large). Without a canonical name → repo binding, every Civitai entry would need to specify its own T5 repo, leading to either 50 different "almost-the-same" T5 downloads or arbitrary-repo trust expansion. By forcing a canonical set, mold ships one T5, one CLIP-L, etc., and any single-file checkpoint that doesn't fit gets `engine_phase: 99`.

### 1.4 Civitai mapping — `civitai_map.rs`

```rust
pub fn map_base_model(base_model: &str) -> Option<(Family, FamilyRole, Option<String>)> {
    use Family::*;
    use FamilyRole::*;
    Some(match base_model {
        // SD1.x
        "SD 1.4" | "SD 1.5" | "SD 1.5 LCM" | "SD 1.5 Hyper" => (Sd15, Finetune, None),

        // SDXL family
        "SDXL 1.0" | "SDXL Lightning" | "SDXL Hyper" => (Sdxl, Finetune, None),
        "Pony" => (Sdxl, Finetune, Some("pony".into())),
        "Pony V7" => (Sdxl, Finetune, Some("pony-v7".into())),
        "Illustrious" => (Sdxl, Finetune, Some("illustrious".into())),
        "NoobAI" => (Sdxl, Finetune, Some("noobai".into())),

        // FLUX 1
        "Flux.1 S" => (Flux, Finetune, Some("flux1-s".into())),
        "Flux.1 D" => (Flux, Finetune, Some("flux1-d".into())),
        "Flux.1 Krea" => (Flux, Finetune, Some("flux1-krea".into())),
        "Flux.1 Kontext" => (Flux, Finetune, Some("flux1-kontext".into())),

        // FLUX 2
        "Flux.2 D" => (Flux2, Finetune, Some("flux2-d".into())),
        "Flux.2 Klein 9B" | "Flux.2 Klein 9B-base" => (Flux2, Finetune, Some("klein-9b".into())),
        "Flux.2 Klein 4B" | "Flux.2 Klein 4B-base" => (Flux2, Finetune, Some("klein-4b".into())),

        // Z-Image
        "ZImageTurbo" => (ZImage, Finetune, Some("turbo".into())),
        "ZImageBase" => (ZImage, Finetune, Some("base".into())),

        // LTX
        "LTXV" => (LtxVideo, Finetune, None),
        "LTXV2" => (Ltx2, Finetune, Some("v2".into())),
        "LTXV 2.3" => (Ltx2, Finetune, Some("v2.3".into())),

        // Qwen
        "Qwen" | "Qwen 2" => (QwenImage, Finetune, None),

        // explicit drops (no mold engine)
        "SD 2.0" | "SD 2.1"
        | "AuraFlow" | "Chroma" | "CogVideoX" | "Ernie" | "Grok"
        | "HiDream" | "Hunyuan 1" | "Hunyuan Video"
        | "Kolors" | "Lumina" | "Mochi"
        | "PixArt a" | "PixArt E"
        | "Wan Video 1.3B t2v" | "Wan Video 14B t2v" | "Wan Video 14B i2v 480p" | "Wan Video 14B i2v 720p"
        | "Wan Video 2.2 TI2V-5B" | "Wan Video 2.2 I2V-A14B" | "Wan Video 2.2 T2V-A14B"
        | "Wan Video 2.5 T2V" | "Wan Video 2.5 I2V" | "Wan Image 2.7" | "Wan Video 2.7"
        | "Anima" | "Other" | "Upscaler" => return None,

        _ => return None, // unknown — drop, don't guess
    })
}

pub fn engine_phase_for(family: Family, bundling: Bundling) -> u8 {
    use Family::*;
    use Bundling::*;
    match (family, bundling) {
        // Diffusers HF entries already work via existing engine paths
        (_, Separated) => 1,
        // Single-file phasing
        (Sd15 | Sdxl, SingleFile) => 2,
        (Flux | Flux2, SingleFile) => 3,
        (ZImage, SingleFile) => 4,
        (LtxVideo | Ltx2, SingleFile) => 5,
        // Other families single-file: not in scope
        (_, SingleFile) => 99,
    }
}
```

### 1.5 HF stage — `stages/hf.rs`

For each `Family` we support, walk a curated seed set + linked finetunes.

```rust
pub async fn scan_family(
    client: &reqwest::Client,
    token: Option<&str>,
    family: Family,
    cap: Option<usize>,
) -> Result<Vec<CatalogEntry>, ScanError> {
    let seeds = hf_seeds::seeds_for(family);
    let mut entries = Vec::new();

    for seed in seeds {
        // 1. Add the seed itself as a Foundation entry (verify it exists)
        if let Some(e) = fetch_repo_as_entry(client, token, seed, family, FamilyRole::Foundation).await? {
            entries.push(e);
        }
        // 2. Walk finetunes via base_model tag
        let mut page = 1;
        loop {
            let url = format!(
                "https://huggingface.co/api/models?filter=base_model:{}&sort=downloads&direction=-1&limit=100&page={}",
                seed, page
            );
            let batch: Vec<HfModelStub> = http_json(client, token, &url).await?;
            if batch.is_empty() { break; }
            for stub in batch {
                if let Some(e) = fetch_repo_as_entry(client, token, &stub.id, family, FamilyRole::Finetune).await? {
                    entries.push(e);
                }
                if let Some(c) = cap { if entries.len() >= c { return Ok(entries); } }
            }
            page += 1;
        }
    }
    Ok(entries)
}

async fn fetch_repo_as_entry(
    client: &reqwest::Client,
    token: Option<&str>,
    repo: &str,
    family: Family,
    role: FamilyRole,
) -> Result<Option<CatalogEntry>, ScanError> {
    let info: HfModelDetail = http_json(client, token, &format!("https://huggingface.co/api/models/{}", repo)).await?;
    let tree: Vec<HfFileEntry> = http_json(client, token, &format!("https://huggingface.co/api/models/{}/tree/main?recursive=true", repo)).await?;

    let (file_format, bundling) = detect_format(&tree, family)?;
    let download_recipe = build_recipe_hf(repo, &tree, family, bundling)?;

    Ok(Some(normalizer::from_hf(info, file_format, bundling, family, role, download_recipe)))
}
```

`hf_seeds::seeds_for`:

```rust
pub fn seeds_for(family: Family) -> &'static [&'static str] {
    use Family::*;
    match family {
        Flux => &["black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-schnell"],
        Flux2 => &["black-forest-labs/FLUX.2-dev", "black-forest-labs/FLUX.2-Klein-9B"],
        Sd15 => &["runwayml/stable-diffusion-v1-5", "stable-diffusion-v1-5/stable-diffusion-v1-5"],
        Sdxl => &["stabilityai/stable-diffusion-xl-base-1.0"],
        ZImage => &["Tongyi-MAI/Z-Image-Turbo", "Tongyi-MAI/Z-Image-Base"],
        LtxVideo => &["Lightricks/LTX-Video"],
        Ltx2 => &["Lightricks/LTX-Video-2", "Lightricks/LTX-Video-2.3"],
        QwenImage => &["Qwen/Qwen-Image"],
        Wuerstchen => &["warp-ai/wuerstchen"],
    }
}
```

### 1.6 Civitai stage — `stages/civitai.rs`

```rust
pub async fn scan(
    client: &reqwest::Client,
    token: Option<&str>,
    options: &ScanOptions,
) -> Result<Vec<CatalogEntry>, ScanError> {
    let mut entries = Vec::new();
    for base_model in CIVITAI_BASE_MODELS {  // every key in civitai_map.rs that maps Some(...)
        let Some((family, role, sub_family)) = map_base_model(base_model) else { continue };
        let mut page = 1;
        loop {
            let url = format!(
                "https://civitai.com/api/v1/models?baseModels={}&types=Checkpoint&sort=Most+Downloaded&limit=100&page={}",
                urlencoding::encode(base_model), page
            );
            let resp: CivitaiModelsResponse = http_json(client, token, &url).await?;
            if resp.items.is_empty() { break; }

            for item in resp.items {
                let Some(version) = item.model_versions.first() else { continue };
                let Some(file) = pick_safetensors_file(&version.files) else { continue }; // hard-skip unsafe formats
                if file.download_count < options.min_downloads { continue; }

                let bundling = if version.base_model_type == Some("Standard".into()) { Bundling::SingleFile } else { Bundling::Separated };
                let companions = companions_for(family, bundling);
                let phase = engine_phase_for(family, bundling);

                entries.push(normalizer::from_civitai(item, version, file, family, role, sub_family.clone(), bundling, companions, phase));
            }

            page += 1;
            if resp.metadata.next_page.is_none() { break; }
        }
    }
    Ok(entries)
}

// Only safetensors are accepted. Civitai's legacy ".pt" binary format is
// dropped at scan time — its deserialization model is unsafe and we will
// not surface it to users.
fn pick_safetensors_file<'a>(files: &'a [CivitaiFile]) -> Option<&'a CivitaiFile> {
    files.iter().find(|f| f.metadata.format.as_deref() == Some("SafeTensor"))
}
```

**Civitai auth:** if `token` is `Some`, sent as `Authorization: Bearer <token>`. Required for early-access content and some NSFW; absent token still returns most public catalog.

### 1.7 Filter — `filter.rs`

```rust
pub fn apply(entries: Vec<CatalogEntry>, options: &ScanOptions) -> Vec<CatalogEntry> {
    entries.into_iter()
        .filter(|e| !e.download_recipe.files.is_empty())
        .filter(|e| match e.source {
            Source::Civitai => e.download_count >= options.min_downloads,
            Source::Hf => true,
        })
        .filter(|e| options.include_nsfw || !e.nsfw)        // scanner-time NSFW; UI also filters
        .collect()
}
```

Note: the scanner does not enforce the user's NSFW preference — that's a UI concern. Scanner-time NSFW filtering is only when the user explicitly passes `--no-nsfw` to refresh.

### 1.8 Sink — `sink.rs`

Two-phase commit per family:

1. Serialize entries to canonical JSON (sorted by `(family_role, download_count desc, name)`, indented 2 spaces, trailing newline).
2. Write to `<dest_dir>/.staging/<family>.json`.
3. Validate: `serde_json::from_str::<Shard>(...)` round-trips byte-equal.
4. `fs::rename` to `<dest_dir>/<family>.json` (POSIX atomic on same filesystem).
5. Per-family DB upsert in one transaction:
   ```sql
   BEGIN;
   DELETE FROM catalog WHERE family = ?;
   INSERT INTO catalog (...) VALUES ... ;
   INSERT INTO catalog_fts(catalog_fts) VALUES('rebuild');
   COMMIT;
   ```

**Determinism:** sort ordering is canonical; if a re-scan finds zero changes, the JSON shard is byte-identical and `git diff` shows nothing. Maintainer can run `mold catalog refresh` weekly without polluting commit history.

### 1.9 mold-db migration

Add to `crates/mold-db/src/migrations.rs`:

```rust
const MIGRATIONS: &[Migration] = &[
    // ... existing migrations ...
    Migration {
        version: SCHEMA_VERSION,    // bump to next
        description: "catalog table for sub-project A",
        sql: include_str!("sql/V<n>__catalog.sql"),
    },
];
```

`sql/V<n>__catalog.sql`:

```sql
CREATE TABLE catalog (
    id            TEXT PRIMARY KEY,           -- "hf:author/repo" | "cv:<modelVersionId>"
    source        TEXT NOT NULL,
    source_id     TEXT NOT NULL,
    name          TEXT NOT NULL,
    author        TEXT,
    family        TEXT NOT NULL,
    family_role   TEXT NOT NULL,
    sub_family    TEXT,
    modality      TEXT NOT NULL,
    kind          TEXT NOT NULL,              -- "checkpoint" (reserved: "lora", "vae", "te", "controlnet")
    file_format   TEXT NOT NULL,
    bundling      TEXT NOT NULL,
    size_bytes    INTEGER,
    download_count INTEGER DEFAULT 0,
    rating        REAL,
    likes         INTEGER DEFAULT 0,
    nsfw          INTEGER DEFAULT 0,
    thumbnail_url TEXT,
    description   TEXT,
    license       TEXT,
    license_flags TEXT,                       -- JSON
    tags          TEXT,                       -- JSON
    companions    TEXT,                       -- JSON
    download_recipe TEXT NOT NULL,            -- JSON
    engine_phase  INTEGER NOT NULL,
    created_at    INTEGER,
    updated_at    INTEGER,
    added_at      INTEGER,
    UNIQUE (source, source_id)
);

CREATE INDEX idx_catalog_family ON catalog(family, family_role);
CREATE INDEX idx_catalog_modality ON catalog(modality);
CREATE INDEX idx_catalog_downloads ON catalog(download_count DESC);
CREATE INDEX idx_catalog_updated ON catalog(updated_at DESC);
CREATE INDEX idx_catalog_rating ON catalog(rating DESC);
CREATE INDEX idx_catalog_phase ON catalog(engine_phase);

CREATE VIRTUAL TABLE catalog_fts USING fts5(
    name, author, description, tags, content='catalog', content_rowid='rowid'
);
```

Catalog is **global per mold install**, not per profile. Rationale: scanning is expensive and the catalog is upstream-objective, not a user preference. The table omits a `profile` column. Per-profile state (downloaded? hidden by user? favorited?) lives in existing tables.

### 1.10 mold-server endpoints — `mold-server/src/catalog_api.rs` (new)

```
GET  /api/catalog                 ?family=&modality=&source=&sub_family=&q=&sort=&page=&page_size=&include_nsfw=
GET  /api/catalog/:id
GET  /api/catalog/families        // for sidebar; returns counts by (family, family_role)
POST /api/catalog/refresh         // long-running; returns job id
GET  /api/catalog/refresh/:id     // poll status
POST /api/catalog/:id/download    // enqueues download via existing downloads.rs
```

`POST /api/catalog/refresh` reuses the same `DownloadQueue`-style single-writer pattern from the model-ui-overhaul work — only one scan at a time per server. Progress events broadcast over the existing event bus.

**Reuses existing downloads.rs.** A catalog `download_recipe.files` array becomes a sequence of `pull_model_with_callback` invocations; companions are enqueued first if missing.

### 1.11 mold-cli — `mold catalog` subcommand

```rust
#[derive(clap::Subcommand)]
pub enum CatalogCmd {
    /// List entries with filters
    List(ListArgs),
    /// Show a single entry by id
    Show { id: String, #[arg(long)] json: bool },
    /// Re-run the scanner, write shards, reseed DB
    Refresh(RefreshArgs),
    /// Print local path for a downloaded entry, or "<not downloaded>"
    Where { id: String },
}

#[derive(clap::Args)]
pub struct ListArgs {
    #[arg(long)] family: Option<String>,
    #[arg(long)] modality: Option<String>,
    #[arg(long)] source: Option<String>,
    #[arg(long)] sub_family: Option<String>,
    #[arg(long)] q: Option<String>,
    #[arg(long, default_value = "downloads")] sort: String,   // downloads | rating | recent | name
    #[arg(long, default_value_t = 20)] limit: usize,
    #[arg(long)] json: bool,
    #[arg(long)] downloaded_only: bool,
    #[arg(long)] include_nsfw: bool,
}

#[derive(clap::Args)]
pub struct RefreshArgs {
    #[arg(long)] family: Option<String>,
    #[arg(long, default_value_t = 100)] min_downloads: u64,
    #[arg(long)] no_nsfw: bool,
    #[arg(long)] dry_run: bool,
    /// For maintainer use: write into `data/catalog/` instead of `$MOLD_HOME/catalog/`
    #[arg(long)] commit_to_repo: bool,
}
```

`mold pull <id>` integration: extend the existing pull logic — if the argument starts with `cv:` or `hf:` and is a colon-prefixed catalog id, look up the row in the catalog table, follow `download_recipe`, auto-enqueue companions. Existing `mold pull <model-name>` keeps working.

### 1.12 Web UI — `/catalog` route

New top-level route in `web/src/router.ts`. Pinia store `useCatalogStore` handles fetch + filters.

**Layout (Approach A from the brainstorm visual):**

- **Sidebar** (`<CatalogSidebar />`): rendered from `GET /api/catalog/families`. Each family row collapses to two children (Foundation / Fine-tunes) with counts. Active row highlighted with left blue bar.
- **Top bar** (`<CatalogTopbar />`):
  - Modality chips: `All / Image / Video`
  - Search input (debounced 250 ms; FTS5 query)
  - Sort dropdown
  - Source chips: `HF / Civitai`
- **Card grid** (`<CatalogCardGrid />`):
  - Responsive 1/2/3/4 column grid via Tailwind
  - Card components show thumb, name, author, size, source badge, downloaded indicator, engine_phase badge if !=1
  - Lazy-load thumbnails with IntersectionObserver
  - Virtual-scroll via `vue-virtual-scroller` for >200 visible rows
- **Detail drawer** (`<CatalogDetailDrawer />`): right-slide panel; full description, license, tags, companions, Download/Cancel/Delete button, "Generate now" button if downloaded.

**Settings additions** (`/settings`):
- "Connect Hugging Face" row: token input, mirrors to `huggingface.token`.
- "Connect Civitai" row: token input, mirrors to `civitai.token`.
- "Show NSFW models" toggle: mirrors to `catalog.show_nsfw`.

### 1.13 Build-time embedding — `mold-catalog/build.rs`

Pattern follows `mold-server/build.rs`:

```rust
fn main() {
    println!("cargo:rerun-if-changed=data/catalog");
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/catalog");
    if !dir.join("flux.json").exists() {
        // Emit a stub shard so first-time `cargo build` doesn't fail
        for fam in &["flux", "flux2", "sd15", "sdxl", "z-image", "ltx-video", "ltx-2", "qwen-image", "wuerstchen"] {
            let stub = dir.join(format!("{}.json", fam));
            if !stub.exists() {
                fs::create_dir_all(&dir).unwrap();
                fs::write(&stub, format!(r#"{{"$schema":"mold.catalog.v1","family":"{}","generated_at":"1970-01-01T00:00:00Z","scanner_version":"0.0","entries":[]}}"#, fam)).unwrap();
            }
        }
    }
}
```

Runtime detects empty shards and shows a "Run `mold catalog refresh` to populate" empty-state in the SPA — same trick as the existing web/dist stub.

### 1.14 First-run seed flow

```rust
pub fn seed_db_from_embedded_if_empty(db: &mold_db::Db) -> Result<(), Error> {
    let count: i64 = db.query_one("SELECT COUNT(*) FROM catalog", [])?;
    if count > 0 { return Ok(()); }

    for shard in shards::EMBEDDED_SHARDS {        // rust-embed iterator
        let shard: Shard = serde_json::from_slice(shard.data.as_ref())?;
        if shard.entries.is_empty() { continue; } // skip stubs
        sink::seed_family(db, shard.family, shard.entries)?;
    }
    Ok(())
}
```

Runs on `mold-server` startup; idempotent.

### 1.15 Phase 1 testing

- **Snapshot tests** (`mold-catalog/tests/normalizer_snapshots.rs`): canned HF + Civitai responses → assert `CatalogEntry` JSON matches `.snap`. ~20 representative cases (each family, separated + single-file, NSFW, missing thumbnail, missing rating, empty file list).
- **Mapping completeness** (`tests/civitai_map_completeness.rs`): static test that asserts every key in `CIVITAI_BASE_MODELS` either maps to a `Family` or is in the explicit drop set. Prevents silent drift when Civitai adds a new `baseModel`.
- **Round-trip** (`tests/shard_roundtrip.rs`): every committed shard parses, re-serializes byte-identical. Catches accidental hand-edits.
- **DB seed** (`tests/db_seed.rs`): load all shards into a `:memory:` DB, run all FTS queries, assert results.
- **CLI golden** (`tests/cli_list.rs`): spawn `mold catalog list --json` against fixture DB, assert stable output.
- **No live HTTP in CI.** Real HF/Civitai calls only run with `cargo test --ignored` (manual on <gpu-host>).

---

## 2. Phase 2 — SD1.5 + SDXL single-file loaders

### 2.1 New module — `mold-inference/src/loader/single_file.rs`

```rust
pub struct SingleFileBundle {
    pub unet: SafeTensors,
    pub vae: Option<SafeTensors>,
    pub clip_l: Option<SafeTensors>,
    pub clip_g: Option<SafeTensors>,
    pub t5: Option<SafeTensors>,           // unused for SD1.5/SDXL but reserved
}

pub fn load(path: &Path, family: Family) -> Result<SingleFileBundle, LoadError> {
    let mmap = unsafe { Mmap::map(&File::open(path)?)? };
    let st = SafeTensors::deserialize(&mmap[..])?;
    match family {
        Family::Sd15 => sd15::load(st),
        Family::Sdxl => sdxl::load(st),
        Family::Flux | Family::Flux2 => flux::load(st),       // phase 3
        Family::ZImage => z_image::load(st),                  // phase 4
        Family::LtxVideo | Family::Ltx2 => ltx::load(st),     // phase 5
        _ => Err(LoadError::Unsupported(family)),
    }
}
```

### 2.2 SD1.5 loader — `mold-inference/src/sd15/single_file.rs`

Tensor prefixes (A1111 convention):
- UNET: `model.diffusion_model.*` → reshape to mold's expected SD1.5 UNET keys
- VAE: `first_stage_model.*` (optional; fall back to companion `sd-vae-ft-mse` if absent)
- CLIP-L: `cond_stage_model.transformer.text_model.*`

Reuses the existing SD1.5 engine paths once components are extracted. Engine entry point becomes:

```rust
pub fn from_single_file(path: &Path, vae_companion: Option<&Path>) -> Result<Sd15Engine, EngineError>;
```

### 2.3 SDXL loader — `mold-inference/src/sdxl/single_file.rs`

Same shape but two text encoders:
- CLIP-L: `conditioner.embedders.0.transformer.text_model.*`
- CLIP-G: `conditioner.embedders.1.model.*`

Pony / Illustrious / NoobAI all use this loader unchanged — they're tensor-compatible. Sub-family is metadata for prompting hints in the UI only.

### 2.4 Wire into pull → run

When the user runs `mold pull cv:618692` (an SDXL Civitai entry), the `download_recipe` writes to `models/sdxl/RunDiffusion/Juggernaut-XL.safetensors`. When `mold run cv:618692 "prompt"` fires, the engine loader sees the catalog entry's `bundling: SingleFile` flag and calls `sdxl::single_file::load(...)` instead of the diffusers loader.

### 2.5 Companion auto-pull on download

In phase 1, companions are tracked but unenforced. In phase 2, `POST /api/catalog/:id/download` enqueues missing companions before the entry itself. UI shows "Downloading companion: clip-l (1.7 GB)" then "Downloading: Juggernaut XL v10 (6.9 GB)".

### 2.6 Phase 2 testing

- **Tensor extraction** (`tests/sdxl_single_file.rs`): fixture safetensors with known structure, assert UNET/VAE/CLIP-L/CLIP-G come out matching expected shapes.
- **End-to-end small generation**: mark `#[ignore]`; <gpu-host> UAT executes manually.
- **Companion auto-pull**: integration test with mocked downloader, assert the order is companions-first.

---

## 3. Phase 3 — FLUX single-file (all versions)

### 3.1 FLUX single-file conventions

FLUX checkpoints on Civitai are *transformer-only* — T5 and CLIP-L are too large to bundle (T5-XXL alone is 9.5 GB). The single file holds:
- `model.diffusion_model.*` (the transformer blocks)
- `vae.*` (sometimes; sometimes baked-out)

**Companions always required:** `["t5-v1_1-xxl", "clip-l"]`. VAE optional (FLUX has its own; companion fallback is `flux-vae` which we add to `companions.rs`).

### 3.2 Loader — `mold-inference/src/flux/single_file.rs`

```rust
pub fn from_single_file(
    path: &Path,
    t5_path: &Path,
    clip_l_path: &Path,
    vae_path: Option<&Path>,
) -> Result<FluxEngine, EngineError>;
```

Reuses the existing FLUX engine; only the transformer source changes from "load from HF repo files" to "load from prefixed tensors in single file".

### 3.3 Variant detection

Civitai sub_family values (`flux1-s`, `flux1-d`, `flux1-krea`, `flux1-kontext`, `flux2-d`, `klein-9b`, `klein-4b`) drive runtime parameters that already exist in mold's FLUX engine config (block count, attention dims, etc.). The single-file loader passes `sub_family` through.

### 3.4 Flux.2 / Klein

Flux.2 architecture differs from Flux.1 (different block layout). Catalog `family: Flux2` triggers the Flux.2 engine, not Flux. Klein 9B / 4B are size variants of Flux.2-D — same loader, different config.

### 3.5 Phase 3 testing

Same shape as phase 2: fixture safetensors, tensor extraction tests, <gpu-host> UAT for end-to-end.

---

## 4. Phase 4 — Z-Image single-file (Turbo + Base)

### 4.1 Z-Image loader — `mold-inference/src/z_image/single_file.rs`

Z-Image already has a bespoke quantized transformer in mold (`zimage/quantized_transformer.rs`). Single-file loader extracts:
- Transformer: family-specific prefix (validated against fixture safetensors during phase 4 implementation; expected `model.transformer.*` based on existing `quantized_transformer.rs` naming)
- VAE: `vae.*`
- Companion: `z-image-te` (canonical companion declared in `companions.rs`; final text-encoder source resolved as part of phase 4)

Variants: `turbo` and `base` differ in step count and CFG defaults — runtime config, not loader concern.

### 4.2 Phase 4 testing

Same shape; tiny ecosystem so manual UAT covers most cases.

---

## 5. Phase 5 — LTX single-file (LTXV / LTXV2 / LTXV 2.3)

### 5.1 LTX loader — `mold-inference/src/ltx*/single_file.rs`

Two loaders (one per engine crate module):
- `ltx_video/single_file.rs` for LTXV (the original)
- `ltx_2/single_file.rs` for LTXV2 + LTXV 2.3

Both are video transformer + VAE; T5 companion. Sub-family `v2` vs `v2.3` is runtime config (frame count, block layout deltas).

### 5.2 CUDA-only constraint

CLAUDE.md notes LTX-2 is CUDA-only for real generation. The single-file loaders inherit this — Metal is unsupported, CPU is correctness-only. Catalog entries for LTX families don't filter by host; the engine surfaces the constraint at run-time as it does today.

---

## 6. Error handling

- **API failure mid-scan**: per-family scan is independent. One family's `ScanError` does not abort the run. Refresh CLI prints summary table with `ok (N) | rate-limited (got N/?) | network-error | auth-required`. Successful families commit their shards; failed ones leave the previous shard untouched.
- **Shard schema drift**: every shard declares `$schema: "mold.catalog.v1"`. Loader refuses unknown schema strings. Future v2 introduces an explicit migration step.
- **Companion missing at run-time**: `EngineError::CompanionMissing { canonical_name, suggested_action }`. CLI hint: `mold pull <canonical-name>`. Web UI shows banner with one-click resolution.
- **NSFW state mismatch**: scanner records what Civitai says. If a download later turns out to be NSFW (e.g., reuploaded content), user can right-click → "Mark as NSFW" → updates DB row.
- **Token required, not provided**: clear error in both surfaces. UI shows "Add a Civitai token in Settings → Connect Civitai" with deep-link.
- **Unsafe format in API response**: never reach `CatalogEntry`. Logged at `info` level so we can audit how often Civitai returns the legacy `.pt` format.

---

## 7. Surfaces touched (cross-reference)

| Layer | New | Modified |
|---|---|---|
| `crates/mold-catalog/` | entire crate | — |
| `crates/mold-db/` | `sql/V<n>__catalog.sql` | `migrations.rs` (bump `SCHEMA_VERSION`) |
| `crates/mold-core/` | `families.rs` re-export check | possibly add `Family` variants if missing (Flux2, ZImage etc. — verify) |
| `crates/mold-cli/` | `commands/catalog.rs` | `main.rs` (subcommand wiring); `commands/pull.rs` (catalog-id resolution) |
| `crates/mold-server/` | `catalog_api.rs` | `lib.rs` (router), reuses `downloads.rs` queue |
| `crates/mold-inference/` (phase 2) | `loader/single_file.rs`, `sd15/single_file.rs`, `sdxl/single_file.rs` | family-engine entry points |
| `crates/mold-inference/` (phase 3) | `flux/single_file.rs` | flux engine entry |
| `crates/mold-inference/` (phase 4) | `z_image/single_file.rs` | z-image engine entry |
| `crates/mold-inference/` (phase 5) | `ltx_video/single_file.rs`, `ltx_2/single_file.rs` | engine entries |
| `web/` | `views/Catalog.vue`, `components/Catalog*`, `stores/catalog.ts`, `router` route | `views/Settings.vue` (HF/Civitai/NSFW), nav link |
| `data/catalog/` | committed shards | — |

---

## 8. CHANGELOG / docs sync

Per CLAUDE.md: every shipped phase updates these in lockstep:

- `CHANGELOG.md` `[Unreleased]` — new endpoints, new CLI commands, new env vars (`CIVITAI_TOKEN`).
- `.claude/skills/mold/SKILL.md` — agent-facing knowledge of `mold catalog *` and `mold pull <catalog-id>`.
- `website/` — VitePress page for the catalog feature, settings page mentions `civitai.token`.
- This spec stays authoritative; PR descriptions reference it.

## 9. Open questions reserved for sub-projects B, C, D

- **B (Civitai as a download source for arbitrary URLs):** how `mold pull <civitai-direct-url>` interacts with the catalog (probably: not in catalog → ad-hoc download path). Catalog auth flow lays groundwork.
- **C (per-stage model in script mode):** chain stages will reference catalog entries by `id`. The catalog table is the source of truth for the picker UI inside the chain composer.
- **D (LoRA browsing + VRAM guardrail):** `kind = lora` rows arrive in this catalog table via a sub-project D scanner pass. UI extension: LoRA picker drawer in chain stages and the generate page.

These dependencies are the reason A ships first.
