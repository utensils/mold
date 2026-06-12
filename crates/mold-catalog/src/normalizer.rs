//! Source-specific JSON → `CatalogEntry`.
//!
//! HF: combine `/api/models/{repo}` detail + `/api/models/{repo}/tree/main`.
//! Civitai: combine the model + a chosen version + a chosen safetensors file.

use serde::Deserialize;

use crate::civitai_map::{engine_phase_for, map_base_model};
use crate::companions::companions_for;
use crate::entry::{
    Bundling, CatalogEntry, CatalogId, DownloadRecipe, FamilyRole, FileFormat, Kind, LicenseFlags,
    Modality, RecipeFile, RecipeFileRole, Source, TokenKind,
};
use crate::families::Family;

#[derive(Clone, Debug, Deserialize)]
pub struct HfDetail {
    pub id: String,
    pub author: Option<String>,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: u64,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default, rename = "pipeline_tag")]
    pub pipeline_tag: Option<String>,
    #[serde(default, rename = "library_name")]
    pub library_name: Option<String>,
    #[serde(default, rename = "createdAt")]
    pub created_at: Option<String>,
    #[serde(default, rename = "lastModified")]
    pub last_modified: Option<String>,
    #[serde(default, rename = "cardData")]
    pub card_data: Option<HfCardData>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct HfCardData {
    pub license: Option<String>,
    #[serde(default)]
    pub extra_gated_eu_disallowed: Option<bool>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct HfTreeEntry {
    #[serde(rename = "type")]
    pub kind: String,
    pub path: String,
    #[serde(default)]
    pub size: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum NormalizeError {
    #[error("no usable safetensors / diffusers payload found")]
    EmptyTree,
}

const HF_RAW: &str = "https://huggingface.co";

/// Map Civitai's `type` string (per `/api/v1/models` schema) to the
/// catalog's `Kind`. Returning `None` drops the entry entirely — used for
/// types mold doesn't yet model (TextualInversion, Hypernetwork, Poses,
/// AestheticGradient). LoCon is grouped with Lora since their inference
/// path is the same patch-on-base-weights merge.
pub(crate) fn civitai_kind_to_catalog_kind(s: &str) -> Option<Kind> {
    match s {
        "Checkpoint" => Some(Kind::Checkpoint),
        "LORA" | "LoCon" | "DoRA" => Some(Kind::Lora),
        "VAE" => Some(Kind::Vae),
        "Controlnet" | "ControlNet" => Some(Kind::ControlNet),
        // TextualInversion / Hypernetwork / Poses / AestheticGradient / Other
        // — drop. The catalog has no slot for these and dropping is safer
        // than misclassifying as Checkpoint (which would let users try to
        // generate from them and fail mysteriously at load time).
        _ => None,
    }
}

/// Heuristic Kind detection for HF entries. HF has no canonical type
/// field, so we read tags, file names, and the repo id in priority order.
///
/// The heuristic is permissive (false negatives are fine — the entry just
/// stays a Checkpoint and the LoRA loader on the engine side won't
/// recognize it for what it is) but correctness matters when LoRA = true,
/// because misclassifying a checkpoint AS a LoRA would skip the companion
/// graph and leave the user with a download that can't run.
fn classify_hf_kind(detail: &HfDetail, files: &[RecipeFile]) -> Kind {
    // Tag-driven: HF curators tag LoRA repos with "lora" or "adapter".
    if detail
        .tags
        .iter()
        .any(|t| t.eq_ignore_ascii_case("lora") || t.eq_ignore_ascii_case("adapter"))
    {
        return Kind::Lora;
    }
    // File-name signal: the HF diffusers convention writes
    // `pytorch_lora_weights.safetensors` for LoRA repos.
    let any_file_is_lora = files.iter().any(|f| {
        let url_lower = f.url.to_ascii_lowercase();
        url_lower.contains("pytorch_lora_weights")
            || url_lower.contains("/lora/")
            || url_lower.ends_with("_lora.safetensors")
            || url_lower.ends_with("-lora.safetensors")
    });
    if any_file_is_lora {
        return Kind::Lora;
    }
    // Repo-id substring: `*-lora`, `lora-*`, `*_lora` are nearly always LoRAs.
    let id_lower = detail.id.to_ascii_lowercase();
    if id_lower.contains("-lora")
        || id_lower.contains("lora-")
        || id_lower.contains("_lora")
        || id_lower.contains("/lora")
    {
        return Kind::Lora;
    }
    Kind::Checkpoint
}

pub fn from_hf(
    detail: HfDetail,
    tree: Vec<HfTreeEntry>,
    family: Family,
    family_role: FamilyRole,
) -> Result<CatalogEntry, NormalizeError> {
    if tree.is_empty() {
        return Err(NormalizeError::EmptyTree);
    }

    let bundling = if tree
        .iter()
        .any(|e| e.kind == "file" && e.path == "model_index.json")
    {
        Bundling::Separated
    } else if tree
        .iter()
        .any(|e| e.kind == "file" && e.path.ends_with(".safetensors") && !e.path.contains('/'))
    {
        Bundling::SingleFile
    } else {
        Bundling::Separated
    };

    let file_format = if tree
        .iter()
        .any(|e| e.kind == "file" && e.path.ends_with(".gguf"))
    {
        FileFormat::Gguf
    } else if tree
        .iter()
        .any(|e| e.kind == "file" && e.path.ends_with(".safetensors") && !e.path.contains('/'))
    {
        FileFormat::Safetensors
    } else if matches!(bundling, Bundling::Separated) {
        FileFormat::Diffusers
    } else {
        FileFormat::Safetensors
    };

    let needs_token = if detail
        .card_data
        .as_ref()
        .and_then(|c| c.extra_gated_eu_disallowed)
        .unwrap_or(false)
        || detail.tags.iter().any(|t| t == "gated")
    {
        Some(TokenKind::Hf)
    } else {
        None
    };

    let mut files: Vec<RecipeFile> = tree
        .iter()
        .filter(|e| {
            e.kind == "file"
                && (e.path.ends_with(".safetensors")
                    || e.path.ends_with(".gguf")
                    || e.path == "model_index.json"
                    || e.path.ends_with("config.json"))
        })
        .map(|e| RecipeFile {
            url: format!("{HF_RAW}/{}/resolve/main/{}", detail.id, e.path),
            dest: format!("{{family}}/{{author}}/{{name}}/{}", e.path),
            sha256: None,
            size_bytes: if e.size > 0 { Some(e.size) } else { None },
            role: None,
        })
        .collect();

    if files.is_empty() {
        return Err(NormalizeError::EmptyTree);
    }
    files.sort_by(|a, b| a.url.cmp(&b.url));

    let total_size = files.iter().filter_map(|f| f.size_bytes).sum::<u64>();
    let modality = match family {
        Family::LtxVideo | Family::Ltx2 => Modality::Video,
        _ => Modality::Image,
    };

    let kind = classify_hf_kind(&detail, &files);
    let companions = match bundling {
        // HF entries don't currently carry a sub_family — pass None and let
        // `companions_for` use its default branch. Single-file HF Flux.2
        // entries are rare; the Civitai path is the load-bearing one.
        Bundling::SingleFile => companions_for(family, None, bundling, kind),
        Bundling::Separated => Vec::new(),
    };
    let phase = engine_phase_for(family, bundling, kind);

    let now = chrono_now_unix();

    Ok(CatalogEntry {
        id: CatalogId::from(format!("hf:{}", detail.id)),
        source: Source::Hf,
        source_id: detail.id.clone(),
        name: detail
            .id
            .split('/')
            .next_back()
            .unwrap_or(&detail.id)
            .to_string(),
        author: detail.author.clone(),
        family,
        family_role,
        sub_family: None,
        modality,
        kind,
        file_format,
        bundling,
        size_bytes: if total_size > 0 {
            Some(total_size)
        } else {
            None
        },
        download_count: detail.downloads,
        rating: None,
        likes: detail.likes,
        nsfw: false,
        thumbnail_url: None,
        description: None,
        license: detail.card_data.as_ref().and_then(|c| c.license.clone()),
        license_flags: LicenseFlags::default(),
        tags: detail.tags.clone(),
        companions,
        download_recipe: DownloadRecipe { files, needs_token },
        engine_phase: phase,
        created_at: parse_iso(&detail.created_at),
        updated_at: parse_iso(&detail.last_modified),
        added_at: now,
        // HF doesn't surface a trigger-words equivalent — the metadata
        // varies per repo (cardData, README, etc.) and isn't worth a
        // brittle scrape. Empty here is the right default; civitai is
        // where 99% of LoRA trigger phrases live anyway.
        trained_words: Vec::new(),
    })
}

fn parse_iso(opt: &Option<String>) -> Option<i64> {
    opt.as_deref().and_then(|s| {
        time::OffsetDateTime::parse(s, &time::format_description::well_known::Iso8601::DEFAULT)
            .ok()
            .map(|dt| dt.unix_timestamp())
    })
}

fn chrono_now_unix() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ── Civitai ────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Deserialize)]
pub struct CivitaiItem {
    pub id: u64,
    pub name: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub nsfw: bool,
    #[serde(default)]
    pub creator: Option<CivitaiCreator>,
    #[serde(default)]
    pub stats: Option<CivitaiStats>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default, rename = "modelVersions")]
    pub model_versions: Vec<CivitaiVersion>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct CivitaiCreator {
    pub username: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct CivitaiStats {
    #[serde(default, rename = "downloadCount")]
    pub download_count: u64,
    #[serde(default)]
    pub rating: Option<f32>,
    #[serde(default, rename = "favoriteCount")]
    pub favorite_count: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CivitaiVersion {
    pub id: u64,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(rename = "baseModel")]
    pub base_model: String,
    #[serde(default, rename = "baseModelType")]
    pub base_model_type: Option<String>,
    #[serde(default)]
    pub files: Vec<CivitaiFile>,
    #[serde(default)]
    pub images: Vec<CivitaiImage>,
    #[serde(default)]
    pub availability: Option<String>,
    /// Trigger phrases the LoRA was trained on (Civitai's `trainedWords`).
    /// Empty for non-LoRA entries; the catalog wire format passes them
    /// through to the web UI which renders click-to-insert chips.
    #[serde(default, rename = "trainedWords")]
    pub trained_words: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CivitaiFile {
    pub id: u64,
    pub name: String,
    #[serde(default, rename = "type")]
    pub file_type: Option<String>,
    #[serde(default, rename = "sizeKB")]
    pub size_kb: Option<f64>,
    #[serde(default, rename = "downloadCount")]
    pub download_count: u64,
    #[serde(default)]
    pub metadata: CivitaiFileMetadata,
    #[serde(default, rename = "downloadUrl")]
    pub download_url: Option<String>,
    #[serde(default)]
    pub hashes: serde_json::Value,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct CivitaiFileMetadata {
    pub format: Option<String>,
    pub size: Option<String>,
    pub fp: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CivitaiImage {
    pub url: String,
    #[serde(default, rename = "nsfwLevel")]
    pub nsfw_level: Option<u32>,
}

pub fn from_civitai(item: CivitaiItem) -> Option<CatalogEntry> {
    let version = item.model_versions.first()?;
    from_civitai_version(&item, version)
}

pub fn from_civitai_search_entries(item: CivitaiItem) -> Vec<CatalogEntry> {
    item.model_versions
        .iter()
        .filter(|version| version_is_public(version))
        .filter_map(|version| from_civitai_version(&item, version))
        .collect()
}

fn version_is_public(version: &CivitaiVersion) -> bool {
    matches!(
        version.availability.as_deref(),
        None | Some("Public") | Some("public")
    )
}

fn from_civitai_version(item: &CivitaiItem, version: &CivitaiVersion) -> Option<CatalogEntry> {
    let (family, family_role, sub_family) = map_base_model(&version.base_model)?;
    let file = pick_safetensors(&version.files)?;
    // Drop entries whose Civitai type isn't representable in the catalog
    // (TextualInversion, Hypernetwork, etc.) before doing any further work.
    let kind = civitai_kind_to_catalog_kind(&item.kind)?;
    let bundling = if version.base_model_type.as_deref() == Some("Standard") {
        Bundling::SingleFile
    } else {
        Bundling::Separated
    };
    let companions = companions_for(family, sub_family.as_deref(), bundling, kind);
    let phase = engine_phase_for(family, bundling, kind);
    let modality = match family {
        Family::LtxVideo | Family::Ltx2 => Modality::Video,
        _ => Modality::Image,
    };

    let mut recipe_files = vec![civitai_recipe_file(version.id, file, None)];
    if family == Family::ZImage {
        if let Some(text_encoder_file) = pick_civitai_text_encoder(&version.files) {
            recipe_files.push(civitai_recipe_file(
                version.id,
                text_encoder_file,
                Some(RecipeFileRole::TextEncoder),
            ));
        }
    }

    let recipe = DownloadRecipe {
        files: recipe_files,
        needs_token: Some(TokenKind::Civitai),
    };

    let stats = item.stats.clone().unwrap_or_default();
    let now = chrono_now_unix();

    let trained_words = version.trained_words.clone();

    Some(CatalogEntry {
        id: CatalogId::from(format!("cv:{}", version.id)),
        source: Source::Civitai,
        source_id: version.id.to_string(),
        name: civitai_entry_name(&item.name, version.name.as_deref()),
        author: item.creator.as_ref().and_then(|c| c.username.clone()),
        family,
        family_role,
        sub_family,
        modality,
        kind,
        file_format: FileFormat::Safetensors,
        bundling,
        size_bytes: file.size_kb.map(|kb| (kb * 1000.0) as u64),
        download_count: stats.download_count,
        rating: stats.rating,
        likes: stats.favorite_count,
        nsfw: item.nsfw,
        thumbnail_url: version.images.first().map(|i| i.url.clone()),
        description: None,
        license: None,
        license_flags: LicenseFlags::default(),
        tags: item.tags.clone(),
        companions,
        download_recipe: recipe,
        engine_phase: phase,
        created_at: None,
        updated_at: None,
        added_at: now,
        trained_words,
    })
}

fn civitai_entry_name(model_name: &str, version_name: Option<&str>) -> String {
    let Some(version_name) = version_name.map(str::trim).filter(|s| !s.is_empty()) else {
        return model_name.to_string();
    };
    if version_name.eq_ignore_ascii_case(model_name) {
        model_name.to_string()
    } else {
        format!("{model_name} - {version_name}")
    }
}

/// Civitai's legacy unsafe `.pt` ("PickleTensor") format is dropped at the
/// scanner. Arbitrary-code-execution risk on deserialization is not worth
/// catalog completeness — only safetensors are surfaced.
fn pick_safetensors(files: &[CivitaiFile]) -> Option<&CivitaiFile> {
    let is_safetensors = |file: &&CivitaiFile| {
        file.metadata.format.as_deref() == Some("SafeTensor")
            || file.name.to_ascii_lowercase().ends_with(".safetensors")
    };
    files
        .iter()
        .filter(is_safetensors)
        .find(|file| {
            file.file_type
                .as_deref()
                .is_some_and(|kind| kind.eq_ignore_ascii_case("Model"))
        })
        .or_else(|| files.iter().find(is_safetensors))
}

fn pick_civitai_text_encoder(files: &[CivitaiFile]) -> Option<&CivitaiFile> {
    files.iter().find(|file| {
        file.download_url.is_some()
            && (file.metadata.format.as_deref() == Some("SafeTensor")
                || file.name.to_ascii_lowercase().ends_with(".safetensors"))
            && file
                .file_type
                .as_deref()
                .is_some_and(|kind| kind.eq_ignore_ascii_case("Text Encoder"))
    })
}

fn civitai_recipe_file(
    version_id: u64,
    file: &CivitaiFile,
    role: Option<RecipeFileRole>,
) -> RecipeFile {
    RecipeFile {
        url: exact_civitai_download_url(version_id, file),
        dest: format!("{{family}}/civitai/{}/{}", version_id, file.name),
        sha256: file
            .hashes
            .get("SHA256")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        size_bytes: file.size_kb.map(|kb| (kb * 1000.0) as u64),
        role,
    }
}

fn exact_civitai_download_url(version_id: u64, file: &CivitaiFile) -> String {
    let raw = file
        .download_url
        .clone()
        .unwrap_or_else(|| format!("https://civitai.com/api/download/models/{version_id}"));
    if raw.contains('?') {
        return raw;
    }

    let mut url = match reqwest::Url::parse(&raw) {
        Ok(url) => url,
        Err(_) => return raw,
    };
    {
        let mut q = url.query_pairs_mut();
        if let Some(file_type) = file.file_type.as_deref().filter(|s| !s.is_empty()) {
            q.append_pair("type", file_type);
        }
        if let Some(format) = file.metadata.format.as_deref().filter(|s| !s.is_empty()) {
            q.append_pair("format", format);
        }
        if let Some(size) = file.metadata.size.as_deref().filter(|s| !s.is_empty()) {
            q.append_pair("size", size);
        }
        if let Some(fp) = file.metadata.fp.as_deref().filter(|s| !s.is_empty()) {
            q.append_pair("fp", fp);
        }
    }
    url.to_string()
}
