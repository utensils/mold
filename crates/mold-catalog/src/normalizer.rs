//! Source-specific JSON → `CatalogEntry`.
//!
//! HF: combine `/api/models/{repo}` detail + `/api/models/{repo}/tree/main`.
//! Civitai: combine the model + first version + a chosen safetensors file.

use serde::Deserialize;

use crate::civitai_map::{engine_phase_for, map_base_model};
use crate::companions::companions_for;
use crate::entry::{
    Bundling, CatalogEntry, CatalogId, DownloadRecipe, FamilyRole, FileFormat, Kind, LicenseFlags,
    Modality, RecipeFile, Source, TokenKind,
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

    let companions = match bundling {
        Bundling::SingleFile => companions_for(family, bundling),
        Bundling::Separated => Vec::new(),
    };
    let phase = engine_phase_for(family, bundling);

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
        kind: Kind::Checkpoint,
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
}

#[derive(Clone, Debug, Deserialize)]
pub struct CivitaiFile {
    pub id: u64,
    pub name: String,
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
}

#[derive(Clone, Debug, Deserialize)]
pub struct CivitaiImage {
    pub url: String,
    #[serde(default, rename = "nsfwLevel")]
    pub nsfw_level: Option<u32>,
}

pub fn from_civitai(item: CivitaiItem) -> Option<CatalogEntry> {
    let version = item.model_versions.first()?;
    let (family, family_role, sub_family) = map_base_model(&version.base_model)?;
    let file = pick_safetensors(&version.files)?;
    let bundling = if version.base_model_type.as_deref() == Some("Standard") {
        Bundling::SingleFile
    } else {
        Bundling::Separated
    };
    let companions = companions_for(family, bundling);
    let phase = engine_phase_for(family, bundling);
    let modality = match family {
        Family::LtxVideo | Family::Ltx2 => Modality::Video,
        _ => Modality::Image,
    };

    let sha256 = file
        .hashes
        .get("SHA256")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let recipe = DownloadRecipe {
        files: vec![RecipeFile {
            url: file.download_url.clone().unwrap_or_else(|| {
                format!("https://civitai.com/api/download/models/{}", version.id)
            }),
            dest: format!("{{family}}/civitai/{}/{}", version.id, file.name),
            sha256,
            size_bytes: file.size_kb.map(|kb| (kb * 1000.0) as u64),
        }],
        needs_token: Some(TokenKind::Civitai),
    };

    let stats = item.stats.unwrap_or_default();
    let now = chrono_now_unix();

    Some(CatalogEntry {
        id: CatalogId::from(format!("cv:{}", version.id)),
        source: Source::Civitai,
        source_id: version.id.to_string(),
        name: item.name.clone(),
        author: item.creator.and_then(|c| c.username),
        family,
        family_role,
        sub_family,
        modality,
        kind: Kind::Checkpoint,
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
        tags: item.tags,
        companions,
        download_recipe: recipe,
        engine_phase: phase,
        created_at: None,
        updated_at: None,
        added_at: now,
    })
}

/// Civitai's legacy unsafe `.pt` ("PickleTensor") format is dropped at the
/// scanner. Arbitrary-code-execution risk on deserialization is not worth
/// catalog completeness — only safetensors are surfaced.
fn pick_safetensors(files: &[CivitaiFile]) -> Option<&CivitaiFile> {
    files
        .iter()
        .find(|f| f.metadata.format.as_deref() == Some("SafeTensor"))
}
