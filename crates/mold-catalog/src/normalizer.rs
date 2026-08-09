//! Source-specific JSON → `CatalogEntry`.
//!
//! HF: combine `/api/models/{repo}` detail + `/api/models/{repo}/tree/main`.
//! Civitai: combine the model + a chosen version + a chosen safetensors file.

use serde::Deserialize;

use crate::civitai_map::{map_base_model, supported_for};
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

/// Canonical Hugging Face host — shared base for `resolve/main` file URLs
/// and human-facing `page_url` composition (also used by `live.rs`).
pub(crate) const HF_RAW: &str = "https://huggingface.co";

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

/// LoRA repositories frequently publish mutually-exclusive precision, step,
/// and fused-checkpoint variants. A pull must select one runnable LoRA payload
/// instead of downloading every safetensors file in the repository.
fn select_hf_lora_payload(files: &[RecipeFile]) -> Option<RecipeFile> {
    files
        .iter()
        .filter(|file| file.url.to_ascii_lowercase().ends_with(".safetensors"))
        // Fused FP8 checkpoints can live beside the actual LoRA weights. They
        // are full models, not an alternative LoRA payload.
        .filter(|file| !file.url.to_ascii_lowercase().contains("fp8_e4m3fn_scaled"))
        .max_by_key(|file| {
            let path = file.url.split("/resolve/main/").nth(1).unwrap_or(&file.url);
            let lower = path.to_ascii_lowercase();
            (
                lower == "pytorch_lora_weights.safetensors",
                !path.contains('/'),
                lower.contains("4step"),
                lower.contains("bf16"),
                lower,
            )
        })
        .cloned()
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
    let modality = if family.is_video() {
        Modality::Video
    } else {
        Modality::Image
    };

    let kind = classify_hf_kind(&detail, &files);
    if kind == Kind::Lora {
        files = select_hf_lora_payload(&files)
            .map(|file| vec![file])
            .ok_or(NormalizeError::EmptyTree)?;
    }
    let total_size = files.iter().filter_map(|f| f.size_bytes).sum::<u64>();
    // HF entries generally carry no sub_family — single-file HF Flux.2 rows
    // are rare and Civitai is the load-bearing path there. Wan is the
    // exception: HF is where it lives, and its 5B variant needs a different
    // VAE from the rest of the family, so that one is inferred from the id.
    let sub_family = crate::live::wan_sub_family_from_id(&detail.id);
    let companions = match bundling {
        Bundling::SingleFile => companions_for(family, sub_family.as_deref(), bundling, kind),
        Bundling::Separated => Vec::new(),
    };
    let supported = supported_for(family, bundling, kind);

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
        sub_family,
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
        supported,
        created_at: parse_iso(&detail.created_at),
        updated_at: parse_iso(&detail.last_modified),
        added_at: now,
        // HF doesn't surface a trigger-words equivalent — the metadata
        // varies per repo (cardData, README, etc.) and isn't worth a
        // brittle scrape. Empty here is the right default; civitai is
        // where 99% of LoRA trigger phrases live anyway.
        trained_words: Vec::new(),
        page_url: Some(format!("{HF_RAW}/{}", detail.id)),
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
    /// Civitai returns HTML here. Normalize it to bounded plain text before
    /// it reaches any client so cards do not expose markup or ingest an
    /// unbounded model-card essay.
    #[serde(default)]
    pub description: Option<String>,
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
    /// Version detail responses carry their own HTML description (usually a
    /// changelog) even when the embedded parent model omits its description.
    #[serde(default)]
    pub description: Option<String>,
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

const CATALOG_THUMBNAIL_WIDTH: usize = 512;

/// Replace Civitai's source-resolution transform with a stable card-sized
/// derivative. The CDN serves this URL with public cache headers, so WebViews
/// reuse the same small response in Grid and Table layouts instead of
/// repeatedly transferring and decoding a multi-megapixel original.
fn civitai_thumbnail_url(raw: &str) -> String {
    let Ok(mut url) = reqwest::Url::parse(raw) else {
        return raw.to_string();
    };
    if !matches!(
        url.host_str(),
        Some("image.civitai.com" | "imagecache.civitai.com")
    ) {
        return raw.to_string();
    }

    let mut segments: Vec<String> = url
        .path_segments()
        .map(|parts| parts.map(str::to_string).collect())
        .unwrap_or_default();
    let Some(transform) = segments
        .iter_mut()
        .find(|segment| segment.starts_with("original=") || segment.starts_with("width="))
    else {
        return raw.to_string();
    };
    *transform = format!("width={CATALOG_THUMBNAIL_WIDTH}");
    url.set_path(&format!("/{}", segments.join("/")));
    url.to_string()
}

pub fn from_civitai(item: CivitaiItem) -> Option<CatalogEntry> {
    let version = item.model_versions.first()?;
    from_civitai_version(&item, version, A14bEmitPolicy::EmitRequested)
}

pub fn from_civitai_search_entries(item: CivitaiItem) -> Vec<CatalogEntry> {
    item.model_versions
        .iter()
        .filter(|version| version_is_public(version))
        .filter_map(|version| from_civitai_version(&item, version, A14bEmitPolicy::SkipLowNoise))
        .collect()
}

pub(crate) fn version_is_public(version: &CivitaiVersion) -> bool {
    matches!(
        version.availability.as_deref(),
        None | Some("Public") | Some("public")
    )
}

/// How to surface a successfully-paired A14B version whose entry point is
/// the *low-noise* expert. Search listings skip it (its high-noise sibling
/// already emits the pair, so one pair is one row); direct `cv:<id>`
/// lookups emit the pair under the requested id so either expert's id
/// resolves to the same runnable install.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum A14bEmitPolicy {
    SkipLowNoise,
    EmitRequested,
}

/// Normalize one version with an explicit A14B emit policy. Only the
/// single-id fetch path (`live::fetch_civitai_version`) needs
/// `EmitRequested` directly.
pub(crate) fn from_civitai_version_with_policy(
    item: &CivitaiItem,
    version: &CivitaiVersion,
    policy: A14bEmitPolicy,
) -> Option<CatalogEntry> {
    from_civitai_version(item, version, policy)
}

fn from_civitai_version(
    item: &CivitaiItem,
    version: &CivitaiVersion,
    a14b_policy: A14bEmitPolicy,
) -> Option<CatalogEntry> {
    let (family, family_role, sub_family) = map_base_model(&version.base_model)?;
    let file = pick_safetensors(&version.files)?;
    // Drop entries whose Civitai type isn't representable in the catalog
    // (TextualInversion, Hypernetwork, etc.) before doing any further work.
    let kind = civitai_kind_to_catalog_kind(&item.kind)?;
    // 4-bit (NF4/NVFP4) safetensors checkpoints have no Wan load path
    // (dense, scaled-FP8, and GGUF only) — drop them like the GGUF
    // publications rather than offering a download that fails at load.
    if kind == Kind::Checkpoint && family == Family::Wan && !wan_precision_is_loadable(file) {
        tracing::debug!(
            target: "catalog.wan_a14b",
            version_id = version.id,
            fp = file.metadata.fp.as_deref().unwrap_or_default(),
            "Wan version dropped: 4-bit safetensors quantization has no Wan load path",
        );
        return None;
    }
    let bundling = if version.base_model_type.as_deref() == Some("Standard") {
        Bundling::SingleFile
    } else {
        Bundling::Separated
    };
    let companions = companions_for(family, sub_family.as_deref(), bundling, kind);
    let mut supported = supported_for(family, bundling, kind);
    let modality = if family.is_video() {
        Modality::Video
    } else {
        Modality::Image
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
    // A14B checkpoints denoise with a high/low expert pair that Civitai
    // publishes as sibling versions. Pair them into one two-file recipe;
    // when the counterpart cannot be identified with confidence the row
    // stays visible but un-installable (fail closed, never single-expert).
    let mut size_bytes = file.size_kb.map(|kb| (kb * 1000.0) as u64);
    if kind == Kind::Checkpoint
        && family == Family::Wan
        && crate::wan_a14b::a14b_sub_family(&version.base_model).is_some()
    {
        match crate::wan_a14b::pair_experts(item, version, file) {
            Ok(pair) => {
                if a14b_policy == A14bEmitPolicy::SkipLowNoise
                    && pair.requested_role == crate::wan_a14b::ExpertRole::LowNoise
                {
                    return None;
                }
                recipe_files = vec![
                    civitai_recipe_file(pair.high.version_id, pair.high.file, None),
                    civitai_recipe_file(
                        pair.low.version_id,
                        pair.low.file,
                        Some(RecipeFileRole::LowNoiseTransformer),
                    ),
                ];
                size_bytes = pair
                    .high
                    .file
                    .size_kb
                    .zip(pair.low.file.size_kb)
                    .map(|(high, low)| ((high + low) * 1000.0) as u64)
                    .or(size_bytes);
            }
            Err(reason) => {
                tracing::debug!(
                    target: "catalog.wan_a14b",
                    version_id = version.id,
                    %reason,
                    "A14B version left un-installable: no confident expert pairing",
                );
                supported = false;
            }
        }
    }

    let recipe = DownloadRecipe {
        files: recipe_files,
        needs_token: Some(TokenKind::Civitai),
    };

    let stats = item.stats.clone().unwrap_or_default();
    let now = chrono_now_unix();

    let trained_words = version.trained_words.clone();

    // The single-version detail path synthesizes a CivitaiItem with id 0
    // when the upstream body lacks `modelId` — no parent id means no
    // model page to point at, so leave the link off rather than compose
    // a URL to civitai.com/models/0.
    let page_url = (item.id != 0).then(|| {
        format!(
            "https://civitai.com/models/{}?modelVersionId={}",
            item.id, version.id
        )
    });

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
        size_bytes,
        download_count: stats.download_count,
        rating: stats.rating,
        likes: stats.favorite_count,
        nsfw: item.nsfw,
        thumbnail_url: version
            .images
            .first()
            .map(|image| civitai_thumbnail_url(&image.url)),
        description: item
            .description
            .as_deref()
            .and_then(plain_text_description)
            .or_else(|| {
                version
                    .description
                    .as_deref()
                    .and_then(plain_text_description)
            }),
        license: None,
        license_flags: LicenseFlags::default(),
        tags: item.tags.clone(),
        companions,
        download_recipe: recipe,
        supported,
        created_at: None,
        updated_at: None,
        added_at: now,
        trained_words,
        page_url,
    })
}

const MAX_DESCRIPTION_CHARS: usize = 1_200;

/// Convert Civitai's model-description HTML into compact display text without
/// adding an HTML parser to the inference binary's dependency tree. Vue still
/// escapes the returned string; this normalization is about readable content
/// and bounded catalog payloads, not trusting the upstream markup.
fn plain_text_description(raw: &str) -> Option<String> {
    let mut without_tags = String::with_capacity(raw.len().min(MAX_DESCRIPTION_CHARS * 2));
    let mut in_tag = false;
    for ch in raw.chars() {
        match ch {
            '<' => {
                in_tag = true;
                without_tags.push(' ');
            }
            '>' if in_tag => {
                in_tag = false;
                without_tags.push(' ');
            }
            _ if !in_tag => without_tags.push(ch),
            _ => {}
        }
    }

    let decoded = decode_html_entities(&without_tags);
    let compact = decoded.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.is_empty() {
        return None;
    }
    if compact.chars().count() <= MAX_DESCRIPTION_CHARS {
        return Some(compact);
    }
    let mut bounded = compact
        .chars()
        .take(MAX_DESCRIPTION_CHARS)
        .collect::<String>();
    bounded.push('…');
    Some(bounded)
}

fn decode_html_entities(raw: &str) -> String {
    let chars = raw.chars().collect::<Vec<_>>();
    let mut out = String::with_capacity(raw.len());
    let mut index = 0;
    while index < chars.len() {
        if chars[index] != '&' {
            out.push(chars[index]);
            index += 1;
            continue;
        }
        let end = ((index + 1)..chars.len().min(index + 14)).find(|&i| chars[i] == ';');
        let Some(end) = end else {
            out.push('&');
            index += 1;
            continue;
        };
        let entity = chars[index + 1..end].iter().collect::<String>();
        let decoded = match entity.as_str() {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" | "#39" => Some('\''),
            "nbsp" => Some(' '),
            value if value.starts_with("#x") || value.starts_with("#X") => {
                u32::from_str_radix(&value[2..], 16)
                    .ok()
                    .and_then(char::from_u32)
            }
            value if value.starts_with('#') => {
                value[1..].parse::<u32>().ok().and_then(char::from_u32)
            }
            _ => None,
        };
        if let Some(ch) = decoded {
            out.push(ch);
            index = end + 1;
        } else {
            out.extend(chars[index..=end].iter());
            index = end + 1;
        }
    }
    out
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
pub(crate) fn pick_safetensors(files: &[CivitaiFile]) -> Option<&CivitaiFile> {
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

/// Whether the Wan loader can read a Civitai safetensors file of this
/// reported precision. The engine loads dense (fp16/bf16/fp32), scaled-FP8,
/// and GGUF weights only — 4-bit safetensors quantizations (Civitai `fp`
/// metadata `nf4` / `fp4` / `nvfp4`, real for A14B I2V expert pairs) have
/// no load path, so surfacing them would offer a multi-gigabyte download
/// that fails at load. Mirrors the GGUF handling: the version is dropped,
/// with the reason logged at the drop site.
pub(crate) fn wan_precision_is_loadable(file: &CivitaiFile) -> bool {
    !matches!(
        file.metadata
            .fp
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("nf4" | "fp4" | "nvfp4")
    )
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
