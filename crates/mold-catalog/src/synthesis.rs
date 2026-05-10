//! Pure synthesis of a catalog entry into a `CatalogModelIntent`.
//!
//! Why "intent" and not the full `ModelConfig`? Catalog-bridge synthesis
//! used to make disk-dependent decisions inside a single function — e.g.
//! the FLUX bundled-VAE probe at the primary file. When that probe ran
//! before the file was on disk, it silently chose the wrong branch and
//! sealed a `cfg.vae = primary` into `state.config.models` that the
//! engine later choked on (`cannot find tensor encoder.conv_in.weight`
//! for transformer-only fine-tunes).
//!
//! Splitting synthesis into a pure stage (this module) and a disk-aware
//! resolution stage (server-side `resolve_intent_to_paths`) makes the
//! race impossible by construction: the intent records *what* a checkpoint
//! needs, never *what's currently on disk*.
//!
//! Both `mold-cli` and `mold-server` consume this single function so
//! their synthesis logic can no longer drift.
//!
//! Disk-dependent decisions (FLUX VAE bundle probe, companion path
//! population) live in the consumer crates' resolution layers.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::companions::companions_for;
use crate::entry::{render_recipe_dest, Bundling, CatalogEntry};
use crate::families::Family;

/// Pure description of what an installed catalog model needs.
///
/// Built by [`synthesize_intent`] from a `CatalogEntry` *without reading
/// disk*. The downstream resolution stage (per-crate, since `ModelConfig`
/// lives in `mold-core`) maps this into runtime paths after verifying
/// the files are actually present.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogModelIntent {
    /// Stable family slug (e.g. `"flux"`, `"sdxl"`). Matches
    /// `ModelManifest.family` for cross-table dispatch.
    pub family: String,
    /// Optional sub-family discriminator (e.g. `"klein-4b"` vs
    /// `"klein-9b"` for Flux.2). Required for Flux.2 to pick the right
    /// text-encoder companion.
    pub sub_family: Option<String>,
    /// Expected on-disk path of the primary checkpoint. Existence is
    /// **not asserted** — resolution stage checks that.
    pub primary_recipe_path: PathBuf,
    /// Companions this model needs, with required/optional flag.
    pub companions: Vec<CompanionIntent>,
    pub bundling: Bundling,
}

/// One row in [`CatalogModelIntent::companions`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompanionIntent {
    /// Canonical companion name (`"flux-vae"`, `"t5-v1_1-xxl"`, ...).
    /// Maps to entries in the manifest companion registry.
    pub name: String,
    /// Whether engine load fails when this companion is unresolvable.
    /// All companions emitted by `companions_for` are required for v1;
    /// future per-entry overrides go here.
    pub required: bool,
}

/// Errors that can arise during pure synthesis. Disk-dependent failure
/// modes (companion not on disk yet, bundled-VAE probe disagrees with
/// current `cfg.vae`) live in the resolution layer, not here.
#[derive(Debug, thiserror::Error)]
pub enum SynthesisError {
    /// The catalog entry's `download_recipe.files` is empty. Real
    /// upstream entries always carry at least one file; this typically
    /// means the upstream API returned a malformed payload (4xx body
    /// that decoded successfully but had no concrete files).
    #[error("catalog entry {id} has empty download_recipe")]
    EmptyRecipe { id: String },

    /// Synthesized primary file path isn't valid UTF-8. Should never
    /// fire — `models_dir` and the recipe template are both guaranteed
    /// UTF-8 by the caller — but we surface it explicitly so a future
    /// non-UTF-8 path doesn't get hidden behind a `.unwrap()`.
    #[error("synthesized path is not valid UTF-8: {path}")]
    NonUtf8Path { path: String },

    /// `Bundling::Separated` isn't supported by the catalog bridge.
    /// Separated diffusers layouts are served by the manifest path
    /// (`[models]` in `config.toml`), not by `cv:*`/`hf:*` synthesis.
    #[error("catalog entry {id} bundling={bundling:?} is not supported (single-file only)")]
    UnsupportedBundling { id: String, bundling: Bundling },
}

/// Synthesize a `CatalogModelIntent` from a `CatalogEntry`.
///
/// Pure: takes only the entry plus the install-root `models_dir` so the
/// primary recipe path can be rendered. Reads no disk state.
pub fn synthesize_intent(
    entry: &CatalogEntry,
    models_dir: &Path,
) -> Result<CatalogModelIntent, SynthesisError> {
    let primary =
        entry
            .download_recipe
            .files
            .first()
            .ok_or_else(|| SynthesisError::EmptyRecipe {
                id: entry.id.0.clone(),
            })?;

    let sanitized = mold_core::download::sanitize_recipe_id(entry.id.as_str());
    let (author, name) = match entry.source_id.split_once('/') {
        Some((a, n)) => (a, n),
        None => ("", entry.source_id.as_str()),
    };
    let rendered_dest = render_recipe_dest(&primary.dest, entry.family.as_str(), author, name);
    let primary_path = models_dir.join(&sanitized).join(&rendered_dest);

    if primary_path.to_str().is_none() {
        return Err(SynthesisError::NonUtf8Path {
            path: primary_path.display().to_string(),
        });
    }

    if !matches!(entry.bundling, Bundling::SingleFile) {
        return Err(SynthesisError::UnsupportedBundling {
            id: entry.id.0.clone(),
            bundling: entry.bundling,
        });
    }

    let companions = companions_for(
        entry.family,
        entry.sub_family.as_deref(),
        Bundling::SingleFile,
        entry.kind,
    )
    .into_iter()
    .map(|name| CompanionIntent {
        name,
        // Every companion emitted by `companions_for` is currently a
        // hard requirement: dropping any one of them breaks engine load.
        // If a future Civitai variant has a soft companion (e.g. a
        // spatial upscaler), add per-entry overrides here.
        required: true,
    })
    .collect();

    Ok(CatalogModelIntent {
        family: entry.family.as_str().to_string(),
        sub_family: entry.sub_family.clone(),
        primary_recipe_path: primary_path,
        companions,
        bundling: entry.bundling,
    })
}

/// Convenience accessor — true if `name` is in the intent's companion
/// list.
pub fn intent_has_companion(intent: &CatalogModelIntent, name: &str) -> bool {
    intent.companions.iter().any(|c| c.name == name)
}

/// Returns true for catalog entries that conventionally bundle their
/// VAE in the primary single-file safetensors. SDXL/SD1.5 are
/// guaranteed bundled; FLUX is mixed (some bundle, some are
/// transformer-only) so this returns false to force the disk-aware
/// resolver to peek the safetensors header. Flux.2 / LTX-Video / LTX-2
/// always need a separate VAE companion.
///
/// Used by the resolution layer to decide whether to even attempt the
/// bundled-VAE probe vs. unconditionally route to the VAE companion.
pub fn family_bundles_vae_unconditionally(family: Family) -> bool {
    match family {
        Family::Sd15 | Family::Sdxl => true,
        // Mixed bundling — let the resolver probe the actual safetensors.
        Family::Flux => false,
        // Always-separate VAE.
        Family::Flux2 | Family::LtxVideo | Family::Ltx2 => false,
        // ZImage is text-encoder-only companion; primary bundles VAE.
        Family::ZImage => true,
        // Single-file unsupported — engine_phase: 99 entries. Doesn't
        // matter what we return; resolution will reject before VAE.
        Family::QwenImage | Family::Wuerstchen => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entry::{
        CatalogId, DownloadRecipe, FamilyRole, FileFormat, Kind, LicenseFlags, Modality,
        RecipeFile, Source, TokenKind,
    };

    fn juggernaut_entry() -> CatalogEntry {
        CatalogEntry {
            id: CatalogId::from("cv:1759168"),
            source: Source::Civitai,
            source_id: "1759168".into(),
            name: "Juggernaut XL Ragnarok".into(),
            author: Some("RunDiffusion".into()),
            family: Family::Sdxl,
            family_role: FamilyRole::Finetune,
            sub_family: None,
            modality: Modality::Image,
            kind: Kind::Checkpoint,
            file_format: FileFormat::Safetensors,
            bundling: Bundling::SingleFile,
            size_bytes: Some(6_938_040_788),
            download_count: 12_345,
            rating: None,
            likes: 0,
            nsfw: false,
            thumbnail_url: None,
            description: None,
            license: None,
            license_flags: LicenseFlags::default(),
            tags: vec![],
            companions: vec!["clip-l".into(), "clip-g".into(), "sdxl-vae".into()],
            download_recipe: DownloadRecipe {
                files: vec![RecipeFile {
                    url: "https://civitai.com/api/download/models/1759168".into(),
                    dest: "{family}/civitai/1759168/juggernautXL_ragnarokBy.safetensors".into(),
                    sha256: Some(
                        "DEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEF".into(),
                    ),
                    size_bytes: Some(6_938_040_788),
                }],
                needs_token: Some(TokenKind::Civitai),
            },
            engine_phase: 1,
            created_at: None,
            updated_at: None,
            added_at: 0,
            trained_words: vec![],
        }
    }

    fn flux_unet_only_entry() -> CatalogEntry {
        CatalogEntry {
            id: CatalogId::from("cv:994561"),
            source: Source::Civitai,
            source_id: "994561".into(),
            name: "Real Horny Pro V3".into(),
            author: Some("someone".into()),
            family: Family::Flux,
            family_role: FamilyRole::Finetune,
            sub_family: None,
            modality: Modality::Image,
            kind: Kind::Checkpoint,
            file_format: FileFormat::Safetensors,
            bundling: Bundling::SingleFile,
            size_bytes: Some(12_000_000_000),
            download_count: 0,
            rating: None,
            likes: 0,
            nsfw: false,
            thumbnail_url: None,
            description: None,
            license: None,
            license_flags: LicenseFlags::default(),
            tags: vec![],
            companions: vec!["t5-v1_1-xxl".into(), "clip-l".into(), "flux-vae".into()],
            download_recipe: DownloadRecipe {
                files: vec![RecipeFile {
                    url: "https://civitai.com/api/download/models/994561".into(),
                    dest: "{family}/civitai/994561/realHornyProV3Unet.safetensors".into(),
                    sha256: None,
                    size_bytes: Some(12_000_000_000),
                }],
                needs_token: Some(TokenKind::Civitai),
            },
            engine_phase: 1,
            created_at: None,
            updated_at: None,
            added_at: 0,
            trained_words: vec![],
        }
    }

    #[test]
    fn synthesize_intent_for_sdxl_single_file_civitai_entry() {
        let models_dir = Path::new("/tmp/mold-test-models");
        let entry = juggernaut_entry();
        let intent = synthesize_intent(&entry, models_dir).expect("synthesis");

        assert_eq!(intent.family, "sdxl");
        assert_eq!(intent.sub_family, None);
        assert_eq!(intent.bundling, Bundling::SingleFile);

        let expected_path = models_dir
            .join("cv-1759168")
            .join("sdxl/civitai/1759168/juggernautXL_ragnarokBy.safetensors");
        assert_eq!(intent.primary_recipe_path, expected_path);

        let names: Vec<&str> = intent.companions.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["clip-l", "clip-g", "sdxl-vae"]);
        assert!(intent.companions.iter().all(|c| c.required));
    }

    /// Synthesis must produce identical intents whether or not the
    /// primary file is on disk. This is the core invariant the lazy
    /// resolution flow leans on: synthesis is pure.
    #[test]
    fn synthesize_intent_is_disk_independent() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path();
        let entry = flux_unet_only_entry();

        // Absent on disk.
        let intent_absent = synthesize_intent(&entry, models_dir).expect("absent synthesis");

        // Now place the file.
        let file_path = models_dir
            .join("cv-994561")
            .join("flux/civitai/994561/realHornyProV3Unet.safetensors");
        std::fs::create_dir_all(file_path.parent().unwrap()).unwrap();
        std::fs::write(&file_path, b"placeholder").unwrap();

        let intent_present = synthesize_intent(&entry, models_dir).expect("present synthesis");

        assert_eq!(
            intent_absent, intent_present,
            "synthesis must not depend on disk state"
        );
    }

    #[test]
    fn synthesize_intent_emits_required_companions_for_flux() {
        let entry = flux_unet_only_entry();
        let intent = synthesize_intent(&entry, Path::new("/tmp")).unwrap();
        let names: Vec<&str> = intent.companions.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["t5-v1_1-xxl", "clip-l", "flux-vae"]);
    }

    #[test]
    fn synthesize_intent_rejects_separated_bundling() {
        let mut entry = juggernaut_entry();
        entry.bundling = Bundling::Separated;
        let err = synthesize_intent(&entry, Path::new("/tmp")).unwrap_err();
        assert!(matches!(err, SynthesisError::UnsupportedBundling { .. }));
    }

    #[test]
    fn synthesize_intent_rejects_empty_recipe() {
        let mut entry = juggernaut_entry();
        entry.download_recipe.files.clear();
        let err = synthesize_intent(&entry, Path::new("/tmp")).unwrap_err();
        assert!(matches!(err, SynthesisError::EmptyRecipe { .. }));
    }
}
