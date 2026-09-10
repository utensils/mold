//! On-disk resolution for the IP-Adapter auxiliary asset bundles.
//!
//! A bundle is three artifacts — the image-prompt adapter, an OpenCLIP
//! ViT-H/14 vision tower, and that tower's `config.json` — and none of them is
//! a transformer or a VAE. [`crate::manifest::paths_from_downloads`] therefore
//! cannot represent it: it exists to build a [`crate::ModelPaths`], which
//! requires a generator. This module resolves the bundle on its own terms
//! instead, and answers only one question: are all three files present, and
//! where?
//!
//! It is a SIBLING of [`crate::pulid_assets`] rather than a generalization of
//! it, deliberately. That module's [`crate::pulid_assets::PulidPaths`] has
//! five named fields and an [`IdentityFamily`]; widening it to also cover a
//! three-file bundle keyed on a different family would produce a struct where
//! most fields are `Option` — the exact shape `pulid_assets`' own header says
//! it was written to avoid. The two bundles share a *pattern*, not a type.
//!
//! There are two bundles, one per base architecture, and they differ in
//! exactly one file. The adapter is architecture-specific — its per-layer
//! projections are `[hidden_size, cross_attention_dim]`, and the two families'
//! widths are 768 and 2048 — while the vision tower is SHARED, because
//! `ip-adapter_sdxl_vit-h` is named for reusing precisely the ViT-H the SD1.5
//! adapter uses. Both manifests carry the same `ip-adapter` family, so
//! [`crate::manifest::storage_path`] lands the tower and its config at
//! identical `shared/ip-adapter/` paths: a machine holding one bundle pulls
//! only the other's adapter.

use std::path::PathBuf;

use crate::config::Config;
use crate::manifest::{
    find_manifest, ModelComponent, ModelFile, ModelManifest, IP_ADAPTER_SD15_MANIFEST,
    IP_ADAPTER_SDXL_MANIFEST,
};

/// A base architecture qualified for image-prompt conditioning.
///
/// IP-Adapter ships one adapter per UNet width and the two are not
/// interchangeable: `image_proj.proj` projects to `tokens * cross_attention_dim`,
/// so the SDXL file's 8192-wide output is not a whole number of SD1.5's
/// 768-wide tokens and vice versa. Every layer that has to behave differently
/// asks this one question rather than re-deriving it from a model name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ImagePromptFamily {
    /// `ip-adapter_sd15.safetensors`, 16 modules at 768 wide.
    Sd15,
    /// `ip-adapter_sdxl_vit-h.safetensors`, 70 modules at 2048 wide.
    ///
    /// The `vit-h` variant specifically: the plain `ip-adapter_sdxl` conditions
    /// on ViT-bigG and would need a second tower.
    Sdxl,
}

impl ImagePromptFamily {
    /// The manifest name of the asset bundle this family conditions with.
    pub fn manifest(self) -> &'static str {
        match self {
            Self::Sd15 => IP_ADAPTER_SD15_MANIFEST,
            Self::Sdxl => IP_ADAPTER_SDXL_MANIFEST,
        }
    }

    /// The generation family string a request's model resolves to.
    pub fn family(self) -> &'static str {
        match self {
            Self::Sd15 => "sd15",
            Self::Sdxl => "sdxl",
        }
    }

    /// Every qualified family, in a stable order.
    pub const ALL: &'static [ImagePromptFamily] =
        &[ImagePromptFamily::Sd15, ImagePromptFamily::Sdxl];

    /// The family a resolved generation family string belongs to, if any.
    ///
    /// The SINGLE authority. Unlike PuLID's, this needs no per-checkpoint
    /// exception: IP-Adapter conditions through the UNet's cross-attention,
    /// which every SD1.5 and SDXL checkpoint has in the same geometry,
    /// including the distilled ones. `sdxl-turbo` is excluded from PuLID
    /// because PuLID v1.1's own release note reports degraded behaviour on
    /// that base; nothing upstream says the same about image prompting, so
    /// inventing an exclusion here would be a guess wearing a contract's
    /// clothes.
    pub fn from_generation_family(family: &str) -> Option<Self> {
        match family {
            "sd15" | "sd1.5" | "stable-diffusion-1.5" => Some(Self::Sd15),
            "sdxl" => Some(Self::Sdxl),
            _ => None,
        }
    }
}

/// Concrete, verified-complete paths to every IP-Adapter asset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IpAdapterPaths {
    /// Which adapter [`Self::adapter`] is, and therefore which UNet width its
    /// projections were trained at.
    ///
    /// Carried rather than inferred from the filename: a path string is not a
    /// contract, and the loader has to know the width before it opens the file.
    pub family: ImagePromptFamily,
    /// The image-prompt adapter: `image_proj.*` plus `ip_adapter.<i>.*`.
    pub adapter: PathBuf,
    /// The OpenCLIP ViT-H/14 tower, in HF safetensors layout.
    ///
    /// Loaded directly, unlike PuLID's EVA02-CLIP `.pt`, which is a conversion
    /// INPUT. There is no derived artifact here and therefore nothing for
    /// removal to clean up beyond the manifest files themselves.
    pub vision_encoder: PathBuf,
    /// The tower's published `config.json`.
    pub vision_config: PathBuf,
}

/// The manifest for one family's bundle.
///
/// Panics only if the manifest registry lost the entry, which a completeness
/// test in `manifest.rs` makes impossible.
pub fn ip_adapter_manifest_for(family: ImagePromptFamily) -> &'static ModelManifest {
    find_manifest(family.manifest()).expect("every IP-Adapter manifest is registered")
}

/// Every registered IP-Adapter bundle manifest, in [`ImagePromptFamily::ALL`] order.
pub fn ip_adapter_manifests() -> Vec<&'static ModelManifest> {
    ImagePromptFamily::ALL
        .iter()
        .copied()
        .map(ip_adapter_manifest_for)
        .collect()
}

fn file_for(
    manifest: &'static ModelManifest,
    component: ModelComponent,
) -> Option<&'static ModelFile> {
    manifest
        .files
        .iter()
        .find(|file| file.component == component)
}

/// Resolve one family's bundle, returning `Some` only when **all three** files
/// are completely on disk.
///
/// A partially present bundle is deliberately `None` rather than a struct with
/// holes: the adapter cannot condition on a picture without the tower that
/// encodes it, so a caller holding an `IpAdapterPaths` holds a runnable
/// bundle. Use [`missing_ip_adapter_files_for`] to report what a repair still
/// needs.
pub fn ip_adapter_paths_for(config: &Config, family: ImagePromptFamily) -> Option<IpAdapterPaths> {
    let manifest = ip_adapter_manifest_for(family);
    let resolve = |component: ModelComponent| -> Option<PathBuf> {
        config.complete_manifest_file_path(manifest, file_for(manifest, component)?)
    };
    Some(IpAdapterPaths {
        family,
        adapter: resolve(ModelComponent::ImagePromptAdapter)?,
        vision_encoder: resolve(ModelComponent::ImagePromptVisionEncoder)?,
        vision_config: resolve(ModelComponent::ImagePromptVisionConfig)?,
    })
}

/// True when every asset of `family`'s bundle is present and complete.
pub fn ip_adapter_is_installed_for(config: &Config, family: ImagePromptFamily) -> bool {
    ip_adapter_paths_for(config, family).is_some()
}

/// The manifest files a repair pull of `family`'s bundle still has to fetch.
///
/// Empty means installed; a non-empty result on an otherwise-present bundle is
/// exactly the "needs repair" signal `mold pull` acts on.
pub fn missing_ip_adapter_files_for(
    config: &Config,
    family: ImagePromptFamily,
) -> Vec<&'static ModelFile> {
    let manifest = ip_adapter_manifest_for(family);
    manifest
        .files
        .iter()
        .filter(|file| config.complete_manifest_file_path(manifest, file).is_none())
        .collect()
}

/// Where each of `family`'s assets WOULD live, present or not.
///
/// The planning counterpart to [`ip_adapter_paths_for`]: a download plan has
/// to name destinations before anything is on disk.
pub fn ip_adapter_storage_paths_for(config: &Config, family: ImagePromptFamily) -> Vec<PathBuf> {
    let manifest = ip_adapter_manifest_for(family);
    let models_dir = config.resolved_models_dir();
    manifest
        .files
        .iter()
        .map(|file| models_dir.join(crate::manifest::storage_path(manifest, file)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_family_has_a_registered_bundle_with_all_three_components() {
        for family in ImagePromptFamily::ALL.iter().copied() {
            let manifest = ip_adapter_manifest_for(family);
            assert!(manifest.hidden, "{:?} bundle must be hidden", family);
            assert!(
                manifest.is_files_only_bundle(),
                "{:?} bundle must never resolve to a ModelPaths",
                family
            );
            assert!(
                !manifest.is_generation_model(),
                "{:?} bundle must never be a default-model candidate",
                family
            );
            for component in [
                ModelComponent::ImagePromptAdapter,
                ModelComponent::ImagePromptVisionEncoder,
                ModelComponent::ImagePromptVisionConfig,
            ] {
                assert!(
                    file_for(manifest, component).is_some(),
                    "{:?} bundle is missing {component:?}",
                    family
                );
            }
        }
    }

    /// The tower is the shared artifact, and sharing is a PATH fact.
    ///
    /// Both manifests carry the same family, so `storage_path` puts the tower
    /// and its config at identical locations. If that ever stopped being true
    /// a machine would download 2.5 GB twice and neither copy would be wrong,
    /// which is exactly the kind of waste no error surfaces.
    #[test]
    fn the_vision_tower_is_one_file_on_disk_for_both_bundles() {
        let sd15 = ip_adapter_manifest_for(ImagePromptFamily::Sd15);
        let sdxl = ip_adapter_manifest_for(ImagePromptFamily::Sdxl);
        for component in [
            ModelComponent::ImagePromptVisionEncoder,
            ModelComponent::ImagePromptVisionConfig,
        ] {
            let a = file_for(sd15, component).expect("sd15 tower file");
            let b = file_for(sdxl, component).expect("sdxl tower file");
            assert_eq!(a, b, "{component:?} must be the same ModelFile");
            assert_eq!(
                crate::manifest::storage_path(sd15, a),
                crate::manifest::storage_path(sdxl, b),
                "{component:?} must land at one path"
            );
        }

        // The adapters, by contrast, must NOT collide.
        let a = file_for(sd15, ModelComponent::ImagePromptAdapter).unwrap();
        let b = file_for(sdxl, ModelComponent::ImagePromptAdapter).unwrap();
        assert_ne!(
            crate::manifest::storage_path(sd15, a),
            crate::manifest::storage_path(sdxl, b)
        );
    }

    /// Family resolution is the single authority and it takes the same legacy
    /// spellings the engine factory accepts.
    #[test]
    fn generation_families_map_to_their_bundles() {
        assert_eq!(
            ImagePromptFamily::from_generation_family("sd15"),
            Some(ImagePromptFamily::Sd15)
        );
        assert_eq!(
            ImagePromptFamily::from_generation_family("stable-diffusion-1.5"),
            Some(ImagePromptFamily::Sd15)
        );
        assert_eq!(
            ImagePromptFamily::from_generation_family("sdxl"),
            Some(ImagePromptFamily::Sdxl)
        );
        for other in ["flux", "flux2", "sd3", "qwen-image", "z-image", "wan"] {
            assert_eq!(ImagePromptFamily::from_generation_family(other), None);
        }
    }

    /// Nothing is installed on a bare config, and everything is reported
    /// missing rather than some of it.
    #[test]
    fn a_bare_config_reports_the_whole_bundle_missing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = Config {
            models_dir: dir.path().to_string_lossy().into_owned(),
            ..Config::default()
        };
        for family in ImagePromptFamily::ALL.iter().copied() {
            assert!(ip_adapter_paths_for(&config, family).is_none());
            assert!(!ip_adapter_is_installed_for(&config, family));
            assert_eq!(
                missing_ip_adapter_files_for(&config, family).len(),
                ip_adapter_manifest_for(family).files.len()
            );
            assert_eq!(
                ip_adapter_storage_paths_for(&config, family).len(),
                ip_adapter_manifest_for(family).files.len()
            );
        }
    }
}
