//! Pinned LTX-2.5 split-pack asset contract.
//!
//! LTX-2.5 no longer ships one checkpoint containing every runtime component.
//! Keep the official filenames, sizes, hashes, and component roles together so
//! download planning, ownership, compatibility probes, and inference cannot
//! silently disagree about what a complete pack means.

use std::path::{Path, PathBuf};

use crate::manifest::{ManifestDefaults, ModelComponent, ModelFile, ModelManifest};
use crate::types::SourceImageCapability;

const REPO: &str = "Lightricks/LTX-2.5";
pub const FAMILY: &str = "ltx2.5-contract";
pub const DEV: &str = "ltx-2.5-22b-dev:bf16";
pub const DEV_CONV: &str = "ltx-2.5-22b-dev:bf16-conv";
pub const DISTILLED: &str = "ltx-2.5-22b-distilled:bf16";
pub const DISTILLED_CONV: &str = "ltx-2.5-22b-distilled:bf16-conv";
const ALL: &[&str] = &[DEV, DEV_CONV, DISTILLED, DISTILLED_CONV];
const DEV_VARIANTS: &[&str] = &[DEV, DEV_CONV];
const DISTILLED_VARIANTS: &[&str] = &[DISTILLED, DISTILLED_CONV];
const DIFFUSION_VAE_VARIANTS: &[&str] = &[DEV, DISTILLED];
const CONV_VAE_VARIANTS: &[&str] = &[DEV_CONV, DISTILLED_CONV];
const REFERENCE_ONLY: &[&str] = &[];

#[derive(Debug, Clone, Copy)]
struct Asset {
    filename: &'static str,
    component: ModelComponent,
    size_bytes: u64,
    sha256: &'static str,
    /// Manifests that currently consume this asset. An empty slice records an
    /// official asset whose storage contract is known but whose native loader
    /// is deliberately still fail-closed.
    manifests: &'static [&'static str],
}

const ASSETS: &[Asset] = &[
    Asset {
        filename: "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
        component: ModelComponent::Transformer,
        size_bytes: 42_018_190_584,
        sha256: "792a2bad501ca03262c0bc2ce7a2949e85b142ce18e30894aad5bc849c8e7584",
        manifests: DEV_VARIANTS,
    },
    Asset {
        filename: "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        component: ModelComponent::Transformer,
        size_bytes: 42_018_190_584,
        sha256: "31eb3cad89b9e54e99dd3baf286f70825ac4f6c660a70d9184d895be76d7bff4",
        manifests: DISTILLED_VARIANTS,
    },
    Asset {
        filename: "diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors",
        component: ModelComponent::Transformer,
        size_bytes: 21_504_034_224,
        sha256: "2edbdb4465cd6c3b532cd67a31ddb38a63e97dcad20be3729675e2a4e8caf92b",
        manifests: REFERENCE_ONLY,
    },
    Asset {
        filename:
            "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        component: ModelComponent::Transformer,
        size_bytes: 21_504_034_224,
        sha256: "c4279eeff115cbeaca494bd2183e7d768c38fe85a184dc6afbb7159157c44334",
        manifests: REFERENCE_ONLY,
    },
    Asset {
        filename: "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
        component: ModelComponent::Transformer,
        size_bytes: 18_721_548_408,
        sha256: "4b94231e734c1950f8f6826cb8bd8715d94be5b3e04f8256ee060c5bc3886c30",
        manifests: REFERENCE_ONLY,
    },
    Asset {
        filename: "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        component: ModelComponent::TextEncoder,
        size_bytes: 26_263_858_182,
        sha256: "ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1",
        manifests: ALL,
    },
    Asset {
        filename: "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        component: ModelComponent::TextEncoder,
        size_bytes: 15_372_969_374,
        sha256: "6ce688a0aa98a5fa36a9f1e6c3f42152a498cc2b53ee8c15674c64244f91487f",
        manifests: REFERENCE_ONLY,
    },
    Asset {
        filename: "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        component: ModelComponent::Vae,
        size_bytes: 1_452_269_922,
        sha256: "685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d",
        manifests: CONV_VAE_VARIANTS,
    },
    Asset {
        filename: "vae/ltx-2.5-video-vae-bf16.safetensors",
        component: ModelComponent::Vae,
        size_bytes: 1_472_223_346,
        sha256: "847e14ca7f3355debca0cea4eaa24ac0fbcdf0061da054ac89ca638a869ddba3",
        manifests: DIFFUSION_VAE_VARIANTS,
    },
    Asset {
        filename: "vae/ltx-2.5-audio-vae-bf16.safetensors",
        component: ModelComponent::AudioVae,
        size_bytes: 364_866_540,
        sha256: "c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5",
        manifests: ALL,
    },
    Asset {
        filename: "model_patches/ltx-2.5-duration-head-bf16.safetensors",
        component: ModelComponent::DurationHead,
        size_bytes: 3_843_690,
        sha256: "2ec71e4206ed365d015f00c05a48caccfb0ee862986809d06ae376c09f5d9190",
        manifests: ALL,
    },
    Asset {
        filename: "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        component: ModelComponent::SpatialUpscaler,
        size_bytes: 995_778_752,
        sha256: "eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8",
        manifests: ALL,
    },
    Asset {
        filename: "latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
        component: ModelComponent::TemporalUpscaler,
        size_bytes: 261_944_000,
        sha256: "2bc3300f2b3c3c1834d72164fbf13a3b9fd73e5a741e8a2c3f4035f89a75c3fe",
        manifests: ALL,
    },
    Asset {
        filename: "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
        component: ModelComponent::DistilledLora,
        size_bytes: 8_899_889_568,
        sha256: "86370bbf79a9eb4edaa158907e2b48a5188fe4c5dc8ce30c7eb8f2f131a9bbf5",
        manifests: DEV_VARIANTS,
    },
];

fn files_for(manifest_name: &str) -> Vec<ModelFile> {
    ASSETS
        .iter()
        .filter(|asset| asset.manifests.contains(&manifest_name))
        .map(|asset| ModelFile {
            hf_repo: REPO.to_string(),
            hf_filename: asset.filename.to_string(),
            component: asset.component,
            size_bytes: asset.size_bytes,
            gated: true,
            sha256: Some(asset.sha256),
        })
        .collect()
}

fn defaults(steps: u32, guidance: f64) -> ManifestDefaults {
    ManifestDefaults {
        steps,
        guidance,
        width: 1216,
        height: 704,
        is_schnell: false,
        scheduler: None,
        negative_prompt: None,
        frames: Some(121),
        fps: Some(24),
        source_image: Some(SourceImageCapability::Optional),
    }
}

pub(crate) fn manifests() -> Vec<ModelManifest> {
    [
        (DEV, "dev", "diffusion", 30, 3.0),
        (DEV_CONV, "dev", "convolutional", 30, 3.0),
        (DISTILLED, "distilled", "diffusion", 8, 1.0),
        (DISTILLED_CONV, "distilled", "convolutional", 8, 1.0),
    ]
    .into_iter()
    .map(
        |(name, checkpoint, decoder, steps, guidance)| ModelManifest {
            name: name.to_string(),
            family: FAMILY.to_string(),
            description: if decoder == "convolutional" {
                format!("LTX-2.5 22B {checkpoint} BF16 — native Conv-VAE split pack")
            } else {
                format!(
                    "LTX-2.5 22B {checkpoint} BF16 — downloadable diffusion-VAE Phase 3 contract"
                )
            },
            files: files_for(name),
            defaults: defaults(steps, guidance),
            hidden: true,
        },
    )
    .collect()
}

pub(crate) fn is_contract_manifest(name: &str) -> bool {
    ALL.contains(&name)
}

/// Keep decoder variants on one physical transformer graph.
pub(crate) fn storage_identity(name: &str) -> &str {
    match name {
        DEV_CONV => DEV,
        DISTILLED_CONV => DISTILLED,
        other => other,
    }
}

/// Lossless, runtime-shaped view of one selected LTX-2.5 split pack.
///
/// This is deliberately separate from the legacy monolithic `ModelPaths`:
/// audio VAE and duration head are first-class, and exactly one complete
/// video VAE occupies the shared encoder/decoder slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ltx25ModelPaths {
    pub transformer: PathBuf,
    pub gemma: PathBuf,
    pub video_vae: PathBuf,
    pub audio_vae: PathBuf,
    pub duration_head: PathBuf,
    pub spatial_upscaler: PathBuf,
    pub temporal_upscaler: PathBuf,
    pub distilled_lora: Option<PathBuf>,
}

impl Ltx25ModelPaths {
    pub fn resolve(config: &crate::Config, model_name: &str) -> Option<Self> {
        Self::resolve_in(&config.resolved_models_dir(), model_name)
    }

    pub fn resolve_in(models_root: &Path, model_name: &str) -> Option<Self> {
        let manifest = crate::manifest::find_manifest(model_name)?;
        let resolved: Vec<_> = manifest
            .files
            .iter()
            .map(|file| {
                (
                    file.component,
                    models_root.join(crate::manifest::storage_path(manifest, file)),
                )
            })
            .collect();
        let one = |component| {
            let mut matches = resolved
                .iter()
                .filter(|(role, _)| *role == component)
                .map(|(_, path)| path.clone());
            let path = matches.next()?;
            matches.next().is_none().then_some(path)
        };
        Some(Self {
            transformer: one(ModelComponent::Transformer)?,
            gemma: one(ModelComponent::TextEncoder)?,
            video_vae: one(ModelComponent::Vae)?,
            audio_vae: one(ModelComponent::AudioVae)?,
            duration_head: one(ModelComponent::DurationHead)?,
            spatial_upscaler: one(ModelComponent::SpatialUpscaler)?,
            temporal_upscaler: one(ModelComponent::TemporalUpscaler)?,
            distilled_lora: one(ModelComponent::DistilledLora),
        })
    }

    pub fn all_file_paths(&self) -> impl Iterator<Item = &Path> {
        [
            Some(self.transformer.as_path()),
            Some(self.gemma.as_path()),
            Some(self.video_vae.as_path()),
            Some(self.audio_vae.as_path()),
            Some(self.duration_head.as_path()),
            Some(self.spatial_upscaler.as_path()),
            Some(self.temporal_upscaler.as_path()),
            self.distilled_lora.as_deref(),
        ]
        .into_iter()
        .flatten()
    }

    pub fn qualify(&self) -> std::io::Result<()> {
        crate::ltx25_probe::validate_ltx25_transformer_gemma(&self.transformer, &self.gemma)?;
        crate::ltx25_probe::probe_ltx25_video_vae(&self.video_vae)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn official_asset_contract_has_unique_paths_and_complete_hashes() {
        let mut paths = HashSet::new();
        assert_eq!(ASSETS.len(), 14);
        for asset in ASSETS {
            assert!(paths.insert(asset.filename), "duplicate {}", asset.filename);
            assert!(asset.size_bytes > 0, "{} has no size", asset.filename);
            assert_eq!(
                asset.sha256.len(),
                64,
                "{} has a partial hash",
                asset.filename
            );
            assert!(asset.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()));
        }
    }

    #[test]
    fn runnable_manifests_have_every_split_runtime_role() {
        for manifest in manifests() {
            assert!(
                manifest.hidden,
                "{} must remain download-only",
                manifest.name
            );
            assert_eq!(manifest.family, FAMILY);
            assert!(manifest.is_files_only_bundle());
            assert!(!manifest.is_generation_model());
            let components: Vec<_> = manifest.files.iter().map(|file| file.component).collect();
            for required in [
                ModelComponent::Transformer,
                ModelComponent::TextEncoder,
                ModelComponent::Vae,
                ModelComponent::AudioVae,
                ModelComponent::DurationHead,
                ModelComponent::SpatialUpscaler,
                ModelComponent::TemporalUpscaler,
            ] {
                assert!(
                    components.contains(&required),
                    "{} missing {required:?}",
                    manifest.name
                );
            }
            assert!(manifest.files.iter().all(|file| file.gated));
        }
    }

    #[test]
    fn minimum_distilled_pack_accounts_for_all_selected_bytes() {
        let manifest = manifests()
            .into_iter()
            .find(|manifest| manifest.name == DISTILLED)
            .expect("distilled manifest");
        assert_eq!(
            crate::manifest::total_download_size(&manifest),
            71_380_705_094
        );
        assert_eq!(
            crate::manifest::compute_download_size(&manifest).0,
            71_380_705_094
        );
        assert_eq!(manifest.total_size_bytes(), 71_380_705_094);
        assert!(crate::download::validate_available_download_space(
            &manifest,
            manifest.total_size_bytes(),
            manifest.total_size_bytes() - 1,
            Path::new("/test-volume"),
        )
        .is_err());
        crate::download::validate_available_download_space(
            &manifest,
            manifest.total_size_bytes(),
            manifest.total_size_bytes(),
            Path::new("/test-volume"),
        )
        .unwrap();
        assert_eq!(
            manifest
                .files
                .iter()
                .filter(|file| file.component == ModelComponent::Vae)
                .count(),
            1
        );
        assert!(!manifest
            .files
            .iter()
            .any(|file| file.component == ModelComponent::DistilledLora));
    }

    #[test]
    fn decoder_variants_share_weights_but_select_different_video_vaes() {
        let diff = manifests()
            .into_iter()
            .find(|manifest| manifest.name == DISTILLED)
            .unwrap();
        let conv = manifests()
            .into_iter()
            .find(|manifest| manifest.name == DISTILLED_CONV)
            .unwrap();
        assert_eq!(storage_identity(&diff.name), storage_identity(&conv.name));
        let selected_vae = |manifest: &ModelManifest| {
            manifest
                .files
                .iter()
                .find(|file| file.component == ModelComponent::Vae)
                .unwrap()
                .hf_filename
                .clone()
        };
        assert_ne!(selected_vae(&diff), selected_vae(&conv));
        assert_eq!(conv.total_size_bytes(), 71_360_751_670);
    }

    #[test]
    fn dedicated_split_paths_preserve_every_runtime_role() {
        let config = crate::Config {
            models_dir: "/ltx25-contract-test".to_string(),
            ..Default::default()
        };
        let dev = Ltx25ModelPaths::resolve(&config, DEV).unwrap();
        let distilled = Ltx25ModelPaths::resolve(&config, DISTILLED).unwrap();
        assert_eq!(dev.all_file_paths().count(), 8);
        assert_eq!(distilled.all_file_paths().count(), 7);
        assert!(dev.distilled_lora.is_some());
        assert!(distilled.distilled_lora.is_none());
        assert!(dev
            .audio_vae
            .ends_with("vae/ltx-2.5-audio-vae-bf16.safetensors"));
        assert!(dev
            .duration_head
            .ends_with("model_patches/ltx-2.5-duration-head-bf16.safetensors"));
    }
}
