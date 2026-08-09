use crate::config::ModelConfig;
use crate::types::Scheduler;
use crate::ModelPaths;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::LazyLock;

/// Model families that are utility models (not image generators).
/// These are excluded from default-model selection and don't produce ModelPaths.
pub const UTILITY_FAMILIES: &[&str] = &["qwen3-expand", "companion"];

/// Model families that are upscaler models (image-to-image enhancement, not generation).
/// These are excluded from default-model selection and use a simplified config path.
pub const UPSCALER_FAMILIES: &[&str] = &["upscaler"];

/// Model families that are auxiliary (not standalone generators).
/// ControlNet models are used via `--control-model`, not as the primary model.
pub const AUXILIARY_FAMILIES: &[&str] = &["controlnet", "ltx2-control", "ltx2-camera-control"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelComponent {
    Transformer,
    TransformerShard, // One shard of a multi-file transformer (Z-Image BF16)
    /// The low-noise half of a two-expert checkpoint (Wan 2.2 A14B).
    ///
    /// Deliberately **not** a [`ModelComponent::TransformerShard`]: shards are
    /// pieces of one network that a loader concatenates into a single
    /// `VarBuilder`, while the two A14B experts are complete, independently
    /// loadable transformers that the sampler switches between at a timestep
    /// boundary. Filing them as shards would mmap 28 GB of weights as one model.
    /// The high-noise expert occupies the ordinary `Transformer` slot because it
    /// runs first.
    LowNoiseTransformer,
    Vae,
    SpatialUpscaler, // LTX latent upsampler / spatial upscaler weights
    TemporalUpscaler,
    DistilledLora,
    /// The distilled adapter belonging to [`ModelComponent::LowNoiseTransformer`].
    /// Each A14B expert is distilled separately, so the pair is not
    /// interchangeable.
    LowNoiseDistilledLora,
    T5Encoder,
    ClipEncoder,
    T5Tokenizer,
    ClipTokenizer,
    ClipEncoder2,   // CLIP-G / OpenCLIP (SDXL)
    ClipTokenizer2, // CLIP-G tokenizer (SDXL)
    TextEncoder,    // Generic text encoder shard (Qwen3 for Z-Image)
    TextTokenizer,  // Generic text encoder tokenizer
    /// MiniMax H3 synchronized-audio VAE. Kept distinct from the video VAE.
    AudioVae,
    /// Tokenizer / multimodal processor data owned by a model family.
    Processor,
    /// Video-side scheduler configuration for a dual-modality model.
    VideoScheduler,
    /// Audio-side scheduler configuration for a dual-modality model.
    AudioScheduler,
    /// Shared architecture/configuration metadata.
    ModelConfig,
    /// Task-transformer configuration. Never shared across tasks; compatible
    /// layouts of the same task may reuse one pinned config identity.
    TaskConfig,
    Decoder,  // Stage B decoder weights (Wuerstchen)
    Upscaler, // Upscaler model weights (Real-ESRGAN, etc.)
}

#[derive(Debug, Clone)]
pub struct ModelFile {
    pub hf_repo: String,
    pub hf_filename: String,
    pub component: ModelComponent,
    pub size_bytes: u64,
    pub gated: bool,
    /// Expected SHA-256 hex digest. None means not yet collected.
    pub sha256: Option<&'static str>,
}

#[derive(Debug, Clone)]
pub struct ManifestDefaults {
    pub steps: u32,
    pub guidance: f64,
    pub width: u32,
    pub height: u32,
    pub is_schnell: bool,
    /// Scheduler algorithm: None for flow-matching models, Some for UNet-based models.
    pub scheduler: Option<Scheduler>,
    /// Default negative prompt for CFG-based models.
    pub negative_prompt: Option<String>,
    /// Default number of video frames. None for image-only models.
    pub frames: Option<u32>,
    /// Default video FPS. None for image-only models.
    pub fps: Option<u32>,
    /// Per-model source-image conditioning contract (#772). Recorded where
    /// the manifest is BUILT — the one place that structurally knows the
    /// task (e.g. which A14B expert repo pair it assembled) — so cold,
    /// not-yet-downloaded tiers advertise correctly without any name
    /// parsing. `None` omits the wire field (image families).
    pub source_image: Option<crate::types::SourceImageCapability>,
}

#[derive(Debug, Clone)]
pub struct ModelManifest {
    pub name: String,
    pub family: String,
    pub description: String,
    pub files: Vec<ModelFile>,
    pub defaults: ManifestDefaults,
    /// Hidden models are excluded from `mold list`, TUI model selector,
    /// and `mold pull` tab completion in release builds. They can still be
    /// used via explicit `mold run <name>` or config.toml entries.
    pub hidden: bool,
}

impl ModelManifest {
    /// Size of the model-specific files in bytes.
    pub fn model_size_bytes(&self) -> u64 {
        self.files
            .iter()
            .filter(|f| is_model_specific_component(f.component))
            .map(|f| f.size_bytes)
            .sum()
    }

    /// Size of the model-specific files in GB (for display).
    pub fn model_size_gb(&self) -> f32 {
        self.model_size_bytes() as f32 / 1_073_741_824.0
    }

    /// True if this is a utility model (e.g., prompt expansion LLM) not an image generator.
    ///
    /// Utility models are downloaded and stored like regular models, but they don't
    /// produce a `ModelPaths` or get written into the config `[models]` section.
    ///
    /// Uses family-based identification (not VAE absence) because auxiliary diffusion
    /// components like ControlNet also lack a VAE but are NOT utility models.
    pub fn is_utility(&self) -> bool {
        UTILITY_FAMILIES.contains(&self.family.as_str())
    }

    /// True if this is an upscaler model (Real-ESRGAN, etc.) not a diffusion generator.
    ///
    /// Upscaler models are downloaded like regular models and get config entries,
    /// but they use a simplified config path (only `transformer` field for weights)
    /// and are not eligible as default generation models.
    pub fn is_upscaler(&self) -> bool {
        UPSCALER_FAMILIES.contains(&self.family.as_str())
    }

    /// True if this is an auxiliary model (e.g., ControlNet) not a standalone generator.
    ///
    /// Auxiliary models are used as modifiers (via `--control-model`) rather than
    /// as the primary generation model.
    pub fn is_auxiliary(&self) -> bool {
        AUXILIARY_FAMILIES.contains(&self.family.as_str())
    }

    /// True if this model can be used as a primary generation model.
    pub fn is_generation_model(&self) -> bool {
        !self.is_upscaler() && !self.is_utility() && !self.is_auxiliary()
    }

    /// True if any file in this model requires HuggingFace authentication.
    pub fn is_gated(&self) -> bool {
        self.files.iter().any(|f| f.gated)
    }

    /// Total size of all files in this model in bytes.
    pub fn total_size_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.size_bytes).sum()
    }

    /// Total size of all files in GB (for display).
    pub fn total_size_gb(&self) -> f32 {
        self.total_size_bytes() as f32 / 1_073_741_824.0
    }

    /// Convert downloaded paths into a `ModelConfig` suitable for saving to config.toml.
    pub fn to_model_config(&self, paths: &ModelPaths) -> ModelConfig {
        ModelConfig {
            transformer: Some(paths.transformer.to_string_lossy().to_string()),
            transformer_shards: if paths.transformer_shards.is_empty() {
                None
            } else {
                Some(
                    paths
                        .transformer_shards
                        .iter()
                        .map(|p| p.to_string_lossy().to_string())
                        .collect(),
                )
            },
            low_noise_transformer: paths
                .low_noise_transformer
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            vae: Some(paths.vae.to_string_lossy().to_string()),
            spatial_upscaler: paths
                .spatial_upscaler
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            temporal_upscaler: paths
                .temporal_upscaler
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            distilled_lora: paths
                .distilled_lora
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            low_noise_distilled_lora: paths
                .low_noise_distilled_lora
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            t5_encoder: paths
                .t5_encoder
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            clip_encoder: paths
                .clip_encoder
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            t5_tokenizer: paths
                .t5_tokenizer
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            clip_tokenizer: paths
                .clip_tokenizer
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            clip_encoder_2: paths
                .clip_encoder_2
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            clip_tokenizer_2: paths
                .clip_tokenizer_2
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            text_encoder_files: if paths.text_encoder_files.is_empty() {
                None
            } else {
                Some(
                    paths
                        .text_encoder_files
                        .iter()
                        .map(|p| p.to_string_lossy().to_string())
                        .collect(),
                )
            },
            text_tokenizer: paths
                .text_tokenizer
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            decoder: paths
                .decoder
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            // Manifest defaults are NOT written to config — resolved at runtime
            // via resolved_model_config() so manifest updates take effect
            // immediately without stale config values. See #129.
            default_steps: None,
            default_guidance: None,
            default_width: None,
            default_height: None,
            is_schnell: None,
            is_turbo: None,
            scheduler: None,
            negative_prompt: None,
            lora: None,
            lora_scale: None,
            default_frames: None,
            default_fps: None,
            description: None,
            family: None,
            placement: None,
        }
    }
}

/// Return a numeric quality rank for a model variant tag.
///
/// Lower numbers mean higher quality. Used to sort model variants within
/// a family so that full-precision appears first and smaller quantizations
/// appear last.
///
/// Ordering: bf16 (0) > fp16 (1) > fp8 (2) > q8 (3) > q6 (4) > q5 (5) > q4 (6) > q3 (7) > q2 (8)
///
/// Unknown tags get rank 100 (sorted last).
pub fn variant_quality_rank(model_name: &str) -> u32 {
    let tag = model_name.rsplit(':').next().unwrap_or("");
    match tag {
        "bf16" => 0,
        "fp16" => 1,
        "fp8" => 2,
        "q8" => 3,
        "q6" => 4,
        "q5" => 5,
        "q4" => 6,
        "q3" => 7,
        "q2" => 8,
        _ => 100,
    }
}

/// Return the base name of a model (everything before the colon tag).
///
/// `"flux-dev:q4"` → `"flux-dev"`, `"sd15:fp16"` → `"sd15"`.
pub fn model_base_name(model_name: &str) -> &str {
    model_name.split(':').next().unwrap_or(model_name)
}

fn is_model_specific_component(component: ModelComponent) -> bool {
    matches!(
        component,
        ModelComponent::Transformer
            | ModelComponent::TransformerShard
            | ModelComponent::LowNoiseTransformer
            | ModelComponent::DistilledLora
            | ModelComponent::LowNoiseDistilledLora
            | ModelComponent::TaskConfig
            | ModelComponent::Upscaler
    )
}

/// Determine the clean storage path for a model file relative to the models directory.
///
/// - **Transformer / TransformerShard**: `<model-name>/<hf_filename>` (model-specific)
/// - **All other components** (VAE, encoders, tokenizers): `shared/<family>/<hf_filename>` (shared)
///
/// Model names are sanitized: colons become dashes (e.g., `flux-schnell:q8` → `flux-schnell-q8`).
/// HF filename paths (e.g., `text_encoder/model-00001-of-00003.safetensors`) are preserved as-is,
/// creating subdirectories under the target directory.
pub fn storage_path(manifest: &ModelManifest, file: &ModelFile) -> PathBuf {
    let sanitized_name = manifest.name.replace(':', "-");

    // H3's official and Comfy transformers use the same task architecture
    // config. Keep one copy per task across layouts while the task-specific
    // source path (`transformer` vs `transformer_ref`) prevents cross-task
    // substitution. The Comfy manifests intentionally omit official sharded
    // weight indexes because those do not describe their single-file weights.
    if manifest.family == crate::minimax_h3::FAMILY
        && file.component == ModelComponent::TaskConfig
        && file.hf_repo == crate::minimax_h3::OFFICIAL_REPO
    {
        return PathBuf::from("shared")
            .join(crate::minimax_h3::FAMILY)
            .join(&file.hf_filename);
    }

    if is_model_specific_component(file.component) {
        PathBuf::from(&sanitized_name).join(&file.hf_filename)
    } else {
        // FLUX.2 [dev] publishes a different Mistral encoder, tokenizer, and
        // VAE under the same relative filenames used by the Klein repos. Do
        // not let installing one variant overwrite the other's shared assets.
        if file.hf_repo == "black-forest-labs/FLUX.2-dev" {
            return PathBuf::from("shared")
                .join("flux2-dev")
                .join(&file.hf_filename);
        }
        // Companion manifests (`t5-v1_1-xxl`, `clip-l`, etc.) route their
        // shared components to the canonical owning family's bucket so an
        // already-installed FLUX/SDXL/SD1.5 model's encoders/tokenizers are
        // found by `ModelPaths::resolve(<companion>, config)` without a
        // duplicate download. Without this routing, t5-v1_1-xxl's tokenizer
        // would land at `shared/companion/...` and the catalog bridge's
        // synthesized `cfg.t5_tokenizer` would stay None even when the file
        // exists at `shared/flux/...` from a prior FLUX install.
        if manifest.family == "companion" {
            let canonical_family = match manifest.name.as_str() {
                "t5-v1_1-xxl" | "clip-l" | "flux-vae" => "flux",
                "clip-g" | "sdxl-vae" => "sdxl",
                "sd-vae-ft-mse" => "sd15",
                "ltx-video-vae" => "ltx-video",
                "ltx2-vae" | "ltx2.3-vae" | "ltx2.3-text-projection" => "ltx2",
                "z-image-te" => "z-image",
                "flux2-vae" | "flux2-te" | "flux2-te-9b" => "flux2",
                "ltx2-te" => "ltx2",
                "wan-umt5" | "wan21-vae" | "wan22-vae" => "wan",
                _ => "companion",
            };
            return PathBuf::from("shared")
                .join(canonical_family)
                .join(&file.hf_filename);
        }
        if manifest.family == "ltx-video" {
            return match file.component {
                // LTX reuses the shared FLUX T5 assets. Keep them under the
                // flux shared bucket so path layout matches the actual source.
                ModelComponent::T5Encoder | ModelComponent::T5Tokenizer => {
                    PathBuf::from("shared").join("flux").join(&file.hf_filename)
                }
                // The currently compatible VAE still comes from the legacy
                // LTX-Video-0.9.5 repo, and the 0.9.8 latent upsampler comes
                // from the current LTX-Video repo. Keep the clean path aligned
                // with the source repo to avoid shared/ltx-video ambiguity.
                ModelComponent::Vae | ModelComponent::SpatialUpscaler => {
                    let repo_leaf = file.hf_repo.rsplit('/').next().unwrap_or("ltx-video");
                    PathBuf::from("shared")
                        .join(repo_leaf)
                        .join(&file.hf_filename)
                }
                _ => PathBuf::from("shared")
                    .join(&manifest.family)
                    .join(&file.hf_filename),
            };
        }
        if manifest.family == "qwen-image" || manifest.family == "qwen-image-edit" {
            match file.hf_repo.as_str() {
                "Qwen/Qwen-Image" => {
                    return PathBuf::from("shared")
                        .join("qwen-image-base")
                        .join(&file.hf_filename);
                }
                "Qwen/Qwen-Image-2512" => {
                    return PathBuf::from("shared")
                        .join("qwen-image")
                        .join(&file.hf_filename);
                }
                "Qwen/Qwen-Image-Edit-2511" => {
                    return PathBuf::from("shared")
                        .join("qwen-image-edit")
                        .join(&file.hf_filename);
                }
                _ => {}
            }
        }
        // Check if this filename collides with another file in the same
        // manifest from a different HF repo. If so, use the repo name
        // as a subfolder to disambiguate (e.g. Wuerstchen has two
        // text_encoder/model.safetensors from different repos).
        let has_collision = manifest
            .files
            .iter()
            .any(|other| other.hf_filename == file.hf_filename && other.hf_repo != file.hf_repo);
        if has_collision {
            let repo_leaf = file.hf_repo.rsplit('/').next().unwrap_or(&manifest.family);
            PathBuf::from("shared")
                .join(repo_leaf)
                .join(&file.hf_filename)
        } else {
            PathBuf::from("shared")
                .join(&manifest.family)
                .join(&file.hf_filename)
        }
    }
}

/// Candidate on-disk storage paths for a manifest file.
///
/// The first path is the canonical current clean path. Additional entries are
/// backwards-compatible legacy locations that should still be discovered and
/// repaired in place by `mold pull`.
pub fn storage_path_candidates(manifest: &ModelManifest, file: &ModelFile) -> Vec<PathBuf> {
    let canonical = storage_path(manifest, file);
    let mut paths = vec![canonical.clone()];

    if manifest.family == "ltx-video" && !is_model_specific_component(file.component) {
        let legacy = PathBuf::from("shared")
            .join(&manifest.family)
            .join(&file.hf_filename);
        if legacy != canonical {
            paths.push(legacy);
        }
    }

    paths
}

/// Shared FLUX component files (VAE, T5, CLIP, tokenizers) — identical across all FLUX models.
fn shared_flux_files() -> Vec<ModelFile> {
    vec![
        ModelFile {
            hf_repo: "black-forest-labs/FLUX.1-schnell".to_string(),
            hf_filename: "ae.safetensors".to_string(),
            component: ModelComponent::Vae,
            size_bytes: 335_304_388,
            gated: true, // BFL repos now require authentication
            sha256: Some("afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38"),
        },
        ModelFile {
            hf_repo: "comfyanonymous/flux_text_encoders".to_string(),
            hf_filename: "t5xxl_fp16.safetensors".to_string(),
            component: ModelComponent::T5Encoder,
            size_bytes: 9_787_841_024,
            gated: false,
            sha256: Some("6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635"),
        },
        ModelFile {
            hf_repo: "comfyanonymous/flux_text_encoders".to_string(),
            hf_filename: "clip_l.safetensors".to_string(),
            component: ModelComponent::ClipEncoder,
            size_bytes: 246_144_152,
            gated: false,
            sha256: Some("660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd"),
        },
        ModelFile {
            hf_repo: "lmz/mt5-tokenizers".to_string(),
            hf_filename: "t5-v1_1-xxl.tokenizer.json".to_string(),
            component: ModelComponent::T5Tokenizer,
            size_bytes: 2_424_257,
            gated: false,
            sha256: None, // non-LFS file, no SHA-256 from HF API
        },
        ModelFile {
            hf_repo: "openai/clip-vit-large-patch14".to_string(),
            hf_filename: "tokenizer.json".to_string(),
            component: ModelComponent::ClipTokenizer,
            size_bytes: 2_224_003,
            gated: false,
            sha256: None, // non-LFS file, no SHA-256 from HF API
        },
    ]
}

/// All known downloadable model manifests, computed once and cached.
static KNOWN_MANIFESTS: LazyLock<Vec<ModelManifest>> = LazyLock::new(build_known_manifests);

/// Index mapping canonical model names to their index in `KNOWN_MANIFESTS`.
static MANIFEST_INDEX: LazyLock<HashMap<String, usize>> = LazyLock::new(|| {
    KNOWN_MANIFESTS
        .iter()
        .enumerate()
        .map(|(i, m)| (m.name.clone(), i))
        .collect()
});

/// All known downloadable model manifests (FLUX, SDXL, SD3, SD1.5, Z-Image, Flux.2, Qwen-Image, LTX Video).
pub fn known_manifests() -> &'static [ModelManifest] {
    &KNOWN_MANIFESTS
}

/// Visible (non-hidden) manifests for user-facing lists (CLI, TUI, tab completion).
/// Hidden models can still be used via explicit `mold run <name>` or config.toml.
pub fn visible_manifests() -> impl Iterator<Item = &'static ModelManifest> {
    known_manifests().iter().filter(|m| !m.hidden)
}

fn build_known_manifests() -> Vec<ModelManifest> {
    let mut manifests = vec![
        ModelManifest {
            name: "flux-schnell:q8".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Schnell Q8 — fast 4-step, general purpose".to_string(),
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "city96/FLUX.1-schnell-gguf".to_string(),
                    hf_filename: "flux1-schnell-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 12_687_821_728,
                    gated: false,
                    sha256: Some(
                        "f6694941193b10148dbf1f0f498d4ccd3e9875c127fc53946213b68580c66f10",
                    ),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 4,
                guidance: 0.0,
                width: 1024,
                height: 1024,
                is_schnell: true,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "flux-dev:q8".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Dev Q8 — full quality, 20+ steps".to_string(),
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "city96/FLUX.1-dev-gguf".to_string(),
                    hf_filename: "flux1-dev-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 12_708_281_504,
                    gated: false,
                    sha256: Some(
                        "129032f32224bf7138f16e18673d8008ba5f84c1ec74063bf4511a8bb4cf553d",
                    ),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 3.5,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "flux-dev:q4".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Dev Q4 — smaller/faster, good quality".to_string(),
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "city96/FLUX.1-dev-gguf".to_string(),
                    hf_filename: "flux1-dev-Q4_1.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 7_530_806_432,
                    gated: false,
                    sha256: Some(
                        "da04c47a9b717bf9a4dd545e46d89e4a62fb44b9497bf9a5d13d622d592fbcda",
                    ),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 3.5,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "flux-dev:q6".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Dev Q6 — best quality/size trade-off".to_string(),
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "city96/FLUX.1-dev-gguf".to_string(),
                    hf_filename: "flux1-dev-Q6_K.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 9_857_000_736,
                    gated: false,
                    sha256: Some(
                        "9566d56031d7f8de184bb5a0393073956ec4e28b32db3f860bd2b87edca04d13",
                    ),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 3.5,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "flux-dev:bf16".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Dev BF16 — full quality, full precision (23.8GB transformer)"
                .to_string(),
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "black-forest-labs/FLUX.1-dev".to_string(),
                    hf_filename: "flux1-dev.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 23_802_932_552,
                    gated: true,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 3.5,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "flux-schnell:bf16".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Schnell BF16 — fast 4-step, full precision (23.8GB transformer)"
                .to_string(),
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "black-forest-labs/FLUX.1-schnell".to_string(),
                    hf_filename: "flux1-schnell.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 23_782_506_688,
                    gated: true,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 4,
                guidance: 0.0,
                width: 1024,
                height: 1024,
                is_schnell: true,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "flux-schnell:q4".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Schnell Q4 — fast 4-step, smaller footprint".to_string(),
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "city96/FLUX.1-schnell-gguf".to_string(),
                    hf_filename: "flux1-schnell-Q4_1.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 7_510_346_656,
                    gated: false,
                    sha256: Some(
                        "a798b7196d2fe614cf9bae9a617dbd9f2c14673e454c7f2f6a500347274630b5",
                    ),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 4,
                guidance: 0.0,
                width: 1024,
                height: 1024,
                is_schnell: true,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "flux-schnell:q6".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Schnell Q6 — fast 4-step, best quality/size trade-off".to_string(),
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "city96/FLUX.1-schnell-gguf".to_string(),
                    hf_filename: "flux1-schnell-Q6_K.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 9_834_955_808,
                    gated: false,
                    sha256: Some(
                        "a42fd143cec4d7194da281dc8d23a8fe54b16875a13423c042cb545d1da6fa50",
                    ),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 4,
                guidance: 0.0,
                width: 1024,
                height: 1024,
                is_schnell: true,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "flux-krea:q8".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Krea Dev Q8 — aesthetic photography fine-tune".to_string(),
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "QuantStack/FLUX.1-Krea-dev-GGUF".to_string(),
                    hf_filename: "flux1-krea-dev-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 12_714_452_256,
                    gated: false,
                    sha256: Some(
                        "0d085b1e3ae0b90e5dbf74da049a80a565617de622a147d28ee37a07761fbd90",
                    ),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 4.5,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "flux-krea:q4".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Krea Dev Q4 — aesthetic photography, smaller footprint"
                .to_string(),
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "QuantStack/FLUX.1-Krea-dev-GGUF".to_string(),
                    hf_filename: "flux1-krea-dev-Q4_1.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 7_536_977_184,
                    gated: false,
                    sha256: Some(
                        "be4c46e5492761f00c0d9ca15e78936fbe54c4ee65b16da8e0dbf5f2115ae6b2",
                    ),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 4.5,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "flux-krea:q6".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Krea Dev Q6 — aesthetic photography, best quality/size trade-off"
                .to_string(),
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "QuantStack/FLUX.1-Krea-dev-GGUF".to_string(),
                    hf_filename: "flux1-krea-dev-Q6_K.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 9_848_349_984,
                    gated: false,
                    sha256: Some(
                        "c50c13ebe1207b2c87b251ccf3a55b9eb54c84f73cee62503d17acd8a460953e",
                    ),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 4.5,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "flux-krea:fp8".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Krea Dev FP8 — aesthetic photography, scaled FP8 quantization"
                .to_string(),
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "Clybius/FLUX.1-Krea-dev-scaled-fp8".to_string(),
                    hf_filename: "flux1-krea-dev_float8_e4m3fn_learned_svd.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 11_904_609_210,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 4.5,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        // ── jibMixFlux v7 PixelHeaven (FLUX-dev fine-tune by J1B) ──────────
        ModelManifest {
            name: "jibmix-flux:fp8".to_string(),
            family: "flux".to_string(),
            description: "jibMixFlux v7.2 PixelHeaven FP8 — photorealistic fine-tune".to_string(),
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "ak199621/jibMixFlux_v72PixelHeaven.safetensors".to_string(),
                    hf_filename: "jibMixFlux_v72PixelHeaven.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 11_901_516_784,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 3.0,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "jibmix-flux:q5".to_string(),
            family: "flux".to_string(),
            description: "jibMixFlux v7 PixelHeaven Q5 — photorealistic, best GGUF quality"
                .to_string(),
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "TheLounger/Jib_Mix_Flux_v7_Beta-GGUF".to_string(),
                    hf_filename: "Jib_Mix_Flux_v7_Beta-Q5_K_M.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 8_421_981_344,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 3.0,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "jibmix-flux:q4".to_string(),
            family: "flux".to_string(),
            description: "jibMixFlux v7 PixelHeaven Q4 — photorealistic, good quality/size"
                .to_string(),
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "TheLounger/Jib_Mix_Flux_v7_Beta-GGUF".to_string(),
                    hf_filename: "Jib_Mix_Flux_v7_Beta-Q4_K_M.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 6_934_297_760,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 3.0,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "jibmix-flux:q3".to_string(),
            family: "flux".to_string(),
            description: "jibMixFlux v7 PixelHeaven Q3 — photorealistic, smaller footprint"
                .to_string(),
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "TheLounger/Jib_Mix_Flux_v7_Beta-GGUF".to_string(),
                    hf_filename: "Jib_Mix_Flux_v7_Beta-Q3_K_M.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 5_370_969_248,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 3.0,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        // ── UltraReal Fine-Tune (photorealistic FLUX-dev fine-tune by Danrisi) ──
        ModelManifest {
            name: "ultrareal-v2:bf16".to_string(),
            family: "flux".to_string(),
            description: "UltraReal Fine-Tune v2.0 BF16 — photorealistic, full precision"
                .to_string(),
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "youknownothing/UltrarealFineTune-Flux".to_string(),
                    hf_filename: "ultrarealFineTune_v20.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 23_802_910_336,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 3.5,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "ultrareal-v3:q8".to_string(),
            family: "flux".to_string(),
            description: "UltraReal Fine-Tune v3 Q8 — photorealistic, best GGUF quality"
                .to_string(),
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "belisarius/FLUX.1-dev-ultrarealFineTune_v3Experimental-GGUF"
                        .to_string(),
                    hf_filename: "ultrarealFineTune_v3Experimental-q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 12_723_103_008,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 3.5,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "ultrareal-v3:q6".to_string(),
            family: "flux".to_string(),
            description: "UltraReal Fine-Tune v3 Q6 — photorealistic, best quality/size trade-off"
                .to_string(),
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "belisarius/FLUX.1-dev-ultrarealFineTune_v3Experimental-GGUF"
                        .to_string(),
                    hf_filename: "ultrarealFineTune_v3Experimental-q6_k.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 9_857_000_736,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 3.5,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "ultrareal-v3:q4".to_string(),
            family: "flux".to_string(),
            description: "UltraReal Fine-Tune v3 Q4 — photorealistic, smaller footprint"
                .to_string(),
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "belisarius/FLUX.1-dev-ultrarealFineTune_v3Experimental-GGUF"
                        .to_string(),
                    hf_filename: "ultrarealFineTune_v3Experimental-q4_k_s.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 6_805_988_640,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 3.5,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "ultrareal-v4:q8".to_string(),
            family: "flux".to_string(),
            description: "UltraReal Fine-Tune v4 Q8 — photorealistic (latest), best GGUF quality"
                .to_string(),
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "void-gryph/ultrareal-fine-tune-GGUF".to_string(),
                    hf_filename: "ultrareal-fine-tune.Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 12_619_809_408,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 3.5,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "ultrareal-v4:q5".to_string(),
            family: "flux".to_string(),
            description: "UltraReal Fine-Tune v4 Q5 — photorealistic (latest), good quality/size"
                .to_string(),
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "void-gryph/ultrareal-fine-tune-GGUF".to_string(),
                    hf_filename: "ultrareal-fine-tune.Q5_K_M.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 8_170_103_424,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 3.5,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "ultrareal-v4:q4".to_string(),
            family: "flux".to_string(),
            description: "UltraReal Fine-Tune v4 Q4 — photorealistic (latest), smaller footprint"
                .to_string(),
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "void-gryph/ultrareal-fine-tune-GGUF".to_string(),
                    hf_filename: "ultrareal-fine-tune.Q4_K_M.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 6_686_868_096,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 3.5,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        // ── iNiverse Mix SFW/NSFW (FLUX-dev fine-tune by JinnGames) ──────────
        ModelManifest {
            name: "iniverse-mix:fp8".to_string(),
            family: "flux".to_string(),
            description: "iNiverse Mix F1D RealNSFW GuoFeng v2 FP8 — realistic SFW/NSFW mix"
                .to_string(),
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "modelzpalace/iniverseMixSFWNSFW_f1dRealnsfwGuofengV2".to_string(),
                    hf_filename: "iniverseMixSFWNSFW_f1dRealnsfwGuofengV2.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 11_901_513_960,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 3.5,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
    ];
    manifests.extend(sd15_manifests());
    manifests.extend(sd3_manifests());
    manifests.extend(sdxl_manifests());
    manifests.extend(zimage_manifests());
    manifests.extend(flux2_manifests());
    manifests.extend(qwen_image_manifests());
    manifests.extend(wuerstchen_manifests());
    manifests.extend(ltx_video_manifests());
    manifests.extend(ltx2_manifests());
    manifests.extend(wan_manifests());
    manifests.extend(crate::minimax_h3::manifests());
    manifests.extend(ltx2_control_manifests());
    manifests.extend(ltx2_camera_control_manifests());
    manifests.extend(controlnet_manifests());
    manifests.extend(qwen3_expand_manifests());
    manifests.extend(upscaler_manifests());
    manifests.extend(companion_manifests());
    manifests
}

/// Shared SD3 component files (VAE, CLIP-L, CLIP-G, T5-XXL, tokenizers) — identical across all SD3.5 models.
///
/// SD3 uses three text encoders: CLIP-L (768-dim), CLIP-G (1280-dim), and T5-XXL (4096-dim).
/// The VAE is embedded in the transformer safetensors for BF16, but GGUF models need a separate VAE.
/// For separate text encoders, we use files from stabilityai/stable-diffusion-3.5-large.
fn shared_sd3_files() -> Vec<ModelFile> {
    vec![
        // VAE: SD3 VAE is embedded in the monolithic safetensors from stabilityai.
        // The mmap approach means only VAE weights (~300MB) get paged in, not the full file.
        // The pipeline uses vb.rename_f(sd3_vae_vb_rename).pp("first_stage_model") prefix.
        ModelFile {
            hf_repo: "stabilityai/stable-diffusion-3.5-large".to_string(),
            hf_filename: "sd3.5_large.safetensors".to_string(),
            component: ModelComponent::Vae,
            size_bytes: 16_460_379_262, // monolithic (VAE portion ~300MB via mmap)
            gated: true,
            sha256: None,
        },
        // CLIP-L encoder
        ModelFile {
            hf_repo: "stabilityai/stable-diffusion-3.5-large".to_string(),
            hf_filename: "text_encoders/clip_l.safetensors".to_string(),
            component: ModelComponent::ClipEncoder,
            size_bytes: 246_144_152,
            gated: true,
            sha256: None,
        },
        // CLIP-G encoder
        ModelFile {
            hf_repo: "stabilityai/stable-diffusion-3.5-large".to_string(),
            hf_filename: "text_encoders/clip_g.safetensors".to_string(),
            component: ModelComponent::ClipEncoder2,
            size_bytes: 1_389_382_176,
            gated: true,
            sha256: None,
        },
        // T5-XXL encoder
        ModelFile {
            hf_repo: "stabilityai/stable-diffusion-3.5-large".to_string(),
            hf_filename: "text_encoders/t5xxl_fp16.safetensors".to_string(),
            component: ModelComponent::T5Encoder,
            size_bytes: 9_787_841_024,
            gated: true,
            sha256: None,
        },
        // CLIP-L tokenizer (same as FLUX/SDXL)
        ModelFile {
            hf_repo: "openai/clip-vit-large-patch14".to_string(),
            hf_filename: "tokenizer.json".to_string(),
            component: ModelComponent::ClipTokenizer,
            size_bytes: 2_224_003,
            gated: false,
            sha256: None,
        },
        // CLIP-G tokenizer (same as SDXL)
        ModelFile {
            hf_repo: "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k".to_string(),
            hf_filename: "tokenizer.json".to_string(),
            component: ModelComponent::ClipTokenizer2,
            size_bytes: 2_224_003,
            gated: false,
            sha256: None,
        },
        // T5 tokenizer (same as FLUX)
        ModelFile {
            hf_repo: "lmz/mt5-tokenizers".to_string(),
            hf_filename: "t5-v1_1-xxl.tokenizer.json".to_string(),
            component: ModelComponent::T5Tokenizer,
            size_bytes: 2_424_257,
            gated: false,
            sha256: None,
        },
    ]
}

/// All known SD3.5 model manifests.
fn sd3_manifests() -> Vec<ModelManifest> {
    vec![
        // --- SD3.5 Large (depth=38, 8.1B) ---
        ModelManifest {
            name: "sd3.5-large:q8".to_string(),
            family: "sd3".to_string(),
            description: "SD3.5 Large Q8 — 8.1B MMDiT, high quality, 28 steps".to_string(),
            files: {
                let mut files = shared_sd3_files();
                files.push(ModelFile {
                    hf_repo: "city96/stable-diffusion-3.5-large-gguf".to_string(),
                    hf_filename: "sd3.5_large-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 8_779_212_512,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 28,
                guidance: 4.0,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "sd3.5-large:q4".to_string(),
            family: "sd3".to_string(),
            description: "SD3.5 Large Q4 — 8.1B MMDiT, smaller footprint, 28 steps".to_string(),
            files: {
                let mut files = shared_sd3_files();
                files.push(ModelFile {
                    hf_repo: "city96/stable-diffusion-3.5-large-gguf".to_string(),
                    hf_filename: "sd3.5_large-Q4_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 4_772_054_752,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 28,
                guidance: 4.0,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        // --- SD3.5 Large Turbo (depth=38, 8.1B, 4 steps, CFG=1.0) ---
        ModelManifest {
            name: "sd3.5-large-turbo:q8".to_string(),
            family: "sd3".to_string(),
            description: "SD3.5 Large Turbo Q8 — 8.1B MMDiT, fast 4-step generation".to_string(),
            files: {
                let mut files = shared_sd3_files();
                files.push(ModelFile {
                    hf_repo: "city96/stable-diffusion-3.5-large-turbo-gguf".to_string(),
                    hf_filename: "sd3.5_large_turbo-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 8_779_212_512,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 4,
                guidance: 1.0,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        // --- SD3.5 Medium (depth=24, 2.5B) ---
        ModelManifest {
            name: "sd3.5-medium:q8".to_string(),
            family: "sd3".to_string(),
            description: "SD3.5 Medium Q8 — 2.5B MMDiT, SLG support, 28 steps".to_string(),
            files: {
                let mut files = shared_sd3_files();
                files.push(ModelFile {
                    hf_repo: "city96/stable-diffusion-3.5-medium-gguf".to_string(),
                    hf_filename: "sd3.5_medium-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 2_855_825_856,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 28,
                guidance: 4.0,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
    ]
}

/// Shared SD1.5 component files (VAE, CLIP-L encoder, tokenizer) — identical across all SD1.5 models.
fn shared_sd15_files() -> Vec<ModelFile> {
    vec![
        ModelFile {
            hf_repo: "stabilityai/sd-vae-ft-mse".to_string(),
            hf_filename: "diffusion_pytorch_model.safetensors".to_string(),
            component: ModelComponent::Vae,
            size_bytes: 334_643_276,
            gated: false,
            sha256: Some("a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815"),
        },
        ModelFile {
            hf_repo: "stable-diffusion-v1-5/stable-diffusion-v1-5".to_string(),
            hf_filename: "text_encoder/model.safetensors".to_string(),
            component: ModelComponent::ClipEncoder,
            size_bytes: 492_265_874,
            gated: false,
            sha256: Some("d008943c017f0092921106440254dbbe00b6a285f7883ec8ba160c3faad88334"),
        },
        ModelFile {
            hf_repo: "openai/clip-vit-large-patch14".to_string(),
            hf_filename: "tokenizer.json".to_string(),
            component: ModelComponent::ClipTokenizer,
            size_bytes: 2_224_003,
            gated: false,
            sha256: None, // non-LFS file, no SHA-256 from HF API
        },
    ]
}

/// All known SD1.5 model manifests.
fn sd15_manifests() -> Vec<ModelManifest> {
    vec![
        ModelManifest {
            name: "sd15:fp16".to_string(),
            family: "sd15".to_string(),
            description: "Stable Diffusion 1.5 — canonical base model, huge LoRA ecosystem"
                .to_string(),
            files: {
                let mut files = shared_sd15_files();
                files.push(ModelFile {
                    hf_repo: "stable-diffusion-v1-5/stable-diffusion-v1-5".to_string(),
                    hf_filename: "unet/diffusion_pytorch_model.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 3_438_167_540,
                    gated: false,
                    sha256: Some(
                        "19da7aaa4b880e59d56843f1fcb4dd9b599c28a1d9d9af7c1143057c8ffae9f1",
                    ),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 7.5,
                width: 512,
                height: 512,
                is_schnell: false,
                scheduler: Some(Scheduler::Ddim),
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "dreamshaper-v8:fp16".to_string(),
            family: "sd15".to_string(),
            description: "DreamShaper v8 — best versatile SD1.5, photorealistic + fantasy"
                .to_string(),
            files: {
                let mut files = shared_sd15_files();
                files.push(ModelFile {
                    hf_repo: "Lykon/dreamshaper-8".to_string(),
                    hf_filename: "unet/diffusion_pytorch_model.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 3_438_167_536,
                    gated: false,
                    sha256: Some(
                        "89b54dc332757e6fff8caef7399e8061833d7d668d42fdbcc02b3e366921c5a6",
                    ),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 7.5,
                width: 512,
                height: 512,
                is_schnell: false,
                scheduler: Some(Scheduler::Ddim),
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "realistic-vision-v5:fp16".to_string(),
            family: "sd15".to_string(),
            description: "Realistic Vision v5.1 — gold standard photorealistic SD1.5".to_string(),
            files: {
                let mut files = shared_sd15_files();
                files.push(ModelFile {
                    hf_repo: "SG161222/Realistic_Vision_V5.1_noVAE".to_string(),
                    hf_filename: "unet/diffusion_pytorch_model.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 3_438_167_536,
                    gated: false,
                    sha256: Some(
                        "4e0868e8fcae7d4ea8f8cdd3051704b3b47d741dc8e8629552d1a07f6efb8e32",
                    ),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 7.5,
                width: 512,
                height: 512,
                is_schnell: false,
                scheduler: Some(Scheduler::Ddim),
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
    ]
}

/// Shared SDXL component files (VAE, dual-CLIP encoders, tokenizers) — identical across all SDXL models.
fn shared_sdxl_files() -> Vec<ModelFile> {
    vec![
        ModelFile {
            hf_repo: "madebyollin/sdxl-vae-fp16-fix".to_string(),
            hf_filename: "diffusion_pytorch_model.safetensors".to_string(),
            component: ModelComponent::Vae,
            size_bytes: 334_643_268,
            gated: false,
            sha256: Some("1b909373b28f2137098b0fd9dbc6f97f8410854f31f84ddc9fa04b077b0ace2c"),
        },
        ModelFile {
            hf_repo: "stabilityai/stable-diffusion-xl-base-1.0".to_string(),
            hf_filename: "text_encoder/model.safetensors".to_string(),
            component: ModelComponent::ClipEncoder,
            size_bytes: 492_265_168,
            gated: false,
            sha256: Some("5c3d6454dd2d23414b56aa1b5858a72487a656937847b6fea8d0606d7a42cdbc"),
        },
        ModelFile {
            hf_repo: "stabilityai/stable-diffusion-xl-base-1.0".to_string(),
            hf_filename: "text_encoder_2/model.safetensors".to_string(),
            component: ModelComponent::ClipEncoder2,
            size_bytes: 2_778_702_264,
            gated: false,
            sha256: Some("3a6032f63d37ae02bbc74ccd6a27440578cd71701f96532229d0154f55a8d3ff"),
        },
        ModelFile {
            hf_repo: "openai/clip-vit-large-patch14".to_string(),
            hf_filename: "tokenizer.json".to_string(),
            component: ModelComponent::ClipTokenizer,
            size_bytes: 2_224_003,
            gated: false,
            sha256: None, // non-LFS file, no SHA-256 from HF API
        },
        ModelFile {
            hf_repo: "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k".to_string(),
            hf_filename: "tokenizer.json".to_string(),
            component: ModelComponent::ClipTokenizer2,
            size_bytes: 2_224_003,
            gated: false,
            sha256: None, // non-LFS file, no SHA-256 from HF API
        },
    ]
}

/// All known SDXL model manifests.
fn sdxl_manifests() -> Vec<ModelManifest> {
    vec![
        // --- Standard SDXL (DDIM scheduler, 20-30 steps, guidance 7.5) ---
        ModelManifest {
            name: "sdxl-base:fp16".to_string(),
            family: "sdxl".to_string(),
            description: "SDXL Base 1.0 — official Stability AI base model".to_string(),
            files: {
                let mut files = shared_sdxl_files();
                files.push(ModelFile {
                    hf_repo: "stabilityai/stable-diffusion-xl-base-1.0".to_string(),
                    hf_filename: "unet/diffusion_pytorch_model.fp16.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 5_135_149_760,
                    gated: false,
                    sha256: Some(
                        "83e012a805b84c7ca28e5646747c90a243c65c8ba4f070e2d7ddc9d74661e139",
                    ),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 7.5,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: Some(Scheduler::Ddim),
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "dreamshaper-xl:fp16".to_string(),
            family: "sdxl".to_string(),
            description: "DreamShaper XL — fantasy, concept art, stylized".to_string(),
            files: {
                let mut files = shared_sdxl_files();
                files.push(ModelFile {
                    hf_repo: "Lykon/dreamshaper-xl-v2-turbo".to_string(),
                    hf_filename: "unet/diffusion_pytorch_model.fp16.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 5_135_149_760,
                    gated: false,
                    sha256: Some(
                        "c1217e273e6fd7570c2ae9d38172323ff0b6f8ac7f2000b3ba99d4851906ee1e",
                    ),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 8,
                guidance: 2.0,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: Some(Scheduler::EulerAncestral),
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "juggernaut-xl:fp16".to_string(),
            family: "sdxl".to_string(),
            description: "Juggernaut XL — photorealism, cinematic lighting".to_string(),
            files: {
                let mut files = shared_sdxl_files();
                files.push(ModelFile {
                    hf_repo: "RunDiffusion/Juggernaut-XL-v9".to_string(),
                    hf_filename: "unet/diffusion_pytorch_model.fp16.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 5_135_149_760,
                    gated: false,
                    sha256: Some(
                        "cf1ee18eb36712683f50c1e674634875e2adf7413d7492d5f9aa7e69e1a8c17a",
                    ),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 30,
                guidance: 7.0,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: Some(Scheduler::Ddim),
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "realvis-xl:fp16".to_string(),
            family: "sdxl".to_string(),
            description: "RealVisXL V5.0 — photorealism, versatile subjects".to_string(),
            files: {
                let mut files = shared_sdxl_files();
                files.push(ModelFile {
                    hf_repo: "SG161222/RealVisXL_V5.0".to_string(),
                    hf_filename: "unet/diffusion_pytorch_model.fp16.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 5_135_149_760,
                    gated: false,
                    sha256: Some(
                        "ea10386073d39ffdde9fda426745b3f5e9dcd2af204c128ece0f4ea84570ffee",
                    ),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 7.5,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: Some(Scheduler::Ddim),
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "playground-v2.5:fp16".to_string(),
            family: "sdxl".to_string(),
            description: "Playground v2.5 — aesthetic quality, artistic".to_string(),
            files: {
                let mut files = shared_sdxl_files();
                files.push(ModelFile {
                    hf_repo: "playgroundai/playground-v2.5-1024px-aesthetic".to_string(),
                    hf_filename: "unet/diffusion_pytorch_model.fp16.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 5_135_149_760,
                    gated: false,
                    sha256: Some(
                        "933778ce76c1fc0ca918b37e1488411b8a99bbd3279c12f527a3ac995a340864",
                    ),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 3.0,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: Some(Scheduler::Ddim),
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        // --- Pony / CyberRealistic (standard SDXL architecture, anime/art/photorealistic) ---
        ModelManifest {
            name: "pony-v6:fp16".to_string(),
            family: "sdxl".to_string(),
            description: "Pony Diffusion V6 XL — anime, art, stylized generation".to_string(),
            files: {
                let mut files = shared_sdxl_files();
                files.push(ModelFile {
                    hf_repo: "kitty7779/ponyDiffusionV6XL".to_string(),
                    hf_filename: "unet/diffusion_pytorch_model.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 5_135_149_760,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 7.0,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: Some(Scheduler::EulerAncestral),
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "cyberrealistic-pony:fp16".to_string(),
            family: "sdxl".to_string(),
            description: "CyberRealistic Pony v16 — photorealistic Pony fine-tune".to_string(),
            files: {
                let mut files = shared_sdxl_files();
                files.push(ModelFile {
                    hf_repo: "LillyCherry/cyberrealisticPony_v160".to_string(),
                    hf_filename: "unet/diffusion_pytorch_model.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 5_135_149_760,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 25,
                guidance: 7.0,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: Some(Scheduler::EulerAncestral),
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        // --- Turbo SDXL (Euler Ancestral, 1-4 steps, guidance 0.0) ---
        ModelManifest {
            name: "sdxl-turbo:fp16".to_string(),
            family: "sdxl".to_string(),
            description: "SDXL Turbo — ultra-fast 1-4 step generation".to_string(),
            files: {
                let mut files = shared_sdxl_files();
                files.push(ModelFile {
                    hf_repo: "stabilityai/sdxl-turbo".to_string(),
                    hf_filename: "unet/diffusion_pytorch_model.fp16.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 5_135_149_760,
                    gated: false,
                    sha256: Some(
                        "48fa46161a745f48d4054df3fe13804ee255486bca893403b60373c188fd1bdb",
                    ),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 4,
                guidance: 0.0,
                width: 512,
                height: 512,
                is_schnell: false,
                scheduler: Some(Scheduler::EulerAncestral),
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
    ]
}

/// Shared Z-Image component files (Qwen3 text encoder, VAE, tokenizer) — identical across all Z-Image models.
fn shared_zimage_files() -> Vec<ModelFile> {
    vec![
        // Qwen3 text encoder (3 shards)
        ModelFile {
            hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
            hf_filename: "text_encoder/model-00001-of-00003.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 3_957_900_840,
            gated: false,
            sha256: Some("328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223"),
        },
        ModelFile {
            hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
            hf_filename: "text_encoder/model-00002-of-00003.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 3_987_450_520,
            gated: false,
            sha256: Some("6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5"),
        },
        ModelFile {
            hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
            hf_filename: "text_encoder/model-00003-of-00003.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 99_630_640,
            gated: false,
            sha256: Some("7ca841ee75b9c61267c0c6148fd8d096d3d21b6d3e161256a9b878154f91fc52"),
        },
        // VAE
        ModelFile {
            hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
            hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
            component: ModelComponent::Vae,
            size_bytes: 167_666_902,
            gated: false,
            sha256: Some("f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3"),
        },
        // Qwen3 tokenizer
        ModelFile {
            hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
            hf_filename: "tokenizer/tokenizer.json".to_string(),
            component: ModelComponent::TextTokenizer,
            size_bytes: 11_422_654,
            gated: false,
            sha256: Some("aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
        },
    ]
}

/// All known Z-Image model manifests.
fn zimage_manifests() -> Vec<ModelManifest> {
    vec![
        // BF16 full precision
        ModelManifest {
            name: "z-image-turbo:bf16".to_string(),
            family: "z-image".to_string(),
            description: "Z-Image Turbo BF16 — 9-step, Alibaba flow-matching".to_string(),
            files: {
                let mut files = shared_zimage_files();
                // Transformer shards (3 files)
                files.push(ModelFile {
                    hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
                    hf_filename: "transformer/diffusion_pytorch_model-00001-of-00003.safetensors"
                        .to_string(),
                    component: ModelComponent::TransformerShard,
                    size_bytes: 9_973_693_184,
                    gated: false,
                    sha256: Some(
                        "95facd593e2549e8252acb571c653d57f7ddb7f1060d4e81712f152555a88804",
                    ),
                });
                files.push(ModelFile {
                    hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
                    hf_filename: "transformer/diffusion_pytorch_model-00002-of-00003.safetensors"
                        .to_string(),
                    component: ModelComponent::TransformerShard,
                    size_bytes: 9_973_693_184,
                    gated: false,
                    sha256: Some(
                        "a4bbe43ee184a1fb5af4b412d27555f532893bdc3165b1149e304ed82b5d7015",
                    ),
                });
                files.push(ModelFile {
                    hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
                    hf_filename: "transformer/diffusion_pytorch_model-00003-of-00003.safetensors"
                        .to_string(),
                    component: ModelComponent::TransformerShard,
                    size_bytes: 4_670_000_000, // ~4.67GB
                    gated: false,
                    sha256: Some(
                        "aba4e37a590e63210878160a718d916d80398f4e1f78ab6c9b2b2a00d92769fa",
                    ),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 9,
                guidance: 0.0,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        // GGUF quantized variants (transformer only; shared components are always BF16)
        ModelManifest {
            name: "z-image-turbo:q8".to_string(),
            family: "z-image".to_string(),
            description: "Z-Image Turbo Q8 — GGUF source, dense runtime fallback".to_string(),
            files: {
                let mut files = shared_zimage_files();
                files.push(ModelFile {
                    hf_repo: "leejet/Z-Image-Turbo-GGUF".to_string(),
                    hf_filename: "z_image_turbo-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 6_577_440_704,
                    gated: false,
                    sha256: Some(
                        "df1c5baa86d1398c979495a6072dbcee79444fdb884a2445582ba0769c44e9a1",
                    ),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 9,
                guidance: 0.0,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "z-image-turbo:q6".to_string(),
            family: "z-image".to_string(),
            description: "Z-Image Turbo Q6 — GGUF source, dense runtime fallback".to_string(),
            files: {
                let mut files = shared_zimage_files();
                files.push(ModelFile {
                    hf_repo: "leejet/Z-Image-Turbo-GGUF".to_string(),
                    hf_filename: "z_image_turbo-Q6_K.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 5_263_239_104,
                    gated: false,
                    sha256: Some(
                        "319f627beac8059b7546f36a7b4d5097b7f4ee6a1fc37585d0f75ca1d12d01af",
                    ),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 9,
                guidance: 0.0,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "z-image-turbo:q4".to_string(),
            family: "z-image".to_string(),
            description: "Z-Image Turbo Q4 — GGUF source, dense runtime fallback".to_string(),
            files: {
                let mut files = shared_zimage_files();
                files.push(ModelFile {
                    hf_repo: "leejet/Z-Image-Turbo-GGUF".to_string(),
                    hf_filename: "z_image_turbo-Q4_K.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 3_683_370_944,
                    gated: false,
                    sha256: Some(
                        "14b375ab4f226bc5378f68f37e899ef3c2242b8541e61e2bc1aff40976086fbd",
                    ),
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 9,
                guidance: 0.0,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
    ]
}

/// Shared Flux.2 Klein-4B component files (Qwen3 text encoder, VAE, tokenizer).
///
/// Klein uses Qwen3 (hidden_size=2560, 36 layers) — same model architecture as Z-Image's
/// text encoder. The encoder stacks 3 hidden state outputs to produce joint_attention_dim=7680.
fn shared_flux2_files() -> Vec<ModelFile> {
    vec![
        // Qwen3 text encoder shard 1 (from the Klein repo, 2 shards)
        ModelFile {
            hf_repo: "black-forest-labs/FLUX.2-klein-4B".to_string(),
            hf_filename: "text_encoder/model-00001-of-00002.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 4_967_215_360,
            gated: false,
            sha256: None,
        },
        // Qwen3 text encoder shard 2
        ModelFile {
            hf_repo: "black-forest-labs/FLUX.2-klein-4B".to_string(),
            hf_filename: "text_encoder/model-00002-of-00002.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 3_077_766_632,
            gated: false,
            sha256: None,
        },
        // VAE
        ModelFile {
            hf_repo: "black-forest-labs/FLUX.2-klein-4B".to_string(),
            hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
            component: ModelComponent::Vae,
            size_bytes: 168_120_878,
            gated: false,
            sha256: None,
        },
        // Qwen3 tokenizer
        ModelFile {
            hf_repo: "black-forest-labs/FLUX.2-klein-4B".to_string(),
            hf_filename: "tokenizer/tokenizer.json".to_string(),
            component: ModelComponent::TextTokenizer,
            size_bytes: 11_422_654,
            gated: false,
            sha256: None,
        },
    ]
}

/// Runtime assets for the gated FLUX.2 [dev] checkpoint.
///
/// The official Mistral3 conditioner concatenates hidden states after decoder
/// layers 10, 20, and 30 (zero-based hidden-state indices 10/20/30). Shards
/// 9-10 contain only later decoder layers and the unused language-model head,
/// so downloading them would add bytes that can never affect Mold output.
fn shared_flux2_dev_files() -> Vec<ModelFile> {
    const REPO: &str = "black-forest-labs/FLUX.2-dev";
    const TEXT_SHARDS: [(&str, u64, &str); 8] = [
        (
            "text_encoder/model-00001-of-00010.safetensors",
            4_883_550_696,
            "91831c2ce219df0ce63bc33c6249e5cb01db8d93816bcebf975f1c406286520e",
        ),
        (
            "text_encoder/model-00002-of-00010.safetensors",
            4_781_593_336,
            "8ffe80706a66b2f5ef1fb058806ccf09f124ec4ad38af7a377e44ab1ee2fd664",
        ),
        (
            "text_encoder/model-00003-of-00010.safetensors",
            4_886_472_224,
            "99ec66e891f9563f568734eadfc5b7701e04620e8e163d4d5755277a3b50cf2f",
        ),
        (
            "text_encoder/model-00004-of-00010.safetensors",
            4_781_593_376,
            "e1df1527b12b1eb5cbd9a50914f9e6eb24e885ec830a3c16b5eed6ad0b53a396",
        ),
        (
            "text_encoder/model-00005-of-00010.safetensors",
            4_781_593_368,
            "3556ac03f47c24eb8ad27c237e25baad639c651d9596fd72cb1523137bf56163",
        ),
        (
            "text_encoder/model-00006-of-00010.safetensors",
            4_886_472_248,
            "2c41e6f80f2b5ca384ce703eac048a13daf2aff689c3acca66a8943f45338aae",
        ),
        (
            "text_encoder/model-00007-of-00010.safetensors",
            4_781_593_376,
            "62a725f154f6ba942a36b5cc450db2b2df32f434e3224558c789bc04fa05fd36",
        ),
        (
            "text_encoder/model-00008-of-00010.safetensors",
            4_781_593_368,
            "3a1a6ac77e6434418bb7273b68a7b3534fed5217c990061c92a8f990dd6ab20e",
        ),
    ];

    let mut files = TEXT_SHARDS
        .into_iter()
        .map(|(hf_filename, size_bytes, sha256)| ModelFile {
            hf_repo: REPO.to_string(),
            hf_filename: hf_filename.to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes,
            gated: true,
            sha256: Some(sha256),
        })
        .collect::<Vec<_>>();
    files.extend([
        ModelFile {
            hf_repo: REPO.to_string(),
            hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
            component: ModelComponent::Vae,
            size_bytes: 336_213_556,
            gated: true,
            sha256: Some("d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5"),
        },
        ModelFile {
            hf_repo: REPO.to_string(),
            hf_filename: "tokenizer/tokenizer.json".to_string(),
            component: ModelComponent::TextTokenizer,
            size_bytes: 17_078_037,
            gated: true,
            sha256: Some("b76085f9923309d873994d444989f7eb6ec074b06f25b58f1e8d7b7741070949"),
        },
    ]);
    files
}

/// All known Flux.2 model manifests.
fn flux2_manifests() -> Vec<ModelManifest> {
    let mut manifests = vec![
        // Flux.2 Klein-4B BF16 (Apache 2.0, NOT gated)
        ModelManifest {
            name: "flux2-klein:bf16".to_string(),
            family: "flux2".to_string(),
            description: "Flux.2 Klein-4B BF16 — Apache 2.0, 4B param distilled flow-matching"
                .to_string(),
            files: {
                let mut files = shared_flux2_files();
                files.push(ModelFile {
                    hf_repo: "black-forest-labs/FLUX.2-klein-4B".to_string(),
                    hf_filename: "transformer/diffusion_pytorch_model.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 7_751_109_744,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 4,
                guidance: 1.0,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None, // Uses flow-matching Euler
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        // Flux.2 Klein-4B GGUF quantizations (from unsloth, Apache 2.0)
        ModelManifest {
            name: "flux2-klein:q8".to_string(),
            family: "flux2".to_string(),
            description: "Flux.2 Klein-4B Q8 GGUF — smaller download, reduced quality".to_string(),
            files: {
                let mut files = shared_flux2_files();
                files.push(ModelFile {
                    hf_repo: "unsloth/FLUX.2-klein-4B-GGUF".to_string(),
                    hf_filename: "flux-2-klein-4b-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 4_300_644_928,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 4,
                guidance: 1.0,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "flux2-klein:q6".to_string(),
            family: "flux2".to_string(),
            description: "Flux.2 Klein-4B Q6 GGUF — smaller download, reduced quality".to_string(),
            files: {
                let mut files = shared_flux2_files();
                files.push(ModelFile {
                    hf_repo: "unsloth/FLUX.2-klein-4B-GGUF".to_string(),
                    hf_filename: "flux-2-klein-4b-Q6_K.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 3_409_273_408,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 4,
                guidance: 1.0,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "flux2-klein:q4".to_string(),
            family: "flux2".to_string(),
            description: "Flux.2 Klein-4B Q4 GGUF — smallest download, reduced quality".to_string(),
            files: {
                let mut files = shared_flux2_files();
                files.push(ModelFile {
                    hf_repo: "unsloth/FLUX.2-klein-4B-GGUF".to_string(),
                    hf_filename: "flux-2-klein-4b-Q4_K_M.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 2_604_311_104,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 4,
                guidance: 1.0,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        // ── Flux.2 Klein-9B (distilled, Non-Commercial) ─────────────────────
        // Klein-9B uses Qwen3-8B (hidden_size=4096) vs Klein-4B's Qwen3-4B (2560).
        ModelManifest {
            name: "flux2-klein-9b:bf16".to_string(),
            family: "flux2".to_string(),
            description:
                "Flux.2 Klein-9B BF16 — 9B param distilled, sub-second generation on RTX 4090"
                    .to_string(),
            files: {
                let mut files = shared_flux2_9b_files();
                files.push(ModelFile {
                    hf_repo: "black-forest-labs/FLUX.2-klein-9B".to_string(),
                    hf_filename: "transformer/diffusion_pytorch_model-00001-of-00002.safetensors"
                        .to_string(),
                    component: ModelComponent::TransformerShard,
                    size_bytes: 9_801_069_272,
                    gated: true,
                    sha256: None,
                });
                files.push(ModelFile {
                    hf_repo: "black-forest-labs/FLUX.2-klein-9B".to_string(),
                    hf_filename: "transformer/diffusion_pytorch_model-00002-of-00002.safetensors"
                        .to_string(),
                    component: ModelComponent::TransformerShard,
                    size_bytes: 8_356_121_608,
                    gated: true,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 4,
                guidance: 1.0,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "flux2-klein-9b:q8".to_string(),
            family: "flux2".to_string(),
            description: "Flux.2 Klein-9B Q8 GGUF — best quantized quality".to_string(),
            files: {
                let mut files = shared_flux2_9b_files();
                files.push(ModelFile {
                    hf_repo: "unsloth/FLUX.2-klein-9B-GGUF".to_string(),
                    hf_filename: "flux-2-klein-9b-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 9_978_304_800,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 4,
                guidance: 1.0,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "flux2-klein-9b:q6".to_string(),
            family: "flux2".to_string(),
            description: "Flux.2 Klein-9B Q6 GGUF — good quality/size trade-off".to_string(),
            files: {
                let mut files = shared_flux2_9b_files();
                files.push(ModelFile {
                    hf_repo: "unsloth/FLUX.2-klein-9B-GGUF".to_string(),
                    hf_filename: "flux-2-klein-9b-Q6_K.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 7_865_424_160,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 4,
                guidance: 1.0,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "flux2-klein-9b:q4".to_string(),
            family: "flux2".to_string(),
            description: "Flux.2 Klein-9B Q4 GGUF — smallest footprint".to_string(),
            files: {
                let mut files = shared_flux2_9b_files();
                files.push(ModelFile {
                    hf_repo: "unsloth/FLUX.2-klein-9B-GGUF".to_string(),
                    hf_filename: "flux-2-klein-9b-Q4_K_M.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 5_909_829_920,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 4,
                guidance: 1.0,
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
    ];

    const REPO: &str = "black-forest-labs/FLUX.2-dev";
    const TRANSFORMER_SHARDS: [(&str, u64, &str); 7] = [
        (
            "transformer/diffusion_pytorch_model-00001-of-00007.safetensors",
            9_935_797_200,
            "9d9b85f75f72fb17c7d29dacf7c430e924da93122d578a559a36a7635e153714",
        ),
        (
            "transformer/diffusion_pytorch_model-00002-of-00007.safetensors",
            9_890_181_048,
            "86adf6f41474b00bd57afbb29a09f008be7d6af8ae914956585ba5bc6bf97c28",
        ),
        (
            "transformer/diffusion_pytorch_model-00003-of-00007.safetensors",
            9_814_681_480,
            "a14e26e8f305dd26d7881f333e6e6ce5b562cbb55282538f46d38e1ff2715179",
        ),
        (
            "transformer/diffusion_pytorch_model-00004-of-00007.safetensors",
            9_814_681_536,
            "5c4f38976fd8d7e5fb2d4cd20562d74eebba3264566987e3ef938d807c75be90",
        ),
        (
            "transformer/diffusion_pytorch_model-00005-of-00007.safetensors",
            9_814_681_536,
            "4d7a74d916fc22117cde8bad76aa4b561e6dc92368cddc23bdae06dbc586ad95",
        ),
        (
            "transformer/diffusion_pytorch_model-00006-of-00007.safetensors",
            9_814_681_536,
            "08f3ad03610651f9d630177ac3a4770d532fa72d788d3b36c39a0301b1595447",
        ),
        (
            "transformer/diffusion_pytorch_model-00007-of-00007.safetensors",
            5_361_898_792,
            "789b9bacb607e9b97597f77c86056fa6cbb747c2a6016588e6e196814b5f9733",
        ),
    ];
    let mut files = shared_flux2_dev_files();
    files.extend(
        TRANSFORMER_SHARDS
            .into_iter()
            .map(|(hf_filename, size_bytes, sha256)| ModelFile {
                hf_repo: REPO.to_string(),
                hf_filename: hf_filename.to_string(),
                component: ModelComponent::TransformerShard,
                size_bytes,
                gated: true,
                sha256: Some(sha256),
            }),
    );
    manifests.push(ModelManifest {
        name: "flux2-dev:bf16".to_string(),
        family: "flux2".to_string(),
        description: "FLUX.2 [dev] BF16 — 32B guidance-distilled image generation and editing"
            .to_string(),
        files,
        defaults: ManifestDefaults {
            steps: 50,
            guidance: 4.0,
            width: 1024,
            height: 1024,
            is_schnell: false,
            scheduler: None,
            negative_prompt: None,
            frames: None,
            fps: None,
            source_image: None,
        },
        hidden: false,
    });
    manifests
}

/// Shared Flux.2 Klein-9B component files (Qwen3 text encoder 4 shards, VAE, tokenizer).
///
/// Klein-9B uses a larger Qwen3 encoder (hidden_size=4096, 4 shards) than Klein-4B
/// (hidden_size=2560, 2 shards). The VAE and tokenizer format are the same.
/// The text encoder and tokenizer come from the gated klein-9B repo.
fn shared_flux2_9b_files() -> Vec<ModelFile> {
    vec![
        ModelFile {
            hf_repo: "black-forest-labs/FLUX.2-klein-9B".to_string(),
            hf_filename: "text_encoder/model-00001-of-00004.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 4_902_257_696,
            gated: true,
            sha256: None,
        },
        ModelFile {
            hf_repo: "black-forest-labs/FLUX.2-klein-9B".to_string(),
            hf_filename: "text_encoder/model-00002-of-00004.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 4_915_960_368,
            gated: true,
            sha256: None,
        },
        ModelFile {
            hf_repo: "black-forest-labs/FLUX.2-klein-9B".to_string(),
            hf_filename: "text_encoder/model-00003-of-00004.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 4_983_068_496,
            gated: true,
            sha256: None,
        },
        ModelFile {
            hf_repo: "black-forest-labs/FLUX.2-klein-9B".to_string(),
            hf_filename: "text_encoder/model-00004-of-00004.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 1_580_230_264,
            gated: true,
            sha256: None,
        },
        ModelFile {
            hf_repo: "black-forest-labs/FLUX.2-klein-9B".to_string(),
            hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
            component: ModelComponent::Vae,
            size_bytes: 168_120_878,
            gated: true,
            sha256: None,
        },
        ModelFile {
            hf_repo: "black-forest-labs/FLUX.2-klein-9B".to_string(),
            hf_filename: "tokenizer/tokenizer.json".to_string(),
            component: ModelComponent::TextTokenizer,
            size_bytes: 11_422_654,
            gated: true,
            sha256: None,
        },
    ]
}

/// Shared base Qwen-Image component files (VAE, text encoder shards, tokenizer).
fn shared_qwen_image_base_files() -> Vec<ModelFile> {
    vec![
        // VAE
        ModelFile {
            hf_repo: "Qwen/Qwen-Image".to_string(),
            hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
            component: ModelComponent::Vae,
            size_bytes: 253_806_966,
            gated: false,
            sha256: None,
        },
        // Qwen2.5-VL text encoder shards
        ModelFile {
            hf_repo: "Qwen/Qwen-Image".to_string(),
            hf_filename: "text_encoder/model-00001-of-00004.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 4_968_243_304,
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "Qwen/Qwen-Image".to_string(),
            hf_filename: "text_encoder/model-00002-of-00004.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 4_991_495_816,
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "Qwen/Qwen-Image".to_string(),
            hf_filename: "text_encoder/model-00003-of-00004.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 4_932_751_040,
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "Qwen/Qwen-Image".to_string(),
            hf_filename: "text_encoder/model-00004-of-00004.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 1_691_924_384,
            gated: false,
            sha256: None,
        },
        // Tokenizer shared across both Qwen-Image releases.
        ModelFile {
            hf_repo: "Qwen/Qwen2.5-7B".to_string(),
            hf_filename: "tokenizer.json".to_string(),
            component: ModelComponent::TextTokenizer,
            size_bytes: 7_031_645,
            gated: false,
            sha256: None,
        },
    ]
}

/// Shared Qwen-Image-2512 component files (VAE, text encoder shards, tokenizer).
fn shared_qwen_image_2512_files() -> Vec<ModelFile> {
    vec![
        ModelFile {
            hf_repo: "Qwen/Qwen-Image-2512".to_string(),
            hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
            component: ModelComponent::Vae,
            size_bytes: 253_806_966,
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "Qwen/Qwen-Image-2512".to_string(),
            hf_filename: "text_encoder/model-00001-of-00004.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 4_968_243_304,
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "Qwen/Qwen-Image-2512".to_string(),
            hf_filename: "text_encoder/model-00002-of-00004.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 4_991_495_816,
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "Qwen/Qwen-Image-2512".to_string(),
            hf_filename: "text_encoder/model-00003-of-00004.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 4_932_751_040,
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "Qwen/Qwen-Image-2512".to_string(),
            hf_filename: "text_encoder/model-00004-of-00004.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 1_691_924_384,
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "Qwen/Qwen2.5-7B".to_string(),
            hf_filename: "tokenizer.json".to_string(),
            component: ModelComponent::TextTokenizer,
            size_bytes: 7_031_645,
            gated: false,
            sha256: None,
        },
    ]
}

/// Shared Qwen-Image-Edit-2511 component files (VAE, text encoder shards, tokenizer).
fn shared_qwen_image_edit_files() -> Vec<ModelFile> {
    vec![
        ModelFile {
            hf_repo: "Qwen/Qwen-Image-Edit-2511".to_string(),
            hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
            component: ModelComponent::Vae,
            size_bytes: 253_806_966,
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "Qwen/Qwen-Image-Edit-2511".to_string(),
            hf_filename: "text_encoder/model-00001-of-00004.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 4_968_243_304,
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "Qwen/Qwen-Image-Edit-2511".to_string(),
            hf_filename: "text_encoder/model-00002-of-00004.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 4_991_495_816,
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "Qwen/Qwen-Image-Edit-2511".to_string(),
            hf_filename: "text_encoder/model-00003-of-00004.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 4_932_751_040,
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "Qwen/Qwen-Image-Edit-2511".to_string(),
            hf_filename: "text_encoder/model-00004-of-00004.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 1_691_924_384,
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "Qwen/Qwen2.5-7B".to_string(),
            hf_filename: "tokenizer.json".to_string(),
            component: ModelComponent::TextTokenizer,
            size_bytes: 7_031_645,
            gated: false,
            sha256: None,
        },
    ]
}

/// All known Qwen-Image model manifests.
fn qwen_image_manifests() -> Vec<ModelManifest> {
    let base_defaults = ManifestDefaults {
        steps: 50,
        guidance: 4.0,
        width: 1328,
        height: 1328,
        is_schnell: false,
        scheduler: None,
        negative_prompt: None,
        frames: None,
        fps: None,
        source_image: None,
    };
    let qwen_2512_defaults = base_defaults.clone();
    let qwen_edit_defaults = ManifestDefaults {
        steps: 50,
        guidance: 4.0,
        width: 1024,
        height: 1024,
        is_schnell: false,
        scheduler: None,
        negative_prompt: None,
        frames: None,
        fps: None,
        source_image: None,
    };

    vec![
        // Base Qwen-Image.
        ModelManifest {
            name: "qwen-image:bf16".to_string(),
            family: "qwen-image".to_string(),
            description: "Qwen-Image BF16 — base model, 60-block flow-matching transformer"
                .to_string(),
            files: {
                let mut files = shared_qwen_image_base_files();
                let shards: &[(&str, u64)] = &[
                    (
                        "transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
                        4_989_364_312,
                    ),
                    (
                        "transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
                        4_984_214_160,
                    ),
                    (
                        "transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
                        4_946_470_000,
                    ),
                    (
                        "transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
                        4_984_213_736,
                    ),
                    (
                        "transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
                        4_946_471_896,
                    ),
                    (
                        "transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
                        4_946_451_560,
                    ),
                    (
                        "transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
                        4_908_690_520,
                    ),
                    (
                        "transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
                        4_984_232_856,
                    ),
                    (
                        "transformer/diffusion_pytorch_model-00009-of-00009.safetensors",
                        1_170_918_840,
                    ),
                ];
                for (filename, size) in shards {
                    files.push(ModelFile {
                        hf_repo: "Qwen/Qwen-Image".to_string(),
                        hf_filename: filename.to_string(),
                        component: ModelComponent::TransformerShard,
                        size_bytes: *size,
                        gated: false,
                        sha256: None,
                    });
                }
                files
            },
            defaults: base_defaults.clone(),
            hidden: false,
        },
        // Base Qwen-Image FP8 E4M3 (ComfyUI-compatible transformer, BF16 text encoder)
        // NOTE: The FP8 text encoder (qwen_2.5_vl_7b_fp8_scaled.safetensors)
        // requires scale_input/scale_weight dequantization that candle doesn't
        // support. We use the BF16 text encoder shared with GGUF variants instead.
        ModelManifest {
            name: "qwen-image:fp8".to_string(),
            family: "qwen-image".to_string(),
            description: "Qwen-Image FP8 — base model, ComfyUI-compatible transformer".to_string(),
            files: {
                let mut files = shared_qwen_image_base_files();
                files.push(ModelFile {
                    hf_repo: "Comfy-Org/Qwen-Image_ComfyUI".to_string(),
                    hf_filename: "split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors"
                        .to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 20_442_787_688,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: base_defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "qwen-image:q8".to_string(),
            family: "qwen-image".to_string(),
            description: "Qwen-Image Q8 — base model quantized transformer, best quality"
                .to_string(),
            files: {
                let mut files = shared_qwen_image_base_files();
                files.push(ModelFile {
                    hf_repo: "city96/Qwen-Image-gguf".to_string(),
                    hf_filename: "qwen-image-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 21_761_817_120,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: base_defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "qwen-image:q6".to_string(),
            family: "qwen-image".to_string(),
            description: "Qwen-Image Q6 — base model quantized, quality/size trade-off".to_string(),
            files: {
                let mut files = shared_qwen_image_base_files();
                files.push(ModelFile {
                    hf_repo: "city96/Qwen-Image-gguf".to_string(),
                    hf_filename: "qwen-image-Q6_K.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 16_824_990_240,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: base_defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "qwen-image:q5".to_string(),
            family: "qwen-image".to_string(),
            description: "Qwen-Image Q5 — base model quantized, dynamic K_M variant".to_string(),
            files: {
                let mut files = shared_qwen_image_base_files();
                files.push(ModelFile {
                    hf_repo: "city96/Qwen-Image-gguf".to_string(),
                    hf_filename: "qwen-image-Q5_K_M.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 14_934_899_232,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: base_defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "qwen-image:q4".to_string(),
            family: "qwen-image".to_string(),
            description: "Qwen-Image Q4 — base model quantized, dynamic K_M variant".to_string(),
            files: {
                let mut files = shared_qwen_image_base_files();
                files.push(ModelFile {
                    hf_repo: "city96/Qwen-Image-gguf".to_string(),
                    hf_filename: "qwen-image-Q4_K_M.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 13_065_746_976,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: base_defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "qwen-image:q3".to_string(),
            family: "qwen-image".to_string(),
            description: "Qwen-Image Q3 — base model quantized, dynamic K_M variant".to_string(),
            files: {
                let mut files = shared_qwen_image_base_files();
                files.push(ModelFile {
                    hf_repo: "city96/Qwen-Image-gguf".to_string(),
                    hf_filename: "qwen-image-Q3_K_M.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 9_679_567_392,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: base_defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "qwen-image:q2".to_string(),
            family: "qwen-image".to_string(),
            description: "Qwen-Image Q2 — base model quantized, smallest published K variant"
                .to_string(),
            files: {
                let mut files = shared_qwen_image_base_files();
                files.push(ModelFile {
                    hf_repo: "city96/Qwen-Image-gguf".to_string(),
                    hf_filename: "qwen-image-Q2_K.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 7_062_518_304,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: base_defaults.clone(),
            hidden: false,
        },
        // Qwen-Image-2512.
        ModelManifest {
            name: "qwen-image-2512:bf16".to_string(),
            family: "qwen-image".to_string(),
            description: "Qwen-Image-2512 BF16 — December update, strongest quality".to_string(),
            files: {
                let mut files = shared_qwen_image_2512_files();
                let shards: &[(&str, u64)] = &[
                    (
                        "transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
                        4_989_364_312,
                    ),
                    (
                        "transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
                        4_984_214_160,
                    ),
                    (
                        "transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
                        4_946_470_000,
                    ),
                    (
                        "transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
                        4_984_213_736,
                    ),
                    (
                        "transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
                        4_946_471_896,
                    ),
                    (
                        "transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
                        4_946_451_560,
                    ),
                    (
                        "transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
                        4_908_690_520,
                    ),
                    (
                        "transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
                        4_984_232_856,
                    ),
                    (
                        "transformer/diffusion_pytorch_model-00009-of-00009.safetensors",
                        1_170_918_840,
                    ),
                ];
                for (filename, size) in shards {
                    files.push(ModelFile {
                        hf_repo: "Qwen/Qwen-Image-2512".to_string(),
                        hf_filename: filename.to_string(),
                        component: ModelComponent::TransformerShard,
                        size_bytes: *size,
                        gated: false,
                        sha256: None,
                    });
                }
                files
            },
            defaults: qwen_2512_defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "qwen-image-2512:q8".to_string(),
            family: "qwen-image".to_string(),
            description: "Qwen-Image-2512 Q8 — Unsloth GGUF, best quality".to_string(),
            files: {
                let mut files = shared_qwen_image_2512_files();
                files.push(ModelFile {
                    hf_repo: "unsloth/Qwen-Image-2512-GGUF".to_string(),
                    hf_filename: "qwen-image-2512-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 21_761_817_120,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: qwen_2512_defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "qwen-image-2512:q6".to_string(),
            family: "qwen-image".to_string(),
            description: "Qwen-Image-2512 Q6 — Unsloth GGUF, quality/size trade-off".to_string(),
            files: {
                let mut files = shared_qwen_image_2512_files();
                files.push(ModelFile {
                    hf_repo: "unsloth/Qwen-Image-2512-GGUF".to_string(),
                    hf_filename: "qwen-image-2512-Q6_K.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 16_824_990_240,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: qwen_2512_defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "qwen-image-2512:q5".to_string(),
            family: "qwen-image".to_string(),
            description: "Qwen-Image-2512 Q5 — Unsloth dynamic K_M GGUF".to_string(),
            files: {
                let mut files = shared_qwen_image_2512_files();
                files.push(ModelFile {
                    hf_repo: "unsloth/Qwen-Image-2512-GGUF".to_string(),
                    hf_filename: "qwen-image-2512-Q5_K_M.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 15_000_074_784,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: qwen_2512_defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "qwen-image-2512:q4".to_string(),
            family: "qwen-image".to_string(),
            description: "Qwen-Image-2512 Q4 — Unsloth dynamic K_M GGUF".to_string(),
            files: {
                let mut files = shared_qwen_image_2512_files();
                files.push(ModelFile {
                    hf_repo: "unsloth/Qwen-Image-2512-GGUF".to_string(),
                    hf_filename: "qwen-image-2512-Q4_K_M.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 13_244_758_560,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: qwen_2512_defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "qwen-image-2512:q3".to_string(),
            family: "qwen-image".to_string(),
            description: "Qwen-Image-2512 Q3 — Unsloth dynamic K_M GGUF".to_string(),
            files: {
                let mut files = shared_qwen_image_2512_files();
                files.push(ModelFile {
                    hf_repo: "unsloth/Qwen-Image-2512-GGUF".to_string(),
                    hf_filename: "qwen-image-2512-Q3_K_M.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 9_932_896_800,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: qwen_2512_defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "qwen-image-2512:q2".to_string(),
            family: "qwen-image".to_string(),
            description: "Qwen-Image-2512 Q2 — Unsloth smallest published K variant".to_string(),
            files: {
                let mut files = shared_qwen_image_2512_files();
                files.push(ModelFile {
                    hf_repo: "unsloth/Qwen-Image-2512-GGUF".to_string(),
                    hf_filename: "qwen-image-2512-Q2_K.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 7_333_837_344,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: qwen_2512_defaults.clone(),
            hidden: false,
        },
        // Lightning distilled variants (step-distilled, no CFG needed)
        ModelManifest {
            name: "qwen-image-lightning:fp8".to_string(),
            family: "qwen-image".to_string(),
            description: "Qwen-Image-2512 Lightning FP8 — 4-step distilled, 12-25x faster"
                .to_string(),
            files: {
                let mut files = shared_qwen_image_2512_files();
                files.push(ModelFile {
                    hf_repo: "lightx2v/Qwen-Image-2512-Lightning".to_string(),
                    hf_filename:
                        "qwen_image_2512_fp8_e4m3fn_scaled_comfyui_4steps_v1.0.safetensors"
                            .to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 20_400_000_000,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 4,
                guidance: 1.0,
                width: 1328,
                height: 1328,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "qwen-image-lightning:fp8-8step".to_string(),
            family: "qwen-image".to_string(),
            description: "Qwen-Image-2512 Lightning FP8 — 8-step distilled, higher quality"
                .to_string(),
            files: {
                let mut files = shared_qwen_image_2512_files();
                files.push(ModelFile {
                    hf_repo: "lightx2v/Qwen-Image-2512-Lightning".to_string(),
                    hf_filename: "qwen_image_2512_fp8_e4m3fn_scaled_8steps_v1.0.safetensors"
                        .to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 20_400_000_000,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 8,
                guidance: 1.0,
                width: 1328,
                height: 1328,
                is_schnell: false,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: false,
        },
        ModelManifest {
            name: "qwen-image-edit-2511:bf16".to_string(),
            family: "qwen-image-edit".to_string(),
            description: "Qwen-Image-Edit-2511 BF16 — multimodal image editing".to_string(),
            files: {
                let mut files = shared_qwen_image_edit_files();
                let shards: &[(&str, u64)] = &[
                    (
                        "transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
                        9_973_578_592,
                    ),
                    (
                        "transformer/diffusion_pytorch_model-00002-of-00005.safetensors",
                        9_987_326_072,
                    ),
                    (
                        "transformer/diffusion_pytorch_model-00003-of-00005.safetensors",
                        9_987_307_440,
                    ),
                    (
                        "transformer/diffusion_pytorch_model-00004-of-00005.safetensors",
                        9_930_685_712,
                    ),
                    (
                        "transformer/diffusion_pytorch_model-00005-of-00005.safetensors",
                        982_130_472,
                    ),
                ];
                for (filename, size) in shards {
                    files.push(ModelFile {
                        hf_repo: "Qwen/Qwen-Image-Edit-2511".to_string(),
                        hf_filename: filename.to_string(),
                        component: ModelComponent::TransformerShard,
                        size_bytes: *size,
                        gated: false,
                        sha256: None,
                    });
                }
                files
            },
            defaults: qwen_edit_defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "qwen-image-edit-2511:q8".to_string(),
            family: "qwen-image-edit".to_string(),
            description: "Qwen-Image-Edit-2511 Q8 — multimodal image editing, best GGUF quality"
                .to_string(),
            files: {
                let mut files = shared_qwen_image_edit_files();
                files.push(ModelFile {
                    hf_repo: "unsloth/Qwen-Image-Edit-2511-GGUF".to_string(),
                    hf_filename: "qwen-image-edit-2511-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 21_761_817_184,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: qwen_edit_defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "qwen-image-edit-2511:q6".to_string(),
            family: "qwen-image-edit".to_string(),
            description:
                "Qwen-Image-Edit-2511 Q6 — multimodal image editing, quality/size trade-off"
                    .to_string(),
            files: {
                let mut files = shared_qwen_image_edit_files();
                files.push(ModelFile {
                    hf_repo: "unsloth/Qwen-Image-Edit-2511-GGUF".to_string(),
                    hf_filename: "qwen-image-edit-2511-Q6_K.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 16_852_417_120,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: qwen_edit_defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "qwen-image-edit-2511:q5".to_string(),
            family: "qwen-image-edit".to_string(),
            description: "Qwen-Image-Edit-2511 Q5 — multimodal image editing, dynamic K_M GGUF"
                .to_string(),
            files: {
                let mut files = shared_qwen_image_edit_files();
                files.push(ModelFile {
                    hf_repo: "unsloth/Qwen-Image-Edit-2511-GGUF".to_string(),
                    hf_filename: "qwen-image-edit-2511-Q5_K_M.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 15_027_501_664,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: qwen_edit_defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "qwen-image-edit-2511:q4".to_string(),
            family: "qwen-image-edit".to_string(),
            description: "Qwen-Image-Edit-2511 Q4 — multimodal image editing, dynamic K_M GGUF"
                .to_string(),
            files: {
                let mut files = shared_qwen_image_edit_files();
                files.push(ModelFile {
                    hf_repo: "unsloth/Qwen-Image-Edit-2511-GGUF".to_string(),
                    hf_filename: "qwen-image-edit-2511-Q4_K_M.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 13_244_758_624,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: qwen_edit_defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "qwen-image-edit-2511:q3".to_string(),
            family: "qwen-image-edit".to_string(),
            description: "Qwen-Image-Edit-2511 Q3 — multimodal image editing, dynamic K_M GGUF"
                .to_string(),
            files: {
                let mut files = shared_qwen_image_edit_files();
                files.push(ModelFile {
                    hf_repo: "unsloth/Qwen-Image-Edit-2511-GGUF".to_string(),
                    hf_filename: "qwen-image-edit-2511-Q3_K_M.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 9_920_805_472,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: qwen_edit_defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "qwen-image-edit-2511:q2".to_string(),
            family: "qwen-image-edit".to_string(),
            description:
                "Qwen-Image-Edit-2511 Q2 — multimodal image editing, smallest published K variant"
                    .to_string(),
            files: {
                let mut files = shared_qwen_image_edit_files();
                files.push(ModelFile {
                    hf_repo: "unsloth/Qwen-Image-Edit-2511-GGUF".to_string(),
                    hf_filename: "qwen-image-edit-2511-Q2_K.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 7_468_022_368,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: qwen_edit_defaults.clone(),
            hidden: false,
        },
    ]
}

fn wuerstchen_manifests() -> Vec<ModelManifest> {
    let defaults = ManifestDefaults {
        steps: 30,
        guidance: 4.0,
        width: 1024,
        height: 1024,
        is_schnell: false,
        scheduler: None,
        negative_prompt: Some(
            "low quality, blurry, distorted, deformed, disfigured, bad anatomy, watermark"
                .to_string(),
        ),
        frames: None,
        fps: None,
        source_image: None,
    };
    vec![ModelManifest {
        name: "wuerstchen-v2:fp16".to_string(),
        family: "wuerstchen".to_string(),
        description: "Wuerstchen v2 FP16 — research model, 3-stage cascade with 42x latent compression, painterly style".to_string(),
        files: vec![
            ModelFile {
                hf_repo: "warp-ai/wuerstchen".to_string(),
                hf_filename: "decoder/diffusion_pytorch_model.safetensors".to_string(),
                component: ModelComponent::Decoder,
                size_bytes: 4_221_568_336,
                gated: false,
                sha256: None,
            },
            ModelFile {
                hf_repo: "warp-ai/wuerstchen".to_string(),
                hf_filename: "vqgan/diffusion_pytorch_model.safetensors".to_string(),
                component: ModelComponent::Vae,
                size_bytes: 73_639_568,
                gated: false,
                sha256: None,
            },
            ModelFile {
                hf_repo: "warp-ai/wuerstchen-prior".to_string(),
                hf_filename: "prior/diffusion_pytorch_model.safetensors".to_string(),
                component: ModelComponent::Transformer,
                size_bytes: 3_974_614_642,
                gated: false,
                sha256: None,
            },
            // Prior CLIP-G encoder (1280-dim, for Stage C)
            ModelFile {
                hf_repo: "warp-ai/wuerstchen-prior".to_string(),
                hf_filename: "text_encoder/model.safetensors".to_string(),
                component: ModelComponent::ClipEncoder2,
                size_bytes: 2_772_149_276,
                gated: false,
                sha256: None,
            },
            ModelFile {
                hf_repo: "warp-ai/wuerstchen-prior".to_string(),
                hf_filename: "tokenizer/tokenizer.json".to_string(),
                component: ModelComponent::ClipTokenizer2,
                size_bytes: 2_224_091,
                gated: false,
                sha256: None,
            },
            // Decoder CLIP encoder (1024-dim, for Stage B)
            // Uses separate TextEncoder component to get model-specific path
            ModelFile {
                hf_repo: "warp-ai/wuerstchen".to_string(),
                hf_filename: "text_encoder/model.safetensors".to_string(),
                component: ModelComponent::ClipEncoder,
                size_bytes: 1_411_983_168,
                gated: false,
                sha256: None,
            },
            ModelFile {
                hf_repo: "warp-ai/wuerstchen".to_string(),
                hf_filename: "tokenizer/tokenizer.json".to_string(),
                component: ModelComponent::ClipTokenizer,
                size_bytes: 2_224_119,
                gated: false,
                sha256: None,
            },
        ],
        defaults,
        hidden: false,
    }]
}

/// Resolve a user-provided model name to its canonical `name:tag` form.
///
/// - `flux-schnell` → `flux-schnell:q8` (FLUX default tag)
/// - `dreamshaper-xl` → `dreamshaper-xl:fp16` (SDXL default tag)
/// - `flux-dev:q4` → `flux-dev:q4` (unchanged)
/// - `flux-dev-q4` → `flux-dev:q4` (legacy format)
pub fn resolve_model_name(input: &str) -> String {
    if let Some(name) = crate::minimax_h3::resolve_model_name(input) {
        return name.to_string();
    }
    // Already has a tag
    if input.contains(':') {
        return input.to_string();
    }
    // Bare-name match — companion manifests ("clip-l", "sdxl-vae", ...)
    // register without a `:tag` because there's only one variant. Catch
    // them here so the tag-resolution loop below doesn't synthesize a
    // bogus `clip-l:q8` and miss the real entry.
    if find_manifest_exact(input).is_some() {
        return input.to_string();
    }
    if input == "qwen-image-edit" {
        return "qwen-image-edit-2511:q8".to_string();
    }
    // These high-memory BF16 variants are opt-in. Bare LTX-2.3 22B names
    // continue to resolve to the smaller FP8 checkpoints.
    if matches!(input, "ltx-2.3-22b-dev" | "ltx-2.3-22b-distilled") {
        return format!("{input}:fp8");
    }
    // The bare 5B name stays on the fp16 safetensors default. The tag loop
    // below tries `:q8` before `:fp16`, so without this pin the small-card
    // GGUF tag added by #794 would silently become the family default.
    if input == "wan22-ti2v-5b" {
        return format!("{input}:fp16");
    }
    // Legacy format: flux-dev-q4 -> flux-dev:q4 and
    // ltx-2.3-22b-dev-fp8 -> ltx-2.3-22b-dev:fp8.
    if let Some((base, suffix)) = input.rsplit_once('-') {
        let legacy_quant = (suffix.starts_with('q')
            && suffix.len() <= 3
            && suffix[1..].chars().all(|c| c.is_ascii_digit()))
            || matches!(suffix, "fp8" | "fp16" | "bf16");
        if legacy_quant {
            return format!("{base}:{suffix}");
        }
    }
    // Try default tags in preference order: :q8 (GGUF, smaller), :fp16 (SDXL), :bf16, :fp8 (community)
    for tag in ["q8", "fp16", "bf16", "fp8"] {
        let candidate = format!("{input}:{tag}");
        if find_manifest_exact(&candidate).is_some() {
            return candidate;
        }
    }
    // Fallback to :q8 for backward compatibility
    format!("{input}:q8")
}

/// Find a manifest by exact name (no resolution). Used internally to avoid
/// circular dependency in `resolve_model_name`.
fn find_manifest_exact(name: &str) -> Option<&'static ModelManifest> {
    MANIFEST_INDEX.get(name).map(|&i| &KNOWN_MANIFESTS[i])
}

/// Find a manifest by name, handling tag resolution and legacy names.
pub fn find_manifest(name: &str) -> Option<&'static ModelManifest> {
    let canonical = resolve_model_name(name);
    MANIFEST_INDEX.get(&canonical).map(|&i| &KNOWN_MANIFESTS[i])
}

/// Find a manifest whose Transformer (or TransformerShard) file points
/// at the given Hugging Face repo (e.g. `"black-forest-labs/FLUX.1-dev"`).
/// Used by the catalog API to map HF catalog rows back to a known model
/// name when computing the `installed` wire field — `find_manifest`
/// itself is keyed on canonical names like `"flux-dev:q8"`, so HF repo
/// paths never resolve through it directly.
///
/// Returns the FIRST matching manifest in `KNOWN_MANIFESTS` order. When
/// multiple quantizations of the same model share an HF repo
/// (e.g. several `flux-dev:qN` variants from `city96/FLUX.1-dev-gguf`),
/// the caller may not get the variant the user actually has on disk —
/// that's fine, because `manifest_model_is_downloaded` then queries
/// the resolved canonical name and returns `false` if the user has
/// no copy of that specific variant.
pub fn find_manifest_by_hf_repo(hf_repo: &str) -> Option<&'static ModelManifest> {
    KNOWN_MANIFESTS.iter().find(|m| {
        m.files.iter().any(|f| {
            matches!(
                f.component,
                ModelComponent::Transformer | ModelComponent::TransformerShard
            ) && f.hf_repo == hf_repo
        })
    })
}

/// Find smaller quantized alternatives for a model.
/// Given `"ultrareal-v2:bf16"`, returns `["ultrareal-v3:q4", "ultrareal-v3:q8", ...]`
/// sorted by model size ascending. Returns empty if no alternatives found.
pub fn find_smaller_alternatives(name: &str) -> Vec<String> {
    let canonical = resolve_model_name(name);
    let current = find_manifest(&canonical);
    let current_size = current.map(|m| m.model_size_bytes()).unwrap_or(u64::MAX);

    // Extract base name (everything before first colon, minus version suffix like -v2)
    let base = canonical.split(':').next().unwrap_or(&canonical);
    // Also try the family prefix (e.g. "ultrareal" from "ultrareal-v2")
    let family_prefix = base
        .rfind("-v")
        .or_else(|| base.rfind("-V"))
        .map(|i| &base[..i])
        .unwrap_or(base);

    let mut alternatives: Vec<(u64, String)> = known_manifests()
        .iter()
        .filter(|m| {
            m.name != canonical
                && (m.name.starts_with(&format!("{base}:"))
                    || m.name.starts_with(&format!("{family_prefix}-")))
                && m.model_size_bytes() < current_size
        })
        .map(|m| (m.model_size_bytes(), m.name.clone()))
        .collect();

    alternatives.sort();
    alternatives.into_iter().map(|(_, name)| name).collect()
}

/// Check if a name resolves to a known model (manifest or config).
pub fn is_known_model(name: &str, config: &crate::Config) -> bool {
    let canonical = resolve_model_name(name);
    config.models.contains_key(name)
        || config.models.contains_key(&canonical)
        || find_manifest(&canonical).is_some()
}

/// All known model names (manifests + config), deduplicated and sorted.
pub fn all_model_names(config: &crate::Config) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    for m in known_manifests() {
        seen.insert(m.name.clone());
    }
    for key in config.models.keys() {
        seen.insert(key.clone());
    }
    let mut names: Vec<String> = seen.into_iter().collect();
    names.sort();
    names
}

/// True if a family string identifies a generation model (not upscaler, utility, or auxiliary).
///
/// Used by `all_generation_model_names` to classify config-only models whose family
/// is a plain string rather than a `ModelManifest` with methods.
pub fn is_generation_family(family: &str) -> bool {
    !UPSCALER_FAMILIES.contains(&family)
        && !UTILITY_FAMILIES.contains(&family)
        && !AUXILIARY_FAMILIES.contains(&family)
}

/// All known generation model names (excludes upscalers, utility, and auxiliary models),
/// deduplicated and sorted.
pub fn all_generation_model_names(config: &crate::Config) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    for m in known_manifests() {
        if m.is_generation_model() {
            seen.insert(m.name.clone());
        }
    }
    for key in config.models.keys() {
        // Use resolved config to get the correct family (inherits from manifest if present).
        let resolved = config.resolved_model_config(key);
        let family = resolved.family.as_deref().unwrap_or("flux");
        if is_generation_family(family) {
            seen.insert(key.clone());
        }
    }
    let mut names: Vec<String> = seen.into_iter().collect();
    names.sort();
    names
}

/// Check if a string structurally resembles a model name without being a known one.
///
/// Returns true if the input contains explicit tag syntax (colon), shares a family
/// prefix with a known model, or has high string similarity to any known model base name.
pub fn looks_like_model_name(input: &str, config: &crate::Config) -> bool {
    // Explicit tag syntax is always model-like
    if input.contains(':') {
        return true;
    }

    // After the colon early-return above, input is guaranteed colon-free
    let input_base = input;

    // Extract family from input by stripping version suffix (e.g. "ultrareal-v8" → "ultrareal")
    let input_family = input_base
        .rfind("-v")
        .or_else(|| input_base.rfind("-V"))
        .and_then(|i| {
            let suffix = &input_base[i + 2..];
            if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit() || c == '.') {
                Some(&input_base[..i])
            } else {
                None
            }
        });

    // Check if input's family prefix matches any known model family
    if let Some(family) = input_family {
        for m in known_manifests() {
            if m.family == family {
                return true;
            }
        }
        // Also check config model base names for the same family prefix
        for key in config.models.keys() {
            let key_base = key.split(':').next().unwrap_or(key);
            let key_family = key_base
                .rfind("-v")
                .or_else(|| key_base.rfind("-V"))
                .and_then(|i| {
                    let s = &key_base[i + 2..];
                    if !s.is_empty() && s.chars().all(|c| c.is_ascii_digit() || c == '.') {
                        Some(&key_base[..i])
                    } else {
                        None
                    }
                })
                .unwrap_or(key_base);
            if key_family == family {
                return true;
            }
        }
    }

    // Fuzzy match: check if any known model base name is very similar.
    // Note: all_model_names() rebuilds the list each call; this is fine since
    // looks_like_model_name runs at most once per CLI invocation on the error path.
    for name in all_model_names(config) {
        let base = name.split(':').next().unwrap_or(&name);
        if strsim::jaro_winkler(input_base, base) >= 0.75 {
            return true;
        }
    }

    false
}

/// Suggest similar model names for a given input, ranked by similarity.
/// Returns up to `max` suggestions.
pub fn suggest_similar_models(input: &str, config: &crate::Config, max: usize) -> Vec<String> {
    let input_base = input.split(':').next().unwrap_or(input);

    // all_generation_model_names already deduplicates via HashSet, so no explicit dedup needed
    let mut scored: Vec<(f64, String)> = all_generation_model_names(config)
        .into_iter()
        .map(|name| {
            let base = name.split(':').next().unwrap_or(&name);
            let sim = strsim::jaro_winkler(input_base, base);
            (sim, name)
        })
        .filter(|(sim, _)| *sim > 0.6)
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(max).map(|(_, name)| name).collect()
}

/// FP16 T5-XXL model size in bytes.
pub const T5_FP16_SIZE: u64 = 9_787_841_024;

/// BF16 Qwen3-4B text encoder size in bytes (3 safetensors shards).
pub const QWEN3_FP16_SIZE: u64 = 8_044_982_000;

/// BF16 Qwen3-8B text encoder size in bytes (4 safetensors shards, Klein-9B).
pub const QWEN3_8B_FP16_SIZE: u64 = 16_388_044_384;

// ── Quantized T5 variant registry ────────────────────────────────────────────

/// A quantized T5 encoder variant available from HuggingFace.
#[derive(Debug, Clone)]
pub struct T5Variant {
    pub tag: &'static str,
    pub hf_repo: &'static str,
    pub hf_filename: &'static str,
    pub size_bytes: u64,
}

/// Known T5 quantized variants, sorted largest → smallest.
pub fn known_t5_variants() -> &'static [T5Variant] {
    static VARIANTS: &[T5Variant] = &[
        T5Variant {
            tag: "q8",
            hf_repo: "city96/t5-v1_1-xxl-encoder-gguf",
            hf_filename: "t5-v1_1-xxl-encoder-Q8_0.gguf",
            size_bytes: 5_061_584_064,
        },
        T5Variant {
            tag: "q6",
            hf_repo: "city96/t5-v1_1-xxl-encoder-gguf",
            hf_filename: "t5-v1_1-xxl-encoder-Q6_K.gguf",
            size_bytes: 3_908_261_056,
        },
        T5Variant {
            tag: "q5",
            hf_repo: "city96/t5-v1_1-xxl-encoder-gguf",
            hf_filename: "t5-v1_1-xxl-encoder-Q5_K_M.gguf",
            size_bytes: 3_386_856_640,
        },
        T5Variant {
            tag: "q4",
            hf_repo: "city96/t5-v1_1-xxl-encoder-gguf",
            hf_filename: "t5-v1_1-xxl-encoder-Q4_K_M.gguf",
            size_bytes: 2_896_123_072,
        },
        T5Variant {
            tag: "q3",
            hf_repo: "city96/t5-v1_1-xxl-encoder-gguf",
            hf_filename: "t5-v1_1-xxl-encoder-Q3_K_S.gguf",
            size_bytes: 2_099_467_456,
        },
    ];
    VARIANTS
}

/// Find a T5 variant by tag (e.g. "q8", "q5").
pub fn find_t5_variant(tag: &str) -> Option<&'static T5Variant> {
    known_t5_variants().iter().find(|v| v.tag == tag)
}

// ── Quantized Qwen3 variant registry ──────────────────────────────────────────

/// A quantized Qwen3 text encoder variant available from HuggingFace.
#[derive(Debug, Clone)]
pub struct Qwen3Variant {
    pub tag: &'static str,
    pub hf_repo: &'static str,
    pub hf_filename: &'static str,
    pub size_bytes: u64,
}

/// A quantized Qwen2.5-VL text encoder variant available from HuggingFace.
#[derive(Debug, Clone)]
pub struct Qwen2VlVariant {
    pub tag: &'static str,
    pub hf_repo: &'static str,
    pub hf_filename: &'static str,
    pub size_bytes: u64,
}

/// Known Qwen3 quantized variants, sorted largest → smallest.
pub fn known_qwen3_variants() -> &'static [Qwen3Variant] {
    static VARIANTS: &[Qwen3Variant] = &[
        Qwen3Variant {
            tag: "q8",
            hf_repo: "worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF",
            hf_filename: "Qwen_3_4b-Q8_0.gguf",
            size_bytes: 4_280_404_704,
        },
        Qwen3Variant {
            tag: "q6",
            hf_repo: "worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF",
            hf_filename: "Qwen_3_4b-Q6_K.gguf",
            size_bytes: 3_306_260_704,
        },
        Qwen3Variant {
            tag: "iq4",
            hf_repo: "worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF",
            hf_filename: "Qwen_3_4b-imatrix-IQ4_XS.gguf",
            size_bytes: 2_270_751_136,
        },
        Qwen3Variant {
            tag: "q3",
            hf_repo: "worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF",
            hf_filename: "Qwen_3_4b-imatrix-Q3_K_M.gguf",
            size_bytes: 2_075_617_696,
        },
    ];
    VARIANTS
}

/// Known Qwen2.5-VL-7B quantized variants, sorted largest → smallest.
pub fn known_qwen2_vl_variants() -> &'static [Qwen2VlVariant] {
    static VARIANTS: &[Qwen2VlVariant] = &[
        Qwen2VlVariant {
            tag: "q8",
            hf_repo: "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
            hf_filename: "Qwen2.5-VL-7B-Instruct-Q8_0.gguf",
            size_bytes: 8_100_000_000,
        },
        Qwen2VlVariant {
            tag: "q6",
            hf_repo: "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
            hf_filename: "Qwen2.5-VL-7B-Instruct-Q6_K.gguf",
            size_bytes: 6_250_000_000,
        },
        Qwen2VlVariant {
            tag: "q5",
            hf_repo: "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
            hf_filename: "Qwen2.5-VL-7B-Instruct-Q5_K_M.gguf",
            size_bytes: 5_440_000_000,
        },
        Qwen2VlVariant {
            tag: "q4",
            hf_repo: "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
            hf_filename: "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
            size_bytes: 4_680_000_000,
        },
        Qwen2VlVariant {
            tag: "q3",
            hf_repo: "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
            hf_filename: "Qwen2.5-VL-7B-Instruct-Q3_K_M.gguf",
            size_bytes: 3_810_000_000,
        },
        Qwen2VlVariant {
            tag: "q2",
            hf_repo: "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
            hf_filename: "Qwen2.5-VL-7B-Instruct-Q2_K.gguf",
            size_bytes: 3_020_000_000,
        },
    ];
    VARIANTS
}

/// Find a Qwen2.5-VL variant by tag (e.g. "q6", "q4", "q3").
pub fn find_qwen2_vl_variant(tag: &str) -> Option<&'static Qwen2VlVariant> {
    known_qwen2_vl_variants().iter().find(|v| v.tag == tag)
}

/// Find a Qwen3-4B variant by tag (e.g. "q8", "q6", "iq4", "q3").
pub fn find_qwen3_variant(tag: &str) -> Option<&'static Qwen3Variant> {
    known_qwen3_variants().iter().find(|v| v.tag == tag)
}

// ── Quantized Qwen3-8B variant registry ─────────────────────────────────────

/// Known Qwen3-8B quantized variants (for Klein-9B), sorted largest → smallest.
pub fn known_qwen3_8b_variants() -> &'static [Qwen3Variant] {
    static VARIANTS: &[Qwen3Variant] = &[
        Qwen3Variant {
            tag: "q8",
            hf_repo: "unsloth/Qwen3-8B-GGUF",
            hf_filename: "Qwen3-8B-Q8_0.gguf",
            size_bytes: 8_709_519_168,
        },
        Qwen3Variant {
            tag: "q6",
            hf_repo: "unsloth/Qwen3-8B-GGUF",
            hf_filename: "Qwen3-8B-Q6_K.gguf",
            size_bytes: 6_725_900_096,
        },
        Qwen3Variant {
            tag: "iq4",
            hf_repo: "unsloth/Qwen3-8B-GGUF",
            hf_filename: "Qwen3-8B-IQ4_XS.gguf",
            size_bytes: 4_581_287_744,
        },
        Qwen3Variant {
            tag: "q3",
            hf_repo: "unsloth/Qwen3-8B-GGUF",
            hf_filename: "Qwen3-8B-Q3_K_M.gguf",
            size_bytes: 4_124_161_856,
        },
    ];
    VARIANTS
}

/// Find a Qwen3-8B variant by tag (e.g. "q8", "q6", "iq4", "q3").
pub fn find_qwen3_8b_variant(tag: &str) -> Option<&'static Qwen3Variant> {
    known_qwen3_8b_variants().iter().find(|v| v.tag == tag)
}

/// Total size of all files in the manifest in bytes.
pub fn total_download_size(manifest: &ModelManifest) -> u64 {
    manifest.total_size_bytes()
}

/// Compute how many bytes still need to be downloaded for a model.
///
/// Checks each manifest file against the hf-hub cache and local storage paths.
/// Returns `(total_bytes, remaining_bytes)` where `remaining_bytes` is the
/// amount that actually needs to be fetched.
pub fn compute_download_size(manifest: &ModelManifest) -> (u64, u64) {
    let mut total = 0u64;
    let mut remaining = 0u64;
    for file in &manifest.files {
        total += file.size_bytes;
        let subdir = storage_path(manifest, file);
        let subdir_str = subdir.parent().map(|p| p.to_string_lossy().to_string());
        if crate::download::cached_file_path(
            &file.hf_repo,
            &file.hf_filename,
            subdir_str.as_deref(),
        )
        .is_none()
        {
            remaining += file.size_bytes;
        }
    }
    (total, remaining)
}

/// Check whether a single manifest file is already cached locally.
pub fn is_file_cached(manifest: &ModelManifest, file: &ModelFile) -> bool {
    let subdir = storage_path(manifest, file);
    let subdir_str = subdir.parent().map(|p| p.to_string_lossy().to_string());
    crate::download::cached_file_path(&file.hf_repo, &file.hf_filename, subdir_str.as_deref())
        .is_some()
}

/// Convert a `ModelManifest` to a `ModelPaths` from resolved download paths.
///
/// For diffusion models, Transformer (or TransformerShards) and VAE are usually required.
/// For upscaler models, only the Upscaler component is required (mapped to `transformer`).
/// For utility models (e.g., qwen3-expand), only the Transformer is required (no VAE).
/// LTX-2 family manifests use a single-file upstream checkpoint and do not require a separate
/// VAE asset. Other components are optional — each engine validates what it needs at load time.
pub fn paths_from_downloads(
    downloads: &[(ModelComponent, PathBuf)],
    family: &str,
) -> Option<ModelPaths> {
    let find = |c: ModelComponent| -> Option<PathBuf> {
        downloads
            .iter()
            .find(|(comp, _)| *comp == c)
            .map(|(_, p)| p.clone())
    };

    let collect = |c: ModelComponent| -> Vec<PathBuf> {
        downloads
            .iter()
            .filter(|(comp, _)| *comp == c)
            .map(|(_, p)| p.clone())
            .collect()
    };

    // Upscaler models: single weights file via Upscaler component, no VAE or encoders
    if UPSCALER_FAMILIES.contains(&family) {
        let transformer = find(ModelComponent::Upscaler)?;
        return Some(ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer,
            transformer_shards: Vec::new(),
            vae: PathBuf::new(),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: Vec::new(),
            text_tokenizer: None,
            decoder: None,
        });
    }

    let transformer_shards = collect(ModelComponent::TransformerShard);

    // Transformer: prefer Transformer / TransformerShard. Pure text-encoder
    // companions (e.g. flux2-te) declare only TextEncoder files — fall back
    // to the first TextEncoder shard so the companion still produces a
    // usable ModelPaths for `manifest_files_exist` and the bridge's
    // `text_encoder_files` collection logic.
    let transformer = find(ModelComponent::Transformer)
        .or_else(|| transformer_shards.first().cloned())
        .or_else(|| {
            if UTILITY_FAMILIES.contains(&family) {
                downloads
                    .iter()
                    .find(|(c, _)| *c == ModelComponent::TextEncoder)
                    .or_else(|| {
                        downloads
                            .iter()
                            .find(|(c, _)| *c == ModelComponent::Decoder)
                    })
                    .map(|(_, p)| p.clone())
            } else {
                None
            }
        })?;

    // Utility models and LTX-2: transformer + optional tokenizer, no standalone VAE asset
    let vae = if UTILITY_FAMILIES.contains(&family) || family == "ltx2" {
        find(ModelComponent::Vae).unwrap_or_default()
    } else {
        find(ModelComponent::Vae)?
    };

    Some(ModelPaths {
        transformer,
        transformer_shards,
        vae,
        low_noise_transformer: find(ModelComponent::LowNoiseTransformer),
        spatial_upscaler: find(ModelComponent::SpatialUpscaler),
        temporal_upscaler: find(ModelComponent::TemporalUpscaler),
        distilled_lora: find(ModelComponent::DistilledLora),
        low_noise_distilled_lora: find(ModelComponent::LowNoiseDistilledLora),
        t5_encoder: find(ModelComponent::T5Encoder),
        clip_encoder: find(ModelComponent::ClipEncoder),
        t5_tokenizer: find(ModelComponent::T5Tokenizer),
        clip_tokenizer: find(ModelComponent::ClipTokenizer),
        clip_encoder_2: find(ModelComponent::ClipEncoder2),
        clip_tokenizer_2: find(ModelComponent::ClipTokenizer2),
        text_encoder_files: collect(ModelComponent::TextEncoder),
        text_tokenizer: find(ModelComponent::TextTokenizer),
        decoder: find(ModelComponent::Decoder),
    })
}

fn ltx_video_manifests() -> Vec<ModelManifest> {
    // Plain LTX-Video has no image-to-video path — its engine never reads
    // `source_image` (chain stages render independently and stitch). Only
    // LTX-2 conditions on a still, so these advertise Unsupported and
    // admission rejects an attached image instead of silently ignoring it.
    let dev_defaults = ManifestDefaults {
        steps: 40,
        guidance: 3.0,
        width: 1216,
        height: 704,
        is_schnell: false,
        scheduler: None,
        negative_prompt: None,
        frames: Some(25),
        fps: Some(30),
        source_image: Some(crate::types::SourceImageCapability::Unsupported),
    };
    let distilled_defaults = ManifestDefaults {
        steps: 8,
        guidance: 1.0,
        width: 1216,
        height: 704,
        is_schnell: false,
        scheduler: None,
        negative_prompt: None,
        frames: Some(25),
        fps: Some(30),
        source_image: Some(crate::types::SourceImageCapability::Unsupported),
    };
    let multiscale_distilled_defaults = ManifestDefaults {
        steps: 7,
        guidance: 1.0,
        width: 1216,
        height: 704,
        is_schnell: false,
        scheduler: None,
        negative_prompt: None,
        frames: Some(25),
        fps: Some(30),
        source_image: Some(crate::types::SourceImageCapability::Unsupported),
    };
    let multiscale_dev_defaults = ManifestDefaults {
        steps: 30,
        guidance: 8.0,
        width: 1216,
        height: 704,
        is_schnell: false,
        scheduler: None,
        negative_prompt: None,
        frames: Some(25),
        fps: Some(30),
        source_image: Some(crate::types::SourceImageCapability::Unsupported),
    };
    let shared_t5_files = vec![
        // T5-XXL FP16 text encoder (shared with FLUX)
        ModelFile {
            hf_repo: "comfyanonymous/flux_text_encoders".to_string(),
            hf_filename: "t5xxl_fp16.safetensors".to_string(),
            component: ModelComponent::T5Encoder,
            size_bytes: 9_787_841_024,
            gated: false,
            sha256: Some("6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635"),
        },
        // T5 tokenizer (shared with FLUX)
        ModelFile {
            hf_repo: "lmz/mt5-tokenizers".to_string(),
            hf_filename: "t5-v1_1-xxl.tokenizer.json".to_string(),
            component: ModelComponent::T5Tokenizer,
            size_bytes: 17_163_758,
            gated: false,
            sha256: Some("812ebb1f7bcb9ec5b9b0efcd45e72fbd2ef5f46ec8c4b29d3b07dc1505ca5af7"),
        },
    ];
    // Current Candle LTX VAE support still targets the published legacy VAE
    // layout from LTX-Video-0.9.5. The newer LTX-Video repo ships a different
    // VAE checkpoint that fails local decode with the current port
    // (`decoder.conv_in.conv.weight` shape mismatch) and needs a follow-up
    // architecture update before we can switch repos.
    let shared_vae_file = ModelFile {
        hf_repo: "Lightricks/LTX-Video-0.9.5".to_string(),
        hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
        component: ModelComponent::Vae,
        size_bytes: 2_493_855_612,
        gated: false,
        sha256: Some("7eb65b16cf8ddfd70ccb1c541384ae49ffd6639d754c6b713a11cb72d097233f"),
    };
    let spatial_upscaler_file = ModelFile {
        hf_repo: "Lightricks/LTX-Video".to_string(),
        hf_filename: "ltxv-spatial-upscaler-0.9.8.safetensors".to_string(),
        component: ModelComponent::SpatialUpscaler,
        size_bytes: 505_024_432,
        gated: false,
        sha256: None,
    };
    vec![
        ModelManifest {
            name: "ltx-video-0.9.6:bf16".to_string(),
            family: "ltx-video".to_string(),
            description: "LTX Video 0.9.6 2B BF16 — improved quality, 30fps, text-to-video"
                .to_string(),
            files: {
                let mut f = vec![
                    ModelFile {
                        hf_repo: "Lightricks/LTX-Video".to_string(),
                        hf_filename: "ltxv-2b-0.9.6-dev-04-25.safetensors".to_string(),
                        component: ModelComponent::Transformer,
                        size_bytes: 6_340_743_924,
                        gated: false,
                        sha256: None,
                    },
                    shared_vae_file.clone(),
                ];
                f.extend(shared_t5_files.clone());
                f
            },
            defaults: dev_defaults,
            hidden: false,
        },
        ModelManifest {
            name: "ltx-video-0.9.6-distilled:bf16".to_string(),
            family: "ltx-video".to_string(),
            description: "LTX Video 0.9.6 distilled 2B BF16 — fast 8-step text-to-video"
                .to_string(),
            files: {
                let mut f = vec![
                    ModelFile {
                        hf_repo: "Lightricks/LTX-Video".to_string(),
                        hf_filename: "ltxv-2b-0.9.6-distilled-04-25.safetensors".to_string(),
                        component: ModelComponent::Transformer,
                        size_bytes: 6_340_744_028,
                        gated: false,
                        sha256: None,
                    },
                    shared_vae_file.clone(),
                ];
                f.extend(shared_t5_files.clone());
                f
            },
            defaults: distilled_defaults,
            hidden: false,
        },
        ModelManifest {
            name: "ltx-video-0.9.8-2b-distilled:bf16".to_string(),
            family: "ltx-video".to_string(),
            description: "LTX Video 0.9.8 distilled 2B BF16 — multiscale-ready low-VRAM path"
                .to_string(),
            files: {
                let mut f = vec![
                    ModelFile {
                        hf_repo: "Lightricks/LTX-Video".to_string(),
                        hf_filename: "ltxv-2b-0.9.8-distilled.safetensors".to_string(),
                        component: ModelComponent::Transformer,
                        size_bytes: 6_340_744_492,
                        gated: false,
                        sha256: None,
                    },
                    shared_vae_file.clone(),
                    spatial_upscaler_file.clone(),
                ];
                f.extend(shared_t5_files.clone());
                f
            },
            defaults: multiscale_distilled_defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "ltx-video-0.9.8-13b-dev:bf16".to_string(),
            family: "ltx-video".to_string(),
            description: "LTX Video 0.9.8 13B dev BF16 — highest-quality LTX checkpoint"
                .to_string(),
            files: {
                let mut f = vec![
                    ModelFile {
                        hf_repo: "Lightricks/LTX-Video".to_string(),
                        hf_filename: "ltxv-13b-0.9.8-dev.safetensors".to_string(),
                        component: ModelComponent::Transformer,
                        size_bytes: 28_579_183_340,
                        gated: false,
                        sha256: None,
                    },
                    shared_vae_file.clone(),
                    spatial_upscaler_file.clone(),
                ];
                f.extend(shared_t5_files.clone());
                f
            },
            defaults: multiscale_dev_defaults,
            hidden: false,
        },
        ModelManifest {
            name: "ltx-video-0.9.8-13b-distilled:bf16".to_string(),
            family: "ltx-video".to_string(),
            description: "LTX Video 0.9.8 13B distilled BF16 — faster 13B-quality LTX video"
                .to_string(),
            files: {
                let mut f = vec![
                    ModelFile {
                        hf_repo: "Lightricks/LTX-Video".to_string(),
                        hf_filename: "ltxv-13b-0.9.8-distilled.safetensors".to_string(),
                        component: ModelComponent::Transformer,
                        size_bytes: 28_579_183_564,
                        gated: false,
                        sha256: None,
                    },
                    shared_vae_file,
                    spatial_upscaler_file,
                ];
                f.extend(shared_t5_files);
                f
            },
            defaults: multiscale_distilled_defaults,
            hidden: false,
        },
    ]
}

fn shared_ltx2_files() -> Vec<ModelFile> {
    let gemma_repo = "google/gemma-3-12b-it-qat-q4_0-unquantized".to_string();
    [
        "config.json",
        "generation_config.json",
        "model-00001-of-00005.safetensors",
        "model-00002-of-00005.safetensors",
        "model-00003-of-00005.safetensors",
        "model-00004-of-00005.safetensors",
        "model-00005-of-00005.safetensors",
        "model.safetensors.index.json",
        "added_tokens.json",
        "chat_template.json",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    ]
    .into_iter()
    .map(|hf_filename| {
        let (size_bytes, sha256) = match hf_filename {
            "model-00001-of-00005.safetensors" => (
                4_979_902_192,
                Some("e6fb899db428481aafb45a20130457df6e247e7cb03b7d9f01ee4bc2a9a08138"),
            ),
            "model-00002-of-00005.safetensors" => (
                4_931_296_592,
                Some("d251e7fe9799d529405ddb61705a44cd700bd30a8b66a8d44ae26ddf8365dbc6"),
            ),
            "model-00003-of-00005.safetensors" => (
                4_931_296_656,
                Some("0684ef801385f0669a0b3e4ab160c50877efdbfa40eb97788595985de2743e78"),
            ),
            "model-00004-of-00005.safetensors" => (
                4_931_296_656,
                Some("b4b964e6526f81ccfa625c900b72ce92d5e0fd2debb75998763038ad06b9c541"),
            ),
            "model-00005-of-00005.safetensors" => (
                4_601_000_928,
                Some("4ef2de8f93e165b4e02425769fc566000b0674256ef0c3a27b23a0d45eb12088"),
            ),
            "tokenizer.json" => (
                33_384_570,
                Some("7d4046bf0505a327dd5a0abbb427ecd4fc82f99c2ceaa170bc61ecde12809b0c"),
            ),
            "tokenizer.model" => (
                4_689_074,
                Some("1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c"),
            ),
            _ => (0, None),
        };
        ModelFile {
            hf_repo: gemma_repo.clone(),
            hf_filename: hf_filename.to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes,
            gated: true,
            sha256,
        }
    })
    .collect()
}

fn ltx2_manifests() -> Vec<ModelManifest> {
    let defaults = ManifestDefaults {
        steps: 8,
        guidance: 3.0,
        width: 1216,
        height: 704,
        is_schnell: false,
        scheduler: None,
        negative_prompt: None,
        frames: Some(97),
        fps: Some(24),
        source_image: Some(crate::types::SourceImageCapability::Optional),
    };

    let make_manifest = |name: &str,
                         description: &str,
                         transformer_repo: &str,
                         transformer_file: &str,
                         shared_repo: &str,
                         spatial_file: &str,
                         spatial_file_x1_5: Option<&str>,
                         temporal_file: &str,
                         distilled_lora: Option<&str>| {
        let mut files = shared_ltx2_files();
        let (transformer_size_bytes, transformer_sha256) = match transformer_file {
            "ltx-2-19b-dev-fp8.safetensors" => (
                27_078_716_018,
                "8a67e709b6d1adc061cb19921887a5c15754178199e45801a04310e9b522760d",
            ),
            "ltx-2-19b-distilled-fp8.safetensors" => (
                27_078_716_346,
                "8ae14327130c6ffdc87705b02c8e7654aa5c6d9a7f28a52d0acc1c30cb0d2932",
            ),
            "ltx-2.3-22b-dev-fp8.safetensors" => (
                29_145_431_166,
                "28606c5b5a06ce56f896d4dfcb20f212739e07a68fbe48e53638188449d26450",
            ),
            "ltx-2.3-22b-distilled-fp8.safetensors" => (
                29_531_884_062,
                "d9646b6f2d5c42d337b23671634c43bfeece6989644f51b4a3aa088465ccd3b2",
            ),
            "ltx-2.3-22b-dev.safetensors" => (
                46_149_344_974,
                "7ab7225325bc403448ea84b6db2269811a880e5118cd2ee2b6282a93d585016f",
            ),
            "ltx-2.3-22b-distilled-1.1.safetensors" => (
                46_149_345_334,
                "b33b7fe4bbfe084f484be4aaf90b0f1d95dca20d403ac4c0e037eb8c4f0af7cc",
            ),
            other => panic!("unexpected LTX-2 transformer file '{other}'"),
        };
        files.push(ModelFile {
            hf_repo: transformer_repo.to_string(),
            hf_filename: transformer_file.to_string(),
            component: ModelComponent::Transformer,
            size_bytes: transformer_size_bytes,
            gated: false,
            sha256: Some(transformer_sha256),
        });
        let (spatial_size_bytes, spatial_sha256) = match spatial_file {
            "ltx-2-spatial-upscaler-x2-1.0.safetensors" => (
                995_765_578,
                "3160fabf8edf0bc4dd8de40353a180813b111ce586b655ad54af9a7b8c6736de",
            ),
            "ltx-2.3-spatial-upscaler-x2-1.0.safetensors" => (
                995_743_504,
                "93800de87dbc448b5b31f3c5c3a1579ba6335151de061a564f6f026b0fc770ad",
            ),
            "ltx-2.3-spatial-upscaler-x2-1.1.safetensors" => (
                995_743_560,
                "5f416311fa8172b65af67530758964708d29a317b830d689a51143b7f91913ed",
            ),
            other => panic!("unexpected LTX-2 spatial upscaler file '{other}'"),
        };
        files.push(ModelFile {
            hf_repo: shared_repo.to_string(),
            hf_filename: spatial_file.to_string(),
            component: ModelComponent::SpatialUpscaler,
            size_bytes: spatial_size_bytes,
            gated: false,
            sha256: Some(spatial_sha256),
        });
        if let Some(spatial_file_x1_5) = spatial_file_x1_5 {
            let (spatial_size_bytes, spatial_sha256) = match spatial_file_x1_5 {
                "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors" => (
                    1_090_125_794,
                    "b2b3193e68cb04b4701e1d59bec4ed5d5e3e84506d9a42d5a129d37d39823df7",
                ),
                other => panic!("unexpected LTX-2 x1.5 spatial upscaler file '{other}'"),
            };
            files.push(ModelFile {
                hf_repo: shared_repo.to_string(),
                hf_filename: spatial_file_x1_5.to_string(),
                component: ModelComponent::SpatialUpscaler,
                size_bytes: spatial_size_bytes,
                gated: false,
                sha256: Some(spatial_sha256),
            });
        }
        let (temporal_size_bytes, temporal_sha256) = match temporal_file {
            "ltx-2-temporal-upscaler-x2-1.0.safetensors" => (
                261_965_800,
                "9a35c2eb92f6ed39369fcb83045daa070bc7c2a97fc7267abd6291203fd05b88",
            ),
            "ltx-2.3-temporal-upscaler-x2-1.0.safetensors" => (
                261_944_000,
                "2bc3300f2b3c3c1834d72164fbf13a3b9fd73e5a741e8a2c3f4035f89a75c3fe",
            ),
            other => panic!("unexpected LTX-2 temporal upscaler file '{other}'"),
        };
        files.push(ModelFile {
            hf_repo: shared_repo.to_string(),
            hf_filename: temporal_file.to_string(),
            component: ModelComponent::TemporalUpscaler,
            size_bytes: temporal_size_bytes,
            gated: false,
            sha256: Some(temporal_sha256),
        });
        if let Some(distilled_lora) = distilled_lora {
            let (distilled_lora_size_bytes, distilled_lora_sha256) = match distilled_lora {
                "ltx-2-19b-distilled-lora-384.safetensors" => (
                    7_674_558_424,
                    "2718f89582003cbb5b616635f18c091641917a3f3e5a2f2ad0fb3d5fdd153534",
                ),
                "ltx-2.3-22b-distilled-lora-384.safetensors" => (
                    7_605_507_256,
                    "2943ab994f3c9d88052e5a2a34cca14e4a2dfc36b1d8c407931d52d5c25dd72b",
                ),
                "ltx-2.3-22b-distilled-lora-384-1.1.safetensors" => (
                    7_605_507_256,
                    "f5d4953f3386197a4b4f5abdb17616ff256171e8075c111d6e7d2dfa6e823b3a",
                ),
                other => panic!("unexpected LTX-2 distilled LoRA file '{other}'"),
            };
            files.push(ModelFile {
                hf_repo: shared_repo.to_string(),
                hf_filename: distilled_lora.to_string(),
                component: ModelComponent::DistilledLora,
                size_bytes: distilled_lora_size_bytes,
                gated: false,
                sha256: Some(distilled_lora_sha256),
            });
        }
        ModelManifest {
            name: name.to_string(),
            family: "ltx2".to_string(),
            description: description.to_string(),
            files,
            defaults: defaults.clone(),
            hidden: false,
        }
    };

    vec![
        make_manifest(
            "ltx-2-19b-dev:fp8",
            "LTX-2 19B dev FP8 — joint audio-video generation",
            "Lightricks/LTX-2",
            "ltx-2-19b-dev-fp8.safetensors",
            "Lightricks/LTX-2",
            "ltx-2-spatial-upscaler-x2-1.0.safetensors",
            None,
            "ltx-2-temporal-upscaler-x2-1.0.safetensors",
            Some("ltx-2-19b-distilled-lora-384.safetensors"),
        ),
        make_manifest(
            "ltx-2-19b-distilled:fp8",
            "LTX-2 19B distilled FP8 — faster joint audio-video generation",
            "Lightricks/LTX-2",
            "ltx-2-19b-distilled-fp8.safetensors",
            "Lightricks/LTX-2",
            "ltx-2-spatial-upscaler-x2-1.0.safetensors",
            None,
            "ltx-2-temporal-upscaler-x2-1.0.safetensors",
            None,
        ),
        make_manifest(
            "ltx-2.3-22b-dev:fp8",
            "LTX-2.3 22B dev FP8 — highest quality joint audio-video model",
            "Lightricks/LTX-2.3-fp8",
            "ltx-2.3-22b-dev-fp8.safetensors",
            "Lightricks/LTX-2.3",
            "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
            Some("ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors"),
            "ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
            Some("ltx-2.3-22b-distilled-lora-384.safetensors"),
        ),
        make_manifest(
            "ltx-2.3-22b-distilled:fp8",
            "LTX-2.3 22B distilled FP8 — fastest joint audio-video pipeline",
            "Lightricks/LTX-2.3-fp8",
            "ltx-2.3-22b-distilled-fp8.safetensors",
            "Lightricks/LTX-2.3",
            "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
            Some("ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors"),
            "ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
            None,
        ),
        make_manifest(
            "ltx-2.3-22b-dev:bf16",
            "LTX-2.3 22B dev BF16 — full-quality trainable joint audio-video model",
            "Lightricks/LTX-2.3",
            "ltx-2.3-22b-dev.safetensors",
            "Lightricks/LTX-2.3",
            "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            Some("ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors"),
            "ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
            Some("ltx-2.3-22b-distilled-lora-384-1.1.safetensors"),
        ),
        make_manifest(
            "ltx-2.3-22b-distilled:bf16",
            "LTX-2.3 22B distilled BF16 — full-precision eight-step pipeline",
            "Lightricks/LTX-2.3",
            "ltx-2.3-22b-distilled-1.1.safetensors",
            "Lightricks/LTX-2.3",
            "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            Some("ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors"),
            "ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
            None,
        ),
    ]
}

fn ltx2_control_manifests() -> Vec<ModelManifest> {
    crate::ltx2_control::LTX2_CONTROL_ADAPTERS
        .iter()
        .map(|adapter| ModelManifest {
            name: adapter.download_model.to_string(),
            family: "ltx2-control".to_string(),
            description: format!("{} — {}", adapter.label, adapter.profile.label()),
            files: adapter
                .files()
                .map(|file| ModelFile {
                    hf_repo: adapter.hf_repo.to_string(),
                    hf_filename: file.hf_filename.to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: file.size_bytes,
                    gated: adapter.gated,
                    sha256: Some(file.sha256),
                })
                .collect(),
            defaults: ManifestDefaults {
                steps: 1,
                guidance: 0.0,
                width: 16,
                height: 16,
                is_schnell: true,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: true,
        })
        .collect()
}

fn ltx2_camera_control_manifests() -> Vec<ModelManifest> {
    crate::ltx2_camera::LTX2_CAMERA_CONTROLS
        .iter()
        .map(|preset| ModelManifest {
            name: preset.download_model.to_string(),
            family: "ltx2-camera-control".to_string(),
            description: format!("{} camera control — LTX-2 19B", preset.label),
            files: vec![ModelFile {
                hf_repo: preset.hf_repo.to_string(),
                hf_filename: preset.hf_filename.to_string(),
                component: ModelComponent::Transformer,
                size_bytes: preset.size_bytes,
                gated: false,
                sha256: Some(preset.sha256),
            }],
            defaults: ManifestDefaults {
                steps: 1,
                guidance: 0.0,
                width: 16,
                height: 16,
                is_schnell: true,
                scheduler: None,
                negative_prompt: None,
                frames: None,
                fps: None,
                source_image: None,
            },
            hidden: true,
        })
        .collect()
}

fn controlnet_manifests() -> Vec<ModelManifest> {
    let defaults = ManifestDefaults {
        steps: 25,
        guidance: 7.5,
        width: 512,
        height: 512,
        is_schnell: false,
        scheduler: Some(Scheduler::Ddim),
        negative_prompt: None,
        frames: None,
        fps: None,
        source_image: None,
    };
    vec![
        ModelManifest {
            name: "controlnet-canny-sd15:fp16".to_string(),
            family: "controlnet".to_string(),
            description: "ControlNet Canny edge detection for SD1.5".to_string(),
            files: vec![ModelFile {
                hf_repo: "lllyasviel/control_v11p_sd15_canny".to_string(),
                hf_filename: "diffusion_pytorch_model.fp16.safetensors".to_string(),
                component: ModelComponent::Transformer,
                size_bytes: 722_598_642,
                gated: false,
                sha256: None,
            }],
            defaults: defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "controlnet-depth-sd15:fp16".to_string(),
            family: "controlnet".to_string(),
            description: "ControlNet depth estimation for SD1.5".to_string(),
            files: vec![ModelFile {
                hf_repo: "lllyasviel/control_v11f1p_sd15_depth".to_string(),
                hf_filename: "diffusion_pytorch_model.fp16.safetensors".to_string(),
                component: ModelComponent::Transformer,
                size_bytes: 722_598_642,
                gated: false,
                sha256: None,
            }],
            defaults: defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "controlnet-openpose-sd15:fp16".to_string(),
            family: "controlnet".to_string(),
            description: "ControlNet OpenPose body detection for SD1.5".to_string(),
            files: vec![ModelFile {
                hf_repo: "lllyasviel/control_v11p_sd15_openpose".to_string(),
                hf_filename: "diffusion_pytorch_model.fp16.safetensors".to_string(),
                component: ModelComponent::Transformer,
                size_bytes: 722_598_642,
                gated: false,
                sha256: None,
            }],
            defaults,
            hidden: false,
        },
    ]
}

fn qwen3_expand_manifests() -> Vec<ModelManifest> {
    let defaults = ManifestDefaults {
        steps: 0,
        guidance: 0.0,
        width: 0,
        height: 0,
        is_schnell: false,
        scheduler: None,
        negative_prompt: None,
        frames: None,
        fps: None,
        source_image: None,
    };

    vec![
        ModelManifest {
            name: "qwen3-expand:q8".to_string(),
            family: "qwen3-expand".to_string(),
            description: "Qwen3-1.7B Q8 — prompt expansion LLM (1.8GB)".to_string(),
            files: vec![
                ModelFile {
                    hf_repo: "Qwen/Qwen3-1.7B-GGUF".to_string(),
                    hf_filename: "Qwen3-1.7B-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 1_834_426_016,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    hf_repo: "Qwen/Qwen3-1.7B".to_string(),
                    hf_filename: "tokenizer.json".to_string(),
                    component: ModelComponent::TextTokenizer,
                    size_bytes: 11_422_654,
                    gated: false,
                    sha256: None,
                },
            ],
            defaults: defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "qwen3-expand-small:q8".to_string(),
            family: "qwen3-expand".to_string(),
            description: "Qwen3-0.6B Q8 — lightweight prompt expansion LLM (0.6GB)".to_string(),
            files: vec![
                ModelFile {
                    hf_repo: "Qwen/Qwen3-0.6B-GGUF".to_string(),
                    hf_filename: "Qwen3-0.6B-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 639_446_688,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    hf_repo: "Qwen/Qwen3-1.7B".to_string(),
                    hf_filename: "tokenizer.json".to_string(),
                    component: ModelComponent::TextTokenizer,
                    size_bytes: 11_422_654,
                    gated: false,
                    sha256: None,
                },
            ],
            defaults,
            hidden: false,
        },
    ]
}

/// Synthetic manifest entries for the canonical companions in
/// `mold_catalog::companions::COMPANIONS`. Single-file Civitai checkpoints
/// (SDXL Pony, FLUX-dev finetunes, etc.) routinely strip their text encoders
/// and VAE; the catalog scanner records `companions: ["clip-l", ...]` on
/// those entries, and `POST /api/catalog/:id/download` enqueues each missing
/// companion before the primary entry. The download queue is keyed on
/// manifest names, so each canonical companion needs an entry here.
///
/// These manifests live in `family: "companion"` (see `UTILITY_FAMILIES`) so
/// they're hidden from `mold list`, never selected as defaults, and routed
/// through `pull_model_files_only_with_callback` instead of the
/// config-writing path.
///
/// File lists are deliberately minimal — just the safetensors + (where
/// applicable) the unified `tokenizer.json`. Single-file engines
/// only need the encoder weights and a tokenizer; richer A1111-style
/// loaders that want config/vocab/merges files are a future extension.
/// Upstream's default negative prompt, shared by every Wan variant
/// (`Wan2.2/wan/configs/shared_config.py`). Applied when a request leaves
/// `negative_prompt` unset: these checkpoints were tuned against it, and an
/// empty uncond quietly degrades CFG quality.
/// Upstream's tuned Chinese negative prompt, shared by every Wan preset
/// (`Wan2.2/wan/configs/shared_config.py`). The manifests plant it in their
/// defaults; the engine falls back to it when a request arrives without one.
pub const WAN_DEFAULT_NEGATIVE_PROMPT: &str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";

/// The tuned default negative prompt a family's *engine* substitutes when a
/// request carries no `negative_prompt` at all. This is the single source for
/// every surface that advertises, materializes, or prefills that default —
/// `/api/models[].default_negative_prompt`, the server's request
/// materialization, the CLI's local fallback, and TUI prefill all resolve
/// through here so they cannot drift from the engine
/// (`wan/pipeline.rs::resolve_negative_prompt`). An explicit value — the
/// empty string included — is always authoritative and never replaced.
pub fn default_negative_prompt_for_family(family: &str) -> Option<&'static str> {
    match family {
        "wan" => Some(WAN_DEFAULT_NEGATIVE_PROMPT),
        _ => None,
    }
}

/// The A14B quality tier's advertised guidance scale. Upstream runs the tier
/// with *per-expert* scales — T2V `(3.0, 4.0)` low/high, I2V `(3.5, 3.5)`
/// (`Wan2.2/wan/configs/wan_{t2v,i2v}_A14B.py:37`) — which a single wire
/// scalar cannot express, so the manifests advertise this compromise value.
/// The engine treats a request carrying exactly this value as "the default"
/// and substitutes upstream's per-expert pair; any other explicit scale is
/// honored uniformly. Keeping the sentinel and the manifest default the same
/// constant is what makes that contract hold.
pub const WAN_A14B_QUALITY_GUIDANCE: f64 = 3.5;

/// Shared Wan component files: the UMT5-XXL text encoder (identical across
/// every Wan 2.1/2.2 variant — Comfy-Org's safetensors repack of upstream's
/// `models_t5_umt5-xxl-enc-bf16.pth`) and its SentencePiece tokenizer in
/// tokenizers-crate JSON form.
fn shared_wan_files() -> Vec<ModelFile> {
    vec![
        ModelFile {
            hf_repo: "Comfy-Org/Wan_2.1_ComfyUI_repackaged".to_string(),
            hf_filename: "split_files/text_encoders/umt5_xxl_fp16.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 11_366_399_385,
            gated: false,
            sha256: Some("7b8850f1961e1cf8a77cca4c964a358d303f490833c6c087d0cff4b2f99db2af"),
        },
        ModelFile {
            hf_repo: "google/umt5-xxl".to_string(),
            hf_filename: "tokenizer.json".to_string(),
            component: ModelComponent::TextTokenizer,
            size_bytes: 16_853_013,
            gated: false,
            sha256: Some("af904105ce1071b1202bba0059a841f4a7b85b48b6ec179c4948e3483476e0dd"),
        },
    ]
}

/// Wan 2.1/2.2 video manifests (ComfyUI-repack weight layout — the format the
/// GGUF/LoRA ecosystem shares). Text-to-video is live; image conditioning on
/// TI2V-5B follows in a later layer.
fn wan_manifests() -> Vec<ModelManifest> {
    let defaults_480p = ManifestDefaults {
        // ComfyUI's proven Wan 2.1 recipe (30 steps, cfg 6, uni_pc); upstream
        // defaults to 50 steps at cfg 5-6, which buys little at 1.3B scale.
        steps: 30,
        guidance: 6.0,
        width: 832,
        height: 480,
        is_schnell: false,
        scheduler: None,
        negative_prompt: Some(WAN_DEFAULT_NEGATIVE_PROMPT.to_string()),
        frames: Some(81),
        fps: Some(16),
        // The 1.3B is a pure T2V DiT (16-channel patch embedding, no
        // conditioning concat or inpaint path) — a supplied source image is
        // rejected, so admission should refuse it before the queue (#772).
        source_image: Some(crate::types::SourceImageCapability::Unsupported),
    };
    let defaults_ti2v = ManifestDefaults {
        // ComfyUI's TI2V-5B template: 20 steps, cfg 5, uni_pc, 121f @ 24 fps.
        steps: 20,
        guidance: 5.0,
        width: 1280,
        height: 704,
        is_schnell: false,
        scheduler: None,
        negative_prompt: Some(WAN_DEFAULT_NEGATIVE_PROMPT.to_string()),
        frames: Some(121),
        fps: Some(24),
        // TI2V renders text-to-video and, when a source is supplied,
        // pins it as frame 0 through the latent-inpaint path (#772).
        source_image: Some(crate::types::SourceImageCapability::Optional),
    };

    vec![
        ModelManifest {
            name: "wan21-t2v-1.3b:bf16".to_string(),
            family: "wan".to_string(),
            description: "Wan 2.1 T2V 1.3B BF16 — 480p text-to-video".to_string(),
            files: {
                let mut files = shared_wan_files();
                files.push(ModelFile {
                    hf_repo: "Comfy-Org/Wan_2.1_ComfyUI_repackaged".to_string(),
                    hf_filename: "split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors"
                        .to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 2_838_104_528,
                    gated: false,
                    sha256: Some(
                        "6f999b0d6cb9a72b3d98ac386ed96f57f8cecae13994a69232514ea4974ad5fd",
                    ),
                });
                files.push(ModelFile {
                    hf_repo: "Comfy-Org/Wan_2.1_ComfyUI_repackaged".to_string(),
                    hf_filename: "split_files/vae/wan_2.1_vae.safetensors".to_string(),
                    component: ModelComponent::Vae,
                    size_bytes: 253_815_318,
                    gated: false,
                    sha256: Some(
                        "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b",
                    ),
                });
                files
            },
            defaults: defaults_480p,
            hidden: false,
        },
        ModelManifest {
            name: "wan22-ti2v-5b:fp16".to_string(),
            family: "wan".to_string(),
            description: "Wan 2.2 TI2V 5B FP16 — 720p24 text- and image-to-video".to_string(),
            files: {
                let mut files = shared_wan_files();
                files.push(ModelFile {
                    hf_repo: "Comfy-Org/Wan_2.2_ComfyUI_Repackaged".to_string(),
                    hf_filename: "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"
                        .to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 9_999_658_848,
                    gated: false,
                    sha256: Some(
                        "456f901338bd9eadbded3828b819109a9b68e8a525ca5cf8d0049a69fcfeca1e",
                    ),
                });
                files.push(ModelFile {
                    hf_repo: "Comfy-Org/Wan_2.2_ComfyUI_Repackaged".to_string(),
                    hf_filename: "split_files/vae/wan2.2_vae.safetensors".to_string(),
                    component: ModelComponent::Vae,
                    size_bytes: 1_409_400_960,
                    gated: false,
                    sha256: Some(
                        "e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156",
                    ),
                });
                files
            },
            defaults: defaults_ti2v.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "wan22-ti2v-5b:q8".to_string(),
            family: "wan".to_string(),
            description: "Wan 2.2 TI2V 5B Q8_0 — 720p24 text- and image-to-video (small-card GGUF)"
                .to_string(),
            files: {
                let mut files = shared_wan_files();
                // QuantStack's Q8_0 repack of the same 5B transformer:
                // 5.4 GB against fp16's 10 GB, which is what reaches
                // 8-12 GB-class cards (#794).
                // VRAM peak: 18,460 MiB at the default 1280x704 x 121f
                // (24 fps, 20 steps) on an RTX 4090 — small cards use
                // reduced frames/resolution (#794).
                files.push(ModelFile {
                    hf_repo: "QuantStack/Wan2.2-TI2V-5B-GGUF".to_string(),
                    hf_filename: "Wan2.2-TI2V-5B-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 5_400_179_040,
                    gated: false,
                    sha256: Some(
                        "57bece983817ab2f957546683bb670f13be7d99022d45674840cd999a050ea8f",
                    ),
                });
                files.push(ModelFile {
                    hf_repo: "Comfy-Org/Wan_2.2_ComfyUI_Repackaged".to_string(),
                    hf_filename: "split_files/vae/wan2.2_vae.safetensors".to_string(),
                    component: ModelComponent::Vae,
                    size_bytes: 1_409_400_960,
                    gated: false,
                    sha256: Some(
                        "e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156",
                    ),
                });
                files
            },
            defaults: defaults_ti2v,
            hidden: false,
        },
        a14b_manifest(A14bTier::Fast, A14bTask::T2v),
        a14b_manifest(A14bTier::Quality, A14bTask::T2v),
        a14b_manifest(A14bTier::Compact, A14bTask::T2v),
        a14b_manifest(A14bTier::Fp8, A14bTask::T2v),
        a14b_manifest(A14bTier::Fast, A14bTask::I2v),
        a14b_manifest(A14bTier::Quality, A14bTask::I2v),
        a14b_manifest(A14bTier::Compact, A14bTask::I2v),
        a14b_manifest(A14bTier::Fp8, A14bTask::I2v),
    ]
}

/// Which A14B task a manifest is built for. The two differ only in the GGUF
/// repository and the distill pair — the DiT is the same 40-block, 5120-wide
/// network, with I2V declaring 36 input channels for the mask-plus-image
/// concat.
#[derive(Clone, Copy)]
enum A14bTask {
    T2v,
    I2v,
}

impl A14bTask {
    fn slug(self) -> &'static str {
        match self {
            Self::T2v => "t2v",
            Self::I2v => "i2v",
        }
    }

    fn gguf_repo(self) -> &'static str {
        match self {
            Self::T2v => "QuantStack/Wan2.2-T2V-A14B-GGUF",
            Self::I2v => "QuantStack/Wan2.2-I2V-A14B-GGUF",
        }
    }
}

/// The four shipped A14B tiers. The GGUF tiers are the same weights at
/// different quantizations; what actually separates them is the four-step
/// distill, which only the Lightning tiers carry. The fp8 tier is a different
/// container entirely (#777).
#[derive(Clone, Copy, PartialEq, Eq)]
enum A14bTier {
    /// Q5_K_M plus the lightx2v four-step distill pair.
    Fast,
    /// Q8_0, no adapter, the family's ordinary step count.
    Quality,
    /// Q4_K_M plus the same four-step distill pair as `Fast` — the same
    /// recipe with a ~1.1 GB smaller resident expert, aimed at 12-16 GB
    /// cards. Q4_K_M is the community floor before the quality cliff; the
    /// Q4/Q5 boundary is where GGUF Wan degrades visibly (#794).
    Compact,
    /// Comfy-Org's `*_fp8_scaled` e4m3 safetensors experts (~14.3 GB each,
    /// max-of-pair residency on 24 GB cards), running the 20-step quality
    /// recipe. Measured against `:q8` at 33f/832x480 on an RTX 4090: step
    /// time is a wash (~28 s/step both — the denoise is compute-bound, not
    /// weight-decode-bound), but peak VRAM drops 20,278 → 17,646 MiB. That
    /// headroom, plus the LoRA story, is the tier: unlike GGUF, user LoRAs
    /// (including the lightx2v pair passed via `--lora`) merge into fp8
    /// weights at load instead of running as a per-step branch. Bare-name
    /// resolution deliberately stays on `:q8` (#777 tier-semantics
    /// decision).
    Fp8,
}

impl A14bTier {
    fn tag(self) -> &'static str {
        match self {
            Self::Fast => "q5",
            Self::Quality => "q8",
            Self::Compact => "q4",
            Self::Fp8 => "fp8",
        }
    }

    fn gguf_quant(self) -> &'static str {
        match self {
            Self::Fast => "Q5_K_M",
            Self::Quality => "Q8_0",
            Self::Compact => "Q4_K_M",
            Self::Fp8 => unreachable!("the fp8 tier ships safetensors, not GGUF"),
        }
    }

    /// Whether the tier ships the lightx2v four-step Lightning distill pair.
    /// The adapters were trained against the bf16 weights, so the Q5 and Q4
    /// tiers pull byte-identical files. The fp8 tier merges user adapters at
    /// load instead of shipping one.
    fn has_distill(self) -> bool {
        matches!(self, Self::Fast | Self::Compact)
    }
}

/// `(size, sha256)` for one shipped A14B artifact, read from the Hugging Face
/// API rather than from any local copy.
type FileFacts = (u64, &'static str);

fn a14b_expert_facts(task: A14bTask, tier: A14bTier, low_noise: bool) -> FileFacts {
    match (task, tier, low_noise) {
        (A14bTask::T2v, A14bTier::Fast, false) => (
            10_790_416_896,
            "fe704eb3541b09edb9cb675d58443bebccbabba0c9f5353305a6e01d9d9a2478",
        ),
        (A14bTask::T2v, A14bTier::Fast, true) => (
            10_790_416_896,
            "67242c61f055eb40c70fb421eb7dc1bdfde9c535f47c165fdfbf0b81b8b535dd",
        ),
        (A14bTask::T2v, A14bTier::Quality, false) => (
            15_404_970_496,
            "e15fecd4ce8f7effe1c4d9ab2c51d37b071bf3dd8b7f1b73e9fc27ac28f6820a",
        ),
        (A14bTask::T2v, A14bTier::Quality, true) => (
            15_404_970_496,
            "71574f62260f3ba305c31085c922a2b1b6672dbfdbe07717c066688ea56966fc",
        ),
        (A14bTask::I2v, A14bTier::Fast, false) => (
            10_792_055_296,
            "163e9e5ae7ff83a4598d55242c767384be8909749dc07240d388a24838e8bac6",
        ),
        (A14bTask::I2v, A14bTier::Fast, true) => (
            10_792_055_296,
            "c5affcba15576959fa6d63c18261275b6407c3edd5ad3d8ab1c420e96f9d05d0",
        ),
        (A14bTask::I2v, A14bTier::Quality, false) => (
            15_406_608_896,
            "619a66032c28e1b27882dfccc0bf93e51edb1491e8d4e4c6f291726abe4de8aa",
        ),
        (A14bTask::I2v, A14bTier::Quality, true) => (
            15_406_608_896,
            "029c7adc74de4f7804905c5e4fb9335d0862cd2fc37191df526aeac13b64425e",
        ),
        // The Q4_K_M pairs (#794). Sizes and hashes read from the Hugging
        // Face API (`lfs.oid`), like every row above.
        // VRAM peaks on an RTX 4090 at the 4-step Lightning defaults
        // (53f @ 16 fps): T2V 21,372 MiB at 832x480; I2V 19,580 MiB at
        // 720x480 (#794).
        (A14bTask::T2v, A14bTier::Compact, false) => (
            9_650_090_496,
            "e0c490c6e316fd91ff52034e4ca66b825717e33ff11624585c0ccfcb5d410c59",
        ),
        (A14bTask::T2v, A14bTier::Compact, true) => (
            9_650_090_496,
            "091a5bae02e14aa016bc9b10a7892efda4c629346b81c5dcebbe30ea2ac8923a",
        ),
        (A14bTask::I2v, A14bTier::Compact, false) => (
            9_651_728_896,
            "836250abfaa3411694e2c9cf3a0cc18265329d5156d81aa116d5366b0f8f02e7",
        ),
        (A14bTask::I2v, A14bTier::Compact, true) => (
            9_651_728_896,
            "e2f98d834af009d035c6b0918268f2eba0aa8a63025ce942277e2384d40b0866",
        ),
        // The Comfy-Org fp8-scaled safetensors pairs (#777), read from the
        // Hugging Face API like every row above. Comfy-Org over Kijai's `_KJ`
        // exports: same e4m3 ComfyUI marker contract the loader shipped
        // against in #747, ~1 GB smaller per expert, and the repository this
        // family's VAE/encoder rows already pull from.
        (A14bTask::T2v, A14bTier::Fp8, false) => (
            14_293_923_632,
            "cad711ae211c8b23455ec68cd6a190a33a3d874234a77eb57266d73f8f0e6c9f",
        ),
        (A14bTask::T2v, A14bTier::Fp8, true) => (
            14_293_923_632,
            "e71b96d7c82e638694c5e7fb98fac4bfb0e4ddc5fbbb4b1df40da8f0f1278a97",
        ),
        (A14bTask::I2v, A14bTier::Fp8, false) => (
            14_294_742_832,
            "6122e79d55e0f235698d11d657f3b196c5273c830da00b2b013c5a048d5e6a42",
        ),
        (A14bTask::I2v, A14bTier::Fp8, true) => (
            14_294_742_832,
            "5471a457b6ac404202a5fbe6c11595a3d5641fc766b00f38763f72303fffc21e",
        ),
    }
}

/// The four-step distill for one expert.
///
/// **Source matters here.** `lightx2v/Wan2.2-Distill-Loras` publishes files with
/// the same names and the same advertised purpose, but its I2V pair is not the
/// pure low-rank format: both halves carry hundreds of `.diff`/`.diff_b`
/// full-weight deltas and the low-noise half additionally targets
/// `cross_attn.{k,v,norm_k}_img` and `img_emb.proj.*` — the Wan 2.1 CLIP-vision
/// branch that Wan 2.2's own I2V checkpoint does not contain (its GGUF has zero
/// `_img` tensors). Kijai's `Wan22_Lightx2v` rank-64 I2V pair has the same
/// problem. The Comfy-Org repack below is the one that is 400 clean low-rank
/// pairs with an I64 alpha of 8 at rank 64 — headers parsed, not assumed.
fn a14b_distill_facts(task: A14bTask, low_noise: bool) -> (&'static str, FileFacts) {
    match (task, low_noise) {
        (A14bTask::T2v, false) => (
            "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
            (
                1_226_977_424,
                "698321cb86bd30c4af06c9b84e656a1048c8cb54e06d50694536fb5de37fde41",
            ),
        ),
        (A14bTask::T2v, true) => (
            "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors",
            (
                1_226_977_424,
                "ec95216e614b3c132c11bfb387b11feedf62163150ccc9068bca8a189771e75a",
            ),
        ),
        (A14bTask::I2v, false) => (
            "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
            (
                1_226_977_424,
                "d176c808d6fc461999b68e321efcb7501b20b8c3797523ed0df14f7d1deff11e",
            ),
        ),
        (A14bTask::I2v, true) => (
            "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
            (
                1_226_977_424,
                "024f21de095bc8fad9809ded3e9e49a2e170dcf27075da8145ba7d60d8aab7f9",
            ),
        ),
    }
}

/// A Wan 2.2 A14B mixture-of-experts manifest.
///
/// Both experts ship as separate files and exactly one is GPU-resident at a
/// time, so the VRAM demand is the max over the pair rather than their sum —
/// 10.8 GB at `:q5`, 15.4 GB at `:q8`, 9.65 GB at `:q4` (measured `:q4` total
/// peaks on an RTX 4090 at the 53f Lightning defaults: T2V 21,372 MiB at
/// 832x480, I2V 19,580 MiB at 720x480; #794). Disk is the sum.
///
/// Resolution defaults to 480p because that is the acceptance target for this
/// tier (#747: "81f 480p in ~1.5-3 min on a 4090"); 720p renders, but not in
/// that budget.
fn a14b_manifest(tier: A14bTier, task: A14bTask) -> ModelManifest {
    let expert_file = |low_noise: bool| {
        let (size_bytes, sha256) = a14b_expert_facts(task, tier, low_noise);
        let (hf_repo, hf_filename) = if tier == A14bTier::Fp8 {
            (
                "Comfy-Org/Wan_2.2_ComfyUI_Repackaged".to_string(),
                format!(
                    "split_files/diffusion_models/wan2.2_{}_{}_noise_14B_fp8_scaled.safetensors",
                    task.slug(),
                    if low_noise { "low" } else { "high" },
                ),
            )
        } else {
            let folder = if low_noise { "LowNoise" } else { "HighNoise" };
            (
                task.gguf_repo().to_string(),
                format!(
                    "{folder}/Wan2.2-{}-A14B-{folder}-{}.gguf",
                    task.slug().to_uppercase(),
                    tier.gguf_quant()
                ),
            )
        };
        ModelFile {
            hf_repo,
            hf_filename,
            component: if low_noise {
                ModelComponent::LowNoiseTransformer
            } else {
                ModelComponent::Transformer
            },
            size_bytes,
            gated: false,
            sha256: Some(sha256),
        }
    };
    let distill_file = |low_noise: bool| {
        let (filename, (size_bytes, sha256)) = a14b_distill_facts(task, low_noise);
        ModelFile {
            hf_repo: "Comfy-Org/Wan_2.2_ComfyUI_Repackaged".to_string(),
            hf_filename: format!("split_files/loras/{filename}"),
            component: if low_noise {
                ModelComponent::LowNoiseDistilledLora
            } else {
                ModelComponent::DistilledLora
            },
            size_bytes,
            gated: false,
            sha256: Some(sha256),
        }
    };

    let mut files = shared_wan_files();
    files.push(expert_file(false));
    files.push(expert_file(true));
    // A14B pairs with the 2.1 VAE, same file the 1.3B uses — it dedupes into
    // `shared/wan/`.
    files.push(ModelFile {
        hf_repo: "Comfy-Org/Wan_2.1_ComfyUI_repackaged".to_string(),
        hf_filename: "split_files/vae/wan_2.1_vae.safetensors".to_string(),
        component: ModelComponent::Vae,
        size_bytes: 253_815_318,
        gated: false,
        sha256: Some("2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b"),
    });
    if tier.has_distill() {
        files.push(distill_file(false));
        files.push(distill_file(true));
    }

    let (steps, guidance, tier_note) = match tier {
        // The lightx2v recipe. Guidance 1.0 is not a weak setting, it is the
        // switch that drops the unconditional pass entirely — one forward per
        // step, which is half of where the speed comes from.
        A14bTier::Fast | A14bTier::Compact => (4, 1.0, "4-step Lightning distill"),
        A14bTier::Quality => (20, WAN_A14B_QUALITY_GUIDANCE, "20-step, no distill"),
        A14bTier::Fp8 => (20, WAN_A14B_QUALITY_GUIDANCE, "20-step, fp8-scaled"),
    };
    let task_label = match task {
        A14bTask::T2v => "text-to-video",
        A14bTask::I2v => "image-to-video",
    };
    let container_label = match tier {
        A14bTier::Fp8 => "FP8-scaled".to_string(),
        _ => tier.gguf_quant().to_string(),
    };

    ModelManifest {
        name: format!("wan22-{}-a14b:{}", task.slug(), tier.tag()),
        family: "wan".to_string(),
        description: format!(
            "Wan 2.2 A14B {} {container_label} — 480p16 {task_label}, two-expert MoE ({tier_note})",
            task.slug().to_uppercase(),
        ),
        files,
        defaults: ManifestDefaults {
            steps,
            guidance,
            width: 832,
            height: 480,
            is_schnell: false,
            scheduler: None,
            negative_prompt: Some(WAN_DEFAULT_NEGATIVE_PROMPT.to_string()),
            // Default clip lengths are the measured 24 GB envelope, not the
            // checkpoint's trained 81: on an RTX 4090 the Q5 pair peaks at
            // 23,975 MiB rendering 53 frames at 832x480 (81 frames peaked at
            // 23.0 GB and then OOM'd), and the Q8 pair's ~4.6 GB larger
            // resident expert moves its edge to ~33 frames by the same
            // activation arithmetic. 53 is also the reference space's own
            // default duration. Larger cards can simply pass --frames 81.
            // The Q4 pair's resident expert is ~1.1 GB smaller than Q5's, so
            // 53 fits wherever Q5's does — confirmed: the Q4 T2V default
            // measured 21,372 MiB at 832x480 x 53f on an RTX 4090 (#794).
            frames: Some(match tier {
                A14bTier::Fast | A14bTier::Compact => 53,
                // The fp8 resident expert (~14.3 GB) sits between Q8's
                // 15.4 GB and Q5's 10.8 GB, so it inherits the quality tier's
                // conservative default until its own envelope is measured.
                A14bTier::Quality | A14bTier::Fp8 => 33,
            }),
            fps: Some(16),
            // Recorded from the task this manifest was assembled for, not
            // its name: the I2V pair denoises a 36-channel mask-plus-image
            // concat and cannot generate without the image; the T2V pair has
            // no conditioning input at all (#772).
            source_image: Some(match task {
                A14bTask::T2v => crate::types::SourceImageCapability::Unsupported,
                A14bTask::I2v => crate::types::SourceImageCapability::Required,
            }),
        },
        hidden: false,
    }
}

fn companion_manifests() -> Vec<ModelManifest> {
    let defaults = ManifestDefaults {
        steps: 0,
        guidance: 0.0,
        width: 0,
        height: 0,
        is_schnell: false,
        scheduler: None,
        negative_prompt: None,
        frames: None,
        fps: None,
        source_image: None,
    };

    vec![
        // CLIP-L — text encoder for SD1.5, SDXL, FLUX, Flux.2.
        ModelManifest {
            name: "clip-l".to_string(),
            family: "companion".to_string(),
            description: "OpenAI CLIP-L companion (single-file SD1.5/SDXL/FLUX checkpoints)"
                .to_string(),
            files: vec![
                ModelFile {
                    hf_repo: "openai/clip-vit-large-patch14".to_string(),
                    hf_filename: "model.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 1_710_540_580,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    hf_repo: "openai/clip-vit-large-patch14".to_string(),
                    hf_filename: "tokenizer.json".to_string(),
                    component: ModelComponent::ClipTokenizer,
                    size_bytes: 2_224_003,
                    gated: false,
                    sha256: None,
                },
            ],
            defaults: defaults.clone(),
            hidden: true,
        },
        // CLIP-G — second text encoder for SDXL.
        ModelManifest {
            name: "clip-g".to_string(),
            family: "companion".to_string(),
            description: "OpenCLIP ViT-bigG companion (single-file SDXL checkpoints)".to_string(),
            files: vec![ModelFile {
                hf_repo: "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k".to_string(),
                // `open_clip_pytorch_model.safetensors` 404s; the actual
                // OpenCLIP-format file is `open_clip_model.safetensors`
                // (10.16 GB).
                hf_filename: "open_clip_model.safetensors".to_string(),
                component: ModelComponent::Transformer,
                size_bytes: 10_158_382_892,
                gated: false,
                sha256: None,
            }],
            defaults: defaults.clone(),
            hidden: true,
        },
        // SDXL VAE — fp16-fix VAE used by every SDXL single-file checkpoint.
        ModelManifest {
            name: "sdxl-vae".to_string(),
            family: "companion".to_string(),
            description: "SDXL fp16-fix VAE companion".to_string(),
            files: vec![ModelFile {
                hf_repo: "madebyollin/sdxl-vae-fp16-fix".to_string(),
                hf_filename: "diffusion_pytorch_model.safetensors".to_string(),
                component: ModelComponent::Transformer,
                size_bytes: 334_643_268,
                gated: false,
                sha256: Some("1b909373b28f2137098b0fd9dbc6f97f8410854f31f84ddc9fa04b077b0ace2c"),
            }],
            defaults: defaults.clone(),
            hidden: true,
        },
        // SD1.5 VAE — ft-mse stability VAE used by SD1.5 single-file checkpoints.
        ModelManifest {
            name: "sd-vae-ft-mse".to_string(),
            family: "companion".to_string(),
            description: "Stability SD1.5 ft-mse VAE companion".to_string(),
            files: vec![ModelFile {
                hf_repo: "stabilityai/sd-vae-ft-mse".to_string(),
                hf_filename: "diffusion_pytorch_model.safetensors".to_string(),
                component: ModelComponent::Transformer,
                size_bytes: 334_643_276,
                gated: false,
                sha256: Some("a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815"),
            }],
            defaults: defaults.clone(),
            hidden: true,
        },
        // T5-XXL — shared text encoder for FLUX, Flux.2, LTX-Video, LTX-2.
        // The bf16 variant is the canonical full-precision companion;
        // FLUX single-file flows can pull this instead of bundling T5 in the
        // checkpoint.
        ModelManifest {
            name: "t5-v1_1-xxl".to_string(),
            family: "companion".to_string(),
            description: "T5-XXL bf16 companion (single-file FLUX / Flux.2 / LTX-Video)"
                .to_string(),
            files: vec![
                ModelFile {
                    hf_repo: "city96/t5-v1_1-xxl-encoder-bf16".to_string(),
                    hf_filename: "model.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 9_787_841_024,
                    gated: false,
                    sha256: None,
                },
                // Tokenizer must ride along: the catalog bridge resolves a
                // companion's ModelPaths and copies `paths.t5_tokenizer` into
                // the synthesized `cfg.t5_tokenizer`. Without this entry,
                // `cfg.t5_tokenizer` would stay None and FLUX/SD3/Flux.2
                // single-file loads bomb with "T5 tokenizer path required".
                ModelFile {
                    hf_repo: "lmz/mt5-tokenizers".to_string(),
                    hf_filename: "t5-v1_1-xxl.tokenizer.json".to_string(),
                    component: ModelComponent::T5Tokenizer,
                    size_bytes: 2_424_257,
                    gated: false,
                    sha256: None,
                },
            ],
            defaults: defaults.clone(),
            hidden: true,
        },
        // LTX-Video VAE — used by LTX-Video single-file Civitai checkpoints.
        // Civitai fine-tunes are transformer-only; the VAE is pulled separately
        // from the same HF repo mold uses for manifest-based LTX-Video models.
        ModelManifest {
            name: "ltx-video-vae".to_string(),
            family: "companion".to_string(),
            description: "LTX-Video VAE companion (single-file Civitai LTX-Video checkpoints)"
                .to_string(),
            files: vec![ModelFile {
                hf_repo: "Lightricks/LTX-Video-0.9.5".to_string(),
                hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
                // Companion manifests are download-only; the engine resolves
                // the VAE from paths.transformer. Same pattern as sdxl-vae /
                // flux-vae / sd-vae-ft-mse companions.
                component: ModelComponent::Transformer,
                size_bytes: 2_493_855_612,
                gated: false,
                sha256: None,
            }],
            defaults: defaults.clone(),
            hidden: true,
        },
        ModelManifest {
            name: "ltx2-vae".to_string(),
            family: "companion".to_string(),
            description: "LTX-2 video VAE companion for transformer-only checkpoints"
                .to_string(),
            files: vec![ModelFile {
                hf_repo: "Lightricks/LTX-2".to_string(),
                hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
                component: ModelComponent::Transformer,
                size_bytes: 2_444_982_370,
                gated: false,
                sha256: Some(
                    "107cc359e3c4bce18c53d98686f4b3fe10c4207b6665d89b38b0741270514bfb",
                ),
            }],
            defaults: defaults.clone(),
            hidden: true,
        },
        ModelManifest {
            name: "ltx2.3-vae".to_string(),
            family: "companion".to_string(),
            description: "LTX-2.3 video VAE companion for transformer-only checkpoints"
                .to_string(),
            files: vec![ModelFile {
                hf_repo: "Kijai/LTX2.3_comfy".to_string(),
                hf_filename: "vae/LTX23_video_vae_bf16.safetensors".to_string(),
                component: ModelComponent::Transformer,
                size_bytes: 1_452_258_578,
                gated: false,
                sha256: Some(
                    "01ea62d09bc139f95c5dee7b5c062ad6a3e6cd8be910a1983ac02e7eb5b8ee3b",
                ),
            }],
            defaults: defaults.clone(),
            hidden: true,
        },
        ModelManifest {
            name: "ltx2.3-text-projection".to_string(),
            family: "companion".to_string(),
            description: "LTX-2.3 Gemma hidden-state projection companion for diffusion-only checkpoints".to_string(),
            files: vec![ModelFile {
                hf_repo: "Kijai/LTX2.3_comfy".to_string(),
                hf_filename: "text_encoders/ltx-2.3_text_projection_bf16.safetensors".to_string(),
                component: ModelComponent::Transformer,
                size_bytes: 2_312_149_072,
                gated: false,
                sha256: Some(
                    "911d59bb4cb7708179c9a0045ea0fe41212ecfb77aed3a02702b7c0a8274911f",
                ),
            }],
            defaults: defaults.clone(),
            hidden: true,
        },
        // Wan UMT5-XXL — shared text encoder for every Wan 2.1/2.2 variant.
        // Single-file catalog checkpoints (Civitai/HF transformer-only) pull
        // this instead of bundling the 11 GB encoder in the checkpoint.
        ModelManifest {
            name: "wan-umt5".to_string(),
            family: "companion".to_string(),
            description: "Wan UMT5-XXL text encoder companion (single-file Wan checkpoints)"
                .to_string(),
            files: vec![
                // TextEncoder (not Transformer) on purpose: like flux2-te,
                // a pure text-encoder companion declares TextEncoder shards
                // so `paths_from_downloads` falls back to them, and the
                // shared-component storage path dedupes with the primary
                // Wan manifests' own copy under `shared/wan/`.
                ModelFile {
                    hf_repo: "Comfy-Org/Wan_2.1_ComfyUI_repackaged".to_string(),
                    hf_filename: "split_files/text_encoders/umt5_xxl_fp16.safetensors"
                        .to_string(),
                    component: ModelComponent::TextEncoder,
                    size_bytes: 11_366_399_385,
                    gated: false,
                    sha256: Some(
                        "7b8850f1961e1cf8a77cca4c964a358d303f490833c6c087d0cff4b2f99db2af",
                    ),
                },
                // Tokenizer rides along so the catalog bridge's synthesized
                // config gets a tokenizer path (same reason as t5-v1_1-xxl).
                ModelFile {
                    hf_repo: "google/umt5-xxl".to_string(),
                    hf_filename: "tokenizer.json".to_string(),
                    component: ModelComponent::TextTokenizer,
                    size_bytes: 16_853_013,
                    gated: false,
                    sha256: Some(
                        "af904105ce1071b1202bba0059a841f4a7b85b48b6ec179c4948e3483476e0dd",
                    ),
                },
            ],
            defaults: defaults.clone(),
            hidden: true,
        },
        // Wan 2.1 VAE — used by every 1.3B/14B/A14B Wan model.
        ModelManifest {
            name: "wan21-vae".to_string(),
            family: "companion".to_string(),
            description: "Wan 2.1 video VAE companion (1.3B/14B/A14B checkpoints)".to_string(),
            files: vec![ModelFile {
                hf_repo: "Comfy-Org/Wan_2.1_ComfyUI_repackaged".to_string(),
                hf_filename: "split_files/vae/wan_2.1_vae.safetensors".to_string(),
                component: ModelComponent::Transformer,
                size_bytes: 253_815_318,
                gated: false,
                sha256: Some("2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b"),
            }],
            defaults: defaults.clone(),
            hidden: true,
        },
        // Wan 2.2 VAE — TI2V-5B's higher-compression (16x16x4, 48ch) VAE.
        ModelManifest {
            name: "wan22-vae".to_string(),
            family: "companion".to_string(),
            description: "Wan 2.2 video VAE companion (TI2V-5B checkpoints)".to_string(),
            files: vec![ModelFile {
                hf_repo: "Comfy-Org/Wan_2.2_ComfyUI_Repackaged".to_string(),
                hf_filename: "split_files/vae/wan2.2_vae.safetensors".to_string(),
                component: ModelComponent::Transformer,
                size_bytes: 1_409_400_960,
                gated: false,
                sha256: Some("e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156"),
            }],
            defaults: defaults.clone(),
            hidden: true,
        },
        // FLUX VAE — used by FLUX single-file checkpoints. BFL gating
        // applies; the catalog UI shows the "needs token" badge.
        ModelManifest {
            name: "flux-vae".to_string(),
            family: "companion".to_string(),
            description: "Black Forest Labs FLUX VAE companion (single-file FLUX)".to_string(),
            files: vec![ModelFile {
                hf_repo: "black-forest-labs/FLUX.1-schnell".to_string(),
                hf_filename: "ae.safetensors".to_string(),
                component: ModelComponent::Transformer,
                size_bytes: 335_304_388,
                gated: true,
                sha256: Some("afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38"),
            }],
            defaults: defaults.clone(),
            hidden: true,
        },
        // Flux.2 text encoder — Qwen3 4B (two safetensors shards) +
        // tokenizer. Pulled from the Apache-2.0 Klein-4B repo so single-file
        // Civitai Flux.2 checkpoints have a runnable text encoder.
        ModelManifest {
            name: "flux2-te".to_string(),
            family: "companion".to_string(),
            description: "Flux.2 Qwen3 text encoder + tokenizer companion (single-file Flux.2)"
                .to_string(),
            files: vec![
                ModelFile {
                    hf_repo: "black-forest-labs/FLUX.2-klein-4B".to_string(),
                    hf_filename: "text_encoder/model-00001-of-00002.safetensors".to_string(),
                    component: ModelComponent::TextEncoder,
                    size_bytes: 4_967_215_360,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    hf_repo: "black-forest-labs/FLUX.2-klein-4B".to_string(),
                    hf_filename: "text_encoder/model-00002-of-00002.safetensors".to_string(),
                    component: ModelComponent::TextEncoder,
                    size_bytes: 3_077_766_632,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    hf_repo: "black-forest-labs/FLUX.2-klein-4B".to_string(),
                    hf_filename: "tokenizer/tokenizer.json".to_string(),
                    component: ModelComponent::TextTokenizer,
                    size_bytes: 11_422_654,
                    gated: false,
                    sha256: None,
                },
            ],
            defaults: defaults.clone(),
            hidden: true,
        },
        // Flux.2 text encoder for Klein-9B / FLUX.2-Dev — Qwen3 8B (four
        // safetensors shards) + tokenizer. Sourced from the gated
        // FLUX.2-klein-9B repo; users need a BFL token to pull it.
        ModelManifest {
            name: "flux2-te-9b".to_string(),
            family: "companion".to_string(),
            description:
                "Flux.2 Qwen3-8B text encoder + tokenizer companion (single-file Klein-9B / Dev)"
                    .to_string(),
            files: vec![
                ModelFile {
                    hf_repo: "black-forest-labs/FLUX.2-klein-9B".to_string(),
                    hf_filename: "text_encoder/model-00001-of-00004.safetensors".to_string(),
                    component: ModelComponent::TextEncoder,
                    size_bytes: 4_902_257_696,
                    gated: true,
                    sha256: None,
                },
                ModelFile {
                    hf_repo: "black-forest-labs/FLUX.2-klein-9B".to_string(),
                    hf_filename: "text_encoder/model-00002-of-00004.safetensors".to_string(),
                    component: ModelComponent::TextEncoder,
                    size_bytes: 4_915_960_368,
                    gated: true,
                    sha256: None,
                },
                ModelFile {
                    hf_repo: "black-forest-labs/FLUX.2-klein-9B".to_string(),
                    hf_filename: "text_encoder/model-00003-of-00004.safetensors".to_string(),
                    component: ModelComponent::TextEncoder,
                    size_bytes: 4_983_068_496,
                    gated: true,
                    sha256: None,
                },
                ModelFile {
                    hf_repo: "black-forest-labs/FLUX.2-klein-9B".to_string(),
                    hf_filename: "text_encoder/model-00004-of-00004.safetensors".to_string(),
                    component: ModelComponent::TextEncoder,
                    size_bytes: 1_580_230_264,
                    gated: true,
                    sha256: None,
                },
                ModelFile {
                    hf_repo: "black-forest-labs/FLUX.2-klein-9B".to_string(),
                    hf_filename: "tokenizer/tokenizer.json".to_string(),
                    component: ModelComponent::TextTokenizer,
                    size_bytes: 11_422_654,
                    gated: true,
                    sha256: None,
                },
            ],
            defaults: defaults.clone(),
            hidden: true,
        },
        // Flux.2 VAE — Klein-specific, ~168 MB. Distinct from `flux-vae`
        // (FLUX.1's 335 MB ae.safetensors) which is incompatible.
        ModelManifest {
            name: "flux2-vae".to_string(),
            family: "companion".to_string(),
            description: "Flux.2 Klein VAE companion (single-file Flux.2)".to_string(),
            files: vec![ModelFile {
                hf_repo: "black-forest-labs/FLUX.2-klein-4B".to_string(),
                hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
                component: ModelComponent::Transformer,
                size_bytes: 168_120_878,
                gated: false,
                sha256: None,
            }],
            defaults: defaults.clone(),
            hidden: true,
        },
        // Z-Image shared runtime assets — Qwen3 BF16 shards, tokenizer, and
        // VAE used by single-file Civitai Z-Image checkpoints. The primary
        // checkpoint is the transformer only; catalog resolution fills
        // cfg.text_encoder_files / cfg.text_tokenizer / cfg.vae from this
        // companion.
        ModelManifest {
            name: "z-image-te".to_string(),
            family: "companion".to_string(),
            description: "Z-Image Qwen3 text encoder + tokenizer companion (single-file Z-Image)"
                .to_string(),
            files: shared_zimage_files()
                .into_iter()
                .filter(|file| {
                    matches!(
                        file.component,
                        ModelComponent::TextEncoder
                            | ModelComponent::TextTokenizer
                            | ModelComponent::Vae
                    )
                })
                .collect(),
            defaults: defaults.clone(),
            hidden: true,
        },
        // Gemma 3 12B text encoder + tokenizer companion for LTX-2 / LTX-2.3
        // single-file Civitai checkpoints. The combined LTX-2 .safetensors
        // bundles transformer + VAE only; the runtime (`ltx2/assets.rs`)
        // requires the Gemma TE separately and reads its directory off the
        // first entry in `paths.text_encoder_files`. Same gated repo + file
        // set the manifest LTX-2 models pull, so a user with the Gemma TE
        // already installed for `ltx-2-19b-distilled:fp8` (etc.) gets cv:*
        // installs essentially for free via the HF cache.
        ModelManifest {
            name: "ltx2-te".to_string(),
            family: "companion".to_string(),
            description: "LTX-2 Gemma-3-12B text encoder + tokenizer companion (single-file LTX-2)"
                .to_string(),
            files: shared_ltx2_files(),
            defaults: defaults.clone(),
            hidden: true,
        },
        ModelManifest {
            name: "qwen-image-runtime".to_string(),
            family: "companion".to_string(),
            description: "Qwen-Image VAE, text encoder, and tokenizer companion (single-file Qwen-Image)"
                .to_string(),
            files: shared_qwen_image_base_files(),
            defaults: defaults.clone(),
            hidden: true,
        },
        ModelManifest {
            name: "wuerstchen-runtime".to_string(),
            family: "companion".to_string(),
            description:
                "Wuerstchen decoder, VQGAN, CLIP encoders, and tokenizers companion (single-file Wuerstchen)"
                    .to_string(),
            files: vec![
                ModelFile {
                    hf_repo: "warp-ai/wuerstchen".to_string(),
                    hf_filename: "decoder/diffusion_pytorch_model.safetensors".to_string(),
                    component: ModelComponent::Decoder,
                    size_bytes: 4_221_568_336,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    hf_repo: "warp-ai/wuerstchen".to_string(),
                    hf_filename: "vqgan/diffusion_pytorch_model.safetensors".to_string(),
                    component: ModelComponent::Vae,
                    size_bytes: 73_639_568,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    hf_repo: "warp-ai/wuerstchen-prior".to_string(),
                    hf_filename: "text_encoder/model.safetensors".to_string(),
                    component: ModelComponent::ClipEncoder2,
                    size_bytes: 2_772_149_276,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    hf_repo: "warp-ai/wuerstchen-prior".to_string(),
                    hf_filename: "tokenizer/tokenizer.json".to_string(),
                    component: ModelComponent::ClipTokenizer2,
                    size_bytes: 2_224_091,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    hf_repo: "warp-ai/wuerstchen".to_string(),
                    hf_filename: "text_encoder/model.safetensors".to_string(),
                    component: ModelComponent::ClipEncoder,
                    size_bytes: 1_411_983_168,
                    gated: false,
                    sha256: None,
                },
                ModelFile {
                    hf_repo: "warp-ai/wuerstchen".to_string(),
                    hf_filename: "tokenizer/tokenizer.json".to_string(),
                    component: ModelComponent::ClipTokenizer,
                    size_bytes: 2_224_119,
                    gated: false,
                    sha256: None,
                },
            ],
            defaults,
            hidden: true,
        },
    ]
}

fn upscaler_manifests() -> Vec<ModelManifest> {
    let defaults = ManifestDefaults {
        steps: 0,
        guidance: 0.0,
        width: 0,
        height: 0,
        is_schnell: false,
        scheduler: None,
        negative_prompt: None,
        frames: None,
        fps: None,
        source_image: None,
    };

    vec![
        ModelManifest {
            name: "real-esrgan-x4plus:fp16".to_string(),
            family: "upscaler".to_string(),
            description: "Real-ESRGAN x4+ FP16 — high quality 4x upscaler (32MB)".to_string(),
            files: vec![ModelFile {
                hf_repo: "hlky/RealESRGAN_x4plus".to_string(),
                hf_filename: "diffusion_pytorch_model.fp16.safetensors".to_string(),
                component: ModelComponent::Upscaler,
                size_bytes: 33_461_662,
                gated: false,
                sha256: None,
            }],
            defaults: defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "real-esrgan-x4plus:fp32".to_string(),
            family: "upscaler".to_string(),
            description: "Real-ESRGAN x4+ FP32 — high quality 4x upscaler (64MB)".to_string(),
            files: vec![ModelFile {
                hf_repo: "hlky/RealESRGAN_x4plus".to_string(),
                hf_filename: "diffusion_pytorch_model.safetensors".to_string(),
                component: ModelComponent::Upscaler,
                size_bytes: 66_857_868,
                gated: false,
                sha256: None,
            }],
            defaults: defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "real-esrgan-x4plus-anime:fp16".to_string(),
            family: "upscaler".to_string(),
            description: "Real-ESRGAN x4+ Anime FP16 — anime-optimized 4x upscaler (8.5MB)"
                .to_string(),
            files: vec![ModelFile {
                hf_repo: "hlky/RealESRGAN_x4plus_anime_6B".to_string(),
                hf_filename: "diffusion_pytorch_model.fp16.safetensors".to_string(),
                component: ModelComponent::Upscaler,
                size_bytes: 8_953_054,
                gated: false,
                sha256: None,
            }],
            defaults: defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "real-esrgan-x4plus-anime:fp32".to_string(),
            family: "upscaler".to_string(),
            description: "Real-ESRGAN x4+ Anime FP32 — anime-optimized 4x upscaler (17MB)"
                .to_string(),
            files: vec![ModelFile {
                hf_repo: "hlky/RealESRGAN_x4plus_anime_6B".to_string(),
                hf_filename: "diffusion_pytorch_model.safetensors".to_string(),
                component: ModelComponent::Upscaler,
                size_bytes: 17_888_804,
                gated: false,
                sha256: None,
            }],
            defaults: defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "real-esrgan-anime-v3:fp32".to_string(),
            family: "upscaler".to_string(),
            description: "Real-ESRGAN Anime Video v3 FP32 — fast anime 4x upscaler (2.4MB)"
                .to_string(),
            files: vec![ModelFile {
                hf_repo: "wkrettek/real-esrgan-models".to_string(),
                hf_filename: "realesr_animevideov3.safetensors".to_string(),
                component: ModelComponent::Upscaler,
                size_bytes: 2_489_904,
                gated: false,
                sha256: None,
            }],
            defaults: defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "real-esrgan-x2plus:fp16".to_string(),
            family: "upscaler".to_string(),
            description: "Real-ESRGAN x2+ FP16 — high quality 2x upscaler (32MB)".to_string(),
            files: vec![ModelFile {
                hf_repo: "hlky/RealESRGAN_x2plus".to_string(),
                hf_filename: "diffusion_pytorch_model.fp16.safetensors".to_string(),
                component: ModelComponent::Upscaler,
                size_bytes: 33_472_030,
                gated: false,
                sha256: None,
            }],
            defaults: defaults.clone(),
            hidden: false,
        },
        ModelManifest {
            name: "real-esrgan-x2plus:fp32".to_string(),
            family: "upscaler".to_string(),
            description: "Real-ESRGAN x2+ FP32 — high quality 2x upscaler (64MB)".to_string(),
            files: vec![ModelFile {
                hf_repo: "hlky/RealESRGAN_x2plus".to_string(),
                hf_filename: "diffusion_pytorch_model.safetensors".to_string(),
                component: ModelComponent::Upscaler,
                size_bytes: 66_878_604,
                gated: false,
                sha256: None,
            }],
            defaults,
            hidden: false,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The engine's absence fallback (`wan/pipeline.rs::resolve_negative_prompt`)
    /// and the advertised default must be one value: every wan manifest
    /// plants exactly the family constant. Only wan gets an engine-side
    /// substitution — wuerstchen's manifest negative is a CLI-layer default
    /// (`resolved_model_config`) the server never applies, so the helper must
    /// keep returning None for it rather than promising HTTP behavior that
    /// does not exist.
    #[test]
    fn family_default_negative_prompt_matches_the_wan_manifests() {
        assert_eq!(
            default_negative_prompt_for_family("wan"),
            Some(WAN_DEFAULT_NEGATIVE_PROMPT)
        );
        for family in [
            "flux",
            "ltx2",
            "ltx-video",
            "sdxl",
            "sd15",
            "qwen-image",
            "wuerstchen",
        ] {
            assert_eq!(default_negative_prompt_for_family(family), None);
        }
        let mut wan_manifests = 0;
        for manifest in known_manifests().iter().filter(|m| m.family == "wan") {
            wan_manifests += 1;
            assert_eq!(
                manifest.defaults.negative_prompt.as_deref(),
                Some(WAN_DEFAULT_NEGATIVE_PROMPT),
                "wan manifest {} disagrees with the engine fallback",
                manifest.name
            );
        }
        assert!(wan_manifests > 0, "wan manifests exist");
    }

    #[test]
    fn camera_control_presets_have_hidden_auxiliary_download_manifests() {
        let visible_names = visible_manifests()
            .map(|manifest| manifest.name.as_str())
            .collect::<std::collections::HashSet<_>>();

        for preset in crate::ltx2_camera::LTX2_CAMERA_CONTROLS {
            let manifest = find_manifest(preset.download_model)
                .unwrap_or_else(|| panic!("missing manifest for {}", preset.download_model));
            assert!(manifest.hidden);
            assert!(manifest.is_auxiliary());
            assert!(!manifest.is_generation_model());
            assert!(!visible_names.contains(preset.download_model));
            assert_eq!(manifest.family, "ltx2-camera-control");
            assert_eq!(manifest.files.len(), 1);

            let file = &manifest.files[0];
            assert_eq!(file.hf_repo, preset.hf_repo);
            assert_eq!(file.hf_filename, preset.hf_filename);
            assert_eq!(file.component, ModelComponent::Transformer);
            assert_eq!(file.size_bytes, preset.size_bytes);
            assert_eq!(file.sha256, Some(preset.sha256));
            assert!(!file.gated);
        }
    }

    #[test]
    fn storage_path_transformer_is_model_specific() {
        let manifest = find_manifest("flux-schnell:q8").unwrap();
        let transformer_file = manifest
            .files
            .iter()
            .find(|f| f.component == ModelComponent::Transformer)
            .unwrap();
        let path = storage_path(manifest, transformer_file);
        assert!(
            path.starts_with("flux-schnell-q8"),
            "transformer should be under model-specific dir, got: {}",
            path.display()
        );
        assert!(path.to_string_lossy().contains("flux1-schnell-Q8_0.gguf"));
    }

    #[test]
    fn storage_path_shared_components_under_family() {
        let manifest = find_manifest("flux-schnell:q8").unwrap();
        for file in &manifest.files {
            let path = storage_path(manifest, file);
            match file.component {
                ModelComponent::Transformer | ModelComponent::TransformerShard => {
                    assert!(path.starts_with("flux-schnell-q8"));
                }
                _ => {
                    assert!(
                        path.starts_with("shared/flux"),
                        "shared component {:?} should be under shared/flux, got: {}",
                        file.component,
                        path.display()
                    );
                }
            }
        }
    }

    #[test]
    fn companion_shared_components_route_to_canonical_owning_family() {
        // Catalog single-file flows resolve a companion's ModelPaths and
        // copy `paths.t5_tokenizer` / `paths.clip_tokenizer` into the
        // synthesized config. Companions must route their *shared* (non
        // model-specific) components to the same on-disk bucket the owning
        // family uses, so an already-installed flux-schnell's tokenizer is
        // found by ModelPaths::resolve("t5-v1_1-xxl", config) without a
        // re-download. The companion's primary weight (Transformer) still
        // lives under <companion>/<file> per is_model_specific_component
        // (intentional and preserved).
        // t5-v1_1-xxl ships the tokenizer; clip-l ships its tokenizer.
        // clip-g/sdxl-vae/etc are Transformer-only companions — the
        // routing branch they take is is_model_specific_component, so they
        // don't exercise this path. Cover only companions with at least
        // one shared file.
        let cases = [
            ("t5-v1_1-xxl", "shared/flux"),
            ("clip-l", "shared/flux"),
            ("z-image-te", "shared/z-image"),
        ];
        for (companion, expected_prefix) in cases {
            let manifest = find_manifest(companion)
                .unwrap_or_else(|| panic!("companion manifest '{companion}' not registered"));
            let mut shared_files = 0;
            for file in &manifest.files {
                if is_model_specific_component(file.component) {
                    continue;
                }
                shared_files += 1;
                let path = storage_path(manifest, file);
                assert!(
                    path.starts_with(expected_prefix),
                    "companion '{}' shared file '{}' routed to '{}', expected '{}'",
                    companion,
                    file.hf_filename,
                    path.display(),
                    expected_prefix
                );
            }
            assert!(
                shared_files > 0,
                "companion '{companion}' has no shared (non-Transformer) files — \
                 the canonical-routing test is meaningless for it"
            );
        }
    }

    #[test]
    fn t5_companion_includes_tokenizer_so_catalog_bridge_finds_it() {
        // Regression: cv:* FLUX single-file loads bombed with "T5 tokenizer
        // path required for FLUX models" because the t5-v1_1-xxl companion
        // declared only the encoder weights — copy_catalog_companion saw
        // paths.t5_tokenizer == None and never set cfg.t5_tokenizer.
        let manifest = find_manifest("t5-v1_1-xxl").unwrap();
        let has_tokenizer = manifest
            .files
            .iter()
            .any(|f| f.component == ModelComponent::T5Tokenizer);
        assert!(
            has_tokenizer,
            "t5-v1_1-xxl companion must declare a T5Tokenizer file"
        );
    }

    #[test]
    fn zimage_text_encoder_companion_includes_shards_and_tokenizer() {
        // Regression: cv:* Z-Image loads declared a required `z-image-te`
        // companion, but there was no synthetic manifest for the auto-pull
        // and catalog resolution failed with "missing required component".
        let manifest = find_manifest("z-image-te").unwrap();
        let text_encoder_shards = manifest
            .files
            .iter()
            .filter(|f| f.component == ModelComponent::TextEncoder)
            .count();
        let has_tokenizer = manifest
            .files
            .iter()
            .any(|f| f.component == ModelComponent::TextTokenizer);
        let has_vae = manifest
            .files
            .iter()
            .any(|f| f.component == ModelComponent::Vae);

        assert_eq!(
            text_encoder_shards, 3,
            "z-image-te must declare all three Qwen3 text encoder shards"
        );
        assert!(
            has_tokenizer,
            "z-image-te companion must declare the Qwen3 tokenizer"
        );
        assert!(has_vae, "z-image-te companion must declare the Z-Image VAE");
    }

    #[test]
    fn qwen_shared_components_do_not_collapse_base_and_2512() {
        let base_manifest = find_manifest("qwen-image:q4").unwrap();
        let base_encoder = base_manifest
            .files
            .iter()
            .find(|f| f.component == ModelComponent::TextEncoder)
            .unwrap();
        let base_path = storage_path(base_manifest, base_encoder);
        assert!(base_path.starts_with("shared/qwen-image-base"));

        let q2512_manifest = find_manifest("qwen-image-2512:q4").unwrap();
        let q2512_encoder = q2512_manifest
            .files
            .iter()
            .find(|f| f.component == ModelComponent::TextEncoder)
            .unwrap();
        let q2512_path = storage_path(q2512_manifest, q2512_encoder);
        assert!(q2512_path.starts_with("shared/qwen-image"));
        assert_ne!(base_path, q2512_path);

        let qedit_manifest = find_manifest("qwen-image-edit-2511:q4").unwrap();
        let qedit_encoder = qedit_manifest
            .files
            .iter()
            .find(|f| f.component == ModelComponent::TextEncoder)
            .unwrap();
        let qedit_path = storage_path(qedit_manifest, qedit_encoder);
        assert!(qedit_path.starts_with("shared/qwen-image-edit"));
        assert_ne!(base_path, qedit_path);
        assert_ne!(q2512_path, qedit_path);
    }

    #[test]
    fn storage_path_zimage_preserves_nested_filenames() {
        let manifest = find_manifest("z-image-turbo:q8").unwrap();
        let encoder_file = manifest
            .files
            .iter()
            .find(|f| f.component == ModelComponent::TextEncoder)
            .unwrap();
        let path = storage_path(manifest, encoder_file);
        // Nested HF filename like "text_encoder/model-00001-of-00003.safetensors"
        // should be preserved under shared/z-image/
        assert!(
            path.starts_with("shared/z-image"),
            "got: {}",
            path.display()
        );
        assert!(path.to_string_lossy().contains("text_encoder/"));
    }

    #[test]
    fn storage_path_sdxl_transformer_is_model_specific() {
        let manifest = find_manifest("sdxl-base:fp16").unwrap();
        let transformer_file = manifest
            .files
            .iter()
            .find(|f| f.component == ModelComponent::Transformer)
            .unwrap();
        let path = storage_path(manifest, transformer_file);
        assert!(
            path.starts_with("sdxl-base-fp16"),
            "got: {}",
            path.display()
        );
    }

    #[test]
    fn storage_path_colon_sanitized() {
        let manifest = find_manifest("flux-dev:q4").unwrap();
        let transformer_file = manifest
            .files
            .iter()
            .find(|f| f.component == ModelComponent::Transformer)
            .unwrap();
        let path = storage_path(manifest, transformer_file);
        assert!(
            !path.to_string_lossy().contains(':'),
            "colons should be replaced with dashes"
        );
        assert!(path.starts_with("flux-dev-q4"));
    }

    #[test]
    fn resolve_name_with_tag() {
        assert_eq!(resolve_model_name("flux-dev:q4"), "flux-dev:q4");
        assert_eq!(resolve_model_name("flux-schnell:q8"), "flux-schnell:q8");
    }

    #[test]
    fn resolve_name_default_tag() {
        assert_eq!(resolve_model_name("flux-schnell"), "flux-schnell:q8");
        assert_eq!(resolve_model_name("flux-dev"), "flux-dev:q8");
        assert_eq!(
            resolve_model_name("qwen-image-edit"),
            "qwen-image-edit-2511:q8"
        );
        // SDXL models default to :fp16
        assert_eq!(resolve_model_name("sdxl-base"), "sdxl-base:fp16");
        assert_eq!(resolve_model_name("sdxl-turbo"), "sdxl-turbo:fp16");
        assert_eq!(resolve_model_name("dreamshaper-xl"), "dreamshaper-xl:fp16");
        assert_eq!(resolve_model_name("ltx-2.3-22b-dev"), "ltx-2.3-22b-dev:fp8");
        assert_eq!(
            resolve_model_name("ltx-2.3-22b-distilled"),
            "ltx-2.3-22b-distilled:fp8"
        );
    }

    #[test]
    fn resolve_name_legacy_format() {
        assert_eq!(resolve_model_name("flux-dev-q4"), "flux-dev:q4");
        assert_eq!(resolve_model_name("flux-dev-q8"), "flux-dev:q8");
        assert_eq!(
            resolve_model_name("ltx-2.3-22b-dev-fp8"),
            "ltx-2.3-22b-dev:fp8"
        );
    }

    #[test]
    fn find_known_manifests() {
        assert!(find_manifest("flux-schnell").is_some());
        assert!(find_manifest("flux-dev:q4").is_some());
        assert!(find_manifest("flux-dev-q4").is_some());
        assert!(find_manifest("flux-krea").is_some());
        assert!(find_manifest("flux-schnell:q6").is_some());
        assert!(find_manifest("flux-dev:q6").is_some());
        assert!(find_manifest("flux-krea:q4").is_some());
        assert!(find_manifest("flux-krea:q6").is_some());
        // SD1.5 models
        assert!(find_manifest("sd15").is_some());
        assert!(find_manifest("sd15:fp16").is_some());
        assert!(find_manifest("dreamshaper-v8").is_some());
        assert!(find_manifest("dreamshaper-v8:fp16").is_some());
        assert!(find_manifest("realistic-vision-v5").is_some());
        assert!(find_manifest("realistic-vision-v5:fp16").is_some());
        // SD3 models
        assert!(find_manifest("sd3.5-large:q8").is_some());
        assert!(find_manifest("sd3.5-large:q4").is_some());
        assert!(find_manifest("sd3.5-large-turbo:q8").is_some());
        assert!(find_manifest("sd3.5-medium:q8").is_some());
        // Flux.2 models
        assert!(find_manifest("flux2-klein:bf16").is_some());
        assert!(find_manifest("nonexistent").is_none());
    }

    #[test]
    fn find_manifest_returns_correct_result() {
        let manifest = find_manifest("flux-schnell:q8").unwrap();
        assert_eq!(manifest.name, "flux-schnell:q8");
        assert_eq!(manifest.family, "flux");
    }

    #[test]
    fn find_manifest_exact_unknown_returns_none() {
        assert!(find_manifest_exact("totally-unknown-model:q99").is_none());
    }

    #[test]
    fn flux2_klein_defaults() {
        let manifest = find_manifest("flux2-klein:bf16").unwrap();
        assert_eq!(manifest.name, "flux2-klein:bf16");
        assert_eq!(manifest.family, "flux2");
        assert_eq!(manifest.defaults.steps, 4);
        assert!((manifest.defaults.guidance - 1.0).abs() < 0.01);
        assert_eq!(manifest.defaults.width, 1024);
        assert_eq!(manifest.defaults.height, 1024);
    }

    #[test]
    fn flux2_dev_manifest_is_exact_and_collision_free() {
        let manifest = find_manifest_exact("flux2-dev:bf16").unwrap();
        assert_eq!(manifest.family, "flux2");
        assert_eq!(manifest.defaults.steps, 50);
        assert_eq!(manifest.defaults.guidance, 4.0);
        assert!(manifest.is_gated());
        assert!(!manifest.hidden);
        assert_eq!(manifest.total_size_bytes(), 103_364_356_713);
        assert_eq!(
            manifest
                .files
                .iter()
                .filter(|file| file.component == ModelComponent::TransformerShard)
                .count(),
            7
        );
        assert_eq!(
            manifest
                .files
                .iter()
                .filter(|file| file.component == ModelComponent::TextEncoder)
                .count(),
            8
        );
        assert!(manifest
            .files
            .iter()
            .all(|file| file.hf_repo == "black-forest-labs/FLUX.2-dev"));
        assert!(manifest.files.iter().all(|file| file.sha256.is_some()));
        assert!(!manifest
            .files
            .iter()
            .any(|file| file.hf_filename == "flux2-dev.safetensors"));

        for file in manifest
            .files
            .iter()
            .filter(|file| !is_model_specific_component(file.component))
        {
            assert!(storage_path(manifest, file).starts_with("shared/flux2-dev"));
        }
        let klein = find_manifest_exact("flux2-klein:bf16").unwrap();
        let dev_vae = manifest
            .files
            .iter()
            .find(|file| file.component == ModelComponent::Vae)
            .unwrap();
        let klein_vae = klein
            .files
            .iter()
            .find(|file| file.component == ModelComponent::Vae)
            .unwrap();
        assert_ne!(
            storage_path(manifest, dev_vae),
            storage_path(klein, klein_vae)
        );
    }

    #[test]
    fn sd3_resolves_to_q8() {
        let manifest = find_manifest("sd3.5-large").unwrap();
        assert_eq!(manifest.name, "sd3.5-large:q8");
        assert_eq!(manifest.family, "sd3");
        assert_eq!(manifest.defaults.steps, 28);
        assert!((manifest.defaults.guidance - 4.0).abs() < 0.01);
    }

    #[test]
    fn sd3_turbo_defaults() {
        let manifest = find_manifest("sd3.5-large-turbo:q8").unwrap();
        assert_eq!(manifest.defaults.steps, 4);
        assert!((manifest.defaults.guidance - 1.0).abs() < 0.01);
    }

    #[test]
    fn flux_krea_resolves_to_q8() {
        let manifest = find_manifest("flux-krea").unwrap();
        assert_eq!(manifest.name, "flux-krea:q8");
        assert!(!manifest.defaults.is_schnell);
        assert_eq!(manifest.defaults.steps, 25);
    }

    #[test]
    fn flux_krea_legacy_format() {
        assert_eq!(resolve_model_name("flux-krea-q4"), "flux-krea:q4");
        assert_eq!(resolve_model_name("flux-krea-q6"), "flux-krea:q6");
    }

    #[test]
    fn flux_krea_fp8_exists() {
        let manifest = find_manifest("flux-krea:fp8").unwrap();
        assert_eq!(manifest.family, "flux");
        assert!(!manifest.defaults.is_schnell);
        assert_eq!(manifest.defaults.guidance, 4.5);
        assert!(manifest
            .files
            .iter()
            .any(|f| f.hf_filename.contains("float8_e4m3fn")));
    }

    #[test]
    fn jibmix_flux_manifests_exist() {
        assert!(find_manifest("jibmix-flux:fp8").is_some());
        assert!(find_manifest("jibmix-flux:q5").is_some());
        assert!(find_manifest("jibmix-flux:q4").is_some());
        assert!(find_manifest("jibmix-flux:q3").is_some());
    }

    #[test]
    fn jibmix_flux_bare_resolves_to_fp8() {
        // No :q8/:fp16/:bf16 tag exists, so bare name resolves to :fp8
        assert_eq!(resolve_model_name("jibmix-flux"), "jibmix-flux:fp8");
    }

    #[test]
    fn jibmix_flux_defaults() {
        let manifest = find_manifest("jibmix-flux:q4").unwrap();
        assert_eq!(manifest.family, "flux");
        assert!(!manifest.defaults.is_schnell);
        assert_eq!(manifest.defaults.steps, 25);
        assert_eq!(manifest.defaults.guidance, 3.0);
    }

    #[test]
    fn ultrareal_v2_exists() {
        let manifest = find_manifest("ultrareal-v2:bf16").unwrap();
        assert_eq!(manifest.family, "flux");
        assert!(!manifest.defaults.is_schnell);
        assert!(manifest
            .files
            .iter()
            .any(|f| f.hf_filename.ends_with(".safetensors")
                && f.component == ModelComponent::Transformer));
    }

    #[test]
    fn ultrareal_v3_manifests_exist() {
        assert!(find_manifest("ultrareal-v3:q8").is_some());
        assert!(find_manifest("ultrareal-v3:q6").is_some());
        assert!(find_manifest("ultrareal-v3:q4").is_some());
    }

    #[test]
    fn ultrareal_v4_manifests_exist() {
        assert!(find_manifest("ultrareal-v4:q8").is_some());
        assert!(find_manifest("ultrareal-v4:q5").is_some());
        assert!(find_manifest("ultrareal-v4:q4").is_some());
    }

    #[test]
    fn ultrareal_v4_defaults() {
        let manifest = find_manifest("ultrareal-v4:q8").unwrap();
        assert_eq!(manifest.family, "flux");
        assert!(!manifest.defaults.is_schnell);
        assert_eq!(manifest.defaults.steps, 25);
        assert_eq!(manifest.defaults.guidance, 3.5);
    }

    #[test]
    fn iniverse_mix_exists() {
        let manifest = find_manifest("iniverse-mix:fp8").unwrap();
        assert_eq!(manifest.family, "flux");
        assert!(!manifest.defaults.is_schnell);
        assert_eq!(manifest.defaults.steps, 25);
        assert!(manifest
            .files
            .iter()
            .any(|f| f.hf_filename.contains("iniverseMixSFWNSFW")));
    }

    #[test]
    fn iniverse_mix_bare_resolves_to_fp8() {
        assert_eq!(resolve_model_name("iniverse-mix"), "iniverse-mix:fp8");
    }

    #[test]
    fn pony_v6_exists() {
        let manifest = find_manifest("pony-v6:fp16").unwrap();
        assert_eq!(manifest.family, "sdxl");
        assert_eq!(manifest.defaults.scheduler, Some(Scheduler::EulerAncestral));
        assert!(manifest
            .files
            .iter()
            .any(|f| f.hf_repo.contains("ponyDiffusionV6XL")));
    }

    #[test]
    fn cyberrealistic_pony_exists() {
        let manifest = find_manifest("cyberrealistic-pony:fp16").unwrap();
        assert_eq!(manifest.family, "sdxl");
        assert_eq!(manifest.defaults.scheduler, Some(Scheduler::EulerAncestral));
        assert!(manifest
            .files
            .iter()
            .any(|f| f.hf_repo.contains("cyberrealisticPony")));
    }

    #[test]
    fn flux2_klein_gguf_exists() {
        assert!(find_manifest("flux2-klein:q8").is_some());
        assert!(find_manifest("flux2-klein:q6").is_some());
        assert!(find_manifest("flux2-klein:q4").is_some());
    }

    #[test]
    fn flux2_klein_resolves_to_q8() {
        // bare "flux2-klein" resolves to :q8 (tried first, matches existing installs)
        let name = resolve_model_name("flux2-klein");
        assert_eq!(name, "flux2-klein:q8");
    }

    #[test]
    fn flux2_klein_9b_defaults() {
        let manifest = find_manifest("flux2-klein-9b:bf16").unwrap();
        assert_eq!(manifest.family, "flux2");
        assert_eq!(manifest.defaults.steps, 4);
        assert!((manifest.defaults.guidance - 1.0).abs() < 0.01);
        assert_eq!(manifest.defaults.width, 1024);
    }

    #[test]
    fn flux2_klein_9b_bf16_is_sharded() {
        let manifest = find_manifest("flux2-klein-9b:bf16").unwrap();
        let shards: Vec<_> = manifest
            .files
            .iter()
            .filter(|f| f.component == ModelComponent::TransformerShard)
            .collect();
        assert_eq!(
            shards.len(),
            2,
            "Klein-9B BF16 should have 2 transformer shards"
        );
    }

    #[test]
    fn flux2_klein_9b_gguf_exists() {
        assert!(find_manifest("flux2-klein-9b:q8").is_some());
        assert!(find_manifest("flux2-klein-9b:q6").is_some());
        assert!(find_manifest("flux2-klein-9b:q4").is_some());
    }

    #[test]
    fn flux2_klein_9b_resolves_to_q8() {
        let name = resolve_model_name("flux2-klein-9b");
        assert_eq!(name, "flux2-klein-9b:q8");
    }

    #[test]
    fn flux2_klein_9b_shared_files_are_gated() {
        let manifest = find_manifest("flux2-klein-9b:q8").unwrap();
        let text_encoders: Vec<_> = manifest
            .files
            .iter()
            .filter(|f| f.component == ModelComponent::TextEncoder)
            .collect();
        assert!(
            text_encoders.iter().all(|f| f.gated),
            "Klein-9B text encoder shards should be gated"
        );
    }

    #[test]
    fn variant_quality_rank_ordering() {
        use super::variant_quality_rank;
        assert!(variant_quality_rank("flux-dev:bf16") < variant_quality_rank("flux-dev:fp16"));
        assert!(variant_quality_rank("flux-dev:fp16") < variant_quality_rank("flux-dev:fp8"));
        assert!(variant_quality_rank("flux-dev:fp8") < variant_quality_rank("flux-dev:q8"));
        assert!(variant_quality_rank("flux-dev:q8") < variant_quality_rank("flux-dev:q6"));
        assert!(variant_quality_rank("flux-dev:q6") < variant_quality_rank("flux-dev:q5"));
        assert!(variant_quality_rank("flux-dev:q5") < variant_quality_rank("flux-dev:q4"));
        assert!(variant_quality_rank("flux-dev:q4") < variant_quality_rank("flux-dev:q3"));
    }

    #[test]
    fn variant_quality_rank_unknown_tag_sorts_last() {
        use super::variant_quality_rank;
        assert!(variant_quality_rank("custom-model") > variant_quality_rank("flux-dev:q3"));
    }

    #[test]
    fn model_base_name_extracts_prefix() {
        use super::model_base_name;
        assert_eq!(model_base_name("flux-dev:q4"), "flux-dev");
        assert_eq!(model_base_name("sd15:fp16"), "sd15");
        assert_eq!(model_base_name("custom-model"), "custom-model");
    }

    #[test]
    fn known_manifests_count() {
        // 24 FLUX + 3 SD1.5 + 4 SD3 + 8 SDXL + 4 Z-Image + 9 Flux.2 + 24 Qwen-Image/Qwen-Image-Edit + 1 Wuerstchen + 5 LTX Video + 6 LTX-2 + 11 Wan + 4 compliance-hidden MiniMax H3 contracts + 7 LTX-2 controls + 7 LTX-2 camera controls + 3 ControlNet + 2 Qwen3-Expand + 7 Upscaler + 20 Companion = 149
        // Wan fp8 bump (#777): +wan22-{t2v,i2v}-a14b:fp8 — the Comfy-Org
        // fp8-scaled expert pairs.
        // Wan bump: +wan22-{t2v,i2v}-a14b:{q5,q8} — the two-expert A14B tiers.
        // Wan low-VRAM bump (#794): +wan22-{t2v,i2v}-a14b:q4 and
        // +wan22-ti2v-5b:q8.
        // Companion bump: +flux2-te, +flux2-te-9b, +flux2-vae for the
        // catalog bridge (single-file Civitai Flux.2 fine-tunes); +z-image-te
        // for single-file Civitai Z-Image checkpoints; +ltx2-te for the
        // catalog bridge (single-file Civitai LTX-2 / LTX-2.3 fine-tunes —
        // Gemma 3 12B text encoder); +wan-umt5, +wan21-vae, +wan22-vae for
        // single-file Wan checkpoints.
        assert_eq!(known_manifests().len(), 149);
    }

    #[test]
    fn legacy_ltx_manifests_removed() {
        assert!(find_manifest("ltx-video-0.9:bf16").is_none());
        assert!(find_manifest("ltx-video-0.9.5:bf16").is_none());
    }

    #[test]
    fn current_ltx_manifests_present() {
        assert!(find_manifest("ltx-video-0.9.6:bf16").is_some());
        assert!(find_manifest("ltx-video-0.9.6-distilled:bf16").is_some());
        assert!(find_manifest("ltx-video-0.9.8-2b-distilled:bf16").is_some());
        assert!(find_manifest("ltx-video-0.9.8-13b-dev:bf16").is_some());
        assert!(find_manifest("ltx-video-0.9.8-13b-distilled:bf16").is_some());
    }

    /// The Wan manifests are the contract every later stack layer builds on:
    /// bare names resolve to the shipped tags, the component set is complete
    /// (transformer + VAE + UMT5 + tokenizer), every payload carries a
    /// SHA-256, and the entries stay hidden until the engine's factory arm
    /// exists — a listed model that cannot load is worse than none.
    #[test]
    fn wan_manifests_resolve_and_carry_full_component_sets() {
        assert_eq!(resolve_model_name("wan21-t2v-1.3b"), "wan21-t2v-1.3b:bf16");
        // The bare 5B name must stay on the fp16 default even though a `:q8`
        // small-card tag now exists — the generic tag loop tries `:q8` first,
        // so this is pinned explicitly in `resolve_model_name` (#794).
        assert_eq!(resolve_model_name("wan22-ti2v-5b"), "wan22-ti2v-5b:fp16");
        // Legacy-dash form resolves like every other family's.
        assert_eq!(
            resolve_model_name("wan22-ti2v-5b-fp16"),
            "wan22-ti2v-5b:fp16"
        );
        assert_eq!(resolve_model_name("wan22-ti2v-5b-q8"), "wan22-ti2v-5b:q8");

        for name in [
            "wan21-t2v-1.3b:bf16",
            "wan22-ti2v-5b:fp16",
            "wan22-ti2v-5b:q8",
        ] {
            let manifest = find_manifest(name).unwrap_or_else(|| panic!("{name} must resolve"));
            assert_eq!(manifest.family, "wan");
            assert!(
                !manifest.hidden,
                "{name} is listable now that the wan engine exists"
            );
            for component in [
                ModelComponent::Transformer,
                ModelComponent::Vae,
                ModelComponent::TextEncoder,
                ModelComponent::TextTokenizer,
            ] {
                assert!(
                    manifest.files.iter().any(|f| f.component == component),
                    "{name} missing {component:?}"
                );
            }
            for file in &manifest.files {
                assert!(
                    file.sha256.is_some(),
                    "{name}: {} must carry a SHA-256",
                    file.hf_filename
                );
            }
            let defaults = &manifest.defaults;
            assert!(defaults.frames.is_some() && defaults.fps.is_some());
            assert!(
                defaults.negative_prompt.is_some(),
                "{name} must default to upstream's tuned negative prompt"
            );
        }

        // The two VAE generations must not be conflated: 2.1 (16ch) for the
        // 1.3B/14B family, 2.2 (48ch) for TI2V-5B.
        let vae_file = |name: &str| {
            find_manifest(name)
                .unwrap()
                .files
                .iter()
                .find(|f| f.component == ModelComponent::Vae)
                .unwrap()
                .hf_filename
                .clone()
        };
        assert!(vae_file("wan21-t2v-1.3b:bf16").contains("wan_2.1_vae"));
        assert!(vae_file("wan22-ti2v-5b:fp16").contains("wan2.2_vae"));
        assert!(vae_file("wan22-ti2v-5b:q8").contains("wan2.2_vae"));

        // The `:q8` tag is the same 5B transformer as QuantStack's Q8_0 GGUF
        // — the ~5.4 GB pull that reaches small cards (#794).
        let q8 = find_manifest("wan22-ti2v-5b:q8").unwrap();
        let transformer = q8
            .files
            .iter()
            .find(|f| f.component == ModelComponent::Transformer)
            .unwrap();
        assert_eq!(transformer.hf_repo, "QuantStack/Wan2.2-TI2V-5B-GGUF");
        assert!(transformer.hf_filename.contains("Q8_0"));
        // Same recipe as fp16: the quantization changes the pull, not the
        // defaults contract.
        let fp16 = find_manifest("wan22-ti2v-5b:fp16").unwrap();
        assert_eq!(q8.defaults.steps, fp16.defaults.steps);
        assert_eq!(q8.defaults.guidance, fp16.defaults.guidance);
        assert_eq!(q8.defaults.frames, fp16.defaults.frames);
        assert_eq!(q8.defaults.fps, fp16.defaults.fps);
        assert_eq!((q8.defaults.width, q8.defaults.height), (1280, 704));
    }

    /// Every A14B tier ships a complete expert pair, and the fast tier ships a
    /// distill for each expert. A manifest with one expert, or with one distill
    /// applied to both, is not a smaller model — it is a broken one.
    #[test]
    fn a14b_manifests_carry_both_experts_and_a_distill_per_expert() {
        // Bare names take the family's `:q8` default, which is the quality tier.
        assert_eq!(resolve_model_name("wan22-t2v-a14b"), "wan22-t2v-a14b:q8");
        assert_eq!(resolve_model_name("wan22-i2v-a14b"), "wan22-i2v-a14b:q8");

        for name in [
            "wan22-t2v-a14b:q4",
            "wan22-t2v-a14b:q5",
            "wan22-t2v-a14b:q8",
            "wan22-i2v-a14b:q4",
            "wan22-i2v-a14b:q5",
            "wan22-i2v-a14b:q8",
        ] {
            let manifest = find_manifest(name).unwrap_or_else(|| panic!("{name} must resolve"));
            assert_eq!(manifest.family, "wan");
            assert!(!manifest.hidden);

            let file = |component: ModelComponent| {
                manifest
                    .files
                    .iter()
                    .find(|f| f.component == component)
                    .unwrap_or_else(|| panic!("{name} missing {component:?}"))
            };
            let high = file(ModelComponent::Transformer);
            let low = file(ModelComponent::LowNoiseTransformer);
            assert!(high.hf_filename.contains("HighNoise"), "{name}");
            assert!(low.hf_filename.contains("LowNoise"), "{name}");
            assert_ne!(
                high.sha256, low.sha256,
                "{name}: the two experts are different networks"
            );
            assert_eq!(high.hf_repo, low.hf_repo);

            // The 2.1 VAE, byte-identical to the 1.3B's so it dedupes.
            assert!(file(ModelComponent::Vae)
                .hf_filename
                .contains("wan_2.1_vae"));
            for component in [ModelComponent::TextEncoder, ModelComponent::TextTokenizer] {
                file(component);
            }
            for f in &manifest.files {
                assert!(f.sha256.is_some(), "{name}: {} needs a SHA", f.hf_filename);
            }

            let defaults = &manifest.defaults;
            // Defaults are the measured RTX 4090 envelope, not the trained
            // 81: the Q5 pair peaks at 23,975 MiB rendering 53 frames at
            // 832x480 (81 OOM'd at a 23.0 GB peak), and the Q8 pair's larger
            // resident expert moves its edge to ~33. The Q4 pair's expert is
            // ~1.1 GB smaller than Q5's, so 53 fits wherever Q5's does — its
            // measured T2V peak is 21,372 MiB (#794). All stay on the 4n+1
            // grid; bigger cards pass --frames 81 explicitly.
            let expected_frames = if name.ends_with(":q8") { 33 } else { 53 };
            assert_eq!(defaults.frames, Some(expected_frames), "{name}");
            assert_eq!(
                (expected_frames - 1) % 4,
                0,
                "{name}: envelope default must sit on the 4n+1 grid"
            );
            assert_eq!(defaults.fps, Some(16));
            assert!(defaults.negative_prompt.is_some());

            // Both `:q5` and `:q4` are Lightning tiers; only `:q8` renders
            // the family's ordinary schedule without the distill.
            let fast = !name.ends_with(":q8");
            let distills: Vec<_> = manifest
                .files
                .iter()
                .filter(|f| {
                    matches!(
                        f.component,
                        ModelComponent::DistilledLora | ModelComponent::LowNoiseDistilledLora
                    )
                })
                .collect();
            if fast {
                assert_eq!(distills.len(), 2, "{name}: one distill per expert");
                assert_ne!(
                    distills[0].sha256, distills[1].sha256,
                    "{name}: each expert is distilled separately"
                );
                assert!(distills
                    .iter()
                    .any(|f| f.hf_filename.contains("high_noise")));
                assert!(distills.iter().any(|f| f.hf_filename.contains("low_noise")));
                // The lightx2v recipe: four steps, and guidance 1.0 so the
                // unconditional pass is skipped rather than merely weakened.
                assert_eq!(defaults.steps, 4);
                assert_eq!(defaults.guidance, 1.0);
            } else {
                assert!(
                    distills.is_empty(),
                    "{name}: the quality tier has no distill"
                );
                assert_eq!(defaults.steps, 20);
                assert_eq!(defaults.guidance, 3.5);
            }
        }

        // The distill pairs must come from the Comfy-Org repack. The
        // similarly-named `lightx2v/Wan2.2-Distill-Loras` I2V files carry
        // `.diff`/`.diff_b` full-weight deltas and target the Wan 2.1 CLIP
        // branch, which this family's checkpoints do not have.
        for name in [
            "wan22-t2v-a14b:q4",
            "wan22-t2v-a14b:q5",
            "wan22-i2v-a14b:q4",
            "wan22-i2v-a14b:q5",
        ] {
            for file in find_manifest(name).unwrap().files.iter().filter(|f| {
                matches!(
                    f.component,
                    ModelComponent::DistilledLora | ModelComponent::LowNoiseDistilledLora
                )
            }) {
                assert_eq!(
                    file.hf_repo, "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                    "{name}"
                );
            }
        }

        // Both experts are model-specific, so neither lands in the shared pool
        // where the other tier's same-named file would collide with it.
        let manifest = find_manifest("wan22-t2v-a14b:q5").unwrap();
        for component in [
            ModelComponent::Transformer,
            ModelComponent::LowNoiseTransformer,
            ModelComponent::DistilledLora,
            ModelComponent::LowNoiseDistilledLora,
        ] {
            let file = manifest
                .files
                .iter()
                .find(|f| f.component == component)
                .unwrap();
            let path = storage_path(manifest, file);
            assert!(
                path.starts_with("wan22-t2v-a14b-q5"),
                "{component:?} must be model-specific, got {}",
                path.display()
            );
        }
    }

    /// The `:q4` tier is the `:q5` recipe with smaller experts (#794): it must
    /// carry byte-identical Lightning distill files — the adapters were
    /// trained once, against the bf16 weights, and are quant-independent —
    /// while its experts are the Q4_K_M GGUFs, distinct from Q5_K_M's.
    #[test]
    fn a14b_q4_tier_reuses_the_q5_lightning_distill_pair() {
        for base in ["wan22-t2v-a14b", "wan22-i2v-a14b"] {
            let q4 = find_manifest(&format!("{base}:q4")).unwrap();
            let q5 = find_manifest(&format!("{base}:q5")).unwrap();

            for component in [
                ModelComponent::DistilledLora,
                ModelComponent::LowNoiseDistilledLora,
            ] {
                let file = |m: &'static ModelManifest| {
                    m.files
                        .iter()
                        .find(|f| f.component == component)
                        .unwrap_or_else(|| panic!("{}: missing {component:?}", m.name))
                };
                let (a, b) = (file(q4), file(q5));
                assert_eq!(a.hf_repo, b.hf_repo, "{base} {component:?}");
                assert_eq!(a.hf_filename, b.hf_filename, "{base} {component:?}");
                assert_eq!(a.sha256, b.sha256, "{base} {component:?}");
                assert_eq!(a.size_bytes, b.size_bytes, "{base} {component:?}");
            }

            for component in [
                ModelComponent::Transformer,
                ModelComponent::LowNoiseTransformer,
            ] {
                let expert = q4.files.iter().find(|f| f.component == component).unwrap();
                assert!(
                    expert.hf_filename.contains("Q4_K_M"),
                    "{base} {component:?}: expected a Q4_K_M expert, got {}",
                    expert.hf_filename
                );
            }

            // Same Lightning recipe as `:q5`, down to the reduced-envelope
            // frame default.
            assert_eq!(q4.defaults.steps, q5.defaults.steps);
            assert_eq!(q4.defaults.guidance, q5.defaults.guidance);
            assert_eq!(q4.defaults.frames, q5.defaults.frames);
        }
    }

    /// The wan-umt5 companion and the primary Wan manifests must agree on
    /// storage: the companion bridge routes its shared components into
    /// `shared/wan/`, where the primary manifests already put the encoder and
    /// tokenizer — so a companion pull and a primary pull dedupe to one
    /// on-disk copy of the 11 GB encoder. (The VAE companions deliberately do
    /// NOT dedupe: like ltx-video-vae/ltx2-vae they carry their payload as a
    /// model-specific Transformer file the catalog bridge points at directly.)
    #[test]
    fn wan_umt5_companion_shares_storage_with_primary_manifests() {
        let primary = find_manifest("wan21-t2v-1.3b:bf16").unwrap();
        let umt5 = find_manifest("wan-umt5").unwrap();
        for component in [ModelComponent::TextEncoder, ModelComponent::TextTokenizer] {
            let companion_file = umt5
                .files
                .iter()
                .find(|f| f.component == component)
                .unwrap();
            let primary_file = primary
                .files
                .iter()
                .find(|f| f.component == component)
                .unwrap();
            assert_eq!(
                storage_path(umt5, companion_file),
                storage_path(primary, primary_file),
                "companion {component:?} must land where the primary manifests put it"
            );
        }
    }

    #[test]
    fn ltx_098_manifests_include_spatial_upscaler() {
        for model in [
            "ltx-video-0.9.8-2b-distilled:bf16",
            "ltx-video-0.9.8-13b-dev:bf16",
            "ltx-video-0.9.8-13b-distilled:bf16",
        ] {
            let manifest = find_manifest(model).expect("manifest should exist");
            assert!(
                manifest
                    .files
                    .iter()
                    .any(|file| file.component == ModelComponent::SpatialUpscaler),
                "{model} missing spatial upscaler component"
            );
        }
    }

    #[test]
    fn ltx2_manifests_include_verified_hashes_for_transformers_and_gemma() {
        for model in [
            "ltx-2-19b-dev:fp8",
            "ltx-2-19b-distilled:fp8",
            "ltx-2.3-22b-dev:fp8",
            "ltx-2.3-22b-distilled:fp8",
            "ltx-2.3-22b-dev:bf16",
            "ltx-2.3-22b-distilled:bf16",
        ] {
            let manifest = find_manifest(model).expect("manifest should exist");
            let transformer = manifest
                .files
                .iter()
                .find(|file| file.component == ModelComponent::Transformer)
                .expect("transformer entry should exist");
            assert!(
                transformer.sha256.is_some(),
                "{model} missing transformer sha256"
            );

            let gemma_shards = manifest
                .files
                .iter()
                .filter(|file| {
                    file.component == ModelComponent::TextEncoder
                        && file.hf_filename.ends_with(".safetensors")
                })
                .count();
            assert_eq!(gemma_shards, 5, "{model} missing Gemma shard entries");
            for file in manifest.files.iter().filter(|file| {
                file.component == ModelComponent::TextEncoder
                    && (file.hf_filename.ends_with(".safetensors")
                        || file.hf_filename == "tokenizer.json"
                        || file.hf_filename == "tokenizer.model")
            }) {
                assert!(
                    file.sha256.is_some(),
                    "{model} missing sha256 for {}",
                    file.hf_filename
                );
                assert!(file.gated, "{model} should mark Gemma asset as gated");
            }
        }
    }

    #[test]
    fn ltx2_22b_bf16_manifests_use_upstream_reference_checkpoints() {
        for (model, filename, size_bytes, sha256) in [
            (
                "ltx-2.3-22b-dev:bf16",
                "ltx-2.3-22b-dev.safetensors",
                46_149_344_974,
                "7ab7225325bc403448ea84b6db2269811a880e5118cd2ee2b6282a93d585016f",
            ),
            (
                "ltx-2.3-22b-distilled:bf16",
                "ltx-2.3-22b-distilled-1.1.safetensors",
                46_149_345_334,
                "b33b7fe4bbfe084f484be4aaf90b0f1d95dca20d403ac4c0e037eb8c4f0af7cc",
            ),
        ] {
            let manifest = find_manifest(model).expect("BF16 manifest should exist");
            assert_eq!(manifest.family, "ltx2");
            let transformer = manifest
                .files
                .iter()
                .find(|file| file.component == ModelComponent::Transformer)
                .expect("transformer entry should exist");
            assert_eq!(transformer.hf_repo, "Lightricks/LTX-2.3");
            assert_eq!(transformer.hf_filename, filename);
            assert_eq!(transformer.size_bytes, size_bytes);
            assert_eq!(transformer.sha256, Some(sha256));
        }
    }

    #[test]
    fn ltx2_22b_manifests_include_x1_5_spatial_upscaler_after_x2_default() {
        for model in [
            "ltx-2.3-22b-dev:fp8",
            "ltx-2.3-22b-distilled:fp8",
            "ltx-2.3-22b-dev:bf16",
            "ltx-2.3-22b-distilled:bf16",
        ] {
            let manifest = find_manifest(model).expect("manifest should exist");
            let spatial_files: Vec<&str> = manifest
                .files
                .iter()
                .filter(|file| file.component == ModelComponent::SpatialUpscaler)
                .map(|file| file.hf_filename.as_str())
                .collect();
            let x2 = if model.ends_with(":bf16") {
                "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
            } else {
                "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
            };
            assert_eq!(
                spatial_files,
                vec![x2, "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors"]
            );
        }
    }

    #[test]
    fn ltx_shared_t5_assets_live_under_shared_flux() {
        let manifest = find_manifest("ltx-video-0.9.8-2b-distilled:bf16").unwrap();
        let t5 = manifest
            .files
            .iter()
            .find(|file| file.component == ModelComponent::T5Encoder)
            .unwrap();
        let tokenizer = manifest
            .files
            .iter()
            .find(|file| file.component == ModelComponent::T5Tokenizer)
            .unwrap();
        assert!(storage_path(manifest, t5).starts_with("shared/flux"));
        assert!(storage_path(manifest, tokenizer).starts_with("shared/flux"));
    }

    #[test]
    fn ltx_shared_vae_and_spatial_upscaler_use_repo_aligned_paths() {
        let manifest = find_manifest("ltx-video-0.9.8-13b-distilled:bf16").unwrap();
        let vae = manifest
            .files
            .iter()
            .find(|file| file.component == ModelComponent::Vae)
            .unwrap();
        let upscaler = manifest
            .files
            .iter()
            .find(|file| file.component == ModelComponent::SpatialUpscaler)
            .unwrap();
        assert!(storage_path(manifest, vae).starts_with("shared/LTX-Video-0.9.5"));
        assert!(storage_path(manifest, upscaler).starts_with("shared/LTX-Video"));
    }

    #[test]
    fn ltx_storage_candidates_include_legacy_shared_bucket() {
        let manifest = find_manifest("ltx-video-0.9.8-2b-distilled:bf16").unwrap();
        let t5 = manifest
            .files
            .iter()
            .find(|file| file.component == ModelComponent::T5Encoder)
            .unwrap();
        let candidates = storage_path_candidates(manifest, t5);
        assert_eq!(candidates.len(), 2);
        assert!(candidates[0].starts_with("shared/flux"));
        assert!(candidates[1].starts_with("shared/ltx-video"));
    }

    #[test]
    fn ltx_manifests_use_the_compatible_legacy_vae_source() {
        for manifest in known_manifests()
            .iter()
            .filter(|manifest| manifest.family == "ltx-video")
        {
            let vae = manifest
                .files
                .iter()
                .find(|file| file.component == ModelComponent::Vae)
                .expect("ltx manifest should include a VAE");
            assert_eq!(
                vae.hf_repo, "Lightricks/LTX-Video-0.9.5",
                "{} should keep using the compatible legacy LTX VAE until the newer VAE layout is ported",
                manifest.name
            );
        }
    }

    #[test]
    fn manifest_has_required_components() {
        for manifest in known_manifests() {
            let components: Vec<_> = manifest.files.iter().map(|f| f.component).collect();
            // All diffusion models need VAE except families that intentionally ship a
            // single-file upstream checkpoint without a standalone VAE asset.
            if !manifest.is_utility()
                && !manifest.is_upscaler()
                && !manifest.is_auxiliary()
                && manifest.family != "ltx2"
            {
                assert!(
                    components.contains(&ModelComponent::Vae),
                    "{} missing Vae",
                    manifest.name
                );
            }

            match manifest.family.as_str() {
                "flux" => {
                    assert!(
                        components.contains(&ModelComponent::Transformer),
                        "{} (flux) missing Transformer",
                        manifest.name
                    );
                    assert!(
                        components.contains(&ModelComponent::ClipEncoder),
                        "{} (flux) missing ClipEncoder",
                        manifest.name
                    );
                    assert!(
                        components.contains(&ModelComponent::ClipTokenizer),
                        "{} (flux) missing ClipTokenizer",
                        manifest.name
                    );
                    assert!(
                        components.contains(&ModelComponent::T5Encoder),
                        "{} (flux) missing T5Encoder",
                        manifest.name
                    );
                    assert!(
                        components.contains(&ModelComponent::T5Tokenizer),
                        "{} (flux) missing T5Tokenizer",
                        manifest.name
                    );
                }
                "sd15" => {
                    assert!(
                        components.contains(&ModelComponent::Transformer),
                        "{} (sd15) missing Transformer",
                        manifest.name
                    );
                    assert!(
                        components.contains(&ModelComponent::ClipEncoder),
                        "{} (sd15) missing ClipEncoder",
                        manifest.name
                    );
                    assert!(
                        components.contains(&ModelComponent::ClipTokenizer),
                        "{} (sd15) missing ClipTokenizer",
                        manifest.name
                    );
                    // SD1.5 does NOT use dual CLIP
                    assert!(
                        !components.contains(&ModelComponent::ClipEncoder2),
                        "{} (sd15) should not have ClipEncoder2",
                        manifest.name
                    );
                }
                "sd3" => {
                    assert!(
                        components.contains(&ModelComponent::Transformer),
                        "{} (sd3) missing Transformer",
                        manifest.name
                    );
                    assert!(
                        components.contains(&ModelComponent::ClipEncoder),
                        "{} (sd3) missing ClipEncoder (CLIP-L)",
                        manifest.name
                    );
                    assert!(
                        components.contains(&ModelComponent::ClipTokenizer),
                        "{} (sd3) missing ClipTokenizer (CLIP-L)",
                        manifest.name
                    );
                    assert!(
                        components.contains(&ModelComponent::ClipEncoder2),
                        "{} (sd3) missing ClipEncoder2 (CLIP-G)",
                        manifest.name
                    );
                    assert!(
                        components.contains(&ModelComponent::ClipTokenizer2),
                        "{} (sd3) missing ClipTokenizer2 (CLIP-G)",
                        manifest.name
                    );
                    assert!(
                        components.contains(&ModelComponent::T5Encoder),
                        "{} (sd3) missing T5Encoder",
                        manifest.name
                    );
                    assert!(
                        components.contains(&ModelComponent::T5Tokenizer),
                        "{} (sd3) missing T5Tokenizer",
                        manifest.name
                    );
                }
                "sdxl" => {
                    assert!(
                        components.contains(&ModelComponent::Transformer),
                        "{} (sdxl) missing Transformer",
                        manifest.name
                    );
                    assert!(
                        components.contains(&ModelComponent::ClipEncoder),
                        "{} (sdxl) missing ClipEncoder",
                        manifest.name
                    );
                    assert!(
                        components.contains(&ModelComponent::ClipTokenizer),
                        "{} (sdxl) missing ClipTokenizer",
                        manifest.name
                    );
                    assert!(
                        components.contains(&ModelComponent::ClipEncoder2),
                        "{} (sdxl) missing ClipEncoder2",
                        manifest.name
                    );
                    assert!(
                        components.contains(&ModelComponent::ClipTokenizer2),
                        "{} (sdxl) missing ClipTokenizer2",
                        manifest.name
                    );
                }
                "z-image" => {
                    // Z-Image uses Transformer (GGUF) or TransformerShard (BF16)
                    assert!(
                        components.contains(&ModelComponent::Transformer)
                            || components.contains(&ModelComponent::TransformerShard),
                        "{} (z-image) missing Transformer or TransformerShard",
                        manifest.name
                    );
                    assert!(
                        components.contains(&ModelComponent::TextEncoder),
                        "{} (z-image) missing TextEncoder",
                        manifest.name
                    );
                    assert!(
                        components.contains(&ModelComponent::TextTokenizer),
                        "{} (z-image) missing TextTokenizer",
                        manifest.name
                    );
                    // Z-Image does NOT use CLIP
                    assert!(
                        !components.contains(&ModelComponent::ClipEncoder),
                        "{} (z-image) should not have ClipEncoder",
                        manifest.name
                    );
                }
                "flux2" => {
                    // Flux.2 uses Transformer (or TransformerShard for sharded BF16) + Qwen3 TextEncoder + TextTokenizer
                    assert!(
                        components.contains(&ModelComponent::Transformer)
                            || components.contains(&ModelComponent::TransformerShard),
                        "{} (flux2) missing Transformer or TransformerShard",
                        manifest.name
                    );
                    assert!(
                        components.contains(&ModelComponent::TextEncoder),
                        "{} (flux2) missing TextEncoder",
                        manifest.name
                    );
                    assert!(
                        components.contains(&ModelComponent::TextTokenizer),
                        "{} (flux2) missing TextTokenizer",
                        manifest.name
                    );
                    // Flux.2 Klein does NOT use CLIP or T5
                    assert!(
                        !components.contains(&ModelComponent::ClipEncoder),
                        "{} (flux2) should not have ClipEncoder",
                        manifest.name
                    );
                    assert!(
                        !components.contains(&ModelComponent::T5Encoder),
                        "{} (flux2) should not have T5Encoder",
                        manifest.name
                    );
                }
                "ltx2" => {
                    assert!(
                        components.contains(&ModelComponent::Transformer),
                        "{} (ltx2) missing Transformer",
                        manifest.name
                    );
                    assert!(
                        components.contains(&ModelComponent::SpatialUpscaler),
                        "{} (ltx2) missing SpatialUpscaler",
                        manifest.name
                    );
                    assert!(
                        components.contains(&ModelComponent::TemporalUpscaler),
                        "{} (ltx2) missing TemporalUpscaler",
                        manifest.name
                    );
                    assert!(
                        components.contains(&ModelComponent::TextEncoder),
                        "{} (ltx2) missing Gemma text encoder files",
                        manifest.name
                    );
                }
                "qwen-image" | "qwen-image-edit" => {
                    // Qwen-Image uses TransformerShard (BF16 sharded)
                    assert!(
                        components.contains(&ModelComponent::Transformer)
                            || components.contains(&ModelComponent::TransformerShard),
                        "{} ({}) missing Transformer or TransformerShard",
                        manifest.name,
                        manifest.family
                    );
                    assert!(
                        components.contains(&ModelComponent::TextEncoder),
                        "{} ({}) missing TextEncoder",
                        manifest.name,
                        manifest.family
                    );
                    assert!(
                        components.contains(&ModelComponent::TextTokenizer),
                        "{} ({}) missing TextTokenizer",
                        manifest.name,
                        manifest.family
                    );
                    // Qwen-Image does NOT use CLIP
                    assert!(
                        !components.contains(&ModelComponent::ClipEncoder),
                        "{} ({}) should not have ClipEncoder",
                        manifest.name,
                        manifest.family
                    );
                }
                "controlnet" => {
                    // ControlNet only needs the transformer weights (no VAE, CLIP, etc.)
                    assert!(
                        components.contains(&ModelComponent::Transformer),
                        "{} (controlnet) missing Transformer",
                        manifest.name
                    );
                }
                "upscaler" => {
                    // Upscaler only needs the upscaler weights file
                    assert!(
                        components.contains(&ModelComponent::Upscaler),
                        "{} (upscaler) missing Upscaler",
                        manifest.name
                    );
                }
                _ => {}
            }
        }
    }

    #[test]
    fn shared_files_gated_flags() {
        for file in shared_flux_files() {
            if file.hf_repo.starts_with("black-forest-labs/") {
                assert!(
                    file.gated,
                    "{} should be gated (BFL repo)",
                    file.hf_filename
                );
            } else {
                assert!(!file.gated, "{} should not be gated", file.hf_filename);
            }
        }
    }

    // --- T5 variant registry tests ---

    #[test]
    fn t5_variants_sorted_largest_first() {
        let variants = known_t5_variants();
        for w in variants.windows(2) {
            assert!(
                w[0].size_bytes >= w[1].size_bytes,
                "{} ({}) should be >= {} ({})",
                w[0].tag,
                w[0].size_bytes,
                w[1].tag,
                w[1].size_bytes,
            );
        }
    }

    #[test]
    fn find_t5_variant_by_tag() {
        assert_eq!(find_t5_variant("q8").unwrap().tag, "q8");
        assert_eq!(find_t5_variant("q5").unwrap().tag, "q5");
        assert_eq!(find_t5_variant("q3").unwrap().tag, "q3");
        assert!(find_t5_variant("nonexistent").is_none());
    }

    #[test]
    fn t5_variants_all_gguf() {
        for v in known_t5_variants() {
            assert!(
                v.hf_filename.ends_with(".gguf"),
                "{} should be a GGUF file",
                v.hf_filename
            );
        }
    }

    #[test]
    fn sdxl_manifests_have_scheduler() {
        for manifest in known_manifests() {
            if manifest.family == "sdxl" {
                assert!(
                    manifest.defaults.scheduler.is_some(),
                    "{} (sdxl) should have a scheduler",
                    manifest.name
                );
            }
        }
    }

    #[test]
    fn sdxl_turbo_uses_euler_ancestral() {
        let manifest = find_manifest("sdxl-turbo").unwrap();
        assert_eq!(manifest.defaults.scheduler, Some(Scheduler::EulerAncestral));
        assert_eq!(manifest.defaults.steps, 4);
        assert_eq!(manifest.defaults.guidance, 0.0);
    }

    #[test]
    fn sdxl_base_uses_ddim() {
        let manifest = find_manifest("sdxl-base").unwrap();
        assert_eq!(manifest.defaults.scheduler, Some(Scheduler::Ddim));
        assert_eq!(manifest.defaults.steps, 25);
    }

    #[test]
    fn sdxl_resolve_fp16_default() {
        assert_eq!(resolve_model_name("sdxl-base"), "sdxl-base:fp16");
        assert_eq!(
            resolve_model_name("playground-v2.5"),
            "playground-v2.5:fp16"
        );
    }

    #[test]
    fn t5_fp16_larger_than_all_quantized() {
        for v in known_t5_variants() {
            assert!(
                T5_FP16_SIZE > v.size_bytes,
                "FP16 should be larger than {}",
                v.tag
            );
        }
    }

    // --- SD1.5 tests ---

    #[test]
    fn sd15_manifest_exists() {
        assert!(find_manifest("sd15").is_some());
        assert!(find_manifest("sd15:fp16").is_some());
    }

    #[test]
    fn sd15_default_tag_resolves() {
        assert_eq!(resolve_model_name("sd15"), "sd15:fp16");
    }

    #[test]
    fn dreamshaper_v8_resolves() {
        let manifest = find_manifest("dreamshaper-v8").unwrap();
        assert_eq!(manifest.name, "dreamshaper-v8:fp16");
        assert_eq!(manifest.family, "sd15");
        assert_eq!(manifest.defaults.steps, 25);
        assert_eq!(manifest.defaults.width, 512);
    }

    #[test]
    fn sd15_defaults() {
        for manifest in known_manifests() {
            if manifest.family == "sd15" {
                assert_eq!(manifest.defaults.scheduler, Some(Scheduler::Ddim));
                assert_eq!(manifest.defaults.width, 512);
                assert_eq!(manifest.defaults.height, 512);
            }
        }
    }

    #[test]
    fn realistic_vision_v5_resolves() {
        let manifest = find_manifest("realistic-vision-v5").unwrap();
        assert_eq!(manifest.name, "realistic-vision-v5:fp16");
        assert_eq!(manifest.family, "sd15");
        assert_eq!(manifest.defaults.steps, 25);
        assert_eq!(manifest.defaults.width, 512);
        assert_eq!(manifest.defaults.guidance, 7.5);
        assert_eq!(manifest.defaults.scheduler, Some(Scheduler::Ddim));
    }

    #[test]
    fn shared_sd15_files_not_gated() {
        for file in shared_sd15_files() {
            assert!(
                !file.gated,
                "SD1.5 shared file {} should not be gated",
                file.hf_filename
            );
        }
    }

    #[test]
    fn sd15_manifests_have_scheduler() {
        for manifest in known_manifests() {
            if manifest.family == "sd15" {
                assert!(
                    manifest.defaults.scheduler.is_some(),
                    "{} (sd15) should have a scheduler",
                    manifest.name
                );
            }
        }
    }

    // --- Z-Image tests ---

    #[test]
    fn zimage_turbo_resolves_to_q8() {
        assert_eq!(resolve_model_name("z-image-turbo"), "z-image-turbo:q8");
    }

    #[test]
    fn zimage_turbo_bf16_found() {
        let manifest = find_manifest("z-image-turbo:bf16").unwrap();
        assert_eq!(manifest.family, "z-image");
        assert_eq!(manifest.defaults.steps, 9);
        assert_eq!(manifest.defaults.guidance, 0.0);
        assert_eq!(manifest.defaults.scheduler, None);
    }

    // --- Qwen3 variant registry tests ---

    #[test]
    fn qwen3_variants_sorted_largest_first() {
        let variants = known_qwen3_variants();
        for w in variants.windows(2) {
            assert!(
                w[0].size_bytes >= w[1].size_bytes,
                "{} ({}) should be >= {} ({})",
                w[0].tag,
                w[0].size_bytes,
                w[1].tag,
                w[1].size_bytes,
            );
        }
    }

    #[test]
    fn find_qwen3_variant_by_tag() {
        assert_eq!(find_qwen3_variant("q8").unwrap().tag, "q8");
        assert_eq!(find_qwen3_variant("q6").unwrap().tag, "q6");
        assert_eq!(find_qwen3_variant("iq4").unwrap().tag, "iq4");
        assert_eq!(find_qwen3_variant("q3").unwrap().tag, "q3");
        assert!(find_qwen3_variant("nonexistent").is_none());
    }

    #[test]
    fn find_qwen2_vl_variant_by_tag() {
        assert_eq!(find_qwen2_vl_variant("q8").unwrap().tag, "q8");
        assert_eq!(find_qwen2_vl_variant("q6").unwrap().tag, "q6");
        assert_eq!(find_qwen2_vl_variant("q5").unwrap().tag, "q5");
        assert_eq!(find_qwen2_vl_variant("q4").unwrap().tag, "q4");
        assert_eq!(find_qwen2_vl_variant("q3").unwrap().tag, "q3");
        assert_eq!(find_qwen2_vl_variant("q2").unwrap().tag, "q2");
        assert!(find_qwen2_vl_variant("nonexistent").is_none());
    }

    #[test]
    fn qwen3_variants_all_gguf() {
        for v in known_qwen3_variants() {
            assert!(
                v.hf_filename.ends_with(".gguf"),
                "{} should be a GGUF file",
                v.hf_filename
            );
        }
    }

    #[test]
    fn qwen3_fp16_larger_than_all_quantized() {
        for v in known_qwen3_variants() {
            assert!(
                QWEN3_FP16_SIZE > v.size_bytes,
                "FP16 should be larger than {}",
                v.tag
            );
        }
    }

    // --- Qwen3-8B variant registry tests ---

    #[test]
    fn qwen3_8b_variants_sorted_largest_first() {
        let variants = known_qwen3_8b_variants();
        for w in variants.windows(2) {
            assert!(
                w[0].size_bytes >= w[1].size_bytes,
                "{} ({}) should be >= {} ({})",
                w[0].tag,
                w[0].size_bytes,
                w[1].tag,
                w[1].size_bytes,
            );
        }
    }

    #[test]
    fn find_qwen3_8b_variant_by_tag() {
        assert_eq!(find_qwen3_8b_variant("q8").unwrap().tag, "q8");
        assert_eq!(find_qwen3_8b_variant("q6").unwrap().tag, "q6");
        assert_eq!(find_qwen3_8b_variant("iq4").unwrap().tag, "iq4");
        assert_eq!(find_qwen3_8b_variant("q3").unwrap().tag, "q3");
        assert!(find_qwen3_8b_variant("nonexistent").is_none());
    }

    #[test]
    fn qwen3_8b_variants_all_gguf() {
        for v in known_qwen3_8b_variants() {
            assert!(
                v.hf_filename.ends_with(".gguf"),
                "{} should be a GGUF file",
                v.hf_filename
            );
        }
    }

    #[test]
    fn qwen3_8b_fp16_larger_than_all_quantized() {
        for v in known_qwen3_8b_variants() {
            assert!(
                QWEN3_8B_FP16_SIZE > v.size_bytes,
                "8B FP16 should be larger than {}",
                v.tag
            );
        }
    }

    #[test]
    fn qwen3_8b_larger_than_4b() {
        // Every 8B variant should be larger than the corresponding 4B variant
        let variants_4b = known_qwen3_variants();
        let variants_8b = known_qwen3_8b_variants();
        for v8 in variants_8b {
            if let Some(v4) = variants_4b.iter().find(|v| v.tag == v8.tag) {
                assert!(
                    v8.size_bytes > v4.size_bytes,
                    "8B {} ({}) should be larger than 4B {} ({})",
                    v8.tag,
                    v8.size_bytes,
                    v4.tag,
                    v4.size_bytes,
                );
            }
        }
    }

    #[test]
    fn zimage_defaults() {
        for manifest in known_manifests() {
            if manifest.family == "z-image" {
                assert_eq!(manifest.defaults.steps, 9);
                assert_eq!(manifest.defaults.guidance, 0.0);
                assert_eq!(manifest.defaults.scheduler, None);
            }
        }
    }

    #[test]
    fn total_download_size_equals_sum_of_file_sizes() {
        for manifest in known_manifests() {
            let total = total_download_size(manifest);
            let sum: u64 = manifest.files.iter().map(|f| f.size_bytes).sum();
            assert_eq!(
                total, sum,
                "total_download_size mismatch for {}",
                manifest.name
            );
        }
    }

    #[test]
    fn compute_download_remaining_lte_total() {
        // remaining_bytes must always be <= total_bytes
        for manifest in known_manifests() {
            let (total, remaining) = compute_download_size(manifest);
            assert!(
                remaining <= total,
                "remaining ({remaining}) > total ({total}) for {}",
                manifest.name
            );
        }
    }

    #[test]
    fn total_file_bytes_is_positive_for_all_manifests() {
        // Every manifest must have files that sum to a positive total
        for manifest in known_manifests() {
            let total = total_download_size(manifest);
            assert!(total > 0, "total_download_size is 0 for {}", manifest.name);
        }
    }

    #[test]
    fn total_size_gb_matches_total_size_bytes() {
        for manifest in known_manifests() {
            let from_bytes = manifest.total_size_bytes() as f32 / 1_073_741_824.0;
            let from_method = manifest.total_size_gb();
            assert!(
                (from_bytes - from_method).abs() < 0.001,
                "total_size_gb mismatch for {}: {} vs {}",
                manifest.name,
                from_bytes,
                from_method
            );
        }
    }

    #[test]
    fn model_size_gb_matches_model_size_bytes() {
        for manifest in known_manifests() {
            let from_bytes = manifest.model_size_bytes() as f32 / 1_073_741_824.0;
            let from_method = manifest.model_size_gb();
            assert!(
                (from_bytes - from_method).abs() < 0.001,
                "model_size_gb mismatch for {}: {} vs {}",
                manifest.name,
                from_bytes,
                from_method
            );
        }
    }

    #[test]
    fn total_size_includes_shared_components() {
        // Models with shared files must have total >= model-specific size.
        for manifest in known_manifests() {
            if manifest.is_auxiliary() || manifest.is_upscaler() {
                continue; // ControlNet and upscalers are single-file models
            }
            let model_specific_bytes: u64 = manifest
                .files
                .iter()
                .filter(|f| is_model_specific_component(f.component))
                .map(|f| f.size_bytes)
                .sum();
            let total = manifest.total_size_bytes();
            assert!(
                total >= model_specific_bytes,
                "{}: total ({}) should be >= model-specific ({})",
                manifest.name,
                total,
                model_specific_bytes
            );
            assert_eq!(
                manifest.model_size_bytes(),
                model_specific_bytes,
                "{}: model size should match model-specific files",
                manifest.name
            );
        }
    }

    #[test]
    fn no_manifest_has_size_gb_field() {
        // Ensures we don't accidentally re-add a static size_gb field.
        // If this test exists and compiles, the field doesn't exist on ModelManifest.
        let manifest = find_manifest("flux-schnell:q8").unwrap();
        let _: f32 = manifest.total_size_gb(); // computed, not stored
    }

    #[test]
    fn flux_schnell_total_exceeds_transformer() {
        let manifest = find_manifest("flux-schnell:q8").unwrap();
        // Transformer is ~12GB, shared components add ~9.8GB
        let total_gb = manifest.total_size_gb();
        assert!(
            total_gb > 20.0,
            "flux-schnell:q8 total should be >20GB (was {})",
            total_gb
        );
    }

    #[test]
    fn flux_schnell_model_size_is_transformer_only() {
        let manifest = find_manifest("flux-schnell:q8").unwrap();
        let model_gb = manifest.model_size_gb();
        assert!(
            model_gb > 11.0 && model_gb < 13.0,
            "flux-schnell:q8 model size should be transformer-only (~11.8GiB), was {}",
            model_gb
        );
        assert!(manifest.total_size_gb() > model_gb);
    }

    #[test]
    fn zimage_q8_size_includes_shared() {
        let manifest = find_manifest("z-image-turbo:q8").unwrap();
        let total = manifest.total_size_gb();
        // Transformer (~6.58GB) + shared (~8.2GB) = ~13.8 GiB
        assert!(total > 13.0);
    }

    #[test]
    fn controlnet_model_size_matches_total_size() {
        let manifest = find_manifest("controlnet-canny-sd15:fp16").unwrap();
        assert_eq!(manifest.model_size_bytes(), manifest.total_size_bytes());
    }

    #[test]
    fn wuerstchen_no_storage_path_collisions() {
        let manifest = find_manifest("wuerstchen-v2:fp16").unwrap();
        let paths: Vec<_> = manifest
            .files
            .iter()
            .map(|f| storage_path(manifest, f))
            .collect();
        // No two files should resolve to the same local path
        let unique: std::collections::HashSet<_> = paths.iter().collect();
        assert_eq!(
            paths.len(),
            unique.len(),
            "storage path collision in wuerstchen manifest: {:?}",
            paths
        );
    }

    #[test]
    fn find_smaller_alternatives_for_bf16_model() {
        let alts = super::find_smaller_alternatives("ultrareal-v2:bf16");
        // Should find ultrareal-v3 and v4 quantized variants (all smaller than 23.8GB)
        assert!(
            !alts.is_empty(),
            "bf16 model should have smaller alternatives"
        );
        for alt in &alts {
            assert!(
                alt.contains("ultrareal"),
                "alternative should be in same family: {alt}"
            );
        }
    }

    #[test]
    fn find_smaller_alternatives_for_smallest_model_is_empty() {
        // flux-schnell:q4 is already the smallest flux-schnell variant
        let alts = super::find_smaller_alternatives("flux-schnell:q4");
        // Should have no smaller alternatives (q4 is the smallest)
        assert!(
            alts.is_empty() || alts.iter().all(|a| a != "flux-schnell:q4"),
            "should not suggest the same model"
        );
    }

    #[test]
    fn looks_like_model_name_family_match() {
        let config = crate::Config::default();
        // ultrareal-v8 shares "ultrareal" family with ultrareal-v2/v3/v4
        assert!(super::looks_like_model_name("ultrareal-v8", &config));
        // dreamshaper-v9 shares "dreamshaper" family with dreamshaper-v8
        assert!(super::looks_like_model_name("dreamshaper-v9", &config));
    }

    #[test]
    fn looks_like_model_name_colon_syntax() {
        let config = crate::Config::default();
        // Explicit colon tag is always model-like, even if totally unknown
        assert!(super::looks_like_model_name("foo:q8", &config));
        assert!(super::looks_like_model_name("flux-dev:q99", &config));
    }

    #[test]
    fn looks_like_model_name_fuzzy_match() {
        let config = crate::Config::default();
        // Close misspelling of "flux-dev"
        assert!(super::looks_like_model_name("flx-dev", &config));
    }

    #[test]
    fn looks_like_model_name_rejects_natural_language() {
        let config = crate::Config::default();
        for word in &[
            "a",
            "sunset",
            "photorealistic",
            "cat",
            "beautiful",
            "oil painting",
        ] {
            assert!(
                !super::looks_like_model_name(word, &config),
                "'{word}' should not look like a model name"
            );
        }
    }

    #[test]
    fn suggest_similar_models_for_near_miss() {
        let config = crate::Config::default();
        let suggestions = super::suggest_similar_models("ultrareal-v8", &config, 5);
        assert!(!suggestions.is_empty(), "should have suggestions");
        // Should include at least one ultrareal variant
        assert!(
            suggestions.iter().any(|s| s.starts_with("ultrareal-")),
            "should suggest ultrareal variants, got: {suggestions:?}"
        );
    }

    #[test]
    fn suggest_similar_models_empty_for_unrelated() {
        let config = crate::Config::default();
        let suggestions = super::suggest_similar_models("zzzzzzzzz", &config, 5);
        assert!(
            suggestions.is_empty(),
            "unrelated string should have no suggestions"
        );
    }

    #[test]
    fn all_generation_model_names_excludes_upscalers() {
        let config = crate::Config::default();
        let gen_names = super::all_generation_model_names(&config);
        for name in &gen_names {
            if let Some(manifest) = find_manifest(name) {
                assert!(
                    !manifest.is_upscaler(),
                    "all_generation_model_names should not contain upscaler '{name}'"
                );
            }
        }
    }

    #[test]
    fn all_generation_model_names_excludes_utility_models() {
        let config = crate::Config::default();
        let gen_names = super::all_generation_model_names(&config);
        for name in &gen_names {
            if let Some(manifest) = find_manifest(name) {
                assert!(
                    !manifest.is_utility(),
                    "all_generation_model_names should not contain utility model '{name}'"
                );
            }
        }
    }

    #[test]
    fn all_generation_model_names_contains_diffusion_models() {
        let config = crate::Config::default();
        let gen_names = super::all_generation_model_names(&config);
        // Should contain at least some well-known diffusion models.
        assert!(
            gen_names.iter().any(|n| n.starts_with("flux-schnell")),
            "generation names should include flux-schnell variants"
        );
        assert!(
            gen_names.iter().any(|n| n.starts_with("flux-dev")),
            "generation names should include flux-dev variants"
        );
    }

    #[test]
    fn all_generation_model_names_excludes_controlnet() {
        let config = crate::Config::default();
        let gen_names = super::all_generation_model_names(&config);
        for name in &gen_names {
            if let Some(manifest) = find_manifest(name) {
                assert!(
                    !manifest.is_auxiliary(),
                    "all_generation_model_names should not contain auxiliary model '{name}'"
                );
            }
        }
        // Verify controlnet models exist in the full list but not generation list.
        let all_names = super::all_model_names(&config);
        assert!(
            all_names.iter().any(|n| n.starts_with("controlnet-")),
            "all_model_names should include controlnet models"
        );
        assert!(
            !gen_names.iter().any(|n| n.starts_with("controlnet-")),
            "all_generation_model_names should not include controlnet models"
        );
    }

    #[test]
    fn all_model_names_includes_upscalers() {
        let config = crate::Config::default();
        let all_names = super::all_model_names(&config);
        assert!(
            all_names.iter().any(|n| n.starts_with("real-esrgan")),
            "all_model_names should still include upscaler models"
        );
    }

    #[test]
    fn suggest_similar_models_excludes_upscalers() {
        let config = crate::Config::default();
        // Suggestions for any input should never include upscaler models.
        let suggestions = super::suggest_similar_models("flux-schnell", &config, 50);
        for s in &suggestions {
            if let Some(m) = find_manifest(s) {
                assert!(
                    !m.is_upscaler(),
                    "suggest_similar_models should not suggest upscaler '{s}'"
                );
            }
        }
    }

    #[test]
    fn qwen3_expand_manifest_has_transformer_and_tokenizer() {
        let manifest = find_manifest("qwen3-expand:q8").unwrap();
        let components: Vec<_> = manifest.files.iter().map(|f| f.component).collect();
        assert!(components.contains(&ModelComponent::Transformer));
        assert!(components.contains(&ModelComponent::TextTokenizer));
        assert!(!components.contains(&ModelComponent::Vae));
    }

    #[test]
    fn qwen3_expand_manifest_ungated() {
        let manifest = find_manifest("qwen3-expand:q8").unwrap();
        assert!(!manifest.is_gated());
        let small = find_manifest("qwen3-expand-small:q8").unwrap();
        assert!(!small.is_gated());
    }

    #[test]
    fn qwen3_expand_family() {
        let manifest = find_manifest("qwen3-expand:q8").unwrap();
        assert_eq!(manifest.family, "qwen3-expand");
    }

    #[test]
    fn qwen3_expand_resolve_bare_name() {
        // "qwen3-expand" should resolve to "qwen3-expand:q8"
        let resolved = resolve_model_name("qwen3-expand");
        assert_eq!(resolved, "qwen3-expand:q8");
    }

    #[test]
    fn qwen3_expand_storage_paths() {
        let manifest = find_manifest("qwen3-expand:q8").unwrap();
        let transformer = manifest
            .files
            .iter()
            .find(|f| f.component == ModelComponent::Transformer)
            .unwrap();
        let path = storage_path(manifest, transformer);
        // Transformer is model-specific: qwen3-expand-q8/<filename>
        assert!(
            path.starts_with("qwen3-expand-q8"),
            "got: {}",
            path.display()
        );

        let tokenizer = manifest
            .files
            .iter()
            .find(|f| f.component == ModelComponent::TextTokenizer)
            .unwrap();
        let tok_path = storage_path(manifest, tokenizer);
        // Tokenizer is shared: shared/qwen3-expand/<filename>
        assert!(
            tok_path.starts_with("shared/qwen3-expand"),
            "got: {}",
            tok_path.display()
        );
    }

    // --- is_utility family-based identification ---

    #[test]
    fn qwen3_expand_is_utility() {
        let manifest = find_manifest("qwen3-expand:q8").unwrap();
        assert!(manifest.is_utility(), "qwen3-expand should be utility");
    }

    #[test]
    fn qwen3_expand_small_is_utility() {
        let manifest = find_manifest("qwen3-expand-small:q8").unwrap();
        assert!(
            manifest.is_utility(),
            "qwen3-expand-small should be utility"
        );
    }

    #[test]
    fn controlnet_is_not_utility() {
        let manifest = find_manifest("controlnet-canny-sd15:fp16").unwrap();
        assert!(!manifest.is_utility(), "controlnet should NOT be utility");
    }

    #[test]
    fn controlnet_depth_is_not_utility() {
        let manifest = find_manifest("controlnet-depth-sd15:fp16").unwrap();
        assert!(
            !manifest.is_utility(),
            "controlnet-depth should NOT be utility"
        );
    }

    #[test]
    fn controlnet_openpose_is_not_utility() {
        let manifest = find_manifest("controlnet-openpose-sd15:fp16").unwrap();
        assert!(
            !manifest.is_utility(),
            "controlnet-openpose should NOT be utility"
        );
    }

    #[test]
    fn flux_models_are_not_utility() {
        for manifest in known_manifests() {
            if manifest.family == "flux" {
                assert!(
                    !manifest.is_utility(),
                    "{} should NOT be utility",
                    manifest.name
                );
            }
        }
    }

    #[test]
    fn sd15_models_are_not_utility() {
        for manifest in known_manifests() {
            if manifest.family == "sd15" {
                assert!(
                    !manifest.is_utility(),
                    "{} should NOT be utility",
                    manifest.name
                );
            }
        }
    }

    #[test]
    fn no_diffusion_model_is_utility() {
        let diffusion_families = [
            "flux",
            "sd15",
            "sdxl",
            "sd3",
            "z-image",
            "flux2",
            "qwen-image",
            "qwen-image-edit",
            "wuerstchen",
            "ltx-video",
        ];
        for manifest in known_manifests() {
            if diffusion_families.contains(&manifest.family.as_str()) {
                assert!(
                    !manifest.is_utility(),
                    "{} (family={}) should NOT be utility",
                    manifest.name,
                    manifest.family
                );
            }
        }
    }

    #[test]
    fn utility_families_constant_matches_expand_manifests() {
        for manifest in known_manifests() {
            if manifest.family == "qwen3-expand" {
                assert!(
                    UTILITY_FAMILIES.contains(&manifest.family.as_str()),
                    "{} family not in UTILITY_FAMILIES",
                    manifest.name
                );
            }
        }
    }

    #[test]
    fn all_utility_models_identified_by_is_utility() {
        let utility_count = known_manifests().iter().filter(|m| m.is_utility()).count();
        // Currently 22: 2 qwen3-expand variants + 20 catalog companions
        // (clip-l, clip-g, sdxl-vae, sd-vae-ft-mse, t5-v1_1-xxl, flux-vae,
        // ltx-video-vae, ltx2-vae, ltx2.3-vae, ltx2.3-text-projection, flux2-te, flux2-te-9b,
        // flux2-vae, z-image-te, ltx2-te, wan-umt5, wan21-vae, wan22-vae,
        // qwen-image-runtime, wuerstchen-runtime).
        assert_eq!(
            utility_count, 22,
            "expected exactly 22 utility models, got {utility_count}"
        );
    }

    // --- is_upscaler family-based identification ---

    #[test]
    fn upscaler_models_are_upscaler() {
        for manifest in known_manifests() {
            if manifest.family == "upscaler" {
                assert!(
                    manifest.is_upscaler(),
                    "{} should be upscaler",
                    manifest.name
                );
                assert!(
                    !manifest.is_utility(),
                    "{} should NOT be utility",
                    manifest.name
                );
            }
        }
    }

    // --- Companion manifests (single-file Civitai checkpoint dependencies) ---

    /// All canonical companion names from `mold_catalog::companions::COMPANIONS`
    /// that the `POST /api/catalog/:id/download` companion auto-pull resolves.
    const COMPANION_NAMES: &[&str] = &[
        "clip-l",
        "clip-g",
        "sdxl-vae",
        "sd-vae-ft-mse",
        "t5-v1_1-xxl",
        "flux-vae",
        "flux2-te",
        "flux2-te-9b",
        "flux2-vae",
        "z-image-te",
        "ltx-video-vae",
        "ltx2-vae",
        "ltx2.3-vae",
        "ltx2.3-text-projection",
        "ltx2-te",
        "wan-umt5",
        "wan21-vae",
        "wan22-vae",
    ];

    #[test]
    fn companion_canonical_names_all_resolve() {
        for name in COMPANION_NAMES {
            let manifest = find_manifest(name).unwrap_or_else(|| {
                panic!("companion {name} must resolve via find_manifest");
            });
            assert_eq!(manifest.name, *name, "{name} canonical-name mismatch");
            assert_eq!(
                manifest.family, "companion",
                "{name} should land in the companion family"
            );
            assert!(
                manifest.hidden,
                "{name} must be hidden — companions never appear in `mold list`"
            );
            assert!(
                !manifest.files.is_empty(),
                "{name} must declare at least one file to pull"
            );
        }
    }

    #[test]
    fn companion_manifests_are_utility() {
        for name in COMPANION_NAMES {
            let manifest = find_manifest(name).expect("companion resolves");
            assert!(
                manifest.is_utility(),
                "{name} must be utility — companions don't get config entries or default-model status",
            );
            assert!(
                !manifest.is_generation_model(),
                "{name} must not be a generation model",
            );
        }
    }

    #[test]
    fn companion_clip_l_points_at_openai_repo() {
        // Mirrors mold_catalog::companions::COMPANIONS["clip-l"].repo —
        // the synthetic manifest must keep tracking the canonical source.
        let manifest = find_manifest("clip-l").expect("clip-l resolves");
        assert!(
            manifest
                .files
                .iter()
                .any(|f| f.hf_repo == "openai/clip-vit-large-patch14"),
            "clip-l must source from openai/clip-vit-large-patch14",
        );
    }

    #[test]
    fn upscaler_manifests_have_upscaler_component() {
        for manifest in known_manifests() {
            if manifest.is_upscaler() {
                assert!(
                    manifest
                        .files
                        .iter()
                        .any(|f| f.component == ModelComponent::Upscaler),
                    "{} missing Upscaler component",
                    manifest.name
                );
                // Upscalers should have exactly one file
                assert_eq!(
                    manifest.files.len(),
                    1,
                    "{} should have exactly 1 file",
                    manifest.name
                );
            }
        }
    }

    #[test]
    fn all_upscaler_models_identified_by_is_upscaler() {
        let count = known_manifests().iter().filter(|m| m.is_upscaler()).count();
        assert_eq!(count, 7, "expected 7 upscaler models, got {count}");
    }

    #[test]
    fn upscaler_storage_path_is_model_specific() {
        let manifest = find_manifest("real-esrgan-x4plus:fp16").unwrap();
        let file = &manifest.files[0];
        let path = storage_path(manifest, file);
        // Upscaler component is model-specific, stored under model name dir
        assert!(
            path.starts_with("real-esrgan-x4plus-fp16"),
            "upscaler should be under model-specific dir, got: {}",
            path.display()
        );
    }

    #[test]
    fn diffusion_models_are_not_upscaler() {
        for manifest in known_manifests() {
            if [
                "flux",
                "sd15",
                "sdxl",
                "sd3",
                "z-image",
                "flux2",
                "qwen-image",
                "qwen-image-edit",
                "wuerstchen",
            ]
            .contains(&manifest.family.as_str())
            {
                assert!(
                    !manifest.is_upscaler(),
                    "{} should NOT be upscaler",
                    manifest.name
                );
            }
        }
    }

    #[test]
    fn no_models_are_alpha() {
        for manifest in known_manifests() {
            let desc = &manifest.description;
            assert!(
                !desc.contains("(alpha)") && !desc.contains("[alpha]"),
                "model '{}' is still marked alpha: {desc}",
                manifest.name
            );
        }
    }

    #[test]
    fn no_diffusion_model_descriptions_have_stale_alpha_prefix() {
        // Ensure no model uses the old [alpha] prefix format (should use (alpha) suffix)
        for manifest in known_manifests() {
            assert!(
                !manifest.description.starts_with("[alpha]"),
                "model '{}' uses stale [alpha] prefix format: {}",
                manifest.name,
                manifest.description
            );
        }
    }

    // ── Pipeline structural tests ────────────────────────────────────────
    // These tests read pipeline source files to verify consistent VRAM
    // management patterns across all model families.

    /// Pipelines that use eager mode must wrap their transformer/UNet in Option
    /// so it can be dropped before VAE decode to free VRAM.
    #[test]
    fn all_pipelines_wrap_transformer_in_option() {
        let pipeline_fields = [
            (
                "flux",
                "crates/mold-inference/src/flux/pipeline.rs",
                "Option<FluxTransformer>",
            ),
            (
                "flux2",
                "crates/mold-inference/src/flux2/pipeline.rs",
                "Option<Flux2TransformerWrapper>",
            ),
            (
                "sd15",
                "crates/mold-inference/src/sd15/pipeline.rs",
                "Option<stable_diffusion::unet_2d::UNet2DConditionModel>",
            ),
            (
                "sdxl",
                "crates/mold-inference/src/sdxl/pipeline.rs",
                "Option<stable_diffusion::unet_2d::UNet2DConditionModel>",
            ),
            (
                "sd3",
                "crates/mold-inference/src/sd3/pipeline.rs",
                "Option<SD3Transformer>",
            ),
            (
                "zimage",
                "crates/mold-inference/src/zimage/pipeline.rs",
                "Option<ZImageTransformer>",
            ),
            (
                "qwen_image",
                "crates/mold-inference/src/qwen_image/pipeline.rs",
                "Option<QwenImageTransformer>",
            ),
            (
                "wuerstchen (prior)",
                "crates/mold-inference/src/wuerstchen/pipeline.rs",
                "Option<WPrior>",
            ),
            (
                "wuerstchen (decoder)",
                "crates/mold-inference/src/wuerstchen/pipeline.rs",
                "Option<WDiffNeXt>",
            ),
        ];

        let workspace = env!("CARGO_MANIFEST_DIR")
            .strip_suffix("/crates/mold-core")
            .or_else(|| env!("CARGO_MANIFEST_DIR").strip_suffix("crates/mold-core"))
            .unwrap_or(env!("CARGO_MANIFEST_DIR"));

        for (family, path, expected_type) in pipeline_fields {
            let full_path = format!("{workspace}/{path}");
            let source = std::fs::read_to_string(&full_path)
                .unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
            assert!(
                source.contains(expected_type),
                "{family} pipeline ({path}) must wrap transformer/UNet in {expected_type} \
                 for VRAM management — field should be Option-wrapped so it can be dropped \
                 before VAE decode"
            );
        }
    }

    /// All pipelines must call device.synchronize() before VAE/VQ-GAN decode
    /// in their eager generate path to ensure CUDA releases freed memory.
    #[test]
    fn all_pipelines_synchronize_before_decode() {
        let pipelines = [
            ("flux", "crates/mold-inference/src/flux/pipeline.rs"),
            ("flux2", "crates/mold-inference/src/flux2/pipeline.rs"),
            ("sd15", "crates/mold-inference/src/sd15/pipeline.rs"),
            ("sdxl", "crates/mold-inference/src/sdxl/pipeline.rs"),
            ("sd3", "crates/mold-inference/src/sd3/pipeline.rs"),
            ("zimage", "crates/mold-inference/src/zimage/pipeline.rs"),
            (
                "qwen_image",
                "crates/mold-inference/src/qwen_image/pipeline.rs",
            ),
            (
                "wuerstchen",
                "crates/mold-inference/src/wuerstchen/pipeline.rs",
            ),
        ];

        let workspace = env!("CARGO_MANIFEST_DIR")
            .strip_suffix("/crates/mold-core")
            .or_else(|| env!("CARGO_MANIFEST_DIR").strip_suffix("crates/mold-core"))
            .unwrap_or(env!("CARGO_MANIFEST_DIR"));

        for (family, path) in pipelines {
            let full_path = format!("{workspace}/{path}");
            let source = std::fs::read_to_string(&full_path)
                .unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
            assert!(
                source.contains("synchronize()"),
                "{family} pipeline ({path}) must call device.synchronize() before decode \
                 to ensure CUDA releases freed transformer/UNet VRAM"
            );
        }
    }

    /// All pipelines with sequential mode must check the prompt cache BEFORE
    /// loading the text encoder, to avoid wasting seconds reloading the encoder
    /// on batch iterations when the prompt is identical.
    #[test]
    fn sequential_pipelines_check_cache_before_encoder_load() {
        // Each entry: (family, path, cache_check_pattern, encoder_load_patterns)
        // The cache check must appear BEFORE the encoder load in generate_sequential().
        let pipelines = [
            (
                "flux2",
                "crates/mold-inference/src/flux2/pipeline.rs",
                "restore_cached_tensor(",
                &["Qwen3Encoder::load_"][..],
            ),
            (
                "flux",
                "crates/mold-inference/src/flux/pipeline.rs",
                "restore_prompt_cache(",
                &["T5Encoder::load"][..],
            ),
            (
                "sd15",
                "crates/mold-inference/src/sd15/pipeline.rs",
                "restore_cached_tensor(",
                &["build_clip_transformer("][..],
            ),
            (
                "sdxl",
                "crates/mold-inference/src/sdxl/pipeline.rs",
                "restore_cached_tensor(",
                &["build_clip_transformer("][..],
            ),
            (
                "sd3",
                "crates/mold-inference/src/sd3/pipeline.rs",
                "restore_cached_tensor_pair(",
                &[
                    "SD3TripleEncoder::load(",
                    "SD3TripleEncoder::load_with_tokenizers(",
                ][..],
            ),
            (
                "zimage",
                "crates/mold-inference/src/zimage/pipeline.rs",
                "restore_cached_tensor(",
                &["Qwen3Encoder::load_"][..],
            ),
            (
                "qwen_image",
                "crates/mold-inference/src/qwen_image/pipeline.rs",
                "get_cloned(&prompt_key)",
                &["load_text_encoder("][..],
            ),
            (
                "wuerstchen",
                "crates/mold-inference/src/wuerstchen/pipeline.rs",
                "restore_cached_tensor_pair(",
                &["build_clip_transformer("][..],
            ),
        ];

        let workspace = env!("CARGO_MANIFEST_DIR")
            .strip_suffix("/crates/mold-core")
            .or_else(|| env!("CARGO_MANIFEST_DIR").strip_suffix("crates/mold-core"))
            .unwrap_or(env!("CARGO_MANIFEST_DIR"));

        for (family, path, cache_pattern, load_patterns) in pipelines {
            let full_path = format!("{workspace}/{path}");
            let source = std::fs::read_to_string(&full_path)
                .unwrap_or_else(|e| panic!("failed to read {path}: {e}"));

            // Find generate_sequential function body
            let seq_start = source.find("fn generate_sequential(").unwrap_or_else(|| {
                panic!("{family} pipeline ({path}) missing generate_sequential()")
            });
            let seq_body = &source[seq_start..];

            let cache_pos = seq_body.find(cache_pattern).unwrap_or_else(|| {
                panic!(
                    "{family} pipeline ({path}) generate_sequential() missing cache check '{cache_pattern}'"
                )
            });
            let (load_pattern, load_pos) = load_patterns
                .iter()
                .filter_map(|pattern| seq_body.find(pattern).map(|pos| (*pattern, pos)))
                .min_by_key(|(_, pos)| *pos)
                .unwrap_or_else(|| {
                    let expected = load_patterns.join("' or '");
                    panic!(
                        "{family} pipeline ({path}) generate_sequential() missing encoder load '{expected}'"
                    )
                });

            assert!(
                cache_pos < load_pos,
                "{family} pipeline ({path}): prompt cache check ('{cache_pattern}' at offset {cache_pos}) \
                 must appear BEFORE encoder load ('{load_pattern}' at offset {load_pos}) \
                 in generate_sequential() — encoder should not be loaded when cache hits"
            );
        }
    }

    // --- paths_from_downloads: upscaler and utility family support (issue #184) ---

    #[test]
    fn paths_from_downloads_upscaler_returns_some() {
        let downloads = vec![(
            ModelComponent::Upscaler,
            PathBuf::from("/models/upscaler/weights.safetensors"),
        )];
        let paths = paths_from_downloads(&downloads, "upscaler");
        assert!(
            paths.is_some(),
            "upscaler downloads should produce ModelPaths"
        );
        let paths = paths.unwrap();
        assert_eq!(
            paths.transformer,
            PathBuf::from("/models/upscaler/weights.safetensors"),
            "upscaler weights should map to transformer field"
        );
        assert_eq!(paths.vae, PathBuf::new(), "upscaler should have empty vae");
    }

    #[test]
    fn paths_from_downloads_upscaler_missing_component_returns_none() {
        // No Upscaler component → should return None
        let downloads: Vec<(ModelComponent, PathBuf)> = vec![];
        assert!(
            paths_from_downloads(&downloads, "upscaler").is_none(),
            "empty downloads for upscaler should return None"
        );
    }

    #[test]
    fn paths_from_downloads_utility_returns_some_without_vae() {
        let downloads = vec![
            (
                ModelComponent::Transformer,
                PathBuf::from("/models/qwen3/model.gguf"),
            ),
            (
                ModelComponent::TextTokenizer,
                PathBuf::from("/models/qwen3/tokenizer.json"),
            ),
        ];
        let paths = paths_from_downloads(&downloads, "qwen3-expand");
        assert!(
            paths.is_some(),
            "utility downloads without VAE should produce ModelPaths"
        );
        let paths = paths.unwrap();
        assert_eq!(paths.transformer, PathBuf::from("/models/qwen3/model.gguf"));
        assert_eq!(paths.vae, PathBuf::new(), "utility should have empty vae");
        assert_eq!(
            paths.text_tokenizer,
            Some(PathBuf::from("/models/qwen3/tokenizer.json"))
        );
    }

    #[test]
    fn paths_from_downloads_utility_missing_transformer_returns_none() {
        // Utility model with only a tokenizer (no transformer) → should return None
        let downloads = vec![(
            ModelComponent::TextTokenizer,
            PathBuf::from("/models/qwen3/tokenizer.json"),
        )];
        assert!(
            paths_from_downloads(&downloads, "qwen3-expand").is_none(),
            "utility without transformer should return None"
        );
    }

    #[test]
    fn paths_from_downloads_diffusion_still_requires_vae() {
        // Diffusion model without VAE → should return None (unchanged behavior)
        let downloads = vec![(
            ModelComponent::Transformer,
            PathBuf::from("/models/flux/transformer.safetensors"),
        )];
        assert!(
            paths_from_downloads(&downloads, "flux").is_none(),
            "diffusion model without VAE should return None"
        );
    }

    #[test]
    fn paths_from_downloads_diffusion_with_all_components() {
        let downloads = vec![
            (
                ModelComponent::Transformer,
                PathBuf::from("/models/flux/transformer.safetensors"),
            ),
            (
                ModelComponent::Vae,
                PathBuf::from("/models/flux/vae.safetensors"),
            ),
        ];
        let paths = paths_from_downloads(&downloads, "flux");
        assert!(
            paths.is_some(),
            "diffusion model with transformer+vae should work"
        );
        let paths = paths.unwrap();
        assert_eq!(
            paths.transformer,
            PathBuf::from("/models/flux/transformer.safetensors")
        );
        assert_eq!(paths.vae, PathBuf::from("/models/flux/vae.safetensors"));
    }

    #[test]
    fn all_upscaler_manifests_produce_paths() {
        // Regression: every upscaler manifest should produce valid ModelPaths
        for manifest in known_manifests() {
            if !manifest.is_upscaler() {
                continue;
            }
            let downloads: Vec<(ModelComponent, PathBuf)> = manifest
                .files
                .iter()
                .map(|f| {
                    (
                        f.component,
                        PathBuf::from(format!("/fake/{}", f.hf_filename)),
                    )
                })
                .collect();
            let paths = paths_from_downloads(&downloads, &manifest.family);
            assert!(
                paths.is_some(),
                "upscaler manifest '{}' should produce ModelPaths from its downloads",
                manifest.name
            );
        }
    }

    #[test]
    fn all_utility_manifests_produce_paths() {
        // Regression: every utility manifest should produce valid ModelPaths
        for manifest in known_manifests() {
            if !manifest.is_utility() {
                continue;
            }
            let downloads: Vec<(ModelComponent, PathBuf)> = manifest
                .files
                .iter()
                .map(|f| {
                    (
                        f.component,
                        PathBuf::from(format!("/fake/{}", f.hf_filename)),
                    )
                })
                .collect();
            let paths = paths_from_downloads(&downloads, &manifest.family);
            assert!(
                paths.is_some(),
                "utility manifest '{}' should produce ModelPaths from its downloads",
                manifest.name
            );
        }
    }

    #[test]
    fn find_manifest_by_hf_repo_matches_transformer_repo() {
        // FLUX.1-dev is a well-known canonical entry — its transformer
        // ships from black-forest-labs/FLUX.1-dev (BF16 path) and from
        // city96/FLUX.1-dev-gguf (GGUF quantizations). Either repo string
        // must surface a matching manifest.
        assert!(find_manifest_by_hf_repo("black-forest-labs/FLUX.1-dev").is_some());
        assert!(find_manifest_by_hf_repo("city96/FLUX.1-dev-gguf").is_some());
    }

    #[test]
    fn find_manifest_by_hf_repo_returns_none_for_unknown_repo() {
        assert!(find_manifest_by_hf_repo("some-random-org/never-shipped").is_none());
    }

    #[test]
    fn find_manifest_by_hf_repo_ignores_non_transformer_repos() {
        // lmz/mt5-tokenizers ships only T5Tokenizer files across every
        // manifest that references it — never as a Transformer or
        // TransformerShard. The lookup must not surface those manifests
        // when keyed on the tokenizer repo, otherwise an HF catalog row
        // pointing at a tokenizer would mis-route to whatever model
        // happens to bundle that tokenizer.
        //
        // (Note: openai/clip-vit-large-patch14 is NOT a safe choice for
        // this assertion — the `clip-l` companion manifest declares its
        // CLIP weights as `ModelComponent::Transformer`, so the lookup
        // legitimately resolves CLIP-L there.)
        assert!(find_manifest_by_hf_repo("lmz/mt5-tokenizers").is_none());
    }
}
