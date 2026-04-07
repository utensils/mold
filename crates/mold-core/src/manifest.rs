use crate::config::ModelConfig;
use crate::types::Scheduler;
use crate::ModelPaths;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::LazyLock;

/// Model families that are utility models (not image generators).
/// These are excluded from default-model selection and don't produce ModelPaths.
pub const UTILITY_FAMILIES: &[&str] = &["qwen3-expand"];

/// Model families that are upscaler models (image-to-image enhancement, not generation).
/// These are excluded from default-model selection and use a simplified config path.
pub const UPSCALER_FAMILIES: &[&str] = &["upscaler"];

/// Model families that are auxiliary (not standalone generators).
/// ControlNet models are used via `--control-model`, not as the primary model.
pub const AUXILIARY_FAMILIES: &[&str] = &["controlnet"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelComponent {
    Transformer,
    TransformerShard, // One shard of a multi-file transformer (Z-Image BF16)
    Vae,
    SpatialUpscaler, // LTX latent upsampler / spatial upscaler weights
    T5Encoder,
    ClipEncoder,
    T5Tokenizer,
    ClipTokenizer,
    ClipEncoder2,   // CLIP-G / OpenCLIP (SDXL)
    ClipTokenizer2, // CLIP-G tokenizer (SDXL)
    TextEncoder,    // Generic text encoder shard (Qwen3 for Z-Image)
    TextTokenizer,  // Generic text encoder tokenizer
    Decoder,        // Stage B decoder weights (Wuerstchen)
    Upscaler,       // Upscaler model weights (Real-ESRGAN, etc.)
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
            vae: Some(paths.vae.to_string_lossy().to_string()),
            spatial_upscaler: paths
                .spatial_upscaler
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
        ModelComponent::Transformer | ModelComponent::TransformerShard | ModelComponent::Upscaler
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

    if is_model_specific_component(file.component) {
        PathBuf::from(&sanitized_name).join(&file.hf_filename)
    } else {
        if manifest.family == "qwen-image" {
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
    manifests.extend(controlnet_manifests());
    manifests.extend(qwen3_expand_manifests());
    manifests.extend(upscaler_manifests());
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
            },
            hidden: false,
        },
        // GGUF quantized variants (transformer only; shared components are always BF16)
        ModelManifest {
            name: "z-image-turbo:q8".to_string(),
            family: "z-image".to_string(),
            description: "Z-Image Turbo Q8 — 9-step, quantized transformer".to_string(),
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
            },
            hidden: false,
        },
        ModelManifest {
            name: "z-image-turbo:q6".to_string(),
            family: "z-image".to_string(),
            description: "Z-Image Turbo Q6 — 9-step, best quality/size trade-off".to_string(),
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
            },
            hidden: false,
        },
        ModelManifest {
            name: "z-image-turbo:q4".to_string(),
            family: "z-image".to_string(),
            description: "Z-Image Turbo Q4 — 9-step, smallest footprint".to_string(),
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

/// All known Flux.2 model manifests.
fn flux2_manifests() -> Vec<ModelManifest> {
    vec![
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
            },
            hidden: false,
        },
    ]
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
    };
    let qwen_2512_defaults = base_defaults.clone();

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
            },
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
    // Already has a tag
    if input.contains(':') {
        return input.to_string();
    }
    // Legacy format: flux-dev-q4 -> flux-dev:q4
    if let Some((base, suffix)) = input.rsplit_once('-') {
        if suffix.starts_with('q')
            && suffix.len() <= 3
            && suffix[1..].chars().all(|c| c.is_ascii_digit())
        {
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
/// For diffusion models, Transformer (or TransformerShards) and VAE are always required.
/// For upscaler models, only the Upscaler component is required (mapped to `transformer`).
/// For utility models (e.g., qwen3-expand), only the Transformer is required (no VAE).
/// Other components are optional — each engine validates what it needs at load time.
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
            transformer,
            transformer_shards: Vec::new(),
            vae: PathBuf::new(),
            spatial_upscaler: None,
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

    // Transformer: use single Transformer file, or first TransformerShard as primary path
    let transformer =
        find(ModelComponent::Transformer).or_else(|| transformer_shards.first().cloned())?;

    // Utility models (e.g., qwen3-expand): transformer + optional tokenizer, no VAE
    let vae = if UTILITY_FAMILIES.contains(&family) {
        find(ModelComponent::Vae).unwrap_or_default()
    } else {
        find(ModelComponent::Vae)?
    };

    Some(ModelPaths {
        transformer,
        transformer_shards,
        vae,
        spatial_upscaler: find(ModelComponent::SpatialUpscaler),
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
        // SDXL models default to :fp16
        assert_eq!(resolve_model_name("sdxl-base"), "sdxl-base:fp16");
        assert_eq!(resolve_model_name("sdxl-turbo"), "sdxl-turbo:fp16");
        assert_eq!(resolve_model_name("dreamshaper-xl"), "dreamshaper-xl:fp16");
    }

    #[test]
    fn resolve_name_legacy_format() {
        assert_eq!(resolve_model_name("flux-dev-q4"), "flux-dev:q4");
        assert_eq!(resolve_model_name("flux-dev-q8"), "flux-dev:q8");
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
        // 24 FLUX + 3 SD1.5 + 4 SD3 + 8 SDXL + 4 Z-Image + 8 Flux.2 + 17 Qwen-Image + 1 Wuerstchen + 5 LTX Video + 3 ControlNet + 2 Qwen3-Expand + 7 Upscaler = 86
        assert_eq!(known_manifests().len(), 86);
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
    fn manifest_has_required_components() {
        for manifest in known_manifests() {
            let components: Vec<_> = manifest.files.iter().map(|f| f.component).collect();
            // All diffusion models need VAE (except ControlNet, utility models, and upscalers)
            if !manifest.is_utility() && !manifest.is_upscaler() && !manifest.is_auxiliary() {
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
                "qwen-image" => {
                    // Qwen-Image uses TransformerShard (BF16 sharded)
                    assert!(
                        components.contains(&ModelComponent::Transformer)
                            || components.contains(&ModelComponent::TransformerShard),
                        "{} (qwen-image) missing Transformer or TransformerShard",
                        manifest.name
                    );
                    assert!(
                        components.contains(&ModelComponent::TextEncoder),
                        "{} (qwen-image) missing TextEncoder",
                        manifest.name
                    );
                    assert!(
                        components.contains(&ModelComponent::TextTokenizer),
                        "{} (qwen-image) missing TextTokenizer",
                        manifest.name
                    );
                    // Qwen-Image does NOT use CLIP
                    assert!(
                        !components.contains(&ModelComponent::ClipEncoder),
                        "{} (qwen-image) should not have ClipEncoder",
                        manifest.name
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
        // Models with shared files must have total > transformer-only size
        for manifest in known_manifests() {
            if manifest.is_auxiliary() || manifest.is_upscaler() {
                continue; // ControlNet and upscalers are single-file models
            }
            let transformer_bytes: u64 = manifest
                .files
                .iter()
                .filter(|f| {
                    f.component == ModelComponent::Transformer
                        || f.component == ModelComponent::TransformerShard
                })
                .map(|f| f.size_bytes)
                .sum();
            let total = manifest.total_size_bytes();
            assert!(
                total >= transformer_bytes,
                "{}: total ({}) should be >= transformer-only ({})",
                manifest.name,
                total,
                transformer_bytes
            );
            assert_eq!(
                manifest.model_size_bytes(),
                transformer_bytes,
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
        // Currently 2: qwen3-expand:q8, qwen3-expand-small:q8
        assert_eq!(
            utility_count, 2,
            "expected exactly 2 utility models, got {utility_count}"
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
        // Each entry: (family, path, cache_check_pattern, encoder_load_pattern)
        // The cache check must appear BEFORE the encoder load in generate_sequential().
        let pipelines = [
            (
                "flux2",
                "crates/mold-inference/src/flux2/pipeline.rs",
                "restore_cached_tensor(",
                "Qwen3Encoder::load_",
            ),
            (
                "flux",
                "crates/mold-inference/src/flux/pipeline.rs",
                "restore_prompt_cache(",
                "T5Encoder::load",
            ),
            (
                "sd15",
                "crates/mold-inference/src/sd15/pipeline.rs",
                "restore_cached_tensor(",
                "build_clip_transformer(",
            ),
            (
                "sdxl",
                "crates/mold-inference/src/sdxl/pipeline.rs",
                "restore_cached_tensor(",
                "build_clip_transformer(",
            ),
            (
                "sd3",
                "crates/mold-inference/src/sd3/pipeline.rs",
                "restore_cached_tensor_pair(",
                "SD3TripleEncoder::load(",
            ),
            (
                "zimage",
                "crates/mold-inference/src/zimage/pipeline.rs",
                "restore_cached_tensor(",
                "Qwen3Encoder::load_",
            ),
            (
                "qwen_image",
                "crates/mold-inference/src/qwen_image/pipeline.rs",
                "get_cloned(&prompt_key)",
                "load_text_encoder(",
            ),
            (
                "wuerstchen",
                "crates/mold-inference/src/wuerstchen/pipeline.rs",
                "restore_cached_tensor_pair(",
                "build_clip_transformer(",
            ),
        ];

        let workspace = env!("CARGO_MANIFEST_DIR")
            .strip_suffix("/crates/mold-core")
            .or_else(|| env!("CARGO_MANIFEST_DIR").strip_suffix("crates/mold-core"))
            .unwrap_or(env!("CARGO_MANIFEST_DIR"));

        for (family, path, cache_pattern, load_pattern) in pipelines {
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
            let load_pos = seq_body.find(load_pattern).unwrap_or_else(|| {
                panic!(
                    "{family} pipeline ({path}) generate_sequential() missing encoder load '{load_pattern}'"
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
}
