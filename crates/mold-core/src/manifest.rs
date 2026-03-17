use crate::config::ModelConfig;
use crate::ModelPaths;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelComponent {
    Transformer,
    TransformerShard, // One shard of a multi-file transformer (Z-Image BF16)
    Vae,
    T5Encoder,
    ClipEncoder,
    T5Tokenizer,
    ClipTokenizer,
    ClipEncoder2,   // CLIP-G / OpenCLIP (SDXL)
    ClipTokenizer2, // CLIP-G tokenizer (SDXL)
    TextEncoder,    // Generic text encoder shard (Qwen3 for Z-Image)
    TextTokenizer,  // Generic text encoder tokenizer
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
    /// Scheduler type: None for FLUX (uses flow-matching), "ddim" or "euler_ancestral" for SDXL.
    pub scheduler: Option<&'static str>,
}

#[derive(Debug, Clone)]
pub struct ModelManifest {
    pub name: String,
    pub family: String,
    pub description: String,
    pub size_gb: f32,
    pub files: Vec<ModelFile>,
    pub defaults: ManifestDefaults,
}

impl ModelManifest {
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
            default_steps: Some(self.defaults.steps),
            default_guidance: Some(self.defaults.guidance),
            default_width: Some(self.defaults.width),
            default_height: Some(self.defaults.height),
            is_schnell: Some(self.defaults.is_schnell),
            scheduler: self.defaults.scheduler.map(|s| s.to_string()),
            description: Some(self.description.clone()),
            family: Some(self.family.clone()),
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
            size_bytes: 335_000_000, // ~335MB
            gated: true,             // BFL repos now require authentication
            sha256: Some("afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38"),
        },
        ModelFile {
            hf_repo: "comfyanonymous/flux_text_encoders".to_string(),
            hf_filename: "t5xxl_fp16.safetensors".to_string(),
            component: ModelComponent::T5Encoder,
            size_bytes: 9_200_000_000, // ~9.2GB
            gated: false,
            sha256: Some("6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635"),
        },
        ModelFile {
            hf_repo: "comfyanonymous/flux_text_encoders".to_string(),
            hf_filename: "clip_l.safetensors".to_string(),
            component: ModelComponent::ClipEncoder,
            size_bytes: 246_000_000, // ~246MB
            gated: false,
            sha256: Some("660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd"),
        },
        ModelFile {
            hf_repo: "lmz/mt5-tokenizers".to_string(),
            hf_filename: "t5-v1_1-xxl.tokenizer.json".to_string(),
            component: ModelComponent::T5Tokenizer,
            size_bytes: 2_400_000, // ~2.4MB
            gated: false,
            sha256: None, // non-LFS file, no SHA-256 from HF API
        },
        ModelFile {
            hf_repo: "openai/clip-vit-large-patch14".to_string(),
            hf_filename: "tokenizer.json".to_string(),
            component: ModelComponent::ClipTokenizer,
            size_bytes: 600_000, // ~600KB
            gated: false,
            sha256: None, // non-LFS file, no SHA-256 from HF API
        },
    ]
}

/// All known downloadable model manifests (FLUX + SDXL).
pub fn known_manifests() -> Vec<ModelManifest> {
    let mut manifests = vec![
        ModelManifest {
            name: "flux-schnell:q8".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Schnell Q8 — fast 4-step, general purpose".to_string(),
            size_gb: 12.0,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "city96/FLUX.1-schnell-gguf".to_string(),
                    hf_filename: "flux1-schnell-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 12_000_000_000, // ~12GB
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
            },
        },
        ModelManifest {
            name: "flux-dev:q8".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Dev Q8 — full quality, 20+ steps".to_string(),
            size_gb: 12.0,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "city96/FLUX.1-dev-gguf".to_string(),
                    hf_filename: "flux1-dev-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 12_000_000_000, // ~12GB
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
            },
        },
        ModelManifest {
            name: "flux-dev:q4".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Dev Q4 — smaller/faster, good quality".to_string(),
            size_gb: 7.0,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "city96/FLUX.1-dev-gguf".to_string(),
                    hf_filename: "flux1-dev-Q4_1.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 7_000_000_000, // ~7GB
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
            },
        },
        ModelManifest {
            name: "flux-dev:q6".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Dev Q6 — best quality/size trade-off".to_string(),
            size_gb: 9.9,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "city96/FLUX.1-dev-gguf".to_string(),
                    hf_filename: "flux1-dev-Q6_K.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 9_860_000_000, // ~9.86GB
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
            },
        },
        ModelManifest {
            name: "flux-schnell:q4".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Schnell Q4 — fast 4-step, smaller footprint".to_string(),
            size_gb: 7.5,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "city96/FLUX.1-schnell-gguf".to_string(),
                    hf_filename: "flux1-schnell-Q4_1.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 7_510_000_000, // ~7.51GB
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
            },
        },
        ModelManifest {
            name: "flux-schnell:q6".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Schnell Q6 — fast 4-step, best quality/size trade-off".to_string(),
            size_gb: 9.8,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "city96/FLUX.1-schnell-gguf".to_string(),
                    hf_filename: "flux1-schnell-Q6_K.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 9_830_000_000, // ~9.83GB
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
            },
        },
        ModelManifest {
            name: "flux-krea:q8".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Krea Dev Q8 — aesthetic photography fine-tune".to_string(),
            size_gb: 12.7,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "QuantStack/FLUX.1-Krea-dev-GGUF".to_string(),
                    hf_filename: "flux1-krea-dev-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 12_700_000_000, // ~12.7GB
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
            },
        },
        ModelManifest {
            name: "flux-krea:q4".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Krea Dev Q4 — aesthetic photography, smaller footprint"
                .to_string(),
            size_gb: 7.5,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "QuantStack/FLUX.1-Krea-dev-GGUF".to_string(),
                    hf_filename: "flux1-krea-dev-Q4_1.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 7_530_000_000, // ~7.53GB
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
            },
        },
        ModelManifest {
            name: "flux-krea:q6".to_string(),
            family: "flux".to_string(),
            description: "FLUX.1 Krea Dev Q6 — aesthetic photography, best quality/size trade-off"
                .to_string(),
            size_gb: 9.9,
            files: {
                let mut files = shared_flux_files();
                files.push(ModelFile {
                    hf_repo: "QuantStack/FLUX.1-Krea-dev-GGUF".to_string(),
                    hf_filename: "flux1-krea-dev-Q6_K.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 9_860_000_000, // ~9.86GB
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
            },
        },
    ];
    manifests.extend(sd15_manifests());
    manifests.extend(sd3_manifests());
    manifests.extend(sdxl_manifests());
    manifests.extend(zimage_manifests());
    manifests.extend(flux2_manifests());
    manifests.extend(qwen_image_manifests());
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
            size_bytes: 16_460_000_000, // ~16.46GB monolithic (VAE portion ~300MB via mmap)
            gated: true,
            sha256: None,
        },
        // CLIP-L encoder
        ModelFile {
            hf_repo: "stabilityai/stable-diffusion-3.5-large".to_string(),
            hf_filename: "text_encoders/clip_l.safetensors".to_string(),
            component: ModelComponent::ClipEncoder,
            size_bytes: 246_000_000, // ~246MB
            gated: true,
            sha256: None,
        },
        // CLIP-G encoder
        ModelFile {
            hf_repo: "stabilityai/stable-diffusion-3.5-large".to_string(),
            hf_filename: "text_encoders/clip_g.safetensors".to_string(),
            component: ModelComponent::ClipEncoder2,
            size_bytes: 1_390_000_000, // ~1.39GB
            gated: true,
            sha256: None,
        },
        // T5-XXL encoder
        ModelFile {
            hf_repo: "stabilityai/stable-diffusion-3.5-large".to_string(),
            hf_filename: "text_encoders/t5xxl_fp16.safetensors".to_string(),
            component: ModelComponent::T5Encoder,
            size_bytes: 9_200_000_000, // ~9.2GB
            gated: true,
            sha256: None,
        },
        // CLIP-L tokenizer (same as FLUX/SDXL)
        ModelFile {
            hf_repo: "openai/clip-vit-large-patch14".to_string(),
            hf_filename: "tokenizer.json".to_string(),
            component: ModelComponent::ClipTokenizer,
            size_bytes: 600_000, // ~600KB
            gated: false,
            sha256: None,
        },
        // CLIP-G tokenizer (same as SDXL)
        ModelFile {
            hf_repo: "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k".to_string(),
            hf_filename: "tokenizer.json".to_string(),
            component: ModelComponent::ClipTokenizer2,
            size_bytes: 600_000, // ~600KB
            gated: false,
            sha256: None,
        },
        // T5 tokenizer (same as FLUX)
        ModelFile {
            hf_repo: "lmz/mt5-tokenizers".to_string(),
            hf_filename: "t5-v1_1-xxl.tokenizer.json".to_string(),
            component: ModelComponent::T5Tokenizer,
            size_bytes: 2_400_000, // ~2.4MB
            gated: false,
            sha256: None,
        },
    ]
}

/// Size of shared SD3 components (CLIP-L, CLIP-G, T5-XXL, tokenizers) in GB.
/// Does not include VAE since it is part of the transformer weights for SD3.
pub const SHARED_SD3_COMPONENTS_GB: f32 = 10.8;

/// All known SD3.5 model manifests.
fn sd3_manifests() -> Vec<ModelManifest> {
    vec![
        // --- SD3.5 Large (depth=38, 8.1B) ---
        ModelManifest {
            name: "sd3.5-large:q8".to_string(),
            family: "sd3".to_string(),
            description: "SD3.5 Large Q8 — 8.1B MMDiT, high quality, 28 steps".to_string(),
            size_gb: 8.5,
            files: {
                let mut files = shared_sd3_files();
                files.push(ModelFile {
                    hf_repo: "city96/stable-diffusion-3.5-large-gguf".to_string(),
                    hf_filename: "sd3.5_large-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 8_500_000_000, // ~8.5GB
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
            },
        },
        ModelManifest {
            name: "sd3.5-large:q4".to_string(),
            family: "sd3".to_string(),
            description: "SD3.5 Large Q4 — 8.1B MMDiT, smaller footprint, 28 steps".to_string(),
            size_gb: 5.0,
            files: {
                let mut files = shared_sd3_files();
                files.push(ModelFile {
                    hf_repo: "city96/stable-diffusion-3.5-large-gguf".to_string(),
                    hf_filename: "sd3.5_large-Q4_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 5_000_000_000, // ~5GB
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
            },
        },
        // --- SD3.5 Large Turbo (depth=38, 8.1B, 4 steps, CFG=1.0) ---
        ModelManifest {
            name: "sd3.5-large-turbo:q8".to_string(),
            family: "sd3".to_string(),
            description: "SD3.5 Large Turbo Q8 — 8.1B MMDiT, fast 4-step generation".to_string(),
            size_gb: 8.5,
            files: {
                let mut files = shared_sd3_files();
                files.push(ModelFile {
                    hf_repo: "city96/stable-diffusion-3.5-large-turbo-gguf".to_string(),
                    hf_filename: "sd3.5_large_turbo-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 8_500_000_000, // ~8.5GB
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
            },
        },
        // --- SD3.5 Medium (depth=24, 2.5B) ---
        ModelManifest {
            name: "sd3.5-medium:q8".to_string(),
            family: "sd3".to_string(),
            description: "SD3.5 Medium Q8 — 2.5B MMDiT, SLG support, 28 steps".to_string(),
            size_gb: 2.7,
            files: {
                let mut files = shared_sd3_files();
                files.push(ModelFile {
                    hf_repo: "city96/stable-diffusion-3.5-medium-gguf".to_string(),
                    hf_filename: "sd3.5_medium-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 2_700_000_000, // ~2.7GB
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
            },
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
            size_bytes: 335_000_000, // ~335MB
            gated: false,
            sha256: Some("a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815"),
        },
        ModelFile {
            hf_repo: "runwayml/stable-diffusion-v1-5".to_string(),
            hf_filename: "text_encoder/model.safetensors".to_string(),
            component: ModelComponent::ClipEncoder,
            size_bytes: 492_000_000, // ~492MB (CLIP-L)
            gated: false,
            sha256: Some("d008943c017f0092921106440254dbbe00b6a285f7883ec8ba160c3faad88334"),
        },
        ModelFile {
            hf_repo: "openai/clip-vit-large-patch14".to_string(),
            hf_filename: "tokenizer.json".to_string(),
            component: ModelComponent::ClipTokenizer,
            size_bytes: 600_000, // ~600KB
            gated: false,
            sha256: None, // non-LFS file, no SHA-256 from HF API
        },
    ]
}

/// Size of shared SD1.5 components (VAE, CLIP-L, tokenizer) in GB.
pub const SHARED_SD15_COMPONENTS_GB: f32 = 0.8;

/// All known SD1.5 model manifests.
fn sd15_manifests() -> Vec<ModelManifest> {
    vec![
        ModelManifest {
            name: "sd15:fp16".to_string(),
            family: "sd15".to_string(),
            description: "Stable Diffusion 1.5 — canonical base model, huge LoRA ecosystem"
                .to_string(),
            size_gb: 1.7,
            files: {
                let mut files = shared_sd15_files();
                files.push(ModelFile {
                    hf_repo: "runwayml/stable-diffusion-v1-5".to_string(),
                    hf_filename: "unet/diffusion_pytorch_model.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 1_720_000_000, // ~1.72GB
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
                scheduler: Some("ddim"),
            },
        },
        ModelManifest {
            name: "dreamshaper-v8:fp16".to_string(),
            family: "sd15".to_string(),
            description: "DreamShaper v8 — best versatile SD1.5, photorealistic + fantasy"
                .to_string(),
            size_gb: 1.7,
            files: {
                let mut files = shared_sd15_files();
                files.push(ModelFile {
                    hf_repo: "Lykon/dreamshaper-8".to_string(),
                    hf_filename: "unet/diffusion_pytorch_model.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 1_720_000_000, // ~1.72GB
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
                scheduler: Some("ddim"),
            },
        },
        ModelManifest {
            name: "realistic-vision-v5:fp16".to_string(),
            family: "sd15".to_string(),
            description: "Realistic Vision v5.1 — gold standard photorealistic SD1.5".to_string(),
            size_gb: 1.7,
            files: {
                let mut files = shared_sd15_files();
                files.push(ModelFile {
                    hf_repo: "SG161222/Realistic_Vision_V5.1_noVAE".to_string(),
                    hf_filename: "unet/diffusion_pytorch_model.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 1_720_000_000, // ~1.72GB
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
                scheduler: Some("ddim"),
            },
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
            size_bytes: 335_000_000, // ~335MB
            gated: false,
            sha256: Some("1b909373b28f2137098b0fd9dbc6f97f8410854f31f84ddc9fa04b077b0ace2c"),
        },
        ModelFile {
            hf_repo: "stabilityai/stable-diffusion-xl-base-1.0".to_string(),
            hf_filename: "text_encoder/model.safetensors".to_string(),
            component: ModelComponent::ClipEncoder,
            size_bytes: 492_000_000, // ~492MB (CLIP-L)
            gated: false,
            sha256: Some("5c3d6454dd2d23414b56aa1b5858a72487a656937847b6fea8d0606d7a42cdbc"),
        },
        ModelFile {
            hf_repo: "stabilityai/stable-diffusion-xl-base-1.0".to_string(),
            hf_filename: "text_encoder_2/model.safetensors".to_string(),
            component: ModelComponent::ClipEncoder2,
            size_bytes: 1_390_000_000, // ~1.39GB (CLIP-G / OpenCLIP)
            gated: false,
            sha256: Some("3a6032f63d37ae02bbc74ccd6a27440578cd71701f96532229d0154f55a8d3ff"),
        },
        ModelFile {
            hf_repo: "openai/clip-vit-large-patch14".to_string(),
            hf_filename: "tokenizer.json".to_string(),
            component: ModelComponent::ClipTokenizer,
            size_bytes: 600_000, // ~600KB
            gated: false,
            sha256: None, // non-LFS file, no SHA-256 from HF API
        },
        ModelFile {
            hf_repo: "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k".to_string(),
            hf_filename: "tokenizer.json".to_string(),
            component: ModelComponent::ClipTokenizer2,
            size_bytes: 600_000, // ~600KB
            gated: false,
            sha256: None, // non-LFS file, no SHA-256 from HF API
        },
    ]
}

/// Size of shared SDXL components (VAE, dual-CLIP, tokenizers) in GB.
pub const SHARED_SDXL_COMPONENTS_GB: f32 = 2.2;

/// All known SDXL model manifests.
fn sdxl_manifests() -> Vec<ModelManifest> {
    vec![
        // --- Standard SDXL (DDIM scheduler, 20-30 steps, guidance 7.5) ---
        ModelManifest {
            name: "sdxl-base:fp16".to_string(),
            family: "sdxl".to_string(),
            description: "SDXL Base 1.0 — official Stability AI base model".to_string(),
            size_gb: 5.14,
            files: {
                let mut files = shared_sdxl_files();
                files.push(ModelFile {
                    hf_repo: "stabilityai/stable-diffusion-xl-base-1.0".to_string(),
                    hf_filename: "unet/diffusion_pytorch_model.fp16.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 5_140_000_000,
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
                scheduler: Some("ddim"),
            },
        },
        ModelManifest {
            name: "dreamshaper-xl:fp16".to_string(),
            family: "sdxl".to_string(),
            description: "DreamShaper XL — fantasy, concept art, stylized".to_string(),
            size_gb: 5.14,
            files: {
                let mut files = shared_sdxl_files();
                files.push(ModelFile {
                    hf_repo: "Lykon/dreamshaper-xl-v2-turbo".to_string(),
                    hf_filename: "unet/diffusion_pytorch_model.fp16.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 5_140_000_000,
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
                scheduler: Some("euler_ancestral"),
            },
        },
        ModelManifest {
            name: "juggernaut-xl:fp16".to_string(),
            family: "sdxl".to_string(),
            description: "Juggernaut XL — photorealism, cinematic lighting".to_string(),
            size_gb: 5.14,
            files: {
                let mut files = shared_sdxl_files();
                files.push(ModelFile {
                    hf_repo: "RunDiffusion/Juggernaut-XL-v9".to_string(),
                    hf_filename: "unet/diffusion_pytorch_model.fp16.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 5_140_000_000,
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
                scheduler: Some("ddim"),
            },
        },
        ModelManifest {
            name: "realvis-xl:fp16".to_string(),
            family: "sdxl".to_string(),
            description: "RealVisXL V5.0 — photorealism, versatile subjects".to_string(),
            size_gb: 5.14,
            files: {
                let mut files = shared_sdxl_files();
                files.push(ModelFile {
                    hf_repo: "SG161222/RealVisXL_V5.0".to_string(),
                    hf_filename: "unet/diffusion_pytorch_model.fp16.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 5_140_000_000,
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
                scheduler: Some("ddim"),
            },
        },
        ModelManifest {
            name: "playground-v2.5:fp16".to_string(),
            family: "sdxl".to_string(),
            description: "Playground v2.5 — aesthetic quality, artistic".to_string(),
            size_gb: 5.14,
            files: {
                let mut files = shared_sdxl_files();
                files.push(ModelFile {
                    hf_repo: "playgroundai/playground-v2.5-1024px-aesthetic".to_string(),
                    hf_filename: "unet/diffusion_pytorch_model.fp16.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 5_140_000_000,
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
                scheduler: Some("ddim"),
            },
        },
        // --- Turbo SDXL (Euler Ancestral, 1-4 steps, guidance 0.0) ---
        ModelManifest {
            name: "sdxl-turbo:fp16".to_string(),
            family: "sdxl".to_string(),
            description: "SDXL Turbo — ultra-fast 1-4 step generation".to_string(),
            size_gb: 5.14,
            files: {
                let mut files = shared_sdxl_files();
                files.push(ModelFile {
                    hf_repo: "stabilityai/sdxl-turbo".to_string(),
                    hf_filename: "unet/diffusion_pytorch_model.fp16.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 5_140_000_000,
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
                scheduler: Some("euler_ancestral"),
            },
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
            size_bytes: 3_960_000_000, // ~3.96GB
            gated: false,
            sha256: Some("328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223"),
        },
        ModelFile {
            hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
            hf_filename: "text_encoder/model-00002-of-00003.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 3_990_000_000, // ~3.99GB
            gated: false,
            sha256: Some("6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5"),
        },
        ModelFile {
            hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
            hf_filename: "text_encoder/model-00003-of-00003.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 99_600_000, // ~99.6MB
            gated: false,
            sha256: Some("7ca841ee75b9c61267c0c6148fd8d096d3d21b6d3e161256a9b878154f91fc52"),
        },
        // VAE
        ModelFile {
            hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
            hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
            component: ModelComponent::Vae,
            size_bytes: 168_000_000, // ~168MB
            gated: false,
            sha256: Some("f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3"),
        },
        // Qwen3 tokenizer
        ModelFile {
            hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
            hf_filename: "tokenizer/tokenizer.json".to_string(),
            component: ModelComponent::TextTokenizer,
            size_bytes: 11_400_000, // ~11.4MB
            gated: false,
            sha256: Some("aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
        },
    ]
}

/// Size of shared Z-Image components (Qwen3 text encoder + VAE + tokenizer) in GB.
pub const SHARED_ZIMAGE_COMPONENTS_GB: f32 = 8.2;

/// All known Z-Image model manifests.
fn zimage_manifests() -> Vec<ModelManifest> {
    vec![
        // BF16 full precision
        ModelManifest {
            name: "z-image-turbo:bf16".to_string(),
            family: "z-image".to_string(),
            description: "Z-Image Turbo BF16 — 9-step, Alibaba flow-matching".to_string(),
            size_gb: 24.6,
            files: {
                let mut files = shared_zimage_files();
                // Transformer shards (3 files)
                files.push(ModelFile {
                    hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
                    hf_filename: "transformer/diffusion_pytorch_model-00001-of-00003.safetensors"
                        .to_string(),
                    component: ModelComponent::TransformerShard,
                    size_bytes: 9_970_000_000, // ~9.97GB
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
                    size_bytes: 9_970_000_000, // ~9.97GB
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
                scheduler: Some("flow_match_euler"),
            },
        },
        // GGUF quantized variants (transformer only; shared components are always BF16)
        ModelManifest {
            name: "z-image-turbo:q8".to_string(),
            family: "z-image".to_string(),
            description: "Z-Image Turbo Q8 — 9-step, quantized transformer".to_string(),
            size_gb: 6.58,
            files: {
                let mut files = shared_zimage_files();
                files.push(ModelFile {
                    hf_repo: "leejet/Z-Image-Turbo-GGUF".to_string(),
                    hf_filename: "z_image_turbo-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 6_580_000_000, // ~6.58GB
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
                scheduler: Some("flow_match_euler"),
            },
        },
        ModelManifest {
            name: "z-image-turbo:q6".to_string(),
            family: "z-image".to_string(),
            description: "Z-Image Turbo Q6 — 9-step, best quality/size trade-off".to_string(),
            size_gb: 5.26,
            files: {
                let mut files = shared_zimage_files();
                files.push(ModelFile {
                    hf_repo: "leejet/Z-Image-Turbo-GGUF".to_string(),
                    hf_filename: "z_image_turbo-Q6_K.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 5_260_000_000, // ~5.26GB
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
                scheduler: Some("flow_match_euler"),
            },
        },
        ModelManifest {
            name: "z-image-turbo:q4".to_string(),
            family: "z-image".to_string(),
            description: "Z-Image Turbo Q4 — 9-step, smallest footprint".to_string(),
            size_gb: 3.86,
            files: {
                let mut files = shared_zimage_files();
                files.push(ModelFile {
                    hf_repo: "leejet/Z-Image-Turbo-GGUF".to_string(),
                    hf_filename: "z_image_turbo-Q4_K.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 3_860_000_000, // ~3.86GB
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
                scheduler: Some("flow_match_euler"),
            },
        },
    ]
}

/// Shared Flux.2 Klein-4B component files (Qwen3 text encoder, VAE, tokenizer).
///
/// Klein uses Qwen3 (hidden_size=2560, 36 layers) — same model architecture as Z-Image's
/// text encoder. The encoder stacks 3 hidden state outputs to produce joint_attention_dim=7680.
fn shared_flux2_files() -> Vec<ModelFile> {
    vec![
        // Qwen3 text encoder (from the Klein repo)
        ModelFile {
            hf_repo: "black-forest-labs/FLUX.2-klein-4B".to_string(),
            hf_filename: "text_encoder/model.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 5_400_000_000, // ~5.4GB estimated
            gated: false,
            sha256: None,
        },
        // VAE
        ModelFile {
            hf_repo: "black-forest-labs/FLUX.2-klein-4B".to_string(),
            hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
            component: ModelComponent::Vae,
            size_bytes: 335_000_000, // ~335MB estimated
            gated: false,
            sha256: None,
        },
        // Qwen3 tokenizer
        ModelFile {
            hf_repo: "black-forest-labs/FLUX.2-klein-4B".to_string(),
            hf_filename: "tokenizer/tokenizer.json".to_string(),
            component: ModelComponent::TextTokenizer,
            size_bytes: 11_400_000, // ~11.4MB
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
            size_gb: 13.5,
            files: {
                let mut files = shared_flux2_files();
                files.push(ModelFile {
                    hf_repo: "black-forest-labs/FLUX.2-klein-4B".to_string(),
                    hf_filename: "transformer/diffusion_pytorch_model.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 7_700_000_000, // ~7.7GB estimated for 4B params in BF16
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: ManifestDefaults {
                steps: 20,
                guidance: 0.0, // Klein is distilled, no CFG needed
                width: 1024,
                height: 1024,
                is_schnell: false,
                scheduler: None, // Uses flow-matching Euler
            },
        },
    ]
}

/// Shared Qwen-Image component files (VAE, text encoder shards, tokenizer).
fn shared_qwen_image_files() -> Vec<ModelFile> {
    vec![
        // VAE
        ModelFile {
            hf_repo: "Qwen/Qwen-Image-2512".to_string(),
            hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
            component: ModelComponent::Vae,
            size_bytes: 300_000_000, // ~300MB estimated
            gated: false,
            sha256: None,
        },
        // Qwen2.5-VL text encoder shards
        ModelFile {
            hf_repo: "Qwen/Qwen-Image-2512".to_string(),
            hf_filename: "text_encoder/model-00001-of-00004.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 4_900_000_000, // ~4.9GB
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "Qwen/Qwen-Image-2512".to_string(),
            hf_filename: "text_encoder/model-00002-of-00004.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 4_700_000_000, // ~4.7GB
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "Qwen/Qwen-Image-2512".to_string(),
            hf_filename: "text_encoder/model-00003-of-00004.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 4_700_000_000, // ~4.7GB
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "Qwen/Qwen-Image-2512".to_string(),
            hf_filename: "text_encoder/model-00004-of-00004.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 1_200_000_000, // ~1.2GB
            gated: false,
            sha256: None,
        },
        // Tokenizer (Qwen2.5 tokenizer — Qwen-Image-2512 repo only ships
        // split BPE files; the compiled tokenizer.json lives in the base model repo)
        ModelFile {
            hf_repo: "Qwen/Qwen2.5-7B".to_string(),
            hf_filename: "tokenizer.json".to_string(),
            component: ModelComponent::TextTokenizer,
            size_bytes: 7_000_000, // ~7MB
            gated: false,
            sha256: None,
        },
    ]
}

/// All known Qwen-Image model manifests.
fn qwen_image_manifests() -> Vec<ModelManifest> {
    let defaults = ManifestDefaults {
        steps: 30,
        guidance: 0.0,
        width: 1024,
        height: 1024,
        is_schnell: false,
        scheduler: Some("flow_match_euler"),
    };

    vec![
        // BF16 full precision (sharded safetensors from official repo)
        ModelManifest {
            name: "qwen-image:bf16".to_string(),
            family: "qwen-image".to_string(),
            description: "Qwen-Image-2512 BF16 — 60-block flow-matching transformer".to_string(),
            size_gb: 30.0,
            files: {
                let mut files = shared_qwen_image_files();
                files.push(ModelFile {
                    hf_repo: "Qwen/Qwen-Image-2512".to_string(),
                    hf_filename: "transformer/diffusion_pytorch_model-00001-of-00002.safetensors"
                        .to_string(),
                    component: ModelComponent::TransformerShard,
                    size_bytes: 9_900_000_000,
                    gated: false,
                    sha256: None,
                });
                files.push(ModelFile {
                    hf_repo: "Qwen/Qwen-Image-2512".to_string(),
                    hf_filename: "transformer/diffusion_pytorch_model-00002-of-00002.safetensors"
                        .to_string(),
                    component: ModelComponent::TransformerShard,
                    size_bytes: 4_700_000_000,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: defaults.clone(),
        },
        // GGUF quantized variants (transformer only; shared components stay BF16)
        ModelManifest {
            name: "qwen-image:q8".to_string(),
            family: "qwen-image".to_string(),
            description: "Qwen-Image-2512 Q8 — quantized transformer, best quality".to_string(),
            size_gb: 21.8,
            files: {
                let mut files = shared_qwen_image_files();
                files.push(ModelFile {
                    hf_repo: "city96/Qwen-Image-gguf".to_string(),
                    hf_filename: "qwen-image-Q8_0.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 21_800_000_000,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: defaults.clone(),
        },
        ModelManifest {
            name: "qwen-image:q6".to_string(),
            family: "qwen-image".to_string(),
            description: "Qwen-Image-2512 Q6 — quantized, best quality/size trade-off".to_string(),
            size_gb: 16.8,
            files: {
                let mut files = shared_qwen_image_files();
                files.push(ModelFile {
                    hf_repo: "city96/Qwen-Image-gguf".to_string(),
                    hf_filename: "qwen-image-Q6_K.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 16_800_000_000,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults: defaults.clone(),
        },
        ModelManifest {
            name: "qwen-image:q4".to_string(),
            family: "qwen-image".to_string(),
            description: "Qwen-Image-2512 Q4 — quantized, smallest practical footprint".to_string(),
            size_gb: 12.3,
            files: {
                let mut files = shared_qwen_image_files();
                files.push(ModelFile {
                    hf_repo: "city96/Qwen-Image-gguf".to_string(),
                    hf_filename: "qwen-image-Q4_K_S.gguf".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 12_300_000_000,
                    gated: false,
                    sha256: None,
                });
                files
            },
            defaults,
        },
    ]
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
    // Try :q8 first (FLUX convention), then :fp16 (SDXL convention), then :bf16 (Z-Image convention)
    let q8 = format!("{input}:q8");
    if find_manifest_exact(&q8).is_some() {
        return q8;
    }
    let fp16 = format!("{input}:fp16");
    if find_manifest_exact(&fp16).is_some() {
        return fp16;
    }
    let bf16 = format!("{input}:bf16");
    if find_manifest_exact(&bf16).is_some() {
        return bf16;
    }
    // Fallback to :q8 for backward compatibility
    format!("{input}:q8")
}

/// Find a manifest by exact name (no resolution). Used internally to avoid
/// circular dependency in `resolve_model_name`.
fn find_manifest_exact(name: &str) -> Option<ModelManifest> {
    known_manifests().into_iter().find(|m| m.name == name)
}

/// Find a manifest by name, handling tag resolution and legacy names.
pub fn find_manifest(name: &str) -> Option<ModelManifest> {
    let canonical = resolve_model_name(name);
    known_manifests().into_iter().find(|m| m.name == canonical)
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

/// FP16 T5-XXL model size in bytes (~9.2GB).
pub const T5_FP16_SIZE: u64 = 9_200_000_000;

/// BF16 Qwen3-4B text encoder size in bytes (~8.2GB, 3 safetensors shards).
pub const QWEN3_FP16_SIZE: u64 = 8_200_000_000;

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
            size_bytes: 5_060_000_000, // ~5.06GB
        },
        T5Variant {
            tag: "q6",
            hf_repo: "city96/t5-v1_1-xxl-encoder-gguf",
            hf_filename: "t5-v1_1-xxl-encoder-Q6_K.gguf",
            size_bytes: 3_910_000_000, // ~3.91GB
        },
        T5Variant {
            tag: "q5",
            hf_repo: "city96/t5-v1_1-xxl-encoder-gguf",
            hf_filename: "t5-v1_1-xxl-encoder-Q5_K_M.gguf",
            size_bytes: 3_390_000_000, // ~3.39GB
        },
        T5Variant {
            tag: "q4",
            hf_repo: "city96/t5-v1_1-xxl-encoder-gguf",
            hf_filename: "t5-v1_1-xxl-encoder-Q4_K_M.gguf",
            size_bytes: 2_900_000_000, // ~2.9GB
        },
        T5Variant {
            tag: "q3",
            hf_repo: "city96/t5-v1_1-xxl-encoder-gguf",
            hf_filename: "t5-v1_1-xxl-encoder-Q3_K_S.gguf",
            size_bytes: 2_100_000_000, // ~2.1GB
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
            size_bytes: 4_280_000_000, // ~4.28GB
        },
        Qwen3Variant {
            tag: "q6",
            hf_repo: "worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF",
            hf_filename: "Qwen_3_4b-Q6_K.gguf",
            size_bytes: 3_310_000_000, // ~3.31GB
        },
        Qwen3Variant {
            tag: "iq4",
            hf_repo: "worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF",
            hf_filename: "Qwen_3_4b-imatrix-IQ4_XS.gguf",
            size_bytes: 2_270_000_000, // ~2.27GB
        },
        Qwen3Variant {
            tag: "q3",
            hf_repo: "worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF",
            hf_filename: "Qwen_3_4b-imatrix-Q3_K_M.gguf",
            size_bytes: 2_080_000_000, // ~2.08GB
        },
    ];
    VARIANTS
}

/// Find a Qwen3 variant by tag (e.g. "q8", "q6", "iq4", "q3").
pub fn find_qwen3_variant(tag: &str) -> Option<&'static Qwen3Variant> {
    known_qwen3_variants().iter().find(|v| v.tag == tag)
}

/// Size of shared FLUX components (VAE, T5, CLIP, tokenizers) in GB.
pub const SHARED_COMPONENTS_GB: f32 = 9.8;

/// Total size of all files in the manifest in bytes.
pub fn total_download_size(manifest: &ModelManifest) -> u64 {
    manifest.files.iter().map(|f| f.size_bytes).sum()
}

/// Convert a `ModelManifest` to a `ModelPaths` from resolved download paths.
/// Transformer (or TransformerShards) and VAE are always required.
/// Other components are optional — each engine validates what it needs at load time.
pub fn paths_from_downloads(downloads: &[(ModelComponent, PathBuf)]) -> Option<ModelPaths> {
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

    let transformer_shards = collect(ModelComponent::TransformerShard);

    // Transformer: use single Transformer file, or first TransformerShard as primary path
    let transformer =
        find(ModelComponent::Transformer).or_else(|| transformer_shards.first().cloned())?;

    Some(ModelPaths {
        transformer,
        transformer_shards,
        vae: find(ModelComponent::Vae)?,
        t5_encoder: find(ModelComponent::T5Encoder),
        clip_encoder: find(ModelComponent::ClipEncoder),
        t5_tokenizer: find(ModelComponent::T5Tokenizer),
        clip_tokenizer: find(ModelComponent::ClipTokenizer),
        clip_encoder_2: find(ModelComponent::ClipEncoder2),
        clip_tokenizer_2: find(ModelComponent::ClipTokenizer2),
        text_encoder_files: collect(ModelComponent::TextEncoder),
        text_tokenizer: find(ModelComponent::TextTokenizer),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn flux2_klein_defaults() {
        let manifest = find_manifest("flux2-klein:bf16").unwrap();
        assert_eq!(manifest.name, "flux2-klein:bf16");
        assert_eq!(manifest.family, "flux2");
        assert_eq!(manifest.defaults.steps, 20);
        assert!((manifest.defaults.guidance - 0.0).abs() < 0.01);
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
    fn known_manifests_count() {
        // 9 FLUX + 3 SD1.5 + 4 SD3 + 6 SDXL + 4 Z-Image + 1 Flux.2 + 4 Qwen-Image = 31
        assert_eq!(known_manifests().len(), 31);
    }

    #[test]
    fn manifest_has_required_components() {
        for manifest in known_manifests() {
            let components: Vec<_> = manifest.files.iter().map(|f| f.component).collect();
            // All models need VAE
            assert!(
                components.contains(&ModelComponent::Vae),
                "{} missing Vae",
                manifest.name
            );

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
                    // Flux.2 uses Transformer + Qwen3 TextEncoder + TextTokenizer
                    assert!(
                        components.contains(&ModelComponent::Transformer),
                        "{} (flux2) missing Transformer",
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
        assert_eq!(manifest.defaults.scheduler, Some("euler_ancestral"));
        assert_eq!(manifest.defaults.steps, 4);
        assert_eq!(manifest.defaults.guidance, 0.0);
    }

    #[test]
    fn sdxl_base_uses_ddim() {
        let manifest = find_manifest("sdxl-base").unwrap();
        assert_eq!(manifest.defaults.scheduler, Some("ddim"));
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
                assert_eq!(manifest.defaults.scheduler, Some("ddim"));
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
        assert_eq!(manifest.defaults.scheduler, Some("ddim"));
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
        assert_eq!(manifest.defaults.scheduler, Some("flow_match_euler"));
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

    #[test]
    fn zimage_defaults() {
        for manifest in known_manifests() {
            if manifest.family == "z-image" {
                assert_eq!(manifest.defaults.steps, 9);
                assert_eq!(manifest.defaults.guidance, 0.0);
                assert_eq!(manifest.defaults.scheduler, Some("flow_match_euler"));
            }
        }
    }
}
