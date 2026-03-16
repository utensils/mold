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
            sha256: None,
        },
        ModelFile {
            hf_repo: "comfyanonymous/flux_text_encoders".to_string(),
            hf_filename: "t5xxl_fp16.safetensors".to_string(),
            component: ModelComponent::T5Encoder,
            size_bytes: 9_200_000_000, // ~9.2GB
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "comfyanonymous/flux_text_encoders".to_string(),
            hf_filename: "clip_l.safetensors".to_string(),
            component: ModelComponent::ClipEncoder,
            size_bytes: 246_000_000, // ~246MB
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "lmz/mt5-tokenizers".to_string(),
            hf_filename: "t5-v1_1-xxl.tokenizer.json".to_string(),
            component: ModelComponent::T5Tokenizer,
            size_bytes: 2_400_000, // ~2.4MB
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "openai/clip-vit-large-patch14".to_string(),
            hf_filename: "tokenizer.json".to_string(),
            component: ModelComponent::ClipTokenizer,
            size_bytes: 600_000, // ~600KB
            gated: false,
            sha256: None,
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
            },
        },
    ];
    manifests.extend(sdxl_manifests());
    manifests.extend(zimage_manifests());
    manifests
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
            sha256: None,
        },
        ModelFile {
            hf_repo: "stabilityai/stable-diffusion-xl-base-1.0".to_string(),
            hf_filename: "text_encoder/model.safetensors".to_string(),
            component: ModelComponent::ClipEncoder,
            size_bytes: 492_000_000, // ~492MB (CLIP-L)
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "stabilityai/stable-diffusion-xl-base-1.0".to_string(),
            hf_filename: "text_encoder_2/model.safetensors".to_string(),
            component: ModelComponent::ClipEncoder2,
            size_bytes: 1_390_000_000, // ~1.39GB (CLIP-G / OpenCLIP)
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "openai/clip-vit-large-patch14".to_string(),
            hf_filename: "tokenizer.json".to_string(),
            component: ModelComponent::ClipTokenizer,
            size_bytes: 600_000, // ~600KB
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k".to_string(),
            hf_filename: "tokenizer.json".to_string(),
            component: ModelComponent::ClipTokenizer2,
            size_bytes: 600_000, // ~600KB
            gated: false,
            sha256: None,
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
                    sha256: None,
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
                    sha256: None,
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
                    hf_filename: "unet/diffusion_pytorch_model.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 5_140_000_000,
                    gated: false,
                    sha256: None,
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
                    hf_filename: "unet/diffusion_pytorch_model.safetensors".to_string(),
                    component: ModelComponent::Transformer,
                    size_bytes: 5_140_000_000,
                    gated: false,
                    sha256: None,
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
                    sha256: None,
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
            sha256: None,
        },
        ModelFile {
            hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
            hf_filename: "text_encoder/model-00002-of-00003.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 3_990_000_000, // ~3.99GB
            gated: false,
            sha256: None,
        },
        ModelFile {
            hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
            hf_filename: "text_encoder/model-00003-of-00003.safetensors".to_string(),
            component: ModelComponent::TextEncoder,
            size_bytes: 99_600_000, // ~99.6MB
            gated: false,
            sha256: None,
        },
        // VAE
        ModelFile {
            hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
            hf_filename: "vae/diffusion_pytorch_model.safetensors".to_string(),
            component: ModelComponent::Vae,
            size_bytes: 168_000_000, // ~168MB
            gated: false,
            sha256: None,
        },
        // Qwen3 tokenizer
        ModelFile {
            hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
            hf_filename: "tokenizer/tokenizer.json".to_string(),
            component: ModelComponent::TextTokenizer,
            size_bytes: 11_400_000, // ~11.4MB
            gated: false,
            sha256: None,
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
                    sha256: None,
                });
                files.push(ModelFile {
                    hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
                    hf_filename: "transformer/diffusion_pytorch_model-00002-of-00003.safetensors"
                        .to_string(),
                    component: ModelComponent::TransformerShard,
                    size_bytes: 9_970_000_000, // ~9.97GB
                    gated: false,
                    sha256: None,
                });
                files.push(ModelFile {
                    hf_repo: "Tongyi-MAI/Z-Image-Turbo".to_string(),
                    hf_filename: "transformer/diffusion_pytorch_model-00003-of-00003.safetensors"
                        .to_string(),
                    component: ModelComponent::TransformerShard,
                    size_bytes: 4_670_000_000, // ~4.67GB
                    gated: false,
                    sha256: None,
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
                    sha256: None,
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
                    sha256: None,
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
                    sha256: None,
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
        assert!(find_manifest("nonexistent").is_none());
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
        // 9 FLUX + 6 SDXL + 4 Z-Image = 19
        assert_eq!(known_manifests().len(), 19);
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
