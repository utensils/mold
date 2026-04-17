use serde::{Deserialize, Serialize};

/// Serde helpers for `Option<Vec<u8>>` as base64 in JSON.
mod base64_opt {
    use base64::Engine as _;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(data: &Option<Vec<u8>>, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match data {
            Some(bytes) => {
                let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
                s.serialize_some(&encoded)
            }
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(d: D) -> Result<Option<Vec<u8>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt: Option<String> = Option::deserialize(d)?;
        match opt {
            Some(encoded) => {
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(&encoded)
                    .map_err(serde::de::Error::custom)?;
                Ok(Some(bytes))
            }
            None => Ok(None),
        }
    }
}

/// Serde helpers for `Option<Vec<Vec<u8>>>` as base64 strings in JSON.
mod base64_vec_opt {
    use base64::Engine as _;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(data: &Option<Vec<Vec<u8>>>, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match data {
            Some(items) => {
                let encoded: Vec<String> = items
                    .iter()
                    .map(|bytes| base64::engine::general_purpose::STANDARD.encode(bytes))
                    .collect();
                s.serialize_some(&encoded)
            }
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(d: D) -> Result<Option<Vec<Vec<u8>>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt: Option<Vec<String>> = Option::deserialize(d)?;
        match opt {
            Some(items) => {
                let mut decoded = Vec::with_capacity(items.len());
                for encoded in items {
                    decoded.push(
                        base64::engine::general_purpose::STANDARD
                            .decode(&encoded)
                            .map_err(serde::de::Error::custom)?,
                    );
                }
                Ok(Some(decoded))
            }
            None => Ok(None),
        }
    }
}

/// Serde helpers for `Vec<u8>` as base64 in JSON (required field).
mod base64_required {
    use base64::Engine as _;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(data: &Vec<u8>, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let encoded = base64::engine::general_purpose::STANDARD.encode(data);
        s.serialize_str(&encoded)
    }

    pub fn deserialize<'de, D>(d: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let encoded: String = String::deserialize(d)?;
        base64::engine::general_purpose::STANDARD
            .decode(&encoded)
            .map_err(serde::de::Error::custom)
    }
}

/// Scheduler algorithm for UNet-based diffusion models (SD1.5, SDXL).
///
/// Flow-matching models (FLUX, SD3, Z-Image, Flux.2, Qwen-Image) ignore this setting.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "kebab-case")]
pub enum Scheduler {
    #[default]
    Ddim,
    EulerAncestral,
    UniPc,
}

impl std::fmt::Display for Scheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scheduler::Ddim => write!(f, "ddim"),
            Scheduler::EulerAncestral => write!(f, "euler-ancestral"),
            Scheduler::UniPc => write!(f, "uni-pc"),
        }
    }
}

impl std::str::FromStr for Scheduler {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ddim" => Ok(Scheduler::Ddim),
            "euler-ancestral" | "euler_ancestral" => Ok(Scheduler::EulerAncestral),
            "uni-pc" | "unipc" | "uni_pc" => Ok(Scheduler::UniPc),
            other => Err(format!(
                "unknown scheduler: '{other}'. Valid: ddim, euler-ancestral, uni-pc"
            )),
        }
    }
}

/// Request to expand a short prompt into detailed image generation prompts.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ExpandRequest {
    /// Short prompt to expand
    #[schema(example = "a cat")]
    pub prompt: String,
    /// Model family for prompt style (flux, sdxl, sd15, sd3, etc.)
    #[serde(default = "default_expand_model_family")]
    #[schema(example = "flux")]
    pub model_family: String,
    /// Number of prompt variations to generate
    #[serde(default = "default_expand_variations")]
    #[schema(example = 1)]
    pub variations: usize,
}

fn default_expand_model_family() -> String {
    "flux".to_string()
}

fn default_expand_variations() -> usize {
    1
}

/// Response from prompt expansion.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ExpandResponse {
    /// The original short prompt
    pub original: String,
    /// Expanded prompt(s)
    pub expanded: Vec<String>,
}

/// Request to upscale an image using a super-resolution model (e.g. Real-ESRGAN).
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct UpscaleRequest {
    /// Upscaler model name (e.g. "real-esrgan-x4plus:fp16").
    #[schema(example = "real-esrgan-x4plus:fp16")]
    pub model: String,
    /// Input image bytes (PNG or JPEG, base64-encoded in JSON).
    #[serde(with = "base64_required")]
    pub image: Vec<u8>,
    /// Output image format.
    #[serde(default)]
    pub output_format: OutputFormat,
    /// Tile size override for memory-efficient tiled inference.
    /// Default is 512. Set to 0 to disable tiling (process entire image at once).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tile_size: Option<u32>,
}

/// Response from image upscaling.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct UpscaleResponse {
    /// The upscaled image.
    pub image: ImageData,
    /// Time spent upscaling in milliseconds.
    #[schema(example = 450)]
    pub upscale_time_ms: u64,
    /// The upscaler model used.
    #[schema(example = "real-esrgan-x4plus:fp16")]
    pub model: String,
    /// The scale factor applied (e.g. 2 or 4).
    #[schema(example = 4)]
    pub scale_factor: u32,
    /// Original input image width.
    #[schema(example = 512)]
    pub original_width: u32,
    /// Original input image height.
    #[schema(example = 512)]
    pub original_height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GenerateRequest {
    #[schema(example = "a cat sitting on a windowsill at sunset")]
    pub prompt: String,
    /// Negative prompt — describes what to avoid generating.
    /// Effective for CFG-based models such as SD1.5, SDXL, SD3, and Wuerstchen.
    /// Ignored by distilled / non-CFG families such as FLUX schnell, Z-Image, etc.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = "blurry, low quality, watermark")]
    pub negative_prompt: Option<String>,
    #[schema(example = "flux-schnell:q8")]
    pub model: String,
    #[schema(example = 1024)]
    pub width: u32,
    #[schema(example = 1024)]
    pub height: u32,
    #[schema(example = 4)]
    pub steps: u32,
    /// Guidance scale. 0.0 for schnell (distilled), ~3.5 for dev/finetuned models.
    #[serde(default = "default_guidance")]
    #[schema(example = 3.5)]
    pub guidance: f64,
    #[schema(example = 42)]
    pub seed: Option<u64>,
    #[serde(default = "default_batch_size")]
    #[schema(example = 1)]
    pub batch_size: u32,
    #[serde(default)]
    pub output_format: OutputFormat,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embed_metadata: Option<bool>,
    /// Scheduler override for UNet-based models (SD1.5, SDXL).
    /// Ignored by flow-matching models (FLUX, SD3, Z-Image, Flux.2, Qwen-Image).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheduler: Option<Scheduler>,
    /// Source image for img2img generation (raw PNG/JPEG bytes, base64-encoded in JSON).
    #[serde(default, skip_serializing_if = "Option::is_none", with = "base64_opt")]
    pub source_image: Option<Vec<u8>>,
    /// Source images for Qwen-Image-Edit generation (raw PNG/JPEG bytes, base64-encoded in JSON).
    /// The first image is the primary edit target; additional images are reference images.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "base64_vec_opt"
    )]
    pub edit_images: Option<Vec<Vec<u8>>>,
    /// Denoising strength for img2img (0.0 = no change, 1.0 = full noise / txt2img).
    #[serde(default = "default_strength")]
    pub strength: f64,
    /// Mask image for inpainting (raw PNG/JPEG bytes, base64-encoded in JSON).
    /// White (255) = repaint, black (0) = preserve. Requires source_image.
    #[serde(default, skip_serializing_if = "Option::is_none", with = "base64_opt")]
    pub mask_image: Option<Vec<u8>>,
    /// Control image for ControlNet conditioning (raw PNG/JPEG bytes, base64-encoded in JSON).
    #[serde(default, skip_serializing_if = "Option::is_none", with = "base64_opt")]
    pub control_image: Option<Vec<u8>>,
    /// ControlNet model name (e.g. "controlnet-canny-sd15").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_model: Option<String>,
    /// ControlNet conditioning scale (0.0 = no effect, 1.0 = full conditioning).
    #[serde(default = "default_control_scale")]
    pub control_scale: f64,
    /// Request server-side prompt expansion before generation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expand: Option<bool>,
    /// Original user prompt before expansion (set by client when expanding locally).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_prompt: Option<String>,
    /// LoRA adapter to apply during generation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora: Option<LoraWeight>,
    /// Number of video frames to generate.
    /// Current LTX-Video / LTX-2 pipelines require 8n+1 (9, 17, 25, 33, …).
    /// Only used by video model families (e.g. ltx-video). Ignored by image models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames: Option<u32>,
    /// Video frames per second for output encoding. Default: 24.
    /// Only used by video model families. Ignored by image models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
    /// Upscaler model to apply after generation (e.g. "real-esrgan-x4plus:fp16").
    /// When set, each generated image is upscaled before being returned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upscale_model: Option<String>,
    /// Request a GIF preview alongside the primary video output.
    /// Used by TUI gallery and CLI `--preview` to get an animated preview without
    /// re-encoding when the primary format is not GIF.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub gif_preview: bool,
    /// Enable synchronized audio generation for audio-video model families such as LTX-2.
    /// Defaults to the model family's preferred behavior when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_audio: Option<bool>,
    /// Optional conditioning audio file for audio-to-video generation.
    #[serde(default, skip_serializing_if = "Option::is_none", with = "base64_opt")]
    pub audio_file: Option<Vec<u8>>,
    /// Optional source video for video-to-video / retake generation.
    #[serde(default, skip_serializing_if = "Option::is_none", with = "base64_opt")]
    pub source_video: Option<Vec<u8>>,
    /// Optional keyframe conditioning images for LTX-2 keyframe interpolation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keyframes: Option<Vec<KeyframeCondition>>,
    /// Explicit LTX-2 pipeline mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pipeline: Option<Ltx2PipelineMode>,
    /// Repeatable LoRA stack for model families that support multiple adapters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loras: Option<Vec<LoraWeight>>,
    /// Optional time range for retake / partial regeneration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retake_range: Option<TimeRange>,
    /// Optional spatial latent upscaling mode for LTX-2 pipelines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spatial_upscale: Option<Ltx2SpatialUpscale>,
    /// Optional temporal latent upscaling mode for LTX-2 pipelines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal_upscale: Option<Ltx2TemporalUpscale>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct KeyframeCondition {
    #[schema(example = 0)]
    pub frame: u32,
    #[serde(with = "base64_required")]
    pub image: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct TimeRange {
    #[schema(example = 0.0)]
    pub start_seconds: f32,
    #[schema(example = 2.5)]
    pub end_seconds: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "kebab-case")]
pub enum Ltx2PipelineMode {
    OneStage,
    TwoStage,
    TwoStageHq,
    Distilled,
    IcLora,
    Keyframe,
    A2Vid,
    Retake,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "kebab-case")]
pub enum Ltx2SpatialUpscale {
    X1_5,
    X2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "kebab-case")]
pub enum Ltx2TemporalUpscale {
    X2,
}

/// A LoRA adapter specification: path to safetensors file and effect scale.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct LoraWeight {
    /// Path to the LoRA safetensors file.
    #[schema(example = "/path/to/lora.safetensors")]
    pub path: String,
    /// Scaling factor for LoRA effect (0.0 = no effect, 1.0 = full strength, up to 2.0).
    #[serde(default = "default_lora_scale")]
    #[schema(example = 1.0)]
    pub scale: f64,
}

fn default_lora_scale() -> f64 {
    1.0
}

fn default_guidance() -> f64 {
    3.5
}

fn default_batch_size() -> u32 {
    1
}

fn default_strength() -> f64 {
    0.75
}

fn default_control_scale() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GenerateResponse {
    pub images: Vec<ImageData>,
    /// Video output data. Present only for video model families (e.g. ltx-video).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video: Option<VideoData>,
    #[schema(example = 1234)]
    pub generation_time_ms: u64,
    #[schema(example = "flux-schnell:q8")]
    pub model: String,
    #[schema(example = 42)]
    pub seed_used: u64,
}

/// Video output from a video model family.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct VideoData {
    /// Encoded video bytes in the requested format (APNG, GIF, WebP, MP4).
    pub data: Vec<u8>,
    /// Output format.
    pub format: OutputFormat,
    #[schema(example = 768)]
    pub width: u32,
    #[schema(example = 512)]
    pub height: u32,
    /// Number of frames in the video.
    #[schema(example = 25)]
    pub frames: u32,
    /// Frames per second.
    #[schema(example = 24)]
    pub fps: u32,
    /// First frame as PNG thumbnail for gallery grid.
    pub thumbnail: Vec<u8>,
    /// Animated GIF preview for gallery detail view / TUI playback.
    /// Always generated regardless of primary output format.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gif_preview: Vec<u8>,
    /// Whether this video includes a synchronized audio track.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub has_audio: bool,
    /// Total encoded duration in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    /// Audio sample rate in Hz when audio is present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_sample_rate: Option<u32>,
    /// Number of audio channels when audio is present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_channels: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ImageData {
    pub data: Vec<u8>,
    pub format: OutputFormat,
    #[schema(example = 1024)]
    pub width: u32,
    #[schema(example = 1024)]
    pub height: u32,
    #[schema(example = 0)]
    pub index: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OutputMetadata {
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_prompt: Option<String>,
    pub model: String,
    pub seed: u64,
    pub steps: u32,
    pub guidance: f64,
    pub width: u32,
    pub height: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strength: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheduler: Option<Scheduler>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_scale: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frames: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
    pub version: String,
}

impl OutputMetadata {
    pub fn from_generate_request(
        req: &GenerateRequest,
        seed: u64,
        scheduler: Option<Scheduler>,
        version: impl Into<String>,
    ) -> Self {
        let (lora, lora_scale) = match &req.lora {
            Some(lw) => {
                let name = std::path::Path::new(&lw.path)
                    .file_name()
                    .map(|f| f.to_string_lossy().to_string())
                    .unwrap_or_else(|| lw.path.clone());
                (Some(name), Some(lw.scale))
            }
            None => (None, None),
        };
        Self {
            prompt: req.prompt.clone(),
            negative_prompt: req.negative_prompt.clone(),
            original_prompt: req.original_prompt.clone(),
            model: req.model.clone(),
            seed,
            steps: req.steps,
            guidance: req.guidance,
            width: req.width,
            height: req.height,
            strength: req.source_image.as_ref().map(|_| req.strength),
            scheduler,
            lora,
            lora_scale,
            frames: req.frames,
            fps: req.fps,
            version: version.into(),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    #[default]
    Png,
    Jpeg,
    Gif,
    Apng,
    Webp,
    Mp4,
}

impl OutputFormat {
    /// Returns the file extension for this format.
    pub fn extension(&self) -> &'static str {
        match self {
            OutputFormat::Png => "png",
            OutputFormat::Jpeg => "jpeg",
            OutputFormat::Gif => "gif",
            OutputFormat::Apng => "png", // APNG files are valid PNGs — .png opens natively everywhere
            OutputFormat::Webp => "webp",
            OutputFormat::Mp4 => "mp4",
        }
    }

    /// Returns the MIME content type for this format.
    pub fn content_type(&self) -> &'static str {
        match self {
            OutputFormat::Png => "image/png",
            OutputFormat::Jpeg => "image/jpeg",
            OutputFormat::Gif => "image/gif",
            OutputFormat::Apng => "image/apng",
            OutputFormat::Webp => "image/webp",
            OutputFormat::Mp4 => "video/mp4",
        }
    }

    /// Whether this format is a video/animation format.
    pub fn is_video(&self) -> bool {
        matches!(
            self,
            OutputFormat::Gif | OutputFormat::Apng | OutputFormat::Webp | OutputFormat::Mp4
        )
    }
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.extension())
    }
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "png" => Ok(OutputFormat::Png),
            "jpeg" | "jpg" => Ok(OutputFormat::Jpeg),
            "gif" => Ok(OutputFormat::Gif),
            "apng" => Ok(OutputFormat::Apng),
            "webp" => Ok(OutputFormat::Webp),
            "mp4" => Ok(OutputFormat::Mp4),
            other => Err(format!("unknown format: {other}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ModelInfo {
    #[schema(example = "flux-schnell:q8")]
    pub name: String,
    #[schema(example = "flux")]
    pub family: String,
    #[schema(example = 4.5)]
    pub size_gb: f32,
    pub is_loaded: bool,
    pub last_used: Option<u64>,
    #[schema(example = "black-forest-labs/FLUX.1-schnell")]
    pub hf_repo: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ModelDefaults {
    #[schema(example = 4)]
    pub default_steps: u32,
    #[schema(example = 3.5)]
    pub default_guidance: f64,
    #[schema(example = 1024)]
    pub default_width: u32,
    #[schema(example = 1024)]
    pub default_height: u32,
    #[schema(example = "FLUX Schnell Q8 — fast 4-step generation")]
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ModelInfoExtended {
    #[serde(flatten)]
    pub info: ModelInfo,
    #[serde(flatten)]
    pub defaults: ModelDefaults,
    #[serde(default)]
    pub downloaded: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disk_usage_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remaining_download_bytes: Option<u64>,
}

impl ModelInfoExtended {
    /// True if this is an upscaler model (Real-ESRGAN, etc.) not a diffusion generator.
    pub fn is_upscaler(&self) -> bool {
        crate::manifest::UPSCALER_FAMILIES.contains(&self.family.as_str())
    }

    /// True if this is a utility model (e.g., prompt expansion LLM).
    pub fn is_utility(&self) -> bool {
        crate::manifest::UTILITY_FAMILIES.contains(&self.family.as_str())
    }

    /// True if this is an auxiliary model (e.g., ControlNet) not a standalone generator.
    pub fn is_auxiliary(&self) -> bool {
        crate::manifest::AUXILIARY_FAMILIES.contains(&self.family.as_str())
    }

    /// True if this is a standalone generation model (not an upscaler, utility, or auxiliary).
    pub fn is_generation_model(&self) -> bool {
        !self.is_upscaler() && !self.is_utility() && !self.is_auxiliary()
    }
}

impl std::ops::Deref for ModelInfoExtended {
    type Target = ModelInfo;

    fn deref(&self) -> &Self::Target {
        &self.info
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ActiveGenerationStatus {
    #[schema(example = "flux-schnell:q8")]
    pub model: String,
    #[schema(example = "3df0d8c4c7c8f7b7c78dc37f2b5f7dd5f9f2acb95c8f3f873f98f2f0fcb1a9d5")]
    pub prompt_sha256: String,
    #[schema(example = 1711305600000_u64)]
    pub started_at_unix_ms: u64,
    #[schema(example = 950)]
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ServerStatus {
    #[schema(example = "0.2.0")]
    pub version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = "da039e1")]
    pub git_sha: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = "2026-03-25")]
    pub build_date: Option<String>,
    pub models_loaded: Vec<String>,
    #[serde(default)]
    pub busy: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_generation: Option<ActiveGenerationStatus>,
    pub gpu_info: Option<GpuInfo>,
    #[schema(example = 3600)]
    pub uptime_secs: u64,
    /// Server hostname (e.g. "hal9000"). Added in v0.6.3.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = "hal9000")]
    pub hostname: Option<String>,
    /// Human-readable memory status (e.g. "VRAM: 16.2 GB free"). Added in v0.6.3.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_status: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, utoipa::ToSchema)]
pub struct GpuInfo {
    #[schema(example = "NVIDIA GeForce RTX 4090")]
    pub name: String,
    #[schema(example = 24564)]
    pub vram_total_mb: u64,
    #[schema(example = 8192)]
    pub vram_used_mb: u64,
}

// ── SSE streaming wire types ─────────────────────────────────────────────────

/// Progress event for SSE streaming. Mirrors `mold_inference::ProgressEvent`
/// but uses `u64` milliseconds instead of `Duration` for JSON serialization.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SseProgressEvent {
    StageStart {
        name: String,
    },
    StageDone {
        name: String,
        elapsed_ms: u64,
    },
    Info {
        message: String,
    },
    CacheHit {
        resource: String,
    },
    DenoiseStep {
        step: usize,
        total: usize,
        elapsed_ms: u64,
    },
    /// Download progress for a single file during model pull.
    DownloadProgress {
        filename: String,
        file_index: usize,
        total_files: usize,
        bytes_downloaded: u64,
        bytes_total: u64,
        batch_bytes_downloaded: u64,
        batch_bytes_total: u64,
        batch_elapsed_ms: u64,
    },
    /// A single file download completed.
    DownloadDone {
        filename: String,
        file_index: usize,
        total_files: usize,
        batch_bytes_downloaded: u64,
        batch_bytes_total: u64,
        batch_elapsed_ms: u64,
    },
    /// All downloads complete for a model pull.
    PullComplete {
        model: String,
    },
    /// Request is queued behind other generations.
    Queued {
        position: usize,
    },
    /// Progress loading model weights from disk.
    WeightLoad {
        bytes_loaded: u64,
        bytes_total: u64,
        component: String,
    },
}

/// Completion event sent when image/video generation finishes successfully.
///
/// For image responses, `image` contains base64-encoded image data and the
/// `video_*` fields are absent.  For video responses, `image` contains
/// base64-encoded video data (MP4/GIF/APNG/WebP) and the `video_*` fields
/// carry the metadata needed to reconstruct [`VideoData`] on the client.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct SseCompleteEvent {
    /// Base64-encoded payload — image bytes for images, video bytes for video.
    pub image: String,
    pub format: OutputFormat,
    #[schema(example = 1024)]
    pub width: u32,
    #[schema(example = 1024)]
    pub height: u32,
    #[schema(example = 42)]
    pub seed_used: u64,
    #[schema(example = 1234)]
    pub generation_time_ms: u64,
    /// The model that actually generated this image (server is source of truth).
    #[serde(default)]
    #[schema(example = "flux-schnell:q8")]
    pub model: String,

    // ── Video-only fields (absent for image responses) ──────────────────
    /// Number of frames.  Presence of this field signals a video response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_frames: Option<u32>,
    /// Frames per second.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_fps: Option<u32>,
    /// Base64-encoded first-frame PNG thumbnail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_thumbnail: Option<String>,
    /// Base64-encoded animated GIF preview.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_gif_preview: Option<String>,
    /// Whether this video includes a synchronized audio track.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub video_has_audio: bool,
    /// Total encoded duration in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_duration_ms: Option<u64>,
    /// Audio sample rate in Hz (when audio is present).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_audio_sample_rate: Option<u32>,
    /// Number of audio channels (when audio is present).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_audio_channels: Option<u32>,
}

/// SSE event emitted when an upscale request completes.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct SseUpscaleCompleteEvent {
    /// Base64-encoded upscaled image data.
    pub image: String,
    pub format: OutputFormat,
    pub model: String,
    pub scale_factor: u32,
    pub original_width: u32,
    pub original_height: u32,
    pub upscale_time_ms: u64,
}

/// Error event sent when generation fails during SSE streaming.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct SseErrorEvent {
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_format_from_str_png() {
        assert_eq!("png".parse::<OutputFormat>().unwrap(), OutputFormat::Png);
        assert_eq!("PNG".parse::<OutputFormat>().unwrap(), OutputFormat::Png);
    }

    #[test]
    fn output_format_from_str_jpeg() {
        assert_eq!("jpeg".parse::<OutputFormat>().unwrap(), OutputFormat::Jpeg);
        assert_eq!("jpg".parse::<OutputFormat>().unwrap(), OutputFormat::Jpeg);
        assert_eq!("JPEG".parse::<OutputFormat>().unwrap(), OutputFormat::Jpeg);
    }

    #[test]
    fn output_format_from_str_invalid() {
        assert!("".parse::<OutputFormat>().is_err());
        assert!("bmp".parse::<OutputFormat>().is_err());
        assert!("tiff".parse::<OutputFormat>().is_err());
    }

    #[test]
    fn output_format_new_formats() {
        assert_eq!("apng".parse::<OutputFormat>().unwrap(), OutputFormat::Apng);
        assert_eq!("webp".parse::<OutputFormat>().unwrap(), OutputFormat::Webp);
        assert_eq!("mp4".parse::<OutputFormat>().unwrap(), OutputFormat::Mp4);
        assert!(OutputFormat::Apng.is_video());
        assert!(OutputFormat::Mp4.is_video());
        assert!(!OutputFormat::Png.is_video());
    }

    #[test]
    fn output_format_display() {
        assert_eq!(OutputFormat::Png.to_string(), "png");
        assert_eq!(OutputFormat::Jpeg.to_string(), "jpeg");
    }

    #[test]
    fn output_format_serde_roundtrip() {
        let fmt = OutputFormat::Png;
        let json = serde_json::to_string(&fmt).unwrap();
        assert_eq!(json, r#""png""#);
        let back: OutputFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(back, fmt);
    }

    #[test]
    fn generate_request_serde_roundtrip() {
        let req = GenerateRequest {
            prompt: "a cat on Mars".to_string(),
            negative_prompt: None,
            model: "flux-schnell".to_string(),
            width: 768,
            height: 768,
            steps: 4,
            guidance: 0.0,
            seed: Some(42),
            batch_size: 1,
            output_format: OutputFormat::Png,
            embed_metadata: Some(true),
            scheduler: None,
            source_image: None,
            edit_images: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            source_video: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: GenerateRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.prompt, req.prompt);
        assert_eq!(back.width, req.width);
        assert_eq!(back.seed, req.seed);
        assert_eq!(back.embed_metadata, req.embed_metadata);
        assert_eq!(back.scheduler, None);
    }

    #[test]
    fn generate_request_optional_seed() {
        let json = r#"{"prompt":"test","model":"flux-schnell","width":768,"height":768,"steps":4,"batch_size":1,"output_format":"png"}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert!(req.seed.is_none());
        assert_eq!(req.embed_metadata, None);
        // guidance should default to 3.5 when omitted
        assert!((req.guidance - 3.5).abs() < 0.001);
    }

    #[test]
    fn generate_request_explicit_guidance() {
        let json = r#"{"prompt":"test","model":"flux-schnell","width":768,"height":768,"steps":4,"guidance":0.0,"batch_size":1,"output_format":"png"}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.guidance, 0.0);
    }

    #[test]
    fn generate_request_output_format_defaults_to_png() {
        // output_format omitted — should default to PNG, not fail deserialization
        let json = r#"{"prompt":"test","model":"flux-schnell","width":768,"height":768,"steps":4,"batch_size":1}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.output_format, OutputFormat::Png);
    }

    #[test]
    fn generate_request_output_format_explicit_jpeg() {
        let json = r#"{"prompt":"test","model":"flux-schnell","width":768,"height":768,"steps":4,"batch_size":1,"output_format":"jpeg"}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.output_format, OutputFormat::Jpeg);
    }

    #[test]
    fn generate_request_minimal_json() {
        // Minimal request: only required fields, all optional fields use defaults
        let json = r#"{"prompt":"a cat","model":"test","width":512,"height":512,"steps":4}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "a cat");
        assert_eq!(req.output_format, OutputFormat::Png);
        assert_eq!(req.batch_size, 1);
        assert!((req.guidance - 3.5).abs() < 0.001);
        assert!(req.seed.is_none());
    }

    #[test]
    fn output_format_default_is_png() {
        assert_eq!(OutputFormat::default(), OutputFormat::Png);
    }

    #[test]
    fn scheduler_serde_roundtrip() {
        let sched = Scheduler::EulerAncestral;
        let json = serde_json::to_string(&sched).unwrap();
        assert_eq!(json, r#""euler-ancestral""#);
        let back: Scheduler = serde_json::from_str(&json).unwrap();
        assert_eq!(back, sched);
    }

    #[test]
    fn scheduler_from_str_aliases() {
        assert_eq!("ddim".parse::<Scheduler>().unwrap(), Scheduler::Ddim);
        assert_eq!(
            "euler-ancestral".parse::<Scheduler>().unwrap(),
            Scheduler::EulerAncestral
        );
        assert_eq!(
            "euler_ancestral".parse::<Scheduler>().unwrap(),
            Scheduler::EulerAncestral
        );
        assert_eq!("uni-pc".parse::<Scheduler>().unwrap(), Scheduler::UniPc);
        assert_eq!("unipc".parse::<Scheduler>().unwrap(), Scheduler::UniPc);
        assert_eq!("uni_pc".parse::<Scheduler>().unwrap(), Scheduler::UniPc);
    }

    #[test]
    fn scheduler_from_str_invalid() {
        assert!("unknown".parse::<Scheduler>().is_err());
    }

    #[test]
    fn scheduler_display() {
        assert_eq!(Scheduler::Ddim.to_string(), "ddim");
        assert_eq!(Scheduler::EulerAncestral.to_string(), "euler-ancestral");
        assert_eq!(Scheduler::UniPc.to_string(), "uni-pc");
    }

    #[test]
    fn scheduler_default_is_ddim() {
        assert_eq!(Scheduler::default(), Scheduler::Ddim);
    }

    #[test]
    fn generate_request_backward_compat_no_scheduler() {
        // Existing JSON without scheduler field should deserialize fine
        let json =
            r#"{"prompt":"test","model":"test","width":512,"height":512,"steps":4,"batch_size":1}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.scheduler, None);
    }

    #[test]
    fn generate_request_backward_compat_no_negative_prompt() {
        let json =
            r#"{"prompt":"test","model":"test","width":512,"height":512,"steps":4,"batch_size":1}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert!(req.negative_prompt.is_none());
    }

    #[test]
    fn generate_request_negative_prompt_roundtrip() {
        let req = GenerateRequest {
            prompt: "a cat".to_string(),
            negative_prompt: Some("blurry, low quality".to_string()),
            model: "sd15:fp16".to_string(),
            width: 512,
            height: 512,
            steps: 25,
            guidance: 7.5,
            seed: None,
            batch_size: 1,
            output_format: OutputFormat::Png,
            embed_metadata: None,
            scheduler: None,
            source_image: None,
            edit_images: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            source_video: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("negative_prompt"));
        assert!(json.contains("blurry, low quality"));
        let back: GenerateRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.negative_prompt.as_deref(), Some("blurry, low quality"));
    }

    #[test]
    fn generate_request_negative_prompt_omitted_when_none() {
        let req = GenerateRequest {
            prompt: "test".to_string(),
            negative_prompt: None,
            model: "test".to_string(),
            width: 512,
            height: 512,
            steps: 4,
            guidance: 3.5,
            seed: None,
            batch_size: 1,
            output_format: OutputFormat::Png,
            embed_metadata: None,
            scheduler: None,
            source_image: None,
            edit_images: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            source_video: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("negative_prompt"));
    }

    #[test]
    fn output_metadata_omits_strength_without_source_image() {
        let req = GenerateRequest {
            prompt: "test".to_string(),
            negative_prompt: None,
            model: "flux-schnell:q8".to_string(),
            width: 1024,
            height: 1024,
            steps: 4,
            guidance: 0.0,
            seed: Some(7),
            batch_size: 1,
            output_format: OutputFormat::Png,
            embed_metadata: Some(true),
            scheduler: None,
            source_image: None,
            edit_images: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            source_video: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
        };

        let metadata = OutputMetadata::from_generate_request(&req, 7, None, "0.1.0");
        assert_eq!(metadata.strength, None);
        assert_eq!(metadata.version, "0.1.0");
    }

    #[test]
    fn output_metadata_includes_negative_prompt_when_provided() {
        let req = GenerateRequest {
            prompt: "a cat".to_string(),
            negative_prompt: Some("blurry, ugly".to_string()),
            model: "sd15:fp16".to_string(),
            width: 512,
            height: 512,
            steps: 25,
            guidance: 7.5,
            seed: Some(1),
            batch_size: 1,
            output_format: OutputFormat::Png,
            embed_metadata: Some(true),
            scheduler: None,
            source_image: None,
            edit_images: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            source_video: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
        };
        let metadata = OutputMetadata::from_generate_request(&req, 1, None, "0.1.0");
        assert_eq!(metadata.negative_prompt.as_deref(), Some("blurry, ugly"));
    }

    #[test]
    fn output_metadata_includes_strength_and_scheduler_when_applicable() {
        let req = GenerateRequest {
            prompt: "test".to_string(),
            negative_prompt: None,
            model: "sd15:fp16".to_string(),
            width: 512,
            height: 512,
            steps: 25,
            guidance: 7.0,
            seed: Some(9),
            batch_size: 1,
            output_format: OutputFormat::Png,
            embed_metadata: Some(true),
            scheduler: Some(Scheduler::UniPc),
            source_image: Some(vec![1, 2, 3]),
            edit_images: None,
            strength: 0.5,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            source_video: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
        };

        let metadata =
            OutputMetadata::from_generate_request(&req, 9, Some(Scheduler::UniPc), "0.1.0");
        assert_eq!(metadata.strength, Some(0.5));
        assert_eq!(metadata.scheduler, Some(Scheduler::UniPc));
    }

    // ── SSE type tests ──────────────────────────────────────────────────────

    #[test]
    fn sse_progress_stage_start_roundtrip() {
        let event = SseProgressEvent::StageStart {
            name: "Loading T5 encoder".to_string(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"stage_start""#));
        let back: SseProgressEvent = serde_json::from_str(&json).unwrap();
        assert!(
            matches!(back, SseProgressEvent::StageStart { name } if name == "Loading T5 encoder")
        );
    }

    #[test]
    fn sse_progress_cache_hit_roundtrip() {
        let event = SseProgressEvent::CacheHit {
            resource: "prompt conditioning".to_string(),
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: SseProgressEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            back,
            SseProgressEvent::CacheHit { resource } if resource == "prompt conditioning"
        ));
    }

    #[test]
    fn sse_progress_denoise_step_roundtrip() {
        let event = SseProgressEvent::DenoiseStep {
            step: 5,
            total: 28,
            elapsed_ms: 1234,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"denoise_step""#));
        let back: SseProgressEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            back,
            SseProgressEvent::DenoiseStep {
                step: 5,
                total: 28,
                elapsed_ms: 1234
            }
        ));
    }

    #[test]
    fn sse_complete_event_roundtrip() {
        let event = SseCompleteEvent {
            image: "iVBOR...".to_string(),
            format: OutputFormat::Png,
            width: 1024,
            height: 1024,
            seed_used: 42,
            generation_time_ms: 5000,
            model: "flux-schnell:q8".to_string(),
            video_frames: None,
            video_fps: None,
            video_thumbnail: None,
            video_gif_preview: None,
            video_has_audio: false,
            video_duration_ms: None,
            video_audio_sample_rate: None,
            video_audio_channels: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        // Video fields should be absent from the serialized JSON
        assert!(!json.contains("video_frames"));
        assert!(!json.contains("video_fps"));
        let back: SseCompleteEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.width, 1024);
        assert_eq!(back.seed_used, 42);
        assert_eq!(back.model, "flux-schnell:q8");
        assert!(back.video_frames.is_none());
    }

    #[test]
    fn sse_complete_event_backward_compat_no_model() {
        // Older servers may not include the model field; #[serde(default)]
        // ensures deserialization still succeeds with an empty string.
        let json = r#"{"image":"data","format":"png","width":512,"height":512,"seed_used":1,"generation_time_ms":100}"#;
        let event: SseCompleteEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.model, "");
        assert_eq!(event.width, 512);
    }

    #[test]
    fn sse_error_event_roundtrip() {
        let event = SseErrorEvent {
            message: "something failed".to_string(),
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: SseErrorEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.message, "something failed");
    }

    #[test]
    fn sse_progress_queued_roundtrip() {
        let event = SseProgressEvent::Queued { position: 3 };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"queued""#));
        assert!(json.contains(r#""position":3"#));
        let back: SseProgressEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, SseProgressEvent::Queued { position: 3 }));
    }

    #[test]
    fn sse_progress_weight_load_roundtrip() {
        let event = SseProgressEvent::WeightLoad {
            bytes_loaded: 5_000_000,
            bytes_total: 10_000_000,
            component: "FLUX transformer".to_string(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"weight_load""#));
        assert!(json.contains(r#""bytes_loaded":5000000"#));
        assert!(json.contains(r#""component":"FLUX transformer""#));
        let back: SseProgressEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            back,
            SseProgressEvent::WeightLoad {
                bytes_loaded: 5_000_000,
                bytes_total: 10_000_000,
                ..
            }
        ));
    }

    #[test]
    fn sse_progress_download_roundtrip() {
        let event = SseProgressEvent::DownloadProgress {
            filename: "text_encoder_2/model.safetensors".to_string(),
            file_index: 1,
            total_files: 5,
            bytes_downloaded: 16_384,
            bytes_total: 2_600_000_000,
            batch_bytes_downloaded: 3_100_000_000,
            batch_bytes_total: 8_800_000_000,
            batch_elapsed_ms: 42_000,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"download_progress""#));
        assert!(json.contains(r#""batch_bytes_total":8800000000"#));
        let back: SseProgressEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            back,
            SseProgressEvent::DownloadProgress {
                file_index: 1,
                total_files: 5,
                bytes_downloaded: 16_384,
                batch_bytes_total: 8_800_000_000,
                batch_elapsed_ms: 42_000,
                ..
            }
        ));
    }

    // ── img2img field tests ────────────────────────────────────────────────

    #[test]
    fn generate_request_source_image_base64_roundtrip() {
        // Minimal PNG-like bytes for testing
        let image_bytes = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        let req = GenerateRequest {
            prompt: "test".to_string(),
            negative_prompt: None,
            model: "test".to_string(),
            width: 512,
            height: 512,
            steps: 4,
            guidance: 3.5,
            seed: None,
            batch_size: 1,
            output_format: OutputFormat::Png,
            embed_metadata: None,
            scheduler: None,
            source_image: Some(image_bytes.clone()),
            edit_images: None,
            strength: 0.5,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            source_video: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        // Verify base64 encoding is in the JSON
        assert!(json.contains("source_image"));
        let back: GenerateRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.source_image, Some(image_bytes));
        assert_eq!(back.strength, 0.5);
    }

    #[test]
    fn generate_request_edit_images_base64_roundtrip() {
        let image_a = vec![0x89, 0x50, 0x4E, 0x47];
        let image_b = vec![0xFF, 0xD8, 0xFF, 0xE0];
        let req = GenerateRequest {
            prompt: "test".to_string(),
            negative_prompt: None,
            model: "qwen-image-edit-2511:q4".to_string(),
            width: 1024,
            height: 1024,
            steps: 4,
            guidance: 4.0,
            seed: None,
            batch_size: 1,
            output_format: OutputFormat::Png,
            embed_metadata: None,
            scheduler: None,
            source_image: None,
            edit_images: Some(vec![image_a.clone(), image_b.clone()]),
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            source_video: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("edit_images"));
        let back: GenerateRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.edit_images, Some(vec![image_a, image_b]));
    }

    #[test]
    fn generate_request_backward_compat_no_source_image() {
        // Existing JSON without source_image/strength should deserialize fine
        let json =
            r#"{"prompt":"test","model":"test","width":512,"height":512,"steps":4,"batch_size":1}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert!(req.source_image.is_none());
        assert!((req.strength - 0.75).abs() < 0.001);
    }

    #[test]
    fn generate_request_strength_defaults_to_075() {
        let json = r#"{"prompt":"test","model":"test","width":512,"height":512,"steps":4}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert!((req.strength - 0.75).abs() < 0.001);
    }

    #[test]
    fn generate_request_source_image_omitted_in_json_when_none() {
        let req = GenerateRequest {
            prompt: "test".to_string(),
            negative_prompt: None,
            model: "test".to_string(),
            width: 512,
            height: 512,
            steps: 4,
            guidance: 3.5,
            seed: None,
            batch_size: 1,
            output_format: OutputFormat::Png,
            embed_metadata: None,
            scheduler: None,
            source_image: None,
            edit_images: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            source_video: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("source_image"));
        assert!(!json.contains("control_image"));
    }

    // ── ControlNet field tests ─────────────────────────────────────────────

    #[test]
    fn generate_request_control_image_base64_roundtrip() {
        let control_bytes = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        let req = GenerateRequest {
            prompt: "test".to_string(),
            negative_prompt: None,
            model: "test".to_string(),
            width: 512,
            height: 512,
            steps: 4,
            guidance: 3.5,
            seed: None,
            batch_size: 1,
            output_format: OutputFormat::Png,
            embed_metadata: None,
            scheduler: None,
            source_image: None,
            edit_images: None,
            strength: 0.75,
            mask_image: None,
            control_image: Some(control_bytes.clone()),
            control_model: Some("controlnet-canny-sd15".to_string()),
            control_scale: 0.8,
            expand: None,
            original_prompt: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            source_video: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("control_image"));
        assert!(json.contains("controlnet-canny-sd15"));
        let back: GenerateRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.control_image, Some(control_bytes));
        assert_eq!(back.control_model.as_deref(), Some("controlnet-canny-sd15"));
        assert_eq!(back.control_scale, 0.8);
    }

    #[test]
    fn generate_request_backward_compat_no_control_fields() {
        let json =
            r#"{"prompt":"test","model":"test","width":512,"height":512,"steps":4,"batch_size":1}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert!(req.control_image.is_none());
        assert!(req.control_model.is_none());
        assert!((req.control_scale - 1.0).abs() < 0.001);
    }

    #[test]
    fn generate_request_control_scale_defaults_to_1() {
        let json = r#"{"prompt":"test","model":"test","width":512,"height":512,"steps":4}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert!((req.control_scale - 1.0).abs() < 0.001);
    }
    // ── Inpainting field tests ────────────────────────────────────────────

    #[test]
    fn generate_request_mask_image_base64_roundtrip() {
        let mask_bytes = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        let source_bytes = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        let req = GenerateRequest {
            prompt: "test".to_string(),
            negative_prompt: None,
            model: "test".to_string(),
            width: 512,
            height: 512,
            steps: 4,
            guidance: 3.5,
            seed: None,
            batch_size: 1,
            output_format: OutputFormat::Png,
            embed_metadata: None,
            scheduler: None,
            source_image: Some(source_bytes),
            edit_images: None,
            strength: 0.75,
            mask_image: Some(mask_bytes.clone()),
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            source_video: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("mask_image"));
        let back: GenerateRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.mask_image, Some(mask_bytes));
    }

    #[test]
    fn generate_request_backward_compat_no_mask_image() {
        let json =
            r#"{"prompt":"test","model":"test","width":512,"height":512,"steps":4,"batch_size":1}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert!(req.mask_image.is_none());
    }

    #[test]
    fn default_output_filename_single() {
        let name = super::default_output_filename("flux-dev:q8", 1700000000, "png", 1, 0);
        assert_eq!(name, "mold-flux-dev-q8-1700000000.png");
    }

    #[test]
    fn default_output_filename_batch() {
        let name = super::default_output_filename("flux-dev:q8", 1700000000, "png", 4, 2);
        assert_eq!(name, "mold-flux-dev-q8-1700000000-2.png");
    }

    #[test]
    fn default_output_filename_jpeg() {
        let name = super::default_output_filename("sdxl-turbo", 12345, "jpeg", 1, 0);
        assert_eq!(name, "mold-sdxl-turbo-12345.jpeg");
    }

    #[test]
    fn default_output_filename_millis_timestamp() {
        // Server uses milliseconds for uniqueness
        let name = super::default_output_filename("flux-dev-q8", 1700000000123, "png", 1, 0);
        assert_eq!(name, "mold-flux-dev-q8-1700000000123.png");
    }

    #[test]
    fn server_status_deserialize_without_busy_field() {
        // Older servers don't send `busy` — #[serde(default)] makes it false
        let json = r#"{
            "version": "0.1.0",
            "models_loaded": ["flux-schnell:q8"],
            "gpu_info": null,
            "uptime_secs": 3600
        }"#;
        let status: super::ServerStatus = serde_json::from_str(json).unwrap();
        assert!(!status.busy, "missing busy field should default to false");
        assert_eq!(status.models_loaded, vec!["flux-schnell:q8"]);
    }

    #[test]
    fn server_status_deserialize_with_busy_true() {
        let json = r#"{
            "version": "0.2.0",
            "models_loaded": [],
            "busy": true,
            "gpu_info": null,
            "uptime_secs": 100
        }"#;
        let status: super::ServerStatus = serde_json::from_str(json).unwrap();
        assert!(status.busy);
    }

    #[test]
    fn server_status_deserialize_without_hostname_or_memory() {
        // Older servers (pre-0.6.3) don't send hostname or memory_status
        let json = r#"{
            "version": "0.5.0",
            "models_loaded": [],
            "gpu_info": null,
            "uptime_secs": 100
        }"#;
        let status: super::ServerStatus = serde_json::from_str(json).unwrap();
        assert!(status.hostname.is_none());
        assert!(status.memory_status.is_none());
    }

    #[test]
    fn server_status_deserialize_with_hostname_and_memory() {
        let json = r#"{
            "version": "0.6.3",
            "models_loaded": ["flux-dev:q4"],
            "gpu_info": {"name": "RTX 4090", "vram_total_mb": 24564, "vram_used_mb": 8192},
            "uptime_secs": 3600,
            "hostname": "hal9000",
            "memory_status": "VRAM: 16.0 GB free"
        }"#;
        let status: super::ServerStatus = serde_json::from_str(json).unwrap();
        assert_eq!(status.hostname.as_deref(), Some("hal9000"));
        assert_eq!(status.memory_status.as_deref(), Some("VRAM: 16.0 GB free"));
    }

    #[test]
    fn server_status_roundtrip_preserves_new_fields() {
        let status = super::ServerStatus {
            version: "0.6.3".to_string(),
            git_sha: None,
            build_date: None,
            models_loaded: vec![],
            busy: false,
            current_generation: None,
            gpu_info: None,
            uptime_secs: 0,
            hostname: Some("bender".to_string()),
            memory_status: Some("Memory: 64.0 GB free, 96.0 GB available".to_string()),
        };
        let json = serde_json::to_string(&status).unwrap();
        let parsed: super::ServerStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.hostname.as_deref(), Some("bender"));
        assert_eq!(
            parsed.memory_status.as_deref(),
            Some("Memory: 64.0 GB free, 96.0 GB available")
        );
    }

    // ── UpscaleRequest / UpscaleResponse tests ────────────────────────────

    #[test]
    fn upscale_request_serde_roundtrip() {
        let image_bytes = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        let req = super::UpscaleRequest {
            model: "real-esrgan-x4plus:fp16".to_string(),
            image: image_bytes.clone(),
            output_format: OutputFormat::Png,
            tile_size: Some(256),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("real-esrgan-x4plus:fp16"));
        assert!(json.contains("tile_size"));
        // image should be base64-encoded
        assert!(!json.contains("[137,"));

        let back: super::UpscaleRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "real-esrgan-x4plus:fp16");
        assert_eq!(back.image, image_bytes);
        assert_eq!(back.tile_size, Some(256));
        assert_eq!(back.output_format, OutputFormat::Png);
    }

    #[test]
    fn upscale_request_tile_size_omitted_when_none() {
        let req = super::UpscaleRequest {
            model: "test".to_string(),
            image: vec![0xFF, 0xD8],
            output_format: OutputFormat::Jpeg,
            tile_size: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("tile_size"));
    }

    #[test]
    fn upscale_response_serde_roundtrip() {
        let resp = super::UpscaleResponse {
            image: super::ImageData {
                data: vec![1, 2, 3],
                format: OutputFormat::Png,
                width: 2048,
                height: 2048,
                index: 0,
            },
            upscale_time_ms: 450,
            model: "real-esrgan-x4plus:fp16".to_string(),
            scale_factor: 4,
            original_width: 512,
            original_height: 512,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: super::UpscaleResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.scale_factor, 4);
        assert_eq!(back.original_width, 512);
        assert_eq!(back.image.width, 2048);
        assert_eq!(back.upscale_time_ms, 450);
    }

    #[test]
    fn generate_request_upscale_model_backward_compat() {
        // Existing JSON without upscale_model should deserialize fine
        let json =
            r#"{"prompt":"test","model":"test","width":512,"height":512,"steps":4,"batch_size":1}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert!(req.upscale_model.is_none());
    }

    // ── Video SSE transport tests ────────────────────────────────────────

    #[test]
    fn sse_complete_event_video_roundtrip() {
        let event = SseCompleteEvent {
            image: "dmlkZW9fYnl0ZXM=".to_string(), // "video_bytes" base64
            format: OutputFormat::Mp4,
            width: 832,
            height: 480,
            seed_used: 99,
            generation_time_ms: 12000,
            model: "ltx-2.3-22b-distilled:fp8".to_string(),
            video_frames: Some(33),
            video_fps: Some(12),
            video_thumbnail: Some("dGh1bWI=".to_string()), // "thumb" base64
            video_gif_preview: Some("Z2lm".to_string()),   // "gif" base64
            video_has_audio: true,
            video_duration_ms: Some(2750),
            video_audio_sample_rate: Some(44100),
            video_audio_channels: Some(2),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("video_frames"));
        assert!(json.contains("video_fps"));
        assert!(json.contains("video_thumbnail"));
        assert!(json.contains("video_gif_preview"));
        assert!(json.contains("video_has_audio"));
        assert!(json.contains("video_duration_ms"));
        assert!(json.contains("video_audio_sample_rate"));
        assert!(json.contains("video_audio_channels"));

        let back: SseCompleteEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.video_frames, Some(33));
        assert_eq!(back.video_fps, Some(12));
        assert!(back.video_has_audio);
        assert_eq!(back.video_duration_ms, Some(2750));
        assert_eq!(back.video_audio_sample_rate, Some(44100));
        assert_eq!(back.video_audio_channels, Some(2));
        assert_eq!(back.video_thumbnail.as_deref(), Some("dGh1bWI="));
        assert_eq!(back.video_gif_preview.as_deref(), Some("Z2lm"));
        assert_eq!(back.format, OutputFormat::Mp4);
    }

    #[test]
    fn sse_complete_event_video_no_audio_omits_audio_fields() {
        let event = SseCompleteEvent {
            image: "data".to_string(),
            format: OutputFormat::Gif,
            width: 512,
            height: 512,
            seed_used: 1,
            generation_time_ms: 100,
            model: "ltx-video:bf16".to_string(),
            video_frames: Some(17),
            video_fps: Some(24),
            video_thumbnail: Some("dGh1bWI=".to_string()),
            video_gif_preview: None,
            video_has_audio: false,
            video_duration_ms: None,
            video_audio_sample_rate: None,
            video_audio_channels: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        // Audio-related fields should be absent when not set
        assert!(!json.contains("video_has_audio"));
        assert!(!json.contains("video_audio_sample_rate"));
        assert!(!json.contains("video_audio_channels"));
        assert!(!json.contains("video_gif_preview"));
        // But video fields should be present
        assert!(json.contains("video_frames"));
        assert!(json.contains("video_fps"));

        let back: SseCompleteEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.video_frames, Some(17));
        assert!(!back.video_has_audio);
        assert!(back.video_gif_preview.is_none());
    }

    #[test]
    fn sse_complete_event_backward_compat_no_video_fields() {
        // Older servers send no video fields at all — everything defaults.
        let json = r#"{"image":"aW1n","format":"png","width":1024,"height":1024,"seed_used":42,"generation_time_ms":5000,"model":"flux-dev:q8"}"#;
        let event: SseCompleteEvent = serde_json::from_str(json).unwrap();
        assert!(event.video_frames.is_none());
        assert!(event.video_fps.is_none());
        assert!(event.video_thumbnail.is_none());
        assert!(event.video_gif_preview.is_none());
        assert!(!event.video_has_audio);
        assert!(event.video_duration_ms.is_none());
        assert!(event.video_audio_sample_rate.is_none());
        assert!(event.video_audio_channels.is_none());
        assert_eq!(event.model, "flux-dev:q8");
        assert_eq!(event.width, 1024);
    }

    #[test]
    fn sse_complete_event_image_no_video_fields_in_json() {
        // An image-only event should not include any video_* keys in JSON
        let event = SseCompleteEvent {
            image: "aW1n".to_string(),
            format: OutputFormat::Png,
            width: 1024,
            height: 1024,
            seed_used: 1,
            generation_time_ms: 100,
            model: "flux-dev:q8".to_string(),
            video_frames: None,
            video_fps: None,
            video_thumbnail: None,
            video_gif_preview: None,
            video_has_audio: false,
            video_duration_ms: None,
            video_audio_sample_rate: None,
            video_audio_channels: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(!json.contains("video_"));
    }
}

/// A gallery image entry returned by the server API.
///
/// Covers still images (png/jpg) and animated/video outputs (gif/apng/webp/mp4).
/// `metadata` is synthesized from the filename when a file has no embedded
/// `mold:parameters` chunk — callers should treat zero-valued fields
/// (seed/steps/width/height) as "unknown" for those entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GalleryImage {
    pub filename: String,
    pub metadata: OutputMetadata,
    pub timestamp: u64,
    /// File format inferred from extension. Omitted for backwards compat when
    /// the server doesn't populate it (older servers).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<OutputFormat>,
    /// On-disk size in bytes, for UI display.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    /// True when `metadata` was synthesized (no mold:parameters chunk found).
    #[serde(default, skip_serializing_if = "is_false")]
    pub metadata_synthetic: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// Server-reported capabilities the SPA uses to decide which UI affordances
/// to surface. Additive — clients that deserialize older responses simply
/// see `None` for fields they don't know about. Opt-in destructive
/// operations (like gallery delete) default to `false` so a client that
/// forgets to check gets the safe behavior.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GalleryCapabilities {
    /// Whether `DELETE /api/gallery/image/:filename` is allowed by the
    /// server configuration. Operators opt in via
    /// `MOLD_GALLERY_ALLOW_DELETE=1` (combined with the existing API-key
    /// middleware when the server is exposed beyond localhost).
    pub can_delete: bool,
}

/// Capabilities payload returned by `GET /api/capabilities`. Grouping keeps
/// the shape extensible — future areas (inpainting, upscaling modes, etc.)
/// can add their own sub-structs without churning existing fields.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServerCapabilities {
    pub gallery: GalleryCapabilities,
}

/// Build a default output filename, sanitizing colons from model names.
pub fn default_output_filename(
    model: &str,
    timestamp: u64,
    ext: &str,
    batch: u32,
    index: u32,
) -> String {
    let safe_model = model.replace(':', "-");
    if batch == 1 {
        format!("mold-{safe_model}-{timestamp}.{ext}")
    } else {
        format!("mold-{safe_model}-{timestamp}-{index}.{ext}")
    }
}
