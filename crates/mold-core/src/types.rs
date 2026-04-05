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
    /// Number of video frames to generate. Must be 8n+1 (9, 17, 25, 33, …).
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

/// Completion event sent when image generation finishes successfully.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct SseCompleteEvent {
    /// Base64-encoded image data.
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
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: SseCompleteEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.width, 1024);
        assert_eq!(back.seed_used, 42);
        assert_eq!(back.model, "flux-schnell:q8");
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
        };
        let json = serde_json::to_string(&req).unwrap();
        // Verify base64 encoding is in the JSON
        assert!(json.contains("source_image"));
        let back: GenerateRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.source_image, Some(image_bytes));
        assert_eq!(back.strength, 0.5);
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
}

/// A gallery image entry returned by the server API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GalleryImage {
    pub filename: String,
    pub metadata: OutputMetadata,
    pub timestamp: u64,
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
