use serde::{Deserialize, Serialize};

/// Serde helpers for `Option<Vec<u8>>` as base64 in JSON.
pub(crate) mod base64_opt {
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

/// Serde helpers for `Vec<u8>` as base64 in JSON (required, non-optional field).
///
/// Used by [`crate::chain::NamedRef`] and any other type that carries raw
/// bytes as a required wire field.
pub(crate) mod base64_bytes {
    use base64::{engine::general_purpose::STANDARD, Engine};
    use serde::{Deserialize, Deserializer, Serializer};

    #[allow(clippy::ptr_arg)]
    pub fn serialize<S: Serializer>(bytes: &Vec<u8>, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&STANDARD.encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let s = String::deserialize(d)?;
        STANDARD.decode(s).map_err(serde::de::Error::custom)
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
            // `eulerancestral` is the legacy debug-lower form written by
            // pre-#265 TUI `save_prefs_for_model`; kept as a read-only
            // alias so existing model_prefs rows still parse after the
            // migration to canonical Display format.
            "euler-ancestral" | "euler_ancestral" | "eulerancestral" => {
                Ok(Scheduler::EulerAncestral)
            }
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
    /// Optional visual style the expansion should absorb (e.g. a style preset
    /// label). Sent as a natural-language instruction to the expander — never
    /// appended to the prompt verbatim. Additive: old clients omit it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = "gritty film noir")]
    pub style: Option<String>,
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
    /// Optional generation metadata to preserve in a post-generation upscale.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Object)]
    pub metadata: Option<OutputMetadata>,
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
    /// Output format for the generated media.
    ///
    /// When omitted the server picks a sensible default based on the model
    /// family: `mp4` for video models (`ltx2`, `ltx-video`), `png` for all
    /// image families. Explicitly setting this field always wins — including
    /// if you set `png` for a video model, which the server will then reject
    /// with a 422 so you get a clear error rather than a silent wrong output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<OutputFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embed_metadata: Option<bool>,
    /// Scheduler override for UNet-based models (SD1.5, SDXL).
    /// Ignored by flow-matching models (FLUX, SD3, Z-Image, Flux.2, Qwen-Image).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheduler: Option<Scheduler>,
    /// Enable CFG++ (manifold-projection classifier-free guidance, Chung et al. 2024).
    /// At each Euler step, x_0 is estimated from the CFG-guided velocity but the
    /// re-noise direction uses the unconditional velocity, keeping the trajectory
    /// on the data manifold and allowing lower CFG scales (e.g. 1.5–2 instead of 7).
    /// SD3 only in this release; SDXL/SD15 ports require scheduler-API extensions
    /// and are tracked separately. Ignored by distilled families (FLUX schnell,
    /// Z-Image) and any model run at cfg ≈ 1.0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg_plus: Option<bool>,
    /// Source image for img2img generation (raw PNG/JPEG bytes, base64-encoded in JSON).
    #[serde(default, skip_serializing_if = "Option::is_none", with = "base64_opt")]
    pub source_image: Option<Vec<u8>>,
    /// Client-supplied provenance label for `source_image` — the gallery
    /// filename or upload name it was picked from. Recorded into
    /// `OutputMetadata::source_image_name` so clients can attempt to restore
    /// the input image when reusing settings; the engine never reads it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_image_name: Option<String>,
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
    /// Durable client-generated identifier shared by prepared batch siblings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_id: Option<String>,
    /// One-based sibling position within `batch_count`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_index: Option<u32>,
    /// Total number of siblings in the prepared batch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_count: Option<u32>,
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
    /// Optional server-local conditioning audio path for trusted LTX-2 deployments.
    /// Resolved only by `mold serve` against configured allow roots.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_file_path: Option<String>,
    /// Optional source video for video-to-video / retake generation.
    #[serde(default, skip_serializing_if = "Option::is_none", with = "base64_opt")]
    pub source_video: Option<Vec<u8>>,
    /// Optional server-local source video path for trusted LTX-2 deployments.
    /// Resolved only by `mold serve` against configured allow roots.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_video_path: Option<String>,
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
    /// Optional per-component device placement override. `None` preserves
    /// the engine's VRAM-aware auto-placement end-to-end. See §3 of the
    /// 2026-04-19 model-ui-overhaul design doc.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement: Option<DevicePlacement>,
}

impl GenerateRequest {
    /// Returns the resolved output format, falling back to the default (`Png`)
    /// when the caller did not supply one.
    ///
    /// In normal server flows `normalise_output_format` is called before this
    /// and fills `output_format` with a family-aware default, so this method
    /// always returns the normalised value. Inference engines and other
    /// consumers that hold a fully-normalised request can use this instead of
    /// accessing the field directly to avoid an `.unwrap()`.
    pub fn resolved_output_format(&self) -> OutputFormat {
        self.output_format.unwrap_or_default()
    }

    /// Fill `output_format` with a family-aware default when the caller did
    /// not supply one.
    ///
    /// - `ltx2` with `enable_audio == Some(true)` → `Mp4` (audio requires mp4)
    /// - `ltx2` (any other case) → `Mp4` (most compatible video container)
    /// - `ltx-video` → `Mp4` (most compatible; engine falls back to APNG when
    ///   the field is non-video, but Mp4 is the right API default)
    /// - all other families → `Png` (existing image default)
    ///
    /// This is a no-op when `output_format` is already `Some(…)` — explicit
    /// caller choices are always preserved, even invalid ones. Validation that
    /// runs after normalisation will then reject them with a clear error.
    pub fn normalise_output_format(&mut self, family: Option<&str>) -> &mut Self {
        if self.output_format.is_some() {
            return self;
        }
        self.output_format = Some(match family {
            Some("ltx2") | Some("ltx-video") => OutputFormat::Mp4,
            _ => OutputFormat::Png,
        });
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct KeyframeCondition {
    #[schema(example = 0)]
    pub frame: u32,
    #[serde(with = "base64_required")]
    pub image: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, utoipa::ToSchema)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, utoipa::ToSchema)]
pub struct LoraWeight {
    /// Path to the LoRA safetensors file.
    #[schema(example = "/path/to/lora.safetensors")]
    pub path: String,
    /// Scaling factor for LoRA effect (0.0 = no effect, 1.0 = full strength, up to 2.0).
    #[serde(default = "default_lora_scale")]
    #[schema(example = 1.0)]
    pub scale: f64,
}

/// Installed LoRA adapter available to generation clients.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct LoraInfo {
    /// Catalog id, for example `cv:8001`.
    #[schema(example = "cv:8001")]
    pub id: String,
    /// Human-readable LoRA name.
    #[schema(example = "Skin Texture Detail")]
    pub name: String,
    /// Compatible model family slug.
    #[schema(example = "flux")]
    pub family: String,
    /// Optional upstream author/creator.
    pub author: Option<String>,
    /// Absolute server-side path to pass as `loras[].path`.
    #[schema(example = "/home/user/.mold/models/cv-8001/lora.safetensors")]
    pub path: String,
    /// Civitai trained words / trigger phrases, when known.
    #[serde(default)]
    pub trained_words: Vec<String>,
    /// Size of the primary LoRA file, when known.
    pub size_bytes: Option<u64>,
    /// Preview image URL from the catalog sidecar, when known.
    pub thumbnail_url: Option<String>,
    /// Unix timestamp from the install sidecar.
    pub added_at: i64,
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
    /// Which GPU ordinal handled this request (multi-GPU only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<usize>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_index: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_count: Option<u32>,
    pub model: String,
    pub seed: u64,
    pub steps: u32,
    pub guidance: f64,
    pub width: u32,
    pub height: u32,
    /// Generation canvas before any post-generation upscaler. These stay
    /// stable when `width` / `height` are updated to describe the saved file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strength: Option<f64>,
    /// Provenance label of the img2img source (client-supplied filename) —
    /// present only when the request carried a source image and a name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_image_name: Option<String>,
    /// SHA-256 (hex) of the exact `source_image` bytes used. Lets clients
    /// look the source back up in a local stash when reusing settings —
    /// names and hashes only, never image payloads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_image_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheduler: Option<Scheduler>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<OutputFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg_plus: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_scale: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loras: Option<Vec<LoraWeight>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_scale: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upscale_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gif_preview: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_audio: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_file_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_video_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pipeline: Option<Ltx2PipelineMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retake_range: Option<TimeRange>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spatial_upscale: Option<Ltx2SpatialUpscale>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal_upscale: Option<Ltx2TemporalUpscale>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frames: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
    /// Durable chain-job id this output was finalized from (additive;
    /// absent for single generations, the ephemeral shim, and legacy rows).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chain_job_id: Option<String>,
    /// Structured multi-clip provenance for chain outputs (additive) —
    /// every clip's prompt, frames, transition, and effective seed, so a
    /// sequence is never recorded under clip 1's prompt alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chain: Option<crate::chain::ChainOutputMetadata>,
    pub version: String,
}

impl OutputMetadata {
    pub fn from_generate_request(
        req: &GenerateRequest,
        seed: u64,
        scheduler: Option<Scheduler>,
        version: impl Into<String>,
    ) -> Self {
        let loras = req
            .loras
            .clone()
            .or_else(|| req.lora.clone().map(|lora| vec![lora]));
        let (lora, lora_scale) = match loras.as_ref().and_then(|items| items.first()) {
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
            batch_id: req.batch_id.clone(),
            batch_index: req.batch_index,
            batch_count: req.batch_count,
            model: req.model.clone(),
            seed,
            steps: req.steps,
            guidance: req.guidance,
            width: req.width,
            height: req.height,
            generation_width: Some(req.width),
            generation_height: Some(req.height),
            strength: req.source_image.as_ref().map(|_| req.strength),
            source_image_name: req
                .source_image
                .as_ref()
                .and_then(|_| req.source_image_name.clone()),
            source_image_sha256: req.source_image.as_ref().map(|bytes| {
                use sha2::{Digest, Sha256};
                let mut hasher = Sha256::new();
                hasher.update(bytes);
                format!("{:x}", hasher.finalize())
            }),
            scheduler,
            output_format: req.output_format,
            cfg_plus: req.cfg_plus,
            lora,
            lora_scale,
            loras,
            control_model: req.control_model.clone(),
            control_scale: (req.control_image.is_some() || req.control_model.is_some())
                .then_some(req.control_scale),
            upscale_model: req.upscale_model.clone(),
            gif_preview: req.gif_preview.then_some(true),
            enable_audio: req.enable_audio,
            audio_file_path: req.audio_file_path.clone(),
            source_video_path: req.source_video_path.clone(),
            pipeline: req.pipeline,
            retake_range: req.retake_range.clone(),
            spatial_upscale: req.spatial_upscale,
            temporal_upscale: req.temporal_upscale,
            frames: req.frames,
            fps: req.fps,
            chain_job_id: None,
            chain: None,
            version: version.into(),
        }
    }

    /// Overwrite the requested dimensions with the actual output raster
    /// size (post-upscale, post-fit). Gallery rows and embedded metadata
    /// should describe the file that exists, not the request that made it.
    pub fn apply_output_dimensions(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
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

#[derive(Debug, Clone, Default, Serialize, Deserialize, utoipa::ToSchema)]
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
    /// Default video frame count (additive; absent for image models).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = 97)]
    pub default_frames: Option<u32>,
    /// Default video FPS (additive; absent for image models).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = 24)]
    pub default_fps: Option<u32>,
    /// Maximum frames a single request may ask for WITHOUT temporal
    /// upscaling (additive; absent for image models). Clients with
    /// temporal-upscale support keep their own doubling math; the server
    /// validator remains authoritative.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = 153)]
    pub max_frames: Option<u32>,
    /// Valid frame counts are `k * frame_step + 1` (additive; absent for
    /// image models). 8 for the LTX families.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = 8)]
    pub frame_step: Option<u32>,
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
    /// Human-readable title for catalog-installed models whose `name` is
    /// an opaque `cv:<id>` / `hf:<repo>` identifier (additive; absent for
    /// manifest models, whose `name` is already readable). Display only —
    /// every API call still addresses the model by `name`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// Catalog classification for installed models. Absent for older servers
    /// and manifest rows whose family is sufficient for client-side inference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// Catalog modality for installed models (`image` / `video`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modality: Option<String>,
    /// Explicit mature-content classification. `None` means unknown; clients
    /// must never render an unclassified older sidecar as known-safe.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nsfw: Option<bool>,
    /// Model-specific LTX-2 audio-output capability. `None` preserves
    /// compatibility with older servers that only advertised family support.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_audio: Option<bool>,
}

impl ModelInfoExtended {
    /// Human-readable presentation label. The stable `name` remains the value
    /// used for API requests, persistence, and model identity.
    pub fn human_name(&self) -> String {
        if let Some(display_name) = self
            .display_name
            .as_deref()
            .map(str::trim)
            .filter(|name| !name.is_empty())
        {
            return display_name.to_string();
        }
        if (self.name.starts_with("cv:") || self.name.starts_with("hf:"))
            && !self.defaults.description.trim().is_empty()
        {
            return self.defaults.description.trim().to_string();
        }
        if let Some(id) = self.name.strip_prefix("cv:") {
            return format!("Civitai model #{id}");
        }
        if let Some(repo) = self.name.strip_prefix("hf:") {
            let title = repo
                .rsplit('/')
                .next()
                .unwrap_or(repo)
                .replace(['-', '_'], " ");
            return title;
        }
        self.name.clone()
    }

    /// Resolve a model id carried by queue/status/history records through an
    /// inventory fetched from `/api/models`.
    pub fn human_name_for(name: &str, models: &[Self]) -> String {
        if let Some(model) = models.iter().find(|model| model.name == name) {
            return model.human_name();
        }
        if let Some(id) = name.strip_prefix("cv:") {
            return format!("Civitai model #{id}");
        }
        if let Some(repo) = name.strip_prefix("hf:") {
            return repo
                .rsplit('/')
                .next()
                .unwrap_or(repo)
                .replace(['-', '_'], " ");
        }
        name.to_string()
    }

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

#[cfg(test)]
mod model_defaults_frame_tests {
    use super::ModelDefaults;

    /// Old wire shape (no frame fields) must keep parsing, and the new
    /// fields must default to None and be omitted from serialized output —
    /// image models never advertise frame semantics.
    #[test]
    fn model_defaults_frame_fields_roundtrip_and_default_to_none() {
        let legacy = r#"{
            "default_steps": 4,
            "default_guidance": 3.5,
            "default_width": 1024,
            "default_height": 1024,
            "description": "flux schnell"
        }"#;
        let parsed: ModelDefaults = serde_json::from_str(legacy).unwrap();
        assert_eq!(parsed.default_frames, None);
        assert_eq!(parsed.default_fps, None);
        assert_eq!(parsed.max_frames, None);
        assert_eq!(parsed.frame_step, None);

        let out = serde_json::to_value(&parsed).unwrap();
        assert!(out.get("default_frames").is_none());
        assert!(out.get("default_fps").is_none());
        assert!(out.get("max_frames").is_none());
        assert!(out.get("frame_step").is_none());

        let video = ModelDefaults {
            default_frames: Some(97),
            default_fps: Some(24),
            max_frames: Some(153),
            frame_step: Some(8),
            ..parsed
        };
        let out = serde_json::to_value(&video).unwrap();
        assert_eq!(out["default_frames"], 97);
        assert_eq!(out["default_fps"], 24);
        assert_eq!(out["max_frames"], 153);
        assert_eq!(out["frame_step"], 8);

        let back: ModelDefaults = serde_json::from_value(out).unwrap();
        assert_eq!(back.default_frames, Some(97));
        assert_eq!(back.default_fps, Some(24));
        assert_eq!(back.max_frames, Some(153));
        assert_eq!(back.frame_step, Some(8));
    }
}

#[cfg(test)]
mod model_display_name_tests {
    use super::{ModelDefaults, ModelInfo, ModelInfoExtended};

    fn model(name: &str, display_name: Option<&str>, description: &str) -> ModelInfoExtended {
        ModelInfoExtended {
            info: ModelInfo {
                name: name.to_string(),
                family: "sdxl".to_string(),
                size_gb: 1.0,
                is_loaded: false,
                last_used: None,
                hf_repo: String::new(),
            },
            defaults: ModelDefaults {
                default_steps: 20,
                default_guidance: 7.0,
                default_width: 1024,
                default_height: 1024,
                description: description.to_string(),
                ..Default::default()
            },
            downloaded: true,
            disk_usage_bytes: None,
            remaining_download_bytes: None,
            display_name: display_name.map(str::to_string),
            kind: None,
            modality: None,
            nsfw: None,
            supports_audio: None,
        }
    }

    #[test]
    fn resolves_wire_ids_to_human_readable_inventory_names() {
        let models = vec![model(
            "cv:1759168",
            Some("Juggernaut XL - Ragnarok"),
            "legacy title",
        )];
        assert_eq!(
            ModelInfoExtended::human_name_for("cv:1759168", &models),
            "Juggernaut XL - Ragnarok"
        );
        assert_eq!(
            ModelInfoExtended::human_name_for("cv:999", &models),
            "Civitai model #999"
        );
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
    /// Per-GPU worker status (multi-GPU only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpus: Option<Vec<GpuWorkerStatus>>,
    /// Current request queue depth (multi-GPU only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_depth: Option<usize>,
    /// Maximum queue capacity (multi-GPU only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_capacity: Option<usize>,
    /// Whether new-job dispatch is currently paused (`POST /api/queue/pause`).
    /// Absent on older servers that don't support pausing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_paused: Option<bool>,
    /// Stable UUID identifying this server installation. Persisted in the
    /// metadata DB on first boot; ephemeral (per-process) when the DB is
    /// unavailable. Absent on older servers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = "0b5c1a4e-9f3d-4c8a-b2e7-6d1f0a9c3e58")]
    pub instance_id: Option<String>,
    /// Disk usage of the filesystem holding the models directory. Absent on
    /// older servers or when the mount cannot be determined.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models_disk: Option<DiskUsage>,
}

/// Total/free bytes for the filesystem backing a directory (currently the
/// models dir, surfaced as `ServerStatus::models_disk`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct DiskUsage {
    #[schema(example = 994662584320_u64)]
    pub total_bytes: u64,
    #[schema(example = 213909504000_u64)]
    pub free_bytes: u64,
}

// ── GET /api/queue wire types ────────────────────────────────────────────────

/// One row of `GET /api/queue` — the client-side, Deserialize-capable twin of
/// mold-server's `job_registry::JobEntry` (which is Serialize-only).
///
/// Forward-compat rules:
/// - `state` is a plain [`String`] rather than an enum — current servers ship
///   `"queued"` / `"running"`, and a future server growing a new lifecycle
///   state must not break older clients.
/// - The additive fields (`gpu`, `target_gpu`, `seed_pinned`, `metadata`)
///   `#[serde(default)]` so older servers that omit them still parse, and
///   `skip_serializing_if` so re-serializing matches the server's contract
///   (queued rows carry no `gpu` key at all — never `"gpu": null`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueueJobEntryWire {
    pub id: String,
    pub model: String,
    /// Job lifecycle state — `"queued"` or `"running"` on current servers.
    pub state: String,
    /// Unix-epoch milliseconds when the job was accepted by the server.
    pub started_at_unix_ms: u64,
    /// 0-based index in the server's dispatch-priority order at snapshot
    /// time — 0 is at the head (about to be dispatched, or already running).
    pub position: usize,
    /// GPU ordinal currently running this job (absent for queued rows).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<usize>,
    /// Preferred GPU ordinal for queued jobs (absent means Auto).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_gpu: Option<usize>,
    /// Whether the submitted request pinned a seed. Additive — absent on
    /// older servers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_pinned: Option<bool>,
    /// The submitted request's parameters, metadata-shaped so any client can
    /// inspect a queued job and reuse its settings. Additive — absent on
    /// older servers; never carries image payloads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Box<OutputMetadata>>,
}

/// Whole-queue listing returned by `GET /api/queue` — the client-side twin of
/// mold-server's `job_registry::QueueListing`. The server wraps the rows in
/// an object (not a bare array) so the response can grow extra fields without
/// a breaking change; unknown fields are ignored here for the same reason.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct QueueListingWire {
    #[serde(default)]
    pub entries: Vec<QueueJobEntryWire>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, utoipa::ToSchema)]
pub struct GpuInfo {
    #[schema(example = "NVIDIA GeForce RTX 4090")]
    pub name: String,
    #[schema(example = 24564)]
    pub vram_total_mb: u64,
    #[schema(example = 8192)]
    pub vram_used_mb: u64,
    /// Compute backend driving this GPU. Additive: absent from older peers
    /// (≤ 0.16), and `None` is elided so older clients keep the old shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = "cuda")]
    pub backend: Option<GpuBackend>,
}

/// GPU selection for multi-GPU setups.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum GpuSelection {
    /// Use all discovered GPUs (default).
    #[default]
    All,
    /// Use only these specific GPU ordinals.
    Specific(Vec<usize>),
}

impl GpuSelection {
    /// Parse from comma-separated string like "0,1,2".
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        if s.is_empty() || s.to_lowercase() == "all" {
            return Ok(Self::All);
        }
        let ordinals: Vec<usize> = s
            .split(',')
            .map(|s| s.trim().parse::<usize>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("invalid GPU ordinal: {e}"))?;
        if ordinals.is_empty() {
            return Ok(Self::All);
        }
        Ok(Self::Specific(ordinals))
    }
}

/// Per-GPU worker status for multi-GPU status reporting.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GpuWorkerStatus {
    pub ordinal: usize,
    pub name: String,
    pub vram_total_bytes: u64,
    pub vram_used_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loaded_model: Option<String>,
    pub state: GpuWorkerState,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum GpuWorkerState {
    Idle,
    Generating,
    Loading,
    Degraded,
}

// ── Device placement (Agent C: model-ui-overhaul §3) ─────────────────────────

/// A user-facing request for where a component should run.
///
/// - `Auto` preserves the existing VRAM-aware auto-placement logic.
/// - `Cpu` pins the component to CPU regardless of available VRAM.
/// - `Gpu { ordinal }` pins to GPU ordinal `n` (CUDA-specific ordinal,
///   or `0` on Metal/unified memory).
///
/// Serialized as an externally-tagged enum: `{"kind":"auto"}`,
/// `{"kind":"cpu"}`, or `{"kind":"gpu","ordinal":1}`. A missing `DeviceRef`
/// field deserializes to `Auto`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default, utoipa::ToSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DeviceRef {
    #[default]
    Auto,
    Cpu,
    Gpu {
        ordinal: usize,
    },
}

impl DeviceRef {
    /// Helper constructor mirroring the compact `Gpu(n)` form used in tests.
    pub const fn gpu(ordinal: usize) -> Self {
        DeviceRef::Gpu { ordinal }
    }
}

/// Top-level placement request attached to `GenerateRequest` and persisted
/// under `[models."name:tag".placement]` in config.
///
/// - `text_encoders` is the Tier 1 "group knob" — a single override applied
///   to all text-encoder components (T5 plus CLIP-L, Qwen3, Qwen2.5-VL, etc.).
/// - `advanced` is Tier 2, available only for families listed in spec §3.2
///   (FLUX, Flux.2, Z-Image, Qwen-Image; SD3.5 stretch). When `Some`, each
///   populated field overrides the Tier 1 group knob for that specific
///   component. When `None`, only the group knob is honored.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, utoipa::ToSchema)]
pub struct DevicePlacement {
    #[serde(default)]
    pub text_encoders: DeviceRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub advanced: Option<AdvancedPlacement>,
}

/// Per-component placement overrides available only for Tier 2 families.
///
/// `transformer` and `vae` are required fields (default `Auto`). The
/// per-encoder fields are `Option<DeviceRef>` because not every family has
/// every encoder — FLUX has T5 plus CLIP-L, Flux.2 and Z-Image have Qwen3,
/// Qwen-Image has Qwen2.5-VL. `None` means "follow the Tier 1 group knob";
/// `Some(DeviceRef::Auto)` means "follow the engine's own auto logic".
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, utoipa::ToSchema)]
pub struct AdvancedPlacement {
    #[serde(default)]
    pub transformer: DeviceRef,
    #[serde(default)]
    pub vae: DeviceRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip_l: Option<DeviceRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip_g: Option<DeviceRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t5: Option<DeviceRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qwen: Option<DeviceRef>,
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
    /// Live low-fidelity preview of the denoising latent: a base64 PNG at
    /// latent resolution (~width/8 × height/8) — clients upscale it. Emitted
    /// throttled between denoise steps; disable with `MOLD_STEP_PREVIEW=0`.
    Preview {
        image: String,
        step: usize,
        total: usize,
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
    /// Request is queued behind other generations. The first event the
    /// server emits per request — clients can latch onto `id` to later
    /// reconcile against `GET /api/queue` (sweep zombie cards whose
    /// SSE stream silently dropped). `id` defaults to an empty string
    /// on legacy servers that predate the field; new clients should
    /// treat empty as "no server-assigned identifier".
    Queued {
        position: usize,
        #[serde(default)]
        id: String,
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
    /// Original generated image before post-generation upscaling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_image: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_height: Option<u32>,
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
    /// GPU ordinal that handled this request (multi-GPU only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<usize>,

    // ── Gallery provenance (additive; absent on older servers) ──────────
    /// Filename this payload was saved under in the server's gallery, when
    /// the server persisted it. Clients that mirror the output locally keep
    /// this name so the copy and the original stay one logical print.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    /// Gallery filename of the pre-upscale original, when one was saved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_filename: Option<String>,
    /// The exact `OutputMetadata` the server recorded for this payload, so
    /// mirroring clients don't have to re-parse (or, for video formats that
    /// embed nothing, synthesize) it from the file bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Object)]
    pub metadata: Option<Box<OutputMetadata>>,
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

// ── Resource telemetry (Agent B scope) ───────────────────────────────────────

/// Point-in-time resource snapshot emitted by the server aggregator at 1 Hz.
/// Serialized over `GET /api/resources` and `GET /api/resources/stream`.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ResourceSnapshot {
    /// Host that produced this snapshot. Useful when pointing `MOLD_HOST` at
    /// a remote GPU — the SPA shows this in the resource side-sheet.
    pub hostname: String,
    /// Unix millis at sample time.
    pub timestamp: i64,
    pub gpus: Vec<GpuSnapshot>,
    pub system_ram: RamSnapshot,
    /// System-wide CPU utilization (averaged across all cores). `None` when
    /// the aggregator hasn't had two samples yet (CPU usage is computed from
    /// deltas — the first snapshot always reports zero).
    #[serde(default)]
    pub cpu: Option<CpuSnapshot>,
}

/// Per-GPU memory snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GpuSnapshot {
    pub ordinal: usize,
    pub name: String,
    pub backend: GpuBackend,
    pub vram_total: u64,
    pub vram_used: u64,
    /// Bytes attributable to the running `mold` process (CUDA only).
    /// `None` on Metal and on CUDA hosts that fell back to `nvidia-smi`.
    pub vram_used_by_mold: Option<u64>,
    /// `vram_used - vram_used_by_mold`. `None` whenever `vram_used_by_mold` is.
    pub vram_used_by_other: Option<u64>,
    /// GPU core utilization in percent (0-100). `None` on Metal and on the
    /// `nvidia-smi` fallback path — only NVML exposes this cheaply.
    #[serde(default)]
    pub gpu_utilization: Option<u8>,
}

/// Aggregate CPU snapshot. `usage_percent` is a 0-100 average across every
/// logical core.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct CpuSnapshot {
    pub cores: u16,
    pub usage_percent: f32,
}

/// System RAM snapshot. Per-process fields are always populated (via sysinfo).
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct RamSnapshot {
    pub total: u64,
    pub used: u64,
    pub used_by_mold: u64,
    pub used_by_other: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum GpuBackend {
    Cuda,
    Metal,
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
    fn expand_request_serde_roundtrip_with_style() {
        let req = ExpandRequest {
            prompt: "a cat".to_string(),
            model_family: "flux".to_string(),
            variations: 4,
            style: Some("gritty film noir".to_string()),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: ExpandRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.style.as_deref(), Some("gritty film noir"));
        assert_eq!(back.prompt, "a cat");
        assert_eq!(back.variations, 4);
    }

    #[test]
    fn expand_request_serde_roundtrip_without_style() {
        let req = ExpandRequest {
            prompt: "a cat".to_string(),
            model_family: "sdxl".to_string(),
            variations: 1,
            style: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        // No style set — the field stays off the wire entirely.
        assert!(!json.contains("style"));
        let back: ExpandRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.style, None);
    }

    #[test]
    fn expand_request_old_client_missing_style_is_none() {
        // Old clients don't know about `style` — the field must default to None.
        let back: ExpandRequest = serde_json::from_str(r#"{"prompt":"a cat"}"#).unwrap();
        assert_eq!(back.style, None);
        assert_eq!(back.model_family, "flux");
        assert_eq!(back.variations, 1);
    }

    // ── GET /api/queue wire types ────────────────────────────────────────

    #[test]
    fn queue_listing_wire_parses_a_realistic_snapshot() {
        // Mirrors mold-server's job_registry serialization contract
        // (`snapshot_serializes_with_snake_case_state_and_omits_gpu_when_queued`
        // and `snapshot_carries_request_metadata_only_when_registered_with_it`):
        // running rows carry `gpu`, queued rows omit it, and the additive
        // `target_gpu` / `seed_pinned` / `metadata` fields ride only when
        // present. Unknown fields (a future server growing the wrapper or a
        // row) must be ignored.
        let body = r#"{
            "entries": [
                {
                    "id": "job-running",
                    "model": "flux-dev:fp16",
                    "state": "running",
                    "started_at_unix_ms": 1711305600000,
                    "position": 0,
                    "gpu": 1
                },
                {
                    "id": "job-queued",
                    "model": "sdxl:q8",
                    "state": "queued",
                    "started_at_unix_ms": 1711305601000,
                    "position": 1,
                    "target_gpu": 0,
                    "seed_pinned": true,
                    "metadata": {
                        "prompt": "a cat",
                        "model": "sdxl:q8",
                        "seed": 42,
                        "steps": 20,
                        "guidance": 5.0,
                        "width": 1024,
                        "height": 1024,
                        "version": "0.20.2"
                    },
                    "some_future_row_field": true
                }
            ],
            "some_future_total": 2
        }"#;
        let listing: QueueListingWire = serde_json::from_str(body).unwrap();
        assert_eq!(listing.entries.len(), 2);

        let running = &listing.entries[0];
        assert_eq!(running.id, "job-running");
        assert_eq!(running.model, "flux-dev:fp16");
        assert_eq!(running.state, "running");
        assert_eq!(running.started_at_unix_ms, 1_711_305_600_000);
        assert_eq!(running.position, 0);
        assert_eq!(running.gpu, Some(1));
        assert_eq!(running.target_gpu, None);

        let queued = &listing.entries[1];
        assert_eq!(queued.state, "queued");
        assert_eq!(queued.position, 1);
        assert_eq!(queued.gpu, None);
        assert_eq!(queued.target_gpu, Some(0));
        assert_eq!(queued.seed_pinned, Some(true));
        let meta = queued.metadata.as_deref().expect("metadata rides");
        assert_eq!(meta.prompt, "a cat");
        assert_eq!(meta.seed, 42);
        assert_eq!(meta.width, 1024);
    }

    #[test]
    fn queue_job_entry_wire_accepts_unknown_future_states() {
        // `state` is deliberately a plain String, not an enum — a server
        // that grows a new lifecycle state must not break older clients.
        let json = r#"{"id":"j1","model":"flux-dev:q8","state":"some-future-state","started_at_unix_ms":1,"position":0}"#;
        let entry: QueueJobEntryWire = serde_json::from_str(json).unwrap();
        assert_eq!(entry.state, "some-future-state");
        // And it round-trips: Serialize is derived too.
        let back: QueueJobEntryWire =
            serde_json::from_str(&serde_json::to_string(&entry).unwrap()).unwrap();
        assert_eq!(back, entry);
    }

    #[test]
    fn queue_job_entry_wire_defaults_fields_absent_on_older_servers() {
        // The endpoint originally shipped only the five required fields;
        // `gpu` / `target_gpu` / `seed_pinned` / `metadata` are additive and
        // must default when an older server omits them.
        let json = r#"{"id":"j1","model":"flux-dev:q8","state":"queued","started_at_unix_ms":1711305600000,"position":3}"#;
        let entry: QueueJobEntryWire = serde_json::from_str(json).unwrap();
        assert_eq!(entry.id, "j1");
        assert_eq!(entry.position, 3);
        assert_eq!(entry.gpu, None);
        assert_eq!(entry.target_gpu, None);
        assert_eq!(entry.seed_pinned, None);
        assert!(entry.metadata.is_none());
    }

    #[test]
    fn queue_job_entry_wire_serializes_like_the_server_omitting_absent_options() {
        // Same wire contract as the server: absent options stay off the wire
        // entirely (clients must not see `"gpu": null` and infer GPU 0).
        let entry = QueueJobEntryWire {
            id: "j1".to_string(),
            model: "flux-dev:q8".to_string(),
            state: "queued".to_string(),
            started_at_unix_ms: 5,
            position: 0,
            gpu: None,
            target_gpu: None,
            seed_pinned: None,
            metadata: None,
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains(r#""state":"queued""#), "got: {json}");
        assert!(!json.contains("gpu"), "leaked a gpu field: {json}");
        assert!(!json.contains("seed_pinned"), "got: {json}");
        assert!(!json.contains("metadata"), "got: {json}");
    }

    #[test]
    fn queue_listing_wire_defaults_entries_when_absent() {
        // Tolerate a hypothetical minimal wrapper — `entries` defaults empty.
        let listing: QueueListingWire = serde_json::from_str("{}").unwrap();
        assert!(listing.entries.is_empty());
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
            output_format: Some(OutputFormat::Png),
            embed_metadata: Some(true),
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            source_image_name: None,
            edit_images: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            audio_file_path: None,
            source_video: None,
            source_video_path: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
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
    fn generate_request_server_local_media_paths_serde_roundtrip() {
        let json = r#"{"prompt":"test","model":"ltx-2-19b-distilled:fp8","width":960,"height":576,"steps":8,"batch_size":1,"output_format":"mp4","audio_file_path":"/srv/mold-media/voice.wav","source_video_path":"/srv/mold-media/clip.mp4"}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();

        assert_eq!(
            req.audio_file_path.as_deref(),
            Some("/srv/mold-media/voice.wav")
        );
        assert_eq!(
            req.source_video_path.as_deref(),
            Some("/srv/mold-media/clip.mp4")
        );

        let encoded = serde_json::to_string(&req).unwrap();
        assert!(encoded.contains(r#""audio_file_path":"/srv/mold-media/voice.wav""#));
        assert!(encoded.contains(r#""source_video_path":"/srv/mold-media/clip.mp4""#));
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
    fn generate_request_output_format_omitted_is_none() {
        // output_format omitted — field is None; normalise_output_format fills it later
        let json = r#"{"prompt":"test","model":"flux-schnell","width":768,"height":768,"steps":4,"batch_size":1}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.output_format, None);
        // resolved_output_format falls back to Png for None
        assert_eq!(req.resolved_output_format(), OutputFormat::Png);
    }

    #[test]
    fn generate_request_output_format_explicit_jpeg() {
        let json = r#"{"prompt":"test","model":"flux-schnell","width":768,"height":768,"steps":4,"batch_size":1,"output_format":"jpeg"}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.output_format, Some(OutputFormat::Jpeg));
    }

    #[test]
    fn generate_request_minimal_json() {
        // Minimal request: only required fields, all optional fields use defaults
        let json = r#"{"prompt":"a cat","model":"test","width":512,"height":512,"steps":4}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "a cat");
        assert_eq!(req.output_format, None);
        assert_eq!(req.resolved_output_format(), OutputFormat::Png);
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
            output_format: Some(OutputFormat::Png),
            embed_metadata: None,
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            source_image_name: None,
            edit_images: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            audio_file_path: None,
            source_video: None,
            source_video_path: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
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
            output_format: Some(OutputFormat::Png),
            embed_metadata: None,
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            source_image_name: None,
            edit_images: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            audio_file_path: None,
            source_video: None,
            source_video_path: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
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
            output_format: Some(OutputFormat::Png),
            embed_metadata: Some(true),
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            source_image_name: None,
            edit_images: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            batch_id: Some("prepared-batch-1".to_string()),
            batch_index: Some(2),
            batch_count: Some(3),
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            audio_file_path: None,
            source_video: None,
            source_video_path: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
        };

        let metadata = OutputMetadata::from_generate_request(&req, 7, None, "0.1.0");
        assert_eq!(metadata.strength, None);
        assert_eq!(metadata.version, "0.1.0");
        assert_eq!(metadata.batch_id.as_deref(), Some("prepared-batch-1"));
        assert_eq!(metadata.batch_index, Some(2));
        assert_eq!(metadata.batch_count, Some(3));
        // No source image → no provenance fields (and the label alone never
        // rides without the image).
        assert_eq!(metadata.source_image_name, None);
        assert_eq!(metadata.source_image_sha256, None);
        let json = serde_json::to_string(&metadata).unwrap();
        assert!(!json.contains("source_image_name"));
        assert!(!json.contains("source_image_sha256"));
    }

    /// Reuse-settings source restore: the metadata records the client's
    /// provenance label and the SHA-256 of the exact source bytes — names
    /// and hashes only, never the image payload.
    #[test]
    fn output_metadata_records_source_image_provenance() {
        let mut req = GenerateRequest {
            prompt: "test".to_string(),
            negative_prompt: None,
            model: "flux-dev:q8".to_string(),
            width: 1024,
            height: 1024,
            steps: 4,
            guidance: 3.5,
            seed: Some(7),
            batch_size: 1,
            output_format: Some(OutputFormat::Png),
            embed_metadata: Some(true),
            scheduler: None,
            cfg_plus: None,
            source_image: Some(b"fake-png-bytes".to_vec()),
            source_image_name: Some("mold-flux-123-456.png".to_string()),
            edit_images: None,
            strength: 0.6,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            audio_file_path: None,
            source_video: None,
            source_video_path: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
        };

        let metadata = OutputMetadata::from_generate_request(&req, 7, None, "0.1.0");
        assert_eq!(
            metadata.source_image_name.as_deref(),
            Some("mold-flux-123-456.png")
        );
        // sha256("fake-png-bytes")
        let expected = {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(b"fake-png-bytes");
            format!("{:x}", hasher.finalize())
        };
        assert_eq!(metadata.source_image_sha256.as_deref(), Some(&expected[..]));
        // The sha never depends on the label...
        req.source_image_name = None;
        let unlabeled = OutputMetadata::from_generate_request(&req, 7, None, "0.1.0");
        assert_eq!(unlabeled.source_image_name, None);
        assert_eq!(
            unlabeled.source_image_sha256.as_deref(),
            Some(&expected[..])
        );
        // ...and both fields are additive: older metadata blobs without them
        // still deserialize.
        let legacy: OutputMetadata = serde_json::from_str(&{
            let mut v: serde_json::Value = serde_json::to_value(&metadata).unwrap();
            let obj = v.as_object_mut().unwrap();
            obj.remove("source_image_name");
            obj.remove("source_image_sha256");
            serde_json::to_string(&v).unwrap()
        })
        .unwrap();
        assert_eq!(legacy.source_image_name, None);
        assert_eq!(legacy.source_image_sha256, None);
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
            output_format: Some(OutputFormat::Png),
            embed_metadata: Some(true),
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            source_image_name: None,
            edit_images: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            audio_file_path: None,
            source_video: None,
            source_video_path: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
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
            output_format: Some(OutputFormat::Png),
            embed_metadata: Some(true),
            scheduler: Some(Scheduler::UniPc),
            cfg_plus: None,
            source_image: Some(vec![1, 2, 3]),
            source_image_name: None,
            edit_images: None,
            strength: 0.5,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            audio_file_path: None,
            source_video: None,
            source_video_path: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
        };

        let metadata =
            OutputMetadata::from_generate_request(&req, 9, Some(Scheduler::UniPc), "0.1.0");
        assert_eq!(metadata.strength, Some(0.5));
        assert_eq!(metadata.scheduler, Some(Scheduler::UniPc));
    }

    #[test]
    fn output_metadata_preserves_recreate_knobs() {
        let req = GenerateRequest {
            prompt: "video".to_string(),
            negative_prompt: Some("blur".to_string()),
            model: "ltx-2.3-22b-distilled:fp8".to_string(),
            width: 960,
            height: 576,
            steps: 8,
            guidance: 3.0,
            seed: Some(9),
            batch_size: 1,
            output_format: Some(OutputFormat::Mp4),
            embed_metadata: Some(true),
            scheduler: None,
            cfg_plus: Some(true),
            source_image: None,
            source_image_name: None,
            edit_images: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: Some("controlnet-canny-sd15".to_string()),
            control_scale: 0.8,
            expand: None,
            original_prompt: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            lora: None,
            frames: Some(97),
            fps: Some(24),
            upscale_model: Some("real-esrgan-x4plus:fp16".to_string()),
            gif_preview: true,
            enable_audio: Some(false),
            audio_file: None,
            audio_file_path: Some("/srv/mold/voice.wav".to_string()),
            source_video: None,
            source_video_path: Some("/srv/mold/source.mp4".to_string()),
            keyframes: None,
            pipeline: Some(Ltx2PipelineMode::Retake),
            loras: Some(vec![LoraWeight {
                path: "/loras/camera.safetensors".to_string(),
                scale: 0.7,
            }]),
            retake_range: Some(TimeRange {
                start_seconds: 1.0,
                end_seconds: 2.5,
            }),
            spatial_upscale: Some(Ltx2SpatialUpscale::X1_5),
            temporal_upscale: Some(Ltx2TemporalUpscale::X2),
            placement: None,
        };

        let metadata = OutputMetadata::from_generate_request(&req, 9, None, "0.1.0");

        assert_eq!(metadata.output_format, Some(OutputFormat::Mp4));
        assert_eq!(metadata.cfg_plus, Some(true));
        assert_eq!(
            metadata.control_model.as_deref(),
            Some("controlnet-canny-sd15")
        );
        assert_eq!(metadata.control_scale, Some(0.8));
        assert_eq!(
            metadata.upscale_model.as_deref(),
            Some("real-esrgan-x4plus:fp16")
        );
        assert_eq!(metadata.gif_preview, Some(true));
        assert_eq!(metadata.enable_audio, Some(false));
        assert_eq!(
            metadata.audio_file_path.as_deref(),
            Some("/srv/mold/voice.wav")
        );
        assert_eq!(
            metadata.source_video_path.as_deref(),
            Some("/srv/mold/source.mp4")
        );
        assert_eq!(metadata.pipeline, Some(Ltx2PipelineMode::Retake));
        assert_eq!(
            metadata.loras.as_ref().unwrap()[0].path,
            "/loras/camera.safetensors"
        );
        assert_eq!(
            metadata.retake_range,
            Some(TimeRange {
                start_seconds: 1.0,
                end_seconds: 2.5,
            }),
        );
        assert_eq!(metadata.spatial_upscale, Some(Ltx2SpatialUpscale::X1_5));
        assert_eq!(metadata.temporal_upscale, Some(Ltx2TemporalUpscale::X2));
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
            original_image: None,
            original_width: None,
            original_height: None,
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
            gpu: Some(1),
            filename: None,
            original_filename: None,
            metadata: None,
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
        assert_eq!(back.gpu, Some(1));
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
    fn preview_event_serde_roundtrip() {
        // Wire contract for the live denoise preview SSE event: tagged
        // "preview" with a base64 PNG + step counters.
        let event = SseProgressEvent::Preview {
            image: "aGVsbG8=".to_string(),
            step: 2,
            total: 4,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"preview""#), "got: {json}");
        assert!(json.contains(r#""image":"aGVsbG8=""#), "got: {json}");
        let back: SseProgressEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            back,
            SseProgressEvent::Preview {
                step: 2,
                total: 4,
                ..
            }
        ));
    }

    #[test]
    fn history_listing_serde_roundtrip() {
        // Wire contract for GET /api/history — `{ "entries": [...] }` with
        // exactly { prompt, model, used_at } per row.
        let listing = HistoryListing {
            entries: vec![HistoryEntry {
                prompt: "a cat".to_string(),
                model: "flux-dev:q8".to_string(),
                used_at: 1_700_000_000_000,
            }],
        };
        let json = serde_json::to_string(&listing).unwrap();
        assert!(json.contains(r#""prompt":"a cat""#), "got: {json}");
        assert!(json.contains(r#""used_at":1700000000000"#), "got: {json}");
        let back: HistoryListing = serde_json::from_str(&json).unwrap();
        assert_eq!(back.entries.len(), 1);
        assert_eq!(back.entries[0].model, "flux-dev:q8");
        assert_eq!(back.entries[0].used_at, 1_700_000_000_000);
    }

    #[test]
    fn model_removal_response_serde_roundtrip() {
        // Wire contract for DELETE /api/models/:model — removed paths,
        // kept shared components with their surviving referents, and the
        // bytes actually freed on disk.
        let resp = ModelRemovalResponse {
            removed: vec!["/models/flux-schnell-q8/transformer.gguf".to_string()],
            kept: vec![KeptComponent {
                component: "/models/shared/flux/ae.safetensors".to_string(),
                used_by: vec!["flux-dev:q8".to_string()],
            }],
            freed_bytes: 12_345,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains(r#""freed_bytes":12345"#), "got: {json}");
        assert!(json.contains(r#""used_by":["flux-dev:q8"]"#), "got: {json}");
        let back: ModelRemovalResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.removed.len(), 1);
        assert_eq!(back.kept[0].used_by, vec!["flux-dev:q8".to_string()]);
        assert_eq!(back.freed_bytes, 12_345);
    }

    #[test]
    fn config_wire_types_serde_roundtrip() {
        // Wire contract for the /api/config surface — typed JSON values with
        // a source tag mirroring `mold config list --json`.
        let listing = ConfigListing {
            profile: Some("default".to_string()),
            entries: vec![ConfigEntry {
                key: "server_port".to_string(),
                value: serde_json::json!(7680),
                source: "file".to_string(),
                env_var: None,
            }],
        };
        let json = serde_json::to_string(&listing).unwrap();
        assert!(json.contains(r#""source":"file""#), "got: {json}");
        assert!(json.contains(r#""value":7680"#), "got: {json}");
        // env_var is omitted from the wire unless the source is "env".
        assert!(!json.contains("env_var"), "got: {json}");
        let back: ConfigListing = serde_json::from_str(&json).unwrap();
        assert_eq!(back.entries[0].key, "server_port");
        assert_eq!(back.profile.as_deref(), Some("default"));

        let env_entry = ConfigEntry {
            key: "embed_metadata".to_string(),
            value: serde_json::json!(true),
            source: "env".to_string(),
            env_var: Some("MOLD_EMBED_METADATA".to_string()),
        };
        let json = serde_json::to_string(&env_entry).unwrap();
        assert!(
            json.contains(r#""env_var":"MOLD_EMBED_METADATA""#),
            "got: {json}"
        );

        let profiles = ConfigProfiles {
            active: "dev".to_string(),
            profiles: vec!["default".to_string(), "dev".to_string()],
        };
        let json = serde_json::to_string(&profiles).unwrap();
        let back: ConfigProfiles = serde_json::from_str(&json).unwrap();
        assert_eq!(back.active, "dev");
        assert_eq!(back.profiles.len(), 2);
    }

    #[test]
    fn sse_progress_queued_roundtrip() {
        let event = SseProgressEvent::Queued {
            position: 3,
            id: "job-7".to_string(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"queued""#));
        assert!(json.contains(r#""position":3"#));
        assert!(json.contains(r#""id":"job-7""#));
        let back: SseProgressEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, SseProgressEvent::Queued { position: 3, ref id } if id == "job-7"));
    }

    /// Forward-compat: a Queued payload missing the `id` field — which
    /// legacy servers (pre-L3) emit — must deserialize cleanly with an
    /// empty id. Without `#[serde(default)]` on the field this would
    /// reject every Queued event from an older mold-serve.
    #[test]
    fn sse_progress_queued_back_compat_missing_id() {
        let legacy = r#"{"type":"queued","position":2}"#;
        let evt: SseProgressEvent = serde_json::from_str(legacy).unwrap();
        match evt {
            SseProgressEvent::Queued { position, id } => {
                assert_eq!(position, 2);
                assert_eq!(id, "");
            }
            other => panic!("expected Queued, got {other:?}"),
        }
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
            output_format: Some(OutputFormat::Png),
            embed_metadata: None,
            scheduler: None,
            cfg_plus: None,
            source_image: Some(image_bytes.clone()),
            source_image_name: None,
            edit_images: None,
            strength: 0.5,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            audio_file_path: None,
            source_video: None,
            source_video_path: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
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
            output_format: Some(OutputFormat::Png),
            embed_metadata: None,
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            source_image_name: None,
            edit_images: Some(vec![image_a.clone(), image_b.clone()]),
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            audio_file_path: None,
            source_video: None,
            source_video_path: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
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
            output_format: Some(OutputFormat::Png),
            embed_metadata: None,
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            source_image_name: None,
            edit_images: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            audio_file_path: None,
            source_video: None,
            source_video_path: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
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
            output_format: Some(OutputFormat::Png),
            embed_metadata: None,
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            source_image_name: None,
            edit_images: None,
            strength: 0.75,
            mask_image: None,
            control_image: Some(control_bytes.clone()),
            control_model: Some("controlnet-canny-sd15".to_string()),
            control_scale: 0.8,
            expand: None,
            original_prompt: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            audio_file_path: None,
            source_video: None,
            source_video_path: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
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
            output_format: Some(OutputFormat::Png),
            embed_metadata: None,
            scheduler: None,
            cfg_plus: None,
            source_image: Some(source_bytes),
            source_image_name: None,
            edit_images: None,
            strength: 0.75,
            mask_image: Some(mask_bytes.clone()),
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            audio_file_path: None,
            source_video: None,
            source_video_path: None,
            keyframes: None,
            pipeline: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
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
            gpus: None,
            queue_depth: None,
            queue_capacity: None,
            queue_paused: Some(true),
            instance_id: None,
            models_disk: None,
        };
        let json = serde_json::to_string(&status).unwrap();
        let parsed: super::ServerStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.hostname.as_deref(), Some("bender"));
        assert_eq!(
            parsed.memory_status.as_deref(),
            Some("Memory: 64.0 GB free, 96.0 GB available")
        );
        assert_eq!(parsed.queue_paused, Some(true));
    }

    #[test]
    fn server_status_omits_queue_paused_when_absent() {
        // Older servers don't emit `queue_paused`; deserializing their
        // responses must default it to None rather than failing.
        let json = r#"{"version":"0.6.3","models_loaded":[],"gpu_info":null,"uptime_secs":0}"#;
        let status: super::ServerStatus = serde_json::from_str(json).unwrap();
        assert_eq!(status.queue_paused, None);
        assert!(!serde_json::to_string(&status)
            .unwrap()
            .contains("queue_paused"));
    }

    #[test]
    fn server_status_deserialize_without_instance_id_or_models_disk() {
        // Older servers don't send `instance_id` / `models_disk` — both must
        // default to None, and neither key may appear on re-serialization.
        let json = r#"{"version":"0.16.0","models_loaded":[],"gpu_info":null,"uptime_secs":0}"#;
        let status: super::ServerStatus = serde_json::from_str(json).unwrap();
        assert_eq!(status.instance_id, None);
        assert_eq!(status.models_disk, None);
        let out = serde_json::to_string(&status).unwrap();
        assert!(!out.contains("instance_id"));
        assert!(!out.contains("models_disk"));
    }

    #[test]
    fn server_status_roundtrip_preserves_instance_id_and_models_disk() {
        let json = r#"{
            "version": "0.17.0",
            "models_loaded": [],
            "gpu_info": null,
            "uptime_secs": 12,
            "instance_id": "0b5c1a4e-9f3d-4c8a-b2e7-6d1f0a9c3e58",
            "models_disk": {"total_bytes": 994662584320, "free_bytes": 213909504000}
        }"#;
        let status: super::ServerStatus = serde_json::from_str(json).unwrap();
        assert_eq!(
            status.instance_id.as_deref(),
            Some("0b5c1a4e-9f3d-4c8a-b2e7-6d1f0a9c3e58")
        );
        assert_eq!(
            status.models_disk,
            Some(super::DiskUsage {
                total_bytes: 994_662_584_320,
                free_bytes: 213_909_504_000,
            })
        );
        let out = serde_json::to_string(&status).unwrap();
        let parsed: super::ServerStatus = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed.instance_id, status.instance_id);
        assert_eq!(parsed.models_disk, status.models_disk);
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
            metadata: None,
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
            metadata: None,
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
            original_image: None,
            original_width: None,
            original_height: None,
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
            gpu: None,
            filename: None,
            original_filename: None,
            metadata: None,
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
            original_image: None,
            original_width: None,
            original_height: None,
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
            gpu: None,
            filename: None,
            original_filename: None,
            metadata: None,
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
    fn sse_complete_event_backward_compat_no_gallery_fields() {
        // Older servers predate filename / original_filename / metadata.
        let json = r#"{"image":"aW1n","format":"png","width":1024,"height":1024,"seed_used":42,"generation_time_ms":5000,"model":"flux-dev:q8"}"#;
        let event: SseCompleteEvent = serde_json::from_str(json).unwrap();
        assert!(event.filename.is_none());
        assert!(event.original_filename.is_none());
        assert!(event.metadata.is_none());
        // And when unset they stay off the wire for older clients.
        let out = serde_json::to_string(&event).unwrap();
        assert!(!out.contains("filename"));
        assert!(!out.contains("\"metadata\""));
    }

    #[test]
    fn sse_complete_event_image_no_video_fields_in_json() {
        // An image-only event should not include any video_* keys in JSON
        let event = SseCompleteEvent {
            image: "aW1n".to_string(),
            format: OutputFormat::Png,
            width: 1024,
            height: 1024,
            original_image: None,
            original_width: None,
            original_height: None,
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
            gpu: None,
            filename: None,
            original_filename: None,
            metadata: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(!json.contains("video_"));
    }

    #[test]
    fn resource_snapshot_serde_roundtrip() {
        let snap = ResourceSnapshot {
            hostname: "hal9000".to_string(),
            timestamp: 1_700_000_000_000,
            gpus: vec![GpuSnapshot {
                ordinal: 0,
                name: "NVIDIA RTX 3090".to_string(),
                backend: GpuBackend::Cuda,
                vram_total: 24_000_000_000,
                vram_used: 14_200_000_000,
                vram_used_by_mold: Some(10_100_000_000),
                vram_used_by_other: Some(4_100_000_000),
                gpu_utilization: Some(42),
            }],
            system_ram: RamSnapshot {
                total: 64_000_000_000,
                used: 38_400_000_000,
                used_by_mold: 22_100_000_000,
                used_by_other: 16_300_000_000,
            },
            cpu: Some(CpuSnapshot {
                cores: 16,
                usage_percent: 27.5,
            }),
        };
        let json = serde_json::to_string(&snap).unwrap();
        let back: ResourceSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(back.hostname, "hal9000");
        assert_eq!(back.gpus.len(), 1);
        assert_eq!(back.gpus[0].ordinal, 0);
        assert_eq!(back.gpus[0].backend, GpuBackend::Cuda);
        assert_eq!(back.gpus[0].vram_used_by_mold, Some(10_100_000_000));
        assert_eq!(back.system_ram.used_by_mold, 22_100_000_000);
    }

    #[test]
    fn gpu_backend_serializes_lowercase() {
        let cuda = serde_json::to_string(&GpuBackend::Cuda).unwrap();
        let metal = serde_json::to_string(&GpuBackend::Metal).unwrap();
        assert_eq!(cuda, "\"cuda\"");
        assert_eq!(metal, "\"metal\"");
    }

    #[test]
    fn gpu_info_without_backend_still_deserializes() {
        // Older peers (≤ 0.16) emit GpuInfo without `backend` — the field is
        // additive and must default to None.
        let legacy: GpuInfo = serde_json::from_str(
            r#"{"name":"NVIDIA GeForce RTX 4090","vram_total_mb":24564,"vram_used_mb":8192}"#,
        )
        .unwrap();
        assert_eq!(legacy.name, "NVIDIA GeForce RTX 4090");
        assert_eq!(legacy.backend, None);
    }

    #[test]
    fn gpu_info_backend_none_is_skipped_and_some_roundtrips() {
        let legacy = GpuInfo {
            name: "NVIDIA GeForce RTX 4090".to_string(),
            vram_total_mb: 24564,
            vram_used_mb: 8192,
            backend: None,
        };
        // None is elided so older clients keep seeing the exact old shape.
        let json = serde_json::to_string(&legacy).unwrap();
        assert!(!json.contains("backend"), "json was: {json}");

        let tagged = GpuInfo {
            backend: Some(GpuBackend::Cuda),
            ..legacy
        };
        let json = serde_json::to_string(&tagged).unwrap();
        assert!(json.contains("\"backend\":\"cuda\""), "json was: {json}");
        let back: GpuInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(back.backend, Some(GpuBackend::Cuda));
    }

    #[test]
    fn metal_snapshot_has_none_per_process_fields() {
        let snap = GpuSnapshot {
            ordinal: 0,
            name: "Apple M3 Max".to_string(),
            backend: GpuBackend::Metal,
            vram_total: 64_000_000_000,
            vram_used: 38_000_000_000,
            vram_used_by_mold: None,
            vram_used_by_other: None,
            gpu_utilization: None,
        };
        let json = serde_json::to_string(&snap).unwrap();
        // Both fields are present as `null` (not elided) so the SPA can
        // reliably `vram_used_by_mold === null` to hide the row.
        assert!(
            json.contains("\"vram_used_by_mold\":null"),
            "json was: {json}"
        );
        assert!(
            json.contains("\"vram_used_by_other\":null"),
            "json was: {json}"
        );
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

/// One server-assisted DNS-SD result returned by `GET /api/discovery/peers`.
/// The browser connects to `url` directly; the serving host is discovery-only
/// and never proxies generation traffic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct DiscoveryPeer {
    pub name: String,
    pub url: String,
    pub host: String,
    pub port: u16,
    pub version: Option<String>,
    pub auth_required: bool,
    pub instance_id: Option<String>,
    pub is_this_machine: bool,
}

/// Whether the server has an active DNS-SD browser backing the discovery API.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiscoveryCapabilities {
    pub can_browse: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GalleryCapabilities {
    /// Whether `DELETE /api/gallery/image/:filename` is allowed by the
    /// server. Always `true` on current builds; kept as a capability field
    /// so older clients that still check it continue to work.
    pub can_delete: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CatalogCapabilities {
    /// Whether the catalog system is available and enabled on this server.
    /// Disabled via `MOLD_CATALOG_DISABLE=1` or `MOLD_CATALOG_DISABLE=true`.
    pub available: bool,
    /// List of model family names available in the catalog.
    pub families: Vec<String>,
    /// Sort orders `GET /api/catalog/search` accepts via `?sort=`
    /// (`"downloads"`, `"recent"`, `"rating"`). Empty on older servers
    /// that ignore the parameter — clients feature-detect against this
    /// before offering sort controls.
    #[serde(default)]
    pub sort: Vec<String>,
}

/// Whether the server exposes the `GET /api/events` broadcast stream.
/// Clients must feature-detect before subscribing — SSE clients that
/// auto-retry would otherwise hammer a 404 on older servers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventsCapabilities {
    pub available: bool,
}

/// Whether the server exposes queue-wide controls. `can_pause` covers
/// `POST /api/queue/pause` and `POST /api/queue/resume`; `can_cancel_all`
/// covers `DELETE /api/queue`; `can_reorder` covers moving a queued job with
/// the `PATCH /api/queue/:id` `position` field. All default to `false` so
/// older servers that omit the fields are treated as lacking the controls.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueueCapabilities {
    pub can_pause: bool,
    pub can_cancel_all: bool,
    #[serde(default)]
    pub can_reorder: bool,
}

/// Prompt-expansion backend category. The API intentionally reports the
/// category rather than the configured URL so capabilities never disclose
/// credentials or internal network details.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpandBackend {
    Local,
    Api,
}

/// Local facts about prompt expansion on this server. API-backed expansion
/// does not probe the external service, so `model_present` is `None` there
/// rather than implying reachability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpandCapabilities {
    /// An API backend is configured, or local expansion support is compiled.
    pub configured: bool,
    /// Whether the configured local model is installed. Unknown/not
    /// applicable for API backends.
    pub model_present: Option<bool>,
    pub backend: ExpandBackend,
}

/// Capabilities payload returned by `GET /api/capabilities`. Grouping keeps
/// the shape extensible — future areas (inpainting, upscaling modes, etc.)
/// can add their own sub-structs without churning existing fields.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServerCapabilities {
    pub gallery: GalleryCapabilities,
    pub catalog: CatalogCapabilities,
    /// Absent on older servers. Missing means LAN browsing is unavailable.
    #[serde(default)]
    pub discovery: DiscoveryCapabilities,
    /// Absent on older servers — `#[serde(default)]` keeps deserialization
    /// of their responses working (events.available = false).
    #[serde(default)]
    pub events: EventsCapabilities,
    /// Absent on older servers — `#[serde(default)]` keeps deserialization
    /// of their responses working (can_pause = can_cancel_all = false).
    #[serde(default)]
    pub queue: QueueCapabilities,
    /// Absent on older servers. Unlike default-false capability groups,
    /// absence here means unknown so newer clients may still try expansion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expand: Option<ExpandCapabilities>,
}

/// One prompt-history row returned by `GET /api/history`. Deliberately a
/// small wire-facing projection of the richer `prompt_history` DB row —
/// clients get what they need to re-run a prompt, nothing more.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct HistoryEntry {
    pub prompt: String,
    pub model: String,
    /// Unix epoch milliseconds when the prompt was recorded.
    pub used_at: i64,
}

/// Whole-history listing returned by `GET /api/history`. Wrapped in a struct
/// so the response can grow extra fields (totals, paging cursors, …) without
/// a breaking change — same rationale as `QueueListing`.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct HistoryListing {
    pub entries: Vec<HistoryEntry>,
}

/// One shared component kept on disk by `DELETE /api/models/:model`
/// because other downloaded models still reference it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct KeptComponent {
    /// Absolute path of the kept file.
    pub component: String,
    /// Models (other than the removed one) that still reference it.
    pub used_by: Vec<String>,
}

/// Response of `DELETE /api/models/:model` — what was deleted, what was
/// kept for other models, and how many bytes were actually freed on disk.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ModelRemovalResponse {
    /// Absolute paths of the files that were deleted.
    pub removed: Vec<String>,
    /// Shared components kept because another model still uses them.
    pub kept: Vec<KeptComponent>,
    /// Bytes freed on disk (hf-cache hardlinks accounted for).
    pub freed_bytes: u64,
}

/// One effective config row returned by the `/api/config` surface. Mirrors
/// a `mold config list --json` entry: a typed JSON value plus the source
/// tag that says which store owns it.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ConfigEntry {
    pub key: String,
    /// Typed JSON value (string/number/bool/null), same encoding as
    /// `mold config list --json`.
    #[schema(value_type = Object)]
    pub value: serde_json::Value,
    /// `"db"` (settings/model_prefs tables), `"file"` (config.toml),
    /// `"env"` (overridden by an environment variable at runtime), or
    /// `"default"` (compiled default after a reset).
    pub source: String,
    /// The environment variable overriding this key — present only when
    /// `source == "env"` so UIs can say "Set by <var> in your environment"
    /// without guessing the key-to-variable mapping.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env_var: Option<String>,
}

/// Whole-config listing returned by `GET /api/config`.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ConfigListing {
    /// Active settings profile, `null` when the metadata DB is disabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    pub entries: Vec<ConfigEntry>,
}

/// Profile listing returned by `GET /api/config/profiles`.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ConfigProfiles {
    /// The currently active profile (env `MOLD_PROFILE` wins over the
    /// stored `profile.active` row).
    pub active: String,
    pub profiles: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GenerationMemoryEstimate {
    pub model: String,
    pub peak_memory_bytes: u64,
    pub activation_memory_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub available_memory_bytes: Option<u64>,
    pub load_strategy: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fits_available_memory: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ModelComponentStatus {
    pub kind: String,
    pub name: String,
    pub present: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repair_model: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub options: Vec<ModelComponentOption>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ModelComponentOption {
    pub label: String,
    pub path: String,
    pub present: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ModelComponentsResponse {
    pub model: String,
    pub components: Vec<ModelComponentStatus>,
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

#[cfg(test)]
#[path = "placement_test.rs"]
mod placement_test;

// ─── Downloads UI (Agent A) ─────────────────────────────────────────────────

/// Lifecycle state of a download job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Active,
    Completed,
    Failed,
    Cancelled,
}

/// Download queue entry. Mirrored 1:1 on the wire; the SPA consumes this as
/// `DownloadJobWire` in `web/src/types.ts`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadJob {
    pub id: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub catalog_id: Option<String>,
    pub status: JobStatus,
    pub files_done: usize,
    pub files_total: usize,
    pub bytes_done: u64,
    pub bytes_total: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Internally tagged enum — on the wire each variant is `{"type": "...", ...}`.
/// Keep `#[serde(tag = "type", rename_all = "snake_case")]` stable; the SPA's
/// `DownloadEventWire` union in `types.ts` depends on this exact shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DownloadEvent {
    /// Sent as the *first* SSE frame to every new subscriber, containing
    /// the queue's current state (active + queued + history). Eliminates
    /// the bootstrap race where a fresh subscriber would otherwise have
    /// to wait for the next delta to know what's running. Mirrors the
    /// pattern already used by `/api/resources/stream`.
    ///
    /// SPA reducer drops this into state via `applyListing` rather than
    /// per-job mutation — it's a full snapshot, not a delta.
    Snapshot {
        listing: DownloadsListing,
    },
    Enqueued {
        id: String,
        model: String,
        position: usize,
    },
    Dequeued {
        id: String,
    },
    Started {
        id: String,
        files_total: usize,
        bytes_total: u64,
    },
    Progress {
        id: String,
        files_done: usize,
        bytes_done: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        current_file: Option<String>,
    },
    FileDone {
        id: String,
        filename: String,
    },
    JobDone {
        id: String,
        model: String,
    },
    JobFailed {
        id: String,
        error: String,
    },
    JobCancelled {
        id: String,
    },
    /// All jobs (primary + every companion) belonging to a catalog entry
    /// have settled. `ok` is true iff every job in the group reached
    /// `Completed`. The SPA listens for this and refreshes its model list
    /// — it replaces the older "refresh on the primary's `JobDone`" path
    /// that fired before companions were necessarily on disk.
    CatalogReady {
        id: String,
        ok: bool,
    },
}

/// Server-wide lifecycle events streamed by `GET /api/events`. One broadcast
/// channel carries every generation job's lifecycle plus gallery mutations so
/// a client can observe the whole server over a single SSE connection —
/// per-job `POST /api/generate/stream` remains the progress/result channel.
///
/// Internally tagged like [`DownloadEvent`]; keep
/// `#[serde(tag = "type", rename_all = "snake_case")]` stable — the desktop
/// app's `ServerEvent` union in `types.ts` depends on this exact shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerEvent {
    JobQueued {
        id: String,
        model: String,
    },
    JobStarted {
        id: String,
        model: String,
        /// GPU ordinal on multi-GPU servers; absent on single-GPU.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gpu: Option<usize>,
    },
    /// The job left the queue for *any* reason — completed, errored, was
    /// cancelled, or its client disconnected. Outcome-aware clients keep
    /// using their per-job stream; `gallery_added` is the durable success
    /// signal.
    JobEnded {
        id: String,
    },
    /// A new output landed in the gallery. `image` carries the full gallery
    /// row when the metadata DB recorded it (clients can insert without a
    /// refetch); `None` when the DB is disabled — refetch `/api/gallery`.
    /// Boxed to keep the enum small (clippy::large_enum_variant); the wire
    /// shape is unchanged.
    GalleryAdded {
        filename: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        image: Option<Box<GalleryImage>>,
    },
    /// An output was deleted via `DELETE /api/gallery/image/:filename`.
    GalleryRemoved {
        filename: String,
    },
    /// A durable chain job entered the queue — created, resumed, retaken,
    /// or amended. Never emitted for the ephemeral legacy-shim jobs.
    /// Distinct from [`Self::JobQueued`] on purpose: old clients ignore
    /// unknown `type` tags, so chain jobs never inherit print-queue
    /// affordances (reorder, `DELETE /api/queue/:id`) they don't support.
    ChainJobQueued {
        id: String,
        model: String,
        stage_count: u32,
    },
    /// The chain runner claimed a chain job and began rendering stages.
    ChainJobStarted {
        id: String,
        model: String,
    },
    /// A chain job settled — completed, failed, or cancelled. Terminal chain
    /// jobs stay listed on `/api/chain-jobs` (that is the resumability
    /// feature); this event only says the runner is done with it.
    ChainJobEnded {
        id: String,
        state: crate::chain_job::ChainJobState,
    },
    /// New-job dispatch was paused via `POST /api/queue/pause`. Emitted only
    /// on the resumed→paused transition — idempotent no-op pauses are silent.
    QueuePaused,
    /// New-job dispatch resumed via `POST /api/queue/resume`. Emitted only on
    /// the paused→resumed transition.
    QueueResumed,
}

/// Listing returned from `GET /api/downloads`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadsListing {
    /// All downloads currently transferring. New clients should use this
    /// field; `active` remains as a compatibility view of the first job.
    #[serde(default)]
    pub active_jobs: Vec<DownloadJob>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active: Option<DownloadJob>,
    pub queued: Vec<DownloadJob>,
    pub history: Vec<DownloadJob>,
}

#[cfg(test)]
mod server_event_tests {
    use super::*;

    #[test]
    fn job_events_serialize_with_snake_case_tags() {
        let queued = ServerEvent::JobQueued {
            id: "j1".into(),
            model: "flux-dev:q4".into(),
        };
        assert_eq!(
            serde_json::to_string(&queued).unwrap(),
            r#"{"type":"job_queued","id":"j1","model":"flux-dev:q4"}"#
        );

        let ended = ServerEvent::JobEnded { id: "j1".into() };
        assert_eq!(
            serde_json::to_string(&ended).unwrap(),
            r#"{"type":"job_ended","id":"j1"}"#
        );
    }

    /// Chain jobs get their own distinct event variants (not a `kind` field
    /// on the print-job events) so old clients — whose reducers ignore
    /// unknown `type` tags — never render chain jobs as reorderable /
    /// queue-cancellable rows aimed at the wrong endpoints.
    #[test]
    fn chain_job_events_serialize_with_snake_case_tags() {
        let queued = ServerEvent::ChainJobQueued {
            id: "c1".into(),
            model: "ltx-2-19b-distilled:fp8".into(),
            stage_count: 3,
        };
        assert_eq!(
            serde_json::to_string(&queued).unwrap(),
            r#"{"type":"chain_job_queued","id":"c1","model":"ltx-2-19b-distilled:fp8","stage_count":3}"#
        );

        let started = ServerEvent::ChainJobStarted {
            id: "c1".into(),
            model: "ltx-2-19b-distilled:fp8".into(),
        };
        assert_eq!(
            serde_json::to_string(&started).unwrap(),
            r#"{"type":"chain_job_started","id":"c1","model":"ltx-2-19b-distilled:fp8"}"#
        );

        let ended = ServerEvent::ChainJobEnded {
            id: "c1".into(),
            state: crate::chain_job::ChainJobState::Completed,
        };
        assert_eq!(
            serde_json::to_string(&ended).unwrap(),
            r#"{"type":"chain_job_ended","id":"c1","state":"completed"}"#
        );
    }

    #[test]
    fn job_started_omits_gpu_when_none() {
        let single = ServerEvent::JobStarted {
            id: "j1".into(),
            model: "sdxl".into(),
            gpu: None,
        };
        let wire = serde_json::to_string(&single).unwrap();
        assert!(
            !wire.contains("gpu"),
            "gpu must be omitted when None: {wire}"
        );

        let multi = ServerEvent::JobStarted {
            id: "j1".into(),
            model: "sdxl".into(),
            gpu: Some(1),
        };
        assert!(serde_json::to_string(&multi)
            .unwrap()
            .contains(r#""gpu":1"#));
    }

    #[test]
    fn gallery_added_omits_image_when_db_disabled() {
        let no_row = ServerEvent::GalleryAdded {
            filename: "cat.png".into(),
            image: None,
        };
        assert_eq!(
            serde_json::to_string(&no_row).unwrap(),
            r#"{"type":"gallery_added","filename":"cat.png"}"#
        );
    }

    #[test]
    fn queue_pause_events_serialize_as_bare_tagged_units() {
        assert_eq!(
            serde_json::to_string(&ServerEvent::QueuePaused).unwrap(),
            r#"{"type":"queue_paused"}"#
        );
        assert_eq!(
            serde_json::to_string(&ServerEvent::QueueResumed).unwrap(),
            r#"{"type":"queue_resumed"}"#
        );
    }

    #[test]
    fn gallery_added_round_trips_with_image_row() {
        let metadata: OutputMetadata = serde_json::from_str(
            r#"{"prompt":"a cat","model":"flux-dev:q4","seed":7,"steps":4,"guidance":3.5,"width":1024,"height":1024,"version":"test"}"#,
        )
        .unwrap();
        let ev = ServerEvent::GalleryAdded {
            filename: "cat.png".into(),
            image: Some(Box::new(GalleryImage {
                filename: "cat.png".into(),
                metadata,
                timestamp: 1_700_000_000,
                format: Some(OutputFormat::Png),
                size_bytes: Some(123),
                metadata_synthetic: false,
            })),
        };
        let wire = serde_json::to_string(&ev).unwrap();
        let back: ServerEvent = serde_json::from_str(&wire).unwrap();
        match back {
            ServerEvent::GalleryAdded {
                filename,
                image: Some(img),
            } => {
                assert_eq!(filename, "cat.png");
                assert_eq!(img.timestamp, 1_700_000_000);
            }
            other => panic!("expected gallery_added with image, got {other:?}"),
        }
    }

    #[test]
    fn capabilities_without_events_field_deserializes_as_unavailable() {
        // An older server's /api/capabilities response has no `events` key.
        let caps: ServerCapabilities = serde_json::from_str(
            r#"{"gallery":{"can_delete":true},"catalog":{"available":false,"families":[]}}"#,
        )
        .unwrap();
        assert!(!caps.events.available);
        assert!(!caps.discovery.can_browse);
    }

    #[test]
    fn capabilities_catalog_without_sort_field_deserializes_as_empty() {
        // Older servers predate server-side catalog sorting and omit
        // `catalog.sort` — clients must see an empty vocabulary, not a
        // deserialization failure.
        let caps: ServerCapabilities = serde_json::from_str(
            r#"{"gallery":{"can_delete":true},"catalog":{"available":true,"families":[]}}"#,
        )
        .unwrap();
        assert!(caps.catalog.sort.is_empty());
    }

    #[test]
    fn capabilities_without_queue_field_deserializes_as_uncontrollable() {
        // An older server omits the `queue` block — both flags default false.
        let caps: ServerCapabilities = serde_json::from_str(
            r#"{"gallery":{"can_delete":true},"catalog":{"available":false,"families":[]},"events":{"available":true}}"#,
        )
        .unwrap();
        assert!(!caps.queue.can_pause);
        assert!(!caps.queue.can_cancel_all);
        assert!(!caps.queue.can_reorder);
    }

    #[test]
    fn capabilities_queue_without_reorder_field_defaults_to_false() {
        // A server that predates reorder support reports `queue` with the two
        // older flags only — `can_reorder` must default false, not fail to
        // deserialize.
        let caps: ServerCapabilities = serde_json::from_str(
            r#"{"gallery":{"can_delete":true},"catalog":{"available":false,"families":[]},"queue":{"can_pause":true,"can_cancel_all":true}}"#,
        )
        .unwrap();
        assert!(caps.queue.can_pause);
        assert!(caps.queue.can_cancel_all);
        assert!(!caps.queue.can_reorder);
    }

    #[test]
    fn capabilities_without_expand_field_deserializes_as_unknown() {
        // Older servers predate expansion capability discovery. Absence must
        // remain distinguishable from a current server reporting local facts.
        let caps: ServerCapabilities = serde_json::from_str(
            r#"{"gallery":{"can_delete":true},"catalog":{"available":false,"families":[]}}"#,
        )
        .unwrap();
        assert!(caps.expand.is_none());
    }

    #[test]
    fn expansion_capabilities_round_trip_nullable_model_presence() {
        let local = ExpandCapabilities {
            configured: true,
            model_present: Some(false),
            backend: ExpandBackend::Local,
        };
        let api = ExpandCapabilities {
            configured: true,
            model_present: None,
            backend: ExpandBackend::Api,
        };

        assert_eq!(
            serde_json::to_value(&local).unwrap(),
            serde_json::json!({
                "configured": true,
                "model_present": false,
                "backend": "local"
            })
        );
        assert_eq!(
            serde_json::to_value(&api).unwrap(),
            serde_json::json!({
                "configured": true,
                "model_present": null,
                "backend": "api"
            })
        );
    }
}

#[cfg(test)]
mod downloads_types_tests {
    use super::*;

    #[test]
    fn job_status_serde_snake_case() {
        let cases = [
            (JobStatus::Queued, "\"queued\""),
            (JobStatus::Active, "\"active\""),
            (JobStatus::Completed, "\"completed\""),
            (JobStatus::Failed, "\"failed\""),
            (JobStatus::Cancelled, "\"cancelled\""),
        ];
        for (status, wire) in cases {
            let s = serde_json::to_string(&status).unwrap();
            assert_eq!(s, wire);
            let back: JobStatus = serde_json::from_str(&s).unwrap();
            assert_eq!(back, status);
        }
    }

    #[test]
    fn download_job_round_trip() {
        let job = DownloadJob {
            id: "11111111-1111-1111-1111-111111111111".to_string(),
            model: "flux-dev:q4".to_string(),
            catalog_id: None,
            status: JobStatus::Active,
            files_done: 2,
            files_total: 5,
            bytes_done: 1_000_000,
            bytes_total: 3_000_000,
            current_file: Some("transformer.gguf".to_string()),
            started_at: Some(1_700_000_000_000),
            completed_at: None,
            error: None,
        };
        let s = serde_json::to_string(&job).unwrap();
        let back: DownloadJob = serde_json::from_str(&s).unwrap();
        assert_eq!(back.id, job.id);
        assert_eq!(back.model, job.model);
        assert_eq!(back.status, JobStatus::Active);
        assert_eq!(back.files_done, 2);
        assert_eq!(back.files_total, 5);
        assert_eq!(back.bytes_done, 1_000_000);
        assert_eq!(back.bytes_total, 3_000_000);
        assert_eq!(back.current_file.as_deref(), Some("transformer.gguf"));
    }

    #[test]
    fn download_event_enqueued_tag_shape() {
        let evt = DownloadEvent::Enqueued {
            id: "abc".to_string(),
            model: "flux-dev:q4".to_string(),
            position: 2,
        };
        let s = serde_json::to_string(&evt).unwrap();
        assert!(s.contains("\"type\":\"enqueued\""), "wire: {s}");
        assert!(s.contains("\"id\":\"abc\""), "wire: {s}");
        assert!(s.contains("\"model\":\"flux-dev:q4\""), "wire: {s}");
        assert!(s.contains("\"position\":2"), "wire: {s}");
    }

    #[test]
    fn download_event_progress_tag_shape() {
        let evt = DownloadEvent::Progress {
            id: "abc".to_string(),
            files_done: 1,
            bytes_done: 2_000_000,
            current_file: Some("clip.safetensors".to_string()),
        };
        let s = serde_json::to_string(&evt).unwrap();
        assert!(s.contains("\"type\":\"progress\""), "wire: {s}");
        assert!(s.contains("\"bytes_done\":2000000"), "wire: {s}");
    }
}
