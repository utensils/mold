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

/// Scheduler / solver selection.
///
/// Two disjoint families share this wire slot:
/// - `Ddim` / `EulerAncestral` / `UniPc` — UNet-based image models (SD1.5,
///   SDXL). Flow-matching image models (FLUX, SD3, Z-Image, Flux.2,
///   Qwen-Image) ignore those.
/// - `Euler` / `DpmPp` — Wan's flow-matching sample solvers (upstream
///   `--sample_solver`), alongside `UniPc` which doubles as Wan's default
///   FlowUniPC. Validation rejects them for every non-wan family (#795).
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS,
)]
#[serde(rename_all = "kebab-case")]
pub enum Scheduler {
    #[default]
    Ddim,
    EulerAncestral,
    UniPc,
    /// Plain flow Euler over the diffusers/Lightning sigma grid — the solver
    /// the lightx2v 4-step recipe specifies (wan only).
    Euler,
    /// Wan's `FlowDPMSolverMultistepScheduler` (`fm_solvers.py`), order 2,
    /// dpmsolver++ midpoint, over upstream's `get_sampling_sigmas` grid
    /// (wan only).
    DpmPp,
}

impl std::fmt::Display for Scheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scheduler::Ddim => write!(f, "ddim"),
            Scheduler::EulerAncestral => write!(f, "euler-ancestral"),
            Scheduler::UniPc => write!(f, "uni-pc"),
            Scheduler::Euler => write!(f, "euler"),
            Scheduler::DpmPp => write!(f, "dpm-pp"),
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
            "euler" => Ok(Scheduler::Euler),
            // `dpm++` is upstream Wan's spelling; kebab-case `dpm-pp` is the
            // wire form.
            "dpm-pp" | "dpm++" | "dpmpp" | "dpm_pp" => Ok(Scheduler::DpmPp),
            other => Err(format!(
                "unknown scheduler: '{other}'. Valid: ddim, euler-ancestral, uni-pc, euler, dpm-pp"
            )),
        }
    }
}

/// Resolved generation task that shapes prompt expansion.
///
/// This is intentionally semantic rather than a media payload: clients tell
/// the expander which conditioning contract the generation route resolved,
/// while source images, videos, keyframes, and audio remain on the generation
/// request that owns them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "kebab-case")]
pub enum ExpandTask {
    /// Ordinary image generation (the backward-compatible default).
    #[default]
    TextToImage,
    /// Video generated from text alone.
    TextToVideo,
    /// Motion generated from an authoritative opening image.
    ImageToVideo,
    /// Transformation or continuation of an authoritative source video.
    VideoToVideo,
    /// Partial regeneration of a bounded source-video range.
    Retake,
    /// Motion connecting authoritative keyframe images.
    KeyframeInterpolation,
    /// Video motion synchronized to authoritative conditioning audio.
    AudioDrivenVideo,
    /// Semantic audio-video resynthesis from ordered heterogeneous references.
    ReferenceToAudioVideo,
    /// Audio-only generation from text.
    TextToAudio,
}

impl ExpandTask {
    /// Backward-compatible policy for callers that only send a family.
    pub fn for_family(family: &str) -> Self {
        match family.trim().to_ascii_lowercase().as_str() {
            "ltx2" | "ltx-2" | "ltx-video" | "wan" | "wan2.1" | "wan2.2" | "minimax-h3"
            | "minimax_h3" | "minimaxh3" => Self::TextToVideo,
            _ => Self::TextToImage,
        }
    }

    /// Resolve the narrow expansion policy from an admitted generation
    /// request. More authoritative conditioning wins over incidental media.
    pub fn for_generation(family: &str, req: &GenerateRequest) -> Self {
        if crate::minimax_h3::is_family(family)
            && req
                .references
                .as_ref()
                .is_some_and(|references| !references.is_empty())
        {
            return Self::ReferenceToAudioVideo;
        }
        Self::for_conditioning(
            family,
            req.pipeline,
            req.source_image.is_some(),
            req.source_video.is_some()
                || req
                    .source_video_path
                    .as_deref()
                    .is_some_and(|path| !path.trim().is_empty())
                || req.extend_video.is_some()
                || req
                    .extend_video_path
                    .as_deref()
                    .is_some_and(|path| !path.trim().is_empty()),
            req.audio_file.is_some()
                || req
                    .audio_file_path
                    .as_deref()
                    .is_some_and(|path| !path.trim().is_empty()),
            req.keyframes.as_ref().map_or(0, Vec::len),
            req.retake_range.is_some(),
            req.frames,
        )
    }

    /// Resolve from the minimal conditioning facts available before a full
    /// generation request is assembled (notably the CLI expansion phase).
    #[allow(clippy::too_many_arguments)]
    pub fn for_conditioning(
        family: &str,
        pipeline: Option<Ltx2PipelineMode>,
        has_source_image: bool,
        has_source_video: bool,
        has_audio: bool,
        keyframe_count: usize,
        has_retake_range: bool,
        frames: Option<u32>,
    ) -> Self {
        let normalized = family.trim().to_ascii_lowercase();
        if !matches!(
            normalized.as_str(),
            "ltx2"
                | "ltx-2"
                | "ltx-video"
                | "wan"
                | "wan2.1"
                | "wan2.2"
                | "minimax-h3"
                | "minimax_h3"
                | "minimaxh3"
        ) {
            return Self::TextToImage;
        }
        match pipeline {
            Some(Ltx2PipelineMode::T2a) => return Self::TextToAudio,
            Some(Ltx2PipelineMode::Retake) => return Self::Retake,
            Some(Ltx2PipelineMode::Keyframe) => return Self::KeyframeInterpolation,
            Some(Ltx2PipelineMode::A2Vid | Ltx2PipelineMode::LipDub) => {
                return Self::AudioDrivenVideo;
            }
            // Other explicit pipelines win over incidental incompatible
            // selectors, but their actual image/video conditioning still
            // distinguishes T2V, I2V, and V2V below.
            Some(_) => {}
            None => {
                // Mirrors `validation::ltx2_implicit_pipeline`: retake wins,
                // then audio, then keyframes, then source video.
                if has_retake_range {
                    return Self::Retake;
                }
                if has_audio {
                    return Self::AudioDrivenVideo;
                }
                if keyframe_count > 1 {
                    return Self::KeyframeInterpolation;
                }
            }
        }
        if has_source_video {
            return Self::VideoToVideo;
        }
        if has_source_image {
            return Self::ImageToVideo;
        }
        // A single-frame Wan render with no conditioning is a still (#798):
        // prompt work is image-style visual description, not chronological
        // shot direction. Deliberately after the source checks — a
        // source-conditioned one-frame request keeps its source-preserving
        // contract even though the output is a still.
        // Twin: `studio/lib/expandTask.ts`.
        if matches!(normalized.as_str(), "wan" | "wan2.1" | "wan2.2") && frames == Some(1) {
            return Self::TextToImage;
        }
        Self::TextToVideo
    }
}

impl std::fmt::Display for ExpandTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::TextToImage => "text-to-image",
            Self::TextToVideo => "text-to-video",
            Self::ImageToVideo => "image-to-video",
            Self::VideoToVideo => "video-to-video",
            Self::Retake => "retake",
            Self::KeyframeInterpolation => "keyframe-interpolation",
            Self::AudioDrivenVideo => "audio-driven-video",
            Self::ReferenceToAudioVideo => "reference-to-audio-video",
            Self::TextToAudio => "text-to-audio",
        })
    }
}

impl std::str::FromStr for ExpandTask {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "text-to-image" => Ok(Self::TextToImage),
            "text-to-video" => Ok(Self::TextToVideo),
            "image-to-video" => Ok(Self::ImageToVideo),
            "video-to-video" => Ok(Self::VideoToVideo),
            "retake" => Ok(Self::Retake),
            "keyframe-interpolation" => Ok(Self::KeyframeInterpolation),
            "audio-driven-video" => Ok(Self::AudioDrivenVideo),
            "reference-to-audio-video" => Ok(Self::ReferenceToAudioVideo),
            "text-to-audio" => Ok(Self::TextToAudio),
            _ => Err(format!(
                "unknown expansion task '{value}'. Valid: text-to-image, text-to-video, \
                 image-to-video, video-to-video, retake, keyframe-interpolation, \
                 audio-driven-video, reference-to-audio-video, text-to-audio"
            )),
        }
    }
}

/// Which creative dimension a prompt remix is allowed to vary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "kebab-case")]
pub enum RemixDimension {
    Composition,
    Camera,
    Lighting,
    Setting,
    Mood,
    Movement,
    Style,
}

impl std::fmt::Display for RemixDimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Composition => "composition",
            Self::Camera => "camera",
            Self::Lighting => "lighting",
            Self::Setting => "setting",
            Self::Mood => "mood",
            Self::Movement => "movement",
            Self::Style => "style",
        })
    }
}

impl std::str::FromStr for RemixDimension {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "composition" => Ok(Self::Composition),
            "camera" => Ok(Self::Camera),
            "lighting" => Ok(Self::Lighting),
            "setting" => Ok(Self::Setting),
            "mood" => Ok(Self::Mood),
            "movement" => Ok(Self::Movement),
            "style" => Ok(Self::Style),
            _ => Err(format!(
                "unknown remix dimension '{value}'. Valid: composition, camera, lighting, setting, mood, movement, style"
            )),
        }
    }
}

/// Which prompt a Remix request used as its immediate source.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "kebab-case")]
pub enum RemixSourceKind {
    Original,
    Current,
    #[default]
    Direct,
}

/// Prompt-transform operation retained with generated output provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "kebab-case")]
pub enum PromptTransformOperation {
    Expand,
    Remix,
}

/// Additive provenance describing the prompt transform that produced a
/// generation prompt. `root_prompt` is the user's earliest known idea while
/// `source_prompt` is the exact text passed to the transform backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct PromptTransformProvenance {
    pub operation: PromptTransformOperation,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::prompt_text::deserialize_optional_prompt"
    )]
    pub root_prompt: Option<String>,
    #[serde(deserialize_with = "crate::prompt_text::deserialize_prompt")]
    pub source_prompt: String,
    #[serde(default)]
    pub source_kind: RemixSourceKind,
    pub task: ExpandTask,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dimensions: Vec<RemixDimension>,
}

/// Request to expand a short prompt into a generation-aware prompt.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ExpandRequest {
    /// Short prompt to expand
    #[schema(example = "a cat")]
    #[serde(deserialize_with = "crate::prompt_text::deserialize_prompt")]
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
    /// Resolved generation/conditioning task. Additive: older clients omit it
    /// and the server infers text-to-video for known video families.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<ExpandTask>,
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

fn default_remix_variations() -> usize {
    3
}

/// Request for subject-preserving prompt alternatives. This deliberately uses
/// a separate endpoint from Expand so older hosts fail closed instead of
/// silently ignoring a mode field and returning the wrong transform.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct RemixRequest {
    #[serde(deserialize_with = "crate::prompt_text::deserialize_prompt")]
    pub source_prompt: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::prompt_text::deserialize_optional_prompt"
    )]
    pub root_prompt: Option<String>,
    #[serde(default)]
    pub source_kind: RemixSourceKind,
    #[serde(default = "default_expand_model_family")]
    pub model_family: String,
    #[serde(default = "default_remix_variations")]
    pub variations: usize,
    /// A fixed style constraint. When present, Style cannot also be varied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<ExpandTask>,
    /// Empty means the server's task-aware default set.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dimensions: Vec<RemixDimension>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct RemixVariant {
    pub prompt: String,
    pub dimensions: Vec<RemixDimension>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct RemixResponse {
    pub source_prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root_prompt: Option<String>,
    pub source_kind: RemixSourceKind,
    pub task: ExpandTask,
    pub variants: Vec<RemixVariant>,
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

/// One media authority for an ordered generation reference.
///
/// The tagged enum makes mixed or ambiguous authorities unrepresentable: a
/// reference carries inline bytes, one request-scoped upload handle, or one
/// server-local path. Server paths are resolved only against configured media
/// roots; upload handles are resolved only inside the authenticated request
/// that owns them. `Descriptor` is a payload-free placement-preview projection
/// and is rejected by ordinary generation validation.
#[derive(Clone, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(tag = "authority", rename_all = "snake_case")]
pub enum GenerationReferenceAuthority {
    Inline {
        #[serde(with = "base64_required")]
        #[schema(value_type = String, format = Byte)]
        data: Vec<u8>,
    },
    Upload {
        handle: String,
    },
    ServerPath {
        path: String,
    },
    Descriptor,
}

impl std::fmt::Debug for GenerationReferenceAuthority {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Inline { data } => formatter
                .debug_struct("Inline")
                .field("data", &format_args!("<redacted {} bytes>", data.len()))
                .finish(),
            Self::Upload { .. } => formatter
                .debug_struct("Upload")
                .field("handle", &"<redacted>")
                .finish(),
            Self::ServerPath { .. } => formatter
                .debug_struct("ServerPath")
                .field("path", &"<redacted>")
                .finish(),
            Self::Descriptor => formatter.write_str("Descriptor"),
        }
    }
}

/// Client provenance that is safe to retain after media resolution.
///
/// `name` is a display label, never a client filesystem path. A digest is
/// optional for inline data because Mold computes it from the received bytes;
/// upload handles and server paths require one before admission so recovery
/// and placement can bind the intended content without persisting the secret
/// handle or unrestricted path.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GenerationReferenceProvenance {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// The user crop a client applied to an IMAGE reference before digesting
    /// and uploading it. The server already received the cropped bytes, so
    /// this is provenance: it is validated as a non-degenerate rectangle
    /// inside its source whose size is the reference's own, then retained
    /// verbatim so Reuse settings can restore the crop.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crop: Option<GenerationReferenceCrop>,
}

/// A client-side crop rectangle in SOURCE pixels of the original photograph.
///
/// `width`/`height` are the cropped reference's own dimensions; `source_*`
/// describe the uncropped photograph and `source_sha256` its digest, which is
/// what lets a reattached original re-apply the same crop exactly. This is
/// never a fit-to-canvas policy: the server normalizes every image reference
/// onto its own 2048-short-edge canvas regardless.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GenerationReferenceCrop {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub source_width: u32,
    pub source_height: u32,
    pub source_sha256: String,
}

impl GenerationReferenceCrop {
    /// Check the rectangle against the reference that carries it: a
    /// non-degenerate rect, inside the source, whose size equals the
    /// reference's own `width`/`height` (a size mismatch is a pre-crop
    /// projection whose bytes were never cropped), with a well-formed source
    /// digest.
    pub fn validate_for_image(
        &self,
        reference_width: u32,
        reference_height: u32,
    ) -> Result<(), &'static str> {
        if self.width == 0 || self.height == 0 {
            return Err("crop must be at least one pixel on each axis");
        }
        if self.source_width == 0 || self.source_height == 0 {
            return Err("crop source dimensions must be positive");
        }
        let inside = self
            .x
            .checked_add(self.width)
            .is_some_and(|right| right <= self.source_width)
            && self
                .y
                .checked_add(self.height)
                .is_some_and(|bottom| bottom <= self.source_height);
        if !inside {
            return Err("crop must lie inside its source image");
        }
        if self.width != reference_width || self.height != reference_height {
            return Err("crop size must equal the cropped reference's own dimensions");
        }
        if self.source_sha256.len() != 64
            || !self
                .source_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("crop source_sha256 must contain exactly 64 hexadecimal characters");
        }
        Ok(())
    }
}

/// Ordered heterogeneous reference input for MiniMax H3 Ref2VA.
///
/// Variant-specific descriptors are required up front so placement preview
/// can reason about row counts without decoding or logging media. The server
/// later content-sniffs and probes the resolved payload and rejects any drift
/// before freezing admission.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GenerationReference {
    Image {
        media: GenerationReferenceAuthority,
        #[serde(default)]
        provenance: GenerationReferenceProvenance,
        mime_type: String,
        width: u32,
        height: u32,
    },
    Video {
        media: GenerationReferenceAuthority,
        #[serde(default)]
        provenance: GenerationReferenceProvenance,
        mime_type: String,
        width: u32,
        height: u32,
        /// Exact decoded source-frame count. Older clients deserialize with
        /// this absent, but Ref2VA planning fails closed until ingress probes
        /// it: duration and a rounded FPS cannot determine CFR resampling at
        /// frame-grid boundaries.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        frame_count: Option<u32>,
        duration_ms: u64,
        fps: f64,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        has_audio: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        audio_duration_ms: Option<u64>,
        /// Exact decoded soundtrack samples per channel at
        /// `audio_sample_rate`. Canonical ingress supplies this authority.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        audio_sample_count: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        audio_sample_rate: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        audio_channels: Option<u16>,
    },
    Audio {
        media: GenerationReferenceAuthority,
        #[serde(default)]
        provenance: GenerationReferenceProvenance,
        mime_type: String,
        duration_ms: u64,
        sample_rate: u32,
        channels: u16,
        /// Exact decoded samples per channel at `sample_rate`. Older clients
        /// remain readable, but Ref2VA planning fails closed until ingress
        /// supplies this authority.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sample_count: Option<u64>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum GenerationReferenceKind {
    Image,
    Video,
    Audio,
}

/// Redacted, durable projection of one ordered reference. This is the only
/// form stored in gallery metadata or emitted in completion events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GenerationReferenceMetadata {
    pub kind: GenerationReferenceKind,
    pub index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub sha256: String,
    pub mime_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<f64>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub has_audio: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_duration_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_sample_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_sample_rate: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_channels: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channels: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_count: Option<u64>,
    /// Exact payload-free versioned preprocessing result used for placement and
    /// admission. Older metadata remains compatible when this is absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prepared_shape: Option<crate::minimax_h3::GenerationReferencePreparedShape>,
    /// The client-side crop an image reference carried (see
    /// [`GenerationReferenceCrop`]); absent for every other reference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crop: Option<GenerationReferenceCrop>,
}

pub fn generation_reference_fingerprint(references: &[GenerationReferenceMetadata]) -> String {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    hash.update(b"mold.minimax-h3.references.v1\0");
    hash.update(
        serde_json::to_vec(references)
            .expect("generation reference metadata serialization is infallible"),
    );
    format!("{:x}", hash.finalize())
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ReferenceUploadSessionRequest {
    /// Complete payload-free request used to bind every upload handle. Every
    /// reference authority must be `descriptor` at this stage.
    pub request: GenerateRequest,
    /// One-based entries whose bytes will arrive through the streaming upload
    /// route. The remaining entries retain inline/server-path authority only in
    /// the final request and are still part of the immutable scope hash.
    pub upload_references: Vec<u32>,
}

#[derive(Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ReferenceUploadSlot {
    pub reference: u32,
    pub handle: String,
}

impl std::fmt::Debug for ReferenceUploadSlot {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReferenceUploadSlot")
            .field("reference", &self.reference)
            .field("handle", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ReferenceUploadSessionResponse {
    pub instance_id: String,
    pub expires_at_ms: u64,
    pub request_scope_sha256: String,
    pub session_handle: String,
    pub uploads: Vec<ReferenceUploadSlot>,
}

impl std::fmt::Debug for ReferenceUploadSessionResponse {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReferenceUploadSessionResponse")
            .field("instance_id", &self.instance_id)
            .field("expires_at_ms", &self.expires_at_ms)
            .field("request_scope_sha256", &self.request_scope_sha256)
            .field("session_handle", &"<redacted>")
            .field("uploads", &self.uploads)
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ReferenceUploadCompleteResponse {
    pub instance_id: String,
    pub reference: u32,
    pub metadata: GenerationReferenceMetadata,
    /// Scope rebound to every canonical descriptor observed so far. This is
    /// authoritative only when `session_complete` is true.
    pub request_scope_sha256: String,
    /// True only after every slot in the request-bound session has been
    /// content-probed and the resulting complete descriptor set has passed
    /// MiniMax H3 validation.
    pub session_complete: bool,
}

impl GenerationReference {
    pub const fn kind(&self) -> GenerationReferenceKind {
        match self {
            Self::Image { .. } => GenerationReferenceKind::Image,
            Self::Video { .. } => GenerationReferenceKind::Video,
            Self::Audio { .. } => GenerationReferenceKind::Audio,
        }
    }

    pub fn media(&self) -> &GenerationReferenceAuthority {
        match self {
            Self::Image { media, .. } | Self::Video { media, .. } | Self::Audio { media, .. } => {
                media
            }
        }
    }

    pub fn provenance(&self) -> &GenerationReferenceProvenance {
        match self {
            Self::Image { provenance, .. }
            | Self::Video { provenance, .. }
            | Self::Audio { provenance, .. } => provenance,
        }
    }

    pub fn content_sha256(&self) -> Option<String> {
        match self.media() {
            GenerationReferenceAuthority::Inline { data } => {
                use sha2::{Digest, Sha256};
                Some(format!("{:x}", Sha256::digest(data)))
            }
            GenerationReferenceAuthority::Upload { .. }
            | GenerationReferenceAuthority::ServerPath { .. }
            | GenerationReferenceAuthority::Descriptor => self
                .provenance()
                .sha256
                .as_deref()
                .map(str::to_ascii_lowercase),
        }
    }

    pub fn redacted_metadata(&self, index: usize) -> Option<GenerationReferenceMetadata> {
        let sha256 = self.content_sha256()?;
        let index = u32::try_from(index).ok()?.checked_add(1)?;
        let prepared_shape = crate::minimax_h3::reference_prepared_shape(self).ok();
        let name = redacted_reference_name(self.provenance());
        Some(match self {
            Self::Image {
                mime_type,
                width,
                height,
                ..
            } => GenerationReferenceMetadata {
                kind: GenerationReferenceKind::Image,
                index,
                name,
                sha256,
                mime_type: mime_type.clone(),
                width: Some(*width),
                height: Some(*height),
                frame_count: None,
                duration_ms: None,
                fps: None,
                has_audio: false,
                audio_duration_ms: None,
                audio_sample_count: None,
                audio_sample_rate: None,
                audio_channels: None,
                sample_rate: None,
                channels: None,
                sample_count: None,
                prepared_shape,
                crop: self.provenance().crop.clone(),
            },
            Self::Video {
                mime_type,
                width,
                height,
                frame_count,
                duration_ms,
                fps,
                has_audio,
                audio_duration_ms,
                audio_sample_count,
                audio_sample_rate,
                audio_channels,
                ..
            } => GenerationReferenceMetadata {
                kind: GenerationReferenceKind::Video,
                index,
                name,
                sha256,
                mime_type: mime_type.clone(),
                width: Some(*width),
                height: Some(*height),
                frame_count: *frame_count,
                duration_ms: Some(*duration_ms),
                fps: Some(*fps),
                has_audio: *has_audio,
                audio_duration_ms: *audio_duration_ms,
                audio_sample_count: *audio_sample_count,
                audio_sample_rate: *audio_sample_rate,
                audio_channels: *audio_channels,
                sample_rate: None,
                channels: None,
                sample_count: None,
                prepared_shape,
                crop: None,
            },
            Self::Audio {
                mime_type,
                duration_ms,
                sample_rate,
                channels,
                sample_count,
                ..
            } => GenerationReferenceMetadata {
                kind: GenerationReferenceKind::Audio,
                index,
                name,
                sha256,
                mime_type: mime_type.clone(),
                width: None,
                height: None,
                frame_count: None,
                duration_ms: Some(*duration_ms),
                fps: None,
                has_audio: false,
                audio_duration_ms: None,
                audio_sample_count: None,
                audio_sample_rate: None,
                audio_channels: None,
                sample_rate: Some(*sample_rate),
                channels: Some(*channels),
                sample_count: *sample_count,
                prepared_shape,
                crop: None,
            },
        })
    }

    /// Redacted metadata for the exact generated duration that will consume
    /// this reference. Long reference video/audio is truncated to the target
    /// duration, so its prepared shape must not inherit the 345-frame planning
    /// ceiling when a shorter request is queued or persisted.
    pub fn redacted_metadata_for_target(
        &self,
        index: usize,
        target_frames: u32,
    ) -> Option<GenerationReferenceMetadata> {
        let mut metadata = self.redacted_metadata(index)?;
        metadata.prepared_shape =
            Some(crate::minimax_h3::reference_prepared_shape_for_target(self, target_frames).ok()?);
        Some(metadata)
    }

    /// Preserve ordered reference metadata even if an internal caller bypassed
    /// request validation. Public ingress rejects a missing digest; this
    /// fallback keeps the offending entry visible instead of silently dropping
    /// the entire list from durable metadata.
    pub fn redacted_metadata_lossless(&self, index: usize) -> GenerationReferenceMetadata {
        if let Some(metadata) = self.redacted_metadata(index) {
            return metadata;
        }
        let index = u32::try_from(index)
            .ok()
            .and_then(|index| index.checked_add(1))
            .unwrap_or(u32::MAX);
        let name = redacted_reference_name(self.provenance());
        let prepared_shape = crate::minimax_h3::reference_prepared_shape(self).ok();
        match self {
            Self::Image {
                mime_type,
                width,
                height,
                ..
            } => GenerationReferenceMetadata {
                kind: GenerationReferenceKind::Image,
                index,
                name,
                sha256: String::new(),
                mime_type: mime_type.clone(),
                width: Some(*width),
                height: Some(*height),
                frame_count: None,
                duration_ms: None,
                fps: None,
                has_audio: false,
                audio_duration_ms: None,
                audio_sample_count: None,
                audio_sample_rate: None,
                audio_channels: None,
                sample_rate: None,
                channels: None,
                sample_count: None,
                prepared_shape,
                crop: self.provenance().crop.clone(),
            },
            Self::Video {
                mime_type,
                width,
                height,
                frame_count,
                duration_ms,
                fps,
                has_audio,
                audio_duration_ms,
                audio_sample_count,
                audio_sample_rate,
                audio_channels,
                ..
            } => GenerationReferenceMetadata {
                kind: GenerationReferenceKind::Video,
                index,
                name,
                sha256: String::new(),
                mime_type: mime_type.clone(),
                width: Some(*width),
                height: Some(*height),
                frame_count: *frame_count,
                duration_ms: Some(*duration_ms),
                fps: Some(*fps),
                has_audio: *has_audio,
                audio_duration_ms: *audio_duration_ms,
                audio_sample_count: *audio_sample_count,
                audio_sample_rate: *audio_sample_rate,
                audio_channels: *audio_channels,
                sample_rate: None,
                channels: None,
                sample_count: None,
                prepared_shape,
                crop: None,
            },
            Self::Audio {
                mime_type,
                duration_ms,
                sample_rate,
                channels,
                sample_count,
                ..
            } => GenerationReferenceMetadata {
                kind: GenerationReferenceKind::Audio,
                index,
                name,
                sha256: String::new(),
                mime_type: mime_type.clone(),
                width: None,
                height: None,
                frame_count: None,
                duration_ms: Some(*duration_ms),
                fps: None,
                has_audio: false,
                audio_duration_ms: None,
                audio_sample_count: None,
                audio_sample_rate: None,
                audio_channels: None,
                sample_rate: Some(*sample_rate),
                channels: Some(*channels),
                sample_count: *sample_count,
                prepared_shape,
                crop: None,
            },
        }
    }

    /// Lossless counterpart to [`Self::redacted_metadata_for_target`].
    /// Invalid internal callers retain the reference entry, but never publish
    /// a prepared shape for a different generated duration.
    pub fn redacted_metadata_lossless_for_target(
        &self,
        index: usize,
        target_frames: u32,
    ) -> GenerationReferenceMetadata {
        let mut metadata = self.redacted_metadata_lossless(index);
        metadata.prepared_shape =
            crate::minimax_h3::reference_prepared_shape_for_target(self, target_frames).ok();
        metadata
    }
}

fn redacted_media_name(name: Option<&str>) -> Option<String> {
    name.map(str::trim)
        .filter(|name| {
            !name.is_empty()
                && name.len() <= crate::minimax_h3::MAX_REFERENCE_NAME_BYTES
                && *name != "."
                && *name != ".."
                && !name.contains(['/', '\\'])
                && !name.chars().any(char::is_control)
        })
        .map(str::to_owned)
}

fn redacted_reference_name(provenance: &GenerationReferenceProvenance) -> Option<String> {
    redacted_media_name(provenance.name.as_deref())
}

/// Which collection a generation should file its finished print into.
///
/// Exactly one of the two fields must be present — a `CollectionRef` with
/// neither set is a 422 at admission rather than a deserialization failure,
/// so the error names the field instead of the JSON shape.
///
/// Clients normally send `name`: collections merge across hosts by slug, so
/// "file this under Smurf Village" is the portable instruction and each host
/// resolves (or creates) its own row. `id` is the exact-row form, used when a
/// client is already looking at one specific host's collection list; the id
/// is resolved to its name at admission so the print's embedded provenance
/// records what it was actually filed under.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct CollectionRef {
    /// Exact collection id (a UUID) on the host that will render this print.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Collection display name, resolved by slug and created when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = "Smurf Village")]
    pub name: Option<String>,
}

impl CollectionRef {
    /// A reference by display name — the portable, cross-host form.
    pub fn by_name(name: impl Into<String>) -> Self {
        Self {
            id: None,
            name: Some(name.into()),
        }
    }

    /// A reference to one host's exact collection row.
    pub fn by_id(id: impl Into<String>) -> Self {
        Self {
            id: Some(id.into()),
            name: None,
        }
    }

    /// True when neither field carries anything usable. Whitespace-only
    /// values count as absent: `{"name": "  "}` is the same mistake as `{}`.
    pub fn is_unset(&self) -> bool {
        let blank = |value: &Option<String>| {
            value
                .as_deref()
                .map(|text| text.trim().is_empty())
                .unwrap_or(true)
        };
        blank(&self.id) && blank(&self.name)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GenerateRequest {
    #[schema(example = "a cat sitting on a windowsill at sunset")]
    #[serde(deserialize_with = "crate::prompt_text::deserialize_prompt")]
    pub prompt: String,
    /// Negative prompt — describes what to avoid generating.
    /// Effective for CFG-based models such as SD1.5, SDXL, SD3, and Wuerstchen.
    /// Ignored by distilled / non-CFG families such as FLUX schnell, Z-Image, etc.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::prompt_text::deserialize_optional_prompt"
    )]
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
    /// Supported by SD3, SDXL, and SD1.5 with DDIM. Ignored by distilled
    /// families (FLUX schnell, Z-Image) and whenever guidance does not
    /// activate CFG.
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
    /// Opaque client-shaped source-fit (crop/pad) policy provenance for the
    /// staged source media. Fitting happens client-side before the bytes
    /// ship, so the engine never reads this — it is recorded verbatim into
    /// `OutputMetadata::source_fit` so clients can restore their crop
    /// controls when reusing settings or selecting a running generation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_fit: Option<serde_json::Value>,
    /// Face-identity reference image for identity conditioning (raw PNG/JPEG
    /// bytes, base64-encoded in JSON). Accepted family-wide on FLUX.1 and SDXL
    /// except for SDXL Turbo, and never alongside a LoRA or an img2img
    /// `source_image`. The payload is bounds-checked
    /// from its header alone before any decode
    /// (`identity::validate_id_image_bytes`).
    #[serde(default, skip_serializing_if = "Option::is_none", with = "base64_opt")]
    pub id_image: Option<Vec<u8>>,
    /// Client-supplied provenance label for `id_image` — the gallery filename
    /// or upload name it was picked from. Recorded into
    /// `OutputMetadata::id_image_name` so clients can attempt to restore the
    /// identity reference when reusing settings; the engine never reads it.
    /// Requires `id_image`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id_image_name: Option<String>,
    /// Strength of the identity conditioning, in `0.0..=identity::ID_WEIGHT_MAX`.
    /// Absent means `identity::ID_WEIGHT_DEFAULT`. Requires `id_image`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id_weight: Option<f64>,
    /// First denoise step at which identity conditioning is applied, so the
    /// composition can settle before the face is pinned. Must be `< steps`.
    /// Absent means `identity::ID_START_STEP_DEFAULT`. Requires `id_image`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id_start_step: Option<u32>,
    /// Ordered face-identity reference images (raw PNG/JPEG bytes,
    /// base64-encoded in JSON). The multi-photograph form of [`Self::id_image`]:
    /// every photo is extracted independently and the resulting identity token
    /// sets are averaged, which is how `cubiq/PuLID_ComfyUI` combines several
    /// references (`pulid.py:406,415-419`).
    ///
    /// `id_image` and `id_images` are the SAME field in two shapes and supplying
    /// both is a validation error, never a silent precedence rule — see
    /// `identity::IDENTITY_IMAGE_FORM_CONFLICT`. The count and the per-image
    /// and whole-set byte and pixel budgets are
    /// `identity::ID_IMAGES_MAX`, `identity::ID_IMAGE_LIMITS`,
    /// `identity::ID_IMAGES_TOTAL_ENCODED_BYTES_MAX`, and
    /// `identity::ID_IMAGES_TOTAL_DECODED_PIXELS_MAX`.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "base64_vec_opt"
    )]
    pub id_images: Option<Vec<Vec<u8>>>,
    /// Client-supplied provenance labels for [`Self::id_images`], in the same
    /// order. Either absent or exactly as long as `id_images`; the engine never
    /// reads it. Requires `id_images`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id_image_names: Option<Vec<String>>,
    /// PuLID true classifier-free guidance scale, in
    /// `identity::TRUE_CFG_OFF..=identity::TRUE_CFG_MAX`.
    ///
    /// Absent — or within `identity::TRUE_CFG_EPSILON` of
    /// `identity::TRUE_CFG_OFF` — keeps FLUX's distilled single-forward
    /// guidance and renders bit-identically to a request that never named it
    /// (`PuLID/flux/sampling.py:120`). Above that, every step from
    /// `cfg_start_step` runs a second forward over `negative_prompt` and the
    /// unconditional identity, combined as
    /// `neg + true_cfg * (pos - neg)` (`PuLID/flux/sampling.py:136-149`).
    /// Qualified only alongside active identity conditioning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub true_cfg: Option<f64>,
    /// First denoise step the true-CFG negative branch runs at. Must be
    /// `< steps`. Absent means `identity::CFG_START_STEP_DEFAULT`. Requires
    /// `true_cfg`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg_start_step: Option<u32>,
    /// Source images for Qwen-Image-Edit generation (raw PNG/JPEG bytes, base64-encoded in JSON).
    /// The first image is the primary edit target; additional images are reference images.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "base64_vec_opt"
    )]
    pub edit_images: Option<Vec<Vec<u8>>>,
    /// Ordered heterogeneous MiniMax H3 Ref2VA inputs. Other families retain
    /// their existing source/edit fields and must reject this additive field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub references: Option<Vec<GenerationReference>>,
    /// Strength for img2img/I2V source conditioning. Two family-specific
    /// conventions share this field (#1055): SD-lineage img2img reads it as
    /// DENOISE strength (0.0 = no change, 1.0 = full noise), while LTX-2
    /// reads it as SOURCE strength (1.0 = pin the opening frame; lower
    /// allows more change). The wire value is never inverted — clients
    /// label it per family via `studio/lib/strengthSemantics.ts`..
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
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::prompt_text::deserialize_optional_prompt"
    )]
    pub original_prompt: Option<String>,
    /// Structured prompt-transform provenance. New clients also populate
    /// `original_prompt` so older hosts retain the root/source prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_transform: Option<PromptTransformProvenance>,
    /// User-authored print title. Validated by
    /// [`crate::validate_print_title`] at admission, embedded into
    /// `OutputMetadata.title`, seeded into the gallery row, and folded into
    /// the output filename as a lossy slug (`crate::title_slug`). Additive;
    /// absent means untitled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = "Smurf village at dusk")]
    pub title: Option<String>,
    /// Tags to file the finished print under. Normalized and capped by
    /// [`crate::normalize_request_tags`] at admission, embedded into
    /// `OutputMetadata.tags`, and seeded onto the gallery row exactly once at
    /// insert. Additive; absent means "file under nothing".
    ///
    /// Organization is user-owned once the print exists — these values seed
    /// the row and are never re-applied on a later refresh or re-publication.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    /// Collection to file the finished print into. Clients normally send
    /// `{ "name": "Smurf Village" }`; the server resolves it by slug and
    /// creates it when it does not exist yet. Additive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collection: Option<CollectionRef>,
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
    /// Existing video to continue, as inline bytes.
    ///
    /// Distinct from `source_video`, which is *reference* conditioning for
    /// video-to-video: `extend_video` makes the request a continuation, so the
    /// delivered output is the original followed by newly generated frames.
    ///
    /// `frames` keeps its usual meaning — the length of the clip the model
    /// renders — and its leading `extend_overlap_frames` reproduce the source
    /// tail, so the run adds `frames - extend_overlap_frames` new frames.
    #[serde(default, skip_serializing_if = "Option::is_none", with = "base64_opt")]
    pub extend_video: Option<Vec<u8>>,
    /// Server-local path of the video to continue, for trusted deployments.
    /// Resolved only by `mold serve` against configured allow roots.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extend_video_path: Option<String>,
    /// Pixel frames of the source tail re-encoded as motion conditioning for
    /// the continuation. Must land on the family's own temporal grid — `8k+1`
    /// for LTX-2's causal VAE, `4k+1` for wan — and be strictly less than
    /// `frames`. `None` resolves to that family's carryover default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extend_overlap_frames: Option<u32>,
    /// Optional keyframe conditioning images for LTX-2 keyframe interpolation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keyframes: Option<Vec<KeyframeCondition>>,
    /// Write the render as an OpenEXR sequence into this directory, in
    /// scene-referred linear HDR, alongside the ordinary tonemapped video.
    ///
    /// A sidecar rather than the primary artifact: a frame sequence is many
    /// files and gigabytes, which the one-file-per-generation gallery cannot
    /// represent. Requires the `hdr` IC-LoRA control, whose adapter is what
    /// makes the render HDR in the first place.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hdr_exr_dir: Option<String>,
    /// Write EXR samples at full 32-bit float instead of 16-bit half.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub hdr_exr_full_float: bool,
    /// Explicit LTX-2 pipeline mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pipeline: Option<Ltx2PipelineMode>,
    /// First-party IC-LoRA control adapter id. The server resolves this to an
    /// exact model-profile-compatible artifact before placement/admission.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ic_lora_control: Option<String>,
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
    /// Optional overrides for the LTX-2 multimodal guider. Absent fields keep
    /// the per-pipeline defaults, so omitting this preserves existing outputs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guidance_overrides: Option<Ltx2GuidanceOverrides>,
    /// Wan flow shift (upstream `--sample_shift`), the family's primary
    /// quality/character knob (#782). Absent keeps the per-tier pipeline
    /// defaults authoritative; precedence is request > `MOLD_WAN_SHIFT` >
    /// per-tier default. Rejected, not ignored, for non-wan families.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_shift: Option<f64>,
    /// Strength for the manifest-shipped Lightning distill on the A14B
    /// high-noise expert (or a single-expert checkpoint's distill). Absent
    /// = 1.0. The community's reduced-motion mitigation runs the high-noise
    /// adapter at 1.5-2.0 (#795). Wan only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distill_strength_high: Option<f64>,
    /// Strength for the low-noise expert's distill. Absent = 1.0. Wan only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distill_strength_low: Option<f64>,
    /// Optional per-component device placement override. `None` preserves
    /// the engine's VRAM-aware auto-placement end-to-end. See §3 of the
    /// 2026-04-19 model-ui-overhaul design doc.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement: Option<DevicePlacement>,
}

impl GenerateRequest {
    /// Whether durable admission must extract request-owned media or media
    /// provenance before persisting the JSON request. Ordered MiniMax H3
    /// references count: their descriptors stay on the request while their
    /// media is sealed beside every other source. Local HDR/LoRA authority
    /// is classified separately.
    pub fn has_durable_media_inputs(&self) -> bool {
        self.references.is_some()
            || self.source_image.is_some()
            || self.source_image_name.is_some()
            || self.id_image.is_some()
            || self.id_image_name.is_some()
            || self.id_images.is_some()
            || self.id_image_names.is_some()
            || self.edit_images.is_some()
            || self.mask_image.is_some()
            || self.control_image.is_some()
            || self.audio_file.is_some()
            || self.audio_file_path.is_some()
            || self.source_video.is_some()
            || self.source_video_path.is_some()
            || self.extend_video.is_some()
            || self.extend_video_path.is_some()
            || self.keyframes.is_some()
    }

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

    /// Whether this request continues an existing video rather than starting
    /// a new one.
    pub fn is_extend(&self) -> bool {
        self.extend_video.is_some() || self.extend_video_path.is_some()
    }

    /// Pixel-frame overlap the continuation conditions on, defaulting to the
    /// family's own chain motion tail so extend and sequence seams behave
    /// identically — 17 latent-carryover frames on LTX-2, the one seeded frame
    /// on wan.
    pub fn effective_extend_overlap_frames_for_family(&self, family: Option<&str>) -> u32 {
        self.extend_overlap_frames
            .unwrap_or_else(|| crate::validation::default_extend_overlap_frames_for_family(family))
    }

    /// [`Self::effective_extend_overlap_frames_for_family`] for callers that
    /// hold no family hint, resolving it from the request's own model through
    /// the manifest. An unclassifiable model (an installed `cv:` / `hf:` id)
    /// falls back to `DEFAULT_EXTEND_OVERLAP_FRAMES`, so a caller that knows
    /// the resolved family — admission and every engine — should pass it.
    pub fn effective_extend_overlap_frames(&self) -> u32 {
        self.effective_extend_overlap_frames_for_family(
            crate::manifest::find_manifest(&self.model).map(|manifest| manifest.family.as_str()),
        )
    }

    /// Net-new pixel frames an extend request appends to its source: the
    /// rendered clip minus the leading overlap that reproduces the tail.
    pub fn extend_new_frames(&self) -> Option<u32> {
        self.is_extend()
            .then(|| {
                self.frames
                    .map(|frames| frames.saturating_sub(self.effective_extend_overlap_frames()))
            })
            .flatten()
    }

    /// Fill `output_format` with a family-aware default when the caller did
    /// not supply one.
    ///
    /// - `ltx2` with `enable_audio == Some(true)` → `Mp4` (audio requires mp4)
    /// - `ltx2` (any other case) → `Mp4` (most compatible video container)
    /// - `ltx-video` → `Mp4` (most compatible; engine falls back to APNG when
    ///   the field is non-video, but Mp4 is the right API default)
    /// - `wan` with `frames == Some(1)` → `Png` (a single-frame render is a
    ///   still, not a one-frame video — #798)
    /// - all other families → `Png` (existing image default)
    ///
    /// This is a no-op when `output_format` is already `Some(…)` — explicit
    /// caller choices are always preserved, even invalid ones. Validation that
    /// runs after normalisation will then reject them with a clear error.
    pub fn normalise_output_format(&mut self, family: Option<&str>) -> &mut Self {
        if self.output_format.is_some() {
            return self;
        }
        // An audio-only pipeline has no frames to encode, so the family
        // default (mp4) would be rejected by the validator. Resolve it to the
        // one container that can hold the artifact it actually produces.
        if self.pipeline.is_some_and(Ltx2PipelineMode::is_audio_only) {
            self.output_format = Some(OutputFormat::Wav);
            return self;
        }
        self.output_format = Some(match family {
            Some("wan") if self.frames == Some(1) => OutputFormat::Png,
            Some(family) if family_output_defaults_to_mp4(family) => OutputFormat::Mp4,
            _ => OutputFormat::Png,
        });
        self
    }
}

/// Whether an unset output format on this family defaults to MP4.
///
/// This is the single family-policy authority behind
/// [`GenerateRequest::normalise_output_format`] and the CLI's client-side
/// container default (`mold run` resolves the container before dispatch so
/// local saves and `--output` extensions agree with what the server would
/// pick). The CLI additionally degrades wan/ltx-video to APNG in builds
/// without the `mp4` feature — that build-time concern stays on the CLI side;
/// the family policy itself lives here so the two cannot drift (#806).
pub fn family_output_defaults_to_mp4(family: &str) -> bool {
    matches!(
        family,
        "ltx2" | "ltx-video" | "wan" | "minimax-h3" | "minimax_h3" | "minimaxh3"
    )
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct KeyframeCondition {
    #[schema(example = 0)]
    pub frame: u32,
    #[serde(with = "base64_required")]
    pub image: Vec<u8>,
    /// Display-only provenance label for this keyframe. The engine ignores it;
    /// saved metadata retains only a sanitized name and content digest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, utoipa::ToSchema)]
pub struct TimeRange {
    #[schema(example = 0.0)]
    pub start_seconds: f32,
    #[schema(example = 2.5)]
    pub end_seconds: f32,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS,
)]
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
    LipDub,
    T2a,
}

impl Ltx2PipelineMode {
    /// Every variant, in wire order. The Studio surfaces mirror this list as a
    /// TypeScript string union, and
    /// `ltx2_pipeline_typescript_unions_match_the_wire_contract` pins them to
    /// it — a member that does not deserialize 422s the whole request.
    pub const ALL: [Self; 10] = [
        Self::OneStage,
        Self::TwoStage,
        Self::TwoStageHq,
        Self::Distilled,
        Self::IcLora,
        Self::Keyframe,
        Self::A2Vid,
        Self::Retake,
        Self::LipDub,
        Self::T2a,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OneStage => "one-stage",
            Self::TwoStage => "two-stage",
            Self::TwoStageHq => "two-stage-hq",
            Self::Distilled => "distilled",
            Self::IcLora => "ic-lora",
            Self::Keyframe => "keyframe",
            Self::A2Vid => "a2-vid",
            Self::Retake => "retake",
            Self::LipDub => "lip-dub",
            Self::T2a => "t2a",
        }
    }

    /// Whether this pipeline produces an audio-only artifact rather than
    /// video frames. Audio-only plans skip every spatial stage, so callers
    /// must not reason about `width`/`height`/`spatial_upscale` for them.
    pub const fn is_audio_only(self) -> bool {
        matches!(self, Self::T2a)
    }

    /// Whether this pipeline renders stage 1 at a reduced size and refines the
    /// upsampled result, rather than denoising the requested shape once.
    ///
    /// This is what decides whether a resolution past the trained RoPE span is
    /// reachable at all, so admission (`mold_core::validation`) and the engine
    /// (`ltx2::runtime::pipeline_uses_two_stage_spatial_refinement`) must agree
    /// on it exactly — the engine delegates here rather than restating the set.
    ///
    /// `LipDub` is deliberately absent: it renders through the two-stage
    /// driver but is pinned to the reference clip's resolution, so it never
    /// composes a larger output. `Retake` and `OneStage` denoise once.
    pub const fn refines_spatially(self) -> bool {
        matches!(
            self,
            Self::Distilled
                | Self::TwoStage
                | Self::TwoStageHq
                | Self::IcLora
                | Self::Keyframe
                | Self::A2Vid
        )
    }

    /// Whether this recipe uses classifier-free guidance and therefore
    /// consumes an unconditional (negative-prompt) text context.
    pub const fn uses_cfg(self) -> bool {
        !matches!(
            self,
            Self::Distilled | Self::IcLora | Self::Retake | Self::LipDub
        )
    }
}

/// User-facing guidance controls for a resolved model/pipeline recipe.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS)]
pub struct GuidanceCapabilities {
    /// Whether the primary guidance scale changes the render.
    pub adjustable: bool,
    /// Whether the recipe encodes and uses a negative prompt for CFG.
    pub supports_negative_prompt: bool,
    /// Effective fixed scale when `adjustable` is false.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fixed_scale: Option<f64>,
}

impl GuidanceCapabilities {
    pub const ADJUSTABLE_CFG: Self = Self {
        adjustable: true,
        supports_negative_prompt: true,
        fixed_scale: None,
    };
    pub const FIXED_ONE: Self = Self {
        adjustable: false,
        supports_negative_prompt: false,
        fixed_scale: Some(1.0),
    };
    pub const ADJUSTABLE_NO_NEGATIVE: Self = Self {
        adjustable: true,
        supports_negative_prompt: false,
        fixed_scale: None,
    };

    /// Resolve the guidance scale a request should carry. A recipe that pins
    /// the scale owns the default, because a per-model default is optional and
    /// the global fallback (3.5) does not survive the recipe's own validation.
    /// An explicitly requested value is always preserved, so a conflict is
    /// still reported to the caller instead of being silently rewritten.
    pub fn resolve_scale(&self, requested: Option<f64>, model_default: f64) -> f64 {
        requested.or(self.fixed_scale).unwrap_or(model_default)
    }

    /// Resolve the effective guidance contract for the selected recipe.
    /// `None` means the model's default pipeline.
    pub fn for_recipe(family: &str, model: &str, pipeline: Option<Ltx2PipelineMode>) -> Self {
        match family.trim().to_ascii_lowercase().as_str() {
            "ltx-video" => {
                if model.to_ascii_lowercase().contains("distilled") {
                    Self::FIXED_ONE
                } else {
                    Self::ADJUSTABLE_CFG
                }
            }
            "ltx2" | "ltx-2" => {
                let uses_cfg = pipeline
                    .map(Ltx2PipelineMode::uses_cfg)
                    .unwrap_or_else(|| !model.to_ascii_lowercase().contains("distilled"));
                if uses_cfg {
                    Self::ADJUSTABLE_CFG
                } else {
                    Self::FIXED_ONE
                }
            }
            "flux" | "flux2" | "flux.2" | "flux-2" | "z-image" | "qwen-image" | "qwen_image" => {
                Self::ADJUSTABLE_NO_NEGATIVE
            }
            _ => Self::ADJUSTABLE_CFG,
        }
    }
}

impl std::fmt::Display for Ltx2PipelineMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
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

/// Advanced overrides for the LTX-2 multimodal guider.
///
/// LTX-2 pipelines pin spatiotemporal-guidance (STG), CFG-rescale, modality,
/// and guidance-skip constants per (pipeline, stage). Every field here is
/// optional and replaces exactly one of those constants for one request; an
/// absent field keeps the pipeline default, so a request without overrides is
/// byte-identical to one made before this contract existed.
///
/// Overrides apply only to guiders a pipeline already runs. A pipeline that
/// deliberately disables a guider entirely (LTX-2's `a2-vid` audio guider)
/// stays disabled — overrides tune guidance, they never switch it on.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct Ltx2GuidanceOverrides {
    /// Spatiotemporal guidance scale. `0.0` disables the perturbed pass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = 1.0)]
    pub stg_scale: Option<f64>,
    /// Transformer block indices perturbed for STG. Must be within the
    /// selected checkpoint's transformer depth.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stg_blocks: Option<Vec<u32>>,
    /// CFG-rescale factor (`0.0` = no rescale, `1.0` = full std matching).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = 0.7)]
    pub rescale_scale: Option<f64>,
    /// Cross-modality (audio ↔ video) guidance scale. `1.0` disables the
    /// isolated-modality pass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = 3.0)]
    pub modality_scale: Option<f64>,
    /// Guidance skip stride. `0` applies guidance on every step; `n` applies
    /// it on every `n + 1`-th step and takes the conditional prediction
    /// otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = 0)]
    pub skip_step: Option<u32>,
}

impl Ltx2GuidanceOverrides {
    /// Maximum accepted STG / modality scale. Well past any useful setting;
    /// the point is to reject nonsense (and NaN) rather than to tune.
    pub const MAX_SCALE: f64 = 10.0;
    /// Maximum accepted guidance skip stride.
    pub const MAX_SKIP_STEP: u32 = 8;

    /// True when no field is set — the request carries no override at all and
    /// callers can drop it rather than serialize an empty object.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// `Some(self)` when at least one field is set, `None` otherwise. Clients
    /// building a request from a form use this so untouched controls never
    /// widen the request payload.
    pub fn into_option(self) -> Option<Self> {
        (!self.is_empty()).then_some(self)
    }
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
    /// Which Wan 2.2 A14B expert this adapter belongs to.
    ///
    /// The community publishes A14B adapters as high/low pairs distilled
    /// together and explicitly not interchangeable — applying a high-noise
    /// adapter to the low-noise expert degrades the render rather than failing.
    /// Absent keeps the historical apply-to-both behavior, which is correct for
    /// a genuinely unpaired adapter and for every single-expert family.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert: Option<LoraExpert>,
}

/// The A14B expert an adapter is bound to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "kebab-case")]
pub enum LoraExpert {
    /// The early, structural half of the schedule.
    High,
    /// The late, detail half.
    Low,
}

impl LoraExpert {
    /// Infer the expert from the conventions publishers actually use.
    ///
    /// Delegates to [`crate::wan_expert_marker`], the same classifier the
    /// catalog uses to pair Civitai's separately-published A14B experts
    /// (#784). Sharing it is the point: a file the catalog reads as the
    /// high-noise half must route the same way when the user passes it as an
    /// adapter, and a second tokenizer here drifted immediately — it missed
    /// the published digit-glued `…A14BHIGH` convention that one already
    /// handles.
    ///
    /// A name carrying both markers is a bundle, not one expert, and stays
    /// unresolved. An explicit field always wins; this is the fallback, and a
    /// caller that uses it must disclose that it did.
    pub fn infer_from_filename(name: &str) -> Option<Self> {
        let stem = std::path::Path::new(name)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or(name);
        match crate::wan_expert_marker::classify_name(stem) {
            crate::wan_expert_marker::NameMarker::High => Some(Self::High),
            crate::wan_expert_marker::NameMarker::Low => Some(Self::Low),
            crate::wan_expert_marker::NameMarker::NoMarker
            | crate::wan_expert_marker::NameMarker::Ambiguous => None,
        }
    }
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
    /// Audio-only output data (additive). Present only for audio-only
    /// pipelines — currently LTX-2 text-to-audio. Deliberately a separate
    /// slot from `video`: every existing consumer reads a populated `video`
    /// as "this response is a video", so reshaping `VideoData` around a
    /// frameless artifact would mis-render it everywhere at once.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioData>,
    #[schema(example = 1234)]
    pub generation_time_ms: u64,
    #[schema(example = "flux-schnell:q8")]
    pub model: String,
    #[schema(example = 42)]
    pub seed_used: u64,
    /// Which GPU ordinal handled this request (multi-GPU only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<usize>,
    /// Advisories the server attached to this response: the request was
    /// accepted and the print rendered, but something the caller asked for
    /// was adjusted or dropped (a lip-dub retiming, a filing the host could
    /// not apply).
    ///
    /// Populated by [`crate::MoldClient`] from the `x-mold-request-warning`
    /// response header, which is why it is empty on the server side — the
    /// route builds that header from its own `RequestWarnings` instead.
    /// Additive; empty on every response that carried no advisory.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub request_warnings: Vec<String>,
}

/// LTX-2 still-image conditioning preprocessing as actually executed —
/// the executed-recipe mirror of `VideoData::pipeline`. `codec` names the
/// real round-trip implementation (e.g. `"openh264-cqp33"`, deliberately
/// not a CRF claim) and `fit_policy` the resize/crop contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct Ltx2SourcePreprocessing {
    pub profile: crate::ltx2_preprocess::Ltx2ImagePreprocessingProfile,
    pub codec: String,
    pub fit_policy: String,
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
    /// Runtime-resolved LTX-2 pipeline. This records implicit/default choices
    /// as executed; absent for non-LTX-2 video engines and older servers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pipeline: Option<Ltx2PipelineMode>,
    /// SHA-256 of the exact runtime pipeline provenance that produced this
    /// video. Present for authenticated H3 outputs; absent for older servers
    /// and pipelines without a terminal provenance contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pipeline_provenance_sha256: Option<String>,
    /// LTX-2 source-image conditioning preprocessing actually applied.
    /// Absent for T2V, other families, and older servers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_preprocessing: Option<Ltx2SourcePreprocessing>,
    /// Which attention arithmetic the LTX-2 transformer actually ran
    /// (`ltx2-bf16-math`, `ltx2-bf16-flash`, `ltx2-f32-chunked`,
    /// `ltx2-metal-sdpa`; the literals live in
    /// `mold_inference::ltx2::provenance`). Output-changing, so it is
    /// recorded rather than inferred. Absent for other families, for a
    /// synthetic placeholder render, and for older servers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_path: Option<String>,
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

/// Audio-only output from an audio-generating pipeline (LTX-2 text-to-audio).
///
/// `thumbnail` is a rendered waveform PNG produced where the samples already
/// are, so every gallery surface — web, desktop, iPhone and the TUI — draws a
/// legible tile from the one artifact instead of each inventing its own glyph.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct AudioData {
    /// Encoded audio bytes in the requested format (currently WAV).
    pub data: Vec<u8>,
    /// Output format. Always an [`OutputFormat::is_audio`] variant.
    pub format: OutputFormat,
    /// Sample rate in Hz, as produced by the vocoder.
    #[schema(example = 48000)]
    pub sample_rate: u32,
    /// Channel count in the encoded stream.
    #[schema(example = 2)]
    pub channels: u32,
    /// Total encoded duration in milliseconds.
    #[schema(example = 5040)]
    pub duration_ms: u64,
    /// Rendered waveform PNG for gallery grids and the TUI cell.
    pub thumbnail: Vec<u8>,
    /// Raster size of `thumbnail`. Audio has no dimensions of its own, so
    /// this is what gallery rows record and what grids lay the tile out with.
    #[schema(example = 640)]
    pub thumbnail_width: u32,
    #[schema(example = 360)]
    pub thumbnail_height: u32,
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

/// Byte-free provenance for a keyframe conditioning input. Additive on
/// `OutputMetadata`; legacy rows omit the field entirely.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KeyframeMetadata {
    pub frame: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub sha256: String,
}

/// Authoring surface that produced a saved output. This is deliberately
/// independent of the execution route: a One shot may be auto-expanded into
/// several internal chain stages, but Reuse settings must still return to the
/// One shot form rather than exposing those implementation-detail stages.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, utoipa::ToSchema)]
#[serde(rename_all = "kebab-case")]
pub enum GenerationOutputMode {
    OneShot,
    Sequence,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OutputMetadata {
    #[serde(deserialize_with = "crate::prompt_text::deserialize_prompt")]
    pub prompt: String,
    /// User-authored print title as it was at creation. Embedded so mirrors
    /// and imports carry it; the gallery row (`generations.title`) is the
    /// editable authority once the print exists. Additive; absent on every
    /// untitled and legacy print.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Tags the print was filed under at creation, exactly as applied. The
    /// gallery row's `generation_tags` links are the editable authority once
    /// the print exists; this embedded copy is what lets a mirror or a
    /// reconcile-from-disk recover the filing. Additive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    /// Display name of the collection the print was filed into at creation,
    /// as applied — never the requested id, and never a name the host did not
    /// actually resolve. Additive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collection: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::prompt_text::deserialize_optional_prompt"
    )]
    pub negative_prompt: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::prompt_text::deserialize_optional_prompt"
    )]
    pub original_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_transform: Option<PromptTransformProvenance>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_index: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_count: Option<u32>,
    /// User-facing output mode. Additive so legacy rows can fall back to
    /// durable chain provenance without conflating automatic chaining with an
    /// authored sequence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_mode: Option<GenerationOutputMode>,
    /// Queue job that produced this print. Additive; absent on every print
    /// written before the durable queue existed and on client-side saves.
    ///
    /// This is the replay idempotence key: output filenames are wall-clock, so
    /// once a job's row survives a restart nothing else can tell a replayed
    /// render from the original. Boot replay looks this id up in
    /// `generations.metadata_json` and drops the row instead of re-rendering.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
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
    /// Provenance label of the identity reference (client-supplied filename)
    /// — present only when the request carried an `id_image` and a name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id_image_name: Option<String>,
    /// SHA-256 (hex) of the exact `id_image` bytes used. Names and hashes
    /// only, never the face payload itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id_image_sha256: Option<String>,
    /// Effective identity-conditioning strength, recorded only when the print
    /// actually carried an identity reference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id_weight: Option<f64>,
    /// Effective first identity-conditioned denoise step, recorded only when
    /// the print actually carried an identity reference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id_start_step: Option<u32>,
    /// Provenance labels of every identity reference, in request order —
    /// present only for the multi-photograph form. A single-photograph print
    /// records `id_image_name` exactly as it always did and leaves this absent,
    /// so its metadata is byte-identical to a pre-`id_images` build's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id_image_names: Option<Vec<String>>,
    /// SHA-256 (hex) of each identity reference, in request order — present
    /// only for the multi-photograph form. Names and hashes only, never the
    /// face payloads themselves.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id_image_sha256s: Option<Vec<String>>,
    /// Effective true-CFG scale, recorded only when the print actually ran the
    /// negative branch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub true_cfg: Option<f64>,
    /// Effective first true-CFG denoise step, recorded only when the print
    /// actually ran the negative branch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg_start_step: Option<u32>,
    /// Opaque client-shaped source-fit (crop/pad) policy provenance. The
    /// engine never reads it — fitting happens client-side before the bytes
    /// ship — but recording it verbatim lets Reuse settings and running-job
    /// selection restore the crop controls exactly as they were.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_fit: Option<serde_json::Value>,
    /// SHA-256 (hex) of each Qwen Image Edit input, in request order. Clients
    /// can restore locally available inputs without persisting image payloads
    /// or private filesystem paths in gallery metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edit_image_sha256s: Option<Vec<String>>,
    /// Ordered, redacted H3 reference provenance. Contains only display-safe
    /// labels, content digests, and probed media facts: never payload bytes,
    /// upload handles, API keys, or server/client filesystem paths.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub references: Option<Vec<GenerationReferenceMetadata>>,
    /// Ordered, byte-free keyframe provenance. FL2VA clients use the entry at
    /// `frames - 1` to restore the closing-frame reattachment requirement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keyframes: Option<Vec<KeyframeMetadata>>,
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
    /// Server-local path of the video this print continues, when it was
    /// produced by an extend request. Inline `extend_video` bytes are
    /// deliberately not recorded — metadata rides inside the output file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extend_video_path: Option<String>,
    /// Pixel-frame overlap used to condition the continuation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extend_overlap_frames: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pipeline: Option<Ltx2PipelineMode>,
    /// Whether `pipeline` was explicitly present on the authored request.
    /// `pipeline` itself records the runtime-resolved mode, so reuse clients
    /// need this additive provenance bit to avoid promoting a default into an
    /// override. Absent on legacy metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pipeline_requested: Option<bool>,
    /// Whether the authored request deliberately omitted `frames` so an
    /// LTX-2.5 duration head could choose the clip length. `frames` is later
    /// replaced with the realized output shape, so this additive bit keeps
    /// request provenance available to Library reuse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_prediction_requested: Option<bool>,
    /// Persisted terminal runtime provenance for pipelines that expose an
    /// exact additive SHA-256 identity. Absent for legacy and other outputs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pipeline_provenance_sha256: Option<String>,
    /// LTX-2 source-image conditioning preprocessing actually applied
    /// (recorded from the response, like `pipeline`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_preprocessing: Option<Ltx2SourcePreprocessing>,
    /// LTX-2 attention arithmetic actually run (recorded from the response,
    /// like `pipeline`; see `VideoData::attention_path`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ic_lora_control: Option<String>,
    /// Where the HDR EXR sidecar was written. The gallery holds the tonemapped
    /// video, so without this the sequence is unfindable from the Library.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hdr_exr_dir: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub hdr_exr_full_float: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retake_range: Option<TimeRange>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spatial_upscale: Option<Ltx2SpatialUpscale>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal_upscale: Option<Ltx2TemporalUpscale>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guidance_overrides: Option<Ltx2GuidanceOverrides>,
    /// Wan flow shift as requested (#782). Absent means the per-tier pipeline
    /// default applied; Reuse settings restoring an absent field reproduces
    /// the render as long as the defaults are unchanged, which is the same
    /// contract every other absent knob keeps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_shift: Option<f64>,
    /// Wan Lightning distill strength overrides (#795). Absent = 1.0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distill_strength_high: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distill_strength_low: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frames: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
    /// Durable chain-job id this output was finalized from (additive;
    /// absent for single generations, the ephemeral shim, and legacy rows).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chain_job_id: Option<String>,
    /// Structured multi-clip execution provenance for chain outputs
    /// (additive) — both authored sequences and automatically chained One
    /// shots carry this. `output_mode`, not this field, selects the Reuse
    /// settings authoring surface.
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
        let references = req.references.as_ref().and_then(|references| {
            (!references.is_empty()).then(|| {
                let target_frames = req
                    .frames
                    .unwrap_or(crate::minimax_h3::REVIEWED_COMPACT_FRAMES);
                references
                    .iter()
                    .enumerate()
                    .map(|(index, reference)| {
                        reference.redacted_metadata_lossless_for_target(index, target_frames)
                    })
                    .collect::<Vec<_>>()
            })
        });
        let keyframes = req.keyframes.as_ref().and_then(|keyframes| {
            (!keyframes.is_empty()).then(|| {
                keyframes
                    .iter()
                    .map(|keyframe| {
                        use sha2::{Digest, Sha256};
                        let mut hasher = Sha256::new();
                        hasher.update(&keyframe.image);
                        KeyframeMetadata {
                            frame: keyframe.frame,
                            name: redacted_media_name(keyframe.name.as_deref()),
                            sha256: format!("{:x}", hasher.finalize()),
                        }
                    })
                    .collect::<Vec<_>>()
            })
        });
        Self {
            prompt: req.prompt.clone(),
            title: req.title.clone(),
            // Creation-time filing rides through as requested. Admission has
            // already normalized the tags and resolved the collection ref to
            // a concrete name, so what lands here is what will be applied.
            tags: req.tags.as_ref().filter(|tags| !tags.is_empty()).cloned(),
            collection: req
                .collection
                .as_ref()
                .and_then(|reference| reference.name.as_deref())
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .map(str::to_owned),
            negative_prompt: req.negative_prompt.clone(),
            original_prompt: req.original_prompt.clone(),
            prompt_transform: req.prompt_transform.clone(),
            batch_id: req.batch_id.clone(),
            batch_index: req.batch_index,
            batch_count: req.batch_count,
            output_mode: Some(GenerationOutputMode::OneShot),
            // Stamped by the worker immediately before the save; the request
            // does not know which queue job is carrying it.
            job_id: None,
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
            // Identity provenance is recorded only when the print actually
            // carried a face reference; a bare knob on an ordinary render
            // would read as conditioning that never happened.
            id_image_name: req
                .id_image
                .as_ref()
                .and_then(|_| req.id_image_name.clone()),
            id_image_sha256: req.id_image.as_ref().map(|bytes| {
                use sha2::{Digest, Sha256};
                let mut hasher = Sha256::new();
                hasher.update(bytes);
                format!("{:x}", hasher.finalize())
            }),
            id_weight: crate::identity::request_carries_identity_photo(req)
                .then(|| crate::identity::effective_id_weight(req)),
            id_start_step: crate::identity::request_carries_identity_photo(req)
                .then(|| crate::identity::effective_id_start_step(req)),
            // The plural provenance is recorded ONLY for the multi form, so a
            // single-photograph print's metadata stays byte-identical to what
            // every build before `id_images` wrote.
            id_image_names: req
                .id_images
                .as_ref()
                .filter(|images| !images.is_empty())
                .and_then(|_| req.id_image_names.clone()),
            id_image_sha256s: req
                .id_images
                .as_ref()
                .filter(|images| !images.is_empty())
                .map(|images| {
                    images
                        .iter()
                        .map(|bytes| crate::identity::id_image_sha256(bytes))
                        .collect()
                }),
            true_cfg: crate::identity::request_uses_true_cfg(req)
                .then(|| crate::identity::effective_true_cfg(req)),
            cfg_start_step: crate::identity::request_uses_true_cfg(req)
                .then(|| crate::identity::effective_cfg_start_step(req)),
            source_fit: req.source_fit.clone(),
            edit_image_sha256s: req.edit_images.as_ref().and_then(|images| {
                (!images.is_empty()).then(|| {
                    images
                        .iter()
                        .map(|bytes| {
                            use sha2::{Digest, Sha256};
                            let mut hasher = Sha256::new();
                            hasher.update(bytes);
                            format!("{:x}", hasher.finalize())
                        })
                        .collect()
                })
            }),
            references,
            keyframes,
            scheduler,
            output_format: req.output_format,
            cfg_plus: req.cfg_plus,
            lora,
            lora_scale,
            loras,
            control_model: req.control_model.clone(),
            control_scale: (req.control_image.is_some() || req.control_model.is_some())
                .then_some(req.control_scale),
            sample_shift: req.sample_shift,
            distill_strength_high: req.distill_strength_high,
            distill_strength_low: req.distill_strength_low,
            upscale_model: req.upscale_model.clone(),
            gif_preview: req.gif_preview.then_some(true),
            enable_audio: req.enable_audio,
            audio_file_path: req.audio_file_path.clone(),
            source_video_path: req.source_video_path.clone(),
            extend_video_path: req.extend_video_path.clone(),
            // Only meaningful when this print actually continued something;
            // recording a bare overlap on an ordinary render would read as
            // provenance that does not exist.
            extend_overlap_frames: (req.extend_video.is_some() || req.extend_video_path.is_some())
                .then_some(req.effective_extend_overlap_frames()),
            pipeline: req.pipeline,
            pipeline_requested: Some(req.pipeline.is_some()),
            duration_prediction_requested: req
                .model
                .starts_with("ltx-2.5")
                .then_some(req.frames.is_none()),
            pipeline_provenance_sha256: None,
            source_preprocessing: None,
            attention_path: None,
            ic_lora_control: req.ic_lora_control.clone(),
            hdr_exr_dir: req.hdr_exr_dir.clone(),
            hdr_exr_full_float: req.hdr_exr_full_float,
            retake_range: req.retake_range.clone(),
            spatial_upscale: req.spatial_upscale,
            temporal_upscale: req.temporal_upscale,
            guidance_overrides: req.guidance_overrides.clone(),
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

    /// Record the video shape and runtime pipeline that actually completed.
    /// The pipeline cannot be derived reliably from the request because LTX-2
    /// resolves implicit/default modes against the loaded checkpoint assets.
    pub fn apply_video_output(&mut self, video: &VideoData) {
        self.apply_output_dimensions(video.width, video.height);
        self.frames = Some(video.frames);
        self.fps = Some(video.fps);
        if let Some(pipeline) = video.pipeline {
            self.pipeline = Some(pipeline);
        }
        self.pipeline_provenance_sha256 = video.pipeline_provenance_sha256.clone();
        if let Some(preprocessing) = &video.source_preprocessing {
            self.source_preprocessing = Some(preprocessing.clone());
        }
        if let Some(attention_path) = &video.attention_path {
            self.attention_path = Some(attention_path.clone());
        }
    }
}

#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS,
)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    #[default]
    Png,
    Jpeg,
    Gif,
    Apng,
    Webp,
    Mp4,
    Wav,
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
            OutputFormat::Wav => "wav",
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
            OutputFormat::Wav => "audio/wav",
        }
    }

    /// Whether this format is a video/animation format.
    pub fn is_video(&self) -> bool {
        matches!(
            self,
            OutputFormat::Gif | OutputFormat::Apng | OutputFormat::Webp | OutputFormat::Mp4
        )
    }

    /// Whether this format carries audio samples and no raster frames.
    ///
    /// Audio-only artifacts are neither images nor videos: gallery grids
    /// render their sidecar waveform thumbnail, and every "is this a video"
    /// branch must stay `false` so nothing tries to seek frames out of them.
    pub fn is_audio(&self) -> bool {
        matches!(self, OutputFormat::Wav)
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
            "wav" => Ok(OutputFormat::Wav),
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
pub struct RecommendedDimensions {
    pub width: u32,
    pub height: u32,
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
    /// Minimum requestable video frame count (additive; absent when the
    /// historical one-frame floor applies).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = 124)]
    pub min_frames: Option<u32>,
    /// Maximum frames a single request may ask for at `default_fps`
    /// (additive; absent for image models). For families whose ceiling is a
    /// duration — see `max_runtime_seconds` — this scalar moves with fps, so
    /// clients that let the user change fps should recompute it from
    /// `max_runtime_seconds` rather than treat it as fixed. The server
    /// validator remains authoritative.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = 484)]
    pub max_frames: Option<u32>,
    /// Single-request runtime ceiling in seconds, when the family's real limit
    /// is a duration rather than a frame count (additive; currently LTX-2 /
    /// LTX-2.3, whose temporal RoPE budget is expressed in seconds). Clients
    /// derive `max_frames` at an arbitrary fps as
    /// `max_runtime_seconds * fps + 4`, clamped to `max_frames_absolute`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = 20)]
    pub max_runtime_seconds: Option<u32>,
    /// Hard frame ceiling that applies regardless of fps (additive). Present
    /// alongside `max_runtime_seconds`; a resource guard, not a model limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = 604)]
    pub max_frames_absolute: Option<u32>,
    /// Step of the valid frame grid (additive; absent for image models).
    /// Combine with `frame_offset`, whose backward-compatible default is 1:
    /// valid counts are `k * frame_step + frame_offset`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = 8)]
    pub frame_step: Option<u32>,
    /// Offset of the valid frame grid. Omitted by older servers and families
    /// whose grid is `k * frame_step + 1`; MiniMax H3 advertises 5.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = 5)]
    pub frame_offset: Option<u32>,
    /// Server-authoritative total-pixel ceiling for generation requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = 1800000)]
    pub max_pixels: Option<u64>,
    /// Server-authoritative per-axis ceiling, independent of `max_pixels`
    /// (additive; absent where the family has no axis limit).
    ///
    /// LTX-2 normalizes RoPE pixel positions by the trained span, so a long
    /// edge past it is out of distribution however small the frame is. The
    /// value is per model, not per family: a checkpoint that ships the spatial
    /// upsampler composes stage 1 at half size plus a tiled stage-2
    /// refinement and reaches twice the span, one that does not cannot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = 2048)]
    pub max_axis_pixels: Option<u32>,
    /// Tuned default negative prompt the engine conditions on when a request
    /// omits `negative_prompt` entirely (additive; wan today, whose
    /// checkpoints were trained against a specific long Chinese negative).
    /// This is the absence fallback, never a floor: an explicit empty string
    /// in the request stays a real empty uncond. Clients prefill their
    /// Negative control with it, keep an untouched field absent on the wire,
    /// and submit `""` when the user clears it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_negative_prompt: Option<String>,
    /// Runnable, family-appropriate buckets used by every Studio surface.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recommended_dimensions: Vec<RecommendedDimensions>,
    /// Required dimension grid for this family.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = 16)]
    pub dimension_alignment: Option<u32>,
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
    /// Whether omitting `GenerateRequest.frames` asks this concrete model to
    /// run its qualified prompt-conditioned duration head. Absent on older
    /// servers and false for models without that exact component contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_duration_prediction: Option<bool>,
    /// Whether every component required by this concrete split pack is
    /// present and header-qualified on this host. LTX-2.5 publishes this even
    /// for incomplete rows so automatic routing can refuse them before queueing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_ready: Option<bool>,
    /// Human-readable reason paired with `runtime_ready == false`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_readiness_error: Option<String>,
    /// Whether this concrete model can accept a face-identity reference
    /// (`GenerateRequest.id_image`). Derived from the same generation-profile
    /// authority as `capabilities.supports_identity`, never a second
    /// predicate: it is true only for an identity-qualified checkpoint on a
    /// binary that actually links the identity adapter. `None` on servers
    /// that predate identity conditioning, which clients read as "no".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_identity: Option<bool>,
    /// Whether this build can execute this model at all.
    ///
    /// `false` marks a model that downloads, verifies, inventories, and
    /// removes normally while having no engine arm — the pinned
    /// `official-bf16` qualification references and the pruned NVFP4 compact
    /// layout. `None` on servers that predate the field and for every family
    /// whose rows are all runnable, which clients read as "runnable": the
    /// browser contract is `runtime_available !== false`, so an older server
    /// keeps behaving exactly as before.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_available: Option<bool>,
    /// One sentence naming *why* `runtime_available` is false.
    ///
    /// Present exactly when `runtime_available` is `Some(false)`, and always
    /// the same sentence the generation refusal carries — both come from
    /// `minimax_h3::RuntimeUnavailableReason`. It exists so a client can warn
    /// before a 21-42 GB pull instead of after it (#1276): "no engine arm for
    /// this weight layout", "Ref2VA execution is not available in any
    /// released build", and "this build was compiled without the H3 engine"
    /// are three different answers with three different remedies. `None` on
    /// servers that predate the field, which clients render as a bare
    /// download-only note exactly as they did before.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_unavailable_reason: Option<String>,
    /// Whether this model can continue an existing video in one request
    /// (`GenerateRequest.extend_video`). `None` on servers that predate
    /// continuation support, which clients must read as "no" — offering the
    /// control would only produce a rejected request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_extend: Option<bool>,
    /// Pixel-frame overlap applied when a continuation omits
    /// `extend_overlap_frames`. Present whenever `supports_extend` is true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extend_default_overlap_frames: Option<u32>,
    /// Whether this model's effective runtime pipeline can render sequence
    /// clips. `None` on servers that predate per-model advertisement; clients
    /// fall back to their own conservative name heuristic, which is the right
    /// answer against a server that would still reject the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_sequence: Option<bool>,
    /// Guidance controls for this model's default resolved recipe. Clients
    /// refine this with an explicitly selected pipeline when applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guidance_capabilities: Option<GuidanceCapabilities>,
    /// Per-model source-image conditioning contract (#772). Wan checkpoints
    /// split three ways — T2V-only, I2V-required, I2V-optional — which no
    /// family-level fact can express. `None` on older servers; clients fall
    /// back to their family heuristics, which is the compatible answer
    /// against a server that enforces nothing at admission.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_image: Option<SourceImageCapability>,
    /// Complete, versioned generation-control contract for this concrete
    /// model and every selectable recipe. New clients use this instead of
    /// reconstructing policy from family names and legacy flattened fields.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_profile: Option<crate::generation_profile::GenerationProfileSet>,
}

/// How a model relates to a conditioning source image (#772).
///
/// Derived from checkpoint structure — the engine's own conditioning
/// classification — never from model names.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS,
)]
#[serde(rename_all = "kebab-case")]
pub enum SourceImageCapability {
    /// The checkpoint has no image-conditioning input; a supplied source is
    /// rejected at admission.
    Unsupported,
    /// The checkpoint accepts but does not require a source image.
    Optional,
    /// The checkpoint cannot generate without a source image; admission
    /// rejects a request lacking one.
    Required,
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
        assert_eq!(parsed.min_frames, None);
        assert_eq!(parsed.max_frames, None);
        assert_eq!(parsed.frame_step, None);
        assert_eq!(parsed.frame_offset, None);
        assert_eq!(parsed.max_pixels, None);
        assert!(parsed.recommended_dimensions.is_empty());
        assert_eq!(parsed.dimension_alignment, None);

        let out = serde_json::to_value(&parsed).unwrap();
        assert!(out.get("default_frames").is_none());
        assert!(out.get("default_fps").is_none());
        assert!(out.get("min_frames").is_none());
        assert!(out.get("max_frames").is_none());
        assert!(out.get("frame_step").is_none());
        assert!(out.get("frame_offset").is_none());
        assert!(out.get("max_pixels").is_none());
        assert!(out.get("recommended_dimensions").is_none());
        assert!(out.get("dimension_alignment").is_none());

        let video = ModelDefaults {
            default_frames: Some(97),
            default_fps: Some(24),
            min_frames: Some(1),
            max_frames: Some(484),
            max_runtime_seconds: Some(20),
            frame_step: Some(8),
            frame_offset: Some(1),
            ..parsed
        };
        let out = serde_json::to_value(&video).unwrap();
        assert_eq!(out["default_frames"], 97);
        assert_eq!(out["default_fps"], 24);
        assert_eq!(out["min_frames"], 1);
        assert_eq!(out["max_frames"], 484);
        assert_eq!(out["max_runtime_seconds"], 20);
        assert_eq!(out["frame_step"], 8);
        assert_eq!(out["frame_offset"], 1);

        let back: ModelDefaults = serde_json::from_value(out).unwrap();
        assert_eq!(back.default_frames, Some(97));
        assert_eq!(back.default_fps, Some(24));
        assert_eq!(back.min_frames, Some(1));
        assert_eq!(back.max_frames, Some(484));
        assert_eq!(back.max_runtime_seconds, Some(20));
        assert_eq!(back.frame_offset, Some(1));
        assert_eq!(back.frame_step, Some(8));
    }

    /// `default_negative_prompt` is additive: absent on the old wire shape,
    /// omitted from serialized output when None (image families and older
    /// servers), and round-tripping the tuned wan value verbatim.
    #[test]
    fn model_defaults_default_negative_prompt_is_additive() {
        let legacy = r#"{
            "default_steps": 30,
            "default_guidance": 6.0,
            "default_width": 832,
            "default_height": 480,
            "description": "wan"
        }"#;
        let parsed: ModelDefaults = serde_json::from_str(legacy).unwrap();
        assert_eq!(parsed.default_negative_prompt, None);
        let out = serde_json::to_value(&parsed).unwrap();
        assert!(out.get("default_negative_prompt").is_none());

        let wan = ModelDefaults {
            default_negative_prompt: Some(crate::manifest::WAN_DEFAULT_NEGATIVE_PROMPT.to_string()),
            ..parsed
        };
        let out = serde_json::to_value(&wan).unwrap();
        assert_eq!(
            out["default_negative_prompt"],
            crate::manifest::WAN_DEFAULT_NEGATIVE_PROMPT
        );
        let back: ModelDefaults = serde_json::from_value(out).unwrap();
        assert_eq!(
            back.default_negative_prompt.as_deref(),
            Some(crate::manifest::WAN_DEFAULT_NEGATIVE_PROMPT)
        );
    }
}

#[cfg(test)]
mod model_display_name_tests {
    use super::{ModelDefaults, ModelInfo, ModelInfoExtended};

    fn model(name: &str, display_name: Option<&str>, description: &str) -> ModelInfoExtended {
        ModelInfoExtended {
            supports_duration_prediction: None,
            runtime_ready: None,
            runtime_readiness_error: None,
            runtime_available: None,
            runtime_unavailable_reason: None,
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
            supports_identity: None,
            supports_extend: None,
            supports_sequence: None,
            extend_default_overlap_frames: None,
            guidance_capabilities: None,
            source_image: None,
            generation_profile: None,
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

/// Why restart-safe encrypted queue media is unavailable, for an operator
/// rather than a client feature check.
///
/// [`DurableMediaCapabilities`] answers "may I submit a request whose replay
/// depends on captured bytes"; this answers "why can't I". The two are
/// deliberately separate surfaces: the capability is a contract clients
/// branch on, while these reasons are free prose that names host paths and
/// therefore only ever travels on the authenticated `/api/status`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct DurableMediaStatus {
    /// True exactly when [`ServerCapabilities::durable_media`] is present.
    pub available: bool,
    /// Empty while available. Each entry is one operator-actionable reason,
    /// retained for the life of the process so a startup log that has aged
    /// out is not the only record.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasons: Vec<String>,
}

/// The subsystem name `/health` reports for [`DurableMediaStatus`].
pub const HEALTH_SUBSYSTEM_DURABLE_MEDIA: &str = "durable_media";

/// Whether an otherwise serving process has a subsystem switched off.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum HealthState {
    #[default]
    Ok,
    Degraded,
}

/// Body of `GET /health`.
///
/// `/health` is auth-exempt, so it names only which subsystems are degraded —
/// never why. Reasons name host filesystem paths and stay on the
/// authenticated `/api/status` as [`DurableMediaStatus::reasons`].
///
/// The status code stays `200` while degraded: a subsystem being off does not
/// make the process unable to serve, and failing the check would pull a
/// still-working server out of a load balancer. Callers that only read the
/// status code keep their existing behaviour; the body is additive.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct HealthStatus {
    pub status: HealthState,
    /// Subsystem names only, sorted. Empty while `status` is `ok`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub degraded: Vec<String>,
}

impl HealthStatus {
    pub fn from_degraded(degraded: Vec<String>) -> Self {
        Self {
            status: if degraded.is_empty() {
                HealthState::Ok
            } else {
                HealthState::Degraded
            },
            degraded,
        }
    }
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
    /// Total waiting generation load owned by this server (multi-GPU only).
    ///
    /// This includes the durable SQLite backlog plus live non-durable jobs,
    /// with hydrated durable jobs counted once. It is not limited by
    /// `queue_capacity`, which describes only the hydrated runtime window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_depth: Option<usize>,
    /// Maximum hydrated runtime queue window (multi-GPU only).
    ///
    /// The durable generation backlog is not capped by this value.
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
    /// Host-RAM telemetry from the scheduler's admission ledger. Absent on
    /// older servers and wherever that ledger does not run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_memory: Option<HostMemorySnapshot>,
    /// Why restart-safe queue media is on or off. Absent on older servers and
    /// wherever the durable generation queue itself is disabled, which is a
    /// configuration rather than a degradation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub durable_media: Option<DurableMediaStatus>,
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
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
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
    /// Whether this exact request is journalled across restart. Additive and
    /// per-job because some request shapes intentionally remain live-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub durable: Option<bool>,
    /// Why a durable row is parked in the additive `held` state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held_reason: Option<String>,
    /// Durable preparation error for a held row — the same sentence as
    /// [`Self::held_reason`] under the field name the batch child uses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Whether `POST /api/queue/{id}/retry` may safely resume this held row.
    /// A held row that answers `false` needs operator repair, not a retry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retryable: Option<bool>,
    /// Whether the row was resumed from the journal rather than submitted by
    /// a live client.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replayed: Option<bool>,
    /// How many times a worker has claimed this row for execution. Diagnoses
    /// a hold: a job that keeps taking the process down shows its count here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dispatch_attempts: Option<u32>,
    /// Durable batch this row is a child of. Retry needs the whole authority
    /// (instance + batch + client batch + job) and only the instance belongs
    /// to the server, so these three are what make a bare job id retryable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_id: Option<String>,
    /// The client-minted idempotency id of [`Self::batch_id`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_batch_id: Option<String>,
    /// One-based position of this row within its batch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_index: Option<u32>,
}

impl QueueJobEntryWire {
    /// The complete retry authority for this row, or `None` when the row is
    /// not a durable batch child and therefore has no batch to retry against.
    pub fn retry_request(&self, instance_id: &str) -> Option<GenerationRetryRequest> {
        Some(GenerationRetryRequest {
            instance_id: instance_id.to_string(),
            batch_id: self.batch_id.clone()?,
            client_batch_id: self.client_batch_id.clone()?,
            job_id: self.id.clone(),
        })
    }
}

/// One queued job in full, as `GET /api/queue/{id}` answers it: the row plus
/// the planner's own work item for it when the job has been placed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct QueueJobDetailWire {
    pub job: QueueJobEntryWire,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub work_item: Option<QueueWorkItem>,
}

/// Response of `POST /api/queue/pause` and `POST /api/queue/resume`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueuePauseState {
    pub paused: bool,
}

/// Response of `DELETE /api/queue` — how many queued jobs were cancelled.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueueCancelAllResult {
    pub cancelled: usize,
}

/// Confidence attached to a learned scheduler ETA.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum QueueEstimateConfidence {
    #[default]
    Low,
    Medium,
    High,
    Unknown(String),
}

impl QueueEstimateConfidence {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Unknown(value) => value,
        }
    }
}

impl std::fmt::Display for QueueEstimateConfidence {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl Serialize for QueueEstimateConfidence {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for QueueEstimateConfidence {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "low" => Self::Low,
            "medium" => Self::Medium,
            "high" => Self::High,
            _ => Self::Unknown(value),
        })
    }
}

/// Typed reason that a scheduler work unit cannot currently advance.
///
/// `reason` remains on [`QueueWorkItem`] as a backward-compatible display
/// alias. New clients should prefer this field so assignment and blocking
/// causes cannot be confused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueueBlockedReason {
    DeviceDisabled,
    DeviceDraining,
    DeviceStartupExcluded,
    DeviceUnavailable,
    DeviceDegraded,
    HardPinUnavailable,
    BackendUnsupported,
    ModelNotInstalled,
    InsufficientVram,
    InsufficientHostRam,
    AggregateHostRamReserved,
    ExecutionPlanIncompatible,
    DependencyWait,
    /// This job's own dependency preparation is running right now.
    ///
    /// Distinct from `DependencyWait`, which is every other reason a job is
    /// not ready. A preparation that authenticates tens of gigabytes of
    /// weights is minutes long on a spinning-disk model store, and reporting
    /// it as a generic wait is what left an idle GPU looking idle for no
    /// stated reason.
    Preparing,
    WarmWait,
    QueuePaused,
    MaintenanceMode,
    Cancelling,
    NoSchedulableDevice,
    NoIdleDevice,
    LowerPriorityOpening,
    Unknown(String),
}

impl QueueBlockedReason {
    pub fn as_str(&self) -> &str {
        match self {
            Self::DeviceDisabled => "device_disabled",
            Self::DeviceDraining => "device_draining",
            Self::DeviceStartupExcluded => "device_startup_excluded",
            Self::DeviceUnavailable => "device_unavailable",
            Self::DeviceDegraded => "device_degraded",
            Self::HardPinUnavailable => "hard_pin_unavailable",
            Self::BackendUnsupported => "backend_unsupported",
            Self::ModelNotInstalled => "model_not_installed",
            Self::InsufficientVram => "insufficient_vram",
            Self::InsufficientHostRam => "insufficient_host_ram",
            Self::AggregateHostRamReserved => "aggregate_host_ram_reserved",
            Self::ExecutionPlanIncompatible => "execution_plan_incompatible",
            Self::DependencyWait => "dependency_wait",
            Self::Preparing => "preparing",
            Self::WarmWait => "warm_wait",
            Self::QueuePaused => "queue_paused",
            Self::MaintenanceMode => "maintenance_mode",
            Self::Cancelling => "cancelling",
            Self::NoSchedulableDevice => "no_schedulable_device",
            Self::NoIdleDevice => "no_idle_device",
            Self::LowerPriorityOpening => "lower_priority_opening",
            Self::Unknown(value) => value,
        }
    }

    /// Parse the wire identifier. The one place the mapping lives, so
    /// `Deserialize` and every caller holding a legacy `reason` string (which
    /// is typed as a bare `String` on the wire) agree by construction.
    pub fn parse(value: &str) -> Self {
        match value {
            "device_disabled" => Self::DeviceDisabled,
            "device_draining" => Self::DeviceDraining,
            "device_startup_excluded" => Self::DeviceStartupExcluded,
            "device_unavailable" => Self::DeviceUnavailable,
            "device_degraded" => Self::DeviceDegraded,
            "hard_pin_unavailable" => Self::HardPinUnavailable,
            "backend_unsupported" => Self::BackendUnsupported,
            "model_not_installed" => Self::ModelNotInstalled,
            "insufficient_vram" => Self::InsufficientVram,
            "insufficient_host_ram" => Self::InsufficientHostRam,
            "aggregate_host_ram_reserved" => Self::AggregateHostRamReserved,
            "execution_plan_incompatible" => Self::ExecutionPlanIncompatible,
            "dependency_wait" => Self::DependencyWait,
            "preparing" => Self::Preparing,
            "warm_wait" => Self::WarmWait,
            "queue_paused" => Self::QueuePaused,
            "maintenance_mode" => Self::MaintenanceMode,
            "cancelling" => Self::Cancelling,
            "no_schedulable_device" => Self::NoSchedulableDevice,
            "no_idle_device" => Self::NoIdleDevice,
            "lower_priority_opening" => Self::LowerPriorityOpening,
            other => Self::Unknown(other.to_string()),
        }
    }
}

impl Serialize for QueueBlockedReason {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for QueueBlockedReason {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self::parse(&String::deserialize(deserializer)?))
    }
}

/// Truthful scheduler-owned phase for a projected work unit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum QueueActivityPhase {
    #[default]
    Queued,
    Blocked,
    WarmWait,
    Dispatching,
    Active,
    Cpu,
    Unknown(String),
}

impl QueueActivityPhase {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Queued => "queued",
            Self::Blocked => "blocked",
            Self::WarmWait => "warm_wait",
            Self::Dispatching => "dispatching",
            Self::Active => "active",
            Self::Cpu => "cpu",
            Self::Unknown(value) => value,
        }
    }
}

impl std::fmt::Display for QueueActivityPhase {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl Serialize for QueueActivityPhase {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for QueueActivityPhase {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "queued" => Self::Queued,
            "blocked" => Self::Blocked,
            "warm_wait" => Self::WarmWait,
            "dispatching" => Self::Dispatching,
            "active" => Self::Active,
            "cpu" => Self::Cpu,
            _ => Self::Unknown(value),
        })
    }
}

/// What a running dependency preparation is currently working through.
///
/// Additive and best-effort: only preparations that report component progress
/// (today, MiniMax H3's artifact authentication pass) fill it in, and an older
/// server omits it entirely.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct QueuePreparationProgress {
    /// Human-readable component being prepared, supplied by the preparer.
    pub component: String,
    pub bytes_done: u64,
    /// `0` means the pass reports no size — render the component name alone,
    /// never a percentage.
    pub bytes_total: u64,
    /// How long THIS phase has been running, as distinct from
    /// `QueueWorkItem.preparation_elapsed_ms`, which covers the whole
    /// preparation. A minutes-long preparation is a sequence of
    /// authentications, opens, and decodes; the total says it was slow and
    /// this says which part is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase_elapsed_ms: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct QueueBatchPartition {
    /// One-based partition/sibling index.
    pub index: u32,
    pub count: u32,
    /// Number of generation outputs owned by this work unit.
    pub size: u32,
}

/// Public class of scheduler lane used by a planned work item.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum QueuePlannedLaneKind {
    #[default]
    Device,
    HostUtility,
    Unknown(String),
}

impl QueuePlannedLaneKind {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Device => "device",
            Self::HostUtility => "host_utility",
            Self::Unknown(value) => value,
        }
    }
}

impl Serialize for QueuePlannedLaneKind {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for QueuePlannedLaneKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "device" => Self::Device,
            "host_utility" => Self::HostUtility,
            _ => Self::Unknown(value),
        })
    }
}

/// One internal scheduler unit in the additive queue-plan projection.
///
/// Strings are used for extensible scheduler states/reasons so an older
/// client can continue rendering a plan after a server adds a work kind.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct QueueWorkItem {
    pub work_id: String,
    pub parent_id: String,
    pub work_kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chain_stage: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_partition: Option<QueueBatchPartition>,
    pub priority_class: String,
    pub queue_rank: u64,
    pub bypass_count: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hard_pinned_device_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_gpu: Option<usize>,
    /// Public lane class. A host utility lane is not a hardware device, so its
    /// `planned_device_id` serializes as `null`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Option<String>)]
    pub planned_lane_kind: Option<QueuePlannedLaneKind>,
    #[serde(default)]
    pub planned_device_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lane_order: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_start_unix_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_finish_unix_ms: Option<u64>,
    #[serde(default)]
    #[schema(value_type = String)]
    pub estimate_confidence: QueueEstimateConfidence,
    /// Backward-compatible display alias. For queued work this contains the
    /// block, warm-wait, or assignment reason used by pre-Phase-E clients.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Option<String>)]
    pub blocked_reason: Option<QueueBlockedReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assignment_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warm_wait_deadline_unix_ms: Option<u64>,
    /// How long this job's dependency preparation has been running. Present
    /// only while `blocked_reason` is `preparing`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preparation_elapsed_ms: Option<u64>,
    /// What that preparation is working through, when it reports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preparation_progress: Option<QueuePreparationProgress>,
    #[serde(default)]
    #[schema(value_type = String)]
    pub activity_phase: QueueActivityPhase,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_fingerprint: Option<String>,
    /// Parent-level deterministic execution identity. Unlike
    /// `execution_fingerprint`, this may match across compatible devices.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_equivalence_fingerprint: Option<String>,
}

impl QueueWorkItem {
    /// Whether this work belongs to the host utility lane.
    ///
    /// Current servers provide the typed lane class. The exact legacy
    /// scheduler sentinel is recognized only when that class is absent so a
    /// future typed lane always remains authoritative.
    pub fn is_host_utility_lane(&self) -> bool {
        matches!(
            self.planned_lane_kind.as_ref(),
            Some(&QueuePlannedLaneKind::HostUtility)
        ) || (self.planned_lane_kind.is_none()
            && self.planned_device_id.as_deref() == Some("cpu:utility:0"))
    }

    /// Normalize the legacy host-utility sentinel for public presentation.
    ///
    /// This removes an internal scheduler identity while retaining device and
    /// unknown future typed lanes exactly as received.
    pub fn normalize_planned_lane_for_presentation(&mut self) {
        if self.is_host_utility_lane() {
            self.planned_lane_kind = Some(QueuePlannedLaneKind::HostUtility);
            self.planned_device_id = None;
        }
    }
}

/// Host-RAM telemetry for the machine serving this request.
///
/// `headroom_bytes` is what admission actually spends: available bytes less
/// the safety floor and every live reservation, so it is not
/// `available_bytes - safety_floor_bytes` and can be zero while the machine
/// still reports free memory. Clients colour pressure from `headroom_bytes`
/// against `safety_floor_bytes`.
///
/// Absent on the parent whenever the host ledger has produced no sample —
/// legacy dispatch, or before the first sample lands. Absent means unknown;
/// the server never reports zeros it did not measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct HostMemorySnapshot {
    #[schema(example = 67_430_000_000_u64)]
    pub total_bytes: u64,
    #[schema(example = 58_000_000_000_u64)]
    pub available_bytes: u64,
    #[schema(example = 48_700_000_000_u64)]
    pub headroom_bytes: u64,
    #[schema(example = 10_114_500_000_u64)]
    pub safety_floor_bytes: u64,
}

/// Versioned scheduler plan appended to `GET /api/queue`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct QueuePlan {
    pub plan_version: u64,
    pub state_version: u64,
    pub optimizer_state: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dirty_since_unix_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_replan_at_unix_ms: Option<u64>,
    #[serde(default)]
    pub work_items: Vec<QueueWorkItem>,
    /// Host-RAM telemetry sampled by the scheduler's own ledger.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_memory: Option<HostMemorySnapshot>,
}

/// Metadata for an explicitly requested durable queue page.
///
/// The cursor is opaque: clients persist and return it unchanged. `offset`
/// counts durable rows traversed by this cursor chain and is informational;
/// continuation is always defined by `next_cursor`, never by arithmetic on
/// the offset.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct QueuePage {
    pub limit: usize,
    pub offset: usize,
    pub returned: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}

/// Whole-queue listing returned by `GET /api/queue` — the client-side twin of
/// mold-server's `job_registry::QueueListing`. The server wraps the rows in
/// an object (not a bare array) so the response can grow extra fields without
/// a breaking change; unknown fields are ignored here for the same reason.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct QueueListingWire {
    #[serde(default)]
    pub entries: Vec<QueueJobEntryWire>,
    /// Active rows without durable backing. Current servers repeat this
    /// bounded set on every explicit durable page; absent on legacy hosts.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub live_only_entries: Vec<QueueJobEntryWire>,
    /// Absent on legacy servers and before the V2 coordinator has produced
    /// its first plan.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan: Option<QueuePlan>,
    /// Present only for an explicitly bounded durable request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page: Option<QueuePage>,
}

// ── GET /api/activity wire types ───────────────────────────────────────────

/// One server-owned, nonterminal unit shown by clients in Now Developing.
///
/// This intentionally carries identification and progress metadata only. In
/// particular, prompts and source media never cross the shared activity
/// boundary merely because another client can authenticate to the host.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ActiveWorkItem {
    /// Stable within this host for the lifetime of the work.
    pub id: String,
    /// Extensible public class: `generation`, `sequence`, `download`, or a
    /// scheduler-owned future work kind.
    pub kind: String,
    /// Extensible nonterminal lifecycle (`queued`, `preparing`, `loading`,
    /// `running`, `downloading`, or `cancelling`).
    pub phase: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub created_at_unix_ms: u64,
    pub updated_at_unix_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    /// Server-confirmed cancellation support for this exact item. Clients may
    /// narrow this further when the item is not part of their local session.
    #[serde(default)]
    pub can_cancel: bool,
}

/// Durable reconciliation authority for the shared, present-tense half of
/// Now Developing. Terminal work is deliberately absent; completed/recent
/// history remains bounded and local to the initiating client.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ActiveWorkSnapshot {
    /// Stable server identity used to fence a remembered host whose URL or
    /// machine has changed since the client cached its last snapshot.
    pub instance_id: String,
    pub observed_at_unix_ms: u64,
    #[serde(default)]
    pub items: Vec<ActiveWorkItem>,
    /// Work kinds whose backing authority could not be read. Clients retain
    /// only the last verified rows of these kinds while replacing healthy
    /// kinds from this snapshot.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unavailable_kinds: Vec<String>,
}

impl QueueListingWire {
    /// Preserve the legacy `entries` view for Rust callers while accepting the
    /// current split durable/live response. The first occurrence of an id is
    /// authoritative, matching Studio consumers.
    pub fn merge_live_only_entries(&mut self) {
        let mut seen = self
            .entries
            .iter()
            .map(|entry| entry.id.clone())
            .collect::<std::collections::HashSet<_>>();
        self.entries.extend(
            std::mem::take(&mut self.live_only_entries)
                .into_iter()
                .filter(|entry| seen.insert(entry.id.clone())),
        );
    }

    /// Normalize legacy internal lane identities before presenting this
    /// client-side projection to another consumer.
    pub fn normalize_planned_lanes_for_presentation(&mut self) {
        if let Some(plan) = &mut self.plan {
            for work in &mut plan.work_items {
                work.normalize_planned_lane_for_presentation();
            }
        }
    }
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

/// One startup GPU selector.
///
/// Ordinals are legacy, process-local selectors resolved against the current
/// CUDA-visible inventory. Identifiers are stable Mold IDs (`cuda:...`) or
/// NVIDIA UUID forms (`GPU-...` / `MIG-...`) and are resolved after discovery.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum GpuSelector {
    Ordinal(usize),
    Identifier(String),
}

impl std::fmt::Display for GpuSelector {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ordinal(ordinal) => ordinal.fmt(formatter),
            Self::Identifier(identifier) => identifier.fmt(formatter),
        }
    }
}

/// GPU selection for multi-GPU setups.
///
/// The custom untagged serde shape preserves the original TOML contract:
/// `gpus = []` and `"all"` select all visible devices, `"none"` selects
/// maintenance mode, numeric arrays are legacy ordinal allowlists, and string
/// arrays contain UUID-based selectors.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum GpuSelection {
    /// Use all discovered GPUs (default).
    #[default]
    All,
    /// Start without GPU workers.
    None,
    /// Use only devices matching these startup selectors.
    Specific(Vec<GpuSelector>),
}

impl GpuSelection {
    /// Parse a comma-separated CLI/environment selector list.
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        let trimmed = s.trim();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("all") {
            return Ok(Self::All);
        }
        if trimmed.eq_ignore_ascii_case("none") {
            return Ok(Self::None);
        }

        let selectors = trimmed
            .split(',')
            .map(|token| {
                let token = token.trim();
                if token.is_empty() {
                    anyhow::bail!("GPU selector entries must not be empty");
                }
                if token.eq_ignore_ascii_case("all") {
                    anyhow::bail!("'all' cannot be combined with specific GPU selectors");
                }
                if token.eq_ignore_ascii_case("none") {
                    anyhow::bail!("'none' cannot be combined with specific GPU selectors");
                }
                if token.bytes().all(|byte| byte.is_ascii_digit()) {
                    return token
                        .parse::<usize>()
                        .map(GpuSelector::Ordinal)
                        .map_err(|error| anyhow::anyhow!("invalid GPU ordinal '{token}': {error}"));
                }
                if token
                    .get(..5)
                    .is_some_and(|prefix| prefix.eq_ignore_ascii_case("cuda:"))
                    || token
                        .get(..6)
                        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("metal:"))
                    || token
                        .get(..4)
                        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("GPU-"))
                    || token
                        .get(..4)
                        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("MIG-"))
                {
                    return Ok(GpuSelector::Identifier(token.to_string()));
                }
                anyhow::bail!(
                    "invalid GPU selector '{token}': expected an ordinal, cuda: ID, metal: ID, GPU- UUID, or MIG- UUID"
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self::Specific(selectors))
    }
}

impl Serialize for GpuSelection {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::All => serializer.serialize_str("all"),
            Self::None => serializer.serialize_str("none"),
            Self::Specific(selectors) => selectors.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for GpuSelection {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Keyword(String),
            Ordinals(Vec<usize>),
            Identifiers(Vec<String>),
        }

        match Repr::deserialize(deserializer)? {
            Repr::Keyword(keyword) if keyword.eq_ignore_ascii_case("all") => Ok(Self::All),
            Repr::Keyword(keyword) if keyword.eq_ignore_ascii_case("none") => Ok(Self::None),
            Repr::Keyword(keyword) => Err(serde::de::Error::custom(format!(
                "invalid gpus keyword '{keyword}': expected 'all' or 'none'"
            ))),
            Repr::Ordinals(ordinals) if ordinals.is_empty() => Ok(Self::All),
            Repr::Ordinals(ordinals) => Ok(Self::Specific(
                ordinals.into_iter().map(GpuSelector::Ordinal).collect(),
            )),
            Repr::Identifiers(identifiers) if identifiers.is_empty() => Ok(Self::All),
            Repr::Identifiers(identifiers) => {
                let selectors = identifiers
                    .into_iter()
                    .map(|identifier| match Self::parse(&identifier) {
                        Ok(Self::Specific(mut selectors))
                            if selectors.len() == 1
                                && matches!(selectors[0], GpuSelector::Identifier(_)) =>
                        {
                            Ok(selectors.remove(0))
                        }
                        Ok(_) => Err(serde::de::Error::custom(format!(
                            "gpus string arrays must contain UUID selectors, not '{identifier}'"
                        ))),
                        Err(error) => Err(serde::de::Error::custom(error)),
                    })
                    .collect::<Result<Vec<_>, D::Error>>()?;
                Ok(Self::Specific(selectors))
            }
        }
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
/// - `Gpu { ordinal }` is the legacy process-local CUDA ordinal pin.
/// - `Device { id }` pins to an exact durable device-registry ID such as
///   `cuda:<uuid>` or `metal:default`.
///
/// Serialized as an externally-tagged enum: `{"kind":"auto"}`,
/// `{"kind":"cpu"}`, `{"kind":"gpu","ordinal":1}`, or
/// `{"kind":"device","id":"cuda:..."}`. A missing `DeviceRef` field
/// deserializes to `Auto`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default, utoipa::ToSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DeviceRef {
    #[default]
    Auto,
    Cpu,
    Gpu {
        ordinal: usize,
    },
    Device {
        id: String,
    },
}

impl DeviceRef {
    /// Helper constructor mirroring the compact `Gpu(n)` form used in tests.
    pub const fn gpu(ordinal: usize) -> Self {
        DeviceRef::Gpu { ordinal }
    }

    pub fn device(id: impl Into<String>) -> Self {
        DeviceRef::Device { id: id.into() }
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SseProgressEvent {
    /// The job is waiting for a concrete model dependency to become locally
    /// available. No accelerator lease has been granted at this point.
    DependencyWait {
        dependency: String,
        reason: String,
    },
    StageStart {
        name: String,
    },
    StageDone {
        name: String,
        elapsed_ms: u64,
    },
    /// Bounded work inside a named stage. Unlike `DenoiseStep`, this does not
    /// claim that a generation step has completed; it keeps long model-layer
    /// evaluations visibly alive without inflating denoise percentage.
    StageProgress {
        name: String,
        current: usize,
        total: usize,
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
    /// latent resolution (~width/8 × height/8 for most families; Wan 2.2
    /// TI2V's VAE compresses 16×) — clients upscale it. Video families
    /// project the clip's middle latent frame. Emitted throttled between
    /// denoise steps; disable with `MOLD_STEP_PREVIEW=0`.
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
    /// Advisories the server attached to this render: it succeeded, but
    /// something the caller asked for was adjusted, dropped, or is worth
    /// knowing — the same set the JSON path returns on
    /// `x-mold-request-warning`. Streaming clients would otherwise never see
    /// them, because an SSE render has no response headers to read (#1223).
    ///
    /// Additive; absent on every event that carried no advisory.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub request_warnings: Vec<String>,
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

    // ── Audio-only fields (additive; absent for image and video responses) ──
    /// Sample rate in Hz. Presence of this field signals an audio-only
    /// response, where `image` carries the encoded audio bytes and `format`
    /// is an [`OutputFormat::is_audio`] variant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_sample_rate: Option<u32>,
    /// Channel count of the encoded audio stream.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_channels: Option<u32>,
    /// Total encoded duration in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_duration_ms: Option<u64>,
    /// Base64-encoded waveform PNG for the gallery tile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_thumbnail: Option<String>,

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
    /// The job did not fail — the host is restarting and the job stays queued
    /// to finish there. A terminal frame is sent rather than a quiet close
    /// because a quiet close leaves the desktop app in `loading` forever and
    /// hard-fails the web client. Old clients ignore the flag and show the
    /// message; new clients treat it as interrupted, not failed.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub retained: bool,
    /// Machine-readable reason, when there is one worth branching on
    /// (currently only `server_restarting`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

/// `SseErrorEvent.code` for a job the host retained across a restart.
pub const SSE_ERROR_CODE_SERVER_RESTARTING: &str = "server_restarting";
/// The host admitted the job and then could not resolve its model.
///
/// Durable admission accepts before it resolves a checkpoint, so "this model
/// is not here" arrives as a terminal frame rather than the `404` the attached
/// path used to answer with. The code is carried so a client's
/// missing-model classifier still fires and auto-pull still works.
pub const SSE_ERROR_CODE_MODEL_NOT_FOUND: &str = "MODEL_NOT_FOUND";
/// As [`SSE_ERROR_CODE_MODEL_NOT_FOUND`], for a model no manifest knows.
pub const SSE_ERROR_CODE_UNKNOWN_MODEL: &str = "UNKNOWN_MODEL";
/// A durable direct observer disconnected after admission. The job remains
/// authoritative in the queue and clients reconcile it by the queued ID.
pub const SSE_ERROR_CODE_DURABLE_OBSERVER_DETACHED: &str = "durable_observer_detached";
/// A queued job was cancelled before its direct observer reached a worker.
pub const SSE_ERROR_CODE_QUEUED_CANCELLED: &str = "queued_cancelled";

impl SseErrorEvent {
    /// An ordinary failure: the job is over and the client should say so.
    pub fn failed(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retained: false,
            code: None,
        }
    }

    /// An ordinary failure that carries the server's own error code, so a
    /// client can branch on it exactly as it branches on an HTTP body's.
    pub fn failed_with_code(message: impl Into<String>, code: Option<String>) -> Self {
        Self {
            message: message.into(),
            retained: false,
            code,
        }
    }

    /// The host is restarting and will finish this job after it comes back.
    pub fn retained(message: impl Into<String>) -> Self {
        Self::retained_with_code(message, SSE_ERROR_CODE_SERVER_RESTARTING)
    }

    pub fn retained_with_code(message: impl Into<String>, code: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retained: true,
            code: Some(code.into()),
        }
    }

    pub fn cancelled(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retained: false,
            code: Some(SSE_ERROR_CODE_QUEUED_CANCELLED.to_string()),
        }
    }
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
    /// OS-reported memory immediately available without swapping. Additive
    /// and optional so older resource snapshots remain wire-compatible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub available: Option<u64>,
    pub used_by_mold: u64,
    pub used_by_other: u64,
}

#[derive(
    Debug,
    Clone,
    Copy,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    utoipa::ToSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum GpuBackend {
    Cuda,
    Metal,
}

impl GpuBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Metal => "metal",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_selection_parses_all_none_and_legacy_empty() {
        assert_eq!(GpuSelection::parse("").unwrap(), GpuSelection::All);
        assert_eq!(GpuSelection::parse("   ").unwrap(), GpuSelection::All);
        assert_eq!(GpuSelection::parse("ALL").unwrap(), GpuSelection::All);
        assert_eq!(GpuSelection::parse("none").unwrap(), GpuSelection::None);
    }

    #[test]
    fn gpu_selection_parses_ordinals_and_uuid_selectors() {
        assert_eq!(
            GpuSelection::parse(
                "1,cuda:0123456789abcdef0123456789abcdef,\
                 GPU-fedcba98-7654-3210-fedc-ba9876543210,\
                 MIG-00112233-4455-6677-8899-aabbccddeeff,\
                 metal:default"
            )
            .unwrap(),
            GpuSelection::Specific(vec![
                GpuSelector::Ordinal(1),
                GpuSelector::Identifier("cuda:0123456789abcdef0123456789abcdef".to_string()),
                GpuSelector::Identifier("GPU-fedcba98-7654-3210-fedc-ba9876543210".to_string()),
                GpuSelector::Identifier("MIG-00112233-4455-6677-8899-aabbccddeeff".to_string()),
                GpuSelector::Identifier("metal:default".to_string()),
            ])
        );
    }

    #[test]
    fn gpu_selection_rejects_keywords_mixed_with_specific_selectors() {
        let error = GpuSelection::parse("all,0").unwrap_err().to_string();
        assert!(error.contains("'all'"));

        let error = GpuSelection::parse("none,GPU-abc").unwrap_err().to_string();
        assert!(error.contains("'none'"));
    }

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
    fn wav_is_audio_and_never_video() {
        assert_eq!("wav".parse::<OutputFormat>().unwrap(), OutputFormat::Wav);
        assert_eq!(OutputFormat::Wav.extension(), "wav");
        assert_eq!(OutputFormat::Wav.content_type(), "audio/wav");
        assert_eq!(
            serde_json::to_value(OutputFormat::Wav).unwrap(),
            serde_json::json!("wav")
        );
        assert!(OutputFormat::Wav.is_audio());
        // The whole point of a separate predicate: every `is_video` branch —
        // thumbnail extraction, frame seeking, the ▶ badge — must skip audio.
        assert!(!OutputFormat::Wav.is_video());
        for format in [
            OutputFormat::Png,
            OutputFormat::Jpeg,
            OutputFormat::Gif,
            OutputFormat::Apng,
            OutputFormat::Webp,
            OutputFormat::Mp4,
        ] {
            assert!(!format.is_audio(), "{format:?} must not be audio");
        }
    }

    #[test]
    fn audio_only_pipeline_defaults_output_format_to_wav() {
        // The ltx2 family default is mp4, which the validator rejects for an
        // audio-only pipeline — normalisation has to know the difference.
        let json = r#"{"prompt":"rain on a tin roof","model":"ltx-2.3-22b-dev:fp8","width":1216,"height":704,"steps":30,"batch_size":1,"pipeline":"t2a"}"#;
        let mut req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.pipeline, Some(Ltx2PipelineMode::T2a));
        req.normalise_output_format(Some("ltx2"));
        assert_eq!(req.resolved_output_format(), OutputFormat::Wav);

        let mut explicit: GenerateRequest = serde_json::from_str(json).unwrap();
        explicit.output_format = Some(OutputFormat::Mp4);
        explicit.normalise_output_format(Some("ltx2"));
        assert_eq!(
            explicit.resolved_output_format(),
            OutputFormat::Mp4,
            "an explicit format still wins; the validator rejects it separately"
        );
    }

    #[test]
    fn expand_request_task_is_additive_and_backward_compatible() {
        let legacy: ExpandRequest =
            serde_json::from_str(r#"{"prompt":"a wave","model_family":"ltx2","variations":1}"#)
                .unwrap();
        assert_eq!(legacy.task, None);

        let contextual: ExpandRequest = serde_json::from_str(
            r#"{"prompt":"a wave","model_family":"ltx2","task":"image-to-video"}"#,
        )
        .unwrap();
        assert_eq!(contextual.task, Some(ExpandTask::ImageToVideo));
        assert!(serde_json::to_string(&legacy)
            .unwrap()
            .find("task")
            .is_none());
    }

    #[test]
    fn generation_expansion_task_follows_video_conditioning_priority() {
        let mut req: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"a wave","model":"ltx-2-19b-distilled:fp8","width":768,"height":512,"steps":8,"batch_size":1}"#,
        )
        .unwrap();
        assert_eq!(
            ExpandTask::for_generation("ltx2", &req),
            ExpandTask::TextToVideo
        );

        req.source_image = Some(vec![1]);
        assert_eq!(
            ExpandTask::for_generation("ltx2", &req),
            ExpandTask::ImageToVideo
        );

        req.keyframes = Some(vec![KeyframeCondition {
            frame: 0,
            image: vec![2],
            name: None,
        }]);
        assert_eq!(
            ExpandTask::for_generation("ltx2", &req),
            ExpandTask::ImageToVideo
        );

        req.keyframes.as_mut().unwrap().push(KeyframeCondition {
            frame: 8,
            image: vec![3],
            name: None,
        });
        assert_eq!(
            ExpandTask::for_generation("ltx2", &req),
            ExpandTask::KeyframeInterpolation
        );

        req.audio_file = Some(vec![4]);
        assert_eq!(
            ExpandTask::for_generation("ltx2", &req),
            ExpandTask::AudioDrivenVideo
        );

        req.audio_file = None;
        req.pipeline = Some(Ltx2PipelineMode::LipDub);
        assert_eq!(
            ExpandTask::for_generation("ltx2", &req),
            ExpandTask::AudioDrivenVideo
        );
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
            task: None,
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
            task: None,
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

    #[test]
    fn remix_request_defaults_are_stable_and_structured() {
        let request: RemixRequest = serde_json::from_str(r#"{"source_prompt":"a cat"}"#).unwrap();
        assert_eq!(request.model_family, "flux");
        assert_eq!(request.variations, 3);
        assert_eq!(request.source_kind, RemixSourceKind::Direct);
        assert!(request.dimensions.is_empty());

        let response = RemixResponse {
            source_prompt: request.source_prompt,
            root_prompt: Some("a cat".into()),
            source_kind: RemixSourceKind::Original,
            task: ExpandTask::TextToImage,
            variants: vec![RemixVariant {
                prompt: "a cat in a low-angle frame".into(),
                dimensions: vec![RemixDimension::Camera],
            }],
        };
        let json = serde_json::to_value(response).unwrap();
        assert_eq!(json["variants"][0]["dimensions"][0], "camera");
        assert_eq!(json["source_kind"], "original");
    }

    #[test]
    fn prompt_transform_provenance_reaches_output_metadata() {
        let mut request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"camera remix","model":"flux-schnell:q8","width":512,"height":512,"steps":4}"#,
        )
        .unwrap();
        request.original_prompt = Some("a cat".into());
        request.prompt_transform = Some(PromptTransformProvenance {
            operation: PromptTransformOperation::Remix,
            root_prompt: Some("a cat".into()),
            source_prompt: "an expanded cat portrait".into(),
            source_kind: RemixSourceKind::Current,
            task: ExpandTask::TextToImage,
            dimensions: vec![RemixDimension::Camera],
        });
        let metadata = OutputMetadata::from_generate_request(&request, 7, None, "test");
        assert_eq!(metadata.original_prompt.as_deref(), Some("a cat"));
        let provenance = metadata.prompt_transform.unwrap();
        assert_eq!(provenance.operation, PromptTransformOperation::Remix);
        assert_eq!(provenance.source_prompt, "an expanded cat portrait");
        assert_eq!(provenance.dimensions, vec![RemixDimension::Camera]);
    }

    #[test]
    fn video_pipeline_provenance_is_additive_and_persisted() {
        let request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"x","model":"minimax-h3-fl2va:comfy","width":768,"height":432,"steps":4}"#,
        )
        .unwrap();
        let provenance = std::iter::repeat_n('a', 64).collect::<String>();
        let video = VideoData {
            attention_path: None,
            data: vec![1],
            format: OutputFormat::Mp4,
            width: 768,
            height: 432,
            frames: 124,
            fps: 24,
            pipeline: Some(Ltx2PipelineMode::Distilled),
            pipeline_provenance_sha256: Some(provenance.clone()),
            source_preprocessing: None,
            thumbnail: vec![2],
            gif_preview: Vec::new(),
            has_audio: true,
            duration_ms: Some(5_166),
            audio_sample_rate: Some(44_100),
            audio_channels: Some(2),
        };
        let mut metadata = OutputMetadata::from_generate_request(&request, 7, None, "test");

        metadata.apply_video_output(&video);

        assert_eq!(metadata.pipeline, Some(Ltx2PipelineMode::Distilled));
        assert_eq!(metadata.pipeline_requested, Some(false));
        assert_eq!(
            metadata.pipeline_provenance_sha256.as_deref(),
            Some(provenance.as_str())
        );
        let encoded = serde_json::to_value(&metadata).unwrap();
        assert_eq!(encoded["pipeline_provenance_sha256"], provenance);

        let legacy: VideoData = serde_json::from_value(serde_json::json!({
            "data": [],
            "format": "mp4",
            "width": 1,
            "height": 1,
            "frames": 1,
            "fps": 24,
            "thumbnail": []
        }))
        .unwrap();
        assert!(legacy.pipeline_provenance_sha256.is_none());
    }

    #[test]
    fn predicted_duration_provenance_survives_video_finalization() {
        let request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"a drummer in rain","model":"ltx-2.5-22b-distilled:int8-conv","width":768,"height":512,"steps":8}"#,
        )
        .unwrap();
        let video = VideoData {
            attention_path: None,
            data: vec![1],
            format: OutputFormat::Mp4,
            width: 768,
            height: 512,
            frames: 121,
            fps: 24,
            pipeline: Some(Ltx2PipelineMode::Distilled),
            pipeline_provenance_sha256: None,
            source_preprocessing: None,
            thumbnail: vec![2],
            gif_preview: Vec::new(),
            has_audio: true,
            duration_ms: Some(5_041),
            audio_sample_rate: Some(48_000),
            audio_channels: Some(2),
        };
        let mut metadata = OutputMetadata::from_generate_request(&request, 42, None, "test");

        assert_eq!(metadata.frames, None);
        assert_eq!(metadata.duration_prediction_requested, Some(true));

        metadata.apply_video_output(&video);

        assert_eq!(metadata.frames, Some(121));
        assert_eq!(metadata.duration_prediction_requested, Some(true));
        let encoded = serde_json::to_value(&metadata).unwrap();
        assert_eq!(encoded["duration_prediction_requested"], true);
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
            durable: None,
            held_reason: None,
            ..Default::default()
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
    fn queue_page_keeps_the_cursor_opaque_and_omits_an_absent_continuation() {
        let terminal = QueuePage {
            limit: 32,
            offset: 64,
            returned: 7,
            next_cursor: None,
        };
        let json = serde_json::to_value(&terminal).unwrap();
        assert_eq!(json["limit"], 32);
        assert_eq!(json["offset"], 64);
        assert_eq!(json["returned"], 7);
        assert!(json.get("next_cursor").is_none());

        let continued: QueuePage = serde_json::from_value(serde_json::json!({
            "limit": 32,
            "offset": 32,
            "returned": 32,
            "next_cursor": "opaque-token"
        }))
        .unwrap();
        assert_eq!(continued.next_cursor.as_deref(), Some("opaque-token"));
    }

    #[test]
    fn generate_request_serde_roundtrip() {
        let req = GenerateRequest {
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
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
            references: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            prompt_transform: None,
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
            extend_video: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            keyframes: None,
            pipeline: None,
            ic_lora_control: Some("union".to_string()),
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
            id_image: None,
            id_image_name: None,
            id_weight: None,
            id_start_step: None,
            id_images: None,
            id_image_names: None,
            true_cfg: None,
            cfg_start_step: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: GenerateRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.prompt, req.prompt);
        assert_eq!(back.width, req.width);
        assert_eq!(back.seed, req.seed);
        assert_eq!(back.embed_metadata, req.embed_metadata);
        assert_eq!(back.scheduler, None);
        assert_eq!(back.ic_lora_control.as_deref(), Some("union"));
    }

    #[test]
    fn ltx25_http_duration_and_audio_contract_round_trips() {
        let predicted = serde_json::json!({
            "prompt": "a drummer in a rainstorm",
            "model": "ltx-2.5-22b-distilled:int8-conv",
            "width": 768,
            "height": 512,
            "steps": 8,
            "batch_size": 1,
            "fps": 24,
            "enable_audio": true
        });
        let predicted_request: GenerateRequest = serde_json::from_value(predicted).unwrap();
        assert_eq!(predicted_request.frames, None);
        assert_eq!(predicted_request.fps, Some(24));
        assert_eq!(predicted_request.enable_audio, Some(true));
        let predicted_wire = serde_json::to_value(&predicted_request).unwrap();
        assert!(predicted_wire.get("frames").is_none());
        assert_eq!(predicted_wire["enable_audio"], true);

        let mut explicit_wire = predicted_wire;
        explicit_wire
            .as_object_mut()
            .unwrap()
            .insert("frames".to_string(), serde_json::json!(97));
        let explicit_request: GenerateRequest = serde_json::from_value(explicit_wire).unwrap();
        assert_eq!(explicit_request.frames, Some(97));
        assert_eq!(explicit_request.enable_audio, Some(true));
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
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
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
            references: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            prompt_transform: None,
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
            extend_video: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            keyframes: None,
            pipeline: None,
            ic_lora_control: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
            id_image: None,
            id_image_name: None,
            id_weight: None,
            id_start_step: None,
            id_images: None,
            id_image_names: None,
            true_cfg: None,
            cfg_start_step: None,
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
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
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
            references: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            prompt_transform: None,
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
            extend_video: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            keyframes: None,
            pipeline: None,
            ic_lora_control: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
            id_image: None,
            id_image_name: None,
            id_weight: None,
            id_start_step: None,
            id_images: None,
            id_image_names: None,
            true_cfg: None,
            cfg_start_step: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("negative_prompt"));
    }

    #[test]
    fn output_metadata_source_preprocessing_serde_is_additive() {
        // Older rows/servers without the field must keep deserializing…
        let metadata: OutputMetadata =
            serde_json::from_str(r#"{"version":"1","prompt":"p","model":"m","seed":1,"steps":8,"guidance":3.0,"width":8,"height":8}"#)
                .unwrap();
        assert!(metadata.source_preprocessing.is_none());
        // …absent values serialize to nothing…
        let json = serde_json::to_string(&metadata).unwrap();
        assert!(!json.contains("source_preprocessing"));
        // …and present values round-trip.
        let preprocessing = Ltx2SourcePreprocessing {
            profile: crate::ltx2_preprocess::ltx2_image_preprocessing_profile(
                crate::ltx2_preprocess::Ltx2Generation::V2_3,
            ),
            codec: "openh264-cqp33".to_string(),
            fit_policy: "fill-center-crop".to_string(),
        };
        let mut stamped = metadata.clone();
        stamped.source_preprocessing = Some(preprocessing.clone());
        let round: OutputMetadata =
            serde_json::from_str(&serde_json::to_string(&stamped).unwrap()).unwrap();
        assert_eq!(round.source_preprocessing, Some(preprocessing));
    }

    /// `attention_path` is additive on both the response and the metadata:
    /// older rows/servers deserialize to `None`, `None` serializes to nothing,
    /// and every vocabulary value round-trips verbatim — the Metal one
    /// included, so a Metal print can never be mis-stamped as a CUDA one.
    #[test]
    fn attention_path_serde_is_additive_and_round_trips_every_value() {
        let metadata: OutputMetadata =
            serde_json::from_str(r#"{"version":"1","prompt":"p","model":"m","seed":1,"steps":8,"guidance":3.0,"width":8,"height":8}"#)
                .unwrap();
        assert!(metadata.attention_path.is_none());
        assert!(!serde_json::to_string(&metadata)
            .unwrap()
            .contains("attention_path"));

        for value in [
            "ltx2-bf16-math",
            "ltx2-bf16-flash",
            "ltx2-f32-chunked",
            "ltx2-metal-sdpa",
        ] {
            let video: VideoData = serde_json::from_value(serde_json::json!({
                "data": [], "format": "mp4", "width": 8, "height": 8,
                "frames": 9, "fps": 24, "thumbnail": [],
                "attention_path": value,
            }))
            .unwrap();
            assert_eq!(video.attention_path.as_deref(), Some(value));
            let round: VideoData =
                serde_json::from_str(&serde_json::to_string(&video).unwrap()).unwrap();
            assert_eq!(round.attention_path.as_deref(), Some(value));

            let mut stamped = metadata.clone();
            stamped.apply_video_output(&video);
            assert_eq!(stamped.attention_path.as_deref(), Some(value));
            let round: OutputMetadata =
                serde_json::from_str(&serde_json::to_string(&stamped).unwrap()).unwrap();
            assert_eq!(round.attention_path.as_deref(), Some(value));
        }

        // A response without the field (older server, other family) must not
        // erase a recorded value.
        let mut stamped = metadata.clone();
        stamped.attention_path = Some("ltx2-bf16-math".to_string());
        let bare: VideoData = serde_json::from_value(serde_json::json!({
            "data": [], "format": "mp4", "width": 8, "height": 8,
            "frames": 9, "fps": 24, "thumbnail": [],
        }))
        .unwrap();
        assert!(bare.attention_path.is_none());
        stamped.apply_video_output(&bare);
        assert_eq!(stamped.attention_path.as_deref(), Some("ltx2-bf16-math"));
    }

    #[test]
    fn apply_video_output_records_source_preprocessing_from_the_response() {
        let mut metadata: OutputMetadata =
            serde_json::from_str(r#"{"version":"1","prompt":"p","model":"m","seed":1,"steps":8,"guidance":3.0,"width":8,"height":8}"#)
                .unwrap();
        let preprocessing = Ltx2SourcePreprocessing {
            profile: crate::ltx2_preprocess::ltx2_image_preprocessing_profile(
                crate::ltx2_preprocess::Ltx2Generation::V2,
            ),
            codec: "openh264-cqp33".to_string(),
            fit_policy: "fill-center-crop".to_string(),
        };
        let video: VideoData = serde_json::from_value(serde_json::json!({
            "data": [], "format": "mp4", "width": 8, "height": 8,
            "frames": 9, "fps": 24, "thumbnail": [],
            "source_preprocessing": preprocessing,
        }))
        .unwrap();
        metadata.apply_video_output(&video);
        assert_eq!(metadata.source_preprocessing, Some(preprocessing));
        // A T2V response (no field) must not erase a recorded value.
        let t2v: VideoData = serde_json::from_value(serde_json::json!({
            "data": [], "format": "mp4", "width": 8, "height": 8,
            "frames": 9, "fps": 24, "thumbnail": [],
        }))
        .unwrap();
        metadata.apply_video_output(&t2v);
        assert!(metadata.source_preprocessing.is_some());
    }

    #[test]
    fn output_metadata_omits_strength_without_source_image() {
        let req = GenerateRequest {
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
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
            references: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            prompt_transform: None,
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
            extend_video: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            keyframes: None,
            pipeline: None,
            ic_lora_control: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
            id_image: None,
            id_image_name: None,
            id_weight: None,
            id_start_step: None,
            id_images: None,
            id_image_names: None,
            true_cfg: None,
            cfg_start_step: None,
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

    /// Identity conditioning rides the wire additively: present fields
    /// survive a round trip, and an ordinary request never grows the keys.
    #[test]
    fn identity_fields_round_trip_and_stay_absent_when_unset() {
        let mut req = crate::test_support::minimal_generate_request("flux-dev:q8");
        let json = serde_json::to_string(&req).unwrap();
        for key in ["id_image", "id_image_name", "id_weight", "id_start_step"] {
            assert!(!json.contains(key), "{key} must be absent: {json}");
        }
        let back: GenerateRequest = serde_json::from_str(&json).unwrap();
        assert!(back.id_image.is_none());
        assert!(back.id_weight.is_none());

        // An older client that never heard of identity still deserializes.
        let legacy: GenerateRequest =
            serde_json::from_str(
                r#"{"prompt":"a cat","model":"flux-dev:q8","width":1024,"height":1024,"steps":4,"guidance":3.5}"#,
            )
            .unwrap();
        assert!(legacy.id_image.is_none());
        assert!(legacy.id_image_name.is_none());
        assert!(legacy.id_weight.is_none());
        assert!(legacy.id_start_step.is_none());

        req.id_image = Some(vec![0x89, 0x50, 0x4E, 0x47]);
        req.id_image_name = Some("face.png".to_string());
        req.id_weight = Some(0.8);
        req.id_start_step = Some(2);
        let json = serde_json::to_string(&req).unwrap();
        let back: GenerateRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(
            back.id_image.as_deref(),
            Some(&[0x89, 0x50, 0x4E, 0x47][..])
        );
        assert_eq!(back.id_image_name.as_deref(), Some("face.png"));
        assert_eq!(back.id_weight, Some(0.8));
        assert_eq!(back.id_start_step, Some(2));
    }

    /// The multi-photograph and true-CFG fields ride the wire additively too,
    /// and an ordinary request never grows their keys.
    #[test]
    fn multi_image_and_true_cfg_fields_round_trip_and_stay_absent_when_unset() {
        let mut req = crate::test_support::minimal_generate_request("flux-dev:q8");
        let json = serde_json::to_string(&req).unwrap();
        for key in ["id_images", "id_image_names", "true_cfg", "cfg_start_step"] {
            assert!(!json.contains(key), "{key} must be absent: {json}");
        }

        req.id_images = Some(vec![vec![0x89, 0x50, 0x4E, 0x47], vec![0xFF, 0xD8, 0xFF]]);
        req.id_image_names = Some(vec!["one.png".to_string(), "two.jpg".to_string()]);
        req.true_cfg = Some(2.5);
        req.cfg_start_step = Some(3);
        let json = serde_json::to_string(&req).unwrap();
        let back: GenerateRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(
            back.id_images.as_deref(),
            Some([vec![0x89, 0x50, 0x4E, 0x47], vec![0xFF, 0xD8, 0xFF]].as_slice())
        );
        assert_eq!(
            back.id_image_names.as_deref(),
            Some(["one.png".to_string(), "two.jpg".to_string()].as_slice())
        );
        assert_eq!(back.true_cfg, Some(2.5));
        assert_eq!(back.cfg_start_step, Some(3));
        assert!(back.id_image.is_none(), "the two shapes never merge");
    }

    /// A multi-photograph print records every source; a single-photograph one
    /// records exactly what it always did and grows no keys, so its metadata
    /// stays byte-identical to a pre-`id_images` build's.
    #[test]
    fn identity_metadata_records_every_photograph_only_for_the_plural_form() {
        let mut single = crate::test_support::minimal_generate_request("flux-dev:q8");
        single.id_image = Some(vec![0x89, 0x50, 0x4E, 0x47]);
        single.id_image_name = Some("face.png".to_string());
        let metadata = OutputMetadata::from_generate_request(&single, 7, None, "0.1.0");
        assert_eq!(metadata.id_image_name.as_deref(), Some("face.png"));
        assert_eq!(
            metadata.id_image_sha256.as_deref(),
            Some(crate::identity::id_image_sha256(&[0x89, 0x50, 0x4E, 0x47]).as_str())
        );
        assert!(metadata.id_image_names.is_none());
        assert!(metadata.id_image_sha256s.is_none());
        let json = serde_json::to_string(&metadata).unwrap();
        for key in [
            "id_image_names",
            "id_image_sha256s",
            "true_cfg",
            "cfg_start_step",
        ] {
            assert!(!json.contains(key), "{key} must be absent: {json}");
        }

        let mut plural = crate::test_support::minimal_generate_request("flux-dev:q8");
        plural.id_images = Some(vec![vec![0x89, 0x50, 0x4E, 0x47], vec![0xFF, 0xD8, 0xFF]]);
        plural.id_image_names = Some(vec!["one.png".to_string(), "two.jpg".to_string()]);
        plural.id_weight = Some(0.9);
        let metadata = OutputMetadata::from_generate_request(&plural, 7, None, "0.1.0");
        assert_eq!(
            metadata.id_image_sha256s.as_deref(),
            Some(
                [
                    crate::identity::id_image_sha256(&[0x89, 0x50, 0x4E, 0x47]),
                    crate::identity::id_image_sha256(&[0xFF, 0xD8, 0xFF]),
                ]
                .as_slice()
            )
        );
        assert_eq!(
            metadata.id_image_names.as_deref(),
            Some(["one.png".to_string(), "two.jpg".to_string()].as_slice())
        );
        // The effective knobs are recorded from the plural form too.
        assert_eq!(metadata.id_weight, Some(0.9));
        assert_eq!(
            metadata.id_start_step,
            Some(crate::identity::ID_START_STEP_DEFAULT)
        );
    }

    /// The true-CFG provenance is recorded only when the print actually ran the
    /// negative branch: an inert scale must leave the metadata untouched.
    #[test]
    fn true_cfg_metadata_is_recorded_only_when_the_branch_actually_ran() {
        let mut req = crate::test_support::minimal_generate_request("flux-dev:q8");
        req.id_image = Some(vec![0x89, 0x50, 0x4E, 0x47]);
        let metadata = OutputMetadata::from_generate_request(&req, 7, None, "0.1.0");
        assert!(metadata.true_cfg.is_none());
        assert!(metadata.cfg_start_step.is_none());

        req.true_cfg = Some(1.0);
        let metadata = OutputMetadata::from_generate_request(&req, 7, None, "0.1.0");
        assert!(metadata.true_cfg.is_none(), "an inert scale ran no branch");

        req.true_cfg = Some(2.0);
        let metadata = OutputMetadata::from_generate_request(&req, 7, None, "0.1.0");
        assert_eq!(metadata.true_cfg, Some(2.0));
        assert_eq!(
            metadata.cfg_start_step,
            Some(crate::identity::CFG_START_STEP_DEFAULT)
        );

        req.id_weight = Some(0.0);
        let metadata = OutputMetadata::from_generate_request(&req, 7, None, "0.1.0");
        assert!(
            metadata.true_cfg.is_none(),
            "a zero weight renders the plain print, so nothing is recorded"
        );
    }

    /// Identity provenance is recorded from the effective values, and only
    /// when the print actually carried a face reference.
    #[test]
    fn identity_metadata_records_effective_values_only_with_an_image() {
        let mut req = crate::test_support::minimal_generate_request("flux-dev:q8");
        req.id_weight = Some(2.5);
        req.id_start_step = Some(3);
        req.id_image_name = Some("face.png".to_string());

        let metadata = OutputMetadata::from_generate_request(&req, 7, None, "0.1.0");
        assert_eq!(metadata.id_image_name, None);
        assert_eq!(metadata.id_image_sha256, None);
        assert_eq!(metadata.id_weight, None);
        assert_eq!(metadata.id_start_step, None);
        let json = serde_json::to_string(&metadata).unwrap();
        for key in [
            "id_image_name",
            "id_image_sha256",
            "id_weight",
            "id_start_step",
        ] {
            assert!(!json.contains(key), "{key} must be absent: {json}");
        }

        req.id_image = Some(b"\x89PNG\r\n\x1a\n".to_vec());
        let metadata = OutputMetadata::from_generate_request(&req, 7, None, "0.1.0");
        assert_eq!(metadata.id_image_name.as_deref(), Some("face.png"));
        assert_eq!(
            metadata.id_image_sha256.as_deref().map(str::len),
            Some(64),
            "the reference is recorded as a digest, never as bytes"
        );
        assert_eq!(metadata.id_weight, Some(2.5));
        assert_eq!(metadata.id_start_step, Some(3));

        // Defaults are materialized so saved provenance records what ran.
        req.id_weight = None;
        req.id_start_step = None;
        let metadata = OutputMetadata::from_generate_request(&req, 7, None, "0.1.0");
        assert_eq!(metadata.id_weight, Some(crate::identity::ID_WEIGHT_DEFAULT));
        assert_eq!(
            metadata.id_start_step,
            Some(crate::identity::ID_START_STEP_DEFAULT)
        );

        // Round trip, plus tolerance for metadata written before identity.
        let json = serde_json::to_string(&metadata).unwrap();
        let back: OutputMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id_weight, metadata.id_weight);
        let legacy: OutputMetadata = serde_json::from_str(
            r#"{"prompt":"a cat","model":"flux-dev:q8","seed":1,"steps":4,"guidance":3.5,"width":8,"height":8,"version":"0.1.0"}"#,
        )
        .unwrap();
        assert_eq!(legacy.id_image_sha256, None);
        assert_eq!(legacy.id_start_step, None);
    }

    /// Reuse-settings source restore: the metadata records the client's
    /// provenance label and the SHA-256 of the exact source bytes — names
    /// and hashes only, never the image payload.
    #[test]
    fn output_metadata_records_source_image_provenance() {
        let mut req = GenerateRequest {
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
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
            references: None,
            strength: 0.6,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            prompt_transform: None,
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
            extend_video: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            keyframes: None,
            pipeline: None,
            ic_lora_control: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
            id_image: None,
            id_image_name: None,
            id_weight: None,
            id_start_step: None,
            id_images: None,
            id_image_names: None,
            true_cfg: None,
            cfg_start_step: None,
        };

        let metadata = OutputMetadata::from_generate_request(&req, 7, None, "0.1.0");
        assert_eq!(metadata.output_mode, Some(GenerationOutputMode::OneShot));
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

        // The client-shaped source-fit policy is echoed verbatim so crop
        // controls restore on Reuse settings and running-job selection; when
        // the client sends none the field stays absent from the JSON.
        req.source_fit = Some(serde_json::json!({ "mode": "crop-fill" }));
        let fitted = OutputMetadata::from_generate_request(&req, 7, None, "0.1.0");
        assert_eq!(
            fitted.source_fit,
            Some(serde_json::json!({ "mode": "crop-fill" }))
        );
        req.source_fit = None;
        let unfitted = OutputMetadata::from_generate_request(&req, 7, None, "0.1.0");
        assert_eq!(unfitted.source_fit, None);
        assert!(!serde_json::to_string(&unfitted)
            .unwrap()
            .contains("source_fit"));
    }

    #[test]
    fn output_metadata_records_byte_free_keyframe_provenance() {
        let req: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "end on the painted doorway",
            "model": "minimax-h3-fl2va:official-bf16",
            "width": 768,
            "height": 512,
            "steps": 30,
            "frames": 124,
            "keyframes": [{
                "frame": 123,
                "image": "Y2xvc2luZw==",
                "name": "closing.png"
            }]
        }))
        .unwrap();

        let metadata = OutputMetadata::from_generate_request(&req, 7, None, "0.1.0");
        let keyframes = metadata.keyframes.as_ref().unwrap();
        assert_eq!(keyframes.len(), 1);
        assert_eq!(keyframes[0].frame, 123);
        assert_eq!(keyframes[0].name.as_deref(), Some("closing.png"));
        assert_eq!(
            keyframes[0].sha256,
            "6f9a48800d2f3095e8a310a4317e009a5dc222e11c45b34ec8d5544feebb8277"
        );
        let serialized = serde_json::to_string(&metadata).unwrap();
        assert!(!serialized.contains("Y2xvc2luZw=="));

        let mut legacy_value = serde_json::to_value(&metadata).unwrap();
        legacy_value.as_object_mut().unwrap().remove("keyframes");
        let legacy: OutputMetadata = serde_json::from_value(legacy_value).unwrap();
        assert_eq!(legacy.keyframes, None);
    }

    #[test]
    fn output_metadata_records_qwen_edit_image_hashes_in_order() {
        let req: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "edit",
            "model": "qwen-image-edit:q4",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "seed": 7,
            "edit_images": ["dGFyZ2V0", "cmVmZXJlbmNl"]
        }))
        .unwrap();

        let metadata = OutputMetadata::from_generate_request(&req, 7, None, "0.1.0");
        let expected = req
            .edit_images
            .as_ref()
            .unwrap()
            .iter()
            .map(|bytes| {
                use sha2::{Digest, Sha256};
                let mut hasher = Sha256::new();
                hasher.update(bytes);
                format!("{:x}", hasher.finalize())
            })
            .collect::<Vec<_>>();

        assert_eq!(
            metadata.edit_image_sha256s.as_deref(),
            Some(expected.as_slice())
        );
    }

    #[test]
    fn h3_output_metadata_freezes_reference_shape_for_requested_frames() {
        let mut req: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "short synchronized print",
            "model": crate::minimax_h3::REF2VA_COMFY,
            "width": crate::minimax_h3::DEFAULT_WIDTH,
            "height": crate::minimax_h3::DEFAULT_HEIGHT,
            "steps": crate::minimax_h3::DEFAULT_STEPS,
            "guidance": 0.0,
            "batch_size": 1,
            "frames": crate::minimax_h3::MIN_FRAMES,
            "fps": crate::minimax_h3::FIXED_FPS,
            "output_format": "mp4"
        }))
        .unwrap();
        let reference = GenerationReference::Audio {
            media: GenerationReferenceAuthority::Descriptor,
            provenance: GenerationReferenceProvenance {
                name: Some("long.wav".to_string()),
                sha256: Some("11".repeat(32)),
                crop: None,
            },
            mime_type: "audio/wav".to_string(),
            duration_ms: 15_000,
            sample_rate: 32_000,
            channels: 2,
            sample_count: Some(480_000),
        };
        req.references = Some(vec![reference.clone()]);

        let short = OutputMetadata::from_generate_request(&req, 7, None, "test");
        let short_shape = short.references.unwrap()[0].prepared_shape.clone().unwrap();
        assert_eq!(
            short_shape,
            crate::minimax_h3::reference_prepared_shape_for_target(
                &reference,
                crate::minimax_h3::MIN_FRAMES,
            )
            .unwrap()
        );
        assert_ne!(
            short_shape,
            crate::minimax_h3::reference_prepared_shape(&reference).unwrap()
        );

        req.frames = Some(crate::minimax_h3::MAX_FRAMES);
        let long = OutputMetadata::from_generate_request(&req, 7, None, "test");
        assert_eq!(
            long.references.unwrap()[0].prepared_shape.as_ref(),
            Some(&crate::minimax_h3::reference_prepared_shape(&reference).unwrap())
        );
    }

    #[test]
    fn output_metadata_includes_negative_prompt_when_provided() {
        let req = GenerateRequest {
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
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
            references: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            prompt_transform: None,
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
            extend_video: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            keyframes: None,
            pipeline: None,
            ic_lora_control: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
            id_image: None,
            id_image_name: None,
            id_weight: None,
            id_start_step: None,
            id_images: None,
            id_image_names: None,
            true_cfg: None,
            cfg_start_step: None,
        };
        let metadata = OutputMetadata::from_generate_request(&req, 1, None, "0.1.0");
        assert_eq!(metadata.negative_prompt.as_deref(), Some("blurry, ugly"));
    }

    #[test]
    fn output_metadata_includes_strength_and_scheduler_when_applicable() {
        let req = GenerateRequest {
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
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
            references: None,
            strength: 0.5,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            prompt_transform: None,
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
            extend_video: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            keyframes: None,
            pipeline: None,
            ic_lora_control: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
            id_image: None,
            id_image_name: None,
            id_weight: None,
            id_start_step: None,
            id_images: None,
            id_image_names: None,
            true_cfg: None,
            cfg_start_step: None,
        };

        let metadata =
            OutputMetadata::from_generate_request(&req, 9, Some(Scheduler::UniPc), "0.1.0");
        assert_eq!(metadata.strength, Some(0.5));
        assert_eq!(metadata.scheduler, Some(Scheduler::UniPc));
    }

    #[test]
    fn output_metadata_preserves_recreate_knobs() {
        let req = GenerateRequest {
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
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
            references: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: Some("controlnet-canny-sd15".to_string()),
            control_scale: 0.8,
            expand: None,
            original_prompt: None,
            prompt_transform: None,
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
            extend_video: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            keyframes: None,
            pipeline: Some(Ltx2PipelineMode::Retake),
            ic_lora_control: None,
            loras: Some(vec![LoraWeight {
                path: "/loras/camera.safetensors".to_string(),
                scale: 0.7,
                expert: None,
            }]),
            retake_range: Some(TimeRange {
                start_seconds: 1.0,
                end_seconds: 2.5,
            }),
            spatial_upscale: Some(Ltx2SpatialUpscale::X1_5),
            temporal_upscale: Some(Ltx2TemporalUpscale::X2),
            placement: None,
            id_image: None,
            id_image_name: None,
            id_weight: None,
            id_start_step: None,
            id_images: None,
            id_image_names: None,
            true_cfg: None,
            cfg_start_step: None,
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
        assert_eq!(metadata.pipeline_requested, Some(true));
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

        let mut control_req = req;
        control_req.pipeline = Some(Ltx2PipelineMode::IcLora);
        control_req.ic_lora_control = Some("motion-track".to_string());
        let control_metadata =
            OutputMetadata::from_generate_request(&control_req, 9, None, "0.1.0");
        assert_eq!(
            control_metadata.ic_lora_control.as_deref(),
            Some("motion-track")
        );
    }

    #[test]
    fn ltx2_pipeline_display_matches_the_wire_contract() {
        // Exhaustive by construction: no wildcard arm, so a new variant fails
        // to compile until its wire string is spelled out here too.
        fn expected_wire(mode: Ltx2PipelineMode) -> &'static str {
            match mode {
                Ltx2PipelineMode::OneStage => "one-stage",
                Ltx2PipelineMode::TwoStage => "two-stage",
                Ltx2PipelineMode::TwoStageHq => "two-stage-hq",
                Ltx2PipelineMode::Distilled => "distilled",
                Ltx2PipelineMode::IcLora => "ic-lora",
                Ltx2PipelineMode::Keyframe => "keyframe",
                Ltx2PipelineMode::A2Vid => "a2-vid",
                Ltx2PipelineMode::Retake => "retake",
                Ltx2PipelineMode::LipDub => "lip-dub",
                Ltx2PipelineMode::T2a => "t2a",
            }
        }

        for mode in Ltx2PipelineMode::ALL {
            let expected = expected_wire(mode);
            assert_eq!(mode.as_str(), expected);
            assert_eq!(mode.to_string(), expected);
            assert_eq!(serde_json::to_value(mode).unwrap(), expected);
            // Deserialization is the direction that actually bit us: a client
            // sending "a2vid" instead of "a2-vid" 422s on the request body.
            assert_eq!(
                serde_json::from_value::<Ltx2PipelineMode>(serde_json::json!(expected)).unwrap(),
                mode,
            );
        }
    }

    /// Every Studio surface mirrors `Ltx2PipelineMode` as a TypeScript string
    /// union. A member that does not deserialize is not a cosmetic drift — the
    /// request 422s on `Option<Ltx2PipelineMode>` — so pin the unions to the
    /// Rust wire strings here, next to the authority.
    #[test]
    fn lip_dub_pipeline_mode_round_trips_its_kebab_case_wire_string() {
        // Easy to get wrong in both directions: the adapter id is `lipdub`,
        // the pipeline is `lip-dub`. Mixing them 422s the whole request.
        assert_eq!(Ltx2PipelineMode::LipDub.as_str(), "lip-dub");
        assert_eq!(
            serde_json::to_value(Ltx2PipelineMode::LipDub).unwrap(),
            "lip-dub"
        );
        assert_eq!(
            serde_json::from_str::<Ltx2PipelineMode>("\"lip-dub\"").unwrap(),
            Ltx2PipelineMode::LipDub
        );
        assert!(serde_json::from_str::<Ltx2PipelineMode>("\"lipdub\"").is_err());
    }

    #[test]
    fn ltx2_pipeline_typescript_unions_match_the_wire_contract() {
        let workspace = env!("CARGO_MANIFEST_DIR")
            .strip_suffix("/crates/mold-core")
            .or_else(|| env!("CARGO_MANIFEST_DIR").strip_suffix("crates/mold-core"))
            .unwrap_or(env!("CARGO_MANIFEST_DIR"));

        for path in ["web/src/types.ts", "desktop/src/lib/api/types.ts"] {
            let full_path = format!("{workspace}/{path}");
            let source = std::fs::read_to_string(&full_path)
                .unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
            let union = source
                .split_once("export type Ltx2PipelineMode =")
                .unwrap_or_else(|| panic!("{path} must declare `export type Ltx2PipelineMode`"))
                .1
                .split_once(';')
                .unwrap_or_else(|| panic!("{path} Ltx2PipelineMode union must end in `;`"))
                .0;

            let members: Vec<&str> = union
                .split('|')
                .map(str::trim)
                .filter(|member| !member.is_empty())
                .map(|member| member.trim_matches('"'))
                .collect();
            let expected: Vec<&str> = Ltx2PipelineMode::ALL.iter().map(|m| m.as_str()).collect();

            assert_eq!(
                members, expected,
                "{path} Ltx2PipelineMode union must match mold-core's wire strings exactly; \
                 a mismatched member makes every request using it fail with 422"
            );
        }
    }

    #[test]
    fn ltx_guidance_capabilities_follow_checkpoint_and_pipeline_recipe() {
        assert_eq!(
            GuidanceCapabilities::for_recipe("ltx2", "ltx-2.3-22b-distilled:fp8", None,),
            GuidanceCapabilities::FIXED_ONE,
        );
        assert_eq!(
            GuidanceCapabilities::for_recipe("ltx2", "ltx-2.3-22b-dev:fp8", None),
            GuidanceCapabilities::ADJUSTABLE_CFG,
        );
        assert_eq!(
            GuidanceCapabilities::for_recipe(
                "ltx2",
                "ltx-2.3-22b-dev:fp8",
                Some(Ltx2PipelineMode::Distilled),
            ),
            GuidanceCapabilities::FIXED_ONE,
        );
        assert_eq!(
            GuidanceCapabilities::for_recipe("flux", "flux-dev:q8", None),
            GuidanceCapabilities::ADJUSTABLE_NO_NEGATIVE,
        );
        assert_eq!(
            GuidanceCapabilities::for_recipe(
                "ltx2",
                "ltx-2.3-22b-distilled:fp8",
                Some(Ltx2PipelineMode::TwoStage),
            ),
            GuidanceCapabilities::ADJUSTABLE_CFG,
        );
        assert_eq!(
            GuidanceCapabilities::for_recipe(
                "ltx-video",
                "ltx-video-0.9.8-13b-distilled:bf16",
                None,
            ),
            GuidanceCapabilities::FIXED_ONE,
        );
    }

    /// A recipe that pins guidance owns the default. The CLI has no per-model
    /// `default_guidance` for the distilled LTX checkpoints, so it used to fall
    /// back to the global 3.5 and then fail its own profile validation with
    /// "guidance is fixed at 1 for this recipe" — the model was unrunnable from
    /// the CLI without passing `--guidance 1` by hand. An explicit conflicting
    /// value must still be refused rather than silently rewritten.
    #[test]
    fn a_pinned_recipe_supplies_the_guidance_default_but_never_rewrites_an_explicit_one() {
        let pinned = GuidanceCapabilities::FIXED_ONE;
        assert_eq!(pinned.resolve_scale(None, 3.5), 1.0);
        assert_eq!(pinned.resolve_scale(Some(3.5), 3.5), 3.5);
        assert_eq!(pinned.resolve_scale(Some(1.0), 3.5), 1.0);

        for adjustable in [
            GuidanceCapabilities::ADJUSTABLE_CFG,
            GuidanceCapabilities::ADJUSTABLE_NO_NEGATIVE,
        ] {
            assert_eq!(adjustable.resolve_scale(None, 3.5), 3.5);
            assert_eq!(adjustable.resolve_scale(Some(7.0), 3.5), 7.0);
        }
    }

    /// Every Studio surface has to know which pipeline renders audio only
    /// *before* it builds a request — it decides the output format, whether
    /// conditioning controls are offered at all, and whether `enable_audio`
    /// may travel. Spelling it inline let desktop send `pipeline=t2a` with a
    /// fresh form's `enable_audio: false`, which the server refuses. The
    /// shared predicate is the one mirror; pin it here, next to the authority.
    #[test]
    fn ltx2_audio_only_pipeline_ts_mirror_matches_the_rust_authority() {
        let audio_only: Vec<&str> = Ltx2PipelineMode::ALL
            .iter()
            .filter(|mode| mode.is_audio_only())
            .map(|mode| mode.as_str())
            .collect();
        assert_eq!(
            audio_only,
            vec!["t2a"],
            "the TS mirror is a single constant; a second audio-only pipeline \
             needs it widened to a set first"
        );

        let workspace = env!("CARGO_MANIFEST_DIR")
            .strip_suffix("/crates/mold-core")
            .or_else(|| env!("CARGO_MANIFEST_DIR").strip_suffix("crates/mold-core"))
            .unwrap_or(env!("CARGO_MANIFEST_DIR"));
        let path = format!("{workspace}/studio/lib/ltx2Pipeline.ts");
        let source =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
        assert!(
            source.contains(&format!(
                "export const AUDIO_ONLY_PIPELINE = \"{}\";",
                audio_only[0]
            )),
            "studio/lib/ltx2Pipeline.ts must pin AUDIO_ONLY_PIPELINE to `{}`",
            audio_only[0]
        );
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
    fn sse_progress_stage_progress_roundtrip() {
        let event = SseProgressEvent::StageProgress {
            name: "Encoding prompt (Gemma)".to_string(),
            current: 17,
            total: 48,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"stage_progress""#));
        let back: SseProgressEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            back,
            SseProgressEvent::StageProgress {
                name,
                current: 17,
                total: 48,
            } if name == "Encoding prompt (Gemma)"
        ));
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
            request_warnings: Vec::new(),
            audio_sample_rate: None,
            audio_channels: None,
            audio_duration_ms: None,
            audio_thumbnail: None,
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
        let event = SseErrorEvent::failed("something failed".to_string());
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
                restart_required: false,
            }],
        };
        let json = serde_json::to_string(&listing).unwrap();
        assert!(json.contains(r#""source":"file""#), "got: {json}");
        assert!(json.contains(r#""value":7680"#), "got: {json}");
        // env_var is omitted from the wire unless the source is "env".
        assert!(!json.contains("env_var"), "got: {json}");
        assert!(!json.contains("restart_required"), "got: {json}");
        let back: ConfigListing = serde_json::from_str(&json).unwrap();
        assert_eq!(back.entries[0].key, "server_port");
        assert_eq!(back.profile.as_deref(), Some("default"));
        assert!(!back.entries[0].restart_required);

        let env_entry = ConfigEntry {
            key: "embed_metadata".to_string(),
            value: serde_json::json!(true),
            source: "env".to_string(),
            env_var: Some("MOLD_EMBED_METADATA".to_string()),
            restart_required: false,
        };
        let json = serde_json::to_string(&env_entry).unwrap();
        assert!(
            json.contains(r#""env_var":"MOLD_EMBED_METADATA""#),
            "got: {json}"
        );
        let restart_entry = ConfigEntry {
            key: "scheduler.replan_debounce_ms".to_string(),
            value: serde_json::json!(2000),
            source: "db".to_string(),
            env_var: None,
            restart_required: true,
        };
        let json = serde_json::to_string(&restart_entry).unwrap();
        assert!(json.contains(r#""restart_required":true"#), "got: {json}");

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
    fn sse_progress_dependency_wait_roundtrip_is_typed() {
        let event = SseProgressEvent::DependencyWait {
            dependency: "Qwen3 q6".to_string(),
            reason: "joining an in-progress encoder dependency download".to_string(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"dependency_wait""#));
        let back: SseProgressEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            back,
            SseProgressEvent::DependencyWait {
                dependency,
                reason,
            } if dependency == "Qwen3 q6" && reason.contains("in-progress")
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
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
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
            references: None,
            strength: 0.5,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            prompt_transform: None,
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
            extend_video: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            keyframes: None,
            pipeline: None,
            ic_lora_control: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
            id_image: None,
            id_image_name: None,
            id_weight: None,
            id_start_step: None,
            id_images: None,
            id_image_names: None,
            true_cfg: None,
            cfg_start_step: None,
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
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
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
            references: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            prompt_transform: None,
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
            extend_video: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            keyframes: None,
            pipeline: None,
            ic_lora_control: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
            id_image: None,
            id_image_name: None,
            id_weight: None,
            id_start_step: None,
            id_images: None,
            id_image_names: None,
            true_cfg: None,
            cfg_start_step: None,
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
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
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
            references: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            prompt_transform: None,
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
            extend_video: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            keyframes: None,
            pipeline: None,
            ic_lora_control: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
            id_image: None,
            id_image_name: None,
            id_weight: None,
            id_start_step: None,
            id_images: None,
            id_image_names: None,
            true_cfg: None,
            cfg_start_step: None,
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
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
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
            references: None,
            strength: 0.75,
            mask_image: None,
            control_image: Some(control_bytes.clone()),
            control_model: Some("controlnet-canny-sd15".to_string()),
            control_scale: 0.8,
            expand: None,
            original_prompt: None,
            prompt_transform: None,
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
            extend_video: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            keyframes: None,
            pipeline: None,
            ic_lora_control: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
            id_image: None,
            id_image_name: None,
            id_weight: None,
            id_start_step: None,
            id_images: None,
            id_image_names: None,
            true_cfg: None,
            cfg_start_step: None,
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
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
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
            references: None,
            strength: 0.75,
            mask_image: Some(mask_bytes.clone()),
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            prompt_transform: None,
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
            extend_video: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            keyframes: None,
            pipeline: None,
            ic_lora_control: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
            id_image: None,
            id_image_name: None,
            id_weight: None,
            id_start_step: None,
            id_images: None,
            id_image_names: None,
            true_cfg: None,
            cfg_start_step: None,
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
            host_memory: None,
            durable_media: Some(DurableMediaStatus {
                available: false,
                reasons: vec!["owner media store unavailable".to_string()],
            }),
        };
        let json = serde_json::to_string(&status).unwrap();
        let parsed: super::ServerStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.hostname.as_deref(), Some("bender"));
        assert_eq!(
            parsed.durable_media.as_ref().map(|media| media.available),
            Some(false)
        );
        assert_eq!(
            parsed
                .durable_media
                .as_ref()
                .map(|media| media.reasons.as_slice()),
            Some(["owner media store unavailable".to_string()].as_slice())
        );
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

    #[test]
    fn generation_batch_status_preserves_legacy_wire_and_enriched_outcome() {
        let legacy = r#"{
            "id":"batch-1",
            "client_batch_id":"client-1",
            "children":[{"index":1,"job_id":"job-1","state":"accepted"}]
        }"#;
        let legacy: super::GenerationBatchStatus = serde_json::from_str(legacy).unwrap();
        assert_eq!(legacy.instance_id, "");
        assert!(!legacy.durable);
        assert_eq!(legacy.children[0].created_at_ms, 0);
        assert_eq!(legacy.children[0].updated_at_ms, 0);
        assert!(legacy.children[0].result.is_none());

        let enriched = super::GenerationBatchStatus {
            id: "batch-1".into(),
            client_batch_id: "client-1".into(),
            instance_id: "instance-1".into(),
            durable: true,
            children: vec![super::GenerationBatchChild {
                index: 1,
                job_id: "job-1".into(),
                state: super::GenerationBatchChildState::Complete,
                error: None,
                error_code: None,
                retryable: None,
                created_at_ms: 10,
                updated_at_ms: 20,
                revision: 2,
                completed_at_ms: Some(20),
                terminal_error: None,
                result: Some(super::GenerationBatchResult {
                    filename: Some("finished.png".into()),
                    original_filename: Some("original.png".into()),
                    seed: Some(4242),
                    generation_time_ms: Some(7_500),
                    gpu: Some(1),
                }),
            }],
        };
        let json = serde_json::to_value(&enriched).unwrap();
        assert_eq!(json["instance_id"], "instance-1");
        assert_eq!(json["durable"], true);
        assert_eq!(json["children"][0]["result"]["filename"], "finished.png");
        assert_eq!(json["children"][0]["result"]["seed"], 4242);
        assert_eq!(json["children"][0]["result"]["generation_time_ms"], 7_500);
        assert_eq!(json["children"][0]["result"]["gpu"], 1);
        assert_eq!(
            serde_json::from_value::<super::GenerationBatchStatus>(json).unwrap(),
            enriched
        );
        // A child settled before the terminal facts existed still reads back,
        // reporting absence rather than a fabricated zero.
        let legacy: super::GenerationBatchResult =
            serde_json::from_str(r#"{"filename":"finished.png"}"#).unwrap();
        assert_eq!(legacy.seed, None);
        assert_eq!(legacy.generation_time_ms, None);
        assert_eq!(legacy.gpu, None);
    }

    #[test]
    fn generation_batch_authority_rejects_identity_changes() {
        let status = super::GenerationBatchStatus {
            id: "batch-1".into(),
            client_batch_id: "client-1".into(),
            instance_id: "instance-1".into(),
            durable: true,
            children: Vec::new(),
        };
        let authority =
            super::GenerationBatchAuthority::from_admission(&status, "client-1").unwrap();
        authority.validate_status(&status).unwrap();

        for (field, changed) in [
            (
                "instance",
                super::GenerationBatchStatus {
                    instance_id: "instance-2".into(),
                    ..status.clone()
                },
            ),
            (
                "batch",
                super::GenerationBatchStatus {
                    id: "batch-2".into(),
                    ..status.clone()
                },
            ),
            (
                "client",
                super::GenerationBatchStatus {
                    client_batch_id: "client-2".into(),
                    ..status.clone()
                },
            ),
        ] {
            assert!(authority
                .validate_status(&changed)
                .unwrap_err()
                .contains(field));
        }
    }

    #[test]
    fn generation_batch_bulk_authority_rejects_foreign_duplicate_and_omitted_rows() {
        let status = |batch: &str, client: &str| super::GenerationBatchStatus {
            id: batch.into(),
            client_batch_id: client.into(),
            instance_id: "instance-1".into(),
            durable: true,
            children: Vec::new(),
        };
        let first = status("batch-1", "client-1");
        let second = status("batch-2", "client-2");
        let authorities = [
            super::GenerationBatchAuthority::from_admission(&first, "client-1").unwrap(),
            super::GenerationBatchAuthority::from_admission(&second, "client-2").unwrap(),
        ];
        let response = |batches, missing| super::GenerationBatchStatusResponse {
            instance_id: "instance-1".into(),
            batches,
            missing: super::GenerationBatchMissing {
                client_batch_ids: missing,
                batch_ids: Vec::new(),
            },
        };
        super::validate_generation_batch_status_response(
            &response(vec![first.clone(), second.clone()], Vec::new()),
            &authorities,
        )
        .unwrap();
        assert!(super::validate_generation_batch_status_response(
            &response(vec![first.clone(), first.clone()], Vec::new()),
            &authorities,
        )
        .unwrap_err()
        .contains("duplicated"));
        assert!(super::validate_generation_batch_status_response(
            &response(vec![first.clone()], Vec::new()),
            &authorities,
        )
        .unwrap_err()
        .contains("omitted"));
        assert!(super::validate_generation_batch_status_response(
            &response(vec![first, status("foreign", "foreign")], Vec::new()),
            &authorities,
        )
        .unwrap_err()
        .contains("foreign"));
        assert!(super::validate_generation_batch_status_response(
            &super::GenerationBatchStatusResponse {
                instance_id: "instance-1".into(),
                batches: vec![second],
                missing: super::GenerationBatchMissing {
                    client_batch_ids: Vec::new(),
                    batch_ids: vec!["foreign-batch".into()],
                },
            },
            &authorities,
        )
        .unwrap_err()
        .contains("foreign batch"));
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
            request_warnings: Vec::new(),
            audio_sample_rate: None,
            audio_channels: None,
            audio_duration_ms: None,
            audio_thumbnail: None,
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
            request_warnings: Vec::new(),
            audio_sample_rate: None,
            audio_channels: None,
            audio_duration_ms: None,
            audio_thumbnail: None,
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
            request_warnings: Vec::new(),
            audio_sample_rate: None,
            audio_channels: None,
            audio_duration_ms: None,
            audio_thumbnail: None,
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
                available: None,
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
    fn ram_available_is_additive_and_wire_backward_compatible() {
        let legacy = r#"{
            "total": 64000000000,
            "used": 50000000000,
            "used_by_mold": 10000000000,
            "used_by_other": 40000000000
        }"#;
        let parsed: RamSnapshot = serde_json::from_str(legacy).unwrap();
        assert_eq!(parsed.available, None);

        let current = RamSnapshot {
            total: 64_000_000_000,
            used: 50_000_000_000,
            available: Some(20_000_000_000),
            used_by_mold: 10_000_000_000,
            used_by_other: 40_000_000_000,
        };
        let json = serde_json::to_string(&current).unwrap();
        assert!(json.contains(r#""available":20000000000"#));
        let round_trip: RamSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(round_trip.available, current.available);
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
    /// Stable identity for the current media bytes. Clients use this in
    /// thumbnail cache keys so replacing a file in place cannot show stale
    /// pixels. Older servers omit it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_version: Option<String>,
    /// True when `metadata` was synthesized (no mold:parameters chunk found).
    #[serde(default, skip_serializing_if = "is_false")]
    pub metadata_synthetic: bool,
    /// Editable print title from the gallery row (`generations.title`).
    /// Additive; older servers omit it and clients fall back to
    /// `metadata.title`, then a prompt excerpt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Tag names attached to this print on the serving host, sorted
    /// case-insensitively. Additive; empty/absent means untagged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// Favorite flag on the serving host. Additive; absent means `false`.
    #[serde(default, skip_serializing_if = "is_false")]
    pub favorite: bool,
    /// Ids of the collections (on the serving host) that contain this print.
    /// Additive; empty/absent means no collection membership.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub collections: Vec<String>,
    /// Unix seconds at which the print was moved to the trash. Present only
    /// on `GET /api/gallery?view=trash` rows; live rows omit it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trashed_at: Option<u64>,
    /// Unix seconds at which the retention sweeper will purge this trashed
    /// print. Absent when retention is "keep forever" or the row is live.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub purge_at: Option<u64>,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// A manual collection of prints on one host. Collections merge across
/// hosts by `slug` (the normalized name); `id` is host-local.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct Collection {
    pub id: String,
    pub name: String,
    pub slug: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Gallery filename used as the cover tile; `None` lets clients pick
    /// the newest member.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cover_filename: Option<String>,
    /// Whether members are omitted from the default Library grid and search.
    #[serde(default, skip_serializing_if = "is_false")]
    pub hidden: bool,
    /// Number of prints in the collection (trashed members included; they
    /// keep their membership until purged).
    pub count: u64,
    /// Unix seconds.
    pub created_at: u64,
    /// Unix seconds.
    pub updated_at: u64,
}

/// `GET /api/gallery/collections/:id`: the collection plus its member
/// gallery filenames in collection order (insertion order). Additive shape:
/// clients that only need the summary read `collection`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct CollectionDetail {
    pub collection: Collection,
    /// Gallery filenames in the collection, ordered by position.
    #[serde(default)]
    pub filenames: Vec<String>,
}

/// One tag and how many prints carry it (trashed prints included), from
/// `GET /api/gallery/tags`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct TagCount {
    pub name: String,
    pub count: u64,
}

/// Trash support advertised under `capabilities.gallery.trash`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GalleryTrashCapabilities {
    /// Whether `DELETE /api/gallery/image/:filename` moves to the trash
    /// (true) or hard-deletes as on older servers (false).
    pub enabled: bool,
    /// Effective `gallery.trash_retention_days` (0 = keep forever).
    pub retention_days: u32,
}

/// Per-print edits for `PATCH /api/gallery/image/:filename`. Every field is
/// optional; absent fields are untouched. `tags` replaces the whole set and
/// wins over `add_tags` / `remove_tags` when both are present.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GalleryPatchRequest {
    /// New title; an empty string clears it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub favorite: Option<bool>,
    /// Replace the full tag set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub add_tags: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remove_tags: Option<Vec<String>>,
}

/// Bulk organization edits for `POST /api/gallery/organize`. Applies the
/// same mutation to every listed filename on the serving host.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GalleryOrganizeRequest {
    pub filenames: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub favorite: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub add_tags: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remove_tags: Option<Vec<String>>,
    /// Collection ids to add every filename to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub add_to_collections: Option<Vec<String>>,
    /// Collection ids to remove every filename from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remove_from_collections: Option<Vec<String>>,
}

/// One title assignment inside a bulk organization request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GalleryTitleAssignment {
    pub filename: String,
    /// Empty clears the title.
    pub title: String,
}

/// Capability-gated bulk gallery mutation. `operation_id` is a client-owned
/// replay key; current mutations are idempotent and servers retain receipts.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GalleryBulkMutationRequest {
    pub operation_id: String,
    #[serde(default)]
    pub filenames: Vec<String>,
    #[serde(default)]
    pub titles: Vec<GalleryTitleAssignment>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub favorite: Option<bool>,
    #[serde(default)]
    pub add_tags: Vec<String>,
    #[serde(default)]
    pub remove_tags: Vec<String>,
    /// Ensure this collection name/slug exists, then add every filename.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub add_to_collection: Option<CollectionRef>,
    /// Collection slug to remove every filename from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remove_from_collection_slug: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GalleryBulkMutationResult {
    pub operation_id: String,
    pub changed: u64,
    pub revision: u64,
}

/// Idempotent heterogeneous generation admission. Every child is validated
/// before any child is persisted; execution remains independent afterwards.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GenerationBatchAdmissionRequest {
    pub client_batch_id: String,
    pub requests: Vec<GenerateRequest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum GenerationBatchChildState {
    Accepted,
    Cancelling,
    Running,
    Complete,
    Failed,
    Cancelled,
    Held,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GenerationBatchChild {
    pub index: u32,
    pub job_id: String,
    pub state: GenerationBatchChildState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Typed cause of a `held` child — the preparation refusal's own code
    /// (`MODEL_NOT_FOUND`, `UNKNOWN_MODEL`, …) beside its sentence, so a
    /// client can offer the pull-and-resume instead of matching prose.
    /// Absent for a hold with no typed cause and for every other state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    /// True only when an explicitly held durable child may be returned to the
    /// queue through the retry endpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retryable: Option<bool>,
    /// Unix-epoch milliseconds when the durable child was admitted.
    #[serde(default)]
    pub created_at_ms: i64,
    /// Unix-epoch milliseconds of the latest authoritative state transition.
    #[serde(default)]
    pub updated_at_ms: i64,
    /// Monotonic per-child version, incremented by every authoritative state
    /// transition and by nothing else.
    ///
    /// This is the ordering authority clients compare to decide whether an
    /// incoming snapshot supersedes the one they hold. `updated_at_ms` cannot
    /// serve that role: several transitions commit inside one millisecond
    /// routinely, and `POST /api/queue/{id}/retry` moves a child BACKWARD
    /// through the client's forward-phase ordering (held -> accepted), so a
    /// same-millisecond collision decides whether the retry is visible at all.
    ///
    /// Additive: a server that predates it sends nothing and every client
    /// deserializes `0`, which reads as "no revision authority" and falls back
    /// to the timestamp comparison. A client MUST NOT treat `0` as a real
    /// revision — rows created before migration v29 also sit at `0` until
    /// their next transition.
    #[serde(default)]
    pub revision: u64,
    /// Unix-epoch milliseconds when the child reached a terminal state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at_ms: Option<i64>,
    /// Structured terminal failure/cancellation details. `error` remains for
    /// compatibility with clients that only understand the legacy string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Option<Object>)]
    pub terminal_error: Option<serde_json::Value>,
    /// Durable gallery identities produced by a completed child.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<GenerationBatchResult>,
}

/// What a completed child produced, recorded once at settlement.
///
/// The terminal facts beside the filenames are the ones a caller cannot
/// recover from the gallery alone at the moment it needs them — the seed it
/// must advance from, how long the render took, and which accelerator ran it.
/// They are additive: a child settled before they existed, and the
/// committed-archive replay path (which knows only the filenames), leave them
/// absent rather than reporting a fabricated zero.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, Default)]
pub struct GenerationBatchResult {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_filename: Option<String>,
    /// The seed the render actually used, including a server-chosen one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_time_ms: Option<u64>,
    /// Index of the accelerator that ran it, the same ordinal
    /// [`GenerateResponse::gpu`] and the SSE complete event report.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GenerationBatchStatus {
    pub id: String,
    pub client_batch_id: String,
    /// Exact serving instance. Clients fence cached lifecycle state when this
    /// differs from the instance that admitted the request.
    #[serde(default)]
    pub instance_id: String,
    /// Always true on this authoritative reconnectable status surface.
    #[serde(default)]
    pub durable: bool,
    pub children: Vec<GenerationBatchChild>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GenerationBatchStatusRequest {
    #[serde(default)]
    pub client_batch_ids: Vec<String>,
    #[serde(default)]
    pub batch_ids: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GenerationBatchMissing {
    pub client_batch_ids: Vec<String>,
    pub batch_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GenerationBatchStatusResponse {
    pub instance_id: String,
    pub batches: Vec<GenerationBatchStatus>,
    pub missing: GenerationBatchMissing,
}

/// Complete authority required to retry one held durable generation child.
/// The route job id is repeated here so the server can reject a mismatched
/// path/body pair before entering the transactional lifecycle mutation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GenerationRetryRequest {
    pub instance_id: String,
    pub batch_id: String,
    pub client_batch_id: String,
    pub job_id: String,
}

impl GenerationRetryRequest {
    pub fn from_authority(authority: &GenerationBatchAuthority, job_id: impl Into<String>) -> Self {
        Self {
            instance_id: authority.instance_id.clone(),
            batch_id: authority.batch_id.clone(),
            client_batch_id: authority.client_batch_id.clone(),
            job_id: job_id.into(),
        }
    }
}

/// Immutable identity captured from canonical generation admission. Rust
/// clients validate every later snapshot against this fence before merging
/// lifecycle state or acting on a returned job id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationBatchAuthority {
    pub instance_id: String,
    pub batch_id: String,
    pub client_batch_id: String,
}

impl GenerationBatchAuthority {
    pub fn from_admission(
        status: &GenerationBatchStatus,
        expected_client_batch_id: &str,
    ) -> Result<Self, String> {
        if !status.durable || status.instance_id.is_empty() || status.id.is_empty() {
            return Err("generation batch admission did not return durable identity".to_string());
        }
        if status.client_batch_id != expected_client_batch_id {
            return Err(format!(
                "generation batch admission returned client id {}, expected {expected_client_batch_id}",
                status.client_batch_id
            ));
        }
        Ok(Self {
            instance_id: status.instance_id.clone(),
            batch_id: status.id.clone(),
            client_batch_id: status.client_batch_id.clone(),
        })
    }

    pub fn validate_status(&self, status: &GenerationBatchStatus) -> Result<(), String> {
        if status.instance_id != self.instance_id {
            return Err(format!(
                "generation batch instance changed from {} to {}",
                self.instance_id, status.instance_id
            ));
        }
        if status.id != self.batch_id {
            return Err(format!(
                "generation batch id changed from {} to {}",
                self.batch_id, status.id
            ));
        }
        if status.client_batch_id != self.client_batch_id {
            return Err(format!(
                "generation batch client id changed from {} to {}",
                self.client_batch_id, status.client_batch_id
            ));
        }
        if !status.durable {
            return Err("generation batch lost its durable authority".to_string());
        }
        Ok(())
    }
}

/// Validate one bulk lifecycle snapshot against the complete accepted set.
/// Each authority must appear exactly once, either as a batch or as an
/// explicit missing client or batch id; foreign and duplicate identities are
/// rejected.
pub fn validate_generation_batch_status_response(
    response: &GenerationBatchStatusResponse,
    authorities: &[GenerationBatchAuthority],
) -> Result<(), String> {
    if authorities.is_empty() {
        return Err("generation batch authority set is empty".to_string());
    }
    if authorities
        .iter()
        .any(|authority| authority.instance_id != response.instance_id)
    {
        return Err(format!(
            "generation batch reconciliation returned unexpected instance {}",
            response.instance_id
        ));
    }
    let mut seen = std::collections::HashSet::new();
    for batch in &response.batches {
        let authority = authorities
            .iter()
            .find(|authority| authority.client_batch_id == batch.client_batch_id)
            .ok_or_else(|| {
                format!(
                    "generation batch reconciliation returned foreign client id {}",
                    batch.client_batch_id
                )
            })?;
        authority.validate_status(batch)?;
        if !seen.insert(batch.client_batch_id.as_str()) {
            return Err(format!(
                "generation batch reconciliation duplicated client id {}",
                batch.client_batch_id
            ));
        }
    }
    for client_batch_id in &response.missing.client_batch_ids {
        if !authorities
            .iter()
            .any(|authority| authority.client_batch_id == *client_batch_id)
        {
            return Err(format!(
                "generation batch reconciliation marked foreign client id {client_batch_id} missing"
            ));
        }
        if !seen.insert(client_batch_id.as_str()) {
            return Err(format!(
                "generation batch reconciliation duplicated client id {client_batch_id}"
            ));
        }
    }
    for batch_id in &response.missing.batch_ids {
        let authority = authorities
            .iter()
            .find(|authority| authority.batch_id == *batch_id)
            .ok_or_else(|| {
                format!(
                    "generation batch reconciliation marked foreign batch id {batch_id} missing"
                )
            })?;
        if !seen.insert(authority.client_batch_id.as_str()) {
            return Err(format!(
                "generation batch reconciliation duplicated batch identity {batch_id}"
            ));
        }
    }
    if seen.len() != authorities.len() {
        let omitted = authorities
            .iter()
            .filter(|authority| !seen.contains(authority.client_batch_id.as_str()))
            .map(|authority| authority.client_batch_id.as_str())
            .collect::<Vec<_>>();
        return Err(format!(
            "generation batch reconciliation omitted client ids: {}",
            omitted.join(", ")
        ));
    }
    Ok(())
}

/// Body of `POST /api/gallery/collections`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct CollectionCreateRequest {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Body of `PATCH /api/gallery/collections/:id`. Absent fields are untouched.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, utoipa::ToSchema)]
pub struct CollectionUpdateRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cover_filename: Option<String>,
    /// Hide/show this collection's members in the default Library and search.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hidden: Option<bool>,
}

/// Body of `PUT /api/gallery/collections/:id/items` — gallery filenames to
/// add to / remove from the collection.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, utoipa::ToSchema)]
pub struct CollectionItemsRequest {
    #[serde(default)]
    pub add: Vec<String>,
    #[serde(default)]
    pub remove: Vec<String>,
}

/// Body of `POST /api/gallery/trash` and `POST /api/gallery/trash/restore`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, utoipa::ToSchema)]
pub struct TrashFilenamesRequest {
    pub filenames: Vec<String>,
}

/// Result of `POST /api/gallery/trash/sweep`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, utoipa::ToSchema)]
pub struct TrashSweepResult {
    /// Trashed prints purged because they exceeded retention.
    pub purged: u64,
    /// Trashed prints still waiting for their purge date.
    pub remaining: u64,
}

/// Result of `POST /api/queue/held/sweep` (durable held-row retention).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, utoipa::ToSchema)]
pub struct HeldSweepResult {
    /// Held rows purged because they exceeded `queue.held_retention_days`.
    pub purged: u64,
    /// Held rows still inside their retention window.
    pub remaining: u64,
    /// Purged rows whose encrypted media could not be collected in this pass.
    /// Their obligation is already `gc_pending`, so startup reconciliation
    /// still owns them — reported rather than hidden.
    #[serde(default)]
    pub media_deferred: u64,
}

/// Result of `POST /api/generation-batches/sweep` (settled-batch retention).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, utoipa::ToSchema)]
pub struct SettledBatchSweepResult {
    /// Fully settled batches purged because their newest child settlement
    /// exceeded `queue.held_retention_days`.
    pub purged: u64,
    /// Fully settled batches still inside their retention window.
    pub remaining: u64,
}

/// Result of `DELETE /api/gallery/trash` (empty trash).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, utoipa::ToSchema)]
pub struct EmptyTrashResult {
    pub purged: u64,
}

/// Body of `PATCH /api/gallery/tags/:name`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct TagRenameRequest {
    pub name: String,
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
    /// Trash support. Absent on older servers (delete is permanent there)
    /// and when the metadata DB is disabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trash: Option<GalleryTrashCapabilities>,
    /// Whether titles, favorites, tags, and collections can be edited on
    /// this server (`PATCH /api/gallery/image/:filename`,
    /// `POST /api/gallery/organize`, `/api/gallery/collections`,
    /// `/api/gallery/tags`). False on older servers and when the metadata
    /// DB is disabled.
    #[serde(default, skip_serializing_if = "is_false")]
    pub organize: bool,
    /// `POST /api/gallery/mutations` and bulk permanent deletion are
    /// available. Absent means clients use the legacy routes.
    #[serde(default, skip_serializing_if = "is_false")]
    pub bulk_mutations: bool,
    /// Gallery rows include `media_version`, and thumbnail responses expose
    /// validators derived from the same file identity.
    #[serde(default, skip_serializing_if = "is_false")]
    pub media_version: bool,
    /// Gallery list and thumbnail endpoints support `If-None-Match`.
    #[serde(default, skip_serializing_if = "is_false")]
    pub conditional_get: bool,
    /// Gallery SSE add/update events carry complete rows, allowing clients to
    /// update in place instead of polling the complete library.
    #[serde(default, skip_serializing_if = "is_false")]
    pub row_events: bool,
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

/// Restart-safe encrypted request-media support for the durable generation
/// queue. Presence of this record is the availability signal: servers must
/// omit it until the owner-scoped media store has passed startup validation
/// and reconciliation for the queue owner they actually claimed.
///
/// The version is intentionally independent from [`QueueCapabilities`]. A
/// client may continue using ordinary media-free durability when this record
/// is absent, while requiring an exact media protocol before submitting a
/// request whose replay depends on captured bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DurableMediaCapabilities {
    pub protocol_version: u32,
    pub encrypted_at_rest: bool,
    pub generate_request_media: bool,
    pub identity: bool,
    pub private_h3: bool,
}

impl DurableMediaCapabilities {
    /// The first queue-media wire contract. MiniMax H3 remains outside this
    /// authority: encrypted bytes do not make an authenticated ingress grant
    /// replayable.
    pub const fn v1() -> Self {
        Self {
            protocol_version: 1,
            encrypted_at_rest: true,
            generate_request_media: true,
            identity: true,
            private_h3: false,
        }
    }

    /// Queue-media V2 can add authenticated, owner/job/request-bound replay
    /// for private H3 admission when that adapter is compiled into the serving
    /// binary. Ordered H3 references ride the same encrypted media set as
    /// every other request media and need no separate bit.
    pub const fn v2(private_h3: bool) -> Self {
        Self {
            protocol_version: 2,
            encrypted_at_rest: true,
            generate_request_media: true,
            identity: true,
            private_h3,
        }
    }
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
    /// Queue pins may use opaque stable device IDs.
    #[serde(default)]
    pub stable_device_pins: bool,
    /// Server can cooperatively cancel already-running work at model safe
    /// points. Older servers omit this and clients keep running rows read-only.
    #[serde(default)]
    pub cooperative_cancellation: bool,
    /// A queued job survives a server restart and is replayed automatically.
    /// Clients that see this must stop dead-lettering a queued job whose
    /// stream died — on this host that job is still going to run.
    #[serde(default)]
    pub durable_queue: bool,
    /// How many singleton children one `POST /api/generation-batches`
    /// operation accepts. Present exactly when this host generates at all —
    /// there is one admission path, so its absence means the host refuses
    /// generation rather than that it offers an older protocol.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heterogeneous_batch_max_outputs: Option<u32>,
}

/// Authenticated, stable-URL reference-media ingress advertised by current
/// servers. Handles remain bearer secrets and therefore travel only in the
/// named headers, never in URLs, logs, or durable request metadata.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceUploadCapabilities {
    /// The request-bound upload protocol is offered exactly when API-key auth
    /// is enabled; a host without it accepts validated inline references.
    pub available: bool,
    pub protocol_version: u32,
    pub requires_api_key: bool,
    pub session_path: String,
    pub upload_path: String,
    pub session_handle_header: String,
    pub upload_handle_header: String,
    pub max_file_bytes: u64,
    pub max_session_bytes: u64,
    /// Open sessions one API-key identity may hold at once. A durable batch
    /// takes one lease per reference-bearing sibling BEFORE it POSTs, so a
    /// client chunks such batches to this many siblings.
    pub max_active_sessions: u32,
    pub session_ttl_ms: u64,
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
    /// Subject-preserving Remix is available through `POST /api/remix`.
    #[serde(default)]
    pub remix: bool,
    /// The manifest model local expansion resolves. Additive: clients that
    /// see it stop hard-coding `qwen3-expand` when offering to pull it.
    /// Absent for API backends and for servers that predate the field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Face-identity conditioning shapes this server understands.
///
/// This block exists because the alternative is silent accept-and-ignore. A
/// server that predates a field simply DROPS it: send `id_images` to one and it
/// renders with no identity at all, send `true_cfg` and it renders the
/// distilled path at a guidance value chosen for a branch that never ran. Both
/// are prints of the wrong thing with nothing to say so, which is exactly what
/// `crate::identity` refuses everywhere else.
///
/// Absence therefore means NO, never unknown — an older server omits the whole
/// block, so a client that needs either shape must refuse rather than submit.
/// The singular `id_image` form predates this and needs no gate: every server
/// that accepts identity at all understands it.
///
/// Every value is derived from `crate::identity`'s own constants, so this can
/// never advertise a bound the validator does not enforce.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentityCapabilities {
    /// The server understands `id_images` / `id_image_names` and averages the
    /// set into one identity.
    pub multi_photo: bool,
    /// Maximum photographs one request may carry. Meaningless — and always
    /// `crate::identity::ID_IMAGES_MAX` — when `multi_photo` is true.
    pub max_photos: u32,
    /// The server understands `true_cfg` / `cfg_start_step` and runs the PuLID
    /// negative branch.
    pub true_cfg: bool,
}

impl IdentityCapabilities {
    /// What a build that can execute identity conditioning advertises.
    ///
    /// One constructor, read straight from the contract module, so the
    /// advertisement and the validator cannot drift.
    pub fn advertised() -> Self {
        Self {
            multi_photo: true,
            max_photos: crate::identity::ID_IMAGES_MAX as u32,
            true_cfg: true,
        }
    }
}

// ── Device inventory ────────────────────────────────────────────────────────

/// Runtime-visible device classification. CUDA ordinals are deliberately not
/// encoded here: clients must treat [`DeviceInfo::id`] as the durable,
/// backend-qualified identity and `ordinal` as a process-local display hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum DeviceKind {
    FullGpu,
    Mig,
    UnknownCuda,
    Metal,
}

/// Administrator-requested lifecycle state. Phase A exposes this state
/// read-only; lifecycle mutation lands in a later phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum DeviceAdminState {
    StartupExcluded,
    Starting,
    Enabled,
    Draining,
    Disabled,
}

/// Administrative enablement mutation accepted by
/// `PATCH /api/devices/{stable-id}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct DeviceMutationRequest {
    pub enabled: bool,
}

impl DeviceAdminState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StartupExcluded => "startup_excluded",
            Self::Starting => "starting",
            Self::Enabled => "enabled",
            Self::Draining => "draining",
            Self::Disabled => "disabled",
        }
    }
}

/// Transient device health. Health is never persisted as a user preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum DeviceHealth {
    Healthy,
    Degraded,
    Unavailable,
    Poisoned,
}

impl DeviceHealth {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Unavailable => "unavailable",
            Self::Poisoned => "poisoned",
        }
    }
}

/// Current worker activity, orthogonal to administrative and health state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum DeviceActivity {
    Idle,
    Loading,
    Generating,
    Upscaling,
    AdminLoading,
    Stopping,
}

/// Device memory snapshot. Unsupported or unavailable operational values are
/// explicit JSON `null`, never fabricated zeroes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct DeviceMemoryInfo {
    pub total_bytes: Option<u64>,
    pub used_bytes: Option<u64>,
    pub mold_used_bytes: Option<u64>,
    pub other_used_bytes: Option<u64>,
}

/// Optional operational telemetry from the server's background sampler.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct DeviceTelemetry {
    pub utilization_percent: Option<u8>,
    pub temperature_c: Option<f32>,
    pub power_w: Option<f32>,
}

/// One runtime-visible compute device returned by `GET /api/devices`.
///
/// Stable IDs are opaque strings such as
/// `cuda:0123456789abcdef0123456789abcdef` or `metal:default`. Consumers must
/// URL-encode them and must not parse their components.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct DeviceInfo {
    pub id: String,
    pub backend: GpuBackend,
    pub ordinal: Option<usize>,
    pub device_kind: DeviceKind,
    pub nvml_uuid: Option<String>,
    pub physical_uuid: Option<String>,
    pub mig_uuid: Option<String>,
    pub mig_parent_uuid: Option<String>,
    pub mig_profile: Option<String>,
    pub name: String,
    pub pci_bus_id: Option<String>,
    pub compute_capability: Option<String>,
    pub memory: DeviceMemoryInfo,
    pub telemetry: DeviceTelemetry,
    pub desired_enabled: bool,
    /// The requested preference is persisted, but a server restart is needed
    /// before this device can become schedulable.
    #[serde(default)]
    pub restart_required: bool,
    pub admin_state: DeviceAdminState,
    pub health: DeviceHealth,
    pub activity: DeviceActivity,
    pub schedulable: bool,
    pub unschedulable_reason: Option<String>,
    pub loaded_models: Vec<String>,
    pub active_work_id: Option<String>,
    pub planned_work_ids: Vec<String>,
}

/// Read-only device inventory response. `plan_version` remains zero until the
/// versioned scheduler plan is introduced; keeping the field now avoids a
/// later response-shape fork.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct DeviceState {
    pub devices: Vec<DeviceInfo>,
    pub plan_version: u64,
}

/// Device API feature detection for clients talking to mixed-version hosts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// `GET /api/devices` is present.
    pub available: bool,
    /// Runtime enable/disable lifecycle mutation is currently authoritative.
    /// This is `true` only while scheduler V2 owns dispatch; legacy, observe,
    /// maintenance, and otherwise unavailable runtimes report `false`.
    pub lifecycle: bool,
    /// A disabled device can be persistently enabled for the next restart
    /// even though live lifecycle mutation is not authoritative.
    #[serde(default)]
    pub restart_enable: bool,
    /// Queue pins may use stable device IDs.
    #[serde(default)]
    pub stable_pins: bool,
    /// Queue snapshots include versioned per-device lanes.
    #[serde(default)]
    pub planned_lanes: bool,
    /// ETAs may be backed by persisted runtime observations.
    #[serde(default)]
    pub learned_eta: bool,
}

/// Restart-time GPU dispatch rollout state.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DispatchCapabilities {
    /// Modes accepted by `MOLD_DISPATCH_MODE`.
    pub modes: Vec<String>,
    /// Active mode for this process. Missing on older servers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_mode: Option<String>,
    /// V2 is the authoritative dispatcher and lease owner.
    pub v2_authoritative: bool,
    /// Legacy owns dispatch while read-only V2 decisions are recorded.
    pub observes_v2_decisions: bool,
    /// The server can run the authoritative scheduler against an exact,
    /// read-only request preview without reserving or enqueueing work.
    #[serde(default)]
    pub request_placement_preview: bool,
}

/// Presentation-only state for one authenticated MiniMax H3 component.
///
/// These facts never authorize a download or loader. The private generation
/// path independently authenticates every byte before admission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct MiniMaxH3ComponentCapability {
    pub id: String,
    pub display_name: String,
    pub kind: String,
    pub role: String,
    pub scope: String,
    pub size_bytes: u64,
    pub state: String,
}

/// One exact task partition implemented by the authenticated private runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct MiniMaxH3PartitionCapability {
    pub task: String,
    pub model: String,
    pub display_name: String,
    pub runtime_available: bool,
    pub tier: String,
    pub component_ids: Vec<String>,
    /// Exact request envelope admitted by this reviewed runtime partition.
    /// This is presentation authority only; private admission independently
    /// authenticates and validates the retained qualification record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<MiniMaxH3RequestCapability>,
    /// Reviewed Turbo LoRA variants of this partition (additive; absent on
    /// older servers). Deliberately not extra `partitions` entries — clients
    /// that predate this field parse exactly one partition per task.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub turbo: Vec<MiniMaxH3TurboVariantCapability>,
}

/// One reviewed Turbo LoRA variant of a compact H3 partition.
///
/// The variant is the same base component stack plus one pinned adapter, so
/// it deliberately does not repeat the partition's `component_ids`; only the
/// adapter's own install state and the tier's reviewed request envelope are
/// carried.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct MiniMaxH3TurboVariantCapability {
    pub model: String,
    pub display_name: String,
    /// Human-facing tier label ("Turbo 8-step").
    pub tier: String,
    pub adapter_size_bytes: u64,
    /// Whether the pinned adapter file is already landed on this host.
    pub installed: bool,
    /// The tier's reviewed request envelope; present only when this variant
    /// is executable (base partition and adapter installed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<MiniMaxH3RequestCapability>,
}

/// Exact client-visible request shape for one reviewed H3 task partition.
///
/// The initial compact FL2VA authority is intentionally a single quality
/// point. Keeping every axis explicit prevents a model picker from widening a
/// narrow runtime qualification into the family's larger theoretical range.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct MiniMaxH3RequestCapability {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
    pub steps: u32,
    pub batch_size: u32,
    pub output_format: String,
    pub required_endpoint: String,
    /// Content address of the sole generation profile clients may use for
    /// this narrow partition.
    pub generation_profile_sha256: String,
}

/// Hardware and implementation boundary for one private H3 capability record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct MiniMaxH3QualificationCapability {
    pub backend: String,
    pub metal_supported: bool,
    pub minimum_host_ram_bytes: u64,
    pub minimum_vram_bytes: u64,
    pub attention_profile: String,
    pub quantization_profile: String,
}

/// Additive host-authored MiniMax H3 inventory.
///
/// This record is deliberately separate from model manifests and catalog
/// recipes. It can reveal an already reviewed private partition but cannot
/// create install or public runtime authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct MiniMaxH3Capability {
    pub runtime_available: bool,
    pub qualification: MiniMaxH3QualificationCapability,
    pub partitions: Vec<MiniMaxH3PartitionCapability>,
    pub components: Vec<MiniMaxH3ComponentCapability>,
}

/// Capabilities payload returned by `GET /api/capabilities`. Grouping keeps
/// the shape extensible — future areas (inpainting, upscaling modes, etc.)
/// can add their own sub-structs without churning existing fields.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServerCapabilities {
    /// `/api/models` rows carry a complete version-1 generation profile.
    /// Absent/false identifies a legacy host whose flattened fields require
    /// the contained one-release client adapter.
    #[serde(default)]
    pub generation_profile_v1: bool,
    pub gallery: GalleryCapabilities,
    pub catalog: CatalogCapabilities,
    /// Explicit model-family restrictions enforced by this server. Absent on
    /// older servers, where clients must still trust server-side rejection.
    #[serde(default)]
    pub model_access: crate::ModelAccessCapabilities,
    /// Present only on an authenticated private-UAT server whose exact runtime
    /// partition has a source-controlled reviewed qualification record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimax_h3: Option<MiniMaxH3Capability>,
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
    /// Restart-safe encrypted request-media support for the durable queue.
    /// Absence means unavailable; servers must keep this dark until the full
    /// admission, reconciliation, hydration, and cleanup lifecycle is live.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub durable_media: Option<DurableMediaCapabilities>,
    /// Absent on older servers. Availability advertises the ingress protocol,
    /// not MiniMax H3 model/license activation; model_access remains the
    /// authority for whether a request may run.
    #[serde(default)]
    pub reference_uploads: ReferenceUploadCapabilities,
    /// Absent on older servers. Missing means the read-only device resource
    /// and lifecycle controls are unavailable. `devices.lifecycle` is a
    /// runtime authority flag, not merely endpoint presence.
    #[serde(default)]
    pub devices: DeviceCapabilities,
    /// Absent on older servers. Dispatch mode changes require a restart.
    #[serde(default)]
    pub dispatch: DispatchCapabilities,
    /// Absent on older servers. Unlike default-false capability groups,
    /// absence here means unknown so newer clients may still try expansion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expand: Option<ExpandCapabilities>,
    /// Absent on older servers, and absence means NO — see
    /// [`IdentityCapabilities`]. A client that wants several photographs or
    /// true CFG must refuse rather than submit into a silent drop.
    #[serde(default)]
    pub identity: IdentityCapabilities,
    /// This server exposes `GET /api/licenses` and honours `accept_licenses`
    /// on its download routes.
    ///
    /// Absent (false) on older servers, where a restricted model can only be
    /// accepted by running `mold pull --accept-license` in a shell ON that
    /// host — so a UI must not offer an in-app acceptance flow it cannot
    /// deliver.
    #[serde(default)]
    pub licenses: bool,
}

/// Why a host cannot admit a particular set of requests.
///
/// A refusal names the request trait the host does not carry, because that is
/// the only thing a caller can act on. Host-level absence is
/// [`Self::GenerationUnavailable`]: there is ONE admission path, so a host
/// that does not advertise it does not generate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonicalRefusal {
    /// No requests were supplied.
    Empty,
    /// This host advertises no generation admission at all.
    GenerationUnavailable,
    /// The host advertises a zero-output limit.
    ZeroOutputLimit,
}

impl std::fmt::Display for CanonicalRefusal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(formatter, "no requests were supplied"),
            Self::GenerationUnavailable => {
                write!(formatter, "this host does not admit generation")
            }
            Self::ZeroOutputLimit => write!(formatter, "this host admits zero outputs per batch"),
        }
    }
}

impl std::error::Error for CanonicalRefusal {}

impl ServerCapabilities {
    /// Exact shared Rust-client gate for canonical durable Batch N admission.
    /// Returns the host's per-operation child limit, or the named reason the
    /// requests are not representable. Gates on the MACHINE alone — the
    /// durable queue — exactly as `studio/lib/generationSubmissionPolicy.ts`
    /// does; the server's typed refusal is the authority for the request.
    pub fn canonical_generation_batch_limit(
        &self,
        requests: &[GenerateRequest],
    ) -> Result<usize, CanonicalRefusal> {
        if requests.is_empty() {
            return Err(CanonicalRefusal::Empty);
        }
        let limit = self
            .queue
            .heterogeneous_batch_max_outputs
            .and_then(|limit| usize::try_from(limit).ok())
            .ok_or(CanonicalRefusal::GenerationUnavailable)?;
        if limit == 0 {
            return Err(CanonicalRefusal::ZeroOutputLimit);
        }
        // Deliberately blind to the request: the durable protocol carries
        // media, LoRAs, identity photos, and H3, and the server's own typed
        // admission refusal is the single authority for anything it cannot
        // take. A client-side per-trait fence could only ever refuse work the
        // host would have accepted, and diverge from web/desktop/iPhone,
        // which gate on the machine alone.
        let _ = requests;
        Ok(limit)
    }
}

/// One third-party model license and this server's acceptance state for it.
///
/// The response element of `GET /api/licenses`. Acceptance is per Mold data
/// root, so this is always the answer for the host that served it — a client
/// holding several hosts must ask each one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ThirdPartyLicenseStatus {
    /// Stable id used by `accept_licenses` and `mold pull --accept-license`.
    pub id: String,
    /// Human-readable license name.
    pub name: String,
    /// Immutable, commit-pinned URL of the exact license text. Together with
    /// `sha256` this is the identity an acceptance is bound to.
    pub url: String,
    /// Browsable page for the project's current terms. Presentation only —
    /// deliberately not part of the accepted identity, because its contents
    /// move.
    pub canonical: String,
    /// SHA-256 of the text served at `url`, verified when the pin landed.
    pub sha256: String,
    /// One-sentence statement of the restriction being accepted.
    pub summary: String,
    /// Whether THIS server has a current acceptance on record. A record bound
    /// to a superseded `(url, sha256)` pair reads as `false`.
    pub accepted: bool,
    /// Manifest names that cannot be downloaded until this is accepted.
    #[serde(default)]
    pub required_by: Vec<String>,
}

/// Response body of `GET /api/licenses`.
///
/// An object rather than a bare array so later fields (paging, a server-wide
/// policy note) stay additive.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct LicenseListing {
    pub licenses: Vec<ThirdPartyLicenseStatus>,
}

/// One license the user accepted, carrying the EXACT terms they were shown.
///
/// Deliberately not a bare id string. Acceptance is recorded on whichever
/// machine runs the download, and that machine may be on a different Mold
/// release with a different pinned revision of the same license. An id alone
/// would let it resolve terms of its own choosing and record consent for text
/// the user never read — the server must be able to prove it is storing
/// agreement to the document that was actually displayed, so the identity
/// travels with the id and a mismatch is refused rather than reconciled.
///
/// There is no bare-string compatibility shape on purpose: accepting one would
/// reintroduce exactly the hole this struct closes, and no client has shipped
/// against the id-only form.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct LicenseAcceptance {
    /// Stable license id.
    pub id: String,
    /// Immutable, commit-pinned URL of the terms the user was shown.
    pub url: String,
    /// SHA-256 of the text at `url`, as displayed.
    pub sha256: String,
}

impl LicenseAcceptance {
    /// True when `self` names the same document as `(url, sha256)`.
    ///
    /// Digest comparison is case-insensitive because hex casing is not part of
    /// the identity; the URL must match exactly.
    pub fn matches(&self, url: &str, sha256: &str) -> bool {
        self.url == url && self.sha256.eq_ignore_ascii_case(sha256)
    }
}

/// Error code for an acceptance whose terms are not the ones this server pins.
///
/// Distinct from [`LICENSE_NOT_ACCEPTED`]: nothing is missing, the two sides
/// simply disagree about what the license says. The refusal carries the
/// server's own terms so a client can display those and retry.
pub const LICENSE_TERMS_MISMATCH: &str = "LICENSE_TERMS_MISMATCH";

/// The machine-readable half of a `LICENSE_NOT_ACCEPTED` refusal.
///
/// Rides the error body beside the human message so a UI can render its own
/// acceptance prompt — name, terms links, and the id to send back in
/// `accept_licenses` — instead of scraping prose.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct LicenseRefusal {
    pub id: String,
    pub name: String,
    pub url: String,
    pub canonical: String,
    /// SHA-256 of the text at `url`. Lets a client that was refused for a
    /// terms mismatch re-display and re-send exactly what this server pins.
    pub sha256: String,
    pub summary: String,
}

/// Error code carried by a download refused for a missing license acceptance.
///
/// Clients match on this rather than on the message text.
pub const LICENSE_NOT_ACCEPTED: &str = "LICENSE_NOT_ACCEPTED";

#[cfg(test)]
mod device_types_tests {
    use super::*;

    #[test]
    fn device_state_serializes_nullable_telemetry_and_stable_enums() {
        let state = DeviceState {
            devices: vec![DeviceInfo {
                id: "cuda:0123456789abcdef0123456789abcdef".into(),
                backend: GpuBackend::Cuda,
                ordinal: Some(0),
                device_kind: DeviceKind::FullGpu,
                nvml_uuid: Some("GPU-01234567-89ab-cdef-0123-456789abcdef".into()),
                physical_uuid: Some("GPU-01234567-89ab-cdef-0123-456789abcdef".into()),
                mig_uuid: None,
                mig_parent_uuid: None,
                mig_profile: None,
                name: "NVIDIA GeForce RTX 3090".into(),
                pci_bus_id: Some("00000000:01:00.0".into()),
                compute_capability: Some("8.6".into()),
                memory: DeviceMemoryInfo {
                    total_bytes: Some(24_000_000_000),
                    used_bytes: None,
                    mold_used_bytes: None,
                    other_used_bytes: None,
                },
                telemetry: DeviceTelemetry {
                    utilization_percent: None,
                    temperature_c: None,
                    power_w: None,
                },
                desired_enabled: true,
                restart_required: false,
                admin_state: DeviceAdminState::Enabled,
                health: DeviceHealth::Healthy,
                activity: DeviceActivity::Idle,
                schedulable: true,
                unschedulable_reason: None,
                loaded_models: vec![],
                active_work_id: None,
                planned_work_ids: vec![],
            }],
            plan_version: 0,
        };

        let json = serde_json::to_value(&state).unwrap();
        assert_eq!(json["devices"][0]["device_kind"], "full_gpu");
        assert_eq!(json["devices"][0]["admin_state"], "enabled");
        assert_eq!(json["devices"][0]["health"], "healthy");
        assert_eq!(json["devices"][0]["activity"], "idle");
        assert_eq!(
            json["devices"][0]["telemetry"]["utilization_percent"],
            serde_json::Value::Null
        );
        assert_eq!(
            json["devices"][0]["memory"]["used_bytes"],
            serde_json::Value::Null
        );

        let round_trip: DeviceState = serde_json::from_value(json).unwrap();
        assert_eq!(round_trip.devices[0].device_kind, DeviceKind::FullGpu);
        assert_eq!(
            round_trip.devices[0].memory.total_bytes,
            Some(24_000_000_000)
        );
    }

    #[test]
    fn old_capabilities_default_device_api_to_unavailable() {
        let caps: ServerCapabilities = serde_json::from_str(
            r#"{"gallery":{"can_delete":true},"catalog":{"available":false,"families":[]}}"#,
        )
        .unwrap();
        assert!(!caps.devices.available);
        assert!(!caps.devices.lifecycle);
        assert!(caps.dispatch.modes.is_empty());
        assert!(caps.dispatch.active_mode.is_none());
        assert!(!caps.dispatch.v2_authoritative);
        assert!(!caps.dispatch.observes_v2_decisions);
        assert!(caps.model_access.restrictions.is_empty());
        assert!(!caps.reference_uploads.available);
        assert_eq!(caps.reference_uploads.protocol_version, 0);
    }
}

#[cfg(test)]
mod minimax_h3_capability_tests {
    use super::*;

    #[test]
    fn ordinary_capabilities_omit_private_h3_inventory() {
        let value = serde_json::to_value(ServerCapabilities::default()).unwrap();
        assert!(value.get("minimax_h3").is_none());
    }

    #[test]
    fn private_h3_inventory_serializes_the_single_task_graph() {
        let capability = MiniMaxH3Capability {
            runtime_available: true,
            qualification: MiniMaxH3QualificationCapability {
                backend: "cuda".into(),
                metal_supported: false,
                minimum_host_ram_bytes: 128,
                minimum_vram_bytes: 40,
                attention_profile: "reviewed attention".into(),
                quantization_profile: "reviewed compact layout".into(),
            },
            partitions: vec![MiniMaxH3PartitionCapability {
                task: "fl2va".into(),
                model: crate::minimax_h3::FL2VA_COMFY.into(),
                display_name: "MiniMax H3 FL2VA".into(),
                runtime_available: true,
                tier: "Compact".into(),
                component_ids: vec!["transformer".into()],
                request: Some(MiniMaxH3RequestCapability {
                    width: crate::minimax_h3::DEFAULT_WIDTH,
                    height: crate::minimax_h3::DEFAULT_HEIGHT,
                    frames: crate::minimax_h3::REVIEWED_COMPACT_FRAMES,
                    fps: crate::minimax_h3::FIXED_FPS,
                    steps: crate::minimax_h3::COMFY_DEFAULT_STEPS,
                    batch_size: 1,
                    output_format: "mp4".into(),
                    required_endpoint: "first".into(),
                    generation_profile_sha256: "a".repeat(64),
                }),
                turbo: Vec::new(),
            }],
            components: vec![MiniMaxH3ComponentCapability {
                id: "transformer".into(),
                display_name: "FL2VA transformer".into(),
                kind: "checkpoint".into(),
                role: "transformer".into(),
                scope: "fl2va".into(),
                size_bytes: 20,
                state: "installed".into(),
            }],
        };
        let server = ServerCapabilities {
            minimax_h3: Some(capability),
            ..ServerCapabilities::default()
        };
        let value = serde_json::to_value(server).unwrap();
        assert_eq!(value["minimax_h3"]["partitions"][0]["task"], "fl2va");
        assert!(value["minimax_h3"]["qualification"]["metal_supported"]
            .as_bool()
            .is_some_and(|supported| !supported));
    }

    #[test]
    fn older_private_h3_partition_without_request_stays_deserializable() {
        let decoded: MiniMaxH3PartitionCapability = serde_json::from_value(serde_json::json!({
            "task": "fl2va",
            "model": crate::minimax_h3::FL2VA_COMFY,
            "display_name": "MiniMax H3 FL2VA",
            "runtime_available": true,
            "tier": "Compact",
            "component_ids": ["transformer"]
        }))
        .expect("an older additive H3 partition must not reject the whole capability response");
        assert!(decoded.request.is_none());
    }
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
    /// This persisted value is read when the engine/coordinator starts and
    /// does not change the running process.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub restart_required: bool,
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
    /// Stable peak estimate resolved against the roomiest physical GPU's
    /// total capacity. Unlike `peak_memory_bytes`, this does not follow
    /// moment-to-moment allocations from work already running on the host.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capacity_peak_memory_bytes: Option<u64>,
    /// Total VRAM of the GPU used for `capacity_peak_memory_bytes`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_capacity_bytes: Option<u64>,
    /// Whether the capacity-resolved execution strategy passes the same
    /// family-specific fit policy used by admission.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fits_device_capacity: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GenerationPlacementPreviewRequest {
    pub request: GenerateRequest,
    #[serde(default = "default_preview_copies")]
    pub copies: u32,
}

fn default_preview_copies() -> u32 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GenerationPlacementCandidate {
    pub device_id: String,
    pub execution_fingerprint: String,
    /// Parent-level deterministic execution identity shared by compatible
    /// device-specific candidates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_equivalence_fingerprint: Option<String>,
    pub predicted_start_after_ms: u64,
    pub predicted_completion_after_ms: u64,
    pub setup_ms: u64,
    pub setup_kind: String,
    #[schema(value_type = String)]
    pub estimate_confidence: QueueEstimateConfidence,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChainStagePlacementCandidate {
    pub stage_index: u32,
    /// Zero-based sibling index for repeated ordinary-generation work. Absent
    /// for a once-per-parent stage such as local prompt expansion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub copy_index: Option<u32>,
    #[serde(flatten)]
    pub candidate: GenerationPlacementCandidate,
}

/// A dependency that an authoritative placement preview proved admission can
/// materialize before generation starts. Placement previews never start the
/// download themselves.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct PendingModelDownload {
    pub kind: String,
    pub name: String,
    pub repo: String,
    pub bytes: u64,
    /// Manifest bundle admission will install to materialize this file.
    ///
    /// Additive and generic: clients use this as the retry target after any
    /// outstanding `licenses` are accepted instead of inferring a model name
    /// from a repository or dependency kind.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub install_model: Option<String>,
    /// Exact, server-pinned terms that still block this dependency.
    /// Empty means admission may transparently materialize it on first use.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub licenses: Vec<LicenseRefusal>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GenerationPlacementPreview {
    pub version: u32,
    pub authoritative: bool,
    pub state_version: u64,
    pub plan_version: u64,
    pub outcome: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate: Option<GenerationPlacementCandidate>,
    /// Exact per-stage assignments for every work item in an ordinary
    /// generation DAG. Durable-chain previews remain unsupported until the
    /// server can freeze the chain runner's per-device stage plans.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stage_candidates: Vec<ChainStagePlacementCandidate>,
    /// Known dependencies that admission will materialize before executing a
    /// planned request. Absent for older servers and when no download is
    /// required.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pending_downloads: Vec<PendingModelDownload>,
    /// Concrete absent model components that can be repaired by pulling
    /// `repair_model`. This is only populated for an infeasible response.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub missing_components: Vec<ModelComponentStatus>,
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
    /// A durable generation batch child committed a new authoritative state.
    /// Unlike `job_ended` and `gallery_added`, this is emitted only after the
    /// SQLite transaction has completed, so reconnecting clients may safely
    /// reconcile the child through `/api/generation-batches/status`.
    JobStateCommitted {
        id: String,
    },
    /// One transaction committed authoritative state for multiple durable
    /// generation children. Clients must reconcile the host once; emitting a
    /// child event per row would turn bulk cancellation into an event storm.
    GenerationStatesCommitted,
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
    /// An output was deleted via `DELETE /api/gallery/image/:filename`
    /// (permanently — also emitted when a trashed print is purged).
    GalleryRemoved {
        filename: String,
    },
    /// A gallery row's organization state changed (title, favorite, tags,
    /// or collection membership). `image` carries the refreshed row when the
    /// DB recorded it; `None` means refetch `/api/gallery`.
    GalleryUpdated {
        filename: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        image: Option<Box<GalleryImage>>,
    },
    /// An output was moved to the trash. It leaves the live listing and
    /// appears under `GET /api/gallery?view=trash`.
    GalleryTrashed {
        filename: String,
    },
    /// A trashed output was restored to the live gallery. `image` carries
    /// the restored row when the DB recorded it; `None` means refetch.
    GalleryRestored {
        filename: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        image: Option<Box<GalleryImage>>,
    },
    /// A collection was created, renamed, re-covered, deleted, or had its
    /// membership changed. Clients refetch `GET /api/gallery/collections`.
    GalleryCollectionsChanged {},
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
    /// The V2 scheduler published a newer versioned plan. Clients replace
    /// their tentative lanes only when `plan_version` advances.
    QueuePlanChanged {
        plan: Box<QueuePlan>,
    },
    /// A machine-wide device lifecycle preference or runtime state changed.
    DeviceStateChanged {
        device_id: String,
        desired_enabled: bool,
        admin_state: DeviceAdminState,
    },
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

        let committed = ServerEvent::JobStateCommitted { id: "j1".into() };
        assert_eq!(
            serde_json::to_string(&committed).unwrap(),
            r#"{"type":"job_state_committed","id":"j1"}"#
        );

        assert_eq!(
            serde_json::to_string(&ServerEvent::GenerationStatesCommitted).unwrap(),
            r#"{"type":"generation_states_committed"}"#
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
    fn device_state_event_serializes_stable_lifecycle_fields() {
        let wire = serde_json::to_value(ServerEvent::DeviceStateChanged {
            device_id: "cuda:0123456789abcdef0123456789abcdef".into(),
            desired_enabled: false,
            admin_state: DeviceAdminState::Draining,
        })
        .unwrap();

        assert_eq!(wire["type"], "device_state_changed");
        assert_eq!(wire["device_id"], "cuda:0123456789abcdef0123456789abcdef");
        assert_eq!(wire["desired_enabled"], false);
        assert_eq!(wire["admin_state"], "draining");
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
    fn plan_and_device_events_are_additive_snake_case_contracts() {
        let event = ServerEvent::QueuePlanChanged {
            plan: Box::new(QueuePlan {
                plan_version: 4,
                state_version: 7,
                optimizer_state: "optimized".into(),
                ..QueuePlan::default()
            }),
        };
        let json = serde_json::to_value(event).unwrap();
        assert_eq!(json["type"], "queue_plan_changed");
        assert_eq!(json["plan"]["plan_version"], 4);
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
                media_version: Some("1700000000000:123".into()),
                metadata_synthetic: false,
                title: None,
                tags: Vec::new(),
                favorite: false,
                collections: Vec::new(),
                trashed_at: None,
                purge_at: None,
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

    fn organized_gallery_image() -> GalleryImage {
        let metadata: OutputMetadata = serde_json::from_str(
            r#"{"prompt":"a cat","title":"Smurf village","model":"flux-dev:q4","seed":7,"steps":4,"guidance":3.5,"width":1024,"height":1024,"version":"test"}"#,
        )
        .unwrap();
        GalleryImage {
            filename: "mold-flux-dev-q4-1700000000000~smurf-village.png".into(),
            metadata,
            timestamp: 1_700_000_000,
            format: Some(OutputFormat::Png),
            size_bytes: Some(123),
            media_version: Some("1700000000000:123".into()),
            metadata_synthetic: false,
            title: Some("Smurf village".into()),
            tags: vec!["blue".into(), "cartoon".into()],
            favorite: true,
            collections: vec!["col-1".into()],
            trashed_at: None,
            purge_at: None,
        }
    }

    #[test]
    fn gallery_image_organization_fields_are_additive() {
        // An older server's row carries none of the organization fields.
        let legacy: GalleryImage = serde_json::from_str(
            r#"{"filename":"cat.png","metadata":{"prompt":"a cat","model":"flux-dev:q4","seed":7,"steps":4,"guidance":3.5,"width":1024,"height":1024,"version":"test"},"timestamp":1700000000}"#,
        )
        .unwrap();
        assert_eq!(legacy.title, None);
        assert!(legacy.tags.is_empty());
        assert!(!legacy.favorite);
        assert!(legacy.collections.is_empty());
        assert_eq!(legacy.trashed_at, None);
        assert_eq!(legacy.purge_at, None);
        assert_eq!(legacy.metadata.title, None);

        // Untouched rows serialize without the new keys at all.
        let wire = serde_json::to_value(&legacy).unwrap();
        for key in [
            "title",
            "tags",
            "favorite",
            "collections",
            "trashed_at",
            "purge_at",
        ] {
            assert!(wire.get(key).is_none(), "{key} should be omitted: {wire}");
        }
        assert!(wire["metadata"].get("title").is_none());

        // Organized rows round-trip every field.
        let organized = organized_gallery_image();
        let wire = serde_json::to_value(&organized).unwrap();
        assert_eq!(wire["title"], "Smurf village");
        assert_eq!(wire["tags"], serde_json::json!(["blue", "cartoon"]));
        assert_eq!(wire["favorite"], true);
        assert_eq!(wire["collections"], serde_json::json!(["col-1"]));
        assert_eq!(wire["metadata"]["title"], "Smurf village");
        let back: GalleryImage = serde_json::from_value(wire).unwrap();
        assert_eq!(back.tags, organized.tags);
        assert!(back.favorite);
        assert_eq!(back.collections, organized.collections);
    }

    #[test]
    fn trashed_gallery_image_carries_trash_timestamps() {
        let mut trashed = organized_gallery_image();
        trashed.trashed_at = Some(1_700_000_100);
        trashed.purge_at = Some(1_700_000_100 + 30 * 86_400);
        let wire = serde_json::to_value(&trashed).unwrap();
        assert_eq!(wire["trashed_at"], 1_700_000_100);
        assert_eq!(wire["purge_at"], 1_702_592_100);
        let back: GalleryImage = serde_json::from_value(wire).unwrap();
        assert_eq!(back.trashed_at, Some(1_700_000_100));
        assert_eq!(back.purge_at, Some(1_702_592_100));
    }

    #[test]
    fn generate_request_title_is_additive_and_flows_into_metadata() {
        let without: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"a cat","model":"flux-dev:q4","width":1024,"height":1024,"steps":4}"#,
        )
        .unwrap();
        assert_eq!(without.title, None);
        let metadata = OutputMetadata::from_generate_request(&without, 7, None, "test");
        assert_eq!(metadata.title, None);

        let with: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"a cat","title":"Smurf village","model":"flux-dev:q4","width":1024,"height":1024,"steps":4}"#,
        )
        .unwrap();
        assert_eq!(with.title.as_deref(), Some("Smurf village"));
        let metadata = OutputMetadata::from_generate_request(&with, 7, None, "test");
        assert_eq!(metadata.title.as_deref(), Some("Smurf village"));
        let wire = serde_json::to_value(&metadata).unwrap();
        assert_eq!(wire["title"], "Smurf village");
    }

    /// `tags` / `collection` are additive on both halves of the wire: an
    /// older client's request omits them, and an untagged print's embedded
    /// metadata carries neither key.
    #[test]
    fn generate_request_filing_is_additive_and_flows_into_metadata() {
        let without: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"a cat","model":"flux-dev:q4","width":1024,"height":1024,"steps":4}"#,
        )
        .unwrap();
        assert_eq!(without.tags, None);
        assert_eq!(without.collection, None);
        let metadata = OutputMetadata::from_generate_request(&without, 7, None, "test");
        assert_eq!(metadata.tags, None);
        assert_eq!(metadata.collection, None);
        let wire = serde_json::to_value(&metadata).unwrap();
        assert!(wire.get("tags").is_none(), "{wire}");
        assert!(wire.get("collection").is_none(), "{wire}");
        assert!(serde_json::to_value(&without)
            .unwrap()
            .get("tags")
            .is_none());

        let with: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"a cat","model":"flux-dev:q4","width":1024,"height":1024,"steps":4,
                "tags":["smurfs","village"],"collection":{"name":"Smurf Village"}}"#,
        )
        .unwrap();
        assert_eq!(
            with.tags.as_deref(),
            Some(["smurfs".to_string(), "village".to_string()].as_slice())
        );
        assert_eq!(
            with.collection,
            Some(CollectionRef::by_name("Smurf Village"))
        );
        let metadata = OutputMetadata::from_generate_request(&with, 7, None, "test");
        assert_eq!(
            metadata.tags.as_deref(),
            Some(["smurfs".to_string(), "village".to_string()].as_slice())
        );
        assert_eq!(metadata.collection.as_deref(), Some("Smurf Village"));
        let wire = serde_json::to_value(&metadata).unwrap();
        assert_eq!(wire["tags"], serde_json::json!(["smurfs", "village"]));
        assert_eq!(wire["collection"], "Smurf Village");
    }

    /// The embedded copy records the collection NAME, never the requested
    /// id — a request that arrives as an id has been resolved by admission,
    /// and one that was not must not stamp a UUID into provenance.
    #[test]
    fn metadata_collection_records_the_name_never_an_unresolved_id() {
        let mut req: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"a cat","model":"flux-dev:q4","width":1024,"height":1024,"steps":4}"#,
        )
        .unwrap();
        req.collection = Some(CollectionRef::by_id("11111111-2222-3333-4444-555555555555"));
        let metadata = OutputMetadata::from_generate_request(&req, 7, None, "test");
        assert_eq!(metadata.collection, None);

        // Once admission resolves the id, the name is what lands.
        req.collection = Some(CollectionRef {
            id: Some("11111111-2222-3333-4444-555555555555".into()),
            name: Some("Smurf Village".into()),
        });
        let metadata = OutputMetadata::from_generate_request(&req, 7, None, "test");
        assert_eq!(metadata.collection.as_deref(), Some("Smurf Village"));
    }

    /// An empty tag list is the same as no tags — it must not stamp an empty
    /// array into every print's provenance.
    #[test]
    fn metadata_omits_an_empty_tag_list() {
        let mut req: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"a cat","model":"flux-dev:q4","width":1024,"height":1024,"steps":4}"#,
        )
        .unwrap();
        req.tags = Some(Vec::new());
        let metadata = OutputMetadata::from_generate_request(&req, 7, None, "test");
        assert_eq!(metadata.tags, None);
    }

    #[test]
    fn collection_ref_detects_the_unset_shape() {
        assert!(CollectionRef::default().is_unset());
        assert!(CollectionRef {
            id: Some("  ".into()),
            name: Some("".into())
        }
        .is_unset());
        assert!(!CollectionRef::by_name("Smurf Village").is_unset());
        assert!(!CollectionRef::by_id("abc").is_unset());

        // `{}` deserializes rather than failing, so the 422 can name the
        // field instead of the JSON shape.
        let bare: CollectionRef = serde_json::from_str("{}").unwrap();
        assert!(bare.is_unset());
        assert_eq!(serde_json::to_value(&bare).unwrap(), serde_json::json!({}));
    }

    #[test]
    fn gallery_updated_round_trips_with_image_row() {
        let ev = ServerEvent::GalleryUpdated {
            filename: "cat.png".into(),
            image: Some(Box::new(organized_gallery_image())),
        };
        let wire = serde_json::to_string(&ev).unwrap();
        assert!(wire.starts_with(r#"{"type":"gallery_updated","#), "{wire}");
        let back: ServerEvent = serde_json::from_str(&wire).unwrap();
        match back {
            ServerEvent::GalleryUpdated {
                filename,
                image: Some(img),
            } => {
                assert_eq!(filename, "cat.png");
                assert!(img.favorite);
                assert_eq!(img.title.as_deref(), Some("Smurf village"));
            }
            other => panic!("expected gallery_updated with image, got {other:?}"),
        }

        let no_row = ServerEvent::GalleryUpdated {
            filename: "cat.png".into(),
            image: None,
        };
        assert_eq!(
            serde_json::to_string(&no_row).unwrap(),
            r#"{"type":"gallery_updated","filename":"cat.png"}"#
        );
    }

    #[test]
    fn gallery_trashed_and_restored_round_trip() {
        let trashed = ServerEvent::GalleryTrashed {
            filename: "cat.png".into(),
        };
        assert_eq!(
            serde_json::to_string(&trashed).unwrap(),
            r#"{"type":"gallery_trashed","filename":"cat.png"}"#
        );

        let restored = ServerEvent::GalleryRestored {
            filename: "cat.png".into(),
            image: Some(Box::new(organized_gallery_image())),
        };
        let wire = serde_json::to_string(&restored).unwrap();
        assert!(wire.starts_with(r#"{"type":"gallery_restored","#), "{wire}");
        match serde_json::from_str::<ServerEvent>(&wire).unwrap() {
            ServerEvent::GalleryRestored {
                filename,
                image: Some(img),
            } => {
                assert_eq!(filename, "cat.png");
                assert_eq!(img.timestamp, 1_700_000_000);
            }
            other => panic!("expected gallery_restored with image, got {other:?}"),
        }
        let bare = ServerEvent::GalleryRestored {
            filename: "cat.png".into(),
            image: None,
        };
        assert_eq!(
            serde_json::to_string(&bare).unwrap(),
            r#"{"type":"gallery_restored","filename":"cat.png"}"#
        );
    }

    #[test]
    fn gallery_collections_changed_serializes_as_bare_tagged_object() {
        let wire = serde_json::to_string(&ServerEvent::GalleryCollectionsChanged {}).unwrap();
        assert_eq!(wire, r#"{"type":"gallery_collections_changed"}"#);
        assert!(matches!(
            serde_json::from_str::<ServerEvent>(&wire).unwrap(),
            ServerEvent::GalleryCollectionsChanged {}
        ));
    }

    #[test]
    fn gallery_capabilities_trash_and_organize_are_additive() {
        // Older server: only can_delete.
        let caps: ServerCapabilities = serde_json::from_str(
            r#"{"gallery":{"can_delete":true},"catalog":{"available":false,"families":[]}}"#,
        )
        .unwrap();
        assert!(caps.gallery.can_delete);
        assert_eq!(caps.gallery.trash, None);
        assert!(!caps.gallery.organize);
        let wire = serde_json::to_value(&caps.gallery).unwrap();
        assert_eq!(wire, serde_json::json!({"can_delete": true}));

        // Current server advertises both.
        let full = GalleryCapabilities {
            can_delete: true,
            trash: Some(GalleryTrashCapabilities {
                enabled: true,
                retention_days: 30,
            }),
            organize: true,
            bulk_mutations: true,
            media_version: true,
            conditional_get: true,
            row_events: true,
        };
        let wire = serde_json::to_value(&full).unwrap();
        assert_eq!(
            wire,
            serde_json::json!({
                "can_delete": true,
                "trash": {"enabled": true, "retention_days": 30},
                "organize": true,
                "bulk_mutations": true,
                "media_version": true,
                "conditional_get": true,
                "row_events": true
            })
        );
        let back: GalleryCapabilities = serde_json::from_value(wire).unwrap();
        assert_eq!(back.trash.unwrap().retention_days, 30);
        assert!(back.organize);
    }

    #[test]
    fn collection_and_tag_count_round_trip() {
        let collection = Collection {
            id: "c1".into(),
            name: "Smurfs".into(),
            slug: "smurfs".into(),
            description: None,
            cover_filename: Some("cat.png".into()),
            hidden: false,
            count: 3,
            created_at: 1_700_000_000,
            updated_at: 1_700_000_050,
        };
        let wire = serde_json::to_value(&collection).unwrap();
        assert!(wire.get("description").is_none());
        assert!(wire.get("hidden").is_none());
        assert_eq!(wire["cover_filename"], "cat.png");
        let back: Collection = serde_json::from_value(wire).unwrap();
        assert_eq!(back, collection);

        let tag = TagCount {
            name: "blue".into(),
            count: 2,
        };
        let back: TagCount = serde_json::from_str(&serde_json::to_string(&tag).unwrap()).unwrap();
        assert_eq!(back, tag);
    }

    #[test]
    fn organization_request_bodies_default_absent_fields() {
        let patch: GalleryPatchRequest = serde_json::from_str(r#"{"favorite":true}"#).unwrap();
        assert_eq!(patch.favorite, Some(true));
        assert_eq!(patch.title, None);
        assert_eq!(patch.tags, None);
        assert_eq!(patch.add_tags, None);
        let cleared: GalleryPatchRequest = serde_json::from_str(r#"{"title":""}"#).unwrap();
        assert_eq!(cleared.title.as_deref(), Some(""));

        let organize: GalleryOrganizeRequest =
            serde_json::from_str(r#"{"filenames":["a.png","b.png"],"add_tags":["x"]}"#).unwrap();
        assert_eq!(organize.filenames.len(), 2);
        assert_eq!(organize.add_tags.as_deref(), Some(&["x".to_string()][..]));
        assert_eq!(organize.add_to_collections, None);
        assert!(serde_json::from_str::<GalleryOrganizeRequest>(r#"{}"#).is_err());

        let create: CollectionCreateRequest = serde_json::from_str(r#"{"name":"Smurfs"}"#).unwrap();
        assert_eq!(create.description, None);
        let update: CollectionUpdateRequest =
            serde_json::from_str(r#"{"cover_filename":"cat.png"}"#).unwrap();
        assert_eq!(update.name, None);
        assert_eq!(update.cover_filename.as_deref(), Some("cat.png"));
        let items: CollectionItemsRequest = serde_json::from_str(r#"{"add":["cat.png"]}"#).unwrap();
        assert_eq!(items.add, vec!["cat.png".to_string()]);
        assert!(items.remove.is_empty());
        let trash: TrashFilenamesRequest =
            serde_json::from_str(r#"{"filenames":["cat.png"]}"#).unwrap();
        assert_eq!(trash.filenames, vec!["cat.png".to_string()]);
        let rename: TagRenameRequest = serde_json::from_str(r#"{"name":"teal"}"#).unwrap();
        assert_eq!(rename.name, "teal");

        let sweep = TrashSweepResult {
            purged: 2,
            remaining: 5,
        };
        assert_eq!(
            serde_json::to_string(&sweep).unwrap(),
            r#"{"purged":2,"remaining":5}"#
        );
        assert_eq!(
            serde_json::to_string(&EmptyTrashResult { purged: 7 }).unwrap(),
            r#"{"purged":7}"#
        );
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
            remix: true,
            model: Some("qwen3-expand".into()),
        };
        let api = ExpandCapabilities {
            configured: true,
            model_present: None,
            backend: ExpandBackend::Api,
            remix: true,
            model: None,
        };

        assert_eq!(
            serde_json::to_value(&local).unwrap(),
            serde_json::json!({
                "configured": true,
                "model_present": false,
                "backend": "local",
                "remix": true,
                "model": "qwen3-expand"
            })
        );
        assert_eq!(
            serde_json::to_value(&api).unwrap(),
            serde_json::json!({
                "configured": true,
                "model_present": null,
                "backend": "api",
                "remix": true
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

#[cfg(test)]
mod queue_plan_wire_tests {
    use super::*;

    #[test]
    fn blocked_reasons_round_trip_without_collapsing_distinct_causes() {
        let expected = [
            "device_disabled",
            "device_draining",
            "device_startup_excluded",
            "device_unavailable",
            "device_degraded",
            "hard_pin_unavailable",
            "backend_unsupported",
            "model_not_installed",
            "insufficient_vram",
            "insufficient_host_ram",
            "aggregate_host_ram_reserved",
            "execution_plan_incompatible",
            "dependency_wait",
            "warm_wait",
            "queue_paused",
            "maintenance_mode",
            "cancelling",
            "no_schedulable_device",
            "no_idle_device",
            "lower_priority_opening",
        ];

        for wire in expected {
            let parsed: QueueBlockedReason =
                serde_json::from_value(serde_json::json!(wire)).unwrap();
            assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
        }
    }

    #[test]
    fn future_blocked_reason_retains_its_exact_wire_value() {
        let parsed: QueueBlockedReason =
            serde_json::from_value(serde_json::json!("thermal_throttle")).unwrap();
        assert_eq!(
            serde_json::to_value(parsed).unwrap(),
            serde_json::json!("thermal_throttle")
        );
    }

    #[test]
    fn future_phase_and_confidence_retain_their_exact_wire_values() {
        let phase: QueueActivityPhase =
            serde_json::from_value(serde_json::json!("thermal_wait")).unwrap();
        let confidence: QueueEstimateConfidence =
            serde_json::from_value(serde_json::json!("calibrating")).unwrap();

        assert_eq!(phase.as_str(), "thermal_wait");
        assert_eq!(confidence.as_str(), "calibrating");
        assert_eq!(
            serde_json::to_value(phase).unwrap(),
            serde_json::json!("thermal_wait")
        );
        assert_eq!(
            serde_json::to_value(confidence).unwrap(),
            serde_json::json!("calibrating")
        );
    }

    #[test]
    fn planned_lane_kind_is_typed_and_retains_future_wire_values() {
        let device: QueuePlannedLaneKind =
            serde_json::from_value(serde_json::json!("device")).unwrap();
        let host: QueuePlannedLaneKind =
            serde_json::from_value(serde_json::json!("host_utility")).unwrap();
        let future: QueuePlannedLaneKind =
            serde_json::from_value(serde_json::json!("remote_utility")).unwrap();

        assert_eq!(device, QueuePlannedLaneKind::Device);
        assert_eq!(host, QueuePlannedLaneKind::HostUtility);
        assert_eq!(future.as_str(), "remote_utility");
        assert_eq!(
            serde_json::to_value(future).unwrap(),
            serde_json::json!("remote_utility")
        );
    }

    /// The exact shape three frontends parse. Renaming a field or emitting
    /// zeros for an unsampled host silently miscolours every pressure meter.
    #[test]
    fn host_memory_telemetry_serializes_its_published_field_names() {
        let plan = QueuePlan {
            host_memory: Some(HostMemorySnapshot {
                total_bytes: 67_430_000_000,
                available_bytes: 58_000_000_000,
                headroom_bytes: 48_700_000_000,
                safety_floor_bytes: 10_114_500_000,
            }),
            ..Default::default()
        };
        let json = serde_json::to_value(&plan).unwrap();
        assert_eq!(
            json["host_memory"],
            serde_json::json!({
                "total_bytes": 67_430_000_000_u64,
                "available_bytes": 58_000_000_000_u64,
                "headroom_bytes": 48_700_000_000_u64,
                "safety_floor_bytes": 10_114_500_000_u64,
            })
        );
        assert_eq!(
            serde_json::from_value::<QueuePlan>(json).unwrap(),
            plan,
            "the wire shape must round-trip"
        );
    }

    #[test]
    fn absent_host_memory_is_omitted_rather_than_zeroed() {
        let plan = QueuePlan::default();
        assert!(plan.host_memory.is_none());
        let json = serde_json::to_value(&plan).unwrap();
        assert!(
            json.get("host_memory").is_none(),
            "an unsampled host reports nothing, never a zeroed snapshot"
        );
        // A legacy payload with no field at all must still parse.
        let legacy = serde_json::json!({
            "plan_version": 1,
            "state_version": 1,
            "optimizer_state": "optimized",
        });
        assert!(serde_json::from_value::<QueuePlan>(legacy)
            .unwrap()
            .host_memory
            .is_none());
    }

    #[test]
    fn server_status_carries_the_same_host_memory_shape() {
        let json = serde_json::json!({
            "version": "0.21.0",
            "models_loaded": [],
            "gpu_info": null,
            "uptime_secs": 1,
            "host_memory": {
                "total_bytes": 67_430_000_000_u64,
                "available_bytes": 58_000_000_000_u64,
                "headroom_bytes": 48_700_000_000_u64,
                "safety_floor_bytes": 10_114_500_000_u64,
            },
        });
        let status: ServerStatus = serde_json::from_value(json).unwrap();
        let host_memory = status.host_memory.expect("host memory parses on status");
        assert_eq!(host_memory.total_bytes, 67_430_000_000);
        assert_eq!(host_memory.safety_floor_bytes, 10_114_500_000);

        let legacy = serde_json::json!({
            "version": "0.20.0",
            "models_loaded": [],
            "gpu_info": null,
            "uptime_secs": 1,
        });
        assert!(serde_json::from_value::<ServerStatus>(legacy)
            .unwrap()
            .host_memory
            .is_none());
    }

    #[test]
    fn presentation_normalization_only_rewrites_host_utility_identities() {
        let mut listing = QueueListingWire {
            entries: vec![],
            live_only_entries: vec![],
            plan: Some(QueuePlan {
                work_items: vec![
                    QueueWorkItem {
                        work_id: "legacy".into(),
                        activity_phase: QueueActivityPhase::Queued,
                        planned_device_id: Some("cpu:utility:0".into()),
                        ..Default::default()
                    },
                    QueueWorkItem {
                        work_id: "typed-host".into(),
                        planned_lane_kind: Some(QueuePlannedLaneKind::HostUtility),
                        planned_device_id: Some("must-not-leak".into()),
                        ..Default::default()
                    },
                    QueueWorkItem {
                        work_id: "gpu".into(),
                        planned_lane_kind: Some(QueuePlannedLaneKind::Device),
                        planned_device_id: Some("cuda:stable-a".into()),
                        ..Default::default()
                    },
                    QueueWorkItem {
                        work_id: "future".into(),
                        planned_lane_kind: Some(QueuePlannedLaneKind::Unknown(
                            "remote_utility".into(),
                        )),
                        planned_device_id: Some("future-public-id".into()),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }),
            page: None,
        };

        listing.normalize_planned_lanes_for_presentation();
        let work = &listing.plan.unwrap().work_items;
        for item in &work[..2] {
            assert_eq!(
                item.planned_lane_kind,
                Some(QueuePlannedLaneKind::HostUtility)
            );
            assert_eq!(item.planned_device_id, None);
        }
        assert_eq!(
            work[2].planned_lane_kind,
            Some(QueuePlannedLaneKind::Device)
        );
        assert_eq!(work[2].planned_device_id.as_deref(), Some("cuda:stable-a"));
        assert_eq!(
            work[3].planned_lane_kind,
            Some(QueuePlannedLaneKind::Unknown("remote_utility".into()))
        );
        assert_eq!(
            work[3].planned_device_id.as_deref(),
            Some("future-public-id")
        );
    }

    #[test]
    fn legacy_reason_only_queue_item_remains_readable() {
        let parsed: QueueWorkItem = serde_json::from_value(serde_json::json!({
            "work_id": "job-1",
            "parent_id": "job-1",
            "work_kind": "generation",
            "priority_class": "user",
            "queue_rank": 4,
            "bypass_count": 0,
            "estimate_confidence": "low",
            "reason": "insufficient_vram"
        }))
        .unwrap();

        assert_eq!(parsed.reason.as_deref(), Some("insufficient_vram"));
        assert_eq!(parsed.blocked_reason, None);
        assert_eq!(parsed.planned_lane_kind, None);
        assert_eq!(parsed.activity_phase, QueueActivityPhase::Queued);
        assert_eq!(parsed.execution_equivalence_fingerprint, None);
    }

    #[test]
    fn execution_equivalence_wire_fields_are_additive_and_optional() {
        let legacy: GenerationPlacementCandidate = serde_json::from_value(serde_json::json!({
            "device_id": "cuda:stable",
            "execution_fingerprint": "device-qualified",
            "predicted_start_after_ms": 0,
            "predicted_completion_after_ms": 10,
            "setup_ms": 1,
            "setup_kind": "cold",
            "estimate_confidence": "low"
        }))
        .unwrap();
        assert_eq!(legacy.execution_equivalence_fingerprint, None);

        let serialized = serde_json::to_value(legacy).unwrap();
        assert!(serialized
            .get("execution_equivalence_fingerprint")
            .is_none());
    }

    /// Every field the durable queue adds is additive in both directions: an
    /// old server's payload deserializes, and a new server's payload stays
    /// readable by an old client (which ignores what it does not know).
    #[test]
    fn durable_queue_wire_fields_are_additive() {
        let legacy: SseErrorEvent = serde_json::from_str(r#"{"message":"boom"}"#).unwrap();
        assert!(!legacy.retained);
        assert_eq!(legacy.code, None);
        assert_eq!(
            serde_json::to_value(&legacy).unwrap(),
            serde_json::json!({"message": "boom"}),
            "a plain failure must not grow fields on the wire"
        );

        let retained = SseErrorEvent::retained("the host is restarting");
        let encoded = serde_json::to_value(&retained).unwrap();
        assert_eq!(encoded["retained"], serde_json::json!(true));
        assert_eq!(
            encoded["code"],
            serde_json::json!(SSE_ERROR_CODE_SERVER_RESTARTING)
        );
        let round_tripped: SseErrorEvent = serde_json::from_value(encoded).unwrap();
        assert!(round_tripped.retained);

        let legacy_queue: QueueCapabilities =
            serde_json::from_str(r#"{"can_pause":true,"can_cancel_all":true}"#).unwrap();
        assert!(!legacy_queue.durable_queue);
        assert!(!legacy_queue.cooperative_cancellation);

        let durable_media = serde_json::to_value(DurableMediaCapabilities::v1()).unwrap();
        assert_eq!(
            durable_media,
            serde_json::json!({
                "protocol_version": 1,
                "encrypted_at_rest": true,
                "generate_request_media": true,
                "identity": true,
                "private_h3": false,
            })
        );

        for private_h3 in [false, true] {
            assert_eq!(
                serde_json::to_value(DurableMediaCapabilities::v2(private_h3)).unwrap(),
                serde_json::json!({
                    "protocol_version": 2,
                    "encrypted_at_rest": true,
                    "generate_request_media": true,
                    "identity": true,
                    "private_h3": private_h3,
                }),
                "protocol v2 must advertise private H3 exactly as supplied by the build"
            );
        }

        let legacy_server: ServerCapabilities = serde_json::from_value(serde_json::json!({
            "gallery": {"can_delete": true},
            "catalog": {"available": false, "families": []}
        }))
        .unwrap();
        assert_eq!(legacy_server.durable_media, None);
        assert!(
            !serde_json::to_value(&legacy_server)
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("durable_media"),
            "an unavailable capability must remain absent rather than serialize as null"
        );

        let advertised = ServerCapabilities {
            durable_media: Some(DurableMediaCapabilities::v1()),
            ..ServerCapabilities::default()
        };
        let advertised_wire = serde_json::to_value(&advertised).unwrap();
        assert_eq!(advertised_wire["durable_media"], durable_media);
        let round_tripped: ServerCapabilities = serde_json::from_value(advertised_wire).unwrap();
        assert_eq!(
            round_tripped.durable_media,
            Some(DurableMediaCapabilities::v1())
        );

        let legacy_metadata: OutputMetadata = serde_json::from_value(serde_json::json!({
            "prompt": "a cat",
            "model": "flux-dev:q4",
            "seed": 7,
            "steps": 4,
            "guidance": 3.5,
            "width": 512,
            "height": 512,
            "format": "png",
            "version": "0.0.0",
        }))
        .unwrap();
        assert_eq!(legacy_metadata.job_id, None);
        assert_eq!(legacy_metadata.pipeline_requested, None);
        assert!(
            !serde_json::to_value(&legacy_metadata)
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("job_id"),
            "an unstamped print must not gain a null job id"
        );
    }
}

#[cfg(test)]
mod reference_crop_tests {
    use super::*;

    fn crop() -> GenerationReferenceCrop {
        GenerationReferenceCrop {
            x: 420,
            y: 0,
            width: 1080,
            height: 1080,
            source_width: 1920,
            source_height: 1080,
            source_sha256: "ab".repeat(32),
        }
    }

    fn image(
        width: u32,
        height: u32,
        crop: Option<GenerationReferenceCrop>,
    ) -> GenerationReference {
        GenerationReference::Image {
            media: GenerationReferenceAuthority::Inline {
                data: b"cropped-bytes".to_vec(),
            },
            provenance: GenerationReferenceProvenance {
                name: Some("subject.png".to_string()),
                sha256: None,
                crop,
            },
            mime_type: "image/png".to_string(),
            width,
            height,
        }
    }

    #[test]
    fn crop_validates_only_a_non_degenerate_rect_inside_its_source_that_matches_the_bytes() {
        crop().validate_for_image(1080, 1080).unwrap();

        let degenerate = GenerationReferenceCrop { width: 0, ..crop() };
        assert!(degenerate.validate_for_image(0, 1080).is_err());

        let outside = GenerationReferenceCrop { x: 900, ..crop() };
        assert!(outside.validate_for_image(1080, 1080).is_err());

        let overflow = GenerationReferenceCrop {
            x: u32::MAX,
            width: 2,
            ..crop()
        };
        assert!(overflow.validate_for_image(2, 1080).is_err());

        // The crop describes the bytes the server received: a rect whose size
        // differs from the reference's own dimensions is a pre-crop
        // projection that was never applied.
        assert!(crop().validate_for_image(1920, 1080).is_err());

        let bad_digest = GenerationReferenceCrop {
            source_sha256: "nope".to_string(),
            ..crop()
        };
        assert!(bad_digest.validate_for_image(1080, 1080).is_err());
    }

    #[test]
    fn crop_provenance_is_additive_and_absent_when_unset() {
        let plain = serde_json::to_value(image(1920, 1080, None)).unwrap();
        assert!(plain["provenance"].get("crop").is_none());

        let cropped = image(1080, 1080, Some(crop()));
        let json = serde_json::to_value(&cropped).unwrap();
        assert_eq!(json["provenance"]["crop"]["source_width"], 1920);
        let back: GenerationReference = serde_json::from_value(json).unwrap();
        assert_eq!(back.provenance().crop.as_ref(), Some(&crop()));
    }

    #[test]
    fn redacted_metadata_carries_the_crop_and_reference_validation_enforces_it() {
        let cropped = image(1080, 1080, Some(crop()));
        let metadata = cropped.redacted_metadata(0).unwrap();
        assert_eq!(metadata.crop.as_ref(), Some(&crop()));
        let metadata_json = serde_json::to_value(&metadata).unwrap();
        assert_eq!(metadata_json["crop"]["x"], 420);
        let plain_json =
            serde_json::to_value(image(1920, 1080, None).redacted_metadata(0).unwrap()).unwrap();
        assert!(plain_json.get("crop").is_none());

        crate::minimax_h3::validate_references(&[cropped]).unwrap();
        let unapplied = image(1920, 1080, Some(crop()));
        let error = crate::minimax_h3::validate_references(&[unapplied]).unwrap_err();
        assert_eq!(error.code, "MINIMAX_H3_REFERENCE_CROP");

        let video = GenerationReference::Video {
            media: GenerationReferenceAuthority::Inline {
                data: b"video".to_vec(),
            },
            provenance: GenerationReferenceProvenance {
                name: Some("clip.mp4".to_string()),
                sha256: None,
                crop: Some(crop()),
            },
            mime_type: "video/mp4".to_string(),
            width: 1080,
            height: 1080,
            frame_count: Some(96),
            duration_ms: 4_000,
            fps: 24.0,
            has_audio: false,
            audio_duration_ms: None,
            audio_sample_count: None,
            audio_sample_rate: None,
            audio_channels: None,
        };
        assert_eq!(
            crate::minimax_h3::validate_references(&[video])
                .unwrap_err()
                .code,
            "MINIMAX_H3_REFERENCE_CROP"
        );
    }

    /// The browser's `referencePadEstimate` (studio/lib/referenceCrop.ts)
    /// mirrors `reference_image_dimensions` + `rows_per_video_latent`; these
    /// fixtures are the values its test pins, so the two can only drift
    /// together.
    #[test]
    fn image_reference_pad_fixtures_match_the_browser_estimate() {
        let fixtures: [(u32, u32, u32, u32, u64); 6] = [
            (1920, 1080, 3648, 2048, 7296),
            (1080, 1080, 2048, 2048, 4096),
            (1024, 768, 2720, 2048, 5440),
            (1080, 1920, 2048, 3648, 7296),
            (1344, 768, 3584, 2048, 7168),
            (1120, 1080, 2112, 2048, 4224),
        ];
        for (width, height, normalized_width, normalized_height, rows) in fixtures {
            let shape =
                crate::minimax_h3::reference_prepared_shape(&image(width, height, None)).unwrap();
            assert_eq!(
                (
                    shape.normalized_width,
                    shape.normalized_height,
                    shape.visual_rows
                ),
                (Some(normalized_width), Some(normalized_height), rows),
                "{width}x{height}"
            );
        }
    }
}
