use serde::{Deserialize, Serialize};

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

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GenerateRequest {
    #[schema(example = "a cat sitting on a windowsill at sunset")]
    pub prompt: String,
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
    /// Scheduler override for UNet-based models (SD1.5, SDXL).
    /// Ignored by flow-matching models (FLUX, SD3, Z-Image, Flux.2, Qwen-Image).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheduler: Option<Scheduler>,
}

fn default_guidance() -> f64 {
    3.5
}

fn default_batch_size() -> u32 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GenerateResponse {
    pub images: Vec<ImageData>,
    #[schema(example = 1234)]
    pub generation_time_ms: u64,
    #[schema(example = "flux-schnell:q8")]
    pub model: String,
    #[schema(example = 42)]
    pub seed_used: u64,
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

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    #[default]
    Png,
    Jpeg,
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputFormat::Png => write!(f, "png"),
            OutputFormat::Jpeg => write!(f, "jpeg"),
        }
    }
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "png" => Ok(OutputFormat::Png),
            "jpeg" | "jpg" => Ok(OutputFormat::Jpeg),
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
pub struct ServerStatus {
    #[schema(example = "0.1.0")]
    pub version: String,
    pub models_loaded: Vec<String>,
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
    },
    /// A single file download completed.
    DownloadDone {
        filename: String,
        file_index: usize,
        total_files: usize,
    },
    /// All downloads complete for a model pull.
    PullComplete {
        model: String,
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
        assert!("webp".parse::<OutputFormat>().is_err());
        assert!("".parse::<OutputFormat>().is_err());
        assert!("bmp".parse::<OutputFormat>().is_err());
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
            model: "flux-schnell".to_string(),
            width: 768,
            height: 768,
            steps: 4,
            guidance: 0.0,
            seed: Some(42),
            batch_size: 1,
            output_format: OutputFormat::Png,
            scheduler: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: GenerateRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.prompt, req.prompt);
        assert_eq!(back.width, req.width);
        assert_eq!(back.seed, req.seed);
        assert_eq!(back.scheduler, None);
    }

    #[test]
    fn generate_request_optional_seed() {
        let json = r#"{"prompt":"test","model":"flux-schnell","width":768,"height":768,"steps":4,"batch_size":1,"output_format":"png"}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert!(req.seed.is_none());
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
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: SseCompleteEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.width, 1024);
        assert_eq!(back.seed_used, 42);
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
}
