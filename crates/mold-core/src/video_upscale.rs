//! Durable framewise video-upscale wire contract.

use serde::{Deserialize, Serialize};

pub const VIDEO_UPSCALE_CONTRACT_VERSION: u32 = 1;
pub const VIDEO_UPSCALE_DISCLOSURE: &str =
    "Framewise upscale processes each frame independently; temporal flicker may remain.";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VideoUpscaleSource {
    /// Exact filename of a committed item in this server's Library.
    Library { filename: String },
    /// A request-scoped server upload. The handle is resolved and retained at admission.
    Upload { handle: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct CreateVideoUpscaleJobRequest {
    pub source: VideoUpscaleSource,
    #[schema(example = "real-esrgan-x4plus:fp16")]
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tile_size: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum VideoUpscaleJobState {
    Queued,
    Running,
    Finalizing,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

impl VideoUpscaleJobState {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct VideoUpscaleMediaFacts {
    pub container: String,
    pub video_codec: String,
    pub width: u32,
    pub height: u32,
    pub frame_count: u64,
    /// Exact source rate as an ffmpeg rational, for example `24000/1001`.
    pub fps: String,
    pub duration_ms: u64,
    pub primary_audio_codec: Option<String>,
    pub primary_audio_sample_rate: Option<u32>,
    pub primary_audio_channels: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct VideoUpscaleJob {
    pub contract_version: u32,
    pub id: String,
    pub state: VideoUpscaleJobState,
    pub source: VideoUpscaleSource,
    pub model: String,
    pub scale_factor: u32,
    pub tile_size: Option<u32>,
    pub completed_frames: u64,
    pub total_frames: u64,
    pub source_facts: Option<VideoUpscaleMediaFacts>,
    pub output_facts: Option<VideoUpscaleMediaFacts>,
    pub output_filename: Option<String>,
    pub error: Option<String>,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    pub disclosure: String,
}

pub fn validate_create_video_upscale_job(
    request: &CreateVideoUpscaleJobRequest,
) -> Result<(), String> {
    if request.model.trim().is_empty() {
        return Err("framewise upscale model must not be empty".into());
    }
    match &request.source {
        VideoUpscaleSource::Library { filename } => {
            let name = filename.trim();
            if name.is_empty()
                || name != filename
                || name.contains('/')
                || name.contains('\\')
                || name == "."
                || name == ".."
            {
                return Err("library source must be one exact gallery filename".into());
            }
            let extension = std::path::Path::new(name)
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            if !matches!(extension.as_str(), "mp4" | "mov" | "webm") {
                return Err("framewise upscale supports MP4, MOV, and WebM sources".into());
            }
        }
        VideoUpscaleSource::Upload { handle } if handle.trim().is_empty() => {
            return Err("upload handle must not be empty".into());
        }
        VideoUpscaleSource::Upload { .. } => {}
    }
    if let Some(tile_size) = request.tile_size {
        if tile_size != 0 && tile_size < 64 {
            return Err(format!(
                "tile_size ({tile_size}) must be 0 (disabled) or >= 64"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(filename: &str) -> CreateVideoUpscaleJobRequest {
        CreateVideoUpscaleJobRequest {
            source: VideoUpscaleSource::Library {
                filename: filename.into(),
            },
            model: "real-esrgan-x4plus:fp16".into(),
            tile_size: None,
        }
    }

    #[test]
    fn library_authority_is_exact_and_video_only() {
        assert!(validate_create_video_upscale_job(&request("clip.mp4")).is_ok());
        for invalid in ["../clip.mp4", "nested/clip.mp4", "still.png", " clip.mov"] {
            assert!(validate_create_video_upscale_job(&request(invalid)).is_err());
        }
    }

    #[test]
    fn disclosure_names_framewise_limit() {
        assert!(VIDEO_UPSCALE_DISCLOSURE.contains("Framewise upscale"));
        assert!(VIDEO_UPSCALE_DISCLOSURE.contains("temporal flicker may remain"));
    }
}
