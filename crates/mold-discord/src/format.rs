use mold_core::{GenerateResponse, ModelInfoExtended, ServerStatus, SseProgressEvent};

/// Structured embed data that the handler converts to serenity `CreateEmbed`.
#[derive(Debug, Clone)]
pub struct EmbedData {
    pub title: String,
    pub description: String,
    pub fields: Vec<(String, String, bool)>,
    pub color: u32,
}

// Colors for embed theming
const COLOR_SUCCESS: u32 = 0x57F287; // green
const COLOR_INFO: u32 = 0x5865F2; // blurple
const COLOR_WARNING: u32 = 0xFEE75C; // yellow
const COLOR_ERROR: u32 = 0xED4245; // red

/// Format an SSE progress event into a human-readable status line.
pub fn format_progress(event: &SseProgressEvent) -> String {
    match event {
        SseProgressEvent::Queued { position } => {
            if *position == 0 {
                "Starting generation...".to_string()
            } else {
                format!("Queued — position #{position}")
            }
        }
        SseProgressEvent::StageStart { name } => {
            format!("Loading: {name}...")
        }
        SseProgressEvent::StageDone { name, elapsed_ms } => {
            format!("{name} ({elapsed_ms}ms)")
        }
        SseProgressEvent::Info { message } => message.clone(),
        SseProgressEvent::CacheHit { resource } => {
            format!("Cache hit: {resource}")
        }
        SseProgressEvent::DenoiseStep {
            step,
            total,
            elapsed_ms,
        } => {
            let bar = progress_bar(*step, *total);
            let ms_per_step = if *step > 0 {
                *elapsed_ms as f64 / *step as f64
            } else {
                0.0
            };
            format!("Denoising {step}/{total} {bar} ({ms_per_step:.0}ms/step)")
        }
        SseProgressEvent::DownloadProgress {
            filename,
            file_index,
            total_files,
            bytes_downloaded,
            bytes_total,
        } => {
            let pct = if *bytes_total > 0 {
                (*bytes_downloaded as f64 / *bytes_total as f64 * 100.0) as u64
            } else {
                0
            };
            format!(
                "Downloading {filename} [{}/{total_files}] {pct}%",
                file_index + 1
            )
        }
        SseProgressEvent::DownloadDone {
            filename,
            file_index,
            total_files,
        } => {
            format!("Downloaded {filename} [{}/{total_files}]", file_index + 1)
        }
        SseProgressEvent::PullComplete { model } => {
            format!("Model {model} downloaded")
        }
    }
}

/// Build a text progress bar: `[=========>          ]`
fn progress_bar(current: usize, total: usize) -> String {
    let width = 20;
    let filled = if total > 0 {
        (current * width) / total
    } else {
        0
    };
    let fill_len = if filled > 0 && filled < width {
        filled - 1
    } else {
        filled
    };
    let has_arrow = filled > 0 && filled < width;
    let space_len = width - fill_len - if has_arrow { 1 } else { 0 };
    format!(
        "[{}{}{}]",
        "=".repeat(fill_len),
        if has_arrow { ">" } else { "" },
        " ".repeat(space_len)
    )
}

/// Format a completed generation result into embed data.
pub fn format_generation_result(resp: &GenerateResponse, prompt: &str) -> EmbedData {
    let time_secs = resp.generation_time_ms as f64 / 1000.0;
    let dims = resp
        .images
        .first()
        .map(|img| format!("{}x{}", img.width, img.height))
        .unwrap_or_default();

    let truncated_prompt = if prompt.len() > 256 {
        format!("{}...", &prompt[..253])
    } else {
        prompt.to_string()
    };

    EmbedData {
        title: "Image Generated".to_string(),
        description: truncated_prompt,
        fields: vec![
            ("Model".to_string(), resp.model.clone(), true),
            ("Seed".to_string(), resp.seed_used.to_string(), true),
            ("Size".to_string(), dims, true),
            ("Time".to_string(), format!("{time_secs:.1}s"), true),
        ],
        color: COLOR_SUCCESS,
    }
}

/// Format the model list into embed data.
/// Returns one embed per chunk of models (Discord embeds have field limits).
pub fn format_model_list(models: &[ModelInfoExtended]) -> EmbedData {
    if models.is_empty() {
        return EmbedData {
            title: "Available Models".to_string(),
            description: "No models configured on the server.".to_string(),
            fields: vec![],
            color: COLOR_WARNING,
        };
    }

    let mut lines = Vec::new();
    for m in models {
        let status = if m.info.is_loaded {
            "loaded"
        } else if m.downloaded {
            "ready"
        } else {
            "not downloaded"
        };
        let size = format!("{:.1}GB", m.info.size_gb);
        lines.push(format!(
            "**{}** — {} `{}` [{}]",
            m.info.name, m.info.family, size, status
        ));
    }

    let description = lines.join("\n");
    // Truncate if too long for Discord (4096 char limit)
    let description = if description.len() > 4000 {
        format!("{}...", &description[..3997])
    } else {
        description
    };

    EmbedData {
        title: "Available Models".to_string(),
        description,
        fields: vec![],
        color: COLOR_INFO,
    }
}

/// Format server status into embed data.
pub fn format_server_status(status: &ServerStatus) -> EmbedData {
    let uptime = format_duration_secs(status.uptime_secs);
    let version = match (&status.git_sha, &status.build_date) {
        (Some(sha), Some(date)) => format!("{} ({sha}, {date})", status.version),
        (Some(sha), None) => format!("{} ({sha})", status.version),
        _ => status.version.clone(),
    };

    let loaded = if status.models_loaded.is_empty() {
        "None".to_string()
    } else {
        status.models_loaded.join(", ")
    };

    let busy_status = if status.busy { "Generating..." } else { "Idle" };

    let mut fields = vec![
        ("Version".to_string(), version, true),
        ("Status".to_string(), busy_status.to_string(), true),
        ("Uptime".to_string(), uptime, true),
        ("Models Loaded".to_string(), loaded, false),
    ];

    if let Some(gpu) = &status.gpu_info {
        fields.push((
            "GPU".to_string(),
            format!(
                "{} ({}MB / {}MB VRAM)",
                gpu.name, gpu.vram_used_mb, gpu.vram_total_mb
            ),
            false,
        ));
    }

    EmbedData {
        title: "Server Status".to_string(),
        description: String::new(),
        fields,
        color: if status.busy {
            COLOR_WARNING
        } else {
            COLOR_INFO
        },
    }
}

/// Format an error message into embed data.
pub fn format_error(msg: &str) -> EmbedData {
    EmbedData {
        title: "Error".to_string(),
        description: msg.to_string(),
        fields: vec![],
        color: COLOR_ERROR,
    }
}

/// Format seconds into a human-readable duration string.
fn format_duration_secs(secs: u64) -> String {
    let days = secs / 86400;
    let hours = (secs % 86400) / 3600;
    let minutes = (secs % 3600) / 60;

    if days > 0 {
        format!("{days}d {hours}h {minutes}m")
    } else if hours > 0 {
        format!("{hours}h {minutes}m")
    } else {
        format!("{minutes}m")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::{GpuInfo, ImageData, OutputFormat};

    #[test]
    fn format_progress_queued_zero() {
        let event = SseProgressEvent::Queued { position: 0 };
        assert_eq!(format_progress(&event), "Starting generation...");
    }

    #[test]
    fn format_progress_queued_nonzero() {
        let event = SseProgressEvent::Queued { position: 3 };
        assert_eq!(format_progress(&event), "Queued — position #3");
    }

    #[test]
    fn format_progress_stage_start() {
        let event = SseProgressEvent::StageStart {
            name: "Loading T5 encoder".to_string(),
        };
        assert_eq!(format_progress(&event), "Loading: Loading T5 encoder...");
    }

    #[test]
    fn format_progress_stage_done() {
        let event = SseProgressEvent::StageDone {
            name: "T5 encoder".to_string(),
            elapsed_ms: 1234,
        };
        assert_eq!(format_progress(&event), "T5 encoder (1234ms)");
    }

    #[test]
    fn format_progress_info() {
        let event = SseProgressEvent::Info {
            message: "Hello world".to_string(),
        };
        assert_eq!(format_progress(&event), "Hello world");
    }

    #[test]
    fn format_progress_cache_hit() {
        let event = SseProgressEvent::CacheHit {
            resource: "prompt conditioning".to_string(),
        };
        assert_eq!(format_progress(&event), "Cache hit: prompt conditioning");
    }

    #[test]
    fn format_progress_denoise_step() {
        let event = SseProgressEvent::DenoiseStep {
            step: 5,
            total: 20,
            elapsed_ms: 500,
        };
        let text = format_progress(&event);
        assert!(text.contains("5/20"));
        assert!(text.contains("100ms/step"));
        assert!(text.contains('['));
    }

    #[test]
    fn format_progress_download() {
        let event = SseProgressEvent::DownloadProgress {
            filename: "model.safetensors".to_string(),
            file_index: 0,
            total_files: 3,
            bytes_downloaded: 500_000_000,
            bytes_total: 1_000_000_000,
        };
        let text = format_progress(&event);
        assert!(text.contains("model.safetensors"));
        assert!(text.contains("[1/3]"));
        assert!(text.contains("50%"));
    }

    #[test]
    fn format_progress_download_done() {
        let event = SseProgressEvent::DownloadDone {
            filename: "model.safetensors".to_string(),
            file_index: 1,
            total_files: 3,
        };
        let text = format_progress(&event);
        assert!(text.contains("Downloaded"));
        assert!(text.contains("[2/3]"));
    }

    #[test]
    fn format_progress_pull_complete() {
        let event = SseProgressEvent::PullComplete {
            model: "flux-schnell:q8".to_string(),
        };
        assert_eq!(format_progress(&event), "Model flux-schnell:q8 downloaded");
    }

    #[test]
    fn generation_result_basic() {
        let resp = GenerateResponse {
            images: vec![ImageData {
                data: vec![],
                format: OutputFormat::Png,
                width: 1024,
                height: 1024,
                index: 0,
            }],
            generation_time_ms: 5500,
            model: "flux-schnell:q8".to_string(),
            seed_used: 42,
        };
        let embed = format_generation_result(&resp, "a cat on mars");
        assert_eq!(embed.title, "Image Generated");
        assert_eq!(embed.description, "a cat on mars");
        assert_eq!(embed.color, COLOR_SUCCESS);
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Model" && v == "flux-schnell:q8"));
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Seed" && v == "42"));
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Size" && v == "1024x1024"));
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Time" && v == "5.5s"));
    }

    #[test]
    fn generation_result_truncates_long_prompt() {
        let long_prompt = "a".repeat(300);
        let resp = GenerateResponse {
            images: vec![ImageData {
                data: vec![],
                format: OutputFormat::Png,
                width: 512,
                height: 512,
                index: 0,
            }],
            generation_time_ms: 1000,
            model: "test".to_string(),
            seed_used: 1,
        };
        let embed = format_generation_result(&resp, &long_prompt);
        assert!(embed.description.len() <= 260);
        assert!(embed.description.ends_with("..."));
    }

    #[test]
    fn model_list_empty() {
        let embed = format_model_list(&[]);
        assert!(embed.description.contains("No models"));
        assert_eq!(embed.color, COLOR_WARNING);
    }

    #[test]
    fn model_list_with_models() {
        let models = vec![ModelInfoExtended {
            info: mold_core::ModelInfo {
                name: "flux-schnell:q8".to_string(),
                family: "flux".to_string(),
                size_gb: 4.5,
                is_loaded: true,
                last_used: None,
                hf_repo: "test/repo".to_string(),
            },
            defaults: mold_core::ModelDefaults {
                default_steps: 4,
                default_guidance: 0.0,
                default_width: 1024,
                default_height: 1024,
                description: "Fast flux".to_string(),
            },
            downloaded: true,
            disk_usage_bytes: None,
            remaining_download_bytes: None,
        }];
        let embed = format_model_list(&models);
        assert!(embed.description.contains("flux-schnell:q8"));
        assert!(embed.description.contains("loaded"));
        assert!(embed.description.contains("4.5GB"));
    }

    #[test]
    fn server_status_idle() {
        let status = ServerStatus {
            version: "0.2.0".to_string(),
            git_sha: Some("abc1234".to_string()),
            build_date: Some("2026-03-25".to_string()),
            models_loaded: vec!["flux-schnell:q8".to_string()],
            busy: false,
            current_generation: None,
            gpu_info: Some(GpuInfo {
                name: "RTX 4090".to_string(),
                vram_total_mb: 24564,
                vram_used_mb: 8192,
            }),
            uptime_secs: 3661,
        };
        let embed = format_server_status(&status);
        assert_eq!(embed.title, "Server Status");
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Status" && v == "Idle"));
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Uptime" && v == "1h 1m"));
        assert!(embed.fields.iter().any(|(k, _, _)| k == "GPU"));
        assert_eq!(embed.color, COLOR_INFO);
    }

    #[test]
    fn server_status_busy() {
        let status = ServerStatus {
            version: "0.2.0".to_string(),
            git_sha: None,
            build_date: None,
            models_loaded: vec![],
            busy: true,
            current_generation: None,
            gpu_info: None,
            uptime_secs: 60,
        };
        let embed = format_server_status(&status);
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Status" && v == "Generating..."));
        assert!(embed
            .fields
            .iter()
            .any(|(k, v, _)| k == "Models Loaded" && v == "None"));
        assert_eq!(embed.color, COLOR_WARNING);
    }

    #[test]
    fn error_embed() {
        let embed = format_error("Something went wrong");
        assert_eq!(embed.title, "Error");
        assert_eq!(embed.description, "Something went wrong");
        assert_eq!(embed.color, COLOR_ERROR);
    }

    #[test]
    fn duration_formatting() {
        assert_eq!(format_duration_secs(0), "0m");
        assert_eq!(format_duration_secs(59), "0m");
        assert_eq!(format_duration_secs(60), "1m");
        assert_eq!(format_duration_secs(3600), "1h 0m");
        assert_eq!(format_duration_secs(3661), "1h 1m");
        assert_eq!(format_duration_secs(86400), "1d 0h 0m");
        assert_eq!(format_duration_secs(90061), "1d 1h 1m");
    }

    #[test]
    fn progress_bar_boundaries() {
        let bar_start = progress_bar(0, 20);
        assert!(bar_start.contains('['));
        assert!(bar_start.contains(']'));

        let bar_mid = progress_bar(10, 20);
        assert!(bar_mid.contains('='));
        assert!(bar_mid.contains('>'));

        let bar_end = progress_bar(20, 20);
        assert!(bar_end.contains('='));
        assert!(!bar_end.contains('>'));
    }

    #[test]
    fn progress_bar_zero_total() {
        let bar = progress_bar(0, 0);
        assert!(bar.contains('['));
        assert!(bar.contains(']'));
    }
}
