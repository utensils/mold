//! `mold upscale` command implementation.

use anyhow::{Context, Result};
use clap_complete::engine::CompletionCandidate;
use mold_core::manifest::{known_manifests, resolve_model_name, UPSCALER_FAMILIES};
use mold_core::{Config, OutputFormat, UpscaleRequest};
use std::io::{IsTerminal, Read, Write};
use std::path::Path;

use crate::theme;

/// Provide upscaler model name completions for shell tab-completion.
pub fn complete_upscaler_model() -> Vec<CompletionCandidate> {
    known_manifests()
        .iter()
        .filter(|m| UPSCALER_FAMILIES.contains(&m.family.as_str()))
        .map(|m| CompletionCandidate::new(&m.name))
        .collect()
}

/// Default upscaler model when none specified.
fn default_upscaler_model(config: &Config) -> String {
    // Check MOLD_UPSCALE_MODEL env var first
    if let Ok(model) = std::env::var("MOLD_UPSCALE_MODEL") {
        return model;
    }
    // Check if any upscaler is already downloaded
    for manifest in known_manifests() {
        if !UPSCALER_FAMILIES.contains(&manifest.family.as_str()) {
            continue;
        }
        if config.models.contains_key(&manifest.name) {
            return manifest.name.clone();
        }
    }
    // Default
    "real-esrgan-x4plus:fp16".to_string()
}

#[allow(clippy::too_many_arguments)]
pub async fn run(
    image_path: String,
    model: Option<String>,
    output: Option<String>,
    format: OutputFormat,
    tile_size: Option<u32>,
    host: Option<String>,
    local: bool,
    preview: bool,
) -> Result<()> {
    let config = Config::load_or_default();
    let model = model.unwrap_or_else(|| default_upscaler_model(&config));
    let model = resolve_model_name(&model);

    // Read input image
    let image_bytes = if image_path == "-" {
        let mut buf = Vec::new();
        std::io::stdin().read_to_end(&mut buf)?;
        buf
    } else {
        let path = Path::new(&image_path);
        if !path.exists() {
            anyhow::bail!("input image not found: {image_path}");
        }
        std::fs::read(path).with_context(|| format!("failed to read image '{image_path}'"))?
    };

    // Validate image format
    let is_png = image_bytes.len() >= 4 && image_bytes[..4] == [0x89, 0x50, 0x4E, 0x47];
    let is_jpeg = image_bytes.len() >= 2 && image_bytes[..2] == [0xFF, 0xD8];
    if !is_png && !is_jpeg {
        anyhow::bail!("input image must be PNG or JPEG");
    }

    let req = UpscaleRequest {
        model: model.clone(),
        image: image_bytes,
        output_format: format,
        tile_size,
    };

    // Try server first (unless --local)
    let base_url = host
        .or_else(|| std::env::var("MOLD_HOST").ok())
        .unwrap_or_else(|| "http://localhost:7680".to_string());

    if !local {
        let client = mold_core::MoldClient::new(&base_url);
        match client.upscale(&req).await {
            Ok(resp) => {
                return write_output(
                    &resp.image.data,
                    &image_path,
                    &output,
                    &format,
                    &resp,
                    preview,
                );
            }
            Err(e) => {
                if mold_core::MoldClient::is_connection_error(&e) {
                    eprintln!(
                        "{} Server unavailable, falling back to local inference",
                        theme::icon_info()
                    );
                } else {
                    return Err(e);
                }
            }
        }
    }

    // Local inference
    upscale_local(req, &image_path, &output, &format, &config, preview).await
}

async fn upscale_local(
    req: UpscaleRequest,
    image_path: &str,
    output: &Option<String>,
    format: &OutputFormat,
    config: &Config,
    preview: bool,
) -> Result<()> {
    // Auto-pull if model not downloaded
    let model_name = &req.model;
    if !config.models.contains_key(model_name) {
        eprintln!(
            "{} Model '{}' not found locally, pulling...",
            theme::icon_info(),
            model_name
        );
        crate::commands::pull::run(model_name, &mold_core::download::PullOptions::default())
            .await?;
        // Reload config after pull
        return Box::pin(upscale_local(
            req,
            image_path,
            output,
            format,
            &Config::load_or_default(),
            preview,
        ))
        .await;
    }

    // Get model weights path from config
    let model_cfg = config
        .models
        .get(model_name)
        .ok_or_else(|| anyhow::anyhow!("model '{}' not configured after pull", model_name))?;

    let weights_path = model_cfg
        .transformer
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no weights path for model '{}'", model_name))?;
    let weights_path = std::path::PathBuf::from(weights_path);

    // Create engine and run upscaling in a blocking thread
    let model_name_owned = model_name.clone();
    let req_clone = req.clone();
    let best_gpu_ordinal =
        mold_inference::device::select_best_gpu(&mold_inference::device::discover_gpus())
            .map(|g| g.ordinal)
            .unwrap_or(0);

    let resp = tokio::task::spawn_blocking(move || -> Result<mold_core::UpscaleResponse> {
        // Local upscale should target the GPU with the most free VRAM instead
        // of hardcoding ordinal 0 on multi-GPU hosts.
        let mut engine = mold_inference::create_upscale_engine(
            model_name_owned,
            weights_path,
            mold_inference::LoadStrategy::Sequential,
            best_gpu_ordinal,
        )?;

        // Set up progress callback for stderr
        engine.set_on_progress(Box::new(|event| {
            use mold_inference::ProgressEvent;
            match event {
                ProgressEvent::StageStart { name } => {
                    eprintln!("{} {name}", theme::icon_info());
                }
                ProgressEvent::StageDone { name, elapsed } => {
                    eprintln!(
                        "{} {name} ({:.1}s)",
                        theme::icon_done(),
                        elapsed.as_secs_f64()
                    );
                }
                ProgressEvent::Info { message } => {
                    eprintln!("{} {message}", theme::icon_info());
                }
                ProgressEvent::DenoiseStep { step, total, .. } if total > 1 => {
                    eprint!("\r{} Tile {step}/{total}", theme::icon_info());
                    if step == total {
                        eprintln!();
                    }
                }
                _ => {}
            }
        }));

        engine.upscale(&req_clone)
    })
    .await??;

    write_output(&resp.image.data, image_path, output, format, &resp, preview)
}

fn write_output(
    data: &[u8],
    image_path: &str,
    output: &Option<String>,
    format: &OutputFormat,
    resp: &mold_core::UpscaleResponse,
    preview: bool,
) -> Result<()> {
    let is_stdout_tty = std::io::stdout().is_terminal();

    // Determine output path
    let output_path = if let Some(ref path) = output {
        if path == "-" {
            None // stdout
        } else {
            Some(path.clone())
        }
    } else if !is_stdout_tty {
        None // piped, write to stdout
    } else if image_path == "-" {
        // stdin input, generate a default name
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Some(format!("mold-upscaled-{timestamp}.{format}"))
    } else {
        // Derive from input: foo.png -> foo_upscaled.png
        let path = Path::new(image_path);
        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
        let ext = format.extension();
        let dir = path.parent().unwrap_or(Path::new("."));
        Some(
            dir.join(format!("{stem}_upscaled.{ext}"))
                .to_string_lossy()
                .to_string(),
        )
    };

    if let Some(ref path) = output_path {
        std::fs::write(path, data)
            .with_context(|| format!("failed to write output to '{path}'"))?;
        eprintln!(
            "{} {}x{} -> {}x{} ({}x) saved to {path}",
            theme::icon_done(),
            resp.original_width,
            resp.original_height,
            resp.image.width,
            resp.image.height,
            resp.scale_factor,
        );
    } else {
        // Write to stdout
        std::io::stdout().write_all(data)?;
        std::io::stdout().flush()?;
        eprintln!(
            "{} {}x{} -> {}x{} ({}x)",
            theme::icon_done(),
            resp.original_width,
            resp.original_height,
            resp.image.width,
            resp.image.height,
            resp.scale_factor,
        );
    }
    if preview {
        super::generate::preview_image(data);
    }
    Ok(())
}
