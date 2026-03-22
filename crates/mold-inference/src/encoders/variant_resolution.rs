//! Shared T5 and Qwen3 encoder variant resolution logic.
//!
//! Both FLUX and SD3 use T5-XXL text encoders with identical variant selection
//! logic. Similarly, Z-Image and Flux.2 share Qwen3 variant resolution. This
//! module deduplicates that code.

use anyhow::{bail, Result};
use candle_core::Device;
use std::path::{Path, PathBuf};

use crate::device::{
    fits_in_memory, fmt_gb, qwen3_vram_threshold, should_use_gpu, t5_vram_threshold,
    QWEN3_FP16_VRAM_THRESHOLD, T5_VRAM_THRESHOLD,
};
use crate::progress::ProgressReporter;

/// Resolve which T5 encoder variant to use and where to place it.
///
/// Returns `(encoder_path, on_gpu, device_label)`.
///
/// - `preference`: explicit variant tag (e.g. "q8", "fp16", "auto"), or `None` for auto.
/// - `default_t5_path`: the FP16 T5 encoder path (already validated to exist).
pub(crate) fn resolve_t5_variant(
    progress: &ProgressReporter,
    preference: Option<&str>,
    gpu_device: &Device,
    free_vram: u64,
    default_t5_path: &Path,
) -> Result<(PathBuf, bool, String)> {
    use mold_core::download::{cached_file_path, download_single_file_sync};
    use mold_core::manifest::{find_t5_variant, known_t5_variants, T5_FP16_SIZE};

    let is_cuda = gpu_device.is_cuda();
    let is_metal = gpu_device.is_metal();

    match preference {
        // Explicit quantized variant requested
        Some(tag) if tag != "fp16" && tag != "auto" => {
            let variant = find_t5_variant(tag).ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown T5 variant '{}'. Valid: fp16, auto, q8, q6, q5, q4, q3",
                    tag,
                )
            })?;
            let path = resolve_t5_gguf_path(progress, variant)?;
            let threshold = t5_vram_threshold(variant.size_bytes);
            let on_gpu = should_use_gpu(is_cuda, is_metal, free_vram, threshold);
            let label = if on_gpu {
                "GPU, quantized"
            } else {
                "CPU, quantized"
            };
            progress.info(&format!(
                "Using T5 {} ({}) on {} (explicit)",
                variant.tag,
                fmt_gb(variant.size_bytes),
                if on_gpu { "GPU" } else { "CPU" },
            ));
            Ok((path, on_gpu, label.to_string()))
        }

        // Explicit FP16 requested
        Some("fp16") => {
            let on_gpu = should_use_gpu(is_cuda, is_metal, free_vram, T5_VRAM_THRESHOLD);
            let label = if on_gpu { "GPU" } else { "CPU" };
            progress.info(&format!("Using FP16 T5 on {} (explicit)", label));
            Ok((default_t5_path.to_path_buf(), on_gpu, label.to_string()))
        }

        // Auto mode (default): try FP16 on GPU, then quantized on GPU, then FP16 on CPU
        _ => {
            // Can FP16 T5 fit on GPU?
            if fits_in_memory(is_cuda, is_metal, free_vram, T5_VRAM_THRESHOLD) {
                if is_metal {
                    progress.info("Loading FP16 T5 on GPU (unified memory)");
                } else {
                    progress.info(&format!(
                        "Loading FP16 T5 on GPU ({} free > {} threshold)",
                        fmt_gb(free_vram),
                        fmt_gb(T5_VRAM_THRESHOLD),
                    ));
                }
                return Ok((default_t5_path.to_path_buf(), true, "GPU".to_string()));
            }

            // FP16 won't fit on GPU — try quantized variants (largest first)
            if is_cuda || is_metal {
                for variant in known_t5_variants() {
                    let threshold = t5_vram_threshold(variant.size_bytes);
                    if fits_in_memory(is_cuda, is_metal, free_vram, threshold) {
                        // Check cache first, download if needed
                        let path = match cached_file_path(
                            variant.hf_repo,
                            variant.hf_filename,
                            Some("shared/t5-gguf"),
                        ) {
                            Some(p) => p,
                            None => {
                                progress.info(&format!(
                                    "Downloading T5 {} ({})...",
                                    variant.tag,
                                    fmt_gb(variant.size_bytes),
                                ));
                                tracing::info!(
                                    variant = variant.tag,
                                    repo = variant.hf_repo,
                                    file = variant.hf_filename,
                                    "downloading quantized T5 encoder"
                                );
                                download_single_file_sync(
                                    variant.hf_repo,
                                    variant.hf_filename,
                                    Some("shared/t5-gguf"),
                                )
                                .map_err(|e| {
                                    anyhow::anyhow!("failed to download T5 {}: {e}", variant.tag)
                                })?
                            }
                        };
                        progress.info(&format!(
                            "FP16 T5 ({}) exceeds remaining VRAM ({}). Using quantized T5 {} ({}) on GPU instead.",
                            fmt_gb(T5_FP16_SIZE),
                            fmt_gb(free_vram),
                            variant.tag,
                            fmt_gb(variant.size_bytes),
                        ));
                        return Ok((path, true, format!("GPU, quantized {}", variant.tag)));
                    }
                }
            }

            // On Metal, never fall back to CPU (same memory pool). Use smallest quantized variant.
            if is_metal {
                let variants = known_t5_variants();
                if let Some(smallest) = variants.last() {
                    let path = resolve_t5_gguf_path(progress, smallest)?;
                    progress.info(&format!(
                        "Memory tight — using smallest T5 {} ({}) on GPU to reduce page pressure",
                        smallest.tag,
                        fmt_gb(smallest.size_bytes),
                    ));
                    return Ok((path, true, format!("GPU, quantized {}", smallest.tag)));
                }
            }

            // No quantized variant fits on GPU either — fall back to FP16 on CPU
            if is_cuda || is_metal {
                progress.info(&format!(
                    "Loading FP16 T5 on CPU ({} free, no variant fits on GPU)",
                    fmt_gb(free_vram),
                ));
            } else {
                progress.info("No GPU detected, loading T5 on CPU");
            }
            Ok((default_t5_path.to_path_buf(), false, "CPU".to_string()))
        }
    }
}

/// Resolve the path for a quantized T5 GGUF file: check cache, download if needed.
pub(crate) fn resolve_t5_gguf_path(
    progress: &ProgressReporter,
    variant: &mold_core::manifest::T5Variant,
) -> Result<PathBuf> {
    use mold_core::download::{cached_file_path, download_single_file_sync};

    if let Some(path) =
        cached_file_path(variant.hf_repo, variant.hf_filename, Some("shared/t5-gguf"))
    {
        return Ok(path);
    }
    progress.info(&format!(
        "Downloading T5 {} ({})...",
        variant.tag,
        fmt_gb(variant.size_bytes),
    ));
    download_single_file_sync(variant.hf_repo, variant.hf_filename, Some("shared/t5-gguf"))
        .map_err(|e| anyhow::anyhow!("failed to download T5 {}: {e}", variant.tag))
}

/// Resolve which Qwen3 encoder variant to use and where to place it.
///
/// Returns `(encoder_paths, is_gguf, on_gpu, device_label)`.
///
/// - `preference`: explicit variant tag (e.g. "q8", "bf16", "auto"), or `None` for auto.
/// - `bf16_paths`: BF16 shard paths (may be empty if not available).
/// - `have_bf16`: whether BF16 shards exist on disk.
/// - `prefer_gguf`: if true, auto mode prefers GGUF over BF16 even when BF16 fits.
///   Flux.2 sets this to true because multi-layer extraction (layers 9, 18, 27)
///   only works with the GGUF encoder.
pub(crate) fn resolve_qwen3_variant(
    progress: &ProgressReporter,
    preference: Option<&str>,
    gpu_device: &Device,
    free_vram: u64,
    bf16_paths: &[PathBuf],
    have_bf16: bool,
    prefer_gguf: bool,
) -> Result<(Vec<PathBuf>, bool, bool, String)> {
    use mold_core::download::{cached_file_path, download_single_file_sync};
    use mold_core::manifest::{find_qwen3_variant, known_qwen3_variants};

    let is_cuda = gpu_device.is_cuda();
    let is_metal = gpu_device.is_metal();

    match preference {
        // Explicit quantized variant requested
        Some(tag) if tag != "bf16" && tag != "auto" => {
            let variant = find_qwen3_variant(tag).ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown Qwen3 variant '{}'. Valid: bf16, auto, q8, q6, iq4, q3",
                    tag,
                )
            })?;
            let path = resolve_qwen3_gguf_path(progress, variant)?;
            let threshold = qwen3_vram_threshold(variant.size_bytes);
            let on_gpu = should_use_gpu(is_cuda, is_metal, free_vram, threshold);
            let label = if on_gpu {
                "GPU, quantized"
            } else {
                "CPU, quantized"
            };
            progress.info(&format!(
                "Using Qwen3 {} ({}) on {} (explicit)",
                variant.tag,
                fmt_gb(variant.size_bytes),
                if on_gpu { "GPU" } else { "CPU" },
            ));
            Ok((vec![path], true, on_gpu, label.to_string()))
        }

        // Explicit BF16 requested
        Some("bf16") => {
            if !have_bf16 {
                bail!(
                    "BF16 Qwen3 encoder requested but shard files are missing or not configured. \
                     Either run `mold pull` for a model with Qwen3 or use --qwen3-variant q8/q6/iq4/q3."
                );
            }
            let on_gpu = should_use_gpu(is_cuda, is_metal, free_vram, QWEN3_FP16_VRAM_THRESHOLD);
            let label = if on_gpu { "GPU" } else { "CPU" };
            progress.info(&format!("Using BF16 Qwen3 on {} (explicit)", label));
            Ok((bf16_paths.to_vec(), false, on_gpu, label.to_string()))
        }

        // Auto mode
        _ => {
            if prefer_gguf {
                // Flux.2 path: prefer GGUF because multi-layer extraction requires it.
                // Try quantized variants (largest first) on GPU.
                if is_cuda || is_metal {
                    for variant in known_qwen3_variants() {
                        let threshold = qwen3_vram_threshold(variant.size_bytes);
                        if fits_in_memory(is_cuda, is_metal, free_vram, threshold) {
                            let path = match cached_file_path(
                                variant.hf_repo,
                                variant.hf_filename,
                                Some("shared/qwen3-gguf"),
                            ) {
                                Some(p) => p,
                                None => {
                                    progress.info(&format!(
                                        "Downloading Qwen3 {} ({})...",
                                        variant.tag,
                                        fmt_gb(variant.size_bytes),
                                    ));
                                    download_single_file_sync(
                                        variant.hf_repo,
                                        variant.hf_filename,
                                        Some("shared/qwen3-gguf"),
                                    )
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "failed to download Qwen3 {}: {e}",
                                            variant.tag
                                        )
                                    })?
                                }
                            };
                            progress.info(&format!(
                                "Using quantized Qwen3 {} ({}) on GPU",
                                variant.tag,
                                fmt_gb(variant.size_bytes),
                            ));
                            return Ok((
                                vec![path],
                                true,
                                true,
                                format!("GPU, quantized {}", variant.tag),
                            ));
                        }
                    }
                }

                // Fall back to BF16 on CPU
                if have_bf16 {
                    progress.info("Loading BF16 Qwen3 on CPU (no variant fits on GPU)");
                    Ok((bf16_paths.to_vec(), false, false, "CPU".to_string()))
                } else {
                    bail!("No Qwen3 encoder available (no BF16 files and no GGUF cached)")
                }
            } else {
                // Z-Image path: try BF16 on GPU first, then quantized, then BF16 on CPU.
                if have_bf16
                    && fits_in_memory(is_cuda, is_metal, free_vram, QWEN3_FP16_VRAM_THRESHOLD)
                {
                    if is_metal {
                        progress.info("Loading BF16 Qwen3 on GPU (unified memory)");
                    } else {
                        progress.info(&format!(
                            "Loading BF16 Qwen3 on GPU ({} free > {} threshold, drop-and-reload)",
                            fmt_gb(free_vram),
                            fmt_gb(QWEN3_FP16_VRAM_THRESHOLD),
                        ));
                    }
                    return Ok((bf16_paths.to_vec(), false, true, "GPU".to_string()));
                }

                // BF16 won't fit (or shards missing) — try quantized variants (largest first)
                if is_cuda || is_metal || !have_bf16 {
                    for variant in known_qwen3_variants() {
                        let threshold = qwen3_vram_threshold(variant.size_bytes);
                        if fits_in_memory(is_cuda, is_metal, free_vram, threshold)
                            || (!is_cuda && !is_metal)
                        {
                            let path = match cached_file_path(
                                variant.hf_repo,
                                variant.hf_filename,
                                Some("shared/qwen3-gguf"),
                            ) {
                                Some(p) => p,
                                None => {
                                    progress.info(&format!(
                                        "Downloading Qwen3 {} ({})...",
                                        variant.tag,
                                        fmt_gb(variant.size_bytes),
                                    ));
                                    tracing::info!(
                                        variant = variant.tag,
                                        repo = variant.hf_repo,
                                        file = variant.hf_filename,
                                        "downloading quantized Qwen3 encoder"
                                    );
                                    download_single_file_sync(
                                        variant.hf_repo,
                                        variant.hf_filename,
                                        Some("shared/qwen3-gguf"),
                                    )
                                    .map_err(|e| {
                                        anyhow::anyhow!(
                                            "failed to download Qwen3 {}: {e}",
                                            variant.tag
                                        )
                                    })?
                                }
                            };
                            let on_gpu = is_cuda || is_metal;
                            progress.info(&format!(
                                "Using Qwen3 {} ({}) on {}",
                                variant.tag,
                                fmt_gb(variant.size_bytes),
                                if on_gpu { "GPU" } else { "CPU" },
                            ));
                            return Ok((
                                vec![path],
                                true,
                                on_gpu,
                                format!(
                                    "{}, quantized {}",
                                    if on_gpu { "GPU" } else { "CPU" },
                                    variant.tag
                                ),
                            ));
                        }
                    }
                }

                // On Metal, never fall back to CPU (same memory pool). Use smallest quantized variant on GPU.
                if is_metal {
                    let variants = known_qwen3_variants();
                    if let Some(smallest) = variants.last() {
                        let path = match cached_file_path(
                            smallest.hf_repo,
                            smallest.hf_filename,
                            Some("shared/qwen3-gguf"),
                        ) {
                            Some(p) => p,
                            None => {
                                progress.info(&format!(
                                    "Downloading Qwen3 {} ({})...",
                                    smallest.tag,
                                    fmt_gb(smallest.size_bytes),
                                ));
                                download_single_file_sync(
                                    smallest.hf_repo,
                                    smallest.hf_filename,
                                    Some("shared/qwen3-gguf"),
                                )
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "failed to download Qwen3 {}: {e}",
                                        smallest.tag
                                    )
                                })?
                            }
                        };
                        progress.info(&format!(
                            "Memory tight — using smallest Qwen3 {} ({}) on GPU to reduce page pressure",
                            smallest.tag,
                            fmt_gb(smallest.size_bytes),
                        ));
                        return Ok((
                            vec![path],
                            true,
                            true,
                            format!("GPU, quantized {}", smallest.tag),
                        ));
                    }
                }

                // Fall back to BF16 on CPU (only if shards are available)
                if have_bf16 {
                    if is_cuda || is_metal {
                        progress.info(&format!(
                            "Loading BF16 Qwen3 on CPU ({} free, no variant fits on GPU)",
                            fmt_gb(free_vram),
                        ));
                    } else {
                        progress.info("No GPU detected, loading Qwen3 on CPU");
                    }
                    return Ok((bf16_paths.to_vec(), false, false, "CPU".to_string()));
                }

                bail!(
                    "no Qwen3 text encoder available: BF16 shards not configured and no \
                     quantized variant could be resolved. Run `mold pull` for a model with \
                     Qwen3 or use --qwen3-variant q8/q6/iq4/q3."
                );
            }
        }
    }
}

/// Resolve the path for a quantized Qwen3 GGUF file: check cache, download if needed.
pub(crate) fn resolve_qwen3_gguf_path(
    progress: &ProgressReporter,
    variant: &mold_core::manifest::Qwen3Variant,
) -> Result<PathBuf> {
    use mold_core::download::{cached_file_path, download_single_file_sync};

    if let Some(path) = cached_file_path(
        variant.hf_repo,
        variant.hf_filename,
        Some("shared/qwen3-gguf"),
    ) {
        return Ok(path);
    }
    progress.info(&format!(
        "Downloading Qwen3 {} ({})...",
        variant.tag,
        fmt_gb(variant.size_bytes),
    ));
    download_single_file_sync(
        variant.hf_repo,
        variant.hf_filename,
        Some("shared/qwen3-gguf"),
    )
    .map_err(|e| anyhow::anyhow!("failed to download Qwen3 {}: {e}", variant.tag))
}
