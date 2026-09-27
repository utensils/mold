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

/// Which Qwen3 architecture to select variants for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Qwen3Size {
    /// Qwen3-4B (hidden_size=2560) — used by Klein-4B and Z-Image.
    B4,
    /// Qwen3-8B (hidden_size=4096) — used by Klein-9B.
    B8,
    /// Qwen3-VL-8B-Instruct's language model (hidden_size=4096, RoPE base 5e6)
    /// — Qwen Image 2.1's text encoder. Its GGUF list is the official
    /// `Qwen/Qwen3-VL-8B-Instruct-GGUF` conversion, sha-pinned; the vision
    /// tower is never quantized and always loads from the BF16 shards.
    Vl8b,
}

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

/// One Qwen3 size's variant registry and thresholds.
struct Qwen3Registry {
    variants: &'static [mold_core::manifest::Qwen3Variant],
    find: fn(&str) -> Option<&'static mold_core::manifest::Qwen3Variant>,
    fp16_threshold: u64,
    cache_subdir: &'static str,
    size_label: &'static str,
    /// Whether auto mode may pick a variant; explicit tags reach every one.
    auto_eligible: fn(&mold_core::manifest::Qwen3Variant) -> bool,
}

fn qwen3_registry(qwen3_size: Qwen3Size) -> Qwen3Registry {
    // The 8B language models (Qwen3-8B and Qwen3-VL-8B's) are ~16.4 GB of
    // BF16 — apply the same 1.25x headroom as the 4B threshold.
    let threshold_8b = (mold_core::manifest::QWEN3_8B_FP16_SIZE as f64 * 1.25) as u64;
    match qwen3_size {
        Qwen3Size::B4 => Qwen3Registry {
            variants: mold_core::manifest::known_qwen3_variants(),
            find: mold_core::manifest::find_qwen3_variant,
            fp16_threshold: QWEN3_FP16_VRAM_THRESHOLD,
            cache_subdir: "shared/qwen3-gguf",
            size_label: "Qwen3-4B",
            auto_eligible: |_| true,
        },
        Qwen3Size::B8 => Qwen3Registry {
            variants: mold_core::manifest::known_qwen3_8b_variants(),
            find: mold_core::manifest::find_qwen3_8b_variant,
            fp16_threshold: threshold_8b,
            cache_subdir: "shared/qwen3-8b-gguf",
            size_label: "Qwen3-8B",
            auto_eligible: |_| true,
        },
        Qwen3Size::Vl8b => Qwen3Registry {
            variants: mold_core::manifest::known_qwen3_vl_8b_variants(),
            find: mold_core::manifest::find_qwen3_vl_8b_variant,
            fp16_threshold: threshold_8b,
            cache_subdir: "shared/qwen3-vl-8b-gguf",
            size_label: "Qwen3-VL-8B",
            auto_eligible: mold_core::manifest::qwen3_vl_8b_variant_auto_eligible,
        },
    }
}

/// Which arm of the policy produced a [`Qwen3Choice`] — only the progress
/// line and the device label depend on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Qwen3Route {
    Explicit,
    PreferGguf,
    Auto,
    MetalSmallest,
    Fallback,
}

/// Which Qwen3 language model a render loads, and where.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen3Choice {
    /// The BF16 shards.
    Bf16 { on_gpu: bool },
    /// One of the size's GGUF variants.
    Gguf {
        variant: &'static mold_core::manifest::Qwen3Variant,
        on_gpu: bool,
    },
}

impl Qwen3Choice {
    pub fn on_gpu(self) -> bool {
        match self {
            Self::Bf16 { on_gpu } | Self::Gguf { on_gpu, .. } => on_gpu,
        }
    }
}

/// The pure half of [`resolve_qwen3_variant`] — no I/O — so an engine and
/// mold-server's planner reach the same answer from the same inputs.
#[allow(clippy::too_many_arguments)]
fn choose_qwen3_variant_routed(
    qwen3_size: Qwen3Size,
    preference: Option<&str>,
    is_cuda: bool,
    is_metal: bool,
    free_vram: u64,
    have_bf16: bool,
    prefer_gguf: bool,
) -> Result<(Qwen3Choice, Qwen3Route)> {
    let registry = qwen3_registry(qwen3_size);
    let size_label = registry.size_label;
    match preference {
        // Explicit quantized variant requested
        Some(tag) if tag != "bf16" && tag != "auto" => {
            let variant = (registry.find)(tag).ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown {} variant '{}'. Valid: bf16, auto, {}",
                    size_label,
                    tag,
                    registry
                        .variants
                        .iter()
                        .map(|variant| variant.tag)
                        .collect::<Vec<_>>()
                        .join(", "),
                )
            })?;
            let on_gpu = should_use_gpu(
                is_cuda,
                is_metal,
                free_vram,
                qwen3_vram_threshold(variant.size_bytes),
            );
            Ok((Qwen3Choice::Gguf { variant, on_gpu }, Qwen3Route::Explicit))
        }

        // Explicit BF16 requested
        Some("bf16") => {
            if !have_bf16 {
                bail!(
                    "BF16 {} encoder requested but shard files are missing or not configured. \
                     Either run `mold pull` for a model with Qwen3 or use --qwen3-variant q8/q6/iq4/q3.",
                    size_label,
                );
            }
            let on_gpu = should_use_gpu(is_cuda, is_metal, free_vram, registry.fp16_threshold);
            Ok((Qwen3Choice::Bf16 { on_gpu }, Qwen3Route::Explicit))
        }

        // Auto mode
        _ => {
            if prefer_gguf {
                // Flux.2 path: prefer GGUF because it's smaller and faster to load.
                // Try quantized variants (largest first) on GPU.
                if is_cuda || is_metal {
                    if let Some(variant) = registry
                        .variants
                        .iter()
                        .filter(|variant| (registry.auto_eligible)(variant))
                        .find(|variant| {
                            fits_in_memory(
                                is_cuda,
                                is_metal,
                                free_vram,
                                qwen3_vram_threshold(variant.size_bytes),
                            )
                        })
                    {
                        return Ok((
                            Qwen3Choice::Gguf {
                                variant,
                                on_gpu: true,
                            },
                            Qwen3Route::PreferGguf,
                        ));
                    }
                }
                // Fall back to BF16 on CPU
                if have_bf16 {
                    return Ok((Qwen3Choice::Bf16 { on_gpu: false }, Qwen3Route::Fallback));
                }
                bail!(
                    "No {} encoder available (no BF16 files and no GGUF cached)",
                    size_label
                )
            }

            // Z-Image path: try BF16 on GPU first, then quantized, then BF16 on CPU.
            if have_bf16 && fits_in_memory(is_cuda, is_metal, free_vram, registry.fp16_threshold) {
                return Ok((Qwen3Choice::Bf16 { on_gpu: true }, Qwen3Route::Auto));
            }

            // BF16 won't fit (or shards missing) — try quantized variants (largest first)
            if is_cuda || is_metal || !have_bf16 {
                if let Some(variant) = registry
                    .variants
                    .iter()
                    .filter(|variant| (registry.auto_eligible)(variant))
                    .find(|variant| {
                        fits_in_memory(
                            is_cuda,
                            is_metal,
                            free_vram,
                            qwen3_vram_threshold(variant.size_bytes),
                        ) || (!is_cuda && !is_metal)
                    })
                {
                    return Ok((
                        Qwen3Choice::Gguf {
                            variant,
                            on_gpu: is_cuda || is_metal,
                        },
                        Qwen3Route::Auto,
                    ));
                }
            }

            // On Metal, never fall back to CPU (same memory pool). Use smallest quantized variant on GPU.
            if is_metal {
                if let Some(variant) = registry
                    .variants
                    .iter()
                    .rev()
                    .find(|variant| (registry.auto_eligible)(variant))
                {
                    return Ok((
                        Qwen3Choice::Gguf {
                            variant,
                            on_gpu: true,
                        },
                        Qwen3Route::MetalSmallest,
                    ));
                }
            }

            // Fall back to BF16 on CPU (only if shards are available)
            if have_bf16 {
                return Ok((Qwen3Choice::Bf16 { on_gpu: false }, Qwen3Route::Fallback));
            }

            bail!(
                "no {} text encoder available: BF16 shards not configured and no \
                 quantized variant could be resolved. Run `mold pull` for a model with \
                 Qwen3 or use --qwen3-variant q8/q6/iq4/q3.",
                size_label,
            );
        }
    }
}

/// [`resolve_qwen3_variant`]'s decision without the acquisition.
#[allow(clippy::too_many_arguments)]
pub fn choose_qwen3_variant(
    qwen3_size: Qwen3Size,
    preference: Option<&str>,
    is_cuda: bool,
    is_metal: bool,
    free_vram: u64,
    have_bf16: bool,
    prefer_gguf: bool,
) -> Result<Qwen3Choice> {
    choose_qwen3_variant_routed(
        qwen3_size,
        preference,
        is_cuda,
        is_metal,
        free_vram,
        have_bf16,
        prefer_gguf,
    )
    .map(|(choice, _)| choice)
}

/// Qwen Image 2.1's Qwen3-VL-8B decision: the BF16 shards are always
/// installed (the vision tower lives there), and auto mode is the Z-Image arm
/// — BF16 when it fits, else the largest official GGUF that fits. `free_vram`
/// is what the card has left once the transformer and VAE are resident.
pub fn choose_qwen3_vl_variant(
    preference: Option<&str>,
    is_cuda: bool,
    is_metal: bool,
    free_vram: u64,
) -> Result<Qwen3Choice> {
    choose_qwen3_variant(
        Qwen3Size::Vl8b,
        preference,
        is_cuda,
        is_metal,
        free_vram,
        true,
        false,
    )
}

/// Resolve which Qwen3 encoder variant to use and where to place it.
///
/// Returns `(encoder_paths, is_gguf, on_gpu, device_label)`.
///
/// - `preference`: explicit variant tag (e.g. "q8", "bf16", "auto"), or `None` for auto.
/// - `bf16_paths`: BF16 shard paths (may be empty if not available).
/// - `have_bf16`: whether BF16 shards exist on disk.
/// - `prefer_gguf`: if true, auto mode prefers GGUF over BF16 even when BF16 fits.
///   Flux.2 sets this to true because GGUF is smaller and faster to load.
///   Both GGUF and BF16 encoders support multi-layer extraction (layers 9, 18, 27).
/// - `qwen3_size`: selects the GGUF variant registry and FP16 size threshold —
///   Qwen3-4B (Klein-4B / Z-Image), Qwen3-8B (Klein-9B), or Qwen3-VL-8B (Qwen
///   Image 2.1, whose list is sha-pinned and verified before use).
///
/// The decision is [`choose_qwen3_variant`]; this adds the acquisition and the
/// progress lines.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn resolve_qwen3_variant(
    progress: &ProgressReporter,
    preference: Option<&str>,
    gpu_device: &Device,
    free_vram: u64,
    bf16_paths: &[PathBuf],
    have_bf16: bool,
    prefer_gguf: bool,
    qwen3_size: Qwen3Size,
) -> Result<(Vec<PathBuf>, bool, bool, String)> {
    let is_cuda = gpu_device.is_cuda();
    let is_metal = gpu_device.is_metal();
    let registry = qwen3_registry(qwen3_size);
    let size_label = registry.size_label;
    let (choice, route) = choose_qwen3_variant_routed(
        qwen3_size,
        preference,
        is_cuda,
        is_metal,
        free_vram,
        have_bf16,
        prefer_gguf,
    )?;
    let device_word = |on_gpu: bool| if on_gpu { "GPU" } else { "CPU" };
    match (choice, route) {
        (Qwen3Choice::Gguf { variant, on_gpu }, Qwen3Route::Explicit) => {
            let path =
                resolve_qwen3_gguf_path_with_cache(progress, variant, registry.cache_subdir)?;
            progress.info(&format!(
                "Using {} {} ({}) on {} (explicit)",
                size_label,
                variant.tag,
                fmt_gb(variant.size_bytes),
                device_word(on_gpu),
            ));
            let label = if on_gpu {
                "GPU, quantized"
            } else {
                "CPU, quantized"
            };
            Ok((vec![path], true, on_gpu, label.to_string()))
        }
        (Qwen3Choice::Bf16 { on_gpu }, Qwen3Route::Explicit) => {
            let label = device_word(on_gpu);
            progress.info(&format!(
                "Using BF16 {} on {} (explicit)",
                size_label, label
            ));
            Ok((bf16_paths.to_vec(), false, on_gpu, label.to_string()))
        }
        (Qwen3Choice::Gguf { variant, on_gpu }, route) => {
            let path =
                resolve_qwen3_gguf_path_with_cache(progress, variant, registry.cache_subdir)?;
            match route {
                Qwen3Route::PreferGguf => progress.info(&format!(
                    "Using quantized {} {} ({}) on GPU",
                    size_label,
                    variant.tag,
                    fmt_gb(variant.size_bytes),
                )),
                Qwen3Route::MetalSmallest => progress.info(&format!(
                    "Memory tight — using smallest {} {} ({}) on GPU to reduce page pressure",
                    size_label,
                    variant.tag,
                    fmt_gb(variant.size_bytes),
                )),
                _ => progress.info(&format!(
                    "Using {} {} ({}) on {}",
                    size_label,
                    variant.tag,
                    fmt_gb(variant.size_bytes),
                    device_word(on_gpu),
                )),
            }
            let label = match route {
                Qwen3Route::PreferGguf | Qwen3Route::MetalSmallest => {
                    format!("GPU, quantized {}", variant.tag)
                }
                _ => format!("{}, quantized {}", device_word(on_gpu), variant.tag),
            };
            Ok((vec![path], true, on_gpu, label))
        }
        (Qwen3Choice::Bf16 { on_gpu: true }, _) => {
            if is_metal {
                progress.info(&format!(
                    "Loading BF16 {} on GPU (unified memory)",
                    size_label
                ));
            } else {
                progress.info(&format!(
                    "Loading BF16 {} on GPU ({} free > {} threshold, drop-and-reload)",
                    size_label,
                    fmt_gb(free_vram),
                    fmt_gb(registry.fp16_threshold),
                ));
            }
            Ok((bf16_paths.to_vec(), false, true, "GPU".to_string()))
        }
        (Qwen3Choice::Bf16 { on_gpu: false }, _) => {
            if prefer_gguf {
                progress.info(&format!(
                    "Loading BF16 {} on CPU (no variant fits on GPU)",
                    size_label
                ));
            } else if is_cuda || is_metal {
                progress.info(&format!(
                    "Loading BF16 {} on CPU ({} free, no variant fits on GPU)",
                    size_label,
                    fmt_gb(free_vram),
                ));
            } else {
                let gpu_available = candle_core::utils::cuda_is_available()
                    || candle_core::utils::metal_is_available();
                progress.info(&cpu_encoder_line(size_label, gpu_available));
            }
            Ok((bf16_paths.to_vec(), false, false, "CPU".to_string()))
        }
    }
}
/// The progress line for a BF16 encoder resolved onto a CPU device. A CPU
/// device on a machine that HAS a usable GPU is a placement (Qwen Image 2.1's
/// sequential plan puts Qwen3-VL on the host for a small card), not a missing
/// GPU, so only a machine with none says "No GPU detected".
fn cpu_encoder_line(size_label: &str, gpu_available: bool) -> String {
    if gpu_available {
        format!("Loading BF16 {size_label} on CPU (the encoder is placed on the host)")
    } else {
        format!("No GPU detected, loading {size_label} on CPU")
    }
}

/// Resolve the path for a quantized Qwen3 GGUF file: check cache, download if needed.
fn resolve_qwen3_gguf_path_with_cache(
    progress: &ProgressReporter,
    variant: &mold_core::manifest::Qwen3Variant,
    cache_subdir: &str,
) -> Result<PathBuf> {
    use mold_core::download::{cached_file_path, download_single_file_sync};

    let path = match cached_file_path(variant.hf_repo, variant.hf_filename, Some(cache_subdir)) {
        Some(path) => path,
        None => {
            progress.info(&format!(
                "Downloading Qwen3 {} ({})...",
                variant.tag,
                fmt_gb(variant.size_bytes),
            ));
            download_single_file_sync(variant.hf_repo, variant.hf_filename, Some(cache_subdir))
                .map_err(|e| anyhow::anyhow!("failed to download Qwen3 {}: {e}", variant.tag))?
        }
    };
    verify_qwen3_variant_pin(&path, variant)?;
    Ok(path)
}

/// Hold a pinned variant to its digest before its bytes are loaded.
///
/// The download writes a `.sha256-verified` marker recording what landed, so
/// a marker naming the pinned digest is the fast path; anything else — an
/// unpinned acquisition, a replaced file, a marker from another revision — is
/// hashed and refused on mismatch. Unpinned variants pass through.
fn verify_qwen3_variant_pin(
    path: &Path,
    variant: &mold_core::manifest::Qwen3Variant,
) -> Result<()> {
    let Some(expected) = variant.sha256 else {
        return Ok(());
    };
    if mold_core::download::recorded_sha256_marker(path)
        .is_some_and(|recorded| recorded.eq_ignore_ascii_case(expected))
        && std::fs::metadata(path).is_ok_and(|meta| meta.len() == variant.size_bytes)
    {
        return Ok(());
    }
    mold_core::download::verify_pinned_file(path, expected, variant.hf_filename, variant.hf_repo)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}

/// Resolve the path for a quantized Qwen2.5-VL GGUF file: check cache, download if needed.
pub(crate) fn resolve_qwen2_vl_gguf_path(
    progress: &ProgressReporter,
    variant: &mold_core::manifest::Qwen2VlVariant,
) -> Result<PathBuf> {
    use mold_core::download::{cached_file_path, download_single_file_sync};

    const CACHE_SUBDIR: &str = "shared/qwen2-vl-gguf";

    if let Some(path) = cached_file_path(variant.hf_repo, variant.hf_filename, Some(CACHE_SUBDIR)) {
        return Ok(path);
    }
    progress.info(&format!(
        "Downloading Qwen2.5-VL {} ({})...",
        variant.tag,
        fmt_gb(variant.size_bytes),
    ));
    download_single_file_sync(variant.hf_repo, variant.hf_filename, Some(CACHE_SUBDIR))
        .map_err(|e| anyhow::anyhow!("failed to download Qwen2.5-VL {}: {e}", variant.tag))
}

// ── UMT5 (Wan) ──────────────────────────────────────────────────────────────

/// Whether an encoder of `size_bytes` fits this device's free VRAM.
///
/// Split out for the prepared route, where admission has already chosen the
/// file and only the placement is still open. It applies the same threshold
/// the auto policy does, so a prepared render and an unprepared one land the
/// same encoder on the same device.
pub(crate) fn umt5_fits_on_gpu(
    gpu_device: &Device,
    free_vram: u64,
    size_bytes: u64,
    quantized: bool,
) -> bool {
    let threshold = if gpu_device.is_metal() && quantized {
        crate::device::t5_metal_gguf_vram_threshold(size_bytes)
    } else {
        t5_vram_threshold(size_bytes)
    };
    should_use_gpu(
        gpu_device.is_cuda(),
        gpu_device.is_metal(),
        free_vram,
        threshold,
    )
}

/// Resolve which UMT5 encoder variant a wan render should use, and where.
///
/// Returns `(encoder_path, on_gpu, device_label)`, matching
/// [`resolve_t5_variant`].
///
/// Wan differs from the T5 families in one way that changes the auto policy:
/// its engine is sequential by construction — the encoder is dropped before
/// the transformer denoises — so the encoder never competes with the DiT for
/// residency. What it does compete with is the *download*, and 11.4 GB of
/// FP16 is the largest single artifact in a wan pull. It is also the floor of
/// the render's memory estimate, since the sequential weight peak is
/// `max(encoder, transformer + vae)`.
///
/// The auto rule therefore prefers the largest GGUF that fits over FP16 when
/// the encoder would otherwise land on CPU. A CPU-parked UMT5 widens to F32
/// (~22.7 GB of host RAM and a slow F32 encode of a 5.7 B-parameter model),
/// which is the case a quantized GPU encode most clearly beats.
pub(crate) fn resolve_umt5_variant(
    progress: &ProgressReporter,
    preference: Option<&str>,
    gpu_device: &Device,
    free_vram: u64,
    default_umt5_path: &Path,
) -> Result<(PathBuf, bool, String)> {
    use mold_core::manifest::{find_umt5_variant, known_umt5_variants, UMT5_FP16_SIZE};

    let is_cuda = gpu_device.is_cuda();
    let is_metal = gpu_device.is_metal();

    match preference {
        Some(tag) if tag != "fp16" && tag != "auto" => {
            let variant = find_umt5_variant(tag).ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown UMT5 variant '{tag}'. Valid: fp16, auto, {}",
                    known_umt5_variants()
                        .iter()
                        .map(|v| v.tag)
                        .collect::<Vec<_>>()
                        .join(", "),
                )
            })?;
            let path = resolve_umt5_gguf_path(progress, variant)?;
            let threshold = if is_metal {
                crate::device::t5_metal_gguf_vram_threshold(variant.size_bytes)
            } else {
                t5_vram_threshold(variant.size_bytes)
            };
            let on_gpu = should_use_gpu(is_cuda, is_metal, free_vram, threshold);
            progress.info(&format!(
                "Using UMT5 {} ({}) on {} (explicit)",
                variant.tag,
                fmt_gb(variant.size_bytes),
                if on_gpu { "GPU" } else { "CPU" },
            ));
            Ok((
                path,
                on_gpu,
                if on_gpu {
                    "GPU, quantized".to_string()
                } else {
                    "CPU, quantized".to_string()
                },
            ))
        }

        Some("fp16") => {
            let on_gpu = should_use_gpu(
                is_cuda,
                is_metal,
                free_vram,
                t5_vram_threshold(UMT5_FP16_SIZE),
            );
            let label = if on_gpu { "GPU" } else { "CPU" };
            progress.info(&format!("Using FP16 UMT5 on {label} (explicit)"));
            Ok((default_umt5_path.to_path_buf(), on_gpu, label.to_string()))
        }

        _ => {
            if fits_in_memory(
                is_cuda,
                is_metal,
                free_vram,
                t5_vram_threshold(UMT5_FP16_SIZE),
            ) {
                progress.info(&format!(
                    "Loading FP16 UMT5 on GPU ({} free)",
                    fmt_gb(free_vram),
                ));
                return Ok((default_umt5_path.to_path_buf(), true, "GPU".to_string()));
            }

            // FP16 does not fit: the largest GGUF that does beats parking the
            // encoder on CPU, where it widens to F32.
            if is_cuda || is_metal {
                for variant in known_umt5_variants() {
                    let threshold = if is_metal {
                        crate::device::t5_metal_gguf_vram_threshold(variant.size_bytes)
                    } else {
                        t5_vram_threshold(variant.size_bytes)
                    };
                    if fits_in_memory(is_cuda, is_metal, free_vram, threshold) {
                        let path = resolve_umt5_gguf_path(progress, variant)?;
                        progress.info(&format!(
                            "FP16 UMT5 ({}) exceeds free VRAM ({}). Using UMT5 {} ({}) on GPU.",
                            fmt_gb(UMT5_FP16_SIZE),
                            fmt_gb(free_vram),
                            variant.tag,
                            fmt_gb(variant.size_bytes),
                        ));
                        return Ok((path, true, format!("GPU, quantized {}", variant.tag)));
                    }
                }
            }

            progress.info(&format!(
                "Loading FP16 UMT5 on CPU ({} free, no variant fits on GPU)",
                fmt_gb(free_vram),
            ));
            Ok((default_umt5_path.to_path_buf(), false, "CPU".to_string()))
        }
    }
}

/// Cache-or-download one UMT5 GGUF, deduped under `shared/wan/umt5-gguf`.
pub(crate) fn resolve_umt5_gguf_path(
    progress: &ProgressReporter,
    variant: &mold_core::manifest::Umt5Variant,
) -> Result<PathBuf> {
    use mold_core::download::{cached_file_path, download_single_file_sync};

    const SUBDIR: &str = "shared/wan/umt5-gguf";
    if let Some(path) = cached_file_path(variant.hf_repo, variant.hf_filename, Some(SUBDIR)) {
        return Ok(path);
    }
    progress.info(&format!(
        "Downloading UMT5 {} ({})...",
        variant.tag,
        fmt_gb(variant.size_bytes),
    ));
    tracing::info!(
        variant = variant.tag,
        repo = variant.hf_repo,
        file = variant.hf_filename,
        "downloading quantized UMT5 encoder"
    );
    download_single_file_sync(variant.hf_repo, variant.hf_filename, Some(SUBDIR))
        .map_err(|error| anyhow::anyhow!("failed to download UMT5 {}: {error}", variant.tag))
}
#[cfg(test)]
mod tests {
    use super::*;

    const ABC_SHA256: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    /// An encoder handed a CPU device is not proof the machine has no GPU:
    /// Qwen Image 2.1's sequential plan places Qwen3-VL on the host on a
    /// 12 GB or 24 GB card, and the line used to say "No GPU detected" on an
    /// L40S (UAT, 24 GB simulation). Only a machine with no usable GPU says so.
    #[test]
    fn a_cpu_placed_encoder_does_not_claim_there_is_no_gpu() {
        assert_eq!(
            cpu_encoder_line("Qwen3-VL-8B", true),
            "Loading BF16 Qwen3-VL-8B on CPU (the encoder is placed on the host)"
        );
        assert_eq!(
            cpu_encoder_line("Qwen3-VL-8B", false),
            "No GPU detected, loading Qwen3-VL-8B on CPU"
        );
    }

    fn pinned(sha256: Option<&'static str>) -> mold_core::manifest::Qwen3Variant {
        mold_core::manifest::Qwen3Variant {
            tag: "q8",
            hf_repo: "Qwen/Qwen3-VL-8B-Instruct-GGUF",
            hf_filename: "fixture.gguf",
            size_bytes: 3,
            sha256,
        }
    }

    /// A pinned variant is held to its digest: matching bytes pass (and are
    /// marked), different bytes are refused and removed, and an unpinned
    /// variant is never hashed.
    #[test]
    fn a_pinned_qwen3_variant_is_verified_before_use() {
        let root = tempfile::tempdir().unwrap();
        let good = root.path().join("good.gguf");
        std::fs::write(&good, b"abc").unwrap();
        verify_qwen3_variant_pin(&good, &pinned(Some(ABC_SHA256))).unwrap();
        assert_eq!(
            mold_core::download::recorded_sha256_marker(&good).as_deref(),
            Some(ABC_SHA256)
        );
        // The marker is now the fast path.
        verify_qwen3_variant_pin(&good, &pinned(Some(ABC_SHA256))).unwrap();

        let bad = root.path().join("bad.gguf");
        std::fs::write(&bad, b"abd").unwrap();
        assert!(verify_qwen3_variant_pin(&bad, &pinned(Some(ABC_SHA256))).is_err());
        assert!(!bad.exists(), "a mismatched pinned file is removed");

        let unpinned = root.path().join("unpinned.gguf");
        std::fs::write(&unpinned, b"anything").unwrap();
        verify_qwen3_variant_pin(&unpinned, &pinned(None)).unwrap();
        assert!(mold_core::download::recorded_sha256_marker(&unpinned).is_none());
    }

    /// Auto mode on CUDA walks BF16 → q8 → q4 → BF16-on-CPU by what the card
    /// has left after the transformer and VAE; Metal never leaves the pool;
    /// an explicit tag wins.
    #[test]
    fn the_vl_variant_choice_follows_the_free_card() {
        const GB: u64 = 1_000_000_000;
        let tag = |choice: Qwen3Choice| match choice {
            Qwen3Choice::Bf16 { on_gpu } => ("bf16", on_gpu),
            Qwen3Choice::Gguf { variant, on_gpu } => (variant.tag, on_gpu),
        };
        let auto = |free| tag(choose_qwen3_vl_variant(None, true, false, free).unwrap());
        assert_eq!(auto(30 * GB), ("bf16", true));
        assert_eq!(auto(12 * GB), ("q8", true));
        // Q4_K_M fails the conditioning gate, so auto mode never picks it.
        assert_eq!(auto(8 * GB), ("bf16", false));
        assert_eq!(auto(5 * GB), ("bf16", false));
        assert_eq!(
            tag(choose_qwen3_vl_variant(Some("auto"), false, true, 5 * GB).unwrap()),
            ("q8", true),
            "Metal takes the smallest auto-eligible GGUF on the unified pool"
        );
        assert_eq!(
            tag(choose_qwen3_vl_variant(Some("q4"), true, false, 40 * GB).unwrap()),
            ("q4", true)
        );
        assert_eq!(
            tag(choose_qwen3_vl_variant(Some("bf16"), true, false, 5 * GB).unwrap()),
            ("bf16", false)
        );
        assert!(choose_qwen3_vl_variant(Some("q6"), true, false, 40 * GB).is_err());
    }

    /// An explicit tag the VL list does not carry names the tags it does.
    #[test]
    fn an_unknown_vl_tag_lists_the_vl_variants() {
        let error = resolve_qwen3_variant(
            &ProgressReporter::default(),
            Some("iq4"),
            &Device::Cpu,
            0,
            &[],
            false,
            false,
            Qwen3Size::Vl8b,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("Qwen3-VL-8B"), "{error}");
        assert!(error.contains("q8, q4"), "{error}");
    }
}
