use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::flux;
use candle_transformers::quantized_var_builder;
use mold_core::{GenerateRequest, GenerateResponse, ImageData, ModelPaths};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::cache::{
    clear_cache, prompt_text_key, restore_cached_tensor_pair, store_cached_tensor_pair,
    CachedTensorPair, LruCache, DEFAULT_PROMPT_CACHE_CAPACITY,
};
use crate::device::{
    check_memory_budget, effective_device_ref, fmt_gb, free_vram_bytes, memory_status_string,
    preflight_memory_check, should_offload, should_use_gpu, usable_free_vram_bytes,
    CLIP_VRAM_THRESHOLD, MIN_OFFLOAD_VRAM,
};
use crate::encoders;
use crate::engine::{rand_seed, InferenceEngine, LoadStrategy, OptionRestoreGuard};
use crate::engine_base::EngineBase;
use crate::image::{build_output_metadata, encode_image};
use crate::progress::{ProgressCallback, ProgressReporter};

use super::transformer::FluxTransformer;

/// Some FLUX safetensors checkpoints store transformer tensors at the root
/// while others nest them under `model.diffusion_model`.
fn flux_transformer_var_builder<'a>(vb: VarBuilder<'a>) -> VarBuilder<'a> {
    if vb.contains_tensor("img_in.weight") {
        vb
    } else if vb.contains_tensor("model.diffusion_model.img_in.weight") {
        vb.pp("model.diffusion_model")
    } else if vb.contains_tensor("diffusion_model.img_in.weight") {
        vb.pp("diffusion_model")
    } else {
        vb
    }
}

/// Some FLUX single-file checkpoints bundle the VAE under a wrapper prefix
/// while the candle FLUX autoencoder expects root `encoder.*` / `decoder.*`
/// keys.
fn flux_vae_var_builder<'a>(vb: VarBuilder<'a>) -> VarBuilder<'a> {
    if vb.contains_tensor("encoder.conv_in.weight") {
        vb
    } else if vb.contains_tensor("first_stage_model.encoder.conv_in.weight") {
        vb.pp("first_stage_model")
    } else if vb.contains_tensor("vae.encoder.conv_in.weight") {
        vb.pp("vae")
    } else {
        vb
    }
}

/// Check if a FLUX safetensors checkpoint stores weights in FP8 (F8_E4M3).
/// Uses candle's DType after loading a single small tensor on CPU (img_in.weight
/// is typically only a few KB).
fn flux_safetensors_transformer_is_fp8(path: &std::path::Path) -> Result<bool> {
    let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&[path])? };
    for key in [
        "img_in.weight",
        "model.diffusion_model.img_in.weight",
        "diffusion_model.img_in.weight",
    ] {
        if let Ok(tensor) = tensors.load(key, &Device::Cpu) {
            return Ok(tensor.dtype() == DType::F8E4M3);
        }
    }
    Ok(false)
}

fn flux_runtime_dtype(is_cuda: bool, is_quantized: bool, transformer_is_fp8: bool) -> DType {
    if is_quantized {
        if is_cuda {
            DType::BF16
        } else {
            DType::F32
        }
    } else if is_cuda && transformer_is_fp8 {
        // FP8 safetensors must go through F16 on CUDA (candle has a kernel naming
        // bug that prevents direct CUDA FP8→BF16 casts). The lazy mmap VarBuilder
        // handles dtype conversion during model construction.
        DType::F16
    } else if is_cuda {
        DType::BF16
    } else {
        DType::F32
    }
}

/// Path for the Q8 GGUF cache of an FP8 safetensors file.
/// Cache key: stem + file size + FNV-1a hash of 4KB sampled from the weight
/// data region (past the JSON header). This avoids collisions between
/// different fine-tunes that share the same tensor layout and header.
fn fp8_gguf_cache_path(path: &Path) -> PathBuf {
    use std::io::{Read, Seek, SeekFrom};
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("transformer");
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    // Sample 4KB from the weight data region (past the safetensors JSON header).
    // The header is typically ~30-60KB; sampling from 25% into the file ensures
    // we're reading actual weight data, not the identical JSON layout.
    let sample_offset = size / 4;
    let content_hash = std::fs::File::open(path)
        .and_then(|mut f| {
            f.seek(SeekFrom::Start(sample_offset))?;
            let mut buf = vec![0u8; 4096];
            let n = f.read(&mut buf)?;
            buf.truncate(n);
            Ok(buf)
        })
        .map(|buf| {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
            for &b in &buf {
                h ^= b as u64;
                h = h.wrapping_mul(0x0100_0000_01b3); // FNV-1a prime
            }
            format!("{h:016x}")
        })
        .unwrap_or_else(|_| "0".to_string());
    let cache_root = mold_core::Config::mold_dir()
        .unwrap_or_else(|| PathBuf::from(".mold"))
        .join("cache")
        .join("flux-q8");
    cache_root.join(format!("{stem}-{size}-{content_hash}.q8_0.gguf"))
}

fn q8_0_can_quantize_dims(dims: &[usize]) -> bool {
    if dims.len() < 2 {
        return false;
    }
    let block_size = candle_core::quantized::GgmlDType::Q8_0.block_size();
    dims.last()
        .is_some_and(|last_dim| *last_dim >= block_size && *last_dim % block_size == 0)
}

fn fp8_cache_should_skip_tensor(name: &str, dims: &[usize]) -> bool {
    dims.is_empty() || name.starts_with("text_encoders.")
}

fn fp8_gguf_tmp_path(cache_path: &Path) -> PathBuf {
    static NEXT_TMP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = NEXT_TMP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    cache_path.with_extension(format!("tmp.{}.{}", std::process::id(), seq))
}

/// Convert an FP8 safetensors checkpoint to Q8_0 GGUF (one-time).
///
/// FP8 safetensors cannot run directly through candle on a 24 GB card because
/// expanding to F16/BF16 doubles the VRAM requirement. Q8_0 GGUF keeps the
/// model at ~12 GB and uses candle's efficient quantized matmul path.
fn ensure_fp8_gguf_cache(path: &Path, progress: &ProgressReporter) -> Result<PathBuf> {
    let cache_path = fp8_gguf_cache_path(path);
    if cache_path.exists() {
        progress.info(&format!("Using cached Q8 GGUF: {}", cache_path.display()));
        return Ok(cache_path);
    }

    let parent = cache_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("invalid cache path: {}", cache_path.display()))?;

    // Clean up orphaned caches from older naming schemes only.
    // v1: {stem}.q8_0.gguf  (no size/hash — exactly "stem.q8_0.gguf")
    // v2: {stem}-{size}.q8_0.gguf  (size only, no content hash — one dash)
    // Current v3: {stem}-{size}-{hash}.q8_0.gguf  (two dashes — NOT cleaned)
    // We only remove v1/v2 formats. Valid v3 caches for other checkpoints
    // (different size/hash) are preserved to avoid expensive re-quantization.
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("transformer");
    std::fs::create_dir_all(parent)?;
    let old_v1 = parent.join(format!("{stem}.q8_0.gguf"));
    if old_v1.exists() {
        tracing::info!(path = %old_v1.display(), "removing v1 orphaned FP8 cache");
        let _ = std::fs::remove_file(&old_v1);
    }
    // v2 format: {stem}-{digits}.q8_0.gguf (one dash, no hash)
    if let Ok(entries) = std::fs::read_dir(parent) {
        let v2_prefix = format!("{stem}-");
        let suffix = ".q8_0.gguf";
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name_str) = name.to_str() else {
                continue;
            };
            if !name_str.starts_with(&v2_prefix) || !name_str.ends_with(suffix) {
                continue;
            }
            // Extract the middle part between prefix and suffix
            let middle = &name_str[v2_prefix.len()..name_str.len() - suffix.len()];
            // v2 has no dash in the middle (just digits for size).
            // v3 has a dash (size-hash). Only remove v2.
            if !middle.contains('-') && middle.chars().all(|c| c.is_ascii_digit()) {
                tracing::info!(path = %entry.path().display(), "removing v2 orphaned FP8 cache");
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }

    progress.info("Converting FP8 checkpoint to Q8 GGUF cache (one-time, may take a few minutes)");
    tracing::info!(
        source = %path.display(),
        cache = %cache_path.display(),
        "converting FP8 safetensors to Q8_0 GGUF cache"
    );

    let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&[path])? };

    // Detect and strip the common prefix used in some checkpoints
    let prefix = if tensors.get("img_in.weight").is_ok() {
        ""
    } else if tensors.get("model.diffusion_model.img_in.weight").is_ok() {
        "model.diffusion_model."
    } else if tensors.get("diffusion_model.img_in.weight").is_ok() {
        "diffusion_model."
    } else {
        ""
    };

    // Enumerate all tensor names via MmapedSafetensors::tensors()
    let all_names: Vec<String> = tensors
        .tensors()
        .into_iter()
        .map(|(name, _)| name)
        .collect();

    let mut qtensors: Vec<(String, candle_core::quantized::QTensor)> = Vec::new();

    let total = all_names.len();
    for (i, name) in all_names.iter().enumerate() {
        if (i + 1) % 50 == 0 || i + 1 == total {
            progress.info(&format!("Quantizing tensor {}/{total}", i + 1));
        }

        let tensor = tensors.load(name, &Device::Cpu)?;
        // Strip prefix for GGUF (quantized model expects unprefixed names)
        let out_name = if !prefix.is_empty() && name.starts_with(prefix) {
            name[prefix.len()..].to_string()
        } else {
            name.clone()
        };

        if fp8_cache_should_skip_tensor(&out_name, tensor.dims()) {
            continue;
        }

        let can_quantize = q8_0_can_quantize_dims(tensor.dims());

        let qt = if can_quantize {
            candle_core::quantized::QTensor::quantize(
                &tensor,
                candle_core::quantized::GgmlDType::Q8_0,
            )?
        } else {
            // Small/odd-shaped tensors (norms, biases): store as F32
            candle_core::quantized::QTensor::quantize(
                &tensor,
                candle_core::quantized::GgmlDType::F32,
            )?
        };
        qtensors.push((out_name, qt));
    }

    // Write GGUF cache (clean up temp file on error)
    let tmp_path = fp8_gguf_tmp_path(&cache_path);
    let write_result = (|| -> Result<()> {
        let file = std::fs::File::create(&tmp_path)?;
        let mut writer = std::io::BufWriter::new(file);
        let tensor_refs: Vec<(&str, &candle_core::quantized::QTensor)> =
            qtensors.iter().map(|(n, q)| (n.as_str(), q)).collect();
        candle_core::quantized::gguf_file::write(&mut writer, &[], &tensor_refs)?;
        Ok(())
    })();
    if let Err(e) = write_result {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }
    if cache_path.exists() {
        let _ = std::fs::remove_file(&tmp_path);
        progress.info(&format!("Using cached Q8 GGUF: {}", cache_path.display()));
        return Ok(cache_path);
    }
    std::fs::rename(&tmp_path, &cache_path)?;

    progress.info(&format!("Q8 GGUF cache created: {}", cache_path.display()));
    tracing::info!(cache = %cache_path.display(), "FP8→Q8_0 GGUF cache created");
    Ok(cache_path)
}

// ── City96-format GGUF embedding patching ──────────────────────────────────

/// Embedding tensors required by all FLUX models (schnell and dev).
const FLUX_EMBEDDING_TENSORS: &[&str] = &[
    "img_in.weight",
    "img_in.bias",
    "time_in.in_layer.weight",
    "time_in.in_layer.bias",
    "time_in.out_layer.weight",
    "time_in.out_layer.bias",
    "vector_in.in_layer.weight",
    "vector_in.in_layer.bias",
    "vector_in.out_layer.weight",
    "vector_in.out_layer.bias",
];

/// Additional embedding tensors for FLUX-dev (guidance-based) models.
const FLUX_GUIDANCE_EMBEDDING_TENSORS: &[&str] = &[
    "guidance_in.in_layer.weight",
    "guidance_in.in_layer.bias",
    "guidance_in.out_layer.weight",
    "guidance_in.out_layer.bias",
];

/// Lightweight check: does a GGUF file contain the FLUX embedding layers?
/// Reads only the GGUF header (tensor_infos), not the tensor data.
///
/// Relies on the city96-format property that embedding tensors are either
/// all present or all absent. A GGUF with `img_in.weight` but missing other
/// embeddings would pass this check.
fn gguf_has_embeddings(path: &Path) -> Result<bool> {
    let mut file = std::fs::File::open(path)?;
    let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    Ok(content.tensor_infos.contains_key("img_in.weight"))
}

/// Does a GGUF contain the flux-dev-only `guidance_in` tensors? Schnell GGUFs
/// return false because the schnell architecture is distilled without guidance.
fn gguf_has_guidance(path: &Path) -> Result<bool> {
    let mut file = std::fs::File::open(path)?;
    let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    Ok(content
        .tensor_infos
        .contains_key("guidance_in.in_layer.weight"))
}

/// Search for a downloaded FLUX GGUF that contains complete embeddings.
///
/// Prefers larger quantizations (more likely downloaded) first. When
/// `needs_guidance` is true, schnell candidates are skipped and dev candidates
/// are verified to contain `guidance_in` tensors — a schnell GGUF passes the
/// basic `img_in` check but cannot supply `guidance_in` for a dev-family target.
///
/// When `models_dir_override` is `Some`, searches that directory instead of
/// the config-resolved models dir (used by tests to avoid global state).
fn find_flux_reference_gguf(
    needs_guidance: bool,
    models_dir_override: Option<&Path>,
) -> Option<PathBuf> {
    let config = mold_core::Config::load_or_default();
    let models_dir = models_dir_override
        .map(PathBuf::from)
        .unwrap_or_else(|| config.resolved_models_dir());

    // Dev candidates satisfy both schnell and dev targets (schnell tensors are a
    // subset of dev). Schnell candidates only satisfy schnell targets.
    // flux-krea is a dev-family fine-tune shipped as complete GGUFs by
    // QuantStack, so it carries the full embedding set including guidance_in —
    // fall back to it before asking the user to download flux-dev.
    let mut candidates: Vec<&str> = vec![
        "flux-dev:q8",
        "flux-dev:q6",
        "flux-dev:q4",
        "flux-krea:q8",
        "flux-krea:q6",
        "flux-krea:q4",
    ];
    if !needs_guidance {
        candidates.extend(["flux-schnell:q8", "flux-schnell:q4"]);
    }

    for name in candidates {
        let Some(manifest) = mold_core::manifest::find_manifest(name) else {
            continue;
        };
        // Find the transformer file in the manifest
        let Some(xformer_file) = manifest
            .files
            .iter()
            .find(|f| f.component == mold_core::manifest::ModelComponent::Transformer)
        else {
            continue;
        };
        let xformer_path =
            models_dir.join(mold_core::manifest::storage_path(manifest, xformer_file));
        if !xformer_path.exists() {
            continue;
        }
        // Verify it actually has the embeddings (don't assume)
        match gguf_has_embeddings(&xformer_path) {
            Ok(true) => {
                if needs_guidance {
                    match gguf_has_guidance(&xformer_path) {
                        Ok(true) => {}
                        Ok(false) => {
                            tracing::debug!(
                                model = name,
                                "reference candidate lacks guidance_in, skipping for dev target"
                            );
                            continue;
                        }
                        Err(e) => {
                            tracing::debug!(
                                model = name,
                                err = %e,
                                "failed to probe guidance tensors"
                            );
                            continue;
                        }
                    }
                }
                tracing::info!(
                    reference = %xformer_path.display(),
                    model = name,
                    needs_guidance,
                    "found reference FLUX GGUF with embeddings"
                );
                return Some(xformer_path);
            }
            Ok(false) => {
                tracing::debug!(
                    model = name,
                    "reference candidate also missing embeddings, skipping"
                );
            }
            Err(e) => {
                tracing::debug!(model = name, err = %e, "failed to probe reference candidate");
            }
        }
    }
    None
}

/// Cache path for a GGUF patched with missing embedding layers.
/// Same FNV-1a content hashing scheme as `fp8_gguf_cache_path`.
fn embedding_patched_cache_path(path: &Path) -> PathBuf {
    use std::io::{Read, Seek, SeekFrom};
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("transformer");
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let sample_offset = size / 4;
    let content_hash = std::fs::File::open(path)
        .and_then(|mut f| {
            f.seek(SeekFrom::Start(sample_offset))?;
            let mut buf = vec![0u8; 4096];
            let n = f.read(&mut buf)?;
            buf.truncate(n);
            Ok(buf)
        })
        .map(|buf| {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for &b in &buf {
                h ^= b as u64;
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
            format!("{h:016x}")
        })
        .unwrap_or_else(|_| "0".to_string());
    let cache_root = mold_core::Config::mold_dir()
        .unwrap_or_else(|| PathBuf::from(".mold"))
        .join("cache")
        .join("flux-embeddings");
    cache_root.join(format!("{stem}-{size}-{content_hash}.patched.gguf"))
}

/// Ensure a GGUF file has complete FLUX embedding layers.
///
/// City96-format GGUFs (used by community fine-tune quantizations like
/// UltraReal) only include the diffusion blocks but omit input embedding
/// layers (`img_in`, `time_in`, `vector_in`, `guidance_in`). This function
/// detects incomplete GGUFs and patches them by sourcing the missing
/// embeddings from a reference FLUX GGUF (e.g. flux-dev:q8).
///
/// Returns the original path if the GGUF is already complete, or the path
/// to a patched cache file.
///
/// `models_dir_override` is forwarded to `find_flux_reference_gguf` and
/// only used by tests to avoid mutating process-global environment variables.
fn ensure_gguf_embeddings(
    path: &Path,
    is_schnell: bool,
    progress: &ProgressReporter,
    models_dir_override: Option<&Path>,
) -> Result<PathBuf> {
    let cache_path = embedding_patched_cache_path(path);
    if cache_path.exists() {
        progress.info(&format!(
            "Using cached embedding-patched GGUF: {}",
            cache_path.display()
        ));
        return Ok(cache_path);
    }

    // Probe whether embeddings are actually missing
    if gguf_has_embeddings(path)? {
        return Ok(path.to_path_buf());
    }

    progress.info(
        "GGUF is missing FLUX embedding layers (city96 format) — patching from reference model",
    );
    tracing::info!(
        path = %path.display(),
        is_schnell,
        "GGUF missing embedding layers, searching for reference model"
    );

    let source_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("<unknown>");
    let needs_guidance = !is_schnell;
    let reference_path =
        find_flux_reference_gguf(needs_guidance, models_dir_override).ok_or_else(|| {
            let family = if needs_guidance { "dev" } else { "schnell" };
            anyhow::anyhow!(
                "{source_name} is a city96-format GGUF that ships only the diffusion \
                 blocks — its FLUX input embedding layers (img_in, time_in, vector_in{guidance}) \
                 must be sourced from a complete flux-{family} GGUF, but none is downloaded.\n\n\
                 To fix this:\n\n  mold pull flux-dev:q8\n\n\
                 Then retry — mold will patch the incomplete GGUF from the reference.",
                guidance = if needs_guidance { ", guidance_in" } else { "" },
            )
        })?;

    // Determine which embedding tensors we need
    let mut needed: Vec<&str> = FLUX_EMBEDDING_TENSORS.to_vec();
    if !is_schnell {
        needed.extend_from_slice(FLUX_GUIDANCE_EMBEDDING_TENSORS);
    }

    // Read source (incomplete) GGUF
    progress.info("Reading source GGUF tensors...");
    let mut src_file = std::fs::File::open(path)?;
    let src_content = candle_core::quantized::gguf_file::Content::read(&mut src_file)?;

    // Read only the needed embedding tensors from the reference GGUF
    progress.info(&format!(
        "Extracting {} embedding tensors from reference: {}",
        needed.len(),
        reference_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("?")
    ));
    let mut ref_file = std::fs::File::open(&reference_path)?;
    let ref_content = candle_core::quantized::gguf_file::Content::read(&mut ref_file)?;

    let cpu = Device::Cpu;

    // Load all source tensors
    let mut qtensors: Vec<(String, candle_core::quantized::QTensor)> = Vec::new();
    let total = src_content.tensor_infos.len();
    for (i, name) in src_content.tensor_infos.keys().enumerate() {
        if (i + 1) % 100 == 0 || i + 1 == total {
            progress.info(&format!("Loading source tensor {}/{total}", i + 1));
        }
        let tensor = src_content.tensor(&mut src_file, name, &cpu)?;
        qtensors.push((name.clone(), tensor));
    }

    // Load missing embedding tensors from reference
    let mut patched_count = 0usize;
    for name in &needed {
        if src_content.tensor_infos.contains_key(*name) {
            continue; // already present in source
        }
        if !ref_content.tensor_infos.contains_key(*name) {
            bail!(
                "while patching {source_name}: the only downloaded reference ({}) \
                 is also missing '{name}'. This model needs a complete flux-dev GGUF \
                 — run 'mold pull flux-dev:q8' and retry.",
                reference_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("<unknown>"),
            );
        }
        let tensor = ref_content.tensor(&mut ref_file, name, &cpu)?;
        tracing::debug!(tensor = name, "patching embedding tensor from reference");
        qtensors.push((name.to_string(), tensor));
        patched_count += 1;
    }

    progress.info(&format!(
        "Patched {patched_count} embedding tensors from reference"
    ));

    // Write patched GGUF
    let parent = cache_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("invalid cache path: {}", cache_path.display()))?;
    std::fs::create_dir_all(parent)?;
    let tmp_path = cache_path.with_extension(format!("tmp.{}", std::process::id()));
    let write_result = (|| -> Result<()> {
        let file = std::fs::File::create(&tmp_path)?;
        let mut writer = std::io::BufWriter::new(file);
        let tensor_refs: Vec<(&str, &candle_core::quantized::QTensor)> =
            qtensors.iter().map(|(n, q)| (n.as_str(), q)).collect();
        candle_core::quantized::gguf_file::write(&mut writer, &[], &tensor_refs)?;
        Ok(())
    })();
    if let Err(e) = write_result {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }
    std::fs::rename(&tmp_path, &cache_path)?;

    progress.info(&format!(
        "Embedding-patched GGUF cache created: {}",
        cache_path.display()
    ));
    tracing::info!(
        cache = %cache_path.display(),
        patched_count,
        "embedding-patched GGUF cache created"
    );
    Ok(cache_path)
}

fn flux_safetensors_var_builder<'a>(
    path: &std::path::Path,
    dtype: DType,
    device: &Device,
    component: &str,
    progress: &ProgressReporter,
) -> Result<VarBuilder<'a>> {
    let aliases = flux_rms_norm_scale_aliases(path)?;
    if aliases.is_empty() {
        crate::weight_loader::load_safetensors_with_progress(
            std::slice::from_ref(&path),
            dtype,
            device,
            component,
            progress,
        )
    } else {
        tracing::info!(
            alias_count = aliases.len(),
            path = %path.display(),
            "FLUX checkpoint uses RMSNorm .weight keys; aliasing .scale lookups"
        );
        crate::weight_loader::load_safetensors_with_aliases(
            std::slice::from_ref(&path),
            dtype,
            device,
            component,
            progress,
            aliases,
        )
    }
}

fn flux_rms_norm_scale_aliases(path: &std::path::Path) -> Result<BTreeMap<String, String>> {
    let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&[path])? };
    let mut aliases = BTreeMap::new();
    for prefix in ["", "model.diffusion_model.", "diffusion_model."] {
        for i in 0..64 {
            for stream in ["img_attn", "txt_attn"] {
                for norm in ["query_norm", "key_norm"] {
                    let target = format!("{prefix}double_blocks.{i}.{stream}.norm.{norm}.scale");
                    let source = format!("{prefix}double_blocks.{i}.{stream}.norm.{norm}.weight");
                    if tensors.get(&target).is_err() && tensors.get(&source).is_ok() {
                        aliases.insert(target, source);
                    }
                }
            }
        }
        for i in 0..128 {
            for norm in ["query_norm", "key_norm"] {
                let target = format!("{prefix}single_blocks.{i}.norm.{norm}.scale");
                let source = format!("{prefix}single_blocks.{i}.norm.{norm}.weight");
                if tensors.get(&target).is_err() && tensors.get(&source).is_ok() {
                    aliases.insert(target, source);
                }
            }
        }
    }
    Ok(aliases)
}

/// Build a LoRA-patching VarBuilder that wraps mmap'd base weights.
///
/// Uses a custom `SimpleBackend` that intercepts every `vb.get()` call during
/// model construction.  Each tensor loads from mmap directly to GPU with LoRA
/// deltas applied inline — identical memory profile to the non-LoRA mmap path.
///
/// Multi-LoRA: pass a slice with more than one weight and the deltas merge
/// additively. Each adapter's contribution is independently cached (per
/// path + scale) so a stack of (cinematic, dramatic-light) reuses both
/// matmuls when the user toggles either back on later.
fn flux_lora_var_builder<'a>(
    transformer_path: &Path,
    loras: &[mold_core::LoraWeight],
    dtype: DType,
    device: &Device,
    progress: &ProgressReporter,
    delta_cache: Option<std::sync::Arc<std::sync::Mutex<super::lora::LoraDeltaCache>>>,
) -> Result<VarBuilder<'a>> {
    use super::lora;

    let adapters: Vec<std::sync::Arc<lora::LoraAdapter>> = loras
        .iter()
        .map(|w| {
            progress.info("Loading LoRA adapter");
            let adapter = lora::get_or_load_adapter(Path::new(&w.path))?;
            progress.info(&format!(
                "LoRA: {} layers, rank {}, scale {:.2}",
                adapter.layers.len(),
                adapter.rank,
                w.scale,
            ));
            anyhow::Ok(adapter)
        })
        .collect::<Result<_>>()?;

    let specs: Vec<lora::LoraSpec<'_>> = adapters
        .iter()
        .zip(loras.iter())
        .map(|(adapter, w)| lora::LoraSpec {
            adapter: adapter.as_ref(),
            scale: w.scale,
            path_hash: lora_path_hash(&w.path),
        })
        .collect();

    lora::lora_var_builder(
        transformer_path,
        &specs,
        dtype,
        device,
        progress,
        delta_cache,
    )
}

/// Stable hash for a LoRA file path. Used as the per-LoRA cache-key
/// component so the delta cache survives transformer rebuilds and
/// disambiguates adapters in a multi-LoRA stack.
fn lora_path_hash(path: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut hasher);
    hasher.finish()
}

/// Same wrapper for the GGUF (quantized) path.
fn flux_gguf_lora_var_builder(
    transformer_path: &Path,
    loras: &[mold_core::LoraWeight],
    device: &Device,
    progress: &ProgressReporter,
    delta_cache: Option<std::sync::Arc<std::sync::Mutex<super::lora::LoraDeltaCache>>>,
) -> Result<candle_transformers::quantized_var_builder::VarBuilder> {
    use super::lora;

    let adapters: Vec<std::sync::Arc<lora::LoraAdapter>> = loras
        .iter()
        .map(|w| {
            progress.info("Loading LoRA adapter");
            let adapter = lora::get_or_load_adapter(Path::new(&w.path))?;
            progress.info(&format!(
                "LoRA: {} layers, rank {}, scale {:.2}",
                adapter.layers.len(),
                adapter.rank,
                w.scale,
            ));
            anyhow::Ok(adapter)
        })
        .collect::<Result<_>>()?;

    let specs: Vec<lora::LoraSpec<'_>> = adapters
        .iter()
        .zip(loras.iter())
        .map(|(adapter, w)| lora::LoraSpec {
            adapter: adapter.as_ref(),
            scale: w.scale,
            path_hash: lora_path_hash(&w.path),
        })
        .collect();

    lora::gguf_lora_var_builder(transformer_path, &specs, device, progress, delta_cache)
}

/// Three-state opt-in for bypass-mode LoRA. `auto` enables bypass on
/// every supported path: the offload transformer (avoids the ~24 GB
/// CPU-resident BF16 merge) and the GGUF transformer (avoids the
/// minutes-long, ~95 GB peak dequant→merge→requant cycle on Q8 with
/// a stack of LoRAs). `on` forces bypass; `off` reverts to the
/// legacy `flux_lora_var_builder` / `gguf_lora_var_builder` so users
/// can regression-check a build.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LoraBypassMode {
    Auto,
    On,
    Off,
}

impl LoraBypassMode {
    fn from_env() -> Self {
        match crate::runtime_env::value("MOLD_LORA_BYPASS")
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("on") | Some("1") | Some("true") => Self::On,
            Some("off") | Some("0") | Some("false") => Self::Off,
            _ => Self::Auto,
        }
    }
}

fn should_use_offload_bypass_registry(
    use_offload: bool,
    has_lora: bool,
    bypass_mode: LoraBypassMode,
) -> bool {
    use_offload && has_lora && bypass_mode != LoraBypassMode::Off
}

/// Build a [`super::lora_bypass::LoraRegistry`] for any bypass-capable
/// path (offload or GGUF/quantized).
///
/// Adapters are placed on `device` at `dtype` (typically GPU + BF16) so
/// the per-step path never round-trips them CPU↔GPU. Both paths use the
/// same registry shape: keys are FLUX candle tensor names, values are
/// the bypass adapters that fire each time that Linear runs forward.
///
/// Returns `Ok(None)` when `loras` is empty so callers keep their
/// no-LoRA hot path.
fn build_lora_registry(
    loras: &[mold_core::LoraWeight],
    cfg: &flux::model::Config,
    device: &Device,
    dtype: DType,
    progress: &ProgressReporter,
) -> Result<Option<super::lora_bypass::LoraRegistry>> {
    use super::lora;
    use super::lora_bypass;

    if loras.is_empty() {
        return Ok(None);
    }

    let adapters: Vec<lora::LoraAdapter> = loras
        .iter()
        .map(|w| {
            progress.info("Loading LoRA adapter (bypass)");
            let adapter = lora::LoraAdapter::load(Path::new(&w.path))?;
            progress.info(&format!(
                "LoRA: {} layers, rank {}, scale {:.2}",
                adapter.layers.len(),
                adapter.rank,
                w.scale,
            ));
            anyhow::Ok(adapter)
        })
        .collect::<Result<_>>()?;

    let specs: Vec<lora::LoraSpec<'_>> = adapters
        .iter()
        .zip(loras.iter())
        .map(|(adapter, w)| lora::LoraSpec {
            adapter,
            scale: w.scale,
            path_hash: lora_path_hash(&w.path),
        })
        .collect();

    // Pre-compute the fused linear out-row counts that bypass-mode
    // needs to translate component-index targets (e.g. "Q only") into
    // absolute slice offsets.
    let h = cfg.hidden_size;
    let mlp_sz = (h as f64 * cfg.mlp_ratio) as usize;
    let mut linear_out_dims: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for idx in 0..cfg.depth {
        // Double blocks: img_attn.qkv / txt_attn.qkv each 3*h.
        linear_out_dims.insert(format!("double_blocks.{idx}.img_attn.qkv.weight"), 3 * h);
        linear_out_dims.insert(format!("double_blocks.{idx}.txt_attn.qkv.weight"), 3 * h);
    }
    for idx in 0..cfg.depth_single_blocks {
        // Single block linear1 fuses [Q, K, V, MLP] = 3*h + mlp_sz.
        linear_out_dims.insert(
            format!("single_blocks.{idx}.linear1.weight"),
            3 * h + mlp_sz,
        );
    }

    let registry = lora_bypass::build_registry(&specs, &linear_out_dims, device, dtype)?;
    progress.info(&format!(
        "LoRA bypass: {} target tensors, adapters resident on {device:?}",
        registry.len()
    ));
    Ok(Some(registry))
}

/// Resolve the effective LoRA list for a request.
///
/// Wire format intentionally accepts both `lora` (single) and `loras`
/// (plural) for back-compat with older clients. When both are set,
/// `loras` wins — single-form callers haven't been updated yet but
/// new clients always populate the plural shape.
///
/// Entries whose `scale.abs() < ZERO_SCALE_EPS` are dropped: a slider
/// pinned to zero is a no-op patch and forcing the transformer to
/// rebuild for it is pure overhead. A `tracing::debug!` records each
/// drop so a user wondering "why didn't my LoRA apply" can spot it
/// in `RUST_LOG=debug` output.
pub(crate) fn effective_loras(req: &mold_core::GenerateRequest) -> Vec<mold_core::LoraWeight> {
    /// Threshold below which a LoRA scale is treated as off. Matches
    /// the precision of an f64 scrubbed by a UI slider — anything
    /// closer to zero than this is the user nudging the slider, not
    /// a deliberate negative weight.
    const ZERO_SCALE_EPS: f64 = 1e-8;

    let raw: Vec<mold_core::LoraWeight> = if let Some(plural) = &req.loras {
        if !plural.is_empty() {
            plural.clone()
        } else {
            req.lora.iter().cloned().collect()
        }
    } else {
        req.lora.iter().cloned().collect()
    };

    raw.into_iter()
        .filter(|w| {
            let keep = w.scale.abs() > ZERO_SCALE_EPS;
            if !keep {
                tracing::debug!(
                    path = w.path.as_str(),
                    scale = w.scale,
                    "dropping zero-scale LoRA from effective stack"
                );
            }
            keep
        })
        .collect()
}

/// Loaded FLUX model components, ready for inference.
/// FLUX transformer and VAE always run on GPU. T5 and CLIP run on GPU or CPU
/// depending on available VRAM (checked at load time after the transformer is loaded).
/// When T5/CLIP are loaded on GPU, they are dropped after encoding to free VRAM
/// for the denoising pass (their weights are only needed for prompt encoding).
struct LoadedFlux {
    /// None after being dropped for VAE decode VRAM; reloaded on next generate.
    flux_model: Option<FluxTransformer>,
    t5: encoders::t5::T5Encoder,
    clip: encoders::clip::ClipEncoder,
    vae: flux::autoencoder::AutoEncoder,
    /// GPU device for FLUX transformer + VAE
    device: Device,
    dtype: DType,
    /// Effective VAE dtype after `MOLD_VAE_DTYPE` resolution. Stored so the
    /// post-denoise cast and the decode forward pass agree on precision —
    /// the eager path loads the VAE once at startup so this is captured at
    /// load time and persists for the engine's lifetime. Sequential reloads
    /// re-resolve per request.
    vae_dtype: DType,
    is_schnell: bool,
    /// True if using quantized GGUF model (state tensors must be F32)
    is_quantized: bool,
    /// Resolved transformer path (may be a GGUF cache for FP8 models).
    transformer_path: PathBuf,
    /// The actual T5 encoder path used (may be a quantized GGUF, not the original FP16 path).
    t5_encoder_path: std::path::PathBuf,
}

/// Fingerprint of a single LoRA adapter (path + scale). Used to detect
/// when the active LoRA stack has changed so we know to rebuild the
/// transformer; an unchanged stack reuses the previously merged weights.
#[derive(Clone, PartialEq, Eq)]
struct LoraFingerprint {
    path_hash: u64,
    scale_bits: u64,
}

impl LoraFingerprint {
    fn from_lora_weight(lora: &mold_core::LoraWeight) -> Self {
        Self {
            path_hash: lora_path_hash(&lora.path),
            scale_bits: lora.scale.to_bits(),
        }
    }
}

/// Fingerprint of an ordered LoRA stack. Equality is order-sensitive —
/// `[A, B]` and `[B, A]` produce identical numerical results in theory
/// (delta sums commute) but the wrapper still considers them distinct
/// because the user-facing intent is order-driven (e.g. style vs
/// override) and the cost of one redundant rebuild is small.
fn fingerprint_stack(loras: &[mold_core::LoraWeight]) -> Vec<LoraFingerprint> {
    loras
        .iter()
        .map(LoraFingerprint::from_lora_weight)
        .collect()
}

/// FLUX inference engine backed by candle.
pub struct FluxEngine {
    base: EngineBase<LoadedFlux>,
    /// Optional explicit override for is_schnell; if None, auto-detect from transformer filename.
    is_schnell_override: Option<bool>,
    /// T5 variant preference: None/"auto" = auto-select, "fp16" = force FP16, "q8"/"q5"/etc = specific quantized.
    t5_variant: Option<String>,
    prompt_cache: Mutex<LruCache<String, CachedTensorPair>>,
    /// Cached result of FP8 safetensors probe (None = not yet checked).
    transformer_is_fp8: Option<bool>,
    /// Cached resolved transformer path (GGUF cache for FP8, or original path).
    /// Avoids re-computing the cache key (file I/O) on every sequential generation.
    cached_transformer_path: Option<PathBuf>,
    /// Force block-level offloading (--offload / MOLD_OFFLOAD=1).
    offload: bool,
    /// Fingerprint of the currently applied LoRA (None = no LoRA baked in).
    /// Empty when no LoRAs are active. Order-sensitive: changing the
    /// stack triggers a transformer rebuild on the next generate.
    active_lora: Vec<LoraFingerprint>,
    /// CPU-resident cache of pre-computed LoRA deltas, shared across transformer rebuilds.
    lora_delta_cache: Arc<Mutex<super::lora::LoraDeltaCache>>,
    /// Optional shared tokenizer pool for cross-engine caching.
    shared_pool: Option<Arc<Mutex<crate::shared_pool::SharedPool>>>,
    /// Per-request placement override. Set at the start of `generate()`,
    /// cleared on exit. `None` preserves the existing VRAM-aware auto logic.
    pending_placement: Option<mold_core::types::DevicePlacement>,
}

impl FluxEngine {
    /// Create a new FluxEngine. Does not load models until `load()` is called.
    /// `is_schnell_override` lets callers explicitly set the scheduler family.
    /// `t5_variant` controls T5 encoder selection: None/"auto" = VRAM-based auto-select,
    /// "fp16" = force FP16, "q8"/"q5"/etc = specific quantized variant.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_name: String,
        paths: ModelPaths,
        is_schnell_override: Option<bool>,
        t5_variant: Option<String>,
        load_strategy: LoadStrategy,
        gpu_ordinal: usize,
        offload: bool,
        shared_pool: Option<Arc<Mutex<crate::shared_pool::SharedPool>>>,
    ) -> Self {
        Self {
            base: EngineBase::new(model_name, paths, load_strategy, gpu_ordinal),
            is_schnell_override,
            t5_variant,
            prompt_cache: Mutex::new(LruCache::new(DEFAULT_PROMPT_CACHE_CAPACITY)),
            transformer_is_fp8: None,
            cached_transformer_path: None,
            offload,
            active_lora: Vec::new(),
            lora_delta_cache: Arc::new(Mutex::new(super::lora::LoraDeltaCache::new())),
            shared_pool,
            pending_placement: None,
        }
    }

    /// Return the LoRA delta cache handle, or `None` when disabled via
    /// `MOLD_FLUX_DELTA_CACHE=0`. The cache stores CPU-resident F32 delta
    /// tensors for every LoRA-touched layer; on a typical FLUX LoRA that's
    /// ~25 GB of standing memory which dominates host RAM use during Q8+LoRA
    /// rebuilds. Disabling forces a sub-second `B@A·scale` recompute on the
    /// next rebuild, which is cheap on GPU.
    fn lora_delta_cache_handle(&self) -> Option<Arc<Mutex<super::lora::LoraDeltaCache>>> {
        if crate::runtime_env::value("MOLD_FLUX_DELTA_CACHE")
            .map(|v| v == "0")
            .unwrap_or(false)
        {
            None
        } else {
            Some(self.lora_delta_cache.clone())
        }
    }

    /// Try to get a cached tokenizer from the shared pool.
    fn get_cached_tokenizer(&self, path: &std::path::Path) -> Option<Arc<tokenizers::Tokenizer>> {
        let pool = self.shared_pool.as_ref()?;
        let pool = pool.lock().unwrap();
        pool.get_tokenizer(&path.to_string_lossy())
    }

    /// Store a tokenizer in the shared pool.
    fn cache_tokenizer(&self, path: &std::path::Path, tokenizer: Arc<tokenizers::Tokenizer>) {
        if let Some(ref pool) = self.shared_pool {
            let mut pool = pool.lock().unwrap();
            pool.insert_tokenizer(path.to_string_lossy().into_owned(), tokenizer);
        }
    }

    /// Load VAE weights through the shared CPU tensor cache when available.
    fn load_vae_var_builder<'a>(
        &self,
        dtype: DType,
        device: &Device,
        component: &str,
    ) -> Result<VarBuilder<'a>> {
        if let Some(pool) = &self.shared_pool {
            let cached = pool
                .lock()
                .unwrap()
                .load_cpu_tensors(std::slice::from_ref(&self.base.paths.vae))?;
            let vb = crate::encoders::park::varbuilder_from_parked(cached.as_ref(), dtype, device);
            return Ok(flux_vae_var_builder(vb));
        }

        let vb = crate::weight_loader::load_safetensors_with_progress(
            std::slice::from_ref(&self.base.paths.vae),
            dtype,
            device,
            component,
            &self.base.progress,
        )?;
        Ok(flux_vae_var_builder(vb))
    }

    fn get_cached_safetensors(&self, path: &Path) -> Result<Option<Arc<HashMap<String, Tensor>>>> {
        let Some(pool) = &self.shared_pool else {
            return Ok(None);
        };
        let paths = [path];
        pool.lock().unwrap().load_safetensors_cpu_tensors(&paths)
    }

    fn restore_prompt_cache(
        progress: &ProgressReporter,
        prompt_cache: &Mutex<LruCache<String, CachedTensorPair>>,
        prompt: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<Option<(candle_core::Tensor, candle_core::Tensor)>> {
        let restored =
            restore_cached_tensor_pair(prompt_cache, &prompt_text_key(prompt), device, dtype)?;
        let Some(restored) = restored else {
            return Ok(None);
        };
        progress.cache_hit("prompt conditioning");
        Ok(Some(restored))
    }

    fn store_prompt_cache(
        prompt_cache: &Mutex<LruCache<String, CachedTensorPair>>,
        prompt: &str,
        t5_emb: &candle_core::Tensor,
        clip_emb: &candle_core::Tensor,
    ) -> Result<()> {
        store_cached_tensor_pair(prompt_cache, prompt_text_key(prompt), t5_emb, clip_emb)
    }
}

/// Move a conditioning tensor to host RAM if it currently lives on GPU.
///
/// ComfyUI keeps text-encoder outputs on CPU between encode and denoise so the
/// transformer load and LoRA merge see ~50–200 MB more headroom. mirroring
/// that here: after `t5.encode(...)` / `clip.encode(...)` we call this, then
/// move the tensor back to GPU only at `State::new` time inside the denoise
/// loop. Idempotent — when the encoder already produced a CPU tensor (the
/// GGUF / Q8 dequant path) this is a cheap pass-through with no copy.
pub(crate) fn park_cond_to_cpu(tensor: &candle_core::Tensor) -> Result<candle_core::Tensor> {
    if tensor.device().is_cpu() {
        return Ok(tensor.clone());
    }
    Ok(tensor.to_device(&Device::Cpu)?)
}

impl FluxEngine {
    /// Detect is_schnell from override, model name, or transformer filename.
    fn detect_is_schnell(&self) -> bool {
        self.is_schnell_override.unwrap_or_else(|| {
            self.base.model_name.contains("schnell")
                || self
                    .base
                    .paths
                    .transformer
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.to_ascii_lowercase().contains("schnell"))
                    .unwrap_or(false)
        })
    }

    /// Detect if the transformer is quantized (GGUF).
    /// Check if the transformer is FP8 safetensors, caching the result so the
    /// file is only probed once (not on every `generate_sequential` call).
    fn check_transformer_is_fp8(&mut self, is_quantized: bool) -> bool {
        if let Some(cached) = self.transformer_is_fp8 {
            return cached;
        }
        let result = !is_quantized
            && flux_safetensors_transformer_is_fp8(&self.base.paths.transformer).unwrap_or(false);
        self.transformer_is_fp8 = Some(result);
        result
    }

    fn detect_is_quantized(&self) -> bool {
        self.base
            .paths
            .transformer
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
    }

    /// Validate that all required paths exist.
    fn validate_paths(
        &self,
    ) -> Result<(
        std::path::PathBuf,
        std::path::PathBuf,
        std::path::PathBuf,
        std::path::PathBuf,
    )> {
        let t5_encoder_path = self
            .base
            .paths
            .t5_encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("T5 encoder path required for FLUX models"))?
            .clone();
        let t5_tokenizer_path = self
            .base
            .paths
            .t5_tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("T5 tokenizer path required for FLUX models"))?
            .clone();
        let clip_encoder_path = self
            .base
            .paths
            .clip_encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP encoder path required for FLUX models"))?
            .clone();
        let clip_tokenizer_path = self
            .base
            .paths
            .clip_tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP tokenizer path required for FLUX models"))?
            .clone();

        for (label, path) in [
            ("transformer", &self.base.paths.transformer),
            ("vae", &self.base.paths.vae),
            ("t5_encoder", &t5_encoder_path),
            ("clip_encoder", &clip_encoder_path),
            ("t5_tokenizer", &t5_tokenizer_path),
            ("clip_tokenizer", &clip_tokenizer_path),
        ] {
            if !path.exists() {
                bail!("{label} file not found: {}", path.display());
            }
        }

        Ok((
            t5_encoder_path,
            t5_tokenizer_path,
            clip_encoder_path,
            clip_tokenizer_path,
        ))
    }

    /// Load all model components into GPU memory (Eager mode).
    ///
    /// On error, `self.base.loaded` remains `None` — all components are assembled into
    /// local variables and only stored in `self.base.loaded` on success, so partial loads
    /// cannot leave the engine in an inconsistent state.
    pub fn load(&mut self) -> Result<()> {
        self.active_lora = Vec::new();
        if self.base.loaded.is_some() {
            return Ok(());
        }

        // Sequential/offloaded mode defers loading to generate_sequential().
        // The offloaded BF16 transformer is built per request after prompt
        // encoding; eager preload would put the full transformer on GPU and
        // bypass block streaming.
        if self.defers_eager_load() {
            return Ok(());
        }

        let is_schnell = self.detect_is_schnell();
        tracing::info!(model = %self.base.model_name, "loading FLUX model components...");

        let (t5_encoder_path, t5_tokenizer_path, clip_encoder_path, clip_tokenizer_path) =
            self.validate_paths()?;

        let cpu = Device::Cpu;
        let transformer_ref = effective_device_ref(
            self.pending_placement.as_ref(),
            |adv| Some(adv.transformer.clone()),
            false,
        );
        let device = crate::device::resolve_device(Some(transformer_ref), || {
            crate::device::create_device(self.base.gpu_ordinal, &self.base.progress)
        })?;
        let mut is_quantized = self.detect_is_quantized();
        let transformer_is_fp8 = self.check_transformer_is_fp8(is_quantized);

        // FP8 safetensors → Q8 GGUF cache: candle lacks native FP8 compute and
        // expanding to F16 doubles VRAM (OOM on 24 GB). Q8 GGUF keeps the model
        // compact (~12 GB) and uses candle's efficient quantized matmul.
        let transformer_path = if transformer_is_fp8 {
            let p = ensure_fp8_gguf_cache(&self.base.paths.transformer, &self.base.progress)?;
            is_quantized = true;
            p
        } else {
            self.base.paths.transformer.clone()
        };

        // Patch city96-format GGUFs missing embedding layers (img_in, time_in, etc.)
        let transformer_path = if is_quantized {
            ensure_gguf_embeddings(&transformer_path, is_schnell, &self.base.progress, None)?
        } else {
            transformer_path
        };

        let gpu_dtype = flux_runtime_dtype(device.is_cuda(), is_quantized, false);

        tracing::info!("GPU device: {:?}, GPU dtype: {:?}", device, gpu_dtype);

        // --- Load FLUX transformer + VAE on GPU first (variable size) ---
        // This must happen before T5/CLIP so we can measure remaining VRAM.

        // Check if full-precision transformer fits in VRAM before attempting load.
        if !is_quantized {
            let xformer_size = std::fs::metadata(&transformer_path)
                .map(|m| m.len())
                .unwrap_or(0);
            // Budget decision: subtract the OS / cuBLAS reserve so we don't
            // promise space the next allocator call cannot deliver.
            let free = usable_free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
            if free > 0 && xformer_size > free {
                bail!(
                    "transformer ({:.1} GB) exceeds available VRAM ({:.1} GB) — \
                     use a quantized model (q8/q4) instead of full-precision for this GPU",
                    xformer_size as f64 / 1e9,
                    free as f64 / 1e9,
                );
            }
        }

        let flux_cfg = if is_schnell {
            flux::model::Config::schnell()
        } else {
            flux::model::Config::dev()
        };

        let xformer_label = if is_quantized {
            "Loading FLUX transformer (GPU, quantized)"
        } else {
            "Loading FLUX transformer (GPU, BF16)"
        };
        self.base.progress.stage_start(xformer_label);
        let xformer_stage = Instant::now();
        tracing::info!(
            path = %transformer_path.display(),
            quantized = is_quantized,
            "loading FLUX transformer on GPU..."
        );

        let flux_model = if is_quantized {
            let vb = quantized_var_builder::VarBuilder::from_gguf(&transformer_path, &device)?;
            FluxTransformer::Quantized(flux::quantized_model::Flux::new(&flux_cfg, vb)?)
        } else {
            let flux_vb = flux_transformer_var_builder(flux_safetensors_var_builder(
                &transformer_path,
                gpu_dtype,
                &device,
                "FLUX transformer",
                &self.base.progress,
            )?);
            FluxTransformer::BF16(flux::model::Flux::new(&flux_cfg, flux_vb)?)
        };
        self.base
            .progress
            .stage_done(xformer_label, xformer_stage.elapsed());
        tracing::info!("FLUX transformer loaded on GPU");

        // Load VAE on GPU (small, ~300MB)
        // Tier 2: honor `advanced.vae` override.
        let vae_ref = effective_device_ref(
            self.pending_placement.as_ref(),
            |adv| Some(adv.vae.clone()),
            false,
        );
        let vae_device = crate::device::resolve_device(Some(vae_ref), || Ok(device.clone()))?;
        self.base.progress.stage_start("Loading VAE (GPU)");
        let vae_stage = Instant::now();
        tracing::info!(path = %self.base.paths.vae.display(), "loading VAE on GPU...");
        // Resolve VAE precision once at load time — see LoadedFlux::vae_dtype.
        let vae_dtype = crate::device::resolve_vae_dtype(gpu_dtype);
        let vae_vb = self.load_vae_var_builder(vae_dtype, &vae_device, "VAE")?;
        let vae_cfg = if is_schnell {
            flux::autoencoder::Config::schnell()
        } else {
            flux::autoencoder::Config::dev()
        };
        let vae = flux::autoencoder::AutoEncoder::new(&vae_cfg, vae_vb)?;
        self.base
            .progress
            .stage_done("Loading VAE (GPU)", vae_stage.elapsed());
        tracing::info!("VAE loaded on GPU");

        // --- Decide where to place T5 and CLIP based on remaining VRAM ---
        // Log the raw driver reading (matches `nvidia-smi`) but pass the
        // reserve-adjusted budget to variant resolution so quantized
        // encoders aren't picked when their footprint would push past the
        // OS / cuBLAS workspace headroom.
        let free_raw = free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
        let free = usable_free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
        if free_raw > 0 {
            self.base.progress.info(&format!(
                "Free VRAM after transformer+VAE: {}",
                fmt_gb(free_raw)
            ));
            tracing::info!(
                free_vram = free_raw,
                free_vram_usable = free,
                "free VRAM after loading transformer + VAE"
            );
        }

        // --- T5 encoder: auto-select variant based on VRAM or explicit preference ---
        self.base.progress.stage_start("Selecting T5 encoder");
        let t5_resolve_start = Instant::now();
        let t5_preference = self.t5_variant.as_deref();
        let (resolved_t5_path, t5_on_gpu, _t5_auto_device_label) =
            crate::encoders::variant_resolution::resolve_t5_variant(
                &self.base.progress,
                t5_preference,
                &device,
                free,
                &t5_encoder_path,
            )?;
        self.base
            .progress
            .stage_done("Selecting T5 encoder", t5_resolve_start.elapsed());
        // Tier 2 (if `advanced.t5` populated) overrides Tier 1 text_encoders group knob.
        let t5_ref =
            effective_device_ref(self.pending_placement.as_ref(), |adv| adv.t5.clone(), true);
        let auto_t5_device = if t5_on_gpu {
            device.clone()
        } else {
            cpu.clone()
        };
        let t5_device_owned =
            crate::device::resolve_device(Some(t5_ref), || Ok(auto_t5_device.clone()))?;
        let t5_device = &t5_device_owned;
        let t5_on_gpu = !t5_device.is_cpu();
        let t5_device_label = if t5_on_gpu { "GPU" } else { "CPU" };
        let t5_dtype = if t5_on_gpu { gpu_dtype } else { DType::F32 };

        // Load T5 encoder
        let t5_stage_label = format!("Loading T5 encoder ({t5_device_label})");
        self.base.progress.stage_start(&t5_stage_label);
        let t5_stage = Instant::now();
        tracing::info!(
            path = %resolved_t5_path.display(),
            device = %t5_device_label,
            "loading T5 encoder..."
        );
        let cached_t5_tok = self.get_cached_tokenizer(&t5_tokenizer_path);
        let cached_t5_tensors = self.get_cached_safetensors(&resolved_t5_path)?;
        let t5 = encoders::t5::T5Encoder::load_with_tokenizer_and_tensors(
            &resolved_t5_path,
            &t5_tokenizer_path,
            t5_device,
            t5_dtype,
            &self.base.progress,
            cached_t5_tok,
            cached_t5_tensors,
        )?;
        self.cache_tokenizer(&t5_tokenizer_path, t5.tokenizer_arc());
        self.base
            .progress
            .stage_done(&t5_stage_label, t5_stage.elapsed());
        tracing::info!(device = %t5_device_label, "T5 encoder loaded");

        // Re-check VRAM after T5 (it may have consumed GPU memory). Budget
        // decision → reserve-adjusted reading.
        let free_after_t5 = usable_free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
        let clip_on_gpu = should_use_gpu(
            device.is_cuda(),
            device.is_metal(),
            free_after_t5,
            CLIP_VRAM_THRESHOLD,
        );
        let clip_ref = effective_device_ref(
            self.pending_placement.as_ref(),
            |adv| adv.clip_l.clone(),
            true,
        );
        let auto_clip_device = if clip_on_gpu {
            device.clone()
        } else {
            cpu.clone()
        };
        let clip_device_owned =
            crate::device::resolve_device(Some(clip_ref), || Ok(auto_clip_device.clone()))?;
        let clip_device = &clip_device_owned;
        let clip_on_gpu = !clip_device.is_cpu();
        let clip_dtype = if clip_on_gpu { gpu_dtype } else { DType::F32 };
        let clip_device_label = if clip_on_gpu { "GPU" } else { "CPU" };

        // Load CLIP encoder
        let clip_stage_label = format!("Loading CLIP encoder ({clip_device_label})");
        self.base.progress.stage_start(&clip_stage_label);
        let clip_stage = Instant::now();
        tracing::info!(
            path = %clip_encoder_path.display(),
            device = clip_device_label,
            "loading CLIP encoder..."
        );
        let cached_clip_tok = self.get_cached_tokenizer(&clip_tokenizer_path);
        let cached_clip_tensors = self.get_cached_safetensors(&clip_encoder_path)?;
        let clip = encoders::clip::ClipEncoder::load_with_tokenizer_and_tensors(
            &clip_encoder_path,
            &clip_tokenizer_path,
            clip_device,
            clip_dtype,
            &self.base.progress,
            cached_clip_tok,
            cached_clip_tensors,
        )?;
        self.cache_tokenizer(&clip_tokenizer_path, clip.tokenizer_arc());
        self.base
            .progress
            .stage_done(&clip_stage_label, clip_stage.elapsed());
        tracing::info!(device = clip_device_label, "CLIP encoder loaded");

        self.base.loaded = Some(LoadedFlux {
            flux_model: Some(flux_model),
            t5,
            clip,
            vae,
            device,
            dtype: gpu_dtype,
            vae_dtype,
            is_schnell,
            is_quantized,
            transformer_path,
            t5_encoder_path: resolved_t5_path,
        });

        tracing::info!(model = %self.base.model_name, "all model components loaded successfully");
        Ok(())
    }

    /// Generate an image using sequential loading strategy.
    ///
    /// Loads components one at a time and drops them when done to minimize peak memory:
    /// 1. Load T5 → encode → drop T5
    /// 2. Load CLIP → encode → drop CLIP
    /// 3. Load transformer + VAE → denoise → drop transformer
    /// 4. VAE decode → drop VAE
    ///
    /// Peak memory: max(T5_size, transformer_size + VAE_size) instead of sum(all).
    fn generate_sequential(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let is_schnell = self.detect_is_schnell();
        let mut is_quantized = self.detect_is_quantized();

        let (t5_encoder_path, t5_tokenizer_path, clip_encoder_path, clip_tokenizer_path) =
            self.validate_paths()?;

        // Check memory budget
        if let Some(warning) = check_memory_budget(&self.base.paths, LoadStrategy::Sequential) {
            self.base.progress.info(&warning);
        }

        let transformer_ref = effective_device_ref(
            self.pending_placement.as_ref(),
            |adv| Some(adv.transformer.clone()),
            false,
        );
        let device = crate::device::resolve_device(Some(transformer_ref), || {
            crate::device::create_device(self.base.gpu_ordinal, &self.base.progress)
        })?;

        // Use cached transformer path to avoid file I/O on every sequential call.
        let transformer_path = if let Some(ref cached) = self.cached_transformer_path {
            if cached
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false)
            {
                is_quantized = true;
            }
            cached.clone()
        } else {
            let transformer_is_fp8 = self.check_transformer_is_fp8(is_quantized);
            let p = if transformer_is_fp8 {
                let p = ensure_fp8_gguf_cache(&self.base.paths.transformer, &self.base.progress)?;
                is_quantized = true;
                p
            } else {
                self.base.paths.transformer.clone()
            };
            // Patch city96-format GGUFs missing embedding layers
            let p = if is_quantized {
                ensure_gguf_embeddings(&p, is_schnell, &self.base.progress, None)?
            } else {
                p
            };
            self.cached_transformer_path = Some(p.clone());
            p
        };

        let gpu_dtype = flux_runtime_dtype(device.is_cuda(), is_quantized, false);

        let start = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);

        let width = req.width as usize;
        let height = req.height as usize;

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            steps = req.steps,
            "starting sequential FLUX generation"
        );

        self.base
            .progress
            .info("Using sequential loading (load-use-drop) to minimize peak memory");

        let (t5_emb, clip_emb) = if let Some((t5_emb, clip_emb)) = Self::restore_prompt_cache(
            &self.base.progress,
            &self.prompt_cache,
            &req.prompt,
            &device,
            gpu_dtype,
        )? {
            (t5_emb, clip_emb)
        } else {
            // --- Phase 1: T5 encoding ---
            // Reserve-adjusted reading drives the variant choice.
            let free = usable_free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
            self.base.progress.stage_start("Selecting T5 encoder");
            let t5_resolve_start = Instant::now();
            let t5_preference = self.t5_variant.as_deref();
            let (resolved_t5_path, t5_on_gpu, _t5_auto_device_label) =
                crate::encoders::variant_resolution::resolve_t5_variant(
                    &self.base.progress,
                    t5_preference,
                    &device,
                    free,
                    &t5_encoder_path,
                )?;
            self.base
                .progress
                .stage_done("Selecting T5 encoder", t5_resolve_start.elapsed());

            let t5_ref =
                effective_device_ref(self.pending_placement.as_ref(), |adv| adv.t5.clone(), true);
            let auto_t5_device = if t5_on_gpu {
                device.clone()
            } else {
                Device::Cpu
            };
            let t5_device_owned =
                crate::device::resolve_device(Some(t5_ref), || Ok(auto_t5_device.clone()))?;
            let t5_device = &t5_device_owned;
            let t5_on_gpu = !t5_device.is_cpu();
            let t5_device_label = if t5_on_gpu { "GPU" } else { "CPU" };
            let t5_dtype = if t5_on_gpu { gpu_dtype } else { DType::F32 };

            let t5_size = std::fs::metadata(&resolved_t5_path)
                .map(|m| m.len())
                .unwrap_or(0);
            // T5 activations: ~256 MB workspace (floor) — small relative to
            // the 9 GB encoder weights and only resident during encoding.
            let t5_activation_budget = crate::device::activation_bytes(
                req.width,
                req.height,
                1,
                crate::device::dtype_bytes(t5_dtype),
                crate::device::ActivationFamily::SmallTransformer,
            );
            preflight_memory_check("T5 encoder", t5_size, t5_activation_budget)?;
            if let Some(status) = memory_status_string() {
                self.base.progress.info(&status);
            }

            let t5_stage_label = format!("Loading T5 encoder ({t5_device_label})");
            self.base.progress.stage_start(&t5_stage_label);
            let t5_stage = Instant::now();
            let cached_t5_tok = self.get_cached_tokenizer(&t5_tokenizer_path);
            let cached_t5_tensors = self.get_cached_safetensors(&resolved_t5_path)?;
            let mut t5 = encoders::t5::T5Encoder::load_with_tokenizer_and_tensors(
                &resolved_t5_path,
                &t5_tokenizer_path,
                t5_device,
                t5_dtype,
                &self.base.progress,
                cached_t5_tok,
                cached_t5_tensors,
            )?;
            self.cache_tokenizer(&t5_tokenizer_path, t5.tokenizer_arc());
            self.base
                .progress
                .stage_done(&t5_stage_label, t5_stage.elapsed());

            self.base.progress.stage_start("Encoding prompt (T5)");
            let encode_t5 = Instant::now();
            // Park to CPU immediately so the transformer load + LoRA merge
            // window (next 200–500 ms) doesn't have to budget for ~12 MB of
            // T5 output sitting on GPU. Idempotent on the GGUF path where T5
            // already produces CPU tensors.
            let t5_emb = park_cond_to_cpu(&t5.encode(&req.prompt, &device, gpu_dtype)?)?;
            self.base.progress.phase_done(
                crate::ProgressPhase::PromptEncode,
                "Encoding prompt (T5)",
                encode_t5.elapsed(),
            );

            drop(t5);
            self.base.progress.info("Freed T5 encoder");
            tracing::info!("T5 encoder dropped (sequential mode)");

            // --- Phase 2: CLIP encoding ---
            // Reserve-adjusted reading — should_use_gpu must respect the
            // same OS / cuBLAS workspace headroom as the T5 placement above.
            let free_for_clip = usable_free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
            let clip_on_gpu = should_use_gpu(
                device.is_cuda(),
                device.is_metal(),
                free_for_clip,
                CLIP_VRAM_THRESHOLD,
            );
            let clip_ref = effective_device_ref(
                self.pending_placement.as_ref(),
                |adv| adv.clip_l.clone(),
                true,
            );
            let auto_clip_device = if clip_on_gpu {
                device.clone()
            } else {
                Device::Cpu
            };
            let clip_device_owned =
                crate::device::resolve_device(Some(clip_ref), || Ok(auto_clip_device.clone()))?;
            let clip_device = &clip_device_owned;
            let clip_on_gpu = !clip_device.is_cpu();
            let clip_dtype = if clip_on_gpu { gpu_dtype } else { DType::F32 };
            let clip_device_label = if clip_on_gpu { "GPU" } else { "CPU" };

            let clip_stage_label = format!("Loading CLIP encoder ({clip_device_label})");
            self.base.progress.stage_start(&clip_stage_label);
            let clip_stage = Instant::now();
            let cached_clip_tok = self.get_cached_tokenizer(&clip_tokenizer_path);
            let cached_clip_tensors = self.get_cached_safetensors(&clip_encoder_path)?;
            let clip = encoders::clip::ClipEncoder::load_with_tokenizer_and_tensors(
                &clip_encoder_path,
                &clip_tokenizer_path,
                clip_device,
                clip_dtype,
                &self.base.progress,
                cached_clip_tok,
                cached_clip_tensors,
            )?;
            self.cache_tokenizer(&clip_tokenizer_path, clip.tokenizer_arc());
            self.base
                .progress
                .stage_done(&clip_stage_label, clip_stage.elapsed());

            self.base.progress.stage_start("Encoding prompt (CLIP)");
            let encode_clip = Instant::now();
            // Park to CPU for the same reason as T5 above — keeps the
            // TE→transformer transition window from carrying GPU residency
            // we don't need.
            let clip_emb = {
                let mut clip = clip;
                park_cond_to_cpu(&clip.encode(&req.prompt, &device, gpu_dtype)?)?
            };
            self.base.progress.phase_done(
                crate::ProgressPhase::PromptEncode,
                "Encoding prompt (CLIP)",
                encode_clip.elapsed(),
            );

            self.base.progress.info("Freed CLIP encoder");
            tracing::info!("CLIP encoder dropped (sequential mode)");

            // Cache stores via `CachedTensor::from_tensor`, which itself
            // moves to CPU; passing CPU tensors here avoids an unnecessary
            // round-trip on the GGUF path.
            Self::store_prompt_cache(&self.prompt_cache, &req.prompt, &t5_emb, &clip_emb)?;
            (t5_emb, clip_emb)
        };

        // Synchronize to ensure freed T5/CLIP VRAM is reclaimed before
        // loading the transformer (critical for FP8 models that expand to F16).
        device.synchronize()?;

        // --- Phase 3: Load transformer, denoise ---
        let xformer_size = std::fs::metadata(&transformer_path)
            .map(|m| m.len())
            .unwrap_or(0);
        let vae_file_size = std::fs::metadata(&self.base.paths.vae)
            .map(|m| m.len())
            .unwrap_or(0);

        // LoRA + GGUF: supported via selective dequantization.
        // LoRA-affected layers are dequantized to F32 on CPU, patched, then
        // re-quantized back to the original GGML dtype. Non-LoRA tensors are
        // left quantized and untouched.

        // Per-request activation budget — replaces the fixed 3 GB
        // INFERENCE_HEADROOM. Scales with resolution and dtype, so a 768²
        // generation isn't false-offloaded on a 16 GB card while a 2048²
        // generation isn't under-budgeted.
        let activation_budget = crate::device::activation_bytes(
            req.width,
            req.height,
            1, // FLUX is guidance-distilled — single forward per step.
            crate::device::dtype_bytes(gpu_dtype),
            crate::device::ActivationFamily::FluxDit,
        );

        // Determine if block-level offloading should be used.
        let use_offload = if !is_quantized {
            // Reserve-adjusted reading: subtract the OS reserve before passing
            // to `should_offload`, which budgets transformer + activation
            // headroom against this number.
            let free = usable_free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
            if self.offload || should_offload(xformer_size, free, activation_budget) {
                if free > 0 && free < MIN_OFFLOAD_VRAM {
                    bail!(
                        "GPU only has {:.1} GB free — at least {:.1} GB is required \
                         for block-level offloading",
                        free as f64 / 1e9,
                        MIN_OFFLOAD_VRAM as f64 / 1e9,
                    );
                }
                true
            } else if free > 0 && xformer_size > free {
                bail!(
                    "transformer ({:.1} GB) exceeds available VRAM ({:.1} GB) — \
                     use a quantized model (q8/q4) or --offload for block-level streaming",
                    xformer_size as f64 / 1e9,
                    free as f64 / 1e9,
                );
            } else {
                false
            }
        } else {
            if self.offload {
                tracing::warn!(
                    "block-level offloading is not supported for quantized models; \
                     --offload / MOLD_OFFLOAD=1 will be ignored"
                );
            }
            false
        };

        // Even when offloading, blocks must still fit in system RAM on unified-memory
        // (Metal) hosts — preflight catches machines with insufficient total memory.
        if !use_offload || device.is_metal() {
            preflight_memory_check(
                "FLUX transformer + VAE",
                xformer_size + vae_file_size,
                activation_budget,
            )?;
        }
        if let Some(status) = memory_status_string() {
            self.base.progress.info(&status);
        }

        let flux_cfg = if is_schnell {
            flux::model::Config::schnell()
        } else {
            flux::model::Config::dev()
        };

        let active_loras = effective_loras(req);
        let has_lora = !active_loras.is_empty();
        let xformer_label = if has_lora && use_offload {
            "Loading FLUX transformer + LoRA (offloaded)"
        } else if has_lora && is_quantized {
            "Loading FLUX transformer + LoRA (GPU, quantized + selective deq)"
        } else if has_lora {
            "Loading FLUX transformer + LoRA (GPU, BF16)"
        } else if use_offload {
            "Loading FLUX transformer (offloaded, blocks on CPU)"
        } else if is_quantized {
            "Loading FLUX transformer (GPU, quantized)"
        } else {
            "Loading FLUX transformer (GPU, BF16)"
        };
        self.base.progress.stage_start(xformer_label);
        let xformer_stage = Instant::now();

        let bypass_mode = LoraBypassMode::from_env();
        // For the offloaded path, bypass is the obvious win whenever
        // LoRAs are active: the legacy merge path runs `B@A·scale` on
        // every targeted CPU-resident BF16 tensor and rebuilds the
        // ~24 GB block buffer on every LoRA swap. Bypass keeps adapters
        // GPU-resident, so a swap is just a registry replace.
        let use_offload_bypass =
            should_use_offload_bypass_registry(use_offload, has_lora, bypass_mode);
        let offload_lora_registry = if use_offload_bypass {
            // Build the registry before adaptive residency planning so its
            // GPU-resident adapter tensors are included in the free-VRAM
            // reading used to decide how many base blocks can stay resident.
            build_lora_registry(
                &active_loras,
                &flux_cfg,
                &device,
                gpu_dtype,
                &self.base.progress,
            )?
        } else {
            None
        };

        let flux_model = if use_offload {
            // Load transformer blocks on CPU. With bypass enabled the
            // base weights are loaded *unmodified* (LoRA contributions
            // are added at forward time); without bypass we fall back
            // to the merge-on-load path.
            let cpu_vb: VarBuilder = if has_lora && !use_offload_bypass {
                // Legacy LoRA backend: loads from mmap to CPU, patches inline
                flux_lora_var_builder(
                    &transformer_path,
                    &active_loras,
                    gpu_dtype,
                    &Device::Cpu,
                    &self.base.progress,
                    self.lora_delta_cache_handle(),
                )?
            } else {
                flux_transformer_var_builder(flux_safetensors_var_builder(
                    &transformer_path,
                    gpu_dtype,
                    &Device::Cpu,
                    "FLUX transformer",
                    &self.base.progress,
                )?)
            };
            let offloaded = crate::flux::offload::OffloadedFluxTransformer::load(
                cpu_vb,
                &flux_cfg,
                &device,
                self.base.gpu_ordinal,
                activation_budget,
                offload_lora_registry,
                &self.base.progress,
            )?;
            FluxTransformer::Offloaded(offloaded)
        } else if is_quantized && has_lora {
            // GGUF + LoRA: bypass-mode keeps base weights untouched and
            // applies LoRA deltas at forward time. Saves the
            // dequant→merge→requant cycle that previously cost minutes
            // and ~95 GB CPU peak per LoRA load on Q8.
            let bypass_quantized = bypass_mode != LoraBypassMode::Off;
            if bypass_quantized {
                let registry = build_lora_registry(
                    &active_loras,
                    &flux_cfg,
                    &device,
                    gpu_dtype,
                    &self.base.progress,
                )?;
                let vb = quantized_var_builder::VarBuilder::from_gguf(&transformer_path, &device)?;
                FluxTransformer::QuantizedBypass(
                    crate::flux::quantized_transformer::QuantizedFluxTransformer::load(
                        &flux_cfg,
                        vb,
                        registry.as_ref(),
                        &self.base.progress,
                    )?,
                )
            } else {
                // Legacy fallback: dequantize LoRA-affected layers, keep rest quantized.
                let vb = flux_gguf_lora_var_builder(
                    &transformer_path,
                    &active_loras,
                    &device,
                    &self.base.progress,
                    self.lora_delta_cache_handle(),
                )?;
                FluxTransformer::Quantized(flux::quantized_model::Flux::new(&flux_cfg, vb)?)
            }
        } else if is_quantized {
            let vb = quantized_var_builder::VarBuilder::from_gguf(&transformer_path, &device)?;
            FluxTransformer::Quantized(flux::quantized_model::Flux::new(&flux_cfg, vb)?)
        } else if has_lora {
            // LoRA without offload (GPU has enough VRAM for full model)
            let flux_vb = flux_lora_var_builder(
                &transformer_path,
                &active_loras,
                gpu_dtype,
                &device,
                &self.base.progress,
                self.lora_delta_cache_handle(),
            )?;
            FluxTransformer::BF16(flux::model::Flux::new(&flux_cfg, flux_vb)?)
        } else {
            let flux_vb = flux_transformer_var_builder(flux_safetensors_var_builder(
                &transformer_path,
                gpu_dtype,
                &device,
                "FLUX transformer",
                &self.base.progress,
            )?);
            FluxTransformer::BF16(flux::model::Flux::new(&flux_cfg, flux_vb)?)
        };
        self.base
            .progress
            .stage_done(xformer_label, xformer_stage.elapsed());
        if let Some(status) = memory_status_string() {
            self.base.progress.info(&status);
        }

        // Generate noise and build state
        let noise_dtype = if is_quantized { DType::F32 } else { gpu_dtype };
        let latent_h = height / 16 * 2;
        let latent_w = width / 16 * 2;
        // Pre-compute timestep schedule (needed before mixing for img2img).
        // For non-schnell models the schedule depends on image_seq_len which
        // we can derive from latent dimensions without the actual tensor.
        let image_seq_len = (latent_h / 2) * (latent_w / 2);
        let mut timesteps = if is_schnell {
            flux::sampling::get_schedule(req.steps as usize, None)
        } else {
            flux::sampling::get_schedule(req.steps as usize, Some((image_seq_len, 0.5, 1.15)))
        };

        if req.source_image.is_some() {
            let start_index = crate::img2img::img2img_start_index(req.steps as usize, req.strength);
            timesteps = timesteps[start_index..].to_vec();
            tracing::info!(
                strength = req.strength,
                start_index,
                start_timestep = timesteps[0],
                schedule = ?timesteps,
                remaining_steps = timesteps.len().saturating_sub(1),
                "img2img: truncated schedule from strength"
            );
        }

        // For img2img we need the VAE before denoising (to encode the source image).
        // For txt2img we defer VAE loading until after denoising to maximize VRAM
        // available for the transformer — critical for FP8 models expanded to F16.
        let vae_cfg = if is_schnell {
            flux::autoencoder::Config::schnell()
        } else {
            flux::autoencoder::Config::dev()
        };
        // Resolve once so the early img2img encode and the later decode load
        // the VAE at the same precision; fixes shape mismatch when
        // `MOLD_VAE_DTYPE=fp32` would otherwise upgrade only the decode path.
        let early_vae_dtype = crate::device::resolve_vae_dtype(gpu_dtype);

        let (img, inpaint_ctx, early_vae) = if let Some(ref source_bytes) = req.source_image {
            let start_t = timesteps[0];

            // Load VAE early for source image encoding
            self.base.progress.stage_start("Loading VAE (GPU)");
            let vae_stage = Instant::now();
            let vae_vb = self.load_vae_var_builder(early_vae_dtype, &device, "VAE")?;
            let vae = flux::autoencoder::AutoEncoder::new(&vae_cfg, vae_vb)?;
            self.base
                .progress
                .stage_done("Loading VAE (GPU)", vae_stage.elapsed());

            self.base
                .progress
                .stage_start("Encoding source image (VAE)");
            let encode_start = Instant::now();
            let source_tensor = crate::img_utils::decode_source_image(
                source_bytes,
                req.width,
                req.height,
                crate::img_utils::NormalizeRange::MinusOneToOne,
                &device,
                early_vae_dtype,
            )?;
            // FLUX VAE expects pixels in [-1, 1]; encode applies shift/scale internally
            let encoded = vae.encode(&source_tensor)?;
            self.base.progress.phase_done(
                crate::ProgressPhase::Vae,
                "Encoding source image (VAE)",
                encode_start.elapsed(),
            );

            // Flow-matching img2img: interpolate between encoded latents and noise
            // at the exact noise level matching the first timestep in the schedule
            let noise = crate::engine::seeded_randn(
                seed,
                &[1, 16, latent_h, latent_w],
                &device,
                noise_dtype,
            )?;
            let encoded = encoded.to_dtype(noise_dtype)?;

            // Build inpaint context if mask provided
            let inpaint_ctx = if let Some(ref mask_bytes) = req.mask_image {
                let mask = crate::img_utils::decode_mask_image(
                    mask_bytes,
                    latent_h,
                    latent_w,
                    &device,
                    noise_dtype,
                )?;
                Some(crate::img_utils::InpaintContext {
                    original_latents: encoded.clone(),
                    mask,
                    noise: noise.clone(),
                })
            } else {
                None
            };

            // latent = (1 - t) * encoded + t * noise
            // t matches the first schedule timestep, so denoising starts at the correct level
            let img = ((&encoded * (1.0 - start_t))? + (&noise * start_t)?)?;
            (img, inpaint_ctx, Some(vae))
        } else {
            let img = crate::engine::seeded_randn(
                seed,
                &[1, 16, latent_h, latent_w],
                &device,
                noise_dtype,
            )?;
            (img, None, None)
        };

        // Migrate the parked conditioning tensors back to GPU now that the
        // transformer load + LoRA merge phase is over. `to_device` on a
        // tensor already on `device` is a no-op clone, so the cache-restore
        // path (which returns GPU tensors) costs nothing here.
        let t5_emb = t5_emb.to_device(&device)?;
        let clip_emb = clip_emb.to_device(&device)?;
        let (t5_emb_state, clip_emb_state, img_state) = if is_quantized {
            (
                t5_emb.to_dtype(DType::F32)?,
                clip_emb.to_dtype(DType::F32)?,
                img.to_dtype(DType::F32)?,
            )
        } else {
            (t5_emb, clip_emb, img)
        };

        let state = flux::sampling::State::new(&t5_emb_state, &clip_emb_state, &img_state)?;
        let inpaint_ctx = inpaint_ctx
            .as_ref()
            .map(crate::img2img::pack_flux_inpaint_context)
            .transpose()?;

        let denoise_label = format!("Denoising ({} steps)", timesteps.len().saturating_sub(1));
        self.base.progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();

        let previewer = crate::latent_preview::LatentPreviewer::flux1(height, width);
        let img = flux_model.denoise(
            &state.img,
            &state.img_ids,
            &state.txt,
            &state.txt_ids,
            &state.vec,
            &timesteps,
            req.guidance,
            &self.base.progress,
            inpaint_ctx.as_ref(),
            Some(&previewer),
        )?;

        let img = flux::sampling::unpack(&img, height, width)?;
        self.base
            .progress
            .stage_done(&denoise_label, denoise_start.elapsed());

        // Drop transformer + state to free memory for VAE decode
        drop(inpaint_ctx);
        drop(flux_model);
        self.base.progress.info("Freed FLUX transformer");
        drop(state);
        drop(t5_emb_state);
        drop(clip_emb_state);
        drop(img_state);
        // Synchronize to ensure CUDA frees dropped memory before VAE allocates
        device.synchronize()?;
        tracing::info!("Transformer dropped (sequential mode), decoding VAE...");

        // --- Phase 4: VAE decode ---
        // Use VAE from img2img path if already loaded, otherwise load now
        // (deferred loading saves ~300MB VRAM during denoising for FP8 models).
        // Sequential path resolves MOLD_VAE_DTYPE per request — env changes
        // take effect on the next generate() without an engine reload.
        let vae_dtype = crate::device::resolve_vae_dtype(gpu_dtype);
        let vae = if let Some(vae) = early_vae {
            vae
        } else {
            self.base.progress.stage_start("Loading VAE (GPU)");
            let vae_stage = Instant::now();
            let vae_vb = self.load_vae_var_builder(vae_dtype, &device, "VAE")?;
            let vae = flux::autoencoder::AutoEncoder::new(&vae_cfg, vae_vb)?;
            self.base
                .progress
                .stage_done("Loading VAE (GPU)", vae_stage.elapsed());
            vae
        };
        self.base.progress.stage_start("VAE decode");
        let vae_decode_start = Instant::now();
        let img_for_vae = img.to_dtype(vae_dtype)?;
        let device_for_sync = device.clone();
        let img = crate::vae_tiling::decode_with_oom_fallback(
            &img_for_vae,
            |latents| vae.decode(latents).map_err(Into::into),
            || {
                if let Err(e) = device_for_sync.synchronize() {
                    tracing::warn!(
                        "FLUX (sequential) device.synchronize() after VAE OOM failed: {e}"
                    );
                }
            },
        )?;

        let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
        let img = img.i(0)?;

        self.base.progress.phase_done(
            crate::ProgressPhase::Vae,
            "VAE decode",
            vae_decode_start.elapsed(),
        );
        // VAE dropped here

        let output_metadata = build_output_metadata(req, seed, None);
        let image_bytes = encode_image(
            &img,
            req.resolved_output_format(),
            req.width,
            req.height,
            output_metadata.as_ref(),
        )?;

        let generation_time_ms = start.elapsed().as_millis() as u64;
        tracing::info!(generation_time_ms, seed, "sequential generation complete");

        Ok(GenerateResponse {
            images: vec![ImageData {
                data: image_bytes,
                format: req.resolved_output_format(),
                width: req.width,
                height: req.height,
                index: 0,
            }],
            generation_time_ms,
            model: req.model.clone(),
            seed_used: seed,
            video: None,
            gpu: None,
        })
    }
}

impl FluxEngine {
    fn defers_eager_load(&mut self) -> bool {
        self.base.load_strategy == LoadStrategy::Sequential
            || (self.offload && !self.detect_is_quantized())
    }

    fn uses_sequential_generate_path(&mut self) -> bool {
        self.defers_eager_load()
    }

    fn generate_inner(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        if req.scheduler.is_some() {
            tracing::warn!("scheduler selection not supported for FLUX (flow-matching), ignoring");
        }

        // Sequential mode: load-use-drop each component. Forced FLUX offload
        // also routes here because block streaming is chosen after prompt
        // encoding; eager preload would put the full BF16 transformer on GPU
        // before offload can take effect.
        if self.uses_sequential_generate_path() {
            return self.generate_sequential(req);
        }

        // Eager mode: use pre-loaded components
        // LoRA is supported — the transformer is rebuilt from disk on each generation
        // (dropped for VAE decode), so LoRA is applied during the rebuild via a
        // patched VarBuilder. No additional overhead compared to non-LoRA eager mode.
        // Borrow progress reporter separately from loaded state.
        let progress = &self.base.progress;
        let prompt_cache = &self.prompt_cache;

        // Grab path references before borrowing loaded mutably
        let t5_encoder_path = self
            .base
            .loaded
            .as_ref()
            .map(|l| l.t5_encoder_path.clone())
            .or_else(|| self.base.paths.t5_encoder.clone())
            .ok_or_else(|| anyhow::anyhow!("T5 encoder path required for FLUX models"))?;
        let clip_encoder_path = self
            .base
            .paths
            .clip_encoder
            .clone()
            .ok_or_else(|| anyhow::anyhow!("CLIP encoder path required for FLUX models"))?;
        let transformer_path = self
            .base
            .loaded
            .as_ref()
            .map(|l| l.transformer_path.clone())
            .unwrap_or_else(|| self.base.paths.transformer.clone());

        // Captured before we mutably borrow `self.base.loaded` via the
        // OptionRestoreGuard below — once that borrow is live, calling
        // `self.lora_delta_cache_handle()` would conflict.
        let cache_handle = self.lora_delta_cache_handle();

        let mut loaded = OptionRestoreGuard::take(&mut self.base.loaded)
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;

        let start = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);

        let width = req.width as usize;
        let height = req.height as usize;
        let loaded_dtype = loaded.dtype;
        let loaded_device = loaded.device.clone();

        tracing::info!(
            prompt = %req.prompt,
            seed,
            width,
            height,
            steps = req.steps,
            "starting generation"
        );

        (|| -> Result<GenerateResponse> {
            // Only rebuild the transformer when the LoRA stack changes
            // (any adapter swap, scale change, add, remove, or reorder).
            let active_loras = effective_loras(req);
            let requested_stack = fingerprint_stack(&active_loras);
            if requested_stack != self.active_lora {
                if loaded.flux_model.is_some() {
                    loaded.flux_model = None;
                    loaded.device.synchronize()?;
                }
                self.active_lora = requested_stack;
            }

            if loaded.flux_model.is_none() {
                let has_lora = !active_loras.is_empty();
                let xformer_label = match (loaded.is_quantized, has_lora) {
                    (true, true) => "Reloading FLUX transformer (GPU, quantized + LoRA)",
                    (true, false) => "Reloading FLUX transformer (GPU, quantized)",
                    (false, true) if loaded.dtype == DType::F16 => {
                        "Reloading FLUX transformer (GPU, FP16 + LoRA)"
                    }
                    (false, true) => "Reloading FLUX transformer (GPU, BF16 + LoRA)",
                    (false, false) if loaded.dtype == DType::F16 => {
                        "Reloading FLUX transformer (GPU, FP16)"
                    }
                    (false, false) => "Reloading FLUX transformer (GPU, BF16)",
                };
                progress.stage_start(xformer_label);
                let reload_start = Instant::now();
                let flux_cfg = if loaded.is_schnell {
                    flux::model::Config::schnell()
                } else {
                    flux::model::Config::dev()
                };
                let bypass_mode = LoraBypassMode::from_env();
                loaded.flux_model = Some(if loaded.is_quantized && has_lora {
                    // Quantized + LoRA stack. Bypass-mode (default `auto`)
                    // installs the LoRA at forward time on top of the
                    // quantized base — no dequant→merge→requant. Legacy
                    // fallback (MOLD_LORA_BYPASS=off) goes through
                    // `gguf_lora_var_builder`.
                    let bypass_quantized = bypass_mode != LoraBypassMode::Off;
                    if bypass_quantized {
                        let registry = build_lora_registry(
                            &active_loras,
                            &flux_cfg,
                            &loaded.device,
                            loaded.dtype,
                            progress,
                        )?;
                        let vb = quantized_var_builder::VarBuilder::from_gguf(
                            &transformer_path,
                            &loaded.device,
                        )?;
                        FluxTransformer::QuantizedBypass(
                            crate::flux::quantized_transformer::QuantizedFluxTransformer::load(
                                &flux_cfg,
                                vb,
                                registry.as_ref(),
                                progress,
                            )?,
                        )
                    } else {
                        let vb = flux_gguf_lora_var_builder(
                            &transformer_path,
                            &active_loras,
                            &loaded.device,
                            progress,
                            cache_handle.clone(),
                        )?;
                        FluxTransformer::Quantized(flux::quantized_model::Flux::new(&flux_cfg, vb)?)
                    }
                } else if loaded.is_quantized {
                    let vb = quantized_var_builder::VarBuilder::from_gguf(
                        &transformer_path,
                        &loaded.device,
                    )?;
                    FluxTransformer::Quantized(flux::quantized_model::Flux::new(&flux_cfg, vb)?)
                } else if has_lora {
                    // BF16 + LoRA stack: merge all deltas during construction
                    let flux_vb = flux_lora_var_builder(
                        &transformer_path,
                        &active_loras,
                        loaded.dtype,
                        &loaded.device,
                        progress,
                        cache_handle.clone(),
                    )?;
                    FluxTransformer::BF16(flux::model::Flux::new(&flux_cfg, flux_vb)?)
                } else {
                    let flux_vb = flux_transformer_var_builder(flux_safetensors_var_builder(
                        &transformer_path,
                        loaded.dtype,
                        &loaded.device,
                        "FLUX transformer",
                        progress,
                    )?);
                    FluxTransformer::BF16(flux::model::Flux::new(&flux_cfg, flux_vb)?)
                });
                progress.stage_done(xformer_label, reload_start.elapsed());
            }

            if let Some((t5_emb, clip_emb)) = Self::restore_prompt_cache(
                progress,
                prompt_cache,
                &req.prompt,
                &loaded_device,
                loaded_dtype,
            )? {
                return Self::generate_with_embeddings(
                    progress,
                    req,
                    &mut loaded,
                    t5_emb,
                    clip_emb,
                    seed,
                    width,
                    height,
                    start,
                    self.base.gpu_ordinal,
                );
            }

            if loaded.t5.model.is_none() {
                let label = if loaded.t5.is_parked() {
                    "Unparking T5 encoder (CPU→GPU)"
                } else {
                    "Reloading T5 encoder (GPU)"
                };
                progress.stage_start(label);
                let reload_start = Instant::now();
                if loaded.t5.is_parked() {
                    loaded.t5.unpark_to_gpu(loaded_dtype, progress)?;
                } else {
                    loaded.t5.reload(&t5_encoder_path, loaded_dtype, progress)?;
                }
                progress.stage_done(label, reload_start.elapsed());
            }
            if loaded.clip.model.is_none() {
                let label = if loaded.clip.is_parked() {
                    "Unparking CLIP encoder (CPU→GPU)"
                } else {
                    "Reloading CLIP encoder (GPU)"
                };
                progress.stage_start(label);
                let reload_start = Instant::now();
                if loaded.clip.is_parked() {
                    loaded.clip.unpark_to_gpu(loaded_dtype, progress)?;
                } else {
                    loaded
                        .clip
                        .reload(&clip_encoder_path, loaded_dtype, progress)?;
                }
                progress.stage_done(label, reload_start.elapsed());
            }

            progress.stage_start("Encoding prompt (T5)");
            let encode_t5 = Instant::now();
            // Park to CPU between encode and denoise so the transformer
            // load + LoRA merge window (next ~200–500 ms) doesn't have to
            // budget for this tensor sitting on GPU. Idempotent on GGUF.
            let t5_emb = park_cond_to_cpu(&loaded.t5.encode(
                &req.prompt,
                &loaded_device,
                loaded_dtype,
            )?)?;
            progress.phase_done(
                crate::ProgressPhase::PromptEncode,
                "Encoding prompt (T5)",
                encode_t5.elapsed(),
            );
            tracing::info!("T5 encoding complete");

            progress.stage_start("Encoding prompt (CLIP)");
            let encode_clip = Instant::now();
            let clip_emb = park_cond_to_cpu(&loaded.clip.encode(
                &req.prompt,
                &loaded_device,
                loaded_dtype,
            )?)?;
            progress.phase_done(
                crate::ProgressPhase::PromptEncode,
                "Encoding prompt (CLIP)",
                encode_clip.elapsed(),
            );
            tracing::info!("CLIP encoding complete");
            // CachedTensor::from_tensor already moves to CPU — passing CPU
            // tensors here avoids the round-trip on the GGUF path.
            Self::store_prompt_cache(prompt_cache, &req.prompt, &t5_emb, &clip_emb)?;

            // Drop or park encoders to free GPU memory for denoising.
            //
            // Default (`MOLD_KEEP_TE_RAM=0`): drop weights from RAM too. The
            // next request re-mmaps from disk (~2-4 s for T5 Q8+LoRAs).
            //
            // Park mode (`MOLD_KEEP_TE_RAM=1`): move parameters to CPU host
            // RAM and drop the GPU copy. Next request only pays a CPU→GPU
            // tensor copy (~100-300 ms vs 2-4 s) — mirrors ComfyUI's
            // `text_encoder_offload_device()` behavior.
            //
            // On Metal (unified memory) parking is not a win since CPU and
            // GPU share the same physical pool, so we still drop there.
            let is_metal = loaded.device.is_metal();
            let park_mode = crate::device::keep_te_in_ram() && !is_metal;
            let mut dropped_gpu_encoder = false;
            if loaded.t5.on_gpu || is_metal {
                if loaded.t5.on_gpu {
                    dropped_gpu_encoder = true;
                }
                if park_mode {
                    loaded.t5.park_to_cpu()?;
                    tracing::info!(
                        on_gpu = loaded.t5.on_gpu,
                        "T5 encoder parked to CPU host RAM"
                    );
                } else {
                    loaded.t5.drop_weights();
                    tracing::info!(
                        on_gpu = loaded.t5.on_gpu,
                        "T5 encoder dropped to free memory for denoising"
                    );
                }
            }
            if loaded.clip.on_gpu || is_metal {
                if loaded.clip.on_gpu {
                    dropped_gpu_encoder = true;
                }
                if park_mode {
                    loaded.clip.park_to_cpu()?;
                    tracing::info!(
                        on_gpu = loaded.clip.on_gpu,
                        "CLIP encoder parked to CPU host RAM"
                    );
                } else {
                    loaded.clip.drop_weights();
                    tracing::info!(
                        on_gpu = loaded.clip.on_gpu,
                        "CLIP encoder dropped to free memory for denoising"
                    );
                }
            }
            // Force CUDA to complete the encoder cuMemFreeAsync before denoising
            // begins. Without this, the freed encoder VRAM (~5–6 GB for T5 Q8 +
            // CLIP) may not be available when the first denoising step allocates,
            // and on a tight 24 GB budget (Q8 transformer kept loaded + LoRAs)
            // that pushes VAE decode past the limit later in the pipeline.
            if dropped_gpu_encoder {
                loaded.device.synchronize()?;
            }

            Self::generate_with_embeddings(
                progress,
                req,
                &mut loaded,
                t5_emb,
                clip_emb,
                seed,
                width,
                height,
                start,
                self.base.gpu_ordinal,
            )
        })()
    }
}

impl InferenceEngine for FluxEngine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        self.base.progress.checkpoint()?;
        self.pending_placement = req.placement.clone();
        let result = self.generate_inner(req);
        self.pending_placement = None;
        result
    }

    fn model_name(&self) -> &str {
        self.base.model_name()
    }

    fn is_loaded(&self) -> bool {
        // Sequential mode is always "ready" — it loads on demand
        self.base.is_loaded()
    }

    fn load(&mut self) -> Result<()> {
        FluxEngine::load(self)
    }

    fn load_for_request(&mut self, req: &GenerateRequest) -> Result<()> {
        self.pending_placement = req.placement.clone();
        let result = FluxEngine::load(self);
        self.pending_placement = None;
        result
    }

    fn unload(&mut self) {
        self.base.unload();
        // prompt_cache holds GPU-resident T5/CLIP embedding tensors; clear so
        // the unload actually frees VRAM.
        clear_cache(&self.prompt_cache);
        // active_lora reflects the LoRA currently merged into the loaded
        // transformer. After unload there is no transformer, so clear the
        // marker — the next reload re-applies whatever is in the request.
        self.active_lora = Vec::new();
        // lora_delta_cache lives on CPU and survives park so the next reload
        // can skip the B @ A · scale recompute. It dies with the engine on Drop.
    }

    fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.base.set_on_progress(callback);
    }

    fn clear_on_progress(&mut self) {
        self.base.clear_on_progress();
    }

    fn set_cancellation_token(&mut self, token: crate::progress::InferenceCancellationToken) {
        self.base.set_cancellation_token(token);
    }

    fn clear_cancellation_token(&mut self) {
        self.base.clear_cancellation_token();
    }

    fn batch_execution_capability(&self) -> crate::BatchExecutionCapability {
        crate::batch_execution_capability_for_family("flux")
            .expect("production FLUX batch capability must be registered")
    }

    fn model_paths(&self) -> Option<&mold_core::ModelPaths> {
        Some(&self.base.paths)
    }
}

impl FluxEngine {
    #[allow(clippy::too_many_arguments)]
    fn generate_with_embeddings(
        progress: &ProgressReporter,
        req: &GenerateRequest,
        loaded: &mut LoadedFlux,
        t5_emb: candle_core::Tensor,
        clip_emb: candle_core::Tensor,
        seed: u64,
        width: usize,
        height: usize,
        start: Instant,
        gpu_ordinal: usize,
    ) -> Result<GenerateResponse> {
        // 3. Generate initial noise (F32 for quantized, gpu_dtype for BF16)
        let noise_dtype = if loaded.is_quantized {
            DType::F32
        } else {
            loaded.dtype
        };
        let latent_h = height / 16 * 2;
        let latent_w = width / 16 * 2;

        // Pre-compute timestep schedule (needed before mixing for img2img).
        let image_seq_len = (latent_h / 2) * (latent_w / 2);
        let mut timesteps = if loaded.is_schnell {
            flux::sampling::get_schedule(req.steps as usize, None)
        } else {
            flux::sampling::get_schedule(req.steps as usize, Some((image_seq_len, 0.5, 1.15)))
        };

        if req.source_image.is_some() {
            let start_index = crate::img2img::img2img_start_index(req.steps as usize, req.strength);
            timesteps = timesteps[start_index..].to_vec();
            tracing::info!(
                strength = req.strength,
                start_index,
                start_timestep = timesteps[0],
                schedule = ?timesteps,
                remaining_steps = timesteps.len().saturating_sub(1),
                "img2img: truncated schedule from strength"
            );
        }

        let (img, inpaint_ctx) = if let Some(ref source_bytes) = req.source_image {
            let start_t = timesteps[0];

            progress.stage_start("Encoding source image (VAE)");
            let encode_start = Instant::now();
            let source_tensor = crate::img_utils::decode_source_image(
                source_bytes,
                req.width,
                req.height,
                crate::img_utils::NormalizeRange::MinusOneToOne,
                &loaded.device,
                loaded.vae_dtype,
            )?;
            let encoded = loaded.vae.encode(&source_tensor)?;
            progress.phase_done(
                crate::ProgressPhase::Vae,
                "Encoding source image (VAE)",
                encode_start.elapsed(),
            );

            let noise = crate::engine::seeded_randn(
                seed,
                &[1, 16, latent_h, latent_w],
                &loaded.device,
                noise_dtype,
            )?;
            let encoded = encoded.to_dtype(noise_dtype)?;

            let inpaint_ctx = if let Some(ref mask_bytes) = req.mask_image {
                let mask = crate::img_utils::decode_mask_image(
                    mask_bytes,
                    latent_h,
                    latent_w,
                    &loaded.device,
                    noise_dtype,
                )?;
                Some(crate::img_utils::InpaintContext {
                    original_latents: encoded.clone(),
                    mask,
                    noise: noise.clone(),
                })
            } else {
                None
            };

            // latent = (1 - t) * encoded + t * noise
            let img = ((&encoded * (1.0 - start_t))? + (&noise * start_t)?)?;
            (img, inpaint_ctx)
        } else {
            let img = crate::engine::seeded_randn(
                seed,
                &[1, 16, latent_h, latent_w],
                &loaded.device,
                noise_dtype,
            )?;
            (img, None)
        };

        // Migrate parked conditioning tensors back to GPU now that the
        // transformer load + LoRA merge phase is over. `to_device` on a
        // tensor already on `loaded.device` is a no-op clone, so the
        // cache-restore path costs nothing here.
        let t5_emb = t5_emb.to_device(&loaded.device)?;
        let clip_emb = clip_emb.to_device(&loaded.device)?;
        // For quantized model, state tensors must be F32
        let (t5_emb_state, clip_emb_state, img_state) = if loaded.is_quantized {
            (
                t5_emb.to_dtype(DType::F32)?,
                clip_emb.to_dtype(DType::F32)?,
                img.to_dtype(DType::F32)?,
            )
        } else {
            (t5_emb, clip_emb, img)
        };

        // Build sampling state
        let state = flux::sampling::State::new(&t5_emb_state, &clip_emb_state, &img_state)?;
        let inpaint_ctx = inpaint_ctx
            .as_ref()
            .map(crate::img2img::pack_flux_inpaint_context)
            .transpose()?;

        let denoise_label = format!("Denoising ({} steps)", timesteps.len().saturating_sub(1));
        progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();
        tracing::info!(
            steps = timesteps.len().saturating_sub(1),
            quantized = loaded.is_quantized,
            "running denoising loop..."
        );

        // Denoise — guidance from request (0.0 for schnell, 3.5+ for dev/finetuned)
        let img = loaded
            .flux_model
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("transformer not loaded"))?
            .denoise(
                &state.img,
                &state.img_ids,
                &state.txt,
                &state.txt_ids,
                &state.vec,
                &timesteps,
                req.guidance,
                progress,
                inpaint_ctx.as_ref(),
                Some(&crate::latent_preview::LatentPreviewer::flux1(
                    height, width,
                )),
            )?;

        // 7. Unpack latent to spatial
        let img = flux::sampling::unpack(&img, height, width)?;
        progress.stage_done(&denoise_label, denoise_start.elapsed());
        tracing::info!("denoising complete, decoding VAE...");

        // Free denoising intermediates and transformer before VAE decode.
        // On discrete GPUs (CUDA), the BF16 transformer alone is ~24GB — VAE
        // decode needs that VRAM for conv2d intermediates. For Q8 (~12GB) on a
        // 24GB GPU, the transformer can stay resident; dropping forces a full
        // `gguf_lora_var_builder` rebuild on the next generation, which peaks
        // at ~95GB CPU when LoRAs are applied. `MOLD_FLUX_KEEP_TRANSFORMER=1`
        // opts into keeping it loaded across same-LoRA generations.
        drop(state);
        drop(t5_emb_state);
        drop(clip_emb_state);
        drop(img_state);
        let keep_transformer_env = crate::runtime_env::value("MOLD_FLUX_KEEP_TRANSFORMER")
            .map(|v| v == "1")
            .unwrap_or(false);

        // Even with KEEP_TRANSFORMER=1 the keep is conditional: VAE decode
        // needs a large contiguous conv2d allocation (~2–3 GB peak at 1024²,
        // ~10–12 GB at 2048²). When the kept transformer + LoRA-merged
        // tensors leave too little headroom (observed at ~3 GB free with a
        // 2-LoRA stack on a 24 GB card), the VAE alloc OOMs even though the
        // resident transformer size is identical to the no-LoRA case. The
        // next request rebuilds — that's the trade-off for not OOMing here.
        //
        // The headroom budget scales with output resolution via
        // [`activation_bytes`] instead of a fixed 5 GB magic — at 1024² the
        // budget is the FluxDit floor (~256 MB, the previous 5 GB was wildly
        // over-conservative on a busy 24 GB card with KEEP_TRANSFORMER=1)
        // while at 2048² it grows past 1 GB, catching what fixed 5 GB only
        // approximated.
        let vae_headroom_bytes = crate::device::activation_bytes(
            req.width,
            req.height,
            1,
            crate::device::dtype_bytes(loaded.dtype),
            crate::device::ActivationFamily::FluxDit,
        );
        let free_before_vae = crate::device::free_vram_bytes(gpu_ordinal).unwrap_or(0);
        let force_drop_for_headroom =
            keep_transformer_env && free_before_vae > 0 && free_before_vae < vae_headroom_bytes;

        if !keep_transformer_env || force_drop_for_headroom {
            loaded.flux_model = None;
            if force_drop_for_headroom {
                tracing::info!(
                    free_mb = free_before_vae / 1024 / 1024,
                    headroom_mb = vae_headroom_bytes / 1024 / 1024,
                    "Transformer force-dropped before VAE decode (free VRAM below \
                     resolution-scaled headroom; overrides MOLD_FLUX_KEEP_TRANSFORMER=1 \
                     for this request)"
                );
            } else {
                tracing::info!("Transformer dropped to free VRAM for VAE decode");
            }
        } else {
            tracing::info!(
                free_mb = free_before_vae / 1024 / 1024,
                "Transformer kept loaded (MOLD_FLUX_KEEP_TRANSFORMER=1)"
            );
        }
        // Force CUDA to complete pending operations and release freed memory
        // before VAE decode allocates its conv2d intermediates. cuMemFree is
        // asynchronous, so the drops above (denoising state + embeddings, plus
        // the optional transformer drop) may not have actually returned VRAM
        // to the allocator yet. Without this synchronize, VAE decode at 1024²
        // OOMs on the first conv allocation — observable on the keep-transformer
        // path even on iteration 1 (the drop path used to synchronize here, the
        // keep path didn't, which made the bug branch-specific).
        loaded.device.synchronize()?;

        // 8. Decode with VAE — cast to the VAE's actual loaded dtype (which
        // may differ from `loaded.dtype` when MOLD_VAE_DTYPE forces fp32 to
        // suppress banding; the quantized-model F32-state case is also
        // handled by this cast).
        progress.stage_start("VAE decode");
        let vae_decode_start = Instant::now();
        let img_for_vae = img.to_dtype(loaded.vae_dtype)?;
        let vae = &loaded.vae;
        let device_for_sync = loaded.device.clone();
        let img = crate::vae_tiling::decode_with_oom_fallback(
            &img_for_vae,
            |latents| vae.decode(latents).map_err(Into::into),
            || {
                if let Err(e) = device_for_sync.synchronize() {
                    tracing::warn!(
                        "FLUX (parallel) device.synchronize() after VAE OOM failed: {e}"
                    );
                }
            },
        )?;

        // 9. Convert to u8 image: clamp to [-1, 1], map to [0, 255]
        let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
        let img = img.i(0)?; // remove batch dim: [3, H, W]

        progress.phase_done(
            crate::ProgressPhase::Vae,
            "VAE decode",
            vae_decode_start.elapsed(),
        );
        tracing::info!("VAE decode complete, encoding output image...");

        // 10. Convert candle tensor to image bytes
        let output_metadata = build_output_metadata(req, seed, None);
        let image_bytes = encode_image(
            &img,
            req.resolved_output_format(),
            req.width,
            req.height,
            output_metadata.as_ref(),
        )?;

        let generation_time_ms = start.elapsed().as_millis() as u64;
        tracing::info!(generation_time_ms, seed, "generation complete");

        Ok(GenerateResponse {
            images: vec![ImageData {
                data: image_bytes,
                format: req.resolved_output_format(),
                width: req.width,
                height: req.height,
                index: 0,
            }],
            generation_time_ms,
            model: req.model.clone(),
            seed_used: seed,
            video: None,
            gpu: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        effective_loras, flux_rms_norm_scale_aliases, flux_runtime_dtype,
        flux_transformer_var_builder, park_cond_to_cpu, should_use_offload_bypass_registry,
        LoraBypassMode,
    };
    use crate::LoadStrategy;
    use candle_core::{DType, Device, Result, Tensor};
    use candle_nn::VarBuilder;
    use mold_core::{GenerateRequest, LoraWeight, ModelPaths, OutputFormat};
    use std::collections::HashMap;
    use std::path::PathBuf;

    /// `MOLD_LORA_BYPASS=on` and `=off` are the two boundaries we
    /// document. Any other value (including unset) must collapse to
    /// `Auto` so we never silently change behaviour because of
    /// stale `MOLD_*` env vars in a developer's shell.
    #[test]
    fn lora_bypass_mode_env_parsing() {
        let with_env = |val: Option<&str>| -> LoraBypassMode {
            unsafe {
                match val {
                    Some(v) => std::env::set_var("MOLD_LORA_BYPASS", v),
                    None => std::env::remove_var("MOLD_LORA_BYPASS"),
                }
            }
            let mode = LoraBypassMode::from_env();
            unsafe {
                std::env::remove_var("MOLD_LORA_BYPASS");
            }
            mode
        };
        assert_eq!(with_env(Some("on")), LoraBypassMode::On);
        assert_eq!(with_env(Some("ON")), LoraBypassMode::On);
        assert_eq!(with_env(Some("1")), LoraBypassMode::On);
        assert_eq!(with_env(Some("off")), LoraBypassMode::Off);
        assert_eq!(with_env(Some("0")), LoraBypassMode::Off);
        assert_eq!(with_env(Some("auto")), LoraBypassMode::Auto);
        assert_eq!(with_env(Some("garbage")), LoraBypassMode::Auto);
        assert_eq!(with_env(None), LoraBypassMode::Auto);
    }

    #[test]
    fn offload_lora_registry_is_built_before_adaptive_planning_when_enabled() {
        assert!(should_use_offload_bypass_registry(
            true,
            true,
            LoraBypassMode::Auto
        ));
        assert!(should_use_offload_bypass_registry(
            true,
            true,
            LoraBypassMode::On
        ));
        assert!(!should_use_offload_bypass_registry(
            true,
            true,
            LoraBypassMode::Off
        ));
        assert!(!should_use_offload_bypass_registry(
            false,
            true,
            LoraBypassMode::Auto
        ));
        assert!(!should_use_offload_bypass_registry(
            true,
            false,
            LoraBypassMode::Auto
        ));
    }

    #[test]
    fn flux_rms_norm_aliases_detect_weight_suffix_checkpoint() {
        use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};

        let dir = std::env::temp_dir().join(format!(
            "mold-flux-rms-alias-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("flux-rms-weight.safetensors");

        let data = 1.0f32.to_le_bytes();
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.diffusion_model.double_blocks.0.img_attn.norm.query_norm.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![1], &data).unwrap(),
        );
        serialize_to_file(&tensors, &None, &path).unwrap();

        let aliases = flux_rms_norm_scale_aliases(&path).unwrap();
        assert_eq!(
            aliases.get("model.diffusion_model.double_blocks.0.img_attn.norm.query_norm.scale"),
            Some(
                &"model.diffusion_model.double_blocks.0.img_attn.norm.query_norm.weight"
                    .to_string()
            )
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    fn dummy_paths(transformer: &str) -> ModelPaths {
        ModelPaths {
            transformer: PathBuf::from(transformer),
            transformer_shards: Vec::new(),
            vae: PathBuf::from("ae.safetensors"),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: Some(PathBuf::from("t5.safetensors")),
            clip_encoder: Some(PathBuf::from("clip.safetensors")),
            t5_tokenizer: Some(PathBuf::from("t5-tokenizer.json")),
            clip_tokenizer: Some(PathBuf::from("clip-tokenizer.json")),
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: Vec::new(),
            text_tokenizer: None,
            decoder: None,
        }
    }

    #[test]
    fn forced_offload_uses_sequential_generation_path_for_bf16_flux() {
        let mut engine = super::FluxEngine::new(
            "flux-dev:bf16".to_string(),
            dummy_paths("flux1-dev.safetensors"),
            Some(false),
            None,
            LoadStrategy::Eager,
            0,
            true,
            None,
        );

        assert!(engine.uses_sequential_generate_path());
    }

    #[test]
    fn forced_offload_defers_eager_load_for_bf16_flux() {
        let mut engine = super::FluxEngine::new(
            "flux-dev:bf16".to_string(),
            dummy_paths("flux1-dev.safetensors"),
            Some(false),
            None,
            LoadStrategy::Eager,
            0,
            true,
            None,
        );

        assert!(engine.defers_eager_load());
    }

    /// Minimal `GenerateRequest` carrying only the fields `effective_loras`
    /// touches (`lora`, `loras`). Every other field is set to a benign
    /// default so the tests don't drift when unrelated request shapes
    /// change.
    fn req_with_loras(
        single: Option<LoraWeight>,
        plural: Option<Vec<LoraWeight>>,
    ) -> GenerateRequest {
        GenerateRequest {
            prompt: String::new(),
            negative_prompt: None,
            model: "flux-dev".to_string(),
            width: 1024,
            height: 1024,
            steps: 4,
            guidance: 0.0,
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
            lora: single,
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
            ic_lora_control: None,
            loras: plural,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
        }
    }

    /// A slider scrubbed to zero on one of three stacked LoRAs must
    /// drop ONLY that entry from the effective stack.
    #[test]
    fn effective_loras_drops_zero_scale() {
        let req = req_with_loras(
            None,
            Some(vec![
                LoraWeight {
                    path: "p1".into(),
                    scale: 0.8,
                },
                LoraWeight {
                    path: "p2".into(),
                    scale: 0.0,
                },
                LoraWeight {
                    path: "p3".into(),
                    scale: 0.5,
                },
            ]),
        );
        let stack = effective_loras(&req);
        let paths: Vec<&str> = stack.iter().map(|w| w.path.as_str()).collect();
        assert_eq!(
            paths,
            vec!["p1", "p3"],
            "p2 (scale=0.0) must be dropped from the effective stack"
        );
        assert!((stack[0].scale - 0.8).abs() < 1e-9);
        assert!((stack[1].scale - 0.5).abs() < 1e-9);
    }

    /// Negative scales are a legitimate "anti-style" use case.
    #[test]
    fn effective_loras_keeps_negative_scales() {
        let req = req_with_loras(
            None,
            Some(vec![LoraWeight {
                path: "p1".into(),
                scale: -0.3,
            }]),
        );
        let stack = effective_loras(&req);
        assert_eq!(stack.len(), 1);
        assert!((stack[0].scale - (-0.3)).abs() < 1e-9);
    }

    /// Single `lora` form: `scale: 0.0` should be dropped too.
    #[test]
    fn effective_loras_drops_zero_scale_on_single_form() {
        let req = req_with_loras(
            Some(LoraWeight {
                path: "p1".into(),
                scale: 0.0,
            }),
            None,
        );
        assert!(effective_loras(&req).is_empty());
    }

    /// Idempotency: a tensor already on CPU comes back on CPU and equals the
    /// input — `park_cond_to_cpu` must not pay for a redundant copy on the
    /// GGUF / Q8 path where T5 already produces CPU tensors.
    #[test]
    fn park_cond_to_cpu_is_idempotent_for_cpu_tensors() {
        let cpu_tensor = Tensor::zeros((2, 4), DType::F32, &Device::Cpu).unwrap();
        let parked = park_cond_to_cpu(&cpu_tensor).unwrap();
        assert!(parked.device().is_cpu(), "CPU input must stay on CPU");
        assert_eq!(parked.shape(), cpu_tensor.shape());
    }

    /// `park_cond_to_cpu` output is on CPU regardless of input device.
    #[test]
    fn park_cond_to_cpu_returns_cpu_tensor_for_any_input() {
        let input = Tensor::ones((1, 3), DType::F32, &Device::Cpu).unwrap();
        let parked = park_cond_to_cpu(&input).unwrap();
        assert!(parked.device().is_cpu(), "output must be on CPU");
        assert_eq!(parked.shape(), input.shape());
        assert_eq!(parked.dtype(), input.dtype());
    }

    #[test]
    fn flux_var_builder_uses_root_tensors_when_present() -> Result<()> {
        let tensors = HashMap::from([(
            "img_in.weight".to_string(),
            Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?,
        )]);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        let resolved = flux_transformer_var_builder(vb);

        assert!(resolved.contains_tensor("img_in.weight"));
        assert_eq!(resolved.prefix(), "");
        Ok(())
    }

    #[test]
    fn flux_var_builder_uses_model_diffusion_model_prefix_when_present() -> Result<()> {
        let tensors = HashMap::from([(
            "model.diffusion_model.img_in.weight".to_string(),
            Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?,
        )]);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        let resolved = flux_transformer_var_builder(vb);

        assert!(resolved.contains_tensor("img_in.weight"));
        assert_eq!(resolved.prefix(), "model.diffusion_model");
        Ok(())
    }

    #[test]
    fn flux_runtime_dtype_prefers_f16_for_cuda_fp8_safetensors() {
        assert_eq!(flux_runtime_dtype(true, false, true), DType::F16);
        assert_eq!(flux_runtime_dtype(true, false, false), DType::BF16);
        assert_eq!(flux_runtime_dtype(false, false, true), DType::F32);
    }

    #[test]
    fn flux_runtime_dtype_quantized_matches_gpu_policy() {
        assert_eq!(flux_runtime_dtype(true, true, false), DType::BF16);
        assert_eq!(flux_runtime_dtype(false, true, false), DType::F32);
        assert_eq!(flux_runtime_dtype(true, true, true), DType::BF16);
        assert_eq!(flux_runtime_dtype(false, true, true), DType::F32);
    }

    #[test]
    fn fp8_cache_path_includes_file_size() {
        // Create a temp file with known size to test cache path generation
        let dir = std::env::temp_dir().join(format!("mold-cache-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fp8_file = dir.join("transformer.safetensors");
        std::fs::write(&fp8_file, vec![0u8; 1024]).unwrap();

        let cache_path = super::fp8_gguf_cache_path(&fp8_file);
        let filename = cache_path.file_name().unwrap().to_str().unwrap();

        // Should contain the file stem and the size
        assert!(
            filename.contains("transformer"),
            "should contain stem: {filename}"
        );
        assert!(
            filename.contains("1024"),
            "should contain file size: {filename}"
        );
        assert!(
            filename.ends_with(".q8_0.gguf"),
            "should end with .q8_0.gguf: {filename}"
        );

        // Different size → different cache path
        std::fs::write(&fp8_file, vec![0u8; 2048]).unwrap();
        let cache_path2 = super::fp8_gguf_cache_path(&fp8_file);
        assert_ne!(
            cache_path, cache_path2,
            "different file sizes should produce different cache paths"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn fp8_q8_cache_quantizes_only_block_aligned_last_dim() {
        assert!(super::q8_0_can_quantize_dims(&[3072, 3072]));
        assert!(super::q8_0_can_quantize_dims(&[1, 64]));
        assert!(
            !super::q8_0_can_quantize_dims(&[256, 256, 3, 3]),
            "conv kernels have total elements divisible by 32, but Q8_0 \
             requires the last dimension itself to be block-aligned"
        );
        assert!(!super::q8_0_can_quantize_dims(&[512, 512, 1, 1]));
        assert!(!super::q8_0_can_quantize_dims(&[3072]));
    }

    #[test]
    fn fp8_q8_cache_skips_bundled_text_encoder_and_scalar_tensors() {
        assert!(super::fp8_cache_should_skip_tensor(
            "text_encoders.clip_l.logit_scale",
            &[]
        ));
        assert!(super::fp8_cache_should_skip_tensor(
            "text_encoders.t5xxl.encoder.block.0.layer.0.SelfAttention.q.weight",
            &[4096, 4096]
        ));
        assert!(super::fp8_cache_should_skip_tensor("some.scalar", &[]));
        assert!(!super::fp8_cache_should_skip_tensor(
            "double_blocks.0.img_attn.qkv.weight",
            &[9216, 3072]
        ));
    }

    #[test]
    fn fp8_cache_path_lives_under_cache_flux_q8() {
        let path = std::path::Path::new("/some/model/my-model.safetensors");
        // File doesn't exist so size will be 0
        let cache_path = super::fp8_gguf_cache_path(path);
        let cache_str = cache_path.to_str().unwrap();
        assert!(
            cache_str.contains("cache/flux-q8"),
            "cache should be under cache/flux-q8: {cache_str}"
        );
    }

    #[test]
    fn fp8_cache_temp_paths_are_unique_per_writer() {
        let cache_path =
            std::path::Path::new("/tmp/agfluxSchnell_realistic23-1234-deadbeef.q8_0.gguf");

        let first = super::fp8_gguf_tmp_path(cache_path);
        let second = super::fp8_gguf_tmp_path(cache_path);

        assert_ne!(first, second);
        assert_ne!(first, cache_path);
        assert_ne!(second, cache_path);
    }

    #[test]
    fn detects_schnell_from_uppercase_filename() {
        let engine = super::FluxEngine::new(
            "cv:1153358".to_string(),
            dummy_paths("agfluxSchnell_realistic23.safetensors"),
            None,
            None,
            LoadStrategy::Sequential,
            0,
            false,
            None,
        );

        assert!(engine.detect_is_schnell());
    }

    #[test]
    fn flux_vae_var_builder_accepts_vae_prefix() {
        use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
        use std::collections::HashMap;

        let dir = std::env::temp_dir().join(format!(
            "mold-flux-vae-prefix-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("vae-prefix.safetensors");

        let data = vec![0u8; 128 * 3 * 3 * 3 * std::mem::size_of::<f32>()];
        let shape = vec![128, 3, 3, 3];
        let view = TensorView::new(SafeDtype::F32, shape, &data).unwrap();
        let mut tensors = HashMap::new();
        tensors.insert("vae.encoder.conv_in.weight".to_string(), view);
        serialize_to_file(&tensors, &None, &path).unwrap();

        let vb = crate::weight_loader::load_safetensors_with_progress(
            std::slice::from_ref(&path),
            DType::F32,
            &Device::Cpu,
            "test VAE",
            &crate::progress::ProgressReporter::default(),
        )
        .unwrap();
        let vb = super::flux_vae_var_builder(vb);

        assert!(vb.contains_tensor("encoder.conv_in.weight"));

        std::fs::remove_dir_all(&dir).ok();
    }

    // ── Embedding patching tests ────────────────────────────────────────

    /// Helper: write a minimal GGUF file containing the given tensor names.
    /// Each tensor is a tiny 1-element F32 QTensor.
    fn write_test_gguf(path: &std::path::Path, tensor_names: &[&str]) {
        let device = Device::Cpu;
        let qtensors: Vec<(String, candle_core::quantized::QTensor)> = tensor_names
            .iter()
            .map(|name| {
                let t = Tensor::zeros(1, DType::F32, &device).unwrap();
                let qt = candle_core::quantized::QTensor::quantize(
                    &t,
                    candle_core::quantized::GgmlDType::F32,
                )
                .unwrap();
                (name.to_string(), qt)
            })
            .collect();
        let refs: Vec<(&str, &candle_core::quantized::QTensor)> =
            qtensors.iter().map(|(n, q)| (n.as_str(), q)).collect();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = std::io::BufWriter::new(file);
        candle_core::quantized::gguf_file::write(&mut writer, &[], &refs).unwrap();
    }

    #[test]
    fn gguf_has_embeddings_true_for_complete() {
        let dir =
            std::env::temp_dir().join(format!("mold-emb-test-complete-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("complete.gguf");
        write_test_gguf(
            &path,
            &[
                "img_in.weight",
                "img_in.bias",
                "double_blocks.0.img_mod.lin.weight",
            ],
        );
        assert!(super::gguf_has_embeddings(&path).unwrap());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn gguf_has_embeddings_false_for_incomplete() {
        let dir =
            std::env::temp_dir().join(format!("mold-emb-test-incomplete-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("incomplete.gguf");
        write_test_gguf(
            &path,
            &[
                "double_blocks.0.img_mod.lin.weight",
                "single_blocks.0.linear1.weight",
                "txt_in.weight",
            ],
        );
        assert!(!super::gguf_has_embeddings(&path).unwrap());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn embedding_patched_cache_path_format() {
        let dir = std::env::temp_dir().join(format!("mold-emb-cache-fmt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let gguf_file = dir.join("ultrareal.gguf");
        std::fs::write(&gguf_file, vec![0u8; 512]).unwrap();

        let cache_path = super::embedding_patched_cache_path(&gguf_file);
        let cache_str = cache_path.to_str().unwrap();
        assert!(
            cache_str.contains("cache/flux-embeddings"),
            "should be under cache/flux-embeddings: {cache_str}"
        );
        let filename = cache_path.file_name().unwrap().to_str().unwrap();
        assert!(
            filename.contains("ultrareal"),
            "should contain stem: {filename}"
        );
        assert!(
            filename.contains("512"),
            "should contain file size: {filename}"
        );
        assert!(
            filename.ends_with(".patched.gguf"),
            "should end with .patched.gguf: {filename}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn ensure_gguf_embeddings_noop_for_complete() {
        let dir = std::env::temp_dir().join(format!("mold-emb-noop-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("complete.gguf");

        // Write a GGUF with img_in.weight present
        write_test_gguf(
            &path,
            &["img_in.weight", "double_blocks.0.img_mod.lin.weight"],
        );

        let progress = crate::progress::ProgressReporter::default();
        let result = super::ensure_gguf_embeddings(&path, false, &progress, None).unwrap();

        // Should return the original path unchanged
        assert_eq!(result, path);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn ensure_gguf_embeddings_patches_incomplete_with_reference() {
        // Test the full patching flow using a synthetic reference GGUF.
        // Uses models_dir_override to avoid mutating process-global env vars.
        let dir = std::env::temp_dir().join(format!("mold-emb-patch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Create an incomplete GGUF (city96 format — only diffusion blocks)
        let incomplete_path = dir.join("ultrareal-test.gguf");
        write_test_gguf(
            &incomplete_path,
            &[
                "double_blocks.0.img_mod.lin.weight",
                "single_blocks.0.linear1.weight",
                "txt_in.weight",
                "txt_in.bias",
                "final_layer.linear.weight",
            ],
        );

        // Create a fake reference model at the expected manifest path.
        // flux-dev:q8 transformer lives at <models_dir>/flux-dev-q8/flux1-dev-Q8_0.gguf
        let models_dir = dir.join("models");
        let ref_model_dir = models_dir.join("flux-dev-q8");
        std::fs::create_dir_all(&ref_model_dir).unwrap();
        let ref_path = ref_model_dir.join("flux1-dev-Q8_0.gguf");

        // The reference GGUF has all embedding tensors
        let mut all_tensors: Vec<&str> = super::FLUX_EMBEDDING_TENSORS.to_vec();
        all_tensors.extend_from_slice(super::FLUX_GUIDANCE_EMBEDDING_TENSORS);
        all_tensors.extend_from_slice(&[
            "double_blocks.0.img_mod.lin.weight",
            "txt_in.weight",
            "txt_in.bias",
        ]);
        write_test_gguf(&ref_path, &all_tensors);

        let progress = crate::progress::ProgressReporter::default();
        let result =
            super::ensure_gguf_embeddings(&incomplete_path, false, &progress, Some(&models_dir));

        let patched_path = result.unwrap();
        assert_ne!(
            patched_path, incomplete_path,
            "should return a different cached path"
        );
        assert!(patched_path.exists(), "patched GGUF should exist on disk");
        assert!(
            patched_path.to_str().unwrap().contains("flux-embeddings"),
            "patched file should be in flux-embeddings cache"
        );

        // Verify the patched file contains the embedding tensors
        assert!(
            super::gguf_has_embeddings(&patched_path).unwrap(),
            "patched GGUF should have embeddings"
        );

        // Clean up
        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_file(&patched_path).ok();
        let _ = std::fs::remove_dir(patched_path.parent().unwrap());
    }

    #[test]
    fn ensure_gguf_embeddings_cache_is_reused() {
        // If a cache file already exists, it should be returned directly
        let dir = std::env::temp_dir().join(format!("mold-emb-reuse-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let incomplete_path = dir.join("model-for-cache.gguf");
        write_test_gguf(&incomplete_path, &["double_blocks.0.img_mod.lin.weight"]);

        // Pre-create the cache file
        let cache_path = super::embedding_patched_cache_path(&incomplete_path);
        std::fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
        write_test_gguf(
            &cache_path,
            &["img_in.weight", "double_blocks.0.img_mod.lin.weight"],
        );

        let progress = crate::progress::ProgressReporter::default();
        let result =
            super::ensure_gguf_embeddings(&incomplete_path, true, &progress, None).unwrap();

        assert_eq!(result, cache_path, "should return cached file");

        // Clean up
        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_file(&cache_path).ok();
        // Try to clean up cache parent dir (may fail if other tests use it)
        let _ = std::fs::remove_dir(cache_path.parent().unwrap());
    }

    #[test]
    fn find_flux_reference_skips_schnell_when_dev_needed() {
        // Regression: if only flux-schnell is downloaded, a dev-family target
        // (e.g. ultrareal-v4:q8) would previously pick schnell as reference and
        // then fail mid-patch because schnell lacks guidance_in.
        let dir = std::env::temp_dir().join(format!(
            "mold-ref-picker-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let models_dir = dir.join("models");
        let schnell_dir = models_dir.join("flux-schnell-q8");
        std::fs::create_dir_all(&schnell_dir).unwrap();
        let schnell_path = schnell_dir.join("flux1-schnell-Q8_0.gguf");

        // Schnell has img_in but not guidance_in — mirrors the real city96 schnell GGUF
        let mut schnell_tensors: Vec<&str> = super::FLUX_EMBEDDING_TENSORS.to_vec();
        schnell_tensors.push("double_blocks.0.img_mod.lin.weight");
        write_test_gguf(&schnell_path, &schnell_tensors);

        // needs_guidance=true must reject the schnell-only state
        let result = super::find_flux_reference_gguf(true, Some(&models_dir));
        assert!(
            result.is_none(),
            "schnell must not be picked as reference for dev targets: got {result:?}"
        );

        // needs_guidance=false (schnell target) accepts the schnell reference
        let result = super::find_flux_reference_gguf(false, Some(&models_dir));
        assert_eq!(result.as_deref(), Some(schnell_path.as_path()));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn find_flux_reference_accepts_dev_candidate_with_guidance() {
        // Happy path for the needs_guidance branch: a dev reference that has
        // guidance_in is accepted; a dev reference lacking guidance (truncated
        // or swapped file) is rejected.
        let dir = std::env::temp_dir().join(format!(
            "mold-ref-dev-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let models_dir = dir.join("models");
        let dev_dir = models_dir.join("flux-dev-q8");
        std::fs::create_dir_all(&dev_dir).unwrap();
        let dev_path = dev_dir.join("flux1-dev-Q8_0.gguf");

        // Reference without guidance — needs_guidance=true must reject it.
        let incomplete: Vec<&str> = super::FLUX_EMBEDDING_TENSORS.to_vec();
        write_test_gguf(&dev_path, &incomplete);
        assert!(
            super::find_flux_reference_gguf(true, Some(&models_dir)).is_none(),
            "dev candidate without guidance_in must be rejected for dev targets"
        );

        // Now add guidance tensors — same path should be accepted.
        let mut complete: Vec<&str> = super::FLUX_EMBEDDING_TENSORS.to_vec();
        complete.extend_from_slice(super::FLUX_GUIDANCE_EMBEDDING_TENSORS);
        write_test_gguf(&dev_path, &complete);
        let picked = super::find_flux_reference_gguf(true, Some(&models_dir))
            .expect("complete dev reference must be accepted");
        assert_eq!(picked, dev_path);

        // Schnell target (needs_guidance=false) also accepts the dev candidate.
        let picked = super::find_flux_reference_gguf(false, Some(&models_dir))
            .expect("dev candidate satisfies schnell targets too");
        assert_eq!(picked, dev_path);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn find_flux_reference_accepts_krea_when_no_base_dev() {
        // flux-krea is a dev-family fine-tune shipped as complete GGUFs — it
        // should serve as a reference for city96-format fine-tunes (UltraReal,
        // etc.) even when the base flux-dev GGUF isn't downloaded.
        let dir = std::env::temp_dir().join(format!(
            "mold-ref-krea-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let models_dir = dir.join("models");
        let krea_dir = models_dir.join("flux-krea-q8");
        std::fs::create_dir_all(&krea_dir).unwrap();
        let krea_path = krea_dir.join("flux1-krea-dev-Q8_0.gguf");

        let mut complete: Vec<&str> = super::FLUX_EMBEDDING_TENSORS.to_vec();
        complete.extend_from_slice(super::FLUX_GUIDANCE_EMBEDDING_TENSORS);
        write_test_gguf(&krea_path, &complete);

        let picked = super::find_flux_reference_gguf(true, Some(&models_dir))
            .expect("complete flux-krea reference must be accepted for dev targets");
        assert_eq!(picked, krea_path);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn embedding_tensor_names_are_exhaustive() {
        // Verify the const arrays cover all non-diffusion-block tensors that
        // Flux::new() in quantized_model.rs expects (lines 378-416).
        // The model loads: img_in, txt_in, time_in, vector_in, guidance_in (optional),
        // double_blocks, single_blocks, final_layer, pe_embedder (computed, no tensors).
        // txt_in is present in city96 GGUFs. double/single/final are the diffusion blocks.
        // Only the embedding layers (img_in, time_in, vector_in, guidance_in) are missing.
        let all_embedding_names: Vec<&str> = super::FLUX_EMBEDDING_TENSORS
            .iter()
            .chain(super::FLUX_GUIDANCE_EMBEDDING_TENSORS.iter())
            .copied()
            .collect();

        // img_in: linear (weight + bias)
        assert!(all_embedding_names.contains(&"img_in.weight"));
        assert!(all_embedding_names.contains(&"img_in.bias"));

        // time_in: MlpEmbedder (in_layer + out_layer, each with weight + bias)
        assert!(all_embedding_names.contains(&"time_in.in_layer.weight"));
        assert!(all_embedding_names.contains(&"time_in.in_layer.bias"));
        assert!(all_embedding_names.contains(&"time_in.out_layer.weight"));
        assert!(all_embedding_names.contains(&"time_in.out_layer.bias"));

        // vector_in: MlpEmbedder
        assert!(all_embedding_names.contains(&"vector_in.in_layer.weight"));
        assert!(all_embedding_names.contains(&"vector_in.in_layer.bias"));
        assert!(all_embedding_names.contains(&"vector_in.out_layer.weight"));
        assert!(all_embedding_names.contains(&"vector_in.out_layer.bias"));

        // guidance_in: MlpEmbedder (dev only)
        assert!(all_embedding_names.contains(&"guidance_in.in_layer.weight"));
        assert!(all_embedding_names.contains(&"guidance_in.in_layer.bias"));
        assert!(all_embedding_names.contains(&"guidance_in.out_layer.weight"));
        assert!(all_embedding_names.contains(&"guidance_in.out_layer.bias"));

        // Total: 14 tensors
        assert_eq!(all_embedding_names.len(), 14);
    }
}
