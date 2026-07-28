//! Header-only validator for Flux.2 single-file safetensors checkpoints.
//!
//! Civitai (and ComfyUI export) Flux.2 fine-tunes ship as a single
//! `.safetensors` whose tensor keys are BFL-native (every key prefixed
//! `model.diffusion_model.`). The diffusers naming the in-tree
//! `Flux2Transformer::new` expects (`x_embedder.weight`, …) is rewritten
//! by `SingleFileBackend::from_flux2_singlefile`.
//!
//! Some uploads also ship NVFP4-quantised weights with extra
//! `*.weight_scale_2` / `*.comfy_quant` markers. Those
//! route through `SingleFileBackend` synthetic `weight.nvfp4_*` subkeys and
//! `Flux2Linear::Nvfp4Streaming` instead of the normal BF16/FP16/FP8 weight
//! lookup.
//!
//! Reads only the safetensors JSON header; tensor data is never touched.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use serde_json::Value;
use thiserror::Error;

/// Detected layout of a Flux.2 single-file safetensors checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flux2SingleFileFormat {
    /// BFL-native single-file with `model.diffusion_model.*` prefix
    /// (typical Civitai/ComfyUI export of Klein/Dev fine-tunes).
    /// Loadable via `SingleFileBackend::from_flux2_singlefile` after key remap.
    BflNative,
    /// BFL-native single-file with **no** `model.diffusion_model.*` prefix
    /// — keys live at the root (`img_in.weight`, `double_blocks.0.*`, …).
    /// Many community FP8 conversions ship this layout. Same remap, just
    /// without the prefix on the source side.
    BflNativeRoot,
    /// Diffusers-style root keys (`x_embedder.*`, `transformer_blocks.*`).
    /// The standard `Flux2Transformer::new` path handles this layout
    /// directly — rare for single-file uploads.
    Diffusers,
    /// NVFP4-quantised single-file. Detected via `*.weight_scale_2` or
    /// `*.comfy_quant` markers in the header. Loadable
    /// through the portable streaming NVFP4 path.
    Nvfp4,
    /// No recognisable Flux.2 signature in the header.
    Unknown,
}

#[derive(Debug, Error)]
pub enum DetectError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("safetensors header parse failed: {0}")]
    Header(String),
}

/// Inspect the safetensors header at `path` and report the detected layout.
pub fn detect_format(path: &Path) -> Result<Flux2SingleFileFormat, DetectError> {
    let keys = read_tensor_keys(path)?;

    // NVFP4 markers must be checked BEFORE BFL-native (an NVFP4 export
    // still uses `model.diffusion_model.*` keys for the underlying tensors).
    if keys.iter().any(|k| is_nvfp4_marker(k)) {
        return Ok(Flux2SingleFileFormat::Nvfp4);
    }

    // BFL-native with `model.diffusion_model.` prefix (Civitai NVFP4 / ComfyUI export).
    if keys.iter().any(|k| k.starts_with(BFL_NATIVE_PREFIX)) {
        return Ok(Flux2SingleFileFormat::BflNative);
    }

    // Diffusers: root-level Flux.2 markers — checked BEFORE BflNativeRoot
    // because the diffusers `single_transformer_blocks.*` shape is
    // distinguishable from BFL-native `single_blocks.*`.
    if keys.iter().any(|k| is_diffusers_marker(k)) {
        return Ok(Flux2SingleFileFormat::Diffusers);
    }

    // BFL-native at the root (no prefix) — common for community FP8 conversions
    // that strip the wrapping namespace before re-export.
    if keys.iter().any(|k| is_bfl_native_root_marker(k)) {
        return Ok(Flux2SingleFileFormat::BflNativeRoot);
    }

    Ok(Flux2SingleFileFormat::Unknown)
}

const BFL_NATIVE_PREFIX: &str = "model.diffusion_model.";

fn is_nvfp4_marker(key: &str) -> bool {
    key.ends_with(".weight_scale_2") || key.ends_with(".comfy_quant")
}

fn is_diffusers_marker(key: &str) -> bool {
    key == "x_embedder.weight"
        || key == "context_embedder.weight"
        || key.starts_with("transformer_blocks.")
        || key.starts_with("single_transformer_blocks.")
}

fn is_bfl_native_root_marker(key: &str) -> bool {
    key == "img_in.weight"
        || key == "txt_in.weight"
        || key.starts_with("double_blocks.")
        || key.starts_with("single_blocks.")
        || key.starts_with("final_layer.")
}

/// Read just the safetensors header, returning every tensor key except the
/// reserved `__metadata__` entry. Does not touch tensor data.
fn read_tensor_keys(path: &Path) -> Result<Vec<String>, DetectError> {
    let mut file = File::open(path)?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;
    let header: BTreeMap<String, Value> =
        serde_json::from_slice(&header_buf).map_err(|e| DetectError::Header(e.to_string()))?;
    Ok(header.into_keys().filter(|k| k != "__metadata__").collect())
}

/// Header-peek the checkpoint to determine its `hidden_size` (= the
/// `Flux2Config` variant: 3072 → klein-4B, 4096 → klein-9B).
///
/// Reads the shape of the first weight tensor whose first dim is the
/// transformer's `hidden_size`. Probes `<prefix>img_in.weight` (first dim
/// is `hidden_size`) where `<prefix>` is `model.diffusion_model.` for
/// BFL-native exports and `""` for community root-level layouts.
///
/// Returns `Ok(None)` if neither marker is present (caller falls back to
/// a default config or model-name heuristic). Touches only the JSON header.
pub fn detect_hidden_size(path: &Path) -> Result<Option<usize>, DetectError> {
    let mut file = File::open(path)?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;
    let header: BTreeMap<String, Value> =
        serde_json::from_slice(&header_buf).map_err(|e| DetectError::Header(e.to_string()))?;

    let first_dim = |key: &str| -> Option<usize> {
        let info = header.get(key)?;
        let shape = info.get("shape")?.as_array()?;
        shape.first()?.as_u64().map(|n| n as usize)
    };

    for prefix in ["model.diffusion_model.", ""] {
        if let Some(d) = first_dim(&format!("{prefix}img_in.weight")) {
            return Ok(Some(d));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn temp_path(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "mold-flux2-detect-{}-{}-{}.safetensors",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        p
    }

    fn write_fixture(path: &Path, keys: &[&str]) {
        let zero = 0.0f32.to_le_bytes().to_vec();
        let bufs: Vec<Vec<u8>> = keys.iter().map(|_| zero.clone()).collect();
        let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
        for (key, buf) in keys.iter().zip(bufs.iter()) {
            tensors.insert(
                (*key).to_string(),
                TensorView::new(SafeDtype::F32, vec![1], buf).unwrap(),
            );
        }
        serialize_to_file(&tensors, &None, path).unwrap();
    }

    #[test]
    fn flux2_detect_format_recognizes_bfl_native() {
        let p = temp_path("bfl");
        write_fixture(&p, &["model.diffusion_model.img_in.weight"]);
        assert_eq!(detect_format(&p).unwrap(), Flux2SingleFileFormat::BflNative);
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn flux2_detect_format_recognizes_nvfp4() {
        let p = temp_path("nvfp4");
        write_fixture(
            &p,
            &[
                "model.diffusion_model.double_blocks.0.img_attn.qkv.weight",
                "model.diffusion_model.double_blocks.0.img_attn.qkv.weight_scale_2",
            ],
        );
        assert_eq!(detect_format(&p).unwrap(), Flux2SingleFileFormat::Nvfp4);
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn flux2_input_scale_without_nvfp4_marker_remains_bfl_fp8() {
        let p = temp_path("fp8-input-scale");
        write_fixture(
            &p,
            &[
                "model.diffusion_model.double_blocks.0.img_attn.qkv.weight",
                "model.diffusion_model.double_blocks.0.img_attn.qkv.input_scale",
            ],
        );
        assert_eq!(detect_format(&p).unwrap(), Flux2SingleFileFormat::BflNative);
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn flux2_detect_format_recognizes_diffusers() {
        let p = temp_path("diffusers");
        write_fixture(
            &p,
            &["x_embedder.weight", "transformer_blocks.0.attn.to_q.weight"],
        );
        assert_eq!(detect_format(&p).unwrap(), Flux2SingleFileFormat::Diffusers);
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn flux2_detect_format_recognizes_bfl_native_root_no_prefix() {
        // Community FP8 conversions strip `model.diffusion_model.` and ship
        // BFL keys at the root. Detect them so the loader can use the same
        // remap with an empty prefix instead of failing as Unknown.
        let p = temp_path("bfl-root");
        write_fixture(
            &p,
            &[
                "img_in.weight",
                "double_blocks.0.img_attn.qkv.weight",
                "single_blocks.0.linear1.weight",
            ],
        );
        assert_eq!(
            detect_format(&p).unwrap(),
            Flux2SingleFileFormat::BflNativeRoot
        );
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn flux2_detect_format_unknown_when_no_markers() {
        let p = temp_path("unknown");
        write_fixture(&p, &["some.other.weight"]);
        assert_eq!(detect_format(&p).unwrap(), Flux2SingleFileFormat::Unknown);
        let _ = std::fs::remove_file(p);
    }
}
