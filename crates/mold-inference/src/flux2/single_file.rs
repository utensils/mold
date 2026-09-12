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

/// The safetensors header's own ceiling, mirrored from the `safetensors`
/// crate's `MAX_HEADER_SIZE`.
const MAX_SAFETENSORS_HEADER_BYTES: u64 = 100_000_000;

/// Read the JSON header of a safetensors file, refusing a length the file
/// cannot possibly hold.
///
/// The leading `u64` is accident-controlled: a truncated download, a sparse
/// placeholder, or any non-safetensors file with a `.safetensors` name yields
/// an arbitrary length, and allocating it unchecked is an ABORT, not an error
/// — `handle_alloc_error` takes the whole process down, and this probe runs on
/// the coordinator's admission path where every planning pass reaches it.
/// Bounded by the file's own length and by the format's own ceiling, so a
/// malformed header is a `DetectError` the caller already handles by falling
/// back to the model-name heuristic.
fn read_safetensors_header(path: &Path) -> Result<BTreeMap<String, Value>, DetectError> {
    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf);
    if header_len > MAX_SAFETENSORS_HEADER_BYTES || header_len > file_len.saturating_sub(8) {
        return Err(DetectError::Header(format!(
            "declared header length {header_len} does not fit a {file_len} byte file"
        )));
    }
    let mut header_buf = vec![0u8; header_len as usize];
    file.read_exact(&mut header_buf)?;
    serde_json::from_slice(&header_buf).map_err(|e| DetectError::Header(e.to_string()))
}

/// Read just the safetensors header, returning every tensor key except the
/// reserved `__metadata__` entry. Does not touch tensor data.
fn read_tensor_keys(path: &Path) -> Result<Vec<String>, DetectError> {
    Ok(read_safetensors_header(path)?
        .into_keys()
        .filter(|k| k != "__metadata__")
        .collect())
}

/// Header-peek the checkpoint to determine its `hidden_size` (= the
/// `Flux2Config` variant: 3072 → Klein-4B, 4096 → Klein-9B, 6144 → Dev).
///
/// Reads the shape of the first weight tensor whose first dim is the
/// transformer's `hidden_size`. Probes `<prefix>img_in.weight` (first dim
/// is `hidden_size`) where `<prefix>` is `model.diffusion_model.` for
/// BFL-native exports and `""` for community root-level layouts.
///
/// Returns `Ok(None)` if neither marker is present (caller falls back to
/// a default config or model-name heuristic). Touches only the JSON header.
pub fn detect_hidden_size(path: &Path) -> Result<Option<usize>, DetectError> {
    let header = read_safetensors_header(path)?;

    let first_dim = |key: &str| -> Option<usize> {
        let info = header.get(key)?;
        let shape = info.get("shape")?.as_array()?;
        shape.first()?.as_u64().map(|n| n as usize)
    };

    for prefix in ["model.diffusion_model.", ""] {
        if let Some(d) = first_dim(&format!("{prefix}img_in.weight")) {
            return Ok(Some(d));
        }
        if let Some(d) = first_dim(&format!("{prefix}x_embedder.weight")) {
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

    /// A header length the file cannot hold is an ERROR, never an allocation.
    ///
    /// `detect_hidden_size` runs on the coordinator for every FLUX.2 plan, and
    /// a sparse placeholder or a truncated download makes the leading `u64`
    /// arbitrary. Allocating it calls `handle_alloc_error`, which ABORTS the
    /// process — a whole server killed by one bad file on disk. The zero-filled
    /// case below is the one a sparse test fixture produces; the huge-length
    /// case is what a truncated real checkpoint produces.
    #[test]
    fn a_header_length_the_file_cannot_hold_is_refused_rather_than_allocated() {
        use std::io::{Seek, SeekFrom, Write};

        let zeroed = temp_path("zeroed");
        let mut file = File::create(&zeroed).unwrap();
        file.seek(SeekFrom::Start(64 * 1024 - 1)).unwrap();
        file.write_all(&[0]).unwrap();
        drop(file);
        assert!(matches!(
            detect_hidden_size(&zeroed),
            Err(DetectError::Header(_))
        ));
        assert!(matches!(
            detect_format(&zeroed),
            Err(DetectError::Header(_)) | Ok(Flux2SingleFileFormat::Unknown)
        ));

        let truncated = temp_path("truncated");
        let mut file = File::create(&truncated).unwrap();
        file.write_all(&u64::MAX.to_le_bytes()).unwrap();
        file.write_all(b"{}").unwrap();
        drop(file);
        assert!(matches!(
            detect_hidden_size(&truncated),
            Err(DetectError::Header(_))
        ));

        let _ = std::fs::remove_file(&zeroed);
        let _ = std::fs::remove_file(&truncated);
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

    #[test]
    fn flux2_detect_hidden_size_recognizes_dev_diffusers_input() {
        let p = temp_path("dev-hidden-size");
        let bytes = vec![0u8; 6_144 * 128 * 2];
        let mut tensors = HashMap::new();
        tensors.insert(
            "x_embedder.weight".to_string(),
            TensorView::new(SafeDtype::BF16, vec![6_144, 128], &bytes).unwrap(),
        );
        serialize_to_file(&tensors, &None, &p).unwrap();

        assert_eq!(detect_hidden_size(&p).unwrap(), Some(6_144));
        let _ = std::fs::remove_file(p);
    }
}
