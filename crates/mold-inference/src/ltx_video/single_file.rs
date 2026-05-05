//! Single-file validator for LTX-Video (LTXV) Civitai checkpoints (phase 5).
//!
//! Reads only the safetensors header to detect the tensor key layout and
//! partition keys into transformer and VAE buckets. No tensor data is
//! materialised.
//!
//! Two transformer key formats are accepted:
//!
//! - **Native** (no prefix): keys like `transformer_blocks.0.attn1.to_q.weight`.
//!   The engine applies `remap_official_ltx_transformer_key` at load time.
//!
//! - **Diffusers** (prefixed): keys like
//!   `model.diffusion_model.transformer_blocks.0.attn1.to_q.weight`.
//!   No remap needed.
//!
//! VAE presence is detected by any key starting with `vae.`. Combined
//! checkpoints (transformer + VAE) set `has_vae = true` and the engine
//! loads the VAE under `vb.pp("vae")`. Transformer-only checkpoints
//! (`has_vae = false`) require the `ltx-video-vae` companion to be
//! present on disk; the factory validates this before constructing the
//! engine.
//!
//! LTX-2 uses `blocks.*` instead of `transformer_blocks.*` — pass those
//! checkpoints to `ltx_2::single_file::load` instead.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use serde_json::Value;
use thiserror::Error;

/// Transformer key layout in the checkpoint file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LtxKeyFormat {
    /// Native LTX format — keys lack the `model.diffusion_model.` prefix
    /// (e.g., `transformer_blocks.0.attn1.to_q.weight`). Requires
    /// `remap_official_ltx_transformer_key` at engine load time.
    Native,
    /// Diffusers format — keys already carry the `model.diffusion_model.`
    /// prefix. No remap needed.
    Diffusers,
}

/// Result of header-parsing an LTX-Video single-file safetensors.
#[derive(Debug, Clone)]
pub struct LtxVideoSingleFileBundle {
    pub format: LtxKeyFormat,
    /// Number of transformer (`transformer_blocks.*`) keys found.
    /// Used in unit tests to verify that non-transformer keys are not counted.
    #[allow(dead_code)]
    pub transformer_key_count: usize,
    /// `true` when the checkpoint contains `vae.*` keys (combined
    /// transformer+VAE layout). `false` for transformer-only checkpoints
    /// (require companion VAE).
    pub has_vae: bool,
}

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("safetensors header parse failed: {0}")]
    Header(String),
    /// No `transformer_blocks.*` or `model.diffusion_model.transformer_blocks.*`
    /// keys found. The file is probably not an LTX-Video checkpoint.
    #[error(
        "no LTX-Video transformer keys found \
         (expected `transformer_blocks.*` or `model.diffusion_model.transformer_blocks.*`)"
    )]
    NoTransformerKeys,
}

/// Header-parse the safetensors at `path` and return the detected layout.
///
/// Only reads the 8-byte length prefix + the JSON header — tensor data on
/// disk is never touched.
pub fn load(path: &Path) -> Result<LtxVideoSingleFileBundle, LoadError> {
    let keys = read_tensor_keys(path)?;

    let mut native_count = 0usize;
    let mut diffusers_count = 0usize;
    let mut vae_count = 0usize;

    for key in &keys {
        if has_prefix(key, NATIVE_TRANSFORMER_KEY) {
            native_count += 1;
        } else if has_prefix(key, DIFFUSERS_TRANSFORMER_KEY) {
            diffusers_count += 1;
        } else if has_prefix(key, VAE_KEY) {
            vae_count += 1;
        }
    }

    let (format, transformer_key_count) = if diffusers_count > 0 {
        (LtxKeyFormat::Diffusers, diffusers_count)
    } else if native_count > 0 {
        (LtxKeyFormat::Native, native_count)
    } else {
        return Err(LoadError::NoTransformerKeys);
    };

    Ok(LtxVideoSingleFileBundle {
        format,
        transformer_key_count,
        has_vae: vae_count > 0,
    })
}

// `transformer_blocks.*` — native LTX-Video transformer key prefix.
const NATIVE_TRANSFORMER_KEY: &str = "transformer_blocks";
// `model.diffusion_model.transformer_blocks.*` — diffusers-style prefix.
const DIFFUSERS_TRANSFORMER_KEY: &str = "model.diffusion_model.transformer_blocks";
// `vae.*` — VAE present in the combined checkpoint.
const VAE_KEY: &str = "vae";

/// `true` iff `key == prefix` or `key` starts with `"<prefix>."`.
/// Prevents `model.diffusion_model_extras.foo` from matching
/// `model.diffusion_model`.
fn has_prefix(key: &str, prefix: &str) -> bool {
    if key.len() < prefix.len() {
        return false;
    }
    if key == prefix {
        return true;
    }
    key.as_bytes().get(prefix.len()) == Some(&b'.') && key.starts_with(prefix)
}

/// Read just the safetensors header, returning all tensor key names
/// except `__metadata__`. Does not mmap or read tensor data.
fn read_tensor_keys(path: &Path) -> Result<Vec<String>, LoadError> {
    let mut file = File::open(path)?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;
    let header: BTreeMap<String, Value> =
        serde_json::from_slice(&header_buf).map_err(|e| LoadError::Header(e.to_string()))?;
    Ok(header.into_keys().filter(|k| k != "__metadata__").collect())
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
            "mold-ltxvideo-sf-{}-{}-{}.safetensors",
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
                key.to_string(),
                TensorView::new(SafeDtype::F32, vec![1], buf).unwrap(),
            );
        }
        serialize_to_file(&tensors, &None, path).unwrap();
    }

    #[test]
    fn native_transformer_only_checkpoint() {
        let p = temp_path("native-no-vae");
        write_fixture(
            &p,
            &[
                "transformer_blocks.0.attn1.to_q.weight",
                "transformer_blocks.0.attn1.to_k.weight",
                "proj_in.weight",
                "caption_projection.linear_1.weight",
            ],
        );

        let bundle = load(&p).expect("native transformer-only load");
        assert_eq!(bundle.format, LtxKeyFormat::Native);
        assert_eq!(bundle.transformer_key_count, 2); // only transformer_blocks.* counted
        assert!(!bundle.has_vae);

        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn native_combined_transformer_and_vae() {
        let p = temp_path("native-with-vae");
        write_fixture(
            &p,
            &[
                "transformer_blocks.0.attn1.to_q.weight",
                "proj_in.weight",
                "vae.encoder.conv_in.weight",
                "vae.decoder.conv_out.weight",
            ],
        );

        let bundle = load(&p).expect("native combined load");
        assert_eq!(bundle.format, LtxKeyFormat::Native);
        assert!(bundle.has_vae);

        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn diffusers_format_detected() {
        let p = temp_path("diffusers");
        write_fixture(
            &p,
            &[
                "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight",
                "model.diffusion_model.patchify_proj.weight",
            ],
        );

        let bundle = load(&p).expect("diffusers load");
        assert_eq!(bundle.format, LtxKeyFormat::Diffusers);
        assert_eq!(bundle.transformer_key_count, 1); // only transformer_blocks.* counted
        assert!(!bundle.has_vae);

        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn diffusers_takes_precedence_over_native() {
        // Pathological checkpoint that has both prefixes — diffusers wins.
        let p = temp_path("both");
        write_fixture(
            &p,
            &[
                "model.diffusion_model.transformer_blocks.0.to_q.weight",
                "transformer_blocks.0.to_q.weight",
            ],
        );

        let bundle = load(&p).expect("both-format load");
        assert_eq!(bundle.format, LtxKeyFormat::Diffusers);

        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn no_transformer_keys_returns_error() {
        let p = temp_path("no-transformer");
        write_fixture(&p, &["vae.encoder.conv_in.weight", "some.other.weight"]);

        assert!(matches!(load(&p), Err(LoadError::NoTransformerKeys)));

        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn has_prefix_is_exact_segment_match() {
        // "blocks" should not match "transformer_blocks"
        assert!(!has_prefix("transformer_blocks.0.to_q.weight", "blocks"));
        // "model.diffusion_model.transformer_blocks" should not match ltx2's
        // "model.diffusion_model.blocks" prefix
        assert!(!has_prefix(
            "model.diffusion_model.transformer_blocks.0.to_q.weight",
            "model.diffusion_model.blocks"
        ));
        // Exact prefix match
        assert!(has_prefix(
            "transformer_blocks.0.to_q.weight",
            "transformer_blocks"
        ));
        // Root-level key == prefix
        assert!(has_prefix("vae", "vae"));
    }
}
