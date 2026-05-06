//! Single-file validator for LTX-2 (LTXV2 / LTXV 2.3) Civitai checkpoints (phase 5).
//!
//! Reads only the safetensors header to detect the tensor key layout.
//! No tensor data is materialised.
//!
//! LTX-2 combined checkpoints (the standard Lightricks format) embed both
//! the video transformer and the VAE in a single file:
//!
//! - Transformer keys: native `blocks.*` prefix (e.g., `blocks.0.attn1.to_q.weight`)
//!   or diffusers `model.diffusion_model.blocks.*`.
//! - VAE keys: `vae.*` prefix (e.g., `vae.encoder.conv_in.weight`).
//!
//! Sub-family (`v2` = 19B, `v2.3` = 22B) is runtime config — this module does
//! not distinguish between them, and neither does the loader.
//!
//! Transformer-only fine-tunes (no `vae.*` keys) are rejected at
//! `Ltx2Engine::from_single_file` because the runtime always loads the
//! VAE from the same checkpoint path via `vb.pp("vae")`.
//!
//! For LTX-Video (using `transformer_blocks.*`) use
//! `ltx_video::single_file::load` instead.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use serde_json::Value;
use thiserror::Error;

/// Transformer key layout in the checkpoint file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LtxKeyFormat {
    /// Native LTX-2 format — keys lack the `model.diffusion_model.` prefix
    /// (e.g., `blocks.0.attn1.to_q.weight`). Requires the engine's
    /// `remap_ltx2_transformer_key` at load time.
    Native,
    /// Diffusers format — keys already carry the `model.diffusion_model.`
    /// prefix. No remap needed.
    Diffusers,
}

/// Result of header-parsing an LTX-2 single-file safetensors.
#[derive(Debug, Clone)]
pub struct Ltx2SingleFileBundle {
    /// Detected key-format of the transformer weights. The LTX-2 native
    /// CUDA runtime handles both formats internally; this field is for
    /// diagnostics and unit tests.
    #[allow(dead_code)]
    pub format: LtxKeyFormat,
    /// Number of transformer (`blocks.*`) keys found. Used in unit tests
    /// to verify non-transformer keys are not mis-counted.
    #[allow(dead_code)]
    pub transformer_key_count: usize,
    /// `true` when the checkpoint contains `vae.*` keys. LTX-2 combined
    /// checkpoints always include the VAE; transformer-only fine-tunes
    /// will have `has_vae = false` and will be rejected by `from_single_file`.
    pub has_vae: bool,
}

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("safetensors header parse failed: {0}")]
    Header(String),
    /// No `blocks.*` or `model.diffusion_model.blocks.*` keys found.
    /// The file is probably not an LTX-2 checkpoint (maybe LTX-Video?).
    #[error(
        "no LTX-2 transformer keys found \
         (expected `blocks.*` or `model.diffusion_model.blocks.*`)"
    )]
    NoTransformerKeys,
}

/// Header-parse the safetensors at `path` and return the detected layout.
///
/// Only reads the 8-byte length prefix + the JSON header — tensor data on
/// disk is never touched.
pub fn load(path: &Path) -> Result<Ltx2SingleFileBundle, LoadError> {
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

    Ok(Ltx2SingleFileBundle {
        format,
        transformer_key_count,
        has_vae: vae_count > 0,
    })
}

// `blocks.*` — native LTX-2 transformer key prefix (not `transformer_blocks.*`).
const NATIVE_TRANSFORMER_KEY: &str = "blocks";
// `model.diffusion_model.blocks.*` — diffusers-style LTX-2 prefix.
const DIFFUSERS_TRANSFORMER_KEY: &str = "model.diffusion_model.blocks";
// `vae.*` — VAE present in the combined checkpoint.
const VAE_KEY: &str = "vae";

/// `true` iff `key == prefix` or `key` starts with `"<prefix>."`.
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
            "mold-ltx2-sf-{}-{}-{}.safetensors",
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
    fn native_combined_checkpoint() {
        let p = temp_path("native-combined");
        write_fixture(
            &p,
            &[
                "blocks.0.attn1.to_q.weight",
                "blocks.0.attn1.to_k.weight",
                "proj_in.weight",
                "vae.encoder.conv_in.weight",
                "vae.decoder.conv_out.weight",
            ],
        );

        let bundle = load(&p).expect("native combined load");
        assert_eq!(bundle.format, LtxKeyFormat::Native);
        assert_eq!(bundle.transformer_key_count, 2); // only blocks.* counted
        assert!(bundle.has_vae);

        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn native_transformer_only_rejected_by_vae_check() {
        let p = temp_path("native-no-vae");
        write_fixture(&p, &["blocks.0.attn1.to_q.weight", "proj_in.weight"]);

        let bundle = load(&p).expect("native transformer-only parses");
        assert_eq!(bundle.format, LtxKeyFormat::Native);
        assert!(
            !bundle.has_vae,
            "transformer-only checkpoint must report no VAE"
        );

        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn diffusers_combined_checkpoint() {
        let p = temp_path("diffusers-combined");
        write_fixture(
            &p,
            &[
                "model.diffusion_model.blocks.0.attn1.to_q.weight",
                "model.diffusion_model.patchify_proj.weight",
                "vae.encoder.conv_in.weight",
            ],
        );

        let bundle = load(&p).expect("diffusers combined load");
        assert_eq!(bundle.format, LtxKeyFormat::Diffusers);
        assert!(bundle.has_vae);

        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn ltx_video_keys_not_confused_with_ltx2() {
        // LTX-Video uses transformer_blocks.*, not blocks.*
        let p = temp_path("ltxvideo-wrong-loader");
        write_fixture(
            &p,
            &[
                "transformer_blocks.0.attn1.to_q.weight",
                "vae.encoder.conv_in.weight",
            ],
        );

        // ltx_2 loader should NOT detect transformer_blocks.* as its keys
        assert!(matches!(load(&p), Err(LoadError::NoTransformerKeys)));

        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn no_transformer_keys_returns_error() {
        let p = temp_path("no-transformer");
        write_fixture(&p, &["vae.encoder.conv_in.weight"]);

        assert!(matches!(load(&p), Err(LoadError::NoTransformerKeys)));

        let _ = std::fs::remove_file(p);
    }
}
