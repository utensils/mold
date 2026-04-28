//! Single-file Civitai checkpoint dispatcher (phase 2.3).
//!
//! Header-parses a `.safetensors` checkpoint and partitions its tensor
//! keys into UNet / VAE / CLIP-L / CLIP-G / unknown buckets, dispatched
//! by `Family`. Carries only the original key names — no tensor data is
//! materialised. The diffusers-key rename pass and tensor materialisation
//! are tasks 2.4 / 2.5.
//!
//! Phase-2.2 audit (`tasks/catalog-expansion-phase-2-tensor-audit.md`)
//! found that:
//!
//! - UNet always lives at `model.diffusion_model.*`
//! - VAE always lives at `first_stage_model.*`
//! - SD1.5 CLIP-L: `cond_stage_model.transformer.text_model.*`
//! - SDXL CLIP-L: `conditioner.embedders.0.transformer.text_model.*`
//! - SDXL CLIP-G: `conditioner.embedders.1.model.*`
//! - Stray tensors (e.g. `denoiser.sigmas`) are inert — they land in
//!   `unknown_keys` and are dropped by downstream consumers, never
//!   erroneous.
//! - Pony is structurally indistinguishable from generic SDXL — no
//!   sub-family branch in the dispatcher.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use mold_catalog::civitai_map::engine_phase_for;
use mold_catalog::entry::Bundling;
use mold_catalog::families::Family;
use serde_json::Value;
use thiserror::Error;

const UNET_PREFIX: &str = "model.diffusion_model";
const VAE_PREFIX: &str = "first_stage_model";
const SD15_CLIP_L_PREFIX: &str = "cond_stage_model.transformer.text_model";
const SDXL_CLIP_L_PREFIX: &str = "conditioner.embedders.0.transformer.text_model";
const SDXL_CLIP_G_PREFIX: &str = "conditioner.embedders.1.model";

/// Result of partitioning a Civitai single-file safetensors into
/// recognised component buckets.
///
/// Holds only the original key names verbatim — phase 2.4 / 2.5 do the
/// A1111 → diffusers rename pass and hand the renamed keys to candle's
/// `MmapedSafetensors::multi(&[path])` to materialise tensors lazily.
/// Keeping 2.3 zero-copy at the tensor layer avoids the `Mmap` +
/// `SafeTensors<'_>` lifetime puzzle.
#[derive(Debug, Clone)]
pub struct SingleFileBundle {
    /// The checkpoint path the bundle was sourced from.
    pub path: PathBuf,
    /// Keys under `model.diffusion_model.*`.
    pub unet_keys: Vec<String>,
    /// Keys under `first_stage_model.*`.
    pub vae_keys: Vec<String>,
    /// CLIP-L keys. SD1.5 → `cond_stage_model.transformer.text_model.*`.
    /// SDXL → `conditioner.embedders.0.transformer.text_model.*`.
    pub clip_l_keys: Vec<String>,
    /// CLIP-G keys (SDXL only) — `conditioner.embedders.1.model.*`.
    /// `None` for SD1.5.
    pub clip_g_keys: Option<Vec<String>>,
    /// Keys that did not match any recognised prefix. Logged + dropped
    /// by downstream consumers (e.g. Juggernaut's stray `denoiser.sigmas`).
    pub unknown_keys: Vec<String>,
}

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("safetensors header: {0}")]
    Header(String),
    /// Returned for any family whose single-file ingest path is not in
    /// scope yet. The `u8` is the canonical phase number from
    /// `mold_catalog::civitai_map::engine_phase_for(family, Bundling::SingleFile)`
    /// so callers can render "arrives in mold phase N" without a second
    /// lookup. `99` is the sentinel for "not in scope for any current phase".
    #[error("family {0:?} is not a single-file family yet (phase {1})")]
    UnsupportedFamily(Family, u8),
}

/// Partition the given safetensors checkpoint into component key
/// buckets per the family's known prefix layout.
///
/// Only the safetensors header is read — tensor data is left untouched.
/// SD1.5 + SDXL produce `Ok(SingleFileBundle)`; every other family
/// returns `Err(LoadError::UnsupportedFamily(family, engine_phase))`
/// per `engine_phase_for(family, Bundling::SingleFile)`.
pub fn load(path: &Path, family: Family) -> Result<SingleFileBundle, LoadError> {
    let clip_l_prefix = match family {
        Family::Sd15 => SD15_CLIP_L_PREFIX,
        Family::Sdxl => SDXL_CLIP_L_PREFIX,
        other => {
            return Err(LoadError::UnsupportedFamily(
                other,
                engine_phase_for(other, Bundling::SingleFile),
            ));
        }
    };

    let keys = read_tensor_keys(path)?;

    let mut unet_keys = Vec::new();
    let mut vae_keys = Vec::new();
    let mut clip_l_keys = Vec::new();
    let mut clip_g_keys: Vec<String> = Vec::new();
    let mut unknown_keys = Vec::new();

    for key in keys {
        if has_prefix(&key, UNET_PREFIX) {
            unet_keys.push(key);
        } else if has_prefix(&key, VAE_PREFIX) {
            vae_keys.push(key);
        } else if has_prefix(&key, clip_l_prefix) {
            clip_l_keys.push(key);
        } else if family == Family::Sdxl && has_prefix(&key, SDXL_CLIP_G_PREFIX) {
            clip_g_keys.push(key);
        } else {
            unknown_keys.push(key);
        }
    }

    Ok(SingleFileBundle {
        path: path.to_path_buf(),
        unet_keys,
        vae_keys,
        clip_l_keys,
        clip_g_keys: match family {
            Family::Sdxl => Some(clip_g_keys),
            _ => None,
        },
        unknown_keys,
    })
}

/// `true` iff `key` is exactly `prefix` or `key` starts with `"<prefix>."`.
/// Stops `model.diffusion_model_extras.foo` from being mistaken for a
/// `model.diffusion_model.*` key.
fn has_prefix(key: &str, prefix: &str) -> bool {
    if key.len() < prefix.len() {
        return false;
    }
    if key == prefix {
        return true;
    }
    key.as_bytes().get(prefix.len()) == Some(&b'.') && key.starts_with(prefix)
}

/// Read just the safetensors header and return the tensor key names,
/// excluding the `__metadata__` slot. Tensor data on disk is not
/// touched — this avoids mmaping multi-GB checkpoints just to inspect
/// their layout.
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

    fn temp_safetensors(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "mold-loader-{}-{}-{}.safetensors",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        path
    }

    /// Synthesize a minimal valid safetensors with a single F32 scalar
    /// per supplied key. The tensor data is just zero bytes — the
    /// loader only inspects keys, never values.
    fn write_fixture(path: &Path, keys: &[&str]) {
        let f32_zero = 0.0f32.to_le_bytes().to_vec();
        // `serialize_to_file` borrows from a single `&HashMap`, so we
        // need one shared zero-byte buffer that outlives every view.
        let buffers: Vec<Vec<u8>> = keys.iter().map(|_| f32_zero.clone()).collect();
        let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
        for (key, buf) in keys.iter().zip(buffers.iter()) {
            tensors.insert(
                (*key).to_string(),
                TensorView::new(SafeDtype::F32, vec![1], buf).unwrap(),
            );
        }
        serialize_to_file(&tensors, &None, path).unwrap();
    }

    #[test]
    fn partition_sd15_dreamshaper_layout() {
        let path = temp_safetensors("sd15");
        write_fixture(
            &path,
            &[
                "model.diffusion_model.input_blocks.0.0.weight",
                "first_stage_model.encoder.conv_in.weight",
                "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
                "denoiser.sigmas",
            ],
        );

        let bundle = load(&path, Family::Sd15).expect("sd15 partition");
        assert_eq!(bundle.path, path);
        assert_eq!(
            bundle.unet_keys,
            vec!["model.diffusion_model.input_blocks.0.0.weight".to_string()]
        );
        assert_eq!(
            bundle.vae_keys,
            vec!["first_stage_model.encoder.conv_in.weight".to_string()]
        );
        assert_eq!(
            bundle.clip_l_keys,
            vec![
                "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"
                    .to_string()
            ]
        );
        assert!(
            bundle.clip_g_keys.is_none(),
            "SD1.5 has no CLIP-G; expected None, got {:?}",
            bundle.clip_g_keys
        );
        assert_eq!(bundle.unknown_keys, vec!["denoiser.sigmas".to_string()]);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn partition_sdxl_layout() {
        let path = temp_safetensors("sdxl");
        write_fixture(
            &path,
            &[
                "model.diffusion_model.input_blocks.0.0.weight",
                "first_stage_model.encoder.conv_in.weight",
                "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
                "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight",
                "denoiser.sigmas",
            ],
        );

        let bundle = load(&path, Family::Sdxl).expect("sdxl partition");
        assert_eq!(
            bundle.unet_keys,
            vec!["model.diffusion_model.input_blocks.0.0.weight".to_string()]
        );
        assert_eq!(
            bundle.vae_keys,
            vec!["first_stage_model.encoder.conv_in.weight".to_string()]
        );
        assert_eq!(
            bundle.clip_l_keys,
            vec![
                "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"
                    .to_string()
            ]
        );
        assert_eq!(
            bundle.clip_g_keys.as_deref(),
            Some(
                vec![
                    "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight"
                        .to_string()
                ]
                .as_slice()
            ),
            "SDXL must populate CLIP-G keys",
        );
        assert_eq!(bundle.unknown_keys, vec!["denoiser.sigmas".to_string()]);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn partition_pony_uses_sdxl_path() {
        // Audit finding: Pony is structurally indistinguishable from
        // generic SDXL. No metadata, identical prefixes. Loader must
        // not branch on any sub-family hint.
        let path = temp_safetensors("pony");
        write_fixture(
            &path,
            &[
                "model.diffusion_model.input_blocks.0.0.weight",
                "model.diffusion_model.output_blocks.0.0.weight",
                "first_stage_model.encoder.conv_in.weight",
                "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
                "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight",
            ],
        );

        let bundle = load(&path, Family::Sdxl).expect("pony-shaped partition");
        assert_eq!(bundle.unet_keys.len(), 2);
        assert_eq!(bundle.vae_keys.len(), 1);
        assert_eq!(bundle.clip_l_keys.len(), 1);
        assert_eq!(
            bundle.clip_g_keys.as_ref().map(|v| v.len()),
            Some(1),
            "Pony must surface CLIP-G keys identically to generic SDXL",
        );
        assert!(
            bundle.unknown_keys.is_empty(),
            "Pony fixture has no strays; got {:?}",
            bundle.unknown_keys
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn unsupported_family_returns_error() {
        // Any non-SD15/SDXL family must be rejected with the canonical
        // phase number from `engine_phase_for`. The fixture path does
        // not need to exist — the family check happens before I/O.
        let path = std::env::temp_dir().join("does-not-exist.safetensors");

        let cases: &[(Family, u8)] = &[
            (Family::Flux, 3),
            (Family::Flux2, 3),
            (Family::ZImage, 4),
            (Family::LtxVideo, 5),
            (Family::Ltx2, 5),
            (Family::QwenImage, 99),
            (Family::Wuerstchen, 99),
        ];

        for (family, expected_phase) in cases {
            match load(&path, *family) {
                Err(LoadError::UnsupportedFamily(got_family, got_phase)) => {
                    assert_eq!(got_family, *family, "family round-trip");
                    assert_eq!(
                        got_phase, *expected_phase,
                        "phase number for {:?} must match engine_phase_for",
                        family,
                    );
                }
                other => panic!(
                    "load(_, {:?}) expected UnsupportedFamily({:?}, {}), got {:?}",
                    family, family, expected_phase, other
                ),
            }
        }
    }
}
