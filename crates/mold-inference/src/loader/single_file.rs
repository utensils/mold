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
use mold_catalog::entry::{Bundling, Kind};
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
    /// `mold_catalog::civitai_map::engine_phase_for(family, Bundling::SingleFile, Kind::Checkpoint)`
    /// so callers can render "arrives in mold phase N" without a second
    /// lookup. Values at or above 6 mean unsupported by this build.
    #[error("family {0:?} is not a single-file family yet (phase {1})")]
    UnsupportedFamily(Family, u8),
}

/// Partition the given safetensors checkpoint into component key
/// buckets per the family's known prefix layout.
///
/// Only the safetensors header is read — tensor data is left untouched.
/// SD1.5 + SDXL produce `Ok(SingleFileBundle)`; every other family
/// returns `Err(LoadError::UnsupportedFamily(family, engine_phase))`
/// per `engine_phase_for(family, Bundling::SingleFile, Kind::Checkpoint)`.
pub fn load(path: &Path, family: Family) -> Result<SingleFileBundle, LoadError> {
    let clip_l_prefix = match family {
        Family::Sd15 => SD15_CLIP_L_PREFIX,
        Family::Sdxl => SDXL_CLIP_L_PREFIX,
        other => {
            return Err(LoadError::UnsupportedFamily(
                other,
                engine_phase_for(other, Bundling::SingleFile, Kind::Checkpoint),
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

/// Peek the safetensors header (8-byte length prefix + JSON, no tensor data
/// read) to determine whether a FLUX single-file checkpoint bundles its VAE.
/// Returns `true` when any VAE-encoder marker key is present.
///
/// The keys we look for are inclusive of the conventions used by Civitai
/// converters and BFL's own export format:
/// - `encoder.conv_in.*`           — diffusers-style root
/// - `first_stage_model.encoder.*` — A1111/ComfyUI-style root
/// - `vae.encoder.*`               — some pruning tools' convention
///
/// Civitai's FLUX fine-tune convention is inconsistent — some bundle the VAE
/// (`*_full.safetensors`), some are transformer-only (`*Unet.safetensors` /
/// `*_diffusion.safetensors`). Without this probe the catalog bridge would
/// unconditionally point `cfg.vae` at the primary checkpoint and the engine
/// would crash with `cannot find tensor encoder.conv_in.weight` on every
/// transformer-only fine-tune.
///
/// Returns `Err` only on bona fide I/O / parse failures — never panics.
pub fn flux_single_file_bundles_vae(path: &Path) -> std::io::Result<bool> {
    let mut file = File::open(path)?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;
    let header: BTreeMap<String, Value> = serde_json::from_slice(&header_buf).map_err(|e| {
        std::io::Error::other(format!(
            "parse safetensors header at {}: {e}",
            path.display()
        ))
    })?;
    Ok(header.keys().any(|k| {
        k != "__metadata__"
            && (k.starts_with("encoder.conv_in")
                || k.starts_with("first_stage_model.encoder.")
                || k.starts_with("vae.encoder."))
    }))
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
            // Flux.2 is wired through the catalog bridge (phase 1), but
            // *this* SD15/SDXL key-partition loader still rejects it — the
            // Flux.2 pipeline has its own loader. The error reports the
            // canonical phase regardless.
            (Family::Flux2, 1),
            (Family::ZImage, 4),
            (Family::LtxVideo, 5),
            (Family::Ltx2, 5),
            (Family::QwenImage, 1),
            (Family::Wuerstchen, 1),
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

    #[test]
    fn flux_single_file_bundles_vae_true_for_bundled_safetensors() {
        // Diffusers-style root: `encoder.conv_in.weight` is the canonical
        // VAE marker. Anything starting with `encoder.conv_in` (incl.
        // `.bias`) trips the probe.
        let path = temp_safetensors("flux-vae-diffusers");
        write_fixture(
            &path,
            &[
                "double_blocks.0.img_attn.proj.weight",
                "encoder.conv_in.weight",
                "decoder.conv_out.weight",
            ],
        );

        let bundled = flux_single_file_bundles_vae(&path).expect("probe must not error");
        assert!(
            bundled,
            "diffusers-style `encoder.conv_in.weight` must mark the file as VAE-bundled"
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn flux_single_file_bundles_vae_false_for_unet_only() {
        // The cv:994561 case: 780 keys, all `double_blocks.*` /
        // `single_blocks.*` / `img_in.*`, zero VAE encoder markers.
        let path = temp_safetensors("flux-unet-only");
        write_fixture(
            &path,
            &[
                "double_blocks.0.img_attn.proj.weight",
                "double_blocks.0.img_attn.norm.query_norm.scale",
                "single_blocks.0.linear1.weight",
                "img_in.weight",
                "txt_in.weight",
                "final_layer.linear.weight",
            ],
        );

        let bundled = flux_single_file_bundles_vae(&path).expect("probe must not error");
        assert!(
            !bundled,
            "transformer-only checkpoint (no encoder.conv_in / first_stage_model / vae prefix) \
             must NOT be marked as VAE-bundled"
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn flux_single_file_bundles_vae_handles_a1111_prefix() {
        // A1111/ComfyUI-style: VAE keys live under `first_stage_model.encoder.*`.
        let path = temp_safetensors("flux-vae-a1111");
        write_fixture(
            &path,
            &[
                "model.diffusion_model.double_blocks.0.img_attn.proj.weight",
                "first_stage_model.encoder.conv_in.weight",
                "first_stage_model.decoder.conv_out.weight",
            ],
        );

        let bundled = flux_single_file_bundles_vae(&path).expect("probe must not error");
        assert!(
            bundled,
            "A1111 `first_stage_model.encoder.*` prefix must mark the file as VAE-bundled"
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn flux_single_file_bundles_vae_handles_pruner_prefix() {
        // Some pruning tools strip `first_stage_model` and emit `vae.encoder.*`.
        let path = temp_safetensors("flux-vae-pruned");
        write_fixture(
            &path,
            &[
                "double_blocks.0.img_attn.proj.weight",
                "vae.encoder.conv_in.weight",
            ],
        );

        let bundled = flux_single_file_bundles_vae(&path).expect("probe must not error");
        assert!(
            bundled,
            "pruner-style `vae.encoder.*` prefix must mark the file as VAE-bundled"
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn flux_single_file_bundles_vae_io_error_on_missing_file() {
        // Bona fide I/O failure surfaces as Err; the bridge translates this
        // into a clear "probe FLUX checkpoint X for bundled VAE" message
        // rather than panicking or silently treating the file as transformer-only.
        let missing = std::env::temp_dir().join("mold-loader-flux-vae-missing.safetensors");
        let _ = std::fs::remove_file(&missing); // ensure it doesn't exist
        let err = flux_single_file_bundles_vae(&missing).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }
}
