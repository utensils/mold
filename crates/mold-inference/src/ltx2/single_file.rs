//! Single-file validator for LTX-2 (LTXV2 / LTXV 2.3) Civitai checkpoints.
//!
//! Reads only the safetensors header to detect the tensor key layout.
//! No tensor data is materialised.
//!
//! LTX-2 combined checkpoints (the standard Lightricks format) embed both
//! the video transformer and the VAE in a single file:
//!
//! - Transformer keys: native `transformer_blocks.*` prefix
//!   (e.g., `transformer_blocks.0.attn1.to_q.weight`) or diffusers
//!   `model.diffusion_model.transformer_blocks.*`. The runtime path
//!   (`runtime.rs::remap_ltx2_transformer_key`) unconditionally prepends
//!   `model.diffusion_model.` and the model code asks for
//!   `vb.pp("transformer_blocks")`, so both layouts map to the same
//!   on-disk keys after remap.
//! - VAE keys: `vae.*` prefix (e.g., `vae.encoder.conv_in.weight`).
//!
//! Sub-family (`v2` = 19B, `v2.3` = 22B) is runtime config — this module does
//! not distinguish between them, and neither does the loader.
//!
//! Transformer-only fine-tunes (no `vae.*` keys) use the version-matched
//! external VAE resolved by the catalog. Combined checkpoints keep loading
//! `vae.*` from the primary file.
//!
//! LTX-Video uses the same `transformer_blocks.*` segment, so this module
//! cannot disambiguate the two families on key shape alone — the call
//! site picks the loader by the resolved model family.

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
    /// checkpoints include the VAE; transformer-only fine-tunes have
    /// `has_vae = false` and require an external VAE companion.
    pub has_vae: bool,
    /// Whether the checkpoint also carries the Gemma hidden-state projection.
    /// Diffusion-only LTX-2.3 exports move this into a separate companion.
    pub has_text_projection: bool,
    /// `true` when the header carries NVFP4 sidecars (`*.weight_scale_2`
    /// or `*.comfy_quant`). The runtime handles these through synthetic
    /// `weight.nvfp4_*` subkeys.
    #[allow(dead_code)]
    pub has_nvfp4: bool,
    /// `__metadata__.model_version` from the safetensors header, when
    /// present. Official Lightricks LTX-2 v2.3 checkpoints stamp `"2.3.0"`
    /// here. Used by `Ltx2Engine::from_single_file` to derive a preset
    /// hint when the model name (e.g. `cv:2752735`) lacks the
    /// `ltx-2.3` / `ltx-2` substring `preset_for_model` looks for.
    pub model_version: Option<String>,
}

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("safetensors header parse failed: {0}")]
    Header(String),
    /// No `transformer_blocks.*` or `model.diffusion_model.transformer_blocks.*`
    /// keys found. The file is probably not an LTX-2 checkpoint.
    #[error(
        "no LTX-2 transformer keys found \
         (expected `transformer_blocks.*` or `model.diffusion_model.transformer_blocks.*`)"
    )]
    NoTransformerKeys,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct AudioOutputAssetFlags {
    pub(crate) audio_vae: bool,
    pub(crate) vocoder: bool,
    /// Some `audio_vae.*` tensor exists even though the sentinel did not
    /// match. Distinguishes "no audio VAE here" from "an audio VAE this build
    /// does not recognise", which need different advice.
    pub(crate) audio_vae_prefix_seen: bool,
    /// Some `vocoder.*` tensor exists even though no generator sentinel
    /// matched — an unrecognised vocoder layout rather than a missing one.
    pub(crate) vocoder_prefix_seen: bool,
}

/// Inspect only the safetensors header for the two asset families required to
/// decode LTX-2 audio. Video-only VAE companions intentionally report false.
///
/// Both sentinel sets come from the modules that own the corresponding
/// loaders — `model::vocoder::VOCODER_GENERATOR_SENTINELS` covers every layout
/// `CheckpointVocoderLayout` accepts, and `model::audio_vae::AUDIO_VAE_SENTINEL`
/// the audio VAE — so this probe cannot recognise a different set of
/// checkpoints than the code that reads them.
pub(crate) fn audio_output_asset_flags(path: &Path) -> Result<AudioOutputAssetFlags, LoadError> {
    use crate::ltx2::model::audio_vae::{AUDIO_VAE_KEY_PREFIX, AUDIO_VAE_SENTINEL};
    use crate::ltx2::model::vocoder::{VOCODER_GENERATOR_SENTINELS, VOCODER_KEY_PREFIX};

    let header = read_header(path)?;
    let audio_vae = header.contains_key(AUDIO_VAE_SENTINEL);
    let vocoder = VOCODER_GENERATOR_SENTINELS
        .iter()
        .any(|sentinel| header.contains_key(*sentinel));
    Ok(AudioOutputAssetFlags {
        audio_vae,
        vocoder,
        audio_vae_prefix_seen: !audio_vae
            && header
                .keys()
                .any(|key| key.starts_with(AUDIO_VAE_KEY_PREFIX)),
        vocoder_prefix_seen: !vocoder
            && header.keys().any(|key| key.starts_with(VOCODER_KEY_PREFIX)),
    })
}

pub(crate) fn supports_audio_output(path: &Path) -> Result<bool, LoadError> {
    let flags = audio_output_asset_flags(path)?;
    Ok(flags.audio_vae && flags.vocoder)
}

/// Header-parse the safetensors at `path` and return the detected layout.
///
/// Only reads the 8-byte length prefix + the JSON header — tensor data on
/// disk is never touched.
pub fn load(path: &Path) -> Result<Ltx2SingleFileBundle, LoadError> {
    let header = read_header(path)?;

    let mut native_count = 0usize;
    let mut diffusers_count = 0usize;
    let mut vae_count = 0usize;
    let mut text_projection_count = 0usize;
    let mut nvfp4 = false;

    for key in header.keys() {
        if key == "__metadata__" {
            continue;
        }
        if key.ends_with(".weight_scale_2") || key.ends_with(".comfy_quant") {
            nvfp4 = true;
            continue;
        }
        if key.ends_with(".weight_scale") || key.ends_with(".input_scale") {
            continue;
        }
        if has_prefix(key, NATIVE_TRANSFORMER_KEY) {
            native_count += 1;
        } else if has_prefix(key, DIFFUSERS_TRANSFORMER_KEY) {
            diffusers_count += 1;
        } else if has_prefix(key, VAE_KEY) {
            vae_count += 1;
        } else if has_prefix(key, "text_embedding_projection") {
            text_projection_count += 1;
        }
    }

    let (format, transformer_key_count) = if diffusers_count > 0 {
        (LtxKeyFormat::Diffusers, diffusers_count)
    } else if native_count > 0 {
        (LtxKeyFormat::Native, native_count)
    } else {
        return Err(LoadError::NoTransformerKeys);
    };

    let model_version = header
        .get("__metadata__")
        .and_then(|v| v.get("model_version"))
        .and_then(|v| v.as_str())
        .map(str::to_owned);

    Ok(Ltx2SingleFileBundle {
        format,
        transformer_key_count,
        has_vae: vae_count > 0,
        has_text_projection: text_projection_count > 0,
        has_nvfp4: nvfp4,
        model_version,
    })
}

// `transformer_blocks.*` — native LTX-2 transformer key prefix. The runtime
// loader (`runtime.rs::remap_ltx2_transformer_key`) prepends
// `model.diffusion_model.` before the file lookup, and the model code
// asks for `vb.pp("transformer_blocks")`, so this matches what is
// actually loaded from disk.
const NATIVE_TRANSFORMER_KEY: &str = "transformer_blocks";
// `model.diffusion_model.transformer_blocks.*` — diffusers-style LTX-2 prefix
// (the layout used by official Lightricks LTX-2 v2.3 checkpoints).
const DIFFUSERS_TRANSFORMER_KEY: &str = "model.diffusion_model.transformer_blocks";
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

/// Read just the safetensors header (the 8-byte length prefix + JSON
/// payload) and return the parsed map — including `__metadata__`. Does
/// not mmap or read tensor data.
fn read_header(path: &Path) -> Result<BTreeMap<String, Value>, LoadError> {
    let mut file = File::open(path)?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;
    serde_json::from_slice(&header_buf).map_err(|e| LoadError::Header(e.to_string()))
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
                "transformer_blocks.0.attn1.to_q.weight",
                "transformer_blocks.0.attn1.to_k.weight",
                "proj_in.weight",
                "vae.encoder.conv_in.weight",
                "vae.decoder.conv_out.weight",
            ],
        );

        let bundle = load(&p).expect("native combined load");
        assert_eq!(bundle.format, LtxKeyFormat::Native);
        assert_eq!(bundle.transformer_key_count, 2); // only transformer_blocks.* counted
        assert!(bundle.has_vae);

        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn native_transformer_only_rejected_by_vae_check() {
        let p = temp_path("native-no-vae");
        write_fixture(
            &p,
            &["transformer_blocks.0.attn1.to_q.weight", "proj_in.weight"],
        );

        let bundle = load(&p).expect("native transformer-only parses");
        assert_eq!(bundle.format, LtxKeyFormat::Native);
        assert!(
            !bundle.has_vae,
            "transformer-only checkpoint must report no VAE"
        );

        let _ = std::fs::remove_file(p);
    }

    /// LTX-2 ships two vocoder layouts and the capability probe has to accept
    /// both, because [`crate::ltx2::model::vocoder::Ltx2VocoderConfig::load`]
    /// already does: v2.0 (19B) stores the generator flat under `vocoder.*`,
    /// while v2.3 (22B) nests it beside the bandwidth-extension stage as
    /// `vocoder.vocoder.*`. Probing only the nested key reported "no vocoder"
    /// for every 19B checkpoint, which made *all* LTX-2 19B audio — joint
    /// audio-video as well as text-to-audio — refuse to run against a
    /// checkpoint that plainly contains the weights.
    #[test]
    fn audio_output_accepts_both_the_flat_and_nested_vocoder_layouts() {
        let nested = temp_path("vocoder-nested");
        write_fixture(
            &nested,
            &[
                "audio_vae.per_channel_statistics.mean-of-means",
                "vocoder.vocoder.conv_pre.weight",
            ],
        );
        assert!(
            supports_audio_output(&nested).unwrap(),
            "LTX-2.3 nests the generator under vocoder.vocoder.*"
        );

        let flat = temp_path("vocoder-flat");
        write_fixture(
            &flat,
            &[
                "audio_vae.per_channel_statistics.mean-of-means",
                "vocoder.conv_pre.weight",
            ],
        );
        assert!(
            supports_audio_output(&flat).unwrap(),
            "LTX-2.0 stores the generator flat under vocoder.*"
        );

        // The audio VAE is still required on its own merits.
        let vocoder_only = temp_path("vocoder-only");
        write_fixture(&vocoder_only, &["vocoder.conv_pre.weight"]);
        assert!(!supports_audio_output(&vocoder_only).unwrap());

        for path in [nested, flat, vocoder_only] {
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn audio_output_requires_both_audio_vae_statistics_and_vocoder_weights() {
        let complete = temp_path("audio-complete");
        write_fixture(
            &complete,
            &[
                "audio_vae.per_channel_statistics.mean-of-means",
                "vocoder.vocoder.conv_pre.weight",
            ],
        );
        assert!(supports_audio_output(&complete).unwrap());

        let video_only = temp_path("video-only");
        write_fixture(&video_only, &["vae.per_channel_statistics.mean-of-means"]);
        assert!(!supports_audio_output(&video_only).unwrap());

        let _ = std::fs::remove_file(complete);
        let _ = std::fs::remove_file(video_only);
    }

    #[test]
    fn diffusers_combined_checkpoint() {
        // Mirrors the on-disk layout of the official Lightricks LTX-2 v2.3
        // safetensors (e.g. `ltx23_full.safetensors`): every transformer
        // tensor is prefixed `model.diffusion_model.transformer_blocks.*`,
        // VAE under `vae.*`.
        let p = temp_path("diffusers-combined");
        write_fixture(
            &p,
            &[
                "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight",
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
    fn nvfp4_checkpoint_parses_as_supported_transformer() {
        // cv:2778348 (`ltx23_nvfp4.safetensors`) is an NVFP4-quantized
        // LTX-2 v2.3 fine-tune. The header parser must accept it and let
        // the runtime's NVFP4 subkey backend route packed FP4 weights.
        let p = temp_path("nvfp4");
        write_fixture(
            &p,
            &[
                "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight",
                "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight_scale",
                "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight_scale_2",
                "vae.encoder.conv_in.weight",
            ],
        );
        let bundle = load(&p).expect("NVFP4 LTX-2 checkpoint should parse");
        assert_eq!(bundle.format, LtxKeyFormat::Diffusers);
        assert_eq!(bundle.transformer_key_count, 1);
        assert!(bundle.has_nvfp4);
        assert!(bundle.has_vae);
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
