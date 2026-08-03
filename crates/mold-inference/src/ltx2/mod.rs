mod assets;
mod backend;
pub mod chain;
mod conditioning;
pub(crate) mod convrot;
mod execution;
mod guidance;
mod hdr;
mod lora;
pub mod media;
mod model;
mod nvfp4;
mod pipeline;
mod plan;
mod preset;
mod runtime;
mod sampler;
pub(crate) mod single_file;
mod text;
mod tiling;

pub use chain::{extract_tail_latents, tail_latent_frame_count};
pub use pipeline::Ltx2Engine;

/// Whether the resolved checkpoint set contains both the audio VAE and
/// vocoder tensors required for native LTX-2 audio output.
pub fn audio_output_supported(paths: &mold_core::ModelPaths) -> bool {
    audio_output_gap(paths).is_none()
}

/// Why native LTX-2 audio output is unavailable for this checkpoint set, or
/// `None` when it is available.
///
/// Returns a specific reason rather than a generic one. The previous message
/// named both assets unconditionally — "the resolved checkpoint assets do not
/// include both the audio VAE and vocoder tensors" — which pointed every
/// reader at a bad download. When a 19B checkpoint's vocoder went unrecognised
/// because only the nested key spelling was probed, that message actively
/// misdirected: the tensors were present, under a prefix the probe did not
/// know. An error that misleads costs more than one that is merely vague.
pub fn audio_output_gap(paths: &mold_core::ModelPaths) -> Option<String> {
    let mut flags = single_file::AudioOutputAssetFlags::default();
    for path in [&paths.transformer, &paths.vae] {
        let Ok(found) = single_file::audio_output_asset_flags(path) else {
            continue;
        };
        flags.audio_vae |= found.audio_vae;
        flags.vocoder |= found.vocoder;
        flags.audio_vae_prefix_seen |= found.audio_vae_prefix_seen;
        flags.vocoder_prefix_seen |= found.vocoder_prefix_seen;
    }

    let mut missing: Vec<String> = Vec::new();
    if !flags.audio_vae {
        missing.push(if flags.audio_vae_prefix_seen {
            "the audio VAE (`audio_vae.*` tensors are present but not in a layout this build recognises)"
                .to_string()
        } else {
            "the audio VAE".to_string()
        });
    }
    if !flags.vocoder {
        missing.push(if flags.vocoder_prefix_seen {
            "the vocoder (`vocoder.*` tensors are present but not in a layout this build recognises)"
                .to_string()
        } else {
            "the vocoder".to_string()
        });
    }
    if missing.is_empty() {
        return None;
    }
    Some(missing.join(" and "))
}

/// Header-only capability check for a single catalog checkpoint.
pub fn checkpoint_supports_audio_output(path: &std::path::Path) -> bool {
    single_file::supports_audio_output(path).unwrap_or(false)
}
