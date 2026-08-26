//! Shared audio primitives, independent of any model family.

use mold_core::{Config, ModelPaths};

#[derive(Debug, Clone)]
pub struct NativeAudioTrack {
    pub interleaved_samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AudioOutputProbe {
    Ltx2,
}

fn output_probe(family: &str) -> Option<AudioOutputProbe> {
    match family {
        "ltx2" => Some(AudioOutputProbe::Ltx2),
        _ => None,
    }
}

/// Whether this inference build knows how to inspect native audio output for
/// the model family without loading its weights.
pub fn output_probe_registered(family: &str) -> bool {
    output_probe(family).is_some()
}

/// Resolve a downloaded model's native audio-output capability through the
/// inference-owned family registry.
///
/// `None` means the family has no registered capability probe. This keeps the
/// server annotation generic: adding a future audio/video family requires one
/// registry arm here, not another server-side family special case.
pub fn output_supported(
    family: &str,
    model_name: &str,
    config: &Config,
    paths: &ModelPaths,
) -> Option<bool> {
    match output_probe(family)? {
        AudioOutputProbe::Ltx2 => {
            // LTX-2.5 is a split pack whose audio VAE/vocoder is deliberately
            // absent from legacy `ModelPaths`. Resolve the lossless manifest
            // graph here, inside the inference-owned family registry, so the
            // generic server annotation cannot accidentally probe only the
            // video assets. Older monolithic LTX-2 checkpoints retain their
            // header-only transformer/VAE path.
            let audio_components =
                mold_core::ltx25_manifest::Ltx25ModelPaths::resolve(config, model_name)
                    .map(|split| split.audio_vae);
            Some(crate::ltx2::audio_output_supported_with_components(
                paths,
                audio_components.as_deref(),
            ))
        }
    }
}

/// Header-only form used for installed catalog checkpoints whose complete
/// companion path set is not represented by a built-in manifest.
pub fn checkpoint_output_supported(family: &str, checkpoint: &std::path::Path) -> Option<bool> {
    match output_probe(family)? {
        AudioOutputProbe::Ltx2 => Some(crate::ltx2::checkpoint_supports_audio_output(checkpoint)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_families_have_no_inferred_audio_capability() {
        assert_eq!(output_probe("future-av-family"), None);
        assert_eq!(output_probe("ltx2"), Some(AudioOutputProbe::Ltx2));
        assert!(!output_probe_registered("future-av-family"));
        assert!(output_probe_registered("ltx2"));
    }
}
