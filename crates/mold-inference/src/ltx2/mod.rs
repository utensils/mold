mod assets;
mod backend;
pub mod chain;
mod conditioning;
pub(crate) mod convrot;
mod execution;
mod guidance;
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
    let mut flags = single_file::AudioOutputAssetFlags::default();
    for path in [&paths.transformer, &paths.vae] {
        let Ok(found) = single_file::audio_output_asset_flags(path) else {
            continue;
        };
        flags.audio_vae |= found.audio_vae;
        flags.vocoder |= found.vocoder;
    }
    flags.audio_vae && flags.vocoder
}

/// Header-only capability check for a single catalog checkpoint.
pub fn checkpoint_supports_audio_output(path: &std::path::Path) -> bool {
    single_file::supports_audio_output(path).unwrap_or(false)
}
