//! Weight-free MiniMax H3 model primitives.
//!
//! Nothing in this module registers a runnable inference engine. The H3
//! license gate and the family capability contract remain the authority for
//! whether callers may construct these primitives with real weights.

pub mod audio;
pub mod audio_config;
pub mod audio_weights;

pub use audio::{
    AudioPosterior, AudioPosteriorSelection, AudioSoundtrackAssociation, AudioVae,
    AudioVaeCancellation, AudioVaePhase, NeverCancel, StereoLatents, StereoWaveform,
};
pub use audio_config::{
    AudioVaeConfig, MiniMaxH3AudioContract, LATENT_CHANNELS, LATENT_ROWS_PER_SECOND,
    SAMPLES_PER_LATENT, SAMPLE_RATE,
};
pub use audio_weights::{
    load_validated_audio_vae, validate_audio_safetensors, AudioTensorLayout, AudioTensorSpec,
};
