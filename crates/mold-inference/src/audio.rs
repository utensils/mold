//! Shared audio primitives, independent of any model family.

#[derive(Debug, Clone)]
pub struct NativeAudioTrack {
    pub interleaved_samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}
