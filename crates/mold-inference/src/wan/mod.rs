//! Wan 2.1/2.2 video generation family.
//!
//! Ships in stacked layers: the UMT5-XXL text encoder (this layer), the
//! causal video VAE, then the DiT + FlowUniPC sampler + T2V/I2V pipelines
//! with the engine factory arm. Upstream source of truth:
//! <https://github.com/Wan-Video/Wan2.1> and <https://github.com/Wan-Video/Wan2.2>
//! (cloned under gitignored `tmp/`).

pub(crate) mod conditioning;
pub(crate) mod model;
pub mod pipeline;
pub(crate) mod sampler;
pub(crate) mod text;
