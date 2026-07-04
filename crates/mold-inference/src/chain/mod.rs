//! Model-agnostic chained-generation orchestration.
//!
//! Owns the per-stage render loop ([`ChainOrchestrator`]), the carryover
//! payload ([`ChainTail`] — decoded RGB tail frames, the one currency every
//! video family speaks), and the renderer abstraction
//! ([`ChainStageRenderer`]). Model families implement the renderer; the
//! LTX-2 impl lives in `crate::ltx2`. VAE-ratio-specific helpers
//! (`tail_latent_frame_count`) deliberately stay with their family.

pub mod capability;
pub mod orchestrator;
pub mod stitch;

pub use crate::audio::NativeAudioTrack;
pub use capability::{capability_for_family, CarryoverKind, ChainCapability};
pub use orchestrator::{
    ChainOrchestrator, ChainOrchestratorError, ChainRunOutput, ChainStageRenderer, ChainTail,
    StageOutcome, StageProgressEvent,
};
