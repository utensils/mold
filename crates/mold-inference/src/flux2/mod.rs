pub(crate) mod lora;
mod pipeline;
pub(crate) mod quantized_transformer;
pub(crate) mod sampling;
pub mod single_file;
pub mod text_encoder_residency;
pub(crate) mod transformer;
pub(crate) mod vae;

pub use pipeline::Flux2Engine;
pub use single_file::{detect_format, Flux2SingleFileFormat};
/// The CFG budget gate is re-exported because the server's execution plan has
/// to reach the same verdict the engine does. `flux2_cfg_batching` is a pure
/// function of three byte counts, so the planner charging it itself — rather
/// than re-deriving the arithmetic — is what keeps the recorded execution
/// class and the executed render from disagreeing.
pub use transformer::{flux2_cfg_batching, Flux2CfgBatching, Flux2Config};
