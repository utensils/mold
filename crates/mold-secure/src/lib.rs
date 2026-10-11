//! Shared E2EE primitives and records. No network, application keys or inference.
//! The fixed Noise protocol is implemented by snow, not a custom key schedule.
mod authorization;
#[cfg(not(target_arch = "wasm32"))]
mod ffi;
mod identity;
mod objects;
mod packet;
mod records;
mod session;
#[cfg(feature = "wasm")]
mod wasm;
pub use authorization::*;
pub use identity::*;
pub use objects::*;
pub use packet::*;
pub use records::*;
pub use session::*;
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid secure transport input")]
    Input,
    #[error("secure transport state unavailable")]
    State,
    #[error("secure peer authentication failed")]
    Authentication,
    #[error("secure transport bound exceeded")]
    Limit,
    #[error("secure random source unavailable")]
    Random,
}
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(any(unix, windows))]
pub mod identity_file;

#[cfg(any(unix, windows))]
pub mod operator_key;
