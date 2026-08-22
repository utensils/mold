mod lora;
mod pipeline;

pub(crate) mod identity;

/// PuLID v1.1 identity cross-attention for SDXL.
///
/// Deliberately NOT behind the `pulid` cargo feature, matching
/// [`crate::flux::pulid`]. The feature decides whether a server *advertises
/// and accepts* identity requests — that gate lives once, at the request
/// contract (`mold_core::identity`), which refuses an identity request
/// outright in a build without it. Gating the adapter as well would take it
/// out of the workspace clippy gate, which runs without `pulid`.
pub mod pulid;

pub use pipeline::SDXLEngine;
