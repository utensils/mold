pub(crate) mod identity;
pub(crate) mod lora;
pub(crate) mod lora_bypass;
pub(crate) mod offload;
pub(crate) mod pinned;
mod pipeline;
/// PuLID-FLUX identity cross-attention.
///
/// Deliberately NOT behind the `pulid` cargo feature. The feature decides
/// whether a server *advertises and accepts* identity requests — that gate
/// lives once, at the request contract (`mold_core::identity`), which refuses
/// an identity request outright in a build without it. Gating the adapter as
/// well would fork `FluxTransformer::denoise`'s signature by feature and put a
/// `#[cfg]` at every transformer call site, and would keep the module out of
/// the workspace clippy gate, which runs without `pulid`.
pub mod pulid;
/// Per-variant PuLID injection coverage on a synthetic FLUX transformer.
#[cfg(test)]
pub(crate) mod pulid_variants;
pub(crate) mod quantized_transformer;
pub(crate) mod transformer;

pub use pipeline::FluxEngine;
