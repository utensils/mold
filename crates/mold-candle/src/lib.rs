//! Mold-specific extensions built on the public upstream Candle API.
//!
//! This crate deliberately accepts and returns upstream Candle types. Model
//! implementations, loaders, and explicit operations belong here; changes to
//! universal tensor/backend semantics belong upstream.

pub mod ltx_video;
pub mod metal_reduce;
pub mod minimax_h3;
pub mod quantized;
pub mod quantized_nn;
pub mod stable_diffusion;
