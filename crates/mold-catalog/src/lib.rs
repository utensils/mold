//! Model-discovery catalog for mold.
//!
//! Live-only proxy to HF + Civitai with a 5-minute in-process cache;
//! the bulk-scrape DB and walker are gone. Only `mold-cli` and
//! `mold-server` depend on this crate. `mold-discord` and `mold-tui`
//! MUST NOT transitively depend on it.

#![forbid(unsafe_code)]

pub mod civitai_map;
pub mod companions;
pub mod defaults;
pub mod entry;
pub mod families;
pub mod hf_seeds;
pub mod live;
pub mod normalizer;
pub mod resolve;
pub mod sidecar;
pub mod synthesis;
pub mod wan_a14b;
