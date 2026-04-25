//! Model-discovery catalog for mold.
//!
//! See `docs/superpowers/specs/2026-04-25-catalog-expansion-design.md` for the
//! full design. Only `mold-cli` and `mold-server` depend on this crate.
//! `mold-discord` and `mold-tui` MUST NOT transitively depend on it — see
//! Task 36 for the dependency-tree check.

#![forbid(unsafe_code)]

pub mod civitai_map;
pub mod companions;
pub mod entry;
pub mod families;
pub mod filter;
pub mod hf_seeds;
pub mod normalizer;
pub mod scanner;
