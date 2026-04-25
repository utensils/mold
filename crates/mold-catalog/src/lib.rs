//! Model-discovery catalog for mold.
//!
//! See `docs/superpowers/specs/2026-04-25-catalog-expansion-design.md` for the
//! full design. Only `mold-cli` and `mold-server` depend on this crate.
//! `mold-discord` and `mold-tui` MUST NOT transitively depend on it — see
//! Task 36 for the dependency-tree check.

#![forbid(unsafe_code)]

pub mod entry;
pub mod families;
