//! Model-discovery catalog for mold.
//!
//! See `docs/superpowers/specs/2026-04-25-catalog-expansion-design.md` for the
//! full design. This crate is intentionally lean: only `mold-cli` and
//! `mold-server` depend on it. `mold-discord` and `mold-tui` MUST NOT
//! transitively depend on this crate — see Task 36 for the dependency
//! tree check.

#![forbid(unsafe_code)]
