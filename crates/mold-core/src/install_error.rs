//! Errors that can arise while *installing* a catalog (`cv:*`/`hf:*`)
//! model into the in-memory `state.config.models` view.
//!
//! The bool-returning predecessor (`install_catalog_model -> bool`)
//! collapsed every failure mode into "not installed", so a Civitai
//! outage and a typo'd model ID surfaced identically to the user. This
//! enum splits them along the three axes the server's `ApiError`
//! mapping cares about: transient (502), client-error (404), server bug
//! / malformed upstream payload (500).
//!
//! Lives in `mold-core` so both `mold-server` (HTTP path) and
//! `mold-cli` (local catalog bridge) can return the same shape; CLI
//! formats them as `anyhow::Error` user messages.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum InstallError {
    /// Live HF/Civitai lookup failed to *reach* the upstream — DNS,
    /// connect, TLS, or timeout. Maps to HTTP 502 Bad Gateway. The
    /// inner string carries the upstream error message for the user.
    #[error("network error contacting catalog upstream: {0}")]
    Network(String),

    /// Upstream returned 404 (or a normalize failure equivalent to
    /// "we don't recognize this ID"). Maps to HTTP 404 Not Found.
    #[error("catalog ID not found upstream: {0}")]
    NotFound(String),

    /// Upstream returned a payload mold couldn't normalize into a
    /// `CatalogEntry` — bad shape, missing files, unsupported family.
    /// Maps to HTTP 500 because it indicates either a mold bug or a
    /// genuinely broken upstream response.
    #[error("malformed catalog recipe: {0}")]
    RecipeMalformed(String),
}
