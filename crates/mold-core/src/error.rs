use thiserror::Error;

/// Primary error type for the mold crate ecosystem.
///
/// Provides structured error variants for the main failure modes so callers
/// can pattern-match on specific categories instead of string-matching.
/// The `Other` variant accepts any `anyhow::Error` as a catch-all.
#[derive(Debug, Error)]
pub enum MoldError {
    /// Request validation failures (bad dimensions, empty prompt, etc.)
    #[error("{0}")]
    Validation(String),

    /// Download/network issues
    #[error("{0}")]
    Download(String),

    /// Config parsing/loading errors
    #[error("{0}")]
    Config(String),

    /// HTTP client communication errors
    #[error("{0}")]
    Client(String),

    /// Model doesn't exist or isn't downloaded
    #[error("{0}")]
    ModelNotFound(String),

    /// Inference/generation errors
    #[error("{0}")]
    Inference(String),

    /// Catch-all for everything else
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Convenience alias used across the crate.
pub type Result<T> = std::result::Result<T, MoldError>;

impl From<crate::download::DownloadError> for MoldError {
    fn from(err: crate::download::DownloadError) -> Self {
        MoldError::Download(err.to_string())
    }
}
