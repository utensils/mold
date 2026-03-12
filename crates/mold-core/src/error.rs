use thiserror::Error;

#[derive(Error, Debug)]
pub enum MoldError {
    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("server error: {0}")]
    ServerError(String),

    #[error("connection failed: {0}")]
    ConnectionFailed(String),

    #[error("inference error: {0}")]
    InferenceError(String),

    #[error("config error: {0}")]
    ConfigError(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}
