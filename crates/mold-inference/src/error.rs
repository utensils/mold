use thiserror::Error;

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("model load error: {0}")]
    ModelLoadError(String),

    #[error("generation error: {0}")]
    GenerationError(String),

    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
}
