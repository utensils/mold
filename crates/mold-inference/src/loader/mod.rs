//! Loaders for non-diffusers checkpoint formats.
//!
//! Today this is the single-file Civitai dispatcher (`single_file`),
//! which header-parses a `.safetensors` and partitions its tensor keys
//! into UNet / VAE / CLIP-L / CLIP-G buckets per family. Future phases
//! may add other ingest paths (kohya, A1111 LyCORIS, etc.) under the
//! same module roof.

pub mod single_file;

pub use single_file::{load, LoadError, SingleFileBundle};
