//! Loaders for non-diffusers checkpoint formats.
//!
//! Today this is the single-file Civitai dispatcher (`single_file`),
//! which header-parses a `.safetensors` and partitions its tensor keys
//! into UNet / VAE / CLIP-L / CLIP-G buckets per family. Future phases
//! may add other ingest paths (kohya, A1111 LyCORIS, etc.) under the
//! same module roof.

pub mod sd15_keys;
pub mod sdxl_keys;
pub mod single_file;
pub mod vae_keys;

pub use sd15_keys::{
    apply_sd15_clip_l_rename, apply_sd15_unet_rename, apply_sd15_vae_rename, build_sd15_remap,
    Sd15Remap,
};
pub use sdxl_keys::{
    apply_sdxl_clip_g_rename, apply_sdxl_clip_l_rename, apply_sdxl_unet_rename, build_sdxl_remap,
    RenameOutput, SdxlRemap,
};
pub use single_file::{load, LoadError, SingleFileBundle};
pub use vae_keys::apply_vae_rename;
