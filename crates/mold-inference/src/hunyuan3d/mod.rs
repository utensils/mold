//! Hunyuan3D 2.0 image-to-3D shape generation.
//!
//! Pipeline, in the order the engine runs it:
//!
//! 1. [`dino2`] — a DINOv2-giant vision tower encodes the (background-removed,
//!    letterboxed) source image into a token sequence. This is the ONLY
//!    conditioning; there is no text encoder anywhere in the family.
//! 2. [`transformer`] — a flow-matching DiT denoises a 1-D latent token
//!    sequence (3072 tokens x 64 channels) conditioned on those tokens.
//! 3. [`shape_vae`] — the vecset VAE turns the latents into an occupancy field
//!    and evaluates it on a dense query grid, producing logits per grid point.
//! 4. [`mesh`] — surface nets extract a triangle mesh from that grid on the CPU.
//! 5. [`glb`] — the mesh is written as binary glTF.
//!
//! Every checkpoint is ONE safetensors file carrying all three networks under
//! the `model.`, `vae.` and `conditioner.main_image_encoder.model.` prefixes;
//! see `crates/mold-core/src/manifest.rs` `hunyuan3d_manifests`.
//!
//! Upstream references, in preference order:
//!   - ComfyUI (the executable oracle): `comfy/ldm/hunyuan3d/{model,vae}.py`,
//!     `comfy_extras/nodes_hunyuan3d.py`, `comfy_extras/nodes_save_3d.py`.
//!   - Tencent: `hy3dgen/shapegen/` in `Tencent-Hunyuan/Hunyuan3D-2`, and the
//!     per-checkpoint `config.yaml` shipped beside each weights file.

pub mod backend;
pub mod dino2;
pub mod engine;
pub mod glb;
pub mod mesh;
pub mod obj;
pub mod paint_raster;
pub mod paint_views;
pub mod poster;
pub mod raster;
pub mod sampler;
pub mod shape_vae;
pub mod transformer;
pub mod transformer21;
pub mod turntable;

#[cfg(feature = "mesh-texture")]
pub mod uv;
