//! Admission memory model for the Hunyuan3D image-to-3D family.
//!
//! The generic `device::activation_bytes` model scales with `width × height`,
//! which is the wrong variable here twice over. A mesh has no output canvas,
//! so the request's `width`/`height` are the CONDITIONING size and never
//! change with the thing being produced; and the stage that actually decides
//! whether a render fits is the occupancy decode, whose cost is set by the
//! query-chunk size and the latent count, neither of which appears in an area.
//!
//! Modelled after `ltx2_admission.rs` and `wan_admission.rs`: a small,
//! explicit, testable peak estimate keyed on the request's own shape hint,
//! rather than a calibration constant multiplied by pixels.
//!
//! # Where the peak actually is
//!
//! Three candidate peaks, and the answer is the largest, not the sum — they
//! are sequential stages and each frees before the next allocates:
//!
//! 1. **Image conditioning.** DINOv2-giant over `(size/14)² + 1` tokens at
//!    hidden 1536, 24 heads. At the mini tier's 1022 px that is 5,330 tokens,
//!    and the score matrix alone is `5330² × 24 × 2 B` ≈ 1.36 GB. This is why
//!    the 0.6B tier is not automatically the cheapest one to run.
//! 2. **Shape sampling.** The DiT attends over the latent tokens concatenated
//!    with the conditioning tokens.
//! 3. **Occupancy decode.** Each chunk cross-attends `chunk` query points
//!    against `num_latents` latents: `chunk × latents × heads × 2 B`.
//!
//! Host memory is dominated by something the GPU never sees: the full logit
//! grid, `(octree + 1)³ × 4 B`, plus the extracted mesh.

/// Latent tokens in every published 2.0 tier (`num_latents` in `config.yaml`).
const NUM_LATENTS: u64 = 3072;
/// DiT attention heads.
const DIT_HEADS: u64 = 16;
/// DINOv2-giant geometry.
const VISION_HEADS: u64 = 24;
const VISION_PATCH: u64 = 14;
/// Half-precision activations everywhere on GPU.
const ACTIVATION_BYTES: u64 = 2;
/// Fragmentation headroom. The decode loop allocates and frees one chunk's
/// tensors thousands of times; an allocator that never quite reuses the same
/// block is the difference between fitting and not.
const FRAGMENTATION_MARGIN: u64 = 512 * 1024 * 1024;
/// GPU floor, matching the generic model's: kernel scratch, cuBLAS workspaces.
const FLOOR_BYTES: u64 = 256_000_000;
/// Host floor. Deliberately much lower than the GPU one — this covers the
/// mesher's own bookkeeping, not a kernel workspace, and borrowing the 256 MB
/// figure would swallow the cubic term at every resolution below 384 and make
/// the estimate report a constant.
const HOST_FLOOR_BYTES: u64 = 64_000_000;

/// The request-derived shape the estimate keys on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hunyuan3dShape {
    /// Edge length the source image is letterboxed to before DINOv2.
    pub conditioning_size: u32,
    /// Latent sequence length of the selected shape checkpoint.
    pub num_latents: u64,
    /// Attention heads in the selected vision encoder.
    pub vision_heads: u64,
    /// Query-grid resolution.
    pub octree_resolution: u32,
    /// Query points per decode chunk.
    pub decode_chunk: u32,
}

impl Default for Hunyuan3dShape {
    fn default() -> Self {
        Self {
            conditioning_size: 512,
            num_latents: NUM_LATENTS,
            vision_heads: VISION_HEADS,
            octree_resolution: 256,
            // Upstream's `num_chunks` and the engine's CUDA/CPU default. The
            // Metal engine defaults to 32,000 (measured: 19% off the decode
            // wall at octree 256, byte-identical mesh), but Metal's math
            // attention tiles queries at 512 rows regardless, so the true
            // Metal peak is far BELOW what this term prices even at 8,000;
            // the estimate stays conservative on both backends.
            decode_chunk: 8_000,
        }
    }
}

impl Hunyuan3dShape {
    /// Read the shape from a request, falling back to the selected recipe.
    ///
    /// `conditioning_size` is taken from the request's `width` because that is
    /// where legacy clients record it. Canvasless requests use the manifest
    /// geometry, including the larger mini encoder and 2.1 latent set. It is not an
    /// output canvas and is never treated as one.
    pub fn from_request(req: &mold_core::GenerateRequest) -> Self {
        let defaults = Self::default();
        let geometry = mold_core::manifest::hunyuan3d_shape_geometry(&req.model);
        Self {
            conditioning_size: if req.width > 0 {
                req.width
            } else {
                geometry.conditioning_size
            },
            num_latents: geometry.num_latents,
            vision_heads: geometry.vision_heads,
            octree_resolution: req
                .mesh
                .as_ref()
                .and_then(|mesh| mesh.octree_resolution)
                .unwrap_or(defaults.octree_resolution),
            decode_chunk: defaults.decode_chunk,
        }
    }

    /// Edge the encoder actually sees.
    ///
    /// `conditioning_size` is the request's `width`, which the manifest seeds
    /// with the LETTERBOX edge (512 for the 1.1B tiers, 1022 for mini) — see
    /// [`Self::from_request`]. The conditioner resizes that square up to the encoder's
    /// own `image_size` before patching — 518 for the 1.1B tiers, per
    /// `hunyuan3d-dit-v2-0/config.yaml` and the `ImageEncoder` in Tencent's
    /// `conditioner.py`; mini's 1022 is already its encoder size. The next
    /// multiple of the patch reproduces both without a per-tier table.
    pub fn encoder_edge(&self) -> u64 {
        (self.conditioning_size as u64).div_ceil(VISION_PATCH) * VISION_PATCH
    }

    /// Token count DINOv2 sees: one patch per 14 px, plus the CLS token.
    pub fn vision_tokens(&self) -> u64 {
        let grid = self.encoder_edge() / VISION_PATCH;
        grid * grid + 1
    }

    /// Points on the query grid.
    pub fn query_points(&self) -> u64 {
        let edge = self.octree_resolution as u64 + 1;
        edge * edge * edge
    }
}

/// Peak GPU workspace above the resident weights.
pub fn activation_peak_bytes(shape: Hunyuan3dShape) -> u64 {
    let vision_tokens = shape.vision_tokens();
    // The score matrix is the whole story for a quadratic attention; the
    // projections are linear in the token count and vanish beside it.
    let vision = vision_tokens
        .saturating_mul(vision_tokens)
        .saturating_mul(shape.vision_heads)
        .saturating_mul(ACTIVATION_BYTES);

    let dit_tokens = shape.num_latents.saturating_add(vision_tokens);
    let dit = dit_tokens
        .saturating_mul(dit_tokens)
        .saturating_mul(DIT_HEADS)
        .saturating_mul(ACTIVATION_BYTES);

    let decode = (shape.decode_chunk as u64)
        .saturating_mul(shape.num_latents)
        .saturating_mul(DIT_HEADS)
        .saturating_mul(ACTIVATION_BYTES);

    // The largest, not the sum: these are sequential stages, and each one's
    // workspace is freed before the next allocates.
    let peak = vision.max(dit).max(decode);
    peak.saturating_add(FRAGMENTATION_MARGIN).max(FLOOR_BYTES)
}

/// Peak HOST memory the CPU stages need, above the process baseline.
///
/// Dominated by the full logit grid, which never reaches the GPU: the decode
/// loop copies each chunk's logits back and accumulates them, so at octree 384
/// this is `385³ × 4 B` ≈ 228 MB before the mesher has allocated anything.
pub fn host_peak_bytes(shape: Hunyuan3dShape) -> u64 {
    let grid = shape.query_points().saturating_mul(4);
    // Surface nets emit on the order of one vertex per surface-crossing cell.
    // A generous surface fraction of the grid, at 3 f32 positions + 3 f32
    // normals + roughly two triangles of u32 indices per vertex, is ~36 bytes
    // per emitted vertex; 2% of cells is a comfortable upper bound for a
    // closed object filling the volume.
    let mesh = shape.query_points().saturating_mul(36) / 50;
    grid.saturating_add(mesh).max(HOST_FLOOR_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 1.1B tiers letterbox to 512 but encode at 518 (37x37 patches), so
    /// an estimate built on the letterbox edge would price a 36x36 grid and
    /// undercount the score matrix. Mini is already a multiple of the patch.
    #[test]
    fn vision_tokens_are_counted_on_the_encoder_edge_not_the_letterbox() {
        let base = Hunyuan3dShape {
            conditioning_size: 512,
            ..Hunyuan3dShape::default()
        };
        assert_eq!(base.encoder_edge(), 518);
        assert_eq!(base.vision_tokens(), 37 * 37 + 1);

        let mini = Hunyuan3dShape {
            conditioning_size: 1022,
            ..Hunyuan3dShape::default()
        };
        assert_eq!(mini.encoder_edge(), 1022);
        assert_eq!(mini.vision_tokens(), 73 * 73 + 1);
    }

    #[test]
    fn the_mini_tiers_conditioning_dominates_its_own_peak() {
        // 1022 px conditioning is 5,330 DINOv2 tokens — a bigger sequence
        // than the DiT's own, which is why the SMALLER checkpoint is not
        // automatically the cheaper one to run.
        let mini = Hunyuan3dShape {
            conditioning_size: 1022,
            ..Hunyuan3dShape::default()
        };
        assert_eq!(mini.vision_tokens(), 73 * 73 + 1);
        // The 512 px letterbox is encoded at 518 px: 37x37 patches, not 36x36.
        let base = Hunyuan3dShape::default();
        assert_eq!(base.vision_tokens(), 37 * 37 + 1);
        assert!(
            activation_peak_bytes(mini) > activation_peak_bytes(base),
            "the 1022 px tier must be estimated above the 512 px one"
        );
    }

    #[test]
    fn octree_resolution_drives_host_memory_cubically() {
        let at = |octree| {
            host_peak_bytes(Hunyuan3dShape {
                octree_resolution: octree,
                ..Hunyuan3dShape::default()
            })
        };
        // Both rungs clear the host floor, so the ratio is the pure cubic
        // term — which is the whole reason the request field is an allowlist
        // rather than a range.
        let ratio = at(384) as f64 / at(320) as f64;
        let expected = (385.0_f64 / 321.0).powi(3);
        assert!(
            (ratio - expected).abs() < 0.02,
            "expected the cubic ratio {expected}, got {ratio}"
        );
        // And the growth is real, not a constant hiding behind the floor.
        assert!(at(384) > at(320) && at(320) > at(192));
    }

    #[test]
    fn octree_resolution_does_not_change_the_gpu_peak() {
        // Deliberate: the decode is CHUNKED, so a bigger grid is more chunks
        // of the same size, not a bigger allocation. If this ever stops being
        // true the chunking has been broken.
        let small = activation_peak_bytes(Hunyuan3dShape {
            octree_resolution: 128,
            ..Hunyuan3dShape::default()
        });
        let large = activation_peak_bytes(Hunyuan3dShape {
            octree_resolution: 384,
            ..Hunyuan3dShape::default()
        });
        assert_eq!(small, large);
    }

    #[test]
    fn every_estimate_clears_the_floor() {
        let tiny = Hunyuan3dShape {
            conditioning_size: 14,
            octree_resolution: 16,
            decode_chunk: 1,
            ..Hunyuan3dShape::default()
        };
        assert!(activation_peak_bytes(tiny) >= FLOOR_BYTES);
        assert!(host_peak_bytes(tiny) >= HOST_FLOOR_BYTES);
    }

    fn request(width: u32, mesh: serde_json::Value) -> mold_core::GenerateRequest {
        serde_json::from_value(serde_json::json!({
            "prompt": "",
            "model": "hunyuan3d-mini-turbo:fp16",
            "width": width,
            "height": width,
            "steps": 5,
            "guidance": 5.0,
            "seed": 42,
            "mesh": mesh,
        }))
        .expect("synthetic hunyuan3d request")
    }

    #[test]
    fn the_shape_reads_the_request_and_falls_back_cleanly() {
        let shape = Hunyuan3dShape::from_request(&request(
            1022,
            serde_json::json!({ "octree_resolution": 320 }),
        ));
        assert_eq!(shape.conditioning_size, 1022);
        assert_eq!(shape.octree_resolution, 320);

        // No `mesh` block and no canvas: the family defaults stand rather
        // than a zero reaching the estimate and collapsing it to the floor.
        let fallback = Hunyuan3dShape::from_request(&request(0, serde_json::Value::Null));
        assert_eq!(fallback.conditioning_size, 1022);
    }

    #[test]
    fn shape21_prices_its_4096_latents_and_large_vision_tower() {
        let mut req = request(0, serde_json::Value::Null);
        req.model = "hunyuan3d-2.1:fp16".into();
        let shape = Hunyuan3dShape::from_request(&req);
        assert_eq!(shape.num_latents, 4096);
        assert_eq!(shape.vision_heads, 16);
        assert!(activation_peak_bytes(shape) > activation_peak_bytes(Hunyuan3dShape::default()));
    }
}
