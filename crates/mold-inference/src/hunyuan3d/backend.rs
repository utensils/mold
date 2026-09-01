//! Hunyuan3D's family-scoped compute-dtype policy: **F16 on every
//! accelerator**, F32 on the CPU.
//!
//! # Why F16 and not BF16
//!
//! ComfyUI runs this family in fp16 on both CUDA and MPS, and mold matches it
//! rather than inheriting the crate's usual "BF16 on CUDA" habit.
//!
//! The DiT: `comfy/supported_models_base.py:47` declares
//! `supported_inference_dtypes = [torch.float16, torch.bfloat16,
//! torch.float32]` (`Hunyuan3Dv2` does not override it), and the published
//! checkpoints ship fp16, so `comfy/model_management.py:1136-1138` takes the
//! `weight_dtype == torch.float16` arm and returns fp16 whenever
//! `should_use_fp16` agrees — which it does unconditionally on mps
//! (`:1848-1849`) and on any CUDA device of compute capability 8 or above
//! (`:1869-1871`).
//!
//! The VAE: `comfy/sd.py:856` sets `working_dtypes = [fp16, bf16, fp32]` for
//! `ShapeVAE`, and `comfy/model_management.py:1258-1273` walks that list in
//! order, so fp16 wins wherever `should_use_fp16` does.
//!
//! The vision tower: `comfy/clip_vision.py:51` takes its dtype from
//! `text_encoder_dtype`, which is fp16 on every device
//! (`comfy/model_management.py:1217-1232`).
//!
//! # The BF16 hazard this policy avoids
//!
//! `super::shape_vae::query_grid_chunk` casts the query-grid coordinates to
//! the compute dtype, exactly as upstream does
//! (`comfy/ldm/hunyuan3d/vae.py:442`, and again per chunk at `:888`). Those
//! coordinates ARE the sampling geometry of the occupancy field, so their
//! quantization is the resolution of the mesh.
//!
//! At the default octree resolution of 256 the grid spacing is
//! `2 * 1.01 / 256` = 0.00789, while BF16 carries 8 significant bits and its
//! ulp on `[1, 2)` is `2^-7` = 0.0078. Adjacent planes are therefore about
//! one ulp apart: they survive as distinct values by a hair, but the
//! quantized spacing swings between 0.0039 and 0.0117 — a plus or minus 50%
//! jitter of every query plane. Push the resolution to 384 and adjacent
//! planes collapse onto each other outright. F16's ulp there is `2^-10`, so
//! the same grid stays uniform to about 5%.

use candle_core::{DType, Device};

/// Resolve Hunyuan3D's compute dtype for the device the render selected.
pub(crate) fn compute_dtype(device: &Device) -> DType {
    if device.is_cpu() {
        // Half-precision matmul on candle's CPU backend is emulated and
        // slower than widening, and the CPU never runs this family for
        // memory reasons anyway.
        DType::F32
    } else {
        DType::F16
    }
}

/// Query points the shape VAE decodes per chunk when
/// `MOLD_HUNYUAN3D_DECODE_CHUNKS` is unset.
///
/// `upstream` is ComfyUI's `num_chunks` default (8,000), which CUDA and the
/// CPU keep. Metal takes 32,000, measured on an M4 Max at octree 256 with
/// `hunyuan3d-mini-turbo:fp16` (2026-09-01, framed armchair fixture):
///
/// | chunk   | volume decode | mesh sha256 |
/// | ------- | ------------- | ----------- |
/// | 8,000   | 140.9 s       | f7822d36…   |
/// | 32,000  | 113.9 s       | f7822d36…   |
/// | 100,000 | 137.1 s       | f7822d36…   |
///
/// The mesh is byte-identical at every size — each query attends to the same
/// cached latent keys on its own row, so the chunk only batches launches —
/// and 32,000 is the knee: past it the larger per-chunk buffers cost more
/// than the launches they save. Metal's math attention tiles queries at 512
/// rows regardless of this value, so the score buffer does not grow with it.
pub(crate) fn decode_chunk_default(device: &Device, upstream: usize) -> usize {
    if device.is_metal() {
        32_000
    } else {
        upstream
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{IndexOp, Tensor};

    #[test]
    fn cpu_keeps_upstreams_decode_chunk() {
        assert_eq!(decode_chunk_default(&Device::Cpu, 8_000), 8_000);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_decodes_in_the_measured_32k_chunks() {
        let Ok(metal) = Device::new_metal(0) else {
            return;
        };
        assert_eq!(decode_chunk_default(&metal, 8_000), 32_000);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_keeps_upstreams_decode_chunk() {
        let Ok(cuda) = Device::new_cuda(0) else {
            return;
        };
        assert_eq!(decode_chunk_default(&cuda, 8_000), 8_000);
    }

    #[test]
    fn cpu_computes_in_f32() {
        assert_eq!(compute_dtype(&Device::Cpu), DType::F32);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_computes_in_f16() {
        let Ok(metal) = Device::new_metal(0) else {
            return;
        };
        assert_eq!(compute_dtype(&metal), DType::F16);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_computes_in_f16() {
        let Ok(cuda) = Device::new_cuda(0) else {
            return;
        };
        assert_eq!(compute_dtype(&cuda), DType::F16);
    }

    /// The z axis of the real query grid, as `query_grid_chunk` builds it:
    /// the first `octree + 1` points share `ix = iy = 0` and walk `iz`.
    fn axis(octree: usize) -> Vec<f32> {
        let grid = super::super::shape_vae::query_grid_chunk(
            octree,
            1.01,
            0,
            octree + 1,
            &Device::Cpu,
            DType::F32,
        )
        .expect("query grid axis");
        grid.i((.., 2))
            .expect("z column")
            .to_vec1::<f32>()
            .expect("f32 axis")
    }

    fn round_trip(values: &[f32], dtype: DType) -> Vec<f32> {
        Tensor::from_slice(values, values.len(), &Device::Cpu)
            .expect("axis tensor")
            .to_dtype(dtype)
            .expect("narrow")
            .to_dtype(DType::F32)
            .expect("widen")
            .to_vec1::<f32>()
            .expect("values")
    }

    fn spacings(values: &[f32]) -> Vec<f32> {
        values.windows(2).map(|pair| pair[1] - pair[0]).collect()
    }

    /// BF16 cannot carry the query grid, and this is the measurement that
    /// says so rather than an appeal to the exponent width.
    ///
    /// At the default resolution BF16 keeps every plane distinct but smears
    /// the spacing by plus or minus 50%; one rung finer it merges planes
    /// outright. F16 holds both to a few percent. The grid coordinates are
    /// cast to the compute dtype on purpose (parity with
    /// `comfy/ldm/hunyuan3d/vae.py:442`), so this is a property of the dtype
    /// choice and nothing else.
    #[test]
    fn bf16_cannot_resolve_the_default_query_grid() {
        const OCTREE: usize = super::super::engine::DEFAULT_OCTREE_RESOLUTION;
        let nominal = 2.0 * 1.01 / OCTREE as f32;

        let exact = axis(OCTREE);
        assert_eq!(exact.len(), OCTREE + 1);

        let bf16 = spacings(&round_trip(&exact, DType::BF16));
        let worst = bf16
            .iter()
            .map(|step| (step / nominal - 1.0).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            worst > 0.4,
            "BF16 is supposed to smear the default grid; worst spacing error was only {worst}"
        );

        let f16 = spacings(&round_trip(&exact, DType::F16));
        for (index, step) in f16.iter().enumerate() {
            let error = (step / nominal - 1.0).abs();
            assert!(
                error < 0.1,
                "F16 plane {index} drifted {error} from the nominal spacing"
            );
        }

        // One rung finer and BF16 stops resolving the grid at all: adjacent
        // planes land on the same value, so the occupancy field is evaluated
        // twice at one place and never at the other.
        let finer = axis(384);
        let collapsed = spacings(&round_trip(&finer, DType::BF16))
            .iter()
            .filter(|step| **step == 0.0)
            .count();
        assert!(
            collapsed > 0,
            "expected BF16 to collapse adjacent planes at octree 384"
        );
        let survived = spacings(&round_trip(&finer, DType::F16))
            .iter()
            .filter(|step| **step == 0.0)
            .count();
        assert_eq!(survived, 0, "F16 must keep every plane at octree 384");
    }
}
