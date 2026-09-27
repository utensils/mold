//! CUDA dispatch for [`super::rms_norm_rope_i`].
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{DeviceRepr, LaunchConfig, PushKernelArg};
use candle::cuda_backend::WrapErr;
use candle::{CpuStorage, CudaStorage, CustomOp3, DType, Layout, Result, Shape, Tensor};
use half::{bf16, f16};

#[rustfmt::skip]
mod kernels { include!(concat!(env!("OUT_DIR"), "/qk_norm_rope_cuda.rs")); }

/// The kernel assigns one warp per row, exactly candle's `rmsnorm` launch
/// for rows under 1024 columns, whose reduction order it reproduces.
pub(super) fn supports(dtype: DType, head_dim: usize) -> bool {
    matches!(dtype, DType::BF16 | DType::F16 | DType::F32) && head_dim < 1024
}

struct QkNormRope {
    eps: f32,
    sin: Tensor,
}

impl QkNormRope {
    // One (storage, layout) pair per operand, candle's CustomOp convention.
    #[allow(clippy::too_many_arguments)]
    fn launch<T: DeviceRepr + candle::cuda_backend::CudaDType + candle::WithDType>(
        &self,
        x: &CudaStorage,
        x_layout: &Layout,
        weight: &CudaStorage,
        weight_layout: &Layout,
        cos: &CudaStorage,
        cos_layout: &Layout,
        name: &str,
    ) -> Result<CudaStorage> {
        let (batch, seq, heads, head_dim) = x_layout.shape().dims4()?;
        let count = x_layout.shape().elem_count();
        let dev = x.device().clone();
        let offsets = |layout: &Layout| {
            layout.contiguous_offsets().ok_or_else(|| {
                candle::Error::Msg("qk_norm_rope requires contiguous operands".into())
            })
        };
        let (s0, s1) = offsets(x_layout)?;
        let x = x.as_cuda_slice::<T>()?.slice(s0..s1);
        let (s0, s1) = offsets(weight_layout)?;
        let weight = weight.as_cuda_slice::<T>()?.slice(s0..s1);
        let (s0, s1) = offsets(cos_layout)?;
        let cos = cos.as_cuda_slice::<f32>()?.slice(s0..s1);
        let (sin_storage, sin_layout) = self.sin.storage_and_layout();
        let sin_storage = match &*sin_storage {
            candle::Storage::Cuda(storage) => storage,
            _ => candle::bail!("qk_norm_rope: sin table is not on the CUDA device"),
        };
        let (s0, s1) = offsets(sin_layout)?;
        let sin = sin_storage.as_cuda_slice::<f32>()?.slice(s0..s1);
        let stream = dev.cuda_stream();
        // SAFETY: the kernel writes every element of the output.
        let mut out = unsafe { stream.alloc::<T>(count) }.w()?;
        let func = dev.get_or_load_custom_func(name, "qk-norm-rope", kernels::QK_NORM_ROPE)?;
        let mut builder = func.builder();
        builder
            .arg(&x)
            .arg(&weight)
            .arg(&cos)
            .arg(&sin)
            .arg(&mut out);
        candle::builder_arg!(builder, seq as u32, heads as u32, head_dim as u32, self.eps);
        let config = LaunchConfig {
            grid_dim: ((batch * seq * heads) as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: one 32-thread warp per (b, s, h) row; every index the kernel
        // forms is bounded by the checked dimensions.
        unsafe { builder.launch(config) }.w()?;
        Ok(CudaStorage::wrap_cuda_slice(out, dev.clone()))
    }
}

impl CustomOp3 for QkNormRope {
    fn name(&self) -> &'static str {
        "qk-norm-rope"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("qk_norm_rope's fused kernel is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        x_layout: &Layout,
        weight: &CudaStorage,
        weight_layout: &Layout,
        cos: &CudaStorage,
        cos_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (batch, seq, heads, head_dim) = x_layout.shape().dims4()?;
        if x_layout.shape().elem_count() > u32::MAX as usize
            || batch * seq * heads > u32::MAX as usize
        {
            candle::bail!("qk_norm_rope: tensor too large for one launch");
        }
        let storage = match x.dtype() {
            DType::BF16 => self.launch::<bf16>(
                x,
                x_layout,
                weight,
                weight_layout,
                cos,
                cos_layout,
                "qk_norm_rope_bf16",
            )?,
            DType::F16 => self.launch::<f16>(
                x,
                x_layout,
                weight,
                weight_layout,
                cos,
                cos_layout,
                "qk_norm_rope_f16",
            )?,
            DType::F32 => self.launch::<f32>(
                x,
                x_layout,
                weight,
                weight_layout,
                cos,
                cos_layout,
                "qk_norm_rope_f32",
            )?,
            other => candle::bail!("qk_norm_rope does not support {other:?}"),
        };
        Ok((storage, Shape::from((batch, heads, seq, head_dim))))
    }
}

pub(super) fn forward(
    x: &Tensor,
    weight: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    x.contiguous()?.apply_op3_no_bwd(
        &weight.contiguous()?,
        &cos.contiguous()?,
        &QkNormRope {
            eps,
            sin: sin.contiguous()?,
        },
    )
}
