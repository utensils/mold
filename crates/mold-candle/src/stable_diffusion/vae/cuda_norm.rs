//! Public Candle custom-op dispatch for the opt-in PyTorch CUDA GroupNorm port.
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle::cuda_backend::WrapErr;
use candle::{CpuStorage, CudaStorage, CustomOp3, DType, Layout, Result, Shape, Tensor};
use half::f16;

#[rustfmt::skip]
mod kernels { include!(concat!(env!("OUT_DIR"), "/vae_precision_cuda.rs")); }

struct GroupNorm {
    groups: usize,
    epsilon: f32,
}
impl CustomOp3 for GroupNorm {
    fn name(&self) -> &'static str {
        "paint-vae-cuda-group-norm"
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
        candle::bail!("native paint normalization is CUDA-only")
    }
    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_layout: &Layout,
        weight: &CudaStorage,
        weight_layout: &Layout,
        bias: &CudaStorage,
        bias_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (batch, channels, height, width) = input_layout.shape().dims4()?;
        let count = input_layout.shape().elem_count();
        if batch == 0
            || channels == 0
            || height == 0
            || width == 0
            || self.groups == 0
            || !channels.is_multiple_of(self.groups)
            || count > u32::MAX as usize
            || weight_layout.dims() != [channels]
            || bias_layout.dims() != [channels]
        {
            candle::bail!("invalid native paint normalization shape")
        }
        let offsets = |layout: &Layout| {
            layout.contiguous_offsets().ok_or_else(|| {
                candle::Error::Msg("native paint normalization requires contiguous operands".into())
            })
        };
        let (start, end) = offsets(input_layout)?;
        let input_slice = input.as_cuda_slice::<f16>()?.slice(start..end);
        let (start, end) = offsets(weight_layout)?;
        let weight_slice = weight.as_cuda_slice::<f32>()?.slice(start..end);
        let (start, end) = offsets(bias_layout)?;
        let bias_slice = bias.as_cuda_slice::<f32>()?.slice(start..end);
        let device = input.device();
        let stream = device.cuda_stream();
        // SAFETY: The stats kernel writes every element of both temporary
        // buffers; the affine kernel writes every returned element on this stream.
        let mut mean = unsafe { stream.alloc::<f16>(batch * self.groups) }.w()?;
        let mut rstd = unsafe { stream.alloc::<f16>(batch * self.groups) }.w()?;
        let mut output = unsafe { stream.alloc::<f16>(count) }.w()?;
        let row_size = count / batch / self.groups;
        let stats = device.get_or_load_custom_func(
            "paint_group_norm_stats",
            "paint-vae-group-norm",
            kernels::GROUP_NORM,
        )?;
        let mut builder = stats.builder();
        builder.arg(&input_slice).arg(&mut mean).arg(&mut rstd);
        candle::builder_arg!(builder, row_size as i64);
        candle::builder_arg!(builder, self.epsilon);
        let config = LaunchConfig {
            grid_dim: ((batch * self.groups) as u32, 1, 1),
            block_dim: (if row_size < 512 { 32 } else { 512 }, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: Arguments and row count match group_norm.cu; each block owns
        // one disjoint row and uses either one warp or the fixed 512-thread tree.
        unsafe { builder.launch(config) }.w()?;
        let apply = device.get_or_load_custom_func(
            "paint_group_norm_apply",
            "paint-vae-group-norm",
            kernels::GROUP_NORM,
        )?;
        let mut builder = apply.builder();
        builder
            .arg(&input_slice)
            .arg(&weight_slice)
            .arg(&bias_slice)
            .arg(&mean)
            .arg(&rstd)
            .arg(&mut output);
        candle::builder_arg!(builder, count as i64);
        candle::builder_arg!(builder, channels as i64);
        candle::builder_arg!(builder, (height * width) as i64);
        candle::builder_arg!(builder, self.groups as i64);
        // SAFETY: Checked dimensions bound every source/destination index;
        // cudarc retains stream dependencies until the queued use completes.
        unsafe { builder.launch(LaunchConfig::for_num_elems(count as u32)) }.w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            input_layout.shape().clone(),
        ))
    }
}

pub(super) fn forward(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    groups: usize,
    epsilon: f32,
) -> Result<Tensor> {
    if !epsilon.is_finite()
        || epsilon <= 0.
        || input.dtype() != DType::F16
        || weight.dtype() != DType::F32
        || bias.dtype() != DType::F32
        || !input.device().same_device(weight.device())
        || !input.device().same_device(bias.device())
    {
        candle::bail!("native paint normalization dtype/device mismatch")
    }
    input.contiguous()?.apply_op3_no_bwd(
        &weight.contiguous()?,
        &bias.contiguous()?,
        &GroupNorm { groups, epsilon },
    )
}
