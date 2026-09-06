//! Shared Diffusers GroupNorm with PyTorch CUDA saved-statistics rounding.
//! PyTorch 2.5.1 group_norm_kernel.cu; see LICENSE-PYTORCH. VAE epsilon is
//! 1e-6; paint UNet residual blocks use 1e-5. Existing SD callers retain Candle arithmetic.
use candle::{DType, Module, Result, Tensor};
use candle_nn::VarBuilder;

#[cfg(feature = "cuda")]
#[path = "vae/cuda_norm.rs"]
mod cuda_norm;

#[derive(Debug)]
pub struct DiffusersGroupNorm {
    weight: Tensor,
    bias: Tensor,
    groups: usize,
    channels: usize,
    epsilon: f32,
    dtype: DType,
}

impl DiffusersGroupNorm {
    pub fn new(vb: VarBuilder, groups: usize, channels: usize, epsilon: f32) -> Result<Self> {
        if groups == 0
            || channels == 0
            || !channels.is_multiple_of(groups)
            || !epsilon.is_finite()
            || epsilon <= 0.
            || !matches!(vb.dtype(), DType::F16 | DType::BF16 | DType::F32)
        {
            candle::bail!("invalid Diffusers normalization configuration")
        }
        Ok(Self {
            weight: vb.get(channels, "weight")?.to_dtype(DType::F32)?,
            bias: vb.get(channels, "bias")?.to_dtype(DType::F32)?,
            groups,
            channels,
            epsilon,
            dtype: vb.dtype(),
        })
    }
}

impl Module for DiffusersGroupNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = xs.dims4()?;
        if batch == 0
            || channels != self.channels
            || height == 0
            || width == 0
            || xs.dtype() != self.dtype
        {
            candle::bail!("invalid Diffusers normalization input shape or dtype")
        }
        #[cfg(feature = "cuda")]
        if xs.device().is_cuda() && xs.dtype() == DType::F16 {
            return cuda_norm::forward(xs, &self.weight, &self.bias, self.groups, self.epsilon);
        }
        let grouped = xs.to_dtype(DType::F32)?.reshape((
            batch,
            self.groups,
            channels / self.groups * height * width,
        ))?;
        let mean = grouped.mean_keepdim(2)?;
        let variance = grouped.broadcast_sub(&mean)?.sqr()?.mean_keepdim(2)?;
        let epsilon = Tensor::new(self.epsilon, xs.device())?
            .to_dtype(xs.dtype())?
            .to_dtype(DType::F32)?;
        let rstd = variance
            .broadcast_add(&epsilon)?
            .sqrt()?
            .recip()?
            .to_dtype(xs.dtype())?
            .to_dtype(DType::F32)?
            .reshape((batch, self.groups, 1, 1))?;
        let mean =
            mean.to_dtype(xs.dtype())?
                .to_dtype(DType::F32)?
                .reshape((batch, self.groups, 1, 1))?;
        let affine_shape = (1, self.groups, channels / self.groups, 1);
        let a = rstd.broadcast_mul(&self.weight.reshape(affine_shape)?)?;
        let b = self
            .bias
            .reshape(affine_shape)?
            .broadcast_sub(&a.broadcast_mul(&mean)?)?;
        let x = grouped.reshape((batch, self.groups, channels / self.groups, height * width))?;
        x.broadcast_mul(&a)?
            .broadcast_add(&b)?
            .reshape(xs.shape())?
            .to_dtype(xs.dtype())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires NVIDIA CUDA"]
    fn both_paint_normalization_epsilons_match_torch_cuda_exactly() -> Result<()> {
        let device = candle::Device::new_cuda(0)?;
        let fixture = candle::safetensors::load_buffer(
            include_bytes!("../../../../tests/fixtures/hunyuan3d/paint-conv.safetensors"),
            &device,
        )?;
        for (name, epsilon) in [("vae", 1e-6), ("unet", 1e-5)] {
            let vb = VarBuilder::from_tensors(fixture.clone(), DType::F16, &device)
                .pp(format!("norm.{name}"));
            let norm = DiffusersGroupNorm::new(vb, 4, 32, epsilon)?;
            let actual = norm.forward(&fixture[&format!("norm.{name}.input")])?;
            let error = (actual.to_dtype(DType::F32)?
                - fixture[&format!("norm.{name}.expected")].to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
            eprintln!("paint normalization {name}: {error}");
            assert_eq!(error, 0.);
        }
        Ok(())
    }
}
