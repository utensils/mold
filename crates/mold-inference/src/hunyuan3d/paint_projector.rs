//! Tencent Hunyuan3D 2.1 paint DINO image-token projector.
//! Reference: hy3dpaint/hunyuanpaintpbr/unet/modules.py:710-754, revision
//! 82920d643c0dc2f7bfd7255f45f62d386edfe60c (ImageProjModel).
use candle_core::{DType, Result, Tensor};
use candle_nn::{Module, VarBuilder};

pub struct PaintImageProjector {
    projection: candle_nn::Linear,
    norm: candle_nn::LayerNorm,
    input_width: usize,
    context_width: usize,
    extra_tokens: usize,
    dtype: DType,
}
impl PaintImageProjector {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        Self::with_dimensions(vb, 1536, 1024, 4)
    }
    fn with_dimensions(
        vb: VarBuilder,
        input_width: usize,
        context_width: usize,
        extra_tokens: usize,
    ) -> Result<Self> {
        let projected_width = context_width
            .checked_mul(extra_tokens)
            .filter(|&n| n > 0)
            .ok_or_else(|| candle_core::Error::Msg("invalid paint projector dimensions".into()))?;
        if input_width == 0 || !matches!(vb.dtype(), DType::F32 | DType::F16) {
            candle_core::bail!("invalid paint projector input width")
        }
        // Round parameters to the model dtype before widening opmath. Torch's
        // Linear adds bias before the half output conversion; LayerNorm then
        // consumes that rounded projection and rounds its own output once.
        let projection = candle_nn::Linear::new(
            vb.pp("proj")
                .get((projected_width, input_width), "weight")?
                .to_dtype(DType::F32)?,
            Some(
                vb.pp("proj")
                    .get(projected_width, "bias")?
                    .to_dtype(DType::F32)?,
            ),
        );
        let norm = candle_nn::LayerNorm::new(
            vb.pp("norm")
                .get(context_width, "weight")?
                .to_dtype(DType::F32)?,
            vb.pp("norm")
                .get(context_width, "bias")?
                .to_dtype(DType::F32)?,
            1e-5,
        );
        Ok(Self {
            projection,
            norm,
            input_width,
            context_width,
            extra_tokens,
            dtype: vb.dtype(),
        })
    }
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, tokens, width) = match *input.dims() {
            [batch, width] => (batch, 1, width),
            [batch, tokens, width] => (batch, tokens, width),
            _ => candle_core::bail!("paint image projector expects rank two or three"),
        };
        if batch == 0 || tokens == 0 || width != self.input_width || input.dtype() != self.dtype {
            candle_core::bail!("invalid paint image projector input shape or dtype")
        }
        let rows = batch
            .checked_mul(tokens)
            .ok_or_else(|| candle_core::Error::Msg("paint projector batch overflow".into()))?;
        let output_tokens = tokens.checked_mul(self.extra_tokens).ok_or_else(|| {
            candle_core::Error::Msg("paint projector token count overflow".into())
        })?;
        if rows
            .checked_mul(self.extra_tokens)
            .and_then(|n| n.checked_mul(self.context_width))
            .is_none_or(|n| n > 64 * 1024 * 1024)
        {
            candle_core::bail!("paint image projector output exceeds its allocation bound")
        }
        let projected = self
            .projection
            .forward(&input.reshape((rows, width))?.to_dtype(DType::F32)?)?
            .to_dtype(input.dtype())?;
        self.norm
            .forward(
                &projected
                    .reshape((rows, self.extra_tokens, self.context_width))?
                    .to_dtype(DType::F32)?,
            )?
            .to_dtype(input.dtype())?
            .reshape((batch, output_tokens, self.context_width))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    #[test]
    fn pooled_and_token_inputs_match_tencent_and_keep_token_order() -> Result<()> {
        let fixture = candle_core::safetensors::load_buffer(
            include_bytes!("../../../../tests/fixtures/hunyuan3d/paint-projector-tiny.safetensors"),
            &Device::Cpu,
        )?;
        let vb = VarBuilder::from_tensors(fixture.clone(), DType::F32, &Device::Cpu).pp("weights");
        let model = PaintImageProjector::with_dimensions(vb, 8, 6, 4)?;
        for (label, shape) in [("pooled", vec![2, 4, 6]), ("tokens", vec![2, 12, 6])] {
            let actual = model.forward(&fixture[label])?;
            assert_eq!(actual.dims(), shape);
            let error = (actual - &fixture[&format!("{label}_expected")])?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(error < 1e-5, "{label}: {error}");
        }
        assert!(model
            .forward(&Tensor::zeros((2, 3, 7), DType::F32, &Device::Cpu)?)
            .is_err());
        assert!(model
            .forward(&Tensor::zeros((1, 2, 3, 8), DType::F32, &Device::Cpu)?)
            .is_err());
        assert!(model
            .forward(&Tensor::zeros((1, 0, 8), DType::F32, &Device::Cpu)?)
            .is_err());
        assert!(model
            .forward(&fixture["pooled"].to_dtype(DType::F16)?)
            .is_err());
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA and retained pretrained projector oracle"]
    fn pretrained_paint_projector_matches_tencent() -> anyhow::Result<()> {
        let oracle = std::path::PathBuf::from(std::env::var("MOLD_PAINT_PROJECTOR_ORACLE")?);
        let output = std::path::PathBuf::from(std::env::var("MOLD_PAINT_PROJECTOR_RESULT")?);
        std::fs::create_dir(&output)?;
        let device = Device::new_cuda(0)?;
        let weights = candle_core::safetensors::load(
            oracle.join("paint-projector-weights.safetensors"),
            &device,
        )?;
        let fixture = candle_core::safetensors::load(
            oracle.join("paint-projector-pretrained.safetensors"),
            &device,
        )?;
        let mut results = std::collections::HashMap::new();
        for (name, dtype, maximum, rms_limit) in [
            ("f32", DType::F32, 0.0001, 0.00001),
            ("f16", DType::F16, 0.01, 0.001),
        ] {
            let model = PaintImageProjector::new(VarBuilder::from_tensors(
                weights.clone(),
                dtype,
                &device,
            ))?;
            let actual = model.forward(&fixture[&format!("input_{name}")])?;
            results.insert(name.to_string(), actual.clone());
            candle_core::safetensors::save(&results, output.join(format!("{name}.safetensors")))?;
            let error = (actual.to_dtype(DType::F32)?
                - fixture[&format!("expected_{name}")].to_dtype(DType::F32)?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
            let max = error.iter().map(|x| x.abs()).fold(0., f32::max);
            let rms = (error.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>()
                / error.len() as f64)
                .sqrt();
            eprintln!("paint projector {name}: max={max}, rms={rms}");
            anyhow::ensure!(
                error.iter().all(|x| x.is_finite()) && max < maximum && rms < rms_limit,
                "paint projector {name} diverges"
            );
        }
        Ok(())
    }
}
