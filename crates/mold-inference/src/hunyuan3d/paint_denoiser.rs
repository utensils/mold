//! Request-owned complete paint denoiser. Tencent 82920d64:
//! pipeline.py:297-329,630-698; unet/modules.py:952-1083.
use super::paint_attention::PaintRope;
use super::paint_block::PaintBlockKind;
use super::paint_guidance::paint_guidance;
use super::paint_positions::position_pyramid;
use super::paint_projector::PaintImageProjector;
use super::paint_sampler::PaintUniPc;
use super::paint_unet::{PaintUnet, PaintUnetCondition};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use std::{collections::HashMap, path::Path};

/// Encoded conditioning, before material replication or guidance branches.
/// Geometry latents: [B,V,4,S,S]; reference latents: [B,R,4,S,S];
/// DINO: [B,T,1536]; position maps: [B,V,3,H,W].
pub struct PaintInputs {
    pub normal: Tensor,
    pub position: Tensor,
    pub reference: Tensor,
    pub dino: Tensor,
    pub position_maps: Tensor,
}
impl PaintInputs {
    fn layout(&self) -> Result<(usize, usize, usize, usize)> {
        let shape = self.normal.dims();
        if shape.len() != 5 {
            candle_core::bail!("paint normal latents must have five dimensions")
        }
        let (batch, views, size) = (shape[0], shape[1], shape[3]);
        let reference = self.reference.dims();
        let dino = self.dino.dims();
        let maps = self.position_maps.dims();
        if !(1..=3).contains(&batch)
            || !(1..=6).contains(&views)
            || shape[2] != 4
            || shape[4] != size
            || !(8..=64).contains(&size)
            || !size.is_power_of_two()
            || self.position.dims() != shape
            || reference.len() != 5
            || reference[0] != batch
            || reference[1] == 0
            || reference[2..] != [4, size, size]
            || dino.len() != 3
            || dino[0] != batch
            || dino[1] == 0
            || dino[2] != 1536
            || maps.len() != 5
            || maps[..3] != [batch, views, 3]
            || maps[3] == 0
            || maps[4] == 0
            || maps[3] > 2048
            || maps[4] > 2048
            || !maps[3].is_multiple_of(size)
            || !maps[4].is_multiple_of(size)
            || !matches!(self.normal.dtype(), DType::F16 | DType::F32)
        {
            candle_core::bail!("invalid paint conditioning dimensions or dtype")
        }
        // Tencent's PIL converter retains half-precision position maps even
        // when the model and encoded conditioning use float32.
        if !matches!(self.position_maps.dtype(), DType::F16 | DType::F32)
            || !self
                .position_maps
                .device()
                .same_device(self.normal.device())
        {
            candle_core::bail!(
                "paint position maps must be floating point on the conditioning device"
            )
        }
        for tensor in [&self.position, &self.reference, &self.dino] {
            if tensor.dtype() != self.normal.dtype()
                || !tensor.device().same_device(self.normal.device())
            {
                candle_core::bail!("paint conditioning must share one dtype and device")
            }
        }
        Ok((batch, views, size, reference[1]))
    }
    fn guidance_branches(&self) -> Result<Self> {
        if self.layout()?.0 != 1 {
            candle_core::bail!("guided paint requires one conditioning batch")
        }
        let zeros = self.dino.zeros_like()?;
        Ok(Self {
            normal: self.normal.repeat((3, 1, 1, 1, 1))?,
            position: self.position.repeat((3, 1, 1, 1, 1))?,
            reference: self.reference.repeat((3, 1, 1, 1, 1))?,
            dino: Tensor::cat(&[&zeros, &zeros, &self.dino], 0)?,
            position_maps: self.position_maps.repeat((3, 1, 1, 1, 1))?,
        })
    }
}

pub struct PaintDenoiser {
    main: PaintUnet,
    reference: PaintUnet,
    projector: PaintImageProjector,
    albedo_text: Tensor,
    mr_text: Tensor,
    reference_text: Tensor,
    dtype: DType,
    device: Device,
    head_width: usize,
}
impl PaintDenoiser {
    pub fn load(path: &Path, dtype: DType, device: &Device) -> anyhow::Result<Self> {
        super::paint_weights::load_pth_exact(path, dtype, device, Self::new)
    }
    pub fn new(vb: VarBuilder) -> Result<Self> {
        Self::with_widths(vb, [320, 640, 1280, 1280], [5, 10, 20, 20])
    }
    fn with_widths(vb: VarBuilder, channels: [usize; 4], heads: [usize; 4]) -> Result<Self> {
        if !matches!(vb.dtype(), DType::F16 | DType::F32)
            || heads.contains(&0)
            || channels
                .iter()
                .zip(heads)
                .any(|(&c, h)| c % h != 0 || c / h != channels[0] / heads[0])
        {
            candle_core::bail!("invalid paint denoiser precision or attention widths")
        }
        Ok(Self {
            main: PaintUnet::with_widths(PaintBlockKind::Main, channels, heads, vb.pp("unet"))?,
            reference: PaintUnet::with_widths(
                PaintBlockKind::Reference,
                channels,
                heads,
                vb.pp("unet_dual"),
            )?,
            projector: PaintImageProjector::new(vb.pp("unet.image_proj_model_dino"))?,
            albedo_text: vb.pp("unet").get((77, 1024), "learned_text_clip_albedo")?,
            mr_text: vb.pp("unet").get((77, 1024), "learned_text_clip_mr")?,
            reference_text: vb.pp("unet").get((77, 1024), "learned_text_clip_ref")?,
            dtype: vb.dtype(),
            device: vb.device().clone(),
            head_width: channels[0] / heads[0],
        })
    }
    fn validate(&self, input: &PaintInputs) -> Result<(usize, usize, usize, usize)> {
        let shape = input.layout()?;
        if input.normal.dtype() != self.dtype || !input.normal.device().same_device(&self.device) {
            candle_core::bail!("paint conditioning does not match loaded denoiser")
        }
        Ok(shape)
    }
    /// Prepare once; the returned session borrows these exact model weights.
    pub fn prepare(
        &self,
        input: &PaintInputs,
        reference_scale: &Tensor,
    ) -> Result<PreparedPaint<'_>> {
        let (batch, views, size, references) = self.validate(input)?;
        if reference_scale.dims() != [batch]
            || reference_scale.dtype() != self.dtype
            || !reference_scale.device().same_device(&self.device)
            || reference_scale
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?
                .iter()
                .any(|v| !v.is_finite())
        {
            candle_core::bail!("invalid paint reference scales")
        }
        let geometry = Tensor::cat(&[&input.normal, &input.position], 2)?
            .unsqueeze(1)?
            .repeat((1, 2, 1, 1, 1, 1))?
            .reshape((batch * 2 * views, 8, size, size))?;
        let text = Tensor::stack(&[&self.albedo_text, &self.mr_text], 0)?
            .unsqueeze(0)?
            .unsqueeze(2)?
            .repeat((batch, 1, views, 1, 1))?
            .reshape((batch * 2 * views, 77, 1024))?;
        let dino = self.projector.forward(&input.dino)?;
        let positions = position_pyramid(&input.position_maps, size)?;
        let mut ropes = HashMap::new();
        for grid in [size, size / 2, size / 4, size / 8] {
            let tokens = views * grid * grid;
            ropes.insert(
                tokens,
                PaintRope::new(&positions[&tokens], self.head_width, grid * 8, 2)?,
            );
        }
        let reference_text =
            self.reference_text
                .unsqueeze(0)?
                .repeat((batch * references, 1, 1))?;
        let (_, reference) = self.reference.forward(
            &input
                .reference
                .reshape((batch * references, 4, size, size))?,
            0.,
            &reference_text,
            references,
            None,
        )?;
        Ok(PreparedPaint {
            owner: self,
            geometry,
            text,
            dino,
            reference,
            ropes,
            reference_scale: reference_scale.clone(),
            batch,
            views,
            size,
        })
    }
    pub fn prepare_guided(&self, input: &PaintInputs) -> Result<PreparedPaint<'_>> {
        self.validate(input)?;
        let branches = input.guidance_branches()?;
        let scales = Tensor::new(&[0f32, 1., 1.], &self.device)?.to_dtype(self.dtype)?;
        self.prepare(&branches, &scales)
    }
    /// Published fifteen-step recipe with explicit initial noise. Callback0 runs
    /// before conditioning inference; later callbacks see every retained sample.
    /// Returning an error cancels the request before its next denoiser call.
    pub fn denoise(
        &self,
        input: &PaintInputs,
        initial: &Tensor,
        mut callback: impl FnMut(usize, usize, &Tensor) -> Result<()>,
    ) -> Result<Tensor> {
        let (batch, views, size, _) = self.validate(input)?;
        if batch != 1
            || initial.dims() != [2 * views, 4, size, size]
            || initial.dtype() != self.dtype
            || !initial.device().same_device(&self.device)
        {
            candle_core::bail!("invalid initial paint noise")
        }
        let mut sampler = PaintUniPc::new(15)?;
        let timesteps = sampler.schedule().timesteps().to_vec();
        callback(0, timesteps.len(), initial)?;
        let prepared = self.prepare_guided(input)?;
        let mut sample = initial.clone();
        for (index, timestep) in timesteps.iter().enumerate() {
            let branches = sample
                .reshape((1, 2, views, 4, size, size))?
                .repeat((3, 1, 1, 1, 1, 1))?;
            let prediction = prepared.forward(&branches, *timestep as f64)?;
            let prediction = paint_guidance(&prediction, views, None)?;
            sample = sampler.step(&prediction, *timestep, &sample)?;
            callback(index + 1, timesteps.len(), &sample)?;
        }
        Ok(sample)
    }
}

/// Immutable per-request state; borrowing its owner prevents cache/weight mixing.
pub struct PreparedPaint<'a> {
    owner: &'a PaintDenoiser,
    geometry: Tensor,
    text: Tensor,
    dino: Tensor,
    reference: HashMap<String, Tensor>,
    ropes: HashMap<usize, PaintRope>,
    reference_scale: Tensor,
    batch: usize,
    views: usize,
    size: usize,
}
impl PreparedPaint<'_> {
    pub fn forward(&self, sample: &Tensor, timestep: f64) -> Result<Tensor> {
        if sample.dims() != [self.batch, 2, self.views, 4, self.size, self.size]
            || sample.dtype() != self.owner.dtype
            || !sample.device().same_device(&self.owner.device)
        {
            candle_core::bail!("paint sample does not match prepared conditioning")
        }
        let sample = sample.reshape((self.batch * 2 * self.views, 4, self.size, self.size))?;
        let input = Tensor::cat(&[&sample, &self.geometry], 1)?;
        let condition = PaintUnetCondition {
            reference: &self.reference,
            dino: &self.dino,
            ropes: &self.ropes,
            reference_scale: &self.reference_scale,
        };
        let (sample, _) =
            self.owner
                .main
                .forward(&input, timestep, &self.text, self.views, Some(&condition))?;
        Ok(sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tiny_inputs() -> Result<PaintInputs> {
        let device = Device::Cpu;
        Ok(PaintInputs {
            normal: Tensor::ones((1, 2, 4, 8, 8), DType::F32, &device)?,
            position: Tensor::zeros((1, 2, 4, 8, 8), DType::F32, &device)?,
            reference: Tensor::ones((1, 2, 4, 8, 8), DType::F32, &device)?,
            dino: Tensor::ones((1, 257, 1536), DType::F32, &device)?,
            position_maps: Tensor::zeros((1, 2, 3, 8, 8), DType::F32, &device)?,
        })
    }
    #[test]
    fn paint_guidance_branches_preserve_geometry_and_zero_only_dino() -> Result<()> {
        let inputs = tiny_inputs()?;
        let branches = inputs.guidance_branches()?;
        assert_eq!(branches.layout()?, (3, 2, 8, 2));
        for branch in 0..3 {
            assert_eq!(
                branches
                    .normal
                    .narrow(0, branch, 1)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                inputs.normal.flatten_all()?.to_vec1::<f32>()?
            );
            assert_eq!(
                branches
                    .reference
                    .narrow(0, branch, 1)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                inputs.reference.flatten_all()?.to_vec1::<f32>()?
            );
            assert_eq!(
                branches
                    .dino
                    .narrow(0, branch, 1)?
                    .min_all()?
                    .to_scalar::<f32>()?,
                if branch == 2 { 1. } else { 0. }
            );
            assert_eq!(
                branches
                    .dino
                    .narrow(0, branch, 1)?
                    .max_all()?
                    .to_scalar::<f32>()?,
                if branch == 2 { 1. } else { 0. }
            );
        }
        assert!(branches.guidance_branches().is_err());
        Ok(())
    }
    #[test]
    fn paint_guidance_retains_half_position_maps_for_float_model() -> Result<()> {
        let mut inputs = tiny_inputs()?;
        inputs.position_maps = inputs.position_maps.to_dtype(DType::F16)?;
        let branches = inputs.guidance_branches()?;
        assert_eq!(branches.layout()?, (3, 2, 8, 2));
        assert_eq!(branches.normal.dtype(), DType::F32);
        assert_eq!(branches.position_maps.dtype(), DType::F16);
        Ok(())
    }
    #[test]
    fn paint_preparation_refuses_inconsistent_conditioning() -> Result<()> {
        let mut inputs = tiny_inputs()?;
        inputs.position = inputs.position.to_dtype(DType::F16)?;
        assert!(inputs.guidance_branches().is_err());
        inputs.position = inputs.position.to_dtype(DType::F32)?;
        inputs.reference = Tensor::zeros((1, 1, 3, 8, 8), DType::F32, &Device::Cpu)?;
        assert!(inputs.guidance_branches().is_err());
        Ok(())
    }
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires retained Tencent denoiser capture"]
    fn prepared_paint_denoiser_matches_tencent() -> Result<()> {
        let root = std::path::PathBuf::from(
            std::env::var("MOLD_PAINT_UNET_ORACLE").expect("oracle directory"),
        );
        let device = Device::new_cuda(0)?;
        let fixture = candle_core::safetensors::load(root.join("paint-unet.safetensors"), &device)?;
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("paint-unet.json")).unwrap()).unwrap();
        let dtype = fixture["input.sample"].dtype();
        let model = if metadata["tiny"].as_bool().unwrap() {
            let weights = candle_core::safetensors::load(
                root.join("paint-unet-tiny-weights.safetensors"),
                &device,
            )?;
            let weights = weights
                .into_iter()
                .map(|(k, v)| Ok((k, v.to_dtype(dtype)?)))
                .collect::<Result<HashMap<_, _>>>()?;
            PaintDenoiser::with_widths(
                VarBuilder::from_tensors(weights, dtype, &device),
                [32, 64, 128, 128],
                [2, 4, 8, 8],
            )?
        } else {
            let path = std::env::var("MOLD_PAINT_UNET_WEIGHTS").expect("installed weights");
            PaintDenoiser::load(Path::new(&path), dtype, &device)
                .map_err(|e| candle_core::Error::Msg(format!("{e:#}")))?
        };
        let input = PaintInputs {
            normal: fixture["input.normal"].clone(),
            position: fixture["input.position"].clone(),
            reference: fixture["input.reference"].clone(),
            dino: fixture["input.dino"].clone(),
            position_maps: fixture["input.position_maps"].clone(),
        };
        let base = metadata["guidance_inputs"]
            .as_bool()
            .unwrap_or(false)
            .then(|| PaintInputs {
                normal: fixture["base.normal"].clone(),
                position: fixture["base.position"].clone(),
                reference: fixture["base.reference"].clone(),
                dino: fixture["base.dino"].clone(),
                position_maps: fixture["base.position_maps"].clone(),
            });
        let prepared = match &base {
            Some(base) => model.prepare_guided(base)?,
            None => model.prepare(&input, &fixture["input.reference_scale"])?,
        };
        let output = std::env::var("MOLD_PAINT_UNET_OUTPUT")
            .ok()
            .map(std::path::PathBuf::from);
        if let Some(path) = &output {
            std::fs::create_dir(path).unwrap();
        }
        let compare = |name: &str, actual: &Tensor, expected: &Tensor| -> Result<bool> {
            if let Some(path) = &output {
                candle_core::safetensors::save(
                    &HashMap::from([("actual", actual.contiguous()?)]),
                    path.join(format!("{name}.safetensors")),
                )?;
            }
            let difference = (actual.to_dtype(DType::F32)? - expected.to_dtype(DType::F32)?)?;
            let max = difference.abs()?.max_all()?.to_scalar::<f32>()?;
            let rms = difference.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
            eprintln!("prepared {name}: max={max} rms={rms}");
            let (max_bound, rms_bound) = if dtype == DType::F16 {
                (0.02, 0.002)
            } else {
                (1e-4, 1e-5)
            };
            Ok(max < max_bound && rms < rms_bound)
        };
        let mut passed = compare("dino", &prepared.dino, &fixture["cache.dino"])?;
        assert_eq!(prepared.reference.len(), 16);
        let mut names = prepared.reference.keys().collect::<Vec<_>>();
        names.sort();
        for name in names {
            passed &= compare(
                name,
                &prepared.reference[name],
                &fixture[&format!("cache.reference.{name}")],
            )?;
        }
        for (index, time) in [500, 400, 500].into_iter().enumerate() {
            let actual = prepared.forward(&fixture["input.sample"], time as f64)?;
            passed &= compare(
                &format!("output.{index}.{time}"),
                &actual,
                &fixture[&format!("expected.{time}")],
            )?;
        }
        if metadata["trajectory"].as_bool().unwrap_or(false) {
            let base = base.as_ref().expect("trajectory requires guided inputs");
            let initial = &fixture["trajectory.initial"];
            for stop in [0, 2] {
                let mut callbacks = Vec::new();
                let result = model.denoise(base, initial, |index, total, _| {
                    assert_eq!(total, 15);
                    callbacks.push(index);
                    if index == stop {
                        candle_core::bail!("test cancellation at {stop}")
                    }
                    Ok(())
                });
                assert!(result
                    .unwrap_err()
                    .to_string()
                    .contains("test cancellation"));
                assert_eq!(callbacks, (0..=stop).collect::<Vec<_>>());
            }
            let mut callbacks = Vec::new();
            let result = model.denoise(base, initial, |index, total, sample| {
                assert_eq!(total, 15);
                callbacks.push(index);
                if index > 0 {
                    passed &= compare(
                        &format!("trajectory.sample.{}", index - 1),
                        sample,
                        &fixture[&format!("trajectory.sample.{}", index - 1)],
                    )?;
                }
                Ok(())
            })?;
            assert_eq!(callbacks, (0..=15).collect::<Vec<_>>());
            passed &= compare(
                "trajectory.final",
                &result,
                &fixture["trajectory.sample.14"],
            )?;
        }
        assert!(
            passed,
            "prepared denoiser parity failed; output tensors retained"
        );
        Ok(())
    }
}
