// Copyright 2024 The HuggingFace Team. All rights reserved.
// Apache-2.0; see THIRD_PARTY_NOTICES.md and LICENSE-APACHE-2.0.
//! Paint's fixed Diffusers 0.30 UniPC recipe: VP v-prediction, order two,
//! bh2, zero-terminal-SNR scaled-linear betas, trailing timesteps, final sigma0.
//! Reference: diffusers 8a79d8ec scheduling_unipc_multistep.py:75-110,
//! 180-361,453-788,822-901. This is not Wan's flow-matching schedule.
use candle_core::{DType, Result, Tensor};

#[derive(Clone, Debug)]
pub struct PaintSchedule {
    timesteps: Vec<i64>,
    sigmas: Vec<f32>,
    betas: Vec<f32>,
    alphas_cumprod: Vec<f32>,
}
fn cumprod(values: &[f32]) -> Vec<f32> {
    let mut product = 1f64;
    values
        .iter()
        .map(|&v| {
            product *= v as f64;
            product as f32
        })
        .collect()
}
impl PaintSchedule {
    pub fn new(steps: usize) -> Result<Self> {
        if !(1..=1000).contains(&steps) {
            candle_core::bail!("paint sampling requires 1..=1000 steps")
        }
        let (start, end) = (0.00085f64.sqrt() as f32, 0.012f64.sqrt() as f32);
        let delta = (end - start) / 999.;
        let betas = (0..1000)
            .map(|i| {
                let beta = if i < 500 {
                    delta.mul_add(i as f32, start)
                } else {
                    (-delta).mul_add((999 - i) as f32, end)
                };
                beta * beta
            })
            .collect::<Vec<_>>();
        let mut roots = cumprod(&betas.iter().map(|v| 1. - v).collect::<Vec<_>>())
            .into_iter()
            .map(f32::sqrt)
            .collect::<Vec<_>>();
        let (first, last) = (roots[0], roots[999]);
        let factor = first / (first - last);
        for root in &mut roots {
            *root = (*root - last) * factor;
        }
        let bars = roots.into_iter().map(|r| r * r).collect::<Vec<_>>();
        let betas = (0..1000)
            .map(|i| {
                1. - if i == 0 {
                    bars[0]
                } else {
                    bars[i] / bars[i - 1]
                }
            })
            .collect::<Vec<_>>();
        let mut alphas_cumprod = cumprod(&betas.iter().map(|v| 1. - v).collect::<Vec<_>>());
        alphas_cumprod[999] = 2f32.powi(-24);
        let ratio = 1000. / steps as f64;
        // NumPy arange first rounds start+step, then uses that difference as
        // its increment. Its ceil span can also produce one extra entry.
        let count = (1000. / ratio).ceil() as usize;
        let increment = (1000. - ratio) - 1000.;
        let timesteps = (0..count)
            .map(|i| (1000. + i as f64 * increment).round_ties_even() as i64 - 1)
            .collect::<Vec<_>>();
        let mut sigmas = timesteps
            .iter()
            .map(|&t| {
                let alpha = alphas_cumprod[t.clamp(0, 999) as usize];
                ((1. - alpha) / alpha).sqrt()
            })
            .collect::<Vec<_>>();
        sigmas.push(0.);
        Ok(Self {
            timesteps,
            sigmas,
            betas,
            alphas_cumprod,
        })
    }
    pub fn timesteps(&self) -> &[i64] {
        &self.timesteps
    }
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }
    pub fn betas(&self) -> &[f32] {
        &self.betas
    }
    pub fn alphas_cumprod(&self) -> &[f32] {
        &self.alphas_cumprod
    }
}
fn alpha_sigma(sigma: f32) -> (f32, f32) {
    let alpha = 1. / (sigma * sigma + 1.).sqrt();
    (alpha, sigma * alpha)
}
fn lambda(sigma: f32) -> f32 {
    let (a, s) = alpha_sigma(sigma);
    a.ln() - s.ln()
}
fn scaled(x: &Tensor, scale: f32) -> Result<Tensor> {
    // These are LEFT scalar products in Diffusers (coefficient * sample).
    // Torch CPU rounds that zero-dimensional F32 coefficient to half; CUDA's
    // CPU-scalar fastpath retains F32 opmath. Reversing operands on CPU would
    // change its scalar fastpath too. Keep the oracle's actual operand order.
    let scale = if x.device().is_cpu() {
        rho(scale, x.dtype())
    } else {
        scale
    };
    x.to_dtype(DType::F32)?
        .affine(scale as f64, 0.)?
        .to_dtype(x.dtype())
}
fn divided(x: &Tensor, denominator: f32) -> Result<Tensor> {
    // ATen BinaryDivTrueKernel.cu:35-48 replaces a CPU scalar denominator
    // with an F32 reciprocal on CUDA; CPU retains actual division.
    if x.device().is_cuda() {
        return scaled(x, 1. / denominator);
    }
    x.to_dtype(DType::F32)?
        .broadcast_div(&Tensor::new(denominator, x.device())?)?
        .to_dtype(x.dtype())
}
fn rho(value: f32, dtype: DType) -> f32 {
    if dtype == DType::F16 {
        half::f16::from_f32(value).to_f32()
    } else {
        value
    }
}

pub struct PaintUniPc {
    schedule: PaintSchedule,
    next: usize,
    history: Vec<Tensor>,
    last_sample: Option<Tensor>,
    last_order: usize,
}
impl PaintUniPc {
    pub fn new(steps: usize) -> Result<Self> {
        let schedule = PaintSchedule::new(steps)?;
        // Some NumPy trailing grids contain both0 and-1, which interpolate
        // to the same sigma. Upstream divides by zero in its last corrector.
        if schedule.sigmas.windows(2).any(|pair| pair[0] <= pair[1]) {
            candle_core::bail!(
                "paint trailing schedule repeats a noise level; choose a different step count"
            )
        }
        Ok(Self {
            schedule,
            next: 0,
            history: Vec::new(),
            last_sample: None,
            last_order: 0,
        })
    }
    pub fn schedule(&self) -> &PaintSchedule {
        &self.schedule
    }
    pub fn last_x0(&self) -> Option<&Tensor> {
        self.history.last()
    }
    pub fn last_corrected_sample(&self) -> Option<&Tensor> {
        self.last_sample.as_ref()
    }
    fn predictor(&self, sample: &Tensor, history: &[Tensor], order: usize) -> Result<Tensor> {
        let index = self.next;
        let (alpha_t, sigma_t) = alpha_sigma(self.schedule.sigmas[index + 1]);
        let (_, sigma_s) = alpha_sigma(self.schedule.sigmas[index]);
        let lambda_s = lambda(self.schedule.sigmas[index]);
        let h = lambda(self.schedule.sigmas[index + 1]) - lambda_s;
        let phi = (-h).exp_m1();
        let m0 = &history[history.len() - 1];
        let base = (scaled(sample, sigma_t / sigma_s)? - scaled(m0, alpha_t * phi)?)?;
        if order == 1 {
            return Ok(base);
        }
        let rk = (lambda(self.schedule.sigmas[index - 1]) - lambda_s) / h;
        let difference = divided(&(&history[history.len() - 2] - m0)?, rk)?;
        let residual = scaled(&difference, 0.5)?;
        base - scaled(&residual, alpha_t * phi)?
    }
    fn corrector(&self, x0: &Tensor) -> Result<Tensor> {
        let index = self.next;
        let previous = self
            .last_sample
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("missing paint corrector history".into()))?;
        let m0 = self
            .history
            .last()
            .ok_or_else(|| candle_core::Error::Msg("missing paint model history".into()))?;
        let (alpha_t, sigma_t) = alpha_sigma(self.schedule.sigmas[index]);
        let (_, sigma_s) = alpha_sigma(self.schedule.sigmas[index - 1]);
        let lambda_s = lambda(self.schedule.sigmas[index - 1]);
        let h = lambda(self.schedule.sigmas[index]) - lambda_s;
        let hh = -h;
        let phi = hh.exp_m1();
        let base = (scaled(previous, sigma_t / sigma_s)? - scaled(m0, alpha_t * phi)?)?;
        let difference = (x0 - m0)?;
        let correction = if self.last_order == 1 {
            scaled(&difference, 0.5)?
        } else {
            let rk = (lambda(self.schedule.sigmas[index - 2]) - lambda_s) / h;
            let phi2 = phi / hh - 1.;
            let b0 = phi2 / phi;
            let b1 = (phi2 / hh - 0.5) * 2. / phi;
            // Solve [[1,1],[rk,1]] rho=b. Only order two is in this recipe.
            let r1 = (b1 - rk * b0) / (1. - rk);
            let r0 = b0 - r1;
            let d1 = divided(&(&self.history[self.history.len() - 2] - m0)?, rk)?;
            (scaled(&d1, rho(r0, x0.dtype()))? + scaled(&difference, rho(r1, x0.dtype()))?)?
        };
        base - scaled(&correction, alpha_t * phi)?
    }
    /// Consume precisely one schedule entry. Failed validation leaves state intact.
    pub fn step(&mut self, model: &Tensor, timestep: i64, sample: &Tensor) -> Result<Tensor> {
        if self.schedule.timesteps.get(self.next) != Some(&timestep)
            || sample.dims() != model.dims()
            || sample.elem_count() == 0
            || sample.dtype() != model.dtype()
            || !matches!(sample.dtype(), DType::F16 | DType::F32)
            || !sample.device().same_device(model.device())
            || self.last_sample.as_ref().is_some_and(|previous| {
                previous.dims() != sample.dims()
                    || previous.dtype() != sample.dtype()
                    || !previous.device().same_device(sample.device())
            })
        {
            candle_core::bail!("invalid paint UniPC timestep, sample or model output")
        }
        let (alpha, sigma) = alpha_sigma(self.schedule.sigmas[self.next]);
        // Convert BEFORE correcting the current sample, then retain that x0.
        let x0 = (scaled(sample, alpha)? - scaled(model, sigma)?)?;
        let corrected = if self.next > 0 {
            self.corrector(&x0)?
        } else {
            sample.clone()
        };
        let mut history = self.history.clone();
        if history.len() == 2 {
            history.remove(0);
        }
        history.push(x0);
        let order = 2
            .min(self.schedule.timesteps.len() - self.next)
            .min(self.next + 1);
        let result = self.predictor(&corrected, &history, order)?;
        self.history = history;
        self.last_sample = Some(corrected);
        self.last_order = order;
        self.next += 1;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    #[test]
    fn paint_sampler_refuses_degenerate_trailing_schedule() -> Result<()> {
        let schedule = PaintSchedule::new(769)?;
        assert_eq!(&schedule.timesteps()[767..], &[2, 0, -1]);
        assert!(PaintUniPc::new(769).is_err());
        Ok(())
    }
    #[test]
    fn paint_sampler_rejects_out_of_order_without_advancing() -> Result<()> {
        let mut sampler = PaintUniPc::new(2)?;
        let sample = Tensor::ones((1, 4, 2, 2), DType::F32, &Device::Cpu)?;
        let model = sample.zeros_like()?;
        let timesteps = sampler.schedule().timesteps().to_vec();
        assert!(sampler.step(&model, timesteps[1], &sample).is_err());
        assert!(sampler.last_x0().is_none());
        let next = sampler.step(&model, timesteps[0], &sample)?;
        assert!(sampler.step(&model, timesteps[0], &next).is_err());
        assert!(sampler
            .step(&model.flatten_all()?, timesteps[1], &next)
            .is_err());
        let last = sampler.step(&model, timesteps[1], &next)?;
        assert!(sampler.step(&model, timesteps[1], &last).is_err());
        Ok(())
    }
    #[test]
    fn paint_cpu_left_scalar_rounds_to_half() -> Result<()> {
        let x = Tensor::new(&[-9.9765625f32], &Device::Cpu)?.to_dtype(DType::F16)?;
        assert_eq!(
            scaled(&x, 0.3)?.to_dtype(DType::F32)?.to_vec1::<f32>()?,
            vec![-2.9941406]
        );
        Ok(())
    }
    #[test]
    fn paint_training_arrays_match_diffusers() -> Result<()> {
        let fixture = candle_core::safetensors::load_buffer(
            include_bytes!("../../../../tests/fixtures/hunyuan3d/paint-sampler.safetensors"),
            &Device::Cpu,
        )?;
        let schedule = PaintSchedule::new(15)?;
        for (key, actual) in [
            ("betas", schedule.betas()),
            ("alphas_cumprod", schedule.alphas_cumprod()),
        ] {
            let expected = fixture[key].to_vec1::<f32>()?;
            let max = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(max < 1e-6, "{key} error {max}");
        }
        Ok(())
    }
    #[test]
    fn paint_trailing_schedule_matches_numpy_rounding() -> Result<()> {
        assert_eq!(PaintSchedule::new(48)?.timesteps()[3], 936);
        let schedule = PaintSchedule::new(242)?;
        assert_eq!(schedule.timesteps().len(), 243);
        assert_eq!(schedule.timesteps().last(), Some(&-1));
        let alpha = schedule.alphas_cumprod()[0];
        assert_eq!(schedule.sigmas()[242], ((1. - alpha) / alpha).sqrt());
        Ok(())
    }
    #[test]
    fn paint_unipc_matches_diffusers_trajectory() -> Result<()> {
        let fixture = candle_core::safetensors::load_buffer(
            include_bytes!("../../../../tests/fixtures/hunyuan3d/paint-sampler.safetensors"),
            &Device::Cpu,
        )?;
        compare(&fixture)
    }
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires retained CUDA sampler oracle"]
    fn paint_unipc_matches_diffusers_cuda() -> Result<()> {
        let root = std::env::var("MOLD_PAINT_SAMPLER_ORACLE").expect("CUDA oracle directory");
        let fixture = candle_core::safetensors::load(
            std::path::Path::new(&root).join("paint-sampler.safetensors"),
            &Device::new_cuda(0)?,
        )?;
        compare(&fixture)
    }
    fn compare(fixture: &std::collections::HashMap<String, Tensor>) -> Result<()> {
        for label in ["f32", "f16"] {
            for steps in [1, 2, 3, 15, 30, 48] {
                let prefix = format!("{label}.{steps}");
                let mut sampler = PaintUniPc::new(steps)?;
                assert_eq!(
                    sampler.schedule().timesteps(),
                    fixture[&format!("{prefix}.timesteps")].to_vec1::<i64>()?
                );
                let actual_sigmas = sampler.schedule().sigmas();
                let expected_sigmas = fixture[&format!("{prefix}.sigmas")].to_vec1::<f32>()?;
                for (a, b) in actual_sigmas.iter().zip(expected_sigmas) {
                    assert!((a - b).abs() <= 1e-5 * b.abs().max(1.), "sigma {a} vs {b}");
                }
                let mut worst_max = 0f32;
                let mut worst_rms = 0f32;
                let mut sample = fixture[&format!("{prefix}.initial")].clone();
                for (index, timestep) in sampler
                    .schedule()
                    .timesteps()
                    .to_vec()
                    .into_iter()
                    .enumerate()
                {
                    sample = sampler.step(
                        &fixture[&format!("{prefix}.model.{index}")],
                        timestep,
                        &sample,
                    )?;
                    for (kind, actual) in [
                        ("sample", &sample),
                        ("x0", sampler.last_x0().unwrap()),
                        ("corrected", sampler.last_corrected_sample().unwrap()),
                    ] {
                        let expected = &fixture[&format!("{prefix}.{kind}.{index}")];
                        let delta =
                            (actual.to_dtype(DType::F32)? - expected.to_dtype(DType::F32)?)?;
                        let max = delta.abs()?.max_all()?.to_scalar::<f32>()?;
                        let rms = delta.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
                        worst_max = worst_max.max(max);
                        worst_rms = worst_rms.max(rms);
                        let (max_bound, rms_bound) = if label == "f32" {
                            (5e-5, 1e-5)
                        } else {
                            (0.005, 0.002)
                        };
                        assert!(
                            max < max_bound && rms < rms_bound,
                            "{prefix}.{kind}.{index}: max={max} rms={rms}"
                        );
                    }
                }
                eprintln!("paint UniPC {prefix}: max={worst_max} rms={worst_rms}");
            }
        }
        Ok(())
    }
}
