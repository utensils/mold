#![allow(dead_code)]

mod perturbations;

use std::collections::BTreeMap;

use anyhow::Result;
use candle_core::Tensor;

#[allow(unused_imports)]
pub use perturbations::{
    BatchedPerturbationConfig, Perturbation, PerturbationConfig, PerturbationType,
};

#[derive(Debug, Clone, PartialEq)]
pub struct MultiModalGuiderParams {
    pub cfg_scale: f64,
    pub stg_scale: f64,
    pub stg_blocks: Vec<usize>,
    pub rescale_scale: f64,
    pub modality_scale: f64,
    pub skip_step: usize,
}

impl Default for MultiModalGuiderParams {
    fn default() -> Self {
        Self {
            cfg_scale: 1.0,
            stg_scale: 0.0,
            stg_blocks: Vec::new(),
            rescale_scale: 0.0,
            modality_scale: 1.0,
            skip_step: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MultiModalGuider {
    pub params: MultiModalGuiderParams,
    pub negative_context: Option<Tensor>,
}

impl MultiModalGuider {
    pub fn new(params: MultiModalGuiderParams, negative_context: Option<Tensor>) -> Self {
        Self {
            params,
            negative_context,
        }
    }

    pub fn calculate(
        &self,
        cond: &Tensor,
        uncond_text: &Tensor,
        uncond_perturbed: &Tensor,
        uncond_modality: &Tensor,
    ) -> Result<Tensor> {
        let cfg = (cond.broadcast_sub(uncond_text)? * (self.params.cfg_scale - 1.0))?;
        let stg = (cond.broadcast_sub(uncond_perturbed)? * self.params.stg_scale)?;
        let modality =
            (cond.broadcast_sub(uncond_modality)? * (self.params.modality_scale - 1.0))?;
        let mut pred = cond.broadcast_add(&cfg)?;
        pred = pred.broadcast_add(&stg)?;
        pred = pred.broadcast_add(&modality)?;

        if self.params.rescale_scale != 0.0 {
            let cond_std = tensor_std(cond)?;
            let pred_std = tensor_std(&pred)?;
            let factor = self.params.rescale_scale
                * (cond_std as f64 / pred_std.max(f32::EPSILON) as f64)
                + (1.0 - self.params.rescale_scale);
            pred = (pred * factor)?;
        }

        Ok(pred)
    }

    pub fn do_unconditional_generation(&self) -> bool {
        !approx_eq(self.params.cfg_scale, 1.0)
    }

    pub fn do_perturbed_generation(&self) -> bool {
        !approx_eq(self.params.stg_scale, 0.0)
    }

    pub fn do_isolated_modality_generation(&self) -> bool {
        !approx_eq(self.params.modality_scale, 1.0)
    }

    pub fn should_skip_step(&self, step: usize) -> bool {
        self.params.skip_step != 0 && step % (self.params.skip_step + 1) != 0
    }
}

#[derive(Debug, Clone)]
pub struct MultiModalGuiderFactory {
    negative_context: Option<Tensor>,
    params_by_sigma: Vec<(f32, MultiModalGuiderParams)>,
}

impl MultiModalGuiderFactory {
    pub fn constant(params: MultiModalGuiderParams, negative_context: Option<Tensor>) -> Self {
        Self {
            negative_context,
            params_by_sigma: vec![(f32::INFINITY, params)],
        }
    }

    pub fn from_dict(
        params_by_sigma: BTreeMap<OrderedSigma, MultiModalGuiderParams>,
        negative_context: Option<Tensor>,
    ) -> Self {
        let mut entries = params_by_sigma
            .into_iter()
            .map(|(sigma, params)| (sigma.0, params))
            .collect::<Vec<_>>();
        entries.sort_by(|lhs, rhs| rhs.0.total_cmp(&lhs.0));
        Self {
            negative_context,
            params_by_sigma: entries,
        }
    }

    pub fn params(&self, sigma: f32) -> &MultiModalGuiderParams {
        self.params_by_sigma
            .iter()
            .filter(|(upper_bound, _)| *upper_bound >= sigma)
            .next_back()
            .or_else(|| self.params_by_sigma.first())
            .map(|(_, params)| params)
            .expect("guider factory requires at least one sigma bin")
    }

    pub fn build_from_sigma(&self, sigma: f32) -> MultiModalGuider {
        MultiModalGuider::new(self.params(sigma).clone(), self.negative_context.clone())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrderedSigma(pub f32);

impl Eq for OrderedSigma {}

impl Ord for OrderedSigma {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd for OrderedSigma {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn approx_eq(lhs: f64, rhs: f64) -> bool {
    (lhs - rhs).abs() < 1e-6
}

fn tensor_std(tensor: &Tensor) -> Result<f32> {
    let flat = tensor.flatten_all()?;
    let mean = flat.mean_all()?.to_scalar::<f32>()?;
    let variance = flat
        .affine(1.0, -(mean as f64))?
        .sqr()?
        .mean_all()?
        .to_scalar::<f32>()?;
    Ok(variance.max(0.0).sqrt())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use candle_core::{Device, Tensor};

    use super::{
        BatchedPerturbationConfig, MultiModalGuider, MultiModalGuiderFactory,
        MultiModalGuiderParams, OrderedSigma, Perturbation, PerturbationConfig,
        PerturbationType,
    };

    #[test]
    fn simple_denoiser_guidance_reduces_to_cond_when_disabled() {
        let device = Device::Cpu;
        let cond = Tensor::new(&[1f32, 2.0], &device).unwrap();
        let guider = MultiModalGuider::new(MultiModalGuiderParams::default(), None);
        let out = guider.calculate(&cond, &cond, &cond, &cond).unwrap();
        assert_eq!(out.to_vec1::<f32>().unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn guided_denoiser_applies_cfg_and_stg_deltas() {
        let device = Device::Cpu;
        let cond = Tensor::new(&[5f32, 10.0], &device).unwrap();
        let uncond = Tensor::new(&[3f32, 4.0], &device).unwrap();
        let perturbed = Tensor::new(&[4f32, 8.0], &device).unwrap();
        let modality = Tensor::new(&[2f32, 6.0], &device).unwrap();
        let guider = MultiModalGuider::new(
            MultiModalGuiderParams {
                cfg_scale: 2.0,
                stg_scale: 0.5,
                modality_scale: 1.5,
                ..Default::default()
            },
            None,
        );

        let out = guider
            .calculate(&cond, &uncond, &perturbed, &modality)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(out, vec![9.0, 19.0]);
    }

    #[test]
    fn modality_isolation_detection_matches_scale() {
        let guider = MultiModalGuider::new(
            MultiModalGuiderParams {
                modality_scale: 1.5,
                ..Default::default()
            },
            None,
        );
        assert!(guider.do_isolated_modality_generation());
        assert!(!guider.do_perturbed_generation());
        assert!(!guider.do_unconditional_generation());
    }

    #[test]
    fn skip_step_logic_matches_upstream_contract() {
        let guider = MultiModalGuider::new(
            MultiModalGuiderParams {
                skip_step: 2,
                ..Default::default()
            },
            None,
        );
        assert!(!guider.should_skip_step(0));
        assert!(guider.should_skip_step(1));
        assert!(guider.should_skip_step(2));
        assert!(!guider.should_skip_step(3));
    }

    #[test]
    fn guider_factory_picks_sigma_bin() {
        let mut bins = BTreeMap::new();
        bins.insert(
            OrderedSigma(1.0),
            MultiModalGuiderParams {
                cfg_scale: 3.0,
                ..Default::default()
            },
        );
        bins.insert(
            OrderedSigma(0.5),
            MultiModalGuiderParams {
                cfg_scale: 2.0,
                ..Default::default()
            },
        );
        let factory = MultiModalGuiderFactory::from_dict(bins, None);
        assert_eq!(factory.params(0.75).cfg_scale, 3.0);
        assert_eq!(factory.params(0.49).cfg_scale, 2.0);
    }

    #[test]
    fn batched_perturbations_mark_expected_samples() {
        let config = BatchedPerturbationConfig::new(vec![
            PerturbationConfig::new(vec![Perturbation::new(
                PerturbationType::SkipVideoSelfAttention,
                Some(vec![4]),
            )]),
            PerturbationConfig::empty(),
        ]);
        assert!(config.any_in_batch(PerturbationType::SkipVideoSelfAttention, 4));
        assert!(!config.all_in_batch(PerturbationType::SkipVideoSelfAttention, 4));
        assert_eq!(
            config
                .mask_values(PerturbationType::SkipVideoSelfAttention, 4),
            vec![0.0, 1.0]
        );
    }
}
