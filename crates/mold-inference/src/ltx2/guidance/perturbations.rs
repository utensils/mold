use candle_core::{Device, Result, Tensor};

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerturbationType {
    SkipA2VCrossAttention,
    SkipV2ACrossAttention,
    SkipVideoSelfAttention,
    SkipAudioSelfAttention,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Perturbation {
    pub kind: PerturbationType,
    pub blocks: Option<Vec<usize>>,
}

impl Perturbation {
    pub fn new(kind: PerturbationType, blocks: Option<Vec<usize>>) -> Self {
        Self { kind, blocks }
    }

    pub fn is_perturbed(&self, kind: PerturbationType, block: usize) -> bool {
        self.kind == kind
            && self
                .blocks
                .as_ref()
                .is_none_or(|blocks| blocks.contains(&block))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerturbationConfig {
    perturbations: Vec<Perturbation>,
}

impl PerturbationConfig {
    pub fn new(perturbations: Vec<Perturbation>) -> Self {
        Self { perturbations }
    }

    pub fn empty() -> Self {
        Self {
            perturbations: Vec::new(),
        }
    }

    pub fn is_perturbed(&self, kind: PerturbationType, block: usize) -> bool {
        self.perturbations
            .iter()
            .any(|perturbation| perturbation.is_perturbed(kind, block))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchedPerturbationConfig {
    perturbations: Vec<PerturbationConfig>,
}

impl BatchedPerturbationConfig {
    pub fn new(perturbations: Vec<PerturbationConfig>) -> Self {
        Self { perturbations }
    }

    pub fn empty(batch_size: usize) -> Self {
        Self {
            perturbations: (0..batch_size)
                .map(|_| PerturbationConfig::empty())
                .collect(),
        }
    }

    pub fn mask_values(&self, kind: PerturbationType, block: usize) -> Vec<f32> {
        self.perturbations
            .iter()
            .map(|perturbation| {
                if perturbation.is_perturbed(kind, block) {
                    0.0
                } else {
                    1.0
                }
            })
            .collect()
    }

    pub fn mask_tensor(
        &self,
        kind: PerturbationType,
        block: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let values = self.mask_values(kind, block);
        Tensor::from_vec(values, self.perturbations.len(), device)
    }

    pub fn mask_like(
        &self,
        kind: PerturbationType,
        block: usize,
        values: &Tensor,
    ) -> Result<Tensor> {
        let mask = self
            .mask_tensor(kind, block, values.device())?
            .to_dtype(values.dtype())?;
        let mut shape = vec![mask.dim(0)?];
        shape.extend(std::iter::repeat_n(1usize, values.rank().saturating_sub(1)));
        mask.reshape(shape)
    }

    pub fn any_in_batch(&self, kind: PerturbationType, block: usize) -> bool {
        self.perturbations
            .iter()
            .any(|perturbation| perturbation.is_perturbed(kind, block))
    }

    pub fn all_in_batch(&self, kind: PerturbationType, block: usize) -> bool {
        self.perturbations
            .iter()
            .all(|perturbation| perturbation.is_perturbed(kind, block))
    }
}
