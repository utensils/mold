/// Euler discrete scheduler for FLUX diffusion sampling.
///
/// Implements the noise schedule and step computation for the
/// FLUX flow-matching architecture.
///
/// Stubbed — will be implemented alongside the full FLUX pipeline.
pub struct EulerScheduler {
    pub num_steps: u32,
    pub sigma_min: f64,
    pub sigma_max: f64,
}

impl EulerScheduler {
    pub fn new(num_steps: u32) -> Self {
        Self {
            num_steps,
            sigma_min: 0.0,
            sigma_max: 1.0,
        }
    }

    /// Compute the sigma schedule for the given number of steps.
    pub fn sigmas(&self) -> Vec<f64> {
        let mut sigmas = Vec::with_capacity(self.num_steps as usize + 1);
        for i in 0..=self.num_steps {
            let t = i as f64 / self.num_steps as f64;
            let sigma = self.sigma_max * (1.0 - t) + self.sigma_min * t;
            sigmas.push(sigma);
        }
        sigmas
    }
}
