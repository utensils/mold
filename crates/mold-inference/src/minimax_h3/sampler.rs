//! Exact dual-modality rectified-flow schedule used by MiniMax H3.
//!
//! H3 advances generated video and audio rows through one transformer forward,
//! but the two modalities use different shifted sigma grids. The iterator in
//! this module therefore yields one immutable descriptor per *coupled* forward;
//! callers can check cancellation before requesting the next descriptor and
//! only replace their two latent tensors after the selected paired update
//! succeeds. Official BF16 uses first-order Euler; the compact Comfy layout
//! uses the deterministic RES multistep rule from its released workflow.
//!
//! `grid_points` deliberately includes the terminal zero. The official default
//! is 50 grid points and thus 49 transformer evaluations, not 50 evaluations.

use std::iter::FusedIterator;

use anyhow::{bail, Result};
use candle_core::{DType, Tensor};

pub(crate) const H3_DEFAULT_GRID_POINTS: usize = 50;
pub(crate) const H3_VIDEO_SHIFT: f32 = 12.0;
pub(crate) const H3_AUDIO_SHIFT: f32 = 3.0;
const H3_COMFY_AUDIO_SCALE: f32 = H3_VIDEO_SHIFT / H3_AUDIO_SHIFT;
pub(crate) const H3_VISUAL_CONDITION_TIMESTEP: f32 = 0.999;
pub(crate) const H3_CLEAN_TIMESTEP: f32 = 1.0;

/// Sampling integration selected by the authenticated checkpoint layout.
///
/// The released BF16 pipeline uses first-order Euler. The official ComfyUI
/// deployment templates pair their pruned INT8 checkpoints with the
/// deterministic second-order RES multistep integrator instead.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum H3SamplerKind {
    #[default]
    OfficialEuler,
    ComfyResMultistep,
}

impl H3SamplerKind {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::OfficialEuler => "official-euler",
            Self::ComfyResMultistep => "comfy-res-multistep",
        }
    }
}

/// One modality's authority for a coupled transformer evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct H3FlowStep {
    /// Current noise level.
    pub sigma: f32,
    /// Noise level after this deterministic update.
    pub next_sigma: f32,
    /// AdaLN timestep supplied to the transformer: `1 - sigma`.
    pub timestep: f32,
}

/// Timestep values assigned to the five packed-row roles during one forward.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct H3RowTimesteps {
    /// Text rows do not have an output head and inherit generated video time.
    pub text: f32,
    pub generated_video: f32,
    /// Visual anchors use at least the official near-clean value; this is a
    /// `max(generated_video, 0.999)` in the pinned reference implementation.
    pub visual_condition: f32,
    pub generated_audio: f32,
    /// Audio references are fully clean and never denoised.
    pub audio_reference: f32,
}

/// Unambiguous user/API and execution counts for one schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct H3StepCount {
    /// Requested sigma grid points, including terminal zero.
    pub requested_grid_points: usize,
    /// Grid points after consecutive float32 collisions are collapsed.
    pub effective_grid_points: usize,
    /// Coupled DiT forwards, always `effective_grid_points - 1`.
    pub transformer_evaluations: usize,
}

/// Progress before or after a coupled transformer evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct H3DenoiseProgress {
    pub completed_evaluations: usize,
    pub total_evaluations: usize,
}

impl H3DenoiseProgress {
    pub fn is_complete(self) -> bool {
        self.completed_evaluations == self.total_evaluations
    }
}

/// One cancellation-safe unit of H3 denoising work.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct H3DualScheduleStep {
    /// Zero-based index of the coupled transformer forward.
    pub evaluation_index: usize,
    pub video: H3FlowStep,
    pub audio: H3FlowStep,
    pub row_timesteps: H3RowTimesteps,
    total_evaluations: usize,
}

impl H3DualScheduleStep {
    /// Progress to report immediately before the forward (and cancellation
    /// boundary) represented by this descriptor.
    pub fn progress_before_forward(self) -> H3DenoiseProgress {
        H3DenoiseProgress {
            completed_evaluations: self.evaluation_index,
            total_evaluations: self.total_evaluations,
        }
    }

    /// Progress to report only after both video and audio updates commit.
    pub fn progress_after_update(self) -> H3DenoiseProgress {
        H3DenoiseProgress {
            completed_evaluations: self.evaluation_index + 1,
            total_evaluations: self.total_evaluations,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct H3ModalitySchedule {
    sigmas: Vec<f32>,
}

fn unique_consecutive(values: impl IntoIterator<Item = f32>) -> Vec<f32> {
    let mut unique = Vec::new();
    for value in values {
        if unique.last().copied() != Some(value) {
            unique.push(value);
        }
    }
    unique
}

impl H3ModalitySchedule {
    fn new(grid_points: usize, shift: f32) -> Result<Self> {
        if grid_points < 2 {
            bail!("MiniMax H3 schedule needs at least two grid points, got {grid_points}");
        }
        if !shift.is_finite() || shift <= 0.0 {
            bail!("MiniMax H3 schedule shift must be finite and positive, got {shift}");
        }

        let denominator = (grid_points - 1) as f64;
        let shifted = (0..grid_points).map(|index| {
            // Spell out both endpoints so terminal zero and initial one remain
            // exact. The pinned float32 linspace rounds the result of the
            // double-precision index division, rather than dividing two f32
            // values; the distinction is one ulp for some short grids.
            let base = match index {
                0 => 1.0,
                index if index + 1 == grid_points => 0.0,
                _ => (1.0 - index as f64 / denominator) as f32,
            };
            let numerator = shift * base;
            let denominator = 1.0 + (shift - 1.0) * base;
            numerator / denominator
        });
        let sigmas = unique_consecutive(shifted);

        if sigmas.len() < 2
            || sigmas.first().copied() != Some(1.0)
            || sigmas.last().copied() != Some(0.0)
            || !sigmas.windows(2).all(|pair| pair[1] < pair[0])
        {
            bail!("MiniMax H3 shift {shift} produced an invalid sigma grid");
        }

        Ok(Self { sigmas })
    }

    fn new_comfy_simple(grid_points: usize, shift: f32) -> Result<Self> {
        if grid_points < 2 {
            bail!("MiniMax H3 schedule needs at least two grid points, got {grid_points}");
        }
        let evaluations = grid_points - 1;
        let shifted = (0..evaluations).map(|index| {
            // Comfy's BasicScheduler("simple") samples its 1000-entry flow
            // table at integer floor(index * 1000 / evaluations), then appends
            // terminal zero. Preserve that discrete table for every count.
            let table_index = index.saturating_mul(1_000) / evaluations;
            let base = (1_000 - table_index) as f32 / 1_000.0;
            shift * base / (1.0 + (shift - 1.0) * base)
        });
        let sigmas = unique_consecutive(shifted.chain(std::iter::once(0.0)));
        if sigmas.len() < 2
            || sigmas.first().copied() != Some(1.0)
            || sigmas.last().copied() != Some(0.0)
            || !sigmas.windows(2).all(|pair| pair[1] < pair[0])
        {
            bail!("MiniMax H3 Comfy simple schedule produced an invalid sigma grid");
        }
        Ok(Self { sigmas })
    }

    fn evaluations(&self) -> usize {
        self.sigmas.len() - 1
    }

    fn step(&self, index: usize) -> H3FlowStep {
        let sigma = self.sigmas[index];
        H3FlowStep {
            sigma,
            next_sigma: self.sigmas[index + 1],
            timestep: 1.0 - sigma,
        }
    }
}

/// H3's synchronized video/audio schedule.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct H3DualSchedule {
    requested_grid_points: usize,
    video: H3ModalitySchedule,
    audio: H3ModalitySchedule,
}

impl H3DualSchedule {
    /// Build the exact released-checkpoint schedule: shift 12 for video and 3
    /// for audio, both derived from the same terminal-inclusive base grid.
    pub fn new(grid_points: usize) -> Result<Self> {
        let video = H3ModalitySchedule::new(grid_points, H3_VIDEO_SHIFT)?;
        let audio = H3ModalitySchedule::new(grid_points, H3_AUDIO_SHIFT)?;
        if video.evaluations() != audio.evaluations() {
            bail!(
                "MiniMax H3 shifted grids collapsed to different lengths (video {}, audio {}); cannot synchronize one-forward updates",
                video.evaluations(),
                audio.evaluations()
            );
        }
        Ok(Self {
            requested_grid_points: grid_points,
            video,
            audio,
        })
    }

    pub fn new_for_sampler(grid_points: usize, kind: H3SamplerKind) -> Result<Self> {
        if kind == H3SamplerKind::OfficialEuler {
            return Self::new(grid_points);
        }
        let video = H3ModalitySchedule::new_comfy_simple(grid_points, H3_VIDEO_SHIFT)?;
        let audio = H3ModalitySchedule {
            // The released model derives native audio sigma from the already
            // rounded video-table sigma, rather than independently sampling a
            // shift-3 table. Preserve that f32 inversion/reapplication order.
            sigmas: video
                .sigmas
                .iter()
                .copied()
                .map(|sigma| {
                    if sigma == 0.0 {
                        return 0.0;
                    }
                    let base = sigma / (H3_VIDEO_SHIFT + sigma * (1.0 - H3_VIDEO_SHIFT));
                    H3_AUDIO_SHIFT * base / (1.0 + (H3_AUDIO_SHIFT - 1.0) * base)
                })
                .collect(),
        };
        if video.evaluations() != audio.evaluations() {
            bail!("MiniMax H3 Comfy shifted grids cannot synchronize one-forward updates");
        }
        Ok(Self {
            requested_grid_points: grid_points,
            video,
            audio,
        })
    }

    pub fn official_default() -> Self {
        Self::new(H3_DEFAULT_GRID_POINTS).expect("the official H3 schedule is valid")
    }

    pub fn counts(&self) -> H3StepCount {
        H3StepCount {
            requested_grid_points: self.requested_grid_points,
            effective_grid_points: self.video.sigmas.len(),
            transformer_evaluations: self.video.evaluations(),
        }
    }

    pub fn video_sigmas(&self) -> &[f32] {
        &self.video.sigmas
    }

    pub fn audio_sigmas(&self) -> &[f32] {
        &self.audio.sigmas
    }

    /// Iterate one descriptor per coupled transformer forward. A caller checks
    /// cancellation before each `next()`/forward and never stores a partial AV
    /// result: [`H3DualSampler::step_pair`] returns both next tensors or an
    /// error.
    pub fn steps(&self) -> H3DualScheduleIter<'_> {
        H3DualScheduleIter {
            schedule: self,
            next_index: 0,
        }
    }

    fn step(&self, index: usize) -> H3DualScheduleStep {
        let video = self.video.step(index);
        let audio = self.audio.step(index);
        H3DualScheduleStep {
            evaluation_index: index,
            video,
            audio,
            row_timesteps: H3RowTimesteps {
                text: video.timestep,
                generated_video: video.timestep,
                visual_condition: video.timestep.max(H3_VISUAL_CONDITION_TIMESTEP),
                generated_audio: audio.timestep,
                audio_reference: H3_CLEAN_TIMESTEP,
            },
            total_evaluations: self.video.evaluations(),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct H3DualScheduleIter<'a> {
    schedule: &'a H3DualSchedule,
    next_index: usize,
}

impl Iterator for H3DualScheduleIter<'_> {
    type Item = H3DualScheduleStep;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index >= self.schedule.video.evaluations() {
            return None;
        }
        let step = self.schedule.step(self.next_index);
        self.next_index += 1;
        Some(step)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.schedule.video.evaluations() - self.next_index;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for H3DualScheduleIter<'_> {}
impl FusedIterator for H3DualScheduleIter<'_> {}

/// Apply one data-ward H3 Euler update, then restore the latent dtype.
///
/// The transformer predicts `v` toward clean data, so `x0 = x + sigma * v`.
/// H3 then interpolates exactly to the next sigma without ancestral noise:
/// `x_next = ratio*x + (1-ratio)*x0`, `ratio = next_sigma/sigma`.
///
/// The pinned scheduler first casts `t` to the latent dtype and recovers
/// `sigma = 1 - t` there. The H3 pipeline supplies an FP32 model output, so
/// that recovered value and the latent are promoted for `x0`; the ratio blend
/// also stays FP32 for half-precision latents. Reproducing that ordering is
/// important near the terminal end of the schedule, where the half-precision
/// `1 - (1 - sigma)` round trip is observably different from `sigma`.
pub(crate) fn euler_step(sample: &Tensor, velocity: &Tensor, step: H3FlowStep) -> Result<Tensor> {
    if sample.dims() != velocity.dims() {
        bail!(
            "MiniMax H3 sample/velocity shape mismatch: {:?} versus {:?}",
            sample.dims(),
            velocity.dims()
        );
    }
    if step.sigma <= 0.0
        || step.next_sigma < 0.0
        || step.next_sigma >= step.sigma
        || !step.sigma.is_finite()
        || !step.next_sigma.is_finite()
    {
        bail!(
            "MiniMax H3 Euler step needs finite decreasing sigmas, got {} -> {}",
            step.sigma,
            step.next_sigma
        );
    }
    let expected_timestep = 1.0 - step.sigma;
    if step.timestep.to_bits() != expected_timestep.to_bits() {
        bail!(
            "MiniMax H3 timestep must be exactly 1 - sigma, got {} for sigma {}",
            step.timestep,
            step.sigma
        );
    }

    let output_dtype = sample.dtype();
    if !matches!(output_dtype, DType::F16 | DType::BF16 | DType::F32) {
        bail!(
            "MiniMax H3 latents must use f16, bf16, or f32, got {}",
            output_dtype.as_str()
        );
    }
    let velocity_dtype = velocity.dtype();
    if !matches!(velocity_dtype, DType::F16 | DType::BF16 | DType::F32) {
        bail!(
            "MiniMax H3 velocity must use f16, bf16, or f32, got {}",
            velocity_dtype.as_str()
        );
    }

    // Match `1 - timestep.to(dtype=sample.dtype)` before the model output
    // promotes the x0 expression to FP32.
    let one = Tensor::new(1.0f32, sample.device())?.to_dtype(output_dtype)?;
    let timestep = Tensor::new(step.timestep, sample.device())?.to_dtype(output_dtype)?;
    let sigma_from_timestep = one.sub(&timestep)?.to_dtype(DType::F32)?;
    let sample = sample.to_dtype(DType::F32)?;
    let velocity = velocity.to_dtype(DType::F32)?;
    let x0 = sample.add(&velocity.broadcast_mul(&sigma_from_timestep)?)?;
    let ratio = (step.next_sigma / step.sigma) as f64;
    sample
        .affine(ratio, 0.0)?
        .add(&x0.affine(1.0 - ratio, 0.0)?)?
        .to_dtype(output_dtype)
        .map_err(Into::into)
}

/// Compute both modality updates before the caller replaces either latent.
pub(crate) fn euler_step_pair(
    video_sample: &Tensor,
    audio_sample: &Tensor,
    video_velocity: &Tensor,
    audio_velocity: &Tensor,
    step: H3DualScheduleStep,
) -> Result<(Tensor, Tensor)> {
    let next_video = euler_step(video_sample, video_velocity, step.video)?;
    let next_audio = euler_step(audio_sample, audio_velocity, step.audio)?;
    Ok((next_video, next_audio))
}

fn denoised_from_velocity(sample: &Tensor, velocity: &Tensor, step: H3FlowStep) -> Result<Tensor> {
    if sample.dims() != velocity.dims() {
        bail!(
            "MiniMax H3 sample/velocity shape mismatch: {:?} versus {:?}",
            sample.dims(),
            velocity.dims()
        );
    }
    let output_dtype = sample.dtype();
    if !matches!(output_dtype, DType::F16 | DType::BF16 | DType::F32) {
        bail!(
            "MiniMax H3 latents must use f16, bf16, or f32, got {}",
            output_dtype.as_str()
        );
    }
    if !matches!(velocity.dtype(), DType::F16 | DType::BF16 | DType::F32) {
        bail!(
            "MiniMax H3 velocity must use f16, bf16, or f32, got {}",
            velocity.dtype().as_str()
        );
    }

    // The Comfy model negates the released checkpoint output before its CONST
    // wrapper computes `x - output * sigma`. Mold retains the checkpoint's
    // data-ward sign, so the same clean estimate is `x + velocity * sigma`.
    let sigma = Tensor::new(step.sigma, sample.device())?;
    sample
        .to_dtype(DType::F32)?
        .add(&velocity.to_dtype(DType::F32)?.broadcast_mul(&sigma)?)
        .map_err(Into::into)
}

fn res_multistep_update(
    sample: &Tensor,
    denoised: &Tensor,
    previous_denoised: Option<&Tensor>,
    previous_sigma: Option<f32>,
    step: H3FlowStep,
) -> Result<Tensor> {
    // ComfyUI deliberately uses the same eager-F32 Euler expression for the
    // first and terminal updates. Spell out to_d and dt in its operation order
    // instead of relying on an algebraically equivalent blend.
    if previous_denoised.is_none() || step.next_sigma == 0.0 {
        let sample_f32 = sample.to_dtype(DType::F32)?;
        let sigma = Tensor::new(step.sigma, sample.device())?;
        let dt = Tensor::new(step.next_sigma - step.sigma, sample.device())?;
        let derivative = sample_f32.sub(denoised)?.broadcast_div(&sigma)?;
        return sample_f32
            .add(&derivative.broadcast_mul(&dt)?)?
            .to_dtype(sample.dtype())
            .map_err(Into::into);
    }
    let previous_denoised = previous_denoised.expect("history checked above");
    let previous_sigma = previous_sigma
        .ok_or_else(|| anyhow::anyhow!("MiniMax H3 RES multistep history lacks its sigma"))?;
    if !(previous_sigma > step.sigma && step.sigma > step.next_sigma) {
        bail!(
            "MiniMax H3 RES multistep needs decreasing historical sigmas, got {previous_sigma} -> {} -> {}",
            step.sigma,
            step.next_sigma
        );
    }

    // ComfyUI evaluates these scalar tensor expressions in float32 because
    // the `simple` scheduler supplies float32 sigmas.
    let t = -step.sigma.ln();
    let t_previous = -previous_sigma.ln();
    let t_next = -step.next_sigma.ln();
    let h = t_next - t;
    let c2 = (t_previous - t) / h;
    let phi1 = (-h).exp_m1() / -h;
    let phi2 = (phi1 - 1.0) / -h;
    let b1 = phi1 - phi2 / c2;
    let b2 = phi2 / c2;
    if ![h, c2, b1, b2].into_iter().all(f32::is_finite) {
        bail!("MiniMax H3 RES multistep produced non-finite integration coefficients");
    }

    let history = denoised
        .affine(f64::from(b1), 0.0)?
        .add(&previous_denoised.affine(f64::from(b2), 0.0)?)?
        .affine(f64::from(h), 0.0)?;
    sample
        .to_dtype(DType::F32)?
        .affine(f64::from((-h).exp()), 0.0)?
        .add(&history)?
        .to_dtype(sample.dtype())
        .map_err(Into::into)
}

fn comfy_audio_carried_sample(sample: &Tensor, step: H3DualScheduleStep) -> Result<Tensor> {
    let carry_scale = step.video.sigma / step.audio.sigma;
    sample
        .affine(f64::from(carry_scale), 0.0)
        .map_err(Into::into)
}

fn comfy_audio_carried_velocity(
    native_sample: &Tensor,
    native_velocity: &Tensor,
    step: H3DualScheduleStep,
) -> Result<Tensor> {
    // Comfy's ModelSamplingAV carries audio on the video sigma coordinate and
    // converts the model's native d(audio)/d(sigma_a) back to d(carry)/d(sigma_v).
    // Keep this expression in F32, matching its released wrapper.
    native_sample
        .to_dtype(DType::F32)?
        .affine(f64::from(H3_COMFY_AUDIO_SCALE - 1.0), 0.0)?
        .add(&native_velocity.to_dtype(DType::F32)?.affine(
            f64::from(1.0 + (H3_COMFY_AUDIO_SCALE - 1.0) * step.audio.sigma),
            0.0,
        )?)
        .map_err(Into::into)
}

fn comfy_audio_native_sample(carried: &Tensor, step: H3DualScheduleStep) -> Result<Tensor> {
    let native_scale = if step.video.next_sigma == 0.0 {
        // lim sigma_a/sigma_v as the shared base grid approaches zero.
        H3_AUDIO_SHIFT / H3_VIDEO_SHIFT
    } else {
        step.audio.next_sigma / step.video.next_sigma
    };
    carried
        .affine(f64::from(native_scale), 0.0)
        .map_err(Into::into)
}

/// Stateful synchronized sampler for the video/audio pair.
pub(crate) struct H3DualSampler {
    kind: H3SamplerKind,
    previous_video_denoised: Option<Tensor>,
    previous_audio_denoised: Option<Tensor>,
    current_audio_carried: Option<Tensor>,
    previous_video_sigma: Option<f32>,
    next_evaluation_index: usize,
}

impl H3DualSampler {
    pub(crate) fn new(kind: H3SamplerKind) -> Self {
        Self {
            kind,
            previous_video_denoised: None,
            previous_audio_denoised: None,
            current_audio_carried: None,
            previous_video_sigma: None,
            next_evaluation_index: 0,
        }
    }

    pub(crate) fn step_pair(
        &mut self,
        video_sample: &Tensor,
        audio_sample: &Tensor,
        video_velocity: &Tensor,
        audio_velocity: &Tensor,
        step: H3DualScheduleStep,
    ) -> Result<(Tensor, Tensor)> {
        if step.evaluation_index != self.next_evaluation_index {
            bail!(
                "MiniMax H3 sampler expected evaluation {}, got {}",
                self.next_evaluation_index,
                step.evaluation_index
            );
        }
        if self.kind == H3SamplerKind::OfficialEuler {
            let next = euler_step_pair(
                video_sample,
                audio_sample,
                video_velocity,
                audio_velocity,
                step,
            )?;
            self.next_evaluation_index += 1;
            return Ok(next);
        }

        let video_denoised = denoised_from_velocity(video_sample, video_velocity, step.video)?;
        let initial_audio_carried;
        let audio_carried = if let Some(carried) = self.current_audio_carried.as_ref() {
            carried
        } else {
            initial_audio_carried = comfy_audio_carried_sample(audio_sample, step)?;
            &initial_audio_carried
        };
        let audio_carried_velocity =
            comfy_audio_carried_velocity(audio_sample, audio_velocity, step)?;
        let audio_denoised =
            denoised_from_velocity(audio_carried, &audio_carried_velocity, step.video)?;
        let next_video = res_multistep_update(
            video_sample,
            &video_denoised,
            self.previous_video_denoised.as_ref(),
            self.previous_video_sigma,
            step.video,
        )?;
        let next_audio_carried = res_multistep_update(
            audio_carried,
            &audio_denoised,
            self.previous_audio_denoised.as_ref(),
            self.previous_video_sigma,
            step.video,
        )?;
        let next_audio =
            comfy_audio_native_sample(&next_audio_carried, step)?.to_dtype(audio_sample.dtype())?;

        self.previous_video_denoised = Some(video_denoised);
        self.previous_audio_denoised = Some(audio_denoised);
        self.current_audio_carried = Some(next_audio_carried);
        self.previous_video_sigma = Some(step.video.sigma);
        self.next_evaluation_index += 1;
        Ok((next_video, next_audio))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};
    use serde::Deserialize;

    use super::{
        euler_step, euler_step_pair, unique_consecutive, H3DualSampler, H3DualSchedule, H3FlowStep,
        H3SamplerKind, H3StepCount, H3_AUDIO_SHIFT, H3_CLEAN_TIMESTEP, H3_DEFAULT_GRID_POINTS,
        H3_VIDEO_SHIFT, H3_VISUAL_CONDITION_TIMESTEP,
    };

    #[derive(Debug, Deserialize)]
    struct SyntheticFixture {
        scheduler: SyntheticScheduler,
        coupled_update: SyntheticCoupledUpdate,
    }

    #[derive(Debug, Deserialize)]
    struct SyntheticScheduler {
        grid_points: usize,
        transformer_evaluations: usize,
        video_shift: f32,
        audio_shift: f32,
        video_sigmas: Vec<f32>,
        audio_sigmas: Vec<f32>,
        video_timesteps: Vec<f32>,
        audio_timesteps: Vec<f32>,
        terminal_zero_included: bool,
        ancestral_noise_reinjected: bool,
    }

    #[derive(Debug, Deserialize)]
    struct SyntheticCoupledUpdate {
        step_index: usize,
        sample: Vec<f32>,
        video_velocity: Vec<f32>,
        audio_velocity: Vec<f32>,
        video_next: Vec<f32>,
        audio_next: Vec<f32>,
        velocity_sign: String,
        update_dtype: String,
    }

    #[derive(Debug, Deserialize)]
    struct SyntheticLayerDocument {
        outputs: Vec<SyntheticLayerTensor>,
    }

    #[derive(Debug, Deserialize)]
    struct SyntheticLayerTensor {
        key: String,
        shape: Vec<usize>,
        dtype: String,
        samples: Vec<SyntheticLayerSample>,
    }

    #[derive(Debug, Deserialize)]
    struct SyntheticLayerSample {
        index: Vec<usize>,
        value: f32,
    }

    fn synthetic_fixture() -> SyntheticFixture {
        serde_json::from_str(include_str!(
            "../../../../tests/fixtures/minimax_h3/synthetic-v1.json"
        ))
        .expect("the checked-in MiniMax H3 synthetic fixture must be valid")
    }

    fn synthetic_oracle_values(key: &str) -> Vec<f32> {
        let document: SyntheticLayerDocument = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/minimax_h3/synthetic-oracle-v1.json"
        ))
        .expect("the checked-in MiniMax H3 synthetic layer oracle must be valid");
        let output = document
            .outputs
            .into_iter()
            .find(|output| output.key == key)
            .unwrap_or_else(|| panic!("synthetic layer oracle lacks {key}"));
        assert_eq!(output.dtype, "float32");
        assert_eq!(output.shape, [output.samples.len()]);
        output
            .samples
            .into_iter()
            .enumerate()
            .map(|(offset, sample)| {
                assert_eq!(sample.index, [offset]);
                sample.value
            })
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "value {index}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn exact_dual_shifts_apply_to_one_terminal_inclusive_grid() {
        assert_eq!(H3_VIDEO_SHIFT, 12.0);
        assert_eq!(H3_AUDIO_SHIFT, 3.0);

        let schedule = H3DualSchedule::new(5).unwrap();
        assert_close(
            schedule.video_sigmas(),
            &[1.0, 0.972_972_9, 0.923_076_9, 0.8, 0.0],
            f32::EPSILON,
        );
        assert_close(
            schedule.audio_sigmas(),
            &[1.0, 0.9, 0.75, 0.5, 0.0],
            f32::EPSILON,
        );

        // Swapping the official modality shifts must fail this explicit middle
        // comparison even though both arrays still look like valid schedules.
        assert_ne!(schedule.video_sigmas()[2], schedule.audio_sigmas()[2]);
    }

    #[test]
    fn weight_free_synthetic_v1_matches_the_pinned_oracle() {
        let fixture = synthetic_fixture();
        let authority = &fixture.scheduler;
        assert_eq!(authority.video_shift, H3_VIDEO_SHIFT);
        assert_eq!(authority.audio_shift, H3_AUDIO_SHIFT);
        assert!(authority.terminal_zero_included);
        assert!(!authority.ancestral_noise_reinjected);

        let schedule = H3DualSchedule::new(authority.grid_points).unwrap();
        assert_eq!(
            schedule.counts().transformer_evaluations,
            authority.transformer_evaluations
        );
        assert_eq!(schedule.video_sigmas(), authority.video_sigmas);
        assert_eq!(schedule.audio_sigmas(), authority.audio_sigmas);

        let steps: Vec<_> = schedule.steps().collect();
        assert_eq!(
            steps
                .iter()
                .map(|step| step.video.timestep)
                .collect::<Vec<_>>(),
            authority.video_timesteps
        );
        assert_eq!(
            steps
                .iter()
                .map(|step| step.audio.timestep)
                .collect::<Vec<_>>(),
            authority.audio_timesteps
        );

        let update = &fixture.coupled_update;
        assert_eq!(update.velocity_sign, "data-ward-plus");
        assert_eq!(update.update_dtype, "float32");
        let device = Device::Cpu;
        let video = Tensor::new(update.sample.as_slice(), &device).unwrap();
        let audio = Tensor::new(update.sample.as_slice(), &device).unwrap();
        let video_velocity = Tensor::new(update.video_velocity.as_slice(), &device).unwrap();
        let audio_velocity = Tensor::new(update.audio_velocity.as_slice(), &device).unwrap();
        let step = steps
            .get(update.step_index)
            .copied()
            .expect("fixture step must exist in its schedule");
        let (video, audio) =
            euler_step_pair(&video, &audio, &video_velocity, &audio_velocity, step).unwrap();
        let video = video.to_vec1::<f32>().unwrap();
        let audio = audio.to_vec1::<f32>().unwrap();
        assert_close(&video, &update.video_next, f32::EPSILON);
        assert_close(&audio, &update.audio_next, f32::EPSILON);
        assert_close(&video, &synthetic_oracle_values("video_next"), f32::EPSILON);
        assert_close(&audio, &synthetic_oracle_values("audio_next"), f32::EPSILON);
    }

    #[test]
    fn official_steps_mean_fifty_grid_points_and_forty_nine_forwards() {
        let schedule = H3DualSchedule::official_default();
        assert_eq!(
            schedule.counts(),
            H3StepCount {
                requested_grid_points: H3_DEFAULT_GRID_POINTS,
                effective_grid_points: H3_DEFAULT_GRID_POINTS,
                transformer_evaluations: H3_DEFAULT_GRID_POINTS - 1,
            }
        );
        assert_eq!(schedule.video_sigmas().first(), Some(&1.0));
        assert_eq!(schedule.audio_sigmas().first(), Some(&1.0));
        assert_eq!(schedule.video_sigmas().last(), Some(&0.0));
        assert_eq!(schedule.audio_sigmas().last(), Some(&0.0));
        assert_eq!(schedule.steps().len(), 49);
    }

    #[test]
    fn comfy_res_multistep_matches_pinned_float32_reference() {
        let device = Device::Cpu;
        let mut video = Tensor::new(&[0.25f32, -0.5, 1.0], &device).unwrap();
        let mut audio = video.clone();
        let velocities = [
            [0.1f32, -0.2, 0.3],
            [-0.4, 0.5, -0.6],
            [0.7, -0.8, 0.9],
            [-1.0, 1.1, -1.2],
        ];
        let expected_video = [
            [0.252_702_7f32, -0.505_405_4, 1.008_108_1],
            [0.208_566_74, -0.446_608_84, 0.934_650_9],
            [0.478_628_78, -0.761_437_65, 1.294_246_4],
            [-0.321_371_23, 0.118_562_4, 0.334_246_4],
        ];
        let expected_audio = [
            [0.259_999_96f32, -0.519_999_9, 1.029_999_9],
            [0.127_316_2, -0.343_242_68, 0.809_169_2],
            [0.651_277_3, -0.948_295_53, 1.495_313_6],
            [0.151_277_3, -0.398_295_52, 0.895_313_6],
        ];
        let schedule =
            H3DualSchedule::new_for_sampler(5, H3SamplerKind::ComfyResMultistep).unwrap();
        let mut sampler = H3DualSampler::new(H3SamplerKind::ComfyResMultistep);

        for (index, step) in schedule.steps().enumerate() {
            let velocity = Tensor::new(&velocities[index], &device).unwrap();
            (video, audio) = sampler
                .step_pair(&video, &audio, &velocity, &velocity, step)
                .unwrap();
            assert_close(
                &video.to_vec1::<f32>().unwrap(),
                &expected_video[index],
                8.0 * f32::EPSILON,
            );
            assert_close(
                &audio.to_vec1::<f32>().unwrap(),
                &expected_audio[index],
                8.0 * f32::EPSILON,
            );
        }
    }

    #[test]
    fn comfy_twenty_evaluation_simple_schedule_matches_upstream_f32_bits() {
        let schedule =
            H3DualSchedule::new_for_sampler(21, H3SamplerKind::ComfyResMultistep).unwrap();
        let expected_video = [
            0x3f800000, 0x3f7ee1d1, 0x3f7da6c0, 0x3f7c4a35, 0x3f7ac688, 0x3f7914c2, 0x3f772c23,
            0x3f750192, 0x3f7286bc, 0x3f6fa8da, 0x3f6c4ec5, 0x3f68560c, 0x3f638e39, 0x3f5db0d3,
            0x3f565359, 0x3f4ccccd, 0x3f400000, 0x3f2de305, 0x3f124925, 0x3ec6318d, 0x00000000,
        ];
        let expected_audio = [
            0x3f800000, 0x3f7b9612, 0x3f76db6d, 0x3f71c71e, 0x3f6c4ec7, 0x3f666665, 0x3f600000,
            0x3f590b1e, 0x3f51745b, 0x3f492493, 0x3f3fffff, 0x3f35e50a, 0x3f2aaaaa, 0x3f1e1e1e,
            0x3f0fffff, 0x3f000001, 0x3edb6db8, 0x3eb13b15, 0x3e800000, 0x3e0ba2e9, 0x00000000,
        ];
        assert_eq!(
            schedule
                .video_sigmas()
                .iter()
                .map(|sigma| sigma.to_bits())
                .collect::<Vec<_>>(),
            expected_video
        );
        assert_eq!(
            schedule
                .audio_sigmas()
                .iter()
                .map(|sigma| sigma.to_bits())
                .collect::<Vec<_>>(),
            expected_audio
        );
    }

    #[test]
    fn sampler_rejects_replayed_or_out_of_order_steps() {
        let device = Device::Cpu;
        let sample = Tensor::new(&[0.0f32], &device).unwrap();
        let velocity = Tensor::new(&[1.0f32], &device).unwrap();
        let schedule = H3DualSchedule::new(4).unwrap();
        let first = schedule.steps().next().unwrap();
        let mut sampler = H3DualSampler::new(H3SamplerKind::ComfyResMultistep);
        sampler
            .step_pair(&sample, &sample, &velocity, &velocity, first)
            .unwrap();
        assert!(sampler
            .step_pair(&sample, &sample, &velocity, &velocity, first)
            .unwrap_err()
            .to_string()
            .contains("expected evaluation 1"));
    }

    #[test]
    fn float32_collisions_collapse_without_losing_terminal_zero() {
        assert_eq!(
            unique_consecutive([1.0, 1.0, 0.75, 0.75, 0.25, 0.0, 0.0]),
            vec![1.0, 0.75, 0.25, 0.0]
        );
        let shortest = H3DualSchedule::new(2).unwrap();
        assert_eq!(shortest.video_sigmas(), &[1.0, 0.0]);
        assert_eq!(shortest.audio_sigmas(), &[1.0, 0.0]);
        assert_eq!(shortest.steps().len(), 1);
    }

    #[test]
    fn timesteps_and_condition_rows_keep_their_authority() {
        let schedule = H3DualSchedule::new(5).unwrap();
        let steps: Vec<_> = schedule.steps().collect();

        assert_eq!(steps[0].video.timestep, 0.0);
        assert_eq!(steps[0].audio.timestep, 0.0);
        assert_eq!(steps[2].video.timestep, 1.0 - 0.923_076_9);
        assert_eq!(steps[2].audio.timestep, 0.25);
        for step in steps {
            assert_eq!(step.row_timesteps.text, step.video.timestep);
            assert_eq!(step.row_timesteps.generated_video, step.video.timestep);
            assert_eq!(
                step.row_timesteps.visual_condition,
                step.video.timestep.max(H3_VISUAL_CONDITION_TIMESTEP)
            );
            assert_eq!(step.row_timesteps.generated_audio, step.audio.timestep);
            assert_eq!(step.row_timesteps.audio_reference, H3_CLEAN_TIMESTEP);
        }
    }

    #[test]
    fn iterator_has_one_cancellation_boundary_per_coupled_forward() {
        let schedule = H3DualSchedule::official_default();
        let mut steps = schedule.steps();
        assert_eq!(steps.len(), 49);

        let first = steps.next().unwrap();
        assert_eq!(first.progress_before_forward().completed_evaluations, 0);
        assert_eq!(first.progress_after_update().completed_evaluations, 1);
        assert!(!first.progress_after_update().is_complete());
        assert_eq!(steps.len(), 48);

        let last = steps.last().unwrap();
        assert_eq!(last.evaluation_index, 48);
        assert_eq!(last.video.next_sigma, 0.0);
        assert_eq!(last.audio.next_sigma, 0.0);
        assert_eq!(last.progress_after_update().completed_evaluations, 49);
        assert_eq!(last.progress_after_update().total_evaluations, 49);
        assert!(last.progress_after_update().is_complete());
    }

    #[test]
    fn data_ward_velocity_uses_the_positive_sign_and_reaches_x0() {
        let device = Device::Cpu;
        let sample = Tensor::new(&[1.0f32, -2.0], &device).unwrap();
        let velocity = Tensor::new(&[0.5f32, 1.0], &device).unwrap();
        let final_step = H3DualSchedule::new(5).unwrap().steps().last().unwrap();
        let output = euler_step(&sample, &velocity, final_step.video)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        // Final video sigma is 0.8: x0 = x + 0.8*v. Reversing the sign would
        // produce [0.6, -2.8] and is intentionally caught here.
        assert_close(&output, &[1.4, -1.2], 1e-6);
    }

    #[test]
    fn coupled_update_uses_distinct_sigmas_without_partial_mutation() {
        let device = Device::Cpu;
        let video = Tensor::new(&[1.0f32], &device).unwrap();
        let audio = Tensor::new(&[1.0f32], &device).unwrap();
        let video_velocity = Tensor::new(&[1.0f32], &device).unwrap();
        let audio_velocity = Tensor::new(&[1.0f32], &device).unwrap();
        let step = H3DualSchedule::new(5).unwrap().steps().nth(2).unwrap();

        let (next_video, next_audio) =
            euler_step_pair(&video, &audio, &video_velocity, &audio_velocity, step).unwrap();
        let next_video = next_video.to_vec1::<f32>().unwrap()[0];
        let next_audio = next_audio.to_vec1::<f32>().unwrap()[0];
        assert_close(&[next_video], &[1.123_076_9], 1e-6);
        assert_close(&[next_audio], &[1.25], 1e-6);

        // Inputs are immutable; a later audio error cannot leave a stored video
        // update pretending the pair is synchronized.
        assert_eq!(video.to_vec1::<f32>().unwrap(), vec![1.0]);
        assert_eq!(audio.to_vec1::<f32>().unwrap(), vec![1.0]);
    }

    #[test]
    fn half_latents_quantize_timestep_before_float32_x0_and_blend() {
        let device = Device::Cpu;
        let sample = Tensor::new(&[-80.0f32, 40.0], &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        // The official denoise block explicitly passes its prediction as
        // float32 even while the latent is bf16/f16.
        let velocity = Tensor::new(&[100.0f32, -50.0], &device).unwrap();
        let step = H3DualSchedule::new(5)
            .unwrap()
            .steps()
            .last()
            .unwrap()
            .video;

        let output = euler_step(&sample, &velocity, step).unwrap();
        assert_eq!(output.dtype(), DType::BF16);

        // Oracle ordering: cast t to the latent dtype, subtract there, then
        // promote the mixed latent/model-output expression and blend in f32.
        let one = Tensor::new(1.0f32, &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let timestep = Tensor::new(step.timestep, &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let sigma_from_timestep = one.sub(&timestep).unwrap().to_dtype(DType::F32).unwrap();
        let sample32 = sample.to_dtype(DType::F32).unwrap();
        let expected = sample32
            .add(&velocity.broadcast_mul(&sigma_from_timestep).unwrap())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let actual = output
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(actual, expected);

        // Reusing the f32 grid sigma directly is close numerically but is not
        // the pinned scheduler's half-precision timestep round trip.
        let unquantized = sample32
            .add(&velocity.affine((1.0 - step.timestep) as f64, 0.0).unwrap())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_ne!(actual, unquantized);
    }

    #[test]
    fn malformed_schedule_inputs_and_steps_fail_closed() {
        assert!(H3DualSchedule::new(0).is_err());
        assert!(H3DualSchedule::new(1).is_err());

        let device = Device::Cpu;
        let sample = Tensor::new(&[1.0f32], &device).unwrap();
        let velocity = Tensor::new(&[1.0f32], &device).unwrap();
        let invalid = H3FlowStep {
            sigma: 0.5,
            next_sigma: 0.5,
            timestep: 0.5,
        };
        assert!(euler_step(&sample, &velocity, invalid).is_err());

        let inconsistent_timestep = H3FlowStep {
            sigma: 0.5,
            next_sigma: 0.0,
            timestep: 0.25,
        };
        assert!(euler_step(&sample, &velocity, inconsistent_timestep).is_err());

        let integer_velocity = Tensor::new(&[1u8], &device).unwrap();
        let valid = H3DualSchedule::new(2).unwrap().steps().next().unwrap();
        assert!(euler_step(&sample, &integer_velocity, valid.video).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[cfg_attr(not(target_os = "linux"), ignore)]
    fn tiny_coupled_update_matches_cpu_and_cuda() {
        let Ok(cuda) = Device::new_cuda(0) else {
            return;
        };
        let cpu = Device::Cpu;
        let video_values = [0.125f32, -1.75, 3.5, 0.03125];
        let audio_values = [-0.5f32, 2.25, -4.0, 0.75];
        let video_velocity_values = [0.75f32, -0.125, 1.25, -3.0];
        let audio_velocity_values = [1.5f32, -2.0, 0.0625, 0.25];
        let step = H3DualSchedule::official_default().steps().nth(24).unwrap();

        let run = |device: &Device| {
            let video = Tensor::new(&video_values, device).unwrap();
            let audio = Tensor::new(&audio_values, device).unwrap();
            let video_velocity = Tensor::new(&video_velocity_values, device).unwrap();
            let audio_velocity = Tensor::new(&audio_velocity_values, device).unwrap();
            let (video, audio) =
                euler_step_pair(&video, &audio, &video_velocity, &audio_velocity, step).unwrap();
            (
                video
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                audio
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
            )
        };

        let (cpu_video, cpu_audio) = run(&cpu);
        let (cuda_video, cuda_audio) = run(&cuda);
        assert_close(&cuda_video, &cpu_video, 1e-6);
        assert_close(&cuda_audio, &cpu_audio, 1e-6);
    }
}
