//! Exact dual-modality rectified-flow schedule used by MiniMax H3.
//!
//! H3 advances generated video and audio rows through one transformer forward,
//! but the two modalities use different shifted sigma grids. The iterator in
//! this module therefore yields one immutable descriptor per *coupled* forward;
//! callers can check cancellation before requesting the next descriptor and
//! only replace their two latent tensors after the selected paired update
//! succeeds. Official BF16 uses first-order Euler; the compact Comfy layout
//! uses the deterministic RES multistep rule from its released workflow.
//! Distilled tiers select Comfy Euler or Comfy LCM over the same Comfy grid;
//! LCM is the one integrator that re-noises, from a seeded per-step stream.
//! Video and audio flow shifts are both schedule parameters.
//!
//! `grid_points` deliberately includes the terminal zero. The official default
//! is 50 grid points and thus 49 transformer evaluations, not 50 evaluations.

use std::iter::FusedIterator;

use anyhow::{bail, Result};
use candle_core::{DType, Tensor};

pub(crate) const H3_DEFAULT_GRID_POINTS: usize = 50;
pub(crate) const H3_VIDEO_SHIFT: f32 = 12.0;
pub(crate) const H3_AUDIO_SHIFT: f32 = 3.0;
/// Video shift for the 768p-trained FL2V Turbo tiers. Two of the three are
/// TRANSCRIBED upstream facts; the third is an inference, and the difference
/// is deliberately recorded here rather than flattened.
///
/// - `4-step v1.0 768p` and `8-step v1.0 768p`: ModelTC's Turbo model-specs
///   table lists `6 / 3` as both tiers' training shifts, and LightX2V's own
///   `configs/minimax_h3/dmd/minimax_h3_fp8_8step.json` and
///   `minimax_h3_int8_convrot_8step.json` run the 8-step tier at
///   `video_flow_shift: 6.0` / `audio_flow_shift: 3.0` (with `infer_steps: 9`,
///   the same terminal-inclusive count mold ships).
/// - `4-step v1.1 768p`: UNDOCUMENTED upstream. It has no row in ModelTC's
///   spec table, and no LightX2V config names its file. Mold INHERITS 6 from
///   its v1.0 768p predecessor — same 768p training resolution, same
///   rank/alpha/scale header metadata — pending the A/B recorded in
///   `docs/qualification/minimax-h3.md`. Treat 6.0 for that one tier as mold's
///   choice, never as a transcribed upstream value.
///
/// Every 544p-trained reviewed tier keeps [`H3_VIDEO_SHIFT`].
pub(crate) const H3_TURBO_768P_VIDEO_SHIFT: f32 = 6.0;
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
    /// The Turbo distillations keep ComfyUI's `BasicScheduler("simple")` sigma
    /// table but integrate it with `KSamplerSelect: euler`, which is what every
    /// published Turbo reference workflow selects. It is therefore exactly the
    /// existing Comfy grid paired with the existing first-order integrator.
    ComfyEuler,
    /// ComfyUI's `lcm` sampler over the same `BasicScheduler("simple")` grid:
    /// every step jumps to the clean estimate and, unless the next sigma is
    /// terminal, re-noises it with FRESH Gaussian noise at that sigma
    /// (`comfy/k_diffusion/sampling.py:1083-1113`). The DMAD distillations
    /// are sampled this way; `Yzmblog/DMAD@9a9f664e dmad_h3/sampling.py:63-98`
    /// is the same rule. It is the only H3 integrator that consumes noise
    /// after initialization, so a sampler of this kind must be built with
    /// [`H3DualSampler::with_noise_seed`].
    ComfyLcm,
}

impl H3SamplerKind {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::OfficialEuler => "official-euler",
            Self::ComfyResMultistep => "comfy-res-multistep",
            Self::ComfyEuler => "comfy-euler",
            Self::ComfyLcm => "comfy-lcm",
        }
    }

    pub(crate) fn parse(value: &str) -> Option<Self> {
        [
            Self::OfficialEuler,
            Self::ComfyResMultistep,
            Self::ComfyEuler,
            Self::ComfyLcm,
        ]
        .into_iter()
        .find(|kind| kind.as_str() == value)
    }

    /// True when the sigma grid comes from Comfy's discrete 1000-entry table
    /// rather than the official continuous linspace.
    pub(crate) const fn uses_comfy_simple_grid(self) -> bool {
        matches!(
            self,
            Self::ComfyResMultistep | Self::ComfyEuler | Self::ComfyLcm
        )
    }

    /// True when the coupled update is the first-order data-ward Euler rule.
    pub(crate) const fn uses_euler_update(self) -> bool {
        matches!(self, Self::OfficialEuler | Self::ComfyEuler)
    }

    /// True when the integrator re-noises between evaluations and therefore
    /// needs a deterministic per-step noise stream.
    pub(crate) const fn consumes_step_noise(self) -> bool {
        matches!(self, Self::ComfyLcm)
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
    /// The two flow shifts the owning schedule was built with. Only the Comfy
    /// carried-audio helpers read them; they are copied here so a sampler can
    /// never pair one schedule's descriptors with another schedule's scale.
    video_shift: f32,
    audio_shift: f32,
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

    /// ComfyUI `ModelSamplingAV.audio_scale` (`comfy/model_sampling.py:353-357`):
    /// `shift / audio_shift`, the factor the packed latent carries audio at.
    fn comfy_audio_scale(self) -> f32 {
        self.video_shift / self.audio_shift
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
        if !shift.is_finite() || shift <= 0.0 {
            bail!("MiniMax H3 schedule shift must be finite and positive, got {shift}");
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
    video_shift: f32,
    audio_shift: f32,
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
            video_shift: H3_VIDEO_SHIFT,
            audio_shift: H3_AUDIO_SHIFT,
        })
    }

    pub fn new_for_sampler(grid_points: usize, kind: H3SamplerKind) -> Result<Self> {
        Self::new_for_sampler_with_shifts(grid_points, kind, H3_VIDEO_SHIFT, H3_AUDIO_SHIFT)
    }

    /// Build the schedule for one sampler kind at an explicit video shift and
    /// the family audio shift ([`H3_AUDIO_SHIFT`]).
    ///
    /// Kept as the wrapper every pre-existing caller uses; it is exactly
    /// [`Self::new_for_sampler_with_shifts`] at audio shift 3.
    pub fn new_for_sampler_with_video_shift(
        grid_points: usize,
        kind: H3SamplerKind,
        video_shift: f32,
    ) -> Result<Self> {
        Self::new_for_sampler_with_shifts(grid_points, kind, video_shift, H3_AUDIO_SHIFT)
    }

    /// Build the schedule for one sampler kind at explicit video AND audio
    /// shifts.
    ///
    /// ComfyUI's `ModelSamplingMiniMaxH3` node sets both coherently
    /// (`comfy_extras/nodes_minimax_h3.py:367-408`): the video shift drives the
    /// `simple` sigma table and the audio shift re-maps each video sigma onto
    /// the audio stream. Distilled tiers move either one — the 768p Turbo
    /// tiers train video at 6, FastH3 V2 at 10/3, DMAD at 12/2 and PDMD 2-step
    /// at 12/6 — so both are schedule parameters rather than constants.
    ///
    /// The official continuous grid is the released 12/3 contract and has no
    /// per-tier shift at all, so it refuses anything else.
    pub fn new_for_sampler_with_shifts(
        grid_points: usize,
        kind: H3SamplerKind,
        video_shift: f32,
        audio_shift: f32,
    ) -> Result<Self> {
        if !audio_shift.is_finite() || audio_shift <= 0.0 {
            bail!("MiniMax H3 schedule audio shift must be finite and positive, got {audio_shift}");
        }
        if !kind.uses_comfy_simple_grid() {
            if video_shift.to_bits() != H3_VIDEO_SHIFT.to_bits() {
                bail!(
                    "MiniMax H3 official schedule has no per-tier video shift; refused {video_shift}"
                );
            }
            if audio_shift.to_bits() != H3_AUDIO_SHIFT.to_bits() {
                bail!(
                    "MiniMax H3 official schedule has no per-tier audio shift; refused {audio_shift}"
                );
            }
            return Self::new(grid_points);
        }
        let video = H3ModalitySchedule::new_comfy_simple(grid_points, video_shift)?;
        let audio = H3ModalitySchedule {
            // The released model derives native audio sigma from the already
            // rounded video-table sigma, rather than independently sampling an
            // audio-shift table: ComfyUI's `time_shift_sigma`
            // (`comfy/ldm/minimax/model.py:36-39`) inverts the video shift to
            // the shared base grid and re-applies the audio shift, in that f32
            // order (`model.py:609-613`, `:653-656`).
            sigmas: video
                .sigmas
                .iter()
                .copied()
                .map(|sigma| {
                    if sigma == 0.0 {
                        return 0.0;
                    }
                    let base = sigma / (video_shift + sigma * (1.0 - video_shift));
                    audio_shift * base / (1.0 + (audio_shift - 1.0) * base)
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
            video_shift,
            audio_shift,
        })
    }

    /// The video shift this grid was built with.
    pub fn video_shift(&self) -> f32 {
        self.video_shift
    }

    /// The audio shift this grid was built with.
    pub fn audio_shift(&self) -> f32 {
        self.audio_shift
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
            video_shift: self.video_shift,
            audio_shift: self.audio_shift,
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
    // converts the model's native d(audio)/d(sigma_a) back to d(carry)/d(sigma_v)
    // (`comfy/ldm/minimax/model.py:638-641`). The scale is the schedule's own
    // `video_shift / audio_shift` (`comfy/model_sampling.py:353-357`), never a
    // family constant, so a 10/3 or 12/2 grid carries audio at its own ratio.
    // Keep this expression in F32, matching its released wrapper.
    let scale = step.comfy_audio_scale();
    native_sample
        .to_dtype(DType::F32)?
        .affine(f64::from(scale - 1.0), 0.0)?
        .add(
            &native_velocity
                .to_dtype(DType::F32)?
                .affine(f64::from(1.0 + (scale - 1.0) * step.audio.sigma), 0.0)?,
        )
        .map_err(Into::into)
}

fn comfy_audio_native_sample(carried: &Tensor, step: H3DualScheduleStep) -> Result<Tensor> {
    let native_scale = if step.video.next_sigma == 0.0 {
        // lim sigma_a/sigma_v as the shared base grid approaches zero, which is
        // `1 / audio_scale` (`comfy/model_base.py:2177-2178`).
        step.audio_shift / step.video_shift
    } else {
        step.audio.next_sigma / step.video.next_sigma
    };
    carried
        .affine(f64::from(native_scale), 0.0)
        .map_err(Into::into)
}

/// One modality's ComfyUI `lcm` update.
///
/// `sample_lcm` (`comfy/k_diffusion/sampling.py:1083-1113`) sets `x =
/// denoised` and, when the next sigma is positive, `x =
/// model_sampling.noise_scaling(sigma', noise, x)`, which for the flow (CONST)
/// parameterization is `sigma' * noise + (1 - sigma') * x0`
/// (`comfy/model_sampling.py:94-97`, `noise_scale` 1). The terminal step
/// returns the clean estimate itself, without noise.
///
/// ## Why per-modality re-noise is exactly Comfy's packed update
///
/// Comfy samples H3's packed AV latent on the VIDEO sigma: audio is carried as
/// `c = k * a` with `k = sigma_v / sigma_a` (`comfy/ldm/minimax/model.py:
/// 604-614`, `comfy/model_base.py:2171-2178`). Write both sigmas over the
/// shared base grid `b`: `sigma_s = s*b / (1 + (s-1)*b)`, so
/// `1 - sigma_s = (1-b) / (1 + (s-1)*b)`, and let `r = s_v / s_a` (the
/// `audio_scale`).
///
/// 1. The carried clean estimate is `r * x0_a`. The carried velocity
///    (`model.py:638-641`) gives `x0_c = c + sigma_v*((r-1)*a + (1+(r-1)*sigma_a)*v)`;
///    the identities `k + (r-1)*sigma_v = r` and
///    `sigma_v*(1+(r-1)*sigma_a) = r*sigma_a` (both follow from the two shift
///    formulas over the same `b`) reduce it to `r*(a + sigma_a*v) = r*x0_a`.
/// 2. The packed re-noise at `sigma_v'` is `c' = sigma_v'*eps + (1-sigma_v')*r*x0_a`;
///    the next model call un-carries it by `sigma_a'/sigma_v'` (the carry the
///    DiT undoes), giving `a' = sigma_a'*eps + (sigma_a'/sigma_v')*(1-sigma_v')*r*x0_a`.
/// 3. `(sigma_a'/sigma_v') * (1-sigma_v') * r
///    = (s_a/s_v) * (1+(s_v-1)b)/(1+(s_a-1)b) * (1-b)/(1+(s_v-1)b) * (s_v/s_a)
///    = (1-b)/(1+(s_a-1)b) = 1 - sigma_a'`.
///
/// So `a' = sigma_a'*eps + (1-sigma_a')*x0_a`: the audio re-noise is the plain
/// flow re-noise at audio's OWN next sigma, with the same standard-normal
/// draw, and at the terminal step `x0_c / r = x0_a`. Nothing about the carry
/// survives, which is also why the LCM path needs no carried-audio state.
/// The identity holds in exact arithmetic; Comfy's own packed f32 evaluation
/// differs from it by rounding only (1 ulp on the DMAD fixture).
fn lcm_update(
    sample: &Tensor,
    velocity: &Tensor,
    noise: Option<&Tensor>,
    step: H3FlowStep,
) -> Result<Tensor> {
    let decreasing = step.sigma.is_finite()
        && step.next_sigma.is_finite()
        && step.sigma > 0.0
        && step.next_sigma >= 0.0
        && step.next_sigma < step.sigma;
    if !decreasing {
        bail!(
            "MiniMax H3 LCM step needs finite decreasing sigmas, got {} -> {}",
            step.sigma,
            step.next_sigma
        );
    }
    let x0 = denoised_from_velocity(sample, velocity, step)?;
    let next = if step.next_sigma == 0.0 {
        if noise.is_some() {
            bail!("MiniMax H3 LCM terminal step must not consume noise");
        }
        x0
    } else {
        let noise = noise.ok_or_else(|| {
            anyhow::anyhow!(
                "MiniMax H3 LCM step to sigma {} needs fresh noise",
                step.next_sigma
            )
        })?;
        if noise.dims() != sample.dims() {
            bail!(
                "MiniMax H3 LCM noise shape {:?} does not match the latent {:?}",
                noise.dims(),
                sample.dims()
            );
        }
        // `sigma * noise + (1 - sigma) * latent`, both terms in F32 like the
        // Comfy sampler state; `1 - sigma'` is an f32 scalar subtraction.
        noise
            .to_dtype(DType::F32)?
            .affine(f64::from(step.next_sigma), 0.0)?
            .add(&x0.affine(f64::from(1.0 - step.next_sigma), 0.0)?)?
    };
    next.to_dtype(sample.dtype()).map_err(Into::into)
}

/// Domain separator for the LCM re-noise stream, so its draws can never
/// replay the request's initial-noise stream (`pipeline.rs` `H3RequestNoise`,
/// which seeds `StdRng` with the bare request seed).
const H3_LCM_STEP_NOISE_DOMAIN: u64 = 0x6833_6c63_6d5f_6e7a; // "h3lcm_nz"

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Deterministic per-step noise for integrators that re-noise.
///
/// One fresh `StdRng` per coupled evaluation, seeded from the request seed,
/// the evaluation index, and a domain separator; it draws the VIDEO tensor
/// first and then the AUDIO tensor, element by element, exactly the way the
/// pipeline draws its initial noise (`StandardNormal` f32 on the host, then
/// moved to the latent's device). A replayed request therefore re-noises
/// identically on every backend, and the noise for evaluation `i` does not
/// depend on how many elements any other evaluation drew.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct H3StepNoise {
    seed: u64,
}

impl H3StepNoise {
    pub(crate) const fn new(seed: u64) -> Self {
        Self { seed }
    }

    fn step_seed(self, evaluation_index: usize) -> u64 {
        splitmix64(
            splitmix64(self.seed ^ H3_LCM_STEP_NOISE_DOMAIN).wrapping_add(evaluation_index as u64),
        )
    }

    /// Draw `(video, audio)` noise for one evaluation, video first.
    pub(crate) fn draw(
        self,
        evaluation_index: usize,
        video_shape: &[usize],
        audio_shape: &[usize],
        device: &candle_core::Device,
    ) -> Result<(Tensor, Tensor)> {
        use rand::SeedableRng;
        use rand_distr::Distribution;

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.step_seed(evaluation_index));
        let mut draw = |shape: &[usize]| -> Result<Tensor> {
            let elements = shape.iter().try_fold(1usize, |total, dim| {
                total
                    .checked_mul(*dim)
                    .ok_or_else(|| anyhow::anyhow!("MiniMax H3 LCM noise shape overflows"))
            })?;
            let values = (0..elements)
                .map(|_| rand_distr::StandardNormal.sample(&mut rng))
                .collect::<Vec<f32>>();
            Tensor::from_vec(values, shape, &candle_core::Device::Cpu)?
                .to_device(device)
                .map_err(Into::into)
        };
        let video = draw(video_shape)?;
        let audio = draw(audio_shape)?;
        Ok((video, audio))
    }
}

/// Stateful synchronized sampler for the video/audio pair.
pub(crate) struct H3DualSampler {
    kind: H3SamplerKind,
    previous_video_denoised: Option<Tensor>,
    previous_audio_denoised: Option<Tensor>,
    current_audio_carried: Option<Tensor>,
    previous_video_sigma: Option<f32>,
    next_evaluation_index: usize,
    step_noise: Option<H3StepNoise>,
}

impl H3DualSampler {
    /// A sampler for a deterministic integrator. A noise-consuming kind
    /// ([`H3SamplerKind::consumes_step_noise`]) built this way refuses its
    /// first non-terminal step rather than inventing noise.
    pub(crate) fn new(kind: H3SamplerKind) -> Self {
        Self {
            kind,
            previous_video_denoised: None,
            previous_audio_denoised: None,
            current_audio_carried: None,
            previous_video_sigma: None,
            next_evaluation_index: 0,
            step_noise: None,
        }
    }

    /// A sampler that also owns the request's deterministic re-noise stream.
    ///
    /// Every pipeline builds its sampler this way with the request seed; the
    /// deterministic kinds never read the stream, so this is byte-identical to
    /// [`Self::new`] for them.
    pub(crate) fn with_noise_seed(kind: H3SamplerKind, seed: u64) -> Self {
        Self {
            step_noise: Some(H3StepNoise::new(seed)),
            ..Self::new(kind)
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
        if self.kind.uses_euler_update() {
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
        if self.kind.consumes_step_noise() {
            let next = self.lcm_step_pair(
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

    /// Both modalities' LCM updates, each at its own next sigma (see
    /// [`lcm_update`] for why that IS Comfy's packed update). Noise is drawn
    /// only for a non-terminal step, video first, and both updates are
    /// computed before either is returned.
    fn lcm_step_pair(
        &self,
        video_sample: &Tensor,
        audio_sample: &Tensor,
        video_velocity: &Tensor,
        audio_velocity: &Tensor,
        step: H3DualScheduleStep,
    ) -> Result<(Tensor, Tensor)> {
        // The shared base grid reaches zero for both modalities together.
        if (step.video.next_sigma == 0.0) != (step.audio.next_sigma == 0.0) {
            bail!("MiniMax H3 LCM video and audio schedules disagree about the terminal step");
        }
        let noise = if step.video.next_sigma == 0.0 {
            None
        } else {
            let stream = self.step_noise.ok_or_else(|| {
                anyhow::anyhow!(
                    "MiniMax H3 {} sampler needs the request's noise seed",
                    self.kind.as_str()
                )
            })?;
            Some(stream.draw(
                step.evaluation_index,
                video_sample.dims(),
                audio_sample.dims(),
                video_sample.device(),
            )?)
        };
        let (video_noise, audio_noise) = match &noise {
            Some((video, audio)) => (Some(video), Some(audio)),
            None => (None, None),
        };
        let next_video = lcm_update(video_sample, video_velocity, video_noise, step.video)?;
        let next_audio = lcm_update(audio_sample, audio_velocity, audio_noise, step.audio)?;
        Ok((next_video, next_audio))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};
    use serde::Deserialize;

    use super::{
        euler_step, euler_step_pair, lcm_update, unique_consecutive, H3DualSampler, H3DualSchedule,
        H3FlowStep, H3SamplerKind, H3StepCount, H3StepNoise, H3_AUDIO_SHIFT, H3_CLEAN_TIMESTEP,
        H3_DEFAULT_GRID_POINTS, H3_TURBO_768P_VIDEO_SHIFT, H3_VIDEO_SHIFT,
        H3_VISUAL_CONDITION_TIMESTEP,
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

    /// The Turbo grids, pinned to exact float32 bits.
    ///
    /// The 4-step shift-12 row is the Turbo README's own published worked
    /// example — `[1, 0.9730, 0.9231, 0.8000] -> 0` — so this is an independent
    /// oracle, not a restatement of mold's arithmetic. The 8-step rows sample
    /// the identical 1000-entry table at the same indices the reviewed 21-point
    /// schedule already pins, which is why 0.972973 (`0x3f7914c2`) and its audio
    /// partner (`0x3f666665`) appear in both.
    #[test]
    fn turbo_comfy_euler_grids_match_their_published_float32_bits() {
        let cases = [
            (
                9usize,
                H3_VIDEO_SHIFT,
                vec![
                    0x3f800000u32,
                    0x3f7cfcfd,
                    0x3f7914c2,
                    0x3f73cf3d,
                    0x3f6c4ec5,
                    0x3f60c7ce,
                    0x3f4ccccd,
                    0x3f21af28,
                    0x00000000,
                ],
                vec![
                    0x3f800000u32,
                    0x3f745d19,
                    0x3f666665,
                    0x3f555557,
                    0x3f3fffff,
                    0x3f24924b,
                    0x3f000001,
                    0x3e999998,
                    0x00000000,
                ],
            ),
            (
                5,
                H3_VIDEO_SHIFT,
                vec![0x3f800000, 0x3f7914c2, 0x3f6c4ec5, 0x3f4ccccd, 0x00000000],
                vec![0x3f800000, 0x3f666665, 0x3f3fffff, 0x3f000001, 0x00000000],
            ),
            (
                5,
                H3_TURBO_768P_VIDEO_SHIFT,
                vec![0x3f800000, 0x3f7286bd, 0x3f5b6db7, 0x3f2aaaab, 0x00000000],
                vec![0x3f800000, 0x3f666668, 0x3f3fffff, 0x3f000001, 0x00000000],
            ),
        ];
        for (grid_points, video_shift, expected_video, expected_audio) in cases {
            let schedule = H3DualSchedule::new_for_sampler_with_video_shift(
                grid_points,
                H3SamplerKind::ComfyEuler,
                video_shift,
            )
            .unwrap();
            assert_eq!(schedule.video_shift(), video_shift);
            assert_eq!(
                schedule
                    .video_sigmas()
                    .iter()
                    .map(|sigma| sigma.to_bits())
                    .collect::<Vec<_>>(),
                expected_video,
                "video {grid_points}@{video_shift}"
            );
            assert_eq!(
                schedule
                    .audio_sigmas()
                    .iter()
                    .map(|sigma| sigma.to_bits())
                    .collect::<Vec<_>>(),
                expected_audio,
                "audio {grid_points}@{video_shift}"
            );

            // No `unique_consecutive` collapse may shorten a Turbo render.
            assert_eq!(
                schedule.counts(),
                H3StepCount {
                    requested_grid_points: grid_points,
                    effective_grid_points: grid_points,
                    transformer_evaluations: grid_points - 1,
                }
            );
        }
    }

    /// The published Turbo README states NFE = 4 gives sigma
    /// `[1, 0.9730, 0.9231, 0.8000]` then terminal zero, for shift 12.
    #[test]
    fn turbo_four_evaluation_grid_matches_the_published_decimal_oracle() {
        let schedule = H3DualSchedule::new_for_sampler(5, H3SamplerKind::ComfyEuler).unwrap();
        let published = [1.0f32, 0.9730, 0.9231, 0.8000, 0.0];
        assert_eq!(schedule.video_sigmas().len(), published.len());
        for (observed, expected) in schedule.video_sigmas().iter().zip(published.iter()) {
            assert!(
                (observed - expected).abs() < 5e-5,
                "{observed} vs {expected}"
            );
        }
    }

    /// mold's `steps` is the terminal-inclusive grid-point count, so a Turbo
    /// "N-step" tier is `N + 1` steps and exactly N transformer evaluations.
    #[test]
    fn turbo_step_counts_are_terminal_inclusive_grid_points() {
        for (comfy_steps, mold_steps) in [(8usize, 9usize), (4, 5), (20, 21)] {
            let schedule =
                H3DualSchedule::new_for_sampler(mold_steps, H3SamplerKind::ComfyEuler).unwrap();
            assert_eq!(schedule.counts().transformer_evaluations, comfy_steps);
            assert_eq!(schedule.steps().len(), comfy_steps);
        }
    }

    #[test]
    fn comfy_euler_shares_the_comfy_grid_and_the_official_integrator() {
        // Same sigma table as RES multistep at the default shift...
        let euler = H3DualSchedule::new_for_sampler(9, H3SamplerKind::ComfyEuler).unwrap();
        let multistep =
            H3DualSchedule::new_for_sampler(9, H3SamplerKind::ComfyResMultistep).unwrap();
        assert_eq!(euler.video_sigmas(), multistep.video_sigmas());
        assert_eq!(euler.audio_sigmas(), multistep.audio_sigmas());

        // At 5, 9, and 21 grid points the discrete 1000-entry table lands
        // exactly on the continuous linspace, because 4, 8, and 20 all divide
        // 1000 — so no Turbo tier is exposed to the two grids disagreeing. A
        // count that does not divide 1000 shows they are genuinely different
        // constructions.
        for grid_points in [5usize, 9, 21] {
            assert_eq!(
                H3DualSchedule::new_for_sampler(grid_points, H3SamplerKind::ComfyEuler)
                    .unwrap()
                    .video_sigmas(),
                H3DualSchedule::new(grid_points).unwrap().video_sigmas(),
                "{grid_points}"
            );
        }
        assert_ne!(
            H3DualSchedule::new_for_sampler(8, H3SamplerKind::ComfyEuler)
                .unwrap()
                .video_sigmas(),
            H3DualSchedule::new(8).unwrap().video_sigmas()
        );

        assert!(H3SamplerKind::ComfyEuler.uses_comfy_simple_grid());
        assert!(H3SamplerKind::ComfyEuler.uses_euler_update());
        assert!(H3SamplerKind::ComfyResMultistep.uses_comfy_simple_grid());
        assert!(!H3SamplerKind::ComfyResMultistep.uses_euler_update());
        assert!(!H3SamplerKind::OfficialEuler.uses_comfy_simple_grid());
        assert!(H3SamplerKind::OfficialEuler.uses_euler_update());

        // The coupled update is bit-identical to the official Euler rule on the
        // same descriptors, so nothing new is being integrated.
        let device = Device::Cpu;
        let sample = Tensor::new(&[0.25f32, -0.5, 1.0], &device).unwrap();
        let velocity = Tensor::new(&[0.1f32, -0.2, 0.3], &device).unwrap();
        let step = euler.steps().next().unwrap();
        let mut sampler = H3DualSampler::new(H3SamplerKind::ComfyEuler);
        let (video, audio) = sampler
            .step_pair(&sample, &sample, &velocity, &velocity, step)
            .unwrap();
        let (reference_video, reference_audio) =
            euler_step_pair(&sample, &sample, &velocity, &velocity, step).unwrap();
        assert_eq!(
            video.to_vec1::<f32>().unwrap(),
            reference_video.to_vec1::<f32>().unwrap()
        );
        assert_eq!(
            audio.to_vec1::<f32>().unwrap(),
            reference_audio.to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn non_default_shifts_build_comfy_grids_and_never_the_official_one() {
        // The carried-audio scale is derived from the schedule since #1812, so
        // RES multistep accepts any reviewed video shift (FastH3 V2 is 10/3).
        for shift in [H3_TURBO_768P_VIDEO_SHIFT, 10.0] {
            let schedule = H3DualSchedule::new_for_sampler_with_video_shift(
                9,
                H3SamplerKind::ComfyResMultistep,
                shift,
            )
            .unwrap();
            assert_eq!(schedule.video_shift(), shift);
            assert_eq!(schedule.audio_shift(), H3_AUDIO_SHIFT);
        }

        // The official continuous grid has no per-tier shift at all.
        let error = H3DualSchedule::new_for_sampler_with_video_shift(
            5,
            H3SamplerKind::OfficialEuler,
            H3_TURBO_768P_VIDEO_SHIFT,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("no per-tier video shift"), "{error}");
        let error = H3DualSchedule::new_for_sampler_with_shifts(
            5,
            H3SamplerKind::OfficialEuler,
            H3_VIDEO_SHIFT,
            2.0,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("no per-tier audio shift"), "{error}");

        // And a nonsense shift never builds a grid, on either axis.
        for shift in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            assert!(H3DualSchedule::new_for_sampler_with_video_shift(
                5,
                H3SamplerKind::ComfyEuler,
                shift
            )
            .is_err());
            assert!(H3DualSchedule::new_for_sampler_with_shifts(
                5,
                H3SamplerKind::ComfyLcm,
                H3_VIDEO_SHIFT,
                shift
            )
            .is_err());
        }
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    /// The pre-existing constructors are wrappers, not parallel builders: every
    /// one of them is bit-identical to the explicit-shift constructor at the
    /// family audio shift, for every kind and every reviewed grid length.
    #[test]
    fn legacy_constructors_are_exact_wrappers_over_explicit_shifts() {
        for kind in [
            H3SamplerKind::OfficialEuler,
            H3SamplerKind::ComfyResMultistep,
            H3SamplerKind::ComfyEuler,
            H3SamplerKind::ComfyLcm,
        ] {
            for grid_points in [2usize, 3, 5, 8, 9, 21, 50] {
                let explicit = H3DualSchedule::new_for_sampler_with_shifts(
                    grid_points,
                    kind,
                    H3_VIDEO_SHIFT,
                    H3_AUDIO_SHIFT,
                )
                .unwrap();
                assert_eq!(
                    H3DualSchedule::new_for_sampler(grid_points, kind).unwrap(),
                    explicit
                );
                assert_eq!(
                    H3DualSchedule::new_for_sampler_with_video_shift(
                        grid_points,
                        kind,
                        H3_VIDEO_SHIFT
                    )
                    .unwrap(),
                    explicit
                );
                assert_eq!(explicit.audio_shift(), H3_AUDIO_SHIFT);
            }
        }
        // The derived comfy carry scale at the family shifts is the 4.0 the
        // retired compile-time constant held, bit for bit.
        let step = H3DualSchedule::new_for_sampler(5, H3SamplerKind::ComfyResMultistep)
            .unwrap()
            .steps()
            .next()
            .unwrap();
        assert_eq!(step.comfy_audio_scale().to_bits(), 4.0f32.to_bits());
        assert_eq!(
            (step.audio_shift / step.video_shift).to_bits(),
            0.25f32.to_bits()
        );
    }

    /// DMAD 4-step: `lcm` + `simple`, video 12 / audio 2. Bits captured by
    /// running ComfyUI's `ModelSamplingDiscreteFlow` table, `simple_scheduler`
    /// (`comfy/samplers.py:645-652`), and `time_shift_sigma`
    /// (`comfy/ldm/minimax/model.py:36-39`) in torch f32; the decimals are the
    /// plan's `[1, .972973, .923077, .8, 0]` / `[1, .857143, .666667, .4, 0]`.
    #[test]
    fn dmad_four_step_lcm_grid_matches_comfy_float32_bits() {
        let schedule =
            H3DualSchedule::new_for_sampler_with_shifts(5, H3SamplerKind::ComfyLcm, 12.0, 2.0)
                .unwrap();
        assert_eq!(
            bits(schedule.video_sigmas()),
            [0x3f800000, 0x3f7914c2, 0x3f6c4ec5, 0x3f4ccccd, 0x00000000]
        );
        assert_eq!(
            bits(schedule.audio_sigmas()),
            [0x3f800000, 0x3f5b6db6, 0x3f2aaaaa, 0x3eccccce, 0x00000000]
        );
        assert_close(
            schedule.audio_sigmas(),
            &[1.0, 0.857_143, 0.666_667, 0.4, 0.0],
            1e-6,
        );
        assert_eq!(schedule.counts().transformer_evaluations, 4);
        assert_eq!(schedule.audio_shift(), 2.0);
    }

    /// PDMD 2-step: Euler + `simple`, video 12 / audio 6, from the same f32
    /// capture. PDMD's own sampler is plain Euler on `shift(linspace)`
    /// (`ZeamoxWang/pdmd@6b6e1063 inference/sample_diffsynth.py:130`), which
    /// the Comfy table reproduces at 3 grid points.
    #[test]
    fn pdmd_two_step_euler_grid_matches_comfy_float32_bits() {
        let schedule =
            H3DualSchedule::new_for_sampler_with_shifts(3, H3SamplerKind::ComfyEuler, 12.0, 6.0)
                .unwrap();
        assert_eq!(
            bits(schedule.video_sigmas()),
            [0x3f800000, 0x3f6c4ec5, 0x00000000]
        );
        assert_eq!(
            bits(schedule.audio_sigmas()),
            [0x3f800000, 0x3f5b6db7, 0x00000000]
        );
        assert_close(schedule.video_sigmas(), &[1.0, 0.923_077, 0.0], 1e-6);
        assert_close(schedule.audio_sigmas(), &[1.0, 0.857_143, 0.0], 1e-6);
        assert_eq!(schedule.counts().transformer_evaluations, 2);
    }

    /// FastH3 V2: `res_multistep` + `simple` at video 10 / audio 3 over 9 grid
    /// points. Sigma bits and the coupled update are a torch-f32 transcription
    /// of ComfyUI `res_multistep` (`comfy/k_diffusion/sampling.py:1462-1528`,
    /// eta 0) over the packed latent with `ModelSamplingAV`'s carried audio
    /// (`comfy/model_sampling.py:338-357`, `comfy/ldm/minimax/model.py:604-643`,
    /// `comfy/model_base.py:2171-2178`); the same script reproduces
    /// `comfy_res_multistep_matches_pinned_float32_reference`'s 12/3 pins.
    #[test]
    fn fasth3_res_multistep_matches_comfy_float32_reference_at_shift_ten_three() {
        let schedule = H3DualSchedule::new_for_sampler_with_shifts(
            9,
            H3SamplerKind::ComfyResMultistep,
            10.0,
            3.0,
        )
        .unwrap();
        assert_eq!(
            bits(schedule.video_sigmas()),
            [
                0x3f800000, 0x3f7c64f5, 0x3f77bdef, 0x3f71826a, 0x3f68ba2f, 0x3f5b6db7, 0x3f44ec4f,
                0x3f169697, 0x00000000,
            ]
        );
        assert_eq!(
            bits(schedule.audio_sigmas()),
            [
                0x3f800000, 0x3f745d19, 0x3f666664, 0x3f555557, 0x3f3fffff, 0x3f24924a, 0x3f000001,
                0x3e99999b, 0x00000000,
            ]
        );

        let device = Device::Cpu;
        let mut video = Tensor::new(&[0.25f32, -0.5, 1.0], &device).unwrap();
        let mut audio = video.clone();
        let velocities = [
            [0.1f32, -0.2, 0.3],
            [-0.4, 0.5, -0.6],
            [0.7, -0.8, 0.9],
            [-1.0, 1.1, -1.2],
            [0.3, 0.2, -0.1],
            [-0.6, -0.5, 0.4],
            [0.9, 0.8, -0.7],
            [-1.2, -1.1, 1.0],
        ];
        let expected_video = [
            [0.251_408_46f32, -0.502_816_9, 1.004_225_4],
            [0.238_161_25, -0.485_361_25, 0.982_561_2],
            [0.273_514_93, -0.526_456_3, 1.029_397_6],
            [0.197_043_18, -0.441_623_96, 0.936_204_55],
            [0.264_859_23, -0.466_574_13, 0.974_456_55],
            [0.143_059_42, -0.571_531_3, 1.054_941_5],
            [0.636_463_8, -0.140_646_31, 0.684_251_5],
            [-0.069_418_61, -0.787_705_2, 1.272_486_8],
        ];
        let expected_audio = [
            [0.254_545_48f32, -0.509_090_96, 1.013_636_6],
            [0.214_785_72, -0.456_7, 0.948_614_3],
            [0.310_965_2, -0.568_349_1, 1.075_733_1],
            [0.127_640_8, -0.365_145_42, 0.852_650_1],
            [0.260_040_55, -0.408_277_4, 0.922_360_2],
            [0.074_822_9, -0.586_749_55, 1.063_100_2],
            [0.591_744_96, -0.145_527_45, 0.681_528_6],
            [0.231_744_89, -0.475_527_5, 0.981_528_64],
        ];
        let mut sampler = H3DualSampler::new(H3SamplerKind::ComfyResMultistep);
        for (index, step) in schedule.steps().enumerate() {
            assert_eq!(step.comfy_audio_scale(), 10.0f32 / 3.0);
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

    /// The LCM rule on synthetic tensors at the DMAD 12/2 grid, against a
    /// hand-computable oracle: `x0 = x + sigma * v`, then
    /// `sigma' * eps + (1 - sigma') * x0` per modality, x0 itself at the
    /// terminal step. The expected values come from the same torch-f32
    /// capture, which also evaluated Comfy's PACKED carried-audio form and
    /// agreed with these per-modality values to 1.5e-7 (step 1) and exactly
    /// (terminal step 3).
    #[test]
    fn lcm_update_matches_the_per_modality_comfy_oracle() {
        let device = Device::Cpu;
        let schedule =
            H3DualSchedule::new_for_sampler_with_shifts(5, H3SamplerKind::ComfyLcm, 12.0, 2.0)
                .unwrap();
        let steps: Vec<_> = schedule.steps().collect();
        let video = Tensor::new(&[0.25f32, -0.5, 1.0, 0.125], &device).unwrap();
        let audio = Tensor::new(&[-0.75f32, 0.5, 0.0625, 2.0], &device).unwrap();
        let video_velocity = Tensor::new(&[0.1f32, -0.2, 0.3, -0.4], &device).unwrap();
        let audio_velocity = Tensor::new(&[-0.6f32, 0.7, 0.05, -1.5], &device).unwrap();
        let video_noise = Tensor::new(&[1.5f32, -0.25, 0.75, -2.0], &device).unwrap();
        let audio_noise = Tensor::new(&[-1.0f32, 0.5, 1.25, 0.3], &device).unwrap();

        let next_video =
            lcm_update(&video, &video_velocity, Some(&video_noise), steps[1].video).unwrap();
        let next_audio =
            lcm_update(&audio, &audio_velocity, Some(&audio_noise), steps[1].audio).unwrap();
        assert_close(
            &next_video.to_vec1::<f32>().unwrap(),
            &[1.411_330_6, -0.284_199_6, 0.791_684_03, -1.866_476],
            2.0 * f32::EPSILON,
        );
        assert_close(
            &next_audio.to_vec1::<f32>().unwrap(),
            &[-1.088_095_2, 0.7, 0.868_452_3, 0.438_095_33],
            2.0 * f32::EPSILON,
        );

        // Terminal: the clean estimate, with no noise accepted.
        let last = steps[3];
        let terminal_video = lcm_update(&video, &video_velocity, None, last.video).unwrap();
        let terminal_audio = lcm_update(&audio, &audio_velocity, None, last.audio).unwrap();
        assert_close(
            &terminal_video.to_vec1::<f32>().unwrap(),
            &[0.33, -0.66, 1.24, -0.195],
            2.0 * f32::EPSILON,
        );
        assert_close(
            &terminal_audio.to_vec1::<f32>().unwrap(),
            &[-0.99, 0.78, 0.0825, 1.4],
            2.0 * f32::EPSILON,
        );
        assert!(lcm_update(&video, &video_velocity, Some(&video_noise), last.video).is_err());
        assert!(lcm_update(&video, &video_velocity, None, steps[1].video).is_err());

        // The half-precision latent keeps its dtype; the blend runs in F32.
        let half = video.to_dtype(DType::BF16).unwrap();
        let next = lcm_update(&half, &video_velocity, Some(&video_noise), steps[1].video).unwrap();
        assert_eq!(next.dtype(), DType::BF16);
    }

    #[test]
    fn lcm_step_noise_is_seeded_per_step_and_draws_video_then_audio() {
        let device = Device::Cpu;
        let video_shape = [1usize, 3, 4];
        let audio_shape = [1usize, 5];
        let draw = |seed: u64, index: usize| {
            let (video, audio) = H3StepNoise::new(seed)
                .draw(index, &video_shape, &audio_shape, &device)
                .unwrap();
            (
                video.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                audio.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            )
        };
        // Same seed and step: identical, on every call.
        assert_eq!(draw(42, 1), draw(42, 1));
        // A different step or seed: different noise.
        assert_ne!(draw(42, 1), draw(42, 2));
        assert_ne!(draw(42, 1), draw(43, 1));

        // Video is drawn first: a video-only draw is the prefix of the stream,
        // and the audio continues it.
        let (video, audio) = draw(7, 0);
        let (combined, _) = H3StepNoise::new(7)
            .draw(0, &[17], &[0], &device)
            .map(|(v, a)| (v.to_vec1::<f32>().unwrap(), a))
            .unwrap();
        assert_eq!(&combined[..12], video.as_slice());
        assert_eq!(&combined[12..], audio.as_slice());

        // The stream is not the request's initial-noise stream (a bare
        // `StdRng::seed_from_u64(seed)`), so re-noise never replays it.
        use rand::SeedableRng;
        use rand_distr::Distribution;
        let mut initial = rand::rngs::StdRng::seed_from_u64(7);
        let first: Vec<f32> = (0..12)
            .map(|_| rand_distr::StandardNormal.sample(&mut initial))
            .collect();
        assert_ne!(first, video);
    }

    #[test]
    fn lcm_sampler_renoises_from_the_seeded_stream_and_ends_clean() {
        let device = Device::Cpu;
        let schedule =
            H3DualSchedule::new_for_sampler_with_shifts(5, H3SamplerKind::ComfyLcm, 12.0, 2.0)
                .unwrap();
        let video = Tensor::new(&[[0.25f32, -0.5], [1.0, 0.125]], &device).unwrap();
        let audio = Tensor::new(&[[-0.75f32, 0.5, 0.0625]], &device).unwrap();
        let video_velocity = Tensor::new(&[[0.1f32, -0.2], [0.3, -0.4]], &device).unwrap();
        let audio_velocity = Tensor::new(&[[-0.6f32, 0.7, 0.05]], &device).unwrap();

        // Without a seed the first non-terminal step is refused.
        let first = schedule.steps().next().unwrap();
        let error = H3DualSampler::new(H3SamplerKind::ComfyLcm)
            .step_pair(&video, &audio, &video_velocity, &audio_velocity, first)
            .unwrap_err()
            .to_string();
        assert!(error.contains("noise seed"), "{error}");

        let run = |seed: u64| {
            let mut sampler = H3DualSampler::with_noise_seed(H3SamplerKind::ComfyLcm, seed);
            let (mut v, mut a) = (video.clone(), audio.clone());
            let mut trace = Vec::new();
            for step in schedule.steps() {
                // Expected: the oracle update with this step's stream draw.
                let noise = (step.video.next_sigma > 0.0).then(|| {
                    H3StepNoise::new(seed)
                        .draw(step.evaluation_index, v.dims(), a.dims(), &device)
                        .unwrap()
                });
                let expected_video = lcm_update(
                    &v,
                    &video_velocity,
                    noise.as_ref().map(|(n, _)| n),
                    step.video,
                )
                .unwrap();
                let expected_audio = lcm_update(
                    &a,
                    &audio_velocity,
                    noise.as_ref().map(|(_, n)| n),
                    step.audio,
                )
                .unwrap();
                (v, a) = sampler
                    .step_pair(&v, &a, &video_velocity, &audio_velocity, step)
                    .unwrap();
                assert_eq!(
                    v.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                    expected_video
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                );
                assert_eq!(
                    a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                    expected_audio
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                );
                trace.push(v.flatten_all().unwrap().to_vec1::<f32>().unwrap());
            }
            trace
        };
        assert_eq!(run(9), run(9));
        assert_ne!(run(9), run(10));
        // The final step is noise-free, so it is x0 of its own input.
        let schedule_steps: Vec<_> = schedule.steps().collect();
        assert_eq!(schedule_steps.last().unwrap().video.next_sigma, 0.0);
    }

    #[test]
    fn deterministic_kinds_ignore_the_noise_seed() {
        let device = Device::Cpu;
        let sample = Tensor::new(&[0.25f32, -0.5, 1.0], &device).unwrap();
        let velocity = Tensor::new(&[0.1f32, -0.2, 0.3], &device).unwrap();
        for kind in [
            H3SamplerKind::OfficialEuler,
            H3SamplerKind::ComfyResMultistep,
            H3SamplerKind::ComfyEuler,
        ] {
            assert!(!kind.consumes_step_noise());
            let schedule = H3DualSchedule::new_for_sampler(5, kind).unwrap();
            let mut plain = H3DualSampler::new(kind);
            let mut seeded = H3DualSampler::with_noise_seed(kind, 123);
            let (mut a, mut b) = (
                (sample.clone(), sample.clone()),
                (sample.clone(), sample.clone()),
            );
            for step in schedule.steps() {
                a = plain
                    .step_pair(&a.0, &a.1, &velocity, &velocity, step)
                    .unwrap();
                b = seeded
                    .step_pair(&b.0, &b.1, &velocity, &velocity, step)
                    .unwrap();
                assert_eq!(a.0.to_vec1::<f32>().unwrap(), b.0.to_vec1::<f32>().unwrap());
                assert_eq!(a.1.to_vec1::<f32>().unwrap(), b.1.to_vec1::<f32>().unwrap());
            }
        }
        assert!(H3SamplerKind::ComfyLcm.consumes_step_noise());
        assert!(H3SamplerKind::ComfyLcm.uses_comfy_simple_grid());
        assert!(!H3SamplerKind::ComfyLcm.uses_euler_update());
    }

    #[test]
    fn sampler_kind_round_trips_through_its_stable_string() {
        for kind in [
            H3SamplerKind::OfficialEuler,
            H3SamplerKind::ComfyResMultistep,
            H3SamplerKind::ComfyEuler,
            H3SamplerKind::ComfyLcm,
        ] {
            assert_eq!(H3SamplerKind::parse(kind.as_str()), Some(kind));
        }
        assert_eq!(
            H3SamplerKind::parse("comfy-lcm"),
            Some(H3SamplerKind::ComfyLcm)
        );
        assert_eq!(
            H3SamplerKind::parse("comfy-euler"),
            Some(H3SamplerKind::ComfyEuler)
        );
        assert_eq!(H3SamplerKind::parse("nope"), None);
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

    /// The LCM sampler draws its re-noise on the host and moves it to the
    /// latent's device, so a CUDA render re-noises with the same values as a
    /// CPU one and the whole 4-step DMAD trajectory agrees across backends.
    #[cfg(feature = "cuda")]
    #[test]
    #[cfg_attr(not(target_os = "linux"), ignore)]
    fn lcm_trajectory_matches_cpu_and_cuda() {
        let Ok(cuda) = Device::new_cuda(0) else {
            return;
        };
        let schedule =
            H3DualSchedule::new_for_sampler_with_shifts(5, H3SamplerKind::ComfyLcm, 12.0, 2.0)
                .unwrap();
        let run = |device: &Device| {
            let mut video = Tensor::new(&[[0.125f32, -1.75], [3.5, 0.031_25]], device).unwrap();
            let mut audio = Tensor::new(&[[-0.5f32, 2.25, -4.0]], device).unwrap();
            let video_velocity = Tensor::new(&[[0.75f32, -0.125], [1.25, -3.0]], device).unwrap();
            let audio_velocity = Tensor::new(&[[1.5f32, -2.0, 0.0625]], device).unwrap();
            let mut sampler = H3DualSampler::with_noise_seed(H3SamplerKind::ComfyLcm, 2024);
            for step in schedule.steps() {
                (video, audio) = sampler
                    .step_pair(&video, &audio, &video_velocity, &audio_velocity, step)
                    .unwrap();
            }
            (
                video
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                audio
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
            )
        };
        let (cpu_video, cpu_audio) = run(&Device::Cpu);
        let (cuda_video, cuda_audio) = run(&cuda);
        assert_close(&cuda_video, &cpu_video, 1e-6);
        assert_close(&cuda_audio, &cpu_audio, 1e-6);
    }
}
