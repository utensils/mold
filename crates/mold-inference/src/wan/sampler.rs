//! Wan's FlowUniPC sampler — a UniPC predictor-corrector over flow-matching
//! sigmas.
//!
//! Upstream is `Wan2.1/wan/utils/fm_solvers_unipc.py` (byte-identical in
//! Wan2.2), itself a flow-matching conversion of diffusers'
//! `UniPCMultistepScheduler`. Wan runs it with `solver_order=2`,
//! `solver_type="bh2"`, `predict_x0=True`, `lower_order_final=True`,
//! `final_sigmas_type="zero"`, and an empty `disable_corrector`, so this port
//! hardcodes that configuration rather than carrying dead switches.
//!
//! **The schedule follows diffusers' flow branch, not Wan's in-repo
//! `set_timesteps`.** See [`WanSchedule::new`] — the two disagree, and the
//! diffusers form is the one that reproduces lightx2v's published Lightning
//! timesteps.

use anyhow::{bail, Result};
use candle_core::{DType, Tensor};

/// `num_train_timesteps` for every Wan checkpoint.
pub(crate) const WAN_NUM_TRAIN_TIMESTEPS: usize = 1000;

/// UniPC order. Wan ships 2 for guided sampling (`fm_solvers_unipc.py:82`).
const SOLVER_ORDER: usize = 2;

/// Diffusers nudges a leading sigma of exactly 1 down by this much so
/// `log(alpha_s0)` stays finite on the first update
/// (`scheduling_unipc_multistep.py`, flow branch).
const SIGMA_ONE_EPSILON: f64 = 1e-6;

/// How the sigma grid is laid out for one sampling run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct WanScheduleConfig {
    pub steps: usize,
    /// Flow shift. Wan ships 5.0 (T2V-1.3B, TI2V-5B), 8.0/12.0 (14B / A14B
    /// T2V), 3.0 (I2V), and 5.0 for the 4-step Lightning distills.
    pub shift: f64,
}

impl WanScheduleConfig {
    pub fn new(steps: usize, shift: f64) -> Self {
        Self { steps, shift }
    }
}

/// The resolved sigma grid and the integer timesteps handed to the DiT.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct WanSchedule {
    /// `steps + 1` entries; the last is always exactly 0.
    pub sigmas: Vec<f64>,
    /// `steps` entries, `int64(sigma * 1000)` — truncated, not rounded.
    pub timesteps: Vec<i64>,
}

impl WanSchedule {
    /// Build the grid.
    ///
    /// ```text
    /// raw      = linspace(1, 1/1000, steps + 1)[:-1]
    /// shifted  = shift * raw / (1 + (shift - 1) * raw)
    /// if |shifted[0] - 1| < 1e-6 { shifted[0] -= 1e-6 }
    /// timesteps = trunc(shifted * 1000)
    /// sigmas    = f32(shifted) ++ [0.0]
    /// ```
    ///
    /// Two things here are deliberate and both come from diffusers' flow branch
    /// rather than Wan's own `set_timesteps`:
    ///
    /// 1. **The grid starts at exactly 1.0.** Wan's file interpolates between
    ///    `sigma_max`/`sigma_min` read off a *training* grid that tops out at
    ///    `1 - 1/1000 = 0.999`, and (because its constructor already applied a
    ///    shift) can end up applying the shift twice. At 4 steps / shift 5 that
    ///    yields a final timestep of 624; diffusers, the golden fixture, and
    ///    lightx2v's published Lightning schedule all give **625**.
    /// 2. **The `1e-6` nudge on the first sigma.** `shift * 1 / (1 + (shift-1))`
    ///    is exactly 1 for every shift, which would make `log(alpha)` `-inf` on
    ///    the first predictor update. Diffusers subtracts an epsilon; Wan's file
    ///    never hits the case because its grid starts below 1.
    ///
    /// Sigmas are rounded through `f32` because both schedulers store them as
    /// `float32`; the timesteps are derived from the unrounded `f64` values,
    /// which is diffusers' order of operations.
    pub fn new(config: WanScheduleConfig) -> Result<Self> {
        if config.steps == 0 {
            bail!("Wan sampler: a schedule needs at least one step");
        }
        if config.shift <= 0.0 || !config.shift.is_finite() {
            bail!(
                "Wan sampler: flow shift must be finite and positive, got {}",
                config.shift
            );
        }
        let train = WAN_NUM_TRAIN_TIMESTEPS as f64;
        let start = 1.0;
        let end = 1.0 / train;
        let steps = config.steps;

        let mut shifted = Vec::with_capacity(steps);
        for i in 0..steps {
            let raw = start + (end - start) * (i as f64) / (steps as f64);
            shifted.push(config.shift * raw / (1.0 + (config.shift - 1.0) * raw));
        }
        if (shifted[0] - 1.0).abs() < SIGMA_ONE_EPSILON {
            shifted[0] -= SIGMA_ONE_EPSILON;
        }

        let timesteps = shifted.iter().map(|s| (s * train) as i64).collect();
        let mut sigmas: Vec<f64> = shifted.iter().map(|s| *s as f32 as f64).collect();
        sigmas.push(0.0);

        Ok(Self { sigmas, timesteps })
    }

    pub fn steps(&self) -> usize {
        self.timesteps.len()
    }
}

/// `lambda_t = log(alpha_t) - log(sigma_t)` with `alpha_t = 1 - sigma_t`
/// (`fm_solvers_unipc.py:274-275, 412-413`).
///
/// Rust's `ln` returns `-inf` at zero exactly like torch's, and that is
/// load-bearing: the terminal sigma is 0, which sends `h` to `+inf`, `hh` to
/// `-inf`, `h_phi_1` and `B_h` to `-1`, and collapses the last update to the
/// bare x0 prediction. Do not "fix" this with a clamp.
fn lambda_at(sigma: f64) -> f64 {
    (1.0 - sigma).ln() - sigma.ln()
}

/// The `R` matrix, `b` vector, and the two `B(h)` scalars shared by the
/// predictor and the corrector (`fm_solvers_unipc.py:435-455`, `578-598`).
struct BhCoefficients {
    h_phi_1: f64,
    b_h: f64,
    r: Vec<Vec<f64>>,
    b: Vec<f64>,
}

fn bh_coefficients(hh: f64, order: usize, rks: &[f64]) -> BhCoefficients {
    let h_phi_1 = hh.exp_m1();
    // solver_type "bh2"; "bh1" would be `B_h = hh`.
    let b_h = hh.exp_m1();
    let mut h_phi_k = h_phi_1 / hh - 1.0;
    let mut factorial = 1.0f64;
    let mut r = Vec::with_capacity(order);
    let mut b = Vec::with_capacity(order);
    for i in 1..=order {
        r.push(rks.iter().map(|rk| rk.powi(i as i32 - 1)).collect());
        b.push(h_phi_k * factorial / b_h);
        factorial *= (i + 1) as f64;
        h_phi_k = h_phi_k / hh - 1.0 / factorial;
    }
    BhCoefficients { h_phi_1, b_h, r, b }
}

/// `torch.linalg.solve` for the tiny dense systems UniPC builds (2x2 at
/// `solver_order = 2`). Gaussian elimination with partial pivoting.
fn solve_linear_system(matrix: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>> {
    let n = rhs.len();
    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        bail!("Wan sampler: UniPC coefficient system is not square ({n} unknowns)");
    }
    let mut aug: Vec<Vec<f64>> = matrix
        .iter()
        .zip(rhs)
        .map(|(row, r)| {
            let mut row = row.clone();
            row.push(*r);
            row
        })
        .collect();

    for col in 0..n {
        let pivot = (col..n)
            .max_by(|a, b| {
                aug[*a][col]
                    .abs()
                    .partial_cmp(&aug[*b][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("column range is non-empty");
        aug.swap(col, pivot);
        let pivot_row = aug[col].clone();
        if pivot_row[col] == 0.0 || !pivot_row[col].is_finite() {
            bail!("Wan sampler: UniPC coefficient system is singular at column {col}");
        }
        for (row, target) in aug.iter_mut().enumerate() {
            if row == col {
                continue;
            }
            let factor = target[col] / pivot_row[col];
            for (cell, source) in target.iter_mut().zip(pivot_row.iter()).skip(col) {
                *cell -= factor * source;
            }
        }
    }
    Ok((0..n).map(|i| aug[i][n] / aug[i][i]).collect())
}

/// Classifier-free guidance: `uncond + scale * (cond - uncond)`.
pub(crate) fn apply_cfg(cond: &Tensor, uncond: &Tensor, scale: f64) -> Result<Tensor> {
    let dtype = cond.dtype();
    let cond32 = cond.to_dtype(DType::F32)?;
    let uncond32 = uncond.to_dtype(DType::F32)?;
    Ok(uncond32
        .add(&cond32.sub(&uncond32)?.affine(scale, 0.0)?)?
        .to_dtype(dtype)?)
}

/// Plain Euler flow step: `x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * v`.
///
/// The fallback when UniPC is not wanted. Because the terminal sigma is 0, the
/// last Euler step is exactly `x - sigma * v`, i.e. the x0 prediction — the same
/// place FlowUniPC lands.
pub(crate) fn euler_step(
    sample: &Tensor,
    velocity: &Tensor,
    schedule: &WanSchedule,
    step_index: usize,
) -> Result<Tensor> {
    if step_index + 1 >= schedule.sigmas.len() {
        bail!(
            "Wan sampler: euler step {step_index} is past the {}-step schedule",
            schedule.steps()
        );
    }
    let dt = schedule.sigmas[step_index + 1] - schedule.sigmas[step_index];
    let dtype = sample.dtype();
    Ok(sample
        .to_dtype(DType::F32)?
        .add(&velocity.to_dtype(DType::F32)?.affine(dt, 0.0)?)?
        .to_dtype(dtype)?)
}

/// UniPC multistep predictor-corrector, order 2, `bh2`, x0-prediction.
///
/// State machine: `step` must be called once per schedule entry with
/// monotonically increasing `step_index` starting at 0, because the corrector
/// reads the sample and order recorded by the previous call.
#[derive(Clone, Debug)]
pub(crate) struct FlowUniPc {
    schedule: WanSchedule,
    /// x0-converted model outputs, oldest first — `model_outputs[-1]` upstream
    /// is the last element here.
    model_outputs: Vec<Option<Tensor>>,
    /// Warm-up counter: the solver cannot use order 2 before it has two
    /// outputs (`fm_solvers_unipc.py:721-722`).
    lower_order_nums: usize,
    last_sample: Option<Tensor>,
    /// Order chosen by the previous `step`; the corrector runs at that order.
    this_order: usize,
    next_step_index: usize,
}

impl FlowUniPc {
    pub fn new(schedule: WanSchedule) -> Self {
        Self {
            schedule,
            model_outputs: vec![None; SOLVER_ORDER],
            lower_order_nums: 0,
            last_sample: None,
            this_order: 0,
            next_step_index: 0,
        }
    }

    pub fn from_config(config: WanScheduleConfig) -> Result<Self> {
        Ok(Self::new(WanSchedule::new(config)?))
    }

    pub fn schedule(&self) -> &WanSchedule {
        &self.schedule
    }

    /// The UniPC order the most recent `step` actually used. 1 during warm-up
    /// and on the final step, 2 in between.
    pub fn last_order_used(&self) -> usize {
        self.this_order
    }

    /// The x0 prediction the most recent `step` converted and stored:
    /// `sample - sigma[step_index] * v` from the *pre-step, uncorrected*
    /// sample (`convert_model_output`, the same value the corrector anchors
    /// on). This is what live previews must project — once the corrector has
    /// rewritten the sample the predictor steps from, the naive post-step
    /// identity `x_next - sigma_next * v` no longer recovers this x0.
    /// `None` before the first `step`. Always F32; borrows the multistep
    /// history slot, so reading it costs nothing.
    pub fn last_x0(&self) -> Option<&Tensor> {
        self.model_outputs.last().and_then(|slot| slot.as_ref())
    }

    /// `model_outputs[-(back + 1)]`.
    fn model_output(&self, back: usize) -> Result<&Tensor> {
        let len = self.model_outputs.len();
        if back >= len {
            bail!("Wan sampler: UniPC asked for history {back} beyond order {len}");
        }
        self.model_outputs[len - 1 - back]
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Wan sampler: UniPC history slot {back} is empty"))
    }

    /// `convert_model_output` (`fm_solvers_unipc.py:320-333`): flow velocity to
    /// an x0 prediction, `x0 = sample - sigma_t * v`.
    fn to_x0(&self, model_output: &Tensor, sample: &Tensor, step_index: usize) -> Result<Tensor> {
        let sigma = self.schedule.sigmas[step_index];
        Ok(sample.sub(&model_output.affine(sigma, 0.0)?)?)
    }

    /// `multistep_uni_p_bh_update` (`fm_solvers_unipc.py:352-486`).
    fn predictor(&self, sample: &Tensor, step_index: usize, order: usize) -> Result<Tensor> {
        let sigmas = &self.schedule.sigmas;
        let sigma_t = sigmas[step_index + 1];
        let sigma_s0 = sigmas[step_index];
        let alpha_t = 1.0 - sigma_t;
        let lambda_s0 = lambda_at(sigma_s0);
        let h = lambda_at(sigma_t) - lambda_s0;
        let m0 = self.model_output(0)?;

        let mut d1s = Vec::new();
        for i in 1..order {
            let si = step_index
                .checked_sub(i)
                .ok_or_else(|| anyhow::anyhow!("Wan sampler: UniPC order {order} at step 0"))?;
            let mi = self.model_output(i)?;
            let rk = (lambda_at(sigmas[si]) - lambda_s0) / h;
            d1s.push(mi.sub(m0)?.affine(1.0 / rk, 0.0)?);
        }

        let hh = -h;
        let h_phi_1 = hh.exp_m1();
        let b_h = hh.exp_m1();

        // `sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0`.
        let x_t = sample
            .affine(sigma_t / sigma_s0, 0.0)?
            .sub(&m0.affine(alpha_t * h_phi_1, 0.0)?)?;
        if d1s.is_empty() {
            return Ok(x_t);
        }

        // Upstream builds R and b here too, but for `solver_order = 2` the
        // predictor's rhos are the hardcoded `[0.5]` and the general solve is
        // never reached (`fm_solvers_unipc.py:460-464`).
        let rhos_p = match order {
            2 => vec![0.5f64],
            other => bail!(
                "Wan sampler: the UniPC predictor is wired for order <= {SOLVER_ORDER}, got {other}"
            ),
        };
        let pred_res = weighted_sum(&rhos_p, &d1s)?;
        Ok(x_t.sub(&pred_res.affine(alpha_t * b_h, 0.0)?)?)
    }

    /// `multistep_uni_c_bh_update` (`fm_solvers_unipc.py:488-628`).
    fn corrector(
        &self,
        this_x0: &Tensor,
        last_sample: &Tensor,
        step_index: usize,
        order: usize,
    ) -> Result<Tensor> {
        let sigmas = &self.schedule.sigmas;
        let sigma_t = sigmas[step_index];
        let sigma_s0 = sigmas[step_index - 1];
        let alpha_t = 1.0 - sigma_t;
        let lambda_s0 = lambda_at(sigma_s0);
        let h = lambda_at(sigma_t) - lambda_s0;
        let m0 = self.model_output(0)?;

        let mut rks = Vec::new();
        let mut d1s = Vec::new();
        for i in 1..order {
            // Note the corrector reaches one step further back than the
            // predictor does: `step_index - (i + 1)` (`:564`) vs `- i` (`:421`).
            let si = step_index
                .checked_sub(i + 1)
                .ok_or_else(|| anyhow::anyhow!("Wan sampler: UniC order {order} too early"))?;
            let mi = self.model_output(i)?;
            let rk = (lambda_at(sigmas[si]) - lambda_s0) / h;
            rks.push(rk);
            d1s.push(mi.sub(m0)?.affine(1.0 / rk, 0.0)?);
        }
        rks.push(1.0);

        let hh = -h;
        let coeff = bh_coefficients(hh, order, &rks);
        let rhos_c = if order == 1 {
            vec![0.5f64]
        } else {
            solve_linear_system(&coeff.r, &coeff.b)?
        };

        let x_t = last_sample
            .affine(sigma_t / sigma_s0, 0.0)?
            .sub(&m0.affine(alpha_t * coeff.h_phi_1, 0.0)?)?;
        let d1_t = this_x0.sub(m0)?;
        let (_, rho_last) = rhos_c
            .split_at_checked(rhos_c.len() - 1)
            .ok_or_else(|| anyhow::anyhow!("Wan sampler: UniC produced no coefficients"))?;
        let mut correction = d1_t.affine(rho_last[0], 0.0)?;
        if !d1s.is_empty() {
            correction = correction.add(&weighted_sum(&rhos_c[..rhos_c.len() - 1], &d1s)?)?;
        }
        Ok(x_t.sub(&correction.affine(alpha_t * coeff.b_h, 0.0)?)?)
    }

    /// Advance one step. `model_output` is the DiT's flow velocity at
    /// `schedule.timesteps[step_index]` (post-CFG); the return is the sample at
    /// `sigma[step_index + 1]`.
    pub fn step(
        &mut self,
        model_output: &Tensor,
        step_index: usize,
        sample: &Tensor,
    ) -> Result<Tensor> {
        if step_index != self.next_step_index {
            bail!(
                "Wan sampler: FlowUniPC is a multistep solver and must be driven in order — \
                 expected step {}, got {step_index}",
                self.next_step_index
            );
        }
        if step_index >= self.schedule.steps() {
            bail!(
                "Wan sampler: step {step_index} is past the {}-step schedule",
                self.schedule.steps()
            );
        }

        let out_dtype = sample.dtype();
        let sample32 = sample.to_dtype(DType::F32)?;
        let model_output32 = model_output.to_dtype(DType::F32)?;

        // The x0 conversion deliberately uses the *uncorrected* sample: upstream
        // converts before the corrector runs and stores that value in the
        // history (`fm_solvers_unipc.py:697-711`).
        let x0 = self.to_x0(&model_output32, &sample32, step_index)?;

        let use_corrector = step_index > 0 && self.last_sample.is_some();
        let sample32 = if use_corrector {
            let last_sample = self
                .last_sample
                .as_ref()
                .expect("corrector guard checked last_sample");
            self.corrector(&x0, last_sample, step_index, self.this_order)?
        } else {
            sample32
        };

        self.model_outputs.rotate_left(1);
        let last = self.model_outputs.len() - 1;
        self.model_outputs[last] = Some(x0);

        // `lower_order_final` plus the multistep warm-up
        // (`fm_solvers_unipc.py:714-723`).
        let capped = SOLVER_ORDER.min(self.schedule.steps() - step_index);
        self.this_order = capped.min(self.lower_order_nums + 1);
        if self.this_order == 0 {
            bail!("Wan sampler: UniPC resolved a zero order at step {step_index}");
        }

        self.last_sample = Some(sample32.clone());
        let prev_sample = self.predictor(&sample32, step_index, self.this_order)?;

        if self.lower_order_nums < SOLVER_ORDER {
            self.lower_order_nums += 1;
        }
        self.next_step_index += 1;

        Ok(prev_sample.to_dtype(out_dtype)?)
    }
}

/// `sum_k weights[k] * terms[k]`, upstream's `einsum("k,bkc...->bc...")`.
fn weighted_sum(weights: &[f64], terms: &[Tensor]) -> Result<Tensor> {
    if weights.len() != terms.len() {
        bail!(
            "Wan sampler: {} UniPC coefficients for {} history terms",
            weights.len(),
            terms.len()
        );
    }
    let mut acc: Option<Tensor> = None;
    for (weight, term) in weights.iter().zip(terms) {
        let scaled = term.affine(*weight, 0.0)?;
        acc = Some(match acc {
            None => scaled,
            Some(prev) => prev.add(&scaled)?,
        });
    }
    acc.ok_or_else(|| anyhow::anyhow!("Wan sampler: weighted sum over an empty history"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    // Golden vectors generated from diffusers 0.38.0 `UniPCMultistepScheduler`
    // with `use_flow_sigmas=True, solver_order=2, solver_type="bh2",
    // predict_x0=True, lower_order_final=True, final_sigmas_type="zero",
    // timestep_spacing="linspace"` — the config Wan ships.

    /// `steps=4`, `shift=5.0`.
    const GOLDEN_S4_SHIFT5_0_SIGMAS: &[f64] = &[
        0.9999989867210388,
        0.9375780820846558,
        0.8336109519004822,
        0.6259360909461975,
        0.0,
    ];
    const GOLDEN_S4_SHIFT5_0_TIMESTEPS: &[i64] = &[999, 937, 833, 625];
    /// `steps=6`, `shift=3.0`.
    const GOLDEN_S6_SHIFT3_0_SIGMAS: &[f64] = &[
        0.9999989867210388,
        0.937570333480835,
        0.8573265075683594,
        0.7503747940063477,
        0.6007194519042969,
        0.3764044940471649,
        0.0,
    ];
    const GOLDEN_S6_SHIFT3_0_TIMESTEPS: &[i64] = &[999, 937, 857, 750, 600, 376];
    /// `steps=8`, `shift=3.0`.
    const GOLDEN_S8_SHIFT3_0_SIGMAS: &[f64] = &[
        0.9999989867210388,
        0.9545950293540955,
        0.900119960308075,
        0.8335554599761963,
        0.7503747940063477,
        0.643468976020813,
        0.5009989738464355,
        0.3016776442527771,
        0.0,
    ];
    const GOLDEN_S8_SHIFT3_0_TIMESTEPS: &[i64] = &[999, 954, 900, 833, 750, 643, 500, 301];
    /// `steps=20`, `shift=8.0`.
    const GOLDEN_S20_SHIFT8_0_SIGMAS: &[f64] = &[
        0.9999989867210388,
        0.9934709072113037,
        0.9863163828849792,
        0.9784421324729919,
        0.969733715057373,
        0.9600511789321899,
        0.9492214918136597,
        0.9370278120040894,
        0.9231951832771301,
        0.9073694348335266,
        0.8890862464904785,
        0.867725133895874,
        0.8424373269081116,
        0.8120304942131042,
        0.7747753262519836,
        0.7280645966529846,
        0.6677752137184143,
        0.5869792699813843,
        0.4730703830718994,
        0.30044594407081604,
        0.0,
    ];
    const GOLDEN_S20_SHIFT8_0_TIMESTEPS: &[i64] = &[
        999, 993, 986, 978, 969, 960, 949, 937, 923, 907, 889, 867, 842, 812, 774, 728, 667, 586,
        473, 300,
    ];
    /// `steps=40`, `shift=12.0`.
    const GOLDEN_S40_SHIFT12_0_SIGMAS: &[f64] = &[
        0.9999989867210388,
        0.9978699684143066,
        0.9956377744674683,
        0.9932957887649536,
        0.9908357858657837,
        0.9882485866546631,
        0.98552405834198,
        0.9826509356498718,
        0.9796168208122253,
        0.9764077067375183,
        0.9730080366134644,
        0.9694002866744995,
        0.9655647873878479,
        0.9614792466163635,
        0.9571183919906616,
        0.9524534940719604,
        0.9474514722824097,
        0.9420745372772217,
        0.936278760433197,
        0.9300133585929871,
        0.9232187867164612,
        0.9158250689506531,
        0.9077492356300354,
        0.8988924026489258,
        0.889135479927063,
        0.8783339262008667,
        0.8663104772567749,
        0.8528453707695007,
        0.8376628160476685,
        0.820411741733551,
        0.8006386160850525,
        0.7777466773986816,
        0.7509349584579468,
        0.7191022634506226,
        0.6806926727294922,
        0.633432924747467,
        0.573866069316864,
        0.49646490812301636,
        0.3918100595474243,
        0.24243131279945374,
        0.0,
    ];
    const GOLDEN_S40_SHIFT12_0_TIMESTEPS: &[i64] = &[
        999, 997, 995, 993, 990, 988, 985, 982, 979, 976, 973, 969, 965, 961, 957, 952, 947, 942,
        936, 930, 923, 915, 907, 898, 889, 878, 866, 852, 837, 820, 800, 777, 750, 719, 680, 633,
        573, 496, 391, 242,
    ];
    /// `steps=50`, `shift=5.0`.
    const GOLDEN_S50_SHIFT5_0_SIGMAS: &[f64] = &[
        0.9999989867210388,
        0.9959390759468079,
        0.9917441010475159,
        0.9874082207679749,
        0.982924222946167,
        0.978284478187561,
        0.9734807014465332,
        0.9685039520263672,
        0.9633448123931885,
        0.9579930305480957,
        0.9524376392364502,
        0.9466667175292969,
        0.9406675696372986,
        0.9344263076782227,
        0.9279280304908752,
        0.9211564660072327,
        0.914094090461731,
        0.9067216515541077,
        0.8990183472633362,
        0.8909614086151123,
        0.8825258612632751,
        0.8736844062805176,
        0.8644070029258728,
        0.8546605706214905,
        0.8444086909294128,
        0.8336109519004822,
        0.8222225308418274,
        0.810193657875061,
        0.7974687218666077,
        0.7839854955673218,
        0.7696741223335266,
        0.7544559240341187,
        0.7382418513298035,
        0.720930814743042,
        0.7024076581001282,
        0.6825404167175293,
        0.6611772775650024,
        0.6381427049636841,
        0.6132325530052185,
        0.5862079858779907,
        0.55678790807724,
        0.5246390700340271,
        0.48936325311660767,
        0.45048099756240845,
        0.4074093997478485,
        0.3594328761100769,
        0.30566298961639404,
        0.24498295783996582,
        0.175969198346138,
        0.09677836298942566,
        0.0,
    ];
    const GOLDEN_S50_SHIFT5_0_TIMESTEPS: &[i64] = &[
        999, 995, 991, 987, 982, 978, 973, 968, 963, 957, 952, 946, 940, 934, 927, 921, 914, 906,
        899, 890, 882, 873, 864, 854, 844, 833, 822, 810, 797, 783, 769, 754, 738, 720, 702, 682,
        661, 638, 613, 586, 556, 524, 489, 450, 407, 359, 305, 244, 175, 96,
    ];

    /// 6 steps, shift 3.0, fake model `v = 0.05 * sample + t / 1000`.
    const GOLDEN_TRACE_INITIAL: [f32; 4] = [0.5, -1.25, 2.0, 0.125];
    // Held at the fixture's own `f64` precision — the solver runs in `f32`, so
    // widening the result to compare is the honest direction.
    const GOLDEN_TRACE_STEPS: &[(i64, [f64; 4])] = &[
        (
            999,
            [
                0.43607306480407715,
                -1.308464527130127,
                1.9313908815383911,
                0.06224360316991806,
            ],
        ),
        (
            937,
            [
                0.36109569668769836,
                -1.3764506578445435,
                1.8504210710525513,
                -0.01123564038425684,
            ],
        ),
        (
            857,
            [
                0.2739112377166748,
                -1.4543702602386475,
                1.7552953958511353,
                -0.09643477201461792,
            ],
        ),
        (
            750,
            [
                0.1705636978149414,
                -1.5448309183120728,
                1.640902042388916,
                -0.19702085852622986,
            ],
        ),
        (
            600,
            [
                0.06120988726615906,
                -1.6350579261779785,
                1.5151540040969849,
                -0.302276074886322,
            ],
        ),
        (
            376,
            [
                -0.08147017657756805,
                -1.7458138465881348,
                1.3451104164123535,
                -0.4381152391433716,
            ],
        ),
    ];

    fn schedule(steps: usize, shift: f64) -> WanSchedule {
        WanSchedule::new(WanScheduleConfig::new(steps, shift)).expect("schedule builds")
    }

    fn tensor(values: &[f32]) -> Tensor {
        Tensor::from_slice(values, (1, values.len()), &Device::Cpu).expect("tensor")
    }

    fn values(t: &Tensor) -> Vec<f32> {
        t.flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("to_vec1")
    }

    #[test]
    fn schedules_match_the_golden_sigmas_and_timesteps() {
        let cases: &[(usize, f64, &[f64], &[i64])] = &[
            (
                4,
                5.0,
                GOLDEN_S4_SHIFT5_0_SIGMAS,
                GOLDEN_S4_SHIFT5_0_TIMESTEPS,
            ),
            (
                6,
                3.0,
                GOLDEN_S6_SHIFT3_0_SIGMAS,
                GOLDEN_S6_SHIFT3_0_TIMESTEPS,
            ),
            (
                8,
                3.0,
                GOLDEN_S8_SHIFT3_0_SIGMAS,
                GOLDEN_S8_SHIFT3_0_TIMESTEPS,
            ),
            (
                20,
                8.0,
                GOLDEN_S20_SHIFT8_0_SIGMAS,
                GOLDEN_S20_SHIFT8_0_TIMESTEPS,
            ),
            (
                40,
                12.0,
                GOLDEN_S40_SHIFT12_0_SIGMAS,
                GOLDEN_S40_SHIFT12_0_TIMESTEPS,
            ),
            (
                50,
                5.0,
                GOLDEN_S50_SHIFT5_0_SIGMAS,
                GOLDEN_S50_SHIFT5_0_TIMESTEPS,
            ),
        ];
        for (steps, shift, want_sigmas, want_timesteps) in cases {
            let got = schedule(*steps, *shift);
            assert_eq!(
                got.sigmas.len(),
                steps + 1,
                "steps={steps} shift={shift}: sigma count"
            );
            assert_eq!(
                got.timesteps, *want_timesteps,
                "steps={steps} shift={shift}: timesteps"
            );
            for (i, (g, w)) in got.sigmas.iter().zip(want_sigmas.iter()).enumerate() {
                assert!(
                    (g - w).abs() < 1e-6,
                    "steps={steps} shift={shift}: sigma[{i}] got {g}, want {w}"
                );
            }
            assert_eq!(
                got.sigmas[*steps], 0.0,
                "steps={steps} shift={shift}: terminal sigma must be exactly zero"
            );
        }
    }

    /// The 4-step shift-5 timesteps are lightx2v's published Lightning
    /// schedule. This is the cross-validation that pins the schedule to
    /// diffusers' flow branch rather than Wan's in-repo `set_timesteps`, which
    /// yields 624 on the last step.
    #[test]
    fn four_step_lightning_schedule_matches_lightx2v() {
        assert_eq!(schedule(4, 5.0).timesteps, vec![999, 937, 833, 625]);
    }

    /// A leading sigma of exactly 1 would make `log(alpha_s0)` infinite on the
    /// first predictor update, so diffusers nudges it down by 1e-6. The nudge
    /// fires for every shift because `shift * 1 / (1 + (shift - 1))` is exactly
    /// 1 — which is why all six golden schedules share a first sigma.
    #[test]
    fn leading_sigma_is_nudged_below_one_for_every_shift() {
        for shift in [1.0, 3.0, 5.0, 8.0, 12.0] {
            let sigmas = schedule(7, shift).sigmas;
            assert!(sigmas[0] < 1.0, "shift={shift}: sigma[0] = {}", sigmas[0]);
            assert!(
                (sigmas[0] - (1.0 - SIGMA_ONE_EPSILON)).abs() < 1e-7,
                "shift={shift}: sigma[0] = {}",
                sigmas[0]
            );
            assert!(lambda_at(sigmas[0]).is_finite());
        }
    }

    /// The full predictor/corrector trace. This is what pins the bh2 multistep
    /// math — the schedule test alone would pass with a wrong solver.
    #[test]
    fn multistep_trace_matches_the_golden_vectors() {
        let mut solver =
            FlowUniPc::from_config(WanScheduleConfig::new(6, 3.0)).expect("solver builds");
        let mut sample = tensor(&GOLDEN_TRACE_INITIAL);
        for (step_index, (want_t, want_sample)) in GOLDEN_TRACE_STEPS.iter().enumerate() {
            let t = solver.schedule().timesteps[step_index];
            assert_eq!(t, *want_t, "step {step_index}: timestep");
            // The fixture's fake model: v = 0.05 * sample + t / 1000.
            let velocity = sample.affine(0.05, t as f64 / 1000.0).expect("fake model");
            sample = solver.step(&velocity, step_index, &sample).expect("step");
            for (i, (g, w)) in values(&sample).iter().zip(want_sample.iter()).enumerate() {
                assert!(
                    (f64::from(*g) - w).abs() < 1e-4,
                    "step {step_index} element {i}: got {g}, want {w}"
                );
            }
        }
    }

    /// Warm-up forces order 1 on the first step and `lower_order_final` forces
    /// it again on the last; everything between runs at the full order 2.
    #[test]
    fn lower_order_final_and_warmup_drive_the_order_sequence() {
        let mut solver =
            FlowUniPc::from_config(WanScheduleConfig::new(6, 3.0)).expect("solver builds");
        let mut sample = tensor(&GOLDEN_TRACE_INITIAL);
        let mut orders = Vec::new();
        for step_index in 0..solver.schedule().steps() {
            let t = solver.schedule().timesteps[step_index];
            let velocity = sample.affine(0.05, t as f64 / 1000.0).expect("fake model");
            sample = solver.step(&velocity, step_index, &sample).expect("step");
            orders.push(solver.last_order_used());
        }
        assert_eq!(orders, vec![1, 2, 2, 2, 2, 1]);

        // A two-step run is order 1 throughout: step 0 is warm-up and step 1 is
        // the final step.
        let mut solver =
            FlowUniPc::from_config(WanScheduleConfig::new(2, 5.0)).expect("solver builds");
        let mut sample = tensor(&GOLDEN_TRACE_INITIAL);
        let mut orders = Vec::new();
        for step_index in 0..2 {
            let velocity = sample.affine(0.05, 0.5).expect("fake model");
            sample = solver.step(&velocity, step_index, &sample).expect("step");
            orders.push(solver.last_order_used());
        }
        assert_eq!(orders, vec![1, 1]);
    }

    /// The terminal sigma is zero, so `h` is infinite and `h_phi_1` and `B_h`
    /// both collapse to -1, leaving the final sample equal to the x0
    /// prediction. Guarding this because a naive epsilon-clamp on `log(0)`
    /// would silently break it.
    ///
    /// It also pins the ordering quirk in `step`: the x0 that lands in the
    /// history is built from the sample as the caller passed it in, *before*
    /// the corrector rewrites that sample. The two candidate answers differ, so
    /// this test fails if the conversion is moved after the correction.
    #[test]
    fn final_unipc_step_lands_on_the_uncorrected_x0_prediction() {
        let mut solver =
            FlowUniPc::from_config(WanScheduleConfig::new(3, 5.0)).expect("solver builds");
        let mut sample = tensor(&[0.4, -0.9, 1.7, 0.05]);
        let sigmas = solver.schedule().sigmas.clone();
        let mut last_input = sample.clone();
        let mut last_velocity = sample.clone();
        for step_index in 0..3 {
            let t = solver.schedule().timesteps[step_index];
            let velocity = sample.affine(0.05, t as f64 / 1000.0).expect("fake model");
            last_input = sample.clone();
            last_velocity = velocity.clone();
            sample = solver.step(&velocity, step_index, &sample).expect("step");
        }

        let scaled_velocity = last_velocity.affine(sigmas[2], 0.0).expect("scale");
        let want = last_input.sub(&scaled_velocity).expect("x0");
        for (g, w) in values(&sample).iter().zip(values(&want).iter()) {
            assert!((g - w).abs() < 1e-6, "got {g}, want {w}");
        }

        // The corrector really did move the sample, so the assertion above is
        // not accidentally true of both candidates.
        let corrected = solver.last_sample.as_ref().expect("last sample recorded");
        let from_corrected = corrected.sub(&scaled_velocity).expect("x0");
        assert!(
            values(&from_corrected)
                .iter()
                .zip(values(&want).iter())
                .any(|(a, b)| (a - b).abs() > 1e-3),
            "corrector left the sample unchanged, so this test proves nothing"
        );
    }

    #[test]
    fn euler_step_matches_the_closed_form() {
        let sched = schedule(4, 5.0);
        let sample = tensor(&[1.0, -2.0, 0.5, 0.0]);
        let velocity = tensor(&[0.25, 0.5, -1.0, 2.0]);
        let got = euler_step(&sample, &velocity, &sched, 1).expect("euler");
        let dt = sched.sigmas[2] - sched.sigmas[1];
        let want: Vec<f32> = values(&sample)
            .iter()
            .zip(values(&velocity).iter())
            .map(|(x, v)| x + (dt as f32) * v)
            .collect();
        for (g, w) in values(&got).iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-6, "got {g}, want {w}");
        }
    }

    #[test]
    fn euler_terminal_step_lands_on_the_x0_prediction() {
        let sched = schedule(4, 5.0);
        let sample = tensor(&[1.0, -2.0, 0.5, 0.0]);
        let velocity = tensor(&[0.25, 0.5, -1.0, 2.0]);
        let last = sched.steps() - 1;
        let got = euler_step(&sample, &velocity, &sched, last).expect("euler");
        let sigma = sched.sigmas[last];
        let want: Vec<f32> = values(&sample)
            .iter()
            .zip(values(&velocity).iter())
            .map(|(x, v)| x - (sigma as f32) * v)
            .collect();
        for (g, w) in values(&got).iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-6, "got {g}, want {w}");
        }
    }

    #[test]
    fn euler_step_rejects_a_step_past_the_schedule() {
        let sched = schedule(4, 5.0);
        let sample = tensor(&[1.0, 2.0]);
        assert!(euler_step(&sample, &sample, &sched, 4).is_err());
    }

    #[test]
    fn apply_cfg_matches_the_definition() {
        let cond = tensor(&[1.0, 2.0, -1.0]);
        let uncond = tensor(&[0.5, 0.0, 1.0]);
        let got = values(&apply_cfg(&cond, &uncond, 5.0).expect("cfg"));
        let want = [0.5 + 5.0 * 0.5, 0.0 + 5.0 * 2.0, 1.0 + 5.0 * -2.0];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-6, "got {g}, want {w}");
        }
        // Scale 1 is a pure conditional pass-through.
        let passthrough = values(&apply_cfg(&cond, &uncond, 1.0).expect("cfg"));
        for (g, w) in passthrough.iter().zip(values(&cond).iter()) {
            assert!((g - w).abs() < 1e-6);
        }
    }

    #[test]
    fn solve_linear_system_handles_the_order_two_case() {
        // R = [[1, 1], [rk, 1]] is the shape UniC builds at order 2.
        let matrix = vec![vec![1.0, 1.0], vec![0.25, 1.0]];
        let rhs = vec![0.5, 0.125];
        let got = solve_linear_system(&matrix, &rhs).expect("solve");
        // Closed form: det = 1 - rk.
        let det = 1.0 - 0.25;
        let want = [(0.5 - 0.125) / det, (0.125 - 0.25 * 0.5) / det];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-12, "got {g}, want {w}");
        }
    }

    #[test]
    fn solve_linear_system_rejects_a_singular_system() {
        let matrix = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        assert!(solve_linear_system(&matrix, &[1.0, 2.0]).is_err());
    }

    /// `expm1(-inf) = -1` is what makes the terminal step work; pin it rather
    /// than trusting the platform libm.
    #[test]
    fn infinite_step_size_collapses_the_bh_coefficients() {
        let coeff = bh_coefficients(f64::NEG_INFINITY, 1, &[1.0]);
        assert_eq!(coeff.h_phi_1, -1.0);
        assert_eq!(coeff.b_h, -1.0);
        assert!(lambda_at(0.0).is_infinite() && lambda_at(0.0) > 0.0);
    }

    #[test]
    fn step_must_be_driven_in_order() {
        let mut solver =
            FlowUniPc::from_config(WanScheduleConfig::new(4, 5.0)).expect("solver builds");
        let sample = tensor(&[1.0, 2.0]);
        assert!(solver.step(&sample, 1, &sample).is_err());
        solver.step(&sample, 0, &sample).expect("first step");
        assert!(solver.step(&sample, 0, &sample).is_err());
    }

    #[test]
    fn step_rejects_running_past_the_schedule() {
        let mut solver =
            FlowUniPc::from_config(WanScheduleConfig::new(2, 5.0)).expect("solver builds");
        let mut sample = tensor(&[1.0, 2.0]);
        for step_index in 0..2 {
            sample = solver.step(&sample, step_index, &sample).expect("step");
        }
        assert!(solver.step(&sample, 2, &sample).is_err());
    }

    #[test]
    fn schedule_rejects_degenerate_configs() {
        assert!(WanSchedule::new(WanScheduleConfig::new(0, 5.0)).is_err());
        assert!(WanSchedule::new(WanScheduleConfig::new(4, 0.0)).is_err());
        assert!(WanSchedule::new(WanScheduleConfig::new(4, -1.0)).is_err());
        assert!(WanSchedule::new(WanScheduleConfig::new(4, f64::NAN)).is_err());
    }

    /// A single-step schedule is the degenerate case the shape contract has to
    /// survive: order 1, sigma 0.999999 to 0.
    #[test]
    fn single_step_schedule_runs() {
        let mut solver =
            FlowUniPc::from_config(WanScheduleConfig::new(1, 5.0)).expect("solver builds");
        assert_eq!(solver.schedule().timesteps, vec![999]);
        let sample = tensor(&[0.25, -0.5]);
        let velocity = tensor(&[1.0, 1.0]);
        let out = solver.step(&velocity, 0, &sample).expect("step");
        // Order 1 with sigma_t = 0: x_t = m0 = sample - sigma_s0 * v.
        let sigma = solver.schedule().sigmas[0];
        for (g, (x, v)) in values(&out)
            .iter()
            .zip(values(&sample).iter().zip(values(&velocity).iter()))
        {
            let want = x - (sigma as f32) * v;
            assert!((g - want).abs() < 1e-6, "got {g}, want {want}");
        }
        assert_eq!(solver.last_order_used(), 1);
    }

    /// `last_x0` must hand back the x0 the solver itself converted — from the
    /// *pre-step, uncorrected* sample at `sigma[index]` — and on the corrector
    /// steps that must NOT equal the naive post-step identity
    /// `x_next - sigma_next * v` (#791 preview fix). Two steps coincide by
    /// construction and are excluded from the divergence check: the order-1
    /// first step (the exponential-integrator update cancels back to exactly
    /// the linear identity) and the terminal step (`sigma_next = 0` and the
    /// final order-1 predictor returns m0 itself).
    #[test]
    fn last_x0_is_the_pre_step_conversion_not_the_post_step_identity() {
        let schedule = schedule(4, 8.0);
        let mut solver = FlowUniPc::new(schedule.clone());
        assert!(
            solver.last_x0().is_none(),
            "no x0 exists before the first step"
        );

        let mut x = tensor(&[0.8, -0.4, 1.2, 0.3]);
        for index in 0..schedule.steps() {
            // A nonlinear velocity field, v = 0.6 x^2 - 0.4 x + 0.2: the
            // corrector's adjustment scales with how much successive x0
            // estimates move, so a near-linear field would make the
            // divergence below vanish into the tolerance.
            let v = x
                .sqr()
                .unwrap()
                .affine(0.6, 0.2)
                .unwrap()
                .sub(&x.affine(0.4, 0.0).unwrap())
                .unwrap();
            let x_pre = x.clone();
            x = solver.step(&v, index, &x).unwrap();

            let sigma = schedule.sigmas[index];
            let expected = values(&x_pre.sub(&v.affine(sigma, 0.0).unwrap()).unwrap());
            let got = values(solver.last_x0().expect("step records an x0"));
            for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
                assert!(
                    (g - e).abs() < 1e-6,
                    "step {index} element {i}: got {g}, want {e}"
                );
            }

            if index >= 1 && index + 1 < schedule.steps() {
                let naive = values(
                    &x.sub(&v.affine(schedule.sigmas[index + 1], 0.0).unwrap())
                        .unwrap(),
                );
                let max_diff = got
                    .iter()
                    .zip(&naive)
                    .map(|(g, n)| (g - n).abs())
                    .fold(0f32, f32::max);
                assert!(
                    max_diff > 1e-4,
                    "step {index}: the corrected trajectory should break the naive \
                     identity, max diff {max_diff}"
                );
            }
        }
    }
}
