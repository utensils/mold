//! Wan 2.2 A14B's two-expert mixture.
//!
//! A14B is not a 14B model with a 14B partner bolted on: it is one 27 B-parameter
//! network split across the *noise* axis. Two complete 14 B transformers share an
//! architecture, a VAE, and a text encoder, and the sampler runs the high-noise
//! expert over the early, structural part of the schedule and the low-noise one
//! over the late, detail part. The switch happens once, at a fixed integer
//! timestep, so exactly one expert is ever GPU-resident.
//!
//! Upstream carries the split point as a fraction of the 1000-step training
//! schedule — `t2v_A14B.boundary = 0.875` and `i2v_A14B.boundary = 0.900`
//! (`Wan2.2/wan/configs/wan_{t2v,i2v}_A14B.py:36`) — and picks the expert with
//! `t >= boundary * num_train_timesteps`. Both numbers live on
//! [`WanExpertPair`] rather than being read at the call site, because a boundary
//! that disagrees with the checkpoint it was loaded for is invisible: the render
//! succeeds and simply uses the wrong network for part of the schedule.
//!
//! **VRAM is the max over the pair, not the sum.** Admission should size an
//! A14B plan against one expert (10.8 GB at Q5_K_M, 15.4 GB at Q8_0) plus the
//! adapters, because [`WanExperts`] drops the resident expert before loading its
//! partner. Disk is the sum.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device};

use crate::progress::{ProgressPhase, ProgressReporter};
use crate::wan::lora::WanLoraRegistry;
use crate::wan::model::transformer::{WanTransformer, WanTransformerConfig};

/// Timesteps in the training schedule, which the boundary fraction scales
/// (`num_train_timesteps` in every Wan config).
const NUM_TRAIN_TIMESTEPS: f64 = 1000.0;

/// `t2v_A14B.boundary` (`wan/configs/wan_t2v_A14B.py:36`).
const T2V_BOUNDARY_FRACTION: f64 = 0.875;

/// `i2v_A14B.boundary` (`wan/configs/wan_i2v_A14B.py:36`).
const I2V_BOUNDARY_FRACTION: f64 = 0.900;

/// Which half of an A14B pair a timestep belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WanExpertRole {
    /// Runs first, over the structural part of the schedule.
    HighNoise,
    /// Runs after the boundary, over detail.
    LowNoise,
}

impl WanExpertRole {
    pub fn label(self) -> &'static str {
        match self {
            Self::HighNoise => "high-noise",
            Self::LowNoise => "low-noise",
        }
    }

    /// The other half of the pair.
    pub fn partner(self) -> Self {
        match self {
            Self::HighNoise => Self::LowNoise,
            Self::LowNoise => Self::HighNoise,
        }
    }

    /// The one boundary comparison, upstream's `t >= boundary * 1000`
    /// (`Wan2.2/wan/text2video.py:343`). Both decisions that depend on the
    /// boundary — the expert swap and per-expert guidance selection — route
    /// through here, so they can never disagree about which side a timestep
    /// is on.
    pub fn at(boundary_timestep: i64, timestep: i64) -> Self {
        if timestep >= boundary_timestep {
            Self::HighNoise
        } else {
            Self::LowNoise
        }
    }
}

/// Upstream A14B guidance is a per-expert pair, not a scalar: each expert was
/// tuned with its own CFG scale, selected by the same boundary rule as the
/// expert swap. A single compromise scale under-guides the structural
/// high-noise phase and over-guides the detail phase relative to the
/// reference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct WanExpertGuidance {
    pub high_noise: f64,
    pub low_noise: f64,
}

impl WanExpertGuidance {
    /// Upstream's `sample_guide_scale = (low, high)` for the task:
    /// T2V `(3.0, 4.0)` (`wan_t2v_A14B.py:37`), I2V `(3.5, 3.5)`
    /// (`wan_i2v_A14B.py:37`). Task identity is the conditioning shape, like
    /// the boundary itself.
    pub fn upstream_for(channel_concat: bool) -> Self {
        if channel_concat {
            Self {
                high_noise: 3.5,
                low_noise: 3.5,
            }
        } else {
            Self {
                high_noise: 4.0,
                low_noise: 3.0,
            }
        }
    }

    pub fn for_role(self, role: WanExpertRole) -> f64 {
        match role {
            WanExpertRole::HighNoise => self.high_noise,
            WanExpertRole::LowNoise => self.low_noise,
        }
    }

    /// The strongest scale any step will use — what decides whether the
    /// unconditional prompt needs encoding at all.
    pub fn max(self) -> f64 {
        self.high_noise.max(self.low_noise)
    }
}

/// One expert's weights and whatever adapter belongs to it.
#[derive(Clone, Debug)]
pub(crate) struct WanExpertSlot {
    pub path: PathBuf,
    /// Each expert of a distilled pair is trained separately, so the adapters
    /// are not interchangeable.
    pub loras: WanLoraRegistry,
}

/// A resolved A14B pair: both experts and the boundary they switch at.
#[derive(Clone, Debug)]
pub(crate) struct WanExpertPair {
    pub high_noise: WanExpertSlot,
    pub low_noise: WanExpertSlot,
    /// Integer timestep at or above which the high-noise expert runs. Same
    /// units as [`crate::wan::sampler::WanSchedule::timesteps`], which is
    /// `int64(sigma * 1000)`.
    pub boundary_timestep: i64,
}

impl WanExpertPair {
    /// The boundary for a checkpoint that conditions by channel concat (I2V) or
    /// not (T2V). Shape-driven like the rest of this family's configuration —
    /// a repack or a fine-tune can carry any name, but the conditioning shape
    /// is the task.
    pub fn boundary_for(channel_concat: bool) -> i64 {
        let fraction = if channel_concat {
            I2V_BOUNDARY_FRACTION
        } else {
            T2V_BOUNDARY_FRACTION
        };
        (fraction * NUM_TRAIN_TIMESTEPS).round() as i64
    }

    pub fn role_for(&self, timestep: i64) -> WanExpertRole {
        WanExpertRole::at(self.boundary_timestep, timestep)
    }

    fn slot(&self, role: WanExpertRole) -> &WanExpertSlot {
        match role {
            WanExpertRole::HighNoise => &self.high_noise,
            WanExpertRole::LowNoise => &self.low_noise,
        }
    }
}

/// Load a Wan transformer from either container, with a LoRA stack applied.
///
/// GGUF is chosen by extension, which is how the whole ecosystem labels these
/// files; there is no in-band marker to prefer. The two containers apply
/// adapters differently — see [`crate::wan::lora`] — but that is invisible here.
/// `files` is every file the DiT's weights span. A diffusers export shards
/// the transformer, so passing only the first would fail on whichever tensor
/// landed in a later shard — `Wan2.2-TI2V-5B-Turbo-Diffusers` puts
/// `blocks.29.ffn.net.2.weight` and the output projection in a second file.
/// GGUF is always one file by construction.
pub(crate) fn load_transformer(
    files: &[PathBuf],
    config: WanTransformerConfig,
    device: &Device,
    dtype: DType,
    loras: &WanLoraRegistry,
) -> Result<WanTransformer> {
    load_transformer_with_offload(files, config, device, dtype, loras, None)
}

/// [`load_transformer`] that may park blocks in host RAM to fit the render
/// (#776 item 3).
///
/// `activation_bytes` is the denoise activation budget this render needs after
/// the weights land. `None` keeps every block resident unless
/// `MOLD_WAN_OFFLOAD_BLOCKS` forces otherwise.
pub(crate) fn load_transformer_with_offload(
    files: &[PathBuf],
    config: WanTransformerConfig,
    device: &Device,
    dtype: DType,
    loras: &WanLoraRegistry,
    activation_bytes: Option<u64>,
) -> Result<WanTransformer> {
    let first = files
        .first()
        .ok_or_else(|| anyhow::anyhow!("Wan: no transformer weight files supplied"))?;
    let transformer = if is_gguf(first) {
        WanTransformer::from_gguf_with_offload(first, config, device, loras, activation_bytes)
    } else {
        WanTransformer::from_safetensors_with_loras(files, config, device, dtype, loras)
    };
    transformer.with_context(|| format!("loading Wan transformer {}", first.display()))
}

pub(crate) fn is_gguf(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
}

/// Warms the operating system's page cache for the expert that has not run
/// yet (#802 item 3).
///
/// The A14B swap is a cold read in the middle of the denoise loop. The
/// existing "the file was just read, so the page cache still holds it"
/// rationale covers the *outgoing* expert; the incoming 10.8-15.4 GB file has
/// not been read this run and, on a 32-64 GB host, has been evicted behind
/// UMT5 (11.4 GB) and the resident expert. On the 4-step fast tier that cold
/// read is a visible slice of a ~90-120 s render.
///
/// This is pure host I/O: the bytes are read sequentially and discarded, so
/// **no VRAM is touched and the max-of-pair invariant is untouched**. The work
/// runs on a background thread while the first expert denoises, and the stop
/// flag is checked per chunk so a cancelled render does not leave a thread
/// grinding through fifteen gigabytes.
struct ExpertPrefetch {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl ExpertPrefetch {
    /// Begin warming `path`. Returns `None` when there is nothing to do.
    ///
    /// `MOLD_WAN_PREFETCH=0` turns it off. That exists so the effect can be
    /// measured at all — the mechanism is the page cache, so an A/B needs a
    /// way to run the same render without it — and as an escape hatch on a
    /// host where the extra sequential read competes for I/O.
    fn start(path: PathBuf) -> Option<Self> {
        if crate::runtime_env::value("MOLD_WAN_PREFETCH").as_deref() == Some("0") {
            return None;
        }
        if !path.is_file() {
            return None;
        }
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let handle = std::thread::Builder::new()
            .name("wan-expert-prefetch".to_string())
            .spawn(move || {
                use std::io::Read;
                let Ok(mut file) = std::fs::File::open(&path) else {
                    return;
                };
                // 8 MiB keeps the syscall count low without holding a large
                // buffer; the bytes themselves are thrown away, the point is
                // the page cache they leave behind.
                let mut buffer = vec![0u8; 8 * 1024 * 1024];
                while !flag.load(Ordering::Relaxed) {
                    match file.read(&mut buffer) {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {}
                    }
                }
            })
            .ok()?;
        Some(Self {
            stop,
            handle: Some(handle),
        })
    }
}

impl Drop for ExpertPrefetch {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// The denoise loop's weight source.
///
/// Every Wan generation goes through this, single-expert checkpoints included,
/// so the loop has one shape rather than a branch that only A14B exercises.
///
/// Both variants are boxed so the enum stays pointer-sized: a loaded
/// transformer and the state a pair needs to swap are both several hundred
/// bytes, and this value is moved around by the pipeline.
pub(crate) enum WanExperts {
    /// One transformer for the whole schedule.
    Single(Box<WanTransformer>),
    /// A pair, with at most one resident.
    Pair(Box<WanExpertState>),
}

/// What a pair needs to keep between steps in order to swap experts.
pub(crate) struct WanExpertState {
    pair: WanExpertPair,
    config: WanTransformerConfig,
    device: Device,
    dtype: DType,
    resident: Option<(WanExpertRole, WanTransformer)>,
    /// Background page-cache warm for the expert that has not run yet. Dropped
    /// (and joined) when the swap happens or the render ends.
    prefetch: Option<ExpertPrefetch>,
    /// Denoise activation budget for the render in flight, so each expert
    /// can park blocks as it loads (#776 item 3).
    ///
    /// A14B loads its experts lazily on the role switch rather than up
    /// front, so the budget has to travel with the state — resolving it at
    /// construction and forgetting it is how the offload policy came to be
    /// wired for single-expert checkpoints only, and silently skipped for
    /// the pair that actually needs it.
    activation_bytes: Option<u64>,
}

impl WanExperts {
    pub fn single(transformer: WanTransformer) -> Self {
        Self::Single(Box::new(transformer))
    }

    /// Bind a pair. Neither expert is read until the first
    /// [`Self::transformer_for`].
    ///
    /// Both experts must declare the same architecture. They always do in a
    /// shipped pair, but a hand-pointed mismatch would otherwise surface as a
    /// tensor-shape error deep inside the *second* load — after the boundary,
    /// halfway through a generation the user has already paid for.
    pub fn pair(
        pair: WanExpertPair,
        config: WanTransformerConfig,
        device: &Device,
        dtype: DType,
        low_noise_config: WanTransformerConfig,
        activation_bytes: Option<u64>,
    ) -> Result<Self> {
        if low_noise_config != config {
            bail!(
                "the two Wan A14B experts declare different architectures — {} is {}x{} over {} \
                 blocks and {} is {}x{} over {} blocks. They must be halves of one pair.",
                pair.high_noise.path.display(),
                config.dim,
                config.in_dim,
                config.num_layers,
                pair.low_noise.path.display(),
                low_noise_config.dim,
                low_noise_config.in_dim,
                low_noise_config.num_layers,
            );
        }
        Ok(Self::Pair(Box::new(WanExpertState {
            pair,
            config,
            device: device.clone(),
            dtype,
            resident: None,
            prefetch: None,
            activation_bytes,
        })))
    }

    /// The pair this source switches between, if it is a pair.
    pub fn expert_pair(&self) -> Option<&WanExpertPair> {
        match self {
            Self::Single(_) => None,
            Self::Pair(state) => Some(&state.pair),
        }
    }

    /// The transformer for `timestep`, loading and swapping experts if the
    /// schedule has crossed the boundary since the last call.
    ///
    /// Which expert this timestep belongs to, or `None` for a single-expert
    /// checkpoint where the question does not arise.
    ///
    /// Read-only: consulted by the step cache (#801), which must invalidate on
    /// the swap and so needs the role without forcing a load.
    pub fn current_role(&self, timestep: i64) -> Option<WanExpertRole> {
        match self {
            Self::Single(_) => None,
            Self::Pair(state) => Some(state.pair.role_for(timestep)),
        }
    }

    /// The transformer for `timestep`, loading and swapping experts if the
    /// schedule has crossed the boundary since the last call.
    ///
    /// The outgoing expert is dropped and the device synchronized *before* the
    /// incoming one is read, so peak VRAM stays at one expert. It is a reload
    /// from disk rather than a park to host RAM: the file was just read, so the
    /// page cache usually still holds it, and a 10.8 GB round trip over PCIe
    /// costs more than re-reading it does.
    ///
    /// After the first expert lands, a background thread warms the page cache
    /// for its partner so the mid-loop swap is not a cold read (#802 item 3).
    pub fn transformer_for(
        &mut self,
        timestep: i64,
        progress: &ProgressReporter,
    ) -> Result<&WanTransformer> {
        match self {
            Self::Single(transformer) => Ok(&**transformer),
            Self::Pair(state) => {
                let WanExpertState {
                    pair,
                    config,
                    device,
                    dtype,
                    resident,
                    prefetch,
                    activation_bytes,
                } = state.as_mut();
                let role = pair.role_for(timestep);
                if resident.as_ref().map(|(current, _)| *current) != Some(role) {
                    let swapping = resident.is_some();
                    if swapping {
                        // The warm is either finished or pointless now; join it
                        // before the load so it cannot compete for I/O with the
                        // read it exists to accelerate.
                        *prefetch = None;
                        // Drop first: the incoming expert must not have to fit
                        // beside the outgoing one.
                        *resident = None;
                        device.synchronize()?;
                    }
                    let stage = format!("Loading {} expert", role.label());
                    progress.stage_start(&stage);
                    let started = std::time::Instant::now();
                    let slot = pair.slot(role);
                    let transformer =
                        // A14B experts are always one file each; a sharded
                        // pair is not something the ecosystem publishes.
                        load_transformer_with_offload(
                            std::slice::from_ref(&slot.path),
                            config.clone(),
                            device,
                            *dtype,
                            &slot.loras,
                            *activation_bytes,
                        )?;
                    progress.phase_done(ProgressPhase::ModelLoad, &stage, started.elapsed());
                    if swapping {
                        progress.info(&format!(
                            "Crossed the A14B boundary at timestep {}: now on the {} expert",
                            pair.boundary_timestep,
                            role.label()
                        ));
                    }
                    *resident = Some((role, transformer));
                    if !swapping {
                        // Host I/O only -- this never allocates on the device,
                        // so both experts are still never GPU-resident.
                        let partner = pair.slot(role.partner()).path.clone();
                        *prefetch = ExpertPrefetch::start(partner);
                    }
                }
                Ok(&resident.as_ref().expect("just loaded").1)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(name: &str) -> WanExpertSlot {
        WanExpertSlot {
            path: PathBuf::from(name),
            loras: WanLoraRegistry::default(),
        }
    }

    fn pair(channel_concat: bool) -> WanExpertPair {
        WanExpertPair {
            high_noise: slot("high.gguf"),
            low_noise: slot("low.gguf"),
            boundary_timestep: WanExpertPair::boundary_for(channel_concat),
        }
    }

    /// The two boundaries are upstream's, scaled off the 1000-step training
    /// schedule. Getting one wrong runs the wrong expert over part of the
    /// schedule and still produces a video.
    #[test]
    fn the_boundary_is_upstreams_fraction_of_the_training_schedule() {
        assert_eq!(WanExpertPair::boundary_for(false), 875, "t2v_A14B.boundary");
        assert_eq!(WanExpertPair::boundary_for(true), 900, "i2v_A14B.boundary");
    }

    /// High noise first, low noise after — and the boundary timestep itself
    /// belongs to the high-noise expert, matching upstream's `>=`.
    #[test]
    fn the_high_noise_expert_runs_at_and_above_the_boundary() {
        let t2v = pair(false);
        assert_eq!(t2v.role_for(1000), WanExpertRole::HighNoise);
        assert_eq!(t2v.role_for(876), WanExpertRole::HighNoise);
        assert_eq!(t2v.role_for(875), WanExpertRole::HighNoise);
        assert_eq!(t2v.role_for(874), WanExpertRole::LowNoise);
        assert_eq!(t2v.role_for(0), WanExpertRole::LowNoise);

        // I2V switches later, so a timestep between the two boundaries lands on
        // a different expert depending on the task.
        let i2v = pair(true);
        assert_eq!(i2v.role_for(880), WanExpertRole::LowNoise);
        assert_eq!(t2v.role_for(880), WanExpertRole::HighNoise);
    }

    /// The per-expert scales are upstream's `sample_guide_scale` pairs. A
    /// wrong value here quietly mis-guides half the schedule and still
    /// renders.
    #[test]
    fn upstream_per_expert_guidance_matches_the_reference_configs() {
        let t2v = WanExpertGuidance::upstream_for(false);
        assert_eq!(
            (t2v.low_noise, t2v.high_noise),
            (3.0, 4.0),
            "wan_t2v_A14B.py sample_guide_scale"
        );
        let i2v = WanExpertGuidance::upstream_for(true);
        assert_eq!(
            (i2v.low_noise, i2v.high_noise),
            (3.5, 3.5),
            "wan_i2v_A14B.py sample_guide_scale"
        );
        assert_eq!(t2v.max(), 4.0);

        // Selection follows the same role the expert swap resolves.
        assert_eq!(t2v.for_role(WanExpertRole::HighNoise), 4.0);
        assert_eq!(t2v.for_role(WanExpertRole::LowNoise), 3.0);
    }

    #[test]
    fn gguf_is_recognized_by_extension_case_insensitively() {
        assert!(is_gguf(Path::new("Wan2.2-T2V-A14B-HighNoise-Q5_K_M.gguf")));
        assert!(is_gguf(Path::new("expert.GGUF")));
        assert!(!is_gguf(Path::new("wan2.1_t2v_1.3B_bf16.safetensors")));
        assert!(!is_gguf(Path::new("no-extension")));
    }

    /// A mismatched pair must fail at construction, naming both files — not
    /// halfway through the generation when the second expert loads.
    #[test]
    fn a_mismatched_pair_is_refused_before_anything_loads() {
        let device = Device::Cpu;
        let high = WanTransformerConfig::t2v_14b();
        let low = WanTransformerConfig::i2v_14b();
        let error = WanExperts::pair(pair(false), high.clone(), &device, DType::F32, low, None)
            .err()
            .expect("a 16-channel expert does not pair with a 36-channel one")
            .to_string();
        assert!(error.contains("high.gguf"), "{error}");
        assert!(error.contains("low.gguf"), "{error}");
        assert!(error.contains("different architectures"), "{error}");

        assert!(
            WanExperts::pair(pair(false), high.clone(), &device, DType::F32, high, None).is_ok()
        );
    }

    fn tiny_config() -> WanTransformerConfig {
        WanTransformerConfig {
            ffn_dim: 32,
            text_dim: 32,
            freq_dim: 16,
            ..WanTransformerConfig::tiny(16, 2, 2)
        }
    }

    /// Write a tiny transformer to GGUF with a deterministic weight offset, so
    /// the two experts of the fixture are distinguishable networks.
    fn write_expert(path: &Path, config: &WanTransformerConfig, offset: f32) -> WanTransformer {
        use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
        use candle_nn::{VarBuilder, VarMap};

        let device = Device::Cpu;
        let varmap = VarMap::new();
        WanTransformer::from_var_builder(
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
            config.clone(),
        )
        .unwrap();
        {
            let data = varmap.data().lock().unwrap();
            for var in data.values() {
                let shifted = var.as_tensor().affine(1.0, f64::from(offset)).unwrap();
                var.set(&shifted).unwrap();
            }
        }

        let quantized: Vec<(String, QTensor)> = {
            let data = varmap.data().lock().unwrap();
            data.iter()
                .map(|(name, var)| {
                    (
                        name.clone(),
                        QTensor::quantize(var.as_tensor(), GgmlDType::F32).unwrap(),
                    )
                })
                .collect()
        };
        let refs: Vec<(&str, &QTensor)> = quantized
            .iter()
            .map(|(name, q)| (name.as_str(), q))
            .collect();
        let arch = gguf_file::Value::String("wan".to_string());
        let mut file = std::fs::File::create(path).unwrap();
        gguf_file::write(&mut file, &[("general.architecture", &arch)], &refs).unwrap();
        drop(file);

        WanTransformer::from_gguf(path, config.clone(), &device).unwrap()
    }

    /// The whole point of the layer: each expert must actually run, on its own
    /// side of the boundary.
    ///
    /// The two fixture experts are the same architecture with different
    /// weights, so a loader that quietly used one for the whole schedule — or
    /// switched at the wrong timestep — produces a different forward here.
    #[test]
    fn each_expert_runs_on_its_own_side_of_the_boundary() {
        let device = Device::Cpu;
        let dir = tempfile::tempdir().unwrap();
        let config = tiny_config();

        let high_path = dir.path().join("high.gguf");
        let low_path = dir.path().join("low.gguf");
        let high_alone = write_expert(&high_path, &config, 0.0);
        let low_alone = write_expert(&low_path, &config, 0.25);

        let boundary = WanExpertPair::boundary_for(false);
        let mut experts = WanExperts::pair(
            WanExpertPair {
                high_noise: WanExpertSlot {
                    path: high_path,
                    loras: WanLoraRegistry::default(),
                },
                low_noise: WanExpertSlot {
                    path: low_path,
                    loras: WanLoraRegistry::default(),
                },
                boundary_timestep: boundary,
            },
            config.clone(),
            &device,
            DType::F32,
            config.clone(),
            None,
        )
        .unwrap();

        let x =
            candle_core::Tensor::randn(0f32, 1f32, (1, config.in_dim, 1, 4, 4), &device).unwrap();
        let timestep = candle_core::Tensor::from_vec(vec![500f32], 1, &device).unwrap();
        let context =
            candle_core::Tensor::randn(0f32, 1f32, (1, 4, config.text_dim), &device).unwrap();
        let values =
            |t: &candle_core::Tensor| -> Vec<f32> { t.flatten_all().unwrap().to_vec1().unwrap() };
        let run = |model: &WanTransformer| values(&model.forward(&x, &timestep, &context).unwrap());

        let want_high = run(&high_alone);
        let want_low = run(&low_alone);
        assert_ne!(
            want_high, want_low,
            "the fixture's two experts must be distinguishable"
        );

        // Walk the schedule downward through the boundary, as a real run does.
        let progress = ProgressReporter::default();
        let at = |experts: &mut WanExperts, t: i64| -> Vec<f32> {
            run(experts.transformer_for(t, &progress).unwrap())
        };
        assert_eq!(at(&mut experts, 1000), want_high);
        assert_eq!(at(&mut experts, boundary), want_high, "the boundary itself");
        assert_eq!(at(&mut experts, boundary - 1), want_low, "one step past it");
        assert_eq!(at(&mut experts, 0), want_low);
    }

    /// The prefetch must never make both experts resident.
    ///
    /// The family's whole VRAM story is "max of the pair, not the sum". A warm
    /// that allocated on the device — or that constructed a transformer to get
    /// the bytes read — would quietly double the envelope and turn every
    /// admission estimate into an under-count. It reads the file and throws
    /// the bytes away; `resident` must stay a single entry throughout.
    #[test]
    fn the_partner_prefetch_never_makes_both_experts_resident() {
        let device = Device::Cpu;
        let dir = tempfile::tempdir().unwrap();
        let config = tiny_config();
        let high_path = dir.path().join("high.gguf");
        let low_path = dir.path().join("low.gguf");
        write_expert(&high_path, &config, 0.0);
        write_expert(&low_path, &config, 0.25);

        let mut experts = WanExperts::pair(
            WanExpertPair {
                high_noise: WanExpertSlot {
                    path: high_path,
                    loras: WanLoraRegistry::default(),
                },
                low_noise: WanExpertSlot {
                    path: low_path,
                    loras: WanLoraRegistry::default(),
                },
                boundary_timestep: WanExpertPair::boundary_for(false),
            },
            config.clone(),
            &device,
            DType::F32,
            config,
            None,
        )
        .unwrap();

        let progress = ProgressReporter::default();
        let resident_count = |experts: &WanExperts| match experts {
            WanExperts::Single(_) => 1,
            WanExperts::Pair(state) => usize::from(state.resident.is_some()),
        };

        // High-noise resident; the warm for its partner is running or done.
        experts.transformer_for(1000, &progress).unwrap();
        assert_eq!(resident_count(&experts), 1);
        match &experts {
            WanExperts::Pair(state) => {
                assert!(
                    state.prefetch.is_some(),
                    "the partner warm must start once the first expert lands",
                );
                assert_eq!(
                    state.resident.as_ref().map(|(role, _)| *role),
                    Some(WanExpertRole::HighNoise),
                );
            }
            WanExperts::Single(_) => panic!("constructed as a pair"),
        }

        // Cross the boundary: still exactly one resident, and the warm is
        // joined rather than left competing with the load it exists to help.
        experts.transformer_for(0, &progress).unwrap();
        assert_eq!(resident_count(&experts), 1);
        match &experts {
            WanExperts::Pair(state) => {
                assert_eq!(
                    state.resident.as_ref().map(|(role, _)| *role),
                    Some(WanExpertRole::LowNoise),
                );
                assert!(
                    state.prefetch.is_none(),
                    "the warm must be joined at the swap, not left running",
                );
            }
            WanExperts::Single(_) => panic!("constructed as a pair"),
        }
    }

    /// A role's partner is the other one, and never itself — the prefetch
    /// target depends on this, and warming the expert that is already resident
    /// would be pure waste.
    #[test]
    fn a_roles_partner_is_the_other_half() {
        assert_eq!(WanExpertRole::HighNoise.partner(), WanExpertRole::LowNoise);
        assert_eq!(WanExpertRole::LowNoise.partner(), WanExpertRole::HighNoise);
        for role in [WanExpertRole::HighNoise, WanExpertRole::LowNoise] {
            assert_ne!(role.partner(), role);
            assert_eq!(role.partner().partner(), role);
        }
    }

    /// Reloading the same expert must not re-read it from disk: the swap is
    /// once per generation, not once per step.
    #[test]
    fn staying_on_one_expert_does_not_reload_it() {
        let device = Device::Cpu;
        let dir = tempfile::tempdir().unwrap();
        let config = tiny_config();
        let high_path = dir.path().join("high.gguf");
        let low_path = dir.path().join("low.gguf");
        write_expert(&high_path, &config, 0.0);
        write_expert(&low_path, &config, 0.25);

        let mut experts = WanExperts::pair(
            WanExpertPair {
                high_noise: WanExpertSlot {
                    path: high_path,
                    loras: WanLoraRegistry::default(),
                },
                low_noise: WanExpertSlot {
                    path: low_path.clone(),
                    loras: WanLoraRegistry::default(),
                },
                boundary_timestep: WanExpertPair::boundary_for(false),
            },
            config.clone(),
            &device,
            DType::F32,
            config,
            None,
        )
        .unwrap();

        let progress = ProgressReporter::default();
        experts.transformer_for(1000, &progress).unwrap();
        experts.transformer_for(950, &progress).unwrap();
        // Deleting the high-noise file proves the next same-expert call served
        // the resident copy rather than reading the file again.
        let high_noise_path = experts
            .expert_pair()
            .expect("constructed as a pair")
            .high_noise
            .path
            .clone();
        std::fs::remove_file(&high_noise_path).unwrap();
        experts
            .transformer_for(900, &progress)
            .expect("the resident high-noise expert must be reused");

        // Crossing the boundary does read from disk, so the low-noise file has
        // to still be there — and it is.
        experts
            .transformer_for(0, &progress)
            .expect("the low-noise expert loads at the boundary");
    }

    /// A single-expert source never reports a pair, so nothing downstream can
    /// mistake an ordinary checkpoint for an A14B one.
    #[test]
    fn a_single_expert_source_reports_no_pair() {
        let device = Device::Cpu;
        let config = WanTransformerConfig::tiny(16, 2, 1);
        let varmap = candle_nn::VarMap::new();
        let transformer = WanTransformer::from_var_builder(
            candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device),
            config.clone(),
        )
        .unwrap();
        let mut experts = WanExperts::single(transformer);
        assert!(experts.expert_pair().is_none());

        // Every timestep resolves to the same transformer.
        let progress = ProgressReporter::default();
        let first = experts
            .transformer_for(1000, &progress)
            .unwrap()
            .config()
            .clone();
        let last = experts
            .transformer_for(0, &progress)
            .unwrap()
            .config()
            .clone();
        assert_eq!(first, config);
        assert_eq!(last, config);
    }
}
