//! CUDA qualification harnesses for Qwen Image 2.1.
//!
//! Every test here is `#[ignore]`d, reads installed weights only (never
//! downloads), writes only beneath `QWEN_IMAGE21_BENCH_OUTPUT`, and expects an
//! idle, exclusively assigned GPU (`CUDA_VISIBLE_DEVICES=<n>`, so ordinal 0
//! inside the process). The environment contract mirrors the Metal harness:
//!
//! | Variable | Meaning | Default |
//! |---|---|---|
//! | `QWEN_IMAGE21_MODEL_ROOT` | the models directory (`$MOLD_HOME/models`) | required |
//! | `QWEN_IMAGE21_BENCH_OUTPUT` | receipt/artifact directory | required |
//! | `QWEN_IMAGE21_BENCH_MODE` | `legacy`, `flash`, `ops`, `fast`, `fast-cfgbatch` | `legacy` |
//! | `QWEN_IMAGE21_BENCH_TIER` | transformer tier | `bf16` |
//! | `QWEN_IMAGE21_BENCH_WIDTH` / `_HEIGHT` | canvas, multiples of 32 | 1024 |
//! | `QWEN_IMAGE21_BENCH_GUIDANCE` | true-CFG scale | 1.0 |
//! | `QWEN_IMAGE21_BENCH_NEGATIVE` | negative prompt (used when guidance > 1) | none |
//! | `QWEN_IMAGE21_BENCH_STEPS` / `_LIMIT` | schedule length / steps executed | 40 / steps |
//! | `QWEN_IMAGE21_BENCH_PROMPT` / `_SEED` | prompt / seed | teapot / 210001 |
//! | `QWEN_IMAGE21_BENCH_CONV` | VAE conv backend: `auto`, `cudnn`, `im2col` | legacy: im2col, else auto |
//! | `QWEN_IMAGE21_BENCH_REFERENCE` | final-latent safetensors to compare against | none |
//! | `QWEN_IMAGE21_BENCH_LABEL` | receipt/artifact file suffix | mode-tier-canvas-guidance |
//!
//! `official_cuda_vae_decode_benchmark` additionally reads
//! `QWEN_IMAGE21_BENCH_SIZES` (`WxH,WxH,..`) and `QWEN_IMAGE21_BENCH_CONVS`
//! (`cudnn,im2col`).

use super::*;
use crate::conv_policy::{ConvBackend, ConvScope};
use crate::progress::ProgressReporter;
use crate::qwen_image21::exec_path::{Qwen21ExecPath, TargetAttention};
use crate::qwen_image21::{
    encode_t2i_prompts, QwenImage21TextConditioning, QWEN_IMAGE_21_CANVAS_ALIGNMENT,
    QWEN_IMAGE_21_LATENT_CHANNELS, QWEN_IMAGE_21_VAE_SCALE_FACTOR,
};
use candle_core::cuda_backend::cudarc::driver::CudaContext;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

pub(super) const DEFAULT_PROMPT: &str = "A small red ceramic teapot on a sunlit wooden windowsill, editorial product photograph, soft morning shadows";

pub(super) fn env_or<T: std::str::FromStr>(name: &str, default: T) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    match std::env::var(name) {
        Ok(value) => value
            .trim()
            .parse()
            .map_err(|error| anyhow::anyhow!("{name}={value}: {error}")),
        Err(_) => Ok(default),
    }
}

/// Samples the device's free memory on a background thread so a receipt
/// records the high-water mark INSIDE a forward, not the post-forward residue:
/// candle frees each intermediate as soon as it is dropped, so a sample taken
/// after `synchronize` sees only what survived. It reads the stream-ordered
/// pool's reservation as other processes see it, which is the quantity
/// admission has to charge.
pub(super) struct PeakSampler {
    stop: Arc<AtomicBool>,
    min_free: Arc<AtomicU64>,
    total: u64,
    context: Arc<CudaContext>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl PeakSampler {
    pub(super) fn start(ordinal: usize) -> Result<Self> {
        let context = CudaContext::new(ordinal)?;
        let (_, total) = context.mem_get_info()?;
        let stop = Arc::new(AtomicBool::new(false));
        let min_free = Arc::new(AtomicU64::new(u64::MAX));
        let handle = {
            let (stop, min_free, context) = (stop.clone(), min_free.clone(), context.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    if let Ok((free, _)) = context.mem_get_info() {
                        min_free.fetch_min(free as u64, Ordering::Relaxed);
                    }
                    std::thread::sleep(std::time::Duration::from_micros(250));
                }
            })
        };
        Ok(Self {
            stop,
            min_free,
            total: total as u64,
            context,
            handle: Some(handle),
        })
    }

    /// Bytes in use right now (one synchronous sample).
    pub(super) fn used_now(&self) -> Result<u64> {
        let (free, _) = self.context.mem_get_info()?;
        Ok(self.total.saturating_sub(free as u64))
    }

    /// Start a new window; returns the in-use bytes at its start.
    pub(super) fn reset(&self) -> Result<u64> {
        let used = self.used_now()?;
        self.min_free
            .store(self.total.saturating_sub(used), Ordering::Relaxed);
        Ok(used)
    }

    /// Peak in-use bytes since the last [`Self::reset`].
    pub(super) fn peak_used(&self) -> u64 {
        let min_free = self.min_free.load(Ordering::Relaxed);
        self.total.saturating_sub(min_free.min(self.total))
    }

    pub(super) fn total(&self) -> u64 {
        self.total
    }
}

impl Drop for PeakSampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// A named harness mode: the execution path it asks the transformer to run,
/// and whether guided steps batch both CFG branches into one forward.
#[derive(Debug, Clone, Copy)]
pub(super) struct BenchMode {
    pub name: &'static str,
    pub path: Qwen21ExecPath,
    pub cfg_batch: bool,
}

pub(super) fn bench_mode(name: &str) -> Result<BenchMode> {
    let legacy = Qwen21ExecPath::legacy();
    let (name, path, cfg_batch) = match name {
        "legacy" => ("legacy", legacy, false),
        // FlashAttention alone; every elementwise op stays legacy.
        "flash" => (
            "flash",
            Qwen21ExecPath {
                attention: TargetAttention::FastStill,
                ..legacy
            },
            false,
        ),
        // The fused elementwise ops alone; attention stays legacy math.
        "ops" => (
            "ops",
            Qwen21ExecPath {
                attention: TargetAttention::Legacy,
                ..Qwen21ExecPath::cuda_fast()
            },
            false,
        ),
        "fast" => ("fast", Qwen21ExecPath::cuda_fast(), false),
        "fast-cfgbatch" => ("fast-cfgbatch", Qwen21ExecPath::cuda_fast(), true),
        other => anyhow::bail!(
            "QWEN_IMAGE21_BENCH_MODE={other}: expected legacy, flash, ops, fast, or fast-cfgbatch"
        ),
    };
    Ok(BenchMode {
        name,
        path,
        cfg_batch,
    })
}

/// Put `mode` into effect on a loaded transformer. The attention dispatch is
/// wired through the joint-layout seam; the elementwise knobs are not yet, so
/// a mode that needs them is refused by name rather than silently measured
/// as something else.
pub(super) fn install_mode(
    transformer: &mut QwenImage21Transformer,
    mode: &BenchMode,
) -> Result<()> {
    anyhow::ensure!(
        !mode.cfg_batch,
        "mode {} needs batched CFG, which this harness drives only through the engine",
        mode.name
    );
    transformer.set_exec_path(mode.path);
    Ok(())
}

pub(super) fn transformer_paths(root: &Path, tier: &str) -> Result<Vec<PathBuf>> {
    match tier {
        "bf16" => Ok((1..=2)
            .map(|i| {
                root.join(format!(
                    "qwen-image-2.1-bf16/transformer/diffusion_pytorch_model-{i:05}-of-00002.safetensors"
                ))
            })
            .collect()),
        other => anyhow::bail!(
            "QWEN_IMAGE21_BENCH_TIER={other}: this build loads only the bf16 transformer"
        ),
    }
}

pub(super) fn parse_conv(raw: &str) -> Result<ConvBackend> {
    match raw.trim() {
        "cudnn" => Ok(ConvBackend::Cudnn),
        "im2col" => Ok(ConvBackend::Im2Col),
        "auto" => Ok(crate::conv_policy::resolve_for(
            crate::conv_policy::policy_for_family("qwen-image21"),
        )),
        other => anyhow::bail!("conv backend {other}: expected auto, cudnn or im2col"),
    }
}

fn conv_backend_for(mode: &BenchMode) -> Result<ConvBackend> {
    match std::env::var("QWEN_IMAGE21_BENCH_CONV") {
        Ok(raw) if !raw.trim().is_empty() => parse_conv(&raw),
        // The legacy mode reproduces v0.32, whose VAE ran on im2col.
        _ if mode.path.is_legacy() => Ok(ConvBackend::Im2Col),
        _ => parse_conv("auto"),
    }
}

/// Sum of squares in F32. Finite exactly when every element is: unlike a
/// max/min reduction (CUDA `fmaxf` drops NaN), a sum propagates both NaN and
/// infinity.
pub(super) fn square_sum(tensor: &Tensor) -> Result<f64> {
    Ok(f64::from(
        tensor
            .to_dtype(DType::F32)?
            .sqr()?
            .sum_all()?
            .to_scalar::<f32>()?,
    ))
}

pub(super) fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mid = sorted.len() / 2;
    Some(if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    })
}

fn relative_rms_against(latent: &Tensor, reference: &Path) -> Result<f64> {
    let tensors = candle_core::safetensors::load(reference, &Device::Cpu)?;
    let expected = tensors
        .get("latent")
        .ok_or_else(|| anyhow::anyhow!("{} has no `latent` tensor", reference.display()))?
        .to_dtype(DType::F32)?;
    let actual = latent.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    anyhow::ensure!(
        actual.dims() == expected.dims(),
        "reference latent shape {:?} does not match {:?}",
        expected.dims(),
        actual.dims()
    );
    let diff = square_sum(&(&actual - &expected)?)?;
    let base = square_sum(&expected)?;
    Ok((diff / base.max(f64::MIN_POSITIVE)).sqrt())
}

pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    format!("{:x}", sha2::Sha256::digest(bytes))
}

/// Decode one packed latent under `conv`, returning `(image [3,H,W] u8, seconds,
/// peak increment over the pre-decode residency, cuDNN dispatches)`.
pub(super) fn timed_decode(
    vae: &crate::qwen_image21::vae::QwenImage21Vae,
    latents: &Tensor,
    latent_height: usize,
    latent_width: usize,
    conv: ConvBackend,
    sampler: &PeakSampler,
) -> Result<(Tensor, f64, u64, u64)> {
    let device = latents.device().clone();
    device.synchronize()?;
    let before = sampler.reset()?;
    let dispatch_before = crate::conv_policy::cudnn_dispatch_count();
    let started = Instant::now();
    let decoded = {
        let _scope = ConvScope::apply(conv);
        let decoded = vae.decode_packed(latents, latent_height, latent_width)?;
        device.synchronize()?;
        decoded
    };
    let seconds = started.elapsed().as_secs_f64();
    let peak = sampler.peak_used().saturating_sub(before);
    let dispatches = crate::conv_policy::cudnn_dispatch_count() - dispatch_before;
    let image = postprocess_image(&decoded)?.narrow(1, 0, 3)?.i(0)?;
    Ok((image, seconds, peak, dispatches))
}

/// Interleaved 8-bit RGB bytes of a `[3, H, W]` u8 image — the same bytes
/// `magick <png> -depth 8 rgb:-` produces, so a harness image and a CLI PNG
/// compare by one hash whatever metadata their PNG chunks carry.
pub(super) fn rgb_bytes(image: &Tensor) -> Result<Vec<u8>> {
    Ok(image.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?)
}

/// End-to-end CUDA mode benchmark: encode, denoise (timed per forward),
/// decode, receipt. `legacy` reproduces the engine's v0.32 arithmetic on this
/// device, including its noise, its dtypes and its im2col VAE, so its RGB hash
/// is comparable with a CLI render of the same request.
#[test]
#[ignore = "requires installed Qwen Image 2.1 weights and an idle, exclusive CUDA GPU"]
fn official_cuda_mode_benchmark() -> Result<()> {
    let root = PathBuf::from(std::env::var("QWEN_IMAGE21_MODEL_ROOT")?);
    let output = PathBuf::from(std::env::var("QWEN_IMAGE21_BENCH_OUTPUT")?);
    std::fs::create_dir_all(&output)?;
    let mode =
        bench_mode(&std::env::var("QWEN_IMAGE21_BENCH_MODE").unwrap_or_else(|_| "legacy".into()))?;
    let tier = std::env::var("QWEN_IMAGE21_BENCH_TIER").unwrap_or_else(|_| "bf16".into());
    let width: usize = env_or("QWEN_IMAGE21_BENCH_WIDTH", 1024)?;
    let height: usize = env_or("QWEN_IMAGE21_BENCH_HEIGHT", 1024)?;
    anyhow::ensure!(
        width > 0
            && height > 0
            && width.is_multiple_of(QWEN_IMAGE_21_CANVAS_ALIGNMENT)
            && height.is_multiple_of(QWEN_IMAGE_21_CANVAS_ALIGNMENT),
        "canvas {width}x{height} must be positive multiples of {QWEN_IMAGE_21_CANVAS_ALIGNMENT}"
    );
    let guidance: f64 = env_or("QWEN_IMAGE21_BENCH_GUIDANCE", 1.0)?;
    let negative = std::env::var("QWEN_IMAGE21_BENCH_NEGATIVE").ok();
    let steps: usize = env_or("QWEN_IMAGE21_BENCH_STEPS", 40)?;
    let limit: usize = env_or("QWEN_IMAGE21_BENCH_LIMIT", steps)?;
    anyhow::ensure!(
        steps > 0 && limit > 0,
        "benchmark steps and limit must be positive"
    );
    let prompt =
        std::env::var("QWEN_IMAGE21_BENCH_PROMPT").unwrap_or_else(|_| DEFAULT_PROMPT.into());
    let seed: u64 = env_or("QWEN_IMAGE21_BENCH_SEED", 210001)?;
    let conv = conv_backend_for(&mode)?;
    let label = std::env::var("QWEN_IMAGE21_BENCH_LABEL")
        .unwrap_or_else(|_| format!("{}-{tier}-{width}x{height}-g{guidance}", mode.name));
    // The engine's own rule: true CFG runs only with a negative prompt.
    let guided = crate::engine::cfg_active(guidance) && negative.is_some();

    let device = Device::new_cuda(0)?;
    let sampler = PeakSampler::start(0)?;
    let idle_used = sampler.used_now()?;
    let progress = ProgressReporter::default();
    let started = Instant::now();
    let mut phases = Map::new();
    let dtype = crate::engine::gpu_dtype(&device);

    let shared = root.join("shared/qwen-image21");
    let text_paths = (1..=4)
        .map(|i| shared.join(format!("text_encoder/model-{i:05}-of-00004.safetensors")))
        .collect::<Vec<_>>();
    let phase = Instant::now();
    let mut encoder = crate::encoders::qwen3::Qwen3Encoder::load_bf16(
        &text_paths,
        &shared.join("processor/tokenizer.json"),
        &device,
        dtype,
        &crate::encoders::qwen3_bf16::Qwen3BF16Config::qwen3_image_21_text_encoder(),
        &progress,
    )?;
    device.synchronize()?;
    phases.insert(
        "encoder_load_seconds".into(),
        json!(phase.elapsed().as_secs_f64()),
    );
    let phase = Instant::now();
    let conditioning = encode_t2i_prompts(&mut encoder, std::slice::from_ref(&prompt))?
        .to_device_dtype(&device, dtype)?;
    let negative_conditioning = match (&negative, guided) {
        (Some(negative), true) => Some(
            encode_t2i_prompts(&mut encoder, std::slice::from_ref(negative))?
                .to_device_dtype(&device, dtype)?,
        ),
        _ => None,
    };
    device.synchronize()?;
    phases.insert(
        "prompt_encode_seconds".into(),
        json!(phase.elapsed().as_secs_f64()),
    );
    let prefix_tokens = conditioning.sequence_length();
    let negative_prefix_tokens = negative_conditioning
        .as_ref()
        .map(QwenImage21TextConditioning::sequence_length);
    // The harness isolates the denoiser: the encoder is not resident during
    // denoise (the sequential engine's phase order).
    drop(encoder);
    device.synchronize()?;

    let phase = Instant::now();
    let mut transformer = QwenImage21Transformer::load(
        &transformer_paths(&root, &tier)?,
        &device,
        crate::qwen_image21::transformer_dtype(&device),
        &progress,
    )?;
    install_mode(&mut transformer, &mode)?;
    device.synchronize()?;
    phases.insert(
        "transformer_load_seconds".into(),
        json!(phase.elapsed().as_secs_f64()),
    );
    let resident_used = sampler.used_now()?;

    let latent_height = height / QWEN_IMAGE_21_VAE_SCALE_FACTOR;
    let latent_width = width / QWEN_IMAGE_21_VAE_SCALE_FACTOR;
    let latent_tokens = latent_height * latent_width;
    let mut scheduler = crate::qwen_image21::scheduler::QwenImage21Scheduler::new(
        steps,
        latent_tokens,
        crate::qwen_image21::scheduler::QwenShiftPolicy::DynamicResolution,
    );
    // Exactly the engine's noise: CPU StdRng, straight into the working dtype.
    let noise = crate::engine::seeded_randn(
        seed,
        &[1, latent_tokens, QWEN_IMAGE_21_LATENT_CHANNELS],
        &device,
        dtype,
    )?;
    let mut latents = (noise * scheduler.initial_sigma())?;
    let total_steps = scheduler.num_steps();
    let executed_steps = limit.min(total_steps);
    // The engine retains the prefix whenever it fits; a 1024-token prompt
    // at most on a 46 GB card always does.
    let retain = crate::qwen_image21::PrefixCacheDecision::Retain;
    let mut conditional =
        transformer.prepare_t2i(&conditioning, latent_height, latent_width, retain)?;
    let mut negative_branch = negative_conditioning
        .as_ref()
        .map(|c| transformer.prepare_t2i(c, latent_height, latent_width, retain))
        .transpose()?;

    let mut step_receipts = Vec::with_capacity(executed_steps);
    let mut steady = Vec::new();
    sampler.reset()?;
    device.synchronize()?;
    let denoise_started = Instant::now();
    for step in 0..executed_steps {
        let timestep = crate::qwen_image21::scheduler::step_timestep(
            &scheduler,
            dtype,
            mode.path.round_timestep_to_dtype,
        );
        device.synchronize()?;
        let step_started = Instant::now();
        let conditional_prediction = conditional.forward(&latents, timestep)?;
        device.synchronize()?;
        let conditional_seconds = step_started.elapsed().as_secs_f64();
        let (prediction, negative_seconds) = match &mut negative_branch {
            Some(negative) => {
                let negative_started = Instant::now();
                let negative_prediction = negative.forward(&latents, timestep)?;
                device.synchronize()?;
                let seconds = negative_started.elapsed().as_secs_f64();
                (
                    (&negative_prediction
                        + ((&conditional_prediction - &negative_prediction)? * guidance)?)?,
                    Some(seconds),
                )
            }
            None => (conditional_prediction, None),
        };
        let prediction_square_sum = square_sum(&prediction)?;
        anyhow::ensure!(
            prediction_square_sum.is_finite(),
            "mode {} tier {tier}: step {} prediction is not finite",
            mode.name,
            step + 1
        );
        latents = scheduler.step(&prediction, &latents)?;
        device.synchronize()?;
        let seconds = step_started.elapsed().as_secs_f64();
        if step > 0 {
            steady.push(seconds);
        }
        eprintln!(
            "mode={} canvas={width}x{height} step={}/{} seconds={seconds:.4}",
            mode.name,
            step + 1,
            executed_steps
        );
        step_receipts.push(json!({
            "step": step + 1,
            "timestep": timestep,
            "seconds": seconds,
            "conditional_forward_seconds": conditional_seconds,
            "negative_forward_seconds": negative_seconds,
            "prediction_rms": (prediction_square_sum / prediction.elem_count() as f64).sqrt(),
        }));
    }
    device.synchronize()?;
    let denoise_seconds = denoise_started.elapsed().as_secs_f64();
    let denoise_peak_used = sampler.peak_used();
    phases.insert("denoise_seconds".into(), json!(denoise_seconds));
    drop(conditional);
    drop(negative_branch);
    drop(transformer);
    drop(conditioning);
    drop(negative_conditioning);
    device.synchronize()?;

    let complete = executed_steps == total_steps;
    let (final_latent, final_latent_stats) = capture("latent", &latents)?;
    let mut decode = Map::new();
    let mut reference_relative_rms = None;
    if complete {
        save_tensors(
            &output.join(format!("final-latent-{label}.safetensors")),
            &[final_latent],
        )?;
        if let Ok(reference) = std::env::var("QWEN_IMAGE21_BENCH_REFERENCE") {
            reference_relative_rms = Some(relative_rms_against(&latents, Path::new(&reference))?);
        }
        let vae_path = shared.join("vae/diffusion_pytorch_model.safetensors");
        let vae_dtype = crate::engine::gpu_dtype(&device);
        let phase = Instant::now();
        let vae = crate::qwen_image21::vae::QwenImage21Vae::load(
            &vae_path, &device, vae_dtype, &progress,
        )?;
        device.synchronize()?;
        decode.insert(
            "vae_load_seconds".into(),
            json!(phase.elapsed().as_secs_f64()),
        );
        let (image, seconds, peak, dispatches) = timed_decode(
            &vae,
            &latents.to_dtype(vae_dtype)?,
            latent_height,
            latent_width,
            conv,
            &sampler,
        )?;
        let rgb = rgb_bytes(&image)?;
        let png = crate::image::encode_image(
            &image,
            mold_core::OutputFormat::Png,
            width as u32,
            height as u32,
            None,
        )?;
        std::fs::write(output.join(format!("image-{label}.png")), &png)?;
        decode.insert("conv_backend".into(), json!(conv.as_str()));
        decode.insert("cudnn_dispatches".into(), json!(dispatches));
        decode.insert("vae_decode_seconds".into(), json!(seconds));
        decode.insert("vae_decode_peak_increment_bytes".into(), json!(peak));
        decode.insert("rgb_sha256".into(), json!(sha256_hex(&rgb)));
    }
    let receipt = json!({
        "label": label,
        "mode": mode.name,
        "exec_path": {
            "label": mode.path.label(),
            "attention": format!("{:?}", mode.path.attention),
            "fused_projection": mode.path.fused_projection,
            "compact_modulation": mode.path.compact_modulation,
            "fused_adaln": mode.path.fused_adaln,
            "f32_rope_tables": mode.path.f32_rope_tables,
            "round_timestep_to_dtype": mode.path.round_timestep_to_dtype,
            "cfg_batch": mode.cfg_batch,
        },
        "tier": tier,
        "width": width,
        "height": height,
        "latent_tokens": latent_tokens,
        "guidance": guidance,
        "negative_prompt": negative,
        "guided": guided,
        "prompt": prompt,
        "seed": seed,
        "prefix_tokens": prefix_tokens,
        "negative_prefix_tokens": negative_prefix_tokens,
        "schedule_steps": total_steps,
        "executed_steps": executed_steps,
        "complete": complete,
        "finite": true,
        "working_dtype": format!("{dtype:?}"),
        "flash_compiled": crate::attention::AttentionBackend::flash_compiled(),
        "cudnn_compiled": crate::conv_policy::cudnn_compiled(),
        "cuda_visible_devices": std::env::var("CUDA_VISIBLE_DEVICES").ok(),
        "device_total_bytes": sampler.total(),
        "idle_used_bytes": idle_used,
        "resident_used_bytes": resident_used,
        "denoise_peak_used_bytes": denoise_peak_used,
        "denoise_peak_increment_bytes": denoise_peak_used.saturating_sub(resident_used),
        "first_step_seconds": step_receipts.first().map(|s| s["seconds"].clone()),
        "steady_step_mean_seconds": (!steady.is_empty()).then(|| steady.iter().sum::<f64>() / steady.len() as f64),
        "steady_step_median_seconds": median(&steady),
        "reference_relative_rms": reference_relative_rms,
        "final_latent": final_latent_stats,
        "phases": phases,
        "decode": decode,
        "total_seconds": started.elapsed().as_secs_f64(),
        "steps": step_receipts,
    });
    std::fs::write(
        output.join(format!("receipt-{label}.json")),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    eprintln!(
        "receipt {label}: denoise {denoise_seconds:.2}s, steady median {}s/step, peak {:.2} GB",
        receipt["steady_step_median_seconds"],
        denoise_peak_used as f64 / 1e9
    );
    Ok(())
}

/// VAE-decode calibration: time and peak memory of one decode per canvas and
/// convolution backend, on seeded standard-normal latents (the decode's
/// memory and time do not depend on latent values). Each cell decodes twice;
/// the second is the warm number (cuDNN plans its algorithms on first use).
/// An allocation failure is recorded in the receipt rather than aborting the
/// sweep, because "this size does not fit under this backend" is the finding.
#[test]
#[ignore = "requires the installed Qwen Image 2.1 VAE and an idle, exclusive CUDA GPU"]
fn official_cuda_vae_decode_benchmark() -> Result<()> {
    let root = PathBuf::from(std::env::var("QWEN_IMAGE21_MODEL_ROOT")?);
    let output = PathBuf::from(std::env::var("QWEN_IMAGE21_BENCH_OUTPUT")?);
    std::fs::create_dir_all(&output)?;
    let sizes = std::env::var("QWEN_IMAGE21_BENCH_SIZES")
        .unwrap_or_else(|_| "1024x1024,1344x768,2048x2048,2752x1536".into());
    let convs = std::env::var("QWEN_IMAGE21_BENCH_CONVS").unwrap_or_else(|_| "cudnn,im2col".into());
    let label = std::env::var("QWEN_IMAGE21_BENCH_LABEL").unwrap_or_else(|_| "vae-decode".into());

    let device = Device::new_cuda(0)?;
    let sampler = PeakSampler::start(0)?;
    let dtype = crate::engine::gpu_dtype(&device);
    let progress = ProgressReporter::default();
    let vae = crate::qwen_image21::vae::QwenImage21Vae::load(
        &root.join("shared/qwen-image21/vae/diffusion_pytorch_model.safetensors"),
        &device,
        dtype,
        &progress,
    )?;
    device.synchronize()?;
    let vae_resident = sampler.used_now()?;
    let mut cells = Vec::new();
    for size in sizes.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (width, height) = size
            .split_once('x')
            .ok_or_else(|| anyhow::anyhow!("size {size} is not WxH"))?;
        let (width, height): (usize, usize) = (width.parse()?, height.parse()?);
        let (latent_height, latent_width) = (
            height / QWEN_IMAGE_21_VAE_SCALE_FACTOR,
            width / QWEN_IMAGE_21_VAE_SCALE_FACTOR,
        );
        let latents = crate::engine::seeded_randn(
            210001,
            &[
                1,
                latent_height * latent_width,
                QWEN_IMAGE_21_LATENT_CHANNELS,
            ],
            &device,
            dtype,
        )?;
        for conv in convs.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let backend = parse_conv(conv)?;
            let mut runs = Vec::new();
            for _ in 0..2 {
                match timed_decode(
                    &vae,
                    &latents,
                    latent_height,
                    latent_width,
                    backend,
                    &sampler,
                ) {
                    Ok((image, seconds, peak, dispatches)) => {
                        let rgb = rgb_bytes(&image)?;
                        runs.push(json!({
                            "seconds": seconds,
                            "peak_increment_bytes": peak,
                            "cudnn_dispatches": dispatches,
                            "rgb_sha256": sha256_hex(&rgb),
                        }));
                    }
                    Err(error) => {
                        runs.push(json!({ "error": format!("{error:#}") }));
                        break;
                    }
                }
                device.synchronize()?;
            }
            eprintln!("vae {width}x{height} {}: {runs:?}", backend.as_str());
            cells.push(json!({
                "width": width,
                "height": height,
                "pixels": width * height,
                "conv_backend": backend.as_str(),
                "runs": runs,
            }));
        }
    }
    let receipt = json!({
        "label": label,
        "vae_dtype": format!("{dtype:?}"),
        "cudnn_compiled": crate::conv_policy::cudnn_compiled(),
        "cuda_visible_devices": std::env::var("CUDA_VISIBLE_DEVICES").ok(),
        "device_total_bytes": sampler.total(),
        "vae_resident_used_bytes": vae_resident,
        "cells": cells,
    });
    std::fs::write(
        output.join(format!("receipt-{label}.json")),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    Ok(())
}

/// The design's "no effect" check for the fork's BF16 reduced-precision cuBLAS
/// switch: in `gemm_strided_batched_bf16` it only changes the compute type
/// from `CUBLAS_COMPUTE_32F` to `CUBLAS_COMPUTE_32F_FAST_16BF`, which for BF16
/// inputs is the same arithmetic. Times the transformer's real GEMM shapes
/// (M = 4,177 joint tokens at 1024², K/N from the 4096-wide projections and
/// the 12,288-wide MLP) with the switch off and on, and records whether the
/// outputs differ. The switch is process-global, so it is restored before
/// returning and this test must run alone.
#[test]
#[ignore = "requires an idle, exclusive CUDA GPU; flips a process-global cuBLAS switch"]
fn official_cuda_reduced_precision_gemm_benchmark() -> Result<()> {
    use candle_core::cuda_backend::{gemm_reduced_precision_bf16, set_gemm_reduced_precision_bf16};
    let output = PathBuf::from(std::env::var("QWEN_IMAGE21_BENCH_OUTPUT")?);
    std::fs::create_dir_all(&output)?;
    let device = Device::new_cuda(0)?;
    let previous = gemm_reduced_precision_bf16();
    let mut cells = Vec::new();
    let result = (|| -> Result<()> {
        for (m, k, n) in [
            (4177usize, 4096usize, 4096usize),
            (4177, 4096, 12288),
            (4177, 12288, 4096),
        ] {
            let a = Tensor::randn(0f32, 1.0, (1, m, k), &device)?.to_dtype(DType::BF16)?;
            let w = Tensor::randn(0f32, 0.02, (n, k), &device)?.to_dtype(DType::BF16)?;
            let mut outputs = Vec::new();
            let mut timings = Map::new();
            for reduced in [false, true] {
                set_gemm_reduced_precision_bf16(reduced);
                for _ in 0..5 {
                    a.broadcast_matmul(&w.t()?)?;
                }
                device.synchronize()?;
                let iterations = 50;
                let started = Instant::now();
                let mut out = None;
                for _ in 0..iterations {
                    out = Some(a.broadcast_matmul(&w.t()?)?);
                }
                device.synchronize()?;
                let ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
                timings.insert(
                    if reduced { "reduced_ms" } else { "default_ms" }.into(),
                    json!(ms),
                );
                outputs.push(out.expect("iterations > 0").to_dtype(DType::F32)?);
            }
            let max_diff = (&outputs[0] - &outputs[1])?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            eprintln!("gemm {m}x{k}x{n}: {timings:?} max_diff={max_diff}");
            cells.push(
                json!({ "m": m, "k": k, "n": n, "timings": timings, "max_abs_diff": max_diff }),
            );
        }
        Ok(())
    })();
    set_gemm_reduced_precision_bf16(previous);
    result?;
    std::fs::write(
        output.join("receipt-reduced-precision-gemm.json"),
        serde_json::to_vec_pretty(&json!({ "cells": cells }))?,
    )?;
    Ok(())
}
/// A8 measured gate: the UPPER BOUND of batched CFG on this card. Both rows
/// of one batch-2 forward carry the same prompt, so the batch needs no key
/// padding at all — the cheapest a real batched guided step could ever be —
/// and its cached decode time is compared against two batch-1 decodes. A
/// real negative prompt of another length would add varlen packing on top.
#[test]
#[ignore = "requires installed Qwen Image 2.1 weights and an idle, exclusive CUDA GPU"]
fn official_cuda_cfg_batch_probe() -> Result<()> {
    let root = PathBuf::from(std::env::var("QWEN_IMAGE21_MODEL_ROOT")?);
    let output = PathBuf::from(std::env::var("QWEN_IMAGE21_BENCH_OUTPUT")?);
    std::fs::create_dir_all(&output)?;
    let sizes = std::env::var("QWEN_IMAGE21_BENCH_SIZES")
        .unwrap_or_else(|_| "1024x1024,1344x768,2048x2048".into());
    let device = Device::new_cuda(0)?;
    let dtype = crate::engine::gpu_dtype(&device);
    let progress = ProgressReporter::default();
    let shared = root.join("shared/qwen-image21");
    let conditioning = {
        let text_paths = (1..=4)
            .map(|i| shared.join(format!("text_encoder/model-{i:05}-of-00004.safetensors")))
            .collect::<Vec<_>>();
        let mut encoder = crate::encoders::qwen3::Qwen3Encoder::load_bf16(
            &text_paths,
            &shared.join("processor/tokenizer.json"),
            &device,
            dtype,
            &crate::encoders::qwen3_bf16::Qwen3BF16Config::qwen3_image_21_text_encoder(),
            &progress,
        )?;
        encode_t2i_prompts(&mut encoder, &[DEFAULT_PROMPT.to_string()])?
            .to_device_dtype(&device, dtype)?
    };
    let doubled = QwenImage21TextConditioning {
        embeddings: Tensor::cat(&[&conditioning.embeddings, &conditioning.embeddings], 0)?,
        valid_tokens: [
            conditioning.valid_tokens.clone(),
            conditioning.valid_tokens.clone(),
        ]
        .concat(),
        image_slots: [
            conditioning.image_slots.clone(),
            conditioning.image_slots.clone(),
        ]
        .concat(),
    };
    let transformer = QwenImage21Transformer::load(
        &transformer_paths(&root, "bf16")?,
        &device,
        crate::qwen_image21::transformer_dtype(&device),
        &progress,
    )?;
    let retain = crate::qwen_image21::PrefixCacheDecision::Retain;
    let mut cells = Vec::new();
    for size in sizes.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (width, height) = size
            .split_once('x')
            .ok_or_else(|| anyhow::anyhow!("size {size} is not WxH"))?;
        let (lh, lw) = (
            height.parse::<usize>()? / QWEN_IMAGE_21_VAE_SCALE_FACTOR,
            width.parse::<usize>()? / QWEN_IMAGE_21_VAE_SCALE_FACTOR,
        );
        let latents = crate::engine::seeded_randn(
            210001,
            &[1, lh * lw, QWEN_IMAGE_21_LATENT_CHANNELS],
            &device,
            dtype,
        )?;
        let pair = Tensor::cat(&[&latents, &latents], 0)?;
        let time = |forward: &mut dyn FnMut() -> Result<Tensor>| -> Result<f64> {
            forward()?; // prefill and warm-up
            forward()?;
            device.synchronize()?;
            let mut samples = Vec::new();
            for _ in 0..6 {
                let started = Instant::now();
                forward()?;
                device.synchronize()?;
                samples.push(started.elapsed().as_secs_f64());
            }
            Ok(median(&samples).expect("six samples"))
        };
        let mut single = transformer.prepare_t2i(&conditioning, lh, lw, retain)?;
        let mut negative = transformer.prepare_t2i(&conditioning, lh, lw, retain)?;
        let sequential = time(&mut || {
            single.forward(&latents, 0.5)?;
            negative.forward(&latents, 0.5)
        })?;
        drop((single, negative));
        let mut batched_branch = transformer.prepare_t2i(&doubled, lh, lw, retain)?;
        let batched = time(&mut || batched_branch.forward(&pair, 0.5))?;
        drop(batched_branch);
        let gain = 1.0 - batched / sequential;
        eprintln!(
            "cfg batch {size}: two B=1 {sequential:.4}s, one B=2 {batched:.4}s, gain {gain:.3}"
        );
        cells.push(json!({
            "canvas": size,
            "two_batch1_seconds": sequential,
            "one_batch2_seconds": batched,
            "gain": gain,
        }));
    }
    std::fs::write(
        output.join("receipt-cfg-batch-probe.json"),
        serde_json::to_vec_pretty(&json!({
            "exec_path": transformer.exec_path().label(),
            "cells": cells,
        }))?,
    )?;
    Ok(())
}
