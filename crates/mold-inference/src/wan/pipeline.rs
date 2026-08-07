//! Wan 2.1 / 2.2 video engine.
//!
//! Ties the Wan layers together: UMT5-XXL prompt encoding, the DiT, the
//! FlowUniPC sampler, the causal 3-D VAE, and the image-conditioning assembly.
//!
//! Three conditioning modes, chosen from the checkpoint's own shapes rather
//! than its name:
//!
//! - **Text-to-video** — no source image.
//! - **Latent inpaint** (TI2V-5B) — the source image is encoded to one latent
//!   frame, pinned at frame 0, and its tokens carry timestep 0 so the DiT sees
//!   them as already clean.
//! - **Channel concat** (I2V checkpoints, `in_dim = 2*z + 4`) — a 20-channel
//!   mask-plus-image block rides alongside the noise every step.
//!
//! Video-to-video and keyframes are still refused rather than silently ignored.
//!
//! Like the other video families this engine is *sequential by construction*:
//! every component is loaded inside `generate` and dropped as soon as it is
//! done. That is not a simplification, it is a VRAM requirement — UMT5-XXL is
//! 11.4 GB at fp16 and TI2V-5B's transformer is another 10 GB, so the encoder
//! has to be gone before the first denoise step. `load()` is therefore a
//! no-op, matching `LtxVideoEngine`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::{safetensors::MmapedSafetensors, DType, Device, IndexOp, Tensor};
use mold_core::{GenerateRequest, GenerateResponse, ModelPaths, OutputFormat, VideoData};

use crate::engine::{gpu_dtype, rand_seed, seeded_randn, LoadStrategy};
use crate::engine_base::EngineBase;
use crate::ltx_video::video_enc;
use crate::progress::{ProgressCallback, ProgressEvent, ProgressPhase};
use crate::shared_pool::SharedPool;
use crate::wan::conditioning::{
    build_a14b_conditioning, WanImageAnchors, WanLatentGeometry, WanTi2vInpaint,
};
use crate::wan::experts::{WanExpertPair, WanExpertSlot, WanExperts};
use crate::wan::lora::WanLoraRegistry;
#[cfg(test)]
use crate::wan::model::transformer::WanTransformer;
use crate::wan::model::transformer::WanTransformerConfig;
use crate::wan::model::vae::{WanVaeConfig, WanVideoVae};
use crate::wan::sampler::{apply_cfg, FlowUniPc, WanSchedule, WanScheduleConfig};
use crate::wan::text::umt5::WanTextEncoder;

/// ComfyUI ships flow shift 8.0 in both its Wan 2.1 and Wan 2.2 templates.
/// Upstream's own CLI defaults differ per task (5.0 for 1.3B, 5.0 for TI2V),
/// but the ComfyUI recipe is the one the community's prompts are tuned
/// against, and it is what the manifest defaults already mirror.
const DEFAULT_FLOW_SHIFT: f64 = 8.0;

/// Override for [`DEFAULT_FLOW_SHIFT`]. Validated, not silently ignored.
const FLOW_SHIFT_ENV: &str = "MOLD_WAN_SHIFT";

/// Temporal compression is 4x for both VAE generations.
const VAE_TEMPORAL_COMPRESSION: usize = 4;

/// Which VAE generation a checkpoint pairs with. This decides latent channel
/// count, spatial compression, and the frame/fps defaults.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WanVaeGeneration {
    /// 16-channel, 8x8 spatial — Wan 2.1 (1.3B, 14B, A14B).
    V2_1,
    /// 48-channel, 16x16 spatial — Wan 2.2 TI2V-5B only.
    V2_2,
}

impl WanVaeGeneration {
    fn config(self) -> WanVaeConfig {
        match self {
            Self::V2_1 => WanVaeConfig::v2_1(),
            Self::V2_2 => WanVaeConfig::v2_2(),
        }
    }

    /// `(frames, fps)` when the request carries neither. These mirror the
    /// manifest defaults, which the CLI and server normally plumb through.
    fn default_timing(self) -> (u32, u32) {
        match self {
            Self::V2_1 => (81, 16),
            Self::V2_2 => (121, 24),
        }
    }
}

/// Read a checkpoint's tensor names and shapes without materializing weights.
///
/// Both containers answer the same question, so every shape-driven probe in
/// this module — config detection, the CLIP-branch refusal — works against
/// GGUF experts without knowing they are GGUF. candle's reader already presents
/// GGUF dimensions in torch order (`gguf_file.rs` reverses ggml's
/// fastest-varying-first layout), so the two sources agree on shape as well as
/// on name.
fn header_shapes(path: &Path) -> Result<Vec<(String, Vec<usize>)>> {
    if crate::wan::experts::is_gguf(path) {
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("open Wan checkpoint at {}", path.display()))?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .with_context(|| format!("read the GGUF header of {}", path.display()))?;
        return Ok(content
            .tensor_infos
            .into_iter()
            .map(|(name, info)| (name, info.shape.dims().to_vec()))
            .collect());
    }
    let st = unsafe { MmapedSafetensors::new(path) }
        .with_context(|| format!("open Wan checkpoint at {}", path.display()))?;
    Ok(st
        .tensors()
        .into_iter()
        .map(|(name, view)| (name, view.shape().to_vec()))
        .collect())
}

/// Detect the VAE generation from the checkpoint's own key layout.
///
/// Wan 2.2 nests its stages (`decoder.upsamples.{s}.upsamples.{j}`) while 2.1
/// keeps one flat `nn.Sequential`. ComfyUI uses exactly this probe
/// (`comfy/sd.py:770-775`); it beats matching on the file name, which varies
/// between repacks, or on the model name, which a `--vae` override can
/// contradict.
pub(crate) fn detect_vae_generation(path: &Path) -> Result<WanVaeGeneration> {
    let shapes = header_shapes(path)?;
    let nested = shapes
        .iter()
        .any(|(name, _)| name.starts_with("decoder.upsamples.0.upsamples."));
    if nested {
        return Ok(WanVaeGeneration::V2_2);
    }
    if shapes
        .iter()
        .any(|(name, _)| name == "decoder.middle.0.residual.0.gamma")
    {
        return Ok(WanVaeGeneration::V2_1);
    }
    bail!(
        "{} does not look like a Wan VAE (no decoder.middle.0.residual.0.gamma)",
        path.display()
    )
}

/// Dtype for the UMT5 encoder given where placement put it. candle's CPU
/// backend has no BF16/F16 matmul, so a CPU-parked encoder — which is what
/// placement chooses when the transformer needs the VRAM, e.g. TI2V-5B on a
/// 24 GB card — must compute in F32. The embeddings are cast to the model
/// dtype on their way to the execution device, so downstream is unaffected.
/// (The 1.3B path masked this: its small transformer leaves room to keep the
/// encoder on GPU, where BF16 matmul exists.)
fn encoder_dtype_for(text_device: &Device, model_dtype: DType) -> DType {
    if text_device.is_cpu() {
        DType::F32
    } else {
        model_dtype
    }
}

/// Collect the request's LoRA stack.
///
/// `lora` is the legacy single-adapter field and `loras` the multi-adapter
/// one; a request may carry either. Both are passed through verbatim — Wan has
/// no preset names to resolve, unlike LTX-2's camera-control shorthand.
fn normalize_loras(req: &GenerateRequest) -> Vec<mold_core::LoraWeight> {
    let mut out: Vec<mold_core::LoraWeight> = Vec::new();
    if let Some(loras) = &req.loras {
        out.extend(loras.iter().cloned());
    }
    if out.is_empty() {
        if let Some(lora) = &req.lora {
            out.push(lora.clone());
        }
    }
    out
}

/// Resolve the unconditional prompt for CFG. Only *absence* falls back to
/// the tuned default the checkpoints were trained against — an explicit
/// value is authoritative, including the empty string `--no-negative`
/// produces, which must stay an empty uncond rather than being silently
/// replaced.
fn resolve_negative_prompt(requested: Option<&str>) -> &str {
    match requested {
        Some(text) => text.trim(),
        None => mold_core::manifest::WAN_DEFAULT_NEGATIVE_PROMPT,
    }
}

/// Derive the DiT config from the checkpoint's tensor shapes.
///
/// Shape-driven rather than name-driven, the way ComfyUI detects its models: a
/// repack, a community fine-tune, or a `--transformer` override can all carry
/// a name this code has never seen, but the shapes are the architecture.
///
/// - `patch_embedding.weight` `[dim, in_dim, pt, ph, pw]` gives the width, the
///   latent channel count, and the patch size.
/// - `blocks.{i}.ffn.0.weight` `[ffn_dim, dim]` gives the MLP width.
/// - `text_embedding.0.weight` / `time_embedding.0.weight` give the two
///   conditioning widths.
/// - `head.head.weight` `[out_dim * patch, dim]` gives the output channels.
/// - The highest `blocks.{i}.` index gives the depth.
pub(crate) fn detect_transformer_config(path: &Path) -> Result<WanTransformerConfig> {
    // The shipped Comfy-Org repacks store every DiT key under
    // `model.diffusion_model.` (verified against the real
    // wan2.1_t2v_1.3B_bf16.safetensors header) while the VAE and encoder
    // files are bare. The loader already strips the prefix; detection must
    // see the same names or the advertised checkpoint fails before the
    // prefix-aware loader ever runs.
    let shapes: Vec<(String, Vec<usize>)> = header_shapes(path)?
        .into_iter()
        .map(|(name, shape)| {
            let bare = name
                .strip_prefix("model.diffusion_model.")
                .map(str::to_string)
                .unwrap_or(name);
            (bare, shape)
        })
        .collect();
    let find = |key: &str| -> Result<&Vec<usize>> {
        shapes
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, shape)| shape)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "{} is missing `{key}` — not a Wan DiT checkpoint in the original key layout",
                    path.display()
                )
            })
    };

    let patch = find("patch_embedding.weight")?;
    if patch.len() != 5 {
        bail!(
            "Wan DiT: patch_embedding.weight must be 5-D [dim, in_dim, pt, ph, pw], got {patch:?}"
        );
    }
    let dim = patch[0];
    let in_dim = patch[1];
    let patch_size = (patch[2], patch[3], patch[4]);
    let patch_elems = patch_size.0 * patch_size.1 * patch_size.2;

    let ffn_dim = find("blocks.0.ffn.0.weight")?[0];
    let text_dim = find("text_embedding.0.weight")?[1];
    let freq_dim = find("time_embedding.0.weight")?[1];
    let head = find("head.head.weight")?;
    if !head[0].is_multiple_of(patch_elems) {
        bail!(
            "Wan DiT: head.head.weight rows {} are not a multiple of the {patch_elems}-element \
             patch",
            head[0]
        );
    }
    let out_dim = head[0] / patch_elems;

    let num_layers = shapes
        .iter()
        .filter_map(|(name, _)| name.strip_prefix("blocks."))
        .filter_map(|rest| rest.split('.').next())
        .filter_map(|index| index.parse::<usize>().ok())
        .max()
        .map(|highest| highest + 1)
        .ok_or_else(|| anyhow::anyhow!("{} has no transformer blocks", path.display()))?;

    // Every shipped Wan variant uses a 128-wide head; the checkpoint does not
    // record the head count directly, so derive it from that invariant.
    if !dim.is_multiple_of(128) {
        bail!("Wan DiT: model width {dim} is not a multiple of the 128-wide attention head");
    }
    let num_heads = dim / 128;

    let config = WanTransformerConfig {
        dim,
        ffn_dim,
        num_heads,
        num_layers,
        in_dim,
        out_dim,
        text_dim,
        freq_dim,
        patch_size,
        eps: 1e-6,
        rope_max_seq_len: WanTransformerConfig::t2v_1_3b().rope_max_seq_len,
    };
    Ok(config)
}

/// Flow shift for the A14B pair.
///
/// Upstream sets `sample_shift` per task and per resolution — 12.0 for T2V and
/// 5.0 for I2V (`wan/configs/wan_{t2v,i2v}_A14B.py:34`) — but those pair with
/// upstream's 720p default. mold's A14B manifests render 480p, where 5.0 is the
/// value both upstream's own 480p path and the lightx2v four-step recipe use.
const A14B_FLOW_SHIFT: f64 = 5.0;

/// Resolve the flow shift, honouring `MOLD_WAN_SHIFT`.
///
/// `two_expert` picks the A14B value: the pair is a different schedule shape
/// from the single-expert checkpoints, not merely a bigger one.
fn resolve_flow_shift(two_expert: bool) -> Result<f64> {
    let default = if two_expert {
        A14B_FLOW_SHIFT
    } else {
        DEFAULT_FLOW_SHIFT
    };
    let Ok(raw) = std::env::var(FLOW_SHIFT_ENV) else {
        return Ok(default);
    };
    let parsed: f64 = raw
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("{FLOW_SHIFT_ENV} must be a number, got {raw:?}"))?;
    if !parsed.is_finite() || parsed <= 0.0 {
        bail!("{FLOW_SHIFT_ENV} must be finite and positive, got {parsed}");
    }
    Ok(parsed)
}

/// Reject the conditioning inputs this layer cannot honour.
///
/// Silently dropping them would render a plain text-to-video clip and look
/// like the conditioning simply had no effect.
fn reject_unsupported_conditioning(req: &GenerateRequest) -> Result<()> {
    let unsupported = [
        (
            req.source_video.is_some() || req.source_video_path.is_some(),
            "source_video",
        ),
        (
            req.extend_video.is_some() || req.extend_video_path.is_some(),
            "extend_video",
        ),
        (req.keyframes.is_some(), "keyframes"),
        // Core validation accepts source_image + mask_image as generic
        // inpainting, but Wan's conditioning never reads the mask — the
        // render would succeed while repainting everything, which reads as
        // "my mask had no effect".
        (req.mask_image.is_some(), "mask_image"),
    ];
    for (present, field) in unsupported {
        if present {
            bail!(
                "{field} is not yet supported for Wan — the family ships text-to-video and \
                 single-image conditioning; video-to-video, masks, and keyframes land later"
            );
        }
    }
    Ok(())
}

/// How a checkpoint expects its conditioning to arrive, derived from the ratio
/// between the DiT's input channels and the VAE's latent channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WanConditioningShape {
    /// `in_dim == z_dim` — the DiT consumes latents directly. Every T2V
    /// checkpoint, plus TI2V-5B, whose image conditioning happens by inpainting
    /// latent frame 0 rather than by widening the input.
    Plain,
    /// `in_dim == 2 * z_dim + 4` — the DiT consumes
    /// `cat([noise(z), mask(4), image(z)])`. Wan 2.1 I2V-14B and Wan 2.2
    /// I2V-A14B both declare 36 channels against the 16-channel 2.1 VAE.
    ChannelConcat,
}

/// Refuse the 36-channel checkpoints this engine cannot run correctly, naming
/// what each is missing.
///
/// Two distinct checkpoints declare 36 input channels. Wan 2.1 I2V additionally
/// carries a CLIP-vision cross-attention branch (`k_img`/`v_img`) the
/// transformer omits, and no amount of pairing fixes that. Wan 2.2 I2V-A14B is
/// architecturally a plain 36-channel DiT and runs correctly — but only as a
/// pair. Given one expert alone, the schedule's late half would be denoised by
/// the network trained for its early half, which renders rather than errors.
fn reject_unwired_channel_concat_checkpoint(
    transformer: &Path,
    low_noise_expert: Option<&Path>,
) -> Result<()> {
    let has_clip_branch = header_shapes(transformer)?.into_iter().any(|(name, _)| {
        name.trim_start_matches("model.diffusion_model.")
            .contains("cross_attn.k_img")
    });
    if has_clip_branch {
        bail!(
            "{} is a Wan 2.1 image-to-video checkpoint, which needs the CLIP-vision \
             cross-attention branch mold does not implement — use a Wan 2.2 checkpoint \
             (wan22-i2v-a14b, or wan22-ti2v-5b) for image conditioning",
            transformer.display()
        );
    }
    if low_noise_expert.is_some() {
        return Ok(());
    }
    bail!(
        "{} is one half of a Wan 2.2 I2V-A14B expert pair, and A14B denoises with both: the \
         high-noise expert down to timestep {}, the low-noise expert below it. Running one \
         alone would render, wrongly. Pull `wan22-i2v-a14b:q5` (or `:q8`) so both experts \
         resolve, or point `--transformer` at a single-expert checkpoint such as \
         wan22-ti2v-5b.",
        transformer.display(),
        WanExpertPair::boundary_for(true),
    )
}

pub(crate) fn conditioning_shape(in_dim: usize, z_dim: usize) -> Result<WanConditioningShape> {
    if in_dim == z_dim {
        return Ok(WanConditioningShape::Plain);
    }
    if in_dim == 2 * z_dim + 4 {
        return Ok(WanConditioningShape::ChannelConcat);
    }
    bail!(
        "Wan: a transformer with {in_dim} input channels does not pair with a {z_dim}-channel \
         VAE — expected {z_dim} (plain) or {} (image concat)",
        2 * z_dim + 4
    )
}

/// The conditioning a request resolved to, carrying whatever tensors the
/// denoise loop needs.
enum WanImageConditioning {
    None,
    /// TI2V-5B: latent frame 0 is pinned to the encoded image and the timestep
    /// vector goes per-token so that frame reports as clean.
    LatentInpaint {
        inpaint: WanTi2vInpaint,
        /// `[1, z, latent_frames, h, w]` — the encoded image broadcast over the
        /// frame axis. Only frame 0 survives the blend.
        condition: Tensor,
    },
    /// I2V checkpoints: a 20-channel `cat([mask, image_latent])` block that
    /// rides alongside the noise every step.
    ChannelConcat {
        conditioning: Tensor,
    },
}

/// Whether a request needs the unconditional pass.
///
/// At guidance <= 1 the CFG combination reduces to the conditional prediction,
/// so the second forward is pure waste. Skipping it is what makes the 4-step
/// Lightning recipe fast.
pub(crate) fn needs_cfg_pass(guidance: f64) -> bool {
    guidance > 1.0
}

/// The scalar timestep tensor handed to the DiT for one denoise step.
///
/// Deliberately F32 regardless of the compute dtype: the DiT's
/// `embed_timestep` immediately reads the value back as F32 → f64 for the
/// sinusoid, so a hop through BF16 (the CUDA compute dtype) is pure
/// information loss — its ulp is 4 in [512, 1024), which rounds the 4-step
/// Lightning grid `[999, 937, 833, 625]` to `[1000, 936, 832, 624]` (#786).
/// CPU already fed F32 here, so this also keeps the backends on the same
/// schedule.
fn scalar_timestep_tensor(timestep: i64, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(vec![timestep as f32], 1, device)?)
}

/// Everything one denoise run needs. Bundled because the loop is shared
/// between `generate` and the CPU smoke tests, and a dozen positional
/// parameters is worse than a struct.
struct DenoiseInputs<'a> {
    /// The weight source. For A14B this swaps experts mid-schedule; for every
    /// other checkpoint it hands back the same transformer every step.
    experts: &'a mut WanExperts,
    conditioning: &'a WanImageConditioning,
    schedule: &'a WanSchedule,
    solver: &'a mut FlowUniPc,
    latents: Tensor,
    cond_embeds: &'a Tensor,
    uncond_embeds: Option<&'a Tensor>,
    guidance: f64,
    /// DiT spatial patch size, needed to size the per-token timestep vector.
    patch: usize,
    rope: &'a (Tensor, Tensor),
    device: &'a Device,
    progress: &'a crate::progress::ProgressReporter,
}

/// The sampling loop for all three conditioning modes.
///
/// Extracted so the CPU smoke tests drive the real branching rather than a
/// copy of it — the conditioning mode decides what the DiT sees and how the
/// timestep is expressed, and those are exactly the parts worth testing.
fn run_denoise_loop(inputs: DenoiseInputs<'_>) -> Result<Tensor> {
    let DenoiseInputs {
        experts,
        conditioning,
        schedule,
        solver,
        mut latents,
        cond_embeds,
        uncond_embeds,
        guidance,
        patch,
        rope,
        device,
        progress,
    } = inputs;

    let total = schedule.timesteps.len();
    for (index, timestep) in schedule.timesteps.iter().enumerate() {
        progress.checkpoint()?;
        let step_start = Instant::now();
        // A14B switches experts here, once, when the schedule crosses the
        // boundary. Every other checkpoint hands back the same transformer.
        let transformer = experts.transformer_for(*timestep, progress)?;

        // Each conditioning mode decides what the DiT sees and how the
        // timestep is expressed. The solver always steps on `latents`.
        let scalar_timestep = || scalar_timestep_tensor(*timestep, device);
        let (model_input, timestep_tensor) = match conditioning {
            WanImageConditioning::None => (latents.clone(), scalar_timestep()?),
            WanImageConditioning::LatentInpaint { inpaint, condition } => {
                // Frame 0 is the clean encoded image; its tokens carry timestep
                // 0 so the DiT treats them as already denoised.
                let blended = inpaint.blend(condition, &latents)?;
                let per_token =
                    inpaint.per_token_timesteps(*timestep as f64, patch, None, device)?;
                (blended, per_token)
            }
            WanImageConditioning::ChannelConcat { conditioning } => (
                Tensor::cat(&[&latents, conditioning], 1)?,
                scalar_timestep()?,
            ),
        };

        let cond =
            transformer.forward_with_rope(&model_input, &timestep_tensor, cond_embeds, rope)?;
        let velocity = match uncond_embeds {
            Some(uncond_embeds) => {
                let uncond = transformer.forward_with_rope(
                    &model_input,
                    &timestep_tensor,
                    uncond_embeds,
                    rope,
                )?;
                apply_cfg(&cond, &uncond, guidance)?
            }
            None => cond,
        };
        latents = solver.step(&velocity, index, &latents)?;

        // Re-impose the clean frame after the step (Wan-native
        // `textimage2video.py:598`). Without this the final latent carries a
        // drifted frame 0 even though every model input was blended.
        if let WanImageConditioning::LatentInpaint { inpaint, condition } = conditioning {
            latents = inpaint.reimpose_clean_frame(condition, &latents)?;
        }

        progress.emit(ProgressEvent::DenoiseStep {
            step: index + 1,
            total,
            elapsed: step_start.elapsed(),
        });
    }
    Ok(latents)
}

pub struct WanEngine {
    base: EngineBase<()>,
    shared_pool: Option<Arc<Mutex<SharedPool>>>,
    pending_placement: Option<mold_core::types::DevicePlacement>,
}

impl WanEngine {
    pub fn new(
        model_name: String,
        paths: ModelPaths,
        load_strategy: LoadStrategy,
        gpu_ordinal: usize,
        shared_pool: Option<Arc<Mutex<SharedPool>>>,
    ) -> Self {
        Self {
            base: EngineBase::new(model_name, paths, load_strategy, gpu_ordinal),
            shared_pool,
            pending_placement: None,
        }
    }

    /// UMT5 weight shards. The manifest ships one fp16 safetensors, but the
    /// multi-shard field is the general contract.
    fn text_encoder_paths(&self) -> Result<Vec<PathBuf>> {
        let paths = &self.base.paths;
        if !paths.text_encoder_files.is_empty() {
            return Ok(paths.text_encoder_files.clone());
        }
        // Fall back to the generic T5 slot so a hand-configured model that
        // routes UMT5 through `t5_encoder` still loads.
        paths
            .t5_encoder
            .as_ref()
            .map(|path| vec![path.clone()])
            .ok_or_else(|| {
                anyhow::anyhow!("Wan: no UMT5 encoder weights configured for this model")
            })
    }

    fn tokenizer_path(&self) -> Result<PathBuf> {
        let paths = &self.base.paths;
        paths
            .text_tokenizer
            .clone()
            .or_else(|| paths.t5_tokenizer.clone())
            .ok_or_else(|| anyhow::anyhow!("Wan: no UMT5 tokenizer configured for this model"))
    }

    /// Decode the source image, encode it with the VAE, and assemble whatever
    /// the checkpoint's conditioning shape needs. The VAE is loaded and dropped
    /// here so it never coexists with the text encoder or the transformer.
    ///
    /// `WanVideoVae::encode` already returns the posterior mean with the
    /// per-channel normalization applied, which is exactly diffusers'
    /// `retrieve_latents(..., sample_mode="argmax")` followed by
    /// `(latent - mean) / std`.
    #[allow(clippy::too_many_arguments)]
    fn build_image_conditioning(
        &self,
        req: &GenerateRequest,
        shape: WanConditioningShape,
        vae_generation: WanVaeGeneration,
        geometry: WanLatentGeometry,
        pixel_frames: usize,
        width: u32,
        height: u32,
        device: &Device,
        dtype: DType,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<WanImageConditioning> {
        let Some(bytes) = req.source_image.as_ref() else {
            if shape == WanConditioningShape::ChannelConcat {
                bail!(
                    "this Wan checkpoint is image-to-video (it declares {} input channels) and \
                     needs a source image",
                    2 * vae_generation.config().z_dim + 4
                );
            }
            return Ok(WanImageConditioning::None);
        };

        if shape == WanConditioningShape::Plain && vae_generation != WanVaeGeneration::V2_2 {
            bail!(
                "this Wan checkpoint is text-to-video only and has no image conditioning path — \
                 use wan22-ti2v-5b, or an I2V checkpoint, for source images"
            );
        }

        progress.stage_start("Encoding source image");
        let encode_start = Instant::now();
        // Fit to the requested frame, matching every other mold engine's source
        // convention. mold deliberately does NOT run upstream's area bucketing,
        // which would silently resize; see `wan::conditioning`.
        let image = crate::img_utils::decode_source_image(
            bytes,
            width,
            height,
            crate::img_utils::NormalizeRange::MinusOneToOne,
            device,
            dtype,
        )?;

        let vae = WanVideoVae::from_safetensors(
            &self.base.paths.vae,
            vae_generation.config(),
            device,
            dtype,
        )?;

        let conditioning = match shape {
            WanConditioningShape::Plain => {
                // TI2V encodes the bare image: one pixel frame in, one latent
                // frame out, broadcast across the clip by the blend.
                let single = image.unsqueeze(2)?;
                let encoded = vae.encode(&single)?;
                let condition = encoded.broadcast_as((
                    1,
                    encoded.dim(1)?,
                    geometry.latent_frames,
                    geometry.latent_height,
                    geometry.latent_width,
                ))?;
                let inpaint = WanTi2vInpaint::new(geometry, device, dtype)?;
                WanImageConditioning::LatentInpaint {
                    inpaint,
                    condition: condition.contiguous()?,
                }
            }
            WanConditioningShape::ChannelConcat => {
                // I2V encodes the image followed by a black canvas, so the
                // conditioning latent spans the whole clip.
                let canvas = Tensor::zeros(
                    (1, 3, pixel_frames - 1, height as usize, width as usize),
                    dtype,
                    device,
                )?;
                let video = Tensor::cat(&[&image.unsqueeze(2)?, &canvas], 2)?;
                let encoded = vae.encode(&video)?;
                WanImageConditioning::ChannelConcat {
                    conditioning: build_a14b_conditioning(
                        &encoded,
                        pixel_frames,
                        WanImageAnchors::FirstFrame,
                        VAE_TEMPORAL_COMPRESSION,
                    )?,
                }
            }
        };
        drop(vae);
        device.synchronize()?;
        progress.phase_done(
            ProgressPhase::Vae,
            "Encoding source image",
            encode_start.elapsed(),
        );
        Ok(conditioning)
    }

    /// Build the denoise loop's weight source, loading the first expert.
    ///
    /// The manifest's distill (when the tier ships one) is stacked *under* the
    /// request's own adapters at full strength, which is how every distilled
    /// tier in mold behaves. The A14B distills are a pair — one per expert —
    /// and they are not interchangeable, so each is bound to its own slot; a
    /// user adapter has no expert affinity and applies to both.
    #[allow(clippy::too_many_arguments)]
    fn resolve_experts(
        &self,
        req: &GenerateRequest,
        config: &WanTransformerConfig,
        shape: WanConditioningShape,
        low_noise_expert: Option<&Path>,
        device: &Device,
        dtype: DType,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<WanExperts> {
        let paths = &self.base.paths;
        let user_loras = normalize_loras(req);
        let stack = |distill: Option<&Path>| -> Result<WanLoraRegistry> {
            let mut weights: Vec<mold_core::LoraWeight> = Vec::new();
            if let Some(path) = distill {
                weights.push(mold_core::LoraWeight {
                    path: path.to_string_lossy().to_string(),
                    scale: 1.0,
                });
            }
            weights.extend(user_loras.iter().cloned());
            WanLoraRegistry::load(&weights)
        };

        let Some(low_noise_path) = low_noise_expert else {
            let loras = stack(paths.distilled_lora.as_deref())?;
            if !loras.is_empty() {
                progress.info(&format!(
                    "Applying {} LoRA patch(es) across {} tensors",
                    loras.patch_count(),
                    loras.tensor_count()
                ));
            }
            progress.stage_start("Loading Wan transformer");
            let started = Instant::now();
            let transformer = crate::wan::experts::load_transformer(
                &paths.transformer,
                config.clone(),
                device,
                dtype,
                &loras,
            )?;
            progress.phase_done(
                ProgressPhase::ModelLoad,
                "Loading Wan transformer",
                started.elapsed(),
            );
            if let Some(marker) = transformer.quantization() {
                progress.info(&format!("fp8-scaled transformer ({marker})"));
            }
            return Ok(WanExperts::single(transformer));
        };

        // Both experts, and the boundary that separates them. The low-noise
        // config is read from its own header so a mismatched pair fails before
        // the first denoise step rather than at the swap.
        let low_noise_config = detect_transformer_config(low_noise_path)?;
        let pair = WanExpertPair {
            high_noise: WanExpertSlot {
                path: paths.transformer.clone(),
                loras: stack(paths.distilled_lora.as_deref())?,
            },
            low_noise: WanExpertSlot {
                path: low_noise_path.to_path_buf(),
                loras: stack(paths.low_noise_distilled_lora.as_deref())?,
            },
            boundary_timestep: WanExpertPair::boundary_for(
                shape == WanConditioningShape::ChannelConcat,
            ),
        };
        progress.info(&format!(
            "Wan A14B: two experts, switching at timestep {} ({} patch(es) on the high-noise \
             expert, {} on the low-noise one); one is resident at a time",
            pair.boundary_timestep,
            pair.high_noise.loras.patch_count(),
            pair.low_noise.loras.patch_count(),
        ));
        WanExperts::pair(pair, config.clone(), device, dtype, low_noise_config)
    }

    fn generate_inner(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let start = Instant::now();
        reject_unsupported_conditioning(req)?;

        let progress = &self.base.progress;
        let paths = &self.base.paths;

        // ------------------------------------------------------------------
        // Shape-driven configuration
        // ------------------------------------------------------------------
        let vae_generation = detect_vae_generation(&paths.vae)?;
        let vae_config = vae_generation.config();
        let transformer_config = detect_transformer_config(&paths.transformer)?;
        let shape = conditioning_shape(transformer_config.in_dim, vae_config.z_dim)?;
        let low_noise_expert = paths.low_noise_transformer.as_deref();
        // Wan 2.1 I2V is refused outright — it needs a CLIP-vision branch the
        // transformer omits. Wan 2.2 I2V-A14B runs, but only with both experts.
        if shape == WanConditioningShape::ChannelConcat {
            reject_unwired_channel_concat_checkpoint(&paths.transformer, low_noise_expert)?;
        }

        let (default_frames, default_fps) = vae_generation.default_timing();
        let num_frames = req.frames.unwrap_or(default_frames);
        let fps = req.fps.unwrap_or(default_fps);
        let steps = req.steps;
        let guidance = req.guidance;
        let seed = req.seed.unwrap_or_else(rand_seed);
        let (width, height) = (req.width, req.height);

        if num_frames == 0 || !(num_frames as usize - 1).is_multiple_of(VAE_TEMPORAL_COMPRESSION) {
            bail!(
                "Wan requires a 4n+1 frame count (1, 5, 9, ... 81, 121), got {}",
                num_frames
            );
        }
        // The DiT patches the latent 2x2 on top of the VAE's spatial stride,
        // so the pixel grid must clear both.
        let spatial_grid = vae_config.spatial_compression() * transformer_config.patch_size.1;
        if !width.is_multiple_of(spatial_grid as u32) || !height.is_multiple_of(spatial_grid as u32)
        {
            bail!(
                "Wan requires width and height to be multiples of {spatial_grid}, got {width}x{height}"
            );
        }

        let latent_frames = (num_frames as usize - 1) / VAE_TEMPORAL_COMPRESSION + 1;
        let latent_h = height as usize / vae_config.spatial_compression();
        let latent_w = width as usize / vae_config.spatial_compression();
        let shift = resolve_flow_shift(low_noise_expert.is_some())?;
        let needs_cfg = needs_cfg_pass(guidance);

        let device = crate::device::create_device(self.base.gpu_ordinal, progress)?;
        let dtype = gpu_dtype(&device);

        progress.info(&format!(
            "Wan: {width}x{height} x {num_frames} frames @ {fps} fps, {steps} steps, \
             guidance {guidance:.1}, shift {shift:.1}, seed {seed}"
        ));
        if !needs_cfg {
            progress.info("Guidance <= 1: running one forward per step (no CFG pass)");
        }

        // ------------------------------------------------------------------
        // 1. Image conditioning, first so the VAE never shares VRAM with the
        //    11 GB text encoder.
        // ------------------------------------------------------------------
        let geometry = WanLatentGeometry {
            latent_frames,
            latent_height: latent_h,
            latent_width: latent_w,
        };
        let conditioning = self.build_image_conditioning(
            req,
            shape,
            vae_generation,
            geometry,
            num_frames as usize,
            width,
            height,
            &device,
            dtype,
            progress,
        )?;

        // ------------------------------------------------------------------
        // 2. Prompt encoding, then drop the encoder before denoise
        // ------------------------------------------------------------------
        progress.stage_start("Loading UMT5-XXL encoder");
        let encoder_start = Instant::now();
        let tokenizer_path = self.tokenizer_path()?;
        let tokenizer = match &self.shared_pool {
            Some(pool) => pool.lock().unwrap().load_tokenizer(&tokenizer_path)?,
            None => Arc::new(
                tokenizers::Tokenizer::from_file(&tokenizer_path)
                    .map_err(|e| anyhow::anyhow!("Wan: loading UMT5 tokenizer failed: {e}"))?,
            ),
        };
        let text_device = crate::device::resolve_device(
            Some(
                self.pending_placement
                    .as_ref()
                    .map(|placement| placement.text_encoders.clone())
                    .unwrap_or_default(),
            ),
            || Ok(device.clone()),
        )?;
        let mut encoder = WanTextEncoder::load_with_tokenizer(
            &self.text_encoder_paths()?,
            &text_device,
            encoder_dtype_for(&text_device, dtype),
            tokenizer,
        )?;
        progress.phase_done(
            ProgressPhase::ModelLoad,
            "Loading UMT5-XXL encoder",
            encoder_start.elapsed(),
        );

        progress.stage_start("Encoding prompt");
        let encode_start = Instant::now();
        let negative = resolve_negative_prompt(req.negative_prompt.as_deref());
        let prompts: Vec<&str> = if needs_cfg {
            vec![req.prompt.as_str(), negative]
        } else {
            vec![req.prompt.as_str()]
        };
        let embeds = encoder
            .encode(&prompts)?
            .to_device(&device)?
            .to_dtype(dtype)?;
        let cond_embeds = embeds.narrow(0, 0, 1)?.contiguous()?;
        let uncond_embeds = if needs_cfg {
            Some(embeds.narrow(0, 1, 1)?.contiguous()?)
        } else {
            None
        };
        drop(embeds);
        progress.phase_done(
            ProgressPhase::PromptEncode,
            "Encoding prompt",
            encode_start.elapsed(),
        );

        // The encoder is 11.4 GB at fp16; it must be gone before the DiT loads.
        encoder.drop_weights();
        drop(encoder);
        device.synchronize()?;
        progress.info("UMT5 encoder dropped, VRAM freed");

        // ------------------------------------------------------------------
        // 2. Denoise
        // ------------------------------------------------------------------
        let schedule = WanSchedule::new(WanScheduleConfig::new(steps as usize, shift))?;
        let mut experts = self.resolve_experts(
            req,
            &transformer_config,
            shape,
            low_noise_expert,
            &device,
            dtype,
            progress,
        )?;

        let mut solver = FlowUniPc::new(schedule.clone());
        let latents = seeded_randn(
            seed,
            &[1, vae_config.z_dim, latent_frames, latent_h, latent_w],
            &device,
            DType::F32,
        )?
        .to_dtype(dtype)?;

        // Hoisted: the rotation tables depend only on the latent grid, which is
        // fixed for the whole run, and both A14B experts share an architecture
        // so one table serves the pair. Probe with the *model input* channel
        // count, not the latent's — the concat path widens it to `in_dim` and
        // `rope_freqs_for` validates that against the config.
        let rope = experts
            .transformer_for(
                schedule.timesteps.first().copied().unwrap_or_default(),
                progress,
            )?
            .rope_freqs_for(&Tensor::zeros(
                (
                    1,
                    transformer_config.in_dim,
                    latent_frames,
                    latent_h,
                    latent_w,
                ),
                dtype,
                &device,
            )?)?;

        progress.stage_start("Denoising");
        let latents = run_denoise_loop(DenoiseInputs {
            experts: &mut experts,
            conditioning: &conditioning,
            schedule: &schedule,
            solver: &mut solver,
            latents,
            cond_embeds: &cond_embeds,
            uncond_embeds: uncond_embeds.as_ref(),
            guidance,
            patch: transformer_config.patch_size.1,
            rope: &rope,
            device: &device,
            progress,
        })?;
        progress.checkpoint()?;
        drop(experts);
        device.synchronize()?;

        // ------------------------------------------------------------------
        // 3. VAE decode
        // ------------------------------------------------------------------
        progress.stage_start("Loading Wan VAE");
        let vae_start = Instant::now();
        let vae = WanVideoVae::from_safetensors(&paths.vae, vae_config, &device, dtype)?;
        progress.phase_done(
            ProgressPhase::ModelLoad,
            "Loading Wan VAE",
            vae_start.elapsed(),
        );

        progress.stage_start("Decoding video frames");
        let decode_start = Instant::now();
        let video = vae.decode(&latents)?;
        drop(vae);
        device.synchronize()?;
        progress.phase_done(
            ProgressPhase::Vae,
            "Decoding video frames",
            decode_start.elapsed(),
        );

        // ------------------------------------------------------------------
        // 4. Encode the artifact
        // ------------------------------------------------------------------
        let output_format = if req.resolved_output_format().is_video() {
            req.resolved_output_format()
        } else {
            OutputFormat::Apng
        };
        let format_name = output_format.extension().to_uppercase();
        progress.stage_start(&format!("Encoding {format_name}"));
        let encode_start = Instant::now();

        let frames = video_frames_to_images(&video, width, height)?;
        let frame_count = frames.len() as u32;
        let video_bytes = match output_format {
            OutputFormat::Apng => {
                let metadata = video_enc::VideoMetadata {
                    prompt: req.prompt.clone(),
                    model: self.base.model_name.clone(),
                    seed,
                    steps,
                    guidance,
                    width,
                    height,
                    frames: frame_count,
                    fps,
                };
                video_enc::encode_apng(&frames, fps, Some(&metadata))?
            }
            OutputFormat::Gif => video_enc::encode_gif(&frames, fps)?,
            #[cfg(feature = "webp")]
            OutputFormat::Webp => video_enc::encode_webp(&frames, fps)?,
            #[cfg(feature = "mp4")]
            OutputFormat::Mp4 => video_enc::encode_mp4(&frames, fps)?,
            #[cfg(not(feature = "webp"))]
            OutputFormat::Webp => {
                bail!("WebP output requires the 'webp' feature — rebuild with --features webp")
            }
            #[cfg(not(feature = "mp4"))]
            OutputFormat::Mp4 => {
                bail!("MP4 output requires the 'mp4' feature — rebuild with --features mp4")
            }
            _ => bail!("{format_name} is not a supported video output format"),
        };
        let thumbnail = video_enc::first_frame_png(&frames)?;
        let gif_preview = if req.gif_preview {
            if output_format == OutputFormat::Gif {
                video_bytes.clone()
            } else {
                video_enc::encode_gif(&frames, fps)?
            }
        } else {
            Vec::new()
        };
        progress.stage_done(&format!("Encoding {format_name}"), encode_start.elapsed());

        let generation_time_ms = start.elapsed().as_millis() as u64;
        progress.info(&format!(
            "Done: {frame_count} frames, {:.1}s total",
            generation_time_ms as f64 / 1000.0
        ));

        Ok(GenerateResponse {
            audio: None,
            images: vec![],
            video: Some(VideoData {
                data: video_bytes,
                format: output_format,
                width,
                height,
                frames: frame_count,
                fps,
                // `pipeline` is LTX-2's provenance slot; Wan has no pipeline
                // variants to record.
                pipeline: None,
                thumbnail,
                gif_preview,
                has_audio: false,
                duration_ms: None,
                audio_sample_rate: None,
                audio_channels: None,
            }),
            generation_time_ms,
            model: self.base.model_name.clone(),
            seed_used: seed,
            gpu: None,
        })
    }
}

/// `[1, 3, F, H, W]` in `[-1, 1]` to RGB frames, resampling if the VAE's
/// output grid differs from what the caller asked for.
fn video_frames_to_images(video: &Tensor, width: u32, height: u32) -> Result<Vec<image::RgbImage>> {
    let (_, channels, frame_count, decoded_h, decoded_w) = video.dims5()?;
    if channels != 3 {
        bail!("Wan VAE decoded {channels} channels, expected 3");
    }
    let bytes = ((video.to_dtype(DType::F32)?.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?
        .to_dtype(DType::U8)?
        .i(0)?;
    let mut frames = Vec::with_capacity(frame_count);
    for index in 0..frame_count {
        let frame = bytes
            .i((.., index, .., ..))?
            .contiguous()?
            .permute((1, 2, 0))?;
        let data: Vec<u8> = frame.flatten_all()?.to_vec1()?;
        let mut rgb = image::RgbImage::from_raw(decoded_w as u32, decoded_h as u32, data)
            .ok_or_else(|| anyhow::anyhow!("Wan: could not build frame {index}"))?;
        if decoded_w as u32 != width || decoded_h as u32 != height {
            rgb =
                image::imageops::resize(&rgb, width, height, image::imageops::FilterType::Triangle);
        }
        frames.push(rgb);
    }
    Ok(frames)
}

impl crate::engine::InferenceEngine for WanEngine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        self.base.progress.checkpoint()?;
        self.pending_placement = req.placement.clone();
        let result = self.generate_inner(req);
        self.pending_placement = None;
        result
    }

    fn model_name(&self) -> &str {
        &self.base.model_name
    }

    fn is_loaded(&self) -> bool {
        self.base.is_loaded()
    }

    fn load(&mut self) -> Result<()> {
        // Components are loaded and dropped inside `generate`; see the module
        // docs for why the 11.4 GB encoder cannot stay resident.
        Ok(())
    }

    fn unload(&mut self) {
        self.base.unload();
    }

    fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.base.set_on_progress(callback);
    }

    fn clear_on_progress(&mut self) {
        self.base.clear_on_progress();
    }

    fn set_cancellation_token(&mut self, token: crate::progress::InferenceCancellationToken) {
        self.base.set_cancellation_token(token);
    }

    fn clear_cancellation_token(&mut self) {
        self.base.clear_cancellation_token();
    }

    fn batch_execution_capability(&self) -> crate::BatchExecutionCapability {
        crate::batch_execution_capability_for_family("wan")
            .expect("production Wan batch capability must be registered")
    }

    fn model_paths(&self) -> Option<&ModelPaths> {
        Some(&self.base.paths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::InferenceEngine;
    use candle_nn::{VarBuilder, VarMap};
    use std::collections::HashMap;

    fn dummy_paths() -> ModelPaths {
        ModelPaths {
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
            transformer: PathBuf::from("/tmp/wan-transformer"),
            transformer_shards: vec![],
            vae: PathBuf::from("/tmp/wan-vae"),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![],
            text_tokenizer: None,
            decoder: None,
        }
    }

    /// #786: the schedule's integer timesteps must reach the DiT exactly.
    ///
    /// On CUDA the compute dtype is BF16 (`gpu_dtype`), whose ulp is 4 in
    /// [512, 1024): a hop through it rounds every step of the 4-step Lightning
    /// grid (999→1000, 937→936, 833→832, 625→624) before `embed_timestep`
    /// reads the value back as F32 for the sinusoid. BF16 rounds identically
    /// on CPU tensors, which is what makes the loss reproducible here.
    #[test]
    fn denoise_timesteps_survive_the_trip_to_the_embedding() {
        let schedule = WanSchedule::new(WanScheduleConfig::new(4, 5.0)).unwrap();
        assert_eq!(
            schedule.timesteps,
            vec![999, 937, 833, 625],
            "the 4-step Lightning grid moved; this test's premise is stale"
        );
        for timestep in &schedule.timesteps {
            let tensor = scalar_timestep_tensor(*timestep, &Device::Cpu).unwrap();
            // F32 until `embed_timestep` — a compute-dtype hop rounds it.
            assert_eq!(tensor.dtype(), DType::F32);
            // Decode exactly as `embed_timestep` does: an F32 read-back.
            let decoded = tensor
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(
                decoded,
                vec![*timestep as f32],
                "timestep {timestep} was corrupted on its way to the embedding"
            );
        }
    }

    fn request() -> GenerateRequest {
        GenerateRequest {
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            prompt: "a cat".to_string(),
            negative_prompt: None,
            model: "wan21-t2v-1.3b:bf16".to_string(),
            width: 832,
            height: 480,
            steps: 4,
            guidance: 6.0,
            seed: Some(7),
            batch_size: 1,
            output_format: None,
            embed_metadata: None,
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            source_image_name: None,
            edit_images: None,
            references: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            lora: None,
            frames: Some(81),
            fps: Some(16),
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            audio_file_path: None,
            source_video: None,
            source_video_path: None,
            extend_video: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            keyframes: None,
            pipeline: None,
            ic_lora_control: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
        }
    }

    /// The registry force-constructs every family with paths that do not
    /// exist; construction must not touch the filesystem.
    #[test]
    fn constructs_without_weights() {
        let engine = WanEngine::new(
            "wan21-t2v-1.3b:bf16".into(),
            dummy_paths(),
            LoadStrategy::Eager,
            0,
            None,
        );
        assert_eq!(engine.model_name(), "wan21-t2v-1.3b:bf16");
        assert!(
            !engine.is_loaded(),
            "an eagerly-strategised engine holds no weights until load"
        );
        assert!(engine.model_paths().is_some());
        assert_eq!(
            engine.batch_execution_capability(),
            crate::batch_execution_capability_for_family("wan").unwrap()
        );
    }

    /// Under the CLI's sequential strategy the engine reports ready because
    /// it loads on demand — the same contract `LtxVideoEngine` publishes.
    #[test]
    fn sequential_strategy_reports_ready() {
        let mut engine = WanEngine::new(
            "wan21-t2v-1.3b:bf16".into(),
            dummy_paths(),
            LoadStrategy::Sequential,
            0,
            None,
        );
        engine.load().unwrap();
        assert!(engine.is_loaded());
    }

    #[test]
    fn video_conditioning_is_rejected_but_images_are_accepted() {
        for mutate in [
            (|req: &mut GenerateRequest| req.source_video = Some(vec![1, 2, 3])) as fn(&mut _),
            |req: &mut GenerateRequest| req.extend_video = Some(vec![1, 2, 3]),
            |req: &mut GenerateRequest| req.source_video_path = Some("clip.mp4".into()),
            |req: &mut GenerateRequest| req.keyframes = Some(Vec::new()),
        ] {
            let mut req = request();
            mutate(&mut req);
            let error = reject_unsupported_conditioning(&req)
                .expect_err("video conditioning must be refused")
                .to_string();
            assert!(
                error.contains("not yet supported for Wan"),
                "unexpected error: {error}"
            );
            assert!(
                error.contains("single-image conditioning"),
                "the error must say what the engine does support: {error}"
            );
        }

        // Images are no longer refused at the request boundary.
        let mut req = request();
        req.source_image = Some(vec![1, 2, 3]);
        req.source_image_name = Some("cat.png".into());
        reject_unsupported_conditioning(&req).unwrap();
        reject_unsupported_conditioning(&request()).unwrap();
    }

    /// The conditioning shape is derived from the checkpoint's channel ratio,
    /// never from its name.
    #[test]
    fn conditioning_shape_comes_from_the_channel_ratio() {
        // T2V-1.3B / T2V-14B against the 16-channel 2.1 VAE.
        assert_eq!(
            conditioning_shape(16, 16).unwrap(),
            WanConditioningShape::Plain
        );
        // TI2V-5B against the 48-channel 2.2 VAE.
        assert_eq!(
            conditioning_shape(48, 48).unwrap(),
            WanConditioningShape::Plain
        );
        // I2V-14B / I2V-A14B: 16 noise + 4 mask + 16 image.
        assert_eq!(
            conditioning_shape(36, 16).unwrap(),
            WanConditioningShape::ChannelConcat
        );
        // A 2.2-VAE image checkpoint would be 48 + 4 + 48.
        assert_eq!(
            conditioning_shape(100, 48).unwrap(),
            WanConditioningShape::ChannelConcat
        );
        // Mismatched pairings are refused rather than silently mis-shaped.
        for (in_dim, z_dim) in [(36, 48), (16, 48), (48, 16), (20, 16)] {
            assert!(
                conditioning_shape(in_dim, z_dim).is_err(),
                "in_dim {in_dim} against z_dim {z_dim} must not resolve"
            );
        }
    }

    /// The Lightning recipe runs at guidance 1.0 and must not pay for a second
    /// forward per step.
    #[test]
    fn cfg_pass_is_skipped_at_or_below_unit_guidance() {
        assert!(!needs_cfg_pass(1.0));
        assert!(!needs_cfg_pass(0.0));
        assert!(needs_cfg_pass(1.0001));
        assert!(needs_cfg_pass(5.0));
        assert!(needs_cfg_pass(6.0));
    }

    #[test]
    fn flow_shift_defaults_and_validates() {
        // The env var is process-global; this test owns it for its duration.
        let previous = std::env::var(FLOW_SHIFT_ENV).ok();
        unsafe { std::env::remove_var(FLOW_SHIFT_ENV) };
        assert_eq!(resolve_flow_shift(false).unwrap(), DEFAULT_FLOW_SHIFT);
        // The A14B pair is a different schedule shape, not a bigger one.
        assert_eq!(resolve_flow_shift(true).unwrap(), A14B_FLOW_SHIFT);
        assert_ne!(A14B_FLOW_SHIFT, DEFAULT_FLOW_SHIFT);

        // An explicit override beats both defaults.
        unsafe { std::env::set_var(FLOW_SHIFT_ENV, "3.5") };
        assert_eq!(resolve_flow_shift(false).unwrap(), 3.5);
        assert_eq!(resolve_flow_shift(true).unwrap(), 3.5);

        for bad in ["", "abc", "0", "-2", "inf"] {
            unsafe { std::env::set_var(FLOW_SHIFT_ENV, bad) };
            assert!(
                resolve_flow_shift(false).is_err(),
                "{bad:?} must be rejected"
            );
        }

        match previous {
            Some(value) => unsafe { std::env::set_var(FLOW_SHIFT_ENV, value) },
            None => unsafe { std::env::remove_var(FLOW_SHIFT_ENV) },
        }
    }

    #[test]
    fn vae_generations_carry_their_own_geometry_and_timing() {
        assert_eq!(WanVaeGeneration::V2_1.config().z_dim, 16);
        assert_eq!(WanVaeGeneration::V2_1.config().spatial_compression(), 8);
        assert_eq!(WanVaeGeneration::V2_1.default_timing(), (81, 16));

        assert_eq!(WanVaeGeneration::V2_2.config().z_dim, 48);
        assert_eq!(WanVaeGeneration::V2_2.config().spatial_compression(), 16);
        assert_eq!(WanVaeGeneration::V2_2.default_timing(), (121, 24));
    }

    /// Write a header-only safetensors file carrying the given shapes, so the
    /// detection probes can be exercised without real weights.
    fn write_header(path: &Path, shapes: &[(&str, &[usize])]) {
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for (name, shape) in shapes {
            tensors.insert(
                (*name).to_string(),
                Tensor::zeros(*shape, DType::F32, &Device::Cpu).unwrap(),
            );
        }
        candle_core::safetensors::save(&tensors, path).unwrap();
    }

    #[test]
    fn transformer_config_is_detected_from_checkpoint_shapes() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("wan.safetensors");
        // The 1.3B geometry: dim 1536, ffn 8960, 30 layers, in/out 16.
        let mut shapes: Vec<(String, Vec<usize>)> = vec![
            ("patch_embedding.weight".into(), vec![1536, 16, 1, 2, 2]),
            ("blocks.0.ffn.0.weight".into(), vec![8960, 1536]),
            ("text_embedding.0.weight".into(), vec![1536, 4096]),
            ("time_embedding.0.weight".into(), vec![1536, 256]),
            ("head.head.weight".into(), vec![64, 1536]),
        ];
        for layer in 0..30 {
            shapes.push((format!("blocks.{layer}.modulation"), vec![1, 6, 1536]));
        }
        let borrowed: Vec<(&str, &[usize])> = shapes
            .iter()
            .map(|(name, shape)| (name.as_str(), shape.as_slice()))
            .collect();
        write_header(&path, &borrowed);

        let config = detect_transformer_config(&path).unwrap();
        assert_eq!(config, WanTransformerConfig::t2v_1_3b());
    }

    /// The same detection has to work on a GGUF expert, because that is the
    /// only container the A14B tiers ship in.
    ///
    /// GGML stores dimensions fastest-varying-first and candle's reader
    /// reverses them into torch order; a probe that reversed them again would
    /// read this checkpoint as 13824 wide with 5120 FFN channels — a config
    /// that loads, and is a transposed model.
    #[test]
    fn transformer_config_is_detected_from_a_gguf_expert() {
        use candle_core::quantized::{gguf_file, GgmlDType, QTensor};

        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("expert.gguf");
        let device = Device::Cpu;

        // The A14B geometry, 36-channel I2V: dim 5120, ffn 13824, 40 layers.
        // Only the tensors detection reads, at their real shapes.
        let mut shapes: Vec<(String, Vec<usize>)> = vec![
            ("patch_embedding.weight".into(), vec![5120, 36, 1, 2, 2]),
            ("blocks.0.ffn.0.weight".into(), vec![13824, 5120]),
            ("text_embedding.0.weight".into(), vec![5120, 4096]),
            ("time_embedding.0.weight".into(), vec![5120, 256]),
            ("head.head.weight".into(), vec![64, 5120]),
        ];
        for layer in 0..40 {
            shapes.push((format!("blocks.{layer}.modulation"), vec![1, 6, 8]));
        }

        let quantized: Vec<(String, QTensor)> = shapes
            .iter()
            .map(|(name, shape)| {
                let tensor = Tensor::zeros(shape.as_slice(), DType::F32, &device).unwrap();
                (
                    name.clone(),
                    QTensor::quantize(&tensor, GgmlDType::F32).unwrap(),
                )
            })
            .collect();
        let refs: Vec<(&str, &QTensor)> = quantized
            .iter()
            .map(|(name, q)| (name.as_str(), q))
            .collect();
        let arch = gguf_file::Value::String("wan".to_string());
        let mut file = std::fs::File::create(&path).unwrap();
        gguf_file::write(&mut file, &[("general.architecture", &arch)], &refs).unwrap();
        drop(file);

        let config = detect_transformer_config(&path).unwrap();
        assert_eq!(config, WanTransformerConfig::i2v_14b());
        // And the conditioning shape that follows from it, which is what routes
        // the expert-pair check.
        assert_eq!(
            conditioning_shape(config.in_dim, 16).unwrap(),
            WanConditioningShape::ChannelConcat
        );
    }

    /// The shipped Comfy-Org repacks prefix every DiT key with
    /// `model.diffusion_model.` (verified against the real 1.3B header, 825
    /// keys, all prefixed). Detection must see through it exactly like the
    /// loader does — this was a hard generate-time failure before the fix.
    #[test]
    fn transformer_detection_sees_through_the_shipped_prefix() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("wan-prefixed.safetensors");
        let mut shapes: Vec<(String, Vec<usize>)> = vec![
            (
                "model.diffusion_model.patch_embedding.weight".into(),
                vec![1536, 16, 1, 2, 2],
            ),
            (
                "model.diffusion_model.blocks.0.ffn.0.weight".into(),
                vec![8960, 1536],
            ),
            (
                "model.diffusion_model.text_embedding.0.weight".into(),
                vec![1536, 4096],
            ),
            (
                "model.diffusion_model.time_embedding.0.weight".into(),
                vec![1536, 256],
            ),
            (
                "model.diffusion_model.head.head.weight".into(),
                vec![64, 1536],
            ),
        ];
        for layer in 0..30 {
            shapes.push((
                format!("model.diffusion_model.blocks.{layer}.modulation"),
                vec![1, 6, 1536],
            ));
        }
        let borrowed: Vec<(&str, &[usize])> = shapes
            .iter()
            .map(|(name, shape)| (name.as_str(), shape.as_slice()))
            .collect();
        write_header(&path, &borrowed);

        let config = detect_transformer_config(&path).unwrap();
        assert_eq!(config, WanTransformerConfig::t2v_1_3b());
    }

    /// A mask the conditioning never reads must be rejected, not silently
    /// ignored — a full-image repaint that "succeeds" reads as a broken mask.
    #[test]
    fn mask_image_is_rejected_with_a_clear_message() {
        let mut req = request();
        req.source_image = Some(vec![1, 2, 3]);
        req.mask_image = Some(vec![4, 5, 6]);
        let err = reject_unsupported_conditioning(&req).unwrap_err();
        assert!(err.to_string().contains("mask_image"), "got: {err}");
    }

    /// 36-channel checkpoints are refused with a message naming what is
    /// missing: the 2.1 CLIP branch, or A14B expert switching.
    #[test]
    fn unwired_channel_concat_checkpoints_are_refused_by_name() {
        let temp = tempfile::tempdir().unwrap();

        // 2.1 I2V: carries cross_attn.k_img — refused for the CLIP branch.
        let clip = temp.path().join("wan21-i2v.safetensors");
        write_header(
            &clip,
            &[
                ("patch_embedding.weight", &[5120, 36, 1, 2, 2][..]),
                ("blocks.0.cross_attn.k_img.weight", &[5120, 1280][..]),
            ],
        );
        // A Wan 2.1 I2V checkpoint is refused with or without a partner: no
        // pairing supplies the CLIP-vision branch the transformer omits.
        let partner = temp.path().join("a14b-low.safetensors");
        write_header(
            &partner,
            &[("patch_embedding.weight", &[5120, 36, 1, 2, 2][..])],
        );
        for low_noise in [None, Some(partner.as_path())] {
            let err = reject_unwired_channel_concat_checkpoint(&clip, low_noise).unwrap_err();
            assert!(err.to_string().contains("CLIP-vision"), "got: {err}");
        }

        // A lone 2.2 A14B expert is refused, naming the boundary it would have
        // switched at and what to pull instead.
        let expert = temp.path().join("a14b-high.safetensors");
        write_header(
            &expert,
            &[("patch_embedding.weight", &[5120, 36, 1, 2, 2][..])],
        );
        let err = reject_unwired_channel_concat_checkpoint(&expert, None).unwrap_err();
        assert!(err.to_string().contains("expert pair"), "got: {err}");
        assert!(err.to_string().contains("900"), "got: {err}");
        assert!(err.to_string().contains("wan22-i2v-a14b"), "got: {err}");

        // With both experts resolved it is accepted — this is the arm the A14B
        // layer replaced.
        reject_unwired_channel_concat_checkpoint(&expert, Some(&partner))
            .expect("a complete A14B pair is runnable");
    }

    /// A CPU-parked encoder must compute in F32 — candle's CPU backend has no
    /// BF16/F16 matmul, and placement parks the encoder exactly when the big
    /// transformer needs the VRAM (found by the first real TI2V-5B run).
    #[test]
    fn cpu_parked_encoder_coerces_to_f32() {
        assert_eq!(encoder_dtype_for(&Device::Cpu, DType::BF16), DType::F32);
        assert_eq!(encoder_dtype_for(&Device::Cpu, DType::F16), DType::F32);
        // Non-CPU devices keep the model dtype; constructing a CUDA device in
        // a CPU test is not possible, so pin the CPU half of the contract and
        // the identity half via the function's own logic on Cpu+F32.
        assert_eq!(encoder_dtype_for(&Device::Cpu, DType::F32), DType::F32);
    }

    /// Absence falls back to the tuned default; an explicit value — the empty
    /// string `--no-negative` produces included — is authoritative.
    #[test]
    fn explicit_empty_negative_prompt_is_preserved() {
        assert_eq!(
            resolve_negative_prompt(None),
            mold_core::manifest::WAN_DEFAULT_NEGATIVE_PROMPT
        );
        assert_eq!(resolve_negative_prompt(Some("")), "");
        assert_eq!(resolve_negative_prompt(Some("   ")), "");
        assert_eq!(resolve_negative_prompt(Some("blurry")), "blurry");
    }

    #[test]
    fn transformer_detection_reads_ti2v_geometry() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("ti2v.safetensors");
        let mut shapes: Vec<(String, Vec<usize>)> = vec![
            ("patch_embedding.weight".into(), vec![3072, 48, 1, 2, 2]),
            ("blocks.0.ffn.0.weight".into(), vec![14336, 3072]),
            ("text_embedding.0.weight".into(), vec![3072, 4096]),
            ("time_embedding.0.weight".into(), vec![3072, 256]),
            // out_dim 48 x patch 4.
            ("head.head.weight".into(), vec![192, 3072]),
        ];
        for layer in 0..30 {
            shapes.push((format!("blocks.{layer}.modulation"), vec![1, 6, 3072]));
        }
        let borrowed: Vec<(&str, &[usize])> = shapes
            .iter()
            .map(|(name, shape)| (name.as_str(), shape.as_slice()))
            .collect();
        write_header(&path, &borrowed);

        let config = detect_transformer_config(&path).unwrap();
        assert_eq!(config, WanTransformerConfig::ti2v_5b());
        assert_eq!(config.num_heads, 24);
        assert_eq!(config.head_dim(), 128);
    }

    #[test]
    fn transformer_detection_rejects_a_foreign_checkpoint() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("not-wan.safetensors");
        write_header(&path, &[("some.other.weight", &[16, 16])]);
        let error = detect_transformer_config(&path).unwrap_err().to_string();
        assert!(error.contains("patch_embedding.weight"), "{error}");
    }

    /// The VAE generation must come from the checkpoint's key layout, since
    /// the 2.2 nesting is the only reliable discriminator.
    #[test]
    fn vae_generation_is_detected_from_the_key_layout() {
        let temp = tempfile::tempdir().unwrap();

        let v21 = temp.path().join("wan_2.1_vae.safetensors");
        write_header(
            &v21,
            &[
                ("decoder.middle.0.residual.0.gamma", &[384, 1, 1, 1]),
                (
                    "decoder.upsamples.0.residual.2.weight",
                    &[384, 384, 3, 3, 3],
                ),
            ],
        );
        assert_eq!(detect_vae_generation(&v21).unwrap(), WanVaeGeneration::V2_1);

        let v22 = temp.path().join("wan2.2_vae.safetensors");
        write_header(
            &v22,
            &[
                ("decoder.middle.0.residual.0.gamma", &[1024, 1, 1, 1]),
                (
                    "decoder.upsamples.0.upsamples.0.residual.2.weight",
                    &[1024, 1024, 3, 3, 3],
                ),
            ],
        );
        assert_eq!(detect_vae_generation(&v22).unwrap(), WanVaeGeneration::V2_2);

        let foreign = temp.path().join("foreign.safetensors");
        write_header(&foreign, &[("encoder.conv_in.weight", &[4, 4])]);
        assert!(detect_vae_generation(&foreign).is_err());
    }

    /// End-to-end on CPU at toy widths: tiny DiT + tiny VAE + a real
    /// FlowUniPC schedule, exercising the whole denoise/decode path including
    /// the CFG branch and the artifact encode.
    fn tiny_engine_run(guidance: f64, steps: u32) -> Vec<image::RgbImage> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let vae_config = WanVaeConfig::tiny_v2_1();
        let varmap = VarMap::new();
        let vae = WanVideoVae::from_var_builder(
            VarBuilder::from_varmap(&varmap, dtype, &device),
            vae_config.clone(),
            &device,
            dtype,
        )
        .unwrap();

        let transformer_config = WanTransformerConfig {
            in_dim: vae_config.z_dim,
            out_dim: vae_config.z_dim,
            ffn_dim: 32,
            text_dim: 32,
            freq_dim: 16,
            ..WanTransformerConfig::tiny(16, 2, 2)
        };
        let transformer_map = VarMap::new();
        let transformer = WanTransformer::from_var_builder(
            VarBuilder::from_varmap(&transformer_map, dtype, &device),
            transformer_config.clone(),
        )
        .unwrap();

        // 5 pixel frames -> 2 latent frames; 32x32 pixels -> 4x4 latent.
        let (frames, width, height) = (5usize, 32u32, 32u32);
        let latent_frames = (frames - 1) / VAE_TEMPORAL_COMPRESSION + 1;
        let latent_h = height as usize / vae_config.spatial_compression();
        let latent_w = width as usize / vae_config.spatial_compression();

        let context = Tensor::zeros((1, 6, 32), dtype, &device).unwrap();
        let schedule = WanSchedule::new(WanScheduleConfig::new(steps as usize, 8.0)).unwrap();
        let mut solver = FlowUniPc::new(schedule.clone());
        let mut latents = seeded_randn(
            7,
            &[1, vae_config.z_dim, latent_frames, latent_h, latent_w],
            &device,
            dtype,
        )
        .unwrap();
        let rope = transformer.rope_freqs_for(&latents).unwrap();

        for (index, timestep) in schedule.timesteps.iter().enumerate() {
            let t = Tensor::from_vec(vec![*timestep as f32], 1, &device).unwrap();
            let cond = transformer
                .forward_with_rope(&latents, &t, &context, &rope)
                .unwrap();
            let velocity = if needs_cfg_pass(guidance) {
                let uncond = transformer
                    .forward_with_rope(&latents, &t, &context, &rope)
                    .unwrap();
                apply_cfg(&cond, &uncond, guidance).unwrap()
            } else {
                cond
            };
            latents = solver.step(&velocity, index, &latents).unwrap();
        }

        let video = vae.decode(&latents).unwrap();
        assert_eq!(
            video.dims(),
            &[1, 3, frames, height as usize, width as usize]
        );
        video_frames_to_images(&video, width, height).unwrap()
    }

    #[test]
    fn tiny_end_to_end_denoise_and_decode_produces_frames() {
        let frames = tiny_engine_run(5.0, 4);
        assert_eq!(frames.len(), 5);
        for frame in &frames {
            assert_eq!(frame.dimensions(), (32, 32));
        }

        // The artifact encoders must accept what the pipeline produces.
        let apng = video_enc::encode_apng(&frames, 16, None).unwrap();
        assert!(!apng.is_empty());
        let thumbnail = video_enc::first_frame_png(&frames).unwrap();
        assert!(!thumbnail.is_empty());
    }

    /// The single-pass path must reach the same shapes as the CFG path — the
    /// guidance branch changes the arithmetic, not the plumbing.
    #[test]
    fn single_pass_guidance_runs_the_same_pipeline() {
        let frames = tiny_engine_run(1.0, 4);
        assert_eq!(frames.len(), 5);
        assert_eq!(frames[0].dimensions(), (32, 32));
    }

    /// Build a tiny model pair and drive the *real* denoise loop for one
    /// conditioning mode. Returns the final latents plus the decoded frames, so
    /// tests can assert both the latent invariant and the pixel outcome.
    fn tiny_i2v_runs(
        shape: WanConditioningShape,
        source_seeds: &[u64],
        guidance: f64,
    ) -> Vec<(Tensor, Tensor, Vec<image::RgbImage>)> {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let (pixel_frames, width, height) = (5usize, 32u32, 32u32);

        let vae_config = WanVaeConfig::tiny_v2_1();
        let vae_map = VarMap::new();
        let vae = WanVideoVae::from_var_builder(
            VarBuilder::from_varmap(&vae_map, dtype, &device),
            vae_config.clone(),
            &device,
            dtype,
        )
        .unwrap();

        let z = vae_config.z_dim;
        let in_dim = match shape {
            WanConditioningShape::Plain => z,
            WanConditioningShape::ChannelConcat => 2 * z + 4,
        };
        assert_eq!(conditioning_shape(in_dim, z).unwrap(), shape);

        let transformer_config = WanTransformerConfig {
            in_dim,
            out_dim: z,
            ffn_dim: 32,
            text_dim: 32,
            freq_dim: 16,
            ..WanTransformerConfig::tiny(16, 2, 2)
        };
        let transformer_map = VarMap::new();
        let transformer = WanTransformer::from_var_builder(
            VarBuilder::from_varmap(&transformer_map, dtype, &device),
            transformer_config.clone(),
        )
        .unwrap();

        let latent_frames = (pixel_frames - 1) / VAE_TEMPORAL_COMPRESSION + 1;
        let latent_h = height as usize / vae_config.spatial_compression();
        let latent_w = width as usize / vae_config.spatial_compression();
        let geometry = WanLatentGeometry {
            latent_frames,
            latent_height: latent_h,
            latent_width: latent_w,
        };

        // One model, many source images: the differential test must vary only
        // the image, and a fresh `VarMap` per call would vary every weight too.
        let mut outcomes = Vec::with_capacity(source_seeds.len());
        for source_seed in source_seeds.iter().copied() {
            // A deterministic "source image" in [-1, 1].
            let image = seeded_randn(
                source_seed,
                &[1, 3, height as usize, width as usize],
                &device,
                dtype,
            )
            .unwrap()
            .clamp(-1f32, 1f32)
            .unwrap();

            let conditioning = match shape {
                WanConditioningShape::Plain => {
                    let encoded = vae.encode(&image.unsqueeze(2).unwrap()).unwrap();
                    let condition = encoded
                        .broadcast_as((1, z, latent_frames, latent_h, latent_w))
                        .unwrap()
                        .contiguous()
                        .unwrap();
                    WanImageConditioning::LatentInpaint {
                        inpaint: WanTi2vInpaint::new(geometry, &device, dtype).unwrap(),
                        condition,
                    }
                }
                WanConditioningShape::ChannelConcat => {
                    let canvas = Tensor::zeros(
                        (1, 3, pixel_frames - 1, height as usize, width as usize),
                        dtype,
                        &device,
                    )
                    .unwrap();
                    let video = Tensor::cat(&[&image.unsqueeze(2).unwrap(), &canvas], 2).unwrap();
                    let encoded = vae.encode(&video).unwrap();
                    WanImageConditioning::ChannelConcat {
                        conditioning: build_a14b_conditioning(
                            &encoded,
                            pixel_frames,
                            WanImageAnchors::FirstFrame,
                            VAE_TEMPORAL_COMPRESSION,
                        )
                        .unwrap(),
                    }
                }
            };

            let context = Tensor::zeros((1, 6, 32), dtype, &device).unwrap();
            let schedule = WanSchedule::new(WanScheduleConfig::new(4, 8.0)).unwrap();
            // A fresh solver per source: `FlowUniPc` carries multistep history.
            let mut solver = FlowUniPc::new(schedule.clone());
            let latents = seeded_randn(
                7,
                &[1, z, latent_frames, latent_h, latent_w],
                &device,
                dtype,
            )
            .unwrap();
            let progress = crate::progress::ProgressReporter::default();
            // `WanTransformer` is `Clone` over Arc-backed weights, so every
            // source image runs against the same numbers, not a copy of them.
            let mut experts = WanExperts::single(transformer.clone());
            let rope = experts
                .transformer_for(schedule.timesteps[0], &progress)
                .unwrap()
                .rope_freqs_for(
                    &Tensor::zeros(
                        (1, in_dim, latent_frames, latent_h, latent_w),
                        dtype,
                        &device,
                    )
                    .unwrap(),
                )
                .unwrap();
            let final_latents = run_denoise_loop(DenoiseInputs {
                experts: &mut experts,
                conditioning: &conditioning,
                schedule: &schedule,
                solver: &mut solver,
                latents,
                cond_embeds: &context,
                uncond_embeds: None,
                guidance,
                patch: transformer_config.patch_size.1,
                rope: &rope,
                device: &device,
                progress: &progress,
            })
            .unwrap();

            let condition_tensor = match &conditioning {
                WanImageConditioning::LatentInpaint { condition, .. } => condition.clone(),
                WanImageConditioning::ChannelConcat { conditioning } => conditioning.clone(),
                WanImageConditioning::None => unreachable!(),
            };
            let video = vae.decode(&final_latents).unwrap();
            let frames = video_frames_to_images(&video, width, height).unwrap();
            outcomes.push((final_latents, condition_tensor, frames));
        }
        outcomes
    }

    fn tiny_i2v_run(
        shape: WanConditioningShape,
        source_seed: u64,
        guidance: f64,
    ) -> (Tensor, Tensor, Vec<image::RgbImage>) {
        tiny_i2v_runs(shape, &[source_seed], guidance)
            .pop()
            .expect("one source seed yields one outcome")
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// TI2V end to end: the loop runs, the clip decodes, and — the load-bearing
    /// part — latent frame 0 comes out bit-identical to the encoded image.
    /// That is the re-imposition contract; without it frame 0 drifts.
    #[test]
    fn tiny_ti2v_run_pins_latent_frame_zero_to_the_source() {
        let (latents, condition, frames) = tiny_i2v_run(WanConditioningShape::Plain, 11, 1.0);
        assert_eq!(latents.dims(), &[1, 4, 2, 4, 4]);
        assert_eq!(frames.len(), 5);
        assert_eq!(frames[0].dimensions(), (32, 32));

        let got = flat(&latents.narrow(2, 0, 1).unwrap().contiguous().unwrap());
        let want = flat(&condition.narrow(2, 0, 1).unwrap().contiguous().unwrap());
        assert_eq!(
            got, want,
            "latent frame 0 must survive every step untouched"
        );

        // And frame 1 must NOT equal the condition, or the loop denoised
        // nothing at all.
        let later = flat(&latents.narrow(2, 1, 1).unwrap().contiguous().unwrap());
        assert_ne!(later, want, "frames after the first must actually denoise");
        assert!(flat(&latents).iter().all(|v| v.is_finite()));
    }

    /// The source image must control latent frame 0 *exactly*, and must reach
    /// the decoded pixels.
    ///
    /// Note what this does NOT assert. The intuitive check — "decoded frame 0
    /// resembles the source more than later frames do" — carries no signal at
    /// tiny scale, and measurably runs the other way: swapping the source moves
    /// pixel frame 0 by ~26 and the last frame by ~39. Two reasons, both
    /// structural rather than bugs. The causal VAE decodes pixel frames 1..4
    /// from latent frames 0 *and* 1, so they inherit frame 0's change on top of
    /// their own; and a random-weight DiT amplifies the perturbation it sees
    /// into latent frame 1 instead of attenuating it. With trained weights the
    /// intuition would hold, but a test may not depend on that.
    ///
    /// The latent-space statement is exact and holds regardless of weights.
    #[test]
    fn tiny_ti2v_source_image_controls_frame_zero() {
        let runs = tiny_i2v_runs(WanConditioningShape::Plain, &[11, 29], 1.0);
        let frame_zero = |t: &Tensor| flat(&t.narrow(2, 0, 1).unwrap().contiguous().unwrap());
        let l1 = frame_zero(&runs[0].0);
        let l2 = frame_zero(&runs[1].0);
        let c1 = frame_zero(&runs[0].1);
        let c2 = frame_zero(&runs[1].1);

        // Latent frame 0 differs between the runs by exactly what the encoded
        // conditions differ by — element for element, not merely in aggregate.
        assert_eq!(l1.len(), c1.len());
        for (index, (((a, b), c), d)) in l1.iter().zip(&l2).zip(&c1).zip(&c2).enumerate() {
            assert_eq!(a - b, c - d, "element {index}");
        }
        let delta: f64 = l1
            .iter()
            .zip(&l2)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
            .sum();
        assert!(delta > 0.0, "the two source images must actually differ");

        // And the change survives the decode into pixels.
        let (a, b) = (&runs[0].2, &runs[1].2);
        let pixel_delta: u64 = a[0]
            .as_raw()
            .iter()
            .zip(b[0].as_raw())
            .map(|(x, y)| u64::from(x.abs_diff(*y)))
            .sum();
        assert!(
            pixel_delta > 0,
            "swapping the source image left decoded frame 0 untouched"
        );
    }

    /// The same source image must reproduce the same clip — the conditioning
    /// path introduces no nondeterminism of its own.
    #[test]
    fn tiny_ti2v_is_deterministic_for_one_source() {
        let runs = tiny_i2v_runs(WanConditioningShape::Plain, &[11, 11], 1.0);
        assert_eq!(flat(&runs[0].0), flat(&runs[1].0), "latents must match");
        for (left, right) in runs[0].2.iter().zip(&runs[1].2) {
            assert_eq!(left.as_raw(), right.as_raw(), "frames must match");
        }
    }

    /// The 36-channel concat path runs end to end. No manifest ships an I2V
    /// checkpoint yet, so the tiny config is the only coverage it has until
    /// layer 6 lands one.
    #[test]
    fn tiny_channel_concat_i2v_runs_end_to_end() {
        let (latents, conditioning, frames) =
            tiny_i2v_run(WanConditioningShape::ChannelConcat, 11, 1.0);
        // The DiT still emits z_dim channels even though it consumed 2z + 4.
        assert_eq!(latents.dims(), &[1, 4, 2, 4, 4]);
        // 4 mask channels + 4 latent channels for the tiny z_dim.
        assert_eq!(conditioning.dims(), &[1, 8, 2, 4, 4]);
        assert_eq!(frames.len(), 5);
        assert!(flat(&latents).iter().all(|v| v.is_finite()));

        // The mask half must be the first-frame pattern, not the latent.
        let mask = flat(&conditioning.narrow(1, 0, 4).unwrap().contiguous().unwrap());
        let per_frame = 4 * 4;
        for channel in 0..4 {
            assert_eq!(
                mask[(channel * 2) * per_frame],
                1.0,
                "channel {channel} frame 0"
            );
            assert_eq!(
                mask[(channel * 2 + 1) * per_frame],
                0.0,
                "channel {channel} frame 1"
            );
        }
    }

    /// Swapping the source image must change the concat conditioning too —
    /// proof the image actually reaches the block rather than only the mask.
    #[test]
    fn channel_concat_conditioning_depends_on_the_source_image() {
        let runs = tiny_i2v_runs(WanConditioningShape::ChannelConcat, &[11, 29], 1.0);
        let (a, b) = (&runs[0].1, &runs[1].1);
        // Mask halves identical, latent halves different.
        let mask_a = flat(&a.narrow(1, 0, 4).unwrap().contiguous().unwrap());
        let mask_b = flat(&b.narrow(1, 0, 4).unwrap().contiguous().unwrap());
        assert_eq!(mask_a, mask_b, "the mask does not depend on the image");
        let latent_a = flat(&a.narrow(1, 4, 4).unwrap().contiguous().unwrap());
        let latent_b = flat(&b.narrow(1, 4, 4).unwrap().contiguous().unwrap());
        assert_ne!(
            latent_a, latent_b,
            "the image latent must vary with the image"
        );
    }
}
