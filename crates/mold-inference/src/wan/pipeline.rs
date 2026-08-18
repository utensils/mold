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
use mold_core::{
    GenerateRequest, GenerateResponse, ImageData, ModelPaths, OutputFormat, VideoData,
};

use crate::engine::{rand_seed, seeded_randn, LoadStrategy};
use crate::engine_base::EngineBase;
use crate::ltx_video::video_enc;
use crate::progress::{ProgressCallback, ProgressEvent, ProgressPhase};
use crate::shared_pool::SharedPool;
use crate::wan::conditioning::{
    build_a14b_conditioning, WanImageAnchors, WanLatentGeometry, WanTi2vInpaint,
};
use crate::wan::experts::{
    WanExpertGuidance, WanExpertPair, WanExpertRole, WanExpertSlot, WanExperts,
};
use crate::wan::lora::WanLoraRegistry;
#[cfg(test)]
use crate::wan::model::transformer::WanTransformer;
use crate::wan::model::transformer::WanTransformerConfig;
use crate::wan::model::vae::{WanVaeConfig, WanVideoVae};
use crate::wan::sampler::{
    apply_cfg, FlowDpmPp, FlowEuler, FlowUniPc, WanSchedule, WanScheduleConfig, WanSolver,
};
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

/// Activation geometry for admission, read from the checkpoint's own header.
///
/// Admission needs the same shape facts the engine derives, and it must get
/// them through the same reader: [`header_shapes`] handles GGUF and
/// safetensors alike, and four of the shipped wan tiers are GGUF. A second
/// hand-rolled safetensors parser on the server side silently fell back for
/// every one of them.
///
/// Returns `None` when the file is not a Wan DiT in a layout this engine
/// recognizes; the caller keeps its conservative fallback rather than
/// inventing a shape.
/// Frames the engine renders when the request omits them, mirroring
/// `WanVaeGeneration::default_timing`.
fn default_frames_for_vae(vae: crate::device::WanVaeGeneration) -> u32 {
    match vae {
        crate::device::WanVaeGeneration::V22 => 121,
        crate::device::WanVaeGeneration::V21 => 81,
    }
}

/// Device memory this render's denoise activations will need, for the block
/// offload policy (#776 item 3).
///
/// `None` when the checkpoint's geometry cannot be read, which leaves the
/// policy to its explicit-request-only path rather than guessing a budget:
/// under-estimating here would park too few blocks and OOM anyway, and
/// over-estimating would park blocks a render did not need.
fn denoise_activation_bytes(
    req: &GenerateRequest,
    files: &[PathBuf],
    config: &WanTransformerConfig,
) -> Option<u64> {
    // Prefer the config the caller already resolved over re-probing the files:
    // the probe can fail on an unfamiliar export, and a `None` here silently
    // disables the whole offload policy rather than failing loudly.
    let geometry = activation_geometry_across(files).or_else(|| {
        let vae = match config.in_dim {
            48 => crate::device::WanVaeGeneration::V22,
            16 | 36 => crate::device::WanVaeGeneration::V21,
            _ => return None,
        };
        Some(crate::device::WanActivationGeometry {
            dim: config.dim as u64,
            ffn_dim: config.ffn_dim as u64,
            num_heads: config.num_heads as u64,
            vae,
            patch_spatial: config.patch_size.1.max(1) as u64,
            per_token_timesteps: false,
        })
    })?;
    // An absent frame count is the engine's own default, not "no video" — and
    // pricing it as zero would size the budget for a single latent frame and
    // disable offload exactly where it is needed most.
    let frames = req
        .frames
        .unwrap_or_else(|| default_frames_for_vae(geometry.vae));
    // The CALIBRATED budget, not the raw derived one. Using the raw sum here
    // under-estimated by 2.14x, so the policy parked nothing and the render
    // OOM'd at a shape admission had just accepted.
    Some(crate::device::wan_calibrated_activation_bytes(
        req.width,
        req.height,
        frames,
        geometry,
        needs_cfg_pass(req.guidance),
    ))
}

pub fn activation_geometry(path: &Path) -> Option<crate::device::WanActivationGeometry> {
    activation_geometry_across(std::slice::from_ref(&path.to_path_buf()))
}

/// Every file the DiT's weights live in: the primary plus any shards.
///
/// A diffusers export splits the transformer, and the split does not respect
/// the detection probe set, so anything reading the architecture out of the
/// header has to see the whole set rather than the first file. Deduped and
/// existence-filtered because a config may name the primary and the shards
/// with overlap, or name a file that is not on disk yet.
pub fn transformer_files(paths: &mold_core::ModelPaths) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = Vec::with_capacity(1 + paths.transformer_shards.len());
    for candidate in std::iter::once(&paths.transformer).chain(paths.transformer_shards.iter()) {
        if candidate.is_file() && !files.contains(candidate) {
            files.push(candidate.clone());
        }
    }
    if files.is_empty() {
        files.push(paths.transformer.clone());
    }
    files
}

/// [`activation_geometry`] over a checkpoint that may span shard files.
pub fn activation_geometry_across(
    paths: &[PathBuf],
) -> Option<crate::device::WanActivationGeometry> {
    let config = detect_transformer_config_across(paths).ok()?;
    let vae = match config.in_dim {
        48 => crate::device::WanVaeGeneration::V22,
        16 | 36 => crate::device::WanVaeGeneration::V21,
        _ => return None,
    };
    Some(crate::device::WanActivationGeometry {
        dim: config.dim as u64,
        ffn_dim: config.ffn_dim as u64,
        num_heads: config.num_heads as u64,
        vae,
        patch_spatial: config.patch_size.1.max(1) as u64,
        // Per-token timesteps are a property of the *request* — only the
        // TI2V latent inpaint drives them — so the geometry does not decide
        // it. The caller sets it from the request it is pricing.
        per_token_timesteps: false,
    })
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

/// One user adapter with the expert it resolved to.
pub(crate) struct RoutedLora {
    pub(crate) weight: mold_core::LoraWeight,
    pub(crate) expert: Option<mold_core::LoraExpert>,
    /// True when the expert came from the filename rather than the request.
    inferred: bool,
}

/// Every user adapter, routed.
pub(crate) struct WanLoraRouting {
    pub(crate) entries: Vec<RoutedLora>,
}

impl WanLoraRouting {
    /// Lines describing any routing the caller did not state explicitly.
    ///
    /// Inference is never silent: a user who names a file `..._high_noise` and
    /// gets it on one expert must be told, because the alternative reading —
    /// that it applied to both — produces a different render.
    pub(crate) fn disclosures(&self) -> Vec<String> {
        self.entries
            .iter()
            .filter(|entry| entry.inferred)
            .filter_map(|entry| {
                let expert = match entry.expert? {
                    mold_core::LoraExpert::High => "high-noise",
                    mold_core::LoraExpert::Low => "low-noise",
                };
                let name = std::path::Path::new(&entry.weight.path)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or(entry.weight.path.as_str());
                Some(format!(
                    "LoRA {name}: routed to the {expert} expert from its filename"
                ))
            })
            .collect()
    }
}

/// Resolve each user adapter to an expert.
///
/// An explicit `expert` field wins. Otherwise the filename is consulted for the
/// conventions publishers use, and anything still unresolved keeps the
/// historical apply-to-both behavior.
pub(crate) fn resolve_user_lora_experts(loras: &[mold_core::LoraWeight]) -> WanLoraRouting {
    WanLoraRouting {
        entries: loras
            .iter()
            .map(|weight| {
                let (expert, inferred) = match weight.expert {
                    Some(expert) => (Some(expert), false),
                    None => (
                        mold_core::LoraExpert::infer_from_filename(&weight.path),
                        true,
                    ),
                };
                RoutedLora {
                    weight: weight.clone(),
                    expert,
                    inferred: inferred && expert.is_some(),
                }
            })
            .collect(),
    }
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
    detect_transformer_config_across(std::slice::from_ref(&path.to_path_buf()))
}

/// Detect the config from a checkpoint that may span several shard files and
/// may be in any of the layouts [`classify_key_layout`] knows.
///
/// Two things force this over the single-file original-layout probe it
/// replaces, and both are properties of real published checkpoints:
///
/// - **Shards.** A diffusers export splits the DiT across files, and the
///   split does not respect the probe set: `Wan2.2-TI2V-5B-Turbo-Diffusers`
///   puts `proj_out.weight` — the output-channel probe — alone in a second
///   89 MB shard. Reading only the first file cannot see it.
/// - **Layout.** #803 taught the *loader* to translate diffusers names, but
///   detection runs first and still demanded the original spelling, so a
///   diffusers checkpoint failed on `blocks.0.ffn.0.weight` before the
///   translating loader was ever reached. Every probe is therefore resolved
///   through the same rename table the loader uses.
pub(crate) fn detect_transformer_config_across(paths: &[PathBuf]) -> Result<WanTransformerConfig> {
    use crate::wan::model::transformer::{classify_key_layout, original_to_diffusers};

    let first = paths
        .first()
        .ok_or_else(|| anyhow::anyhow!("Wan DiT: no transformer files supplied"))?;
    let mut shapes: Vec<(String, Vec<usize>)> = Vec::new();
    for path in paths {
        shapes.extend(header_shapes(path)?);
    }

    // Classify from the merged names, so a checkpoint whose discriminating
    // block lives in a later shard is still classified correctly.
    let layout = classify_key_layout(|probe| shapes.iter().any(|(name, _)| name == probe))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{} is not a Wan DiT checkpoint in any layout this build can address",
                first.display()
            )
        })?;

    // The shipped Comfy-Org repacks store every DiT key under
    // `model.diffusion_model.` while the VAE and encoder files are bare. The
    // loader strips that prefix; detection must see the same names.
    let shapes: Vec<(String, Vec<usize>)> = shapes
        .into_iter()
        .filter_map(|(name, shape)| {
            let bare = if layout.prefix.is_empty() {
                Some(name)
            } else {
                name.strip_prefix(layout.prefix).map(str::to_string)
            }?;
            Some((bare, shape))
        })
        .collect();

    let find = |key: &str| -> Result<&Vec<usize>> {
        // Probes are written in the original spelling; translate them into
        // the checkpoint's own when it is a diffusers export. A key the table
        // does not cover falls through unchanged, which fails by name below
        // rather than silently matching something else.
        let spelled = if layout.diffusers_names {
            original_to_diffusers(key).unwrap_or_else(|| key.to_string())
        } else {
            key.to_string()
        };
        shapes
            .iter()
            .find(|(name, _)| *name == spelled)
            .map(|(_, shape)| shape)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "{} is missing `{spelled}` — not a Wan DiT checkpoint in a layout this build \
                     can address",
                    first.display()
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
        .ok_or_else(|| anyhow::anyhow!("{} has no transformer blocks", first.display()))?;

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

/// Resolve the flow shift: request > `MOLD_WAN_SHIFT` > per-tier default
/// (#782). The request field arrives validated finite/positive; the env var
/// is validated here because `mold serve` reads it process-wide.
///
/// `two_expert` picks the A14B value: the pair is a different schedule shape
/// from the single-expert checkpoints, not merely a bigger one.
fn resolve_flow_shift(request_shift: Option<f64>, two_expert: bool) -> Result<f64> {
    if let Some(shift) = request_shift {
        if !shift.is_finite() || shift <= 0.0 {
            bail!("sample_shift must be finite and positive, got {shift}");
        }
        return Ok(shift);
    }
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

/// Env fallback for the solver selection (#795); the request's `scheduler`
/// slot wins.
const SOLVER_ENV: &str = "MOLD_WAN_SOLVER";

/// Which flow solver drives the denoise loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WanSolverKind {
    UniPc,
    Euler,
    DpmPp,
}

impl WanSolverKind {
    fn label(self) -> &'static str {
        match self {
            Self::UniPc => "unipc",
            Self::Euler => "euler",
            Self::DpmPp => "dpm++",
        }
    }
}

/// Resolve the solver: request `scheduler` > `MOLD_WAN_SOLVER` > FlowUniPC.
///
/// The UNet schedulers are rejected by name — admission already refuses them
/// for wan, but forced-local callers can skip validation and a silently
/// ignored selection would render with the wrong algorithm.
fn resolve_wan_solver(
    requested: Option<mold_core::Scheduler>,
    _two_expert: bool,
) -> Result<WanSolverKind> {
    use mold_core::Scheduler;
    if let Some(scheduler) = requested {
        return match scheduler {
            Scheduler::UniPc => Ok(WanSolverKind::UniPc),
            Scheduler::Euler => Ok(WanSolverKind::Euler),
            Scheduler::DpmPp => Ok(WanSolverKind::DpmPp),
            Scheduler::Ddim | Scheduler::EulerAncestral => bail!(
                "Wan supports the uni-pc, euler, and dpm-pp sample solvers; '{scheduler}' is a \
                 UNet scheduler"
            ),
        };
    }
    let Ok(raw) = std::env::var(SOLVER_ENV) else {
        return Ok(WanSolverKind::UniPc);
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "unipc" | "uni-pc" | "uni_pc" => Ok(WanSolverKind::UniPc),
        "euler" => Ok(WanSolverKind::Euler),
        "dpm++" | "dpmpp" | "dpm-pp" | "dpm_pp" => Ok(WanSolverKind::DpmPp),
        other => bail!("{SOLVER_ENV} must be unipc, euler, or dpm++, got {other:?}"),
    }
}

/// Build the selected solver over its own grid: dpm++ uses upstream's
/// `get_sampling_sigmas` layout, UniPC and euler share the diffusers/
/// Lightning grid.
fn build_wan_solver(kind: WanSolverKind, config: WanScheduleConfig) -> Result<WanSolver> {
    Ok(match kind {
        WanSolverKind::UniPc => WanSolver::UniPc(FlowUniPc::new(WanSchedule::new(config)?)),
        WanSolverKind::Euler => WanSolver::Euler(FlowEuler::new(WanSchedule::new(config)?)),
        WanSolverKind::DpmPp => WanSolver::DpmPp(FlowDpmPp::new(WanSchedule::dpmpp(config)?)),
    })
}

/// Validate a distill-strength override (#795). Absent = 1.0; the accepted
/// band covers the community's documented range with headroom (high 1.5-2.0
/// is the reduced-motion mitigation) while refusing values that can only be
/// typos.
fn resolve_distill_strength(label: &str, requested: Option<f64>) -> Result<f64> {
    let Some(strength) = requested else {
        return Ok(1.0);
    };
    if !strength.is_finite() || strength <= 0.0 || strength > 4.0 {
        bail!("distill_strength_{label} must be in (0, 4], got {strength}");
    }
    Ok(strength)
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
        // `extend_video` is handled before this guard (`extend_inner`), so
        // reaching here with one set would mean the route was missed.
        // Core validation accepts source_image + mask_image as generic
        // inpainting, but Wan's conditioning never reads the mask — the
        // render would succeed while repainting everything, which reads as
        // "my mask had no effect".
        (req.mask_image.is_some(), "mask_image"),
    ];
    for (present, field) in unsupported {
        if present {
            bail!(
                "{field} is not yet supported for Wan — the family ships text-to-video, \
                 single-image, and first/last-frame conditioning; video-to-video and masks \
                 land later"
            );
        }
    }
    Ok(())
}

/// The endpoint images a wan request conditions on, resolved from
/// `source_image` and/or the two-entry first/last `keyframes` layout (#779).
///
/// Every other keyframe layout is refused by name: wan's conditioning
/// contracts anchor pixel frames 0 and F-1 only — there is no mid-clip
/// keyframe path in the family.
#[derive(Debug)]
struct WanEndpointImages<'a> {
    first: &'a [u8],
    last: Option<&'a [u8]>,
}

fn resolve_endpoint_images(
    req: &GenerateRequest,
    num_frames: u32,
) -> Result<Option<WanEndpointImages<'_>>> {
    let keyframes = req.keyframes.as_deref().filter(|list| !list.is_empty());
    match (req.source_image.as_deref(), keyframes) {
        (None, None) => Ok(None),
        (Some(first), None) => Ok(Some(WanEndpointImages { first, last: None })),
        (Some(_), Some(_)) => bail!(
            "Wan takes the first frame from either source_image or keyframes[0], not both — \
             for a first/last-frame render, put both endpoints in keyframes"
        ),
        (None, Some(keyframes)) => {
            if keyframes.len() != 2 {
                bail!(
                    "Wan supports exactly two keyframes — the first and last pixel frames — got \
                     {}",
                    keyframes.len()
                );
            }
            // Coincident endpoints (frames <= 1) must be refused here, not
            // just at server admission: the forced-local path reaches this
            // resolver directly, and the I2V canvas math below computes
            // `frames - 2` interior frames, which would underflow.
            if num_frames < 2 {
                bail!(
                    "Wan first/last-frame keyframes need a multi-frame clip — frames={num_frames} \
                     has no distinct last frame"
                );
            }
            let last_index = num_frames.saturating_sub(1);
            if keyframes[0].frame != 0 || keyframes[1].frame != last_index {
                bail!(
                    "Wan first/last-frame keyframes must anchor frames 0 and {last_index} (the \
                     clip's endpoints), got frames {} and {}",
                    keyframes[0].frame,
                    keyframes[1].frame
                );
            }
            Ok(Some(WanEndpointImages {
                first: &keyframes[0].image,
                last: Some(&keyframes[1].image),
            }))
        }
    }
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
///
/// Both halves of the probe are set-wide rather than file-wide, and both had
/// to be (#803):
///
/// - **Spelling.** The CLIP branch is named differently per key layout, and
///   probing only the original `cross_attn.k_img` let every diffusers-layout
///   2.1 I2V export past this refusal. The diffusers markers are derived from
///   the same rename table the loader uses, so the two cannot drift apart.
/// - **Shards.** A diffusers export splits the DiT across files and the split
///   does not respect any one probe set — the reason
///   [`detect_transformer_config_across`] reads the whole set — so reading
///   only `paths.transformer` let a *sharded* diffusers I2V export through
///   whenever the CLIP adapter landed in a later shard. Take every file the
///   config detection takes.
fn reject_unwired_channel_concat_checkpoint(
    transformer_files: &[PathBuf],
    low_noise_expert: Option<&Path>,
) -> Result<()> {
    let transformer = transformer_files
        .first()
        .map(PathBuf::as_path)
        .ok_or_else(|| anyhow::anyhow!("Wan DiT: no transformer files supplied"))?;
    let markers = crate::wan::model::transformer::clip_vision_branch_markers();
    let mut has_clip_branch = false;
    for file in transformer_files {
        if header_shapes(file)?.into_iter().any(|(name, _)| {
            let bare = name.trim_start_matches("model.diffusion_model.");
            markers.iter().any(|marker| bare.contains(marker))
        }) {
            has_clip_branch = true;
            break;
        }
    }
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

/// Classify a Wan checkpoint's source-image contract from its own headers —
/// the exact classification `generate` applies, exported so `/api/models`
/// advertises what the engine will actually accept (#772).
///
/// - `ChannelConcat` (36-channel patch embedding): the image is half the
///   model input — required.
/// - `Plain` over the 48-channel 2.2 VAE: TI2V latent inpaint — optional.
/// - `Plain` over the 16-channel 2.1 VAE: text-to-video only — unsupported.
///
/// `None` when either header cannot be read or classified; callers must
/// treat that as "unknown", never as one of the three contracts.
pub fn source_image_capability(
    transformer: &Path,
    vae: &Path,
) -> Option<mold_core::SourceImageCapability> {
    let config = detect_transformer_config(transformer).ok()?;
    let vae_config = detect_vae_generation(vae).ok()?.config();
    match conditioning_shape(config.in_dim, vae_config.z_dim).ok()? {
        WanConditioningShape::ChannelConcat => Some(mold_core::SourceImageCapability::Required),
        WanConditioningShape::Plain if vae_config.z_dim == 48 => {
            Some(mold_core::SourceImageCapability::Optional)
        }
        WanConditioningShape::Plain => Some(mold_core::SourceImageCapability::Unsupported),
    }
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

/// How guidance varies over the schedule.
///
/// Upstream A14B guides each expert with its own scale; everything else — and
/// an A14B run where the user picked a scale explicitly — uses one scale for
/// the whole schedule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum WanGuidancePlan {
    Uniform(f64),
    /// Upstream's per-expert pair, selected per timestep with the same
    /// `>= boundary` comparison as the expert swap ([`WanExpertRole::at`]).
    PerExpert {
        boundary_timestep: i64,
        guidance: WanExpertGuidance,
    },
}

impl WanGuidancePlan {
    /// The scale for one denoise step.
    fn for_timestep(self, timestep: i64) -> f64 {
        match self {
            Self::Uniform(guidance) => guidance,
            Self::PerExpert {
                boundary_timestep,
                guidance,
            } => guidance.for_role(WanExpertRole::at(boundary_timestep, timestep)),
        }
    }

    /// The strongest scale any step will use. Decides whether the negative
    /// prompt is encoded at all; individual steps still skip their uncond
    /// forward whenever their own scale is <= 1 ([`needs_cfg_pass`]).
    fn max(self) -> f64 {
        match self {
            Self::Uniform(guidance) => guidance,
            Self::PerExpert { guidance, .. } => guidance.max(),
        }
    }
}

/// Decide between uniform and per-expert guidance.
///
/// Per-expert scales engage only when the request's scale is *this model's
/// own* advertised default AND that default is the quality tier's
/// [`WAN_A14B_QUALITY_GUIDANCE`](mold_core::manifest::WAN_A14B_QUALITY_GUIDANCE)
/// — and the checkpoint actually is a pair. Gating on the model's default,
/// not the family constant alone, keeps an explicit `--guidance 3.5` on a
/// Lightning tier (whose default is 1.0, so 3.5 there is always a choice)
/// honored uniformly, and keeps community pair installs without a manifest
/// on uniform guidance.
///
/// The one case the wire cannot express: on the quality tier itself an
/// explicit 3.5 is byte-identical to the default and receives the upstream
/// pair. `GenerateRequest.guidance` is a bare scalar, so "typed 3.5" and
/// "left the default" are the same request; a dedicated
/// `guidance_high`/`guidance_low` surface is deliberately deferred until
/// demanded (#796).
fn resolve_guidance_plan(
    requested: f64,
    pair_boundary: Option<i64>,
    channel_concat: bool,
    model_default: Option<f64>,
) -> WanGuidancePlan {
    let is_advertised_default = model_default == Some(requested)
        && requested == mold_core::manifest::WAN_A14B_QUALITY_GUIDANCE;
    match pair_boundary {
        Some(boundary_timestep) if is_advertised_default => WanGuidancePlan::PerExpert {
            boundary_timestep,
            guidance: WanExpertGuidance::upstream_for(channel_concat),
        },
        _ => WanGuidancePlan::Uniform(requested),
    }
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
    solver: &'a mut WanSolver,
    latents: Tensor,
    cond_embeds: &'a Tensor,
    uncond_embeds: Option<&'a Tensor>,
    guidance: WanGuidancePlan,
    /// DiT spatial patch size, needed to size the per-token timestep vector.
    patch: usize,
    rope: &'a (Tensor, Tensor),
    device: &'a Device,
    progress: &'a crate::progress::ProgressReporter,
    /// Live denoise previews. `None` when the checkpoint's latent channel
    /// count has no factor table.
    previewer: Option<&'a crate::latent_preview::LatentPreviewer>,
    /// First-block residual reuse (#801). `Off` runs every block on every
    /// step, which is the default and is bit-identical to the pre-cache
    /// engine.
    step_cache: crate::wan::step_cache::WanStepCachePolicy,
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
        previewer,
        step_cache,
    } = inputs;

    // Two caches, never one: the conditional and unconditional forwards are
    // different trajectories, and replaying one's residual into the other
    // would blend them.
    let (mut cond_cache, mut uncond_cache) = match step_cache {
        crate::wan::step_cache::WanStepCachePolicy::Off => (None, None),
        crate::wan::step_cache::WanStepCachePolicy::Threshold(threshold) => (
            Some(crate::wan::step_cache::WanStepCache::new(threshold)),
            Some(crate::wan::step_cache::WanStepCache::new(threshold)),
        ),
    };
    let mut cached_expert: Option<crate::wan::experts::WanExpertRole> = None;

    // #775 A/B switch: `MOLD_WAN_FORCE_DMMV=1` forces candle's quantized
    // matmuls onto the dequantize-per-forward fallback, so a normal run vs a
    // forced run measures whether the MMQ fast path is engaging and what it
    // is worth. Diagnostic only — never set in production.
    //
    // The switch it flips is process-global and never cleared, and candle
    // exposes no reader, so it goes through `crate::quantized_dmmv` — the
    // engines that dispatch on the quantized fast path (Qwen-Image's GGUF
    // transformer) have to be able to see this run's flip afterwards.
    // The mirror now self-initializes from the frozen env on first read, so
    // the process-global is constant before any engine's kernel dispatch —
    // flipping it here raced a concurrent Qwen GGUF forward. This site only
    // reports the diagnostic once.
    {
        static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        if crate::quantized_dmmv::force_dmmv_enabled() {
            WARNED.get_or_init(|| {
                tracing::warn!(
                    "wan: MOLD_WAN_FORCE_DMMV=1 — quantized matmuls on the dequant fallback"
                );
            });
        }
    }

    let total = schedule.timesteps.len();
    for (index, timestep) in schedule.timesteps.iter().enumerate() {
        progress.checkpoint()?;
        let step_start = Instant::now();
        // Resolved before the transformer borrow: per-expert guidance follows
        // the same boundary the swap below is about to consult.
        let step_guidance = guidance.for_timestep(*timestep);
        // The expert swap changes the network, so nothing recorded against
        // the previous one describes it. Reset both caches on the transition
        // rather than trusting a residual across it.
        if let Some(role) = experts.current_role(*timestep) {
            if cached_expert.is_some_and(|previous| previous != role) {
                if let Some(cache) = cond_cache.as_mut() {
                    cache.reset();
                }
                if let Some(cache) = uncond_cache.as_mut() {
                    cache.reset();
                }
            }
            cached_expert = Some(role);
        }

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

        let cond = transformer.forward_with_rope_cached(
            &model_input,
            &timestep_tensor,
            cond_embeds,
            rope,
            cond_cache.as_mut(),
        )?;
        // Wan video steps can spend a long time in each transformer pass.
        // Stop before launching the second CFG pass when cancellation arrived
        // during the conditional pass instead of making the user wait for a
        // full additional forward.
        progress.checkpoint()?;
        // A step whose own scale is <= 1 skips its uncond forward even when
        // the uncond embedding was encoded for other steps' sake.
        let velocity = match uncond_embeds {
            Some(uncond_embeds) if needs_cfg_pass(step_guidance) => {
                let uncond = transformer.forward_with_rope_cached(
                    &model_input,
                    &timestep_tensor,
                    uncond_embeds,
                    rope,
                    uncond_cache.as_mut(),
                )?;
                apply_cfg(&cond, &uncond, step_guidance)?
            }
            _ => cond,
        };
        latents = solver.step(&velocity, index, &latents)?;
        crate::wan::model::transformer::step_profile::report(index + 1);

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
        if let Some(previewer) = previewer {
            if previewer.due(index + 1, total) {
                // Preview the predicted clean latent, not the still-noisy
                // working latent — composition is visible from the first
                // step. The solver's own recorded x0 is the only valid
                // estimate: UniPC's corrector rewrites the sample the
                // predictor steps from, so recomputing `x - sigma * v` from
                // the post-step state projects a stale velocity. The final
                // step's x0 is what the order-1 terminal predictor returns,
                // so the last preview still matches the finished latent.
                match solver.last_x0() {
                    Some(x0) => previewer.maybe_emit(progress, x0, index + 1, total),
                    None => tracing::warn!("skipping denoise preview: solver has no x0 yet"),
                }
            }
        }
    }
    Ok(latents)
}

/// Where the UMT5 encode runs.
///
/// Wan is sequential by construction: the encoder is loaded, used, and gone
/// before the DiT is built, so the two never coexist in VRAM. The scheduler's
/// text-encoder placement is shaped for families where they do, which on a
/// 24 GB card means it parks UMT5 on CPU for every 5B and A14B checkpoint —
/// and a CPU-parked UMT5 has to widen to F32 (candle has no CPU BF16 matmul),
/// costing ~22.7 GB of host RAM and minutes per encode.
///
/// So when the variant resolver has *just measured* that the selected encoder
/// fits in free VRAM right now, that measurement wins over the placement. This
/// is not overriding the scheduler's grant: the grant covers the denoise peak,
/// which is unchanged, and the encoder is released before that peak is
/// reached. Placement still decides when the encoder genuinely does not fit.
fn encode_device(placement: Device, execution: &Device, fits_on_gpu: bool) -> Device {
    match (fits_on_gpu, placement.is_cpu(), execution.is_cpu()) {
        // Placement parked it, but it fits and there is a real device to run
        // it on — promote to the execution device.
        (true, true, false) => execution.clone(),
        // It does not fit: CPU regardless of what placement said.
        (false, _, _) => Device::Cpu,
        _ => placement,
    }
}

pub struct WanEngine {
    base: EngineBase<()>,
    shared_pool: Option<Arc<Mutex<SharedPool>>>,
    pending_placement: Option<mold_core::types::DevicePlacement>,
    /// Explicit UMT5 variant tag, or `None` for the auto policy.
    umt5_variant: Option<String>,
    /// Exact encoder file admission already materialized, when it selected a
    /// GGUF variant. Set, it is authoritative: the download happened before
    /// the lease, so re-running the auto policy here could pick a tier that
    /// is not on disk.
    selected_umt5_path: Option<PathBuf>,
    /// Encoder retained between requests under `MOLD_KEEP_TE_RAM=1`, parked on
    /// host RAM. `None` in the default drop-and-reload mode.
    retained_encoder: Option<WanTextEncoder>,
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
            umt5_variant: None,
            selected_umt5_path: None,
            pending_placement: None,
            retained_encoder: None,
        }
    }

    /// UMT5 weight shards. The manifest ships one fp16 safetensors, but the
    /// multi-shard field is the general contract.
    /// Freeze the explicit UMT5 variant this render was planned with.
    pub fn with_umt5_variant(mut self, variant: Option<String>) -> Self {
        self.umt5_variant = variant;
        self
    }

    /// Freeze the exact encoder file admission materialized for this render.
    pub fn with_selected_umt5_path(mut self, path: Option<PathBuf>) -> Self {
        self.selected_umt5_path = path;
        self
    }

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

    /// Pick the UMT5 weights for this render.
    ///
    /// An explicit `--umt5-variant` / `MOLD_UMT5_VARIANT` wins. Otherwise the
    /// shared resolver prefers FP16 on GPU when it fits and the largest GGUF
    /// that does when it does not — a quantized GPU encode beats parking UMT5
    /// on CPU, where it widens to F32.
    ///
    /// A configured `text_encoder_files` list that is already a single GGUF is
    /// honoured as-is: a hand-pointed encoder is an explicit choice.
    fn resolve_umt5_weights(
        &self,
        progress: &crate::progress::ProgressReporter,
        device: &Device,
    ) -> Result<(Vec<PathBuf>, bool)> {
        let configured = self.text_encoder_paths()?;
        // Admission already chose and downloaded a variant: that file is the
        // route, and only the device is still open. Re-running the auto policy
        // here would re-measure free VRAM against a different resident set and
        // could name a tier that was never fetched.
        if let Some(path) = &self.selected_umt5_path {
            let on_gpu = crate::encoders::variant_resolution::umt5_fits_on_gpu(
                device,
                crate::device::usable_free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0),
                std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0),
            );
            progress.info(&format!(
                "Using prepared UMT5 encoder {} on {}",
                path.display(),
                if on_gpu { "GPU" } else { "CPU" },
            ));
            return Ok((vec![path.clone()], on_gpu));
        }
        let preference = self.umt5_variant.clone();
        let already_gguf = configured
            .iter()
            .all(|path| crate::wan::experts::is_gguf(path));
        if already_gguf || (preference.is_none() && configured.len() != 1) {
            // Nothing to choose between: either the operator already pointed
            // at a quantized file, or the model ships sharded FP16 weights the
            // variant registry does not describe.
            return Ok((configured, true));
        }

        let free_vram = crate::device::usable_free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
        let (path, on_gpu, _label) = crate::encoders::variant_resolution::resolve_umt5_variant(
            progress,
            preference.as_deref(),
            device,
            free_vram,
            configured
                .first()
                .map(PathBuf::as_path)
                .unwrap_or_else(|| Path::new("")),
        )?;
        Ok((vec![path], on_gpu))
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
        let Some(endpoints) = resolve_endpoint_images(req, pixel_frames as u32)? else {
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
                 use wan22-ti2v-5b, or an I2V checkpoint, for source images and keyframes"
            );
        }
        let anchors = if endpoints.last.is_some() {
            WanImageAnchors::FirstAndLastFrame
        } else {
            WanImageAnchors::FirstFrame
        };

        progress.stage_start("Encoding source image");
        let encode_start = Instant::now();
        // Fit to the requested frame, matching every other mold engine's source
        // convention. mold deliberately does NOT run upstream's area bucketing,
        // which would silently resize; see `wan::conditioning`.
        let decode = |bytes: &[u8]| {
            crate::img_utils::decode_source_image(
                bytes,
                width,
                height,
                crate::img_utils::NormalizeRange::MinusOneToOne,
                device,
                dtype,
            )
        };
        let image = decode(endpoints.first)?;
        let last_image = endpoints.last.map(decode).transpose()?;

        let vae = WanVideoVae::from_safetensors(
            &self.base.paths.vae,
            vae_generation.config(),
            device,
            dtype,
        )?;

        let conditioning = match shape {
            WanConditioningShape::Plain => {
                // TI2V encodes the bare image: one pixel frame in, one latent
                // frame out, broadcast across the clip by the blend. FLF
                // additionally pins the final latent frame to the encoded
                // last image — diffusers' `last_image` shape (#779).
                let single = image.unsqueeze(2)?;
                let encoded = vae.encode(&single)?;
                let mut condition = encoded
                    .broadcast_as((
                        1,
                        encoded.dim(1)?,
                        geometry.latent_frames,
                        geometry.latent_height,
                        geometry.latent_width,
                    ))?
                    .contiguous()?;
                if let Some(last_image) = &last_image {
                    let last_encoded = vae.encode(&last_image.unsqueeze(2)?)?;
                    let head = condition.narrow(2, 0, geometry.latent_frames - 1)?;
                    condition = Tensor::cat(&[&head, &last_encoded], 2)?.contiguous()?;
                }
                let inpaint = WanTi2vInpaint::with_anchors(geometry, anchors, device, dtype)?;
                WanImageConditioning::LatentInpaint { inpaint, condition }
            }
            WanConditioningShape::ChannelConcat => {
                // I2V encodes the endpoints with a black canvas between —
                // upstream FLF2V's `y = [first, zeros(F-2), last]`
                // (`first_last_frame2video.py:186-277`); plain I2V is the
                // degenerate one-anchor case.
                let gap = pixel_frames - 1 - usize::from(last_image.is_some());
                let mut segments = vec![image.unsqueeze(2)?];
                if gap > 0 {
                    segments.push(Tensor::zeros(
                        (1, 3, gap, height as usize, width as usize),
                        dtype,
                        device,
                    )?);
                }
                if let Some(last_image) = &last_image {
                    segments.push(last_image.unsqueeze(2)?);
                }
                let refs: Vec<&Tensor> = segments.iter().collect();
                let video = Tensor::cat(&refs, 2)?;
                let encoded = vae.encode(&video)?;
                WanImageConditioning::ChannelConcat {
                    conditioning: build_a14b_conditioning(
                        &encoded,
                        pixel_frames,
                        anchors,
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
        // Per-expert distill strengths (#795): the community's reduced-motion
        // mitigation runs the high-noise adapter above 1.0. A strength on a
        // model that ships no distill in that slot is refused, not ignored —
        // a silently inert knob looks like the mitigation failing.
        let distill_high = resolve_distill_strength("high", req.distill_strength_high)?;
        let distill_low = resolve_distill_strength("low", req.distill_strength_low)?;
        if req.distill_strength_high.is_some() && paths.distilled_lora.is_none() {
            bail!(
                "distill_strength_high was set, but {} ships no distill adapter (the quality \
                 tier is undistilled)",
                self.base.model_name
            );
        }
        if req.distill_strength_low.is_some()
            && (low_noise_expert.is_none() || paths.low_noise_distilled_lora.is_none())
        {
            bail!(
                "distill_strength_low was set, but {} has no low-noise distill slot",
                self.base.model_name
            );
        }
        if distill_high != 1.0 || distill_low != 1.0 {
            progress.info(&format!(
                "Lightning distill strength: high {distill_high:.2}, low {distill_low:.2}"
            ));
        }
        // Which expert each user adapter belongs to, and whether that was
        // stated or inferred. Resolved once so the disclosure below describes
        // the same decision the registries are built from.
        let routing = resolve_user_lora_experts(&user_loras);
        for note in routing.disclosures() {
            progress.info(&note);
        }

        let stack = |distill: Option<&Path>,
                     distill_scale: f64,
                     slot: Option<mold_core::LoraExpert>|
         -> Result<WanLoraRegistry> {
            let mut weights: Vec<mold_core::LoraWeight> = Vec::new();
            if let Some(path) = distill {
                weights.push(mold_core::LoraWeight {
                    path: path.to_string_lossy().to_string(),
                    scale: distill_scale,
                    expert: None,
                });
            }
            // A pair's halves are distilled together and are not
            // interchangeable: putting the high-noise adapter on the low-noise
            // expert degrades the render rather than failing, which is why
            // this is a correctness concern and not a nicety. An adapter with
            // no expert affinity still applies to both.
            weights.extend(
                routing
                    .entries
                    .iter()
                    .filter(|entry| match (slot, entry.expert) {
                        (Some(slot), Some(bound)) => slot == bound,
                        _ => true,
                    })
                    .map(|entry| entry.weight.clone()),
            );
            WanLoraRegistry::load(&weights)
        };

        let Some(low_noise_path) = low_noise_expert else {
            let loras = stack(paths.distilled_lora.as_deref(), distill_high, None)?;
            if !loras.is_empty() {
                progress.info(&format!(
                    "Applying {} LoRA patch(es) across {} tensors",
                    loras.patch_count(),
                    loras.tensor_count()
                ));
            }
            progress.stage_start("Loading Wan transformer");
            let started = Instant::now();
            let files = transformer_files(paths);
            let transformer = crate::wan::experts::load_transformer_with_offload(
                &files,
                config.clone(),
                device,
                dtype,
                &loras,
                denoise_activation_bytes(req, &files, config),
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
                loras: stack(
                    paths.distilled_lora.as_deref(),
                    distill_high,
                    Some(mold_core::LoraExpert::High),
                )?,
            },
            low_noise: WanExpertSlot {
                path: low_noise_path.to_path_buf(),
                loras: stack(
                    paths.low_noise_distilled_lora.as_deref(),
                    distill_low,
                    Some(mold_core::LoraExpert::Low),
                )?,
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
        WanExperts::pair(
            pair,
            config.clone(),
            device,
            dtype,
            low_noise_config,
            denoise_activation_bytes(req, &transformer_files(paths), config),
        )
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
        let dit_files = transformer_files(paths);
        let transformer_config = detect_transformer_config_across(&dit_files)?;
        let shape = conditioning_shape(transformer_config.in_dim, vae_config.z_dim)?;
        let low_noise_expert = paths.low_noise_transformer.as_deref();
        // Wan 2.1 I2V is refused outright — it needs a CLIP-vision branch the
        // transformer omits. Wan 2.2 I2V-A14B runs, but only with both experts.
        // The refusal reads the same shard set the detection above does: the
        // CLIP adapter is not guaranteed to live in the primary file.
        if shape == WanConditioningShape::ChannelConcat {
            reject_unwired_channel_concat_checkpoint(&dit_files, low_noise_expert)?;
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
        let shift = resolve_flow_shift(req.sample_shift, low_noise_expert.is_some())?;
        let solver_kind = resolve_wan_solver(req.scheduler, low_noise_expert.is_some())?;
        let channel_concat = shape == WanConditioningShape::ChannelConcat;
        // The model's own advertised default decides whether the request's
        // scale means "I did not choose" — a community pair without a
        // manifest stays on uniform guidance.
        let model_default_guidance = mold_core::manifest::find_manifest(&self.base.model_name)
            .map(|manifest| manifest.defaults.guidance);
        let guidance_plan = resolve_guidance_plan(
            guidance,
            low_noise_expert.map(|_| WanExpertPair::boundary_for(channel_concat)),
            channel_concat,
            model_default_guidance,
        );
        let needs_cfg = needs_cfg_pass(guidance_plan.max());

        let device = crate::device::create_device(self.base.gpu_ordinal, progress)?;
        let dtype = super::backend::compute_dtype(&device);

        let guidance_label = match guidance_plan {
            WanGuidancePlan::Uniform(scale) => format!("{scale:.1}"),
            WanGuidancePlan::PerExpert { guidance, .. } => format!(
                "{:.1} (high-noise) / {:.1} (low-noise)",
                guidance.high_noise, guidance.low_noise
            ),
        };
        progress.info(&format!(
            "Wan: {width}x{height} x {num_frames} frames @ {fps} fps, {steps} steps, \
             guidance {guidance_label}, shift {shift:.1}, solver {}, seed {seed}",
            solver_kind.label()
        ));
        if let WanGuidancePlan::PerExpert { guidance, .. } = guidance_plan {
            progress.info(&format!(
                "Using upstream per-expert guidance ({:.1} while the high-noise expert runs, \
                 {:.1} after the boundary); pass an explicit --guidance to pin one scale",
                guidance.high_noise, guidance.low_noise
            ));
        }
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
        let placement_device = crate::device::resolve_device(
            Some(
                self.pending_placement
                    .as_ref()
                    .map(|placement| placement.text_encoders.clone())
                    .unwrap_or_default(),
            ),
            || Ok(device.clone()),
        )?;
        // Which UMT5 weights to read. The GGUF loader has existed and been
        // tested since the family landed; what was missing was a way to select
        // one, so every wan render read the 11.4 GB FP16 file.
        let (encoder_paths, encoder_on_gpu) = self.resolve_umt5_weights(progress, &device)?;
        let text_device = encode_device(placement_device, &device, encoder_on_gpu);
        let encoder_dtype = encoder_dtype_for(&text_device, dtype);
        // Reuse the parked encoder when this render was planned for the same
        // weights, device, and dtype; otherwise the retained one is stale and
        // is dropped before the fresh load so both are never resident.
        let retained = self.retained_encoder.take();
        let mut encoder = match retained {
            Some(retained) if retained.matches(&encoder_paths, &text_device, encoder_dtype) => {
                retained
            }
            stale => {
                // Explicit: a `_` arm would keep the stale encoder's ~11.4 GB
                // alive until the end of the match, i.e. across the fresh load.
                drop(stale);
                WanTextEncoder::load_with_tokenizer(
                    &encoder_paths,
                    &text_device,
                    encoder_dtype,
                    tokenizer,
                )?
            }
        };
        encoder.unpark()?;
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

        // The encoder is 11.4 GB at fp16; it must be gone from the compute
        // device before the DiT loads. Under `MOLD_KEEP_TE_RAM=1` it moves to
        // host RAM instead of vanishing, so the next request skips the disk
        // read; the VRAM is released either way.
        // Metal is unified memory: "parking on CPU" copies within the same
        // physical RAM, so every other family skips it there and so does wan.
        if crate::device::keep_te_in_ram() && !text_device.is_metal() {
            encoder.park_to_cpu()?;
            let parked = encoder.is_parked();
            self.retained_encoder = Some(encoder);
            device.synchronize()?;
            progress.info(if parked {
                "UMT5 encoder parked on host RAM, VRAM freed"
            } else {
                // GGUF: device-tied storage parks by dropping, so the next
                // request reloads — cheaper than the FP16 path's read anyway.
                "UMT5 encoder dropped, VRAM freed"
            });
        } else {
            encoder.drop_weights();
            drop(encoder);
            device.synchronize()?;
            progress.info("UMT5 encoder dropped, VRAM freed");
        }

        // ------------------------------------------------------------------
        // 2. Denoise
        // ------------------------------------------------------------------
        // The solver owns its grid: dpm++ lays sigmas out differently from
        // the diffusers/Lightning grid UniPC and euler share (#795), so the
        // schedule must come FROM the solver, never be built beside it.
        let mut solver =
            build_wan_solver(solver_kind, WanScheduleConfig::new(steps as usize, shift))?;
        let schedule = solver.schedule().clone();
        let mut experts = self.resolve_experts(
            req,
            &transformer_config,
            shape,
            low_noise_expert,
            &device,
            dtype,
            progress,
        )?;
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
        // Live previews project the working latent through the checkpoint
        // generation's own factor table, selected by latent channel count.
        let previewer = crate::latent_preview::LatentPreviewer::wan(vae_config.z_dim);
        // First-block residual reuse (#801). Off unless asked for, and refused
        // outright on the schedules where it cannot help -- a distill adapter
        // active, or too few steps to have redundant ones. A refusal is
        // disclosed: a silently ignored knob reads as "the feature does not
        // work".
        // A shipped distill adapter is the signal: the `:q5` fast tiers carry
        // one, the quality tiers do not. That is exactly the split where
        // residual reuse has something to reuse.
        let distill_is_active = self.base.paths.distilled_lora.is_some()
            || self.base.paths.low_noise_distilled_lora.is_some();
        let (step_cache, refusal) = crate::wan::step_cache::WanStepCachePolicy::resolve(
            crate::wan::step_cache::requested_threshold()?,
            steps,
            distill_is_active,
        );
        if let Some(refusal) = refusal {
            progress.info(refusal.message());
        }
        if let crate::wan::step_cache::WanStepCachePolicy::Threshold(threshold) = step_cache {
            progress.info(&format!(
                "Step cache on at relative-L1 threshold {threshold}"
            ));
        }
        let latents = run_denoise_loop(DenoiseInputs {
            experts: &mut experts,
            conditioning: &conditioning,
            schedule: &schedule,
            solver: &mut solver,
            latents,
            cond_embeds: &cond_embeds,
            uncond_embeds: uncond_embeds.as_ref(),
            guidance: guidance_plan,
            patch: transformer_config.patch_size.1,
            rope: &rope,
            device: &device,
            progress,
            previewer: previewer.as_ref(),
            step_cache,
        })?;
        progress.checkpoint()?;
        drop(experts);
        device.synchronize()?;

        // ------------------------------------------------------------------
        // 3. VAE decode
        // ------------------------------------------------------------------
        progress.checkpoint()?;
        progress.stage_start("Loading Wan VAE");
        let vae_start = Instant::now();
        let vae = WanVideoVae::from_safetensors(&paths.vae, vae_config, &device, dtype)?;
        progress.phase_done(
            ProgressPhase::ModelLoad,
            "Loading Wan VAE",
            vae_start.elapsed(),
        );
        progress.checkpoint()?;

        progress.stage_start("Decoding video frames");
        let decode_start = Instant::now();
        let video = vae.decode(&latents)?;
        progress.checkpoint()?;
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
        progress.checkpoint()?;
        let frames = video_frames_to_images(&video, width, height)?;

        // A single-frame render with an image format is a still (#798):
        // it takes the standard still pipeline — embedded `mold:parameters`,
        // a gallery row, upscale eligibility — not a one-frame video.
        // Admission only admits png/jpeg at frames == 1; the length check
        // covers forced-local callers that skipped validation.
        let resolved_format = req.resolved_output_format();
        if matches!(resolved_format, OutputFormat::Png | OutputFormat::Jpeg) {
            if frames.len() != 1 {
                bail!(
                    "Wan renders a {} still only at frames=1, got {} frames",
                    resolved_format.extension(),
                    frames.len()
                );
            }
            let response = still_response(
                req,
                &self.base.model_name,
                seed,
                &frames[0],
                width,
                height,
                start,
            )?;
            progress.info(&format!(
                "Done: 1 frame ({} still), {:.1}s total",
                resolved_format.extension(),
                response.generation_time_ms as f64 / 1000.0
            ));
            return Ok(response);
        }

        let output_format = if resolved_format.is_video() {
            resolved_format
        } else {
            OutputFormat::Apng
        };
        let format_name = output_format.extension().to_uppercase();
        progress.stage_start(&format!("Encoding {format_name}"));
        let encode_start = Instant::now();

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
                pipeline_provenance_sha256: None,
                source_preprocessing: None,
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

/// Build the still artifact for a single-frame Wan render (#798).
///
/// The frame goes through the same encoder every image family uses —
/// `encode_rgb_image` with `build_output_metadata` — so the PNG/JPEG carries
/// the embedded `mold:parameters` block, the gallery records a still row, and
/// post-generation upscale sees an ordinary image response.
fn still_response(
    req: &GenerateRequest,
    model_name: &str,
    seed: u64,
    frame: &image::RgbImage,
    width: u32,
    height: u32,
    start: Instant,
) -> Result<GenerateResponse> {
    let format = req.resolved_output_format();
    let metadata = crate::image::build_output_metadata(req, seed, None);
    let data = crate::image::encode_rgb_image(frame, format, metadata.as_ref())?;
    Ok(GenerateResponse {
        audio: None,
        images: vec![ImageData {
            data,
            format,
            width,
            height,
            index: 0,
        }],
        video: None,
        generation_time_ms: start.elapsed().as_millis() as u64,
        model: model_name.to_string(),
        seed_used: seed,
        gpu: None,
    })
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
        let result = if req.is_extend() {
            self.extend_inner(req)
        } else {
            self.generate_inner(req)
        };
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
        // A parked encoder is ~11.4 GB of host RAM held by this engine. Unload
        // means "give the resources back", so the retention opt-in does not
        // survive it — the cache evicting this engine, or an explicit unload,
        // releases it.
        self.retained_encoder = None;
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

    fn as_chain_renderer(&mut self) -> Option<&mut dyn crate::chain::ChainStageRenderer> {
        Some(self)
    }
}

/// The pixel frames a wan continuation duplicates from the clip before it.
///
/// Re-exported from `mold-core`, which is where every surface that must agree
/// with this engine can reach it — admission, `/api/models`, and the CLI's
/// chain planner all derive their overlap from the same constant instead of
/// restating 1 (#783).
pub use mold_core::validation::WAN_HANDOFF_DUPLICATED_FRAMES;

/// The overlap [`WanEngine::extend_inner`] will run with, or the error it
/// refuses the request with.
///
/// Extracted from `extend_inner` so the acceptance itself is testable without
/// a checkpoint: this is the gate, not a restatement of it.
fn accepted_extend_overlap(req: &GenerateRequest) -> Result<u32> {
    // Name the family: an installed `cv:` / `hf:` wan checkpoint has no
    // manifest for the request to resolve itself through, and the
    // family-blind fallback is LTX-2's 17 — which this gate refuses.
    let overlap = req.effective_extend_overlap_frames_for_family(Some("wan"));
    if overlap != WAN_HANDOFF_DUPLICATED_FRAMES {
        bail!(
            "Wan continuations carry exactly {WAN_HANDOFF_DUPLICATED_FRAMES} frame of \
             context — the source's final frame becomes the continuation's conditioning — \
             so extend_overlap_frames must be {WAN_HANDOFF_DUPLICATED_FRAMES}, not {overlap}. \
             (LTX-2's multi-frame overlap is a latent motion tail wan does not have.)"
        );
    }
    Ok(overlap)
}

impl WanEngine {
    /// Continue an existing clip in one request.
    ///
    /// This is the chain seam with the carryover coming from a file rather
    /// than from the previous stage: the source's final frame becomes the
    /// continuation's image conditioning, the continuation re-renders exactly
    /// that frame, and the stitch drops it before appending. `overlap` is
    /// therefore always [`WAN_HANDOFF_DUPLICATED_FRAMES`] in effect — a larger
    /// value is accepted by validation for grid symmetry with LTX-2 but wan
    /// has only ever one frame of real carryover, so anything above one is
    /// refused here rather than silently trimming good frames.
    ///
    /// Resolution and fps are locked to the source: the stitched output is one
    /// video, and rescaling or retiming mid-clip is always a surprise.
    fn extend_inner(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let overlap = accepted_extend_overlap(req)?;

        let work_dir = tempfile::tempdir().context("Wan: creating the continuation temp dir")?;
        let path = match (&req.extend_video, &req.extend_video_path) {
            (Some(bytes), _) => {
                let staged = work_dir.path().join("extend-source.mp4");
                std::fs::write(&staged, bytes).context("Wan: staging the video to extend")?;
                staged
            }
            (None, Some(path)) => PathBuf::from(path),
            (None, None) => {
                bail!("Wan: extend requested without extend_video or extend_video_path")
            }
        };
        let (probe, source_frames) = crate::ltx2::media::decode_video_frames_from_path(&path)
            .with_context(|| format!("Wan: decoding the video to extend ({})", path.display()))?;
        let Some(last) = source_frames.last() else {
            bail!(
                "Wan: the video to extend ({}) decoded to zero frames",
                path.display()
            );
        };

        // Reject rather than rescale: the stitched result is one video.
        if req.width != probe.width || req.height != probe.height {
            bail!(
                "the video to extend is {}x{} but this request renders {width}x{height}; \
                 continuations must render at the source's resolution",
                probe.width,
                probe.height,
                width = req.width,
                height = req.height,
            );
        }
        let fps = req.fps.unwrap_or(probe.fps);
        if fps != probe.fps {
            bail!(
                "the video to extend runs at {} fps but this request renders {fps} fps; \
                 continuations must render at the source's frame rate (pass --fps {} to match)",
                probe.fps,
                probe.fps,
            );
        }

        let mut png = std::io::Cursor::new(Vec::new());
        last.write_to(&mut png, image::ImageFormat::Png)
            .context("Wan: encoding the continuation's seed frame")?;

        let mut continuation_req = req.clone();
        continuation_req.extend_video = None;
        continuation_req.extend_video_path = None;
        continuation_req.extend_overlap_frames = None;
        continuation_req.source_image = Some(png.into_inner());
        continuation_req.width = probe.width;
        continuation_req.height = probe.height;
        continuation_req.fps = Some(probe.fps);
        continuation_req.output_format = Some(mold_core::OutputFormat::Apng);
        continuation_req.gif_preview = false;

        let response = self.generate_inner(&continuation_req)?;
        let video = response
            .video
            .ok_or_else(|| anyhow::anyhow!("Wan: the continuation returned no video data"))?;
        let continuation = crate::chain::decode_apng_to_rgb_frames(&video.data)?;
        let stitched = crate::ltx2::stitch_extend_frames(source_frames, &continuation, overlap)?;

        let requested_format = req.output_format.unwrap_or(mold_core::OutputFormat::Mp4);
        let encoded =
            crate::chain::encode_chain_frames(&stitched, probe.fps, requested_format, None)?;
        for warning in &encoded.warnings {
            self.base.progress.info(&warning.message());
        }
        Ok(GenerateResponse {
            video: Some(mold_core::VideoData {
                data: encoded.bytes,
                format: encoded.format,
                frames: stitched.len() as u32,
                fps: probe.fps,
                gif_preview: encoded.gif_preview,
                ..video
            }),
            ..response
        })
    }

    /// Seed a continuation stage from the previous clip's last frame.
    ///
    /// Returns `false` when the carryover cannot be applied, which is not an
    /// error: a text-to-video checkpoint has no conditioning channel, so its
    /// stages render independently and the stitch concatenates them.
    ///
    /// A stage that already carries its own source image keeps it — an
    /// authored sequence may pin a specific still for a clip, and silently
    /// replacing it with the previous clip's tail would discard the user's
    /// input.
    fn seed_stage_from_carry(
        &self,
        stage_req: &mut GenerateRequest,
        carry: Option<&crate::chain::ChainTail>,
    ) -> Result<bool> {
        let Some(carry) = carry else { return Ok(false) };
        if stage_req.source_image.is_some() {
            return Ok(false);
        }
        let Some(last) = carry.tail_rgb_frames.last() else {
            return Ok(false);
        };
        // Ask the checkpoint, not the model name: this is the same
        // classification `/api/models` advertises and the chain capability
        // derives its carryover from, so the three cannot disagree.
        let conditions_on_image = matches!(
            source_image_capability(&self.base.paths.transformer, &self.base.paths.vae),
            Some(mold_core::SourceImageCapability::Required)
                | Some(mold_core::SourceImageCapability::Optional)
        );
        if !conditions_on_image {
            return Ok(false);
        }
        let mut png = std::io::Cursor::new(Vec::new());
        last.write_to(&mut png, image::ImageFormat::Png)
            .context("Wan: encoding the chain carryover frame as PNG")?;
        stage_req.source_image = Some(png.into_inner());
        Ok(true)
    }
}

impl crate::chain::ChainStageRenderer for WanEngine {
    fn render_stage(
        &mut self,
        stage_req: &GenerateRequest,
        carry: Option<&crate::chain::ChainTail>,
        motion_tail_pixel_frames: u32,
        hdr_sidecar: Option<&crate::chain::StageSidecar>,
        _stage_progress: Option<&mut dyn FnMut(crate::chain::StageProgressEvent)>,
    ) -> Result<crate::chain::StageOutcome> {
        if hdr_sidecar.is_some() {
            bail!(
                "WanEngine.render_stage: the HDR EXR sidecar is an LTX-2 feature; \
                 wan stages cannot honour it"
            );
        }
        let mut stage_req = stage_req.clone();
        self.seed_stage_from_carry(&mut stage_req, carry)?;

        let start = Instant::now();
        // APNG is lossless, so the encode/decode round-trip preserves every
        // pixel; it costs tens of milliseconds against a multi-second denoise.
        // Same approach as the ltx-video stage renderer.
        stage_req.output_format = Some(mold_core::OutputFormat::Apng);
        stage_req.gif_preview = false;
        let response = self.generate_inner(&stage_req)?;
        let generation_time_ms = start.elapsed().as_millis() as u64;
        let video = response
            .video
            .ok_or_else(|| anyhow::anyhow!("WanEngine.render_stage: no video data"))?;
        let frames = crate::chain::decode_apng_to_rgb_frames(&video.data)?;
        if frames.is_empty() {
            bail!("WanEngine.render_stage: pipeline produced zero frames");
        }

        // Hand back the trailing frames so the next stage can be seeded. The
        // orchestrator threads this through whether or not the carryover is
        // used, and `ChainTail` consumers require a non-empty window.
        let tail_count = (motion_tail_pixel_frames as usize).clamp(1, frames.len());
        let tail_frames: Vec<image::RgbImage> = frames
            .iter()
            .skip(frames.len() - tail_count)
            .cloned()
            .collect();

        Ok(crate::chain::StageOutcome {
            frames,
            tail: crate::chain::ChainTail {
                frames: tail_frames.len() as u32,
                tail_rgb_frames: tail_frames,
            },
            // The family has no audio decode path; chain validation rejects
            // `enable_audio: true` for wan upstream of here.
            audio: None,
            hdr_frames_written: None,
            generation_time_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::InferenceEngine;
    use candle_nn::{VarBuilder, VarMap};
    use std::collections::HashMap;

    fn user_lora(path: &str, expert: Option<mold_core::LoraExpert>) -> mold_core::LoraWeight {
        mold_core::LoraWeight {
            path: path.to_string(),
            scale: 1.0,
            expert,
        }
    }

    /// The engine must actually be a chain renderer.
    ///
    /// This is the defect the capability arm shipped without: `/api/models`
    /// advertised `supports_sequence` for wan while `as_chain_renderer()`
    /// returned the trait's default `None`, so every wan sequence would have
    /// failed at worker dispatch — an advertised capability the engine could
    /// not execute.
    #[test]
    fn the_wan_engine_is_registered_as_a_chain_renderer() {
        let mut engine = WanEngine::new(
            "wan22-ti2v-5b:fp16".to_string(),
            dummy_paths(),
            LoadStrategy::default(),
            0,
            None,
        );
        assert!(
            crate::engine::InferenceEngine::as_chain_renderer(&mut engine).is_some(),
            "wan advertises sequence support, so it must provide a stage renderer",
        );
    }

    /// The seam duplicates one frame, not LTX-2's seventeen.
    ///
    /// LTX-2's 17 is the pixel window its VAE turns into three latent slots of
    /// carryover. Wan has no latent motion tail: its continuation is seeded
    /// with the previous clip's final frame and re-renders exactly that one
    /// frame. Carrying 17 over would discard sixteen good frames at every
    /// boundary while looking entirely plausible.
    #[test]
    fn the_wan_handoff_duplicates_exactly_the_seeded_frame() {
        assert_eq!(WAN_HANDOFF_DUPLICATED_FRAMES, 1);
        assert_ne!(
            WAN_HANDOFF_DUPLICATED_FRAMES,
            crate::chain::capability::LTX_VIDEO_FRAMES_PER_CLIP_CAP,
        );
        // It must stay strictly below any clip length wan can route to, or a
        // continuation would emit no new frames at all.
        for clip in [5u32, 53, 121, 257] {
            assert!(WAN_HANDOFF_DUPLICATED_FRAMES < clip, "clip {clip}");
        }
    }

    /// `extend_inner` must accept a continuation that names no overlap.
    ///
    /// This drives the engine's own gate, not the core helper it consults:
    /// the advertised default used to be LTX-2's 17 for every family, 17 sits
    /// on wan's `4k+1` grid so validation waved it through, and the request
    /// then died right here — after the model load had been paid for (#783).
    #[test]
    fn the_engine_accepts_an_unset_overlap_and_refuses_ltx2s() {
        let continuation = || {
            let mut req = request();
            // An installed catalog id, so the request cannot resolve its own
            // family through the manifest — the case that made the
            // family-blind fallback fatal.
            req.model = "cv:2041121".to_string();
            req.frames = Some(49);
            req.extend_video_path = Some("/srv/mold/clip.mp4".to_string());
            req
        };

        let unset = continuation();
        assert!(unset.extend_overlap_frames.is_none());
        assert_eq!(
            accepted_extend_overlap(&unset).unwrap(),
            WAN_HANDOFF_DUPLICATED_FRAMES
        );

        let mut ltx2_overlap = continuation();
        ltx2_overlap.extend_overlap_frames =
            Some(mold_core::validation::DEFAULT_EXTEND_OVERLAP_FRAMES);
        let error = accepted_extend_overlap(&ltx2_overlap)
            .unwrap_err()
            .to_string();
        assert!(error.contains("must be 1, not 17"), "got: {error}");
    }

    /// A CPU placement is not the last word when the encoder measurably fits.
    ///
    /// Placement is shaped for families where the encoder and the transformer
    /// coexist. Wan releases the encoder before the DiT is built, so honouring
    /// a CPU park it does not need costs an F32 widening (~22.7 GB host RAM)
    /// and a CPU encode of a 5.7B-parameter model. The measurement the variant
    /// resolver just took is the better authority — but only in that one
    /// direction: it must never move an encoder onto a device it does not fit.
    #[test]
    fn a_fitting_encoder_beats_a_cpu_placement_but_never_the_reverse() {
        let cuda = Device::new_cuda(0).ok();
        let execution = cuda.clone().unwrap_or(Device::Cpu);

        // Fits, placement parked it on CPU: promoted to the execution device.
        let promoted = encode_device(Device::Cpu, &execution, true);
        assert_eq!(promoted.is_cpu(), execution.is_cpu());

        // Does not fit: CPU, whatever placement said.
        assert!(encode_device(execution.clone(), &execution, false).is_cpu());
        assert!(encode_device(Device::Cpu, &execution, false).is_cpu());

        // A CPU-only host has nowhere to promote to; the fit flag must not
        // conjure a device.
        assert!(encode_device(Device::Cpu, &Device::Cpu, true).is_cpu());

        // Placement already chose the device and it fits: left alone.
        assert_eq!(
            encode_device(execution.clone(), &execution, true).is_cpu(),
            execution.is_cpu()
        );
    }

    /// The defect this exists for: a HIGH/LOW pair applied to both experts.
    ///
    /// lightx2v distills the halves together and states they are not
    /// interchangeable, so landing both on both experts degrades the render
    /// while succeeding — the same bug the reference HF space's loader had.
    #[test]
    fn a_paired_adapter_reaches_only_its_own_expert() {
        let routing = resolve_user_lora_experts(&[
            user_lora("/loras/high_noise_model.safetensors", None),
            user_lora("/loras/low_noise_model.safetensors", None),
        ]);
        let experts: Vec<_> = routing.entries.iter().map(|entry| entry.expert).collect();
        assert_eq!(
            experts,
            vec![
                Some(mold_core::LoraExpert::High),
                Some(mold_core::LoraExpert::Low)
            ]
        );

        // Inference is disclosed, never silent — the alternative reading
        // ("it applied to both") produces a different render.
        let notes = routing.disclosures();
        assert_eq!(notes.len(), 2, "{notes:?}");
        assert!(notes[0].contains("high-noise"), "{notes:?}");
        assert!(notes[1].contains("low-noise"), "{notes:?}");
    }

    #[test]
    fn an_explicit_expert_beats_the_filename_and_is_not_disclosed() {
        // The file says low, the request says high: the request wins, and
        // nothing is disclosed because nothing was guessed.
        let routing = resolve_user_lora_experts(&[user_lora(
            "/loras/low_noise_model.safetensors",
            Some(mold_core::LoraExpert::High),
        )]);
        assert_eq!(routing.entries[0].expert, Some(mold_core::LoraExpert::High));
        assert!(routing.disclosures().is_empty());
    }

    #[test]
    fn an_unpaired_adapter_keeps_applying_to_both_experts() {
        let routing = resolve_user_lora_experts(&[user_lora("/loras/style.safetensors", None)]);
        assert_eq!(routing.entries[0].expert, None);
        assert!(routing.disclosures().is_empty());
    }

    /// Substring matching would read "highlight" and "slow" as markers.
    #[test]
    fn expert_inference_matches_tokens_not_substrings() {
        for name in [
            "/loras/highlight-boost.safetensors",
            "/loras/slow-motion.safetensors",
            "/loras/shallow.safetensors",
        ] {
            assert_eq!(
                mold_core::LoraExpert::infer_from_filename(name),
                None,
                "{name} must not be read as an expert marker"
            );
        }
        for (name, expected) in [
            (
                "/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
                mold_core::LoraExpert::High,
            ),
            ("/loras/HighNoise.safetensors", mold_core::LoraExpert::High),
            ("/loras/model-LOW.safetensors", mold_core::LoraExpert::Low),
        ] {
            assert_eq!(
                mold_core::LoraExpert::infer_from_filename(name),
                Some(expected),
                "{name}"
            );
        }
        // Both markers present is a merged or ambiguous publication: guessing
        // either half is worse than not guessing.
        assert_eq!(
            mold_core::LoraExpert::infer_from_filename("/loras/high_and_low_merged.safetensors"),
            None
        );

        // The published digit-glued convention, which the catalog already had
        // to handle for Civitai pairing (#784). Sharing that classifier rather
        // than writing a second one is what makes this work — a token-only
        // matcher reads `A14BHIGH` as one non-marker word.
        assert_eq!(
            mold_core::LoraExpert::infer_from_filename("/loras/wan2_2_t2vA14BHIGH.safetensors"),
            Some(mold_core::LoraExpert::High)
        );
        assert_eq!(
            mold_core::LoraExpert::infer_from_filename("/loras/wan2_2_t2vA14BLOW.safetensors"),
            Some(mold_core::LoraExpert::Low)
        );
        // "SLOW" is never digit-preceded, so the glued rule cannot claim it.
        assert_eq!(
            mold_core::LoraExpert::infer_from_filename("/loras/SLOW-motion.safetensors"),
            None
        );
        // Consecutive capitals still split correctly.
        assert_eq!(
            mold_core::LoraExpert::infer_from_filename("/loras/HIGHNoise.safetensors"),
            Some(mold_core::LoraExpert::High)
        );
    }

    /// A directory named after one expert must not route an adapter — only
    /// the file's own name carries the marker.
    #[test]
    fn expert_inference_ignores_the_directory_path() {
        assert_eq!(
            mold_core::LoraExpert::infer_from_filename("/loras/high_noise/style.safetensors"),
            None
        );
    }

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
            source_fit: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
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

    /// #798: a frames=1 render with an image format takes the standard still
    /// pipeline — an image response (never a one-frame video) whose PNG
    /// carries the embedded `mold:parameters` provenance the gallery and
    /// upscaler read.
    #[test]
    fn single_frame_still_takes_the_image_pipeline() {
        let mut req = request();
        req.frames = Some(1);
        req.output_format = Some(OutputFormat::Png);
        let frame = image::RgbImage::from_pixel(8, 8, image::Rgb([200u8, 40, 40]));

        let response =
            still_response(&req, "wan22-t2v-a14b:q5", 7, &frame, 8, 8, Instant::now()).unwrap();

        assert!(response.video.is_none(), "a still is not a one-frame video");
        assert!(response.audio.is_none());
        assert_eq!(response.images.len(), 1);
        let image = &response.images[0];
        assert_eq!(image.format, OutputFormat::Png);
        assert_eq!((image.width, image.height), (8, 8));
        assert!(image.data.starts_with(&[0x89, 0x50, 0x4E, 0x47]));

        // The embedded provenance is the same contract every image family
        // writes: a `mold:parameters` JSON block carrying the request.
        let decoder = png::Decoder::new(std::io::Cursor::new(image.data.clone()));
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        reader.next_frame(&mut buf).unwrap();
        let info = reader.info();
        assert!(
            info.utf8_text.iter().any(|chunk| {
                chunk.keyword == "mold:parameters"
                    && chunk.get_text().unwrap().contains("\"prompt\":\"a cat\"")
            }),
            "the still must embed mold:parameters"
        );

        // JPEG output is the other admitted still format.
        req.output_format = Some(OutputFormat::Jpeg);
        let jpeg =
            still_response(&req, "wan22-t2v-a14b:q5", 7, &frame, 8, 8, Instant::now()).unwrap();
        assert!(jpeg.images[0].data.starts_with(&[0xFF, 0xD8]));

        // Respecting the metadata opt-out keeps parity with image families.
        req.output_format = Some(OutputFormat::Png);
        req.embed_metadata = Some(false);
        let bare =
            still_response(&req, "wan22-t2v-a14b:q5", 7, &frame, 8, 8, Instant::now()).unwrap();
        let decoder = png::Decoder::new(std::io::Cursor::new(bare.images[0].data.clone()));
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        reader.next_frame(&mut buf).unwrap();
        assert!(reader.info().utf8_text.is_empty(), "opt-out must hold");
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

    /// `extend_video` is deliberately absent from this list now (#783): it
    /// routes to `extend_inner` before the guard, so rejecting it here would
    /// refuse a continuation the engine can render.
    #[test]
    fn video_conditioning_is_rejected_but_images_are_accepted() {
        for mutate in [
            (|req: &mut GenerateRequest| req.source_video = Some(vec![1, 2, 3])) as fn(&mut _),
            |req: &mut GenerateRequest| req.source_video_path = Some("clip.mp4".into()),
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
                error.contains("first/last-frame conditioning"),
                "the error must say what the engine does support: {error}"
            );
        }

        // Images are no longer refused at the request boundary, and neither
        // are keyframes — the endpoint layout is validated in
        // `resolve_endpoint_images` (#779).
        let mut req = request();
        req.source_image = Some(vec![1, 2, 3]);
        req.source_image_name = Some("cat.png".into());
        reject_unsupported_conditioning(&req).unwrap();
        reject_unsupported_conditioning(&request()).unwrap();
    }

    /// #779: the endpoint resolver accepts exactly the layouts wan renders —
    /// a lone source image, or a two-entry first/last keyframe pair anchored
    /// at frames 0 and F-1 — and names why anything else is refused.
    #[test]
    fn endpoint_images_resolve_source_and_flf_layouts_only() {
        let frames = 33u32;
        let keyframe = |frame: u32, byte: u8| mold_core::KeyframeCondition {
            frame,
            image: vec![byte],
            name: None,
        };

        // No conditioning at all.
        assert!(resolve_endpoint_images(&request(), frames)
            .unwrap()
            .is_none());

        // A lone source image is the existing I2V shape.
        let mut source_only = request();
        source_only.source_image = Some(vec![7]);
        let endpoints = resolve_endpoint_images(&source_only, frames)
            .unwrap()
            .expect("source resolves");
        assert_eq!(endpoints.first, &[7]);
        assert!(endpoints.last.is_none());

        // The FLF pair anchors the clip's endpoints.
        let mut flf = request();
        flf.keyframes = Some(vec![keyframe(0, 1), keyframe(frames - 1, 2)]);
        let endpoints = resolve_endpoint_images(&flf, frames)
            .unwrap()
            .expect("FLF resolves");
        assert_eq!(endpoints.first, &[1]);
        assert_eq!(endpoints.last, Some(&[2u8][..]));

        // Wrong count, wrong anchors, and ambiguous mixes are refused by name.
        let mut one = request();
        one.keyframes = Some(vec![keyframe(0, 1)]);
        let error = resolve_endpoint_images(&one, frames)
            .unwrap_err()
            .to_string();
        assert!(error.contains("exactly two keyframes"), "{error}");

        let mut misplaced = request();
        misplaced.keyframes = Some(vec![keyframe(0, 1), keyframe(7, 2)]);
        let error = resolve_endpoint_images(&misplaced, frames)
            .unwrap_err()
            .to_string();
        assert!(error.contains("frames 0 and 32"), "{error}");

        let mut ambiguous = request();
        ambiguous.source_image = Some(vec![7]);
        ambiguous.keyframes = Some(vec![keyframe(0, 1), keyframe(frames - 1, 2)]);
        let error = resolve_endpoint_images(&ambiguous, frames)
            .unwrap_err()
            .to_string();
        assert!(error.contains("not both"), "{error}");

        // frames=1 makes the endpoints coincide (both anchor frame 0, which
        // IS frames-1) — refused before the I2V canvas math can compute the
        // underflowing `frames - 2` interior gap.
        let mut coincident = request();
        coincident.keyframes = Some(vec![keyframe(0, 1), keyframe(0, 2)]);
        let error = resolve_endpoint_images(&coincident, 1)
            .unwrap_err()
            .to_string();
        assert!(error.contains("multi-frame clip"), "{error}");

        // An empty keyframes vec is treated as absent, matching serde's
        // omitted-field shape.
        let mut empty = request();
        empty.keyframes = Some(Vec::new());
        assert!(resolve_endpoint_images(&empty, frames).unwrap().is_none());
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

    /// The quality tier's advertised 3.5 means "I did not choose": on a pair
    /// it becomes upstream's per-expert scales, applied on exactly the same
    /// side of the boundary as the expert swap (`text2video.py:343-344`).
    #[test]
    fn quality_tier_default_guidance_becomes_the_upstream_per_expert_pair() {
        let boundary = WanExpertPair::boundary_for(false);
        let quality_default = Some(mold_core::manifest::WAN_A14B_QUALITY_GUIDANCE);
        let plan = resolve_guidance_plan(
            mold_core::manifest::WAN_A14B_QUALITY_GUIDANCE,
            Some(boundary),
            false,
            quality_default,
        );
        assert_eq!(plan.for_timestep(1000), 4.0);
        assert_eq!(
            plan.for_timestep(boundary),
            4.0,
            "the boundary itself is high-noise, matching the expert swap"
        );
        assert_eq!(plan.for_timestep(boundary - 1), 3.0);
        assert_eq!(plan.for_timestep(0), 3.0);
        assert_eq!(plan.max(), 4.0, "the uncond prompt is encoded once");

        // I2V's upstream pair is flat, so the plan is per-expert in name and
        // 3.5 at every step — identical to today's behavior.
        let i2v_boundary = WanExpertPair::boundary_for(true);
        let i2v = resolve_guidance_plan(
            mold_core::manifest::WAN_A14B_QUALITY_GUIDANCE,
            Some(i2v_boundary),
            true,
            quality_default,
        );
        assert_eq!(i2v.for_timestep(1000), 3.5);
        assert_eq!(i2v.for_timestep(0), 3.5);
    }

    /// An explicit scale is an instruction, not a hint: it applies uniformly
    /// even on a pair, and single-expert checkpoints never see per-expert
    /// scales no matter what the request carries.
    #[test]
    fn explicit_or_single_expert_guidance_stays_uniform() {
        let boundary = WanExpertPair::boundary_for(false);
        let quality_default = Some(mold_core::manifest::WAN_A14B_QUALITY_GUIDANCE);
        assert_eq!(
            resolve_guidance_plan(5.0, Some(boundary), false, quality_default),
            WanGuidancePlan::Uniform(5.0)
        );
        assert_eq!(
            resolve_guidance_plan(
                mold_core::manifest::WAN_A14B_QUALITY_GUIDANCE,
                None,
                false,
                quality_default,
            ),
            WanGuidancePlan::Uniform(mold_core::manifest::WAN_A14B_QUALITY_GUIDANCE)
        );
        // The Lightning tiers default to 1.0: uniform, CFG pass skipped —
        // bit-identical to the pre-per-expert behavior.
        let lightning = resolve_guidance_plan(1.0, Some(boundary), false, Some(1.0));
        assert_eq!(lightning, WanGuidancePlan::Uniform(1.0));
        assert!(!needs_cfg_pass(lightning.max()));

        // An explicit 3.5 on a Lightning pair is always a user choice — the
        // tier's own default is 1.0 — and must stay uniform even though the
        // value collides with the quality tier's sentinel (codex review).
        assert_eq!(
            resolve_guidance_plan(
                mold_core::manifest::WAN_A14B_QUALITY_GUIDANCE,
                Some(boundary),
                false,
                Some(1.0),
            ),
            WanGuidancePlan::Uniform(mold_core::manifest::WAN_A14B_QUALITY_GUIDANCE)
        );
        // A community pair with no manifest default never engages per-expert
        // scales.
        assert_eq!(
            resolve_guidance_plan(
                mold_core::manifest::WAN_A14B_QUALITY_GUIDANCE,
                Some(boundary),
                false,
                None,
            ),
            WanGuidancePlan::Uniform(mold_core::manifest::WAN_A14B_QUALITY_GUIDANCE)
        );
    }

    /// Mixed plans keep `needs_cfg_pass` semantics per step: a phase at or
    /// below unit guidance skips its uncond forward even though the negative
    /// prompt was encoded for the other phase.
    #[test]
    fn a_mixed_plan_skips_the_uncond_pass_only_where_guidance_allows() {
        let plan = WanGuidancePlan::PerExpert {
            boundary_timestep: 875,
            guidance: WanExpertGuidance {
                high_noise: 2.0,
                low_noise: 1.0,
            },
        };
        assert!(needs_cfg_pass(plan.max()), "the uncond prompt is encoded");
        assert!(needs_cfg_pass(plan.for_timestep(1000)));
        assert!(
            !needs_cfg_pass(plan.for_timestep(0)),
            "skipped below the boundary"
        );
    }

    #[test]
    fn flow_shift_defaults_and_validates() {
        // The env var is process-global; this test owns it for its duration.
        let previous = std::env::var(FLOW_SHIFT_ENV).ok();
        unsafe { std::env::remove_var(FLOW_SHIFT_ENV) };
        assert_eq!(resolve_flow_shift(None, false).unwrap(), DEFAULT_FLOW_SHIFT);
        // The A14B pair is a different schedule shape, not a bigger one.
        assert_eq!(resolve_flow_shift(None, true).unwrap(), A14B_FLOW_SHIFT);
        assert_ne!(A14B_FLOW_SHIFT, DEFAULT_FLOW_SHIFT);

        // A request-level shift (#782) wins outright.
        assert_eq!(resolve_flow_shift(Some(12.0), true).unwrap(), 12.0);
        for bad in [0.0, -2.0, f64::NAN, f64::INFINITY] {
            assert!(
                resolve_flow_shift(Some(bad), false).is_err(),
                "request shift {bad} must be rejected"
            );
        }

        // The env override beats both defaults…
        unsafe { std::env::set_var(FLOW_SHIFT_ENV, "3.5") };
        assert_eq!(resolve_flow_shift(None, false).unwrap(), 3.5);
        assert_eq!(resolve_flow_shift(None, true).unwrap(), 3.5);
        // …but never the request.
        assert_eq!(resolve_flow_shift(Some(12.0), false).unwrap(), 12.0);

        for bad in ["", "abc", "0", "-2", "inf"] {
            unsafe { std::env::set_var(FLOW_SHIFT_ENV, bad) };
            assert!(
                resolve_flow_shift(None, false).is_err(),
                "{bad:?} must be rejected"
            );
        }

        match previous {
            Some(value) => unsafe { std::env::set_var(FLOW_SHIFT_ENV, value) },
            None => unsafe { std::env::remove_var(FLOW_SHIFT_ENV) },
        }
    }

    /// Solver selection (#795): request wins, env falls back, default stays
    /// FlowUniPC, and the UNet schedulers are refused by name.
    #[test]
    fn wan_solver_resolves_request_env_and_default() {
        use mold_core::Scheduler;
        let previous = std::env::var(SOLVER_ENV).ok();
        unsafe { std::env::remove_var(SOLVER_ENV) };

        assert_eq!(
            resolve_wan_solver(None, false).unwrap(),
            WanSolverKind::UniPc
        );
        assert_eq!(
            resolve_wan_solver(Some(Scheduler::Euler), false).unwrap(),
            WanSolverKind::Euler
        );
        assert_eq!(
            resolve_wan_solver(Some(Scheduler::DpmPp), true).unwrap(),
            WanSolverKind::DpmPp
        );
        assert_eq!(
            resolve_wan_solver(Some(Scheduler::UniPc), false).unwrap(),
            WanSolverKind::UniPc
        );
        for unet in [Scheduler::Ddim, Scheduler::EulerAncestral] {
            let error = resolve_wan_solver(Some(unet), false)
                .unwrap_err()
                .to_string();
            assert!(error.contains("UNet scheduler"), "{error}");
        }

        unsafe { std::env::set_var(SOLVER_ENV, "euler") };
        assert_eq!(
            resolve_wan_solver(None, false).unwrap(),
            WanSolverKind::Euler
        );
        // The request still wins over the env.
        assert_eq!(
            resolve_wan_solver(Some(Scheduler::DpmPp), false).unwrap(),
            WanSolverKind::DpmPp
        );
        unsafe { std::env::set_var(SOLVER_ENV, "dpm++") };
        assert_eq!(
            resolve_wan_solver(None, false).unwrap(),
            WanSolverKind::DpmPp
        );
        unsafe { std::env::set_var(SOLVER_ENV, "sgd") };
        assert!(resolve_wan_solver(None, false).is_err());

        match previous {
            Some(value) => unsafe { std::env::set_var(SOLVER_ENV, value) },
            None => unsafe { std::env::remove_var(SOLVER_ENV) },
        }
    }

    /// Distill strengths: absent = 1.0, the community band is accepted, and
    /// typo-class values are refused.
    #[test]
    fn distill_strength_validates_the_community_band() {
        assert_eq!(resolve_distill_strength("high", None).unwrap(), 1.0);
        assert_eq!(resolve_distill_strength("high", Some(1.8)).unwrap(), 1.8);
        assert_eq!(resolve_distill_strength("low", Some(0.5)).unwrap(), 0.5);
        for bad in [0.0, -1.0, 4.5, f64::NAN, f64::INFINITY] {
            assert!(
                resolve_distill_strength("high", Some(bad)).is_err(),
                "{bad} must be rejected"
            );
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
            // The layout discriminator. Every real Wan DiT carries it; the
            // probe set alone cannot tell the original layout from the
            // diffusers one, because `patch_embedding.weight` is spelled
            // identically in both.
            ("blocks.0.self_attn.q.weight".into(), vec![1536, 1536]),
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
            ("blocks.0.self_attn.q.weight".into(), vec![5120, 5120]),
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
            (
                "model.diffusion_model.blocks.0.self_attn.q.weight".into(),
                vec![1536, 1536],
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
            let err =
                reject_unwired_channel_concat_checkpoint(std::slice::from_ref(&clip), low_noise)
                    .unwrap_err();
            assert!(err.to_string().contains("CLIP-vision"), "got: {err}");
        }

        // A lone 2.2 A14B expert is refused, naming the boundary it would have
        // switched at and what to pull instead.
        let expert = temp.path().join("a14b-high.safetensors");
        write_header(
            &expert,
            &[("patch_embedding.weight", &[5120, 36, 1, 2, 2][..])],
        );
        let err = reject_unwired_channel_concat_checkpoint(std::slice::from_ref(&expert), None)
            .unwrap_err();
        assert!(err.to_string().contains("expert pair"), "got: {err}");
        assert!(err.to_string().contains("900"), "got: {err}");
        assert!(err.to_string().contains("wan22-i2v-a14b"), "got: {err}");

        // With both experts resolved it is accepted — this is the arm the A14B
        // layer replaced.
        reject_unwired_channel_concat_checkpoint(std::slice::from_ref(&expert), Some(&partner))
            .expect("a complete A14B pair is runnable");
    }

    /// The 2.1 I2V refusal must be layout-blind. #803 taught the loader the
    /// diffusers key layout, but this check kept probing only the original
    /// `cross_attn.k_img` spelling — so a diffusers-layout I2V export walked
    /// straight past a refusal that exists precisely because the CLIP-vision
    /// branch is unimplemented.
    #[test]
    fn diffusers_layout_i2v_checkpoints_are_refused_by_name() {
        let temp = tempfile::tempdir().unwrap();
        // Both diffusers markers, each on its own, plus the Comfy-Org repack
        // of one: the prefix and the naming are independent facts.
        for (file, clip_key) in [
            (
                "wan21-i2v-diffusers.safetensors",
                "blocks.0.attn2.add_k_proj.weight",
            ),
            (
                "wan21-i2v-diffusers-embedder.safetensors",
                "condition_embedder.image_embedder.ff.net.0.proj.weight",
            ),
            (
                "wan21-i2v-diffusers-repack.safetensors",
                "model.diffusion_model.blocks.0.attn2.add_k_proj.weight",
            ),
        ] {
            let path = temp.path().join(file);
            write_header(
                &path,
                &[
                    ("patch_embedding.weight", &[5120, 36, 1, 2, 2][..]),
                    (clip_key, &[5120, 1280][..]),
                ],
            );
            let err = reject_unwired_channel_concat_checkpoint(std::slice::from_ref(&path), None)
                .unwrap_err();
            assert!(err.to_string().contains("CLIP-vision"), "{file}: got {err}");
        }
    }

    /// The refusal must read every shard, not just the primary file.
    ///
    /// The diffusers layout is precisely the sharded one, and the split does
    /// not respect any probe set — `Wan2.2-TI2V-5B-Turbo-Diffusers` strands
    /// `proj_out.weight` alone in a second shard, which is why
    /// [`detect_transformer_config_across`] reads the whole set. Probing only
    /// `paths.transformer` therefore admitted a sharded diffusers-layout 2.1
    /// I2V export outright whenever the CLIP adapter landed in a later shard:
    /// with a partner supplied this returned `Ok(())` before the fix, i.e. the
    /// exact export this branch exists to refuse was queued and would have
    /// rendered from an unwired CLIP-vision branch.
    #[test]
    fn sharded_diffusers_layout_i2v_is_refused_across_every_shard() {
        let temp = tempfile::tempdir().unwrap();
        let partner = temp.path().join("partner.safetensors");
        write_header(
            &partner,
            &[("patch_embedding.weight", &[5120, 36, 1, 2, 2][..])],
        );

        // Each diffusers marker in turn, always in the *second* shard, and once
        // more behind the Comfy-Org repack prefix: the shard the adapter lands
        // in, its spelling, and the prefix are three independent facts.
        for (stem, clip_key) in [
            ("attn-adapter", "blocks.0.attn2.add_k_proj.weight"),
            (
                "image-embedder",
                "condition_embedder.image_embedder.ff.net.0.proj.weight",
            ),
            (
                "repacked",
                "model.diffusion_model.blocks.0.attn2.add_k_proj.weight",
            ),
        ] {
            let primary = temp
                .path()
                .join(format!("{stem}-00001-of-00002.safetensors"));
            // The primary carries the 36-channel patch embedding and nothing
            // that names the CLIP branch — this file alone looks runnable.
            write_header(
                &primary,
                &[("patch_embedding.weight", &[5120, 36, 1, 2, 2][..])],
            );
            let shard = temp
                .path()
                .join(format!("{stem}-00002-of-00002.safetensors"));
            write_header(&shard, &[(clip_key, &[5120, 1280][..])]);
            let files = vec![primary, shard];

            for low_noise in [None, Some(partner.as_path())] {
                let err = reject_unwired_channel_concat_checkpoint(&files, low_noise).unwrap_err();
                assert!(err.to_string().contains("CLIP-vision"), "{stem}: got {err}");
            }
        }
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

    /// A sharded diffusers export must detect exactly like the native
    /// single-file checkpoint it was converted from.
    ///
    /// Both halves of this are real properties of
    /// `yetter-ai/Wan2.2-TI2V-5B-Turbo-Diffusers`, verified against its own
    /// safetensors headers: the names are diffusers spellings, and the
    /// output-channel probe sits alone in an 89 MB second shard. Before this,
    /// detection read one file in one spelling, so the checkpoint failed on
    /// `blocks.0.ffn.0.weight` — and admission, left without a geometry, fell
    /// back to the A14B shape and refused it at ~67 GB.
    #[test]
    fn a_sharded_diffusers_export_detects_like_its_native_twin() {
        let temp = tempfile::tempdir().unwrap();
        let first = temp
            .path()
            .join("diffusion_pytorch_model-00001-of-00002.safetensors");
        let second = temp
            .path()
            .join("diffusion_pytorch_model-00002-of-00002.safetensors");

        let mut head: Vec<(String, Vec<usize>)> = vec![
            ("patch_embedding.weight".into(), vec![3072, 48, 1, 2, 2]),
            ("blocks.0.ffn.net.0.proj.weight".into(), vec![14336, 3072]),
            (
                "condition_embedder.text_embedder.linear_1.weight".into(),
                vec![3072, 4096],
            ),
            (
                "condition_embedder.time_embedder.linear_1.weight".into(),
                vec![3072, 256],
            ),
            ("blocks.0.attn1.to_q.weight".into(), vec![3072, 3072]),
        ];
        for layer in 0..30 {
            head.push((
                format!("blocks.{layer}.scale_shift_table"),
                vec![1, 6, 3072],
            ));
        }
        let borrowed: Vec<(&str, &[usize])> = head
            .iter()
            .map(|(name, shape)| (name.as_str(), shape.as_slice()))
            .collect();
        write_header(&first, &borrowed);
        // The output projection alone in the second shard, as published.
        write_header(&second, &[("proj_out.weight", &[192, 3072][..])]);

        let config = detect_transformer_config_across(&[first.clone(), second.clone()]).unwrap();
        assert_eq!(config, WanTransformerConfig::ti2v_5b());

        // The first shard on its own is not enough, and says so by naming the
        // probe it could not find rather than guessing an output width.
        let error = detect_transformer_config(&first).unwrap_err().to_string();
        assert!(error.contains("proj_out.weight"), "{error}");
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
            ("blocks.0.self_attn.q.weight".into(), vec![3072, 3072]),
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
        // A file carrying no Wan attention projection at all is refused at
        // classification, before any shape probe runs.
        assert!(error.contains("not a Wan DiT checkpoint"), "{error}");

        // One that classifies but is missing a probe is named by the probe.
        let partial = temp.path().join("partial.safetensors");
        write_header(
            &partial,
            &[("blocks.0.self_attn.q.weight", &[1536, 1536][..])],
        );
        let error = detect_transformer_config(&partial).unwrap_err().to_string();
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

    /// Deterministically fill a test VarMap: per-name seeds in sorted-name
    /// order, norm gains near 1, biases small, weights at 0.05 sigma. This is
    /// what makes [`tiny_engine_run`] reproducible run-to-run so its pixels
    /// can be pinned (#789) — `VarMap`'s own `Init::Randn` draws from a
    /// process-global RNG.
    fn seed_varmap(varmap: &VarMap, base: u64) {
        let data = varmap.data().lock().unwrap();
        let mut names: Vec<String> = data.keys().cloned().collect();
        names.sort();
        for (index, name) in names.iter().enumerate() {
            let var = &data[name];
            let (scale, shift) = if name.ends_with("gamma") {
                (0.1, 1.0)
            } else if name.ends_with("bias") {
                (0.02, 0.0)
            } else {
                (0.05, 0.0)
            };
            let values = seeded_randn(base + index as u64, var.dims(), &Device::Cpu, DType::F32)
                .unwrap()
                .affine(scale, shift)
                .unwrap();
            var.set(&values).unwrap();
        }
    }

    /// End-to-end on CPU at toy widths: tiny DiT + tiny VAE + a real
    /// FlowUniPC schedule, exercising the whole denoise/decode path including
    /// the CFG branch and the artifact encode. Weights are deterministic, so
    /// two calls (and two processes) produce identical frames.
    fn tiny_engine_run(guidance: f64, steps: u32) -> Vec<image::RgbImage> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let vae_config = WanVaeConfig::tiny_v2_1();
        let varmap = VarMap::new();
        WanVideoVae::from_var_builder(
            VarBuilder::from_varmap(&varmap, dtype, &device),
            vae_config.clone(),
            &device,
            dtype,
        )
        .unwrap();
        seed_varmap(&varmap, 300);
        // Rebuild after seeding: `WanCausalConv3d` slices its 3-D weight into
        // per-tap `Conv2d`s at load time, so mutating the Vars afterwards is
        // invisible to an already-built model.
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
        WanTransformer::from_var_builder(
            VarBuilder::from_varmap(&transformer_map, dtype, &device),
            transformer_config.clone(),
        )
        .unwrap();
        seed_varmap(&transformer_map, 700);
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

        // Distinct cond/uncond contexts: identical ones make `cond == uncond`,
        // so the CFG scale drops out and the pinned pixels cannot see a
        // guidance regression (codex review).
        let context = seeded_randn(11, &[1, 6, 32], &device, dtype).unwrap();
        let uncond_context = seeded_randn(13, &[1, 6, 32], &device, dtype).unwrap();
        let schedule = WanSchedule::new(WanScheduleConfig::new(steps as usize, 8.0)).unwrap();
        let mut solver = WanSolver::UniPc(FlowUniPc::new(schedule.clone()));
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
                    .forward_with_rope(&latents, &t, &uncond_context, &rope)
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

    /// Pinned pixels for the deterministic tiny run (#789): seed 7 noise,
    /// 4-step FlowUniPC at shift 8, CFG 5 over DISTINCT seed-11 cond /
    /// seed-13 uncond contexts (identical ones would cancel the guidance
    /// term), seeded weights. Any schedule,
    /// solver, CFG, noise-layout, or VAE-decode regression moves these
    /// numbers; before this pin such a change could only be caught by UAT.
    ///
    /// Per-frame RGB channel means (5 frames x 3, u8 units) and the pixels at
    /// (3,5), (17,11), (29,27) of every frame. The run is exactly
    /// reproducible on CPU (verified across processes and test-thread
    /// counts), so the tolerances only absorb hypothetical cross-platform
    /// f32 wiggle at the u8 quantization boundary: ±0.35 on a mean, ±1 on a
    /// probe. Regenerate by printing the same reductions from
    /// `tiny_engine_run(5.0, 4)` after an *intentional* pipeline change.
    const TINY_RUN_FRAME_CHANNEL_MEANS: &[f64] = &[
        120.32421875,
        121.3203125,
        130.8427734375,
        143.0107421875,
        140.90234375,
        149.431640625,
        153.169921875,
        157.9775390625,
        146.2822265625,
        134.2734375,
        155.7646484375,
        148.080078125,
        146.45703125,
        175.630859375,
        125.720703125,
    ];
    const TINY_RUN_PROBE_PIXELS: &[u8] = &[
        141, 80, 170, 120, 130, 121, 128, 131, 89, 137, 140, 158, 120, 152, 113, 168, 137, 85, 95,
        210, 182, 136, 101, 164, 163, 137, 188, 153, 179, 143, 110, 196, 94, 170, 197, 102, 160,
        158, 150, 117, 185, 91, 227, 150, 180,
    ];

    #[test]
    fn tiny_end_to_end_pixels_are_pinned() {
        let frames = tiny_engine_run(5.0, 4);
        assert_eq!(frames.len(), 5);
        let mut means = Vec::new();
        let mut probes = Vec::new();
        for frame in &frames {
            for c in 0..3 {
                let sum: u64 = frame.pixels().map(|p| u64::from(p.0[c])).sum();
                means.push(sum as f64 / (32.0 * 32.0));
            }
            for (x, y) in [(3u32, 5u32), (17, 11), (29, 27)] {
                probes.extend_from_slice(&frame.get_pixel(x, y).0);
            }
        }
        assert_eq!(means.len(), TINY_RUN_FRAME_CHANNEL_MEANS.len());
        for (i, (got, want)) in means
            .iter()
            .zip(TINY_RUN_FRAME_CHANNEL_MEANS.iter())
            .enumerate()
        {
            assert!(
                (got - want).abs() <= 0.35,
                "frame-channel mean {i}: got {got}, want {want}"
            );
        }
        assert_eq!(probes.len(), TINY_RUN_PROBE_PIXELS.len());
        for (i, (got, want)) in probes.iter().zip(TINY_RUN_PROBE_PIXELS.iter()).enumerate() {
            let diff = (i16::from(*got) - i16::from(*want)).abs();
            assert!(diff <= 1, "probe {i}: got {got}, want {want}");
        }
    }

    /// The single-pass path must reach the same shapes as the CFG path — the
    /// guidance branch changes the arithmetic, not the plumbing.
    #[test]
    fn single_pass_guidance_runs_the_same_pipeline() {
        let frames = tiny_engine_run(1.0, 4);
        assert_eq!(frames.len(), 5);
        assert_eq!(frames[0].dimensions(), (32, 32));
    }

    /// The preview must project the solver's own x0 — the pre-step
    /// conversion `FlowUniPc::step` records — not a naive post-step
    /// `x_next - sigma_next * v`, which is invalid once the corrector has
    /// rewritten the sample the predictor stepped from (#791 review). The
    /// first (order-1) and terminal steps coincide with the naive identity by
    /// construction, so the fixture must discriminate on a corrector step in
    /// between — a zero preview interval makes the loop emit all of them.
    /// The step cache must be invisible when off and real when on.
    ///
    /// Both halves matter. "Off is bit-identical" is the determinism contract
    /// the family advertises, and it would be easy to break by restructuring
    /// the block loop. "On actually skips" is the part a threshold check can
    /// silently fail at — a cache that never fires looks exactly like a cache
    /// that is working, since the output is then also unchanged.
    #[test]
    fn the_step_cache_is_invisible_when_off_and_skips_blocks_when_on() {
        use crate::wan::step_cache::WanStepCachePolicy;

        let device = Device::Cpu;
        let dtype = DType::F32;
        let z = 16usize;
        let config = WanTransformerConfig::tiny(z, 2, 4);
        let map = VarMap::new();
        let transformer = WanTransformer::from_var_builder(
            VarBuilder::from_varmap(&map, dtype, &device),
            config.clone(),
        )
        .unwrap();
        for (index, var) in map.all_vars().iter().enumerate() {
            let noise = seeded_randn(4242 + index as u64, var.dims(), &device, dtype)
                .unwrap()
                .affine(0.2, 0.0)
                .unwrap();
            var.set(&noise).unwrap();
        }

        let (latent_frames, latent_h, latent_w) = (2usize, 4usize, 4usize);
        let total = 16usize;
        let schedule = WanSchedule::new(WanScheduleConfig::new(total, 8.0)).unwrap();
        let latents0 = seeded_randn(
            11,
            &[1, z, latent_frames, latent_h, latent_w],
            &device,
            dtype,
        )
        .unwrap();
        let context = Tensor::zeros((1, 6, config.text_dim), dtype, &device).unwrap();
        let quiet = crate::progress::ProgressReporter::default();

        let run = |policy: WanStepCachePolicy| {
            let mut experts = WanExperts::single(transformer.clone());
            let rope = experts
                .transformer_for(schedule.timesteps[0], &quiet)
                .unwrap()
                .rope_freqs_for(
                    &Tensor::zeros((1, z, latent_frames, latent_h, latent_w), dtype, &device)
                        .unwrap(),
                )
                .unwrap();
            let mut solver = WanSolver::UniPc(FlowUniPc::new(schedule.clone()));
            run_denoise_loop(DenoiseInputs {
                experts: &mut experts,
                conditioning: &WanImageConditioning::None,
                schedule: &schedule,
                solver: &mut solver,
                latents: latents0.clone(),
                cond_embeds: &context,
                uncond_embeds: None,
                guidance: WanGuidancePlan::Uniform(1.0),
                patch: config.patch_size.1,
                rope: &rope,
                device: &device,
                progress: &quiet,
                previewer: None,
                step_cache: policy,
            })
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
        };

        let baseline = run(WanStepCachePolicy::Off);
        // Off twice is the same run: the loop is deterministic, so any
        // difference below is the cache and not sampling noise.
        assert_eq!(baseline, run(WanStepCachePolicy::Off));

        // A threshold this large accepts every step after the first full one,
        // so the tail residual is replayed and the result must diverge. If it
        // does not, the cache never fired and the "off is identical" half of
        // this test would be passing vacuously.
        let cached = run(WanStepCachePolicy::Threshold(f64::MAX));
        assert_eq!(baseline.len(), cached.len());
        assert!(
            baseline
                .iter()
                .zip(cached.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "an always-accept threshold must skip blocks and change the result",
        );

        // A threshold of zero can never accept a step, so it must reproduce
        // the uncached run exactly — the cache path itself adds no drift.
        assert_eq!(baseline, run(WanStepCachePolicy::Threshold(0.0)));
    }

    #[test]
    fn preview_projects_the_solvers_own_x0() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let z = 16usize;
        let config = WanTransformerConfig::tiny(z, 2, 2);
        let map = VarMap::new();
        let transformer = WanTransformer::from_var_builder(
            VarBuilder::from_varmap(&map, dtype, &device),
            config.clone(),
        )
        .unwrap();
        // Seed every weight so the velocity field is nontrivial — an all-zero
        // DiT would let a wrong-sigma preview hide inside v = 0.
        for (index, var) in map.all_vars().iter().enumerate() {
            let noise = seeded_randn(1000 + index as u64, var.dims(), &device, dtype)
                .unwrap()
                .affine(0.2, 0.0)
                .unwrap();
            var.set(&noise).unwrap();
        }

        let (latent_frames, latent_h, latent_w) = (2usize, 4usize, 4usize);
        let total = 4usize;
        let schedule = WanSchedule::new(WanScheduleConfig::new(total, 8.0)).unwrap();
        let latents0 = seeded_randn(
            7,
            &[1, z, latent_frames, latent_h, latent_w],
            &device,
            dtype,
        )
        .unwrap();
        let context = Tensor::zeros((1, 6, config.text_dim), dtype, &device).unwrap();
        let progress_quiet = crate::progress::ProgressReporter::default();
        let mut experts = WanExperts::single(transformer.clone());
        let rope = experts
            .transformer_for(schedule.timesteps[0], &progress_quiet)
            .unwrap()
            .rope_freqs_for(
                &Tensor::zeros((1, z, latent_frames, latent_h, latent_w), dtype, &device).unwrap(),
            )
            .unwrap();

        // Run the real loop with an unthrottled capturing previewer: every
        // step emits, including the corrector steps under test.
        let mut progress = crate::progress::ProgressReporter::default();
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = events.clone();
        progress.set_callback(Box::new(move |e| sink.lock().unwrap().push(e)));
        let previewer = crate::latent_preview::LatentPreviewer::wan(z)
            .expect("16-channel Wan checkpoints have a preview table")
            .force_enabled()
            .with_min_interval(std::time::Duration::ZERO);
        let mut solver = WanSolver::UniPc(FlowUniPc::new(schedule.clone()));
        run_denoise_loop(DenoiseInputs {
            experts: &mut experts,
            conditioning: &WanImageConditioning::None,
            schedule: &schedule,
            solver: &mut solver,
            latents: latents0.clone(),
            cond_embeds: &context,
            uncond_embeds: None,
            guidance: WanGuidancePlan::Uniform(1.0),
            patch: config.patch_size.1,
            rope: &rope,
            device: &device,
            progress: &progress,
            previewer: Some(&previewer),
            step_cache: crate::wan::step_cache::WanStepCachePolicy::Off,
        })
        .unwrap();
        let loop_pngs: Vec<(usize, Vec<u8>)> = {
            let events = events.lock().unwrap();
            events
                .iter()
                .filter_map(|e| match e {
                    ProgressEvent::Preview {
                        image_png, step, ..
                    } => Some((*step, image_png.as_ref().clone())),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(
            loop_pngs.iter().map(|(s, _)| *s).collect::<Vec<_>>(),
            (1..=total).collect::<Vec<_>>(),
            "an unthrottled previewer emits every step"
        );

        // Replay the identical trajectory, capturing per-step the solver's
        // recorded x0 and the naive post-step estimate.
        let render = |latent: &Tensor, step: usize| -> Vec<u8> {
            let previewer = crate::latent_preview::LatentPreviewer::wan(z)
                .unwrap()
                .force_enabled();
            let mut reporter = crate::progress::ProgressReporter::default();
            let captured = Arc::new(Mutex::new(Vec::new()));
            let sink = captured.clone();
            reporter.set_callback(Box::new(move |e| sink.lock().unwrap().push(e)));
            previewer.maybe_emit(&reporter, latent, step, total);
            let events = captured.lock().unwrap();
            events
                .iter()
                .find_map(|e| match e {
                    ProgressEvent::Preview { image_png, .. } => Some(image_png.as_ref().clone()),
                    _ => None,
                })
                .expect("render emits")
        };
        let mut replay = FlowUniPc::new(schedule.clone());
        let mut latents = latents0;
        let mut corrector_steps_discriminate = false;
        for (index, timestep) in schedule.timesteps.iter().enumerate() {
            let step = index + 1;
            let timestep_tensor = scalar_timestep_tensor(*timestep, &device).unwrap();
            let velocity = transformer
                .forward_with_rope(&latents, &timestep_tensor, &context, &rope)
                .unwrap();
            latents = replay.step(&velocity, index, &latents).unwrap();

            let solver_png = render(replay.last_x0().expect("step records an x0"), step);
            assert_eq!(
                loop_pngs[index].1, solver_png,
                "step {step}: the emitted preview must be the solver's own x0"
            );
            if index >= 1 && step < total {
                let naive = latents
                    .sub(&velocity.affine(schedule.sigmas[index + 1], 0.0).unwrap())
                    .unwrap();
                if render(&naive, step) != solver_png {
                    corrector_steps_discriminate = true;
                }
            }
        }
        assert!(
            corrector_steps_discriminate,
            "fixture has no power: every corrector step rendered the naive \
             estimate identically to the solver's x0"
        );
    }

    /// The real denoise loop must emit `Preview` events when a previewer is
    /// attached: the final step is always rendered, at latent resolution.
    #[test]
    fn tiny_denoise_loop_emits_previews() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        // z = 16 so the real Wan 2.1 factor table applies.
        let z = 16usize;
        let config = WanTransformerConfig::tiny(z, 2, 2);
        let map = VarMap::new();
        let transformer = WanTransformer::from_var_builder(
            VarBuilder::from_varmap(&map, dtype, &device),
            config.clone(),
        )
        .unwrap();

        let (latent_frames, latent_h, latent_w) = (2usize, 4usize, 4usize);
        let schedule = WanSchedule::new(WanScheduleConfig::new(4, 8.0)).unwrap();
        let mut solver = WanSolver::UniPc(FlowUniPc::new(schedule.clone()));
        let latents = seeded_randn(
            7,
            &[1, z, latent_frames, latent_h, latent_w],
            &device,
            dtype,
        )
        .unwrap();

        let mut progress = crate::progress::ProgressReporter::default();
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = events.clone();
        progress.set_callback(Box::new(move |e| sink.lock().unwrap().push(e)));

        let context = Tensor::zeros((1, 6, config.text_dim), dtype, &device).unwrap();
        let mut experts = WanExperts::single(transformer);
        let rope = experts
            .transformer_for(schedule.timesteps[0], &progress)
            .unwrap()
            .rope_freqs_for(
                &Tensor::zeros((1, z, latent_frames, latent_h, latent_w), dtype, &device).unwrap(),
            )
            .unwrap();

        // `force_enabled` keeps the test hermetic against MOLD_STEP_PREVIEW.
        let previewer = crate::latent_preview::LatentPreviewer::wan(z)
            .expect("16-channel Wan checkpoints have a preview table")
            .force_enabled();
        run_denoise_loop(DenoiseInputs {
            experts: &mut experts,
            conditioning: &WanImageConditioning::None,
            schedule: &schedule,
            solver: &mut solver,
            latents,
            cond_embeds: &context,
            uncond_embeds: None,
            guidance: WanGuidancePlan::Uniform(1.0),
            patch: config.patch_size.1,
            rope: &rope,
            device: &device,
            progress: &progress,
            previewer: Some(&previewer),
            step_cache: crate::wan::step_cache::WanStepCachePolicy::Off,
        })
        .unwrap();

        let events = events.lock().unwrap();
        let previews: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                ProgressEvent::Preview {
                    image_png,
                    step,
                    total,
                } => Some((image_png.clone(), *step, *total)),
                _ => None,
            })
            .collect();
        assert!(
            !previews.is_empty(),
            "the wan denoise loop must emit at least one preview frame"
        );
        // The final step is always emitted, at latent resolution.
        let (png, step, total) = previews.last().unwrap();
        assert_eq!((*step, *total), (4, 4));
        assert!(png.starts_with(b"\x89PNG"));
        let decoded = image::load_from_memory(png).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (4, 4));
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
            let mut solver = WanSolver::UniPc(FlowUniPc::new(schedule.clone()));
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
                guidance: WanGuidancePlan::Uniform(guidance),
                patch: transformer_config.patch_size.1,
                rope: &rope,
                device: &device,
                progress: &progress,
                previewer: None,
                step_cache: crate::wan::step_cache::WanStepCachePolicy::Off,
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
