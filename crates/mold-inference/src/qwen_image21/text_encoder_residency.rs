//! Where Qwen Image 2.1's Qwen3-VL text encoder lives while the transformer
//! denoises, for an EAGER engine.
//!
//! The eager engine keeps the transformer and VAE resident across requests.
//! The text encoder is needed only while prompts encode, so on a card that
//! cannot hold everything at once it can leave the device for the denoise and
//! decode phases. The decision is binary per request, like
//! [`crate::flux2::text_encoder_residency::decide_text_encoder_residency`],
//! and has exactly one function, [`decide`], read by BOTH the engine and
//! mold-server's `execution_plan::build_plan` (and priced by
//! `memory_preflight`'s qwen-image21 arm), so the plan and the render cannot
//! disagree:
//!
//! * **Resident** — weights + the TE + the render workspace fit the card: keep
//!   it on the device (today's eager behaviour, and every 48 GB card).
//! * **ParkHost** — the TE does not fit beside the denoise workspace, but the
//!   host can hold it above the scheduler's safety floor: `Qwen3Encoder::
//!   park_to_cpu` after the encode, `unpark_to_gpu` before the next (a
//!   device↔host copy of weights already in memory; ~1.5 s each way for the
//!   16.4 GB BF16 LM, less for a GGUF tier).
//! * **Drop** — neither: release after the encode and reload from disk next
//!   request.
//!
//! Metal and CPU placements are always Resident: unified memory makes a park a
//! second allocation of memory already reachable, and a CPU-placed encoder is
//! on the host already. An unmeasurable card (`usable_free_bytes == 0`) also
//! answers Resident — today's behaviour — because every residency decision
//! falls back to what shipped when it cannot measure.

use std::path::{Path, PathBuf};

use crate::flux2::text_encoder_residency::{host_safety_floor_bytes, KeepTeRamMode};

/// The Qwen3-VL-8B variant decision (BF16 shards or an official GGUF), shared
/// by the engine and the planner so both price the same encoder.
pub use crate::encoders::variant_resolution::{choose_qwen3_vl_variant, Qwen3Choice};

/// Allocator margin charged once on top of every phase, as FLUX's
/// `still_transformer_residency` does.
pub const ALLOCATOR_MARGIN_BYTES: u64 = 1024 * 1024 * 1024;

/// The device the text encoder was placed on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TeDevice {
    Cuda,
    Metal,
    Cpu,
}

/// Everything [`decide`] reads. Bytes throughout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen21TeBudget {
    pub device: TeDevice,
    /// Free device bytes as if NOTHING this render loads were resident — a
    /// caller sampling free VRAM with the engine's own weights on the card
    /// adds their bytes back (the `memory_preflight` `active_vram_bytes`
    /// convention). Zero means unmeasurable.
    pub usable_free_bytes: u64,
    /// Device bytes of the resident transformer (quantized tiers at rest).
    pub transformer_bytes: u64,
    /// Device bytes of the resident VAE.
    pub vae_bytes: u64,
    /// Device bytes of the loaded text encoder ([`text_encoder_device_bytes`]).
    pub text_encoder_bytes: u64,
    /// Device bytes of the reference encoders (vision tower and VAE encoder)
    /// an eager engine keeps loaded after a reference request — resident
    /// through every phase ([`Qwen21RenderPhases::reference_encoder_bytes`]).
    pub reference_encoder_bytes: u64,
    /// Encode-phase working set above the resident weights (vision attention,
    /// the multimodal prompt's rows and scores). Gone before the denoise.
    pub encode_workspace_bytes: u64,
    /// Denoise-phase workspace (activations, prefix KV cache, both CFG
    /// branches).
    pub denoise_workspace_bytes: u64,
    /// VAE-decode peak (`device::qwen_image21_vae_decode_peak_bytes`).
    pub decode_peak_bytes: u64,
    /// Host RAM, total and available now (`MemAvailable` plus any credit the
    /// caller's ledger applies). Zero total means unmeasurable.
    pub host_total_bytes: u64,
    pub host_available_bytes: u64,
    /// Host bytes THIS engine's park already holds, credited back so a warm
    /// request does not ask for room for a second copy.
    pub already_parked_bytes: u64,
    pub keep_te_ram: KeepTeRamMode,
}

/// Where the text encoder lives between encodes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen21TeResidency {
    Resident,
    ParkHost,
    Drop,
}

/// Where the transformer lives while the VAE decodes.
///
/// At 2K the decode alone peaks at ~27.6 GB under cuDNN
/// (`device::qwen_image21_vae_decode_peak_bytes`), which with the 14.8 GB BF16
/// transformer does not fit a 46 GB card. Parking the transformer to host RAM
/// for the decode and restoring it afterwards keeps the eager engine's speed
/// (a device↔host copy, no reload from disk).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransformerDecode {
    /// The transformer stays on the card through the decode.
    Resident,
    /// Park to host RAM before the decode, restore after.
    ParkHost,
    /// The host cannot hold it either: release it (the engine reloads on the
    /// next request).
    Drop,
}

/// The decision and the reason a log line names.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen21TeDecision {
    pub residency: Qwen21TeResidency,
    pub transformer_decode: TransformerDecode,
    pub reason: &'static str,
    /// Device bytes the eager render needs under this decision — the largest
    /// of its phases (encode, denoise, decode) with the allocator margin.
    /// This is what `memory_preflight` prices eager at.
    pub eager_peak_bytes: u64,
}

impl Qwen21TeBudget {
    fn te(&self, on_device: bool) -> u64 {
        if on_device {
            self.text_encoder_bytes
        } else {
            0
        }
    }

    fn transformer(&self, on_device: bool) -> u64 {
        if on_device {
            self.transformer_bytes
        } else {
            0
        }
    }

    /// Prompt encode: transformer, VAE, the encoder and the reference
    /// encoders all on the card, plus the encode working set.
    fn encode_phase(&self) -> u64 {
        self.transformer_bytes
            .saturating_add(self.vae_bytes)
            .saturating_add(self.text_encoder_bytes)
            .saturating_add(self.reference_encoder_bytes)
            .saturating_add(self.encode_workspace_bytes)
            .saturating_add(ALLOCATOR_MARGIN_BYTES)
    }

    fn denoise_phase(&self, te: bool) -> u64 {
        self.transformer_bytes
            .saturating_add(self.vae_bytes)
            .saturating_add(self.te(te))
            .saturating_add(self.reference_encoder_bytes)
            .saturating_add(self.denoise_workspace_bytes)
            .saturating_add(ALLOCATOR_MARGIN_BYTES)
    }

    fn decode_phase(&self, te: bool, transformer: bool) -> u64 {
        self.transformer(transformer)
            .saturating_add(self.vae_bytes)
            .saturating_add(self.te(te))
            .saturating_add(self.reference_encoder_bytes)
            .saturating_add(self.decode_peak_bytes)
            .saturating_add(ALLOCATOR_MARGIN_BYTES)
    }

    /// Every phase's peak under one placement. The phases run one after
    /// another — the encode's working set is freed before the denoise
    /// allocates its own, and the denoise's before the decode — so the
    /// render needs the LARGEST phase, never their sum; what stays resident
    /// across phases (weights, a resident encoder) is counted in each.
    fn peak(&self, te: bool, transformer_through_decode: bool) -> u64 {
        self.encode_phase()
            .max(self.denoise_phase(te))
            .max(self.decode_phase(te, transformer_through_decode))
    }

    fn host_floor(&self) -> u64 {
        host_safety_floor_bytes(self.host_total_bytes)
    }

    fn host_room(&self) -> u64 {
        self.host_available_bytes
            .saturating_add(self.already_parked_bytes)
    }

    /// Whether the host can hold a park of the TE, by the same floor rule the
    /// FLUX.2 decision uses (`Force` asks only for the TE's own room; `Auto`
    /// also leaves the transformer's page-cache room).
    fn host_can_park_te(&self) -> bool {
        if self.host_total_bytes == 0 || self.text_encoder_bytes == 0 {
            return false;
        }
        let required = match self.keep_te_ram {
            KeepTeRamMode::Never => return false,
            KeepTeRamMode::Force => self.text_encoder_bytes.saturating_add(self.host_floor()),
            KeepTeRamMode::Auto => self
                .text_encoder_bytes
                .saturating_add(self.transformer_bytes)
                .saturating_add(self.host_floor()),
        };
        self.host_room() >= required
    }

    /// Whether the host can take the transformer for the decode on top of a
    /// parked encoder, above the floor.
    fn host_can_park_transformer(&self, te_parked: bool) -> bool {
        self.host_total_bytes > 0
            && self.host_room()
                >= self
                    .transformer_bytes
                    .saturating_add(if te_parked {
                        self.text_encoder_bytes
                    } else {
                        0
                    })
                    .saturating_add(self.host_floor())
    }
}

/// The one Qwen Image 2.1 residency decision for an eager engine: where the
/// text encoder lives between encodes, and where the transformer lives while
/// the VAE decodes.
///
/// 1. Everything resident when `transformer + VAE + TE + max(activation,
///    decode)` (and the encode phase) fits.
/// 2. Otherwise the encoder leaves the card after encoding — parked in host
///    RAM when the host has room, dropped otherwise.
/// 3. If the decode still does not fit beside the transformer, the
///    transformer is parked to host RAM for the decode and restored after
///    (dropped, and reloaded next request, if the host cannot take it).
pub fn decide(budget: &Qwen21TeBudget) -> Qwen21TeDecision {
    // A host-placed encoder never occupies the card; only the transformer's
    // decode placement is still a question for it.
    let on_host = Qwen21TeBudget {
        text_encoder_bytes: 0,
        ..*budget
    };
    let budget = if budget.device == TeDevice::Cpu {
        &on_host
    } else {
        budget
    };
    let resident = |reason| Qwen21TeDecision {
        residency: Qwen21TeResidency::Resident,
        transformer_decode: TransformerDecode::Resident,
        reason,
        eager_peak_bytes: budget.peak(true, true),
    };
    if budget.device == TeDevice::Metal {
        return resident("unified memory: a park copies nothing");
    }
    if budget.usable_free_bytes == 0 {
        return resident("the card could not be measured");
    }
    if budget.peak(true, true) <= budget.usable_free_bytes {
        return resident(if budget.device == TeDevice::Cpu {
            "the encoder is placed on the host; weights and workspace fit the card"
        } else {
            "weights, encoder and workspace fit the card"
        });
    }
    if budget.device == TeDevice::Cpu {
        let transformer_parks = budget.host_can_park_transformer(false);
        return Qwen21TeDecision {
            residency: Qwen21TeResidency::Resident,
            transformer_decode: if transformer_parks {
                TransformerDecode::ParkHost
            } else {
                TransformerDecode::Drop
            },
            reason:
                "the encoder is on the host; the transformer leaves the card for the VAE decode",
            eager_peak_bytes: budget.peak(true, false),
        };
    }
    let te_parks = budget.host_can_park_te();
    let residency = if te_parks {
        Qwen21TeResidency::ParkHost
    } else {
        Qwen21TeResidency::Drop
    };
    if budget.peak(false, true) <= budget.usable_free_bytes {
        return Qwen21TeDecision {
            residency,
            transformer_decode: TransformerDecode::Resident,
            reason: if te_parks {
                "the encoder leaves the card for denoise; the host holds it"
            } else if budget.keep_te_ram == KeepTeRamMode::Never {
                "the encoder leaves the card for denoise; MOLD_KEEP_TE_RAM=0 forbids a park"
            } else {
                "the encoder leaves the card for denoise; the host has no room to park it"
            },
            eager_peak_bytes: budget.peak(false, true),
        };
    }
    let transformer_parks = budget.host_can_park_transformer(te_parks);
    Qwen21TeDecision {
        residency,
        transformer_decode: if transformer_parks {
            TransformerDecode::ParkHost
        } else {
            TransformerDecode::Drop
        },
        reason: if transformer_parks {
            "the encoder leaves the card, and the transformer is parked to host for the VAE decode"
        } else {
            "the encoder leaves the card, and the transformer is released for the VAE decode"
        },
        eager_peak_bytes: budget.peak(false, false),
    }
}

/// Everything the planner and the engine both know before the encoder loads.
#[derive(Clone, Copy, Debug)]
pub struct Qwen21PlanInputs<'a> {
    pub paths: &'a mold_core::ModelPaths,
    /// `MOLD_QWEN3_VARIANT`, as the engine reads it.
    pub qwen3_variant: Option<&'a str>,
    pub device: TeDevice,
    /// Free device bytes as if nothing this render loads were resident.
    pub usable_free_bytes: u64,
    /// The request's phase working sets ([`render_phases`]).
    pub phases: Qwen21RenderPhases,
    pub host_total_bytes: u64,
    pub host_available_bytes: u64,
    pub already_parked_bytes: u64,
    pub keep_te_ram: KeepTeRamMode,
}

/// The encoder the render will load and where it lives between encodes.
#[derive(Clone, Copy, Debug)]
pub struct Qwen21TePlan {
    pub choice: Qwen3Choice,
    pub text_encoder_bytes: u64,
    pub decision: Qwen21TeDecision,
}

fn file_bytes(path: &Path) -> u64 {
    std::fs::metadata(path).map_or(0, |metadata| metadata.len())
}

/// Device bytes of the transformer at rest: its file(s). Every tier loads its
/// storage as-is (GGUF and INT8 stay quantized, FP8 stays F8).
pub fn transformer_device_bytes(paths: &mold_core::ModelPaths) -> u64 {
    if paths.transformer_shards.is_empty() {
        file_bytes(&paths.transformer)
    } else {
        paths.transformer_shards.iter().map(|p| file_bytes(p)).sum()
    }
}

/// The transformer tier's storage format, read from the first checkpoint
/// file's header (`None` when it cannot be read — the caller then charges
/// the BF16 workspace).
pub fn transformer_format(
    paths: &mold_core::ModelPaths,
) -> Option<crate::artifact_format::QwenImage21TransformerFormat> {
    let first = paths
        .transformer_shards
        .first()
        .unwrap_or(&paths.transformer);
    crate::artifact_format::probe_qwen_image21_transformer(first).ok()
}

/// Device bytes of the Qwen3-VL-8B GGUF `variant`: measured from its header
/// when the file is installed, else its file size plus the F32 embedding the
/// loader materializes (an upper bound — the quantized embedding it replaces
/// is not subtracted).
fn gguf_variant_device_bytes(variant: &mold_core::manifest::Qwen3Variant) -> u64 {
    const EMBEDDING_F32_BYTES: u64 = 151_936 * 4096 * 4;
    mold_core::download::cached_file_path(
        variant.hf_repo,
        variant.hf_filename,
        Some("shared/qwen3-vl-8b-gguf"),
    )
    .and_then(|path| text_encoder_device_bytes(&[path]).ok())
    .unwrap_or(variant.size_bytes + EMBEDDING_F32_BYTES)
}

/// The one plan both the engine and mold-server's `build_plan` /
/// `memory_preflight` read: which encoder loads (decided on the card left
/// after the transformer and VAE, exactly as the engine measures it after
/// loading them), its device bytes, and [`decide`]'s residency.
pub fn plan(inputs: &Qwen21PlanInputs<'_>) -> anyhow::Result<Qwen21TePlan> {
    let transformer_bytes = transformer_device_bytes(inputs.paths);
    let vae_bytes = file_bytes(&inputs.paths.vae);
    let (is_cuda, is_metal) = match inputs.device {
        TeDevice::Cuda => (true, false),
        TeDevice::Metal => (false, true),
        TeDevice::Cpu => (false, false),
    };
    let free_for_encoder = inputs
        .usable_free_bytes
        .saturating_sub(transformer_bytes)
        .saturating_sub(vae_bytes);
    let choice =
        choose_qwen3_vl_variant(inputs.qwen3_variant, is_cuda, is_metal, free_for_encoder)?;
    let text_encoder_bytes = match choice {
        Qwen3Choice::Bf16 { .. } => text_encoder_device_bytes(&inputs.paths.text_encoder_files)
            .unwrap_or(mold_core::manifest::QWEN3_8B_FP16_SIZE),
        Qwen3Choice::Gguf { variant, .. } => gguf_variant_device_bytes(variant),
    };
    let device = if choice.on_gpu() {
        inputs.device
    } else {
        TeDevice::Cpu
    };
    let decision = decide(&Qwen21TeBudget {
        device,
        usable_free_bytes: inputs.usable_free_bytes,
        transformer_bytes,
        vae_bytes,
        text_encoder_bytes,
        // The vision tower lives on the text encoder's device: a host-placed
        // encoder puts it on the host with it.
        reference_encoder_bytes: if choice.on_gpu() {
            inputs.phases.reference_encoder_bytes
        } else {
            inputs.phases.reference_vae_encoder_bytes
        },
        encode_workspace_bytes: inputs.phases.encode_workspace_bytes,
        denoise_workspace_bytes: inputs.phases.denoise_workspace_bytes,
        decode_peak_bytes: inputs.phases.decode_peak_bytes,
        host_total_bytes: inputs.host_total_bytes,
        host_available_bytes: inputs.host_available_bytes,
        already_parked_bytes: inputs.already_parked_bytes,
        keep_te_ram: inputs.keep_te_ram,
    });
    Ok(Qwen21TePlan {
        choice,
        text_encoder_bytes,
        decision,
    })
}

/// The render workspace both sides charge at `width`x`height`: the denoise
/// activation estimate (`device::activation_bytes`, which carries the prefix
/// KV cache) and the measured VAE-decode peak
/// (`device::qwen_image21_vae_decode_peak_bytes`) under the conv backend the
/// family resolves to, at the VAE's `vae_dtype_bytes` (2 on CUDA's BF16 VAE).
/// `format` is the transformer tier ([`transformer_format`]); the `int8-conv`
/// tier adds its measured W8A8 activation workspace
/// (`device::qwen_image21_linear_workspace_bytes`) to the denoise term.
pub fn render_workspace_bytes(
    format: Option<crate::artifact_format::QwenImage21TransformerFormat>,
    width: u32,
    height: u32,
    batch: u32,
    vae_dtype_bytes: u32,
) -> (u64, u64) {
    let joint = crate::device::QwenImage21SequenceShape::for_request(width, height, &[])
        .joint_tokens() as u64;
    let denoise = crate::device::activation_bytes(
        width,
        height,
        batch,
        2,
        crate::device::ActivationFamily::QwenImage21Dit,
    )
    .saturating_add(crate::device::qwen_image21_linear_workspace_bytes(
        format, joint, batch,
    ));
    let conv =
        crate::conv_policy::resolve_for(crate::conv_policy::policy_for_family("qwen-image21"));
    (
        denoise,
        crate::device::qwen_image21_vae_decode_peak_bytes(width, height, conv, vae_dtype_bytes),
    )
}

/// One Qwen Image 2.1 request, as the phase sizing reads it.
#[derive(Clone, Copy, Debug)]
pub struct Qwen21RenderRequest<'a> {
    /// The transformer tier ([`transformer_format`]).
    pub format: Option<crate::artifact_format::QwenImage21TransformerFormat>,
    pub width: u32,
    pub height: u32,
    pub batch: u32,
    /// Source dimensions of every reference image, in order.
    pub references: &'a [(u32, u32)],
    /// CFG branches that carry a prefix (2 with guidance and a negative).
    pub branches: usize,
    /// The VAE's working dtype bytes (2 on CUDA's BF16 VAE, 4 on Metal).
    pub vae_dtype_bytes: u32,
}

/// Each phase's working set for one request — the ONE sizing the engine
/// (`settle_text_encoder_residency`), [`decide`] / [`plan`], mold-server's
/// `memory_preflight` and `execution_plan` all read.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qwen21RenderPhases {
    /// Vision tower + VAE encoder weights, resident through every phase once
    /// a reference request has loaded them.
    pub reference_encoder_bytes: u64,
    /// The VAE encoder's share of [`Self::reference_encoder_bytes`] — what
    /// stays on the card when the text encoder (and with it the vision tower)
    /// is placed on the host.
    pub reference_vae_encoder_bytes: u64,
    /// The encode phase's working set above the resident weights.
    pub encode_workspace_bytes: u64,
    /// The denoise phase's working set: activations and the retained prefix
    /// cache.
    pub denoise_workspace_bytes: u64,
    /// The VAE decode's peak.
    pub decode_peak_bytes: u64,
}

/// The prefix-cache budget of `request` on a card with `usable_free_bytes`
/// usable (as if nothing of this render were resident; `None` when no card
/// is known) holding `transformer_and_vae_bytes` of weights. The cache is
/// planned beside what the DENOISE holds — the workspace and the resident
/// reference encoders — and never beside the text encoder, which the
/// residency decision parks when the cache needs its room: a retained cache
/// is worth ~8x the denoise, a resident encoder one park per request.
pub fn prefix_cache_budget(
    request: &Qwen21RenderRequest<'_>,
    usable_free_bytes: Option<u64>,
    transformer_and_vae_bytes: u64,
) -> super::PrefixCacheBudget {
    crate::device::qwen_image21_prefix_cache_budget(
        usable_free_bytes,
        transformer_and_vae_bytes,
        crate::device::qwen_image21_planned_denoise_bytes(
            request.format,
            request.width,
            request.height,
            request.references,
            request.batch,
            2,
        ),
    )
}

/// [`Qwen21RenderPhases`] for `request`, with the prefix cache retained under
/// `cache_budget` ([`prefix_cache_budget`]).
pub fn render_phases(
    request: &Qwen21RenderRequest<'_>,
    cache_budget: super::PrefixCacheBudget,
) -> Qwen21RenderPhases {
    let (denoise, decode_peak_bytes) = render_workspace_bytes(
        request.format,
        request.width,
        request.height,
        request.batch,
        request.vae_dtype_bytes,
    );
    let shape = crate::device::QwenImage21SequenceShape::for_request(
        request.width,
        request.height,
        request.references,
    );
    Qwen21RenderPhases {
        reference_encoder_bytes: crate::device::qwen_image21_reference_encoder_bytes(shape),
        reference_vae_encoder_bytes: if request.references.is_empty() {
            0
        } else {
            crate::device::QWEN_IMAGE21_VAE_ENCODER_F32_BYTES
        },
        encode_workspace_bytes: crate::device::qwen_image21_encode_workspace_bytes(shape, 2),
        denoise_workspace_bytes: denoise.saturating_add(
            crate::device::qwen_image21_reference_extra_bytes(
                request.format,
                request.width,
                request.height,
                request.batch,
                2,
                request.references,
                request.branches,
                cache_budget,
            ),
        ),
        decode_peak_bytes,
    }
}

/// Device bytes the Qwen3-VL language model occupies once loaded from
/// `paths`.
///
/// * BF16 shards: the `model.language_model.*` tensors only — the vision tower
///   and `lm_head` share the files but the text encoder never loads them.
/// * A GGUF: every tensor at rest, except `token_embd`, which
///   `GgufQwen3Encoder` dequantizes to F32 on the device (and parks its
///   quantized source on the host).
pub fn text_encoder_device_bytes(paths: &[PathBuf]) -> anyhow::Result<u64> {
    match paths {
        [single] if is_gguf(single) => gguf_text_encoder_bytes(single),
        _ => {
            let mut total = 0u64;
            for path in paths {
                for (name, value) in crate::weight_loader::read_safetensors_header(path)? {
                    if !name.starts_with("model.language_model.") {
                        continue;
                    }
                    let info: safetensors::tensor::TensorInfo = serde_json::from_value(value)?;
                    total += (info.data_offsets.1 - info.data_offsets.0) as u64;
                }
            }
            Ok(total)
        }
    }
}

fn is_gguf(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
}

fn gguf_text_encoder_bytes(path: &Path) -> anyhow::Result<u64> {
    let mut file = std::fs::File::open(path)?;
    let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    let mut total = 0u64;
    for (name, info) in &content.tensor_infos {
        let elements = info.shape.elem_count() as u64;
        if name == "token_embd.weight" {
            total += elements * 4;
            continue;
        }
        let block = info.ggml_dtype.block_size() as u64;
        total += elements / block * info.ggml_dtype.type_size() as u64;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;
    const GIB: u64 = 1 << 30;

    fn decode_peak(width: u32, height: u32) -> u64 {
        crate::device::qwen_image21_vae_decode_peak_bytes(
            width,
            height,
            crate::conv_policy::ConvBackend::Cudnn,
            2,
        )
    }

    /// A realistic int8-conv + q8 text encoder engine: the transformer at rest
    /// (7.26 GB), the Qwen3-VL-8B Q8_0 LM on the device (10.53 GB measured:
    /// the 8.71 GB file with its quantized embedding replaced by F32), the VAE,
    /// and the calibrated cuDNN decode peak.
    fn quantized_engine(usable: u64, width: u32, height: u32) -> Qwen21TeBudget {
        let activation = crate::device::activation_bytes(
            width,
            height,
            1,
            2,
            crate::device::ActivationFamily::QwenImage21Dit,
        );
        Qwen21TeBudget {
            device: TeDevice::Cuda,
            usable_free_bytes: usable,
            transformer_bytes: 7_256_783_064,
            vae_bytes: 675_509_688,
            text_encoder_bytes: 10_531_655_680,
            reference_encoder_bytes: 0,
            encode_workspace_bytes: 0,
            denoise_workspace_bytes: activation,
            decode_peak_bytes: decode_peak(width, height),
            host_total_bytes: 64 * GIB,
            host_available_bytes: 48 * GIB,
            already_parked_bytes: 0,
            keep_te_ram: KeepTeRamMode::Auto,
        }
    }

    fn bf16_engine(usable: u64, width: u32, height: u32) -> Qwen21TeBudget {
        Qwen21TeBudget {
            transformer_bytes: 14_230_280_616,
            text_encoder_bytes: 15_136_811_008,
            host_total_bytes: 256 * GIB,
            host_available_bytes: 200 * GIB,
            ..quantized_engine(usable, width, height)
        }
    }

    /// A 24 GB card after the CUDA context (~22 GiB usable).
    const CARD_24GB: u64 = 22 * GIB;
    /// An L40S / 48 GB card (~44 GiB usable).
    const CARD_48GB: u64 = 44 * GIB;

    /// A 40 GB card (A100-40GB class, ~40 GiB usable).
    const CARD_40GB: u64 = 40 * GIB;
    /// A 32 GB card (~30 GiB usable).
    const CARD_32GB: u64 = 30 * GIB;

    /// 24 GB, 1024²: the q8 encoder (10.5 GB) beside a quantized transformer
    /// and the measured ~7.2 GB 1024² decode peak does not fit, so the encoder
    /// parks; the transformer stays through the decode and eager fits. A
    /// 32 GB card keeps everything resident.
    #[test]
    fn a_24gb_card_at_1024_parks_the_encoder_and_keeps_the_transformer() {
        for transformer_bytes in [7_256_783_064, 4_197_494_816] {
            let budget = Qwen21TeBudget {
                transformer_bytes,
                ..quantized_engine(CARD_24GB, 1024, 1024)
            };
            let decision = decide(&budget);
            assert_eq!(
                decision.residency,
                Qwen21TeResidency::ParkHost,
                "{decision:?}"
            );
            assert_eq!(decision.transformer_decode, TransformerDecode::Resident);
            assert!(decision.eager_peak_bytes <= CARD_24GB);
        }
        let decision = decide(&quantized_engine(CARD_32GB, 1024, 1024));
        assert_eq!(
            decision.residency,
            Qwen21TeResidency::Resident,
            "{decision:?}"
        );
        assert_eq!(decision.transformer_decode, TransformerDecode::Resident);
    }

    /// 24 GB at 2K: the encoder parks, and the ~27.5 GB decode cannot fit the
    /// card even with the transformer parked, so the eager peak honestly
    /// exceeds the card and the planner does not choose Eager.
    #[test]
    fn a_24gb_card_at_2k_parks_everything_and_is_not_eager_feasible() {
        for (width, height) in [(2048, 2048), (2752, 1536), (2400, 1792)] {
            let decision = decide(&quantized_engine(CARD_24GB, width, height));
            assert_eq!(
                decision.residency,
                Qwen21TeResidency::ParkHost,
                "{width}x{height}: {decision:?}"
            );
            assert_eq!(decision.transformer_decode, TransformerDecode::ParkHost);
            assert!(decision.eager_peak_bytes > CARD_24GB, "{width}x{height}");
        }
    }

    /// 48 GB BF16 at 1024²: 14.2 + 15.1 + 0.7 + max(activation, 7.2) + 1 GB
    /// fits, so nothing moves.
    #[test]
    fn a_48gb_card_keeps_bf16_everything_resident_at_1024() {
        let decision = decide(&bf16_engine(CARD_48GB, 1024, 1024));
        assert_eq!(
            decision.residency,
            Qwen21TeResidency::Resident,
            "{decision:?}"
        );
        assert_eq!(decision.transformer_decode, TransformerDecode::Resident);
    }

    /// 48 GB BF16 at 2K: transformer + encoder + the ~27.5 GB decode is ~59 GB,
    /// so the encoder parks; the transformer (14.2 GB) still fits beside the
    /// decode on an L40S and stays.
    #[test]
    fn a_48gb_card_at_2k_parks_the_encoder_only() {
        for (width, height) in [(2048, 2048), (2752, 1536)] {
            let budget = bf16_engine(CARD_48GB, width, height);
            assert!(budget.peak(true, true) > CARD_48GB);
            let decision = decide(&budget);
            assert_eq!(
                decision.residency,
                Qwen21TeResidency::ParkHost,
                "{width}x{height}: {decision:?}"
            );
            assert_eq!(decision.transformer_decode, TransformerDecode::Resident);
            assert!(decision.eager_peak_bytes <= CARD_48GB, "{width}x{height}");
        }
    }

    /// 40 GB BF16 at 2K: the transformer does not fit beside the decode
    /// either, so it parks to host for the decode and eager still fits.
    #[test]
    fn a_40gb_card_at_2k_also_parks_the_transformer_for_decode() {
        for (width, height) in [(2048, 2048), (2752, 1536)] {
            let budget = bf16_engine(CARD_40GB, width, height);
            assert!(budget.peak(false, true) > CARD_40GB);
            let decision = decide(&budget);
            assert_eq!(decision.residency, Qwen21TeResidency::ParkHost);
            assert_eq!(
                decision.transformer_decode,
                TransformerDecode::ParkHost,
                "{width}x{height}: {decision:?}"
            );
            assert!(decision.eager_peak_bytes <= CARD_40GB, "{width}x{height}");
        }
    }

    /// A BF16 engine on a `usable` card sized for `references` (source
    /// dimensions) and `branches`, through the SAME two functions the engine
    /// and the planner call, on the CUDA fast path's memory-following cache
    /// rule (asked explicitly so the test does not depend on the build's
    /// `cuda` feature).
    fn bf16_reference_engine(
        usable: u64,
        width: u32,
        height: u32,
        references: &[(u32, u32)],
        branches: usize,
    ) -> (Qwen21TeBudget, Qwen21RenderPhases) {
        let base = bf16_engine(usable, width, height);
        let request = Qwen21RenderRequest {
            format: None,
            width,
            height,
            batch: 1,
            references,
            branches,
            vae_dtype_bytes: 2,
        };
        let cache_budget =
            super::super::PrefixCacheBudget::Headroom(super::super::prefix_cache_headroom(
                usable,
                base.transformer_bytes + base.vae_bytes,
                crate::device::qwen_image21_planned_denoise_bytes(
                    None, width, height, references, 1, 2,
                ),
            ));
        let phases = render_phases(&request, cache_budget);
        (
            Qwen21TeBudget {
                reference_encoder_bytes: phases.reference_encoder_bytes,
                encode_workspace_bytes: phases.encode_workspace_bytes,
                denoise_workspace_bytes: phases.denoise_workspace_bytes,
                // The CUDA build's cuDNN decode curve, as every other case
                // here reads it (this test build may resolve im2col).
                ..base
            },
            phases,
        )
    }

    /// The phases run one after another, so the render needs the largest of
    /// them — never encode activations PLUS the denoise workspace, which was
    /// what parked the text encoder for every reference request on a 46 GB
    /// L40S although the measured peak with it resident was ~37.5 GB.
    #[test]
    fn the_budget_is_the_largest_phase_not_the_sum_of_phases() {
        let budget = Qwen21TeBudget {
            reference_encoder_bytes: 2 * GIB,
            encode_workspace_bytes: 8 * GIB,
            denoise_workspace_bytes: 8 * GIB,
            decode_peak_bytes: 4 * GIB,
            ..bf16_engine(CARD_48GB, 1024, 1024)
        };
        let resident = budget.transformer_bytes
            + budget.vae_bytes
            + budget.text_encoder_bytes
            + budget.reference_encoder_bytes;
        assert_eq!(
            budget.peak(true, true),
            resident + 8 * GIB + ALLOCATOR_MARGIN_BYTES
        );
        let decision = decide(&budget);
        assert_eq!(
            decision.residency,
            Qwen21TeResidency::Resident,
            "{decision:?}"
        );
        // Summing the two working sets would not have fit.
        assert!(resident + 16 * GIB + ALLOCATOR_MARGIN_BYTES > CARD_48GB);
    }

    /// 1024² with one reference, and with three under guidance and a negative
    /// prompt (two retained caches), keeps the BF16 encoder resident on a
    /// 46/48 GB card AND retains every prefix cache.
    #[test]
    fn a_48gb_card_keeps_the_encoder_for_reference_renders_at_1024() {
        for (references, branches) in [
            (vec![(1024, 1024)], 1),
            (vec![(1024, 1024)], 2),
            (vec![(1024, 1024); 3], 1),
            (vec![(1344, 768); 3], 1),
            (vec![(1024, 1024); 2], 2),
        ] {
            let (budget, phases) =
                bf16_reference_engine(CARD_48GB, 1024, 1024, &references, branches);
            let shape =
                crate::device::QwenImage21SequenceShape::for_request(1024, 1024, &references);
            let cache = crate::qwen_image21::prefix_cache_bytes(shape.prefix_tokens(), 1, 2)
                * branches as u64;
            assert!(
                phases.denoise_workspace_bytes > cache,
                "the cache is retained: {phases:?}"
            );
            let decision = decide(&budget);
            assert_eq!(
                decision.residency,
                Qwen21TeResidency::Resident,
                "{} refs x{branches}: {decision:?} {phases:?}",
                references.len()
            );
            assert_eq!(decision.transformer_decode, TransformerDecode::Resident);
            assert!(decision.eager_peak_bytes <= CARD_48GB);
        }
    }

    /// Three 1024² references under guidance and a negative prompt retain
    /// two ~5 GB prefix caches. Beside them the 15 GB encoder does not fit a
    /// 46/48 GB card (measured on an L40S: the denoise held ~30 GiB with the
    /// encoder parked, 45 GiB total), and a retained cache is worth ~8x the
    /// denoise against one park per request — so the encoder parks and both
    /// caches stay.
    #[test]
    fn a_48gb_card_prefers_both_caches_to_a_resident_encoder_for_three_references() {
        let references = vec![(1024, 1024); 3];
        let (budget, phases) = bf16_reference_engine(CARD_48GB, 1024, 1024, &references, 2);
        let shape = crate::device::QwenImage21SequenceShape::for_request(1024, 1024, &references);
        let caches = 2 * crate::qwen_image21::prefix_cache_bytes(shape.prefix_tokens(), 1, 2);
        assert!(phases.denoise_workspace_bytes > caches, "{phases:?}");
        let decision = decide(&budget);
        assert_eq!(
            decision.residency,
            Qwen21TeResidency::ParkHost,
            "{decision:?}"
        );
        assert!(decision.eager_peak_bytes <= CARD_48GB);
    }

    /// A 24 GB card still parks for a reference render, and every phase
    /// still charges the reference encoders the eager engine keeps loaded.
    #[test]
    fn a_24gb_card_still_parks_the_encoder_for_a_reference_render() {
        let (budget, phases) = bf16_reference_engine(CARD_24GB, 1024, 1024, &[(1024, 1024)], 1);
        assert!(phases.reference_encoder_bytes > 2 * GB);
        assert!(phases.encode_workspace_bytes > 0);
        let quantized = Qwen21TeBudget {
            transformer_bytes: 7_256_783_064,
            text_encoder_bytes: 10_531_655_680,
            ..budget
        };
        assert_ne!(decide(&quantized).residency, Qwen21TeResidency::Resident);
        // Text-to-image carries no reference phase at all.
        let (_, t2i) = bf16_reference_engine(CARD_24GB, 1024, 1024, &[], 1);
        assert_eq!(t2i.reference_encoder_bytes, 0);
        assert_eq!(t2i.encode_workspace_bytes, 0);
    }

    #[test]
    fn a_host_without_room_drops_and_never_forbids_a_park() {
        let tight_host = Qwen21TeBudget {
            host_total_bytes: 32 * GIB,
            host_available_bytes: 12 * GIB,
            ..quantized_engine(CARD_24GB, 1024, 1024)
        };
        assert_eq!(decide(&tight_host).residency, Qwen21TeResidency::Drop);
        // Force asks only for the encoder's own room above the floor.
        let forced = Qwen21TeBudget {
            host_available_bytes: 20 * GIB,
            keep_te_ram: KeepTeRamMode::Force,
            ..tight_host
        };
        assert_eq!(decide(&forced).residency, Qwen21TeResidency::ParkHost);
        let never = Qwen21TeBudget {
            keep_te_ram: KeepTeRamMode::Never,
            ..quantized_engine(CARD_24GB, 1024, 1024)
        };
        let decision = decide(&never);
        assert_eq!(decision.residency, Qwen21TeResidency::Drop);
        assert!(decision.reason.contains("MOLD_KEEP_TE_RAM=0"));
        // A host that cannot take the transformer either releases it.
        let no_room = Qwen21TeBudget {
            host_total_bytes: 32 * GIB,
            host_available_bytes: 10 * GIB,
            ..bf16_engine(CARD_40GB, 2048, 2048)
        };
        let decision = decide(&no_room);
        assert_eq!(decision.residency, Qwen21TeResidency::Drop);
        assert_eq!(decision.transformer_decode, TransformerDecode::Drop);
    }

    #[test]
    fn a_warm_park_is_credited_back_so_it_does_not_flap() {
        let cold = Qwen21TeBudget {
            host_total_bytes: 48 * GIB,
            host_available_bytes: 26 * GIB,
            ..quantized_engine(CARD_24GB, 1024, 1024)
        };
        assert_eq!(decide(&cold).residency, Qwen21TeResidency::ParkHost);
        // The park now holds ~10.5 GB, which MemAvailable no longer shows.
        let warm = Qwen21TeBudget {
            host_available_bytes: 26 * GIB - cold.text_encoder_bytes,
            already_parked_bytes: cold.text_encoder_bytes,
            ..cold
        };
        assert_eq!(decide(&warm).residency, Qwen21TeResidency::ParkHost);
    }

    #[test]
    fn metal_cpu_and_unmeasurable_cards_keep_todays_behaviour() {
        let metal = Qwen21TeBudget {
            device: TeDevice::Metal,
            ..quantized_engine(8 * GIB, 2048, 2048)
        };
        let decision = decide(&metal);
        assert_eq!(decision.residency, Qwen21TeResidency::Resident);
        assert_eq!(decision.transformer_decode, TransformerDecode::Resident);
        // A host-placed encoder is not charged to the card at all.
        let cpu = Qwen21TeBudget {
            device: TeDevice::Cpu,
            ..quantized_engine(CARD_24GB, 1024, 1024)
        };
        let decision = decide(&cpu);
        assert_eq!(decision.residency, Qwen21TeResidency::Resident);
        assert_eq!(decision.transformer_decode, TransformerDecode::Resident);
        assert!(decision.eager_peak_bytes < cpu.peak(true, true));
        assert_eq!(
            decide(&quantized_engine(0, 2048, 2048)).residency,
            Qwen21TeResidency::Resident
        );
    }

    /// The workspace both sides charge reads C's calibrated decode curve.
    #[test]
    fn the_render_workspace_uses_the_calibrated_decode_peak() {
        let (_, decode) = render_workspace_bytes(None, 2048, 2048, 1, 2);
        let conv =
            crate::conv_policy::resolve_for(crate::conv_policy::policy_for_family("qwen-image21"));
        assert_eq!(
            decode,
            crate::device::qwen_image21_vae_decode_peak_bytes(2048, 2048, conv, 2)
        );
        assert!(decode > 20 * GB);
    }

    /// The int8-conv tier's W8A8 activation workspace rides on the denoise
    /// term (it was measured 2.2 GB above bf16 at 2K); every other tier's
    /// denoise workspace is the BF16 estimate.
    #[test]
    fn the_render_workspace_charges_the_int8_tier_its_linear_workspace() {
        use crate::artifact_format::QwenImage21TransformerFormat as Format;
        let (bf16, decode) = render_workspace_bytes(Some(Format::Bf16), 2048, 2048, 1, 2);
        let (none, _) = render_workspace_bytes(None, 2048, 2048, 1, 2);
        let (int8, int8_decode) =
            render_workspace_bytes(Some(Format::ComfyInt8ConvRot), 2048, 2048, 1, 2);
        assert_eq!(bf16, none);
        assert_eq!(decode, int8_decode);
        let joint = 2048 * 2048 / 256 + super::super::LEGACY_PREFIX_CACHE_TOKENS as u64;
        assert_eq!(
            int8 - bf16,
            crate::device::qwen_image21_linear_workspace_bytes(
                Some(Format::ComfyInt8ConvRot),
                joint,
                1
            )
        );
        assert!(int8 - bf16 > GB);
    }

    #[test]
    fn gguf_text_encoder_bytes_price_the_embedding_as_f32() {
        use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
        use candle_core::{Device, Tensor};
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("te.gguf");
        let embd = QTensor::quantize(
            &Tensor::zeros((8, 256), candle_core::DType::F32, &Device::Cpu).unwrap(),
            GgmlDType::Q8_0,
        )
        .unwrap();
        let q = QTensor::quantize(
            &Tensor::zeros((4, 256), candle_core::DType::F32, &Device::Cpu).unwrap(),
            GgmlDType::Q8_0,
        )
        .unwrap();
        let mut file = std::fs::File::create(&path).unwrap();
        gguf_file::write(
            &mut file,
            &[],
            &[("token_embd.weight", &embd), ("blk.0.attn_q.weight", &q)],
        )
        .unwrap();
        drop(file);
        // Q8_0: 34 bytes per 32 values.
        assert_eq!(
            text_encoder_device_bytes(&[path]).unwrap(),
            8 * 256 * 4 + 4 * 256 / 32 * 34
        );
    }

    #[test]
    fn safetensors_text_encoder_bytes_count_only_the_language_model() {
        use candle_core::{DType, Device, Tensor};
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("te.safetensors");
        let tensors = std::collections::HashMap::from([
            (
                "model.language_model.layers.0.mlp.up_proj.weight".to_string(),
                Tensor::zeros((4, 8), DType::BF16, &Device::Cpu).unwrap(),
            ),
            (
                "model.visual.blocks.0.attn.qkv.weight".to_string(),
                Tensor::zeros((100, 100), DType::BF16, &Device::Cpu).unwrap(),
            ),
            (
                "lm_head.weight".to_string(),
                Tensor::zeros((100, 8), DType::BF16, &Device::Cpu).unwrap(),
            ),
        ]);
        candle_core::safetensors::save(&tensors, &path).unwrap();
        assert_eq!(text_encoder_device_bytes(&[path]).unwrap(), 4 * 8 * 2);
    }

    /// The real files, when staged: the q8 LM is ~10.5 GB on the device and
    /// the BF16 LM ~16.4 GB.
    #[test]
    #[ignore = "needs the staged Qwen3-VL-8B GGUF and the Qwen Image 2.1 text-encoder shards"]
    fn the_real_text_encoders_price_as_measured() {
        let tiers = PathBuf::from(std::env::var("MOLD_QWEN_IMAGE21_TIERS_DIR").unwrap());
        let q8 = text_encoder_device_bytes(&[tiers.join("Qwen3VL-8B-Instruct-Q8_0.gguf")]).unwrap();
        let q4 =
            text_encoder_device_bytes(&[tiers.join("Qwen3VL-8B-Instruct-Q4_K_M.gguf")]).unwrap();
        let shared = PathBuf::from(std::env::var("MOLD_QWEN_IMAGE21_SHARED_DIR").unwrap());
        let bf16 = text_encoder_device_bytes(
            &(1..=4)
                .map(|i| shared.join(format!("text_encoder/model-0000{i}-of-00004.safetensors")))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        eprintln!("TE-BYTES q8 {q8} q4 {q4} bf16 {bf16}");
        assert!((10 * GB..11 * GB).contains(&q8), "{q8}");
        assert!((7 * GB..8 * GB).contains(&q4), "{q4}");
        assert!((15 * GB..16 * GB).contains(&bf16), "{bf16}");
    }
}
