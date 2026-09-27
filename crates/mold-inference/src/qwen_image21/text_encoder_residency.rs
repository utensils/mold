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

    /// Prompt encode: transformer, VAE and the encoder all on the card.
    fn encode_phase(&self) -> u64 {
        self.transformer_bytes
            .saturating_add(self.vae_bytes)
            .saturating_add(self.text_encoder_bytes)
            .saturating_add(ALLOCATOR_MARGIN_BYTES)
    }

    fn denoise_phase(&self, te: bool) -> u64 {
        self.transformer_bytes
            .saturating_add(self.vae_bytes)
            .saturating_add(self.te(te))
            .saturating_add(self.denoise_workspace_bytes)
            .saturating_add(ALLOCATOR_MARGIN_BYTES)
    }

    fn decode_phase(&self, te: bool, transformer: bool) -> u64 {
        self.transformer(transformer)
            .saturating_add(self.vae_bytes)
            .saturating_add(self.te(te))
            .saturating_add(self.decode_peak_bytes)
            .saturating_add(ALLOCATOR_MARGIN_BYTES)
    }

    /// Every phase's peak under one placement. With everything resident this
    /// is `transformer + VAE + TE + max(activation, decode)`.
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
    pub denoise_workspace_bytes: u64,
    pub decode_peak_bytes: u64,
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
        denoise_workspace_bytes: inputs.denoise_workspace_bytes,
        decode_peak_bytes: inputs.decode_peak_bytes,
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
pub fn render_workspace_bytes(
    width: u32,
    height: u32,
    batch: u32,
    vae_dtype_bytes: u32,
) -> (u64, u64) {
    let denoise = crate::device::activation_bytes(
        width,
        height,
        batch,
        2,
        crate::device::ActivationFamily::QwenImage21Dit,
    );
    let conv =
        crate::conv_policy::resolve_for(crate::conv_policy::policy_for_family("qwen-image21"));
    (
        denoise,
        crate::device::qwen_image21_vae_decode_peak_bytes(width, height, conv, vae_dtype_bytes),
    )
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
        let (_, decode) = render_workspace_bytes(2048, 2048, 1, 2);
        let conv =
            crate::conv_policy::resolve_for(crate::conv_policy::policy_for_family("qwen-image21"));
        assert_eq!(
            decode,
            crate::device::qwen_image21_vae_decode_peak_bytes(2048, 2048, conv, 2)
        );
        assert!(decode > 20 * GB);
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
