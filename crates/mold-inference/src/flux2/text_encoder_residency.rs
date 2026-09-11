//! Residency arithmetic for FLUX.2 [dev]'s streamed Mistral3 text encoder.
//!
//! This module is the SINGLE authority on what a Mistral3 conditioner costs
//! while it runs. Admission, the placement planner, and the engine all read it
//! rather than pricing the encoder from its file length.
//!
//! The distinction matters because the encoder never materializes its
//! checkpoint. `encoders::mistral3::Mistral3Encoder::encode` builds a
//! memory-mapped `VarBuilder` over the shards and then constructs ONE decoder
//! layer at a time, dropping it before the next is built — exactly the shape
//! `ltx2::cpu_gemma_streaming_anon_peak_bytes` prices for LTX-2's Gemma. The
//! shard pages are file-backed and reclaimable; only the embedding table, the
//! layers in flight, and the retained hidden states are real allocations.
//!
//! Charging the file instead is not conservatism, it is a wrong answer with
//! two measured consequences on plato (4x L40S 46 GB):
//!
//! * the 36 GB encoder file made `eager_peak` exceed 90 % of a 46 GB card, so
//!   the planner declared memory pressure and auto-parked the encoder to the
//!   CPU, where FLUX.2 picks F32 and the encode took 78.8 seconds with the GPU
//!   at 0 % SM for its entire duration; and
//! * the CPU arm then reserved those same 36 GB as irreclaimable HOST RAM,
//!   which is an outright refusal on a 64 GB desktop.
//!
//! The geometry below is BFL's `Mistral-Small-3.2-24B` language model, mirrored
//! from the constants `crate::encoders::mistral3` actually constructs. The
//! prefix is layers 0..=29 plus the token embedding, because FLUX conditioning
//! reads `hidden_states[10 | 20 | 30]`
//! (`tmp/flux2-upstream/src/flux2/text_encoder.py:26`, `:230-232`) — index `k`
//! is the output of decoder layer `k - 1`, so layer 29 is the last one that
//! runs and layers 30-39, the final norm, the vision tower, the projector, and
//! the LM head are never touched.
//!
//! WP6's later host-residency commit adds its decision function here; the
//! arithmetic below is deliberately free of any policy so both can share it.

use candle_core::DType;
use std::path::Path;

/// Decoder layers the streamed prefix covers: 0..=29.
///
/// `crate::encoders::mistral3::CAPTURE_LAYERS` ends at 29 because BFL reads
/// `hidden_states[30]`, the output of layer 29
/// (`tmp/flux2-upstream/src/flux2/text_encoder.py:26`).
pub const MISTRAL3_PREFIX_LAYERS: u64 = 30;

/// Decoder layers held on the device at once while the encoder streams.
///
/// One layer is running while the next is being uploaded. `lookahead = 0` is
/// the strictly serial loader; `lookahead = 1` is the prefetching one.
pub const MISTRAL3_DEFAULT_LOOKAHEAD: u64 = 1;

/// Token positions the encoder always runs. BFL pads to `max_length`
/// unconditionally (`text_encoder.py:28`, `:247`), so this is not a bound on
/// the prompt — it is the exact shape of every encode.
const MAX_LENGTH: u64 = 512;

/// Hidden states alive at the peak: the three captured tensors plus the one
/// being produced.
const RETAINED_HIDDEN_STATES: u64 = 4;

const VOCAB_SIZE: u64 = 131_072;
const HIDDEN_SIZE: u64 = 5_120;
const INTERMEDIATE_SIZE: u64 = 32_768;
const NUM_ATTENTION_HEADS: u64 = 32;
const NUM_KV_HEADS: u64 = 8;
const HEAD_DIM: u64 = 128;

fn dtype_bytes(dtype: DType) -> u64 {
    crate::device::dtype_bytes(dtype) as u64
}

/// Parameters in one Mistral3 decoder layer, from the shapes
/// `encoders::mistral3::DecoderLayer::new` actually constructs.
fn layer_elements() -> u64 {
    let attention_dim = NUM_ATTENTION_HEADS * HEAD_DIM;
    let kv_dim = NUM_KV_HEADS * HEAD_DIM;
    // q_proj and o_proj span the full attention width; k_proj and v_proj are
    // grouped-query and span the smaller key/value width. None carries a bias.
    let attention = HIDDEN_SIZE * attention_dim * 2 + HIDDEN_SIZE * kv_dim * 2;
    // gate_proj, up_proj, down_proj.
    let mlp = HIDDEN_SIZE * INTERMEDIATE_SIZE * 3;
    // input_layernorm and post_attention_layernorm.
    let layernorms = HIDDEN_SIZE * 2;
    attention + mlp + layernorms
}

/// Bytes one streamed Mistral3 decoder layer occupies at `dtype`.
pub fn mistral3_layer_bytes(dtype: DType) -> u64 {
    layer_elements().saturating_mul(dtype_bytes(dtype))
}

/// Bytes the token embedding table occupies at `dtype`.
///
/// This is the largest single allocation in the encoder and the figure
/// `encoders::mistral3::streamed_peak_weight_bytes` has always reported.
pub fn mistral3_embed_bytes(dtype: DType) -> u64 {
    VOCAB_SIZE
        .saturating_mul(HIDDEN_SIZE)
        .saturating_mul(dtype_bytes(dtype))
}

/// Bytes the WHOLE streamed prefix would occupy if it were resident at once:
/// the embedding table plus every decoder layer the encoder runs.
///
/// Nothing charges this as a peak — the encoder never holds it — but it is the
/// figure a host-residency decision has to fit when it parks the prefix in RAM
/// instead of streaming it, so it belongs beside the streaming arithmetic.
pub fn mistral3_prefix_bytes(dtype: DType) -> u64 {
    mistral3_embed_bytes(dtype)
        .saturating_add(MISTRAL3_PREFIX_LAYERS.saturating_mul(mistral3_layer_bytes(dtype)))
}

/// Peak bytes the streamed encoder allocates on its target device.
///
/// The embedding table is charged alongside the layers rather than against
/// them: the loader is free to begin prefetching layer 0 while the embedding
/// lookup is still running, and an admission figure that assumed otherwise
/// would be a floor rather than a peak.
pub fn mistral3_streamed_device_peak_bytes(dtype: DType, lookahead: u64) -> u64 {
    let layers_in_flight = lookahead.saturating_add(1);
    let hidden = RETAINED_HIDDEN_STATES
        .saturating_mul(MAX_LENGTH)
        .saturating_mul(HIDDEN_SIZE)
        .saturating_mul(dtype_bytes(dtype));
    mistral3_embed_bytes(dtype)
        .saturating_add(layers_in_flight.saturating_mul(mistral3_layer_bytes(dtype)))
        .saturating_add(hidden)
}

/// [`mistral3_prefix_bytes`] at the dtype a GPU-placed encoder runs at.
///
/// The entry point for consumers outside this crate, for the same reason
/// [`mistral3_admission_charge_for_gpu`] is one: `mold-server` names no candle
/// type by design, and the dtype is the engine's choice rather than the
/// planner's.
pub fn mistral3_prefix_bytes_bf16() -> u64 {
    mistral3_prefix_bytes(DType::BF16)
}

/// The page-locked host cap, re-exported so a planner can ask the residency
/// decision the same question the engine asks without reaching into
/// `flux::pinned` — which is otherwise a FLUX.1 block-offload detail.
pub fn host_pinned_cap_bytes() -> u64 {
    crate::flux::pinned::pinned_cap_bytes()
}

/// Whether a text-encoder artifact is a FLUX.2 [dev] Mistral3 shard the
/// encoder STREAMS rather than materializes.
///
/// Coordinator-safe: a name test and a path extension, never a file read. The
/// family gate keeps Klein (Qwen3, materialized) and every other family on
/// their file-size charge, and the `.safetensors` gate keeps a hypothetical
/// quantized republication — which would take a different loader with its own
/// residency — out of this arithmetic.
pub fn mistral3_streams_from_mmap(family: &str, model: &str, path: &Path) -> bool {
    matches!(family, "flux2" | "flux.2")
        && mold_core::validation::is_flux2_dev_model(model)
        && path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"))
}

/// What admission charges for a streamed Mistral3 encoder, on each side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamedEncoderCharge {
    /// Device bytes while the encoder runs on the GPU at the engine's GPU
    /// dtype.
    pub device_peak: u64,
    /// Irreclaimable HOST bytes when the encoder is parked on the CPU. The
    /// shards stay a reclaimable mapping there too, so this is the same
    /// streaming heap priced at the F32 the CPU arm of
    /// `flux2::pipeline` selects — never the file length.
    pub host_anon_peak: u64,
}

/// The admission charge for `model`'s text encoders, or `None` when this is
/// not a streamed Mistral3 conditioner and the file-size estimate stands.
///
/// `dtype` is the GPU dtype the engine would run the encoder at. Every
/// `text_encoder_files` entry must look like a Mistral3 shard, because the
/// charge replaces the encoder phase as a whole.
pub fn mistral3_admission_charge(
    model: &str,
    paths: &mold_core::ModelPaths,
    dtype: DType,
) -> Option<StreamedEncoderCharge> {
    if !mold_core::validation::is_flux2_dev_model(model) {
        return None;
    }
    if paths.text_encoder_files.is_empty()
        || !paths
            .text_encoder_files
            .iter()
            .all(|path| mistral3_streams_from_mmap("flux2", model, path))
    {
        return None;
    }
    Some(StreamedEncoderCharge {
        device_peak: mistral3_streamed_device_peak_bytes(dtype, MISTRAL3_DEFAULT_LOOKAHEAD),
        host_anon_peak: mistral3_streamed_device_peak_bytes(DType::F32, MISTRAL3_DEFAULT_LOOKAHEAD),
    })
}

/// [`mistral3_admission_charge`] at the dtype FLUX.2 actually runs a
/// GPU-placed encoder at.
///
/// `crate::engine::gpu_dtype` answers BF16 on CUDA and Metal (F16 would price
/// identically), and `flux2::pipeline` hands the encoder that value. This is
/// the entry point for consumers outside this crate — `mold-server` names no
/// candle type by design, and the dtype is the engine's choice rather than the
/// planner's.
pub fn mistral3_admission_charge_for_gpu(
    model: &str,
    paths: &mold_core::ModelPaths,
) -> Option<StreamedEncoderCharge> {
    mistral3_admission_charge(model, paths, DType::BF16)
}

// ── Host residency ───────────────────────────────────────────────────────────

/// The accelerator a text encoder would run on.
///
/// Only three answers matter to this decision and none of them is a candle
/// type, so the planner can ask the same question the engine asks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextEncoderDevice {
    /// Discrete VRAM with a real host-to-device copy — the only case where
    /// holding weights in host RAM buys anything.
    Cuda,
    /// Unified memory. "Parking in host RAM" is where the weights already are,
    /// so it saves no copy and spends the same budget twice.
    Metal,
    /// The encoder is running on the host already.
    Cpu,
}

/// What `MOLD_KEEP_TE_RAM` asks for.
///
/// Tri-state because the variable now answers two different questions. It has
/// always been the opt-in that keeps FLUX's T5, SD3's and Wan's encoders in
/// host RAM ([`crate::device::keep_te_in_ram`] is exactly `Force`, so those
/// families are untouched), and it is now also the OVERRIDE on a decision
/// that has a real default. `Auto` is that default and is what an unset
/// variable means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeepTeRamMode {
    /// Decide from the host's own memory. Unset, or any value that is neither
    /// `1` nor `0`.
    Auto,
    /// `MOLD_KEEP_TE_RAM=1` — park whenever the encoder itself fits above the
    /// floor, without asking whether the transformer would also fit.
    Force,
    /// `MOLD_KEEP_TE_RAM=0` — never park.
    Never,
}

/// Everything the residency decision reads. Bytes throughout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextEncoderResidencyInputs {
    /// Host bytes the parked prefix would occupy — for Mistral3,
    /// [`mistral3_prefix_bytes`] at the encoder's own dtype.
    pub encoder_bytes: u64,
    /// Host bytes the TRANSFORMER's own load competes for: its mapping's page
    /// cache, and on a LoRA request the merge's staging copies.
    pub transformer_bytes: u64,
    /// Total host RAM.
    pub host_total_bytes: u64,
    /// Host RAM available right now — `MemAvailable`, plus whatever credit the
    /// caller's own ledger already applies.
    pub host_available_bytes: u64,
    /// Soft cap on page-locked host memory, from
    /// [`crate::flux::pinned::pinned_cap_bytes`].
    pub pinned_cap_bytes: u64,
    pub keep_te_ram: KeepTeRamMode,
    pub device: TextEncoderDevice,
    /// Bytes THIS engine's park is already holding, or zero.
    ///
    /// Credited back before the comparison, exactly as the VRAM side adds a
    /// resident transformer's bytes back. `host_available_bytes` is live
    /// `MemAvailable`, which already excludes an existing park, so without
    /// this the warm question becomes "can I park a SECOND copy" — the answer
    /// flips to no, the prefix is released, `MemAvailable` recovers, and the
    /// next request parks again. A ~4 s host copy on alternating requests
    /// forever is worse than never parking.
    pub already_parked_bytes: u64,
}

/// Where a text encoder's weights live between requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextEncoderResidency {
    /// Build each layer from the memory mapping, as the streaming loader
    /// always has. The shard pages are file-backed and reclaimable, so this
    /// costs the host nothing it cannot get back.
    StreamFromMmap,
    /// Hold the prefix in host RAM across requests.
    HostParked {
        /// Whether the host allocation is page-locked for fast async H2D.
        pinned: bool,
    },
}

impl TextEncoderResidency {
    pub fn parks(self) -> bool {
        matches!(self, Self::HostParked { .. })
    }

    pub fn is_pinned(self) -> bool {
        matches!(self, Self::HostParked { pinned: true })
    }
}

/// The host-memory safety floor: 15 % of the machine, never below 8 GiB.
///
/// This is the scheduler's OWN floor (`h3_admission::H3HostMemory::
/// safety_floor_bytes`), reproduced here rather than imported because
/// `mold-inference` cannot depend on `mold-server` and because the engine has
/// to be able to answer this question with no scheduler in the process at all
/// — the forced-local CLI path. The arithmetic is identical, including the
/// remainder term that keeps it exact on a total that is not a multiple of
/// 100.
pub fn host_safety_floor_bytes(host_total_bytes: u64) -> u64 {
    const GIB: u64 = 1024 * 1024 * 1024;
    let whole = host_total_bytes / 100;
    let remainder = host_total_bytes % 100;
    whole
        .saturating_mul(15)
        .saturating_add(remainder.saturating_mul(15) / 100)
        .max(8 * GIB)
}

/// Whether a text encoder's weights should live in host RAM between requests.
///
/// The decision is BINARY, not partial, and that is the whole shape of it:
/// parking P of the prefix's Q bytes buys nothing, because the remaining
/// `Q - P` still has to be built from the mapping every request and the P is
/// now unavailable to the page cache that was serving it. So either the whole
/// prefix fits above the floor or none of it does.
///
/// * **Metal and CPU stream.** Unified memory means the "parked" copy is in
///   the same pool the encoder runs from, so the park is a second allocation
///   of memory that was already reachable, charged against a budget the Metal
///   policy is already tight on. On the CPU the weights are already host-side.
/// * **`Never` streams**, and is the documented way to get today's behaviour
///   back on a host where parking turns out to be wrong.
/// * **`Force` parks whenever the encoder alone fits above the floor.** It is
///   an operator saying "I know this machine"; it still respects the floor,
///   because the floor is what keeps the process from being OOM-killed rather
///   than merely slow.
/// * **`Auto` also requires room for the TRANSFORMER.** The transformer's own
///   load wants host RAM at the same time — its mapping's pages, and on a LoRA
///   request the merge's staging copies — and the campaign's rule is that no
///   residency decision may make a smaller machine worse. A 64 GB desktop
///   therefore streams, a 128 GB host parks one engine's prefix, and plato's
///   1.5 TB parks and pins.
///
/// `pinned` is decided last and never widens the park: page-locked memory is
/// a scarcer resource than host RAM (it cannot be paged out at all), so it is
/// granted only when the prefix fits the pinned cap AND the host still has the
/// floor's worth of room left after the park.
pub fn decide_text_encoder_residency(inputs: &TextEncoderResidencyInputs) -> TextEncoderResidency {
    if matches!(
        inputs.device,
        TextEncoderDevice::Metal | TextEncoderDevice::Cpu
    ) {
        return TextEncoderResidency::StreamFromMmap;
    }
    if inputs.keep_te_ram == KeepTeRamMode::Never {
        return TextEncoderResidency::StreamFromMmap;
    }
    if inputs.encoder_bytes == 0 || inputs.host_total_bytes == 0 {
        return TextEncoderResidency::StreamFromMmap;
    }

    let floor = host_safety_floor_bytes(inputs.host_total_bytes);
    let required = match inputs.keep_te_ram {
        KeepTeRamMode::Force => inputs.encoder_bytes.saturating_add(floor),
        KeepTeRamMode::Auto => inputs
            .encoder_bytes
            .saturating_add(inputs.transformer_bytes)
            .saturating_add(floor),
        KeepTeRamMode::Never => unreachable!("handled above"),
    };
    // The budget is asked against the host as if this engine's own park were
    // not on it — the `usable_free` convention the VRAM side uses, and the
    // only way a warm engine asks the same question it answered cold.
    let spendable = inputs
        .host_available_bytes
        .saturating_add(inputs.already_parked_bytes);
    if spendable < required {
        return TextEncoderResidency::StreamFromMmap;
    }

    // Strictly above the floor, not merely at it: page-locked pages cannot be
    // paged out at all, so a park that lands exactly on the safety floor is
    // the one that must not also make its bytes unreclaimable.
    let pinned = inputs.encoder_bytes <= inputs.pinned_cap_bytes
        && spendable.saturating_sub(inputs.encoder_bytes) > floor;
    TextEncoderResidency::HostParked { pinned }
}

/// Whether a materialized Qwen3 encoder should stay in host RAM between
/// requests.
///
/// The shared decision, asked with the encoder's own on-disk size — Qwen3 is
/// MATERIALIZED, unlike the streamed Mistral3, so its file length IS what a
/// park would hold, in either precision and for GGUF as well as BF16.
///
/// Z-Image asks the identical question through this function's twin in its own
/// pipeline; both read `text_encoder_decide_text_encoder_residency`
/// so a host can never take opposite decisions for two encoders of the same
/// size.
pub fn qwen3_park_residency(
    device: &candle_core::Device,
    encoder_paths: &[std::path::PathBuf],
    transformer_bytes: u64,
    already_parked_bytes: u64,
) -> TextEncoderResidency {
    let encoder_bytes: u64 = encoder_paths
        .iter()
        .filter_map(|path| std::fs::metadata(path).ok().map(|metadata| metadata.len()))
        .sum();
    let device_class = if device.is_metal() {
        TextEncoderDevice::Metal
    } else if device.is_cuda() {
        TextEncoderDevice::Cuda
    } else {
        TextEncoderDevice::Cpu
    };
    decide_text_encoder_residency(&TextEncoderResidencyInputs {
        encoder_bytes,
        transformer_bytes,
        host_total_bytes: crate::flux::pinned::total_system_ram_bytes().unwrap_or(0),
        host_available_bytes: crate::device::available_host_ram_bytes().unwrap_or(0),
        pinned_cap_bytes: crate::flux::pinned::pinned_cap_bytes(),
        keep_te_ram: crate::device::keep_te_ram_mode(),
        device: device_class,
        already_parked_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const GIB: u64 = 1024 * 1024 * 1024;
    /// Decimal GB — the unit a machine's RAM is advertised in, and the one the
    /// plan's matrix rows are written in.
    const GB: u64 = 1_000_000_000;

    fn inputs(
        host_total: u64,
        host_available: u64,
        transformer: u64,
        keep_te_ram: KeepTeRamMode,
        device: TextEncoderDevice,
    ) -> TextEncoderResidencyInputs {
        TextEncoderResidencyInputs {
            encoder_bytes: mistral3_prefix_bytes(DType::BF16),
            transformer_bytes: transformer,
            host_total_bytes: host_total,
            host_available_bytes: host_available,
            // Linux's default: half the machine.
            pinned_cap_bytes: host_total / 2,
            keep_te_ram,
            device,
            already_parked_bytes: 0,
        }
    }

    /// The floor is the scheduler's own, reproduced rather than imported.
    #[test]
    fn the_host_floor_is_fifteen_percent_or_eight_gibibytes() {
        let fifteen_percent = |total: u64| (u128::from(total) * 15 / 100) as u64;
        assert_eq!(
            host_safety_floor_bytes(32 * GB),
            8 * GIB,
            "15 % of a 32 GB machine is under the 8 GiB minimum"
        );
        for total in [64 * GB, 128 * GB, 1_500 * GB] {
            assert_eq!(host_safety_floor_bytes(total), fifteen_percent(total));
        }
        assert_eq!(
            host_safety_floor_bytes(0),
            8 * GIB,
            "an unmeasured host still floors"
        );
    }

    /// How many of `gpus` workers park, each asking the same question after
    /// the ones before it have taken their bytes.
    fn parks_across(
        gpus: usize,
        host_total: u64,
        host_available: u64,
        mode: KeepTeRamMode,
    ) -> usize {
        const TRANSFORMER: u64 = 33_000_000_000;
        let mut available = host_available;
        let mut parked = 0;
        for _ in 0..gpus {
            let decision = decide_text_encoder_residency(&inputs(
                host_total,
                available,
                TRANSFORMER,
                mode,
                TextEncoderDevice::Cuda,
            ));
            if !decision.parks() {
                break;
            }
            parked += 1;
            available = available.saturating_sub(mistral3_prefix_bytes(DType::BF16));
        }
        parked
    }

    /// The matrix the campaign's "no regression on a smaller machine" rule is
    /// measured against: host RAM against the number of engines that might
    /// want to park, plus the override and device rows.
    ///
    /// The prefix is ~34.7 GB at BF16 and the transformer term is a 33 GB Q8
    /// FLUX.2 [dev]. Each GPU's worker asks the SAME question after the ones
    /// before it have taken their bytes, so what the matrix pins is how many
    /// of them the host can afford — never a per-machine constant.
    #[test]
    fn residency_matrix_over_host_ram_and_gpu_count() {
        // 32 GB desktop: the floor alone is 8 GiB and the prefix is 34.7 GB.
        // Nothing fits, under any mode, at any GPU count.
        for mode in [KeepTeRamMode::Auto, KeepTeRamMode::Force] {
            for gpus in [1, 4] {
                assert_eq!(
                    parks_across(gpus, 32 * GB, 30 * GB, mode),
                    0,
                    "a 32 GB desktop cannot hold the prefix at all ({mode:?}, {gpus} GPUs)"
                );
            }
        }

        // 64 GB desktop with 58 GB free: the prefix plus the floor fits, but
        // the prefix plus the TRANSFORMER plus the floor does not — so Auto
        // streams and an operator who insists gets exactly one park.
        assert_eq!(
            parks_across(4, 64 * GB, 58 * GB, KeepTeRamMode::Auto),
            0,
            "a 64 GB desktop must stream: this is the no-regression rule"
        );
        assert_eq!(
            parks_across(4, 64 * GB, 58 * GB, KeepTeRamMode::Force),
            1,
            "an explicit keep is an operator saying they know the machine, and \
             it still stops at the floor"
        );

        // 128 GB host: some park and not all four. The exact count is the
        // host's business; what must hold is that a four-GPU box does not
        // reserve four prefixes on a machine this size.
        let mid = parks_across(4, 128 * GB, 120 * GB, KeepTeRamMode::Auto);
        assert!(
            (1..4).contains(&mid),
            "a 128 GB host parks some engines and not every one; parked {mid}"
        );

        // plato: 1.5 TB, four GPUs. Every one of the four parks, and pins.
        assert_eq!(
            parks_across(4, 1_500 * GB, 1_400 * GB, KeepTeRamMode::Auto),
            4
        );
        assert_eq!(
            decide_text_encoder_residency(&inputs(
                1_500 * GB,
                1_400 * GB,
                33_000_000_000,
                KeepTeRamMode::Auto,
                TextEncoderDevice::Cuda
            )),
            TextEncoderResidency::HostParked { pinned: true },
            "1.5 TB is the machine UAT runs on: parked and pinned"
        );

        // `Never` streams everywhere, whatever the machine.
        assert_eq!(
            parks_across(4, 1_500 * GB, 1_400 * GB, KeepTeRamMode::Never),
            0
        );

        // Metal and the CPU stream on the largest host there is: unified
        // memory means the park is a second allocation of memory that was
        // already reachable.
        for device in [TextEncoderDevice::Metal, TextEncoderDevice::Cpu] {
            for mode in [KeepTeRamMode::Auto, KeepTeRamMode::Force] {
                assert_eq!(
                    decide_text_encoder_residency(&inputs(
                        1_500 * GB,
                        1_400 * GB,
                        33_000_000_000,
                        mode,
                        device
                    )),
                    TextEncoderResidency::StreamFromMmap,
                    "{device:?} / {mode:?}"
                );
            }
        }
    }

    /// The same engine asked twice must give the same answer.
    ///
    /// `host_available_bytes` is live `MemAvailable`, which ALREADY excludes
    /// the bytes this engine parked last request — so without a credit the
    /// warm question is really "can I park a SECOND copy", the answer flips to
    /// no, the prefix is unparked, `MemAvailable` recovers, and the next
    /// request parks again. That is a ~4 s host copy on alternating requests
    /// forever, which is worse than never parking at all.
    ///
    /// The VRAM side has had this right from the start — the resident
    /// transformer's bytes are added back before the comparison — and
    /// `parks_across` could not catch it because it models DISTINCT engines
    /// taking one decision each.
    #[test]
    fn an_engine_that_already_parked_does_not_unpark_itself() {
        const TRANSFORMER: u64 = 33_000_000_000;
        let prefix = mistral3_prefix_bytes(DType::BF16);

        // The peer review's worked example: 128 GB host, 120 GiB available,
        // flux2-dev Q8. The cold decision parks.
        let cold = inputs(
            128 * GIB,
            120 * GIB,
            TRANSFORMER,
            KeepTeRamMode::Auto,
            TextEncoderDevice::Cuda,
        );
        assert!(decide_text_encoder_residency(&cold).parks());

        // Now the same engine asks again, with the park it just took already
        // subtracted from MemAvailable by the kernel.
        let mut warm = cold;
        warm.host_available_bytes = 120 * GIB - prefix;
        warm.already_parked_bytes = prefix;
        assert!(
            decide_text_encoder_residency(&warm).parks(),
            "an engine holding its own park must not be told there is no room for it"
        );

        // The credit is exactly the park, not a blanket pass: an engine
        // holding a park on a host that has SINCE lost most of its memory to
        // something else still gives it up.
        let mut starved = warm;
        starved.host_available_bytes = 4 * GIB;
        assert!(
            !decide_text_encoder_residency(&starved).parks(),
            "the credit must not survive a host that genuinely ran out"
        );

        // And an engine holding nothing is unaffected — the cold path is the
        // one every existing row exercises.
        let mut cold_again = cold;
        cold_again.already_parked_bytes = 0;
        assert_eq!(
            decide_text_encoder_residency(&cold_again),
            decide_text_encoder_residency(&cold)
        );
    }

    /// Pinning never widens the park, and it is refused on its own cap.
    #[test]
    fn pinning_is_decided_after_the_park_and_only_under_its_own_cap() {
        let mut narrow = inputs(
            1_500 * GB,
            1_400 * GB,
            33_000_000_000,
            KeepTeRamMode::Auto,
            TextEncoderDevice::Cuda,
        );
        assert_eq!(
            decide_text_encoder_residency(&narrow),
            TextEncoderResidency::HostParked { pinned: true }
        );

        narrow.pinned_cap_bytes = narrow.encoder_bytes - 1;
        assert_eq!(
            decide_text_encoder_residency(&narrow),
            TextEncoderResidency::HostParked { pinned: false },
            "a prefix over the page-locked cap still parks, just not pinned"
        );

        // Exactly enough for the park, with the floor and not a byte more
        // left afterwards: parked, unpinned.
        let mut tight = inputs(
            1_500 * GB,
            0,
            0,
            KeepTeRamMode::Force,
            TextEncoderDevice::Cuda,
        );
        tight.host_available_bytes =
            tight.encoder_bytes + host_safety_floor_bytes(tight.host_total_bytes);
        assert_eq!(
            decide_text_encoder_residency(&tight),
            TextEncoderResidency::HostParked { pinned: false },
            "page-locked memory cannot be paged out, so it needs room to spare"
        );
    }

    /// An encoder with no bytes, or an unmeasured host, is not evidence for a
    /// park — the fallback is always today's streaming behaviour.
    #[test]
    fn an_unmeasured_host_streams() {
        let mut unknown = inputs(0, 0, 0, KeepTeRamMode::Force, TextEncoderDevice::Cuda);
        assert_eq!(
            decide_text_encoder_residency(&unknown),
            TextEncoderResidency::StreamFromMmap
        );
        unknown.host_total_bytes = 1_500 * GB;
        unknown.encoder_bytes = 0;
        assert_eq!(
            decide_text_encoder_residency(&unknown),
            TextEncoderResidency::StreamFromMmap
        );
    }

    fn within_one_percent(got: u64, anchor: u64) -> bool {
        let delta = got.abs_diff(anchor) as f64;
        delta / anchor as f64 <= 0.01
    }

    /// The documented anchors, each pinned within 1 %.
    ///
    /// * 1.34 GB embedding table — the figure
    ///   `encoders::mistral3::streamed_peak_weight_bytes` reports and the
    ///   audit's "runtime peak is really 1.34 GB".
    /// * ~1.11 GB per decoder layer.
    /// * ~34.7 GB for the whole 30-layer prefix — the "~35 GB" the plan's
    ///   residency note prices, and the reason it cannot co-reside with a
    ///   33 GB Q8 transformer on a 46 GB card.
    /// * ~3.6 GB streamed device peak at one layer of look-ahead, which is
    ///   what makes the encoder fit anywhere at all.
    #[test]
    fn mistral3_residency_constants_match_the_documented_anchors() {
        assert_eq!(MISTRAL3_PREFIX_LAYERS, 30);
        assert!(within_one_percent(
            mistral3_embed_bytes(DType::BF16),
            1_342_177_280
        ));
        assert!(within_one_percent(
            mistral3_layer_bytes(DType::BF16),
            1_111_511_040
        ));
        assert!(within_one_percent(
            mistral3_prefix_bytes(DType::BF16),
            34_687_508_480
        ));
        assert!(
            within_one_percent(
                mistral3_streamed_device_peak_bytes(DType::BF16, MISTRAL3_DEFAULT_LOOKAHEAD),
                3_586_170_880
            ),
            "streamed peak was {}",
            mistral3_streamed_device_peak_bytes(DType::BF16, MISTRAL3_DEFAULT_LOOKAHEAD)
        );
    }

    /// The streamed peak is what the encoder allocates, never a second copy of
    /// the checkpoint: it must stay small beside the real 36 GB file.
    #[test]
    fn the_streamed_peak_is_not_a_second_copy_of_the_checkpoint() {
        const REAL_MISTRAL3_BF16_BYTES: u64 = 36_000_000_000;
        assert!(
            mistral3_streamed_device_peak_bytes(DType::BF16, MISTRAL3_DEFAULT_LOOKAHEAD)
                < REAL_MISTRAL3_BF16_BYTES / 8,
            "a streaming encoder does not materialize its checkpoint"
        );
    }

    /// The engine's own peak-weight figure and this module must not drift:
    /// `streamed_peak_weight_bytes` delegates here.
    #[test]
    fn the_engine_peak_weight_figure_is_this_modules_embedding_table() {
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            assert_eq!(
                crate::encoders::mistral3::streamed_peak_weight_bytes(dtype),
                mistral3_embed_bytes(dtype)
            );
        }
    }

    #[test]
    fn look_ahead_charges_exactly_one_more_layer() {
        let serial = mistral3_streamed_device_peak_bytes(DType::BF16, 0);
        let prefetching = mistral3_streamed_device_peak_bytes(DType::BF16, 1);
        assert_eq!(
            prefetching - serial,
            mistral3_layer_bytes(DType::BF16),
            "one layer of look-ahead is one more resident layer, nothing else"
        );
    }

    #[test]
    fn only_flux2_dev_safetensors_shards_stream() {
        let shard = PathBuf::from("mistral_3_small_flux2_bf16.safetensors");
        assert!(mistral3_streams_from_mmap("flux2", "flux2-dev:q8", &shard));
        assert!(mistral3_streams_from_mmap("flux.2", "FLUX.2-dev", &shard));
        assert!(
            !mistral3_streams_from_mmap("flux2", "flux2-klein:q8", &shard),
            "Klein conditions on a materialized Qwen3, not a streamed Mistral3"
        );
        assert!(
            !mistral3_streams_from_mmap("flux", "flux2-dev:q8", &shard),
            "the family gate is not optional"
        );
        assert!(
            !mistral3_streams_from_mmap(
                "flux2",
                "flux2-dev:q8",
                &PathBuf::from("mistral3-Q8_0.gguf")
            ),
            "a quantized republication would take a different loader"
        );
    }

    fn paths_with_encoders(files: &[&str]) -> mold_core::ModelPaths {
        mold_core::ModelPaths {
            transformer: PathBuf::from("/models/flux2-dev-Q8_0.gguf"),
            transformer_shards: Vec::new(),
            low_noise_transformer: None,
            vae: PathBuf::from("/models/flux2-vae.safetensors"),
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            low_noise_distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: files.iter().map(PathBuf::from).collect(),
            text_tokenizer: None,
            decoder: None,
        }
    }

    #[test]
    fn the_admission_charge_covers_only_a_complete_mistral3_encoder_set() {
        let dev = paths_with_encoders(&[
            "mistral-00001-of-00002.safetensors",
            "mistral-00002-of-00002.safetensors",
        ]);
        let charge = mistral3_admission_charge("flux2-dev:q8", &dev, DType::BF16)
            .expect("a dev encoder streams");
        assert_eq!(
            charge.device_peak,
            mistral3_streamed_device_peak_bytes(DType::BF16, MISTRAL3_DEFAULT_LOOKAHEAD)
        );
        assert_eq!(
            charge.host_anon_peak,
            mistral3_streamed_device_peak_bytes(DType::F32, MISTRAL3_DEFAULT_LOOKAHEAD),
            "a CPU-placed FLUX.2 encoder runs at F32"
        );

        assert!(
            mistral3_admission_charge("flux2-klein:q8", &dev, DType::BF16).is_none(),
            "Klein keeps its file-size charge"
        );
        assert!(
            mistral3_admission_charge("flux2-dev:q8", &paths_with_encoders(&[]), DType::BF16)
                .is_none(),
            "no encoder files means nothing to re-price"
        );
        assert!(
            mistral3_admission_charge(
                "flux2-dev:q8",
                &paths_with_encoders(&["mistral-00001-of-00002.safetensors", "qwen3-Q8_0.gguf"]),
                DType::BF16
            )
            .is_none(),
            "a mixed set is not a streamed Mistral3 encoder phase"
        );
    }
}
