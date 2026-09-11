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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

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
