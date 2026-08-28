mod assets;
mod backend;
pub mod chain;
mod conditioning;
pub(crate) mod convrot;
mod execution;
pub(crate) mod exr;
pub(crate) mod fp8_widen;
pub(crate) mod gguf;
mod guidance;
mod hdr;
mod lora;
pub mod media;
mod model;
mod nvfp4;
mod pipeline;
mod plan;
pub(crate) mod preprocess;
mod preset;
mod runtime;
mod sampler;
pub(crate) mod single_file;
mod text;
mod tiling;

pub use chain::{extract_tail_latents, tail_latent_frame_count};
pub(crate) use model::DecodedAudio;
pub use pipeline::Ltx2Engine;
pub use runtime::{ltx2_transformer_weight_sizes, Ltx2TransformerWeightSizes};
// Wan continues a clip the same way at the stitch layer -- drop the duplicated
// leading frames, append the rest -- so the two share one implementation
// rather than keeping a second chance to be off by one.
pub(crate) use pipeline::stitch_extend_frames;

/// Whether the resolved checkpoint set contains both the audio VAE and
/// vocoder tensors required for native LTX-2 audio output.
pub fn audio_output_supported(paths: &mold_core::ModelPaths) -> bool {
    audio_output_gap(paths).is_none()
}

/// Whether a resolved LTX-2 checkpoint set plus an optional split-pack audio
/// component contains every tensor required for native synchronized audio.
/// LTX-2.5 keeps that component outside legacy [`mold_core::ModelPaths`].
pub(crate) fn audio_output_supported_with_components(
    paths: &mold_core::ModelPaths,
    audio_components_path: Option<&std::path::Path>,
) -> bool {
    audio_output_gap_with_components(paths, audio_components_path).is_none()
}

/// Why native LTX-2 audio output is unavailable for this checkpoint set, or
/// `None` when it is available.
///
/// Returns a specific reason rather than a generic one. The previous message
/// named both assets unconditionally — "the resolved checkpoint assets do not
/// include both the audio VAE and vocoder tensors" — which pointed every
/// reader at a bad download. When a 19B checkpoint's vocoder went unrecognised
/// because only the nested key spelling was probed, that message actively
/// misdirected: the tensors were present, under a prefix the probe did not
/// know. An error that misleads costs more than one that is merely vague.
pub fn audio_output_gap(paths: &mold_core::ModelPaths) -> Option<String> {
    audio_output_gap_with_components(paths, None)
}

fn audio_output_gap_with_components(
    paths: &mold_core::ModelPaths,
    audio_components_path: Option<&std::path::Path>,
) -> Option<String> {
    let mut flags = single_file::AudioOutputAssetFlags::default();
    for path in [paths.transformer.as_path(), paths.vae.as_path()]
        .into_iter()
        .chain(audio_components_path)
    {
        let Ok(found) = single_file::audio_output_asset_flags(path) else {
            continue;
        };
        flags.audio_vae |= found.audio_vae;
        flags.vocoder |= found.vocoder;
        flags.audio_vae_prefix_seen |= found.audio_vae_prefix_seen;
        flags.vocoder_prefix_seen |= found.vocoder_prefix_seen;
    }

    let mut missing: Vec<String> = Vec::new();
    if !flags.audio_vae {
        missing.push(if flags.audio_vae_prefix_seen {
            "the audio VAE (`audio_vae.*` tensors are present but not in a layout this build recognises)"
                .to_string()
        } else {
            "the audio VAE".to_string()
        });
    }
    if !flags.vocoder {
        missing.push(if flags.vocoder_prefix_seen {
            "the vocoder (`vocoder.*` tensors are present but not in a layout this build recognises)"
                .to_string()
        } else {
            "the vocoder".to_string()
        });
    }
    if missing.is_empty() {
        return None;
    }
    Some(missing.join(" and "))
}

/// Header-only capability check for a single catalog checkpoint.
pub fn checkpoint_supports_audio_output(path: &std::path::Path) -> bool {
    single_file::supports_audio_output(path).unwrap_or(false)
}

/// Whether a CPU-placed Gemma artifact pays the streaming encoder's
/// anonymous-heap peak on top of its own bytes.
///
/// The Q4 GGUF variant is loaded through `GgufGemmaEncoder::load`, which takes
/// no dtype and keeps its own quantized residency
/// (`text/prompt_encoder.rs:111-119`), so only the safetensors variant does.
pub fn cpu_gemma_allocates_anon_peak(artifact: &std::path::Path) -> bool {
    !artifact
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
}

/// Host bytes a CPU-placed Gemma prompt encoder allocates on top of its
/// memory-mapped weight files, for admission's host-RAM reservation.
///
/// The CPU encoder does not materialize the checkpoint. `load_from_assets`
/// builds through `new_streaming` (`text/encoder.rs:445-451`), so the shards
/// stay an mmap'd `VarBuilder` and `forward_hidden_states` constructs each
/// decoder layer inside the loop and drops it before the next
/// (`text/encoder.rs:507-522`). Pricing the whole file at the CPU compute
/// dtype would therefore charge a second full copy of ~24.7 GB that is never
/// allocated, and park work that runs comfortably.
///
/// What is anonymous heap is the composition
/// [`crate::device::LTX2_GEMMA_VRAM_THRESHOLD`] documents: the retained token
/// embedding table, at most two in-flight decoder layers, and the retained
/// hidden states. That note states the arithmetic at BF16 — a peak near
/// 3.3 GB — while on CPU every one of those tensors is built at
/// [`backend::Ltx2Backend::Cpu`]'s compute dtype, so the same element count is
/// priced at F32 here.
pub fn cpu_gemma_streaming_anon_peak_bytes() -> u64 {
    /// `forward_hidden_states` holds the layer it is running and the one it
    /// just produced output from; they are never all co-resident.
    const LAYERS_IN_FLIGHT: u64 = 2;
    /// The context [`crate::device::LTX2_GEMMA_VRAM_THRESHOLD`]'s note prices.
    /// Conservative against today's `DEFAULT_GEMMA_MAX_LENGTH`, which is
    /// smaller, and one arithmetic rather than two that can drift apart.
    const HIDDEN_STATE_CONTEXT_TOKENS: u64 = 1_024;

    let cfg = text::encoder::ltx_gemma_config();
    let hidden_size = cfg.hidden_size as u64;
    let embedding_table = cfg.vocab_size as u64 * hidden_size;
    let hidden_states =
        (cfg.num_hidden_layers as u64 + 1) * HIDDEN_STATE_CONTEXT_TOKENS * hidden_size;
    let elements =
        embedding_table + LAYERS_IN_FLIGHT * gemma_decoder_layer_elements(&cfg) + hidden_states;
    elements * backend::Ltx2Backend::Cpu.compute_dtype().size_in_bytes() as u64
}

/// Parameter count of one Gemma decoder layer, from the shapes
/// `DecoderLayer::new` actually constructs (`text/encoder.rs:323-350`).
fn gemma_decoder_layer_elements(cfg: &text::encoder::GemmaConfig) -> u64 {
    let hidden = cfg.hidden_size as u64;
    let attention_dim = (cfg.num_attention_heads * cfg.head_dim) as u64;
    let kv_dim = (cfg.num_key_value_heads * cfg.head_dim) as u64;
    // q_proj and o_proj span the full attention width; k_proj and v_proj are
    // grouped-query and span the smaller key/value width. q_norm and k_norm
    // are per-head.
    let attention = hidden * attention_dim * 2 + hidden * kv_dim * 2 + (cfg.head_dim as u64) * 2;
    // gate_proj, up_proj, down_proj.
    let mlp = hidden * cfg.intermediate_size as u64 * 3;
    // input, post-attention, pre-feedforward, and post-feedforward norms.
    let layernorms = hidden * 4;
    attention + mlp + layernorms
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    /// The streaming heap must reproduce the composition
    /// `device::LTX2_GEMMA_VRAM_THRESHOLD` documents, priced at the CPU dtype.
    ///
    /// That note derives its figures at BF16 — a 2.01 GB embedding table,
    /// ~0.45 GB per in-flight decoder layer, ~0.39 GB of retained hidden
    /// states, "a peak near 3.3 GB". Halving the F32 answer must land back on
    /// it, or the two arithmetics have drifted.
    #[test]
    fn cpu_gemma_streaming_heap_matches_the_documented_bf16_peak() {
        let f32_peak = super::cpu_gemma_streaming_anon_peak_bytes();
        assert_eq!(f32_peak, 6_591_410_176);
        let bf16_peak = f32_peak / 2;
        assert!(
            (3_200_000_000..3_400_000_000).contains(&bf16_peak),
            "BF16 peak {bf16_peak} must still be the documented ~3.3 GB"
        );
    }

    /// The heap is what the encoder allocates, never a second copy of the
    /// weights: it must stay small beside a real 24.7 GB BF16 checkpoint.
    #[test]
    fn cpu_gemma_streaming_heap_is_not_a_second_copy_of_the_weights() {
        const REAL_GEMMA_BF16_BYTES: u64 = 24_700_000_000;
        assert!(
            super::cpu_gemma_streaming_anon_peak_bytes() < REAL_GEMMA_BF16_BYTES / 3,
            "a streaming encoder does not materialize its checkpoint"
        );
    }

    #[test]
    fn a_quantized_gemma_allocates_no_streaming_heap() {
        assert!(super::cpu_gemma_allocates_anon_peak(Path::new(
            "model-00001-of-00005.safetensors"
        )));
        assert!(!super::cpu_gemma_allocates_anon_peak(Path::new(
            "gemma-3-12b-it-q4_0.gguf"
        )));
    }
}
