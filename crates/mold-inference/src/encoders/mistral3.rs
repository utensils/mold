//! Streamed Mistral3 language encoder for FLUX.2 [dev].
//!
//! The upstream checkpoint contains a 40-layer, 24B-class language model, but
//! FLUX conditioning consumes only hidden states after layers 10, 20, and 30.
//! This implementation mmaps the eight shards containing that exact prefix and
//! materializes one decoder layer at a time on the target device. Later layers,
//! the final norm, vision tower, projector, and LM head are intentionally absent.
//! This matches BFL's `Mistral3SmallEmbedder::forward`, which passes text-only
//! chat-template inputs to the language model; editing references enter the
//! denoiser separately through `prepare_reference` VAE latent tokens. See
//! `src/flux2/text_encoder.py` and `src/flux2/sampling.py` in BFL's `flux2`
//! reference repository.

use crate::flux2::text_encoder_residency as residency;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::VarBuilder;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

const VOCAB_SIZE: usize = 131_072;
const HIDDEN_SIZE: usize = 5_120;
const INTERMEDIATE_SIZE: usize = 32_768;
const NUM_ATTENTION_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const KV_REPEAT: usize = NUM_ATTENTION_HEADS / NUM_KV_HEADS;
const ROPE_THETA: f64 = 1_000_000_000.0;
const RMS_NORM_EPS: f64 = 1e-5;
const MAX_LENGTH: usize = 512;
const PAD_TOKEN_ID: u32 = 11;
const CAPTURE_LAYERS: [usize; 3] = [9, 19, 29];

/// The progress component name for the whole streamed prefix. One label for
/// the mapping and for every layer it streams, so the bar advances instead of
/// jumping from 0 % to 100 % once.
pub(crate) const STREAMED_ENCODER_COMPONENT: &str = "FLUX.2 [dev] Mistral3 encoder";

/// Largest simultaneously resident weight allocation in the streamed encoder:
/// the 131072 x 5120 token embedding table. Decoder layers are smaller.
///
/// Delegates to [`crate::flux2::text_encoder_residency`], which is the single
/// authority on this encoder's residency arithmetic — admission, the placement
/// planner, and this engine must not carry two copies of the geometry.
pub(crate) fn streamed_peak_weight_bytes(dtype: DType) -> u64 {
    crate::flux2::text_encoder_residency::mistral3_embed_bytes(dtype)
}

const SYSTEM_PROMPT: &str = "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\nattribution and actions without speculation.";

fn format_prompt(prompt: &str) -> String {
    let prompt = prompt.replace("[IMG]", "");
    format!("<s>[SYSTEM_PROMPT]{SYSTEM_PROMPT}[/SYSTEM_PROMPT][INST]{prompt}[/INST]")
}

struct RmsNorm {
    weight: Tensor,
}

impl RmsNorm {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(HIDDEN_SIZE, "weight")?,
        })
    }

    /// Match Transformers' Mistral RMSNorm: variance and normalization are
    /// performed in float32, then cast back before applying the learned scale.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let xs = xs_f32.broadcast_div(&(variance + RMS_NORM_EPS)?.sqrt()?)?;
        xs.to_dtype(dtype)?
            .broadcast_mul(&self.weight)
            .map_err(Into::into)
    }
}

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, device: &Device) -> Result<Self> {
        let inv_freq = (0..HEAD_DIM)
            .step_by(2)
            .map(|i| 1f32 / ROPE_THETA.powf(i as f64 / HEAD_DIM as f64) as f32)
            .collect::<Vec<_>>();
        let inv_freq = Tensor::from_vec(inv_freq, (1, HEAD_DIM / 2), device)?;
        let positions = Tensor::arange(0u32, MAX_LENGTH as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_LENGTH, 1))?;
        let freqs = positions.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((
            candle_nn::rotary_emb::rope(&q.contiguous()?, &self.cos, &self.sin)?,
            candle_nn::rotary_emb::rope(&k.contiguous()?, &self.cos, &self.sin)?,
        ))
    }
}

fn repeat_kv(x: Tensor) -> Result<Tensor> {
    let (batch, heads, seq, dim) = x.dims4()?;
    x.unsqueeze(2)?
        .broadcast_as((batch, heads, KV_REPEAT, seq, dim))?
        .reshape((batch, heads * KV_REPEAT, seq, dim))
        .map_err(Into::into)
}

struct Attention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    rotary: Arc<RotaryEmbedding>,
}

impl Attention {
    fn new(rotary: Arc<RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q_proj: candle_nn::linear_no_bias(
                HIDDEN_SIZE,
                NUM_ATTENTION_HEADS * HEAD_DIM,
                vb.pp("q_proj"),
            )?,
            k_proj: candle_nn::linear_no_bias(
                HIDDEN_SIZE,
                NUM_KV_HEADS * HEAD_DIM,
                vb.pp("k_proj"),
            )?,
            v_proj: candle_nn::linear_no_bias(
                HIDDEN_SIZE,
                NUM_KV_HEADS * HEAD_DIM,
                vb.pp("v_proj"),
            )?,
            o_proj: candle_nn::linear_no_bias(
                NUM_ATTENTION_HEADS * HEAD_DIM,
                HIDDEN_SIZE,
                vb.pp("o_proj"),
            )?,
            rotary,
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (batch, seq, _) = xs.dims3()?;
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((batch, seq, NUM_ATTENTION_HEADS, HEAD_DIM))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((batch, seq, NUM_KV_HEADS, HEAD_DIM))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((batch, seq, NUM_KV_HEADS, HEAD_DIM))?
            .transpose(1, 2)?;
        let (q, k) = self.rotary.apply(&q, &k)?;
        let k = repeat_kv(k)?.contiguous()?;
        let v = repeat_kv(v)?.contiguous()?;
        let scores = (q.matmul(&k.transpose(2, 3)?)? / (HEAD_DIM as f64).sqrt())?;
        let scores = scores.broadcast_add(mask)?;
        let context = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?;
        context
            .transpose(1, 2)?
            .reshape((batch, seq, NUM_ATTENTION_HEADS * HEAD_DIM))?
            .apply(&self.o_proj)
            .map_err(Into::into)
    }
}

struct Mlp {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl Mlp {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: candle_nn::linear_no_bias(
                HIDDEN_SIZE,
                INTERMEDIATE_SIZE,
                vb.pp("gate_proj"),
            )?,
            up_proj: candle_nn::linear_no_bias(HIDDEN_SIZE, INTERMEDIATE_SIZE, vb.pp("up_proj"))?,
            down_proj: candle_nn::linear_no_bias(
                INTERMEDIATE_SIZE,
                HIDDEN_SIZE,
                vb.pp("down_proj"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::Activation::Silu.forward(&xs.apply(&self.gate_proj)?)?;
        (gate * xs.apply(&self.up_proj)?)?
            .apply(&self.down_proj)
            .map_err(Into::into)
    }
}

struct DecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl DecoderLayer {
    fn new(rotary: Arc<RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input_layernorm: RmsNorm::new(vb.pp("input_layernorm"))?,
            self_attn: Attention::new(rotary, vb.pp("self_attn"))?,
            post_attention_layernorm: RmsNorm::new(vb.pp("post_attention_layernorm"))?,
            mlp: Mlp::new(vb.pp("mlp"))?,
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = (residual + self.self_attn.forward(&xs, mask)?)?;
        let residual = &xs;
        let hidden = self.post_attention_layernorm.forward(&xs)?;
        (residual + self.mlp.forward(&hidden)?).map_err(Into::into)
    }
}

/// The two key namespaces FLUX.2's Mistral3 conditioner ships under.
///
/// BFL's own `FLUX.2-dev/text_encoder` shards keep the full
/// `Mistral3ForConditionalGeneration` layout, so the language model sits under
/// `language_model.model.*`. Comfy-Org's single-file republication of the same
/// encoder strips that wrapper and puts it at `model.*`, with the vision tower
/// and projector beside it at the root. Both carry identical tensors under the
/// prefix; only the namespace differs.
const MISTRAL3_LM_PREFIXES: [&str; 2] = ["language_model.model", "model"];

/// The last decoder layer the streamed encoder runs. Layers past it, the final
/// norm, the vision tower, and the LM head are never touched.
fn last_required_layer() -> usize {
    *CAPTURE_LAYERS.last().expect("CAPTURE_LAYERS is non-empty")
}

/// Header-peek the configured files and resolve which namespace holds the
/// language model, requiring the exact prefix this encoder streams: the token
/// embedding plus every decoder layer up to the last captured one.
///
/// This replaces a shard COUNT check. The count was a proxy for "the runtime
/// prefix is present", and it is the wrong proxy: it refuses a single-file
/// republication of the same weights, and it would accept eight shards of some
/// other checkpoint. Asking the headers what is actually there answers the
/// real question and names what is missing when it is not.
fn resolve_lm_prefix(encoder_paths: &[PathBuf]) -> Result<&'static str> {
    if encoder_paths.is_empty() {
        anyhow::bail!("FLUX.2 [dev] requires a Mistral3 text encoder; none is configured");
    }
    let mut keys = std::collections::BTreeSet::new();
    for path in encoder_paths {
        let header = crate::weight_loader::read_safetensors_header(path)
            .with_context(|| format!("peek Mistral3 encoder header at {}", path.display()))?;
        keys.extend(header.into_keys());
    }

    let last_layer = last_required_layer();
    for prefix in MISTRAL3_LM_PREFIXES {
        if !keys.contains(&format!("{prefix}.embed_tokens.weight")) {
            continue;
        }
        if let Some(missing) = (0..=last_layer).find(|layer| {
            !keys.contains(&format!("{prefix}.layers.{layer}.input_layernorm.weight"))
        }) {
            anyhow::bail!(
                "Mistral3 text encoder at {} is incomplete: FLUX.2 [dev] streams decoder layers 0-{last_layer} under `{prefix}`, but layer {missing} is missing",
                encoder_paths
                    .first()
                    .map(|path| path.display().to_string())
                    .unwrap_or_default(),
            );
        }
        return Ok(prefix);
    }
    anyhow::bail!(
        "no Mistral3 language model found in the configured FLUX.2 [dev] text encoder ({} file(s)); expected `{}.embed_tokens.weight` or `{}.embed_tokens.weight`",
        encoder_paths.len(),
        MISTRAL3_LM_PREFIXES[0],
        MISTRAL3_LM_PREFIXES[1],
    );
}

/// Whether a tensor name belongs to the prefix the encoder actually runs.
///
/// The prefix is the token embedding plus decoder layers `0..=last`. Nothing
/// else may be parked: the single-file republication of this checkpoint ships
/// a vision tower, a multimodal projector and decoder layers 30-39 beside the
/// prefix, and a whole-file park would charge host RAM for every byte of them
/// — roughly a third again on top of the 34.7 GB that IS read. A memory
/// mapping never pages those in because nothing asks for them; an eager park
/// would.
///
/// `prefix` is the resolved namespace (`language_model.model` or `model`), so
/// the filter is exact rather than a substring guess.
pub(crate) fn parked_prefix_tensor(prefix: &str, last_layer: usize, name: &str) -> bool {
    let Some(rest) = name
        .strip_prefix(prefix)
        .and_then(|rest| rest.strip_prefix('.'))
    else {
        return false;
    };
    if rest.starts_with("embed_tokens.") {
        return true;
    }
    let Some(rest) = rest.strip_prefix("layers.") else {
        return false;
    };
    let Some((index, _)) = rest.split_once('.') else {
        return false;
    };
    index
        .parse::<usize>()
        .is_ok_and(|index| index <= last_layer)
}

/// The streamed prefix, held in host RAM between requests.
///
/// Every tensor is a CPU tensor at the encoder's working dtype, so an encode
/// is a host-to-device copy per layer instead of a page fault, a dtype
/// conversion and a copy. `_pinned` holds the page-locked registrations for
/// the lifetime of the tensors they cover — dropping it unregisters them, so
/// it must not be replaced with a bare bool.
pub(crate) struct ParkedPrefix {
    tensors: std::collections::HashMap<String, Tensor>,
    _pinned: Vec<crate::flux::pinned::PinnedRegion>,
}

impl ParkedPrefix {
    /// Host bytes this park is holding.
    pub(crate) fn bytes(&self) -> u64 {
        self.tensors
            .values()
            .map(|tensor| (tensor.elem_count() * tensor.dtype().size_in_bytes()) as u64)
            .sum()
    }
}

pub(crate) struct Mistral3Encoder {
    encoder_paths: Vec<PathBuf>,
    /// Resolved at load from the checkpoint's own headers.
    lm_prefix: &'static str,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    dtype: DType,
    /// The prefix held in host RAM, when the residency budget allowed it.
    parked: Option<ParkedPrefix>,
}

impl Mistral3Encoder {
    pub fn load(
        encoder_paths: &[PathBuf],
        tokenizer: Arc<Tokenizer>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let lm_prefix = resolve_lm_prefix(encoder_paths)?;
        Ok(Self {
            encoder_paths: encoder_paths.to_vec(),
            lm_prefix,
            tokenizer,
            device: device.clone(),
            dtype,
            parked: None,
        })
    }

    /// Whether this encoder is holding its prefix in host RAM.
    pub(crate) fn is_parked(&self) -> bool {
        self.parked.is_some()
    }

    /// Host bytes the park is holding, or zero.
    pub(crate) fn parked_bytes(&self) -> u64 {
        self.parked.as_ref().map_or(0, ParkedPrefix::bytes)
    }

    /// Read the prefix into host RAM, page-locking it when asked.
    ///
    /// ONE copy per tensor, out of a mapping, through the shared
    /// `encoders::park` loader — and FILTERED, so the vision tower, the
    /// projector and layers 30-39 are never touched at all.
    ///
    /// `pinned` is the residency decision's own answer, never re-derived here.
    /// A pin that the driver or the tracker's soft cap declines is a no-op
    /// rather than an error: the park is still worth having without it.
    pub(crate) fn park_prefix(&mut self, pinned: bool) -> Result<()> {
        if self.parked.is_some() {
            return Ok(());
        }
        let prefix = self.lm_prefix;
        let last_layer = last_required_layer();
        let tensors =
            crate::encoders::park::load_tensors_to_cpu_filtered(&self.encoder_paths, |name| {
                parked_prefix_tensor(prefix, last_layer, name)
            })?;
        if tensors.is_empty() {
            anyhow::bail!(
                "Mistral3 prefix park found no tensors under `{prefix}` — refusing to park an \
                 empty set rather than render from one"
            );
        }
        let mut regions = Vec::new();
        if pinned {
            let tracker = crate::flux::pinned::PinnedMemoryTracker::new(
                crate::flux::pinned::pinned_cap_bytes(),
            );
            for tensor in tensors.values() {
                match crate::flux::pinned::try_pin_to_host(tensor, &tracker) {
                    Ok(Some(region)) => regions.push(region),
                    Ok(None) => {}
                    Err(error) => {
                        tracing::debug!(%error, "Mistral3 prefix pin declined; parking unpinned");
                        regions.clear();
                        break;
                    }
                }
            }
        }
        self.parked = Some(ParkedPrefix {
            tensors,
            _pinned: regions,
        });
        Ok(())
    }

    /// Release the host park.
    pub(crate) fn unpark(&mut self) {
        self.parked = None;
    }

    pub fn encode(
        &self,
        prompt: &str,
        target_device: &Device,
        target_dtype: DType,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<(Tensor, usize)> {
        let formatted = format_prompt(prompt);
        let encoding = self
            .tokenizer
            .encode(formatted, false)
            .map_err(|error| anyhow::anyhow!("Mistral3 tokenization failed: {error}"))?;
        let token_count = encoding.len().min(MAX_LENGTH);
        let mut tokens = encoding.get_ids()[..token_count].to_vec();
        tokens.resize(MAX_LENGTH, PAD_TOKEN_ID);
        let attention = (0..MAX_LENGTH)
            .map(|index| index < token_count)
            .collect::<Vec<_>>();

        // The parked prefix and the mapping produce the SAME weights: the park
        // is `MmapedSafetensors::multi` plus one CPU load per tensor, which is
        // what the mapping-backed `VarBuilder` would have done lazily. What
        // changes is where the bytes come from on the second and later
        // requests — host RAM, already at the working dtype and optionally
        // page-locked, instead of a page fault plus a conversion.
        //
        // `pp(lm_prefix)` is applied in both arms because the park keeps the
        // checkpoint's own key namespace, so the two builders are addressed
        // identically and the layer loop below cannot tell them apart.
        let vb = match self.parked.as_ref() {
            Some(parked) => crate::encoders::park::varbuilder_from_parked(
                &parked.tensors,
                self.dtype,
                &self.device,
            )
            .pp(self.lm_prefix),
            None => crate::weight_loader::load_safetensors_with_progress(
                &self.encoder_paths,
                self.dtype,
                &self.device,
                STREAMED_ENCODER_COMPONENT,
                &crate::progress::ProgressReporter::default(),
            )?
            .pp(self.lm_prefix),
        };

        let input_ids = Tensor::from_vec(tokens, (1, MAX_LENGTH), &self.device)?;
        let hidden = {
            let embedding = candle_nn::embedding(VOCAB_SIZE, HIDDEN_SIZE, vb.pp("embed_tokens"))?;
            embedding.forward(&input_ids)?
        };

        let mask = causal_padding_mask(&attention, self.dtype, &self.device)?;
        let rotary = Arc::new(RotaryEmbedding::new(self.dtype, &self.device)?);

        // The streamed prefix, priced exactly as admission prices it, so the
        // progress bar and the memory plan describe the same thing.
        let embed_bytes = residency::mistral3_embed_bytes(self.dtype);
        let layer_bytes = residency::mistral3_layer_bytes(self.dtype);
        let prefix_bytes = residency::mistral3_prefix_bytes(self.dtype);
        let built = std::sync::atomic::AtomicU64::new(0);
        let layers = vb.pp("layers");
        let build = |index: usize| -> Result<DecoderLayer> {
            let layer = DecoderLayer::new(rotary.clone(), layers.pp(index))
                .with_context(|| format!("loading Mistral3 decoder layer {index}"))?;
            let done = built.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            progress.weight_load(
                STREAMED_ENCODER_COMPONENT,
                embed_bytes.saturating_add(done.saturating_mul(layer_bytes)),
                prefix_bytes,
            );
            Ok(layer)
        };

        let captured = stream_layers(
            last_required_layer(),
            hidden,
            build,
            |layer: &DecoderLayer, hidden: &Tensor| layer.forward(hidden, &mask),
            &CAPTURE_LAYERS,
        )?;
        let output = Tensor::cat(&captured, D::Minus1)?
            .to_device(target_device)?
            .to_dtype(target_dtype)?;
        Ok((output, token_count))
    }
}

/// Drive a streamed layer stack with one layer of look-ahead.
///
/// `build(k + 1)` runs on a scoped thread while `run` executes layer `k`, so
/// the host-side work of materializing the next layer — faulting in its
/// mapped pages, converting its dtype, and issuing its host-to-device copies —
/// overlaps the layer already on the accelerator instead of following it.
/// Exactly `MISTRAL3_LOOKAHEAD + 1` layers are ever alive, which is what
/// [`crate::flux2::text_encoder_residency::mistral3_streamed_device_peak_bytes`]
/// charges admission.
///
/// Three facts make this safe on CUDA, and none of them is an assumption about
/// timing:
///
/// * Candle's `CudaDevice` is `Clone` and every clone shares ONE
///   `Arc<CudaStream>` (`candle-core/src/cuda_backend/device.rs:59-70`), so
///   the prefetch's uploads and the forward's launches are issued onto the
///   same stream and execute in the order the driver receives them.
/// * Every cudarc entry point binds the context to the calling thread before
///   touching the driver (`cudarc-0.19.9/src/driver/safe/core.rs:1538` for
///   `alloc`, `:1612` for `memcpy_htod`), so a second OS thread needs no
///   setup of its own.
/// * `CudaSlice::drop` is stream-ordered (`core.rs:800-819`): with async
///   allocation it issues `cuMemFreeAsync` on the slice's OWN stream, and
///   without it, it synchronizes that stream first. So releasing layer `k`
///   after `k + 1`'s uploads have been issued cannot free memory the stream
///   is still reading.
///
/// The two operations touch disjoint allocations — the forward reads layer
/// `k`'s weights and the state tensor, the prefetch writes fresh
/// allocations — so the result is bit-identical to the serial loop, which is
/// what `prefetching_matches_the_serial_stack` pins.
///
/// This replaces a `device.synchronize()` after every layer. Those calls
/// blocked the host on the GPU without ordering anything the stream did not
/// already order, which is the whole reason there was nothing to overlap.
fn stream_layers<Layer, State, Build, Run>(
    last_layer: usize,
    initial: State,
    build: Build,
    mut run: Run,
    capture: &[usize],
) -> Result<Vec<State>>
where
    Layer: Send,
    State: Clone,
    Build: Fn(usize) -> Result<Layer> + Sync,
    Run: FnMut(&Layer, &State) -> Result<State>,
{
    let build = &build;
    let mut state = initial;
    let mut captured = Vec::with_capacity(capture.len());
    let mut current = build(0)?;
    for index in 0..=last_layer {
        let (next, output) = std::thread::scope(|scope| -> Result<(Option<Layer>, State)> {
            let prefetch = (index < last_layer).then(|| scope.spawn(move || build(index + 1)));
            // Run first: the prefetch is already in flight, and a failure here
            // must still join the thread before it propagates.
            let output = run(&current, &state);
            let next = match prefetch {
                Some(handle) => Some(handle.join().map_err(|_| {
                    anyhow::anyhow!("streamed layer {} failed to build", index + 1)
                })??),
                None => None,
            };
            let output =
                output.with_context(|| format!("running Mistral3 decoder layer {index}"))?;
            Ok((next, output))
        })?;
        state = output;
        if capture.contains(&index) {
            captured.push(state.clone());
        }
        // Layer `index` is released HERE, after `index + 1` is already built:
        // two layers alive at the seam, never three.
        match next {
            Some(layer) => current = layer,
            None => break,
        }
    }
    Ok(captured)
}

fn causal_padding_mask(attention: &[bool], dtype: DType, device: &Device) -> Result<Tensor> {
    if attention.len() != MAX_LENGTH {
        anyhow::bail!("Mistral3 attention mask must contain exactly {MAX_LENGTH} positions");
    }
    let values = (0..MAX_LENGTH)
        .flat_map(|query| {
            (0..MAX_LENGTH).map(move |key| {
                if key <= query && attention[key] {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (1, 1, MAX_LENGTH, MAX_LENGTH), device)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// The park's filter is what keeps ~12 GB of weights the encoder never runs
    /// out of host RAM.
    ///
    /// The single-file republication of this checkpoint carries a vision tower, a
    /// multimodal projector, decoder layers 30-39, the final norm and the LM head
    /// beside the prefix. A memory mapping never pages them in, because nothing
    /// asks for them; an eager whole-file park would charge every byte. So the
    /// filter is not an optimization, it is the difference between a 34.7 GB park
    /// and one half again as large.
    #[test]
    fn the_park_filter_never_admits_the_vision_tower_or_the_unused_layers() {
        let last = last_required_layer();
        assert_eq!(last, 29, "the prefix is layers 0..=29");

        for prefix in MISTRAL3_LM_PREFIXES {
            assert!(parked_prefix_tensor(
                prefix,
                last,
                &format!("{prefix}.embed_tokens.weight")
            ));
            for layer in [0, 1, 15, 28, 29] {
                for leaf in [
                    "input_layernorm.weight",
                    "self_attn.q_proj.weight",
                    "mlp.down_proj.weight",
                ] {
                    assert!(
                        parked_prefix_tensor(
                            prefix,
                            last,
                            &format!("{prefix}.layers.{layer}.{leaf}")
                        ),
                        "{prefix}.layers.{layer}.{leaf} is part of the prefix"
                    );
                }
            }

            // Everything the encoder never reads.
            for name in [
                format!("{prefix}.layers.30.input_layernorm.weight"),
                format!("{prefix}.layers.39.mlp.down_proj.weight"),
                format!("{prefix}.norm.weight"),
                "vision_tower.transformer.layers.0.attention.q_proj.weight".to_string(),
                "multi_modal_projector.linear_1.weight".to_string(),
                "lm_head.weight".to_string(),
            ] {
                assert!(
                    !parked_prefix_tensor(prefix, last, &name),
                    "{name} must never be parked"
                );
            }
        }

        // A name under the OTHER namespace is not this checkpoint's prefix. The
        // two spellings differ only by a wrapper, and `model.layers.0...` is a
        // suffix of `language_model.model.layers.0...`, so a substring test would
        // have accepted both — which is why this is an exact prefix strip.
        assert!(!parked_prefix_tensor(
            "language_model.model",
            last,
            "model.layers.0.input_layernorm.weight"
        ));
        // And a layer index that merely starts with an admitted one is refused.
        assert!(!parked_prefix_tensor(
            "model",
            last,
            "model.layers.290.input_layernorm.weight"
        ));
        assert!(!parked_prefix_tensor(
            "model",
            last,
            "model.layers.30x.weight"
        ));
    }

    /// A stand-in for a decoder layer that reports its own residency.
    ///
    /// A real two-layer Mistral3 fixture is not a test: one layer of this
    /// geometry is 1.1 GB, and the dimensions are compile-time constants of
    /// the checkpoint. What `stream_layers` owns is the SCHEDULE — how many
    /// layers are alive, in what order they run, and whether the answer
    /// depends on the overlap — and a synthetic layer exercises all three.
    struct CountedLayer {
        index: usize,
        live: Arc<AtomicUsize>,
    }

    impl CountedLayer {
        fn build(index: usize, live: &Arc<AtomicUsize>, peak: &Arc<AtomicUsize>) -> Self {
            let now = live.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(now, Ordering::SeqCst);
            Self {
                index,
                live: live.clone(),
            }
        }
    }

    impl Drop for CountedLayer {
        fn drop(&mut self) {
            self.live.fetch_sub(1, Ordering::SeqCst);
        }
    }

    /// An order-sensitive, non-commutative step, so a schedule that ran two
    /// layers out of order could not produce the same number.
    fn advance(layer: &CountedLayer, state: &f64) -> Result<f64> {
        Ok(state.mul_add(1.5, layer.index as f64 + 1.0).sqrt())
    }

    fn serial(last_layer: usize, capture: &[usize]) -> Vec<f64> {
        let live = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let mut state = 1.0f64;
        let mut captured = Vec::new();
        for index in 0..=last_layer {
            let layer = CountedLayer::build(index, &live, &peak);
            state = advance(&layer, &state).unwrap();
            drop(layer);
            if capture.contains(&index) {
                captured.push(state);
            }
        }
        captured
    }

    /// One layer of look-ahead must not change the answer. The prefetch runs
    /// on another thread and on CUDA shares the forward's stream, so if the
    /// schedule could perturb the result it would perturb every render.
    #[test]
    fn prefetching_matches_the_serial_stack() {
        for last_layer in [0usize, 1, 2, 29] {
            let capture: Vec<usize> = (0..=last_layer).filter(|i| i % 3 == 0).collect();
            let live = Arc::new(AtomicUsize::new(0));
            let peak = Arc::new(AtomicUsize::new(0));
            let (build_live, build_peak) = (live.clone(), peak.clone());
            let got = stream_layers(
                last_layer,
                1.0f64,
                move |index| Ok(CountedLayer::build(index, &build_live, &build_peak)),
                |layer: &CountedLayer, state: &f64| advance(layer, state),
                &capture,
            )
            .unwrap();
            assert_eq!(
                got,
                serial(last_layer, &capture),
                "look-ahead changed the result at last_layer={last_layer}"
            );
        }
    }

    /// At most `lookahead + 1` layers are ever alive — exactly what
    /// `flux2::text_encoder_residency::mistral3_streamed_device_peak_bytes`
    /// charges admission at `MISTRAL3_DEFAULT_LOOKAHEAD`. A driver that held
    /// three would silently break the budget the planner admitted on.
    #[test]
    fn never_more_than_the_charged_look_ahead_is_resident() {
        let last_layer = last_required_layer();
        let live = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let (build_live, build_peak) = (live.clone(), peak.clone());
        stream_layers(
            last_layer,
            1.0f64,
            move |index| Ok(CountedLayer::build(index, &build_live, &build_peak)),
            |layer: &CountedLayer, state: &f64| advance(layer, state),
            &CAPTURE_LAYERS,
        )
        .unwrap();
        assert_eq!(
            peak.load(Ordering::SeqCst),
            (residency::MISTRAL3_DEFAULT_LOOKAHEAD + 1) as usize,
            "the streamed encoder holds the running layer and the prefetched one"
        );
        assert_eq!(
            live.load(Ordering::SeqCst),
            0,
            "every layer is released before the stack returns"
        );
    }

    /// A build failure surfaces as an error naming the layer, never a panic
    /// escaping the scoped thread.
    #[test]
    fn a_failed_prefetch_is_reported_not_swallowed() {
        let error = stream_layers(
            5,
            1.0f64,
            |index| {
                if index == 3 {
                    anyhow::bail!("synthetic failure")
                }
                Ok(index)
            },
            |layer: &usize, state: &f64| Ok(state + *layer as f64),
            &[],
        )
        .unwrap_err();
        assert!(error.to_string().contains("synthetic failure"));
    }

    #[test]
    fn prompt_matches_official_flux2_dev_template_and_removes_image_markers() {
        assert_eq!(
            format_prompt("put [IMG] beside [IMG] the tree"),
            format!(
                "<s>[SYSTEM_PROMPT]{SYSTEM_PROMPT}[/SYSTEM_PROMPT][INST]put  beside  the tree[/INST]"
            )
        );
    }

    #[test]
    fn hidden_state_indices_are_after_layers_ten_twenty_and_thirty() {
        assert_eq!(CAPTURE_LAYERS, [9, 19, 29]);
    }

    /// Write a safetensors file whose header carries exactly `keys` (each a
    /// 1-element F32 tensor). Only the header is ever read by the resolver.
    fn write_header_fixture(name: &str, keys: &[String]) -> PathBuf {
        use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
        let path = std::env::temp_dir().join(format!(
            "mold-mistral3-{name}-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let payload = 0.0f32.to_le_bytes();
        let mut views = std::collections::HashMap::new();
        for key in keys {
            views.insert(
                key.clone(),
                TensorView::new(Dtype::F32, vec![1], &payload).unwrap(),
            );
        }
        serialize_to_file(&views, &None, &path).unwrap();
        path
    }

    fn language_model_keys(prefix: &str, through_layer: usize) -> Vec<String> {
        let mut keys = vec![format!("{prefix}.embed_tokens.weight")];
        for layer in 0..=through_layer {
            keys.push(format!("{prefix}.layers.{layer}.input_layernorm.weight"));
        }
        keys
    }

    /// The encoder is identified by the tensors it needs, not by a shard
    /// count: BFL ships it as eight `language_model.model.*` shards and
    /// Comfy-Org ships the same weights as one `model.*` file.
    #[test]
    fn both_published_layouts_resolve_to_their_language_model_prefix() {
        for prefix in MISTRAL3_LM_PREFIXES {
            let path = write_header_fixture(
                "layout",
                &language_model_keys(prefix, last_required_layer()),
            );
            assert_eq!(
                resolve_lm_prefix(std::slice::from_ref(&path)).unwrap(),
                prefix,
            );
            std::fs::remove_file(&path).ok();
        }
    }

    /// A sharded encoder is resolved across its files, exactly as the streamed
    /// load reads it.
    #[test]
    fn a_sharded_encoder_resolves_across_its_shards() {
        let all = language_model_keys("language_model.model", last_required_layer());
        let (head, tail) = all.split_at(all.len() / 2);
        let first = write_header_fixture("shard1", head);
        let second = write_header_fixture("shard2", tail);
        assert_eq!(
            resolve_lm_prefix(&[first.clone(), second.clone()]).unwrap(),
            "language_model.model",
        );
        std::fs::remove_file(&first).ok();
        std::fs::remove_file(&second).ok();
    }

    /// A truncated encoder must be refused at load with the missing layer
    /// named, rather than failing mid-stream on layer 29 after minutes of
    /// setup.
    #[test]
    fn an_encoder_missing_a_streamed_layer_is_refused_by_name() {
        let path = write_header_fixture(
            "truncated",
            &language_model_keys("model", last_required_layer() - 1),
        );
        let error = resolve_lm_prefix(std::slice::from_ref(&path))
            .expect_err("a truncated encoder must be refused")
            .to_string();
        assert!(
            error.contains(&format!("layer {} is missing", last_required_layer())),
            "unhelpful error: {error}",
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_checkpoint_without_a_language_model_is_refused() {
        let path = write_header_fixture("no-lm", &["vision_tower.ln_pre.weight".to_string()]);
        let error = resolve_lm_prefix(std::slice::from_ref(&path))
            .expect_err("a checkpoint with no language model must be refused")
            .to_string();
        assert!(error.contains("no Mistral3 language model"), "{error}");
        std::fs::remove_file(&path).ok();

        assert!(resolve_lm_prefix(&[]).is_err());
    }

    #[test]
    fn streamed_peak_accounts_for_the_resolved_weight_dtype() {
        assert_eq!(
            streamed_peak_weight_bytes(DType::F32),
            streamed_peak_weight_bytes(DType::BF16) * 2
        );
    }

    #[test]
    fn attention_mask_is_causal_and_excludes_padding_keys() {
        let mut attention = vec![false; MAX_LENGTH];
        attention[..3].fill(true);
        let mask = causal_padding_mask(&attention, DType::F32, &Device::Cpu).unwrap();

        assert_eq!(
            mask.i((0, 0, 2, 0)).unwrap().to_scalar::<f32>().unwrap(),
            0.0
        );
        assert_eq!(
            mask.i((0, 0, 2, 2)).unwrap().to_scalar::<f32>().unwrap(),
            0.0
        );
        assert!(mask
            .i((0, 0, 1, 2))
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .is_infinite());
        assert!(mask
            .i((0, 0, 511, 3))
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .is_infinite());
    }
}
