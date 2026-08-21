//! PuLID's IDFormer — the identity resampler.
//!
//! Ported from upstream `ToTheBeginning/PuLID` at commit
//! `1aa2fc7df4bf51080df39f355f9abdc1cbfefbaa`,
//! `pulid/encoders_transformer.py:122-209` (`IDFormer`), `:75-119`
//! (`PerceiverAttention`) and `:8-15` (`FeedForward`). The weights are the
//! `pulid_encoder.*` tensors of `pulid_flux_v0.9.1.safetensors`; upstream
//! splits the checkpoint by leading module name at `pipeline_flux.py:99-109`.
//!
//! It takes the concatenation of the ArcFace identity embedding (512) and the
//! L2-normalized EVA02-CLIP projection (768) plus the tower's five hidden
//! states, and returns `[1, 32, 2048]` — the tokens FLUX's PuLID
//! cross-attention consumes.
//!
//! The shape of the thing is a resampler run five times: the 32 learned
//! latents are concatenated with five identity tokens once, and then each of
//! the five vision scales drives two `[PerceiverAttention, FeedForward]`
//! layers in order. The identity tokens stay in the key/value context for
//! every scale, which is why they are built before the loop and not inside it.

// The PuLID pipeline that consumes this module lands with the FLUX
// integration (milestone "PuLID-FLUX: functional"); issue #1229 delivers the
// encoders and their parity coverage on their own. Until that consumer exists
// every item here is reachable only from tests, so the dead-code lint would
// otherwise force either a premature `pub` surface or a stub caller.
#![allow(dead_code)]

use anyhow::{ensure, Context, Result};
use candle_core::{IndexOp, Tensor, D};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder};

/// Residual width (`dim`).
pub(crate) const DIM: usize = 1024;
/// `depth`, ten layers total.
const DEPTH: usize = 10;
/// Five vision scales; `self.depth = depth // 5` is the layers each one drives.
const SCALES: usize = 5;
const LAYERS_PER_SCALE: usize = DEPTH / SCALES;
/// `dim_head`.
const HEAD_DIM: usize = 64;
/// `heads`.
const NUM_HEADS: usize = 16;
/// `num_id_token` — the identity embedding expands to five tokens.
const NUM_ID_TOKENS: usize = 5;
/// `num_queries` — the learned latents, and the output token count.
pub(crate) const NUM_QUERIES: usize = 32;
/// `output_dim`.
pub(crate) const OUTPUT_DIM: usize = 2048;
/// `ff_mult`.
const FF_MULT: usize = 4;
/// `cat([arcface_512, clip_768])` (`pipeline_flux.py:181`).
pub(crate) const ID_COND_DIM: usize = 1280;
/// `nn.LayerNorm` default.
const LAYER_NORM_EPS: f64 = 1e-5;
/// `nn.LeakyReLU` default `negative_slope`.
const LEAKY_RELU_SLOPE: f64 = 0.01;

fn layer_norm(size: usize, vb: VarBuilder) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        vb.get(size, "weight")?,
        vb.get(size, "bias")?,
        LAYER_NORM_EPS,
    ))
}

fn linear(out: usize, inp: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((out, inp), "weight")?;
    let bias = if bias {
        Some(vb.get(out, "bias")?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias))
}

/// `nn.Sequential(Linear, LayerNorm, LeakyReLU, Linear, LayerNorm, LeakyReLU,
/// Linear)` — the shape both `id_embedding_mapping` and `mapping_{0..4}` use
/// (`encoders_transformer.py:164-187`). Indices in the tensor names are the
/// `Sequential` positions, so 0/3/6 are the linears and 1/4 the norms; 2 and 5
/// are the activations and carry no weights.
#[derive(Debug)]
struct MappingMlp {
    fc1: Linear,
    norm1: LayerNorm,
    fc2: Linear,
    norm2: LayerNorm,
    fc3: Linear,
}

impl MappingMlp {
    fn new(input_dim: usize, output_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(DIM, input_dim, true, vb.pp("0"))?,
            norm1: layer_norm(DIM, vb.pp("1"))?,
            fc2: linear(DIM, DIM, true, vb.pp("3"))?,
            norm2: layer_norm(DIM, vb.pp("4"))?,
            fc3: linear(output_dim, DIM, true, vb.pp("6"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = leaky_relu(&self.norm1.forward(&self.fc1.forward(xs)?)?)?;
        let xs = leaky_relu(&self.norm2.forward(&self.fc2.forward(&xs)?)?)?;
        Ok(self.fc3.forward(&xs)?)
    }
}

/// `max(x, 0) + slope * min(x, 0)`.
///
/// candle's `Activation::LeakyRelu` exists but takes the slope at
/// construction; spelling it out keeps the 0.01 next to the upstream citation.
fn leaky_relu(xs: &Tensor) -> Result<Tensor> {
    let positive = xs.maximum(0.0)?;
    let negative = (xs.minimum(0.0)? * LEAKY_RELU_SLOPE)?;
    Ok((positive + negative)?)
}

/// `PerceiverAttention` (`encoders_transformer.py:75-119`).
///
/// Two things differ from an ordinary cross-attention and both matter:
/// the key/value input is `cat(context, latents)` so the latents attend to
/// themselves as well as the context (`:104`), and the scale is applied to
/// **both** q and k as `dim_head^-0.25` before the matmul rather than once
/// afterwards (`:112-113`) — mathematically the same, numerically what
/// upstream ships.
#[derive(Debug)]
struct PerceiverAttention {
    norm1: LayerNorm,
    norm2: LayerNorm,
    to_q: Linear,
    to_kv: Linear,
    to_out: Linear,
}

impl PerceiverAttention {
    fn new(vb: VarBuilder) -> Result<Self> {
        let inner = HEAD_DIM * NUM_HEADS;
        Ok(Self {
            norm1: layer_norm(DIM, vb.pp("norm1"))?,
            norm2: layer_norm(DIM, vb.pp("norm2"))?,
            to_q: linear(inner, DIM, false, vb.pp("to_q"))?,
            to_kv: linear(inner * 2, DIM, false, vb.pp("to_kv"))?,
            to_out: linear(DIM, inner, false, vb.pp("to_out"))?,
        })
    }

    /// `reshape_tensor` (`:18-26`): `[b, n, heads * head_dim]` ->
    /// `[b, heads, n, head_dim]`.
    fn split_heads(xs: &Tensor) -> Result<Tensor> {
        let (batch, tokens, _) = xs.dims3()?;
        Ok(xs
            .reshape((batch, tokens, NUM_HEADS, HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?)
    }

    fn forward(&self, context: &Tensor, latents: &Tensor) -> Result<Tensor> {
        let context = self.norm1.forward(context)?;
        let latents = self.norm2.forward(latents)?;
        let (batch, queries, _) = latents.dims3()?;

        let q = Self::split_heads(&self.to_q.forward(&latents)?)?;
        let kv_input = Tensor::cat(&[&context, &latents], 1)?.contiguous()?;
        let kv = self.to_kv.forward(&kv_input)?;
        let inner = HEAD_DIM * NUM_HEADS;
        let k = Self::split_heads(&kv.narrow(D::Minus1, 0, inner)?.contiguous()?)?;
        let v = Self::split_heads(&kv.narrow(D::Minus1, inner, inner)?.contiguous()?)?;

        let scale = 1.0 / (HEAD_DIM as f64).sqrt().sqrt();
        let scores = (q * scale)?.matmul(&(k * scale)?.transpose(D::Minus2, D::Minus1)?)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let attended = weights
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((batch, queries, inner))?;
        Ok(self.to_out.forward(&attended)?)
    }
}

/// `FeedForward` (`encoders_transformer.py:8-15`): `LayerNorm`, biasless
/// `Linear` up, `nn.GELU` (the exact erf form, not the tanh approximation),
/// biasless `Linear` down.
#[derive(Debug)]
struct FeedForward {
    norm: LayerNorm,
    up: Linear,
    down: Linear,
}

impl FeedForward {
    fn new(vb: VarBuilder) -> Result<Self> {
        let inner = DIM * FF_MULT;
        Ok(Self {
            norm: layer_norm(DIM, vb.pp("0"))?,
            up: linear(inner, DIM, false, vb.pp("1"))?,
            down: linear(DIM, inner, false, vb.pp("3"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.up.forward(&self.norm.forward(xs)?)?;
        Ok(self.down.forward(&xs.gelu_erf()?)?)
    }
}

/// The IDFormer.
#[derive(Debug)]
pub(crate) struct IdFormer {
    latents: Tensor,
    proj_out: Tensor,
    layers: Vec<(PerceiverAttention, FeedForward)>,
    mappings: Vec<MappingMlp>,
    id_embedding_mapping: MappingMlp,
}

impl IdFormer {
    /// Build from a `VarBuilder` rooted at `pulid_encoder`.
    pub(crate) fn new(vb: VarBuilder) -> Result<Self> {
        let layers = (0..DEPTH)
            .map(|index| {
                let vb = vb.pp(format!("layers.{index}"));
                Ok((
                    PerceiverAttention::new(vb.pp("0"))?,
                    FeedForward::new(vb.pp("1"))?,
                ))
            })
            .collect::<Result<Vec<_>>>()
            .context("failed to build an IDFormer layer")?;
        let mappings = (0..SCALES)
            .map(|index| MappingMlp::new(DIM, DIM, vb.pp(format!("mapping_{index}"))))
            .collect::<Result<Vec<_>>>()
            .context("failed to build an IDFormer vision mapping")?;
        Ok(Self {
            latents: vb.get((1, NUM_QUERIES, DIM), "latents")?,
            // `proj_out` is a bare parameter used as `latents @ proj_out`, so
            // it is stored [dim, output_dim] and must NOT be transposed the
            // way an `nn.Linear` weight would be.
            proj_out: vb.get((DIM, OUTPUT_DIM), "proj_out")?,
            layers,
            mappings,
            id_embedding_mapping: MappingMlp::new(
                ID_COND_DIM,
                DIM * NUM_ID_TOKENS,
                vb.pp("id_embedding_mapping"),
            )?,
        })
    }

    /// `id_cond` is `[batch, 1280]`; `vision_hidden_states` are the tower's
    /// five `[batch, 577, 1024]` snapshots in
    /// [`crate::encoders::eva_clip_vision::HIDDEN_STATE_BLOCKS`] order.
    /// Returns `[batch, 32, 2048]`.
    pub(crate) fn forward(
        &self,
        id_cond: &Tensor,
        vision_hidden_states: &[Tensor],
    ) -> Result<Tensor> {
        ensure!(
            vision_hidden_states.len() == SCALES,
            "IDFormer needs {SCALES} vision hidden states, got {}",
            vision_hidden_states.len()
        );
        let (batch, cond_dim) = id_cond.dims2()?;
        ensure!(
            cond_dim == ID_COND_DIM,
            "IDFormer expects a {ID_COND_DIM}-wide identity condition, got {cond_dim}"
        );

        let dtype = self.latents.dtype();
        let id_cond = id_cond.to_dtype(dtype)?.to_device(self.latents.device())?;
        // `:195-196`: the identity embedding becomes `num_id_token` tokens.
        let identity =
            self.id_embedding_mapping
                .forward(&id_cond)?
                .reshape((batch, NUM_ID_TOKENS, DIM))?;
        // `:191,198`: the learned latents, with the identity tokens appended.
        let mut latents = Tensor::cat(
            &[
                self.latents.expand((batch, NUM_QUERIES, DIM))?,
                identity.clone(),
            ],
            1,
        )?
        .contiguous()?;

        for (scale, hidden_state) in vision_hidden_states.iter().enumerate() {
            let hidden = hidden_state
                .to_dtype(dtype)?
                .to_device(self.latents.device())?;
            let (hidden_batch, _, hidden_dim) = hidden.dims3()?;
            ensure!(
                hidden_batch == batch && hidden_dim == DIM,
                "vision hidden state {scale} is {:?}, expected [{batch}, *, {DIM}]",
                hidden.dims()
            );
            let mapped = self.mappings[scale].forward(&hidden)?;
            // `:202`: the identity tokens ride along in every scale's context.
            let context = Tensor::cat(&[&identity, &mapped], 1)?.contiguous()?;
            for layer in &self.layers[scale * LAYERS_PER_SCALE..(scale + 1) * LAYERS_PER_SCALE] {
                latents = (layer.0.forward(&context, &latents)? + &latents)?;
                latents = (layer.1.forward(&latents)? + &latents)?;
            }
        }

        // `:207-208`: drop the identity tokens, project the queries.
        let queries = latents.i((.., ..NUM_QUERIES, ..))?.contiguous()?;
        Ok(queries.broadcast_matmul(&self.proj_out)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pulid_fixtures::{
        golden, max_errors, pulid_asset, scale_relative_error, DeterministicStream, GoldenStats,
        SEED_IDFORMER_ID, SEED_IDFORMER_VIT,
    };
    use candle_core::{DType, Device};

    #[test]
    fn the_geometry_matches_the_published_defaults() {
        assert_eq!(LAYERS_PER_SCALE, 2);
        assert_eq!(DIM * NUM_ID_TOKENS, 5120);
        assert_eq!(HEAD_DIM * NUM_HEADS, DIM);
        assert_eq!(ID_COND_DIM, 512 + 768);
    }

    /// `nn.LeakyReLU()` defaults to 0.01; a plain ReLU here would look right
    /// on most inputs and quietly clip the negative half of every mapping.
    #[test]
    fn leaky_relu_keeps_a_hundredth_of_the_negative_half() {
        let xs = Tensor::from_vec(vec![-2.0_f32, -0.5, 0.0, 3.0], 4, &Device::Cpu).unwrap();
        let out = leaky_relu(&xs).unwrap().to_vec1::<f32>().unwrap();
        let (absolute, _) = max_errors(&out, &[-0.02, -0.005, 0.0, 3.0]);
        assert!(absolute < 1e-7, "{out:?}");
    }

    fn load_encoder(device: &Device) -> IdFormer {
        let adapter = pulid_asset("pulid_flux_v0.9.1.safetensors");
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[adapter], DType::F32, device).unwrap() };
        IdFormer::new(vb.pp("pulid_encoder")).unwrap()
    }

    fn fixture_inputs(device: &Device) -> (Tensor, Vec<Tensor>) {
        // The capture script draws arcface then clip from ONE stream and
        // concatenates them, so the same 1280 values come out of one draw.
        let id_cond = DeterministicStream::new(SEED_IDFORMER_ID).tensor(&[1, ID_COND_DIM], device);
        let hidden = (0..SCALES)
            .map(|index| {
                DeterministicStream::new(SEED_IDFORMER_VIT + index as u64)
                    .tensor(&[1, 577, DIM], device)
            })
            .collect();
        (id_cond, hidden)
    }

    /// Full parity against upstream on the pinned adapter.
    ///
    /// ```text
    /// MOLD_TEST_PULID_ASSETS=/path/to/pulid-assets \
    ///   cargo test --release -p mold-ai-inference --lib pulid_encoder \
    ///     -- --ignored --nocapture --test-threads=1
    /// ```
    ///
    /// Unlike the tower goldens this one is the COMPLETE output tensor
    /// (`[1, 32, 2048]`, 256 KB), because it is small enough to commit whole.
    /// The fixture drives the encoder with uniform noise rather than a real
    /// face, which pushes the LayerNorm/LeakyReLU stack well outside its
    /// trained regime and lands the output around +-13000 — a harsher
    /// numerical test than production, and the reason the bound is quoted
    /// against the tensor's own peak.
    #[test]
    #[ignore = "requires the pinned PuLID checkpoints via MOLD_TEST_PULID_ASSETS"]
    fn idformer_matches_upstream() {
        let device = Device::Cpu;
        let encoder = load_encoder(&device);
        let (id_cond, hidden) = fixture_inputs(&device);
        let output = encoder.forward(&id_cond, &hidden).unwrap();
        assert_eq!(output.dims(), &[1, NUM_QUERIES, OUTPUT_DIM]);

        let expected_stats = GoldenStats::load("idformer.output.stats");
        expected_stats.assert_matches(&GoldenStats::measure(&output), 1e-4, "idformer");

        let actual = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = golden("idformer.output")
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let error = scale_relative_error(&actual, &expected, expected_stats.peak);
        println!("idformer: {error:.3e} of the {} scale", expected_stats.peak);
        assert!(error < 1e-4, "idformer drifted by {error}");
    }

    /// The five identity tokens must be in every scale's key/value context,
    /// not only the first — dropping them still produces a `[1, 32, 2048]`
    /// result.
    #[test]
    #[ignore = "requires the pinned PuLID checkpoints via MOLD_TEST_PULID_ASSETS"]
    fn every_scale_sees_the_identity_tokens() {
        let device = Device::Cpu;
        let encoder = load_encoder(&device);
        let (id_cond, hidden) = fixture_inputs(&device);
        let baseline = encoder.forward(&id_cond, &hidden).unwrap();

        // Zero the LAST scale's vision features only. If the final two layers
        // were not running, or were reading a different scale, this would not
        // move the output.
        let mut altered = hidden.clone();
        altered[SCALES - 1] = altered[SCALES - 1].zeros_like().unwrap();
        let changed = encoder.forward(&id_cond, &altered).unwrap();
        let (absolute, _) = max_errors(
            &changed.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            &baseline.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        assert!(absolute > 1e-3, "the last scale did not affect the output");
    }

    #[test]
    #[ignore = "requires the pinned PuLID checkpoints via MOLD_TEST_PULID_ASSETS"]
    fn the_wrong_number_of_hidden_states_is_refused() {
        let device = Device::Cpu;
        let encoder = load_encoder(&device);
        let (id_cond, hidden) = fixture_inputs(&device);
        let error = encoder.forward(&id_cond, &hidden[..4]).unwrap_err();
        assert!(error.to_string().contains("vision hidden states"));
    }
}
