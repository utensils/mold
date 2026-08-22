//! EVA02-CLIP-L-14-336 vision tower — PuLID's identity image encoder.
//!
//! Ported from upstream `ToTheBeginning/PuLID` at commit
//! `1aa2fc7df4bf51080df39f355f9abdc1cbfefbaa`:
//!   - `eva_clip/eva_vit_model.py` (`EVAVisionTransformer`, `Attention`,
//!     `SwiGLU`, `Block`, `PatchEmbed`)
//!   - `eva_clip/rope.py` (`VisionRotaryEmbeddingFast`)
//!   - `eva_clip/model.py:110-131` (`_build_vision_tower`, which fixes every
//!     hyper-parameter this file hard-codes)
//!   - `eva_clip/model_configs/EVA02-CLIP-L-14-336.json` (the config itself)
//!
//! candle ships an EVA-02 in `candle-transformers/src/models/eva2.rs` whose
//! attention and SwiGLU *shapes* match, but it is a fixed-448 ImageNet
//! classifier with a different weight layout and no hidden-state taps, so this
//! is a port rather than a reuse — exactly as issue #1229 requires.
//!
//! Only two things leave this module, because they are the only two things
//! `pulid/pipeline_flux.py:175-183` consumes:
//!   - the residual stream captured on the way into blocks 4, 8, 12, 16 and 20
//!     (five tensors of `[1, 577, 1024]`), and
//!   - the L2-normalized `visual.head` projection of the CLS token (768).
//!
//! The tower is ~609 MB of f16 weights, so it follows the crate's
//! drop-and-reload rule: build it, encode, drop it. Nothing here caches.

// The PuLID pipeline that consumes this module lands with the FLUX
// integration (milestone "PuLID-FLUX: functional"); issue #1229 delivers the
// encoders and their parity coverage on their own. Until that consumer exists
// every item here is reachable only from tests, so the dead-code lint would
// otherwise force either a premature `pub` surface or a stub caller.
#![allow(dead_code)]

use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, Linear, Module, VarBuilder};

/// Square input edge in pixels (`EVA02-CLIP-L-14-336.json`, `image_size`).
pub(crate) const IMAGE_SIZE: usize = 336;
/// Patch edge in pixels (`patch_size`).
pub(crate) const PATCH_SIZE: usize = 14;
/// Patch grid edge: 336 / 14.
pub(crate) const GRID_SIZE: usize = IMAGE_SIZE / PATCH_SIZE;
/// CLS + 24 x 24 patches.
pub(crate) const SEQUENCE_LEN: usize = GRID_SIZE * GRID_SIZE + 1;
/// Residual width (`width`).
pub(crate) const EMBED_DIM: usize = 1024;
/// Transformer depth (`layers`).
pub(crate) const DEPTH: usize = 24;
/// `width / head_width` = 1024 / 64.
pub(crate) const NUM_HEADS: usize = 16;
/// `head_width`.
pub(crate) const HEAD_DIM: usize = EMBED_DIM / NUM_HEADS;
/// `int(1024 * 2.6667)` — upstream truncates, so this is 2730 and not 2731
/// (`eva_vit_model.py:261`, `mlp_hidden_dim = int(dim * mlp_ratio)`).
pub(crate) const MLP_HIDDEN_DIM: usize = 2730;
/// CLIP joint-embedding width, the `visual.head` output (`embed_dim`).
pub(crate) const PROJECTION_DIM: usize = 768;
/// `partial(norm_layer, eps=1e-6)` in `model.py:123`. Without apex,
/// `FusedLayerNorm` is eva_clip's own `LayerNorm` (`model.py:25-27`), which is
/// `nn.LayerNorm` with a dtype round-trip, so the arithmetic is plain.
const LAYER_NORM_EPS: f64 = 1e-6;
/// The RoPE grid the checkpoint was trained on (`pt_hw_seq_len`, 224 / 14).
const ROPE_PRETRAINED_GRID: usize = 16;
/// `theta` in `rope.py:96`.
const ROPE_THETA: f64 = 10000.0;

/// Which blocks the hidden states are taken on the way into.
///
/// `eva_vit_model.py:526` appends the residual stream *before* running block
/// `idx` whenever `0 < idx <= 20 && idx % 4 == 0`, so entry 0 is the state
/// after blocks 0..=3 — not the output of block 4. Getting this off by one
/// costs one whole transformer block of drift and nothing else complains.
pub(crate) const HIDDEN_STATE_BLOCKS: [usize; 5] = [4, 8, 12, 16, 20];

/// Everything PuLID reads off the tower.
#[derive(Debug, Clone)]
pub(crate) struct EvaClipVisionOutput {
    /// Five `[1, 577, 1024]` residual snapshots, in `HIDDEN_STATE_BLOCKS`
    /// order. These feed `IDFormer::forward`'s `y`.
    pub(crate) hidden_states: Vec<Tensor>,
    /// `[1, 768]`, L2-normalized along the feature axis
    /// (`pipeline_flux.py:178-179`). This is the CLIP half of the IDFormer's
    /// `cat([arcface_512, clip_768])`.
    pub(crate) cls_projection: Tensor,
}

/// `VisionRotaryEmbeddingFast` (`rope.py:79-137`) materialized as its two
/// `[576, 64]` tables.
///
/// The interpolation the config asks for (`intp_freq: true`) is entirely in
/// the position ramp: positions run `arange(24) / 24 * 16`, so the 24x24
/// inference grid is sampled across the span the 16x16 training grid covered.
#[derive(Debug, Clone)]
pub(crate) struct VisionRotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl VisionRotaryEmbedding {
    /// `dim` is the *half* head dim (`eva_vit_model.py:402`,
    /// `half_head_dim = embed_dim // num_heads // 2`); each of the two spatial
    /// axes contributes `dim` columns, so the tables are `head_dim` wide.
    pub(crate) fn new(
        half_head_dim: usize,
        pretrained_grid: usize,
        grid: usize,
        device: &Device,
    ) -> Result<Self> {
        // `rope.py:96`: freqs = 1 / theta^(arange(0, dim, 2)[:dim/2] / dim).
        let freqs: Vec<f64> = (0..half_head_dim / 2)
            .map(|i| 1.0 / ROPE_THETA.powf((2 * i) as f64 / half_head_dim as f64))
            .collect();
        // `rope.py:105`: t = arange(ft_seq_len) / ft_seq_len * pt_seq_len.
        let positions: Vec<f64> = (0..grid)
            .map(|i| i as f64 / grid as f64 * pretrained_grid as f64)
            .collect();

        // `rope.py:107-109`: outer product, each frequency repeated twice
        // (`'... n -> ... (n r)', r = 2`, i.e. interleaved, not halves), then
        // broadcast-concatenated over the two axes into [grid, grid, 2*dim].
        let axis_width = half_head_dim;
        let width = axis_width * 2;
        let mut cos = vec![0.0_f32; grid * grid * width];
        let mut sin = vec![0.0_f32; grid * grid * width];
        for (row, &y) in positions.iter().enumerate() {
            for (column, &x) in positions.iter().enumerate() {
                let base = (row * grid + column) * width;
                for (index, &frequency) in freqs.iter().enumerate() {
                    let (angle_h, angle_w) = (y * frequency, x * frequency);
                    for repeat in 0..2 {
                        let h = base + index * 2 + repeat;
                        let w = h + axis_width;
                        cos[h] = angle_h.cos() as f32;
                        sin[h] = angle_h.sin() as f32;
                        cos[w] = angle_w.cos() as f32;
                        sin[w] = angle_w.sin() as f32;
                    }
                }
            }
        }
        let shape = (grid * grid, width);
        Ok(Self {
            cos: Tensor::from_vec(cos, shape, device)?,
            sin: Tensor::from_vec(sin, shape, device)?,
        })
    }

    fn for_tower(device: &Device) -> Result<Self> {
        Self::new(HEAD_DIM / 2, ROPE_PRETRAINED_GRID, GRID_SIZE, device)
    }

    pub(crate) fn cos(&self) -> &Tensor {
        &self.cos
    }

    pub(crate) fn sin(&self) -> &Tensor {
        &self.sin
    }

    /// `rope.py:23-27` `rotate_half`: pairs are *adjacent* (`(d r) -> d r`
    /// with `r = 2`), so this is the interleaved rotation, not the
    /// split-in-halves one most of candle uses. Swapping the two silently
    /// produces a plausible, wrong image embedding.
    fn rotate_half(xs: &Tensor) -> Result<Tensor> {
        let (batch, heads, tokens, dim) = xs.dims4()?;
        let pairs = xs.reshape((batch, heads, tokens, dim / 2, 2))?;
        let even = pairs.i((.., .., .., .., 0))?;
        let odd = pairs.i((.., .., .., .., 1))?;
        Tensor::stack(&[odd.neg()?, even], D::Minus1)?
            .reshape((batch, heads, tokens, dim))
            .map_err(Into::into)
    }

    /// `x * cos + rotate_half(x) * sin` over `[batch, heads, tokens, dim]`.
    pub(crate) fn apply(&self, xs: &Tensor) -> Result<Tensor> {
        let cos = self.cos.to_dtype(xs.dtype())?;
        let sin = self.sin.to_dtype(xs.dtype())?;
        let rotated = Self::rotate_half(xs)?;
        Ok((xs.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?)
    }
}

/// `Attention` with `subln=True` (`eva_vit_model.py:106-243`).
///
/// Three quirks all come from the same place and all matter: q/k/v are
/// separate biasless `nn.Linear`s whose bias is supplied out of band as
/// `q_bias` / `v_bias` with **no k bias at all** (`:176-178`), an
/// `inner_attn_ln` sits between the attention output and `proj` (`:240-241`),
/// and RoPE is applied to the patch tokens only, never to CLS (`:195-201`).
#[derive(Debug)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    inner_attn_ln: LayerNorm,
    proj: Linear,
    scale: f64,
}

impl Attention {
    fn new(vb: VarBuilder) -> Result<Self> {
        let shape = (EMBED_DIM, EMBED_DIM);
        let q_bias = vb.get(EMBED_DIM, "q_bias")?;
        let v_bias = vb.get(EMBED_DIM, "v_bias")?;
        Ok(Self {
            q_proj: Linear::new(vb.get(shape, "q_proj.weight")?, Some(q_bias)),
            k_proj: Linear::new(vb.get(shape, "k_proj.weight")?, None),
            v_proj: Linear::new(vb.get(shape, "v_proj.weight")?, Some(v_bias)),
            inner_attn_ln: layer_norm(EMBED_DIM, vb.pp("inner_attn_ln"))?,
            proj: Linear::new(
                vb.get(shape, "proj.weight")?,
                Some(vb.get(EMBED_DIM, "proj.bias")?),
            ),
            scale: (HEAD_DIM as f64).powf(-0.5),
        })
    }

    fn split_heads(xs: &Tensor) -> Result<Tensor> {
        let (batch, tokens, _) = xs.dims3()?;
        Ok(xs
            .reshape((batch, tokens, NUM_HEADS, HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?)
    }

    /// `eva_vit_model.py:195-201` — split CLS off, rotate the rest, re-join.
    fn apply_rope(rope: &VisionRotaryEmbedding, xs: &Tensor) -> Result<Tensor> {
        let cls = xs.i((.., .., ..1, ..))?;
        let patches = xs.i((.., .., 1.., ..))?.contiguous()?;
        Ok(Tensor::cat(&[cls, rope.apply(&patches)?], 2)?.contiguous()?)
    }

    fn forward(&self, xs: &Tensor, rope: &VisionRotaryEmbedding) -> Result<Tensor> {
        let (batch, tokens, _) = xs.dims3()?;
        let q = Self::apply_rope(rope, &Self::split_heads(&self.q_proj.forward(xs)?)?)?;
        let k = Self::apply_rope(rope, &Self::split_heads(&self.k_proj.forward(xs)?)?)?;
        let v = Self::split_heads(&self.v_proj.forward(xs)?)?;

        // `:218-239`: q is scaled before the matmul, softmax in the working
        // dtype, then heads are merged back.
        let scores = (q * self.scale)?.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let attended = weights
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((batch, tokens, EMBED_DIM))?;
        Ok(self.proj.forward(&self.inner_attn_ln.forward(&attended)?)?)
    }
}

/// Naive SwiGLU (`eva_vit_model.py:81-104`): `w3(ffn_ln(silu(w1(x)) * w2(x)))`.
/// All three linears carry biases here, unlike the gated MLPs elsewhere in the
/// crate.
#[derive(Debug)]
struct SwiGlu {
    w1: Linear,
    w2: Linear,
    ffn_ln: LayerNorm,
    w3: Linear,
}

impl SwiGlu {
    fn new(vb: VarBuilder) -> Result<Self> {
        let load = |name: &str, out: usize, inp: usize| -> Result<Linear> {
            Ok(Linear::new(
                vb.get((out, inp), &format!("{name}.weight"))?,
                Some(vb.get(out, &format!("{name}.bias"))?),
            ))
        };
        Ok(Self {
            w1: load("w1", MLP_HIDDEN_DIM, EMBED_DIM)?,
            w2: load("w2", MLP_HIDDEN_DIM, EMBED_DIM)?,
            ffn_ln: layer_norm(MLP_HIDDEN_DIM, vb.pp("ffn_ln"))?,
            w3: load("w3", EMBED_DIM, MLP_HIDDEN_DIM)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gated = (self.w1.forward(xs)?.silu()? * self.w2.forward(xs)?)?;
        Ok(self.w3.forward(&self.ffn_ln.forward(&gated)?)?)
    }
}

/// `Block` (`eva_vit_model.py:246-302`) in the one configuration this
/// checkpoint uses: pre-norm, no layer scale (`ls_init_value` is `None`, so
/// `gamma_1`/`gamma_2` do not exist) and no drop path at inference.
#[derive(Debug)]
struct Block {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    mlp: SwiGlu,
}

impl Block {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: layer_norm(EMBED_DIM, vb.pp("norm1"))?,
            attn: Attention::new(vb.pp("attn"))?,
            norm2: layer_norm(EMBED_DIM, vb.pp("norm2"))?,
            mlp: SwiGlu::new(vb.pp("mlp"))?,
        })
    }

    fn forward(&self, xs: &Tensor, rope: &VisionRotaryEmbedding) -> Result<Tensor> {
        let xs = (xs + self.attn.forward(&self.norm1.forward(xs)?, rope)?)?;
        Ok((&xs + self.mlp.forward(&self.norm2.forward(&xs)?)?)?)
    }
}

fn layer_norm(size: usize, vb: VarBuilder) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        vb.get(size, "weight")?,
        vb.get(size, "bias")?,
        LAYER_NORM_EPS,
    ))
}

/// L2-normalize each row, **in the tensor's own dtype**.
///
/// `pulid/pipeline_flux.py:178-179`:
///
/// ```python
/// id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
/// id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)
/// ```
///
/// `id_cond_vit` is whatever the tower returned, and the tower was run in
/// `weight_dtype` (`pipeline_flux.py:176`,
/// `face_features_image.to(self.weight_dtype)`) — bfloat16 by default. So
/// upstream takes the norm and the division in the narrow dtype, and mold has
/// to as well or a BF16 deployment rounds differently from the reference it is
/// supposed to match. Widening to f32 here looks like a free accuracy win and
/// is actually a divergence.
///
/// The result stays in the input dtype for the same reason, and because it is
/// concatenated with the ArcFace embedding before the IDFormer sees it — an F32
/// half of that concatenation would be a dtype mismatch at the join.
fn l2_normalize_rows(xs: &Tensor) -> Result<Tensor> {
    let norm = xs.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    Ok(xs.broadcast_div(&norm)?)
}

/// The tower.
///
/// Build it from a `VarBuilder` rooted at the **`visual.` prefix already
/// stripped** — that is what [`super::pickle_convert`] writes, and what
/// keeps this file from knowing anything about the CLIP text half.
#[derive(Debug)]
pub(crate) struct EvaClipVisionTower {
    patch_embed: Conv2d,
    cls_token: Tensor,
    pos_embed: Tensor,
    rope: VisionRotaryEmbedding,
    blocks: Vec<Block>,
    norm: LayerNorm,
    head: Linear,
    device: Device,
    dtype: DType,
}

impl EvaClipVisionTower {
    /// Build from an artifact [`super::pickle_convert`] has already
    /// authenticated.
    ///
    /// The production entry point, and deliberately not a path: see
    /// [`super::pickle_convert::AuthenticatedArtifact`] for why a loader that
    /// hashed a pathname and then reopened it would be re-resolving a name a
    /// shared model root lets another member rename. [`Self::new`] stays for
    /// the golden test, which builds a `VarBuilder` over a file it converted
    /// itself moments earlier inside its own temporary directory.
    /// `dtype` is the tower's WORKING dtype, chosen by
    /// [`crate::identity::extraction::eva_working_dtype`] from the device it
    /// will run on. It is a parameter rather than a constant because the
    /// derived file stores f16 and asking for f32 costs a widening pass over
    /// 609 MB into ~1.2 GB — `pulid-perf.md` §4 measured that pass as half of
    /// the single largest line item in the whole extraction. Upstream runs
    /// this tower narrow too (`PuLID/pulid/pipeline_flux.py:60` casts it to
    /// `weight_dtype`, bf16 in `app_flux.py:45`), so the narrow arm is
    /// upstream's own behaviour and the wide one is mold's CPU concession.
    pub(crate) fn from_authenticated(
        artifact: &super::pickle_convert::AuthenticatedArtifact,
        device: &Device,
        dtype: candle_core::DType,
    ) -> Result<Self> {
        let vb = VarBuilder::from_slice_safetensors(artifact.bytes(), dtype, device).with_context(
            || {
                format!(
                    "reading the vision tower {}",
                    artifact.display_path().display()
                )
            },
        )?;
        Self::new(vb, device)
    }

    pub(crate) fn new(vb: VarBuilder, device: &Device) -> Result<Self> {
        let dtype = vb.dtype();
        let patch_embed = Conv2d::new(
            vb.get(
                (EMBED_DIM, 3, PATCH_SIZE, PATCH_SIZE),
                "patch_embed.proj.weight",
            )?,
            Some(vb.get(EMBED_DIM, "patch_embed.proj.bias")?),
            Conv2dConfig {
                stride: PATCH_SIZE,
                ..Default::default()
            },
        );
        let blocks = (0..DEPTH)
            .map(|index| Block::new(vb.pp(format!("blocks.{index}"))))
            .collect::<Result<Vec<_>>>()
            .context("failed to build an EVA02-CLIP vision block")?;
        Ok(Self {
            patch_embed,
            cls_token: vb.get((1, 1, EMBED_DIM), "cls_token")?,
            pos_embed: vb.get((1, SEQUENCE_LEN, EMBED_DIM), "pos_embed")?,
            rope: VisionRotaryEmbedding::for_tower(device)?,
            blocks,
            norm: layer_norm(EMBED_DIM, vb.pp("norm"))?,
            head: Linear::new(
                vb.get((PROJECTION_DIM, EMBED_DIM), "head.weight")?,
                Some(vb.get(PROJECTION_DIM, "head.bias")?),
            ),
            device: device.clone(),
            dtype,
        })
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn dtype(&self) -> DType {
        self.dtype
    }

    /// `pixels` is `[batch, 3, 336, 336]`, already resized and normalized by
    /// [`super::eva_clip_preprocess`].
    pub(crate) fn forward(&self, pixels: &Tensor) -> Result<EvaClipVisionOutput> {
        let (batch, channels, height, width) = pixels.dims4()?;
        ensure!(
            channels == 3 && height == IMAGE_SIZE && width == IMAGE_SIZE,
            "EVA02-CLIP expects [batch, 3, {IMAGE_SIZE}, {IMAGE_SIZE}], got \
             [{batch}, {channels}, {height}, {width}]"
        );
        let pixels = pixels.to_dtype(self.dtype)?.to_device(&self.device)?;

        // `eva_vit_model.py:325`: conv, flatten the grid, put tokens last.
        let patches = self
            .patch_embed
            .forward(&pixels)?
            .flatten_from(2)?
            .transpose(1, 2)?;
        // `:504-509`: prepend CLS, then add the absolute position embedding.
        let cls = self.cls_token.expand((batch, 1, EMBED_DIM))?;
        let mut xs = Tensor::cat(&[cls, patches], 1)?
            .broadcast_add(&self.pos_embed)?
            .contiguous()?;

        let mut hidden_states = Vec::with_capacity(HIDDEN_STATE_BLOCKS.len());
        for (index, block) in self.blocks.iter().enumerate() {
            if HIDDEN_STATE_BLOCKS.contains(&index) {
                hidden_states.push(xs.clone());
            }
            xs = block.forward(&xs, &self.rope)?;
        }
        ensure!(
            hidden_states.len() == HIDDEN_STATE_BLOCKS.len(),
            "expected {} hidden states, captured {}",
            HIDDEN_STATE_BLOCKS.len(),
            hidden_states.len()
        );

        // `:534-538`: `use_mean_pooling` is false for this config
        // (`model.py:114`, `global_average_pool` defaults to `False`), so
        // `fc_norm` does not exist and the pooled feature is the normed CLS
        // token. `:545` then projects it through `head`.
        let cls = self.norm.forward(&xs)?.i((.., 0, ..))?;
        let projection = self.head.forward(&cls)?;
        let cls_projection = l2_normalize_rows(&projection)?;

        Ok(EvaClipVisionOutput {
            hidden_states,
            cls_projection,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pulid_fixtures::{
        gather_probe, golden, max_errors, pulid_asset, scale_relative_error, DeterministicStream,
        GoldenStats, SEED_TOWER_INPUT, SEED_TOWER_PROBE,
    };

    #[test]
    fn the_geometry_matches_the_published_config() {
        assert_eq!(GRID_SIZE, 24);
        assert_eq!(SEQUENCE_LEN, 577);
        assert_eq!(HEAD_DIM, 64);
        // `int(1024 * 2.6667)` truncates; the checkpoint's `mlp.w1.weight` is
        // [2730, 1024], so an accidental round to 2731 fails to load.
        assert_eq!(MLP_HIDDEN_DIM, 2730);
        assert_eq!(MLP_HIDDEN_DIM, (EMBED_DIM as f64 * 2.6667) as usize);
    }

    /// The RoPE tables are derivable, but the checkpoint also ships them as
    /// `visual.rope.freqs_{cos,sin}`. Six rows of upstream's own buffer are
    /// committed, so this runs without any weights.
    #[test]
    fn the_rope_table_matches_the_checkpoint_buffer() {
        let rope = VisionRotaryEmbedding::for_tower(&Device::Cpu).unwrap();
        assert_eq!(rope.cos().dims(), &[576, 64]);
        let rows = [0_usize, 1, 23, 24, 300, 575];
        for (name, table) in [("cos", rope.cos()), ("sin", rope.sin())] {
            let expected = golden(&format!("rope.freqs_{name}.rows"));
            let indices = Tensor::from_vec(
                rows.iter().map(|&r| r as u32).collect::<Vec<_>>(),
                rows.len(),
                &Device::Cpu,
            )
            .unwrap();
            let actual = table.index_select(&indices, 0).unwrap();
            let (absolute, _) = max_errors(
                &actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                &expected.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            );
            // The checkpoint stores these as f16, so the tolerance is f16
            // resolution near 1.0 and not f32 parity.
            assert!(absolute < 1e-3, "rope {name} drifted by {absolute}");
        }
    }

    /// Interpolation is in the position ramp, nowhere else: row `r` of the
    /// table is patch `(r / 24, r % 24)`, and the 24-step ramp must cover the
    /// 16-wide trained span (`rope.py:105`).
    #[test]
    fn rope_interpolates_the_trained_grid_onto_the_inference_grid() {
        let rope = VisionRotaryEmbedding::for_tower(&Device::Cpu).unwrap();
        let cos = rope.cos().to_vec2::<f32>().unwrap();
        // Position 0 has angle 0 on both axes, so the whole row is cos(0).
        assert!(cos[0].iter().all(|value| (value - 1.0).abs() < 1e-6));
        // The lowest frequency is 1.0, so column 0 of row `r` is
        // cos(row_index * 16 / 24).
        for row in [1_usize, 5, 23, 575] {
            let y = (row / GRID_SIZE) as f64 / GRID_SIZE as f64 * ROPE_PRETRAINED_GRID as f64;
            assert!(
                (cos[row][0] as f64 - y.cos()).abs() < 1e-6,
                "row {row} height angle"
            );
            let x = (row % GRID_SIZE) as f64 / GRID_SIZE as f64 * ROPE_PRETRAINED_GRID as f64;
            assert!(
                (cos[row][HEAD_DIM / 2] as f64 - x.cos()).abs() < 1e-6,
                "row {row} width angle"
            );
        }
        // Each frequency is repeated over an adjacent pair, which is what
        // makes `rotate_half` interleaved.
        for row in [3_usize, 100] {
            assert_eq!(cos[row][0], cos[row][1]);
            assert_eq!(cos[row][2], cos[row][3]);
        }
    }

    /// `rotate_half` pairs adjacent lanes. A half-split implementation would
    /// pass every shape check and quietly change the embedding.
    #[test]
    fn rotate_half_pairs_adjacent_lanes() {
        let xs =
            Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (1, 1, 1, 4), &Device::Cpu).unwrap();
        let rotated = VisionRotaryEmbedding::rotate_half(&xs).unwrap();
        assert_eq!(
            rotated.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![-2.0, 1.0, -4.0, 3.0]
        );
    }

    /// RoPE must not touch CLS. Rotating it would mix a positionless token
    /// into the grid and only show up as a slightly wrong identity.
    #[test]
    fn rope_leaves_the_cls_token_alone() {
        let device = Device::Cpu;
        let rope = VisionRotaryEmbedding::for_tower(&device).unwrap();
        let xs = DeterministicStream::new(SEED_TOWER_INPUT)
            .tensor(&[1, NUM_HEADS, SEQUENCE_LEN, HEAD_DIM], &device);
        let out = Attention::apply_rope(&rope, &xs).unwrap();
        assert_eq!(out.dims(), xs.dims());
        let before = xs.i((.., .., 0, ..)).unwrap().flatten_all().unwrap();
        let after = out.i((.., .., 0, ..)).unwrap().flatten_all().unwrap();
        assert_eq!(
            before.to_vec1::<f32>().unwrap(),
            after.to_vec1::<f32>().unwrap()
        );
        // ...and it must touch everything else.
        let patch_before = xs.i((.., .., 5, ..)).unwrap().flatten_all().unwrap();
        let patch_after = out.i((.., .., 5, ..)).unwrap().flatten_all().unwrap();
        assert_ne!(
            patch_before.to_vec1::<f32>().unwrap(),
            patch_after.to_vec1::<f32>().unwrap()
        );
    }

    /// `silu(w1 x) * w2 x` and not the other way round — swapping the gate is
    /// the classic SwiGLU port bug and both orders type-check.
    #[test]
    fn swiglu_gates_with_w1() {
        // Width 3, not 2: LayerNorm over two elements always yields the same
        // +-1 pattern, which would hide the very swap this test exists for.
        const WIDTH: usize = 3;
        let device = Device::Cpu;
        let diagonal = |scale: f32| {
            let mut data = vec![0.0_f32; WIDTH * WIDTH];
            for i in 0..WIDTH {
                data[i * WIDTH + i] = scale;
            }
            Tensor::from_vec(data, (WIDTH, WIDTH), &device).unwrap()
        };
        let zeros = Tensor::zeros(WIDTH, DType::F32, &device).unwrap();
        let ones = Tensor::ones(WIDTH, DType::F32, &device).unwrap();
        // w2 doubles its input, so `silu(w1 x) * w2(x)` and `silu(w2 x) * w1(x)`
        // are numerically distinguishable.
        let module = SwiGlu {
            w1: Linear::new(diagonal(1.0), Some(zeros.clone())),
            w2: Linear::new(diagonal(2.0), Some(zeros.clone())),
            ffn_ln: LayerNorm::new(ones.clone(), zeros.clone(), LAYER_NORM_EPS),
            w3: Linear::new(diagonal(1.0), Some(zeros.clone())),
        };
        let xs = Tensor::from_vec(vec![1.0_f32, -1.0, 0.25], (1, 1, WIDTH), &device).unwrap();
        let reference = |gate: &Tensor, value: &Tensor| {
            let hidden = (gate.silu().unwrap() * value).unwrap();
            LayerNorm::new(ones.clone(), zeros.clone(), LAYER_NORM_EPS)
                .forward(&hidden)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let doubled = (&xs * 2.0).unwrap();
        let actual = module
            .forward(&xs)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let (absolute, _) = max_errors(&actual, &reference(&xs, &doubled));
        assert!(absolute < 1e-6, "swiglu drifted by {absolute}");
        // The swapped order is a genuinely different function here.
        let (swapped, _) = max_errors(&actual, &reference(&doubled, &xs));
        assert!(swapped > 1e-3, "the gate order is not observable");
    }

    /// Upstream normalizes `id_cond_vit` in `weight_dtype`
    /// (`pipeline_flux.py:178-179`, on the tensor the tower returned at
    /// `:176`), so mold must not widen to f32 first. In f32 this is a no-op,
    /// which the parity goldens still cover; in a narrow dtype it is the whole
    /// difference.
    #[test]
    fn l2_normalization_stays_in_the_working_dtype() {
        let device = Device::Cpu;
        let raw = DeterministicStream::new(SEED_TOWER_INPUT).tensor(&[1, PROJECTION_DIM], &device);

        // F32: unit length, and the reference behaviour is unchanged.
        let f32_out = l2_normalize_rows(&raw).unwrap();
        assert_eq!(f32_out.dtype(), DType::F32);
        let norm: f32 = f32_out
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "f32 norm {norm}");

        // F16: the result stays F16 — an F32 result would also be a dtype
        // mismatch where this is concatenated with the ArcFace embedding.
        let narrow = raw.to_dtype(DType::F16).unwrap();
        let f16_out = l2_normalize_rows(&narrow).unwrap();
        assert_eq!(f16_out.dtype(), DType::F16, "the working dtype was widened");

        // ...and it matches a torch-style reference computed the same way:
        // norm and division both in F16, exactly `torch.div(x, torch.norm(x))`.
        let values = narrow
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let reference_norm =
            half::f16::from_f32(values.iter().map(|v| v * v).sum::<f32>().sqrt()).to_f32();
        let expected: Vec<f32> = values
            .iter()
            .map(|v| half::f16::from_f32(v / reference_norm).to_f32())
            .collect();
        let actual = f16_out
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let (absolute, _) = max_errors(&actual, &expected);
        // Measured 2.4e-4. The reference above sums the 768 squares in f32 and
        // rounds once; candle reduces in the storage dtype, and F16 addition
        // near the resulting norm (~9) quantizes at ~8e-3, so the two norms
        // differ by a few ulps and that propagates. Which accumulator a
        // reduction uses is a backend detail neither upstream nor mold pins —
        // what this test pins is that the DIVISION and the RESULT stay in the
        // working dtype, which is what `pipeline_flux.py:178-179` does.
        assert!(absolute < 1e-3, "f16 normalization drifted by {absolute}");

        // The widened path this replaced is measurably different, so the test
        // is not vacuous.
        let widened = l2_normalize_rows(&narrow.to_dtype(DType::F32).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let (widened_difference, _) = max_errors(&actual, &widened);
        assert!(
            widened_difference > 0.0,
            "widening would have produced identical bytes"
        );
    }

    fn load_tower(device: &Device) -> EvaClipVisionTower {
        let source = pulid_asset("EVA02_CLIP_L_336_psz14_s6B.pt");
        let staging = std::env::temp_dir().join("mold-pulid-eva-parity");
        std::fs::create_dir_all(&staging).unwrap();
        let converted = staging.join(super::super::pickle_convert::EVA_DERIVED_FILENAME);
        super::super::pickle_convert::convert_eva_clip_vision(&source, &converted).unwrap();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[converted], DType::F32, device).unwrap()
        };
        EvaClipVisionTower::new(vb, device).unwrap()
    }

    /// Full parity against upstream on the pinned checkpoint.
    ///
    /// ```text
    /// MOLD_TEST_PULID_ASSETS=/path/to/pulid-assets \
    ///   cargo test --release -p mold-ai-inference --lib eva_clip \
    ///     -- --ignored --nocapture --test-threads=1
    /// ```
    ///
    /// `--test-threads=1` is not cosmetic: candle's CPU gemm splits work by
    /// the parallelism it finds, so running beside other heavy tests changes
    /// the accumulation order and moves the last few digits.
    ///
    /// ## Tolerances, and why they differ by output
    ///
    /// Measured on an aarch64-darwin CPU in f32 against the goldens:
    ///
    /// ```text
    /// hidden_0  abs 7.4e-4  peak  40.4  ->  1.8e-5 of scale
    /// hidden_1  abs 5.9e-3  peak  48.8  ->  1.2e-4
    /// hidden_2  abs 6.5e-3  peak  57.6  ->  1.1e-4
    /// hidden_3  abs 2.3e-2  peak 120.0  ->  1.9e-4
    /// hidden_4  abs 3.4e-2  peak 256.6  ->  1.3e-4
    /// cls_projection                    abs 1.3e-5 (unit-length vector)
    /// ```
    ///
    /// The hidden states are raw residual stream, and EVA02 is notorious for
    /// activations two orders of magnitude above the bulk; the pre-norm
    /// LayerNorms then subtract nearly-equal large numbers, so f32 rounding
    /// amplifies with depth. Three things say that is what this is rather
    /// than a structural difference:
    ///
    /// - It GROWS with depth and then plateaus (1.8e-5 after four blocks,
    ///   ~1.3e-4 from eight on). A wrong epsilon, a misplaced bias, or a wrong
    ///   RoPE convention is present in block 0 and would be just as large
    ///   relative to scale at `hidden_0` as at `hidden_4`.
    /// - The CLS projection, which runs strictly deeper, is a hundred times
    ///   TIGHTER, because the final LayerNorm and the L2 normalization divide
    ///   the noise out along with the outliers that produced it.
    /// - The IDFormer parity test, sharing this harness and this metric,
    ///   lands at 1.5e-7 — so the comparison is capable of showing exactness
    ///   when exactness is there.
    ///
    /// An f64 cross-check would settle it outright but is not available:
    /// candle's fused LayerNorm has no F64 kernel ("unsupported dtype for
    /// rmsnorm F64").
    ///
    /// The CLS projection is the strong evidence and gets the strict bound: it
    /// runs the full 24 blocks plus the final norm, `head`, and L2
    /// normalization, and lands within 1.3e-5 on a unit vector. Nothing with a
    /// wrong weight, a wrong RoPE convention, or a wrong tap index survives
    /// that.
    #[test]
    #[ignore = "requires the pinned PuLID checkpoints via MOLD_TEST_PULID_ASSETS"]
    fn tower_matches_upstream_hidden_states_and_projection() {
        let device = Device::Cpu;
        let tower = load_tower(&device);
        let pixels = DeterministicStream::new(SEED_TOWER_INPUT)
            .tensor(&[1, 3, IMAGE_SIZE, IMAGE_SIZE], &device);
        let output = tower.forward(&pixels).unwrap();
        assert_eq!(output.hidden_states.len(), HIDDEN_STATE_BLOCKS.len());

        for (index, hidden) in output.hidden_states.iter().enumerate() {
            assert_eq!(hidden.dims(), &[1, SEQUENCE_LEN, EMBED_DIM]);
            // Whole-tensor statistics first: a defect that misses all 512
            // probe indices still moves these.
            let expected_stats = GoldenStats::load(&format!("tower.hidden_{index}.stats"));
            let actual_stats = GoldenStats::measure(hidden);
            expected_stats.assert_matches(&actual_stats, 1e-3, &format!("hidden_{index}"));

            let actual = gather_probe(hidden, SEED_TOWER_PROBE + index as u64);
            let expected = golden(&format!("tower.hidden_{index}.probe"))
                .to_vec1::<f32>()
                .unwrap();
            let error = scale_relative_error(&actual, &expected, expected_stats.peak);
            println!(
                "hidden_{index}: {error:.3e} of the {} scale",
                expected_stats.peak
            );
            assert!(error < 1e-3, "hidden state {index} drifted by {error}");
        }

        assert_eq!(output.cls_projection.dims(), &[1, PROJECTION_DIM]);
        let actual = output
            .cls_projection
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected = golden("tower.cls_projection_normalized")
            .to_vec1::<f32>()
            .unwrap();
        let (absolute, _) = max_errors(&actual, &expected);
        println!("cls_projection: abs {absolute:.3e}");
        assert!(absolute < 1e-4, "projection absolute error {absolute}");
        // Unit length by construction, so this catches a missing or
        // double-applied normalization independently of the golden.
        let norm: f32 = actual.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "projection norm {norm}");
    }

    /// The hidden states are the residual stream *entering* blocks 4/8/12/16/20.
    /// Reading them as block *outputs* is a one-block shift that still produces
    /// five correctly shaped tensors, so it needs its own check.
    #[test]
    #[ignore = "requires the pinned PuLID checkpoints via MOLD_TEST_PULID_ASSETS"]
    fn hidden_states_are_taken_entering_the_tapped_blocks() {
        let device = Device::Cpu;
        let tower = load_tower(&device);
        let pixels = DeterministicStream::new(SEED_TOWER_INPUT)
            .tensor(&[1, 3, IMAGE_SIZE, IMAGE_SIZE], &device);
        let output = tower.forward(&pixels).unwrap();
        // Re-run the first four blocks by hand; the result must be the first
        // hidden state exactly.
        let (batch, ..) = pixels.dims4().unwrap();
        let patches = tower
            .patch_embed
            .forward(&pixels)
            .unwrap()
            .flatten_from(2)
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let cls = tower.cls_token.expand((batch, 1, EMBED_DIM)).unwrap();
        let mut xs = Tensor::cat(&[cls, patches], 1)
            .unwrap()
            .broadcast_add(&tower.pos_embed)
            .unwrap()
            .contiguous()
            .unwrap();
        for block in tower.blocks.iter().take(HIDDEN_STATE_BLOCKS[0]) {
            xs = block.forward(&xs, &tower.rope).unwrap();
        }
        let expected = output.hidden_states[0]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let peak = expected.iter().fold(0.0_f32, |peak, &v| peak.max(v.abs()));
        let error = scale_relative_error(
            &xs.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            &expected,
            peak,
        );
        // Same arithmetic, so this would be exact but for candle's CPU gemm
        // reassociating when the thread pool is contended. A one-block shift
        // is an O(1) difference, so 1e-5 still catches what this is for.
        assert!(
            error < 1e-5,
            "hidden state 0 is not the block-4 input ({error:.3e})"
        );
    }
}
