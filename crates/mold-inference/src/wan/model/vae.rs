//! Wan causal 3-D video VAE (2.1 and 2.2).
//!
//! Ports two upstream reference implementations:
//!
//! - Wan 2.1 — `Wan2.1/wan/modules/vae.py` (4x temporal, 8x8 spatial, z=16,
//!   `dim=96`). Used by T2V-1.3B/14B, I2V-14B, and both Wan 2.2 A14B experts.
//!   `Wan2.2/wan/modules/vae2_1.py` is the same file with the class renamed.
//! - Wan 2.2 — `Wan2.2/wan/modules/vae2_2.py` (4x temporal, 16x16 spatial,
//!   z=48, encoder `dim=160` / decoder `dim=256`). Used by TI2V-5B only.
//!
//! Both are *streaming* autoencoders: every convolution is causal on the time
//! axis and carries a two-frame tail from the previous chunk in a flat cache
//! indexed by conv-visit order. Encode walks the pixel clip in chunks of 1,
//! then 4, 4, 4, ...; decode feeds exactly one latent frame per iteration. The
//! visit order is what binds a cache slot to a convolution, so any divergence
//! in the module walk silently desynchronizes every downstream causal conv —
//! [`tests::wan21_encode_is_chunk_plan_invariant`] and its siblings exist to
//! catch precisely that.
//!
//! candle has no cuDNN conv3d, so [`WanCausalConv3d`] decomposes each 3-D
//! kernel into `kt` `Conv2d` slices summed over the temporal taps, the same
//! strategy as [`crate::ltx2::model::video_vae::Ltx2VideoCausalConv3d`].

use candle_core::{bail, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{ops, Conv2d, Conv2dConfig, Init, VarBuilder};
use std::path::Path;

/// Trailing frames each causal conv keeps from the previous chunk
/// (`vae.py:14`, `vae2_2.py:14`).
const CACHE_T: usize = 2;

/// Pixel frames per encode chunk after the leading single frame
/// (`vae.py:519-534`, `vae2_2.py:786-802`).
const ENCODE_CHUNK_FRAMES: usize = 4;

/// Weight init used when a [`VarBuilder`] has nothing on disk (tests over a
/// `VarMap`). File-backed builders ignore these entirely.
const TEST_WEIGHT_INIT: Init = Init::Randn {
    mean: 0.0,
    stdev: 0.05,
};

// ---------------------------------------------------------------------------
// small tensor helpers
// ---------------------------------------------------------------------------

fn cat2(a: &Tensor, b: &Tensor, dim: usize) -> Result<Tensor> {
    Tensor::cat(&[a, b], dim)
}

fn cat_all(xs: &[Tensor], dim: usize) -> Result<Tensor> {
    let refs = xs.iter().collect::<Vec<_>>();
    Tensor::cat(&refs, dim)
}

/// `x[:, :, -n:]` on a `[b, c, t, h, w]` tensor, clamped to what exists.
fn tail_frames(x: &Tensor, n: usize) -> Result<Tensor> {
    let t = x.dim(2)?;
    let n = n.min(t);
    x.narrow(2, t - n, n)?.contiguous()
}

/// `x[:, :, -1:]` on a `[b, c, t, h, w]` tensor.
fn last_frame(x: &Tensor) -> Result<Tensor> {
    tail_frames(x, 1)
}

/// Apply a 2-D op to every frame of a `[b, c, t, h, w]` tensor.
///
/// Upstream writes this as `rearrange(x, 'b c t h w -> (b t) c h w')`, which is
/// equivalent because the op is independent per sample and per frame. Slicing
/// frame by frame keeps the transient copy at 1/t of the activation, which
/// matters: the decoder's last spatial upsample runs at full output resolution.
fn map_frames<F>(x: &Tensor, mut f: F) -> Result<Tensor>
where
    F: FnMut(&Tensor) -> Result<Tensor>,
{
    let t = x.dim(2)?;
    let mut out = Vec::with_capacity(t);
    for ti in 0..t {
        let frame = x.narrow(2, ti, 1)?.squeeze(2)?.contiguous()?;
        out.push(f(&frame)?.unsqueeze(2)?);
    }
    cat_all(&out, 2)
}

/// `repeat_interleave(x, repeats, dim=1)` — each channel duplicated in place.
fn repeat_interleave_channels(x: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats == 1 {
        return x.contiguous();
    }
    let (b, c, t, h, w) = x.dims5()?;
    x.unsqueeze(2)?
        .broadcast_as(&[b, c, repeats, t, h, w])?
        .contiguous()?
        .reshape(&[b, c * repeats, t, h, w])
}

// ---------------------------------------------------------------------------
// configuration
// ---------------------------------------------------------------------------

/// Which upstream VAE a config describes. The variant selects both the block
/// topology (2.2 wraps every stage in an average-pool / duplicate shortcut) and
/// the checkpoint key layout (2.2 nests `downsamples.{stage}.downsamples.{j}`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WanVaeVariant {
    /// `Wan2.1/wan/modules/vae.py`
    V2_1,
    /// `Wan2.2/wan/modules/vae2_2.py`
    V2_2,
}

/// Per-channel latent statistics, `vae.py:629-639`.
const WAN21_LATENT_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];
const WAN21_LATENT_STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.916,
];

/// Per-channel latent statistics, `vae2_2.py:904-1011`.
const WAN22_LATENT_MEAN: [f32; 48] = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557, -0.1382, 0.0542, 0.2813,
    0.0891, 0.157, -0.0098, 0.0375, -0.1825, -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108,
    -0.2158, 0.2502, -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.123, -0.0313,
    -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.052, 0.3748, 0.0152, 0.1957, 0.1433, -0.2944,
    0.3573, -0.0548, -0.1681, -0.0667,
];
const WAN22_LATENT_STD: [f32; 48] = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.499, 0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901,
    0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944,
    0.5726, 1.2042, 0.5458, 1.6887, 0.3971, 1.06, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
];

#[derive(Clone, Debug)]
pub(crate) struct WanVaeConfig {
    pub variant: WanVaeVariant,
    /// Encoder base width. 2.2's decoder is wider than its encoder.
    pub encoder_dim: usize,
    pub decoder_dim: usize,
    pub z_dim: usize,
    pub dim_mult: Vec<usize>,
    pub num_res_blocks: usize,
    /// One flag per downsampling stage, so `dim_mult.len() - 1` entries. The
    /// decoder's temporal upsample flags are this list reversed
    /// (`vae.py:500`).
    pub temporal_downsample: Vec<bool>,
    /// Pixel-space patchify factor applied before the encoder: 1 for 2.1,
    /// 2 for 2.2 (`vae2_2.py:280-299`).
    pub patch_size: usize,
    pub latent_mean: Vec<f32>,
    pub latent_std: Vec<f32>,
}

impl WanVaeConfig {
    /// The released `Wan2.1_VAE.pth` / `wan_2.1_vae.safetensors`
    /// (`vae.py:592-616`, `vae.py:621-639`).
    pub fn v2_1() -> Self {
        Self {
            variant: WanVaeVariant::V2_1,
            encoder_dim: 96,
            decoder_dim: 96,
            z_dim: 16,
            dim_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            temporal_downsample: vec![false, true, true],
            patch_size: 1,
            latent_mean: WAN21_LATENT_MEAN.to_vec(),
            latent_std: WAN21_LATENT_STD.to_vec(),
        }
    }

    /// The released `Wan2.2_VAE.pth` / `wan2.2_vae.safetensors`
    /// (`vae2_2.py:863-885`, `vae2_2.py:888-1012`; note `dec_dim` is not
    /// forwarded by `_video_vae`, so the decoder keeps `WanVAE_`'s 256).
    pub fn v2_2() -> Self {
        Self {
            variant: WanVaeVariant::V2_2,
            encoder_dim: 160,
            decoder_dim: 256,
            z_dim: 48,
            dim_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            temporal_downsample: vec![false, true, true],
            patch_size: 2,
            latent_mean: WAN22_LATENT_MEAN.to_vec(),
            latent_std: WAN22_LATENT_STD.to_vec(),
        }
    }

    /// The 2.1 topology at `dim=8`, `z=4`. Same stage/key layout as
    /// [`Self::v2_1`] — only the widths shrink — so tests exercise the real
    /// module walk and the real checkpoint key indices.
    pub fn tiny_v2_1() -> Self {
        let mut cfg = Self::v2_1();
        cfg.encoder_dim = 8;
        cfg.decoder_dim = 8;
        cfg.z_dim = 4;
        cfg.latent_mean = WAN21_LATENT_MEAN[..4].to_vec();
        cfg.latent_std = WAN21_LATENT_STD[..4].to_vec();
        cfg
    }

    /// The 2.2 topology at `dim=8`, `z=4`, keeping patchify and the
    /// average-pool / duplicate shortcuts.
    pub fn tiny_v2_2() -> Self {
        let mut cfg = Self::v2_2();
        cfg.encoder_dim = 8;
        cfg.decoder_dim = 8;
        cfg.z_dim = 4;
        cfg.latent_mean = WAN22_LATENT_MEAN[..4].to_vec();
        cfg.latent_std = WAN22_LATENT_STD[..4].to_vec();
        cfg
    }

    /// Pixel channels the encoder's `conv1` sees: 3 for 2.1, 12 for 2.2.
    pub fn encoder_in_channels(&self) -> usize {
        3 * self.patch_size * self.patch_size
    }

    /// `2^(temporal downsample stages)` — 4 for every shipped config.
    pub fn temporal_compression(&self) -> usize {
        1 << self.temporal_downsample.iter().filter(|d| **d).count()
    }

    /// Patchify factor times one halving per resampling stage: 8 for 2.1,
    /// 16 for 2.2.
    pub fn spatial_compression(&self) -> usize {
        self.patch_size * (1 << (self.dim_mult.len() - 1))
    }

    /// Encoder widths, `[dim * u for u in [1] + dim_mult]` (`vae.py:284`).
    pub fn encoder_dims(&self) -> Vec<usize> {
        std::iter::once(1)
            .chain(self.dim_mult.iter().copied())
            .map(|u| self.encoder_dim * u)
            .collect()
    }

    /// Decoder widths, `[dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]`
    /// (`vae.py:388`).
    pub fn decoder_dims(&self) -> Vec<usize> {
        std::iter::once(self.dim_mult[self.dim_mult.len() - 1])
            .chain(self.dim_mult.iter().rev().copied())
            .map(|u| self.decoder_dim * u)
            .collect()
    }

    /// Decoder temporal upsample flags — the downsample flags reversed
    /// (`vae.py:500`).
    fn temporal_upsample(&self) -> Vec<bool> {
        self.temporal_downsample.iter().rev().copied().collect()
    }

    fn validate(&self) -> Result<()> {
        if self.dim_mult.len() < 2 {
            bail!("Wan VAE: dim_mult needs at least two stages");
        }
        if self.temporal_downsample.len() != self.dim_mult.len() - 1 {
            bail!(
                "Wan VAE: temporal_downsample must carry one flag per resampling stage ({} \
                 expected, got {})",
                self.dim_mult.len() - 1,
                self.temporal_downsample.len()
            );
        }
        if self.latent_mean.len() != self.z_dim || self.latent_std.len() != self.z_dim {
            bail!(
                "Wan VAE: latent statistics must cover every latent channel (z_dim={}, mean={}, \
                 std={})",
                self.z_dim,
                self.latent_mean.len(),
                self.latent_std.len()
            );
        }
        if self.patch_size == 0 {
            bail!("Wan VAE: patch_size must be positive");
        }
        if self.variant == WanVaeVariant::V2_2 {
            // `Down_ResidualBlock`'s final stage keeps its width because its
            // AvgDown3D has factor 1 (`vae2_2.py:332`); a dim_mult that grows
            // at the last stage would trip upstream's own assert.
            let mult = &self.dim_mult;
            if mult[mult.len() - 1] != mult[mult.len() - 2] {
                bail!(
                    "Wan 2.2 VAE: the last two dim_mult entries must match so the final \
                     AvgDown3D stays width-preserving, got {:?}",
                    mult
                );
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// feature cache
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum FeatCacheEntry {
    /// First-decode-chunk sentinel for `upsample3d`: there is no temporal
    /// history yet, so the next chunk runs its time conv zero-filled rather
    /// than cache-filled (`vae.py:104-132`).
    Rep,
    Frames(Tensor),
}

/// The flat `feat_cache` / `feat_idx` pair upstream threads through the whole
/// module tree (`vae.py:475-480`, `582-589`). The slot for a convolution is its
/// position in the forward walk, so `rewind` must be called once per chunk and
/// the walk must be identical on every chunk.
#[derive(Debug, Default)]
struct FeatCache {
    slots: Vec<Option<FeatCacheEntry>>,
    idx: usize,
}

impl FeatCache {
    fn new() -> Self {
        Self::default()
    }

    /// `feat_idx = [0]` before each chunk.
    fn rewind(&mut self) {
        self.idx = 0;
    }

    fn next_idx(&mut self) -> usize {
        let idx = self.idx;
        self.idx += 1;
        if self.slots.len() <= idx {
            self.slots.resize(idx + 1, None);
        }
        idx
    }

    fn get(&self, idx: usize) -> Option<&FeatCacheEntry> {
        self.slots[idx].as_ref()
    }

    fn set(&mut self, idx: usize, entry: FeatCacheEntry) {
        self.slots[idx] = Some(entry);
    }

    /// The cache/update dance every plain `CausalConv3d` call site repeats
    /// (`vae.py:202-220`, `318-331`, `350-365`): hand the previous chunk's tail
    /// to the conv, then store this chunk's tail — topped up from the previous
    /// slot when this chunk is a single frame.
    fn causal_conv(&mut self, conv: &WanCausalConv3d, x: &Tensor) -> Result<Tensor> {
        let idx = self.next_idx();
        let prev = match self.get(idx) {
            None => None,
            Some(FeatCacheEntry::Frames(prev)) => Some(prev.clone()),
            Some(FeatCacheEntry::Rep) => bail!(
                "Wan VAE: cache slot {idx} holds the upsample3d 'Rep' sentinel but a plain causal \
                 conv is reading it — the feat-cache visit order desynchronized"
            ),
        };
        let mut cache_x = tail_frames(x, CACHE_T)?;
        if cache_x.dim(2)? < CACHE_T {
            if let Some(prev) = prev.as_ref() {
                cache_x = cat2(&last_frame(prev)?, &cache_x, 2)?;
            }
        }
        let y = conv.forward(x, prev.as_ref())?;
        self.set(idx, FeatCacheEntry::Frames(cache_x));
        Ok(y)
    }
}

// ---------------------------------------------------------------------------
// primitives
// ---------------------------------------------------------------------------

/// `nn.Conv3d` with left-only temporal padding (`vae.py:17-36`).
///
/// The temporal pad is `2 * padding[0]` frames in front and none behind, so the
/// output never sees the future. When the previous chunk's tail is supplied it
/// is prepended and the zero pad shrinks by that many frames, which makes a
/// chunked walk identical to one pass over the zero-padded clip.
#[derive(Debug, Clone)]
struct WanCausalConv3d {
    kt: usize,
    stride_t: usize,
    front_pad: usize,
    pad_h: usize,
    pad_w: usize,
    /// One `Conv2d` per temporal tap — candle has no conv3d.
    slices: Vec<Conv2d>,
    bias: Tensor,
}

impl WanCausalConv3d {
    fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
    ) -> Result<Self> {
        let (kt, kh, kw) = kernel;
        let (stride_t, stride_h, stride_w) = stride;
        if stride_h != stride_w {
            bail!("Wan VAE: asymmetric spatial stride ({stride_h}, {stride_w}) is unsupported");
        }
        let weight = vb.get_with_hints(
            (out_channels, in_channels, kt, kh, kw),
            "weight",
            TEST_WEIGHT_INIT,
        )?;
        let bias = vb.get_with_hints(out_channels, "bias", Init::Const(0.0))?;

        let cfg = Conv2dConfig {
            stride: stride_h,
            ..Default::default()
        };
        let mut slices = Vec::with_capacity(kt);
        for ti in 0..kt {
            let weight2d = weight.i((.., .., ti, .., ..))?.contiguous()?;
            slices.push(Conv2d::new(weight2d, None, cfg));
        }

        Ok(Self {
            kt,
            stride_t,
            front_pad: 2 * padding.0,
            pad_h: padding.1,
            pad_w: padding.2,
            slices,
            bias,
        })
    }

    fn forward(&self, x: &Tensor, cache: Option<&Tensor>) -> Result<Tensor> {
        let mut x = x.clone();
        let mut front = self.front_pad;
        // Upstream ignores the cache when the conv has no temporal padding
        // (`vae.py:30`); `downsample3d` relies on that and splices its own.
        if let Some(cache) = cache {
            if front > 0 {
                let carried = cache.dim(2)?;
                if carried > front {
                    bail!(
                        "Wan VAE: causal cache carries {carried} frames but the conv only pads \
                         {front}"
                    );
                }
                x = cat2(cache, &x, 2)?;
                front -= carried;
            }
        }
        if front > 0 {
            x = x.pad_with_zeros(2, front, 0)?;
        }
        if self.pad_h > 0 {
            x = x.pad_with_zeros(3, self.pad_h, self.pad_h)?;
        }
        if self.pad_w > 0 {
            x = x.pad_with_zeros(4, self.pad_w, self.pad_w)?;
        }

        let t_pad = x.dim(2)?;
        if t_pad < self.kt {
            bail!(
                "Wan VAE: {t_pad} frames after causal padding is short of the {} needed",
                self.kt
            );
        }
        let t_out = (t_pad - self.kt) / self.stride_t + 1;

        let mut ys = Vec::with_capacity(t_out);
        for oi in 0..t_out {
            let base = oi * self.stride_t;
            let mut acc: Option<Tensor> = None;
            for ki in 0..self.kt {
                let frame = x.i((.., .., base + ki, .., ..))?.contiguous()?;
                let y = frame.apply(&self.slices[ki])?;
                acc = Some(match acc {
                    None => y,
                    Some(prev) => prev.add(&y)?,
                });
            }
            ys.push(acc.expect("temporal kernel is non-empty").unsqueeze(2)?);
        }

        let y = cat_all(&ys, 2)?;
        let bias = self.bias.reshape((1, self.bias.dim(0)?, 1, 1, 1))?;
        y.broadcast_add(&bias)
    }
}

/// `RMS_norm` (`vae.py:39-54`): `F.normalize(x, dim=1) * sqrt(dim) * gamma`.
///
/// `F.normalize` is an L2 normalize with the denominator clamped at 1e-12, so
/// this is RMS-norm over the channel axis with no epsilon inside the mean.
#[derive(Debug, Clone)]
struct WanRmsNorm {
    /// Flattened to `[dim]`; reshaped per call to match the input rank.
    gamma: Tensor,
    scale: f64,
}

impl WanRmsNorm {
    /// `images` selects the on-disk gamma shape: `[dim, 1, 1]` for the 4-D
    /// per-frame attention norm, `[dim, 1, 1, 1]` for the 5-D video norms
    /// (`vae.py:41-48`).
    fn load(vb: VarBuilder, dim: usize, images: bool) -> Result<Self> {
        let gamma = if images {
            vb.get_with_hints((dim, 1, 1), "gamma", Init::Const(1.0))?
        } else {
            vb.get_with_hints((dim, 1, 1, 1), "gamma", Init::Const(1.0))?
        };
        Ok(Self {
            gamma: gamma.reshape(dim)?,
            scale: (dim as f64).sqrt(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x32 = x.to_dtype(DType::F32)?;
        // 1e-24 under the sqrt is F.normalize's 1e-12 floor on the norm.
        // Rank-5 `[b, c, t, h, w]` reduced over the channel axis: candle's
        // Metal reduction is wrong there, and the undersized sum-of-squares
        // reaches `sqrt` negative, NaN-ing every pixel. See `metal_reduce`.
        let norm = crate::metal_reduce::sum_keepdim_stable(&x32.sqr()?, 1)?
            .affine(1.0, 1e-24)?
            .sqrt()?;
        let mut shape = vec![1usize; x.rank()];
        shape[1] = self.gamma.dim(0)?;
        let gamma = self.gamma.to_dtype(DType::F32)?.reshape(shape)?;
        x32.broadcast_div(&norm)?
            .affine(self.scale, 0.0)?
            .broadcast_mul(&gamma)?
            .to_dtype(dtype)
    }
}

/// Single-head 2-D self-attention run independently per frame
/// (`vae.py:223-262`).
#[derive(Debug, Clone)]
struct WanAttentionBlock {
    norm: WanRmsNorm,
    to_qkv: Conv2d,
    proj: Conv2d,
}

impl WanAttentionBlock {
    fn load(vb: VarBuilder, dim: usize) -> Result<Self> {
        let cfg = Conv2dConfig::default();
        Ok(Self {
            norm: WanRmsNorm::load(vb.pp("norm"), dim, true)?,
            to_qkv: candle_nn::conv2d(dim, dim * 3, 1, cfg, vb.pp("to_qkv"))?,
            proj: candle_nn::conv2d(dim, dim, 1, cfg, vb.pp("proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let projected = map_frames(x, |frame| {
            let (b, c, h, w) = frame.dims4()?;
            let normed = self.norm.forward(frame)?;
            // [b, 3c, h, w] -> [b, hw, 3c], then split q|k|v along the channel
            // thirds, matching upstream's reshape/permute/chunk.
            let qkv = normed
                .apply(&self.to_qkv)?
                .reshape((b, 3 * c, h * w))?
                .transpose(1, 2)?
                .contiguous()?;
            let head = |offset: usize| -> Result<Tensor> {
                qkv.narrow(2, offset, c)?.unsqueeze(1)?.contiguous()
            };
            // `math_attention` deliberately, not `attention`: routing the VAE
            // through the FlashAttention backend would make reconstruction
            // depend on how the binary was built.
            let attn = crate::attention::math_attention(
                &head(0)?,
                &head(c)?,
                &head(2 * c)?,
                1.0 / (c as f32).sqrt(),
            )?;
            attn.squeeze(1)?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b, c, h, w))?
                .apply(&self.proj)
        })?;
        projected.add(x)
    }
}

/// `ResidualBlock` (`vae.py:186-220`). The two cached convs are `residual.2`
/// and `residual.6`; the 1x1x1 shortcut is called outside the cache walk and
/// consumes no slot (`vae.py:203`).
#[derive(Debug, Clone)]
struct WanResidualBlock {
    norm1: WanRmsNorm,
    conv1: WanCausalConv3d,
    norm2: WanRmsNorm,
    conv2: WanCausalConv3d,
    shortcut: Option<WanCausalConv3d>,
}

impl WanResidualBlock {
    fn load(vb: VarBuilder, in_dim: usize, out_dim: usize) -> Result<Self> {
        let residual = vb.pp("residual");
        let shortcut = if in_dim != out_dim {
            Some(WanCausalConv3d::load(
                vb.pp("shortcut"),
                in_dim,
                out_dim,
                (1, 1, 1),
                (1, 1, 1),
                (0, 0, 0),
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1: WanRmsNorm::load(residual.pp("0"), in_dim, false)?,
            conv1: WanCausalConv3d::load(
                residual.pp("2"),
                in_dim,
                out_dim,
                (3, 3, 3),
                (1, 1, 1),
                (1, 1, 1),
            )?,
            norm2: WanRmsNorm::load(residual.pp("3"), out_dim, false)?,
            conv2: WanCausalConv3d::load(
                residual.pp("6"),
                out_dim,
                out_dim,
                (3, 3, 3),
                (1, 1, 1),
                (1, 1, 1),
            )?,
            shortcut,
        })
    }

    fn forward(&self, x: &Tensor, cache: &mut FeatCache) -> Result<Tensor> {
        let identity = match &self.shortcut {
            Some(conv) => conv.forward(x, None)?,
            None => x.clone(),
        };
        let h = ops::silu(&self.norm1.forward(x)?)?;
        let h = cache.causal_conv(&self.conv1, &h)?;
        let h = ops::silu(&self.norm2.forward(&h)?)?;
        let h = cache.causal_conv(&self.conv2, &h)?;
        h.add(&identity)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResampleMode {
    Upsample2d,
    Upsample3d,
    Downsample2d,
    Downsample3d,
}

/// `Resample` (`vae.py:66-160`, `vae2_2.py:71-169`). `resample.1` is the
/// Conv2d; index 0 of the Sequential is a parameter-free `Upsample` or
/// `ZeroPad2d`.
#[derive(Debug, Clone)]
struct WanResample {
    mode: ResampleMode,
    conv: Conv2d,
    time_conv: Option<WanCausalConv3d>,
}

impl WanResample {
    fn load(
        vb: VarBuilder,
        dim: usize,
        mode: ResampleMode,
        variant: WanVaeVariant,
    ) -> Result<Self> {
        let (conv, time_conv) = match mode {
            ResampleMode::Upsample2d | ResampleMode::Upsample3d => {
                // 2.1 halves the width on the way up (`vae.py:79,83`); 2.2
                // keeps it (`vae2_2.py:89,94`). Downsample convs are
                // width-preserving in both.
                let out = match variant {
                    WanVaeVariant::V2_1 => dim / 2,
                    WanVaeVariant::V2_2 => dim,
                };
                let cfg = Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                };
                let conv = candle_nn::conv2d(dim, out, 3, cfg, vb.pp("resample").pp("1"))?;
                let time_conv = if mode == ResampleMode::Upsample3d {
                    Some(WanCausalConv3d::load(
                        vb.pp("time_conv"),
                        dim,
                        dim * 2,
                        (3, 1, 1),
                        (1, 1, 1),
                        (1, 0, 0),
                    )?)
                } else {
                    None
                };
                (conv, time_conv)
            }
            ResampleMode::Downsample2d | ResampleMode::Downsample3d => {
                let cfg = Conv2dConfig {
                    stride: 2,
                    ..Default::default()
                };
                let conv = candle_nn::conv2d(dim, dim, 3, cfg, vb.pp("resample").pp("1"))?;
                let time_conv = if mode == ResampleMode::Downsample3d {
                    Some(WanCausalConv3d::load(
                        vb.pp("time_conv"),
                        dim,
                        dim,
                        (3, 1, 1),
                        (2, 1, 1),
                        (0, 0, 0),
                    )?)
                } else {
                    None
                };
                (conv, time_conv)
            }
        };
        Ok(Self {
            mode,
            conv,
            time_conv,
        })
    }

    fn time_conv(&self) -> Result<&WanCausalConv3d> {
        self.time_conv
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("Wan VAE: 3-D resample without a time conv"))
    }

    fn forward(&self, x: &Tensor, cache: &mut FeatCache) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        let mut x = x.clone();

        if self.mode == ResampleMode::Upsample3d {
            let idx = cache.next_idx();
            match cache.get(idx).cloned() {
                // First decode chunk: no temporal history, so the time conv is
                // skipped entirely and T is *not* doubled (`vae.py:106-108`).
                // This is why one latent frame decodes to one pixel frame and
                // every later frame decodes to four.
                None => cache.set(idx, FeatCacheEntry::Rep),
                Some(entry) => {
                    let mut cache_x = tail_frames(&x, CACHE_T)?;
                    if cache_x.dim(2)? < CACHE_T {
                        cache_x = match &entry {
                            FeatCacheEntry::Frames(prev) => cat2(&last_frame(prev)?, &cache_x, 2)?,
                            FeatCacheEntry::Rep => cat2(&cache_x.zeros_like()?, &cache_x, 2)?,
                        };
                    }
                    x = match &entry {
                        FeatCacheEntry::Rep => self.time_conv()?.forward(&x, None)?,
                        FeatCacheEntry::Frames(prev) => {
                            self.time_conv()?.forward(&x, Some(prev))?
                        }
                    };
                    cache.set(idx, FeatCacheEntry::Frames(cache_x));
                    // [b, 2c, t, h, w] -> interleaved [b, c, 2t, h, w]
                    // (`vae.py:134-137`).
                    let split = x.reshape((b, 2, c, t, h, w))?;
                    let even = split.i((.., 0))?.contiguous()?;
                    let odd = split.i((.., 1))?.contiguous()?;
                    x = Tensor::stack(&[&even, &odd], 3)?.reshape((b, c, t * 2, h, w))?;
                }
            }
        }

        x = map_frames(&x, |frame| match self.mode {
            ResampleMode::Upsample2d | ResampleMode::Upsample3d => {
                let (_, _, fh, fw) = frame.dims4()?;
                // `nearest-exact` and `nearest` agree exactly at an integer
                // scale factor of 2, so candle's nearest resize is the same op.
                frame.upsample_nearest2d(fh * 2, fw * 2)?.apply(&self.conv)
            }
            // `nn.ZeroPad2d((0, 1, 0, 1))` then a stride-2 unpadded conv.
            ResampleMode::Downsample2d | ResampleMode::Downsample3d => frame
                .pad_with_zeros(2, 0, 1)?
                .pad_with_zeros(3, 0, 1)?
                .apply(&self.conv),
        })?;

        if self.mode == ResampleMode::Downsample3d {
            let idx = cache.next_idx();
            match cache.get(idx).cloned() {
                // First encode chunk passes through untouched; the layer caches
                // the whole (single-frame) output (`vae.py:146-148`).
                None => cache.set(idx, FeatCacheEntry::Frames(x.clone())),
                Some(FeatCacheEntry::Frames(prev)) => {
                    let cache_x = last_frame(&x)?;
                    // The time conv is unpadded, so upstream splices the carry
                    // frame itself rather than routing it through the cache arg
                    // (`vae.py:156-157`).
                    let joined = cat2(&last_frame(&prev)?, &x, 2)?;
                    x = self.time_conv()?.forward(&joined, None)?;
                    cache.set(idx, FeatCacheEntry::Frames(cache_x));
                }
                Some(FeatCacheEntry::Rep) => bail!(
                    "Wan VAE: downsample3d cache slot {idx} holds the upsample 'Rep' sentinel — \
                     the feat-cache visit order desynchronized"
                ),
            }
        }

        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// 2.2-only parameter-free shortcuts
// ---------------------------------------------------------------------------

/// `AvgDown3D` (`vae2_2.py:316-367`): space/time-to-channel folding followed by
/// a group mean. Parameter-free, so it contributes no checkpoint keys.
#[derive(Debug, Clone, Copy)]
struct AvgDown3d {
    out_channels: usize,
    factor_t: usize,
    factor_s: usize,
    group_size: usize,
}

impl AvgDown3d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        factor_t: usize,
        factor_s: usize,
    ) -> Result<Self> {
        let factor = factor_t * factor_s * factor_s;
        if !(in_channels * factor).is_multiple_of(out_channels) {
            bail!(
                "Wan 2.2 VAE: AvgDown3D needs out_channels to divide in_channels*factor \
                 ({in_channels}*{factor} % {out_channels} != 0)"
            );
        }
        Ok(Self {
            out_channels,
            factor_t,
            factor_s,
            group_size: in_channels * factor / out_channels,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Front-pad the time axis up to a whole factor, so a leading 1-frame
        // chunk averages against a zero frame (`vae2_2.py:336-338`).
        let pad_t = (self.factor_t - x.dim(2)? % self.factor_t) % self.factor_t;
        let x = if pad_t > 0 {
            x.pad_with_zeros(2, pad_t, 0)?
        } else {
            x.clone()
        };
        let (b, c, t, h, w) = x.dims5()?;
        let (ft, fs) = (self.factor_t, self.factor_s);
        let (tn, hn, wn) = (t / ft, h / fs, w / fs);
        let grouped = x
            .reshape(&[b, c, tn, ft, hn, fs, wn, fs])?
            .permute(vec![0, 1, 3, 5, 7, 2, 4, 6])?
            .contiguous()?
            .reshape(&[b, c * ft * fs * fs, tn, hn, wn])?
            .reshape(&[b, self.out_channels, self.group_size, tn, hn, wn])?;
        // Rank-6 reduced over dim 2 hits the same Metal defect as the norm
        // above; the fold keeps the group mean correct there.
        crate::metal_reduce::mean_stable(&grouped, 2)
    }
}

/// `DupUp3D` (`vae2_2.py:370-412`): channel-to-space/time duplication.
#[derive(Debug, Clone, Copy)]
struct DupUp3d {
    out_channels: usize,
    factor_t: usize,
    factor_s: usize,
    repeats: usize,
}

impl DupUp3d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        factor_t: usize,
        factor_s: usize,
    ) -> Result<Self> {
        let factor = factor_t * factor_s * factor_s;
        if !(out_channels * factor).is_multiple_of(in_channels) {
            bail!(
                "Wan 2.2 VAE: DupUp3D needs in_channels to divide out_channels*factor \
                 ({out_channels}*{factor} % {in_channels} != 0)"
            );
        }
        Ok(Self {
            out_channels,
            factor_t,
            factor_s,
            repeats: out_channels * factor / in_channels,
        })
    }

    fn forward(&self, x: &Tensor, first_chunk: bool) -> Result<Tensor> {
        let (b, _, t, h, w) = x.dims5()?;
        let (ft, fs) = (self.factor_t, self.factor_s);
        let repeated = repeat_interleave_channels(x, self.repeats)?;
        let y = repeated
            .reshape(&[b, self.out_channels, ft, fs, fs, t, h, w])?
            .permute(vec![0, 1, 5, 2, 6, 3, 7, 4])?
            .contiguous()?
            .reshape(&[b, self.out_channels, t * ft, h * fs, w * fs])?;
        if first_chunk {
            // The first latent frame decodes to one pixel frame, so the
            // duplicate shortcut drops its leading `factor_t - 1`
            // (`vae2_2.py:410-411`) to match the main path's undoubled output.
            let total = t * ft;
            y.narrow(2, ft - 1, total - (ft - 1))?.contiguous()
        } else {
            Ok(y)
        }
    }
}

// ---------------------------------------------------------------------------
// stage assembly
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum StageBlock {
    Residual(WanResidualBlock),
    Resample(WanResample),
}

impl StageBlock {
    fn forward(&self, x: &Tensor, cache: &mut FeatCache) -> Result<Tensor> {
        match self {
            StageBlock::Residual(block) => block.forward(x, cache),
            StageBlock::Resample(resample) => resample.forward(x, cache),
        }
    }
}

/// `Down_ResidualBlock` (`vae2_2.py:415-452`).
#[derive(Debug, Clone)]
struct WanDownResidualBlock {
    avg_shortcut: AvgDown3d,
    blocks: Vec<StageBlock>,
}

impl WanDownResidualBlock {
    fn forward(&self, x: &Tensor, cache: &mut FeatCache) -> Result<Tensor> {
        let shortcut = self.avg_shortcut.forward(x)?;
        let mut h = x.clone();
        for block in &self.blocks {
            h = block.forward(&h, cache)?;
        }
        h.add(&shortcut)
    }
}

/// `Up_ResidualBlock` (`vae2_2.py:455-497`).
#[derive(Debug, Clone)]
struct WanUpResidualBlock {
    avg_shortcut: Option<DupUp3d>,
    blocks: Vec<StageBlock>,
}

impl WanUpResidualBlock {
    fn forward(&self, x: &Tensor, cache: &mut FeatCache, first_chunk: bool) -> Result<Tensor> {
        let mut h = x.clone();
        for block in &self.blocks {
            h = block.forward(&h, cache)?;
        }
        match &self.avg_shortcut {
            Some(shortcut) => h.add(&shortcut.forward(x, first_chunk)?),
            None => Ok(h),
        }
    }
}

/// One entry of `encoder.downsamples`: a flat Sequential element for 2.1, a
/// whole nested stage for 2.2.
#[derive(Debug, Clone)]
enum DownLevel {
    Flat(StageBlock),
    Nested(WanDownResidualBlock),
}

impl DownLevel {
    fn forward(&self, x: &Tensor, cache: &mut FeatCache) -> Result<Tensor> {
        match self {
            DownLevel::Flat(block) => block.forward(x, cache),
            DownLevel::Nested(stage) => stage.forward(x, cache),
        }
    }
}

/// One entry of `decoder.upsamples`.
#[derive(Debug, Clone)]
enum UpLevel {
    Flat(StageBlock),
    Nested(WanUpResidualBlock),
}

impl UpLevel {
    fn forward(&self, x: &Tensor, cache: &mut FeatCache, first_chunk: bool) -> Result<Tensor> {
        match self {
            UpLevel::Flat(block) => block.forward(x, cache),
            UpLevel::Nested(stage) => stage.forward(x, cache, first_chunk),
        }
    }
}

/// `middle = Sequential(ResidualBlock, AttentionBlock, ResidualBlock)`.
#[derive(Debug, Clone)]
struct WanMiddleBlock {
    first: WanResidualBlock,
    attention: WanAttentionBlock,
    second: WanResidualBlock,
}

impl WanMiddleBlock {
    fn load(vb: VarBuilder, dim: usize) -> Result<Self> {
        Ok(Self {
            first: WanResidualBlock::load(vb.pp("0"), dim, dim)?,
            attention: WanAttentionBlock::load(vb.pp("1"), dim)?,
            second: WanResidualBlock::load(vb.pp("2"), dim, dim)?,
        })
    }

    fn forward(&self, x: &Tensor, cache: &mut FeatCache) -> Result<Tensor> {
        let h = self.first.forward(x, cache)?;
        // The attention block holds no causal conv, so it consumes no cache
        // slot (`vae.py:344-347`).
        let h = self.attention.forward(&h)?;
        self.second.forward(&h, cache)
    }
}

// ---------------------------------------------------------------------------
// encoder / decoder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct WanEncoder3d {
    conv1: WanCausalConv3d,
    downsamples: Vec<DownLevel>,
    middle: WanMiddleBlock,
    head_norm: WanRmsNorm,
    head_conv: WanCausalConv3d,
}

impl WanEncoder3d {
    fn load(vb: VarBuilder, cfg: &WanVaeConfig) -> Result<Self> {
        let dims = cfg.encoder_dims();
        let stages = cfg.dim_mult.len();
        let conv1 = WanCausalConv3d::load(
            vb.pp("conv1"),
            cfg.encoder_in_channels(),
            dims[0],
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
        )?;

        let vb_down = vb.pp("downsamples");
        let mut downsamples = Vec::new();
        match cfg.variant {
            WanVaeVariant::V2_1 => {
                // Flat `nn.Sequential` (`vae.py:291-306`), so the checkpoint
                // index counts every residual block *and* every resample.
                let mut flat = 0usize;
                for i in 0..stages {
                    let mut in_dim = dims[i];
                    let out_dim = dims[i + 1];
                    for _ in 0..cfg.num_res_blocks {
                        downsamples.push(DownLevel::Flat(StageBlock::Residual(
                            WanResidualBlock::load(vb_down.pp(flat.to_string()), in_dim, out_dim)?,
                        )));
                        flat += 1;
                        in_dim = out_dim;
                    }
                    if i != stages - 1 {
                        let mode = if cfg.temporal_downsample[i] {
                            ResampleMode::Downsample3d
                        } else {
                            ResampleMode::Downsample2d
                        };
                        downsamples.push(DownLevel::Flat(StageBlock::Resample(WanResample::load(
                            vb_down.pp(flat.to_string()),
                            out_dim,
                            mode,
                            cfg.variant,
                        )?)));
                        flat += 1;
                    }
                }
            }
            WanVaeVariant::V2_2 => {
                // One `Down_ResidualBlock` per stage, its own Sequential nested
                // under `downsamples.{stage}.downsamples.{j}`
                // (`vae2_2.py:529-543`).
                for i in 0..stages {
                    let stage_vb = vb_down.pp(i.to_string()).pp("downsamples");
                    let mut in_dim = dims[i];
                    let out_dim = dims[i + 1];
                    let down_flag = i != stages - 1;
                    let temporal = cfg.temporal_downsample.get(i).copied().unwrap_or(false);
                    let avg_shortcut = AvgDown3d::new(
                        dims[i],
                        out_dim,
                        if temporal { 2 } else { 1 },
                        if down_flag { 2 } else { 1 },
                    )?;
                    let mut blocks = Vec::new();
                    for j in 0..cfg.num_res_blocks {
                        blocks.push(StageBlock::Residual(WanResidualBlock::load(
                            stage_vb.pp(j.to_string()),
                            in_dim,
                            out_dim,
                        )?));
                        in_dim = out_dim;
                    }
                    if down_flag {
                        let mode = if temporal {
                            ResampleMode::Downsample3d
                        } else {
                            ResampleMode::Downsample2d
                        };
                        blocks.push(StageBlock::Resample(WanResample::load(
                            stage_vb.pp(cfg.num_res_blocks.to_string()),
                            out_dim,
                            mode,
                            cfg.variant,
                        )?));
                    }
                    downsamples.push(DownLevel::Nested(WanDownResidualBlock {
                        avg_shortcut,
                        blocks,
                    }));
                }
            }
        }

        let bottleneck = dims[stages];
        Ok(Self {
            conv1,
            downsamples,
            middle: WanMiddleBlock::load(vb.pp("middle"), bottleneck)?,
            head_norm: WanRmsNorm::load(vb.pp("head").pp("0"), bottleneck, false)?,
            head_conv: WanCausalConv3d::load(
                vb.pp("head").pp("2"),
                bottleneck,
                cfg.z_dim * 2,
                (3, 3, 3),
                (1, 1, 1),
                (1, 1, 1),
            )?,
        })
    }

    fn forward(&self, x: &Tensor, cache: &mut FeatCache) -> Result<Tensor> {
        let mut h = cache.causal_conv(&self.conv1, x)?;
        for level in &self.downsamples {
            h = level.forward(&h, cache)?;
        }
        h = self.middle.forward(&h, cache)?;
        h = ops::silu(&self.head_norm.forward(&h)?)?;
        cache.causal_conv(&self.head_conv, &h)
    }
}

#[derive(Debug, Clone)]
struct WanDecoder3d {
    conv1: WanCausalConv3d,
    middle: WanMiddleBlock,
    upsamples: Vec<UpLevel>,
    head_norm: WanRmsNorm,
    head_conv: WanCausalConv3d,
}

impl WanDecoder3d {
    fn load(vb: VarBuilder, cfg: &WanVaeConfig) -> Result<Self> {
        let dims = cfg.decoder_dims();
        let stages = cfg.dim_mult.len();
        let temporal_upsample = cfg.temporal_upsample();
        let conv1 = WanCausalConv3d::load(
            vb.pp("conv1"),
            cfg.z_dim,
            dims[0],
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
        )?;
        let middle = WanMiddleBlock::load(vb.pp("middle"), dims[0])?;

        let vb_up = vb.pp("upsamples");
        let mut upsamples = Vec::new();
        match cfg.variant {
            WanVaeVariant::V2_1 => {
                let mut flat = 0usize;
                for i in 0..stages {
                    // Every stage after the first is fed by an upsample whose
                    // conv halved the width (`vae.py:403-404`).
                    let mut in_dim = if i >= 1 { dims[i] / 2 } else { dims[i] };
                    let out_dim = dims[i + 1];
                    for _ in 0..(cfg.num_res_blocks + 1) {
                        upsamples.push(UpLevel::Flat(StageBlock::Residual(
                            WanResidualBlock::load(vb_up.pp(flat.to_string()), in_dim, out_dim)?,
                        )));
                        flat += 1;
                        in_dim = out_dim;
                    }
                    if i != stages - 1 {
                        let mode = if temporal_upsample[i] {
                            ResampleMode::Upsample3d
                        } else {
                            ResampleMode::Upsample2d
                        };
                        upsamples.push(UpLevel::Flat(StageBlock::Resample(WanResample::load(
                            vb_up.pp(flat.to_string()),
                            out_dim,
                            mode,
                            cfg.variant,
                        )?)));
                        flat += 1;
                    }
                }
            }
            WanVaeVariant::V2_2 => {
                for i in 0..stages {
                    let stage_vb = vb_up.pp(i.to_string()).pp("upsamples");
                    let mut in_dim = dims[i];
                    let out_dim = dims[i + 1];
                    let up_flag = i != stages - 1;
                    let temporal = temporal_upsample.get(i).copied().unwrap_or(false);
                    let avg_shortcut = if up_flag {
                        Some(DupUp3d::new(
                            dims[i],
                            out_dim,
                            if temporal { 2 } else { 1 },
                            2,
                        )?)
                    } else {
                        None
                    };
                    let mut blocks = Vec::new();
                    for j in 0..(cfg.num_res_blocks + 1) {
                        blocks.push(StageBlock::Residual(WanResidualBlock::load(
                            stage_vb.pp(j.to_string()),
                            in_dim,
                            out_dim,
                        )?));
                        in_dim = out_dim;
                    }
                    if up_flag {
                        let mode = if temporal {
                            ResampleMode::Upsample3d
                        } else {
                            ResampleMode::Upsample2d
                        };
                        blocks.push(StageBlock::Resample(WanResample::load(
                            stage_vb.pp((cfg.num_res_blocks + 1).to_string()),
                            out_dim,
                            mode,
                            cfg.variant,
                        )?));
                    }
                    upsamples.push(UpLevel::Nested(WanUpResidualBlock {
                        avg_shortcut,
                        blocks,
                    }));
                }
            }
        }

        let out_dim = dims[stages];
        Ok(Self {
            conv1,
            middle,
            upsamples,
            head_norm: WanRmsNorm::load(vb.pp("head").pp("0"), out_dim, false)?,
            head_conv: WanCausalConv3d::load(
                vb.pp("head").pp("2"),
                out_dim,
                cfg.encoder_in_channels(),
                (3, 3, 3),
                (1, 1, 1),
                (1, 1, 1),
            )?,
        })
    }

    fn forward(&self, x: &Tensor, cache: &mut FeatCache, first_chunk: bool) -> Result<Tensor> {
        let mut h = cache.causal_conv(&self.conv1, x)?;
        h = self.middle.forward(&h, cache)?;
        for level in &self.upsamples {
            h = level.forward(&h, cache, first_chunk)?;
        }
        h = ops::silu(&self.head_norm.forward(&h)?)?;
        cache.causal_conv(&self.head_conv, &h)
    }
}

// ---------------------------------------------------------------------------
// public surface
// ---------------------------------------------------------------------------

/// The full Wan video VAE. Both directions stream internally with a fresh
/// feature cache per call, so `&self` is enough and calls are independent.
#[derive(Debug, Clone)]
pub(crate) struct WanVideoVae {
    config: WanVaeConfig,
    encoder: WanEncoder3d,
    decoder: WanDecoder3d,
    /// 1x1x1 over the encoder head's `2*z` output (`vae.py:505`).
    conv1: WanCausalConv3d,
    /// 1x1x1 over the incoming latent (`vae.py:506`).
    conv2: WanCausalConv3d,
    latent_mean: Tensor,
    latent_std: Tensor,
    dtype: DType,
}

impl WanVideoVae {
    /// Load from a Comfy-Org repack (`wan_2.1_vae.safetensors`,
    /// `wan2.2_vae.safetensors`), which keeps the original checkpoint's bare
    /// `encoder.*` / `decoder.*` / `conv1` / `conv2` layout. A full-checkpoint
    /// `vae.` prefix is also accepted.
    pub fn from_safetensors(
        path: &Path,
        config: WanVaeConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device)? };
        let vb = if vb.contains_tensor("encoder.conv1.weight") {
            vb
        } else {
            vb.pp("vae")
        };
        Self::from_var_builder(vb, config, device, dtype)
    }

    /// Build from a `VarBuilder` rooted at the checkpoint layout. Split out so
    /// tests can materialize tiny models from a `VarMap`.
    pub fn from_var_builder(
        vb: VarBuilder,
        config: WanVaeConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        config.validate()?;
        let encoder = WanEncoder3d::load(vb.pp("encoder"), &config)?;
        let decoder = WanDecoder3d::load(vb.pp("decoder"), &config)?;
        let conv1 = WanCausalConv3d::load(
            vb.pp("conv1"),
            config.z_dim * 2,
            config.z_dim * 2,
            (1, 1, 1),
            (1, 1, 1),
            (0, 0, 0),
        )?;
        let conv2 = WanCausalConv3d::load(
            vb.pp("conv2"),
            config.z_dim,
            config.z_dim,
            (1, 1, 1),
            (1, 1, 1),
            (0, 0, 0),
        )?;
        let stat = |values: &[f32]| -> Result<Tensor> {
            Tensor::from_slice(values, (1, values.len(), 1, 1, 1), device)?.to_dtype(dtype)
        };
        let latent_mean = stat(&config.latent_mean)?;
        let latent_std = stat(&config.latent_std)?;
        Ok(Self {
            config,
            encoder,
            decoder,
            conv1,
            conv2,
            latent_mean,
            latent_std,
            dtype,
        })
    }

    pub fn config(&self) -> &WanVaeConfig {
        &self.config
    }

    /// Encode `[b, 3, f, h, w]` pixels in `[-1, 1]` to normalized latents
    /// `[b, z, (f - 1) / 4 + 1, h / 8, w / 8]` (`/16` for 2.2).
    ///
    /// Returns the posterior *mean*; upstream never samples here
    /// (`vae.py:535-542`).
    pub fn encode(&self, video: &Tensor) -> Result<Tensor> {
        let (_, channels, frames, height, width) = video.dims5()?;
        if channels != 3 {
            bail!("Wan VAE: encode expects 3 pixel channels, got {channels}");
        }
        if frames == 0 || (frames - 1) % ENCODE_CHUNK_FRAMES != 0 {
            // Upstream would silently drop the ragged tail
            // (`vae.py:520-534`); refuse instead.
            bail!(
                "Wan VAE: encode needs 4k+1 frames so the causal chunking covers the clip, got \
                 {frames}"
            );
        }
        let spatial = self.config.spatial_compression();
        if height % spatial != 0 || width % spatial != 0 {
            bail!("Wan VAE: {height}x{width} is not a multiple of the {spatial}x spatial stride");
        }

        let x = patchify(&video.to_dtype(self.dtype)?, self.config.patch_size)?;
        let out = self.encode_stream(&x, &encode_chunk_plan(frames))?;
        let mu = self
            .conv1
            .forward(&out, None)?
            .narrow(1, 0, self.config.z_dim)?;
        mu.broadcast_sub(&self.latent_mean)?
            .broadcast_div(&self.latent_std)
    }

    /// Decode normalized latents back to `[b, 3, f, h, w]` pixels in `[-1, 1]`.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let channels = latents.dim(1)?;
        if channels != self.config.z_dim {
            bail!(
                "Wan VAE: decode expects {} latent channels, got {channels}",
                self.config.z_dim
            );
        }
        let z = latents
            .to_dtype(self.dtype)?
            .broadcast_mul(&self.latent_std)?
            .broadcast_add(&self.latent_mean)?;
        let x = self.conv2.forward(&z, None)?;
        let plan = vec![1usize; x.dim(2)?];
        let out = self.decode_stream(&x, &plan)?;
        unpatchify(&out, self.config.patch_size)?.clamp(-1.0, 1.0)
    }

    /// Spatially tiled decode, for a canvas whose single-pass decode does not
    /// fit.
    ///
    /// Wan's decoder already streams the TEMPORAL axis — `decode` walks one
    /// latent frame per chunk — so what is left to bound is the spatial
    /// working set, which is the whole `[b, c, t, H, W]` activation at full
    /// output resolution. This tiles `h` and `w`, decodes each tile through
    /// the ordinary path, and blends the overlaps.
    ///
    /// Ported from ComfyUI's `tiled_scale_multidim`
    /// (`comfy/utils.py:1145-1258`), which is also what its own
    /// `decode_tiled_3d` uses: tile starts step by `tile - overlap`, each tile
    /// is clamped so the last one ends flush with the edge, and the
    /// accumulator is a weighted mean under a mask that ramps linearly over
    /// `overlap` pixels at every tiled edge. Upstream Wan has no tiling of its
    /// own (`Wan2.2/wan/modules/vae2_1.py` decodes whole frames), so ComfyUI is
    /// the reference here.
    ///
    /// The blend is a true weighted mean — accumulate `tile * mask`, and
    /// `mask` separately, then divide — so it does not matter that the ramps
    /// do not sum to one at the canvas edges, where only one tile contributes.
    /// `tile` and `overlap` are LATENT units.
    pub fn decode_tiled(&self, latents: &Tensor, tile: usize, overlap: usize) -> Result<Tensor> {
        let (_, _, _, lat_h, lat_w) = latents.dims5()?;
        let tile = tile.max(1);
        // An overlap at or above the tile size would make the stride zero and
        // never terminate.
        let overlap = overlap.min(tile.saturating_sub(1));
        if lat_h <= tile && lat_w <= tile {
            return self.decode(latents);
        }

        let scale = self.config.spatial_compression();
        let starts = |len: usize| -> Vec<usize> {
            if len <= tile {
                return vec![0];
            }
            let stride = tile - overlap;
            let mut out = Vec::new();
            let mut pos = 0usize;
            while pos < len.saturating_sub(overlap) {
                out.push(pos.min(len.saturating_sub(overlap)));
                pos += stride;
            }
            out
        };

        let device = latents.device().clone();
        // The accumulators live on the CPU, not on the device that just ran
        // out of memory. They are full-video F32 buffers — about 1.6 GiB
        // together at 1280x704 x 121 before a single tile temporary — and
        // holding them on the constrained device is how a recovery path
        // reproduces the failure it is recovering from. `crate::vae_tiling`
        // accumulates on the host for the same reason. Only one tile's decode
        // is on the device at a time.
        let accumulator_device = Device::Cpu;
        let mut acc: Option<Tensor> = None;
        let mut weight: Option<Tensor> = None;

        for h0 in starts(lat_h) {
            for w0 in starts(lat_w) {
                let th = tile.min(lat_h - h0);
                let tw = tile.min(lat_w - w0);
                let piece = latents.narrow(3, h0, th)?.narrow(4, w0, tw)?.contiguous()?;
                let decoded = self.decode(&piece)?.to_dtype(DType::F32)?;
                let (_, _, frames, out_h, out_w) = decoded.dims5()?;

                let mask = tile_blend_mask(out_h, out_w, overlap * scale, &device)?;
                let contribution = decoded.broadcast_mul(&mask)?;

                // Allocated on the first tile, where the decoded frame count
                // is finally known: the temporal expansion is the decoder's,
                // not something the latent shape states.
                if acc.is_none() {
                    acc = Some(Tensor::zeros(
                        (1, 3, frames, lat_h * scale, lat_w * scale),
                        DType::F32,
                        &accumulator_device,
                    )?);
                    weight = Some(Tensor::zeros(
                        (1, 1, frames, lat_h * scale, lat_w * scale),
                        DType::F32,
                        &accumulator_device,
                    )?);
                }

                let (oh, ow) = (h0 * scale, w0 * scale);
                let acc_ref = acc.as_mut().expect("allocated above");
                *acc_ref = acc_ref.slice_assign(
                    &[0..1, 0..3, 0..frames, oh..oh + out_h, ow..ow + out_w],
                    &acc_ref
                        .narrow(3, oh, out_h)?
                        .narrow(4, ow, out_w)?
                        .add(&contribution)?,
                )?;
                let weight_ref = weight.as_mut().expect("allocated above");
                *weight_ref = weight_ref.slice_assign(
                    &[0..1, 0..1, 0..frames, oh..oh + out_h, ow..ow + out_w],
                    &weight_ref
                        .narrow(3, oh, out_h)?
                        .narrow(4, ow, out_w)?
                        .broadcast_add(&mask)?,
                )?;
            }
        }

        let acc = acc.ok_or_else(|| {
            candle_core::Error::Msg("Wan VAE: tiled decode produced no tiles".into())
        })?;
        let weight = weight.expect("allocated beside acc");
        // Every output pixel is covered by at least one tile, so the divisor
        // is never zero; the clamp is belt and braces against a degenerate
        // mask rather than a real case.
        acc.broadcast_div(&weight.clamp(1e-6, f64::INFINITY)?)?
            .to_device(&device)?
            .to_dtype(self.dtype)?
            .clamp(-1.0, 1.0)
    }

    /// Walk the encoder over an explicit chunk plan. Causal padding makes the
    /// result independent of how the clip is split (after the mandatory leading
    /// single frame), which is what the equivalence tests assert.
    fn encode_stream(&self, patched: &Tensor, plan: &[usize]) -> Result<Tensor> {
        let mut cache = FeatCache::new();
        let mut outputs = Vec::with_capacity(plan.len());
        let mut start = 0usize;
        for len in plan {
            cache.rewind();
            let chunk = patched.narrow(2, start, *len)?.contiguous()?;
            outputs.push(self.encoder.forward(&chunk, &mut cache)?);
            start += len;
        }
        cat_all(&outputs, 2)
    }

    /// Walk the decoder over an explicit latent-frame plan. The first entry
    /// must be 1: the leading chunk is what primes every `upsample3d` slot with
    /// its `Rep` sentinel and, for 2.2, what sets `first_chunk`.
    fn decode_stream(&self, x: &Tensor, plan: &[usize]) -> Result<Tensor> {
        if plan.first() != Some(&1) {
            bail!("Wan VAE: the first decode chunk must be exactly one latent frame");
        }
        let mut cache = FeatCache::new();
        let mut outputs = Vec::with_capacity(plan.len());
        let mut start = 0usize;
        for (i, len) in plan.iter().enumerate() {
            cache.rewind();
            let chunk = x.narrow(2, start, *len)?.contiguous()?;
            outputs.push(self.decoder.forward(&chunk, &mut cache, i == 0)?);
            start += len;
        }
        cat_all(&outputs, 2)
    }
}

/// `1, 4, 4, 4, ...` over the pixel frame axis (`vae.py:519-534`).
fn encode_chunk_plan(frames: usize) -> Vec<usize> {
    let mut plan = vec![1usize];
    let mut remaining = frames - 1;
    while remaining > 0 {
        let len = remaining.min(ENCODE_CHUNK_FRAMES);
        plan.push(len);
        remaining -= len;
    }
    plan
}

/// `patchify` (`vae2_2.py:280-296`): `b c f (h q) (w r) -> b (c r q) f h w`.
/// ComfyUI's tile feather, as a separable `[1, 1, 1, h, w]` mask.
///
/// `comfy/utils.py:1229-1236` walks `t in 0..feather` and MULTIPLIES both the
/// `t`-th and the `(len-1-t)`-th row by `(t+1)/feather`. When `2*feather`
/// exceeds the axis the two ramps overlap and their factors multiply, which is
/// reproduced here rather than clamped — a tile narrower than twice the
/// feather is exactly the edge case where matching upstream matters.
fn tile_blend_mask(height: usize, width: usize, feather: usize, device: &Device) -> Result<Tensor> {
    let ramp = |len: usize| -> Vec<f32> {
        let mut values = vec![1.0f32; len];
        if feather == 0 || feather >= len {
            return values;
        }
        for t in 0..feather {
            let a = (t + 1) as f32 / feather as f32;
            values[t] *= a;
            values[len - 1 - t] *= a;
        }
        values
    };

    let rows = Tensor::from_vec(ramp(height), (1, 1, 1, height, 1), device)?;
    let cols = Tensor::from_vec(ramp(width), (1, 1, 1, 1, width), device)?;
    rows.broadcast_mul(&cols)
}

fn patchify(x: &Tensor, patch_size: usize) -> Result<Tensor> {
    if patch_size == 1 {
        return x.contiguous();
    }
    let (b, c, f, h, w) = x.dims5()?;
    let p = patch_size;
    x.reshape(&[b, c, f, h / p, p, w / p, p])?
        .permute(vec![0, 1, 6, 4, 2, 3, 5])?
        .contiguous()?
        .reshape(&[b, c * p * p, f, h / p, w / p])
}

/// `unpatchify` (`vae2_2.py:299-313`): `b (c r q) f h w -> b c f (h q) (w r)`.
fn unpatchify(x: &Tensor, patch_size: usize) -> Result<Tensor> {
    if patch_size == 1 {
        return x.contiguous();
    }
    let (b, c_packed, f, h, w) = x.dims5()?;
    let p = patch_size;
    let c = c_packed / (p * p);
    x.reshape(&[b, c, p, p, f, h, w])?
        .permute(vec![0, 1, 4, 5, 3, 6, 2])?
        .contiguous()?
        .reshape(&[b, c, f, h * p, w * p])
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn build(config: WanVaeConfig) -> (WanVideoVae, VarMap) {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let vae = WanVideoVae::from_var_builder(vb, config, &device, DType::F32)
            .expect("tiny Wan VAE constructs from a VarMap");
        (vae, varmap)
    }

    fn randn(shape: &[usize]) -> Tensor {
        Tensor::randn(0f32, 1f32, shape, &Device::Cpu).expect("random tensor")
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        assert_eq!(a.dims(), b.dims(), "shape mismatch");
        a.sub(b)
            .expect("subtract")
            .abs()
            .expect("abs")
            .flatten_all()
            .expect("flatten")
            .max(0)
            .expect("max")
            .to_scalar::<f32>()
            .expect("scalar")
    }

    /// The `Conv2d`-slice decomposition must equal a literal 3-D convolution
    /// with left-only temporal padding. Computed here by hand from the raw
    /// weight so nothing about the port is assumed.
    #[test]
    fn causal_conv3d_matches_a_hand_rolled_reference() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let (cin, cout, kt, kh, kw) = (2usize, 3usize, 3usize, 3usize, 3usize);
        let conv = WanCausalConv3d::load(vb.pp("c"), cin, cout, (kt, kh, kw), (1, 1, 1), (1, 1, 1))
            .unwrap();

        let (t, h, w) = (4usize, 5usize, 4usize);
        let x = randn(&[1, cin, t, h, w]);
        let got = conv.forward(&x, None).unwrap();
        assert_eq!(got.dims(), &[1, cout, t, h, w]);

        let (weight, bias) = {
            let data = varmap.data().lock().unwrap();
            (
                data["c.weight"]
                    .as_tensor()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                data["c.bias"].as_tensor().to_vec1::<f32>().unwrap(),
            )
        };
        let xv = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let at_x = |c: usize, ti: isize, hi: isize, wi: isize| -> f32 {
            // Zero outside the clip: two frames in front (2 * padding[0]) and
            // one row/column of spatial zero padding on each side.
            if ti < 0 || hi < 0 || wi < 0 || hi >= h as isize || wi >= w as isize {
                return 0.0;
            }
            if ti >= t as isize {
                return 0.0;
            }
            xv[((c * t + ti as usize) * h + hi as usize) * w + wi as usize]
        };
        let at_wt = |co: usize, ci: usize, kti: usize, khi: usize, kwi: usize| -> f32 {
            weight[(((co * cin + ci) * kt + kti) * kh + khi) * kw + kwi]
        };

        let mut want = vec![0f32; cout * t * h * w];
        for co in 0..cout {
            for to in 0..t {
                for ho in 0..h {
                    for wo in 0..w {
                        let mut acc = bias[co];
                        for ci in 0..cin {
                            for kti in 0..kt {
                                // Two frames of causal front padding shift the
                                // window back by (kt - 1).
                                let ti = to as isize + kti as isize - (kt as isize - 1);
                                for khi in 0..kh {
                                    let hi = ho as isize + khi as isize - 1;
                                    for kwi in 0..kw {
                                        let wi = wo as isize + kwi as isize - 1;
                                        acc += at_wt(co, ci, kti, khi, kwi) * at_x(ci, ti, hi, wi);
                                    }
                                }
                            }
                        }
                        want[((co * t + to) * h + ho) * w + wo] = acc;
                    }
                }
            }
        }
        let want = Tensor::from_vec(want, (1, cout, t, h, w), &device).unwrap();
        assert!(max_abs_diff(&got, &want) < 1e-4);
    }

    /// A causal conv fed the previous chunk's tail must equal one pass over the
    /// whole zero-padded clip.
    #[test]
    fn causal_conv3d_streaming_equals_one_pass() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let conv =
            WanCausalConv3d::load(vb.pp("c"), 2, 2, (3, 3, 3), (1, 1, 1), (1, 1, 1)).unwrap();

        let x = randn(&[1, 2, 7, 4, 4]);
        let full = conv.forward(&x, None).unwrap();

        let head = x.narrow(2, 0, 1).unwrap().contiguous().unwrap();
        let tail = x.narrow(2, 1, 6).unwrap().contiguous().unwrap();
        let a = conv.forward(&head, None).unwrap();
        let b = conv
            .forward(&tail, Some(&tail_frames(&head, CACHE_T).unwrap()))
            .unwrap();
        let streamed = Tensor::cat(&[&a, &b], 2).unwrap();
        assert!(max_abs_diff(&full, &streamed) < 1e-5);
    }

    /// `RMS_norm` must be `x / ||x||_2 * sqrt(dim) * gamma` over the channel
    /// axis.
    #[test]
    fn rms_norm_matches_l2_normalize() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let norm = WanRmsNorm::load(vb.pp("n"), 4, false).unwrap();

        let x = Tensor::from_vec(
            vec![3f32, 4.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            (1, 4, 2, 1, 1),
            &device,
        )
        .unwrap();
        let got = norm
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // gamma initializes to 1, so the expectation is x / ||x|| * 2.
        // Position 0 has channels (3, 0, 1, 1) -> norm sqrt(11);
        // position 1 has (4, 0, 1, 1) -> norm sqrt(18).
        let s = 2.0f32;
        let n0 = 11f32.sqrt();
        let n1 = 18f32.sqrt();
        let want = [
            3.0 / n0 * s,
            4.0 / n1 * s,
            0.0,
            0.0,
            1.0 / n0 * s,
            1.0 / n1 * s,
            1.0 / n0 * s,
            1.0 / n1 * s,
        ];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
        }
    }

    /// Encoding is a causal stream, so how the clip is chunked past the leading
    /// single frame must not change the latents. A wrong cache index, a wrong
    /// tail length, or a wrong padding reduction all break this.
    #[test]
    fn wan21_encode_is_chunk_plan_invariant() {
        let (vae, _map) = build(WanVaeConfig::tiny_v2_1());
        let video = randn(&[1, 3, 13, 32, 32]);
        let patched = patchify(&video, 1).unwrap();
        let reference = vae.encode_stream(&patched, &[1, 4, 4, 4]).unwrap();
        assert_eq!(reference.dims(), &[1, 8, 4, 4, 4]);
        for plan in [vec![1, 8, 4], vec![1, 4, 8], vec![1, 12]] {
            let other = vae.encode_stream(&patched, &plan).unwrap();
            assert!(
                max_abs_diff(&reference, &other) < 1e-4,
                "encode plan {plan:?} diverged"
            );
        }
    }

    /// The structural contract of the tiler, which is what a unit test can
    /// actually hold it to.
    ///
    /// Deliberately NOT a numerical-closeness assertion against the untiled
    /// decode. A tile's convolutions see zero padding where the full canvas
    /// had neighbouring pixels — the approximation every tiler upstream makes
    /// — and against these *untrained* fixture weights the receptive field
    /// spans a whole 4-px latent tile, so the difference is large and spread
    /// evenly across the tile rather than banded at the seams. Measured on
    /// this fixture it is ~1.6 with the error map uniform, which says the
    /// oracle is bad, not the blend. Quality is judged on a real checkpoint in
    /// UAT; what belongs here is that the plan, the shapes, and the
    /// short-circuit are right.
    #[test]
    fn wan21_tiled_decode_covers_the_canvas_and_short_circuits() {
        let (vae, _map) = build(WanVaeConfig::tiny_v2_1());
        let latents = randn(&[1, 4, 2, 8, 8]);

        let reference = vae.decode(&latents).unwrap();
        assert_eq!(reference.dims(), &[1, 3, 5, 64, 64]);

        // A tile at or above both axes must be the plain decode itself, not an
        // approximation of it — otherwise every render that does not need
        // tiling would still pay for the blend.
        let untiled = vae.decode_tiled(&latents, 8, 2).unwrap();
        assert_eq!(
            max_abs_diff(&reference, &untiled),
            0.0,
            "a single-tile plan must be the plain decode"
        );

        // Genuinely subdivided: same shape, every pixel written, all finite,
        // and inside the decoder's own output range.
        let tiled = vae.decode_tiled(&latents, 4, 1).unwrap();
        assert_eq!(tiled.dims(), reference.dims());
        let values = tiled.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            values.iter().all(|v| v.is_finite()),
            "a gap in the tile plan or a zero divisor shows up as NaN"
        );
        assert!(
            values.iter().all(|v| (-1.0..=1.0).contains(v)),
            "tiled decode left the clamped output range"
        );

        // An overlap at or above the tile would make the stride zero; it is
        // clamped rather than hanging.
        let clamped = vae.decode_tiled(&latents, 4, 9).unwrap();
        assert_eq!(clamped.dims(), reference.dims());
    }

    /// The blend mask is ComfyUI's, and the property that matters is that it
    /// ramps from near zero at a feathered edge to one in the interior.
    #[test]
    fn the_tile_blend_mask_ramps_at_the_edges_and_is_flat_inside() {
        let device = Device::Cpu;
        let mask = tile_blend_mask(8, 8, 2, &device).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 1, 8, 8]);
        let values = mask.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let at = |r: usize, c: usize| values[r * 8 + c];

        // Interior is untouched.
        assert!((at(4, 4) - 1.0).abs() < 1e-6);
        // Corners get both ramps, so they are the product of the two.
        assert!((at(0, 0) - 0.25).abs() < 1e-6, "corner was {}", at(0, 0));
        // Edges ramp on one axis only.
        assert!((at(0, 4) - 0.5).abs() < 1e-6, "top edge was {}", at(0, 4));
        assert!((at(4, 0) - 0.5).abs() < 1e-6, "left edge was {}", at(4, 0));
        // The ramp is symmetric.
        assert!((at(0, 4) - at(7, 4)).abs() < 1e-6);
        assert!((at(4, 0) - at(4, 7)).abs() < 1e-6);

        // A feather at or above the axis leaves the mask flat rather than
        // collapsing it to zero.
        let flat = tile_blend_mask(4, 4, 4, &device).unwrap();
        let flat = flat.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(flat.iter().all(|v| (*v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn wan21_decode_is_chunk_plan_invariant() {
        let (vae, _map) = build(WanVaeConfig::tiny_v2_1());
        let latents = randn(&[1, 4, 4, 4, 4]);
        let x = vae.conv2.forward(&latents, None).unwrap();
        let reference = vae.decode_stream(&x, &[1, 1, 1, 1]).unwrap();
        assert_eq!(reference.dims(), &[1, 3, 13, 32, 32]);
        for plan in [vec![1, 3], vec![1, 2, 1], vec![1, 1, 2]] {
            let other = vae.decode_stream(&x, &plan).unwrap();
            assert!(
                max_abs_diff(&reference, &other) < 1e-4,
                "decode plan {plan:?} diverged"
            );
        }
    }

    #[test]
    fn wan22_encode_is_chunk_plan_invariant() {
        let (vae, _map) = build(WanVaeConfig::tiny_v2_2());
        let video = randn(&[1, 3, 13, 32, 32]);
        let patched = patchify(&video, 2).unwrap();
        let reference = vae.encode_stream(&patched, &[1, 4, 4, 4]).unwrap();
        assert_eq!(reference.dims(), &[1, 8, 4, 2, 2]);
        for plan in [vec![1, 8, 4], vec![1, 4, 8], vec![1, 12]] {
            let other = vae.encode_stream(&patched, &plan).unwrap();
            assert!(
                max_abs_diff(&reference, &other) < 1e-4,
                "encode plan {plan:?} diverged"
            );
        }
    }

    #[test]
    fn wan22_decode_is_chunk_plan_invariant() {
        let (vae, _map) = build(WanVaeConfig::tiny_v2_2());
        let latents = randn(&[1, 4, 4, 2, 2]);
        let x = vae.conv2.forward(&latents, None).unwrap();
        let reference = vae.decode_stream(&x, &[1, 1, 1, 1]).unwrap();
        // Still packed: 12 channels that unpatchify to 3 at 2x the spatial size.
        assert_eq!(reference.dims(), &[1, 12, 13, 16, 16]);
        for plan in [vec![1, 3], vec![1, 2, 1], vec![1, 1, 2]] {
            let other = vae.decode_stream(&x, &plan).unwrap();
            assert!(
                max_abs_diff(&reference, &other) < 1e-4,
                "decode plan {plan:?} diverged"
            );
        }
    }

    /// Causality means a prefix of the clip encodes to a prefix of the latents.
    #[test]
    fn wan21_encode_prefix_is_stable() {
        let (vae, _map) = build(WanVaeConfig::tiny_v2_1());
        let long = randn(&[1, 3, 13, 32, 32]);
        let short = long.narrow(2, 0, 5).unwrap().contiguous().unwrap();
        let z_long = vae.encode(&long).unwrap();
        let z_short = vae.encode(&short).unwrap();
        assert_eq!(z_long.dims(), &[1, 4, 4, 4, 4]);
        assert_eq!(z_short.dims(), &[1, 4, 2, 4, 4]);
        let prefix = z_long.narrow(2, 0, 2).unwrap().contiguous().unwrap();
        assert!(max_abs_diff(&prefix, &z_short) < 1e-4);
    }

    #[test]
    fn wan21_shape_contract() {
        let cfg = WanVaeConfig::tiny_v2_1();
        assert_eq!(cfg.temporal_compression(), 4);
        assert_eq!(cfg.spatial_compression(), 8);
        let (vae, _map) = build(cfg);
        for frames in [1usize, 5, 9, 13] {
            let video = randn(&[1, 3, frames, 32, 24]);
            let latents = vae.encode(&video).unwrap();
            assert_eq!(latents.dims(), &[1, 4, (frames - 1) / 4 + 1, 4, 3]);
            let decoded = vae.decode(&latents).unwrap();
            assert_eq!(decoded.dims(), &[1, 3, frames, 32, 24]);
        }
    }

    #[test]
    fn wan22_shape_contract() {
        let cfg = WanVaeConfig::tiny_v2_2();
        assert_eq!(cfg.temporal_compression(), 4);
        assert_eq!(cfg.spatial_compression(), 16);
        assert_eq!(cfg.encoder_in_channels(), 12);
        let (vae, _map) = build(cfg);
        for frames in [1usize, 5, 9] {
            let video = randn(&[1, 3, frames, 32, 16]);
            let latents = vae.encode(&video).unwrap();
            assert_eq!(latents.dims(), &[1, 4, (frames - 1) / 4 + 1, 2, 1]);
            let decoded = vae.decode(&latents).unwrap();
            assert_eq!(decoded.dims(), &[1, 3, frames, 32, 16]);
        }
    }

    /// Decode output must be clamped into the pixel range upstream promises.
    #[test]
    fn decode_clamps_to_unit_range() {
        let (vae, _map) = build(WanVaeConfig::tiny_v2_1());
        let latents = randn(&[1, 4, 3, 4, 4]).affine(50.0, 0.0).unwrap();
        let decoded = vae.decode(&latents).unwrap();
        let flat = decoded.flatten_all().unwrap();
        let lo = flat.min(0).unwrap().to_scalar::<f32>().unwrap();
        let hi = flat.max(0).unwrap().to_scalar::<f32>().unwrap();
        assert!(lo >= -1.0 && hi <= 1.0, "range [{lo}, {hi}]");
    }

    /// `patchify`/`unpatchify` must be exact inverses, including the channel
    /// ordering upstream's einops pattern implies.
    #[test]
    fn patchify_round_trips() {
        let x = randn(&[2, 3, 5, 8, 6]);
        let packed = patchify(&x, 2).unwrap();
        assert_eq!(packed.dims(), &[2, 12, 5, 4, 3]);
        let back = unpatchify(&packed, 2).unwrap();
        assert_eq!(max_abs_diff(&x, &back), 0.0);
    }

    /// A round trip cannot tell one packing convention from another, so pin the
    /// exact channel order `b c f (h q) (w r) -> b (c r q) f h w` implies:
    /// channel `r * 2 + q` carries the sub-pixel at row `q`, column `r`.
    #[test]
    fn patchify_channel_order_matches_einops() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![0f32, 1.0, 2.0, 3.0], (1, 1, 1, 2, 2), &device).unwrap();
        let packed = patchify(&x, 2).unwrap();
        assert_eq!(packed.dims(), &[1, 4, 1, 1, 1]);
        assert_eq!(
            packed.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.0, 2.0, 1.0, 3.0]
        );
    }

    #[test]
    fn encode_chunk_plan_matches_upstream() {
        assert_eq!(encode_chunk_plan(1), vec![1]);
        assert_eq!(encode_chunk_plan(5), vec![1, 4]);
        assert_eq!(encode_chunk_plan(13), vec![1, 4, 4, 4]);
        // The shipped 81-frame default: one leading frame then twenty quads.
        let mut want = vec![1usize];
        want.extend(std::iter::repeat_n(4, 20));
        assert_eq!(encode_chunk_plan(81), want);
    }

    #[test]
    fn encode_rejects_ragged_frame_counts() {
        let (vae, _map) = build(WanVaeConfig::tiny_v2_1());
        let err = vae
            .encode(&randn(&[1, 3, 6, 32, 32]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("4k+1"), "unexpected error: {err}");
    }

    /// The per-channel latent tables are hardcoded, so pin their length and
    /// their endpoints against the Python sources.
    #[test]
    fn latent_statistics_match_upstream() {
        let v21 = WanVaeConfig::v2_1();
        assert_eq!(v21.latent_mean.len(), 16);
        assert_eq!(v21.latent_std.len(), 16);
        // vae.py:629-636
        assert_eq!(v21.latent_mean[0], -0.7571);
        assert_eq!(v21.latent_mean[15], -0.2921);
        assert_eq!(v21.latent_std[0], 2.8184);
        assert_eq!(v21.latent_std[15], 1.916);

        let v22 = WanVaeConfig::v2_2();
        assert_eq!(v22.latent_mean.len(), 48);
        assert_eq!(v22.latent_std.len(), 48);
        // vae2_2.py:904-1011
        assert_eq!(v22.latent_mean[0], -0.2289);
        assert_eq!(v22.latent_mean[47], -0.0667);
        assert_eq!(v22.latent_std[0], 0.4765);
        assert_eq!(v22.latent_std[47], 0.7744);
        // A middle sample from each table, so a truncated paste is caught.
        assert_eq!(v22.latent_mean[19], 0.5109);
        assert_eq!(v22.latent_std[39], 1.6887);
    }

    /// The production stage widths and the parameter-free 2.2 shortcut factors
    /// are pure config arithmetic; pin them rather than materializing 127 M
    /// (2.1) and 705 M (2.2) parameters in a unit test.
    #[test]
    fn production_stage_widths_match_upstream() {
        let v21 = WanVaeConfig::v2_1();
        v21.validate().unwrap();
        assert_eq!(v21.encoder_dims(), vec![96, 96, 192, 384, 384]);
        assert_eq!(v21.decoder_dims(), vec![384, 384, 384, 192, 96]);
        assert_eq!(v21.temporal_upsample(), vec![true, true, false]);
        assert_eq!(v21.encoder_in_channels(), 3);
        assert_eq!(v21.temporal_compression(), 4);
        assert_eq!(v21.spatial_compression(), 8);

        let v22 = WanVaeConfig::v2_2();
        v22.validate().unwrap();
        assert_eq!(v22.encoder_dims(), vec![160, 160, 320, 640, 640]);
        assert_eq!(v22.decoder_dims(), vec![1024, 1024, 1024, 512, 256]);
        assert_eq!(v22.encoder_in_channels(), 12);
        assert_eq!(v22.temporal_compression(), 4);
        assert_eq!(v22.spatial_compression(), 16);

        // Every 2.2 stage's shortcut must satisfy upstream's divisibility
        // asserts at production widths (vae2_2.py:332, 387).
        let enc = v22.encoder_dims();
        let temporal = &v22.temporal_downsample;
        for i in 0..v22.dim_mult.len() {
            let down = i != v22.dim_mult.len() - 1;
            AvgDown3d::new(
                enc[i],
                enc[i + 1],
                if temporal.get(i).copied().unwrap_or(false) {
                    2
                } else {
                    1
                },
                if down { 2 } else { 1 },
            )
            .unwrap();
        }
        let dec = v22.decoder_dims();
        let up = v22.temporal_upsample();
        for i in 0..(v22.dim_mult.len() - 1) {
            DupUp3d::new(dec[i], dec[i + 1], if up[i] { 2 } else { 1 }, 2).unwrap();
        }
    }

    fn key_set(varmap: &VarMap) -> Vec<String> {
        let mut keys: Vec<String> = varmap.data().lock().unwrap().keys().cloned().collect();
        keys.sort();
        keys
    }

    /// The tiny configs carry the production topology, so the `VarMap` they
    /// materialize is the production checkpoint key layout. Pin the indices
    /// that the flat/nested Sequential arithmetic produces.
    #[test]
    fn wan21_key_layout_matches_the_checkpoint() {
        let (_vae, varmap) = build(WanVaeConfig::tiny_v2_1());
        let keys = key_set(&varmap);
        let has = |k: &str| keys.iter().any(|s| s == k);
        for key in [
            "conv1.weight",
            "conv2.weight",
            "encoder.conv1.weight",
            // stage 0: two residual blocks then a spatial-only downsample
            "encoder.downsamples.0.residual.0.gamma",
            "encoder.downsamples.0.residual.2.weight",
            "encoder.downsamples.0.residual.3.gamma",
            "encoder.downsamples.0.residual.6.weight",
            "encoder.downsamples.2.resample.1.weight",
            // stage 1 widens, so its first block owns a shortcut
            "encoder.downsamples.3.shortcut.weight",
            "encoder.downsamples.5.time_conv.weight",
            "encoder.downsamples.8.time_conv.weight",
            "encoder.downsamples.10.residual.6.weight",
            "encoder.middle.0.residual.0.gamma",
            "encoder.middle.1.to_qkv.weight",
            "encoder.middle.1.proj.bias",
            "encoder.middle.2.residual.6.weight",
            "encoder.head.0.gamma",
            "encoder.head.2.weight",
            "decoder.conv1.weight",
            "decoder.middle.1.norm.gamma",
            "decoder.upsamples.3.time_conv.weight",
            "decoder.upsamples.7.time_conv.weight",
            "decoder.upsamples.11.resample.1.weight",
            "decoder.upsamples.14.residual.6.weight",
            "decoder.head.2.bias",
        ] {
            assert!(has(key), "missing {key}");
        }
        // Flat indices stop at 10 / 14 (vae.py:291-306, 400-416).
        assert!(!has("encoder.downsamples.11.residual.0.gamma"));
        assert!(!has("decoder.upsamples.15.residual.0.gamma"));
        // upsample2d has no time conv.
        assert!(!has("decoder.upsamples.11.time_conv.weight"));
        // The 2.2 nesting must not appear.
        assert!(!keys.iter().any(|k| k.contains("downsamples.0.downsamples")));
    }

    #[test]
    fn wan22_key_layout_matches_the_checkpoint() {
        let (_vae, varmap) = build(WanVaeConfig::tiny_v2_2());
        let keys = key_set(&varmap);
        let has = |k: &str| keys.iter().any(|s| s == k);
        for key in [
            "encoder.downsamples.0.downsamples.0.residual.2.weight",
            "encoder.downsamples.0.downsamples.2.resample.1.weight",
            "encoder.downsamples.1.downsamples.2.time_conv.weight",
            "encoder.downsamples.2.downsamples.2.time_conv.weight",
            "encoder.downsamples.3.downsamples.1.residual.6.weight",
            // ComfyUI's 2.2 detection probe (report-comfyui.md 1.3).
            "decoder.upsamples.0.upsamples.0.residual.2.weight",
            "decoder.upsamples.0.upsamples.3.time_conv.weight",
            "decoder.upsamples.1.upsamples.3.time_conv.weight",
            "decoder.upsamples.2.upsamples.3.resample.1.weight",
            "decoder.upsamples.3.upsamples.2.residual.6.weight",
            "decoder.middle.0.residual.0.gamma",
        ] {
            assert!(has(key), "missing {key}");
        }
        // The last decoder stage has no resample and no shortcut parameters.
        assert!(!has("decoder.upsamples.3.upsamples.3.resample.1.weight"));
        // Spatial-only stage 0 has no time conv.
        assert!(!has("encoder.downsamples.0.downsamples.2.time_conv.weight"));
        // The average-pool / duplicate shortcuts are parameter-free.
        assert!(!keys.iter().any(|k| k.contains("avg_shortcut")));
    }

    #[test]
    fn config_validation_rejects_mismatched_tables() {
        let mut cfg = WanVaeConfig::v2_1();
        cfg.latent_std.pop();
        assert!(cfg.validate().is_err());

        let mut cfg = WanVaeConfig::v2_1();
        cfg.temporal_downsample = vec![true, true];
        assert!(cfg.validate().is_err());
    }
    const GOLDEN_V21_ENCODE: &[f64] = &[
        0.28541356900077336,
        0.2854779783399573,
        0.2849592488731112,
        0.28462007658875604,
        0.28587250766186384,
        0.285986133376414,
        0.2849381118891893,
        0.2842693655482159,
        0.28633305768391243,
        0.2864878131362548,
        0.28490583362153965,
        0.2839131283182757,
        0.5158684705079641,
        0.5164692006638391,
        0.5160656614792941,
        0.5152511828001655,
        0.5164174426594018,
        0.5175868689652523,
        0.5167235093951997,
        0.5150967749390558,
        0.5169855807057634,
        0.5187058639863205,
        0.5173538172854946,
        0.514917375935914,
        0.39303017523591066,
        0.3935343282129457,
        0.3937714086148528,
        0.3933928333662576,
        0.39300643045117384,
        0.3940020907060312,
        0.39443450165483596,
        0.393667882673651,
        0.3929993074630696,
        0.3944789798771493,
        0.395084397724926,
        0.3939254056543181,
        -0.05469010139812205,
        -0.05433367355321547,
        -0.053790444613519506,
        -0.05385913258616982,
        -0.05502295205111126,
        -0.05430974754519419,
        -0.05324922885475277,
        -0.05340069320659375,
        -0.05534370094135719,
        -0.05427414125996143,
        -0.05271087420627365,
        -0.052952395765917405,
    ];
    const GOLDEN_V21_DECODE_PROBES: &[f64] = &[
        0.16625932155248263,
        -0.12115336595087728,
        0.027538322902072615,
        0.12264302361714854,
        -0.08837984426123152,
        -0.06631215999373327,
        -0.28519628547689696,
        0.23146329319588269,
        0.39532782841697867,
        -0.26331574592236706,
        -0.5727164088170484,
        0.5265851918424986,
        0.5110159750901077,
        -0.1897449280286741,
        0.29411696320988934,
        -0.47864286709180404,
        0.23114645354550642,
        0.02400039745187403,
        -0.4884599190628749,
        0.3611099804399578,
        0.5184515210192773,
        -0.24576728369224254,
        -0.3724960791007724,
        0.3915723188535392,
        -0.17333997898291761,
        0.5052012177363836,
        -0.2936996039566173,
        -0.3504736460609408,
        0.5049720152940144,
        0.5886325473406522,
        -0.4533156902236791,
        -0.2570521878789533,
        0.2179693138017445,
        -0.3210584571652122,
        0.635608484919898,
        -0.3856066354753301,
        0.45984046349587687,
        0.6048214440671047,
        -0.4086633552888846,
        -0.5012196938273352,
        0.5098503351666063,
        0.5198939560991449,
        -0.35869520289199397,
        0.24202374084373682,
        -0.13532571848624592,
        0.10618413079126568,
        0.06892443768290694,
        -0.11062519960151651,
        -0.044060631040436574,
        0.25437073019322304,
        -0.09473325301937456,
        -0.22455337145888782,
        0.1952248939703132,
        -0.0733028116944058,
        0.3814629179080815,
        -0.30506174737504843,
        -0.4643352959392758,
        0.28970039727062047,
        0.652335508563302,
        -0.28809907696241344,
        -0.2606882834716587,
        0.23494724859243044,
        -0.287183137617253,
        0.43970306092245753,
        -0.02636397819298452,
        0.4436718597942721,
        0.5011157971688744,
        -0.34067491004138445,
        -0.3708122573222803,
        0.33323682135272736,
        0.5643696223080777,
        -0.25058925367638224,
        0.09779817856647358,
        -0.5718459754651022,
        0.07354787736003637,
        0.10389367277988812,
        -0.2999616647044435,
        -0.2549566073708456,
        0.30636661246979424,
        0.0597126813992296,
        -0.16918803162140256,
        0.2534171006122905,
        -0.12783982846532221,
        0.3736702615935804,
        -0.42620958907381534,
        -0.4911651820788838,
        0.2932027074074878,
        0.15040861283186308,
        -0.0709132783302632,
        -0.12282590247737019,
        -0.025621703596399772,
        -0.06315050523861535,
        0.19413230200946854,
        -0.07457217911362429,
        0.18778319166284482,
        0.21397598528452927,
        0.02523663510195856,
        -0.6500622999069068,
        0.057271340595747566,
        0.33768817318769684,
        -0.09106082999472911,
        -0.11830054099948066,
        -0.6582814038620244,
        0.06726527844425154,
        0.15903580010900387,
        -0.16461331234939713,
        -0.15161422221270363,
        0.06797850685802607,
        0.007421079167583659,
        -0.3401335158953277,
        0.02207817067026556,
        -0.06386815517163284,
        0.011396135414276282,
        -0.4254440951535722,
        -0.5442627209986586,
        0.1783024835459659,
        0.5342845850446121,
        -0.16105436183193161,
        -0.5092218153457324,
        0.12678379516030483,
        -0.12548043509036422,
        0.6089659197954271,
        0.004572699448491396,
        0.2845932560254896,
        0.28410811882297937,
        -0.022281984524130695,
        -0.15220816498438913,
        0.0344479352060832,
        0.2969360798600738,
        -0.09214966476273068,
        -0.11762290329740069,
    ];
    const GOLDEN_V21_DECODE_MEANS: &[f64] = &[
        0.03805848636620867,
        0.029949290005471727,
        0.022720027557631597,
        0.028222674154902185,
        0.03577845297749539,
        0.043093200782860844,
        0.04678087930184359,
        0.04755195734439171,
        0.04764026180434472,
        0.007742924255844283,
        0.0003078346608151763,
        -0.006243895476616,
        -0.00030125979172917713,
        0.007818820949232993,
        0.01558219956587743,
        0.01918395823390586,
        0.01966682917099028,
        0.019471755364957218,
        -0.03332768896357434,
        -0.038894175330822345,
        -0.043715792389939855,
        -0.038287891871119965,
        -0.03090805360605773,
        -0.02394328178263102,
        -0.021006089103686114,
        -0.02088900166170011,
        -0.02133611384316795,
    ];
    const V21_DECODE_PROBE_STRIDE: usize = 53;
    const GOLDEN_V22_ENCODE: &[f64] = &[
        0.578887949169003,
        0.5793844800510664,
        0.5772781594778871,
        0.5756659434769039,
        0.5810756431177437,
        0.5822339211663885,
        0.5777497421074441,
        0.5742162645161261,
        0.5833517485095251,
        0.5850537456855252,
        0.5780463988829697,
        0.5726327195600197,
        0.04447544122814935,
        0.04520356153221614,
        0.04497720916019134,
        0.04412919353601716,
        0.044884712080283495,
        0.04649920638166051,
        0.04608723295222926,
        0.04425452269446394,
        0.045382196300032526,
        0.04783641284937808,
        0.0471180079553113,
        0.04428326346545179,
        0.3004770962800196,
        0.3025460464921017,
        0.30396336944733837,
        0.3026451530635571,
        0.29962531177949525,
        0.3041524057302224,
        0.3074187468976297,
        0.30462178254311867,
        0.2989943583236483,
        0.3059380438413197,
        0.3107765443565484,
        0.30639573666569436,
        0.16807672104390503,
        0.1686710649916246,
        0.16972182594982288,
        0.16968405367106348,
        0.16720271993153316,
        0.1684843184524432,
        0.17080854376223992,
        0.17075805096213623,
        0.1663828400314555,
        0.16836818350833738,
        0.17190708805257915,
        0.17179624692351533,
    ];
    const GOLDEN_V22_DECODE_PROBES: &[f64] = &[
        0.03669274765739762,
        -0.011128194758272921,
        0.0006995477855355724,
        -0.15836558351372154,
        0.010408146608942272,
        0.08491417841329593,
        0.07940674351099837,
        0.21184352104285128,
        0.07831760888139341,
        -0.22721896519028079,
        0.09804939270958955,
        -0.01854592919511735,
        0.4051273570950103,
        -0.2473833837059608,
        0.0773347164196987,
        -0.09078650369417247,
        -0.0950901630386532,
        0.098378756209385,
        0.4223827080395486,
        -0.2636111636607852,
        -0.03045316516055916,
        -0.40348865385555116,
        0.07061612110164783,
        0.30408648266732813,
        0.04586649339639466,
        0.047575886761668656,
        -0.29638326435119816,
        -0.15647109582256097,
        -0.06737639866289115,
        0.002406507714996904,
        0.03755530293737295,
        -0.19122729890868476,
        -0.161107002730314,
        0.07331025220413621,
        -0.03547937750535386,
        0.03682013843970408,
        -0.1267639257974948,
        -0.3090213013471819,
        -0.034859653116398756,
        0.19506506772703425,
        0.1548596140855497,
        0.379998915300898,
        0.0883853426638555,
        -0.1373185991157287,
        -0.1051356081875133,
        0.08699093449834817,
        0.02910379451329255,
        0.0006543025261491683,
        -0.07965089820606422,
        -0.046667611395539974,
        0.13285787034116348,
        0.21955911882056886,
        0.1757171825956764,
        0.0043100009403996345,
        0.05318229900423725,
        -0.003916562160084007,
        0.2442580766413583,
        -0.14491234747368464,
        0.05575482216131211,
        0.21038301287687852,
        -0.2041597819074612,
        -0.022161674969508827,
        0.1278914507880594,
        0.0690253976953623,
        -0.2169801067987848,
        -0.038023863404272995,
        -0.12613555504726703,
        0.2777114652650926,
        0.26569563686976966,
        0.010002675967217614,
        -0.12523672833449623,
        -0.12563625199087994,
        -0.17823531257423803,
        0.1698869883239067,
        -0.0003575802822664764,
        -0.07540047010987183,
        -0.2510119010649144,
        0.11484937330224267,
        -0.17401204040500445,
        0.2899673768989237,
        0.05789762409011828,
        -0.06788048489059438,
        -0.28340047081467745,
        -0.017198521592735758,
        0.25070424050138335,
        0.30791201840281635,
        0.04434960204304954,
        0.06804030717011932,
        0.015188933101956488,
        0.04110964319013985,
        0.030998861058837243,
        -0.05383384247873771,
        0.03123244405461103,
        -0.054241924280065947,
        0.16947762091020238,
        0.08578853134926417,
        -0.027643949340217897,
        -0.06918610250684525,
        -0.3065127427494863,
        0.057642551139872045,
        0.13306853322095738,
        0.1099227069030648,
        0.015138930179777186,
        -0.14254285898827462,
        0.0511254952588006,
        0.1120086768008419,
        0.0013522161603959412,
        -0.31894585221690663,
        0.07270183580737968,
        -0.4884396810524618,
        0.10997978375321443,
        0.3355239861032197,
        0.020131903089855768,
        -0.6370070857777832,
        -0.48253880085056866,
        -0.40776325698406246,
        0.010136268171453834,
        0.09215352628106208,
        0.45255659463452913,
        -0.17097309263432992,
        -0.4006765738528502,
        0.23750678967630034,
        0.09703001550862543,
        0.10163547696086705,
        -0.03367749892484226,
        -0.1324564457726731,
        -0.02513917366713095,
        -0.06817185882573065,
        0.2616706199220633,
        0.17983172416411217,
        -0.037453460630965024,
        -0.047318683768292194,
    ];
    const GOLDEN_V22_DECODE_MEANS: &[f64] = &[
        -0.007222604528508958,
        -0.011413653339580736,
        -0.015625826746444263,
        -0.014894179068466268,
        -0.013456312980547497,
        -0.012070595098062662,
        -0.01190152479021657,
        -0.012109961904850398,
        -0.012291641691096774,
        0.0181608029397836,
        0.019907239187345955,
        0.02116209165604031,
        0.02022396368012063,
        0.02041935470005219,
        0.021053413804810524,
        0.021124672354297838,
        0.020944329735150008,
        0.020754481748075957,
        -0.020674824450836952,
        -0.016635368206490652,
        -0.012532116713242218,
        -0.013182334130490829,
        -0.014637160510478437,
        -0.016077915438201362,
        -0.01625317106179453,
        -0.016029080006687867,
        -0.01583092120729866,
    ];
    const V22_DECODE_PROBE_STRIDE: usize = 211;

    // ---------------------------------------------------------------------
    // Golden parity with the upstream reference implementations (#789).
    //
    // Weights, inputs, and expected outputs are shared with the capture
    // script `tmp/wan-research/gen-wan-vae-goldens.py`, which runs upstream's
    // own modules — `tmp/Wan2.1/wan/modules/vae.py` `WanVAE_.encode`/`decode`
    // (vae.py:516-568) and `tmp/Wan2.2/wan/modules/vae2_2.py`
    // `WanVAE_.encode`/`decode` (vae2_2.py:783-839) — in float64 over the
    // same float32-cast synthesized weights before emitting the literals
    // (`wan-vae-golden.json`). The 2.1 fixture is additionally cross-checked
    // against diffusers 0.38.0 `AutoencoderKLWan` (exact key-set match after
    // renaming); the 2.2 fixture is pinned against upstream `vae2_2.py`
    // alone — diffusers' `is_residual` key naming diverges enough that a
    // second rename map would itself be a fixture risk, and upstream
    // Wan-Video is the canonical reference per CLAUDE.md.
    //
    // Every golden covers three streaming chunks (encode plan [1, 4, 4];
    // decode plan [1, 1, 1]), so the feat-cache protocol — slot visit order,
    // two-frame tail carry, the upsample3d 'Rep' sentinel, downsample3d's
    // self-spliced carry, and 2.2's `first_chunk` frame drop — sits inside
    // the parity boundary instead of only being self-consistent. Mutation
    // spot-checks recorded when the fixture landed: truncating the causal
    // tail carry to one frame moves the decode probes by 8.5e-3 (and the
    // weakest signal, 2.2 encode, by 3.5e-5); swapping the upsample3d
    // even/odd interleave moves the decode probes by 4.3e-4 while every
    // chunk-plan-invariance test stays green — exactly the class of
    // wrong-but-consistent port this fixture exists to catch.
    // ---------------------------------------------------------------------

    /// The capture script's deterministic parameter fill, keyed on the
    /// checkpoint tensor name (identical in both layouts):
    /// gammas `1 + 0.25*sin(0.5*i + off)`, biases `0.05*sin(0.9*i + off)`,
    /// weights `0.015*sin(0.7*i + off)`, with
    /// `off(name) = (sum of utf8 bytes % 97) * 0.1`, row-major, f64 -> f32.
    fn synth_offset(name: &str) -> f64 {
        (name.bytes().map(u64::from).sum::<u64>() % 97) as f64 * 0.1
    }

    fn synth_param(name: &str, count: usize) -> Vec<f32> {
        let off = synth_offset(name);
        (0..count)
            .map(|i| {
                let i = i as f64;
                let v = if name.ends_with("gamma") {
                    1.0 + 0.25 * (0.5 * i + off).sin()
                } else if name.ends_with("bias") {
                    0.05 * (0.9 * i + off).sin()
                } else {
                    0.015 * (0.7 * i + off).sin()
                };
                v as f32
            })
            .collect()
    }

    /// `scale * sin(freq * i + off(name))`, the script's input synthesis.
    fn synth_input(name: &str, shape: &[usize], scale: f64, freq: f64) -> Tensor {
        let off = synth_offset(name);
        let count: usize = shape.iter().product();
        let values: Vec<f32> = (0..count)
            .map(|i| (scale * (freq * i as f64 + off).sin()) as f32)
            .collect();
        Tensor::from_vec(values, shape, &Device::Cpu).unwrap()
    }

    /// Materialize the tiny config once through a VarMap to learn the exact
    /// (name, shape) set mold's loader reads, then rebuild from synthesized
    /// tensors under those names. The Python side fills upstream's
    /// `state_dict()` the same way, so any key-layout divergence between the
    /// two ports shows up as parity error, not silence.
    fn golden_vae(config: WanVaeConfig) -> WanVideoVae {
        let device = Device::Cpu;
        let (_probe, varmap) = build(config.clone());
        let mut tensors = std::collections::HashMap::new();
        for (name, var) in varmap.data().lock().unwrap().iter() {
            let dims = var.dims().to_vec();
            let count: usize = dims.iter().product();
            let tensor = Tensor::from_vec(synth_param(name, count), dims, &device).unwrap();
            tensors.insert(name.clone(), tensor);
        }
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        WanVideoVae::from_var_builder(vb, config, &device, DType::F32)
            .expect("golden Wan VAE constructs from synthesized tensors")
    }

    /// Goldens carry the reference's f64 precision; mold runs in f32, so
    /// widening our values to compare is the honest direction.
    fn assert_matches_golden(got: &[f32], want: &[f64], tolerance: f64, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length mismatch");
        let mut worst = 0.0f64;
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let err = (f64::from(*g) - w).abs();
            assert!(
                err < tolerance,
                "{what}[{i}]: got {g}, want {w}, err {err:e}"
            );
            worst = worst.max(err);
        }
        assert!(worst < tolerance, "{what}: worst error {worst:e}");
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// Per-(channel, frame) means, row-major over (c, f) — the reference's
    /// `dec.mean(dim=(0, 3, 4))`.
    fn frame_channel_means(t: &Tensor) -> Vec<f32> {
        flat(&t.mean(4).unwrap().mean(3).unwrap())
    }

    fn probe(values: &[f32], stride: usize) -> Vec<f32> {
        values.iter().copied().step_by(stride).collect()
    }

    /// Observed agreement with the f64 reference is 7.5e-7 at worst (f32
    /// accumulation through the full streamed decoder). 5e-6 keeps ~7x
    /// headroom over that noise while sitting 7x below the weakest mutation
    /// signal observed during capture (3.5e-5).
    const VAE_GOLDEN_TOLERANCE: f64 = 5e-6;

    #[test]
    fn wan21_encode_matches_the_upstream_golden() {
        let vae = golden_vae(WanVaeConfig::tiny_v2_1());
        let video = synth_input("encode.video.v21", &[1, 3, 9, 16, 16], 0.9, 0.31);
        let latents = vae.encode(&video).unwrap();
        assert_eq!(latents.dims(), &[1, 4, 3, 2, 2]);
        assert_matches_golden(
            &flat(&latents),
            GOLDEN_V21_ENCODE,
            VAE_GOLDEN_TOLERANCE,
            "wan2.1 encode",
        );
    }

    #[test]
    fn wan21_decode_matches_the_upstream_golden() {
        let vae = golden_vae(WanVaeConfig::tiny_v2_1());
        let latents = synth_input("decode.latents.v21", &[1, 4, 3, 2, 2], 0.8, 0.53);
        let decoded = vae.decode(&latents).unwrap();
        assert_eq!(decoded.dims(), &[1, 3, 9, 16, 16]);
        // The fixture keeps |pixels| < 0.98 so mold's clamp is the identity
        // and parity stays visible at every probe.
        let values = flat(&decoded);
        assert_matches_golden(
            &probe(&values, V21_DECODE_PROBE_STRIDE),
            GOLDEN_V21_DECODE_PROBES,
            VAE_GOLDEN_TOLERANCE,
            "wan2.1 decode probes",
        );
        assert_matches_golden(
            &frame_channel_means(&decoded),
            GOLDEN_V21_DECODE_MEANS,
            VAE_GOLDEN_TOLERANCE,
            "wan2.1 decode means",
        );
    }

    #[test]
    fn wan22_encode_matches_the_upstream_golden() {
        let vae = golden_vae(WanVaeConfig::tiny_v2_2());
        let video = synth_input("encode.video.v22", &[1, 3, 9, 32, 32], 0.9, 0.31);
        let latents = vae.encode(&video).unwrap();
        assert_eq!(latents.dims(), &[1, 4, 3, 2, 2]);
        assert_matches_golden(
            &flat(&latents),
            GOLDEN_V22_ENCODE,
            VAE_GOLDEN_TOLERANCE,
            "wan2.2 encode",
        );
    }

    #[test]
    fn wan22_decode_matches_the_upstream_golden() {
        let vae = golden_vae(WanVaeConfig::tiny_v2_2());
        let latents = synth_input("decode.latents.v22", &[1, 4, 3, 2, 2], 0.8, 0.53);
        let decoded = vae.decode(&latents).unwrap();
        assert_eq!(decoded.dims(), &[1, 3, 9, 32, 32]);
        let values = flat(&decoded);
        assert_matches_golden(
            &probe(&values, V22_DECODE_PROBE_STRIDE),
            GOLDEN_V22_DECODE_PROBES,
            VAE_GOLDEN_TOLERANCE,
            "wan2.2 decode probes",
        );
        assert_matches_golden(
            &frame_channel_means(&decoded),
            GOLDEN_V22_DECODE_MEANS,
            VAE_GOLDEN_TOLERANCE,
            "wan2.2 decode means",
        );
    }
}
