//! Exact row-banded 3x3 convolution for the Qwen Image 2.1 VAE.
//!
//! Candle's im2col convolution materializes a `B * H_out * W_out` by
//! `C_in * k * k` column buffer before its single GEMM. The 2.1 decoder's
//! last stages run 288- and 144-channel 3x3 convolutions at full output
//! resolution, so that buffer is 5.4 GB at 1024² and 21.9 GB at 2752x1536 in
//! BF16 — the VAE decode, not the transformer, is what stops a 2K render on a
//! 46 GB card whenever the convolution runs on im2col (a build without
//! `cudnn`, or `MOLD_CONV=im2col`).
//!
//! [`BandedConv2d`] keeps the arithmetic and bounds the buffer: it pads the
//! input ONCE with the convolution's own zero padding, then convolves
//! horizontal bands of output rows, each band reading its rows plus a halo of
//! `kernel - 1` input rows, with padding 0. Every column-buffer entry is the
//! same value the unbanded im2col writes (an out-of-range tap is a zero either
//! way) and every output element is the same dot product over the same `K`, so
//! on CPU the result is bit-identical (pinned below). On CUDA it is the same
//! mathematics but NOT always the same bits: cuBLAS picks its GEMM algorithm
//! (tile, split-K) from the problem shape, and a band has a smaller `M`.
//! Measured on an L40S at the decoder's 288->144 shape: two bands agreed bit
//! for bit, three did not (the CUDA test below pins a rounding-level bound
//! instead). It is internal to the convolution, which is why the family's
//! `tiled_vae` capability stays `Unsupported`: nothing about the decode is
//! tiled, overlapped or blended.
//!
//! Because banding can move bits on CUDA, it must never touch a canvas v0.32
//! could render, or `MOLD_ATTN=math MOLD_CONV=im2col` would stop reproducing
//! archived bytes. [`BandScope`] carries that decision from the decode, which
//! knows the canvas, down to every convolution: a decode whose canvas fits
//! inside [`V032_MAX_PIXELS`] never bands; a larger (2K) canvas bands every
//! column buffer above [`BAND_COLUMN_BYTES`]. Outside a scope (the VAE
//! encoder, which v0.32 did not have) the 2 GiB limit applies directly.
//! Banding is also confined to CUDA with cuDNN not in effect — under cuDNN the
//! fork dispatches these shapes to cuDNN, which keeps no column buffer — so
//! Metal and CPU keep the plain convolution and their bytes never move.

use anyhow::Result;
use candle_core::{Device, Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder};
use std::cell::Cell;

/// Largest im2col column buffer a single convolution may allocate before it
/// is split into row bands.
pub(crate) const BAND_COLUMN_BYTES: u64 = 2 << 30;

/// The largest canvas v0.32 admitted for this family
/// (`mold_core::validation::MAX_PIXELS` before the 2K presets). Every decode
/// at or below it runs unbanded, byte-for-byte what v0.32 ran.
pub(crate) const V032_MAX_PIXELS: u64 = 1_800_000;

/// How a scoped decode lets its convolutions band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BandPolicy {
    /// No decode scope is active: band above [`BAND_COLUMN_BYTES`].
    Default,
    /// Never band (a v0.32-renderable canvas).
    Never,
    /// Band above this many column bytes.
    Above(u64),
}

thread_local! {
    static BAND_POLICY: Cell<BandPolicy> = const { Cell::new(BandPolicy::Default) };
}

/// Applies a decode's banding decision to every [`BandedConv2d`] on this
/// thread while alive, restoring the previous one on drop (the
/// `conv_policy::ConvScope` shape, so an error return cannot leak it).
#[must_use = "the band policy is only in effect while the scope is alive"]
pub(crate) struct BandScope {
    previous: BandPolicy,
}

impl BandScope {
    /// The decision for a decode producing `pixels` output pixels.
    pub(crate) fn for_canvas(pixels: u64) -> Self {
        Self::apply(if pixels <= V032_MAX_PIXELS {
            BandPolicy::Never
        } else {
            BandPolicy::Above(BAND_COLUMN_BYTES)
        })
    }

    /// Band above an explicit limit (tests use a lowered one).
    #[cfg(test)]
    pub(crate) fn above(limit: u64) -> Self {
        Self::apply(BandPolicy::Above(limit))
    }

    fn apply(policy: BandPolicy) -> Self {
        Self {
            previous: BAND_POLICY.with(|cell| cell.replace(policy)),
        }
    }
}

impl Drop for BandScope {
    fn drop(&mut self) {
        BAND_POLICY.with(|cell| cell.set(self.previous));
    }
}

/// A `candle_nn::Conv2d` whose forward is banded when its im2col column
/// buffer would exceed the limit in effect (see [`BandScope`]).
#[derive(Debug, Clone)]
pub(crate) struct BandedConv2d {
    conv: Conv2d,
}

/// `candle_nn::conv2d`, returning the banded wrapper.
pub(crate) fn banded_conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv2dConfig,
    vb: VarBuilder<'_>,
) -> Result<BandedConv2d> {
    Ok(BandedConv2d {
        conv: candle_nn::conv2d(in_channels, out_channels, kernel_size, cfg, vb)?,
    })
}

impl BandedConv2d {
    /// Whether this convolution's shape can be banded exactly: a square
    /// kernel larger than 1x1 with unit stride, unit dilation and one group.
    /// Anything else always runs whole.
    fn bandable(&self) -> bool {
        let cfg = self.conv.config();
        let (_, _, k_h, k_w) = self.conv.weight().dims4().unwrap_or((0, 0, 0, 0));
        cfg.stride == 1 && cfg.dilation == 1 && cfg.groups == 1 && k_h > 1 && k_h == k_w
    }

    /// Bytes of the im2col column buffer this convolution would allocate for
    /// `xs`.
    pub(crate) fn column_bytes(&self, xs: &Tensor) -> Result<u64> {
        let (batch, c_in, height, width) = xs.dims4()?;
        let (_, _, k_h, k_w) = self.conv.weight().dims4()?;
        let cfg = self.conv.config();
        let out_h = (height + 2 * cfg.padding).saturating_sub(k_h) / cfg.stride.max(1) + 1;
        let out_w = (width + 2 * cfg.padding).saturating_sub(k_w) / cfg.stride.max(1) + 1;
        Ok((batch * out_h * out_w * c_in * k_h * k_w * xs.dtype().size_in_bytes()) as u64)
    }

    /// The column-buffer limit on `device` under the convolution backend in
    /// effect on this thread, or `None` when the convolution never bands.
    fn band_limit(device: &Device) -> Option<u64> {
        let cudnn_in_effect =
            candle_core::cudnn_policy::is_compiled() && candle_core::cudnn_policy::is_enabled();
        if !device.is_cuda() || cudnn_in_effect {
            return None;
        }
        match BAND_POLICY.with(Cell::get) {
            BandPolicy::Default => Some(BAND_COLUMN_BYTES),
            BandPolicy::Never => None,
            BandPolicy::Above(limit) => Some(limit),
        }
    }

    /// Forward with an explicit column-buffer limit (`None` = never band).
    pub(crate) fn forward_with_limit(&self, xs: &Tensor, limit: Option<u64>) -> Result<Tensor> {
        let Some(limit) = limit else {
            return Ok(self.conv.forward(xs)?);
        };
        if !self.bandable() || self.column_bytes(xs)? <= limit {
            return Ok(self.conv.forward(xs)?);
        }
        let (batch, c_in, _, width) = xs.dims4()?;
        let (_, _, kernel, _) = self.conv.weight().dims4()?;
        let padding = self.conv.config().padding;
        let padded = xs
            .pad_with_zeros(2, padding, padding)?
            .pad_with_zeros(3, padding, padding)?;
        let out_h = padded.dim(2)? + 1 - kernel;
        let out_w = width + 2 * padding + 1 - kernel;
        let row_bytes =
            (batch * out_w * c_in * kernel * kernel * xs.dtype().size_in_bytes()) as u64;
        let band_rows = ((limit / row_bytes.max(1)) as usize).clamp(1, out_h);
        let weight = self.conv.weight();
        let bias = self
            .conv
            .bias()
            .map(|bias| bias.reshape((1, bias.dim(0)?, 1, 1)))
            .transpose()?;
        let mut bands = Vec::with_capacity(out_h.div_ceil(band_rows));
        let mut start = 0;
        while start < out_h {
            let rows = band_rows.min(out_h - start);
            // Output rows [start, start + rows) read padded input rows
            // [start, start + rows + kernel - 1).
            let band = padded
                .narrow(2, start, rows + kernel - 1)?
                .contiguous()?
                .conv2d(weight, 0, 1, 1, 1)?;
            bands.push(match &bias {
                Some(bias) => band.broadcast_add(bias)?,
                None => band,
            });
            start += rows;
        }
        Ok(Tensor::cat(&bands, 2)?)
    }
}

impl Module for BandedConv2d {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.forward_with_limit(xs, Self::band_limit(xs.device()))
            .map_err(candle_core::Error::wrap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::VarMap;

    fn conv(c_in: usize, c_out: usize, device: &Device, dtype: DType) -> BandedConv2d {
        let map = VarMap::new();
        let vb = VarBuilder::from_varmap(&map, dtype, device);
        banded_conv2d(
            c_in,
            c_out,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb,
        )
        .unwrap()
    }

    fn bits(t: &Tensor) -> Vec<u32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    /// Every band size — single rows, an uneven split, two-plus-remainder,
    /// and one band short of the whole — reproduces the whole convolution
    /// bit for bit.
    #[test]
    fn banded_convolution_is_bitwise_the_whole_convolution() {
        let device = Device::Cpu;
        let layer = conv(5, 7, &device, DType::F32);
        let xs = Tensor::randn(0f32, 1.0, (2, 5, 13, 11), &device).unwrap();
        let whole = layer.forward_with_limit(&xs, None).unwrap();
        let row = layer.column_bytes(&xs).unwrap() / 13;
        // 13 output rows: bands of 1, 3 (uneven), 6 (two plus one) and 12.
        for rows in [1u64, 3, 6, 12] {
            let banded = layer.forward_with_limit(&xs, Some(row * rows)).unwrap();
            assert_eq!(banded.dims(), whole.dims());
            assert_eq!(bits(&banded), bits(&whole), "band of {rows} rows");
        }
    }

    #[test]
    fn a_buffer_under_the_limit_is_never_banded() {
        let device = Device::Cpu;
        let layer = conv(3, 4, &device, DType::F32);
        let xs = Tensor::randn(0f32, 1.0, (1, 3, 8, 8), &device).unwrap();
        let bytes = layer.column_bytes(&xs).unwrap();
        assert_eq!(bytes, (8 * 8 * 3 * 9 * 4) as u64);
        let whole = layer.forward_with_limit(&xs, None).unwrap();
        let at_limit = layer.forward_with_limit(&xs, Some(bytes)).unwrap();
        assert_eq!(bits(&at_limit), bits(&whole));
    }

    /// A 1x1 convolution has no halo and no column-buffer blow-up worth
    /// banding; it always runs whole.
    #[test]
    fn pointwise_convolutions_never_band() {
        let device = Device::Cpu;
        let map = VarMap::new();
        let vb = VarBuilder::from_varmap(&map, DType::F32, &device);
        let layer = banded_conv2d(4, 4, 1, Default::default(), vb).unwrap();
        assert!(!layer.bandable());
    }

    #[test]
    fn only_cuda_without_cudnn_bands() {
        assert_eq!(BandedConv2d::band_limit(&Device::Cpu), None);
        let _scope = BandScope::above(1);
        assert_eq!(BandedConv2d::band_limit(&Device::Cpu), None);
    }

    /// Every canvas v0.32 could render decodes unbanded; a larger one bands
    /// at 2 GiB; the scope restores what was in effect, even nested.
    #[test]
    fn the_band_scope_keeps_every_v032_canvas_unbanded() {
        let policy = || BAND_POLICY.with(Cell::get);
        assert_eq!(policy(), BandPolicy::Default);
        {
            let _v032 = BandScope::for_canvas(1344 * 768);
            assert_eq!(policy(), BandPolicy::Never);
            {
                let _two_k = BandScope::for_canvas(2752 * 1536);
                assert_eq!(policy(), BandPolicy::Above(BAND_COLUMN_BYTES));
            }
            assert_eq!(policy(), BandPolicy::Never);
        }
        assert_eq!(policy(), BandPolicy::Default);
        let _edge = BandScope::for_canvas(V032_MAX_PIXELS);
        assert_eq!(policy(), BandPolicy::Never);
    }

    /// The production shapes on CUDA: the decoder's full-resolution 288- and
    /// 144-channel convolutions in BF16, banded under a lowered limit, against
    /// the whole convolution. cuBLAS may pick a different GEMM algorithm for
    /// a band's smaller `M`, so the pin is a BF16 rounding bound, not bit
    /// equality (measured on an L40S: two bands were bitwise, three were
    /// not) — which is exactly why `BandScope` keeps v0.32 canvases unbanded.
    /// Skips without a CUDA device (CI has none).
    #[cfg(feature = "cuda")]
    #[test]
    fn banded_convolution_matches_the_whole_convolution_on_cuda_at_vae_shapes() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let _im2col = crate::conv_policy::ConvScope::apply(crate::conv_policy::ConvBackend::Im2Col);
        for (c_in, c_out, height, width) in [(288, 144, 256, 384), (144, 144, 320, 192)] {
            let layer = conv(c_in, c_out, &device, DType::BF16);
            let xs = Tensor::randn(0f32, 1.0, (1, c_in, height, width), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let whole = layer
                .forward_with_limit(&xs, None)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let scale = whole
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let total = layer.column_bytes(&xs).unwrap();
            for parts in [2u64, 3, 7] {
                let banded = layer
                    .forward_with_limit(&xs, Some(total / parts))
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap();
                assert_eq!(banded.dims(), whole.dims());
                let diff = (&banded - &whole)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .max_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();
                // One BF16 ulp of the output's magnitude.
                assert!(
                    diff <= scale / 128.0,
                    "{c_in}->{c_out} at {height}x{width}, {parts} bands: {diff} vs {scale}"
                );
            }
        }
    }
}
