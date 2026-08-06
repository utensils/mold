//! Wan's 3-D rotary position embedding.
//!
//! The head dimension is split across three axes — time, height, width — with
//! widths `[d - 4*(d//6), 2*(d//6), 2*(d//6)]`, so at the production head_dim
//! of 128 that is 44/42/42 (`Wan2.1/wan/modules/model.py:480-484`). Each axis
//! gets its own geometric frequency ladder over its own sub-dimension, and the
//! three tables are concatenated along the feature axis.
//!
//! Token order is `(frame, height, width)` row-major, matching the order
//! patchify flattens its output in — the two must agree or every position is
//! rotated by the wrong angle.

use candle_core::{bail, DType, Device, IndexOp, Result, Tensor};

/// RoPE base. Wan uses 10000 on every axis (`model.py:31-37`).
pub(crate) const WAN_ROPE_THETA: f64 = 10000.0;

/// Precomputed per-axis rotation tables.
///
/// Diffusers materializes `cos`/`sin` at the full head width by
/// `repeat_interleave(2)`, then immediately slices `freqs_cos[..., 0::2]` and
/// `freqs_sin[..., 1::2]` back out when applying them
/// (`transformer_wan.py:104-117`). Those two slices are exactly the
/// un-repeated tables, so this stores the half-width form and skips the
/// round trip. The slicing is only valid because every axis width is even, so
/// no rotation pair straddles an axis boundary — [`WanRope::new`] enforces it.
#[derive(Debug, Clone)]
pub(crate) struct WanRope {
    head_dim: usize,
    max_seq_len: usize,
    /// Full per-axis widths in `(time, height, width)` order.
    axis_dims: [usize; 3],
    /// `cos(pos * freq)` per axis, `[max_seq_len, axis_dim / 2]` row-major.
    cos: [Vec<f64>; 3],
    sin: [Vec<f64>; 3],
}

impl WanRope {
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f64) -> Result<Self> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            bail!("Wan RoPE: head dimension must be positive and even, got {head_dim}");
        }
        if max_seq_len == 0 {
            bail!("Wan RoPE: max_seq_len must be positive");
        }
        let spatial = 2 * (head_dim / 6);
        let temporal = head_dim
            .checked_sub(2 * spatial)
            .filter(|t| *t > 0)
            .ok_or_else(|| {
                candle_core::Error::msg(format!(
                    "Wan RoPE: head dimension {head_dim} leaves no room for the temporal axis"
                ))
            })?;
        if !temporal.is_multiple_of(2) || !spatial.is_multiple_of(2) {
            bail!(
                "Wan RoPE: every axis width must be even so rotation pairs stay inside one axis, \
                 got {temporal}/{spatial}/{spatial}"
            );
        }
        let axis_dims = [temporal, spatial, spatial];

        let table = |dim: usize| -> (Vec<f64>, Vec<f64>) {
            let half = dim / 2;
            let mut cos = Vec::with_capacity(max_seq_len * half);
            let mut sin = Vec::with_capacity(max_seq_len * half);
            for pos in 0..max_seq_len {
                for i in 0..half {
                    // `1 / theta^(2i / dim)` — the exponent walks `arange(0, dim, 2)`.
                    let freq = 1.0 / theta.powf((2 * i) as f64 / dim as f64);
                    let angle = pos as f64 * freq;
                    cos.push(angle.cos());
                    sin.push(angle.sin());
                }
            }
            (cos, sin)
        };
        let (cos_t, sin_t) = table(axis_dims[0]);
        let (cos_h, sin_h) = table(axis_dims[1]);
        let (cos_w, sin_w) = table(axis_dims[2]);

        Ok(Self {
            head_dim,
            max_seq_len,
            axis_dims,
            cos: [cos_t, cos_h, cos_w],
            sin: [sin_t, sin_h, sin_w],
        })
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Per-axis widths, `(time, height, width)`.
    pub fn axis_dims(&self) -> [usize; 3] {
        self.axis_dims
    }

    /// Rotation tables for a `frames x height x width` patch grid, laid out in
    /// the same `(f, h, w)` row-major order patchify produces.
    ///
    /// Returns `(cos, sin)`, both `[frames * height * width, head_dim / 2]` in
    /// F32 on `device`.
    pub fn freqs(
        &self,
        frames: usize,
        height: usize,
        width: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        for (axis, len) in ["frames", "height", "width"]
            .iter()
            .zip([frames, height, width])
        {
            if len > self.max_seq_len {
                bail!(
                    "Wan RoPE: {axis} extent {len} exceeds the precomputed {} positions",
                    self.max_seq_len
                );
            }
        }
        let halves = self.axis_dims.map(|d| d / 2);
        let row = halves.iter().sum::<usize>();
        let tokens = frames * height * width;
        let mut cos = Vec::with_capacity(tokens * row);
        let mut sin = Vec::with_capacity(tokens * row);
        for f in 0..frames {
            for h in 0..height {
                for w in 0..width {
                    for (axis, pos) in [f, h, w].into_iter().enumerate() {
                        let half = halves[axis];
                        let start = pos * half;
                        let slice = start..start + half;
                        cos.extend(self.cos[axis][slice.clone()].iter().map(|v| *v as f32));
                        sin.extend(self.sin[axis][slice].iter().map(|v| *v as f32));
                    }
                }
            }
        }
        Ok((
            Tensor::from_vec(cos, (tokens, row), device)?,
            Tensor::from_vec(sin, (tokens, row), device)?,
        ))
    }
}

/// Rotate `x` in place along its head dimension.
///
/// `x` is `[batch, tokens, heads, head_dim]`; `cos`/`sin` are
/// `[tokens, head_dim / 2]`. Pairs are adjacent, so
/// `out[2k] = x[2k]*cos - x[2k+1]*sin` and
/// `out[2k+1] = x[2k]*sin + x[2k+1]*cos` (`transformer_wan.py:111-116`).
/// Always evaluated in F32 and cast back, as diffusers does.
pub(crate) fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (batch, tokens, heads, head_dim) = x.dims4()?;
    let (cos_tokens, half) = cos.dims2()?;
    if cos_tokens != tokens || half * 2 != head_dim {
        bail!(
            "Wan RoPE: tables are [{cos_tokens}, {half}] but the tensor is \
             [{batch}, {tokens}, {heads}, {head_dim}]"
        );
    }
    if cos.dims() != sin.dims() {
        bail!("Wan RoPE: cos and sin tables disagree in shape");
    }

    let dtype = x.dtype();
    let pairs = x
        .to_dtype(DType::F32)?
        .reshape((batch, tokens, heads, half, 2))?;
    let x1 = pairs.i((.., .., .., .., 0))?.contiguous()?;
    let x2 = pairs.i((.., .., .., .., 1))?.contiguous()?;
    let cos = cos.to_dtype(DType::F32)?.reshape((1, tokens, 1, half))?;
    let sin = sin.to_dtype(DType::F32)?.reshape((1, tokens, 1, half))?;

    let even = x1.broadcast_mul(&cos)?.sub(&x2.broadcast_mul(&sin)?)?;
    let odd = x1.broadcast_mul(&sin)?.add(&x2.broadcast_mul(&cos)?)?;
    Tensor::stack(&[&even, &odd], 4)?
        .reshape((batch, tokens, heads, head_dim))?
        .to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_split_matches_upstream() {
        let rope = WanRope::new(128, 1024, WAN_ROPE_THETA).unwrap();
        assert_eq!(rope.axis_dims(), [44, 42, 42]);
        assert_eq!(rope.axis_dims().iter().sum::<usize>(), 128);

        // The tiny golden config.
        let rope = WanRope::new(8, 32, WAN_ROPE_THETA).unwrap();
        assert_eq!(rope.axis_dims(), [4, 2, 2]);
    }

    #[test]
    fn rejects_degenerate_head_dims() {
        assert!(WanRope::new(0, 32, WAN_ROPE_THETA).is_err());
        assert!(WanRope::new(7, 32, WAN_ROPE_THETA).is_err());
        assert!(WanRope::new(128, 0, WAN_ROPE_THETA).is_err());
        // head_dim 4 gives spatial 0, which would leave the h/w axes unrotated
        // rather than erroring; head_dim 2 leaves temporal 2 and spatial 0.
        assert!(WanRope::new(128, 8, WAN_ROPE_THETA)
            .unwrap()
            .freqs(9, 2, 2, &Device::Cpu)
            .is_err());
    }

    #[test]
    fn position_zero_is_the_identity_rotation() {
        let rope = WanRope::new(8, 32, WAN_ROPE_THETA).unwrap();
        let (cos, sin) = rope.freqs(1, 1, 1, &Device::Cpu).unwrap();
        assert_eq!(cos.dims(), &[1, 4]);
        let cos_v = cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let sin_v = sin.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(cos_v.iter().all(|c| (c - 1.0).abs() < 1e-7));
        assert!(sin_v.iter().all(|s| s.abs() < 1e-7));

        let x = Tensor::arange(0f32, 8f32, &Device::Cpu)
            .unwrap()
            .reshape((1, 1, 1, 8))
            .unwrap();
        let rotated = apply_rope(&x, &cos, &sin).unwrap();
        let a = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = rotated.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (l, r) in a.iter().zip(b.iter()) {
            assert!((l - r).abs() < 1e-6, "{l} vs {r}");
        }
    }

    /// The table for one token must be the three axes' half-width tables
    /// concatenated in `(t, h, w)` order.
    #[test]
    fn freqs_concatenate_the_axes_in_order() {
        let rope = WanRope::new(8, 32, WAN_ROPE_THETA).unwrap();
        let (cos, _) = rope.freqs(2, 2, 2, &Device::Cpu).unwrap();
        assert_eq!(cos.dims(), &[8, 4]);
        let v = cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Token 5 is (f=1, h=0, w=1) in row-major (f, h, w).
        let token = 5;
        let row = &v[token * 4..token * 4 + 4];
        // Temporal axis is 4 wide -> 2 pairs, at position f = 1.
        for (i, entry) in row.iter().take(2).enumerate() {
            let freq = 1.0f64 / WAN_ROPE_THETA.powf((2 * i) as f64 / 4.0);
            assert!((f64::from(*entry) - (1.0 * freq).cos()).abs() < 1e-6);
        }
        // Height axis is 2 wide -> 1 pair, at position h = 0.
        assert!((row[2] - 1.0).abs() < 1e-6);
        // Width axis is 2 wide -> 1 pair, at position w = 1, freq = 1.
        assert!((f64::from(row[3]) - 1.0f64.cos()).abs() < 1e-6);
    }

    /// Pin the rotation *direction* against the closed form.
    ///
    /// This cannot be delegated to the DiT golden fixture: its synthetic
    /// weights leave the pre-softmax logits near zero (~6e-4), so the softmax
    /// sits in its uniform regime and the attention output is effectively
    /// `mean(v)` regardless of how q and k were rotated. Reversing the rotation
    /// there moves the model output by less than 1e-8 — about as much as
    /// removing RoPE entirely. So the sign has to be nailed down here.
    #[test]
    fn rotation_direction_matches_the_closed_form() {
        let rope = WanRope::new(8, 32, WAN_ROPE_THETA).unwrap();
        // Token (f=0, h=0, w=1): the width pair rotates by exactly 1 radian,
        // every other pair by 0.
        let (cos, sin) = rope.freqs(1, 1, 2, &Device::Cpu).unwrap();
        let cos = cos.narrow(0, 1, 1).unwrap();
        let sin = sin.narrow(0, 1, 1).unwrap();

        // Unit vector on the first element of the width pair (indices 6, 7).
        let mut raw = vec![0f32; 8];
        raw[6] = 1.0;
        let x = Tensor::from_vec(raw, (1, 1, 1, 8), &Device::Cpu).unwrap();
        let out = apply_rope(&x, &cos, &sin).unwrap();
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // (1, 0) rotated by +1 rad is (cos 1, sin 1). The reverse rotation
        // would give (cos 1, -sin 1), so the sign of v[7] is the whole test.
        assert!(
            (f64::from(v[6]) - 1.0f64.cos()).abs() < 1e-6,
            "got {}",
            v[6]
        );
        assert!(
            (f64::from(v[7]) - 1.0f64.sin()).abs() < 1e-6,
            "got {}",
            v[7]
        );
        assert!(v[7] > 0.0, "rotation ran backwards");
        // The untouched pairs stay at zero.
        assert!(v[..6].iter().all(|c| c.abs() < 1e-7));

        // And the second element of the pair maps (0, 1) -> (-sin 1, cos 1).
        let mut raw = vec![0f32; 8];
        raw[7] = 1.0;
        let x = Tensor::from_vec(raw, (1, 1, 1, 8), &Device::Cpu).unwrap();
        let v = apply_rope(&x, &cos, &sin)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            (f64::from(v[6]) + 1.0f64.sin()).abs() < 1e-6,
            "got {}",
            v[6]
        );
        assert!(
            (f64::from(v[7]) - 1.0f64.cos()).abs() < 1e-6,
            "got {}",
            v[7]
        );
    }

    /// Rotation preserves the norm of each adjacent pair.
    #[test]
    fn rotation_is_norm_preserving() {
        let rope = WanRope::new(8, 32, WAN_ROPE_THETA).unwrap();
        let (cos, sin) = rope.freqs(2, 2, 2, &Device::Cpu).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 8, 2, 8), &Device::Cpu).unwrap();
        let y = apply_rope(&x, &cos, &sin).unwrap();
        let norm = |t: &Tensor| -> Vec<f32> {
            t.reshape((1, 8, 2, 4, 2))
                .unwrap()
                .sqr()
                .unwrap()
                .sum(4)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        for (a, b) in norm(&x).iter().zip(norm(&y).iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn apply_rope_rejects_mismatched_tables() {
        let rope = WanRope::new(8, 32, WAN_ROPE_THETA).unwrap();
        let (cos, sin) = rope.freqs(2, 2, 2, &Device::Cpu).unwrap();
        let wrong = Tensor::zeros((1, 4, 2, 8), DType::F32, &Device::Cpu).unwrap();
        assert!(apply_rope(&wrong, &cos, &sin).is_err());
    }
}
