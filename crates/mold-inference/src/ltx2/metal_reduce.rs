//! Reductions on shapes candle's Metal backend gets right.
//!
//! candle's Metal reduction returns wrong values when a tensor of rank > 3 is
//! reduced over a non-final dimension. Reducing `[1, 8, 3, 4, 4]` over dim 1
//! yields `0.21675` on Metal where both the CPU and a rank-3 view of the very
//! same bytes yield `2.60200`. `sum` is wrong the same way, contiguous or not,
//! so it is the reduction itself rather than a stride or an averaging step.
//! Rank <= 3 and last-dimension reductions are correct.
//!
//! LTX-2 hit this in two places, and together they made every Metal render
//! useless while reporting success:
//!
//! * the video decoder's final `PerChannelRmsNorm` reduces the channel axis of
//!   a rank-5 activation. The too-small mean-of-squares still reaches `sqrt`,
//!   and with real activations it goes negative, so `sqrt` returns NaN and the
//!   following `broadcast_div` spreads it over every pixel — a fully black clip
//!   from healthy latents, with no error raised anywhere.
//! * the text connector's `norm_and_concat_per_token_rms` reduces dim 2 of a
//!   rank-4 `[batch, seq, hidden, layers]` stack. That rescales the prompt
//!   embedding by a wrong factor, so the render is coherent and well-formed but
//!   ignores the prompt entirely.
//!
//! Folding to rank 3 around the reduced axis keeps the same elements in the
//! same groups while staying on a shape Metal reduces correctly. It is a no-op
//! reshape on contiguous data, so CUDA and CPU are unaffected.

use candle_core::{Result, Tensor};

/// Mean over `dim`, keeping rank, on shapes Metal reduces correctly.
pub(crate) fn mean_keepdim_stable(x: &Tensor, dim: usize) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    if dims.len() <= 3 || dim + 1 == dims.len() {
        return x.mean_keepdim(dim);
    }
    let lead: usize = dims[..dim].iter().product();
    let mid = dims[dim];
    let trail: usize = dims[dim + 1..].iter().product();
    let reduced = x
        .contiguous()?
        .reshape((lead, mid, trail))?
        .mean_keepdim(1)?;
    let mut kept = dims;
    kept[dim] = 1;
    reduced.reshape(kept)
}

#[cfg(test)]
mod tests {
    use super::mean_keepdim_stable;
    use candle_core::{Device, Tensor};

    fn sample(device: &Device, dims: &[usize]) -> Tensor {
        let count: usize = dims.iter().product();
        Tensor::arange(0f32, count as f32, device)
            .unwrap()
            .reshape(dims)
            .unwrap()
            .affine(0.01, -0.5)
            .unwrap()
            .sqr()
            .unwrap()
    }

    fn values(tensor: &Tensor) -> Vec<f32> {
        tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// Folding must not change the answer on a backend that was already right.
    #[test]
    fn folding_preserves_the_cpu_result() {
        let device = Device::Cpu;
        for dims in [
            vec![2usize, 8, 3, 4, 4],
            vec![2usize, 8, 3, 4],
            vec![2usize, 8, 3],
        ] {
            let tensor = sample(&device, &dims);
            for dim in 0..dims.len() {
                let folded = values(&mean_keepdim_stable(&tensor, dim).unwrap());
                let direct = values(&tensor.mean_keepdim(dim).unwrap());
                assert_eq!(folded.len(), direct.len(), "{dims:?} dim {dim}");
                for (a, b) in folded.iter().zip(direct.iter()) {
                    assert!(
                        (a - b).abs() <= 1e-5 * a.abs().max(1.0),
                        "{dims:?} dim {dim}"
                    );
                }
            }
        }
    }

    /// The shapes LTX-2 actually reduces, checked against the CPU reference.
    ///
    /// Rank-5 dim 1 is the video decoder's channel norm; rank-4 dim 2 is the
    /// text connector's per-token RMS. Both returned wrong values on Metal.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_matches_the_cpu_on_the_shapes_ltx2_reduces() {
        let metal = Device::new_metal(0).unwrap();
        for (dims, dim) in [
            (vec![2usize, 8, 3, 4, 4], 1usize),
            (vec![2usize, 6, 8, 4], 2usize),
            (vec![2usize, 8, 3, 4, 4], 4usize),
        ] {
            let expected = values(&mean_keepdim_stable(&sample(&Device::Cpu, &dims), dim).unwrap());
            let actual = values(&mean_keepdim_stable(&sample(&metal, &dims), dim).unwrap());
            assert_eq!(expected.len(), actual.len(), "{dims:?} dim {dim}");
            for (cpu, gpu) in expected.iter().zip(actual.iter()) {
                assert!(
                    (cpu - gpu).abs() <= 1e-4 * cpu.abs().max(1.0),
                    "{dims:?} dim {dim}: cpu {cpu} metal {gpu}"
                );
            }
        }
    }

    /// Every value must stay finite: a negative mean-of-squares is what made
    /// the decoder's `sqrt` return NaN and blacken the frame.
    #[cfg(feature = "metal")]
    #[test]
    fn a_mean_of_squares_stays_non_negative_on_metal() {
        let metal = Device::new_metal(0).unwrap();
        let reduced = mean_keepdim_stable(&sample(&metal, &[2, 8, 3, 4, 4]), 1).unwrap();
        for value in values(&reduced) {
            assert!(
                value.is_finite() && value >= 0.0,
                "mean of squares must stay finite and non-negative, got {value}"
            );
        }
    }
}
