//! Reductions on shapes candle's Metal backend gets right.
//!
//! candle's Metal reduction returns wrong values when a tensor of rank > 3 is
//! reduced over a non-final dimension. Reducing `[1, 8, 3, 4, 4]` over dim 1
//! yields `0.21675` on Metal where both the CPU and a rank-3 view of the very
//! same bytes yield `2.60200`. `sum` is wrong the same way, contiguous or not,
//! so it is the reduction itself rather than a stride or an averaging step.
//! Rank <= 3 and last-dimension reductions are correct.
//!
//! The defect is in `reduce.metal`'s strided indexer: the rank-5/6 arms of
//! `get_strided_index_t` are commented out upstream, and the fallback loop
//! peels dim 0 where it should peel the trailing dims, so dim 0 is counted
//! twice and the highest dim never. LTX-2 hit it first (#1011) — a rank-5
//! channel norm made every Metal render a black clip while reporting success,
//! and a rank-4 per-token RMS rescaled the prompt embedding — and Wan hits the
//! identical pattern in its VAE (`WanRmsNorm` reduces the channel axis of a
//! rank-5 activation; `AvgDown3d` means over dim 2 of a rank-6 tensor).
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

/// Sum over `dim`, keeping rank, on shapes Metal reduces correctly.
pub(crate) fn sum_keepdim_stable(x: &Tensor, dim: usize) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    if dims.len() <= 3 || dim + 1 == dims.len() {
        return x.sum_keepdim(dim);
    }
    let lead: usize = dims[..dim].iter().product();
    let mid = dims[dim];
    let trail: usize = dims[dim + 1..].iter().product();
    let reduced = x
        .contiguous()?
        .reshape((lead, mid, trail))?
        .sum_keepdim(1)?;
    let mut kept = dims;
    kept[dim] = 1;
    reduced.reshape(kept)
}

/// Mean over `dim`, dropping it, on shapes Metal reduces correctly.
pub(crate) fn mean_stable(x: &Tensor, dim: usize) -> Result<Tensor> {
    mean_keepdim_stable(x, dim)?.squeeze(dim)
}

#[cfg(test)]
mod tests {
    use super::{mean_keepdim_stable, mean_stable, sum_keepdim_stable};
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

    fn assert_close(folded: &[f32], direct: &[f32], context: &str) {
        assert_eq!(folded.len(), direct.len(), "{context}");
        for (a, b) in folded.iter().zip(direct.iter()) {
            assert!((a - b).abs() <= 1e-5 * a.abs().max(1.0), "{context}");
        }
    }

    /// Folding must not change the answer on a backend that was already right.
    #[test]
    fn folding_preserves_the_cpu_result() {
        let device = Device::Cpu;
        for dims in [
            vec![2usize, 8, 3, 4, 4],
            vec![2usize, 8, 3, 4],
            vec![2usize, 8, 3],
            vec![2usize, 4, 2, 3, 4, 4],
        ] {
            let tensor = sample(&device, &dims);
            for dim in 0..dims.len() {
                let context = format!("{dims:?} dim {dim}");
                assert_close(
                    &values(&mean_keepdim_stable(&tensor, dim).unwrap()),
                    &values(&tensor.mean_keepdim(dim).unwrap()),
                    &context,
                );
                assert_close(
                    &values(&sum_keepdim_stable(&tensor, dim).unwrap()),
                    &values(&tensor.sum_keepdim(dim).unwrap()),
                    &context,
                );
                assert_close(
                    &values(&mean_stable(&tensor, dim).unwrap()),
                    &values(&tensor.mean(dim).unwrap()),
                    &context,
                );
            }
        }
    }

    /// `mean_stable` must drop the reduced dim like `Tensor::mean` does.
    #[test]
    fn mean_stable_drops_the_reduced_dim() {
        let tensor = sample(&Device::Cpu, &[2, 4, 2, 3, 4, 4]);
        assert_eq!(
            mean_stable(&tensor, 2).unwrap().dims(),
            &[2usize, 4, 3, 4, 4]
        );
    }

    /// The shapes LTX-2 and Wan actually reduce, checked against the CPU
    /// reference.
    ///
    /// Rank-5 dim 1 is both video decoders' channel norm (LTX-2's
    /// `PerChannelRmsNorm`, Wan's `WanRmsNorm`); rank-4 dim 2 is LTX-2's
    /// per-token text RMS; rank-6 dim 2 is Wan's `AvgDown3d` group mean. All
    /// returned wrong values through the direct reductions on Metal.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_matches_the_cpu_on_the_shapes_the_engines_reduce() {
        let metal = Device::new_metal(0).unwrap();
        for (dims, dim) in [
            (vec![2usize, 8, 3, 4, 4], 1usize),
            (vec![2usize, 6, 8, 4], 2usize),
            (vec![2usize, 8, 3, 4, 4], 4usize),
            (vec![1usize, 4, 4, 3, 4, 4], 2usize),
        ] {
            let context = format!("{dims:?} dim {dim}");
            let cpu = sample(&Device::Cpu, &dims);
            let gpu = sample(&metal, &dims);
            assert_close(
                &values(&mean_keepdim_stable(&cpu, dim).unwrap()),
                &values(&mean_keepdim_stable(&gpu, dim).unwrap()),
                &context,
            );
            assert_close(
                &values(&sum_keepdim_stable(&cpu, dim).unwrap()),
                &values(&sum_keepdim_stable(&gpu, dim).unwrap()),
                &context,
            );
            assert_close(
                &values(&mean_stable(&cpu, dim).unwrap()),
                &values(&mean_stable(&gpu, dim).unwrap()),
                &context,
            );
        }
    }

    /// Every value must stay finite: a negative mean-of-squares is what made
    /// the decoders' `sqrt` return NaN and blacken the frame.
    #[cfg(feature = "metal")]
    #[test]
    fn a_mean_of_squares_stays_non_negative_on_metal() {
        let metal = Device::new_metal(0).unwrap();
        for reduced in [
            mean_keepdim_stable(&sample(&metal, &[2, 8, 3, 4, 4]), 1).unwrap(),
            sum_keepdim_stable(&sample(&metal, &[2, 8, 3, 4, 4]), 1).unwrap(),
        ] {
            for value in values(&reduced) {
                assert!(
                    value.is_finite() && value >= 0.0,
                    "reduction of squares must stay finite and non-negative, got {value}"
                );
            }
        }
    }
}
