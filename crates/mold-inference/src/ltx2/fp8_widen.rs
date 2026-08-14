//! Host-side widening of `F8E4M3` weights.
//!
//! candle's Metal backend has no `F8E4M3` cast kernel, so every FP8 weight an
//! LTX-2 forward pass touches is widened on the host and copied back. candle's
//! CPU path does that through a serial `unary_map` calling `F8E4M3::to_f32`
//! once per element.
//!
//! A 120-second stack sample of an LTX-2 Metal denoise on an M4 Max put
//! **47% of the worker thread's self time inside `F8E4M3::to_f32` alone**, and
//! 0.1% in the matmul it feeds — the transformer spends its render converting
//! one byte at a time. `F8E4M3` has 256 possible values, so that per-element
//! call is a table lookup.
//!
//! The tables are built from candle's own conversion expressions
//! (`candle-core/src/cpu_backend/mod.rs`, e.g. `bf16::from_f32(v.to_f32())`),
//! so the result is bit-identical **by construction** rather than by
//! approximation. `widening_matches_candles_own_conversion` pins that for all
//! 256 inputs; if candle changes its expression the test fails rather than the
//! renders drifting.

use candle_core::{DType, Device, Result, Tensor};
use float8::F8E4M3;
use half::{bf16, f16};
use rayon::prelude::*;
use std::sync::LazyLock;

/// Below this many elements the rayon split costs more than the conversion.
/// LTX-2's FP8 weight chunks are millions of elements, so this only keeps the
/// small biases and scales off the thread pool.
const PARALLEL_MIN_ELEMENTS: usize = 1 << 16;

/// Elements per rayon task. Large enough that each task amortises its own
/// scheduling, small enough that 12 performance cores all get work.
const PARALLEL_CHUNK: usize = 1 << 14;

macro_rules! widening_table {
    ($name:ident, $ty:ty, |$value:ident| $convert:expr) => {
        static $name: LazyLock<[$ty; 256]> = LazyLock::new(|| {
            let mut table = [<$ty as Default>::default(); 256];
            for (bits, slot) in table.iter_mut().enumerate() {
                let $value = F8E4M3::from_bits(bits as u8);
                *slot = $convert;
            }
            table
        });
    };
}

widening_table!(TABLE_F32, f32, |value| value.to_f32());
widening_table!(TABLE_BF16, bf16, |value| bf16::from_f32(value.to_f32()));
widening_table!(TABLE_F16, f16, |value| f16::from_f32(value.to_f32()));

fn widen_slice<T: Copy + Send + Sync>(src: &[F8E4M3], table: &[T; 256]) -> Vec<T> {
    if src.len() < PARALLEL_MIN_ELEMENTS {
        return src.iter().map(|v| table[v.to_bits() as usize]).collect();
    }
    src.par_iter()
        .with_min_len(PARALLEL_CHUNK)
        .map(|v| table[v.to_bits() as usize])
        .collect()
}

/// Widen an `F8E4M3` tensor to `dtype` and place the result on `device`.
///
/// Returns `Ok(None)` for a target dtype without a table, so the caller keeps
/// candle's generic path rather than this one silently becoming authoritative
/// for a conversion it does not implement.
pub(crate) fn widen_f8e4m3(
    weight: &Tensor,
    dtype: DType,
    device: &Device,
) -> Result<Option<Tensor>> {
    if weight.dtype() != DType::F8E4M3 {
        return Ok(None);
    }
    if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
        return Ok(None);
    }

    let host = weight.to_device(&Device::Cpu)?.contiguous()?;
    let shape = host.shape().clone();
    let src = host.flatten_all()?.to_vec1::<F8E4M3>()?;

    let widened = match dtype {
        DType::F32 => Tensor::from_vec(widen_slice(&src, &TABLE_F32), shape, device)?,
        DType::BF16 => Tensor::from_vec(widen_slice(&src, &TABLE_BF16), shape, device)?,
        DType::F16 => Tensor::from_vec(widen_slice(&src, &TABLE_F16), shape, device)?,
        _ => unreachable!("dtype guarded above"),
    };
    Ok(Some(widened))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn every_f8e4m3_value() -> Tensor {
        let values: Vec<F8E4M3> = (0..=u8::MAX).map(F8E4M3::from_bits).collect();
        Tensor::from_vec(values, 256, &Device::Cpu).expect("f8e4m3 tensor")
    }

    /// The tables must agree with candle bit for bit, NaN payloads included —
    /// these are model weights, so a differing low bit is a differing render.
    /// Compared as raw bits because `NaN != NaN` would pass a value check by
    /// accident.
    #[test]
    fn widening_matches_candles_own_conversion() {
        let source = every_f8e4m3_value();

        let ours = widen_f8e4m3(&source, DType::BF16, &Device::Cpu)
            .expect("widen")
            .expect("bf16 is a supported target");
        let theirs = source.to_dtype(DType::BF16).expect("candle bf16");
        assert_eq!(
            ours.to_vec1::<bf16>()
                .expect("ours")
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            theirs
                .to_vec1::<bf16>()
                .expect("theirs")
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
        );

        let ours = widen_f8e4m3(&source, DType::F16, &Device::Cpu)
            .expect("widen")
            .expect("f16 is a supported target");
        let theirs = source.to_dtype(DType::F16).expect("candle f16");
        assert_eq!(
            ours.to_vec1::<f16>()
                .expect("ours")
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            theirs
                .to_vec1::<f16>()
                .expect("theirs")
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
        );

        let ours = widen_f8e4m3(&source, DType::F32, &Device::Cpu)
            .expect("widen")
            .expect("f32 is a supported target");
        let theirs = source.to_dtype(DType::F32).expect("candle f32");
        assert_eq!(
            ours.to_vec1::<f32>()
                .expect("ours")
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            theirs
                .to_vec1::<f32>()
                .expect("theirs")
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
        );
    }

    /// The parallel branch has to produce the serial answer. The threshold is
    /// crossed here on purpose — a tensor below it never exercises rayon.
    #[test]
    fn the_parallel_branch_agrees_with_the_serial_one() {
        let repeats = PARALLEL_MIN_ELEMENTS.div_ceil(256) + 3;
        let values: Vec<F8E4M3> = (0..repeats)
            .flat_map(|_| (0..=u8::MAX).map(F8E4M3::from_bits))
            .collect();
        assert!(values.len() > PARALLEL_MIN_ELEMENTS);

        let parallel = widen_slice(&values, &TABLE_BF16);
        let serial: Vec<bf16> = values
            .iter()
            .map(|v| TABLE_BF16[v.to_bits() as usize])
            .collect();
        assert_eq!(
            parallel.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            serial.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        );
    }

    /// Shape and layout survive the round trip — the caller feeds the result
    /// straight into a transposed matmul, so a flattened or reordered result
    /// would render garbage rather than fail.
    #[test]
    fn widening_preserves_shape_for_a_narrowed_weight_chunk() {
        let values: Vec<F8E4M3> = (0..24).map(|i| F8E4M3::from_bits(i as u8 + 60)).collect();
        let weight = Tensor::from_vec(values, (4, 6), &Device::Cpu).expect("weight");
        let chunk = weight.narrow(0, 1, 2).expect("narrow");

        let ours = widen_f8e4m3(&chunk, DType::BF16, &Device::Cpu)
            .expect("widen")
            .expect("bf16 is a supported target");
        let theirs = chunk.to_dtype(DType::BF16).expect("candle bf16");

        assert_eq!(ours.dims(), &[2, 6]);
        assert_eq!(
            ours.to_vec2::<bf16>().expect("ours"),
            theirs.to_vec2::<bf16>().expect("theirs")
        );
    }

    /// A dtype without a table must decline rather than guess, so candle stays
    /// the authority for conversions this module does not implement.
    #[test]
    fn an_unsupported_target_dtype_declines() {
        let source = every_f8e4m3_value();
        assert!(widen_f8e4m3(&source, DType::F64, &Device::Cpu)
            .expect("widen")
            .is_none());

        let already_wide = Tensor::zeros(4, DType::BF16, &Device::Cpu).expect("bf16 tensor");
        assert!(widen_f8e4m3(&already_wide, DType::BF16, &Device::Cpu)
            .expect("widen")
            .is_none());
    }
}
